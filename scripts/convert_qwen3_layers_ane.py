"""2c Phase 1: convert K Qwen3-4B layers to one CoreML model for ANE.

Chains K Qwen3 transformer blocks with external KV cache (one cache per
layer). Extracts real weights from mlx-community/Qwen3-4B-bf16. LUT6
quantizes. Profiles standalone ANE latency.

Goal: validate per-layer latency scaling. Expect K × ~1.55ms + small
fixed overhead. If measured >> projection, chunking strategy needs
revision.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import coremltools as ct
import coremltools.optimize.coreml as cto


# Qwen3-4B-bf16 config
HIDDEN = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000
HALF_DIM = HEAD_DIM // 2  # 64


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=RMS_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        x2 = x.float() ** 2
        r = torch.rsqrt(x2.mean(-1, keepdim=True) + self.eps)
        return (x.float() * r).to(x.dtype) * self.weight


def rotate_half(x):
    x1 = x[..., :HALF_DIM]
    x2 = x[..., HALF_DIM:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class Qwen3Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_ln = RMSNorm(HIDDEN)
        self.post_ln = RMSNorm(HIDDEN)
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM)
        self.k_norm = RMSNorm(HEAD_DIM)
        self.gate = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x, cos_q, sin_q, cos_k_new, sin_k_new,
                cache_k, cache_v, causal_mask):
        # x: [1, 16, 2560], cache: [1, 8, state, 128]
        h = x
        y = self.input_ln(x)
        q = self.q_proj(y).reshape(1, 16, N_HEADS, HEAD_DIM).transpose(1, 2)
        k = self.k_proj(y).reshape(1, 16, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        v = self.v_proj(y).reshape(1, 16, N_KV_HEADS, HEAD_DIM).transpose(1, 2)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = apply_rope(q, cos_q, sin_q)
        new_k = apply_rope(k, cos_k_new, sin_k_new)
        new_v = v
        full_k = torch.cat([cache_k, new_k], dim=2)
        full_v = torch.cat([cache_v, new_v], dim=2)
        full_k = full_k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        full_v = full_v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.matmul(q, full_k.transpose(-2, -1)) * scale
        scores = scores + causal_mask
        attn = scores.softmax(-1)
        out = torch.matmul(attn, full_v)
        out = out.transpose(1, 2).reshape(1, 16, N_HEADS * HEAD_DIM)
        out = self.o_proj(out)
        h = h + out
        y = self.post_ln(h)
        mlp_out = self.down(F.silu(self.gate(y)) * self.up(y))
        return h + mlp_out, new_k, new_v


class Qwen3MultiLayer(nn.Module):
    """K chained Qwen3 blocks with external KV cache.

    Outputs final hidden + stacked K/V for cache update + optional
    intermediate hiddens at `capture_indices` (0-based within this K layers,
    for DFlash's target-layer capture).
    """
    def __init__(self, num_layers: int, capture_indices: tuple[int, ...] = ()):
        super().__init__()
        self.K = num_layers
        self.capture_indices = sorted(capture_indices)
        self.layers = nn.ModuleList([Qwen3Block() for _ in range(num_layers)])

    def forward(self, x, cos_q, sin_q, cos_k_new, sin_k_new,
                cache_k_all, cache_v_all, causal_mask):
        new_ks = []
        new_vs = []
        captures = []
        for i, layer in enumerate(self.layers):
            x, nk, nv = layer(x, cos_q, sin_q, cos_k_new, sin_k_new,
                              cache_k_all[i], cache_v_all[i], causal_mask)
            new_ks.append(nk)
            new_vs.append(nv)
            if i in self.capture_indices:
                captures.append(x)
        new_k_all = torch.stack(new_ks, dim=0)
        new_v_all = torch.stack(new_vs, dim=0)
        if captures:
            # Stack captures along new dim 0: [n_captures, 1, 16, 2560]
            cap_tensor = torch.stack(captures, dim=0)
            return x, new_k_all, new_v_all, cap_tensor
        return x, new_k_all, new_v_all


def extract_layers_weights(num_layers: int, start_layer: int = 0) -> list[dict]:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    end = start_layer + num_layers
    print(f"[extract] loading mlx-community/Qwen3-4B-bf16 and "
          f"extracting layers {start_layer}..{end-1}...", flush=True)
    model, _ = mlx_load("mlx-community/Qwen3-4B-bf16")
    out = []
    for i in range(start_layer, end):
        ly = model.model.layers[i]
        w = {}
        w["q_proj"] = np.asarray(ly.self_attn.q_proj.weight.astype(mx.float32)).astype(np.float16)
        w["k_proj"] = np.asarray(ly.self_attn.k_proj.weight.astype(mx.float32)).astype(np.float16)
        w["v_proj"] = np.asarray(ly.self_attn.v_proj.weight.astype(mx.float32)).astype(np.float16)
        w["o_proj"] = np.asarray(ly.self_attn.o_proj.weight.astype(mx.float32)).astype(np.float16)
        w["q_norm"] = np.asarray(ly.self_attn.q_norm.weight.astype(mx.float32)).astype(np.float16)
        w["k_norm"] = np.asarray(ly.self_attn.k_norm.weight.astype(mx.float32)).astype(np.float16)
        w["gate"] = np.asarray(ly.mlp.gate_proj.weight.astype(mx.float32)).astype(np.float16)
        w["up"] = np.asarray(ly.mlp.up_proj.weight.astype(mx.float32)).astype(np.float16)
        w["down"] = np.asarray(ly.mlp.down_proj.weight.astype(mx.float32)).astype(np.float16)
        w["input_ln"] = np.asarray(ly.input_layernorm.weight.astype(mx.float32)).astype(np.float16)
        w["post_ln"] = np.asarray(ly.post_attention_layernorm.weight.astype(mx.float32)).astype(np.float16)
        out.append(w)
    return out


def load_weights_into_model(model: Qwen3MultiLayer, weights: list[dict]):
    with torch.no_grad():
        for i, w in enumerate(weights):
            layer = model.layers[i]
            layer.q_proj.weight.copy_(torch.from_numpy(w["q_proj"]))
            layer.k_proj.weight.copy_(torch.from_numpy(w["k_proj"]))
            layer.v_proj.weight.copy_(torch.from_numpy(w["v_proj"]))
            layer.o_proj.weight.copy_(torch.from_numpy(w["o_proj"]))
            layer.q_norm.weight.copy_(torch.from_numpy(w["q_norm"]))
            layer.k_norm.weight.copy_(torch.from_numpy(w["k_norm"]))
            layer.gate.weight.copy_(torch.from_numpy(w["gate"]))
            layer.up.weight.copy_(torch.from_numpy(w["up"]))
            layer.down.weight.copy_(torch.from_numpy(w["down"]))
            layer.input_ln.weight.copy_(torch.from_numpy(w["input_ln"]))
            layer.post_ln.weight.copy_(torch.from_numpy(w["post_ln"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-layers", type=int, default=5)
    ap.add_argument("--start-layer", type=int, default=0,
                    help="Source layer offset into Qwen3-4B (for multi-chunk). "
                         "K=18 start=0 = chunk 1 (layers 0-17); start=18 = chunk 2 (layers 18-35).")
    ap.add_argument("--state-len", type=int, default=256)
    ap.add_argument("--out-dir", default="/tmp/qwen3_klayers")
    ap.add_argument("--skip-lut6", action="store_true")
    ap.add_argument("--capture-indices", type=str, default="",
                    help="Comma-separated 0-based layer indices within K to expose (for DFlash captures). E.g. '1' for K>=2, '1,9' for K>=10, '1,9,17' for K>=18. For chunk 2 starting at global layer 18, captures [25, 33] map to local [7, 15].")
    args = ap.parse_args()

    K = args.num_layers
    START = args.start_layer
    capture_indices = tuple(int(x) for x in args.capture_indices.split(",") if x.strip())
    if capture_indices:
        print(f"[config] start={START} K={K}, capture at local indices {capture_indices}")
        for idx in capture_indices:
            assert idx < K, f"capture {idx} out of range for K={K}"
    L = 16  # block_size
    STATE = args.state_len
    # Include start in dir name when > 0 so different chunks don't collide
    subdir = f"K{K}" if START == 0 else f"K{K}_s{START}"
    out_dir = Path(args.out_dir) / subdir
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"[config] start={START}, K={K} layers, state={STATE}, block_size={L}")

    weights = extract_layers_weights(K, start_layer=START)

    model = Qwen3MultiLayer(K, capture_indices=capture_indices).eval().to(torch.float16)
    load_weights_into_model(model, weights)

    # Example inputs
    x = torch.randn(1, L, HIDDEN, dtype=torch.float16) * 0.1
    cos_q = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    sin_q = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    cos_k_new = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    sin_k_new = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    cache_k_all = torch.zeros(K, 1, N_KV_HEADS, STATE, HEAD_DIM, dtype=torch.float16)
    cache_v_all = torch.zeros(K, 1, N_KV_HEADS, STATE, HEAD_DIM, dtype=torch.float16)
    causal_mask = torch.zeros(1, 1, L, STATE + L, dtype=torch.float16)

    print("[test] torch forward...")
    with torch.no_grad():
        result = model(x, cos_q, sin_q, cos_k_new, sin_k_new,
                        cache_k_all, cache_v_all, causal_mask)
    if capture_indices:
        out, new_k_all, new_v_all, captures = result
        print(f"[test] out {tuple(out.shape)}, new_k_all {tuple(new_k_all.shape)}, "
              f"captures {tuple(captures.shape)}")
    else:
        out, new_k_all, new_v_all = result
        print(f"[test] out {tuple(out.shape)}, new_k_all {tuple(new_k_all.shape)}")

    print("[trace]...")
    traced = torch.jit.trace(model, (x, cos_q, sin_q, cos_k_new, sin_k_new,
                                       cache_k_all, cache_v_all, causal_mask))

    fp16_pkg = out_dir / f"qwen3_K{K}.mlpackage"
    lut6_pkg = out_dir / f"qwen3_K{K}_lut6.mlpackage"

    print("[convert] fp16 ML Program...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="cos_q", shape=cos_q.shape, dtype=np.float16),
            ct.TensorType(name="sin_q", shape=sin_q.shape, dtype=np.float16),
            ct.TensorType(name="cos_k_new", shape=cos_k_new.shape, dtype=np.float16),
            ct.TensorType(name="sin_k_new", shape=sin_k_new.shape, dtype=np.float16),
            ct.TensorType(name="cache_k_all", shape=cache_k_all.shape, dtype=np.float16),
            ct.TensorType(name="cache_v_all", shape=cache_v_all.shape, dtype=np.float16),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ],
        outputs=(
            [ct.TensorType(name="out", dtype=np.float16),
             ct.TensorType(name="new_k_all", dtype=np.float16),
             ct.TensorType(name="new_v_all", dtype=np.float16)]
            + ([ct.TensorType(name="captures", dtype=np.float16)] if capture_indices else [])
        ),
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.iOS18,  # required for per_grouped_channel LUT
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")
    if fp16_pkg.exists():
        shutil.rmtree(fp16_pkg)
    mlmodel.save(str(fp16_pkg))

    if not args.skip_lut6:
        print(f"[palettize] LUT6 per_grouped_channel gs=16 (K={K}, may take {K*10}+s)...")
        config = cto.OptimizationConfig(
            global_config=cto.OpPalettizerConfig(
                nbits=6, mode="kmeans",
                granularity="per_grouped_channel", group_size=16,
            )
        )
        t0 = time.perf_counter()
        mlmodel_q = cto.palettize_weights(mlmodel, config)
        print(f"[palettize] done in {time.perf_counter()-t0:.1f}s")
        if lut6_pkg.exists():
            shutil.rmtree(lut6_pkg)
        mlmodel_q.save(str(lut6_pkg))

    packages = [("fp16", fp16_pkg)]
    if not args.skip_lut6:
        packages.append(("lut6", lut6_pkg))

    for label, pkg in packages:
        mlmc = pkg.with_suffix(".mlmodelc")
        if mlmc.exists():
            shutil.rmtree(mlmc)
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)],
                       check=True, capture_output=True)
        print(f"\n=== K={K} {label} ===")
        result = subprocess.run(
            [str(Path.home() / "projects/anemll-profile/anemll-profile"), str(mlmc)],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            for key in ["Model size:", "ANE ops:", "CPU ops:", "Measured:",
                         "ANE compilation", "CPU fallback:", "ANE graph interruptions:",
                         "Weight BW:", "Compute:"]:
                if key in line and "ms/prediction" in line or key in line:
                    print(f"  {line.strip()}")
                    break


if __name__ == "__main__":
    main()
