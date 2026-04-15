"""2c probe: port one Qwen3-4B transformer layer to CoreML, measure ANE latency.

If per-layer ANE cost < GPU's 2.0ms (bf16 target), partial-target-on-ANE is
viable. If ~equal or slower, skip.

Uses layer 0's weights from mlx-community/Qwen3-4B-bf16. Measures standalone
forward pass for [1, 16, 2560] input (same shape as our SD target_verify).
Uses external KV cache pattern (stateless model) for ANE-friendly compile.
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


# Qwen3-4B-bf16 config (from model's config.json)
HIDDEN = 2560
N_HEADS = 32
N_KV_HEADS = 8
HEAD_DIM = 128
INTERMEDIATE = 9728  # MLP hidden dim
RMS_EPS = 1e-6
ROPE_THETA = 1_000_000


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        # fp16-safe RMS norm
        x2 = x.float() ** 2
        r = torch.rsqrt(x2.mean(-1, keepdim=True) + self.eps)
        return (x.float() * r).to(x.dtype) * self.weight


HALF_DIM = HEAD_DIM // 2  # 64


def rotate_half(x):
    # x: [..., head_dim]; split into (x1, x2) of head_dim/2 each
    x1 = x[..., :HALF_DIM]
    x2 = x[..., HALF_DIM:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x, cos, sin):
    # x: [1, heads, L, head_dim]. cos/sin: [L, head_dim]
    cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, L, dh]
    sin = sin.unsqueeze(0).unsqueeze(0)
    return x * cos + rotate_half(x) * sin


class Qwen3Block(nn.Module):
    """Qwen3 transformer block with external KV cache. Stateless for ANE."""
    def __init__(self):
        super().__init__()
        self.input_ln = RMSNorm(HIDDEN, RMS_EPS)
        self.post_ln = RMSNorm(HIDDEN, RMS_EPS)
        self.q_proj = nn.Linear(HIDDEN, N_HEADS * HEAD_DIM, bias=False)
        self.k_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.v_proj = nn.Linear(HIDDEN, N_KV_HEADS * HEAD_DIM, bias=False)
        self.o_proj = nn.Linear(N_HEADS * HEAD_DIM, HIDDEN, bias=False)
        self.q_norm = RMSNorm(HEAD_DIM, RMS_EPS)
        self.k_norm = RMSNorm(HEAD_DIM, RMS_EPS)
        self.gate = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.up = nn.Linear(HIDDEN, INTERMEDIATE, bias=False)
        self.down = nn.Linear(INTERMEDIATE, HIDDEN, bias=False)

    def forward(self, x, cos_q, sin_q, cos_k_new, sin_k_new, cache_k, cache_v, causal_mask):
        # Shapes are all hardcoded for static-shape CoreML compile:
        # x: [1, L=16, hidden=2560]
        # cos_q/sin_q/cos_k_new/sin_k_new: [L=16, head_dim=128]
        # cache_k, cache_v: [1, n_kv_heads=8, state_len, head_dim]
        # causal_mask: [1, 1, L, state+L]
        h = x
        y = self.input_ln(x)
        # Projections with explicit shape reshape
        q = self.q_proj(y).reshape(1, 16, N_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 32, 16, 128]
        k = self.k_proj(y).reshape(1, 16, N_KV_HEADS, HEAD_DIM).transpose(1, 2)  # [1, 8, 16, 128]
        v = self.v_proj(y).reshape(1, 16, N_KV_HEADS, HEAD_DIM).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        q = apply_rope(q, cos_q, sin_q)
        new_k = apply_rope(k, cos_k_new, sin_k_new)  # [1, 8, 16, 128]
        new_v = v

        # Concat past + new
        full_k = torch.cat([cache_k, new_k], dim=2)  # [1, 8, state+16, 128]
        full_v = torch.cat([cache_v, new_v], dim=2)

        # GQA: expand kv heads 8 -> 32 (repeat by factor 4)
        full_k = full_k.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)  # [1, 32, state+16, 128]
        full_v = full_v.repeat_interleave(N_HEADS // N_KV_HEADS, dim=1)

        # Attention
        scale = 1.0 / (HEAD_DIM ** 0.5)
        scores = torch.matmul(q, full_k.transpose(-2, -1)) * scale  # [1, 32, 16, state+16]
        scores = scores + causal_mask
        attn = scores.softmax(-1)
        out = torch.matmul(attn, full_v)  # [1, 32, 16, 128]
        out = out.transpose(1, 2).reshape(1, 16, N_HEADS * HEAD_DIM)
        out = self.o_proj(out)
        h = h + out

        # MLP
        y = self.post_ln(h)
        mlp_out = self.down(F.silu(self.gate(y)) * self.up(y))
        return h + mlp_out, new_k, new_v


def extract_layer0_weights() -> dict:
    import mlx.core as mx
    from mlx_lm import load as mlx_load
    print("[extract] loading mlx-community/Qwen3-4B-bf16...", flush=True)
    model, _ = mlx_load("mlx-community/Qwen3-4B-bf16")
    layer0 = model.model.layers[0]
    weights = {}
    weights["q_proj"] = np.asarray(layer0.self_attn.q_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["k_proj"] = np.asarray(layer0.self_attn.k_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["v_proj"] = np.asarray(layer0.self_attn.v_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["o_proj"] = np.asarray(layer0.self_attn.o_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["q_norm"] = np.asarray(layer0.self_attn.q_norm.weight.astype(mx.float32)).astype(np.float16)
    weights["k_norm"] = np.asarray(layer0.self_attn.k_norm.weight.astype(mx.float32)).astype(np.float16)
    weights["gate"] = np.asarray(layer0.mlp.gate_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["up"] = np.asarray(layer0.mlp.up_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["down"] = np.asarray(layer0.mlp.down_proj.weight.astype(mx.float32)).astype(np.float16)
    weights["input_ln"] = np.asarray(layer0.input_layernorm.weight.astype(mx.float32)).astype(np.float16)
    weights["post_ln"] = np.asarray(layer0.post_attention_layernorm.weight.astype(mx.float32)).astype(np.float16)
    return weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-len", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out-dir", default="/tmp/qwen3_block_ane")
    args = ap.parse_args()

    L = args.block_size
    STATE = args.state_len

    print(f"[config] block_size={L}, state_len={STATE}, "
          f"hidden={HIDDEN}, heads={N_HEADS}, kv_heads={N_KV_HEADS}")

    w = extract_layer0_weights()
    print(f"[extracted] q_proj {w['q_proj'].shape}, "
          f"k_proj {w['k_proj'].shape}, gate {w['gate'].shape}")

    # Build torch model with real weights
    model = Qwen3Block().eval()
    model = model.to(torch.float16)
    with torch.no_grad():
        model.q_proj.weight.copy_(torch.from_numpy(w["q_proj"]))
        model.k_proj.weight.copy_(torch.from_numpy(w["k_proj"]))
        model.v_proj.weight.copy_(torch.from_numpy(w["v_proj"]))
        model.o_proj.weight.copy_(torch.from_numpy(w["o_proj"]))
        model.q_norm.weight.copy_(torch.from_numpy(w["q_norm"]))
        model.k_norm.weight.copy_(torch.from_numpy(w["k_norm"]))
        model.gate.weight.copy_(torch.from_numpy(w["gate"]))
        model.up.weight.copy_(torch.from_numpy(w["up"]))
        model.down.weight.copy_(torch.from_numpy(w["down"]))
        model.input_ln.weight.copy_(torch.from_numpy(w["input_ln"]))
        model.post_ln.weight.copy_(torch.from_numpy(w["post_ln"]))

    # Dummy inputs for trace (only cos_k_new passed, which is for L new positions)
    x = torch.randn(1, L, HIDDEN, dtype=torch.float16) * 0.1
    cos_q = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    sin_q = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    cos_k_new = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    sin_k_new = torch.randn(L, HEAD_DIM, dtype=torch.float16)
    cache_k = torch.zeros(1, N_KV_HEADS, STATE, HEAD_DIM, dtype=torch.float16)
    cache_v = torch.zeros(1, N_KV_HEADS, STATE, HEAD_DIM, dtype=torch.float16)
    causal_mask = torch.zeros(1, 1, L, STATE + L, dtype=torch.float16)

    # Test forward
    print("[test] torch forward...")
    with torch.no_grad():
        out, new_k, new_v = model(x, cos_q, sin_q, cos_k_new, sin_k_new, cache_k, cache_v, causal_mask)
    print(f"[test] output shape {out.shape}, new_k {new_k.shape}, new_v {new_v.shape}")

    print("[trace]...")
    traced = torch.jit.trace(model, (x, cos_q, sin_q, cos_k_new, sin_k_new, cache_k, cache_v, causal_mask))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    fp16_pkg = out_dir / "qwen3_block.mlpackage"
    lut6_pkg = out_dir / "qwen3_block_lut6.mlpackage"

    print("[convert] to CoreML fp16...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="x", shape=x.shape, dtype=np.float16),
            ct.TensorType(name="cos_q", shape=cos_q.shape, dtype=np.float16),
            ct.TensorType(name="sin_q", shape=sin_q.shape, dtype=np.float16),
            ct.TensorType(name="cos_k_new", shape=cos_k_new.shape, dtype=np.float16),
            ct.TensorType(name="sin_k_new", shape=sin_k_new.shape, dtype=np.float16),
            ct.TensorType(name="cache_k", shape=cache_k.shape, dtype=np.float16),
            ct.TensorType(name="cache_v", shape=cache_v.shape, dtype=np.float16),
            ct.TensorType(name="causal_mask", shape=causal_mask.shape, dtype=np.float16),
        ],
        outputs=[
            ct.TensorType(name="out", dtype=np.float16),
            ct.TensorType(name="new_k", dtype=np.float16),
            ct.TensorType(name="new_v", dtype=np.float16),
        ],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")
    if fp16_pkg.exists():
        shutil.rmtree(fp16_pkg)
    mlmodel.save(str(fp16_pkg))

    print("[palettize] LUT6 per_tensor...")
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=6, mode="kmeans", granularity="per_tensor"
        )
    )
    t0 = time.perf_counter()
    mlmodel_q = cto.palettize_weights(mlmodel, config)
    print(f"[palettize] done in {time.perf_counter()-t0:.1f}s")
    if lut6_pkg.exists():
        shutil.rmtree(lut6_pkg)
    mlmodel_q.save(str(lut6_pkg))

    for pkg in [fp16_pkg, lut6_pkg]:
        mlmc = pkg.with_suffix(".mlmodelc")
        if mlmc.exists():
            shutil.rmtree(mlmc)
        subprocess.run(["xcrun", "coremlcompiler", "compile", str(pkg), str(out_dir)],
                       check=True, capture_output=True)

    # Profile both
    for label, mlmc in [("fp16", fp16_pkg.with_suffix(".mlmodelc")),
                         ("lut6", lut6_pkg.with_suffix(".mlmodelc"))]:
        print(f"\n=== {label} ===")
        result = subprocess.run(
            [str(Path.home() / "projects/anemll-profile/anemll-profile"), str(mlmc)],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            for key in ["Model size:", "ANE ops:", "CPU ops:", "Measured:",
                         "ANE compilation", "CPU fallback:", "ANE graph interruptions:"]:
                if key in line:
                    print(f"  {line.strip()}")
                    break


if __name__ == "__main__":
    main()
