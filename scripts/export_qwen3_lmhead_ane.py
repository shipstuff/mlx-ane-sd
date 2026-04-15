"""Extract Qwen3-4B lm_head weight, export as CoreML model, LUT6 quantize.

Output: /tmp/lmhead_qwen3/
  lmhead_fp16.mlpackage      -- fp16, pre-quantization
  lmhead_fp16.mlmodelc       -- compiled
  lmhead_lut6.mlpackage       -- LUT6 palettized
  lmhead_lut6.mlmodelc        -- compiled

The model takes fp16 [1, 15, 2560] hidden input (draft's last 15 positions
pre-sliced by caller) and returns fp16 [1, 15, vocab] logits. Host does argmax.

Quick quality check: compare top-1 argmax against MLX bf16 reference on a
hidden state from an actual DFlash draft output.
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
import coremltools as ct
import coremltools.optimize.coreml as cto


def extract_mlx_lmhead_weight() -> np.ndarray:
    """Load mlx-community/Qwen3-4B-bf16 and return lm_head weight as fp16 numpy."""
    import mlx.core as mx
    from mlx_lm import load as mlx_load

    print("[extract] loading mlx-community/Qwen3-4B-bf16...", flush=True)
    model, _ = mlx_load("mlx-community/Qwen3-4B-bf16")

    if hasattr(model, "lm_head") and model.lm_head is not None:
        w = model.lm_head.weight  # [vocab, hidden] bf16
        tied = False
    else:
        # Tied embedding (Qwen3-4B has untied but check anyway)
        w = model.model.embed_tokens.weight
        tied = True
    print(f"[extract] lm_head.weight shape={w.shape} dtype={w.dtype} tied={tied}")
    # bf16 -> fp32 -> fp16 (numpy fp16 is IEEE half, loses some range but OK for lm_head)
    w_np = np.asarray(w.astype(mx.float32)).astype(np.float16)
    return w_np


class Qwen3LMHead(nn.Module):
    """Real Qwen3-4B lm_head: linear to vocab, no bias."""
    def __init__(self, weight: np.ndarray):
        super().__init__()
        V, H = weight.shape
        self.weight = nn.Parameter(torch.from_numpy(weight.copy()))

    def forward(self, hidden):
        # hidden: [1, L, H] -> logits: [1, L, V]
        return torch.matmul(hidden, self.weight.T)


def convert_and_quantize(weight: np.ndarray, block_size_out: int, out_dir: Path) -> Path:
    out_dir.mkdir(exist_ok=True, parents=True)
    fp16_mlpackage = out_dir / "lmhead_fp16.mlpackage"
    fp16_mlmodelc = out_dir / "lmhead_fp16.mlmodelc"
    lut6_mlpackage = out_dir / "lmhead_lut6.mlpackage"
    lut6_mlmodelc = out_dir / "lmhead_lut6.mlmodelc"

    # 1. torch model with REAL Qwen3-4B lm_head weights
    print(f"[torch] building Qwen3LMHead with weight shape={weight.shape}")
    model = Qwen3LMHead(weight.astype(np.float16)).eval()
    model = model.to(torch.float16)

    hidden = weight.shape[1]
    example = torch.randn(1, block_size_out, hidden, dtype=torch.float16)
    traced = torch.jit.trace(model, example)

    # 2. Convert to CoreML fp16
    print(f"[convert] fp16 ML Program...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="hidden", shape=example.shape, dtype=np.float16)],
        outputs=[ct.TensorType(name="logits", dtype=np.float16)],
        convert_to="mlprogram",
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")
    if fp16_mlpackage.exists():
        shutil.rmtree(fp16_mlpackage)
    mlmodel.save(str(fp16_mlpackage))

    # 3. LUT6 palettize
    print(f"[palettize] LUT6 group_size=16...")
    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=6, mode="kmeans",
            granularity="per_grouped_channel", group_size=16,
        )
    )
    t0 = time.perf_counter()
    mlmodel_q = cto.palettize_weights(mlmodel, config)
    print(f"[palettize] done in {time.perf_counter()-t0:.1f}s")
    if lut6_mlpackage.exists():
        shutil.rmtree(lut6_mlpackage)
    mlmodel_q.save(str(lut6_mlpackage))

    # 4. Compile both
    for mlpkg, mlmc in [(fp16_mlpackage, fp16_mlmodelc), (lut6_mlpackage, lut6_mlmodelc)]:
        if mlmc.exists():
            shutil.rmtree(mlmc)
        print(f"[compile] {mlpkg.name} -> {mlmc.name}")
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpkg), str(out_dir)],
            check=True, capture_output=True,
        )
    return lut6_mlmodelc


def quality_check(lut6_mlmodelc: Path, weight: np.ndarray, n_samples: int = 20,
                   block_size: int = 15):
    """Compare ANE LUT6 top-1 argmax to fp32 reference on random hiddens.

    Returns (agreement_rate, mean_top5_overlap).
    """
    print(f"\n[quality] running {n_samples} random inputs at bs={block_size}...")
    model = ct.models.CompiledMLModel(str(lut6_mlmodelc), ct.ComputeUnit.CPU_AND_NE)

    V, H = weight.shape
    w_f32 = weight.astype(np.float32)

    agreements = []
    top5_overlaps = []
    for trial in range(n_samples):
        np.random.seed(trial)
        hidden = np.random.randn(1, block_size, H).astype(np.float16) * 0.5

        h_f32 = hidden.astype(np.float32)
        logits_ref = h_f32 @ w_f32.T
        argmax_ref = np.argmax(logits_ref, axis=-1)[0]
        top5_ref = np.argsort(logits_ref[0], axis=-1)[:, -5:]

        out = model.predict({"hidden": hidden})
        logits_ane = out["logits"].astype(np.float32)
        argmax_ane = np.argmax(logits_ane, axis=-1)[0]
        top5_ane = np.argsort(logits_ane[0], axis=-1)[:, -5:]

        agreement = (argmax_ref == argmax_ane).mean()
        top5_overlap = np.mean([
            len(set(top5_ref[i]) & set(top5_ane[i])) / 5.0 for i in range(block_size)
        ])
        agreements.append(agreement)
        top5_overlaps.append(top5_overlap)

    mean_agreement = np.mean(agreements)
    mean_top5 = np.mean(top5_overlaps)
    print(f"[quality] top-1 argmax agreement:  {mean_agreement*100:.2f}%")
    print(f"[quality] top-5 average overlap:   {mean_top5*100:.2f}%")
    print(f"[quality] per-trial min agreement: {min(agreements)*100:.1f}%")
    return mean_agreement, mean_top5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/tmp/lmhead_qwen3")
    ap.add_argument("--skip-extract", action="store_true",
                    help="Skip weight extraction (use cached /tmp/qwen3_lmhead_weight.npy)")
    ap.add_argument("--block-size-out", type=int, default=15,
                    help="Number of positions to project (15 for draft's last-15; 16 for target_verify's full block)")
    args = ap.parse_args()

    cache = Path("/tmp/qwen3_lmhead_weight.npy")
    if args.skip_extract and cache.exists():
        print(f"[cache] loading {cache}")
        weight = np.load(str(cache))
    else:
        weight = extract_mlx_lmhead_weight()
        np.save(str(cache), weight)
        print(f"[cache] saved to {cache}")

    print(f"\n[weight] shape={weight.shape} dtype={weight.dtype} "
          f"range=[{weight.min():.4f}, {weight.max():.4f}] std={weight.std():.4f}")

    # Output dir disambiguated by bs when != 15 to avoid overwriting the draft-bs-15 artifact
    base_out = Path(args.out_dir)
    out_dir = base_out if args.block_size_out == 15 else base_out / f"bs{args.block_size_out}"
    lut6_mlmodelc = convert_and_quantize(weight, block_size_out=args.block_size_out, out_dir=out_dir)
    quality_check(lut6_mlmodelc, weight, n_samples=20, block_size=args.block_size_out)

    print(f"\n[done] LUT6 model at {lut6_mlmodelc}")


if __name__ == "__main__":
    main()
