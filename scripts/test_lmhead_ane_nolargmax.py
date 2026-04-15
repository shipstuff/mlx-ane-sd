"""Variant 2: lm_head only (no argmax) — argmax stays on host.

If ANE can't handle reduce_argmax efficiently, keep the matmul on ANE and
do argmax on CPU after. The matmul is the expensive part (~778MB weight).
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


class LMHeadOnly(nn.Module):
    """Just the linear projection — no argmax, no slice."""
    def __init__(self, hidden_size: int, vocab_size: int, bs_out: int):
        super().__init__()
        self.bs_out = bs_out
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=torch.float16) * 0.02)

    def forward(self, hidden):
        # hidden: [1, bs_out, hidden_size] (pre-sliced by caller)
        return torch.matmul(hidden, self.weight.T)  # [1, bs_out, vocab]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=151936)
    ap.add_argument("--hidden-size", type=int, default=2560)
    ap.add_argument("--block-size-out", type=int, default=15,
                    help="number of positions to project (bs-1 = 15)")
    ap.add_argument("--out-dir", default="/tmp/lmhead_ane_v2")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    mlpackage = out_dir / "lmhead_nolarg.mlpackage"
    mlmodelc = out_dir / "lmhead_nolarg.mlmodelc"
    out_dir.mkdir(exist_ok=True, parents=True)

    print(f"[build] lm_head (no argmax) [{args.vocab_size} x {args.hidden_size}], "
          f"bs_out={args.block_size_out}")
    model = LMHeadOnly(args.hidden_size, args.vocab_size, args.block_size_out).eval()

    example = torch.randn(1, args.block_size_out, args.hidden_size, dtype=torch.float16)
    traced = torch.jit.trace(model, example)

    print("[convert] to CoreML ML Program, fp16, CPU_AND_NE")
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

    if mlpackage.exists():
        shutil.rmtree(mlpackage)
    mlmodel.save(str(mlpackage))
    print(f"[save] {mlpackage}")

    if mlmodelc.exists():
        shutil.rmtree(mlmodelc)
    subprocess.run(
        ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(out_dir)],
        check=True, capture_output=True,
    )
    print(f"[compiled] {mlmodelc}")

    # Bench
    print("\n[load] compiled model on CPU_AND_NE...")
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(str(mlmodelc), ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] in {time.perf_counter()-t0:.2f}s")

    example_np = np.random.randn(1, args.block_size_out, args.hidden_size).astype(np.float16)

    for _ in range(10):
        model.predict({"hidden": example_np})

    iters = 100
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        out = model.predict({"hidden": example_np})
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    mean = np.mean(samples)

    # Check output shape
    logits = out["logits"]
    print(f"\n=== lm_head (no argmax) latency, output shape {logits.shape} ===")
    print(f"  mean:   {mean:.3f} ms  median: {samples[50]:.3f} ms  "
          f"p10: {samples[10]:.3f} p90: {samples[90]:.3f}")
    print(f"  + host argmax: ~0.1 ms (for [1, {args.block_size_out}, {args.vocab_size}] tensor)")

    print(f"\n[profile] anemll-profile...")
    result = subprocess.run(
        [str(Path.home() / "projects/anemll-profile/anemll-profile"), str(mlmodelc)],
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
