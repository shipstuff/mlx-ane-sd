"""Quick CoreML test: can ANE host a full-vocab lm_head + argmax efficiently?

Builds a standalone torch module of just [hidden -> vocab] linear + argmax,
converts to CoreML with CPU_AND_NE target, profiles latency.

If ANE predict < 20ms (current GPU draft_lmhead cost), it's a win.
If > 20ms, need to reconsider (maybe top-k + CPU final argmax, etc.)
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


class LMHeadArgmax(nn.Module):
    """lm_head matmul + argmax over vocab dim. Takes only the last (bs-1) positions
    since those are the draft's predictions.
    """
    def __init__(self, hidden_size: int, vocab_size: int, bs: int):
        super().__init__()
        self.bs = bs
        # Use tied weight layout that matches Qwen3's lm_head: [vocab, hidden]
        self.weight = nn.Parameter(torch.randn(vocab_size, hidden_size, dtype=torch.float16) * 0.02)

    def forward(self, hidden):
        # hidden: [1, bs, hidden_size]
        # Slice to last bs-1 positions (draft predictions)
        sliced = hidden[:, 1 - self.bs:, :]  # [1, bs-1, hidden_size]
        logits = torch.matmul(sliced, self.weight.T)  # [1, bs-1, vocab]
        return torch.argmax(logits, dim=-1)  # [1, bs-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab-size", type=int, default=151936)  # Qwen3 vocab
    ap.add_argument("--hidden-size", type=int, default=2560)   # Qwen3-4B hidden
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--out-dir", default="/tmp/lmhead_ane")
    ap.add_argument("--skip-convert", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    mlpackage = out_dir / "lmhead.mlpackage"
    mlmodelc = out_dir / "lmhead.mlmodelc"
    out_dir.mkdir(exist_ok=True, parents=True)

    if not args.skip_convert:
        # Build torch model, trace, convert
        print(f"[build] lm_head [{args.vocab_size} x {args.hidden_size}], block={args.block_size}")
        model = LMHeadArgmax(args.hidden_size, args.vocab_size, args.block_size).eval()

        example = torch.randn(1, args.block_size, args.hidden_size, dtype=torch.float16)
        print(f"[trace] input shape = {tuple(example.shape)}")
        traced = torch.jit.trace(model, example)

        print("[convert] to CoreML ML Program, fp16, CPU_AND_NE")
        t0 = time.perf_counter()
        mlmodel = ct.convert(
            traced,
            inputs=[ct.TensorType(name="hidden", shape=example.shape, dtype=np.float16)],
            outputs=[ct.TensorType(name="tokens", dtype=np.int32)],
            convert_to="mlprogram",
            compute_units=ct.ComputeUnit.CPU_AND_NE,
            compute_precision=ct.precision.FLOAT16,
            minimum_deployment_target=ct.target.macOS15,
        )
        t_convert = time.perf_counter() - t0
        print(f"[convert] done in {t_convert:.1f}s")

        if mlpackage.exists():
            shutil.rmtree(mlpackage)
        mlmodel.save(str(mlpackage))
        print(f"[save] {mlpackage}")

        # Compile to .mlmodelc (which is what anemll-profile expects)
        if mlmodelc.exists():
            shutil.rmtree(mlmodelc)
        print(f"[compile] to {mlmodelc}...")
        compiled = ct.models.CompiledMLModel(str(mlpackage), ct.ComputeUnit.CPU_AND_NE)
        # If the compile step went through, we can copy the .mlmodelc from the auto location
        # Actually, easier: use xcrun coremlcompiler
        subprocess.run(
            ["xcrun", "coremlcompiler", "compile", str(mlpackage), str(out_dir)],
            check=True, capture_output=True,
        )
        # Compiler outputs lmhead.mlmodelc in out_dir
        print(f"[compiled] {mlmodelc}")

    # Load + bench
    print("\n[load] compiled model on CPU_AND_NE...")
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(str(mlmodelc), ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] in {time.perf_counter()-t0:.2f}s")

    example_np = np.random.randn(1, args.block_size, args.hidden_size).astype(np.float16)

    # Warmup
    for _ in range(10):
        model.predict({"hidden": example_np})

    # Benchmark
    iters = 100
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict({"hidden": example_np})
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    mean = np.mean(samples)
    median = samples[iters // 2]
    p10 = samples[iters // 10]
    p90 = samples[iters * 9 // 10]

    print(f"\n=== lm_head ANE latency ({iters} iters, after 10 warmup) ===")
    print(f"  mean:   {mean:.3f} ms")
    print(f"  median: {median:.3f} ms")
    print(f"  p10:    {p10:.3f} ms")
    print(f"  p90:    {p90:.3f} ms")
    print(f"  min:    {min(samples):.3f} ms")
    print(f"  max:    {max(samples):.3f} ms")

    # Compare to current GPU cost
    print(f"\n=== Comparison ===")
    print(f"  Current GPU draft_lmhead: ~19.5 ms/cycle")
    print(f"  ANE lm_head test:         {mean:.2f} ms/cycle")
    if mean < 19.5:
        print(f"  --> ANE is {19.5/mean:.2f}x faster. Pursue integration.")
    else:
        print(f"  --> ANE is {mean/19.5:.2f}x slower. Stick with GPU.")

    # Also run anemll-profile for deeper info
    print(f"\n[profile] running anemll-profile...")
    result = subprocess.run(
        [str(Path.home() / "projects/anemll-profile/anemll-profile"), str(mlmodelc)],
        capture_output=True, text=True,
    )
    # Show the key sections
    out = result.stdout
    for section in ["Model size:", "ANE ops:", "Measured:", "ANE graph interruptions:",
                     "CPU ops:"]:
        for line in out.splitlines():
            if section in line:
                print(f"  {line.strip()}")
                break


if __name__ == "__main__":
    main()
