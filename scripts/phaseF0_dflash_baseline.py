"""Phase F.0: reproduce z-lab DFlash GPU-only baseline on M4 Pro.

Uses z-lab's native MLX reference implementation (dflash/model_mlx.py)
with Qwen3-4B bf16 target and z-lab/Qwen3-4B-DFlash-b16 draft. This
establishes the GPU-only ground truth we'll try to beat with ANE offload
in F.1.

Measures:
- target-only throughput (no SD)
- DFlash throughput (GPU-only, target + draft both on MLX)
- acceptance rate (from how often draft blocks were fully accepted)

We also measure per-prompt variance since DFlash is known to swing
widely (math@4028 best case ~4x, prose worst case <1x per CLAUDE.md).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

# Expose z-lab's dflash package (cloned at /tmp/dflash)
sys.path.insert(0, "/tmp/dflash")

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

from dflash.model_mlx import load_draft, stream_generate as dflash_stream


PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("quantum", "Explain the Heisenberg uncertainty principle. "),
    ("recipe", "The recipe called for two cups of flour, one teaspoon of baking soda, and a pinch of salt."),
    ("story", "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who"),
    ("math", "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:"),
]


def bench_target_only(model, tok, prompt, max_new):
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    tokens = []
    for resp in stream_generate(model, tok, prompt, max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def bench_dflash(model, draft, tok, prompt, max_new):
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    tokens = []
    accepted_total = 0
    cycles = 0
    for resp in dflash_stream(model, draft, tok, prompt,
                               max_tokens=max_new, sampler=sampler):
        tokens.extend(resp.tokens)
        accepted_total += resp.accepted
        cycles += 1
    elapsed = time.perf_counter() - t0
    return tokens, elapsed, accepted_total, cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/Qwen3-4B-bf16")
    ap.add_argument("--draft", default="z-lab/Qwen3-4B-DFlash-b16")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--skip-target-only", action="store_true")
    args = ap.parse_args()

    print(f"[load] target {args.target}...")
    model, tok = load(args.target)
    print(f"[load] draft  {args.draft}...")
    draft = load_draft(args.draft)

    print(f"\n[info] target: {type(model).__name__}, "
          f"n_layers={len(model.model.layers) if hasattr(model,'model') else '?'}")
    print(f"[info] draft: {type(draft).__name__}, "
          f"n_layers={draft.config.num_hidden_layers}, "
          f"block={draft.config.block_size}, "
          f"target_layer_ids={draft.config.target_layer_ids}")

    # Warmup once
    print("\n[warmup]...")
    list(bench_dflash(model, draft, tok, "The weather is", 20))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        print(f"prompt: {prompt[:70]!r}...")

        # target-only
        to_tps = None
        to_tok_count = None
        if not args.skip_target_only:
            to_tokens, to_elapsed = bench_target_only(model, tok, prompt, args.max_new)
            to_tok_count = len(to_tokens)
            to_tps = to_tok_count / to_elapsed
            print(f"  target-only: {to_tok_count} tok in {to_elapsed:.2f}s = {to_tps:.2f} tok/s")

        # DFlash
        df_tokens, df_elapsed, accepted, cycles = bench_dflash(model, draft, tok, prompt, args.max_new)
        df_tps = len(df_tokens) / df_elapsed
        accept_rate = accepted / (cycles * (draft.config.block_size - 1)) if cycles else 0
        print(f"  dflash:      {len(df_tokens)} tok in {df_elapsed:.2f}s = {df_tps:.2f} tok/s  "
              f"(cycles={cycles}, accepted={accepted}, accept_rate={accept_rate:.0%})")
        if to_tps:
            print(f"  speedup:     {df_tps/to_tps:.2f}x")

        rows.append({
            "name": name,
            "target_tps": to_tps,
            "dflash_tps": df_tps,
            "speedup": df_tps / to_tps if to_tps else None,
            "accept_rate": accept_rate,
            "cycles": cycles,
        })

    print("\n=== Summary ===")
    print(f"{'prompt':<12} {'target':>8} {'dflash':>8} {'speedup':>8} {'accept':>8}")
    for r in rows:
        ts = f"{r['target_tps']:.2f}" if r['target_tps'] else "-"
        sp = f"{r['speedup']:.2f}x" if r['speedup'] else "-"
        print(f"{r['name']:<12} {ts:>8} {r['dflash_tps']:>8.2f} {sp:>8} {r['accept_rate']:>7.0%}")

    if not args.skip_target_only:
        tgt = [r['target_tps'] for r in rows if r['target_tps']]
        df = [r['dflash_tps'] for r in rows]
        sp = [r['speedup'] for r in rows if r['speedup']]
        print(f"\nmean target: {statistics.mean(tgt):.2f} tok/s")
        print(f"mean dflash: {statistics.mean(df):.2f} tok/s")
        print(f"mean speedup: {statistics.mean(sp):.2f}x (min {min(sp):.2f}, max {max(sp):.2f})")


if __name__ == "__main__":
    main()
