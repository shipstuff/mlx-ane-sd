"""End-to-end benchmark: Swift dflash-sd with ANE lm_head vs MLX-GPU baseline.

Runs both configurations on the 4 standard prompts at max_new=100. Verifies
text output matches (correctness check), then reports tok/s comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
from pathlib import Path

REPO = Path(__file__).parent.parent
SWIFT_BIN = REPO / "swift-bench" / ".build" / "release" / "dflash-sd"

PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci",
     "def fibonacci(n):\n    if n <= 1:\n        return n\n"
     "    return fibonacci(n-1) + fibonacci(n-2)\n\n"
     "# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("math",
     "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. "
     "Now solve 3y - 7 = 20:"),
    ("story",
     "Once upon a time in a small village nestled between two mountains, there "
     "lived a young girl named Elara who"),
]


def run_swift(prompt: str, max_new: int, draft: str, ane_lmhead: str = "") -> dict:
    cmd = [str(SWIFT_BIN), "--prompt", prompt, "--max-new", str(max_new),
           "--draft", draft, "--json"]
    if ane_lmhead:
        cmd += ["--ane-lmhead", ane_lmhead]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return json.loads(result.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--draft",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc")
    ap.add_argument("--ane-lmhead",
                    default="/tmp/lmhead_qwen3/lmhead_lut6.mlmodelc")
    ap.add_argument("--out", default=str(REPO / "notes" / "bench_ane_lmhead.json"))
    args = ap.parse_args()

    if not Path(args.ane_lmhead).exists():
        raise SystemExit(f"ANE lm_head model not found: {args.ane_lmhead}\n"
                          f"Run: python scripts/export_qwen3_lmhead_ane.py")

    rows = []

    # Warmup both stacks
    print("[warmup] gpu...", flush=True)
    run_swift("Hello world", 10, args.draft)
    print("[warmup] ane...", flush=True)
    run_swift("Hello world", 10, args.draft, args.ane_lmhead)

    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===", flush=True)
        gpu = run_swift(prompt, args.max_new, args.draft)
        gpu["impl"] = "gpu_lmhead"
        gpu["name"] = name
        ane = run_swift(prompt, args.max_new, args.draft, args.ane_lmhead)
        ane["impl"] = "ane_lmhead"
        ane["name"] = name

        # Correctness: text agreement (first 80 chars to allow EOS truncation diffs)
        text_match = gpu["text"][:80] == ane["text"][:80]
        cycles_match = gpu["cycles"] == ane["cycles"]
        accept_close = abs(gpu["avg_accepted_per_cycle"] -
                           ane["avg_accepted_per_cycle"]) < 0.1

        print(f"  gpu: {gpu['tok_per_s_decode']:.2f} tok/s, "
              f"cyc={gpu['cycles']}, acc/cyc={gpu['avg_accepted_per_cycle']:.2f}")
        print(f"  ane: {ane['tok_per_s_decode']:.2f} tok/s, "
              f"cyc={ane['cycles']}, acc/cyc={ane['avg_accepted_per_cycle']:.2f}")
        print(f"  speedup: {ane['tok_per_s_decode']/gpu['tok_per_s_decode']:.3f}x")
        print(f"  correctness: text_match={text_match} cycles_match={cycles_match} "
              f"accept_close={accept_close}")
        if not text_match:
            print(f"    GPU first80: {gpu['text'][:80]!r}")
            print(f"    ANE first80: {ane['text'][:80]!r}")

        # Phase comparison
        gpu_phases = gpu["phases"]
        ane_phases = ane["phases"]
        print(f"  per-cycle: target_verify gpu={gpu_phases['target_verify']['meanMs']:.1f}ms "
              f"ane={ane_phases['target_verify']['meanMs']:.1f}ms")
        print(f"  per-cycle: draft_lmhead gpu={gpu_phases['draft_lmhead']['meanMs']:.2f}ms "
              f"ane={ane_phases['draft_lmhead']['meanMs']:.2f}ms")
        print(f"  per-cycle: draft_predict gpu={gpu_phases['draft_predict']['meanMs']:.1f}ms "
              f"ane={ane_phases['draft_predict']['meanMs']:.1f}ms")

        rows.extend([gpu, ane])

    print()
    print("=" * 90)
    print(f"{'prompt':<12} {'impl':<12} {'tok/s':>8} {'cyc':>4} {'acc/cyc':>8} "
          f"{'tv ms':>7} {'dlh ms':>7}")
    print("-" * 90)
    for r in rows:
        p = r["phases"]
        print(f"{r['name']:<12} {r['impl']:<12} {r['tok_per_s_decode']:>8.2f} "
              f"{r['cycles']:>4} {r['avg_accepted_per_cycle']:>8.2f} "
              f"{p['target_verify']['meanMs']:>7.1f} "
              f"{p['draft_lmhead']['meanMs']:>7.2f}")
    print("-" * 90)
    for impl in sorted(set(r["impl"] for r in rows)):
        subset = [r for r in rows if r["impl"] == impl]
        m = statistics.mean(r["tok_per_s_decode"] for r in subset)
        print(f"{'MEAN':<12} {impl:<12} {m:>8.2f}")
    print("=" * 90)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
