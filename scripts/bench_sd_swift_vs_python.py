"""Side-by-side Swift dflash-sd vs Python F.1 end-to-end SD benchmark.

Runs both implementations on the same 4 standard prompts at max_new=100 and
reports tok/s (decode-only), accept rate, and per-phase timing breakdown.

Swift dflash-sd emits --json output. Python F.1 is timed in-process.

Run:
    python scripts/bench_sd_swift_vs_python.py
    python scripts/bench_sd_swift_vs_python.py --max-new 100 --warmup 10
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
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


def run_swift(prompt: str, max_new: int, state_length: int,
              mlmodelc: str) -> dict:
    result = subprocess.run(
        [str(SWIFT_BIN),
         "--prompt", prompt,
         "--max-new", str(max_new),
         "--state-length", str(state_length),
         "--draft", mlmodelc,
         "--json"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip())


def run_python(target, tok, draft, prompt: str, max_new: int) -> dict:
    # Import here so Swift-only runs don't pay the import cost.
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, "/tmp/dflash")
    import phaseF1_ane_stream_accum as F

    t0 = time.perf_counter()
    gen, t_decode, accepted, cycles, t_prefill = F.stream_generate_ane_accum(
        target, draft, tok, prompt, max_new
    )
    t_total = time.perf_counter() - t0
    text = tok.decode(gen)
    return {
        "tokens": len(gen),
        "cycles": cycles,
        "accepted_total": accepted,
        "avg_accepted_per_cycle": accepted / max(cycles, 1),
        "decode_ms": t_decode * 1000,
        "total_wall_ms": t_total * 1000,
        "tok_per_s_decode": len(gen) / t_decode if t_decode > 0 else 0,
        "tok_per_s_wall": len(gen) / t_total if t_total > 0 else 0,
        "prefill_ms": t_prefill * 1000,
        "text": text,
    }


def load_python_env(draft_mlmodelc: str, state_length: int):
    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, "/tmp/dflash")
    from huggingface_hub import snapshot_download
    import mlx_lm
    from dflash_torch import DFlashConfig
    import phaseF1_ane_stream_accum as F

    print("[py] loading draft config...", flush=True)
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print("[py] loading target Qwen3-4B-bf16...", flush=True)
    target, tok = mlx_lm.load("mlx-community/Qwen3-4B-bf16")
    print("[py] loading draft ANE model...", flush=True)
    draft = F.DFlashANEAccumDraft(draft_mlmodelc, config, state_length=state_length)
    return target, tok, draft


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--mlmodelc",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc")
    ap.add_argument("--warmup-tokens", type=int, default=20,
                    help="per-stack warmup run to amortize first-call compile")
    ap.add_argument("--swift-only", action="store_true")
    ap.add_argument("--python-only", action="store_true")
    ap.add_argument("--out", default=str(REPO / "notes" / "bench_swift_vs_python_f1.json"))
    args = ap.parse_args()

    rows = []

    if not args.swift_only:
        target, tok, draft = load_python_env(args.mlmodelc, args.state_length)
        print(f"[py] warmup ({args.warmup_tokens} tok)...", flush=True)
        run_python(target, tok, draft, "Hello world", args.warmup_tokens)
        for name, prompt in PROMPTS:
            print(f"[py] {name}...", flush=True)
            r = run_python(target, tok, draft, prompt, args.max_new)
            r["name"] = name
            r["impl"] = "python_f1"
            rows.append(r)
            print(f"   tokens={r['tokens']} decode={r['decode_ms']:.0f}ms "
                  f"tps={r['tok_per_s_decode']:.2f} "
                  f"avg/cyc={r['avg_accepted_per_cycle']:.2f}")

    if not args.python_only:
        print(f"[sw] warmup ({args.warmup_tokens} tok)...", flush=True)
        run_swift("Hello world", args.warmup_tokens, args.state_length, args.mlmodelc)
        for name, prompt in PROMPTS:
            print(f"[sw] {name}...", flush=True)
            r = run_swift(prompt, args.max_new, args.state_length, args.mlmodelc)
            r["name"] = name
            r["impl"] = "swift_dflash_sd"
            rows.append(r)
            print(f"   tokens={r['tokens']} decode={r['decode_ms']:.0f}ms "
                  f"tps={r['tok_per_s_decode']:.2f} "
                  f"avg/cyc={r['avg_accepted_per_cycle']:.2f}")

    # Print comparison table
    print()
    print("=" * 88)
    print(f"{'prompt':<12} {'impl':<18} {'tok':>4} {'cyc':>4} "
          f"{'dec ms':>8} {'tps':>7} {'avg/cyc':>8}")
    print("-" * 88)
    for r in rows:
        print(f"{r['name']:<12} {r['impl']:<18} "
              f"{r['tokens']:>4} {r['cycles']:>4} "
              f"{r['decode_ms']:>8.0f} {r['tok_per_s_decode']:>7.2f} "
              f"{r['avg_accepted_per_cycle']:>8.2f}")

    # Per-impl mean
    print("-" * 88)
    for impl in sorted(set(r["impl"] for r in rows)):
        subset = [r for r in rows if r["impl"] == impl]
        mean_tps = statistics.mean(r["tok_per_s_decode"] for r in subset)
        mean_acc = statistics.mean(r["avg_accepted_per_cycle"] for r in subset)
        print(f"{'MEAN':<12} {impl:<18} {'':>4} {'':>4} {'':>8} "
              f"{mean_tps:>7.2f} {mean_acc:>8.2f}")
    print("=" * 88)

    # Save JSON
    out_path = Path(args.out)
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
