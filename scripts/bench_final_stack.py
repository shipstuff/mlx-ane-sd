"""Final 2c bench: full-ANE stack vs MLX bf16 baseline vs Swift-F.1 parity.

Runs each config on the 4 standard prompts at max_new=100 and reports:
  - Mean tok/s (decode-only)
  - Per-prompt tok/s, cycles, accept/cycle
  - Per-phase timing (where available)
  - Text sample (first 100 chars)

Configs:
  0. MLX bf16 (no SD) — reference
  1. Swift dflash-sd (fp16 draft + GPU lm_head) — parity with Python F.1
  2. + ANE LUT6 lm_head (draft side)
  3. + LUT6 draft body
  4. + K=18 partial target on ANE (half target, byte-identical)
  5. + chunked full target (2 x K=18, per_tensor LUT6)
  6. + chunked full target (per_grouped_channel LUT6, current best)
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
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


def run_mlx_baseline(max_new: int) -> list[dict]:
    """MLX bf16 baseline decode-only tok/s (excludes prefill)."""
    print("[bench] MLX bf16 baseline (decode-only)...", flush=True)
    from mlx_lm import load, stream_generate
    model, tok = load("mlx-community/Qwen3-4B-bf16")
    # warmup
    for _ in stream_generate(model, tok, prompt="Hello", max_tokens=5):
        pass

    rows = []
    for name, prompt in PROMPTS:
        # Time only the decode phase by skipping the first sample (includes prefill).
        responses = []
        for resp in stream_generate(model, tok, prompt=prompt, max_tokens=max_new):
            responses.append(resp)
        # stream_generate yields Response objects with .token, .text, etc.
        # Decode time = time from first token to last token.
        if len(responses) >= 2:
            # Use the last response's stats if available
            last = responses[-1]
            tps = getattr(last, "generation_tps", None)
            if tps is None:
                # Fallback: compute from timings
                tps = 0
            text = "".join(r.text for r in responses)[:100]
        else:
            tps = 0
            text = ""
        rows.append({
            "impl": "MLX_bf16_baseline",
            "name": name,
            "tokens": len(responses),
            "cycles": len(responses),
            "decode_ms": (len(responses) / tps * 1000) if tps else 0,
            "tok_per_s_decode": tps,
            "avg_accepted_per_cycle": 1.0,
            "text": text,
        })
        print(f"  {name}: {len(responses)} tok = {tps:.2f} tok/s (decode only)")
    return rows


def run_swift(label: str, max_new: int, base_args: list[str]) -> list[dict]:
    print(f"[bench] {label}...", flush=True)
    subprocess.run(base_args + ["--max-new", "10", "--prompt", "Hello", "--json"],
                   check=True, capture_output=True)

    rows = []
    for name, prompt in PROMPTS:
        r = subprocess.run(
            base_args + ["--max-new", str(max_new), "--prompt", prompt, "--json"],
            check=True, capture_output=True, text=True,
        )
        d = json.loads(r.stdout.strip())
        d["impl"] = label
        d["name"] = name
        rows.append(d)
        print(f"  {name}: {d['tokens']} tok, {d['tok_per_s_decode']:.2f} tok/s, "
              f"acc/cyc={d['avg_accepted_per_cycle']:.2f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--out", default=str(REPO / "notes" / "bench_final_stack.json"))
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    DRAFT_LUT6 = "/tmp/dflash_ane_accum_lut6_c/dflash_ane_accum_lut6.mlmodelc"
    DRAFT_FP16 = "/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc"
    LMHEAD_15 = "/tmp/lmhead_qwen3/lmhead_lut6.mlmodelc"
    LMHEAD_16 = "/tmp/lmhead_qwen3/bs16/lmhead_lut6.mlmodelc"
    CHUNK1_PT = "/tmp/qwen3_klayers_cap/K18/qwen3_K18_lut6.mlmodelc"
    CHUNK2_PT = "/tmp/qwen3_klayers_cap/K18_s18/qwen3_K18_lut6.mlmodelc"
    CHUNK1_PGC = "/tmp/qwen3_klayers_cap_pgc/K18/qwen3_K18_lut6.mlmodelc"
    CHUNK2_PGC = "/tmp/qwen3_klayers_cap_pgc/K18_s18/qwen3_K18_lut6.mlmodelc"

    all_rows = []

    if not args.skip_baseline:
        all_rows.extend(run_mlx_baseline(args.max_new))

    # Swift dflash-sd plain (matches Python F.1)
    all_rows.extend(run_swift("Swift_plain", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_FP16]))

    # + ANE LUT6 lm_head
    all_rows.extend(run_swift("+ane_lmhead", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_FP16,
                                "--ane-lmhead", LMHEAD_15]))

    # + LUT6 draft body
    all_rows.extend(run_swift("+lut6_draft", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_LUT6,
                                "--ane-lmhead", LMHEAD_15]))

    # + K=18 partial target (per_tensor LUT6, byte-identical)
    all_rows.extend(run_swift("+K18_partial", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_LUT6,
                                "--ane-lmhead", LMHEAD_15,
                                "--ane-target-layers", CHUNK1_PT,
                                "--ane-target-k", "18", "--ane-target-captures", "1,9,17"]))

    # + chunked full target (per_tensor LUT6, no ANE target lmhead)
    all_rows.extend(run_swift("+chunked_pt", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_LUT6,
                                "--ane-lmhead", LMHEAD_15,
                                "--ane-target-layers", CHUNK1_PT,
                                "--ane-target-k", "18", "--ane-target-captures", "1,9,17",
                                "--ane-target-layers2", CHUNK2_PT,
                                "--ane-target-k2", "18", "--ane-target-captures2", "7,15"]))

    # + chunked full target + ANE target lm_head (per_tensor)
    all_rows.extend(run_swift("+chunked_pt_ane_lmhead", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_LUT6,
                                "--ane-lmhead", LMHEAD_15,
                                "--ane-target-layers", CHUNK1_PT,
                                "--ane-target-k", "18", "--ane-target-captures", "1,9,17",
                                "--ane-target-layers2", CHUNK2_PT,
                                "--ane-target-k2", "18", "--ane-target-captures2", "7,15",
                                "--ane-target-lmhead", LMHEAD_16]))

    # + chunked full target (pgc LUT6) + ANE target lm_head — CURRENT BEST
    all_rows.extend(run_swift("+chunked_pgc_ane_lmhead (best)", args.max_new,
                               [str(SWIFT_BIN), "--draft", DRAFT_LUT6,
                                "--ane-lmhead", LMHEAD_15,
                                "--ane-target-layers", CHUNK1_PGC,
                                "--ane-target-k", "18", "--ane-target-captures", "1,9,17",
                                "--ane-target-layers2", CHUNK2_PGC,
                                "--ane-target-k2", "18", "--ane-target-captures2", "7,15",
                                "--ane-target-lmhead", LMHEAD_16]))

    # Table
    print("\n" + "=" * 110)
    print(f"{'impl':<32} {'capital':>10} {'fibonacci':>10} {'math':>10} {'story':>10} {'MEAN':>10}")
    print("-" * 110)

    impls = []
    for r in all_rows:
        if r["impl"] not in impls:
            impls.append(r["impl"])
    for impl in impls:
        subset = [r for r in all_rows if r["impl"] == impl]
        by_name = {r["name"]: r for r in subset}
        tpss = [r["tok_per_s_decode"] for r in subset]
        mean = statistics.mean(tpss) if tpss else 0
        cells = [f"{by_name[p][0] if isinstance(by_name.get(p, ''), str) else by_name[p]['tok_per_s_decode']:.2f}"
                 if p in by_name else "—"
                 for p in ["capital", "fibonacci", "math", "story"]]
        print(f"{impl:<32} " + " ".join(f"{c:>10}" for c in cells)
              + f" {mean:>10.2f}")
    print("=" * 110)

    Path(args.out).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out, "w") as f:
        json.dump(all_rows, f, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
