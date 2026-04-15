"""Multi-stream SD worker: one subprocess runs one SD stream end-to-end.

Supports three modes:
- baseline: target-only decoding (no draft)
- f0: DFlash GPU-only (target + draft both on MLX)
- f1: DFlash ANE-hosted draft (target on MLX, draft on ANE)

Reports wall-clock + tok/s on stdout. Parent process aggregates.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))


PROMPT_POOL = [
    "The capital of France is Paris, which is known for",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')",
    "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:",
    "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who",
    "The economic impact of climate change on coastal communities has been",
    "Write a Python function that computes the nth prime number:",
    "Explain how neural networks learn from data in one paragraph:",
    "Translate the following English phrase to French: 'The weather is nice today.'",
]


def run_baseline(target, tok, prompt, max_new):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    tokens = []
    t0 = time.perf_counter()
    for resp in stream_generate(target, tok, prompt, max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return len(tokens), elapsed


def run_f0(target, draft, tok, prompt, max_new):
    from mlx_lm.sample_utils import make_sampler
    from dflash.model_mlx import stream_generate as dflash_stream
    sampler = make_sampler(temp=0.0)
    tokens = []
    t0 = time.perf_counter()
    for resp in dflash_stream(target, draft, tok, prompt,
                               max_tokens=max_new, sampler=sampler):
        tokens.extend(resp.tokens)
    elapsed = time.perf_counter() - t0
    return len(tokens), elapsed


def run_f1(target, draft, tok, prompt, max_new):
    from phaseF1_ane_stream_accum import stream_generate_ane_accum
    gen, elapsed, accepted, cycles, _ = stream_generate_ane_accum(
        target, draft, tok, prompt, max_new
    )
    return len(gen), elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "f0", "f1"], required=True)
    ap.add_argument("--stream-id", type=int, required=True,
                    help="0-indexed stream id")
    ap.add_argument("--prompt-id", type=int, default=None,
                    help="If set, all streams use PROMPT_POOL[prompt_id]. "
                         "If unset, use stream_id%pool_size (different prompt per stream).")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--mlmodelc",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc",
                    help="F.1 only: path to compiled ANE DFlash draft")
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--warmup-prompt", default="The weather is")
    ap.add_argument("--warmup-tokens", type=int, default=20)
    ap.add_argument("--report-file", type=Path, default=None,
                    help="If set, write JSON result here (else stdout)")
    ap.add_argument("--start-barrier", type=Path, default=None,
                    help="If set, wait for this file to exist before starting (sync start)")
    args = ap.parse_args()

    # Pick a prompt
    if args.prompt_id is not None:
        prompt = PROMPT_POOL[args.prompt_id % len(PROMPT_POOL)]
    else:
        prompt = PROMPT_POOL[args.stream_id % len(PROMPT_POOL)]

    # Load target (always needed)
    print(f"[stream-{args.stream_id}/{args.mode}] loading target...", file=sys.stderr, flush=True)
    t0 = time.perf_counter()
    from mlx_lm import load as mlx_load
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    load_target_s = time.perf_counter() - t0
    print(f"[stream-{args.stream_id}] target loaded in {load_target_s:.1f}s", file=sys.stderr, flush=True)

    draft = None
    load_draft_s = 0.0
    if args.mode == "f0":
        t0 = time.perf_counter()
        from dflash.model_mlx import load_draft as dflash_load_draft
        draft = dflash_load_draft("z-lab/Qwen3-4B-DFlash-b16")
        draft.bind(target)
        load_draft_s = time.perf_counter() - t0
    elif args.mode == "f1":
        from huggingface_hub import snapshot_download
        from dflash_torch import DFlashConfig
        from phaseF1_ane_stream_accum import DFlashANEAccumDraft
        draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
        config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
        t0 = time.perf_counter()
        draft = DFlashANEAccumDraft(args.mlmodelc, config, state_length=args.state_length)
        load_draft_s = time.perf_counter() - t0

    # Warmup (single-run, not benchmarked)
    print(f"[stream-{args.stream_id}] warmup...", file=sys.stderr, flush=True)
    if args.mode == "baseline":
        run_baseline(target, tok, args.warmup_prompt, args.warmup_tokens)
    elif args.mode == "f0":
        run_f0(target, draft, tok, args.warmup_prompt, args.warmup_tokens)
    else:
        run_f1(target, draft, tok, args.warmup_prompt, args.warmup_tokens)

    # Sync start across processes via file barrier.
    # Post our "ready" file so the orchestrator knows we finished loading.
    if args.start_barrier is not None:
        ready_file = args.start_barrier.parent / f"ready_{args.mode}_{args.stream_id}"
        ready_file.touch()
        print(f"[stream-{args.stream_id}] ready, waiting for start barrier...",
              file=sys.stderr, flush=True)
        timeout = 600.0
        waited = 0.0
        while not args.start_barrier.exists() and waited < timeout:
            time.sleep(0.02)
            waited += 0.02
        if waited >= timeout:
            print(f"[stream-{args.stream_id}] barrier timeout",
                  file=sys.stderr, flush=True)
            sys.exit(1)

    # Actual benchmark run
    t_start = time.perf_counter()
    if args.mode == "baseline":
        n_tokens, elapsed = run_baseline(target, tok, prompt, args.max_new)
    elif args.mode == "f0":
        n_tokens, elapsed = run_f0(target, draft, tok, prompt, args.max_new)
    else:
        n_tokens, elapsed = run_f1(target, draft, tok, prompt, args.max_new)
    t_wall = time.perf_counter() - t_start

    result = {
        "stream_id": args.stream_id,
        "mode": args.mode,
        "prompt_head": prompt[:40],
        "tokens_generated": n_tokens,
        "elapsed_s": elapsed,
        "wall_s": t_wall,
        "tok_per_s": n_tokens / elapsed if elapsed > 0 else 0,
        "load_target_s": load_target_s,
        "load_draft_s": load_draft_s,
    }

    if args.report_file:
        args.report_file.parent.mkdir(parents=True, exist_ok=True)
        args.report_file.write_text(json.dumps(result, indent=2))
    else:
        print(json.dumps(result), flush=True)

    print(f"[stream-{args.stream_id}/{args.mode}] done: "
          f"{n_tokens} tok in {elapsed:.2f}s = {result['tok_per_s']:.2f} tok/s",
          file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
