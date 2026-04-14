"""Phase F.0 contention: DFlash GPU-only under background GPU workload.

Same shape as Phase C (phaseC_parallel_workload.py) but for the Qwen3-4B
DFlash pair. Measures how much DFlash degrades when a secondary workload
contends for the Metal queue. This is the baseline F.1 will try to beat
by moving the draft to the ANE.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time

sys.path.insert(0, "/tmp/dflash")


def background_load_worker(model_name: str, stop_event, ready_event):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"[bg] loading {model_name}...", flush=True)
    model, tok = load(model_name)
    sampler = make_sampler(temp=0.0)
    ready_event.set()
    print(f"[bg] ready, generating continuously", flush=True)

    prompts = [
        "Explain neural networks in one paragraph:",
        "Write a haiku about the sea:",
        "The best programming language is",
    ]
    total = 0
    t0 = time.perf_counter()
    while not stop_event.is_set():
        for p in prompts:
            for resp in stream_generate(model, tok, p, max_tokens=80, sampler=sampler):
                if stop_event.is_set():
                    break
                total += 1
            if stop_event.is_set():
                break
    elapsed = time.perf_counter() - t0
    print(f"[bg] generated {total} tokens in {elapsed:.1f}s ({total/elapsed:.2f} tok/s)",
          flush=True)


def bench_target_only_each(target, tok, prompt, max_new):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    tokens = []
    for resp in stream_generate(target, tok, prompt, max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def bench_dflash_each(target, draft, tok, prompt, max_new):
    from mlx_lm.sample_utils import make_sampler
    from dflash.model_mlx import stream_generate as dflash_stream

    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    tokens = []
    accepted_total = 0
    cycles = 0
    for resp in dflash_stream(target, draft, tok, prompt,
                               max_tokens=max_new, sampler=sampler):
        tokens.extend(resp.tokens)
        accepted_total += resp.accepted
        cycles += 1
    elapsed = time.perf_counter() - t0
    return tokens, elapsed, accepted_total, cycles


PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("math", "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:"),
    ("story", "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "parallel"], default="solo")
    ap.add_argument("--approach", choices=["target", "dflash"], required=True)
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--bg-model", default="mlx-community/gemma-3-270m-it-bf16",
                    help="small MLX model as background GPU load (tiny so contention is moderate)")
    args = ap.parse_args()

    bg_proc = None
    stop_event = mp.Event()
    ready_event = mp.Event()

    if args.mode == "parallel":
        print(f"[main] starting background workload: {args.bg_model}")
        bg_proc = mp.Process(target=background_load_worker,
                              args=(args.bg_model, stop_event, ready_event),
                              daemon=True)
        bg_proc.start()
        if not ready_event.wait(timeout=120):
            print("[main] background failed to load in 120s")
            sys.exit(1)
        time.sleep(3)

    try:
        # Load once, reuse across prompts
        from mlx_lm import load
        target, tok = load("mlx-community/Qwen3-4B-bf16")
        draft = None
        if args.approach == "dflash":
            from dflash.model_mlx import load_draft, stream_generate as dflash_stream
            from mlx_lm.sample_utils import make_sampler
            draft = load_draft("z-lab/Qwen3-4B-DFlash-b16")
            sampler = make_sampler(temp=0.0)
            # Warmup
            print("[main] warmup...")
            list(dflash_stream(target, draft, tok, "The weather is",
                                max_tokens=20, sampler=sampler))

        rows = []
        for name, prompt in PROMPTS:
            print(f"\n=== {name} ({args.mode}, {args.approach}) ===")
            if args.approach == "target":
                toks, elapsed = bench_target_only_each(target, tok, prompt, args.max_new)
                tps = len(toks) / elapsed
                print(f"  {len(toks)} tok in {elapsed:.2f}s = {tps:.2f} tok/s")
                rows.append({"name": name, "tps": tps, "tokens": len(toks)})
            else:
                toks, elapsed, accepted, cycles = bench_dflash_each(target, draft, tok,
                                                                    prompt, args.max_new)
                tps = len(toks) / elapsed
                print(f"  {len(toks)} tok in {elapsed:.2f}s = {tps:.2f} tok/s  "
                      f"(cycles={cycles}, accepted={accepted})")
                rows.append({"name": name, "tps": tps, "tokens": len(toks),
                             "cycles": cycles, "accepted": accepted})

        print("\n=== Summary ===")
        tpss = [r['tps'] for r in rows]
        print(f"mean tok/s: {statistics.mean(tpss):.2f}  "
              f"(min {min(tpss):.2f}, max {max(tpss):.2f})")

    finally:
        if bg_proc is not None:
            stop_event.set()
            bg_proc.join(timeout=10)
            if bg_proc.is_alive():
                bg_proc.terminate()


if __name__ == "__main__":
    main()
