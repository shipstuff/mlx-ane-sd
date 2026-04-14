"""Phase C: heterogeneous SD while another workload competes for GPU.

Hypothesis: when a second workload is loading the GPU, pure-MLX SD's
draft competes for Metal queue slots and its throughput drops. Our
heterogeneous setup (draft on ANE) keeps the draft off the GPU entirely,
so the draft portion doesn't contend.

Setup:
    Background load: second MLX model continuously decoding on GPU
    Foreground SD:   either pure-MLX SD (target + draft on GPU)
                     or heterogeneous (target on GPU, draft on ANE)

We measure foreground SD throughput under contention vs solo.

This tests the Exp E thesis applied to SD.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import signal
import sys
import time
from pathlib import Path

# Queue message format: (time_ns, event_type, payload)


def background_load_worker(model_name: str, stop_event, ready_event):
    """Runs in a separate process. Continuously generates tokens using an
    MLX model to put constant load on the GPU."""
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
            tokens = []
            for resp in stream_generate(model, tok, p, max_tokens=50, sampler=sampler):
                if stop_event.is_set():
                    break
                tokens.append(resp.token)
            total += len(tokens)
            if stop_event.is_set():
                break
    elapsed = time.perf_counter() - t0
    print(f"[bg] generated {total} tokens in {elapsed:.1f}s ({total/elapsed:.2f} tok/s)",
          flush=True)


def run_pure_mlx_sd(prompt_ids, num_draft, max_new):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    target, tok = load("mlx-community/gemma-3-12b-it-bf16")
    draft, _ = load("mlx-community/gemma-3-270m-it-bf16")
    sampler = make_sampler(temp=0.0)
    prompt = tok.decode(prompt_ids)
    t0 = time.perf_counter()
    tokens = []
    from_draft_count = 0
    for resp in stream_generate(target, tok, prompt, max_tokens=max_new,
                                 draft_model=draft, num_draft_tokens=num_draft,
                                 sampler=sampler):
        tokens.append(resp.token)
        if getattr(resp, 'from_draft', False):
            from_draft_count += 1
    elapsed = time.perf_counter() - t0
    return tokens, elapsed, from_draft_count


def run_hetero_sd(prompt_ids, num_draft, max_new):
    sys.path.insert(0, '/Users/carl/projects/mlx-ane-sd/scripts')
    import phaseB_sequential_optimized as pb
    target = pb.MLXTarget("mlx-community/gemma-3-12b-it-bf16")
    draft = pb.ANEDraft()
    t0 = time.perf_counter()
    gen, stats = pb.run_sd(draft, target, prompt_ids, max_new, num_draft, verbose=False)
    elapsed = time.perf_counter() - t0
    return gen, elapsed, stats


def run_target_only(prompt_ids, max_new):
    """Target-only decode (no SD) — baseline for contention comparison."""
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    target, tok = load("mlx-community/gemma-3-12b-it-bf16")
    sampler = make_sampler(temp=0.0)
    prompt = tok.decode(prompt_ids)
    t0 = time.perf_counter()
    tokens = []
    for resp in stream_generate(target, tok, prompt, max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "parallel"], default="solo",
                    help="solo = no background load; parallel = 0.8B MLX model loads GPU")
    ap.add_argument("--approach", choices=["mlx", "hetero", "baseline"], required=True,
                    help="mlx = pure-MLX SD; hetero = ANE draft + MLX target; baseline = target-only, no SD")
    ap.add_argument("--num-draft", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=100)
    ap.add_argument("--bg-model", default="mlx-community/gemma-3-270m-it-bf16",
                    help="background workload model (kept tiny so GPU contention is moderate)")
    args = ap.parse_args()

    # Background load (if parallel mode)
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
            print("[main] background model failed to load in 120s")
            sys.exit(1)
        time.sleep(3)  # let bg warm up

    # Prepare prompt tokens using our own tokenizer
    from mlx_lm import load as mlx_load
    # Load tokenizer only — full model will be loaded by the runner
    _, tok = mlx_load("mlx-community/gemma-3-12b-it-bf16")
    prompt = "The capital of France is"
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    del _  # release before running

    print(f"\n[main] running {args.approach} SD in {args.mode} mode, "
          f"num_draft={args.num_draft}, max_new={args.max_new_tokens}")

    if args.approach == "mlx":
        tokens, elapsed, from_draft = run_pure_mlx_sd(prompt_ids, args.num_draft, args.max_new_tokens)
        print(f"[main] tokens: {len(tokens)}, from_draft: {from_draft} ({from_draft/len(tokens)*100:.0f}%)")
    elif args.approach == "baseline":
        tokens, elapsed, _ = run_target_only(prompt_ids, args.max_new_tokens)
        print(f"[main] tokens: {len(tokens)} (target-only, no SD)")
    else:
        tokens, elapsed, stats = run_hetero_sd(prompt_ids, args.num_draft, args.max_new_tokens)
        print(f"[main] tokens: {len(tokens)}, stats: {stats['draft_accepted']}/{stats['draft_generated']} "
              f"= {stats['draft_accepted']/max(stats['draft_generated'],1):.0%} accept")

    print(f"[main] time: {elapsed*1000:.0f} ms  tok/s: {len(tokens)/elapsed:.2f}")

    # Stop background
    if bg_proc is not None:
        stop_event.set()
        bg_proc.join(timeout=10)
        if bg_proc.is_alive():
            bg_proc.terminate()


if __name__ == "__main__":
    main()
