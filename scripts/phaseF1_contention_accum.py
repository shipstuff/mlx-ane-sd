"""Phase F.1 contention (accumulating-cache variant — 100% ANE)."""
from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))


def background_load_worker(model_name, stop_event, ready_event):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    print(f"[bg] loading {model_name}...", flush=True)
    model, tok = load(model_name)
    sampler = make_sampler(temp=0.0)
    ready_event.set()
    print("[bg] ready", flush=True)
    prompts = ["Explain neural networks.", "Write a haiku.", "The best language is"]
    total = 0
    t0 = time.perf_counter()
    while not stop_event.is_set():
        for p in prompts:
            for resp in stream_generate(model, tok, p, max_tokens=80, sampler=sampler):
                if stop_event.is_set(): break
                total += 1
            if stop_event.is_set(): break
    elapsed = time.perf_counter() - t0
    print(f"[bg] {total} tok in {elapsed:.1f}s ({total/elapsed:.1f} tok/s)", flush=True)


PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("math", "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:"),
    ("story", "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "parallel"], default="solo")
    ap.add_argument("--mlmodelc",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--bg-model", default="mlx-community/gemma-3-270m-it-bf16")
    args = ap.parse_args()

    bg_proc = None
    stop_event = mp.Event()
    ready_event = mp.Event()
    if args.mode == "parallel":
        print(f"[main] bg: {args.bg_model}")
        bg_proc = mp.Process(target=background_load_worker,
                              args=(args.bg_model, stop_event, ready_event),
                              daemon=True)
        bg_proc.start()
        if not ready_event.wait(timeout=120):
            sys.exit("bg load failed")
        time.sleep(3)

    try:
        from huggingface_hub import snapshot_download
        from dflash_torch import DFlashConfig
        from phaseF1_ane_stream_accum import DFlashANEAccumDraft, stream_generate_ane_accum
        from mlx_lm import load as mlx_load

        draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
        config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
        print("[load] target...")
        target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
        print("[load] draft (ANE accum)...")
        draft = DFlashANEAccumDraft(args.mlmodelc, config, state_length=args.state_length)
        print("[warmup]...")
        list(stream_generate_ane_accum(target, draft, tok, "The weather", 20))

        rows = []
        for name, prompt in PROMPTS:
            print(f"\n=== {name} ({args.mode}) ===")
            gen, t, accepted, cycles, _ = stream_generate_ane_accum(
                target, draft, tok, prompt, args.max_new)
            tps = len(gen) / t
            print(f"  {len(gen)} tok in {t:.2f}s = {tps:.2f} tok/s (cycles={cycles})")
            rows.append({"name": name, "tps": tps})

        print("\n=== Summary ===")
        tpss = [r['tps'] for r in rows]
        print(f"mean: {statistics.mean(tpss):.2f} tok/s (min {min(tpss):.2f}, max {max(tpss):.2f})")
    finally:
        if bg_proc is not None:
            stop_event.set()
            bg_proc.join(timeout=10)
            if bg_proc.is_alive(): bg_proc.terminate()


if __name__ == "__main__":
    main()
