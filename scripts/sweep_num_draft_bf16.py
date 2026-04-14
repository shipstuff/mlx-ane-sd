"""Gemma-3-12B bf16 + 270M bf16 draft — SD test with num_draft sweep.

At bf16 the target is memory-bandwidth-bound, where SD has its largest headroom.
dflash-mlx reported 4.6× at bf16 (vs 1.4× at 4bit) for Qwen3-4B — we test
whether that pattern holds for Gemma-3-12B on our M4 Pro.
"""
import time
import statistics
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

TARGET = "mlx-community/gemma-3-12b-it-bf16"
DRAFT = "mlx-community/gemma-3-270m-it-bf16"

PROMPTS = [
    "The capital of France is",
    "1 + 1 equals",
    "def fibonacci(n):",
    "Write a Python function that reverses a string.",
    "To make a good cup of coffee, first",
    "In machine learning, gradient descent is",
]

MAX_TOKENS = 80


def bench(target_model, tokenizer, draft_model, prompts, num_draft):
    sampler = make_sampler(temp=0.0)
    res = []
    for prompt in prompts:
        t0 = time.perf_counter()
        base_tokens = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
            base_tokens.append(resp.token)
        t_base = time.perf_counter() - t0
        base_tps = len(base_tokens) / t_base

        t0 = time.perf_counter()
        sd_tokens = []
        from_draft_flags = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler,
                                     draft_model=draft_model, num_draft_tokens=num_draft):
            sd_tokens.append(resp.token)
            from_draft_flags.append(getattr(resp, 'from_draft', False))
        t_sd = time.perf_counter() - t0
        sd_tps = len(sd_tokens) / t_sd

        n_accept = sum(from_draft_flags)
        accept = n_accept / len(sd_tokens) if sd_tokens else 0
        speedup = t_base / t_sd
        res.append({
            "base_tps": base_tps, "sd_tps": sd_tps, "speedup": speedup, "accept": accept
        })
    return res


def main():
    print("Loading target (Gemma-3-12B bf16, ~24 GB)...", flush=True)
    target_model, tokenizer = load(TARGET)
    print("Loading draft (Gemma-3-270M bf16)...", flush=True)
    draft_model, _ = load(DRAFT)

    # Warmup
    print("Warmup...", flush=True)
    sampler = make_sampler(temp=0.0)
    for _ in stream_generate(target_model, tokenizer, "hello", max_tokens=5, sampler=sampler):
        pass
    for _ in stream_generate(target_model, tokenizer, "hello", max_tokens=5, sampler=sampler,
                              draft_model=draft_model, num_draft_tokens=4):
        pass
    print("Warmup done.\n", flush=True)

    all_results = []
    for num_draft in [2, 4, 6, 8]:
        print(f"=== num_draft_tokens={num_draft} ===", flush=True)
        prompt_results = bench(target_model, tokenizer, draft_model, PROMPTS, num_draft)
        speedups = [r["speedup"] for r in prompt_results]
        accepts = [r["accept"] for r in prompt_results]
        base_tps = statistics.mean([r["base_tps"] for r in prompt_results])
        sd_tps = statistics.mean([r["sd_tps"] for r in prompt_results])
        mean_speedup = statistics.mean(speedups)
        max_speedup = max(speedups)
        mean_accept = statistics.mean(accepts)
        all_results.append({
            "num_draft": num_draft, "base_tps": base_tps, "sd_tps": sd_tps,
            "mean_speedup": mean_speedup, "max_speedup": max_speedup, "mean_accept": mean_accept,
        })
        print(f"  base: {base_tps:.2f} tok/s | sd: {sd_tps:.2f} tok/s | "
              f"speedup: mean={mean_speedup:.2f}× max={max_speedup:.2f}× | accept: {mean_accept:.1%}\n",
              flush=True)

    print("=" * 72)
    print(f"{'num_draft':>10s}  {'base tok/s':>11s}  {'sd tok/s':>10s}  {'mean sp':>8s}  {'max sp':>7s}  {'accept':>7s}")
    print("-" * 72)
    for r in all_results:
        print(f"{r['num_draft']:>10d}  {r['base_tps']:>11.2f}  {r['sd_tps']:>10.2f}  "
              f"{r['mean_speedup']:>7.2f}×  {r['max_speedup']:>6.2f}×  {r['mean_accept']:>6.1%}")
    print("=" * 72)


if __name__ == "__main__":
    main()
