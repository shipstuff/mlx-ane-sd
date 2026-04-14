"""Sweep num_draft_tokens to find optimal block size for Gemma-3 4bit SD."""
import time
import statistics
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

TARGET = "mlx-community/gemma-3-12b-it-4bit"
DRAFT = "mlx-community/gemma-3-270m-it-bf16"

# Short set covering a mix of acceptance profiles from Phase 0
PROMPTS = [
    "The capital of France is",                             # 76% accept at num=4
    "1 + 1 equals",                                         # 56% accept
    "def fibonacci(n):",                                    # 75%
    "Write a Python function that reverses a string.",      # 77%
    "To make a good cup of coffee, first",                  # 62%
]

MAX_TOKENS = 80


def bench(target_model, tokenizer, draft_model, prompts, num_draft):
    sampler = make_sampler(temp=0.0)
    speedups = []
    accepts = []
    sd_tps_list = []
    base_tps_list = []
    for prompt in prompts:
        t0 = time.perf_counter()
        base_tokens = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler):
            base_tokens.append(resp.token)
        t_base = time.perf_counter() - t0
        base_tps = len(base_tokens) / t_base
        base_tps_list.append(base_tps)

        t0 = time.perf_counter()
        sd_tokens = []
        from_draft_flags = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX_TOKENS, sampler=sampler,
                                     draft_model=draft_model, num_draft_tokens=num_draft):
            sd_tokens.append(resp.token)
            from_draft_flags.append(getattr(resp, 'from_draft', False))
        t_sd = time.perf_counter() - t0
        sd_tps = len(sd_tokens) / t_sd
        sd_tps_list.append(sd_tps)

        n_accept = sum(from_draft_flags)
        accept = n_accept / len(sd_tokens) if sd_tokens else 0
        speedup = t_base / t_sd
        accepts.append(accept)
        speedups.append(speedup)
    return {
        "num_draft": num_draft,
        "mean_speedup": statistics.mean(speedups),
        "median_speedup": statistics.median(speedups),
        "max_speedup": max(speedups),
        "min_speedup": min(speedups),
        "mean_accept": statistics.mean(accepts),
        "mean_base_tps": statistics.mean(base_tps_list),
        "mean_sd_tps": statistics.mean(sd_tps_list),
    }


def main():
    print("Loading target (Gemma-3-12B 4bit)...", flush=True)
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

    results = []
    for num_draft in [2, 3, 4, 6, 8]:
        print(f"=== num_draft_tokens={num_draft} ===", flush=True)
        r = bench(target_model, tokenizer, draft_model, PROMPTS, num_draft)
        results.append(r)
        print(f"  base: {r['mean_base_tps']:.1f} tok/s")
        print(f"  sd:   {r['mean_sd_tps']:.1f} tok/s")
        print(f"  speedup: mean={r['mean_speedup']:.2f}× median={r['median_speedup']:.2f}× "
              f"min={r['min_speedup']:.2f}× max={r['max_speedup']:.2f}×")
        print(f"  accept: {r['mean_accept']:.1%}\n", flush=True)

    print("=" * 72)
    print(f"{'num_draft':>10s}  {'base tok/s':>11s}  {'sd tok/s':>10s}  {'mean sp':>8s}  {'max sp':>7s}  {'accept':>7s}")
    print("-" * 72)
    for r in results:
        print(f"{r['num_draft']:>10d}  {r['mean_base_tps']:>11.1f}  {r['mean_sd_tps']:>10.1f}  "
              f"{r['mean_speedup']:>7.2f}×  {r['max_speedup']:>6.2f}×  {r['mean_accept']:>6.1%}")
    print("=" * 72)
    best = max(results, key=lambda r: r['mean_speedup'])
    print(f"\nBest num_draft: {best['num_draft']} → {best['mean_speedup']:.2f}× mean speedup")


if __name__ == "__main__":
    main()
