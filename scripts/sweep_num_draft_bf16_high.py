"""Push num_draft higher on bf16 to find ceiling."""
import time, statistics
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

target_model, tokenizer = load("mlx-community/gemma-3-12b-it-bf16")
draft_model, _ = load("mlx-community/gemma-3-270m-it-bf16")

PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "Write a Python function that reverses a string.",
    "To make a good cup of coffee, first",
]
MAX = 80
sampler = make_sampler(temp=0.0)

# warmup
for _ in stream_generate(target_model, tokenizer, "hello", max_tokens=5, sampler=sampler, draft_model=draft_model, num_draft_tokens=8):
    pass

print(f"{'num_draft':>10}  {'sd tok/s':>9}  {'mean sp':>8}  {'max sp':>7}  {'accept':>7}")
print("-" * 55)
for num_draft in [8, 10, 12, 16, 20]:
    speedups = []
    accepts = []
    sd_tps_list = []
    for prompt in PROMPTS:
        # baseline
        t0 = time.perf_counter()
        base = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX, sampler=sampler):
            base.append(resp.token)
        t_base = time.perf_counter() - t0

        t0 = time.perf_counter()
        sd = []
        fd = []
        for resp in stream_generate(target_model, tokenizer, prompt, max_tokens=MAX, sampler=sampler,
                                     draft_model=draft_model, num_draft_tokens=num_draft):
            sd.append(resp.token)
            fd.append(getattr(resp, 'from_draft', False))
        t_sd = time.perf_counter() - t0
        sd_tps_list.append(len(sd)/t_sd)
        accepts.append(sum(fd) / len(sd))
        speedups.append(t_base / t_sd)

    print(f"{num_draft:>10d}  {statistics.mean(sd_tps_list):>9.2f}  "
          f"{statistics.mean(speedups):>7.2f}×  {max(speedups):>6.2f}×  {statistics.mean(accepts):>6.1%}")
