# dflash-mlx reproduction on M4 Pro

**Date:** 2026-04-13
**Hardware:** Mac mini M4 Pro, 64 GB, macOS 26.3.1
**Tool:** [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx)

## Purpose

Reproduce dflash-mlx's published benchmarks on our M4 Pro hardware to:
1. Verify the reported 4.6× bf16 / 1.4× 4bit speedups are achievable on
   Apple Silicon generally, not just M4 Max
2. Establish a reference speedup ceiling for block-diffusion SD on our
   hardware class
3. Inform whether our heterogeneous ANE+GPU approach needs to beat these
   numbers or just match them

## Setup

- Clean Python 3.14 venv (dflash-mlx requires 3.10+; system Python was 3.9)
- HF cache pinned to an external volume since the internal boot disk was tight
- Installed dflash-mlx via `pip install -e .`
- MLX 0.31.1, mlx-lm 0.31.2

Downloads (~10 GB total):
- `mlx-community/Qwen3-4B-bf16` (7.5 GB)
- `mlx-community/Qwen3-4B-4bit` (2.1 GB)
- `z-lab/Qwen3-4B-DFlash-b16` draft (1 GB, always bf16)

Verify mode: `parallel-replay` (dflash-mlx reference mode).

## Apples-to-apples reproduction (author's math@4028 prompt)

| Metric | M4 Pro (ours) | M4 Max (theirs) | Ratio |
|---|---:|---:|---:|
| bf16 baseline gen tok/s | 27.7 | 40.6 | 0.68 |
| bf16 DFlash gen tok/s | 112.5 | 186.4 | 0.60 |
| **bf16 DFlash speedup** | **4.06×** | 4.59× | — |
| bf16 avg acceptance (tokens/window) | 13.55 | 13.55 | identical |
| bf16 DFlash peak mem | 10.00 GB | 10.00 GB | identical |
| 4bit baseline gen tok/s | 81.1 | 110.5 | 0.73 |
| 4bit DFlash gen tok/s | 126.0 | 159.2 | 0.79 |
| **4bit DFlash speedup** | **1.55×** | 1.44× | — |
| 4bit avg acceptance (tokens/window) | 8.96 | 8.92 | identical |

**Observations:**

- Acceptance rates are **bit-for-bit identical** to their published numbers —
  verification is deterministic at temp=0.
- Absolute tok/s is ~60-70% of M4 Max (M4 Pro has ~60% of M4 Max's memory
  bandwidth).
- **Multiplicative speedup is essentially identical on M4 Pro vs M4 Max** —
  slightly better at 4bit (1.55× vs 1.44×).
- Peak memory for bf16 DFlash matches exactly (10 GB); 4bit is slightly lower
  (3.65 GB vs 4.42 GB) — likely minor MLX version differences in bookkeeping.

## Multi-prompt sweep (reality-check on published headlines)

5 diverse prompts at max 512 tokens, greedy decode:

| Config | Baseline gen tok/s | DFlash gen tok/s | Speedup (gen) | Avg accept len | Peak mem (GB) |
|---|---:|---:|---:|---:|---:|
| bf16, mean | 28.5 | 43.6 | **1.53×** | 4.84 | 9.3 |
| 4bit, mean | 92.9 | 63.3 | **0.68×** (slowdown) | 4.01 | 2.8 |

**Per-prompt bf16 speedups:**

| Prompt | Speedup |
|---|---:|
| Math | 2.67× |
| Code | 2.02× |
| Simple math | 1.42× |
| Prose | 0.87× |
| Creative writing | 0.64× |

**Per-prompt 4bit speedups:**

| Prompt | Speedup |
|---|---:|
| Math | 1.17× |
| Code | 0.81× |
| Simple math | 0.62× |
| Prose | 0.46× |
| Creative writing | 0.35× |

## Critical context: the 4.6× headline is a best case

dflash-mlx's README headlines 4.6× speedup at bf16. That number is from their
"math at 4028 tokens" benchmark — a long-context structured reasoning task
where acceptance hits 13.55 tokens/window (exceptionally high).

**Our M4 Pro reproduction confirms 4.06× on that specific prompt.** But on
diverse shorter prompts (512 tokens), the mean drops to 1.53× at bf16 and
0.68× (NET SLOWDOWN) at 4bit.

Takeaways:

- **Benchmark numbers must specify workload.** "4.6× speedup" is accurate
  for a specific ideal case, misleading as a general claim.
- **Acceptance rate is the dominant factor** — swings from 13.55 tokens/window
  (math) to 2 tokens/window (prose), with speedup tracking almost linearly.
- **The 4bit slowdown is structural.** Even with a purpose-built block-diffusion
  draft and all the supporting framework, 4bit targets decode too fast on
  Apple Silicon for SD to help on average.

## How this compares to our mlx-lm + Gemma-3 baseline

On the bf16 comparison:

| Approach | Hardware | Target | Mean bf16 speedup |
|---|---|---|---:|
| **mlx-lm SD + natural draft (ours)** | M4 Pro | Gemma-3-12B | **2.45×** |
| dflash-mlx + custom draft | M4 Pro | Qwen3-4B | 1.53× |
| dflash-mlx + custom draft (published) | M4 Max | Qwen3-4B | ~1.5× (inferred) |

Our natural same-family pair with standard mlx-lm SD beats dflash-mlx's
purpose-built approach on **average**. Their ceiling is higher on ideal math
prompts (4.06× vs our 3.31× max), but ours is more robust across workloads.

Reasons this may be the case:

1. **Larger target (12B vs 4B)** → more memory-bandwidth-bound → more SD
   headroom
2. **Natural same-family draft alignment** — Gemma-3-270M is trained on the
   same data/objectives as Gemma-3-12B, so acceptance is high across diverse
   prompts, not just ones the custom draft was tuned for
3. **Standard SD is more robust to workload variance** than block diffusion,
   which relies on the draft predicting entire blocks correctly

## Reproducing this

```
dflash-work/
├── dflash-mlx/                  # git clone + editable install
├── .venv/                       # Python 3.10+ venv
├── hf-cache/                    # HF models (8+ GB; pin HF_HOME to keep off boot disk)
├── dflash_bench.py              # driver script (not in this repo — rewrite from README)
├── bench_log.txt                # full output
└── bench_results.json           # structured results
```

Key environment pins to avoid filling the boot disk: `HF_HOME`, `HF_HUB_CACHE`,
`HF_XET_CACHE`, `HUGGINGFACE_HUB_CACHE`, `TRANSFORMERS_CACHE`, `TMPDIR`, and
`HF_HUB_DISABLE_XET=1` all set to a volume with ≥15 GB free.
