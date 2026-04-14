# Day 0 findings — speculative decoding baseline measurements

**Date:** 2026-04-13
**Hardware:** Mac mini M4 Pro, 64 GB (two machines used — one for our local
mlx-lm SD sweeps, one for the dflash-mlx reproduction)
**Duration:** single afternoon

## Summary

Established the SD speedup ceiling on Apple Silicon for dense transformer
targets using two approaches:

1. **Our approach:** mlx-lm's built-in speculative decoding with a natural
   same-family draft (Gemma-3-12B + Gemma-3-270M)
2. **dflash-mlx:** Block-diffusion speculative decoding with a custom-trained
   draft (Qwen3-4B + z-lab/Qwen3-4B-DFlash-b16)

Key finding: **bf16 is the only precision where SD gives meaningful speedup
on Apple Silicon.** At 4bit, the target decodes fast enough that the draft
overhead eats most of the benefit regardless of approach.

## The datasets

**Our Gemma-3-12B sweep (M4 Pro):**
- Target: `mlx-community/gemma-3-12b-it-{4bit,bf16}`
- Draft: `mlx-community/gemma-3-270m-it-bf16`
- Engine: `mlx_lm.stream_generate` with `draft_model=` parameter
- Prompts: 5-10 diverse (capital, math, fibonacci, Pythagorean, gradient
  descent, speed of light, coffee, planets, string reverse, photosynthesis)
- Decode: greedy (temp=0), max 80 new tokens
- Output correctness: 10/10 exact match vs target-only decode (SD is lossless)

**dflash-mlx baseline (M4 Pro):**
- Target: `mlx-community/Qwen3-4B-{bf16,4bit}`
- Draft: `z-lab/Qwen3-4B-DFlash-b16` (block-diffusion, same size as target)
- Engine: dflash-mlx CLI, `parallel-replay` verify mode
- Prompts: 5 diverse + their math@4028 ideal case
- Decode: greedy (temp=0), up to 4028 new tokens

## Phase 0 — initial acceptance check

First ran mlx-lm SD out of the box at `num_draft_tokens=4`, 4bit target:

```
Prompt                                  base tok/s  sd tok/s  speedup  accept
The capital of France is                30.9        40.2      1.30×    76.2%
1 + 1 equals                            30.7        23.4      0.76×    56.2%
def fibonacci(n):                       30.6        37.8      1.24×    75.0%
The Pythagorean theorem states that     30.6        35.8      1.17×    71.2%
In machine learning, gradient descent   30.5        25.4      0.83×    60.0%
The speed of light ...                  30.4        29.0      0.95×    66.2%
To make a good cup of coffee, first     30.0        27.0      0.90×    62.5%
The largest planet in our solar system  30.4        23.5      0.77×    60.0%
Write a Python function that reverses   29.9        39.4      1.32×    77.5%
Explain photosynthesis ...              30.5        24.8      0.81×    60.0%

Mean: base 30.5 tok/s, SD 30.6 tok/s, speedup 1.01×, acceptance 66.5%
```

Conclusion at 4bit: high acceptance (66.5% mean) does NOT translate to wall-clock
speedup. Draft overhead eats the benefit. Some prompts actually SLOW DOWN under SD.

## Phase 0b — `num_draft_tokens` sweep at 4bit

Tried to find the sweet spot:

```
num_draft  base tok/s  sd tok/s  mean speedup  max speedup  accept
    2      30.3        35.2      1.16×         1.37×        58.2%
    3      30.4        35.5      1.17×         1.46×        65.2%  ← peak
    4      30.0        30.6      1.02×         1.24×        69.5%
    6      29.8        26.5      0.89×         1.15×        73.2%
    8      29.6        25.1      0.85×         1.25×        74.8%
```

**Peak at `num_draft=3` with 1.17× mean.** Above that, draft overhead dominates
despite rising acceptance. Below that, we don't amortize enough.

## Phase 1 — bf16 sweep (downloaded bf16 Gemma-3-12B, ~24 GB)

```
num_draft  base tok/s  sd tok/s  mean speedup  max speedup  accept
    2      9.20        10.48     1.13×         1.31×        58.5%
    4      8.93        12.27     1.37×         1.69×        71.0%
    6      9.05        14.52     1.59×         2.03×        75.8%
    8      9.11        16.63     1.80×         2.67×        76.2%
```

And higher values:

```
num_draft  sd tok/s  mean speedup  max speedup  accept
    8      20.71     2.15×         2.75×        78.4%
   10      21.84     2.28×         3.14×        80.0%
   12      23.21     2.45×         3.31×        81.2%  ← peak
   16      21.50     2.30×         3.10×        83.4%
   20      19.86     2.13×         3.05×        83.7%
```

**Peak at `num_draft=12` with 2.45× mean, 3.31× max.** Baseline 9.1 tok/s → 23.2
SD tok/s. Acceptance keeps climbing past the peak, but draft time starts
dominating.

Critical observation: **the limiting factor above num_draft=12 is draft
serial time on the GPU, not acceptance.** If the draft ran in parallel on a
separate engine, we'd expect the peak to shift right and speedup to keep
climbing.

## dflash-mlx reproduction (M4 Pro)

Author reports: M4 Max, Qwen3-4B, 4.6× bf16 / 1.4× 4bit.

Our reproduction on M4 Pro:

**Apples-to-apples on their default math@4028 prompt:**

| Metric | M4 Pro (ours) | M4 Max (theirs) |
|---|---:|---:|
| bf16 baseline gen tok/s | 27.7 | 40.6 |
| bf16 DFlash gen tok/s | 112.5 | 186.4 |
| bf16 DFlash speedup | **4.06×** | 4.59× |
| bf16 avg acceptance (tokens/window) | 13.55 | 13.55 |
| 4bit baseline gen tok/s | 81.1 | 110.5 |
| 4bit DFlash gen tok/s | 126.0 | 159.2 |
| 4bit DFlash speedup | 1.55× | 1.44× |
| 4bit avg acceptance (tokens/window) | 8.96 | 8.92 |

M4 Pro is at ~60-70% of M4 Max's absolute throughput (bandwidth limited), but
the **multiplicative speedup matches cleanly** — actually slightly better on
4bit (1.55× vs 1.44×).

**Multi-prompt sweep (5 diverse prompts, 512 max tokens):**

| Config | Baseline | DFlash | Speedup |
|---|---:|---:|---:|
| bf16 mean | 28.5 tok/s | 43.6 | **1.53×** |
| 4bit mean | 92.9 tok/s | 63.3 | **0.68× (slowdown)** |

Per-prompt bf16: math 2.67× / code 2.02× / prose 0.87× / simple math 1.42× /
creative 0.64×.

Per-prompt 4bit: math 1.17× / code 0.81× / prose 0.46× / simple math 0.62× /
creative 0.35×.

## Cross-approach comparison

| Approach | Hardware | Target | Precision | Mean speedup | Best-case |
|---|---|---|---|---:|---:|
| **mlx-lm SD (ours)** | M4 Pro | Gemma-3-12B | bf16 | **2.45×** | 3.31× |
| **mlx-lm SD (ours)** | M4 Pro | Gemma-3-12B | 4bit | 1.17× | 1.46× |
| dflash-mlx | M4 Pro | Qwen3-4B | bf16 | 1.53× | 4.06× (math@4028) |
| dflash-mlx | M4 Pro | Qwen3-4B | 4bit | 0.68× | 1.55× (math) |
| dflash-mlx (theirs) | M4 Max | Qwen3-4B | bf16 | — | 4.59× (math@4028) |

**Takeaways:**

1. **We beat dflash-mlx on average** (2.45× vs 1.53× at bf16). Their
   purpose-built block-diffusion draft isn't obviously better than a natural
   same-family small draft when the target is large enough (12B vs 4B).
2. **dflash-mlx wins on specific workloads** (math@4028: 4.06×). Their ceiling
   is higher on ideal inputs, ours is more robust across prompts.
3. **4bit is a structural dead-end** across approaches. Even dflash-mlx is net
   slower than baseline on 4bit averaged over diverse prompts.
4. **Acceptance alone doesn't predict speedup.** Our sweeps hit 80%+ acceptance
   but speedup capped because draft time dominated. Speedup requires
   acceptance AND fast-enough draft.

## Why this matters for heterogeneous ANE + GPU

The bf16 sweep shows acceptance keeps climbing past `num_draft=12` (to 84% at
num_draft=20) but speedup peaks and declines. **Draft time on GPU is the
bottleneck above 12.**

If we move the draft to ANE and run it in parallel with target verification on
GPU, the draft becomes free (hidden under target's forward pass). Predicted
outcome: push num_draft to 16-20 with draft time hidden, extending speedup
toward 3-4× mean.

This is the load-bearing experiment for this project.

## Files

- `scripts/phase0_acceptance_test.py` — initial 10-prompt acceptance test at 4bit, num_draft=4
- `scripts/sweep_num_draft_4bit.py` — num_draft sweep at 4bit
- `scripts/sweep_num_draft_bf16.py` — num_draft sweep at bf16 (2, 4, 6, 8)
- `scripts/sweep_num_draft_bf16_high.py` — higher num_draft at bf16 (8, 10, 12, 16, 20)

## Next

See [next_steps.md](./next_steps.md) for the ANE draft bridge plan.
