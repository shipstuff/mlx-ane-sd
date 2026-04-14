# Phase C: heterogeneous SD under GPU contention — **the real win**

**Date:** 2026-04-13
**Hardware:** Mac mini M4 Pro, 64 GB
**Setup:** Gemma-3-12B bf16 target (foreground), Gemma-3-270M bf16 background
load running continuously on GPU.

## Headline result

When the GPU is contended by another workload, **heterogeneous ANE+MLX SD
preserves far more of its solo throughput than pure-MLX SD**.

| Approach | Solo tok/s | Parallel tok/s | Slowdown | Retained |
|---|---:|---:|---:|---:|
| Target-only, no SD | 9.05 | 7.91 | -13% | 87% |
| Pure-MLX SD (target + draft on GPU) | 19.60 | 13.88 | **-29%** | 71% |
| **Heterogeneous (draft on ANE)** | **23.26** | **20.42** | **-12%** | **88%** |

Under contention, heterogeneous is **47% faster** than pure-MLX SD
(20.42 vs 13.88 tok/s). Background workload got the same throughput in
both cases (~141 tok/s) — ANE offload doesn't penalize the secondary
workload either.

**The target-only baseline is the critical diagnostic.** Target-only
degrades by only 13% under contention — roughly the same as
heterogeneous (12%). That tells us the contention penalty in pure-MLX
SD (29%) isn't coming from target-vs-background competition. It's
coming from having **two** MLX models on the GPU (target + draft), each
queuing behind the background workload *and* each other. Heterogeneous
avoids this by putting the draft on separate silicon.

Put another way: under contention, SD is still worth doing (20.42 vs
7.91 = 2.58× for heterogeneous) — but only if you don't pile the draft
onto the same contended GPU.

## Why this is the real value proposition

Phase B showed that heterogeneous SD is only ~10% faster than pure-MLX SD
on solo single-stream inference (throughput within noise, power slightly
worse). Phase C reveals what the heterogeneous architecture actually buys
you:

1. **GPU contention resilience**: the draft runs on separate silicon so
   it doesn't compete with the target (or with any other GPU workload)
   for Metal queue slots and memory bandwidth.
2. **Composability with other workloads**: a running agent using
   heterogeneous SD doesn't get slowed down when the user also runs an
   MLX model in another app.
3. **Better peak speedup at the pair level**: the foreground SD throughput
   is 47% higher under contention. In a 2-workload setup, total tokens/s
   across both workloads is higher with heterogeneous.

This matches the `anemll-qwen35` Exp E finding (35B MLX + 0.8B ANE in
parallel preserves 86% of foreground throughput) applied specifically to
the speculative decoding pattern.

## Setup details

- Foreground workload: `mlx-community/gemma-3-12b-it-bf16` target,
  `mlx-community/gemma-3-270m-it-bf16` MLX draft or ANEMLL 270M ANE draft
- Background workload: same 270M model running `stream_generate`
  continuously on rotating prompts
- num_draft=12, max_new=50 (the sweet spot from Phase B sweeps)
- Same prompt "The capital of France is" for reproducibility
- 3 processes total (main + target + background)

The background Gemma-3-270M saturates the GPU at ~141 tok/s continuously —
representative of a real secondary workload.

## Reproducing

```bash
# Target-only (no SD) solo + parallel — the diagnostic baseline
python scripts/phaseC_parallel_workload.py --approach baseline --mode solo --max-new-tokens 50
python scripts/phaseC_parallel_workload.py --approach baseline --mode parallel --max-new-tokens 50

# Pure-MLX SD solo + parallel
python scripts/phaseC_parallel_workload.py --approach mlx --mode solo --num-draft 12 --max-new-tokens 50
python scripts/phaseC_parallel_workload.py --approach mlx --mode parallel --num-draft 12 --max-new-tokens 50

# Heterogeneous SD solo + parallel
python scripts/phaseC_parallel_workload.py --approach hetero --mode solo --num-draft 12 --max-new-tokens 50
python scripts/phaseC_parallel_workload.py --approach hetero --mode parallel --num-draft 12 --max-new-tokens 50
```

## Combined throughput across both workloads

Sum across foreground + background:

| Config | Foreground tok/s | Background tok/s | Combined |
|---|---:|---:|---:|
| Target-only solo | 9.05 | — | 9.05 |
| Target-only parallel | 7.91 | ~117 | ~125 |
| Pure-MLX solo | 19.60 | — | 19.60 |
| Pure-MLX parallel | 13.88 | 141.84 | 155.72 |
| **Heterogeneous parallel** | **20.42** | **141.21** | **161.63** |

Heterogeneous produces 3.8% more total tokens/sec across both workloads
while delivering 47% better foreground speed.

## Connection to other project findings

This is the same pattern the `anemll-qwen35` project documented across
multiple contexts:

- **Exp E (35B + 0.8B parallel)**: 35B MLX on GPU preserved 86% of solo
  throughput while 0.8B ANE ran at 94% of its solo throughput. Combined
  125 tok/s on a single M4 Pro.
- **Exp B (ANE/MLX split)**: splitting a model across engines gives +65%
  over pure-ANE because GPU handles the compute-heavy portion.

Phase C validates the same principle for speculative decoding: the value
of the ANE is not that it beats the GPU at a given task, but that it lets
you run MORE tasks in parallel without interference.

## Interpretation

The "right" way to think about heterogeneous SD on Apple Silicon:

- It's not a faster SD algorithm — it's a better **deployment topology**
- Solo, pure-MLX SD is fine (in fact slightly more efficient power-wise)
- Under contention (which is the realistic deployment scenario for any
  always-on agent running alongside user workloads), heterogeneous SD
  wins cleanly

For production: if your agent runs by itself, use pure-MLX SD. If it runs
alongside any other GPU workload (user-facing model, video/image gen,
graphics), use heterogeneous ANE+MLX SD.

## Absolute numbers to remember

At num_draft=12 on Gemma-3-12B bf16 target, same 270M-class draft:

- **Target-only solo**: 9.05 tok/s (1.00×)
- **Target-only contended**: 7.91 tok/s (0.87×)
- **Pure-MLX SD solo**: 19.60 tok/s (2.17×)
- **Pure-MLX SD contended**: 13.88 tok/s (1.53×)
- **Heterogeneous solo**: 23.26 tok/s (2.57×)
- **Heterogeneous contended**: 20.42 tok/s (2.26×)

The contended-heterogeneous number (2.26×) roughly matches the solo pure-MLX
number (2.17×). Which means: **running heterogeneous SD under load gets you
the same speed as running pure-MLX SD alone.**

That's the headline for real-world deployment.
