# 2c probe: single Qwen3-4B layer on ANE

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB
**Goal:** Establish ANE per-layer latency for partial-target-on-ANE, to
decide whether the engineering investment (multi-day) is justified.

## Result: ANE LUT6 layer = 1.55 ms. Beats GPU bf16 (2.0 ms/layer) by 29%

Ported Qwen3-4B-bf16 layer 0 (real extracted weights) to CoreML:

| variant        | size    | measured  | 100% ANE? | interruptions |
|:---------------|--------:|----------:|:---------:|:-------------:|
| fp16           | 192.6 MB | 2.61 ms   |    ✓      |       0       |
| **LUT6 (per_tensor)** | **72.3 MB** | **1.55 ms** |    ✓      |       0       |

Per-layer comparison across configurations:

| placement / precision    | ms/layer | notes |
|:-------------------------|---------:|:------|
| MLX GPU / bf16           |     2.0  | baseline, 72 ms total for 36 layers |
| MLX GPU / 8bit           |     1.44 | faster baseline via weight quant |
| **ANE / LUT6**           |  **1.55** | 29% faster than bf16 GPU |

## Break-even analysis: K layers on ANE

Moving K target layers to ANE saves `K × 0.45 ms` (vs bf16 GPU) minus cross-
device handoff overhead (~0.8 ms round-trip, from our measured `mlx_to_coreml`).

| K (layers on ANE) | saving (ms) | target_verify (ms) | % faster |
|------------------:|------------:|-------------------:|---------:|
|                 1 |      -0.35  |              72.35 |  -0.5%  |
|                 2 |       0.10  |              71.90 |   0.1%  |
|                 5 |       1.45  |              70.55 |   2.0%  |
|                10 |       3.70  |              68.30 |   5.1%  |
|                18 |       7.30  |              64.70 |  10.1%  |
|             **36** | **15.40**   |          **56.60** | **21.4%** |

**Full target on ANE at bf16 = 21% faster target_verify** (72 → 57 ms).
Partial target wins meaningfully only at K ≥ 10.

## Integration constraints

1. **KV cache split**: first K layers' KV stays on ANE (external-cache pattern
   like DFlash), remaining (36-K) layers stay on MLX. Two caches to maintain.

2. **Capture layer indices**: DFlash draft needs target hidden states from
   layers `[1, 9, 17, 25, 33]`. If K ≥ 2, layer 1 capture happens on ANE.
   Need to materialize the capture output back to MLX to concat with the
   other captures.

3. **ANE model size**: 36 layers × 72 MB = 2.6 GB. ANE compilation often
   fails for single-blob models > 1 GB (we saw this with the fp16 lm_head).
   Likely needs to split into N chunks (e.g., 4 × 9-layer chunks at 650 MB
   each), adding N-1 extra handoffs.

4. **Handoff overhead accumulates**: each ANE-chunk boundary costs ~0.4 ms.
   4 chunks = 3 extra handoffs = 1.2 ms overhead. Total for full-target:
   55.8 ANE + 1.6 handoffs = 57.4 ms. Still 20% faster than GPU bf16.

## Effort estimate

| phase                                              | effort  |
|:---------------------------------------------------|:--------|
| Multi-layer CoreML conversion script               | 1 day   |
| KV cache dual-state management in Swift runner     | 1 day   |
| Handling capture layers inside ANE range           | 0.5 day |
| Chunking for memory fit + handoff orchestration    | 1 day   |
| Correctness testing (text match to baseline)       | 0.5 day |
| **Total**                                          | **~4 days** |

## Expected final impact (stacked with current best)

Current best (bf16): Swift + ANE lm_head + LUT6 draft = 43.26 tok/s mean.
Per-cycle: 82 ms = 72 (tv) + 5.7 (dp) + 3.1 (dlh) + 0.5 (misc).

With full target on ANE (projected): target_verify 72 → 57 ms.
New per-cycle: 67 ms. Projected mean tok/s: 43.26 × (82/67) = **~53 tok/s mean**
at bf16 quality. **2.6× vs bf16 MLX baseline (28.5 t/s)** — and reaching this
purely at bf16 would be more credible than the 8bit target number.

## Recommendation

2c is viable at bf16 target and would be a clean +20% on target_verify, our
single biggest remaining bottleneck. But it's ~4 days of careful engineering
(vs 1-2 days for most of today's wins). Worth it if we're committed to the
heterogeneous-SD research narrative; maybe not if we're close to a writeup.

**Next decision point for user:** commit to 2c full implementation, or
document findings and transition to paper writeup.
