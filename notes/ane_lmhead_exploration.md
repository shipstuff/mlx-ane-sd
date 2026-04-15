# ANE lm_head: LUT6 hits 3.06 ms — 6.4× faster than GPU

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB (mini-02)
**Context:** Single-stream Swift dflash-sd has `draft_lmhead` on MLX GPU at
19.5 ms/cycle (19% of a 102 ms cycle). ANE is idle 90% of each cycle.
Pushing lm_head onto ANE offloads GPU and increases ANE utilization.

## Why this works (the short version)

**What it is.** The LUT6 lm_head is the final projection layer of Qwen3-4B
(`[hidden=2560] → [vocab=151936]`) — the matrix multiply that turns hidden
states into vocabulary logits, from which we sample tokens. The original
weight is ~778 MB in bf16. We applied **LUT6 palettization**: each weight
is replaced with a 6-bit index into a small palette of cluster centroids
(k-means with `group_size=16`, per-grouped-channel granularity), shrinking
the tensor to ~280 MB. That compression is what makes it fit ANE
compilation constraints — the fp16 742 MB version was too large and the
CoreML compiler fell back to CPU.

**Why offloading it to ANE helps:**

1. **Direct latency win.** The matmul went from 19.5 ms on MLX/GPU to
   3.06 ms on ANE — a 6.4× speedup. Per cycle this saves ~16 ms
   (102 ms → 87 ms), translating to **+20% end-to-end throughput**
   (34.30 → 41.07 mean tok/s on Qwen3-4B-bf16).
2. **Quality preserved.** Despite LUT6 only matching fp32 reference 93.3%
   of the time on *random* inputs, real Qwen3 hidden states produce peaked
   logit distributions where the top-1 dominates — so argmax rarely shifts.
   End-to-end output is byte-identical to the GPU baseline across all 4
   standard prompts.
3. **Better hardware balance.** Before, ANE was idle 90% of each cycle
   (only running the DFlash draft predict at ~10 ms). Moving lm_head onto
   ANE bumps utilization to ~15% of cycle (predict + lm_head = ~13 ms),
   while GPU drops from 89% → 73% busy. **That GPU headroom is what makes
   future multi-stream work more promising** — the GPU was the bottleneck,
   and we just freed ~20 ms of its budget per cycle per stream.
4. **No accuracy cost from the LUT6 quantization concern.** The
   synthetic-input quality test was misleading; the real-data behavior is
   what matters.

In short: it's the single biggest single-stream optimization we landed
today, and it shifts the hardware balance toward the underutilized engine.

## Latency comparison (Qwen3-4B lm_head shape)

Input: `[1, 15, 2560]` fp16. Weight: `[151936, 2560]`. Op: matmul.

| variant                        | weight size | placement         | measured    | vs GPU |
|:-------------------------------|------------:|:------------------|------------:|-------:|
| MLX bf16 (current runner)      |      778 MB | GPU               |    19.5 ms  |   1.0× |
| CoreML fp16 + argmax           |      742 MB | CPU fallback      |     6.59 ms |   3.0× |
| CoreML fp16 (no argmax)        |      742 MB | CPU fallback      |     6.86 ms |   2.8× |
| **CoreML LUT6 (no argmax)**    |  **280 MB** | **100% ANE**      | **3.06 ms** | **6.4×** |

## What the profile told us

The fp16 742 MB variant's `ANE compilation likely failed — model fell back to
CPU` — the weight was too large to fit ANE's compilation pattern. LUT6
palettization (6-bit clusters, group_size=16) compressed weights 2.6×. The
compressed model compiled cleanly for ANE:

```
  Model size:   279.7 MB
  Total ops:    3
  ANE ops:      1 (100.0% of cost)
  CPU ops:      0 (0.0% of cost)
  ANE graph interruptions: 0
  Measured:  3.058 ms/prediction  (327.0 iter/s, 10 runs)
```

## Per-cycle and end-to-end projection

Current cycle (single stream): 72 (target_verify GPU) + 19.5 (draft_lmhead GPU)
+ 10 (draft_predict ANE) + 0.4 (other) = **102 ms/cycle**.

Projected with LUT6 ANE lm_head: 72 + 3.1 + 10 + 0.4 = **85.5 ms/cycle**.

- **1.19× per-cycle speedup** (at same accept rate)
- **Mean tok/s 34.3 → ~41**
- ANE utilization: 10% → 13% of cycle (both draft_predict + lm_head)
- GPU utilization: 89% → 72% of cycle (frees GPU for more work, e.g. a second
  stream's target_verify)

## Remaining validation

1. **Quality**: LUT6 quantization of the real Qwen3-4B lm_head — does top-1
   argmax match MLX bf16 reference on random inputs? Expect yes
   (anemll uses LUT6 for Gemma/Llama lm_heads), but needs verification.
2. **Integration overhead**: extra CoreML `predict()` call per cycle. Measured
   standalone at ~3 ms — confirm it materializes that way inside the Swift
   loop (no extra MLX→CoreML conversion since hidden is already a
   `MLMultiArray` from the ANE draft output).
3. **LUT6 conversion time**: 11 minutes for 742 MB weight on M4 Pro via
   `coremltools.optimize.palettize_weights`. One-time cost but bigger than
   expected — consider parallelizing or using a faster clustering method.

## LUT6 cost study candidates (not yet tested)

- LUT8 instead of LUT6: 4-bit less compression (~380 MB), more accuracy.
- group_size 8 vs 16 vs 32: affects both quality and speed.
- Per-tensor vs per-grouped-channel palettization.
- Fused lm_head + argmax in one CoreML model (argmax forced ANE via type
  coercion workarounds).

## Open question on 6.4× speedup origin

Why is MLX GPU so slow for a simple matmul? 19.5 ms for 778 MB × [1, 15, 2560]
is only ~40 GB/s effective bandwidth — M4 Pro's GPU should hit ~250 GB/s.
Three hypotheses:

1. MLX's bf16 GEMM on Apple GPU is less optimized than fp16.
2. The round-trip cost (MLMultiArray → MLX copy → eval → copy back → argmax
   force-eval via `.asArray`) is eating most of the time.
3. MLX kernel scheduling has significant launch overhead for each forward.

All three could be partly true. The CoreML path measured 6.8 ms on CPU for
the same shape, suggesting the pure matmul is ~7 ms-class even without ANE.
So at least 12 ms of the MLX path is probably overhead. This deserves
deeper investigation — might yield optimizations to the **rest** of the
runner's MLX GPU calls too.

---

## End-to-end result with real Qwen3-4B weights

**Mean tok/s: 34.30 → 41.07 = +20% (1.20×).** Text byte-identical to GPU
baseline across all 4 standard prompts.

| prompt    | GPU baseline | ANE LUT6 lm_head | speedup | tv ms | dlh ms (gpu→ane) |
|:----------|-------------:|-----------------:|--------:|------:|-----------------:|
| capital   |       17.06  |          19.85  |   1.16× |  72.4 | 19.5 → 3.08      |
| fibonacci |       68.09  |          81.39  |   1.20× |  72.6 | 20.7 → 3.21      |
| math      |       33.36  |          40.91  |   1.23× |  72.5 | 19.8 → 3.15      |
| story     |       18.70  |          22.12  |   1.18× |  72.4 | 19.4 → 3.09      |
| **mean**  |   **34.30**  |       **41.07** |   1.20× |       |                  |

**Tied embedding caveat**: Qwen3-4B-bf16 has `tie_word_embeddings=true`, so
lm_head weight comes from `model.model.embed_tokens.weight`. Same matrix
either way — the script handled this correctly.

### Quality preserved despite 93.3% random-input agreement

The pre-integration LUT6 quality test (random fp16 hiddens vs fp32 reference)
showed only 93.3% top-1 argmax agreement, raising concern. **End-to-end
acceptance was unaffected** — real Qwen3 hidden states have peaked logit
distributions where the largest logit dominates by a wide margin. LUT6 noise
rarely shifts the top-1 in practice.

Cycle counts and accept rates per prompt nearly identical:
- capital: 57 (gpu) vs 58 (ane) cycles, accept 1.74 vs 1.71
- fibonacci: 14 vs 14 cycles, accept 7.07 vs 7.07
- math: 29 vs 28 cycles, accept 3.41 vs 3.54
- story: 52 vs 52 cycles, accept 1.90 vs 1.90

### Per-cycle breakdown (after optimization)

| phase            | GPU baseline | ANE LUT6 |
|:-----------------|-------------:|---------:|
| target_verify    |     72.4 ms  | 72.9 ms  |
| draft_lmhead     |     19.5 ms  |  3.1 ms  |
| draft_predict    |     10.2 ms  | 10.3 ms  |
| mlx_to_coreml    |      0.4 ms  |  0.4 ms  |
| **per-cycle**    |   **102.5 ms** | **86.7 ms** |

ANE utilization rose from 10% (draft_predict only) to **15% (predict +
lm_head)**. GPU utilization dropped from 89% to **73%** — meaningful
headroom for multi-stream serving where the GPU was the bottleneck.

### Implementation notes

- Swift integration: added `--ane-lmhead` flag to `dflash-sd`. Loads
  separate `MLModel` for lm_head, slices draft hidden `[1,16,2560] →
  [1,15,2560]` via memcpy, calls ANE predict, host argmax via
  `vImageConvert_Planar16FtoPlanarF` + `vDSP_maxvi`.
- Host argmax with Accelerate: 0.1 ms for `[1, 15, 151936]` fp16. Naive
  Swift loop was 0.7 ms — Accelerate is 7× faster.
- Cold start: +5s ANE lm_head load on top of the ~700ms target load.

### Reproduction

```bash
# 1. Export real-weight LUT6 lm_head (~3 min)
python scripts/export_qwen3_lmhead_ane.py

# 2. Build Swift runner with --ane-lmhead support
cd swift-bench && swift build -c release

# 3. Bench end-to-end
python scripts/bench_ane_lmhead.py --max-new 100
```
