# 2c Phase 4: Full-ANE stack — 64.76 tok/s at bf16 (2.21× MLX baseline)

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB
**Milestone:** Essentially the entire SD pipeline runs on ANE. MLX is
only used for token embedding and the final RMSNorm. Quantization is
LUT6 throughout (per_grouped_channel for best quality).

## Final bench (decode-only, max_new=100, 4 prompts)

| config                                            | mean t/s | vs MLX bf16 baseline |
|:--------------------------------------------------|---------:|---------------------:|
| **MLX bf16 baseline (no SD)**                     |  **29.27** |              1.00× |
| Swift dflash-sd (fp16 draft + GPU lm_head)        |    34.05 |                1.16× |
| + ANE LUT6 lm_head (draft-side, bs=15)            |    40.96 |                1.40× |
| + LUT6 draft body                                 |    43.05 |                1.47× |
| + K=18 partial target (byte-identical)            |    52.81 |                1.80× |
| + chunked full target (2×K=18 per_tensor)         |    55.90 |                1.91× |
| + chunked + ANE target lm_head (per_tensor)       |    62.78 |                2.14× |
| **+ chunked + ANE target lm_head (pgc LUT6)**     | **64.76** |           **2.21×** |

### Per-prompt at current best (pgc LUT6 full-ANE)

| prompt    | MLX bf16 | current | speedup | cycles | acc/cyc |
|:----------|---------:|--------:|--------:|-------:|--------:|
| capital   |    29.39 |   32.99 |   1.12× |     57 |    1.81 |
| fibonacci |    29.25 |  140.82 |   4.81× |     13 |    7.62 |
| math      |    29.17 |   48.48 |   1.66× |     39 |    2.56 |
| story     |    29.26 |   36.74 |   1.26× |     51 |    1.94 |

**Fibonacci hits 4.8×** because the draft accepts ~7.6 of 15 positions
per cycle (structured, predictable prompt). Prose prompts (capital,
story) land at 1.1-1.3× because draft acceptance is ~1.8-1.9/cycle and
per-cycle cost (~45 ms) is still sizable vs baseline's single-token
autoregressive forward (~34 ms).

## Pipeline (final)

```
  verifyInput = [last_tok, draft_0, ..., draft_14]  (16 tokens)
              |
              v
     [MLX] embed_tokens  (0.5 ms)
              |
              v    fp16 MLMultiArray
     [ANE] Qwen3 chunk 1 (layers 0-17, LUT6 pgc)  (19.3 ms)
              |
              v    fp16 MLMultiArray
     [ANE] Qwen3 chunk 2 (layers 18-35, LUT6 pgc) (19.3 ms)
              |
              v    bf16 MLXArray
     [MLX] final RMSNorm  (0.5 ms)
              |
              v    fp16 MLMultiArray
     [ANE] lm_head (bs=16, LUT6 pgc)  (3.2 ms)
              |
              v    fp16 MLMultiArray [1, 16, 151936]
     [CPU] host argmax (Accelerate vDSP)  (0.1 ms)
              |
              v   [Int32] × 16
          targetTokenArray -> accept_check
```

**Per-cycle target_verify: ~43 ms** (vs 72 ms baseline — 40% faster).

ANE is now ~96% of per-cycle work; MLX is ~2% (embed + norm).

## Key engineering in Phase 4

### bs=16 ANE lm_head for target

The existing ANE lm_head was bs=15 (slice of 15 positions for the
draft's predictions). target_verify needs bs=16 (all positions) to get
the "bonus" target token at the mismatch position.

Added `--block-size-out` flag to `scripts/export_qwen3_lmhead_ane.py`:

```bash
python scripts/export_qwen3_lmhead_ane.py --block-size-out 16 \
    --skip-extract   # uses cached weights from bs=15 export
```

Output at `/tmp/lmhead_qwen3/bs16/lmhead_lut6.mlmodelc` (~280 MB). Same
quality metric (93% top-1 argmax agreement on random inputs) as bs=15.

### Qwen3InspModelInner.applyNorm

Added a public `applyNorm(hidden) -> MLXArray` that applies the final
RMSNorm without running any transformer layers. Let us plug MLX norm
between ANE chunks and ANE lm_head.

### The "unified quantization cancellation" effect

When both draft lm_head and target lm_head use LUT6, quantization noise
shifts the logits consistently — so the argmax winners in the SAME
quantized space agree more often. Concretely:

- Per_tensor chunks + GPU target lm_head: accept/cycle on capital = 1.67
- Per_tensor chunks + **ANE target lm_head**: accept/cycle on capital = 1.87 (**up**)

This is a happy side effect: running target through the same LUT6
lm_head the draft uses makes draft/target more likely to agree. Accept
rate on capital recovered past baseline's 1.74 to 1.87.

### Per_grouped_channel LUT6 for chunks

The original chunk converter used `macOS15` target which only supports
`per_tensor` palettization (one palette per whole tensor). Upgrading
to `iOS18` target enables `per_grouped_channel` with `group_size=16`
(one palette per group of 16 output channels — finer quantization).

Same disk size (1.3 GB), same ANE latency (18.5 ms), **better
accuracy**: mean tok/s 62.78 → 64.76 (+3%), with quality variance
shifted. Fibonacci +10%, story +18% (more consistent); capital -7%,
math -13% (slightly different near-tie argmax winners).

Net: pgc LUT6 is the better default for the chunk converter.

## Quality trade-off (honest framing)

Text divergence from bf16 baseline:

| config                                 | capital | fib | math | story |
|:---------------------------------------|:-------:|:---:|:----:|:-----:|
| K=18 partial (18 MLX layers intact)    |   ✓     |  ✓  |  ✓   |   ✓   |
| chunked per_tensor (+ MLX lm_head)     |   ✓     |  ✓  |  ≈   |   ≈   |
| chunked per_tensor + ANE target lm_head|   ✓     |  ≈  |  ≈   |   ≈   |
| chunked pgc + ANE target lm_head       |   ≈     |  ≈  |  ≈   |   ≈   |

Legend: ✓ = byte-identical first ~80 chars; ≈ = semantically valid but
different continuation (near-tie argmax flipped).

For strict quality, K=18 partial (52.81 t/s, 1.80×) gives byte-identical
output. Going beyond that trades mild text variance for throughput.

## ANE utilization saturated

### Per-cycle compute breakdown (profiler-measured)

| hardware | ms/cycle | share | what it does |
|:---------|---------:|------:|:-------------|
| **ANE**  | **49.9** | **93.7%** | chunk 1 (19.3) + chunk 2 (19.3) + draft predict (5.7) + target lm_head (3.2) + draft lm_head (3.1) |
| CPU      |    1.9   |   3.5%| MLMultiArray memcpy for cache sync/commit, host argmax (vDSP), RoPE table build, accept_check |
| GPU      |    1.5   |   2.8%| token embed (0.5 ms), final RMSNorm (0.6 ms), noise embed (0.4 ms) |
| **total**|   **53** |        |            |

The hardware balance has flipped from the original 90% GPU / 10% ANE to
~94% ANE / 3% GPU / 3% CPU.

### System-level during bench (sudo powermetrics, 500ms samples)

Captured live while running the bench in a loop:

| metric                         | value                | interpretation |
|:-------------------------------|:--------------------:|:---------------|
| GPU active residency           | **0.8–10%** (338 MHz only) | GPU is mostly idle — brief low-freq kernels for embed + norm |
| GPU power                      | 2–40 mW              | effectively off (M4 Pro GPU max ≈ 15 W) |
| CPU E-cluster active residency | 55–73%               | efficiency cores handle coordination |
| CPU P1-cluster active residency| 25–45%               | performance cores for heavier async work |
| CPU power                      | 180–630 mW (avg ~300)| modest — runs at low freqs (1–2.5 GHz mostly) |
| ANE power (powermetrics)       | 0 mW (reported)      | **`ane_power` sampler reports 0 on this workload — a known reporting quirk, NOT actual idle.** Profiler's ~50 ms/cycle of ANE call latency is the real number. |

### Interpreting the CPU load

CPU E-cluster at 55–73% active residency looks high at first glance, but
**it's coordination work, not compute**:

- Swift async/await scheduling for CoreML predictions
- MLMultiArray allocation + memcpy for KV cache sync, commit, conversions
- CoreML feature-dict setup per call
- Tokenizer ops + accept-check loop
- Host-side vDSP argmax over `[1, 16, 151936]` fp16 logits

The P-cores stay mostly at 1.3–2 GHz (not their 4.5 GHz max). CPU averages
~300 mW out of a chip budget measured in tens of watts. Everything about
the CPU picture says "doing small things frequently, at low frequency" —
classic coordination pattern.

### Implications for multi-stream

**The bottleneck flipped from GPU to ANE.** In Phase 2 data (GPU lm_head +
MLX target), multi-stream was GPU-bound: target_verify at 72 ms/cycle on
GPU serialized across streams. Under 2 streams, aggregate was 1.28× solo
(not 2×) because GPU couldn't keep up.

Now:
- **GPU is essentially free** (0.8–10% residency, 2–40 mW). It could host
  a second stream's target_verify with minimal contention.
- **ANE is ~94% utilized per stream**. Two streams on ANE would fully
  serialize — aggregate near 1.0× (not 2×).
- **CPU has headroom** at 300 mW / ~60% on E-cores. Could coordinate
  many concurrent streams.

So the **Phase C preservation pattern inverts**: instead of "ANE preserves
throughput when GPU is contended", it becomes "**GPU preserves throughput
when ANE is contended.**" This suggests a potential new play:

- Stream A on the full-ANE stack (what we have now)
- Stream B on the MLX target path (GPU-bound, previously "the slow path")
- Both run concurrently since they use disjoint hardware

Projected 2-stream aggregate in this layout:
- Stream A (ANE): ~65 tok/s × 1.0 contention factor = ~65 tok/s
- Stream B (GPU): ~43 tok/s × some contention factor
- If GPU isn't contended by stream A (only 0.5-1 ms/cycle), stream B's
  MLX forward stays near 72 ms/cycle → ~43 tok/s
- **Aggregate: ~108 tok/s** (vs our best single-stream 65 tok/s)

This is an open experiment, not measured. But the ANE 94% / GPU 3%
distribution is a strong hint that heterogeneous routing between streams
(not identical copies) could unlock more aggregate throughput than
same-stack multi-streaming.

## Reproduction

See the [README Quick Start](../README.md#quick-start--reproduce-the-best-221-result)
for the exact convert + run commands. Full bench:

```bash
python scripts/bench_final_stack.py --max-new 100
```

## What's still on the table

1. **Compile-caching**: first-load of the K=18 chunks takes ~20 seconds
   each (ANE compile). Subsequent loads cached by macOS. For production,
   ship the ANE-compiled artifacts.
2. **Draft on ANE optimization**: draft_predict is already 5.7 ms
   (LUT6). Could push to LUT4 for ~4 ms at some quality cost.
3. **Fused final norm into chunk 2**: save ~0.5 ms by baking norm at
   chunk 2's output. Marginal win.
4. **Multi-stream with new hardware balance**: previous Phase C data
   showed GPU serialized under 2-stream. Now ANE is the hot path —
   would the pattern invert? Worth re-measuring.
5. **LUT8 alternative**: 256-entry palette vs LUT6's 64. Could preserve
   byte-identical quality. Size ~+50%, latency similar.
