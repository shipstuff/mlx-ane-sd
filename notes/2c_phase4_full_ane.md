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

With the full-ANE stack, per-cycle ANE busy time:

| ANE op         | ms/cycle |
|:---------------|---------:|
| chunk 1        |    19.3  |
| chunk 2        |    19.3  |
| target lm_head |     3.2  |
| draft predict  |     5.7  |
| draft lm_head  |     3.1  |
| **total ANE**  | **~50.6** |

Per-cycle total: ~53 ms. ANE is ~96% of cycle — saturated. GPU does ~1
ms of work (embed + norm). The hardware balance has flipped from the
original 90% GPU / 10% ANE to ~2% GPU / 96% ANE.

This has implications for multi-stream: ANE now serializes target work
between streams. The previous GPU-bottleneck multi-stream ceiling no
longer applies in the same way.

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
