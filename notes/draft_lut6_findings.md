# DFlash draft LUT6: 1.83× faster predict, stacks to 43.26 tok/s mean

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB (mini-02)
**Context:** The DFlash ANE draft was not actually LUT-compressed (1.0 GB
fp16 package), despite me previously assuming it was. anemll-profile's
"22.5 MB weights streamed/iter" was ANE's working-set per iteration, not
total model size. Applying LUT6 quantization (per_tensor for macOS15
compatibility) shrinks the draft 2.6× and makes it substantially faster.

## Result: 9.82 → 5.36 ms per predict (1.83× faster)

| variant          | size    | measured    | TOPS   | Weight BW   |
|:-----------------|--------:|------------:|-------:|------------:|
| fp16 (prior)     | 1.0 GB  |   9.82 ms   |   3.15 |   2.29 GB/s |
| **LUT6 per_tensor** | **385 MB** | **5.36 ms** | **5.92** | **75.9 GB/s** |

The **33× jump in effective weight bandwidth** reveals the fp16 draft
was actually memory-bound — weights weren't fitting ANE's on-chip storage
well. Compression let more weights stay resident, doubling effective TOPS.
My earlier "compute-bound" reading was wrong; the profile metric was
masking the true bottleneck.

## End-to-end stack (LUT6 draft + ANE LUT6 lm_head)

Solo, 4 standard prompts, max_new=100:

| prompt    | LUT6 draft + GPU lm_head | LUT6 draft + ANE lm_head |
|:----------|-------------------------:|-------------------------:|
| capital   |                    17.87 |                    20.94 |
| fibonacci |                    71.34 |                    86.12 |
| math      |                    32.60 |                **43.12** |
| story     |                    19.16 |                    22.88 |
| **mean**  |               **35.24**  |               **43.26**  |

Correctness preserved: text byte-identical to GPU baseline across all
prompts. Cycle counts within 1-3 of baseline (capital 57→58, math 31→28,
story 53→53, fibonacci 14→14).

## Full progression from baseline

| config                                      | mean tok/s | vs MLX baseline |
|:--------------------------------------------|-----------:|----------------:|
| MLX-only (no SD)                            |      28.5  |          1.00×  |
| Swift dflash-sd (fp16 draft + GPU lm_head)  |     34.30  |          1.20×  |
| + ANE lm_head                               |     41.07  |          1.44×  |
| **+ LUT6 draft** (current best)             |  **43.26** |      **1.52×**  |

Mean tok/s improvement from yesterday's baseline: **+26%** (34.30 → 43.26).
Now matching dflash-mlx's 1.53× headline — but on ANE+GPU heterogeneous
hardware instead of pure MLX/GPU.

## Per-cycle breakdown (current best)

| phase            | ms/cycle  |
|:-----------------|----------:|
| target_verify    |    72.6   |
| draft_predict    |     5.7   |
| draft_lmhead     |     3.1   |
| other            |    <0.5   |
| **total**        | **~82**   |

ANE busy: 8.8 ms (10.7%). GPU busy: 72.6 ms (88.5%).

Notice that ANE utilization actually DROPPED (15% → 11%) after adding
the LUT6 draft speedup, because the draft predict window shrank. The
GPU is now even more dominant as the bottleneck. Multi-stream headroom
is unchanged.

## Multi-stream stacked: aggregate 23 tok/s (1.10× solo)

| config                          | solo | 2-stream agg | ratio |
|:--------------------------------|-----:|-------------:|------:|
| GPU lm_head (yesterday)         | 17.13| 19.02        | 1.11× |
| ANE lm_head only (morning)      | 19.91| 22.93        | 1.15× |
| **LUT6 draft + ANE lm_head**    |**20.93** | **23.03** | **1.10×** |

Solo improved from 19.91 → 20.93 (+5%) but 2-stream aggregate was nearly
flat (22.93 → 23.03, +0.4%). Under 2-stream contention, GPU is fully
serialized at 120 ms/cycle target_verify, and savings on draft_predict
don't land on the critical path. **LUT6 draft is a solo optimization,
not a multi-stream one.**

## LUT6 granularity caveat

Used `per_tensor` granularity instead of `per_grouped_channel` because:

1. The draft was converted with `minimum_deployment_target=macOS15`
2. `per_grouped_channel` palettization requires iOS18/macOS15+ target at
   the quantization step, which errored with the stricter convert target
3. Re-converting with iOS18/macOS15 target is a separate one-time job

Per-tensor LUT6 is a coarser quantization (single palette for all weights
in a layer) than per-grouped-channel, but quality held across all 4 prompts.
If we wanted finer quantization, re-converting to iOS18 target could
enable per-grouped-channel — probably wouldn't change throughput much
since we're already 43.26 mean, but could reduce LUT-related accept rate
drift on harder prompts.

## Reproduction

```bash
# Convert (one-time, already done)
python scripts/dflash_coreml_convert_accum.py \
    --output /tmp/dflash_ane_accum.mlpackage

# LUT6 per_tensor quantize (42s)
python scripts/dflash_lut_quantize.py \
    --input /tmp/dflash_ane_accum.mlpackage \
    --output /tmp/dflash_ane_accum_lut6.mlpackage \
    --bits 6 --granularity per_tensor

# Compile + bench
xcrun coremlcompiler compile /tmp/dflash_ane_accum_lut6.mlpackage /tmp/dflash_ane_accum_lut6_c/

# Run
.build/release/dflash-sd \
    --draft /tmp/dflash_ane_accum_lut6_c/dflash_ane_accum_lut6.mlmodelc \
    --ane-lmhead /tmp/lmhead_qwen3/lmhead_lut6.mlmodelc \
    --prompt "..." --max-new 100
```
