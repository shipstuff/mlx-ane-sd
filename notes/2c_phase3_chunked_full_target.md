# 2c Phase 3: Chunked full-target on ANE

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB
**Milestone:** Full Qwen3-4B target on ANE via 2 × K=18 LUT6 chunks.
Only MLX work in target_verify is the final norm + lm_head.

## Result

| prompt    | tok/s  | cycles | acc/cyc | target_verify |
|:----------|-------:|-------:|--------:|--------------:|
| capital   |  27.81 |     61 |    1.67 |        47.3 ms |
| fibonacci | 110.90 |     14 |    7.07 |        50.3 ms |
| math      |  50.80 |     33 |    3.09 |        48.1 ms |
| story     |  28.11 |     60 |    1.65 |        47.8 ms |
| **mean**  |**54.40** |         |          |      **~48 ms** |

**1.91× over MLX bf16 baseline** (28.5 tok/s → 54.40 tok/s), **+5% over
K=18 single chunk** (51.80 tok/s).

## Full progression

| config                                                 | mean tok/s |  speedup |
|:-------------------------------------------------------|-----------:|---------:|
| MLX bf16 baseline (no SD)                              |       28.5 |    1.00× |
| dflash-mlx (MLX draft, bf16 target)                    |       43.6 |    1.53× |
| Python F.1 / Swift parity                              |       34.97 / 34.32 | 1.22× |
| + ANE LUT6 lm_head                                     |       41.07 |    1.44× |
| + LUT6 draft                                           |       43.26 |    1.52× |
| + K=18 partial target on ANE                           |       51.80 |    1.82× |
| **+ chunked full target on ANE** (current best)        |   **54.40** | **1.91×** |

## Per-cycle breakdown (chunked full-target)

| phase                       | ms/cycle |
|:----------------------------|---------:|
| target_verify total         |     48.5 |
|   → ANE chunk 1 (0-17)      |     19.3 |
|   → ANE chunk 2 (18-35)     |     19.3 |
|   → MLX norm + lm_head      |     ~9   |
|   → embed + conversions     |     0.9  |
| draft_predict (ANE)         |      5.7 |
| draft_lmhead (ANE)          |      3.1 |
| other                       |      2   |
| **total per-cycle**         | **~59.3** |

## Quality trade-off

All 36 target layers now run through LUT6 palettization (vs half at K=18
single-chunk). The additional quantization noise shifts argmax on
near-ties:

| prompt    | text match to baseline |
|:----------|:-----------------------|
| capital   | first ~80 chars byte-identical, diverges later |
| fibonacci | byte-identical |
| math      | semantically equivalent but different wording |
| story     | coherent but different continuation |

This is the same quality trade-off as 8bit target — quantization at
sufficient scale eventually shifts decoder output. For research we can:
1. Accept the drift (valid continuations, acceptable for most deployments)
2. Upgrade to per_grouped_channel LUT6 (needs iOS18 target at convert step)
3. Use LUT8 instead of LUT6 (bigger, more accurate)

The per_grouped_channel path is worth exploring — we already saw the
draft-body LUT6 with per_tensor works; per_grouped_channel should
preserve more accuracy for full-target.

## Integration details

### Handoff costs between chunks
Chunk 1 output is an MLMultiArray `[1, 16, 2560]` fp16. Chunk 2 takes the
same-shaped fp16 MLMultiArray as its "x" input. No dtype conversion
needed — we just pass the pointer through. Verified ~0.01 ms for the
between-chunk transition (negligible).

### Dual prefill cache sync
Extract MLX cache `state` for layers 0-17 → chunk 1 cache (promptLen
positions). Layers 18-35 → chunk 2 cache. Both writePos = promptLen,
globalOffset = promptLen. One `ane_cache_sync` phase at ~11 ms on the
capital prompt (twice the K=18 time, as expected).

### Capture routing
Global capture indices `[1, 9, 17, 25, 33]` split:
- Chunk 1 (layers 0-17): captures at local `[1, 9, 17]` → 3 tensors output
- Chunk 2 (layers 18-35): captures at local `[7, 15]` (= global 25, 33) → 2 tensors
- MLX: captures = empty (startIdx = 36, no layers run)

Runner reassembles in declared global-index order before handing to
DFlash's target_hidden input.

### MLX norm-only path
When `startIdx = 36 = layers.count`, `forwardFromLayerCapturing` skips the
loop and just applies `norm(h)` + `lm_head`. Added an early-return path
that avoids calling `createAttentionMask(..., cache: cache?[36])` which
would index out-of-bounds. This was the first bug encountered — caught
via SIGTRAP exit code 133.

## Headroom still open

The chunked full-target is our new best but leaves some wins on the table:

1. **Fused final norm into chunk 2**: save ~1 ms by moving norm to ANE.
2. **Fused lm_head onto ANE output**: we already have an ANE lm_head
   model. Could wire it after chunk 2 and skip MLX entirely for
   target_verify. Saves 9 ms (the tv_mlx_layers cost). **Projected
   mean tok/s: ~58-60**.
3. **Per_grouped_channel LUT6**: preserve quality so all prompts stay
   byte-identical.
4. **Three chunks of K=12**: less per-chunk LUT6 aggregation noise per
   chunk. Might preserve quality at same speed.

## Next

- Option A: **wire ANE lm_head after chunk 2** to eliminate the MLX norm
  + lm_head path. Cheap (a few hours of Swift), projected +5-10% mean.
- Option B: **fix quality drift** via per_grouped_channel LUT6 or LUT8.
  Requires re-converting chunks. Preserves bf16-level output across all prompts.
- Option C: **lock in the 54.40 result**, update README, move to
  writeup/paper.

## Reproduction

```bash
# Build both chunks (~6 min total palettize)
python scripts/convert_qwen3_layers_ane.py \
  --num-layers 18 --start-layer 0 --capture-indices 1,9,17 \
  --out-dir /tmp/qwen3_klayers_cap
python scripts/convert_qwen3_layers_ane.py \
  --num-layers 18 --start-layer 18 --capture-indices 7,15 \
  --out-dir /tmp/qwen3_klayers_cap

# Run with full stack
.build/release/dflash-sd \
  --prompt "..." --max-new 100 \
  --draft /tmp/dflash_ane_accum_lut6_c/dflash_ane_accum_lut6.mlmodelc \
  --ane-lmhead /tmp/lmhead_qwen3/lmhead_lut6.mlmodelc \
  --ane-target-layers /tmp/qwen3_klayers_cap/K18/qwen3_K18_lut6.mlmodelc \
  --ane-target-k 18 --ane-target-captures 1,9,17 \
  --ane-target-layers2 /tmp/qwen3_klayers_cap/K18_s18/qwen3_K18_lut6.mlmodelc \
  --ane-target-k2 18 --ane-target-captures2 7,15
```
