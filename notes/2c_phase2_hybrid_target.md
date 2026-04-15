# 2c Phase 2: Hybrid target_verify WORKS — 51.80 tok/s mean at bf16

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB
**Milestone:** K=18 partial target on ANE end-to-end, byte-identical output
to baseline, **+20% over our previous best and 1.82× over MLX bf16**.

## Result

Full 4-prompt bench, max_new=100, Qwen3-4B-bf16 target:

| prompt    | tok/s  | cycles | acc/cyc | target_verify | ANE 18-layer | MLX 18-layer |
|:----------|-------:|-------:|--------:|--------------:|-------------:|-------------:|
| capital   |  25.14 |     57 |    1.74 |        60.1 ms |     18.82 ms |     40.4 ms |
| fibonacci | 103.71 |     13 |    7.62 |        60.9 ms |     19.41 ms |     40.5 ms |
| math      |  49.10 |     29 |    3.41 |        60.4 ms |     18.90 ms |     40.5 ms |
| story     |  29.25 |     49 |    2.02 |        60.1 ms |     18.77 ms |     40.4 ms |
| **mean**  |**51.80** |         |          |     **~60 ms** |    **~19 ms** |   **~40 ms** |

Text: byte-identical to full-MLX baseline on all 4 prompts. Accept rate
identical. Correctness preserved end-to-end.

## Full progression

| config                                                | mean tok/s | vs MLX bf16 baseline |
|:------------------------------------------------------|-----------:|---------------------:|
| MLX bf16 baseline (no SD)                             |       28.5 |                1.00× |
| dflash-mlx (MLX draft)                                |       43.6 |                1.53× |
| Swift dflash-sd (matches Python F.1)                  |       34.32 |                1.20× |
| + ANE LUT6 lm_head                                    |       41.07 |                1.44× |
| + LUT6 draft                                          |       43.26 |                1.52× |
| **+ K=18 partial target on ANE** (current best)       |   **51.80** |            **1.82×** |

## Per-cycle breakdown (current best)

| phase                    | ms/cycle |
|:-------------------------|---------:|
| target_verify total      |     60.3 |
|   → ANE 18 layers        |     19.0 |
|   → MLX 18 layers        |     40.4 |
|   → embed + conversions  |      0.9 |
| draft_predict (ANE)      |      5.7 |
| draft_lmhead (ANE)       |      3.1 |
| other (mlx_to_coreml etc.)|     0.5 |
| **total per-cycle**      | **~69.6**|

Vs previous best per-cycle: 82 ms → 69.6 ms = **15% cycle speedup**.
Throughput scales directly with accept rate, so mean tok/s goes 43.26 →
51.80 (+20%).

## Implementation highlights

### Cache sync at prefill
After MLX prefill populates the full 36-layer target cache, we extract
each of layers 0..17's `(keys, values)` via `cache[i].state`, convert each
to `MLMultiArray` fp16, and write into the ANE's accumulating KV cache at
positions `[0, promptLen)` with `writePos = promptLen`. One-time cost,
~2.2 ms on the capital prompt.

### The dtype-cast gotcha
First working version had target_verify at **194 ms** (worse than baseline).
Root cause: ANE outputs fp16, MLX target is bf16. Without an explicit
`hiddenAfterK.asType(.bfloat16)` cast, MLX was doing per-layer dtype
conversion inside each of the 31 (→now 18) layers. Casting once up front
dropped MLX 18-layer latency from **188 ms → 40 ms** (near-baseline 2 ms/layer).
The fp16→bf16 cast is one tensor, fast.

### Capture layer splitting
DFlash draft needs target captures at layers `[1, 9, 17, 25, 33]`. With
K=18 split:
- Layers 1, 9, 17 captured inside ANE (stacked tensor output)
- Layers 25, 33 captured on MLX side (existing flow)
- Reassembled in declared order before passing to next-cycle draft input

### Integration surface
- `Qwen3ANELayers` class in `DFlashCore` (225 lines): loads CoreML model,
  manages external KV cache, `forward()/commit()/trim()/loadFromPrefill()`
- `Qwen3InspModelInner.forwardFromLayerCapturing(startIdx, hidden, cache, captureAt)`:
  runs layers `[startIdx..35]` + final norm with pre-computed hidden input
- `DFlashSDRunner`: new flags `--ane-target-layers`, `--ane-target-k`,
  `--ane-target-captures`; hybrid path wired with per-stage profiling

## Per-layer cost analysis

| placement    | per-layer ms | observed total              |
|:-------------|-------------:|:----------------------------|
| MLX GPU bf16 | 2.0          | 72 ms / 36 layers           |
| MLX GPU bf16 (starting layer 18) | 2.24  | 40.4 ms / 18 layers (slightly higher) |
| ANE LUT6 (K=18) | 1.06       | 19 ms / 18 layers           |

ANE per-layer is ~half the GPU per-layer cost. The MLX partial forward
is ~12% slower per-layer than the full forward — probably mask setup or
cache indexing has fixed overhead that doesn't amortize as well when
starting from layer 18. Not a deal-breaker.

## Load time caveat

Loading the 1.3 GB LUT6 K=18 mlmodelc takes ~20 seconds on the first run
due to ANE compilation. Subsequent runs in the same session use the
compiled model cache and load in <100 ms. For production deployment,
the ANE-compiled artifact should be cached to disk (this is standard
ANEMLL practice).

## What's next

### Further optimization within this architecture
- **Chunked full-target (2 × K=18)**: eliminate MLX 18 layers entirely,
  projected target_verify ~38 ms vs current 60 ms = +58% more speedup.
  Adds a second ANE model load + inter-chunk handoff. Projected mean
  tok/s: ~77 tok/s at bf16.
- **Pre-compile caching**: avoid the 20s first-load cost by persisting
  the ANE-compiled model artifact.
- **Per-grouped-channel LUT6**: the partial target was per_tensor LUT6
  for macOS15 compatibility. Upgrading to iOS18 target + per_grouped_channel
  might preserve more accuracy while keeping ANE compile fit.

### Further strategic wins
- **RoPE-precomputation caching**: currently rebuild RoPE tables per
  cycle in Swift. Could cache across cycles since positions advance
  predictably.
- **Reduce embed→MLMultiArray conversion** by compiling embed into ANE
  chunk 1 (takes token ids directly instead of hidden).

## Reproduction

```bash
# Convert K=18 with capture indices (~3 min)
python scripts/convert_qwen3_layers_ane.py \
  --num-layers 18 --capture-indices 1,9,17 \
  --out-dir /tmp/qwen3_klayers_cap

# Build Swift runner
cd swift-bench && swift build -c release

# Run with full stack
.build/release/dflash-sd \
  --prompt "The capital of France is Paris, which is known for" \
  --max-new 100 \
  --draft /tmp/dflash_ane_accum_lut6_c/dflash_ane_accum_lut6.mlmodelc \
  --ane-lmhead /tmp/lmhead_qwen3/lmhead_lut6.mlmodelc \
  --ane-target-layers /tmp/qwen3_klayers_cap/K18/qwen3_K18_lut6.mlmodelc \
  --ane-target-k 18 \
  --ane-target-captures 1,9,17
```
