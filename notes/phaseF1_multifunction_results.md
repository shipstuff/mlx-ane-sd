# Phase F.1 multi-function ANE variant (mini-03)

**Date:** 2026-04-14
**Status:** PARTIAL -- multi-function compile works end-to-end (33 functions,
weights shared), but runtime acceptance is significantly below the
single-function accumcache baseline. Escalation trigger hit (brief:
"benchmark numbers worse than S=2048 means architecture isn't paying off").

## What was built

1. `scripts/dflash_ane_multifn.py` -- DFlash model with `write_pos` and
   `rotate` as Python-int constants at trace time. Each variant bakes in a
   specific write position, and attention scope is `write_pos + T`
   (not STATE_LEN). All slice bounds are static -> ANE-legal.
2. `scripts/dflash_coreml_convert_multifn.py` -- converts per-variant
   .mlpackages then combines via `ct.utils.MultiFunctionDescriptor` +
   `save_multifunction` into a single .mlpackage with N+1 named functions:
   `write_0`, `write_32`, ..., `write_992`, and `rotate`.
3. `scripts/phaseF1_ane_stream_multifn.py` -- runtime SD loop that
   dispatches to the right variant per cycle.
4. Compiled artifact: `/tmp/dflash_ane_multifn.mlpackage` on mini-03
   (1.03 GB, 33 functions, weights fully deduplicated).

## What worked

- **Multi-function compile** across 33 variants: all variants converted
  cleanly through coremltools. Combine via `save_multifunction` succeeded
  (~7 minutes for 33 variants -- O(N^2)-ish cost for weight-dedup).
- **Weight sharing**: combined size 1,029 MB vs. 33,833 MB for the sum of
  per-variant scratch packages. Dedup ratio 0.030 -- perfect sharing.
  Disk cost does NOT explode with N.
- **Per-function loading**: `ct.models.CompiledMLModel(mlpackage,
  function_name=name)` works as expected. Each variant loads in ~5s on
  cold cache; 33 variants take ~3 min startup.
- **Variant latency scales with attention scope** (measured as
  single-function mlpackages on synthetic inputs):
  - `write_0` (attend_len=32): 15.2 ms
  - `write_32` (attend_len=64): 15.4 ms
  - `rotate` (attend_len=1024): 19.2 ms
  - Baseline S=1024 accumcache single-function: 19.3 ms
- **Correctness on a single call**: fp32 eager-torch vs fp16 CoreML
  cosine similarity 0.9998 on `hidden` output for all three variant types
  (write_0, write_32, rotate). The model itself works.

## What did not work

The runtime SD loop gets only 8.81 tok/s mean over the standard 4 prompts
at max_new=100, vs. 21.20 tok/s for the single-function accumcache
baseline at S=1024 (same hardware). 42% of baseline. The regression is
driven entirely by acceptance collapse.

### Benchmark tables (mini-03 M4 Pro, Qwen3-4B bf16 target)

**100 tokens** (all configs, no sliding yet)

| Config              | mean tok/s | fibonacci | math  | capital | story | avg/cycle | predict ms |
|---------------------|-----------:|----------:|------:|--------:|------:|----------:|-----------:|
| accum S=1024        |     21.20  |    41.64  | 22.03 |   10.47 | 10.67 | 1.77-7.07 |     21.49  |
| accum S=2048        |     21.76  |    41.28  | 20.71 |   13.94 | 11.11 | 1.77-7.07 |     24.56  |
| **multifn S=1024**  |  **8.90**  |    10.83  | 10.97 |    7.65 |  6.15 | 1.22-1.80 |     28.02  |

**500 tokens** (S=1024 sliding since cycle 32; S=2048 sliding since cycle 64)

| Config              | mean tok/s | fibonacci | math  | capital | story | avg/cycle | predict ms |
|---------------------|-----------:|----------:|------:|--------:|------:|----------:|-----------:|
| accum S=1024        |     15.81  |    21.22  | 18.23 |   12.36 | 11.44 | 2.00-3.75 |     16.29  |
| accum S=2048        |     16.09  |    21.77  | 19.60 |   11.71 | 11.27 | 2.02-3.76 |     24.52  |
| multifn S=1024      |  not run   |        -- |   --  |      -- |    -- |        -- |         -- |

(multifn at 500 tok not run because 100-tok numbers already hit the escalation
trigger; architecture is wrong regardless of length. accumcache numbers match
within noise; single-call profile also matches notes/phaseF3_cache_sizing.md,
scaled for mini-03 hardware.)

### Why multifn loses

The root cause is an architectural mismatch between multi-function's
T-aligned `baked_wp` grid and the fine-grained `committed`-based cache
advancement used by accumcache:

- accumcache advances `write_pos` by `committed = s_real + accepted + 1`
  each cycle (1..T per cycle). Rejected-token cache slots get overwritten
  by the next cycle's T-position write that starts `committed` positions
  later. Cache stays tightly packed with valid content.
- Multifn only has variants at `baked_wp in {0, T, 2T, ...}`. There are
  two ways to reconcile:
  1. **Advance `write_pos` by T each cycle** (first attempt): cache slots
     between `committed` and `T` fill with junk K/V (zero-padded ctx or
     rejected positions). These entries remain in cache through
     subsequent cycles. Also: `global_offset` for RoPE either matches
     cycle * T (drifts from real text position, breaks relative-position
     structure when generation acceptance varies) or matches committed
     (misaligns cache layout with RoPE positions). Neither is correct.
  2. **Advance `write_pos` by `committed`, round up to pick baked_wp**
     (second attempt): Python writes valid content contiguously, but the
     variant's model attends over `[0, baked_wp + T)` which includes
     gap regions between each cycle's committed window and the next
     T-aligned boundary. The mask can mark these as `-inf`, but that
     leaves the attention softmax normalizing over a variable number of
     valid keys per cycle -- and the draft wasn't trained for that.

Both attempts land at ~1.2-1.8 tokens/cycle acceptance (vs. 1.8-7.1 for
accumcache), i.e. near the "draft random" baseline. The ANE model
itself is computing correct outputs (cos similarity 0.9998 confirms
that), so the problem is end-to-end cache/RoPE semantics, not numerical
precision.

### Why the brief's theoretical win evaporates in practice

The brief predicted multifn at S=1024 would match S=2048 quality at
S=1024 per-call latency. In principle the per-call latency does drop for
early cycles (attend_len=32..T variants are ~15 ms vs. 19 ms). But:

- At 100 tokens, the accumcache at S=1024 *already* has no sliding, so
  acceptance is already 85% of F.0 (per notes/phaseF3_cache_sizing.md).
  Multifn can at best match this, not beat.
- At 1000 tokens, the sliding fallback fires after cycle 32 regardless
  of whether it's Python-side (accumcache) or inside-model (multifn
  `rotate`). Semantics are equivalent; no accuracy win.
- The early-cycle latency gain (~4 ms per call x ~32 cycles pre-fill) is
  ~128 ms per 1000-token run, vs. the total decode time of ~40 s for
  accumcache at 1000 tok. Negligible.

## Caveats

- **mini-03 hardware runs slower than the notes' reference** (mini-02).
  accumcache at 100 tok here is 21.20 tok/s vs. notes' 32.94. Per-call
  latency here is 19 ms wall vs. notes' 16 ms wall. The architecture
  analysis above holds across hardware; only absolute numbers differ.
- **Load time** for 33 variants is ~3 min on cold cache. Not a showstopper
  for a long-running server but impractical for an interactive dev loop.
- **coremltools combine step (save_multifunction)** for N=33 variants
  took ~7 minutes and peaked at ~37 GB memory on mini-03. O(N^2) in N;
  may fail at N >= ~64 on a 64 GB machine.

## Files

- Model: `scripts/dflash_ane_multifn.py`
- Converter: `scripts/dflash_coreml_convert_multifn.py`
- Runner: `scripts/phaseF1_ane_stream_multifn.py`
- Artifact on mini-03: `/tmp/dflash_ane_multifn.mlpackage` (1.03 GB,
  33 functions: `write_0`, `write_32`, ..., `write_992`, `rotate`)

## Recommendation

Accept the escalation. The multi-function approach **does not beat
accumcache as configured in the brief**. It is technically functional
(weights share, variants compile, dispatch works) but architecturally
incompatible with DFlash's `committed`-based cache-advance semantics.

Options to revisit later:

1. **Re-purpose the multi-function toolchain for STATE_LEN variants**
   rather than write_pos variants. Compile a few variants with different
   STATE_LEN (e.g. 256, 512, 1024, 2048) and dispatch to the smallest
   variant that can hold the current history. This preserves
   accumcache's write_pos semantics (no T-alignment issue) and gives
   a pay-as-you-grow attention-cost curve without changing cache
   layout. Expect modest gains on short workloads since accum S=1024 is
   already good there.
2. **Modify the accumcache model to output
   (hidden, new_K, new_V, full_updated_cache)** and replace Python-side
   np.concatenate sliding with an in-model shift for the rotate case
   only. Avoids the T-aligned write_pos grid issue entirely. Unclear if
   it saves enough to matter (accumcache's sliding is already cheap
   in Python -- 0.16 ms/cycle).
3. **Leave accumcache as the recommended config** and focus the
   remaining research budget on LUT quantization (orthogonal 2-4x
   latency win) + the F.2 tree-speculation work.
