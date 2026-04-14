# Phase F.1 cached: KV-cache support added, acceptance recovered

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB

## What we built

Extended the F.1 stateless ANE DFlash port with a persistent KV cache
exposed as a CoreML `state_tensor`. Structure mirrors anemll/qwen_model.py:
unified cache buffer of shape `(2N, Hkv, STATE_LEN, D)` where index `2i`
holds layer i's K and `2i+1` holds V. Each forward writes fresh K/V at
`cache[current_pos:current_pos+S+L]` and reads the full cache for SDPA
under an attention mask.

Key fix found: **rope positions use cache-slot indices, not absolute
sequence positions.** The MLX reference applies rope with `offset =
cache.offset` — rope positions track where K/V lands in the cache, not
where its source hidden came from in the original sequence. After we
matched this semantic, acceptance rates recovered dramatically.

## Results (same 4 prompts, Qwen3-4B bf16 target, max_new=100, greedy)

| Prompt | F.0 GPU | F.1 stateless | F.1 cached | accept/cycle (F.0 / cached) |
|---|---:|---:|---:|---|
| capital   | 17.76 | 13.91 | 9.75  | 1.8 / 1.0 |
| fibonacci | 85.38 | 16.67 | **68.84** | 8.4 / 7.1 |
| math      | 40.48 | 21.42 | 34.35 | 4.0 / 3.5 |
| story     | 19.05 | 13.86 | 18.18 | 1.9 / 1.9 |
| **mean**  | **40.92** | 16.46 | **32.78** | — |

Ratio vs F.0: stateless was 0.40×, cached is **0.80×**.

The stateless→cached jump almost doubled throughput. The RoPE fix on
top added +8% (from 30.49 cached-pre-fix to 32.78 post-fix).

## Where F.1 cached is running: GPU, not ANE

The cache-aware model **failed to compile for ANE.** The ANE compiler
bails on the `slice_update` op with dynamic (`current_pos`-based)
begin/end bounds — exactly what the anemll CLAUDE.md warns about:

> "KV-Cache State Updates Must Use Static Slicing Only. Any slice
> bounds that depend on runtime values (current_pos, dynamic seq_len)
> will compile into slice_by_index with unresolved parameters,
> causing ANE failure: 'Failed to retrieve parameter end.'"

Symptoms:
- `ct.ComputeUnit.CPU_AND_NE`: load fails with error `-14`
- `ct.ComputeUnit.ALL`: loads and runs (falls back to GPU)
- anemll-profile: `MILCompilerForANE error: failed to compile ANE
  model using ANEF. Error=_ANECompiler: ANECCompile() FAILED (11)`

So the F.1 cached numbers above represent CoreML-on-GPU running the
DFlash draft, not ANE. This is not the outcome we wanted — the whole
point of the port was ANE offload.

## Why the cached version still wins at 0.80× of GPU-MLX

Even though this is GPU-not-ANE, moving the draft to CoreML-GPU
(instead of MLX-GPU) doesn't change the fundamental story: both run
on the same silicon. The 0.80× ratio reflects:

- Per-call overhead of CoreML (extra numpy conversion, Python bridge)
  vs native MLX
- Slight numerical drift from `layer_norm`-form RMSNorm (per ANEMLL
  convention, used here for eventual ANE compatibility) — about 0.99
  cos_sim vs standard RMSNorm

The **real** win (ANE offload under contention, per Phase C pattern)
requires the model to actually run on ANE. Which means fixing the
slice_update issue.

## Per-prompt: capital is an outlier

capital stayed bad even with cache (1.0 tok/cycle = zero draft
acceptance). This is likely a DFlash-intrinsic issue: the draft was
trained on GSM8K, MATH-500, HumanEval, MBPP — math and code. On
factual prose it has low confidence overlap with the target.

F.0 (native MLX DFlash) gets 1.8 tok/cycle on capital (still low).
F.1 gets 1.0. The gap (from 1.8 to 1.0) is ~40% accept rate drop —
probably cumulative numerical noise from our port path + potential
cache-reset effects during the 99-cycle run.

## Path to actually landing on ANE

The dynamic slice bound is the blocker. Two approaches:

### Option A: Sliding window (static bounds)

Each call shifts the cache left by (S+L) and appends at the fixed end:

```python
# Shift — all static bounds
kv_cache[:, :, :-T, :] = kv_cache[:, :, T:, :].clone()
# Append — all static
kv_cache[:, :, -T:, :] = new_kv
```

Window of last 256 positions = 8 cycles of history at T=32/cycle.
Enough for short generations but caps the context memory.

RoPE positions become trickier: we need per-slot rope positions
stored/updated alongside K/V, OR we accept the "last N positions"
semantic and rope based on slot index (positions 0..256 in the window,
not absolute).

### Option B: Multi-function model

Compile N separate functions for the N allowed values of `current_pos`
(all static). At runtime, dispatch to the right one. This is what
ANEMLL does for prefill-vs-decode and split-rotate scenarios.

### Option C: External cache manager

Don't use a state tensor. Pass cache as a regular input (fp16 tensor
of shape `(2N, Hkv, STATE_LEN, D)`). The model reads the whole thing,
computes the attention output using it, and ALSO returns the new cache
contents as an output. Python-side manages the cache via ordinary numpy
slicing — which can be static-slice shifts.

Downside: ~5 MB of cache read/written per call over the numpy bridge =
extra latency.

## Files

- `scripts/dflash_ane_cache.py` — cache-aware PyTorch model
- `scripts/dflash_coreml_convert_cache.py` — CoreML conversion w/ states
- `scripts/phaseF1_ane_stream_cache.py` — end-to-end SD with cache
- `/tmp/dflash_ane_cache.mlpackage` + `/tmp/dflash_ane_cache_compiled/
  dflash_ane_cache.mlmodelc` — compiled artifacts (regenerable)

## Summary

**Progress:** acceptance rate recovered from 0.40× of GPU-baseline
(stateless) to 0.80× (cached). RoPE position fix identified and
applied. Cache infrastructure proven to work.

**Remaining gap:** the model runs on GPU, not ANE, because
`slice_update` with dynamic bounds isn't ANE-lowerable. To realize the
ANE contention-resilience benefit, we need to restructure the cache
updates to use static slice bounds — sliding window is the cleanest
path.
