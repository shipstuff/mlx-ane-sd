# Phase F.1: three DFlash-on-ANE variants compared

**Date:** 2026-04-14

Three approaches to getting DFlash running on the ANE, each with different
cache architectures. All produce coherent output. Benchmarked on
Qwen3-4B bf16 target, max_new=100, greedy, same 4 prompts.

## Variant matrix

| Variant | Cache architecture | ANE placement | Mean tok/s |
|---|---|---|---:|
| **Stateless** (`phaseF1_ane_stream.py`) | None — re-process each cycle | 100% ANE ✓ | 16.46 |
| **State tensor** (`phaseF1_ane_stream_cache.py`) | In-model state_tensor, dynamic slice_update | 0% ANE — falls back to GPU | 32.78 |
| **External sliding** (`phaseF1_ane_stream_ext.py`) | Passed as I/O, static shift-and-append | 100% ANE ✓ | 16.68 |
| F.0 GPU-native MLX (reference) | Dynamic MLX KVCache | N/A (GPU) | 40.92 |

## Per-prompt breakdown

| Prompt | F.0 GPU | Stateless (ANE) | StateTensor (GPU) | ExtSlide (ANE) |
|---|---:|---:|---:|---:|
| capital | 17.76 | 13.91 | 9.75 | 11.83 |
| fibonacci | 85.38 | 16.67 | 68.84 | 27.24 |
| math | 40.48 | 21.42 | 34.35 | 14.93 |
| story | 19.05 | 13.86 | 18.18 | 12.73 |
| **mean** | **40.92** | 16.46 | 32.78 | 16.68 |

## Observations

1. **State-tensor path gives best acceptance, but not on ANE.** Writing
   fresh K/V at a dynamic `current_pos` keeps a clean, growing cache
   that matches MLX semantics closely. Acceptance recovers to 80% of
   F.0 on fibonacci. But the ANE compiler rejects `slice_update` with
   dynamic begin/end bounds (error -14 at MIL→ANEF translation). Only
   the GPU path works.

2. **External sliding cache gets ANE placement, loses context memory.**
   Shift-left-by-T, append-at-tail is all-static and lowers cleanly.
   Verified 100% ANE ops, 0 graph interruptions. But the 256-slot
   window holds only 8 cycles of history (T=32 per cycle). After that,
   earliest entries shift out. Acceptance doesn't recover like the
   state-tensor path — similar to stateless on average.

3. **Stateless and external-sliding are roughly equivalent.**
   Stateless processes each cycle fresh; external-sliding keeps ~8
   cycles of history which, for the workloads tested, adds marginal
   info. The DFlash draft is apparently trained to benefit from
   LONGER history than 8 cycles.

## The fundamental tradeoff

Getting DFlash to run on the ANE requires static-bound cache updates.
But dynamic-offset cache updates are what the draft was trained to use.
The three options we've tried resolve this differently, none fully
recovers F.0 performance:

- Drop cache → can't match trained expectations (50% of F.0)
- Use cache, accept GPU fallback → gets trained behavior but on GPU (80% F.0)
- Use static cache via sliding window → ANE works but loses context (41% F.0)

## Bottlenecks to push through

1. **State_tensor ANE compatibility** — either a coremltools / Core ML
   update fixes dynamic slice_update on ANE, or we find a way to
   express the same semantics with static slicing. The ANEMLL approach
   for Qwen3 (single-token decode with fixed `[pos:pos+1]` slice)
   works, but DFlash writes multi-token `[pos:pos+32]`. Might need
   separate compiled functions for each `pos mod STATE_LEN` value.

2. **Larger sliding window** — bump STATE_LEN from 256 to 1024 or
   higher. Gives 32+ cycles of history. Cost: 4× cache I/O per call
   (~20 MB instead of 5 MB). Easy to test.

3. **Multi-function model** — compile N versions of the model with
   different fixed `pos` values baked in. Caller dispatches to the
   right one based on current cache offset.

## Files added this iteration

- `scripts/dflash_ane_cache.py` — state_tensor variant model
- `scripts/dflash_coreml_convert_cache.py` — converter for it
- `scripts/phaseF1_ane_stream_cache.py` — runner (GPU fallback)
- `scripts/dflash_ane_slidecache.py` — in-model sliding state_tensor (also fails ANE)
- `scripts/dflash_coreml_convert_slide.py` — converter
- `scripts/dflash_ane_extcache.py` — external-cache I/O-threaded variant
- `scripts/dflash_coreml_convert_ext.py` — converter (CLEAN ANE place)
- `scripts/phaseF1_ane_stream_ext.py` — runner (100% ANE)

## Session summary

**What we know works end-to-end:**
- Three variants of DFlash-on-ANE compile, load, run, produce coherent text
- Two of them (stateless, external-sliding) get full 100% ANE placement
  with 0 CPU ops and 0 graph interruptions — confirmed via anemll-profile
- Per-call ANE latency is ~8-10 ms for the 0.5B draft at block_size=16
- PyTorch→CoreML numerical parity is preserved (cos_sim 0.99 vs
  standard-RMSNorm reference)

**What's still blocked:**
- Full acceptance-rate parity requires dynamic cache offsets. Those
  don't lower to ANE. Workaround: larger static window, or
  multi-function. Both are week-scale follow-on work.

**Net result across variants:** best ANE variant is currently 41% of
GPU baseline. Best non-ANE variant (GPU-fallback state_tensor) is 80%.
The gap is the lost history in the sliding window.
