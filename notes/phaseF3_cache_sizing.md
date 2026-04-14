# Phase F.3: cache-size scaling for long generations

**Date:** 2026-04-14

## The problem F.3 targets

F.1 accumulating cache has a sliding-window fallback when
`write_pos + T > STATE_LENGTH`. Once sliding kicks in, earliest cache
entries are discarded forever, and acceptance rate drops sharply.

Originally framed as "multi-function model with N baked-in write
positions" (ANEMLL-style). Tested first whether the simpler fix —
just bigger STATE_LENGTH — suffices.

## Results across lengths and cache sizes

F.1 accum ANE, unquantized, all 100% on ANE:

| Length | F.0 | S=1024 | S=2048 | S=4096 | Winner |
|---:|---:|---:|---:|---:|---|
| 100 | 40.92 | **32.94 (85%)** | 30.79 (80%) | 27.40 (70%) | S=1024 |
| 500 | 29.11 | **24.33 (84%)** | 22.79 (79%) | 20.36 (70%) | S=1024 |
| 1000 | 32.58 | 14.20 (44%) | **25.71 (79%)** | 22.99 (72%) | S=2048 |

Per-call latency (measured):
- S=1024: 13 ms (profile), ~16 ms wall
- S=2048: 24 ms (profile), ~24 ms wall
- S=4096: 28 ms (profile), ~38 ms wall

## Analysis

Two forces compete:

1. **Per-call latency** grows roughly linearly with cache size. Attention
   computes Q × K over `STATE_LEN + T` positions. Going from S=1024 to
   S=2048 adds ~50% attention work.
2. **Sliding penalty** kicks in at `STATE_LEN / T` cycles. At T=32:
   - S=1024 → sliding at cycle 32
   - S=2048 → sliding at cycle 64
   - S=4096 → sliding at cycle 128

At short lengths (≤ ~16 cycles), no sliding fires at any S. So the
smallest S with lowest latency wins: S=1024.

At long lengths (≥ 100 cycles) sliding fires in all S. S=2048 wins
because per-call latency is still lower than S=4096, and both lose
history to sliding eventually.

## S=4096 is NOT strictly better than S=2048

Initially I expected "bigger cache = always better at long gen." Not
true — at 1000 tokens, S=2048 beats S=4096 by +12%. Reason: both hit
sliding (sliding starts at ~64 cycles for S=2048, ~128 for S=4096, but
story goes 500+ cycles for 1000 tokens), so neither preserves full
history. S=2048 wins on per-call cost.

Implication: there's a sweet-spot STATE_LEN for each workload. For
100-1000 tokens: S=2048. For >1000 tokens: would need larger S or
true multi-function dispatch.

## Recommendation

- **Short workloads (≤ 500 tokens)**: use S=1024 (best case)
- **Mixed workloads (500-1500 tokens)**: use S=2048 (best overall)
- **Long workloads (> 1500 tokens)**: benchmark needed; may still want S=2048

A runtime-dispatch approach that starts with S=1024 and upgrades to
S=2048 when cache nears capacity is possible but adds complexity. The
gain at 1000 tokens is from 24.3 (S=1024 crash) to 25.7 (S=2048) —
modest. Static S=2048 as a safe default captures most of the value.

## Where F.1 accum currently sits vs F.0

With the right cache size per regime:

| Workload | F.0 GPU | F.1 ANE (best S) | F.1/F.0 |
|---|---:|---:|---:|
| 100 tok | 40.92 | 32.94 (S=1024) | 85% |
| 500 tok | 29.11 | 24.33 (S=1024) | 84% |
| 1000 tok | 32.58 | 25.71 (S=2048) | 79% |

Consistent 79-85% of F.0 across all realistic generation lengths. The
remaining 15-21% gap is the ANE fp16-compute tax (slightly lower
acceptance rate per cycle) + Python/CoreML bridge overhead.

## Files

No new script files. Added to existing pipeline via configurable
`--state-length` flag on `phaseF1_ane_stream_accum.py` and 
`dflash_coreml_convert_accum.py`. Compiled variants at:

- `/tmp/dflash_ane_accum_1024_c/dflash_ane_accum_1024.mlmodelc` (short)
- `/tmp/dflash_ane_accum_2048_c/dflash_ane_accum_2048.mlmodelc` (default)
- `/tmp/dflash_ane_accum_4096_c/dflash_ane_accum_4096.mlmodelc` (very long, probably suboptimal)

## What a proper multi-function dispatcher would do

If we wanted to unlock longer generations without sliding loss, the
multi-function pattern would compile N variants with different fixed
cache write positions. Python tracks current write_pos and dispatches
to the variant that handles that position. Implementation sketch:

- Compile models with write_pos ∈ {0, 64, 128, ...} baked in as
  constants in the model (not cache-size changes)
- Each model has same STATE_LEN but writes at the fixed pos
- Python manages which variant to call each cycle
- Effectively: a "permanent accumulating" cache with no sliding

This is larger engineering (~3-5 days) for gains mostly at
>1500-token generations. Not pursued this iteration.
