# Phase F.1 accumulating cache: the right design

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB

## The design

The first three F.1 variants hit a wall:
- **Stateless**: 40% of F.0 — no inter-cycle memory
- **In-model state_tensor**: 80% of F.0 but runs on GPU (ANE compile fails
  on dynamic slice_update)
- **External sliding**: 41% of F.0 — static-bound cache updates work on ANE,
  but sliding window continually discards history

The fix: **externalize the cache entirely, manage it in Python with
MLX-style trim semantics.** The ANE model takes cache as a regular input
(no state_tensor), uses `cat(cache, new_K)` for attention (static shape),
and returns `(hidden, new_K, new_V)` so the caller can write the cache
with knowledge of which positions were actually committed.

Python-side: write `T = S + L = 32` fresh K/V into cache at `write_pos`
each cycle, but advance `write_pos` by only `s_real + accepted + 1`
(committed count). Next cycle overwrites the rejected positions. This
matches MLX's `trim_prompt_cache` behavior.

When `write_pos + T` would overflow STATE_LEN=256, fall back to sliding
(shift-left + append). For 100-token generations this happens late in the
run.

## Results

All three comparisons on same 4 prompts, Qwen3-4B bf16 target, max_new=100,
greedy decode.

### Solo

| Prompt | F.0 GPU | F.1 accum ANE | Ratio |
|---|---:|---:|---:|
| capital | 17.76 | **17.18** | 97% |
| fibonacci | 85.38 | 68.71 | 80% |
| math | 40.48 | 34.36 | 85% |
| story | 19.05 | **18.47** | 97% |
| **mean** | **40.92** | **34.68** | **85%** |

### Under moderate contention (gemma-270m bg)

| Config | Solo | Parallel | Slowdown | Retained |
|---|---:|---:|---:|---:|
| F.0 GPU | 40.92 | 32.79 | -20% | 80% |
| F.1 accum ANE | 34.68 | 28.83 | -17% | 83% |

### Under heavy contention (Qwen3-4B bg, same size as target)

| Prompt | F.0 | F.1 accum ANE | Winner |
|---|---:|---:|---|
| capital | 9.21 | 9.53 | **F.1 +3.5%** |
| fibonacci | 45.14 | 39.30 | F.0 +15% |
| math | 20.81 | 19.23 | F.0 +8% |
| story | 9.66 | 10.43 | **F.1 +8%** |
| **mean** | **21.21** | 19.62 | F.0 +8% |

F.1 wins on 2 of 4 prompts under heavy contention; overall F.0 stays
marginally ahead. The prose/factual prompts where acceptance is low
are where F.1 already pulls ahead because there's less "amortization
benefit" from F.0's GPU residency.

## Why fibonacci and math stay F.0-favored

DFlash on fibonacci/math gets high draft acceptance (56-27%). In
F.0, each cycle amortizes weight bandwidth across ~8 tokens —
bandwidth utilization multiplies speedup. F.1 ANE has per-cycle
overhead (numpy cache copy + CoreML bridge) that doesn't shrink at
high-acceptance rates. So F.0 widens its lead on these prompts.

F.1 wins on low-acceptance prompts (capital, story) because their
per-cycle cost is dominated by verify, not draft bandwidth amortization.
There F.1's ANE offload just buys raw GPU time for the target.

## Why this matters

1. **First competitive ANE port of a block-diffusion SD draft.** 85% of
   native-MLX throughput with 100% ANE placement, 0 CPU ops, 0 graph
   interruptions. The port is real.
2. **Contention-resilience confirmed for this regime.** F.1's solo→parallel
   slowdown is -17% vs F.0's -20%. At heavier contention the curves
   converge and F.1 starts beating F.0 on low-acceptance prompts.
3. **Paved road to F.2.** Tree speculation (DDTree proper) now has a
   working ANE-hosted draft to build on. Tree branching adds parallel
   candidates on ANE's batch=64 path without touching the cache
   machinery we just solved.

## Implementation notes

Key files:
- `scripts/dflash_ane_accumcache.py` — model definition
- `scripts/dflash_coreml_convert_accum.py` — coremltools conversion
- `scripts/phaseF1_ane_stream_accum.py` — Python-side cache mgmt + SD loop
- `scripts/phaseF1_contention_accum.py` — contention harness

Cache architecture reminder for future-us:
- Cache is a regular model input, not a state_tensor
- Cache shape: `(N, Hkv, STATE_LEN, D) = (5, 8, 256, 128)` = ~5 MB fp16
- Model attends over `cat(cache, new_K)` — total `STATE_LEN + T = 288` positions
- Mask suppresses `[write_pos, STATE_LEN)` of cache (unwritten positions)
- Python accumulates for first 256-T cycles, slides after

Model size: 1025 MB unquantized fp16. LUT4/6 quantization (ANEMLL-style)
would shrink to ~260 MB and likely reduce per-call latency, not yet tried.

## Paths forward (ranked)

1. **LUT quantization** on the DFlash weights — shrink model, potentially
   reduce latency. ANEMLL's existing toolchain handles this. Low risk,
   potentially meaningful wall-clock improvement on ANE (current is
   3.5 TOPS out of 38 TOPS peak).

2. **F.2: DDTree tree branching** — generate κ candidate paths per cycle
   using the ANE batch=64 path. Original plan. Should stack on the
   accumulating cache architecture.

3. **Measure combined fg+bg throughput more precisely.** Record bg
   tok/s alongside fg under various loads to quantify "2-workload
   pair wins for F.1 vs F.0."

4. **Multi-function for cycles >8** to eliminate the sliding-window
   fallback. The accumulating phase is fine; the sliding phase loses
   context. Multi-function could give full MLX cache semantics
   indefinitely.
