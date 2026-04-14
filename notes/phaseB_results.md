# Phase B: state snapshotting & concurrency — findings

**Date:** 2026-04-13
**Hardware:** Mac mini M4 Pro, 64 GB

## Summary

**Phase B.1 (snapshot-based, no re-prefill) was the big win: +29% over Phase A.**
Phase B.2 and B.3 added concurrency but concurrency **did not** further
improve throughput for this setup. For small-draft + large-target SD on
Apple Silicon, the benefit of running draft on ANE vs GPU is negligible
because draft is already tiny relative to target verify.

## Phases compared

| Phase | Approach | Throughput (K=12, mean) |
|---|---|---:|
| A | Sequential, re-prefill every cycle | 16.45 tok/s |
| B.1 | Sequential, snapshot committed state | **21.19 tok/s** |
| B.2 | Concurrent + cross-cycle speculation | 21.01 tok/s |
| B.3 | Concurrent (worker does K+1 steps), no cross-cycle spec | 20.68 tok/s |

Baseline (target-only): ~9 tok/s.

### Sweep across num_draft (mean over 3 diverse prompts)

| num_draft | B.1 | B.2 | B.3 |
|---:|---:|---:|---:|
| 4 | 13.63 | 13.91 | 13.46 |
| 6 | 16.83 | 17.15 | 16.62 |
| 8 | 18.44 | — | 18.20 |
| 10 | 19.92 | 20.14 | 19.50 |
| **12** | **21.19** | 21.01 | 20.68 |
| 16 | 20.39 | 20.24 | 20.10 |

All three phases peak at num_draft=12 on Gemma-3-12B bf16 + ANE 270M. B.2
and B.3 are within noise of B.1.

## Why concurrency didn't help

The hypothesis was that hiding draft time under verify time would give
another 20% speedup (since draft was 20% of total in B.1). The actual
reason it didn't:

1. **Draft and verify are fundamentally sequential within a cycle**: verify
   needs the draft's tokens as input, so verify can't start until draft
   completes.
2. **Cross-cycle speculation (B.2) fails often**: starting cycle N+1's
   draft before verify of cycle N completes requires guessing the bonus
   token. ANE's guess matches target's ~35% of the time (full-accept rate
   × per-token accept rate). On miss, we waste the speculative draft.
3. **Main-thread serial overhead eats the gain**: step_sync for finalize,
   snapshot read, submit/collect synchronization each add ~3-5 ms per
   cycle. Over 5-10 cycles that's 30-50 ms of serial overhead that didn't
   exist in B.1.
4. **Draft is not the bottleneck**: on a ~2s wall-clock run, draft is ~400-600
   ms. Even hiding it entirely would save <30% of total time. But with
   partial-reject recoveries and serial overhead, we net out.

## What Phase B proved (and what it didn't)

**Did:**
- Correctness of cross-framework SD: ANE draft + MLX target produces
  identical output to target-only decode across num_draft values
- MLState snapshot/restore works bit-exactly
- The architecture composes cleanly — ANE draft is a drop-in replacement
  for an MLX draft in the SD loop
- Snapshot-based state handling is feasible and low-overhead (<20 ms total
  per 50-token generation)
- Threading model works: coremltools predict() and MLX forward() both
  release the GIL and run concurrently on separate silicon

**Did not:**
- Heterogeneous SD does not beat pure-MLX SD for this small-draft setup
  (pure-MLX SD at K=12: 23.2 tok/s mean; our B.1 at K=12: 21.19 mean —
  we're at 91% of pure-MLX)
- Concurrency did not add meaningful throughput; the bottleneck is
  verify time, not draft time

## Where heterogeneous ANE+MLX SD likely DOES win

The setups where moving draft to ANE should beat pure-MLX SD (not tested
here):

1. **Larger drafts (2B+)** — GPU draft time starts mattering; moving it off
   GPU frees target bandwidth
2. **Long generations at high num_draft** — draft time accumulates; ANE's
   separate silicon runs in parallel without contending
3. **Multi-workload scenarios (Exp E pattern)** — when the GPU is busy
   with another task (second model, graphics), ANE draft preserves SD
   speedup for the primary model
4. **Power-constrained inference** — ANE at ~5 W vs GPU at ~14 W; for
   24/7 agents doing SD, total energy-per-token drops meaningfully
5. **Multi-mini clusters** — each mini has an ANE; distributing drafts
   across minis via the RDMA interconnect could scale

None of these are tested by this single-prompt, single-user benchmark.

## Absolute numbers and speedups

| Setup | tok/s | Speedup vs 9.1 baseline |
|---|---:|---:|
| Target-only (Gemma-3-12B bf16) | ~9.1 | 1.00× |
| Pure-MLX SD (mlx-lm builtin, K=12) | ~23.2 | 2.55× |
| Our heterogeneous SD B.1 (K=12) | 21.19 | 2.33× |
| Our heterogeneous SD best case | 24.87 | 2.73× |
| dflash-mlx on Qwen3-4B bf16, math@4028 | ~40 | 4.6× |

We're at 91% of pure-MLX SD. Competitive but not a clear win for this
specific benchmark shape.

## Files

- `scripts/phaseA_ane_draft_mlx_target.py` — Phase A (re-prefill)
- `scripts/phaseB_sequential_optimized.py` — Phase B.1 (snapshot, sequential)
- `scripts/phaseB2_concurrent.py` — Phase B.2 (concurrent + spec)
- `scripts/phaseB3_all_worker.py` — Phase B.3 (concurrent, no spec)

## Next

The question after Phase B is not "can we make this faster on a single
stream" — we've hit diminishing returns there. The more interesting
questions are:

1. **Power**: does ANE draft actually reduce energy per token?
   `sudo powermetrics --samplers cpu_power,gpu_power,ane_power` on our SD
   run vs pure-MLX SD. Expected: meaningful reduction.
2. **Multi-workload**: run the same heterogeneous SD while ALSO running
   another MLX model on the GPU. Expected: our speedup holds, pure-MLX
   SD would not.
3. **Larger draft**: train or find a 2-4B Gemma 3 draft, compare. Expected:
   ANE draft starts to win because GPU draft would steal target bandwidth.
4. **Mirror-SD bidirectional protocol**: the actual Apple paper approach.
   Likely gives another 10-30% on top of B.1.
