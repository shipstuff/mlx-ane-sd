# Phase G: multi-stream SD serving findings

**Date:** 2026-04-14 (Week 1)
**Hardware:** Mac mini M4 Pro (mini-02), 64 GB, M4 Pro
**Target:** Qwen3-4B bf16 per-stream
**Draft:** Qwen3-4B-DFlash-b16 (F.0 GPU) / ANE port (F.1)

## Summary

Ran N-stream concurrent SD serving across F.0 (DFlash GPU-only) and F.1
(DFlash ANE-hosted draft). Tested solo, under moderate contention
(gemma-270m bg), and heavy contention (Qwen3-4B bg). **The Phase-C
preservation advantage of F.1 doesn't translate to a decisive
multi-stream win for block-diffusion drafts.**

F.0 keeps winning on total throughput at every N where it runs (N≤6).
F.1's advantage is narrow: +3-5pp better per-stream efficiency, and
the ability to run at N=8 where F.0 OOMs at our 64 GB.

## Solo (no bg contention)

| Mode | N=1 | N=2 | N=4 | N=6 | N=8 |
|---|---:|---:|---:|---:|---:|
| F.0 total | 67.75 | 85.61 | 94.03 | 95.95 | OOM (3/8 ran) |
| F.1 total | 51.95 | 68.76 | 78.11 | 80.16 | 23.69 |
| Ratio (F.1/F.0) | 77% | 80% | 83% | 84% | — |

F.0 plateaus at ~95 tok/s around N=4-6. F.1 plateaus at ~80 tok/s. At
N=8 F.0 fails memory-wise; F.1 runs but heavy contention on the ANE
daemon (serialized draft calls across 8 processes) drops throughput
substantially.

## Moderate contention (gemma-270m bg)

| Mode | N=1 | N=2 | N=4 | Per-stream eff @N=4 |
|---|---:|---:|---:|---:|
| F.0 | 55.90 | 74.07 | 86.57 | 34.7% of solo |
| F.1 | 43.41 | 61.54 | 71.79 | 37.5% of solo |
| bg tok/s | ~140 | ~113 | ~83 | (degrades with N) |

F.1 preserves slightly better (37.5% vs 34.7% per-stream efficiency).
bg throughput is nearly identical across modes (F.1's ANE offload
doesn't meaningfully free GPU for bg either).

## Heavy contention (Qwen3-4B bg)

| Mode | N=2 total (SD only) | N=4 total (SD only) | SD+bg @N=4 |
|---|---:|---:|---:|
| F.0 | 55.76 | 70.26 | 83.76 |
| F.1 | 47.92 | 60.15 | 72.80 |
| F.1/F.0 | 86% | 86% | 87% |

F.0 continues to win on total. Heavy contention doesn't flip the
ranking. Both modes degrade the bg workload similarly (~13 tok/s).

## Why F.1 doesn't win decisively here

**DFlash's draft is already bandwidth-amortized.** One forward per cycle
processes all 16 block positions in parallel (batch=16). On GPU this
takes ~5-10 ms. Moving it to ANE saves ~5-10 ms per cycle.

This is much smaller than what the Phase C (Gemma-270M) autoregressive
draft saved: there the draft was K=12 sequential steps of ~5 ms each =
~60 ms per cycle. Offloading that to ANE saved substantially more
proportionally.

For DFlash-class block-diffusion drafts, the ANE offload advantage is
genuine but small:
- Solo: 85% of F.0 throughput (draft savings not enough to flip)
- Contention: 3-5pp better preservation (Phase C is real but diminished)
- Total: F.0 still wins at all tested N×bg combinations

## Where F.1 does win

1. **Memory scaling**: at 64 GB, F.0 fails at N=8 (each worker loads its
   own draft, 8 × ~9 GB ≥ available memory). F.1 shares the compiled
   CoreML draft on disk — all workers load the same mlmodelc without
   duplicating weights in MLX memory. F.1 runs N=8; F.0 doesn't.

2. **Energy efficiency (not measured here)**: ANE is lower-power than
   GPU. For continuous agent workloads, F.1 likely gives better
   tokens-per-joule. Deprioritized by user preference.

3. **Frees GPU headroom** (marginally): F.1's GPU usage per SD cycle
   is lower, so theoretically leaves more for other GPU workloads.
   Our contention tests show this effect is small in practice (bg
   throughput ~same in both F.0 and F.1 contention tests).

## Implications for the paper

The Week-1 hypothesis ("F.1 decisively wins under multi-stream or
contention") did NOT hold for block-diffusion drafts on this hardware.
The paper narrative needs to pivot:

**From:** "Heterogeneous ANE offload is the winning architecture for
concurrent SD serving."

**To:** "Heterogeneous ANE offload is a viable alternative that
unlocks higher concurrency on memory-constrained hardware, at a small
(~15-20%) throughput cost relative to GPU-only. For autoregressive
drafts (Phase B/C), the ANE advantage is more pronounced; for
block-diffusion drafts (DFlash), the draft-side savings are smaller
because the draft is already parallelized on GPU."

Still publishable but a different story. The Phase B/C autoregressive
findings remain valid; this just characterizes where the ANE advantage
does/doesn't transfer.

## What would change the story

1. **Swift native runner** (Week 2) — eliminates Python/CoreML bridge
   (~5 ms/call). Could flip F.1 solo ≥ F.0 solo, strengthening the
   solo-competitive case.
2. **Larger target** (e.g., Qwen3-14B if drafts exist) — memory
   bandwidth dominates, ANE's reduced GPU load matters more.
3. **Larger draft** (e.g., 2B+) — draft work becomes meaningful
   fraction of cycle; offloading matters more.
4. **Energy/power analysis** (deprioritized) — F.1 may win
   tokens-per-joule even if losing tokens-per-second.

## Files

- `scripts/multistream_worker.py` — one-stream SD subprocess
- `scripts/phaseG_multistream.py` — N-stream orchestrator with optional
  bg workload; sync-start barrier, true-total throughput measurement

## Remaining Week 1 work

- Wait for mini-03 agents (LUT grid, multi-function) to finish — may
  provide data that shifts F.1 numbers up
- Consider narrative pivot given these results
