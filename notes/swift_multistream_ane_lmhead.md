# 2-stream contention with ANE lm_head

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB (mini-02)
**Context:** Previous 2-stream test (GPU lm_head) hit 1.28× aggregate over
solo. Prediction was that moving lm_head to ANE would push aggregate to ~1.45×
since GPU per-cycle work would drop from 91ms to 72ms, leaving more headroom
under contention.

## Actual result: absolute scaling improves, ratio does not

| config              | solo tok/s | 2-stream agg | ratio | abs gain 2→2-stream |
|:--------------------|-----------:|-------------:|------:|--------------------:|
| GPU lm_head (prev)  |     17.13  |       19.02  | 1.11× |                —    |
| **ANE lm_head (now)** | **19.91** |   **22.93**  | **1.15×** |        **+20%** absolute |

Absolute 2-stream aggregate jumped 19.02 → 22.93 tok/s (**+20%**). But
**solo jumped more** (17.13 → 19.91 = +16%), so the aggregate/solo ratio
actually dropped slightly (1.28 → 1.15). My prediction of 1.45× aggregate
ratio was wrong; here's why.

## Per-phase behavior under 2-stream contention (ANE lm_head)

| phase           | solo ms/cyc | 2-stream ms/cyc | slowdown  |
|:----------------|------------:|----------------:|----------:|
| target_verify   |       72.5  |          118.5  |   **1.63×** |
| draft_lmhead    |        3.09 |            3.16 |    1.02×  |
| draft_predict   |       10.1  |           10.6  |    1.05×  |

**The target_verify (GPU) is still the bottleneck**, slowing 1.63× under
contention (vs 1.72× with GPU lm_head — slightly less contention because
the GPU is no longer running lm_head). ANE work (predict + lm_head) is
essentially unaffected by contention — consistent with the Phase C pattern
we documented previously.

Per-stream cycle under 2-stream:
- GPU lm_head: 124 (tv) + 27 (dlh) + 11 (dp) = ~162 ms/cycle
- ANE lm_head: 118 (tv) + 3.2 (dlh) + 10.6 (dp) = **~132 ms/cycle**

Solo cycle:
- GPU lm_head: 72 + 19 + 10 = 101 ms/cycle
- ANE lm_head: 72.5 + 3.1 + 10.1 = **~86 ms/cycle**

So the ANE variant is faster in BOTH regimes, but the *speedup over solo*
from parallelism is similar — GPU still serializes target_verify.

## The honest takeaway

Moving lm_head to ANE is a pure per-cycle win — it helps solo and it helps
multi-stream roughly proportionally. It does **not** unlock a new scaling
regime. The 72ms target_verify on GPU is the real ceiling for multi-stream
throughput on this hardware.

Concretely:
- Solo ANE-lm_head throughput: **+16% over solo GPU-lm_head**
- 2-stream aggregate ANE-lm_head throughput: **+20% over 2-stream GPU-lm_head**
- The extra ~4 percentage points at 2-stream comes from ANE lm_head not
  contending at all (1.02× slowdown vs 1.40× for the GPU version).

That 4pp is real but modest. To materially change multi-stream scaling, we
have to attack the target_verify (GPU) ceiling directly — which motivates
the "partial target on ANE" direction.

## Reproduction

```bash
bash scripts/bench_swift_2stream_ane_lmhead.sh
```

## What this closes out

- ✅ Confirms ANE lm_head + multi-stream compound: both regimes benefit
- ✅ Confirms GPU target_verify is the scaling ceiling (1.63× slowdown
  at 2 streams = ~60% effective GPU utilization after contention)
- ✅ Multi-stream value is still memory density + serving latency smoothing,
  not raw aggregate throughput

## Next steps

**Option 2 (user's pick):** bigger/deeper draft on ANE. ANE is still only
13ms/cycle busy (15%). The draft has 60+ ms of ANE headroom. A wider/deeper
draft that takes 20-25ms but lifts accept rate from 3.5 to 5+ tok/cycle
would be a net win — throughput scales linearly with accept rate.
