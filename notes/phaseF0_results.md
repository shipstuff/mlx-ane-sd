# Phase F.0: DFlash GPU-only baseline on M4 Pro (Qwen3-4B bf16)

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB
**Target:** `mlx-community/Qwen3-4B-bf16` (36 layers, 2560 hidden, GQA 32/8)
**Draft:** `z-lab/Qwen3-4B-DFlash-b16` (5 layers, block_size=16, target_layer_ids=[1,9,17,25,33])
**Runner:** z-lab's own `dflash/model_mlx.py` stream_generate (native MLX)
**Decode:** greedy (temp=0), max_new=100

## Results

| Prompt | target-only | DFlash | speedup | cycles | accept_rate |
|---|---:|---:|---:|---:|---:|
| capital    | 27.66 | 17.76 | 0.64× | 56 | 12% |
| fibonacci  | 27.43 | 85.38 | **3.11×** | 12 | 56% |
| quantum    | 27.58 | 22.09 | 0.80× | 46 | 15% |
| recipe     | 27.53 | 25.73 | 0.93× | 39 | 17% |
| story      | 27.56 | 19.05 | 0.69× | 53 | 13% |
| math       | 27.45 | 40.48 | 1.47× | 25 | 27% |

**Mean target-only:** 27.53 tok/s
**Mean DFlash:** 35.08 tok/s
**Mean speedup:** 1.28× (min 0.64×, max 3.11×)

## Interpretation

DFlash is **net slower on 4/6 prompts**. It only wins on structured
workloads (code, math). On prose/factual prompts the per-cycle overhead
(target verify at batch=16, draft forward, cache management, sampling)
isn't amortized because acceptance rates are too low (12-17%).

Per-cycle timing estimate: ~75-95 ms per cycle regardless of acceptance.
Target-only at 27.5 tok/s is 36 ms/token. A DFlash cycle needs to yield
at least ~2.5 tokens on average to beat target-only, and at least ~5
tokens/cycle for a meaningful speedup.

This matches CLAUDE.md's recorded dflash-mlx results on M4 Pro: 1.53×
mean at bf16, with math@4028 peaking at 4.06× and prose@512 underwater.
Our 1.28× is slightly below CLAUDE.md's 1.53× — likely because
max_new=100 doesn't amortize cycle overhead as well as longer sequences.

## What this tells us about F.1

The F.1 hypothesis is: moving the draft off the GPU to the ANE reduces
per-cycle overhead by the draft-forward time (plus whatever GPU-contention
benefit we get under load).

Rough accounting:
- Per-cycle GPU work = target verify @ batch=16 + draft forward @ batch=16
- Target verify dominates (Qwen3-4B is ~8× bigger than draft)
- Draft forward on GPU: ~5-10 ms estimated
- ANE draft forward: ~2-5 ms + ~1 ms memory transfer

Potential F.1 solo gain: ~3-7% per cycle reduction → 3-7% throughput
improvement. Small. But two other effects compound:

1. **Under contention (Phase C pattern):** GPU draft time balloons when
   another workload competes for Metal queues. ANE is unaffected.
   Expected: DFlash preserves more of its solo throughput when draft is
   on ANE. This is the main expected win.

2. **Concurrent draft+verify:** with draft on separate silicon, we can
   potentially run draft(cycle N+1) in parallel with verify(cycle N).
   But acceptance rates of 12-17% on prose mean cross-cycle speculation
   fails often (same trap we hit in Phase B.2). May not help on prose.

## Contention measurement

Same four prompts (capital, fibonacci, math, story) run solo vs with a
background `gemma-3-270m-it-bf16` MLX workload looping on the GPU.

| Config | Solo | Parallel | Slowdown | Retained |
|---|---:|---:|---:|---:|
| Target-only | 27.39 | 22.54 | -17.7% | 82% |
| DFlash GPU-only | 40.92 | 32.79 | -19.9% | 80% |

Background workload got ~120 tok/s continuously (not meaningfully
different between runs).

**Contrast with Phase C on Gemma-3-12B:** pure-MLX SD there degraded
-29% under contention. DFlash on Qwen3-4B only degrades -20%. Two
reasons:

1. Qwen3-4B target is ~3× smaller than Gemma-3-12B → less memory-bandwidth
   pressure → less contention sensitivity. The background 270M is a
   bigger fraction of the target's size in Phase C.
2. DFlash's batch=16 verify per cycle produces a different contention
   profile than standard SD's alternating sequential-draft +
   parallel-verify pattern. Fewer Metal queue submissions per cycle.

## Notes on throughput ceiling

DFlash fibonacci clocks 85 tok/s, well above Qwen3-4B's bandwidth-bound
ceiling (~36 tok/s). How: each target verify processes batch=16 tokens
but fetches weights once, so bandwidth cost is amortized across all 16.
When acceptance is high (56% on fibonacci), most of those 16 tokens
land, so effective throughput × 2-3× the single-token ceiling. Target-only
is autoregressive — weights fetched every token — so is bandwidth-bound.

This reinforces the F.1 case: the *draft's* forward is additional bandwidth
pressure on the GPU during each cycle. Moving it to the ANE frees that
bandwidth for the target's verify to go even faster.

## F.0 baseline for F.1 comparison

F.1 targets to beat (same 4-prompt mean):

| Metric | F.0 solo | F.0 parallel |
|---|---:|---:|
| DFlash tok/s | **40.92** | **32.79** |
| Slowdown under contention | — | **-20%** |

F.1 hypothesis:
- Solo: marginal improvement (+5-10%) from draft not competing for GPU bandwidth
- Parallel: larger improvement (closer to solo) because ANE draft is
  unaffected by Metal queue contention

Scripts: `scripts/phaseF0_dflash_baseline.py` (solo full 6-prompt sweep),
`scripts/phaseF0_contention.py` (solo vs parallel 4-prompt).
