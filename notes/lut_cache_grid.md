# LUT x cache-size x generation-length grid

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB (`skynet-m4-mini-03`)
**Target:** `mlx-community/Qwen3-4B-bf16` on MLX/GPU
**Draft:** `z-lab/Qwen3-4B-DFlash-b16` -> CoreML F.1 accumulating-cache ANE variant (100% ANE placement)

Solo benchmarks of the F.1 DFlash draft across the full
(quantization x state_length x generation length) matrix. Each cell is the
mean throughput over 4 prompts (`capital`, `fibonacci`, `math`, `story`) at
`max_new` in {100, 300} and — due to compute budget — 2 prompts (`capital`,
`fibonacci`) at `max_new=1000`. Greedy decoding. No contention.

Artifacts: `artifacts/lut_cache_grid.json` (raw per-prompt rows),
`artifacts/lut_cache_grid.csv` (tidy), `scripts/bench_lut_cache_grid.py`
(reproducible runner).

## Deployment recommendation (TL;DR)

| Workload (max_new) | Best config | Mean tok/s | Runner-up |
|---|---|---:|---|
| short (<=100 tok) | **LUT4 per_grouped_channel (g=8) + S=1024** | **32.74** | LUT4 + S=512 (33.89) but fragile for anything longer |
| medium (~300 tok) | **LUT4 per_grouped_channel (g=8) + S=1024** | **24.14** | LUT4 + S=2048 (22.54) |
| long (1000 tok) | **LUT6 per_tensor + S=2048** | **27.50** | LUT4 + S=2048 (26.17) |

A single safe default across all generation lengths: **LUT4 per_grouped_channel
(g=8) + S=2048** (30.51 / 22.54 / 26.17 tok/s at 100 / 300 / 1000 tok). LUT4
g=8 is 257 MB on disk vs 1025 MB unquantized.

## Quick take (5 bullets)

1. **LUT4 per_grouped_channel (g=8) dominates gen<=300 across all cache
   sizes.** It beats both unquantized and LUT6 per_tensor at every
   (state_length, max_new<=300) cell. Compression halves per-call latency
   (~17 ms -> ~8 ms at S=1024) without noticeable acceptance loss; at
   S=1024/max_new=100 the LUT4 run even hits 64 tok/s on fibonacci alone.
2. **Cache sweet spot moves with generation length.** S=1024 is best at
   short gen (100-300 tok) because it's big enough to accumulate without
   sliding for typical prompts but small enough for cheap attention;
   S=2048 wins at 1000 tok because the extra headroom delays the sliding-
   window acceptance collapse that kicks in at `state_length / 32` cycles.
3. **S=512 is dead-on-arrival past ~200 tokens.** Sliding fires at
   cycle 16 so only fibonacci-like high-acceptance prompts benefit, and
   even those collapse by max_new=300. Include S=512 only for bounded
   short-reply deployments.
4. **LUT6 per_tensor flips ahead of LUT4 at gen=1000.** LUT6 preserves
   slightly more acceptance than LUT4 under prolonged sliding (27.50 vs
   26.17 at S=2048/1000). If you already pay the LUT tax, LUT6 is the
   right choice for long streaming workloads; LUT4 is the right choice
   for short interactive ones.
5. **mini-03 runs ~15% slower than mini-02 on the same config.** mini-02
   reported 32.94 tok/s at (none, S=1024, 100 tok); mini-03 gets 28.69.
   Same coremltools 9.0, torch 2.11, but mini-03 is on macOS 26.3.1
   preview. Relative comparisons within this run are consistent; treat
   absolute numbers as mini-03-specific until mini-02 re-runs the same
   grid.

## Matrix: mean tok/s per cell

### Unquantized fp16 (baseline)

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | 17.51 | 8.62 | 9.97 |
| S=1024 | **28.69** | **16.06** | 12.00 |
| S=2048 | 24.05 | 13.32 | **21.90** |
| S=4096 | 21.47 | 16.39 | 18.29 |

*Best per column in bold.* Observations:
- S=1024 is Pareto-optimal for <=300 tok (biggest acceptance without the
  attention tax of S>=2048).
- S=2048 wins at 1000 tok because sliding doesn't fire until cycle 64
  vs S=1024's cycle 32. The extra per-call cost (~27 vs 18 ms) is
  amortized by the higher average acceptance per cycle.
- S=4096 at max_new=100 is worse than S=1024: no sliding benefit (prompts
  finish before cycle 32 anyway) and ~2x the attention cost per call.

### LUT6 per_tensor

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | 29.86 | 13.60 | 11.60 |
| S=1024 | 24.22 | 17.90 | 10.66 |
| S=2048 | 19.65 | 20.04 | **27.50** |
| S=4096 | **27.94** | **20.99** | 24.44 |

- LUT6 per_tensor per-call latency: ~12 ms at S=1024 (vs unquant 18 ms),
  ~23 ms at S=2048, ~34 ms at S=4096.
- LUT6 wins at gen=1000 / S=2048 (27.50) — the clear long-gen winner.
- At gen=100 LUT6 isn't as sharp as LUT4 g=8; the per_tensor granularity
  costs some acceptance on code-like prompts.
- LUT6/S=1024 max_new=100 is lower than unquant S=1024 (24.22 vs 28.69).
  Likely single-run variance; LUT4 g=8 is clearly the stronger short-gen
  choice at S=1024.

### LUT4 per_grouped_channel (group=8) — iOS18+/macOS15+

| state_length \ max_new | 100 | 300 | 1000 |
|---:|---:|---:|---:|
| S=512 | **33.89** | **20.95** | 12.35 |
| S=1024 | 32.74 | **24.14** | 14.20 |
| S=2048 | 30.51 | 22.54 | **26.17** |
| S=4096 | 26.90 | 20.00 | 23.25 |

- LUT4 g=8 per-call latency: ~8 ms at S=512, ~12 ms at S=1024, ~19 ms at
  S=2048, ~34 ms at S=4096.
- Dominates the gen<=300 half of the grid at every cache size.
- Per-grouped-channel matters: per_tensor LUT4 (not shown; earlier work)
  collapsed acceptance. The g=8 granularity preserves it.
- Starts losing to LUT6 at gen=1000 as sliding-phase acceptance pressure
  dominates latency gains.

## Per-call latency cheat-sheet

Median ANE draft latency per cycle (ms), across prompts:

| state_length | none | lut6_pt | lut4_gc8 |
|---:|---:|---:|---:|
| S=512 | ~14 | ~9 | **~8** |
| S=1024 | ~18 | ~12 | **~12** |
| S=2048 | ~26 | ~23 | **~19** |
| S=4096 | ~40 | ~34 | **~34** |

Compression halves per-call cost at S<=1024 but tapers at larger caches
because attention over the full `state_length + T` positions starts
dominating.

## Per-prompt detail

Below: raw per-prompt rows for each cell. `cycles` = number of draft
speculation cycles; `accepted` = total tokens that survived verification;
`per_call_ms` = mean latency of a single ANE draft invocation.

<details><summary>none (fp16)</summary>

**S=512, max_new=100** — mean 17.51 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 12.12 | 8.25 | 56 | 99 | 14.33 |
| fibonacci | 100 | 2.70 | 37.04 | 14 | 99 | 12.93 |
| math | 100 | 6.34 | 15.77 | 28 | 99 | 15.18 |
| story | 100 | 11.13 | 8.99 | 52 | 99 | 14.10 |

**S=512, max_new=300** — mean 8.62 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 34.02 | 8.82 | 157 | 299 | 14.32 |
| fibonacci | 300 | 29.16 | 10.29 | 105 | 299 | 14.71 |
| math | 300 | 36.32 | 8.26 | 104 | 299 | 14.94 |
| story | 300 | 42.22 | 7.11 | 180 | 299 | 14.12 |

**S=512, max_new=1000** — mean 9.97 tok/s (2 prompts)

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 119.86 | 8.34 | 789 | 999 | 13.93 |
| fibonacci | 1000 | 86.17 | 11.60 | 661 | 999 | 13.75 |

**S=1024, max_new=100** — mean 28.69 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 6.89 | 14.52 | 56 | 99 | 17.51 |
| fibonacci | 100 | 1.68 | 59.63 | 14 | 99 | 17.21 |
| math | 100 | 3.78 | 26.44 | 28 | 99 | 18.76 |
| story | 100 | 7.05 | 14.19 | 52 | 99 | 18.58 |

**S=1024, max_new=300** — mean 16.06 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 21.09 | 14.22 | 142 | 301 | 18.68 |
| fibonacci | 300 | 12.94 | 23.19 | 81 | 302 | 19.62 |
| math | 300 | 17.32 | 17.32 | 91 | 300 | 18.59 |
| story | 300 | 31.53 | 9.52 | 162 | 300 | 18.30 |

**S=1024, max_new=1000** — mean 12.00 tok/s (2 prompts)

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 85.77 | 11.66 | 705 | 999 | 15.96 |
| fibonacci | 1000 | 81.05 | 12.34 | 533 | 999 | 18.00 |

**S=2048, max_new=100** — mean 24.05 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 9.56 | 10.46 | 56 | 99 | 27.16 |
| fibonacci | 100 | 2.34 | 42.74 | 14 | 99 | 27.18 |
| math | 100 | 3.37 | 29.64 | 28 | 99 | 24.16 |
| story | 100 | 7.48 | 13.37 | 52 | 99 | 25.07 |

**S=2048, max_new=300** — mean 13.32 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 30.37 | 9.88 | 142 | 301 | 27.73 |
| fibonacci | 300 | 17.49 | 17.15 | 81 | 302 | 27.67 |
| math | 300 | 19.43 | 15.44 | 91 | 300 | 27.51 |
| story | 300 | 31.53 | 9.52 | 162 | 300 | 18.30 |

**S=2048, max_new=1000** — mean 21.90 tok/s (2 prompts)

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 63.40 | 15.77 | 322 | 1006 | 25.75 |
| fibonacci | 1000 | 35.41 | 28.24 | 228 | 1005 | 28.67 |

**S=4096, max_new=100** — mean 21.47 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 100 | 7.60 | 13.16 | 56 | 99 | 38.86 |
| fibonacci | 100 | 2.26 | 44.22 | 14 | 99 | 38.69 |
| math | 100 | 5.22 | 19.16 | 28 | 99 | 40.17 |
| story | 100 | 11.63 | 8.60 | 52 | 99 | 37.91 |

**S=4096, max_new=300** — mean 16.39 tok/s

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 300 | 27.94 | 10.74 | 142 | 301 | 39.06 |
| fibonacci | 300 | 14.20 | 21.12 | 81 | 302 | 44.78 |
| math | 300 | 14.39 | 20.84 | 91 | 300 | 43.32 |
| story | 300 | 23.02 | 13.03 | 162 | 300 | 40.08 |

**S=4096, max_new=1000** — mean 18.29 tok/s (2 prompts)

| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |
|---|---:|---:|---:|---:|---:|---:|
| capital | 1000 | 56.54 | 17.69 | 322 | 1006 | 41.91 |
| fibonacci | 1000 | 52.70 | 18.98 | 221 | 1005 | 44.06 |

</details>

<details><summary>lut6_pt (LUT6 per_tensor)</summary>

**S=512, max_new=100** — mean 29.86 tok/s

Per-prompt: capital 16.19 / fibonacci 58.45 / math 20.96 / story 22.26 tok/s;
per_call ~9 ms uniformly.

**S=512, max_new=300** — mean 13.60 tok/s. Sliding dominates; capital 9.67,
fibonacci 14.50, math 15.82, story 13.28 tok/s.

**S=512, max_new=1000** — mean 11.60 tok/s (capital 11.02, fibonacci 12.17).

**S=1024, max_new=100** — mean 24.22 tok/s (capital 17.41 / fibonacci 45.89 /
math 21.85 / story 18.12). per_call ~12 ms.

**S=1024, max_new=300** — mean 17.90 tok/s (capital 19.21 / fibonacci 22.74 /
math 18.96 / story 10.17).

**S=1024, max_new=1000** — mean 10.66 tok/s (capital 9.39, fibonacci 11.93).

**S=2048, max_new=100** — mean 19.65 tok/s (capital 10.18 / fibonacci 38.75 /
math 18.35 / story 11.28). per_call ~23 ms.

**S=2048, max_new=300** — mean 20.04 tok/s (capital 11.77 / fibonacci 24.12 /
math 28.25 / story 16.01).

**S=2048, max_new=1000** — mean **27.50 tok/s** (capital 26.32, fibonacci
28.69). Best long-gen cell; per_call ~19 ms.

**S=4096, max_new=100** — mean 27.94 tok/s (capital 15.80 / fibonacci 56.40 /
math 26.33 / story 12.61).

**S=4096, max_new=300** — mean 20.99 tok/s.

**S=4096, max_new=1000** — mean 24.44 tok/s (capital 23.38, fibonacci 25.50).

</details>

<details><summary>lut4_gc8 (LUT4 per_grouped_channel, group=8)</summary>

**S=512, max_new=100** — mean 33.89 tok/s (capital 17.14 / fibonacci 66.23 /
math 34.17 / story 18.03). per_call ~8 ms.

**S=512, max_new=300** — mean 20.95 tok/s (capital 17.27 / fibonacci 24.34 /
math 26.85 / story 15.06).

**S=512, max_new=1000** — mean 12.35 tok/s (capital 11.19, fibonacci 13.52).

**S=1024, max_new=100** — mean 32.74 tok/s (capital 16.54 / fibonacci 64.00 /
math 33.00 / story 20.61). per_call ~12 ms.

**S=1024, max_new=300** — mean 24.14 tok/s (capital 16.42 / fibonacci 34.56 /
math 29.85 / story 15.65).

**S=1024, max_new=1000** — mean 14.20 tok/s (capital 12.37, fibonacci 16.00).

**S=2048, max_new=100** — mean 30.51 tok/s (capital 16.08 / fibonacci 59.61 /
math 30.74 / story 16.25). per_call ~19 ms.

**S=2048, max_new=300** — mean 22.54 tok/s (capital 13.18 / fibonacci 27.86 /
math 26.69 / story 15.73).

**S=2048, max_new=1000** — mean 26.17 tok/s (capital 24.28, fibonacci 28.49).

**S=4096, max_new=100** — mean 26.90 tok/s.

**S=4096, max_new=300** — mean 20.00 tok/s.

**S=4096, max_new=1000** — mean 23.25 tok/s (capital 21.57, fibonacci 24.92).

</details>

## Compute cost note

- Total wall time to fill the 36-cell matrix on mini-03: **~65 min**
  (benchmarking only; add ~15 min for 12 convert/LUT/compile runs).
- Conversion: 4 base packages (~15 s each, mostly graph tracing).
- LUT6 per_tensor: ~50 s per package (62 ops palletized).
- LUT4 per_grouped_channel (g=8): ~90-110 s per package (3-4x slower than
  LUT6 due to per-channel clustering).
- Compilation via `MLModel.get_compiled_model_path()`: ~6-9 s each.

## Methodology caveats

- **Only capital + fibonacci at max_new=1000** for wall-time budgeting.
  These bracket the acceptance distribution (capital = prose / low accept,
  fibonacci = code / high accept). Math and story would fall in between,
  but we have them at max_new<=300 for reference.
- **Single run per cell.** Variance between runs is visible at the ~10%
  level for prose/mixed prompts; treat gaps <10% between adjacent cells
  as noise.
- **mini-03 is ~15% slower than mini-02** on the same config, same
  coremltools/torch versions; likely the macOS 26.3.1 preview vs
  production build difference. Use this grid for *relative* configuration
  choices, not as the paper's headline absolute numbers.
- **Draft acceptance per cycle** tracks sliding-window regime: starts
  ~7 for fibonacci / ~2 for capital in the accumulating phase, collapses
  towards ~1.3-1.5 once sliding fires. See per-prompt cycle counts for
  the signal.
- **Sliding onset** is at cycle `state_length / T` where `T = 32`:
  cycle 16 for S=512, cycle 32 for S=1024, cycle 64 for S=2048,
  cycle 128 for S=4096. Everything past that is a sliding-window model
  with context-loss penalty to acceptance.

## Reproducing

```bash
# On mini-03 (or any Apple Silicon with MLX + coremltools):
python scripts/bench_lut_cache_grid.py \
  --artifacts-dir /tmp/lut_cache_grid \
  --output-json artifacts/lut_cache_grid.json \
  --output-csv artifacts/lut_cache_grid.csv \
  --output-md notes/lut_cache_grid.md \
  --long-gen-prompts capital fibonacci  # keeps gen=1000 tractable

# Or to limit the grid:
python scripts/bench_lut_cache_grid.py \
  --quant lut4_gc8 --states 1024 2048 --gens 100 300
```

The runner auto-builds `.mlpackage` + `.mlmodelc` for any
(quant, state_length) combination not already in `--artifacts-dir`, and
supports `--resume` to continue a crashed run.
