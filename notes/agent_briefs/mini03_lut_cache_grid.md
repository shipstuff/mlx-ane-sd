# Agent brief: LUT × cache-size grid sweep (mini-03)

**For:** agent running on mini-03
**Duration:** ~2-3 days (mostly compute time, light engineering)
**Converges with:** paper's deployment-config recommendation table

## Mission

Fill in the full (quantization × cache size × workload length) matrix
for the F.1 accum ANE DFlash variant. Current data is sparse: we have
LUT6/LUT4-gc at S=1024 for 100-tok, LUT0 at S=∈{1024, 2048, 4096} for
various lengths. Fill the gaps so the paper has a clean deployment-
recommendation table.

## Grid to fill

Axes:
- Quantization: **none, LUT6 per_tensor, LUT4 per_grouped_channel (group=8)**
- State length: **S=512, 1024, 2048, 4096**
- Generation length: **max_new ∈ {100, 300, 1000}**

That's 3 × 4 × 3 = 36 cells to benchmark. Each cell runs the 4 standard
prompts (capital, fibonacci, math, story) and reports mean tok/s.

## Reference materials

- Convert script: `scripts/dflash_coreml_convert_accum.py` (change `--state-length`)
- LUT script: `scripts/dflash_lut_quantize.py` (handles `--bits` and `--granularity`)
- Runner: `scripts/phaseF1_ane_stream_accum.py`
- Current sparse data: `notes/phaseF1_accum_findings.md`, `notes/phaseF3_cache_sizing.md`,
  `notes/week_of_2026-04-14_summary.md`

## Deliverables

1. **`notes/lut_cache_grid.md`** — the full matrix as markdown tables.
   One table per LUT variant; rows = state_length, cols = gen_length.
2. **`scripts/bench_lut_cache_grid.py`** — runner that generates the matrix
   end-to-end, so it's reproducible.
3. Brief commentary per table interpreting the numbers.

## Compute budget

Each cell takes ~1-5 min depending on gen length. 36 cells × 3 min
average = ~2 hours of GPU+ANE time. Parallel compilation of mlpackages
takes ~10 min each. Total wall: ~half day of compute + a day of setup +
a day of writeup.

## Starting steps

1. `ssh 192.168.0.63 && cd ~/projects && git clone <repo>`
2. Convert all needed mlpackages (for each S ∈ {512, 1024, 2048, 4096}).
   - Note: S=512 requires adding support in the convert script
     (currently does 1024+ only) — small tweak.
3. For each mlpackage: apply LUT6 per_tensor AND LUT4 per_grouped_channel
   versions. That's 4 × 3 = 12 mlpackages total.
4. Compile all 12 to mlmodelc.
5. Write the grid runner that iterates (mlpackage, state_length, max_new)
   and calls the stream_generate. Output CSV/JSON.
6. Analyze and write the markdown tables.

## Known risks

- S=512 may hit the sliding fallback immediately for even short
  generations at low acceptance rates. Interesting data either way.
- S=4096 at max_new=1000 with LUT takes the longest per run (~3-5
  min per prompt × 4 prompts = 15-20 min). Budget accordingly.
- LUT4 per_grouped_channel requires iOS18 target — make sure
  convert script uses `minimum_deployment_target=ct.target.macOS15`.

## Stop-and-ask triggers

- Any variant fails to compile: report back with the MIL error
- Numerical outputs diverge from baseline (gibberish): stop, report,
  debug before filling more cells
- Compile step takes more than 30 min per package (unusual): flag it

## Out of scope

Do NOT benchmark under contention in this grid — that doubles the
effort and the contention data we already have at a few points is
enough for the paper. Solo only for the grid.
