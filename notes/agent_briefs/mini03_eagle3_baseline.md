# Agent brief: EAGLE-3 baseline on Qwen3-4B (mini-03)

**For:** agent running on mini-03 (`ssh 192.168.0.63`, skynet-m4-mini-03.local)
**Duration:** ~3 days
**Converges with:** paper writing in Week 4 (benchmark tables)

## Mission

Run EAGLE-3 speculative decoding on Qwen3-4B bf16 target using MLX.
Produce directly-comparable numbers to our F.0 (DFlash GPU baseline) and
F.1 (DFlash ANE port). Gives the paper a "compared against other SD
methods" data point that isn't just DFlash-only.

## Scope

- Hardware: mini-03 (M4 Pro, 64 GB, ANE + GPU)
- Target model: `mlx-community/Qwen3-4B-bf16` (already downloaded on HF cache)
- Draft: EAGLE-3's published draft for Qwen3 (check
  [github.com/SafeAILab/EAGLE](https://github.com/SafeAILab/EAGLE) or
  HuggingFace for released checkpoints)
- Same 4 prompts as our benchmarks: capital, fibonacci, math, story
  (see `scripts/phaseF0_contention.py` for exact text)
- Same metrics: solo tok/s, tok/s under moderate/heavy contention,
  acceptance rate per cycle

## Reference materials

- Our F.0 and F.1 benchmark scripts at
  `~/projects/mlx-ane-sd/scripts/phaseF0_*.py` and `phaseF1_*.py`
- F.0 numbers at 100 tok: 40.92 tok/s mean; under Qwen3-4B bg contention: 21.21
- F.1 numbers at 100 tok: 34.68 tok/s mean; under Qwen3-4B bg contention: 19.62
- Full context: `notes/week_of_2026-04-14_summary.md`

## Deliverable

A markdown file at `notes/eagle3_baseline.md` with:
- Solo tok/s per prompt (and mean)
- Contention tok/s per prompt (moderate and heavy)
- Acceptance rate per cycle per prompt
- Notes on any implementation caveats (e.g., did you have to port
  EAGLE-3's inference from CUDA to MLX? What compromises?)

Also commit the runner as `scripts/baseline_eagle3.py`.

## Starting steps

1. `ssh 192.168.0.63 && cd ~/projects && git clone <mlx-ane-sd-repo>`
2. Check Python venv: `ls ~/models/mlx-venv/bin/python` — should exist if
   following our convention. If not, create one with mlx and mlx-lm.
3. Find and download a published EAGLE-3 draft for Qwen3. Likely
   `SafeAILab/EAGLE3-Qwen3-4B` or similar — check their repo.
4. Port EAGLE-3's inference loop to MLX. EAGLE-3's reference code is
   PyTorch; port the key loop (draft, tree spec, verify, accept).
5. Validate output coherency on one prompt before running full sweep.
6. Run the 4 prompts at max_new=100 solo.
7. Run contention sweep with bg=gemma-3-270m (moderate) and Qwen3-4B (heavy).
8. Record everything, commit, open a PR (or hand the branch back).

## Known risks

- EAGLE-3's draft might require PyTorch to run natively — if porting
  to MLX is >2 days of effort, **report back before spending more**
  and consider running EAGLE-3 via PyTorch on CPU/Metal-PyTorch as a
  fallback for baseline comparison (slower but still gives us numbers).
- If no Qwen3 draft is released by EAGLE-3 team, fall back to their
  Llama-3 draft on a Llama target — this gives us a different SD
  method comparison even if targets differ.

## Stop-and-ask triggers

- EAGLE-3 code doesn't have an MLX runner and porting is >2 days
- No compatible draft exists for any target we have downloaded
- Something about the approach differs enough that "directly
  comparable" numbers aren't meaningful

**Keep the scope bounded.** Better to ship 4 solid EAGLE-3 numbers
than to chase ideal methodology.
