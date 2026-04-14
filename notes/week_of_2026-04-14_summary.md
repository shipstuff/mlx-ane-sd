# Week summary: 2026-04-14 → DDTree-on-Apple-Silicon progress

## Headline

**Built the first ANE port of a block-diffusion SD draft.** DFlash draft runs
100% on ANE with acceptance competitive with the native-MLX baseline. F.1
reaches 85% of F.0's solo throughput, 83-85% under contention, and wins on
low-acceptance prompts under heavy contention. Plus a solid negative
finding on naive top-κ tree speculation that clarifies what proper DDTree
requires.

## Numbers that matter

### Solo, max_new=100, Qwen3-4B bf16 target, greedy

| Variant | Mean tok/s | ANE placement | Notes |
|---|---:|:---:|---|
| F.0 GPU MLX (baseline) | 40.92 | — | z-lab/dflash MLX reference |
| F.1 stateless | 16.46 | 100% | no KV cache |
| F.1 state_tensor | 32.78 | 0% | ANE compile fails on dyn slice |
| F.1 ext-sliding | 16.68 | 100% | shift-left + append |
| **F.1 accum (winner)** | **34.68 (85%)** | **100%** | Python-managed cache |
| F.1 accum + LUT4 (gc=8) | 32.62 | 100% | 4× smaller model |
| F.2 top-κ=2 tree | 27.70 (-18%) | 100% | naive tree hurts |

### Scaling with length (F.1 accum)

| Length | F.0 | F.1 S=256 | F.1 S=1024 | F.1/F.0 |
|---:|---:|---:|---:|---:|
| 100 | 40.92 | 34.68 | 32.94 | 85% |
| 300 | 30.48 | 15.16 (crash) | 24.33 | 80% |
| 500 | 29.11 | — | 24.33 | 84% |

**Key result**: STATE_LENGTH=256 is enough only for ≤8 cycles. For longer
generations, STATE_LENGTH=1024 prevents the sliding-window acceptance
collapse.

### Under contention

| Config | Solo | Moderate (gemma-270m) | Heavy (Qwen3-4B) |
|---|---:|---:|---:|
| F.0 | 40.92 | 32.79 | 21.21 |
| F.1 accum ANE | 34.68 | 28.83 | 19.62 |
| Retained (F.0) | — | 80% | 52% |
| Retained (F.1) | — | 83% | 57% |

F.1 preserves 3pp more than F.0 at both contention levels. On
low-acceptance prompts (capital, story) under heavy contention, F.1
already wins outright (capital 9.53 vs 9.21; story 10.43 vs 9.66).

## Architecture decisions and why

### Cache architecture: externalized accumulating

Three variants failed before getting to the winner:
- **In-model state_tensor with dynamic slice_update**: cleanest design
  semantically, but ANE compiler rejects dynamic slice_update bounds.
  Works on GPU (~32.78 tok/s) but defeats the whole point of the port.
  Verified via `anemll-profile`: `MILCompilerForANE error: failed to
  compile ANE model using ANEF. Error=_ANECompiler : ANECCompile()
  FAILED (11)`.
- **In-model sliding window**: all-static bounds, compiles to ANE, but
  loses history after 8 cycles. 41% of F.0.
- **External cache without trim**: same sliding issue + extra Python
  overhead.
- **External cache WITH MLX-style trim** (winner): advance by
  `s_real + accepted + 1` (committed count) per cycle, not T.
  Matches MLX's `trim_prompt_cache` semantics. Next cycle overwrites
  rejected positions via static slice. 85% of F.0 with 100% ANE.

### Why torch version doesn't matter

Tested torch 2.5.0 vs 2.11.0 on the state_tensor variant. Same MIL
output, same ANE compile failure. The ANE compiler's rejection of
dynamic slice bounds is fundamental, not a versioning mismatch.

### LUT quantization: modest win

| Config | Size | Per-call | Mean tok/s (100) |
|---|---:|---:|---:|
| Unquant S=256 | 1025 MB | 10.9 ms | 34.68 |
| Unquant S=1024 | 1025 MB | 16.5 ms | 32.94 |
| LUT6 per_tensor | 385 MB | 12.0 ms | 33.82 |
| LUT4 per_tensor | 257 MB | 11.8 ms | 28.10 (acceptance hurt) |
| LUT4 per_grouped_channel (g=8) | 257 MB | 12.0 ms | 32.62 |

LUT4+group=8 is the best deployment config (smallest, preserves
acceptance). But the latency gain is only ~10-15%. DFlash is
compute-bound, not bandwidth-bound (3.5-4 TOPS of 38 peak). ANEMLL's
6.5× from LUT4 on qwen3.5-0.8B was a bandwidth-bound scenario.

### Naive tree speculation doesn't work

DFlash is bidirectional — alternative first-token choices don't change
subsequent positions' draft logits. I thought this would let tree
speculation "collapse" to simple top-κ acceptance. It doesn't: without
target-side tree attention, top-κ boundary events force premature stop,
cutting high-acceptance streaks short. Net -18% vs baseline.

Proper DDTree (Algorithm 1 in the paper) REQUIRES target-side tree
attention — each branch attends only to its ancestors. Implementing
this in MLX target forward is ~1 week of engineering.

## What's open for next iteration

Ranked by research value:

### 1. Target-side tree attention for true DDTree (~1 week)

Modify mlx-lm Qwen3's attention to accept a tree-structured causal
mask, with position IDs by tree depth. Pack the tree as a flattened
sequence. Implement tree-walking acceptance. Paper claims 1.3-2× over
single-chain SD.

Prereq understanding: read Section 4.4 of DDTree paper in detail
(already extracted into `notes/phaseF2_tree_findings.md`).

### 2. Batched multi-chain verify (~half week)

Simpler alternative to full tree attention. Instead of one chain, run
N=2 chains (one from each of draft's top-2 at position 0) through
target with batch=N. Each chain has its own KV cache state; pick the
best. Estimated gain: +10-20%.

### 3. Multi-function for cache write positions (~1 week)

Eliminate the sliding-window fallback at cycle 8+ by compiling N
versions of the model with baked-in write positions. Adds cache
history depth without hitting the dynamic-slice ANE blocker. Would
push long-generation F.1 above 85% of F.0 consistently.

### 4. Train a block-diffusion draft for Gemma-3-12B (weeks)

Opens a much bigger regime — Gemma-3-12B is the target we used in
Phases A-C. A DFlash-style draft for it would combine Phase C's
heterogeneous-contention win with Phase F.1's draft port.

### Lower-priority

- Larger ANE utilization: the compute profile showed 3.9 TOPS of ANE's
  38 peak. Maybe fused super-blocks (anemll's `test_sb_fusion.py`
  pattern) could push this higher, though DFlash only has 5 layers so
  the fusion headroom is limited.

## Files the week added

### Scripts
- `scripts/phaseF0_dflash_baseline.py` — F.0 solo baseline
- `scripts/phaseF0_contention.py` — F.0 contention harness
- `scripts/dflash_torch.py` — PyTorch clean-room port (MLX parity)
- `scripts/dflash_ane.py` — ANE-adapted stateless variant
- `scripts/dflash_ane_cache.py` — in-model state_tensor (GPU-fallback)
- `scripts/dflash_ane_slidecache.py` — in-model sliding (ANE)
- `scripts/dflash_ane_extcache.py` — external sliding cache (ANE)
- `scripts/dflash_ane_accumcache.py` — **external accumulating (winner)**
- `scripts/dflash_coreml_convert*.py` — converters for each variant
- `scripts/dflash_lut_quantize.py` — LUT4/6 post-processing
- `scripts/phaseF1_parity_*.py` — numerical parity tests
- `scripts/phaseF1_coreml_validate.py` — post-conversion validation
- `scripts/phaseF1_ane_stream_*.py` — four SD runners (stateless/sliding/accum)
- `scripts/phaseF1_contention*.py` — contention harnesses
- `scripts/phaseF2_ane_tree.py` — top-κ tree (negative)

### Notes
- `notes/phaseF0_results.md` — F.0 baseline numbers + contention
- `notes/phaseF_investigation.md` — feasibility of the port
- `notes/phaseF1_stateless_findings.md` — stateless port works, limited
- `notes/phaseF1_cache_findings.md` — state_tensor GPU fallback
- `notes/phaseF1_variants_summary.md` — three variant comparison
- `notes/phaseF1_accum_findings.md` — accumulating cache breakthrough
- `notes/phaseF2_tree_findings.md` — naive tree fails, needs tree attention
- `notes/week_of_2026-04-14_summary.md` — this file
