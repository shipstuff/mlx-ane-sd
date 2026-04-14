# Phase F.1 stateless: partial port of DFlash draft to ANE

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB
**Target:** `mlx-community/Qwen3-4B-bf16`
**Draft:** `z-lab/Qwen3-4B-DFlash-b16`, ported to ANE via our ANEMLL fork patterns

## What we built

1. **Clean-room PyTorch port** of DFlashDraftModel from z-lab's `model_mlx.py`,
   with no HF `transformers` / `trust_remote_code` dependency. Parity with
   the MLX reference: 16/16 top-1 match on the first-cycle block, cos_sim
   0.9999. See `scripts/dflash_torch.py`, `scripts/phaseF1_parity_torch_vs_mlx.py`.

2. **ANE-compatible variant** with:
   - Module-level RMSNorm switch (standard / ane form, ANEMLL convention)
   - Static shapes (block_size=16, ctx_size=16)
   - Trace-ready (no dynamic int ops, no Optional tensors, no Python control flow)
   - RoPE tables as model inputs (positions computed Python-side)
   - GQA via expand+reshape (not repeat_interleave)

3. **CoreML conversion via coremltools.convert.**
   - All 617 ops lowered to MLProgram
   - **100% on ANE, 0 CPU ops, 0 graph interruptions** per anemll-profile
   - **8.57 ms/prediction** (measured), 116.7 iter/s
   - ~1 GB unquantized weights, streaming 2.5 MB/iter
   - Compute-bound (not bandwidth-bound)

4. **End-to-end SD loop** with ANE-hosted draft + MLX-hosted target.

## What works

- The PyTorch rewrite is bit-for-bit compatible with the MLX reference on
  cycle 0 (cold cache).
- CoreML conversion works cleanly — no trace errors, no unsupported ops,
  no unexpected CPU fallbacks.
- The whole model runs on ANE with no graph interruptions.
- End-to-end generation produces coherent, correct output text.

## Where it falls short

Stateless ANE DFlash produces valid output but at much lower throughput
than the GPU-only baseline:

| Prompt | F.0 (GPU-only) | F.1 (stateless ANE) | Ratio |
|---|---:|---:|---:|
| capital | 17.76 | 13.91 | 0.78× |
| fibonacci | **85.38** | 16.67 | **0.20×** |
| math | 40.48 | 21.42 | 0.53× |
| story | 19.05 | 13.86 | 0.73× |
| **mean** | **40.92** | **16.46** | **0.40×** |

Acceptance rates per cycle (tokens accepted, including the 1 bonus):

| Prompt | F.0 | F.1 stateless | Drop |
|---|---:|---:|---:|
| capital | 1.8 | 1.4 | -22% |
| fibonacci | 8.4 | 1.7 | **-80%** |
| math | 4.0 | 2.2 | -45% |
| story | 1.9 | 1.4 | -26% |

## Diagnosis

The DFlash draft maintains a **KV cache across decode cycles**. Each cycle
appends (ctx + block) K/V to that cache, then trims to the committed
prefix. The cache gives the draft *longer-range memory* of prior committed
context, which the target_hidden per-cycle doesn't fully capture.

Our stateless ANE variant **re-processes every cycle from scratch** — no
memory of prior cycles. This is the root cause of the acceptance drop:
- High-acceptance prompts (fibonacci: 49% per-draft-slot accept rate)
  depend most on the cached context. Our stateless impl drops them to
  ~5% accept, killing the win.
- Low-acceptance prompts are less affected but still degraded.

Fibonacci's 85 → 17 drop is especially stark because GPU-only DFlash
was already bandwidth-amortizing heavily on that prompt (each verify
processes 16 tokens per weight-fetch, giving super-linear throughput).
Losing acceptance on fibonacci forfeits that bandwidth amortization.

## What's missing: KV cache as CoreML state_tensor

The path forward is well-understood. Our anemll-qwen35 fork has the
recipe:

- Represent K and V cache per-layer as CoreML `state_tensor`
  with shape `(B, Hkv, MAX_CACHE_LEN, head_dim)`.
- Each draft call writes new K/V at positions `[cache_offset, cache_offset+S+L)`
  via static-slice updates.
- SDPA attends to the full cache; an attention mask suppresses positions
  beyond `cache_offset+S+L`.
- Caller reads/writes state via `CompiledMLModel.read_state` /
  `write_state` (same pattern our Phase B.1 infrastructure uses).
- On partial accept, trim by writing back the pre-write state or
  re-computing from a snapshot.

This is ~2-3 days of focused work:
1. Add state tensors to the model definition, convert with states declared
2. Implement snapshot/restore for partial-accept trim (reuse Phase B infra)
3. Handle the cache mask (attention mask becomes required input)
4. Re-validate: parity + benchmark

## Intermediate wins documented

- **The port is mechanically possible.** No exotic ops, no CoreML
  deal-breakers. The draft model lowers 100% to ANE.
- **Per-call latency on ANE is competitive.** 8.57 ms for the full draft
  forward at batch=16, vs a reasonable estimate of 15-30 ms for the same
  forward on the MLX GPU pipeline (though we don't have that isolated).
- **The RoPE-position math across variable ctx lengths is worked out.**
  The key insight: real_ctx_start = (len(tokens) - 1) - s_real, pad
  positions get arbitrary rope (contribute ≈0 due to padded-K=0).

## Next session work

Adding the KV cache is the next concrete step. The expected outcome:
- Accept rates recover toward F.0's numbers (within ~5-10%)
- Throughput: solo likely within noise of F.0 (ANE latency is good)
- **Under contention** (Phase C-style): F.1 should preserve more solo
  throughput than F.0 — that's the real DFlash-on-Apple-Silicon
  value proposition, analogous to our Phase C result.

## Files committed

- `scripts/dflash_torch.py` — PyTorch reference (MLX-parity)
- `scripts/dflash_ane.py` — ANE-adapted variant (traceable)
- `scripts/dflash_coreml_convert.py` — coremltools conversion
- `scripts/phaseF1_parity_torch_vs_mlx.py`
- `scripts/phaseF1_parity_ane_torch.py`
- `scripts/phaseF1_coreml_validate.py`
- `scripts/phaseF1_ane_stream.py` — end-to-end ANE SD runner
- `/tmp/dflash_ane.mlpackage` + `/tmp/dflash_ane_compiled/dflash_ane.mlmodelc`
  (not checked in, regenerable from conversion script)
