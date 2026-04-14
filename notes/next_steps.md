# Next steps — heterogeneous ANE + GPU speculative decoding

## The hypothesis to test

Our bf16 sweep showed speedup peaks at `num_draft=12` (2.45× mean, 3.31× max)
and declines above that, even though **acceptance keeps climbing** (81% at
num_draft=12, 84% at num_draft=20). The bottleneck above `num_draft=12` is
**draft serial time on the GPU**, not draft quality.

If the draft runs on ANE in parallel with target verification on GPU, draft
time is hidden under the target's forward pass. This should let us push
`num_draft` to 16-20 without paying the serial cost, extending speedup
meaningfully.

**Predicted outcome:** 3-4× mean speedup at bf16, potentially matching
dflash-mlx's math-only ceiling on our natural-draft average.

## The target setup

- **Target:** `mlx-community/gemma-3-12b-it-bf16` running on MLX/Metal GPU
  (~24 GB, 9.1 tok/s baseline)
- **Draft:** `anemll-gemma-3-270m-it-MONO-ctx512-lut6` on ANE
  (already compiled, monolithic CoreML, LUT6/LUT4, ctx=512)
- **Engine:** custom speculative loop (not `mlx-lm`'s `speculative_generate_step`
  — see below)

## The engineering challenge

`mlx-lm`'s `speculative_generate_step` expects the draft model to be an MLX
`nn.Module` with a trimmable prompt cache. Our ANE draft:

- Is a `coremltools.MLModel`, not `nn.Module`
- Has a static-slice KV cache baked into the CoreML graph (shift-left +
  append), not a trimmable array
- Uses ANEMLL's tensor naming convention (`hidden_states`, `position_ids`,
  `causal_mask`, `current_pos`)
- Runs synchronously via `model.predict()` which returns numpy arrays

**We cannot plug it into `mlx-lm`'s SD loop directly.** We need to rebuild
the SD verify loop with our own orchestration:

```
while not done:
    # Draft phase (on ANE)
    draft_tokens = []
    for _ in range(num_draft):
        tok, logits = ane_draft_step(current_state)
        draft_tokens.append((tok, logits))

    # Verify phase (on GPU/MLX)
    # Target processes entire draft sequence in one forward pass
    target_logits = target_model(prefix + draft_tokens_only)

    # Compare draft predictions vs target predictions
    accepted_prefix = find_longest_agreement(draft_tokens, target_logits)

    # Commit accepted tokens, generate correction from target if any rejection
    commit(accepted_prefix)
    if rejected:
        commit(target_correction_token)
        ane_draft_rollback(state, rejected_position)
        mlx_target_trim_cache(accepted_prefix_length + 1)
```

The tricky parts:

1. **ANE draft rollback.** The ANEMLL monolithic model's cache is inside
   CoreML, not exposed. We have two options:
   - Save/restore cache state snapshots (needs access to internal CoreML
     tensors — may not be possible without custom ANEMLL build)
   - Re-prefill the draft from scratch on rollback (slow but correct)
2. **Concurrency.** To actually hide draft time, draft and target must run
   concurrently. Options:
   - Two Python threads (`threading`) — coremltools and MLX release the GIL
     during their respective calls
   - Two processes (`multiprocessing`) — cleaner isolation, Exp E from
     anemll-qwen35 proves this works at ~0 handoff cost
3. **Pipeline structure.** The "mirror" opportunity is to draft batch N+1 on
   ANE WHILE verifying batch N on GPU. This requires a dual-buffer design.

## Proposed implementation phases

### Phase A — sequential ANE+GPU (proof of correctness)

- Run draft on ANE, then verify on GPU, then draft again — all sequential
- No parallelism, no speedup expected (probably slower than pure-MLX due to
  ANE dispatch overhead)
- **Goal:** validate the loop produces correct output matching target-only
  decode, and that the ANE draft predictions are what we expect
- **Timebox:** 2-4 hours

### Phase B — concurrent ANE+GPU (the payoff)

- Draft on ANE runs in a thread while target on GPU runs in main thread
- Synchronize via queue: draft produces tokens, target consumes
- Measure speedup vs Phase A (sequential) and vs pure-MLX (2.45× baseline)
- **Goal:** demonstrate hidden draft time → extended speedup
- **Timebox:** 4-6 hours

### Phase C — tuning

- Sweep `num_draft` 8, 12, 16, 20, 24 on the concurrent pipeline to find
  the new peak
- Measure acceptance and speedup across diverse prompts
- Compare to dflash-mlx headlines and our pure-MLX ceiling
- **Timebox:** 2-4 hours

### Phase D (optional) — Mirror-SD protocol

If Phase B/C shows meaningful improvement, consider implementing the full
Mirror-SD bidirectional protocol:
- Target emits top-κ at an early-exit layer (~N/2)
- Draft branches from each of the κ candidates in parallel
- Reduces wasted work on rejected tokens via precomputed continuations

This is significantly more work (~2-3 days) and only worth it if Phase B/C
numbers warrant it.

## Decision gates

- **After Phase A:** if the loop doesn't produce correct output, debug before
  moving on. If ANE draft predictions are too different from target (despite
  natural family pairing), investigate acceptance on its own before
  proceeding.
- **After Phase B:** if concurrent ANE+GPU gives <1.3× speedup over pure-MLX
  SD, the heterogeneous thesis is dead on this hardware and we should
  reconsider.
- **After Phase C:** if best config gives <3× mean speedup at bf16, we're not
  meaningfully beating dflash-mlx's best case and should ask whether this
  approach is worth pursuing vs just training a better MLX draft.

## What we're NOT doing (saved effort)

- **Not training a custom SS-objective draft.** The natural Gemma-3-270M
  draft gives 2.45× at bf16 already; custom training is a second-order
  optimization that only makes sense after we've validated the ANE
  heterogeneous pattern.
- **Not implementing tensor parallelism across minis.** Single-mini scope
  is enough for the PoC. Multi-mini can come later via `exo` if we want to
  scale to larger targets.
- **Not using EAGLE-3 / Medusa / other pretrained draft heads.** The
  natural-family draft path is cleaner and gives comparable numbers.
- **Not pursuing 4bit SD.** Structural dead-end per multiple data points.
