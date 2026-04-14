# Phase D.0: early-exit logit-lens accuracy on Gemma-3-12B

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB
**Model:** `mlx-community/gemma-3-12b-it-bf16` (48 layers)

## The question

Mirror Speculative Decoding (Apple, arxiv 2510.13161) assumes that the
target's intermediate hidden state hₜ^(ℓₑ) at an early-exit layer ℓₑ,
when passed through the model's LM head, yields a top-κ distribution
that's a good proxy for the final top-1 token. The whole protocol
depends on this: the draft starts generating speculatively seeded on
the Top-κ from layer ℓₑ while the target finishes layers ℓₑ+1..N in
parallel.

Does this assumption hold for Gemma-3-12B bf16 with the model's native
LM head?

## Method

- Run Gemma-3-12B forward on 8 diverse prompts (219 total token positions)
- At each candidate ℓₑ ∈ {12, 18, 24, 30, 36, 42}, apply `norm` + `lm_head`
  to the layer's output hidden state
- Compare to the same operation applied at layer 48 (final)
- Measure: P(proxy top-1 == final top-1), P(final top-1 ∈ proxy top-κ)
  for κ ∈ {1, 2, 4, 8, 16}

Script: `scripts/phaseD0_early_exit_accuracy.py`.

## Result

| Depth | top-1 | top-2 | top-4 | top-8 | top-16 |
|---:|---:|---:|---:|---:|---:|
| 12 | 0.0% | 0.5% | 1.8% | 4.1% | 8.7% |
| 18 | 2.3% | 3.7% | 7.3% | 8.2% | 11.4% |
| 24 | 11.4% | 13.7% | 18.3% | 24.2% | 28.8% |
| 30 | 32.9% | 45.2% | 57.1% | 62.1% | 69.9% |
| 36 | 44.3% | 57.5% | 68.9% | 71.7% | 79.5% |
| 42 | 49.8% | 63.5% | 68.9% | 78.1% | 82.2% |

At the paper's canonical depth (ℓₑ = N/2 = 24), **the proxy top-1 matches
the final top-1 only 11% of the time**. Even with κ=8, containment is
only 24%.

Accuracy climbs monotonically with depth, but so does the lost overlap
budget: at depth 42, only 6 of 48 layers remain to hide draft work under.

## Why this kills naive Mirror-SD on Gemma-3-12B

Mirror-SD's speedup relies on the draft's generation time Tdraft^gen
fitting entirely under the target's remaining-layer budget Δ:

```
Tdraft^gen(γ) ≤ Δ = T_target^(ℓₑ+1..N)
```

The paper argues Δ is generous because ℓₑ = N/2 gives you half the target
forward as overlap. But that requires the proxy at ℓₑ = N/2 to be
predictive enough that the draft's branches land on the target's final
choice.

On Gemma-3-12B with the native LM head, ℓₑ = N/2 gives 11% top-1 and
24% top-8 containment. Even κ=16 only reaches 29%. **The draft spends
most of its time computing branches that the target will reject.**

At deep exits (ℓₑ = 36 or 42) the proxy gets decent (44-50% top-1) but
Δ shrinks to 12-6 layers — not enough budget for γ≥7 draft tokens.

## Why this happens

Gemma-3 (like most modern dense LLMs) uses its late layers to do the
final "what token should come next" computation. Mid-layer hidden states
are still in feature space, not yet projected to output space. Applying
the model's LM head to them gives garbage until the representation is
nearly aligned with the output vocabulary.

This isn't a Gemma-specific quirk — most decoder-only transformers
behave this way unless **explicitly trained** with an early-exit
objective (as in DeeBERT, CALM, or per Mirror-SD's claim, a
purpose-built model). The Mirror-SD paper used Qwen3-14B and -32B; we
don't know whether those models happen to have decodable mid-layer
residuals, or whether the paper also trained an auxiliary early-exit
head (Section 3 is ambiguous; appendices not re-checked).

## Implications for Phase D

Three options:

1. **Drop the early-exit proxy, keep the overlap structure.** Run target
   and draft concurrently using the draft's own prediction as the seed
   for next-cycle. This is essentially Phase B.2 cross-cycle
   speculation, which we measured at ~35% hit rate and didn't give
   meaningful speedup. Not a path forward.

2. **Train an auxiliary LM head on mid-layer hidden states.** Distill
   the final LM head's behavior from mid-layer residuals. Requires
   training data + a training run. Significant project. Would give
   us a fair reproduction of Mirror-SD's technique.

3. **Move to Phase E (tree speculation) first.** This gets us value
   from the ANE batch=64 capacity independent of the early-exit
   assumption. Standard SD with tree branching: at each draft step,
   emit top-2 or top-4 candidates, evaluate all via batched ANE
   forward, verify best path on target. No assumption on mid-layer
   decodability.

## Decision

Skip Phase D.1/D.2 as designed. The Mirror-SD protocol as written
assumes model architecture (or training) that Gemma-3-12B does not
satisfy. Rather than chasing an auxiliary LM head (out of scope for
this phase of research), we proceed directly to Phase E (tree
speculation), which:

- Uses the same ANE batch=64 capacity Mirror-SD would use on its draft
- Captures the "parallel candidate exploration" value proposition
- Has no assumption on logit-lens decodability
- Is a clean baseline to layer Mirror-SD on top of LATER if we train
  an auxiliary head

This D.0 result is the real deliverable of Phase D: a concrete
measurement showing that off-the-shelf Gemma-3 is not a drop-in target
for Mirror-SD, and that any future Mirror-SD work needs to start with
an auxiliary early-exit head.
