# Phase F.2 tree speculation: naive top-κ is a net loss

**Date:** 2026-04-14

## What we tried

For DFlash's bidirectional draft, I initially thought tree speculation
would "collapse" to a simpler top-κ acceptance criterion (since
alternative first-token choices don't change draft logits at other
positions under bidirectional attention). I implemented:

- Draft outputs logits for all bs=16 positions in one pass
- At each position, compute top-κ candidate tokens
- Verify as usual: target processes draft's top-1 chain
- Accept at position i if target's choice is in draft's top-κ at i
- Stop at first mismatch (target's choice not in top-κ) OR first top-κ boundary hit

## Why it doesn't work

When target picks draft's top-2 at position i (not top-1), we MUST stop:
target's subsequent logits at position i+1 were computed assuming the
chain contained draft's top-1 at position i, so they're invalid after
we commit a different token.

So top-κ boundary events can fire BEFORE the natural top-1 mismatch
would have. If we would have had 7 top-1 matches in a row (high-
acceptance cycle), top-κ might trigger at position 2 and stop us at
3 commits. **Net loss** for high-acceptance prompts.

## Results

Same 4 prompts, Qwen3-4B bf16 target, max_new=100, κ=2 vs κ=1:

| Prompt | κ=1 (baseline) | κ=2 | Δ |
|---|---:|---:|---:|
| capital | 16.80 | 29.15 | **+74%** |
| fibonacci | 67.50 | 38.89 | -42% |
| math | 33.52 | 26.60 | -21% |
| story | 18.08 | 16.09 | -11% |
| mean | 33.84 | 27.70 | **-18%** |

On prompts where standard SD rarely has a full top-1 streak (capital,
story), top-κ sometimes catches the first mismatch and gains a token.
But on high-acceptance prompts (fibonacci, math), top-κ triggers
*too early* and cuts streaks short.

## What proper DDTree needs

The DDTree paper's Algorithm 1 (read from the PDF) builds a
best-first tree with κ candidates per position, pruned by a node
budget B=512. Then:

> To verify the selected draft tree in one target-model forward pass,
> we flatten it into a sequence of token ids rooted at the bonus
> token b. We assign position ids by tree depth so that the verifier
> applies the correct positional encoding. We then use **tree
> attention [19]**, under which each drafted node attends to the past
> context through the KV cache and, within the drafted tree, **only
> to the root, its ancestors, and itself**.

Without tree attention, each candidate branch needs its own target
forward, making even small trees (B=16) cost 16× the single-chain
target work. Tree attention lets target evaluate the ENTIRE tree in
one forward, keeping target compute constant.

## Implementing tree attention in MLX

Would require:
1. Custom attention mask for target forward (tree-structured, not
   causal)
2. Packed sequence of candidate branches with depth-based position IDs
3. Per-branch KV cache rollback on reject
4. Walk-based acceptance over the tree

This is a ~1-week engineering project — modifies mlx-lm's attention
module for the target side. Doable but much larger scope than the F.1
port.

## Alternative: multi-draft forward, single-candidate verify

Simpler approach — exploit that the draft is fast on ANE:
1. Run N draft forwards with different sampling temperatures (or
   DIFFERENT top-1 choices at first position by explicitly setting it)
2. Build N candidate chains
3. Verify via batch=N target forward (single shot on GPU)
4. Accept the best-length chain

For N=2, this is ~2× draft cost (on ANE, still cheap) + 1 batch=2
target forward (cheaper than 2 separate forwards). Gain from the
extra chain should exceed the cost.

But: MLX target forward with batch>1 is non-trivial to set up with
the trim-on-partial-reject pattern. Each batch element would need
its own KV cache state. Tractable but another half-week.

## Decision

Abandon the naive top-κ approach. Document the negative finding. Two
paths forward for true DDTree:

**Path A**: Implement tree attention for target. Paper-faithful, ~1 week.
Highest confidence of delivering the paper's claimed 1.3-2× over DFlash.

**Path B**: Batched multi-chain verify (simpler, ~half week). Likely
smaller gain (~10-20%) but simpler engineering.

Neither gets started this week. The F.1 accumulating cache work is
the week's main deliverable — first ANE port of DFlash with
acceptance competitive with native MLX, fully validated under
contention.

## Current F.1 state recap

Best configurations:
- **Short generations (≤8 cycles, ~250 tok)**: unquant S=256 → 34.68 tok/s
  (85% of F.0 solo 40.92)
- **Long generations**: LUT4 per_grouped_channel S=1024 → 24-32 tok/s
  across 100-500 tok (80-85% of F.0)
- **Under contention**: F.1 preserves 83-85% vs F.0's 80%. At heavy
  contention, F.1 wins on prose/factual prompts.

Files:
- `scripts/dflash_ane_accumcache.py` — model (100% ANE)
- `scripts/phaseF1_ane_stream_accum.py` — SD runner with Python-managed cache
- `scripts/dflash_lut_quantize.py` — LUT4/6 post-processing
- `scripts/phaseF2_ane_tree.py` — F.2 top-κ experiment (negative result)
