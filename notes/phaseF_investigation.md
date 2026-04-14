# Phase F investigation: porting DFlash draft to ANE

**Date:** 2026-04-14

Goal: assess feasibility of moving DFlash's block-diffusion draft from
GPU (MLX) to ANE (CoreML via our ANEMLL fork). This is the core of
DDTree-on-Apple-Silicon: heterogeneous compute + block-diffusion draft +
(eventually) tree branching.

## Key finding: the draft is 150 LOC of standard MLX

z-lab's own repo ships `dflash/model_mlx.py` — a native MLX port of the
DFlash draft. This is authoritative (written by the paper's authors,
not a third-party port like Aryagm/dflash-mlx). Architecture summary:

- **`DFlashAttention`**: almost a vanilla transformer attention block.
  The only non-standard piece: queries come from current tokens `x`,
  but keys and values are computed from the concatenation
  `c = concatenate([x_ctx, x])` where `x_ctx` is a projection of target
  hidden states. RoPE + SDPA + standard output projection.
- **`DFlashDecoderLayer`**: RMSNorm → DFlashAttention (+residual) →
  RMSNorm → MLP (+residual). Qwen3-style.
- **`DFlashDraftModel`**: embedding + hidden-state projection (`fc` +
  RMSNorm of target hidden concat) + stack of decoder layers + final
  norm + LM head.
- **Crucially**: `embed_tokens` and `lm_head` are **shared with the
  target model** via `.bind(target)`. They are not separate weights in
  the draft checkpoint.

No custom kernels, no exotic ops, nothing in the architecture that
ANEMLL can't already express.

## Inference call pattern

From `stream_generate` in `model_mlx.py`:

```
block = [last_accepted, MASK, MASK, ..., MASK]        # length K = block_size (15)
draft_logits = draft(block, target_hidden, cache)     # one parallel forward
draft_tokens = sampler(draft_logits[:, 1-K:])         # K-1 tokens in one shot

verify_input = [last_accepted] ++ draft_tokens        # length K
target_logits = target(verify_input, target_cache)
target_hidden = concat(target._hidden_states)         # captured via monkey-patch
target_tokens = sampler(target_logits)

accepted = first index where draft[i] != target[i]    # commit prefix + 1 bonus
```

The draft emits K-1 tokens per forward — one forward, not K sequential
forwards. This is exactly what the ANE's `batch=64` prefill path is
optimized for. Moving the draft to ANE plays to the ANE's strengths
instead of fighting them (our current Gemma-3-270M draft is
autoregressive — it fights the ANE).

## What the ANE port requires

**Existing ANEMLL machinery (our fork at `~/projects/anemll`) already
has:**
- Qwen3 converter (`anemll/ane_converter/qwen_converter.py`)
- Qwen3 model with ANE idioms (`anemll/models/qwen_model.py`):
  Conv2d-for-Linear, static KV cache, vocab splitting, LUT quant
- CoreML conversion pipeline with LUT6/LUT4 weight quantization
- Based on llama_model.py pattern, fp16, static context

**New pieces we need to build:**

1. **`DFlashAttention` ANE variant** — modified attention that takes
   both `x` and `x_ctx`, concatenates before computing K/V. This is a
   one-function modification to `qwen_model.py`'s attention class.
2. **External context input** — the CoreML model must accept
   `target_hidden` (shape `(B, K, concat_dim)` where concat_dim =
   num_target_layers × target_hidden_size) as an extra input. Straight
   CoreML input spec change.
3. **`fc` + `hidden_norm`** — linear + RMSNorm on the external hidden
   input. Trivial.
4. **Shared embed/lm_head handling** — on CoreML we can't share tensors
   with MLX. We duplicate the weights (target has ~780M params for
   embed+lm_head at Qwen3-4B; fits easily). The draft's bind() is
   replaced with "load target weights into ANE draft's embed/lm_head".
5. **Static cache** — draft uses dynamic MLX KVCache; ANE needs static
   slice-based cache. Our Phase B infrastructure (snapshot/restore for
   ANEDraft) handles exactly this pattern; reusable.
6. **Memory transfer** — target hidden must move MLX→numpy→CoreML each
   cycle. Per-cycle size ≈ K × concat_dim × 2 bytes = ~300 KB for
   K=15, Qwen3-4B, 4 target layers. Not a bottleneck.

## What's NOT in scope for the port

- The `GDNStateCapture` / gated-delta rollback machinery in
  `model_mlx.py` — that's only used for Qwen3.5 (which the model card
  flagged as incomplete). We target Qwen3 (dense), skip it.
- Training — we use z-lab's pretrained `z-lab/Qwen3-4B-DFlash-b16`
  checkpoint directly.

## Target model choice

The DFlash draft was trained to complement **Qwen3-4B**. The released
checkpoint `z-lab/Qwen3-4B-DFlash-b16` only works with this target.

Implication: our Phase F benchmarks use Qwen3-4B bf16 as target, not
Gemma-3-12B. That's a **different benchmark shape** from Phase B/C
(our Gemma-3-12B heterogeneous SD). We need both:

- Phase F.0 baseline: reproduce Aryagm/dflash-mlx benchmark on M4 Pro
  for Qwen3-4B bf16 + DFlash draft (GPU-only). From CLAUDE.md:
  expected ~43 tok/s (1.53× over 28.5 tok/s baseline).
- Phase F.1: same but draft on ANE. Measure vs baseline.
- Phase F.2: add tree branching (DDTree) on top of F.1.

Can't directly compare F.1 to Phase C numbers (different target model).
But we can compare F.1 to F.0 to isolate the ANE offload effect for
block-diffusion drafts specifically.

## Decision

Port is tractable. Estimated effort:

- F.0 (baseline reproduction on our M4 Pro): ½ day
- F.1 (ANE port of DFlash draft + integration): 3-5 days
- F.2 (DDTree tree branching on top of F.1): 3-5 days

Total: ~2 weeks of focused work. Delivers a genuinely novel result —
first heterogeneous ANE+MLX block-diffusion SD on Apple Silicon, and
first DDTree-style tree-branching variant on the ANE.

## Next concrete action

Start with F.0: clone Aryagm/dflash-mlx locally, run its benchmark on
Qwen3-4B bf16, establish ground truth numbers. This confirms:
- DFlash actually works end-to-end on this hardware
- Our baseline is comparable to CLAUDE.md's recorded numbers
- Target-only baseline for comparison is established

Before starting F.0, we also want to confirm: does z-lab/dflash/model_mlx.py
work as a drop-in alternative to Aryagm/dflash-mlx? (Both are MLX, but
the z-lab version is authoritative and actively maintained.) If yes,
use z-lab's — fewer intermediate dependencies.
