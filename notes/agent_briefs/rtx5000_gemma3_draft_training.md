# Agent brief: train a DFlash-style block-diffusion draft for Gemma-3

**For:** an agent on the RTX 5000 workstation (16 GB VRAM, 64 GB RAM, 14 vCPU)
**Dependencies converge with:** M4 Pro agents (mini-02, mini-03) after Week 2
**Hand-off point:** trained draft checkpoint in z-lab DFlash format on HuggingFace Hub
**Timeline:** 2 weeks

## Mission

Train a block-diffusion speculative-decoding draft for Google's **Gemma-3-4B**
target, using z-lab's [DFlash](https://github.com/z-lab/dflash) approach.
This unlocks our existing ANE port pipeline (currently Qwen3-4B-only) to
work on Gemma-3 targets, giving us a second family of benchmarks and
eventually paving the path to Gemma-3-12B (which our broader research has
already characterized deeply on the MLX side).

## Context (why this matters)

We've completed an ANE port of z-lab's Qwen3-4B DFlash draft that runs
100% on Apple Neural Engine at 79-85% of the GPU baseline's throughput and
wins under GPU contention (details in the `mlx-ane-sd` repo under `notes/`,
especially `week_of_2026-04-14_summary.md`). The port only works on
Qwen3-4B because z-lab only released one draft checkpoint.

To extend to Gemma-3 (our phases A-C target), we need a Gemma-3-specific
block-diffusion draft — **architectural mismatches prevent reusing
Qwen3-4B-DFlash**: different tokenizer, hidden size, layer count, layer
types, and RoPE theta.

Your draft becomes a key input for Week 3+ of the broader research program.

## Scope (two milestones, stop-and-report between them)

### Milestone 1 (Week 1): Gemma-3-1B sanity training

**Goal:** validate the training pipeline end-to-end on the smallest target.

- Target: `google/gemma-3-1b-it` (~1B params fp16, ~2 GB)
- Draft: 5 transformer layers, ~300-500M params (scale fc/hidden accordingly)
- Training: ~1-2 hours on the RTX 5000, 10-50M tokens of UltraChat
- Success criterion: coherent generation using the draft + positive
  acceptance rate vs mlx-lm's stock SD on the same pair

**Stop-and-report checkpoint** — do not proceed to Milestone 2 without
a green light. Report:
- Training loss curve
- Generated output samples (5 diverse prompts)
- Acceptance rate measured against stock Gemma-3-1B SD

### Milestone 2 (Week 2): Gemma-3-4B production training

**Goal:** a real, usable DFlash-style draft for Gemma-3-4B.

- Target: `google/gemma-3-4b-it` (~8 GB fp16)
- Draft: 5 transformer layers, scale to match Gemma-3-4B's hidden_size (2560)
- Training: 12-24 hours, ~100-200M tokens of UltraChat
- Success criterion: comparable or better SD speedup than mlx-lm's stock
  SD on Gemma-3-4B

**Deliverable:** HuggingFace Hub repo (name suggestion:
`carl/Gemma-3-4b-DFlash-b16`) with checkpoint in z-lab's exact format.

## Hardware constraints

- **RTX 5000**: 16 GB VRAM. Gemma-3-4B fp16 (~8 GB) + draft (~1 GB) +
  activations + gradients fits with batch_size=1-2 and gradient
  accumulation. **Do NOT try Gemma-3-12B — it won't fit.**
- No cloud compute available for now. Work within local limits.
- 64 GB system RAM is fine for offloading if needed.
- Python: whatever z-lab recommends (their repo requires CUDA + specific
  PyTorch version)

## Reference materials

**z-lab/dflash repo**: `git clone https://github.com/z-lab/dflash.git`
- `dflash/model.py` — PyTorch training architecture (use this as your base)
- `dflash/model_mlx.py` — MLX reference (read for semantics, don't train with)
- `dflash/benchmark.py` — their eval harness (useful for your success criterion)

**Released draft (format reference, clone locally):**
- `huggingface-cli download z-lab/Qwen3-4B-DFlash-b16`
- Study `config.json` — your output config must match this structure
- Study `model.safetensors` keys — your output weights must use same naming

**Training data:**
- `HuggingFaceH4/ultrachat_200k` via `datasets.load_dataset`
- Standard instruction-tuning corpus, ~200k multi-turn conversations

**Our integration pipeline** (so you know what your output feeds into):
- Repo: `mlx-ane-sd` (see README + `notes/week_of_2026-04-14_summary.md`)
- Loader: `scripts/dflash_torch.py::load_dflash_from_hf` — this is what
  will load your checkpoint. Make sure your format works with it.

## Gemma-3 adaptations required

z-lab's training script targets Qwen3. Adapt these:

**1. Target model loader**
- Replace `Qwen3ForCausalLM` / `AutoModelForCausalLM` loading with Gemma-3
- Gemma-3-1B config: `hidden_size=1152, num_hidden_layers=26, num_attention_heads=4, num_key_value_heads=1, head_dim=256, rope_theta=10000`
- Gemma-3-4B config: `hidden_size=2560, num_hidden_layers=34, num_attention_heads=8, num_key_value_heads=4, head_dim=256, rope_theta=10000`
- Verify against `config.json` on HF — don't trust my numbers above blindly.

**2. target_layer_ids**
- z-lab's Qwen3-4B picks `[1, 9, 17, 25, 33]` — 5 roughly-evenly-spaced
  layers from a 36-layer target.
- For Gemma-3-4B (34 layers): try `[1, 8, 16, 24, 32]`
- For Gemma-3-1B (26 layers): try `[1, 6, 12, 18, 24]`
- Their choice is heuristic; if you have time, do a quick ablation.

**3. Vocabulary / embed sharing**
- **CRITICAL**: DFlash draft shares `embed_tokens` and `lm_head` with target
  (via `.bind()` at inference, or equivalent during training).
- Gemma-3 vocab size: 262208 (not Qwen's 151936). Make sure the bind call
  loads Gemma's tokens.
- Gemma-3 uses `tie_word_embeddings=True` — embed and lm_head are the
  same tensor. Qwen3-4B is also tied — should be clean.

**4. Gemma-3 attention pattern**
- Gemma-3 uses sliding-window + global attention (alternating, pattern
  `sliding_window_pattern=6` in HF config). DFlash's `target_hidden`
  extraction should still work (just captures whatever hidden state is
  produced by the layer, regardless of attention type). But verify that
  your hidden-state extraction is capturing sensible states across both
  attention types.

**5. Mask token ID**
- Qwen3 uses `151669` (random vocab token) as DFlash's MASK. For Gemma-3,
  pick a similarly-unused token. Gemma-3 has ~100 reserved `<unused_*>`
  tokens in its vocab — pick one of those.

**6. RMSNorm**
- Gemma-3 uses standard RMSNorm (not the +1 shift that Qwen3.5 has).
  DFlash's draft RMSNorm is standard — should match.

## Deliverable format

Final HF repo structure (mirror z-lab/Qwen3-4B-DFlash-b16):

```
repo/
├── config.json           # z-lab DFlash schema
├── model.safetensors     # trained weights
├── tokenizer.json        # Gemma-3 tokenizer (copy from target)
├── tokenizer_config.json
└── README.md             # brief description
```

**config.json must include:**
```json
{
  "architectures": ["DFlashDraftModel"],
  "model_type": "qwen3",   // this is what z-lab uses; keep for compatibility
  "block_size": 16,
  "dflash_config": {
    "mask_token_id": <your chosen unused token>,
    "target_layer_ids": [1, 8, 16, 24, 32]   // or whatever you picked
  },
  "num_target_layers": 34,   // 34 for Gemma-3-4B target
  "hidden_size": 2560,        // your draft's hidden (can match target)
  "num_hidden_layers": 5,     // draft depth
  "num_attention_heads": 32,  // your draft's heads
  "num_key_value_heads": 8,   // your draft's kv heads
  "head_dim": 128,
  "intermediate_size": 9728,  // draft MLP dim
  "vocab_size": 262208,        // Gemma-3 vocab
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000,         // Gemma-3's theta
  "max_position_embeddings": 131072,
  "tie_word_embeddings": true,
  "dtype": "bfloat16"
}
```

**safetensors key naming (must match z-lab exactly):**
- `fc.weight`
- `hidden_norm.weight`
- `norm.weight`
- `layers.{i}.input_layernorm.weight`
- `layers.{i}.post_attention_layernorm.weight`
- `layers.{i}.self_attn.q_proj.weight`
- `layers.{i}.self_attn.k_proj.weight`
- `layers.{i}.self_attn.v_proj.weight`
- `layers.{i}.self_attn.o_proj.weight`
- `layers.{i}.self_attn.q_norm.weight`
- `layers.{i}.self_attn.k_norm.weight`
- `layers.{i}.mlp.gate_proj.weight`
- `layers.{i}.mlp.up_proj.weight`
- `layers.{i}.mlp.down_proj.weight`

You can validate the format by running our loader on your checkpoint:
```bash
# On the RTX 5000, clone our repo
git clone <mlx-ane-sd-repo-url>
cd mlx-ane-sd
pip install safetensors torch huggingface-hub
python -c "
from scripts.dflash_torch import load_dflash_from_hf
model = load_dflash_from_hf('./your_checkpoint_dir')
print('params:', sum(p.numel() for p in model.parameters()) / 1e6, 'M')
"
```
If this runs without errors, format is correct.

## Training recipe starting points

Based on z-lab's Qwen3-4B-DFlash release (infer from what's published):
- Optimizer: AdamW, lr=1e-4, warmup 1k steps, cosine decay to 1e-5
- Batch size: effective 64-256 (accumulate on RTX 5000)
- Context length: 2048 or 4096 depending on memory
- Objective: block diffusion denoising (their custom loss from `dflash.model.DFlashDraftModel`)
- Mixed precision: bf16 (RTX 5000 doesn't support native bf16 — use fp16
  with loss scaling, OR use fp32 master weights. z-lab's code may need
  adaptation for this.)

**If bf16 is critical (z-lab uses it)** and fp16 numerically diverges, fall
back to float32 training — slower but works. Document the choice.

## Escalation triggers

Stop and ping the user if any of these fire:

1. **OOM that can't be solved with batch=1 + grad accum + grad checkpointing.**
   Don't burn days on memory debugging.
2. **z-lab's training code uses ops that don't port to Gemma-3.** E.g.,
   if they use Qwen3-specific MoE ops or flash-attention configs that
   break on Gemma's architecture, report back before spending more than
   a day adapting.
3. **Training loss diverges or stalls** at any milestone. Report with
   loss curves and config, don't keep running.
4. **Milestone 1 output is incoherent** — that's a sign of fundamental
   pipeline bug. Don't proceed to expensive Milestone 2 training.

## Working style

- Commit often to a clean git branch. Clear commit messages.
- Write `notes/training_log.md` with running observations — problems
  encountered, what you tried, what worked.
- Preserve all checkpoints (disk is cheap; reproducing is expensive).
- If you need to deviate from this plan for good reason, document why
  and proceed. Just don't silently swap approaches.

## What to expect from us after hand-off

Once you deliver the Gemma-3-4B draft:

1. We'll port it to our ANE pipeline (convert PyTorch → CoreML, compile for ANE)
2. Benchmark against stock mlx-lm Gemma-3-4B SD
3. Run contention studies (Phase C pattern)
4. Integrate with multi-stream serving work from Week 1-2
5. Consolidate findings into the final paper/release

## Questions you should ask before starting

If unclear about any of these, ping the user before burning cycles:

- Data preprocessing: use z-lab's exact UltraChat formatting or a simpler version?
- Checkpoint granularity: save every N steps, keep best-k by eval loss?
- HF Hub upload: push to org account or personal? Which org?

**Ready to start. Good luck.**
