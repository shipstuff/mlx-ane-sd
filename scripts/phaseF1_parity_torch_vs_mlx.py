"""Phase F.1 step 1: PyTorch DFlash parity vs MLX DFlash.

Verifies that our PyTorch rewrite of DFlashDraftModel produces numerically
identical output to z-lab's MLX reference on the same input. This is
the foundation for the ANE port — we need a tensor-only PyTorch model
that traces cleanly, and it must match the authoritative implementation.

Procedure:
1. Load Qwen3-4B bf16 target (MLX).
2. Run target on a real prompt, capturing hidden states at target_layer_ids.
3. Load MLX DFlash draft and our PyTorch DFlash draft.
4. Build the same block input (= [last_token, MASK, ..., MASK]) for both.
5. Run both drafts with cold cache.
6. Compare: max abs diff, cos sim, top-1 agreement.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import KVCache, make_prompt_cache

from dflash.model_mlx import load_draft as mlx_load_draft
from dflash_torch import DFlashConfig, DFlashDraftModel, load_dflash_from_hf
from huggingface_hub import snapshot_download


TARGET_ID = "mlx-community/Qwen3-4B-bf16"
DRAFT_ID = "z-lab/Qwen3-4B-DFlash-b16"


def mlx_to_torch(a: mx.array) -> torch.Tensor:
    """Cast MLX bf16 array to torch via numpy fp32 intermediary (bf16 numpy is iffy)."""
    return torch.from_numpy(np.asarray(a.astype(mx.float32))).to(torch.bfloat16)


def torch_to_mlx(t: torch.Tensor) -> mx.array:
    arr = t.to(torch.float32).cpu().numpy()
    return mx.array(arr).astype(mx.bfloat16)


def main():
    print(f"[load] MLX target {TARGET_ID}...")
    target, tok = mlx_load(TARGET_ID)

    print(f"[load] MLX draft  {DRAFT_ID}...")
    mlx_draft = mlx_load_draft(DRAFT_ID)
    mlx_draft.bind(target)

    draft_path = snapshot_download(DRAFT_ID)
    print(f"[load] PyTorch draft from {draft_path}...")
    torch_draft = load_dflash_from_hf(draft_path)

    torch_params = sum(p.numel() for p in torch_draft.parameters())
    print(f"[info] torch params: {torch_params/1e6:.1f}M")

    # ---- Build input via a real target forward ----
    prompt = "The capital of France is Paris, which is known for"
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    print(f"[info] prompt ids len={len(prompt_ids)}")

    # Patch the MLX target model to capture hidden states (same as dflash.model_mlx does)
    from dflash.model_mlx import _patch_model
    _patch_model(target, mlx_draft.config.target_layer_ids)

    prompt_arr = mx.array(prompt_ids)[None]
    target_cache = make_prompt_cache(target)
    with mx.stream(mx.default_stream(mx.default_device())):
        logits = target(prompt_arr, target_cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
    mx.eval(logits, hidden)

    # Hidden is the target_hidden context. Only the last block_size positions matter
    # during decode, but for parity we can feed any slice. Use the last `block_size`
    # positions to mirror real usage.
    block_size = mlx_draft.config.block_size
    ctx = hidden[:, -block_size:, :]  # (1, S=block_size, num_target_layers * H)
    print(f"[info] target_hidden ctx shape (mlx): {ctx.shape}, dtype={ctx.dtype}")

    # Build a dummy block: [bos_or_prev_token, MASK, MASK, ...]
    mask_id = mlx_draft.config.mask_token_id
    block_tokens = [int(prompt_ids[-1])] + [mask_id] * (block_size - 1)
    block = mx.array([block_tokens])
    print(f"[info] block tokens: {block_tokens[:4]}...(len {len(block_tokens)})")

    # Get noise_embedding via target.embed_tokens
    # In MLX the target is Model with model.embed_tokens
    embed = target.model.embed_tokens
    noise_emb = embed(block)  # (1, block_size, H)
    mx.eval(noise_emb)

    # ---- Run MLX draft (cold cache) ----
    print("\n[run] MLX draft...")
    draft_cache = [KVCache() for _ in range(mlx_draft.config.num_hidden_layers)]

    # Looking at DFlashDraftModel.__call__(inputs, target_hidden, cache), where inputs
    # is token ids. The MLX version computes embeddings internally via embed_tokens
    # (bound). We want to feed the same thing.
    t0 = time.perf_counter()
    mlx_logits = mlx_draft(block, ctx, draft_cache)
    mx.eval(mlx_logits)
    t_mlx = time.perf_counter() - t0
    print(f"  MLX draft logits shape: {mlx_logits.shape}, dtype: {mlx_logits.dtype}, {t_mlx*1000:.1f}ms")

    # MLX runs lm_head inside (via bind). Our PyTorch does NOT — it returns hidden.
    # So to compare, we need to invert: run MLX without lm_head, OR apply target.lm_head
    # to our PyTorch output.

    # Cleanest: compute what MLX would have produced PRE-lm_head by running its forward
    # step by step. But simpler — apply lm_head to our PyTorch output and compare logits.

    # ---- Run PyTorch draft (cold cache) ----
    print("\n[run] PyTorch draft...")
    noise_emb_t = mlx_to_torch(noise_emb)   # (1, block_size, H)
    ctx_t = mlx_to_torch(ctx)                # (1, block_size, num_target_layers * H)

    with torch.no_grad():
        t0 = time.perf_counter()
        hidden_out, new_ks, new_vs = torch_draft(noise_emb_t, ctx_t, cache_offset=0)
        t_torch = time.perf_counter() - t0
    print(f"  PyTorch draft hidden shape: {hidden_out.shape}, dtype: {hidden_out.dtype}, {t_torch*1000:.1f}ms")

    # Apply MLX target LM head to torch output. For Qwen3-4B,
    # tie_word_embeddings=True so the LM head is embed_tokens.as_linear.
    hidden_out_mx = torch_to_mlx(hidden_out)
    if hasattr(target, "lm_head"):
        torch_logits_via_lmhead = target.lm_head(hidden_out_mx)
    else:
        torch_logits_via_lmhead = target.model.embed_tokens.as_linear(hidden_out_mx)
    mx.eval(torch_logits_via_lmhead)

    # ---- Compare ----
    print("\n[compare]")
    a = np.asarray(mlx_logits.astype(mx.float32))[0]                  # (block_size, vocab)
    b = np.asarray(torch_logits_via_lmhead.astype(mx.float32))[0]     # (block_size, vocab)

    max_abs = np.max(np.abs(a - b))
    mean_abs = np.mean(np.abs(a - b))
    cos_per_pos = np.array([
        np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-8)
        for i in range(a.shape[0])
    ])

    top1_mlx = a.argmax(axis=-1)
    top1_torch = b.argmax(axis=-1)
    top1_match = (top1_mlx == top1_torch).mean()

    print(f"  shape: a={a.shape} b={b.shape}")
    print(f"  max_abs diff: {max_abs:.4f}")
    print(f"  mean_abs diff: {mean_abs:.4f}")
    print(f"  cos_sim per position: min={cos_per_pos.min():.6f} mean={cos_per_pos.mean():.6f}")
    print(f"  top-1 agreement: {top1_match:.1%}  ({np.sum(top1_mlx==top1_torch)}/{len(top1_mlx)})")
    print(f"  mlx top-1: {top1_mlx[:8].tolist()}")
    print(f"  tch top-1: {top1_torch[:8].tolist()}")

    if top1_match >= 0.9 and cos_per_pos.mean() > 0.999:
        print("\n[OK] parity looks good — ready for ANE conversion")
    else:
        print("\n[WARN] parity is too loose — debug the PyTorch rewrite before converting")


if __name__ == "__main__":
    main()
