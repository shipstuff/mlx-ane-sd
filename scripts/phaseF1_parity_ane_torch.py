"""Phase F.1 step 2: ANE-variant PyTorch DFlash parity.

Two parity checks:
1. dflash_ane.DFlashDraftModelANE with mode='standard' vs dflash_torch.DFlashDraftModel
   This validates the stateless/traceable refactor produces the same outputs
   as the reference PyTorch — i.e., no behavioral change, just a shape/tracing
   friendly rewrite.

2. dflash_ane with mode='ane' (mean-subtract + layer_norm RMSNorm) vs mode='standard'.
   This measures the ANE-RMSNorm precision tax. Expected: ~1-4% top-1 flip
   on random positions, but similar on realistic prompts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download

from dflash_torch import DFlashConfig, DFlashDraftModel, load_dflash_from_hf
from dflash_ane import DFlashDraftModelANE, build_rope_for_offset, copy_weights, set_rmsnorm_mode


def mlx_to_torch(a: mx.array) -> torch.Tensor:
    return torch.from_numpy(np.asarray(a.astype(mx.float32))).to(torch.bfloat16)


def main():
    print("[load] target + drafts...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    torch_draft = load_dflash_from_hf(draft_path)

    # Use ctx_size = prompt_len to match the parity inputs exactly
    block_size = torch_draft.config.block_size
    ctx_size_for_test = len(tok.encode("The capital of France is Paris, which is known for",
                                         add_special_tokens=True))
    ane_draft = DFlashDraftModelANE(torch_draft.config,
                                     block_size=block_size,
                                     ctx_size=ctx_size_for_test).to(torch.bfloat16).eval()
    loaded, skipped = copy_weights(torch_draft, ane_draft)
    print(f"[info] ANE variant: {loaded} weights copied, {len(skipped)} skipped ({skipped[:3]}...)")

    # Build realistic inputs from a prompt
    from dflash.model_mlx import _patch_model
    prompt = "The capital of France is Paris, which is known for"
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    mlx_draft_cfg = torch_draft.config
    _patch_model(target, mlx_draft_cfg.target_layer_ids)
    prompt_arr = mx.array(prompt_ids)[None]
    target_cache = make_prompt_cache(target)
    with mx.stream(mx.default_stream(mx.default_device())):
        _ = target(prompt_arr, target_cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
    mx.eval(hidden)

    block_size = mlx_draft_cfg.block_size
    ctx_len = hidden.shape[1]
    ctx = hidden  # (1, prompt_len, concat_dim)
    mask_id = mlx_draft_cfg.mask_token_id
    block_tokens = [int(prompt_ids[-1])] + [mask_id] * (block_size - 1)
    block = mx.array([block_tokens])
    embed = target.model.embed_tokens
    noise_emb = embed(block)
    mx.eval(noise_emb)

    noise_emb_t = mlx_to_torch(noise_emb)
    ctx_t = mlx_to_torch(ctx)

    # Build RoPE for offset=0 (first cycle)
    cos_q, sin_q, cos_k, sin_k = build_rope_for_offset(mlx_draft_cfg, offset=0,
                                                       ctx_len=ctx_len, block_size=block_size)
    print(f"[info] RoPE shapes: cos_q={cos_q.shape}, cos_k={cos_k.shape}")
    print(f"[info] noise_emb={noise_emb_t.shape}, ctx={ctx_t.shape}")

    # Reference: plain dflash_torch (this was proven to match MLX)
    with torch.no_grad():
        ref_hidden, _, _ = torch_draft(noise_emb_t, ctx_t, cache_offset=0)

    # ANE variant, standard mode
    set_rmsnorm_mode("standard")
    with torch.no_grad():
        ane_std_hidden = ane_draft(noise_emb_t, ctx_t, cos_q, sin_q, cos_k, sin_k)

    # ANE variant, ane mode (mean-subtract + layer_norm)
    set_rmsnorm_mode("ane")
    with torch.no_grad():
        ane_ane_hidden = ane_draft(noise_emb_t, ctx_t, cos_q, sin_q, cos_k, sin_k)

    # Apply target LM head to each (via MLX round-trip)
    def apply_lm_head(h: torch.Tensor) -> np.ndarray:
        arr = h.to(torch.float32).cpu().numpy()
        mx_h = mx.array(arr).astype(mx.bfloat16)
        if hasattr(target, "lm_head"):
            logits = target.lm_head(mx_h)
        else:
            logits = target.model.embed_tokens.as_linear(mx_h)
        mx.eval(logits)
        return np.asarray(logits.astype(mx.float32))[0]   # (L, V)

    ref_logits = apply_lm_head(ref_hidden)
    std_logits = apply_lm_head(ane_std_hidden)
    ane_logits = apply_lm_head(ane_ane_hidden)

    def compare(a, b, name):
        cos_per_pos = np.array([
            np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-8)
            for i in range(a.shape[0])
        ])
        top1_a = a.argmax(axis=-1)
        top1_b = b.argmax(axis=-1)
        match = (top1_a == top1_b).mean()
        print(f"  {name:40} cos_mean={cos_per_pos.mean():.6f}  "
              f"top1={match:.1%} ({int(match*a.shape[0])}/{a.shape[0]})")
        return cos_per_pos.mean(), match

    print("\n[parity] dflash_torch (ref) vs dflash_ane-standard:")
    compare(ref_logits, std_logits, "std vs ref")

    print("\n[parity] dflash_ane-standard vs dflash_ane-ane (RMSNorm mode tax):")
    compare(std_logits, ane_logits, "ane vs std")

    print("\n[parity] dflash_torch (ref) vs dflash_ane-ane:")
    compare(ref_logits, ane_logits, "ane vs ref")


if __name__ == "__main__":
    main()
