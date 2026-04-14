"""PyTorch reference implementation of DFlash draft model.

Goal: a tensor-in / tensor-out PyTorch module that mirrors
z-lab/Qwen3-4B-DFlash-b16 bit-for-bit, loadable from the HF safetensors
checkpoint, with NO dependency on the `transformers` library or the
`trust_remote_code` path.

Why: we need this as an intermediate step to port the draft to the ANE
via coremltools.convert. HF's DFlash code uses DynamicCache +
Qwen3PreTrainedModel which don't trace cleanly. A plain PyTorch module
with tensor-only inputs/outputs does.

The reference we match:
- z-lab/dflash/model_mlx.py (authoritative MLX port) for runtime shape/
  dataflow
- z-lab/dflash/model.py (PyTorch via transformers) for weight layout

DFlash attention twist: queries come from current block hidden states,
but keys and values are computed on the concatenation of
(target_hidden projected through fc, current block hidden states).
Attention is fully BIDIRECTIONAL within the attended positions — the
draft is essentially doing a diffusion denoising step, so it sees
"future" block positions freely. No causal mask.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class DFlashConfig:
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: list[int]
    num_target_layers: int
    mask_token_id: int

    @classmethod
    def from_hf_json(cls, path: str | Path) -> "DFlashConfig":
        data = json.loads(Path(path).read_text())
        dflash_cfg = data["dflash_config"]
        return cls(
            hidden_size=data["hidden_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data["head_dim"],
            intermediate_size=data["intermediate_size"],
            vocab_size=data["vocab_size"],
            rms_norm_eps=data["rms_norm_eps"],
            rope_theta=data["rope_theta"],
            max_position_embeddings=data["max_position_embeddings"],
            block_size=data["block_size"],
            target_layer_ids=list(dflash_cfg["target_layer_ids"]),
            num_target_layers=data["num_target_layers"],
            mask_token_id=dflash_cfg["mask_token_id"],
        )


# ---------------------------------------------------------------------------
# Building blocks — plain PyTorch (not yet ANE-legal; we port later)
# ---------------------------------------------------------------------------


class RMSNorm(nn.Module):
    """Standard RMSNorm (matches mlx nn.RMSNorm and transformers Qwen3RMSNorm)."""
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_ = x.to(torch.float32)
        variance = x_.pow(2).mean(-1, keepdim=True)
        x_ = x_ * torch.rsqrt(variance + self.eps)
        return (self.weight * x_).to(orig_dtype)


def _rope_base_freqs(dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


def _build_cos_sin(positions: torch.Tensor, inv_freq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # positions: (L,), inv_freq: (D/2,)
    freqs = torch.outer(positions.to(torch.float32), inv_freq)  # (L, D/2)
    emb = torch.cat([freqs, freqs], dim=-1)  # (L, D)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary(q: torch.Tensor, k: torch.Tensor,
                  cos_q: torch.Tensor, sin_q: torch.Tensor,
                  cos_k: torch.Tensor, sin_k: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q: (B, H, Lq, D), k: (B, H, Lk, D)
    # cos_q/sin_q: (Lq, D), cos_k/sin_k: (Lk, D) — broadcast to (1,1,L,D)
    q_embed = q * cos_q.unsqueeze(0).unsqueeze(0) + rotate_half(q) * sin_q.unsqueeze(0).unsqueeze(0)
    k_embed = k * cos_k.unsqueeze(0).unsqueeze(0) + rotate_half(k) * sin_k.unsqueeze(0).unsqueeze(0)
    return q_embed, k_embed


class DFlashAttention(nn.Module):
    """Draft attention with cross-stream K/V from concat(target_ctx, current_block)."""
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor,
                inv_freq: torch.Tensor, cache_offset: int,
                past_k: Optional[torch.Tensor] = None,
                past_v: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x:      (B, L, H)      current block embeddings
        x_ctx:  (B, S, H)      projected target hidden context
        inv_freq: (head_dim/2,) rope frequencies
        cache_offset: int      how many prior draft tokens are in past_k/v
        past_k: (B, Hkv, P, D) or None
        past_v: same

        Returns:
            output (B, L, H),
            new_past_k (B, Hkv, P+S+L, D),
            new_past_v same
        """
        B, L, H = x.shape
        S = x_ctx.shape[1]

        # Concat ctx || block on the token axis for K/V computation
        c = torch.cat([x_ctx, x], dim=1)            # (B, S+L, H)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)     # (B, L, Hq, D)
        k_raw = self.k_proj(c).view(B, S + L, self.n_kv_heads, self.head_dim)  # (B, S+L, Hkv, D)
        v_raw = self.v_proj(c).view(B, S + L, self.n_kv_heads, self.head_dim)

        # Per-head norm, then transpose to (B, H, T, D)
        q = self.q_norm(q).transpose(1, 2)           # (B, Hq, L, D)
        k_raw = self.k_norm(k_raw).transpose(1, 2)   # (B, Hkv, S+L, D)
        v_raw = v_raw.transpose(1, 2)                # (B, Hkv, S+L, D)

        # RoPE positions — mirror MLX behavior:
        #   queries: offset = cache_offset + S (x_noise starts past prior cache + ctx)
        #   keys: offset = cache_offset (start of concat'd ctx+x)
        q_positions = torch.arange(L, device=q.device) + cache_offset + S
        k_positions = torch.arange(S + L, device=k_raw.device) + cache_offset
        cos_q, sin_q = _build_cos_sin(q_positions, inv_freq.to(q.device))
        cos_k, sin_k = _build_cos_sin(k_positions, inv_freq.to(k_raw.device))
        # Cast rope to activations dtype
        cos_q, sin_q = cos_q.to(q.dtype), sin_q.to(q.dtype)
        cos_k, sin_k = cos_k.to(k_raw.dtype), sin_k.to(k_raw.dtype)
        q, k_raw = apply_rotary(q, k_raw, cos_q, sin_q, cos_k, sin_k)

        # Append to past cache: full K/V = concat(past, fresh)
        if past_k is not None:
            k_full = torch.cat([past_k, k_raw], dim=2)   # (B, Hkv, P+S+L, D)
            v_full = torch.cat([past_v, v_raw], dim=2)
        else:
            k_full = k_raw
            v_full = v_raw

        # GQA expand: repeat k/v to match query heads
        if self.n_kv_heads != self.n_heads:
            rep = self.n_heads // self.n_kv_heads
            k_exp = k_full.repeat_interleave(rep, dim=1)
            v_exp = v_full.repeat_interleave(rep, dim=1)
        else:
            k_exp = k_full
            v_exp = v_full

        # Fully bidirectional attention (no mask) — matches MLX mx.fast.SDPA with None mask
        out = F.scaled_dot_product_attention(q, k_exp, v_exp, attn_mask=None,
                                              is_causal=False, scale=self.scale)
        # (B, Hq, L, D) -> (B, L, H)
        out = out.transpose(1, 2).reshape(B, L, -1)
        out = self.o_proj(out)

        return out, k_full, v_full


class DFlashMLP(nn.Module):
    """Qwen3-style gated MLP: silu(gate(x)) * up(x) -> down."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.self_attn = DFlashAttention(config)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor,
                inv_freq: torch.Tensor, cache_offset: int,
                past_k: Optional[torch.Tensor] = None,
                past_v: Optional[torch.Tensor] = None
                ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_attn, new_k, new_v = self.self_attn(
            self.input_layernorm(x), x_ctx, inv_freq, cache_offset, past_k, past_v
        )
        x = x + h_attn
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_k, new_v


class DFlashDraftModel(nn.Module):
    """Complete DFlash draft.

    Inputs:
        noise_embedding: (B, L, H)            — embeddings of the block tokens
                                                  (computed externally via target.embed_tokens)
        target_hidden:   (B, S, num_target_layers * H) — concat of hidden states
                                                  captured at target_layer_ids
        cache_offset: int                     — current length of the draft KV cache
        past_ks: list[Optional[Tensor]] len=num_layers — per-layer past K
        past_vs: same

    Returns:
        out_hidden: (B, L, H)                  — what the caller feeds to target.lm_head
        new_ks, new_vs: updated caches
    """
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        inv_freq = _rope_base_freqs(config.head_dim, config.rope_theta)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor,
                cache_offset: int = 0,
                past_ks: Optional[list] = None, past_vs: Optional[list] = None
                ) -> tuple[torch.Tensor, list, list]:
        if past_ks is None:
            past_ks = [None] * self.config.num_hidden_layers
            past_vs = [None] * self.config.num_hidden_layers

        # Project target hidden context
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding

        new_ks, new_vs = [], []
        for layer, pk, pv in zip(self.layers, past_ks, past_vs):
            x, k, v = layer(x, x_ctx, self.inv_freq, cache_offset, pk, pv)
            new_ks.append(k)
            new_vs.append(v)

        x = self.norm(x)
        return x, new_ks, new_vs


# ---------------------------------------------------------------------------
# Weight loader (from HF safetensors DFlashDraftModel checkpoint)
# ---------------------------------------------------------------------------


def _hf_key_map(config: DFlashConfig) -> dict[str, str]:
    """Map HF z-lab DFlash checkpoint keys → our PyTorch model keys."""
    out = {
        "fc.weight": "fc.weight",
        "hidden_norm.weight": "hidden_norm.weight",
        "norm.weight": "norm.weight",
    }
    for i in range(config.num_hidden_layers):
        pfx_hf = f"layers.{i}"
        pfx = f"layers.{i}"
        out.update({
            f"{pfx_hf}.input_layernorm.weight": f"{pfx}.input_layernorm.weight",
            f"{pfx_hf}.post_attention_layernorm.weight": f"{pfx}.post_attention_layernorm.weight",
            f"{pfx_hf}.self_attn.q_proj.weight": f"{pfx}.self_attn.q_proj.weight",
            f"{pfx_hf}.self_attn.k_proj.weight": f"{pfx}.self_attn.k_proj.weight",
            f"{pfx_hf}.self_attn.v_proj.weight": f"{pfx}.self_attn.v_proj.weight",
            f"{pfx_hf}.self_attn.o_proj.weight": f"{pfx}.self_attn.o_proj.weight",
            f"{pfx_hf}.self_attn.q_norm.weight": f"{pfx}.self_attn.q_norm.weight",
            f"{pfx_hf}.self_attn.k_norm.weight": f"{pfx}.self_attn.k_norm.weight",
            f"{pfx_hf}.mlp.gate_proj.weight": f"{pfx}.mlp.gate_proj.weight",
            f"{pfx_hf}.mlp.up_proj.weight": f"{pfx}.mlp.up_proj.weight",
            f"{pfx_hf}.mlp.down_proj.weight": f"{pfx}.mlp.down_proj.weight",
        })
    return out


def load_dflash_from_hf(checkpoint_dir: str | Path) -> DFlashDraftModel:
    """Load a DFlashDraftModel from the HF z-lab checkpoint dir.

    Expected files:
        config.json
        model.safetensors (or sharded)
    """
    import safetensors.torch

    path = Path(checkpoint_dir)
    config = DFlashConfig.from_hf_json(path / "config.json")
    model = DFlashDraftModel(config)

    # Collect weights
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(safetensors.torch.load_file(str(f)))

    key_map = _hf_key_map(config)
    missing = []
    for hf_key, our_key in key_map.items():
        if hf_key not in weights:
            missing.append(hf_key)
    if missing:
        raise RuntimeError(f"Missing weights in checkpoint: {missing[:5]}...")

    sd = {our_key: weights[hf_key].to(torch.bfloat16) for hf_key, our_key in key_map.items()}
    model.load_state_dict(sd, strict=False)  # strict=False because inv_freq is buffer
    model = model.to(torch.bfloat16).eval()
    return model
