"""DFlash draft model, ANE-friendly variant — traceable for coremltools.convert.

Differences from dflash_torch.py:
1. Stateless — no KV cache across cycles. Each forward sees only (ctx, block),
   no accumulated past. Simplifies tracing; we'll validate that quality is
   preserved empirically in the benchmark.
2. RMSNorm has a mode switch (standard/ane). ANE path uses mean-subtract +
   F.layer_norm per ANEMLL CLAUDE.md convention. Standard path matches
   mlx-lm bit-for-bit for parity tests.
3. Static shapes: ctx_len and block_size are fixed per model instance.
4. No Optional tensors or None-checks inside forward — all inputs concrete.
5. Forward takes (noise_embedding, target_hidden) only. Returns hidden states
   (pre-LM-head). The caller applies target.lm_head externally.

This makes the model traceable by torch.jit.trace and convertable via
coremltools.convert to MLProgram. The trace path is exercised by the
companion conversion script.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash_torch import DFlashConfig  # reuse config dataclass


# ---------------------------------------------------------------------------
# RMSNorm mode switch
# ---------------------------------------------------------------------------

_RMSNORM_MODE = {"mode": "standard"}


def set_rmsnorm_mode(mode: str) -> None:
    """Flip between 'standard' (mlx-lm parity) and 'ane' (layer_norm form)."""
    assert mode in ("ane", "standard")
    _RMSNORM_MODE["mode"] = mode


class RMSNormSwitchable(nn.Module):
    """RMSNorm with standard / ANE forms. Same weights, different compute."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if _RMSNORM_MODE["mode"] == "standard":
            orig_dtype = x.dtype
            x_ = x.to(torch.float32)
            variance = x_.pow(2).mean(-1, keepdim=True)
            x_ = x_ * torch.rsqrt(variance + self.eps)
            return (self.weight * x_).to(orig_dtype)
        # ANE mode: mean-subtract + layer_norm
        mean = x.mean(-1, keepdim=True)
        x = x - mean
        return F.layer_norm(x, self.weight.shape, self.weight, bias=None,
                             eps=float(self.eps))


# ---------------------------------------------------------------------------
# Rotary embedding — precomputed cos/sin tables
# ---------------------------------------------------------------------------


def _rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def precompute_rope_table(head_dim: int, base: float, max_positions: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute (cos, sin) tables of shape (max_positions, head_dim)."""
    inv_freq = _rope_inv_freq(head_dim, base)
    positions = torch.arange(max_positions, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)   # (T, head_dim/2)
    emb = torch.cat([freqs, freqs], dim=-1)     # (T, head_dim)
    return emb.cos(), emb.sin()


def rotate_half_fixed(x: torch.Tensor, half: int) -> torch.Tensor:
    """rotate_half with an explicit half-dim constant so tracer doesn't emit dynamic int ops."""
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_fixed(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                      half: int) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (T, D) — broadcast over B, H; half = D // 2 (compile-time const)
    return x * cos + rotate_half_fixed(x, half) * sin


# ---------------------------------------------------------------------------
# Attention — stateless, fixed-shape
# ---------------------------------------------------------------------------


class DFlashAttentionANE(nn.Module):
    """Stateless DFlash attention with cross-stream K/V.

    Given:
      x:     (B, L, H)             current block (noise embeddings)
      x_ctx: (B, S, H)             projected target hidden context
      cos_q, sin_q: (L, D)         rope at positions [offset+S, offset+S+L)
      cos_k, sin_k: (S+L, D)       rope at positions [offset, offset+S+L)

    Produces: out (B, L, H). No K/V cache state returned — stateless.
    """
    def __init__(self, config: DFlashConfig, block_size: int, ctx_size: int):
        super().__init__()
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.half_dim = config.head_dim // 2
        self.scale = self.head_dim ** -0.5
        self.L = block_size
        self.S = ctx_size
        self.total_kv = ctx_size + block_size
        # GQA replication factor
        self.rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNormSwitchable(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormSwitchable(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor,
                cos_q: torch.Tensor, sin_q: torch.Tensor,
                cos_k: torch.Tensor, sin_k: torch.Tensor) -> torch.Tensor:
        B = 1
        L = self.L
        S = self.S
        T = self.total_kv

        c = torch.cat([x_ctx, x], dim=1)  # (B, S+L, H)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)     # (B, Hq, L, D)
        k = self.k_norm(k).transpose(1, 2)     # (B, Hkv, T, D)
        v = v.transpose(1, 2)                   # (B, Hkv, T, D)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # GQA expand via expand + reshape (ANE-friendlier than repeat_interleave)
        # (B, Hkv, T, D) -> unsqueeze(2) -> (B, Hkv, 1, T, D) -> expand -> (B, Hkv, rep, T, D) -> reshape -> (B, H, T, D)
        k = k.unsqueeze(2).expand(B, self.n_kv_heads, self.rep, T, self.head_dim)
        k = k.reshape(B, self.n_heads, T, self.head_dim)
        v = v.unsqueeze(2).expand(B, self.n_kv_heads, self.rep, T, self.head_dim)
        v = v.reshape(B, self.n_heads, T, self.head_dim)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None,
                                              is_causal=False, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(out)


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayerANE(nn.Module):
    def __init__(self, config: DFlashConfig, block_size: int, ctx_size: int):
        super().__init__()
        self.self_attn = DFlashAttentionANE(config, block_size, ctx_size)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor,
                cos_q: torch.Tensor, sin_q: torch.Tensor,
                cos_k: torch.Tensor, sin_k: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x), x_ctx, cos_q, sin_q, cos_k, sin_k)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DFlashDraftModelANE(nn.Module):
    """Stateless traceable DFlash draft. Takes rope tables as inputs so
    position offset is implicit in the tables (caller computes cos/sin for
    the right positions before calling).

    Shapes are fixed at construction:
        block_size: size of the speculated block (L, typically 16)
        ctx_size:   size of the target_hidden context (S, padded/truncated by caller)
    """
    def __init__(self, config: DFlashConfig, block_size: int | None = None,
                 ctx_size: int | None = None):
        super().__init__()
        self.config = config
        self.block_size = block_size if block_size is not None else config.block_size
        self.ctx_size = ctx_size if ctx_size is not None else config.block_size

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayerANE(config, self.block_size, self.ctx_size)
             for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor,
                cos_q: torch.Tensor, sin_q: torch.Tensor,
                cos_k: torch.Tensor, sin_k: torch.Tensor) -> torch.Tensor:
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding
        for layer in self.layers:
            x = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k)
        return self.norm(x)


def build_rope_for_offset(config: DFlashConfig, offset: int, ctx_len: int,
                          block_size: int) -> tuple[torch.Tensor, torch.Tensor,
                                                     torch.Tensor, torch.Tensor]:
    """Build the (cos_q, sin_q, cos_k, sin_k) RoPE tensors for a given offset.

    - queries (block tokens) at positions [offset+ctx_len, offset+ctx_len+block_size)
    - keys (ctx+block) at positions [offset, offset+ctx_len+block_size)
    """
    inv_freq = _rope_inv_freq(config.head_dim, config.rope_theta)

    q_positions = torch.arange(block_size, dtype=torch.float32) + (offset + ctx_len)
    k_positions = torch.arange(ctx_len + block_size, dtype=torch.float32) + offset

    def mktbl(positions):
        freqs = torch.outer(positions, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()

    cos_q, sin_q = mktbl(q_positions)
    cos_k, sin_k = mktbl(k_positions)
    return cos_q, sin_q, cos_k, sin_k


def copy_weights(src: nn.Module, dst: nn.Module) -> None:
    """Copy weights from dflash_torch DFlashDraftModel into DFlashDraftModelANE.

    The two models have the same structure (mostly). This loads by name.
    """
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    loaded = 0
    skipped = []
    for key in dst_sd.keys():
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
            loaded += 1
        else:
            skipped.append(key)
    dst.load_state_dict(dst_sd, strict=False)
    return loaded, skipped
