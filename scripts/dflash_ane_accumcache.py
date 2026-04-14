"""DFlash ANE model with externalized accumulating cache.

Design: model is stateless w.r.t. the KV cache. Cache is a regular input,
the model uses cat(cache, new_K) for attention (static shapes = ANE-legal),
and returns (hidden, new_K, new_V) separately so Python manages the cache.

Python-side:
- For cycles fitting in [0, STATE_LEN): accumulate at write_pos, advance by T
- When write_pos + T > STATE_LEN: shift-left + append (same semantic as the
  external-sliding variant, but only kicks in after 8 cycles, so shorter
  generations keep FULL accumulating cache)

Key difference from dflash_ane_extcache.py:
- No internal shift-left inside the model
- Returns new K/V (shape (N, Hkv, T, D) per layer merged) so Python can place
  them wherever it wants
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash_torch import DFlashConfig


_RMSNORM_MODE = {"mode": "ane"}


def set_rmsnorm_mode(mode: str) -> None:
    assert mode in ("ane", "standard")
    _RMSNORM_MODE["mode"] = mode


class RMSNormSwitchable(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        if _RMSNORM_MODE["mode"] == "standard":
            orig = x.dtype
            x_ = x.to(torch.float32)
            var = x_.pow(2).mean(-1, keepdim=True)
            x_ = x_ * torch.rsqrt(var + self.eps)
            return (self.weight * x_).to(orig)
        mean = x.mean(-1, keepdim=True)
        x = x - mean
        return F.layer_norm(x, self.weight.shape, self.weight, bias=None,
                             eps=float(self.eps))


def rotate_half_fixed(x, half: int):
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope_fixed(x, cos, sin, half: int):
    return x * cos + rotate_half_fixed(x, half) * sin


class DFlashAttentionAccum(nn.Module):
    """Attention that takes cache_in, returns attn_out + new_K/V (NOT merged)."""
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.T = ctx_size + block_size
        self.attend_len = state_length + self.T  # Q attends to this many positions
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.half_dim = config.head_dim // 2
        self.scale = self.head_dim ** -0.5
        self.rep = self.n_heads // self.n_kv_heads

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNormSwitchable(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNormSwitchable(self.head_dim, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                cache_K, cache_V, causal_mask):
        """
        cache_K, cache_V: (1, Hkv, STATE_LEN, D)
        Returns: attn_out (1, L, H), new_K (1, Hkv, T, D), new_V same.
        """
        B = 1
        L = self.block_size
        T = self.T

        c = torch.cat([x_ctx, x], dim=1)   # (B, T, H)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)       # (1, Hq, L, D)
        k = self.k_norm(k).transpose(1, 2)       # (1, Hkv, T, D)
        v = v.transpose(1, 2)                     # (1, Hkv, T, D)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # Assemble full K/V for attention: cat(cache, new). Static shape.
        k_full = torch.cat([cache_K, k], dim=2)   # (1, Hkv, STATE_LEN + T, D)
        v_full = torch.cat([cache_V, v], dim=2)

        # GQA expand
        k_full = k_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.attend_len, self.head_dim)
        k_full = k_full.reshape(1, self.n_heads, self.attend_len, self.head_dim)
        v_full = v_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.attend_len, self.head_dim)
        v_full = v_full.reshape(1, self.n_heads, self.attend_len, self.head_dim)

        attn = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=causal_mask,
                                                is_causal=False, scale=self.scale)
        attn = attn.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(attn), k, v  # new K/V are returned BEFORE GQA expand


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayerAccum(nn.Module):
    def __init__(self, config, layer_idx, block_size, ctx_size, state_length):
        super().__init__()
        self.self_attn = DFlashAttentionAccum(config, layer_idx, block_size,
                                                ctx_size, state_length)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                cache_K, cache_V, causal_mask):
        attn_out, new_k, new_v = self.self_attn(
            self.input_layernorm(x), x_ctx, cos_q, sin_q, cos_k, sin_k,
            cache_K, cache_V, causal_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_k, new_v


class DFlashDraftModelAccum(nn.Module):
    """Accumulating-cache DFlash. Cache is input; new K/V returned separately.
    Python-side caller accumulates, optionally slides after STATE_LEN."""
    def __init__(self, config: DFlashConfig, block_size: int = 16,
                 ctx_size: int = 16, state_length: int = 256):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.T = ctx_size + block_size

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayerAccum(config, i, block_size, ctx_size, state_length)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, noise_embedding, target_hidden,
                cos_q, sin_q, cos_k, sin_k,
                cache_K, cache_V, causal_mask):
        """
        cache_K, cache_V: (N, Hkv, STATE_LEN, D) unified across layers.
        Returns: hidden (1, L, H), new_K (N, Hkv, T, D), new_V same.
        """
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding

        new_ks = []
        new_vs = []
        for i, layer in enumerate(self.layers):
            k_in = cache_K[i : i + 1]
            v_in = cache_V[i : i + 1]
            x, nk, nv = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                                k_in, v_in, causal_mask)
            new_ks.append(nk)
            new_vs.append(nv)

        new_K = torch.cat(new_ks, dim=0)   # (N, Hkv, T, D)
        new_V = torch.cat(new_vs, dim=0)
        return self.norm(x), new_K, new_V


def copy_weights_to_accum(src, dst: DFlashDraftModelAccum):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd.keys():
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst.load_state_dict(dst_sd, strict=False)
