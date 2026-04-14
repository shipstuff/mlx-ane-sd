"""DFlash ANE model with persistent KV cache as CoreML state_tensor.

Based on dflash_ane.py but adds per-layer KV cache as register_buffer,
following ANEMLL's qwen_model.py pattern. Each forward:
 - Reads current_pos (tensor int32) — where to write
 - Writes fresh K/V at [current_pos, current_pos + S + L) with static size S+L=32
 - Attends to the full cache, with causal_mask suppressing invalid positions

Unified KV cache pattern: one big buffer of shape
(num_layers*2, num_kv_heads, STATE_LENGTH, head_dim).
  - layer i, K at index 2i
  - layer i, V at index 2i+1
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash_torch import DFlashConfig


# Module-level RMSNorm mode (shared with dflash_ane.py for parity tests)
_RMSNORM_MODE = {"mode": "ane"}


def set_rmsnorm_mode(mode: str) -> None:
    assert mode in ("ane", "standard")
    _RMSNORM_MODE["mode"] = mode


class RMSNormSwitchable(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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


def _rope_inv_freq(head_dim: int, base: float) -> torch.Tensor:
    return 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))


def rotate_half_fixed(x: torch.Tensor, half: int) -> torch.Tensor:
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_fixed(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     half: int) -> torch.Tensor:
    return x * cos + rotate_half_fixed(x, half) * sin


class DFlashAttentionANECache(nn.Module):
    """DFlash attention using external persistent KV cache buffer (written in-place).

    Cache layout (unified tensor passed in by the DFlashDraftModelANECache):
        shape (2 * num_layers, num_kv_heads, STATE_LENGTH, head_dim)
        cache[2*layer_idx]     = K
        cache[2*layer_idx + 1] = V
    """
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.total_new = ctx_size + block_size  # S+L, size of each write
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

    def forward(self, x: torch.Tensor, x_ctx: torch.Tensor,
                cos_q: torch.Tensor, sin_q: torch.Tensor,
                cos_k: torch.Tensor, sin_k: torch.Tensor,
                kv_cache: torch.Tensor, current_pos: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (1, L, H)          block embeddings
            x_ctx: (1, S, H)      projected target hidden context
            cos_q/sin_q: (L, D)   rope for queries (block positions)
            cos_k/sin_k: (S+L, D) rope for newly-computed K positions
            kv_cache: (2N, Hkv, STATE_LENGTH, D) — unified cache, in-place written
            current_pos: (,) int32 — where to write new K/V in the cache
            causal_mask: (1, 1, L, STATE_LENGTH) — additive mask for SDPA over cache
        """
        B = 1
        L = self.block_size
        S = self.ctx_size
        T = self.total_new

        c = torch.cat([x_ctx, x], dim=1)                         # (1, S+L, H)

        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)                       # (1, Hq, L, D)
        k = self.k_norm(k).transpose(1, 2)                       # (1, Hkv, T, D)
        v = v.transpose(1, 2)                                     # (1, Hkv, T, D)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # Write new K/V into cache via direct slice assignment — mirrors
        # anemll/qwen_model.py pattern: kv_cache[idx:idx+1, :, pos:pos+T, :] = k
        k_idx = 2 * self.layer_idx
        v_idx = 2 * self.layer_idx + 1
        kv_cache[k_idx : k_idx + 1, :, current_pos : current_pos + T, :] = k
        kv_cache[v_idx : v_idx + 1, :, current_pos : current_pos + T, :] = v

        # Read full cache for this layer
        k_full = kv_cache[k_idx : k_idx + 1]                     # (1, Hkv, STATE_LEN, D)
        v_full = kv_cache[v_idx : v_idx + 1]                     # (1, Hkv, STATE_LEN, D)

        # GQA expand
        k_full = k_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.state_length, self.head_dim)
        k_full = k_full.reshape(1, self.n_heads, self.state_length, self.head_dim)
        v_full = v_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.state_length, self.head_dim)
        v_full = v_full.reshape(1, self.n_heads, self.state_length, self.head_dim)

        # SDPA with causal_mask suppressing invalid positions
        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=causal_mask,
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


class DFlashDecoderLayerANECache(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.self_attn = DFlashAttentionANECache(config, layer_idx, block_size,
                                                   ctx_size, state_length)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                kv_cache, current_pos, causal_mask):
        x = x + self.self_attn(self.input_layernorm(x), x_ctx,
                                cos_q, sin_q, cos_k, sin_k,
                                kv_cache, current_pos, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DFlashDraftModelANECache(nn.Module):
    def __init__(self, config: DFlashConfig, block_size: int = 16,
                 ctx_size: int = 16, state_length: int = 256):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayerANECache(config, i, block_size, ctx_size, state_length)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

        # Unified KV cache buffer (stateful)
        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            state_length,
            config.head_dim,
        )
        self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=torch.float16))

    def forward(self, noise_embedding: torch.Tensor, target_hidden: torch.Tensor,
                cos_q: torch.Tensor, sin_q: torch.Tensor,
                cos_k: torch.Tensor, sin_k: torch.Tensor,
                current_pos: torch.Tensor,
                causal_mask: torch.Tensor) -> torch.Tensor:
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding
        for layer in self.layers:
            x = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                      self.kv_cache, current_pos, causal_mask)
        return self.norm(x)


def copy_weights_to_cache(src, dst: DFlashDraftModelANECache):
    """Copy weights from dflash_torch.DFlashDraftModel into cache-aware variant."""
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd.keys():
        if key == "kv_cache":  # skip state buffer
            continue
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst.load_state_dict(dst_sd, strict=False)
