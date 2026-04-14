"""DFlash ANE model with sliding-window KV cache — all slice bounds static.

Goal: ANE placement. The dynamic-offset state_tensor variant (dflash_ane_cache.py)
compiles but ANE compiler rejects it because slice_update with dynamic bounds
fails ANECCompile.

Sliding window pattern:
- Each forward: cache.shift_left_by(T=S+L=32), append fresh K/V at end.
- All slice bounds are compile-time constants → ANE-lowerable.
- Cache stores last STATE_LENGTH positions worth of K/V regardless of which
  cycle they came from.

RoPE positions are relative to a running "cycle counter" that the caller passes
in via the cos/sin tables — the model doesn't know absolute sequence positions.
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


def rotate_half_fixed(x: torch.Tensor, half: int) -> torch.Tensor:
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope_fixed(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                     half: int) -> torch.Tensor:
    return x * cos + rotate_half_fixed(x, half) * sin


class DFlashAttentionSlide(nn.Module):
    """Sliding-window KV attention. All cache writes are static slices."""
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.T = ctx_size + block_size  # window slide amount
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
                kv_cache, causal_mask):
        B = 1
        L = self.block_size
        T = self.T

        c = torch.cat([x_ctx, x], dim=1)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)                     # (1, Hkv, T, D)
        v = v.transpose(1, 2)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # Sliding-window update: shift left by T, write new at tail.
        # All static slices.
        k_idx = 2 * self.layer_idx
        v_idx = 2 * self.layer_idx + 1

        # Shift-left: new[0..STATE_LEN-T] = old[T..STATE_LEN]
        # Append: new[STATE_LEN-T..STATE_LEN] = k
        # Using torch.cat with static slices
        k_head = kv_cache[k_idx : k_idx + 1, :, T:, :]         # (1, Hkv, STATE_LEN-T, D)
        k_new = k.squeeze(0).unsqueeze(0)                      # (1, Hkv, T, D)
        k_stream = torch.cat([k_head, k_new], dim=2)           # (1, Hkv, STATE_LEN, D)

        v_head = kv_cache[v_idx : v_idx + 1, :, T:, :]
        v_new = v.squeeze(0).unsqueeze(0)
        v_stream = torch.cat([v_head, v_new], dim=2)

        # In-place write to state tensor
        kv_cache[k_idx : k_idx + 1] = k_stream
        kv_cache[v_idx : v_idx + 1] = v_stream

        # Read full cache for this layer (for attention)
        k_full = k_stream                                       # (1, Hkv, STATE_LEN, D)
        v_full = v_stream

        # GQA expand
        k_full = k_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.state_length, self.head_dim)
        k_full = k_full.reshape(1, self.n_heads, self.state_length, self.head_dim)
        v_full = v_full.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                              self.state_length, self.head_dim)
        v_full = v_full.reshape(1, self.n_heads, self.state_length, self.head_dim)

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

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayerSlide(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.self_attn = DFlashAttentionSlide(config, layer_idx, block_size,
                                                ctx_size, state_length)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                kv_cache, causal_mask):
        x = x + self.self_attn(self.input_layernorm(x), x_ctx,
                                cos_q, sin_q, cos_k, sin_k,
                                kv_cache, causal_mask)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class DFlashDraftModelSlide(nn.Module):
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
            [DFlashDecoderLayerSlide(config, i, block_size, ctx_size, state_length)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

        cache_shape = (
            2 * config.num_hidden_layers,
            config.num_key_value_heads,
            state_length,
            config.head_dim,
        )
        self.register_buffer("kv_cache", torch.zeros(cache_shape, dtype=torch.float16))

    def forward(self, noise_embedding, target_hidden,
                cos_q, sin_q, cos_k, sin_k, causal_mask):
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding
        for layer in self.layers:
            x = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                      self.kv_cache, causal_mask)
        return self.norm(x)


def copy_weights_to_slide(src, dst: DFlashDraftModelSlide):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd.keys():
        if key == "kv_cache":
            continue
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst.load_state_dict(dst_sd, strict=False)
