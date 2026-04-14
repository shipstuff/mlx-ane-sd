"""DFlash ANE model with EXTERNAL KV cache — cache passed as input/output.

No state_tensor. The caller maintains the cache tensor in Python/numpy and
passes it fresh each call. The model reads the input cache, does
sliding-window update internally, and returns the updated cache.

Downside: 5 MB cache copied through the coremltools bridge each call.
Expected cost: ~1-2 ms per call. Tractable.

Upside: no state_tensor read/write ops → should lower cleanly to ANE.
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


def rotate_half_fixed(x, half: int):
    return torch.cat([-x[..., half:], x[..., :half]], dim=-1)


def apply_rope_fixed(x, cos, sin, half: int):
    return x * cos + rotate_half_fixed(x, half) * sin


class DFlashAttentionExt(nn.Module):
    """Attention that takes cache_in, returns cache_out."""
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.T = ctx_size + block_size
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
                k_cache_in, v_cache_in, causal_mask):
        """
        k_cache_in, v_cache_in: (1, Hkv, STATE_LEN, D) each
        Returns: out, k_cache_out, v_cache_out
        """
        B = 1
        L = self.block_size
        T = self.T

        c = torch.cat([x_ctx, x], dim=1)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # Sliding-window update: shift + append, all static
        k_head = k_cache_in[:, :, T:, :]              # (1, Hkv, STATE_LEN-T, D)
        k_out = torch.cat([k_head, k], dim=2)          # (1, Hkv, STATE_LEN, D)
        v_head = v_cache_in[:, :, T:, :]
        v_out = torch.cat([v_head, v], dim=2)

        # GQA expand for attention
        k_full = k_out.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                             self.state_length, self.head_dim)
        k_full = k_full.reshape(1, self.n_heads, self.state_length, self.head_dim)
        v_full = v_out.unsqueeze(2).expand(1, self.n_kv_heads, self.rep,
                                             self.state_length, self.head_dim)
        v_full = v_full.reshape(1, self.n_heads, self.state_length, self.head_dim)

        out = F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=causal_mask,
                                              is_causal=False, scale=self.scale)
        out = out.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(out), k_out, v_out


class DFlashMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DFlashDecoderLayerExt(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int,
                 block_size: int, ctx_size: int, state_length: int):
        super().__init__()
        self.self_attn = DFlashAttentionExt(config, layer_idx, block_size,
                                              ctx_size, state_length)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                k_cache_in, v_cache_in, causal_mask):
        attn_out, k_out, v_out = self.self_attn(
            self.input_layernorm(x), x_ctx, cos_q, sin_q, cos_k, sin_k,
            k_cache_in, v_cache_in, causal_mask,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, k_out, v_out


class DFlashDraftModelExt(nn.Module):
    """External-cache DFlash. Cache is input + output, not state_tensor."""
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
            [DFlashDecoderLayerExt(config, i, block_size, ctx_size, state_length)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, noise_embedding, target_hidden,
                cos_q, sin_q, cos_k, sin_k,
                k_cache_in, v_cache_in, causal_mask):
        """
        k_cache_in, v_cache_in: (2N, Hkv, STATE_LEN, D) unified across layers.
                                cache[2i] = layer i's K, cache[2i+1] = layer i's V
        Returns: hidden (1, L, H), k_cache_out, v_cache_out
        """
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding

        # Split unified cache into per-layer inputs, process, re-stack
        k_per_layer = []
        v_per_layer = []
        for i, layer in enumerate(self.layers):
            k_in = k_cache_in[2 * i : 2 * i + 1]       # (1, Hkv, STATE_LEN, D)
            v_in = v_cache_in[2 * i : 2 * i + 1]       # note: v stored at odd slots in a unified view
            # Actually our caller passes separate k_cache_in and v_cache_in of shape (N, Hkv, STATE_LEN, D)
            # Let me fix the interface — use per-layer indexing instead
            # (rewrite below)
            pass

        # Simpler: accept separate K and V, each (N, Hkv, STATE_LEN, D)
        k_layers_out = []
        v_layers_out = []
        for i, layer in enumerate(self.layers):
            k_in = k_cache_in[i : i + 1]   # (1, Hkv, STATE_LEN, D)
            v_in = v_cache_in[i : i + 1]
            x, k_out, v_out = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                                      k_in, v_in, causal_mask)
            k_layers_out.append(k_out)
            v_layers_out.append(v_out)

        k_cache_out = torch.cat(k_layers_out, dim=0)   # (N, Hkv, STATE_LEN, D)
        v_cache_out = torch.cat(v_layers_out, dim=0)
        return self.norm(x), k_cache_out, v_cache_out


def copy_weights_to_ext(src, dst: DFlashDraftModelExt):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd.keys():
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst.load_state_dict(dst_sd, strict=False)
