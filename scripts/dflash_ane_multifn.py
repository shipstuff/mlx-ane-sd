"""DFlash ANE model for multi-function compilation.

The multi-function trick: each compiled variant bakes in a specific write_pos
as a Python integer constant known at trace time. Because write_pos is
constant per variant, all index expressions that depend on it are static
slices -- ANE-legal and cheap.

The architectural win: attention scope = write_pos + T (the VALID prefix of
the cache), not STATE_LEN. Early cycles attend over just 32 positions; late
cycles over 1024. Same weights across variants, amortized over the whole
generation.

Python still owns the cache; it writes the returned T-sized new K/V into
cache[:, :, write_pos:write_pos+T, :] (a static Python slice per call).
There is no sliding as long as we have a compiled variant for the current
write_pos. When we run out (write_pos would exceed STATE_LEN - T), Python
falls back to a `rotate` variant that shifts-left inside the model.

API:
- DFlashDraftModelMultiFn(config, block_size, ctx_size, state_length,
                          write_pos, rotate)
  -- write_pos: compile-time constant int
  -- rotate: if True, attention sees the shift-left'd cache with new K/V
     appended at tail (write_pos ignored).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dflash_torch import DFlashConfig
from dflash_ane_accumcache import (
    RMSNormSwitchable, apply_rope_fixed, set_rmsnorm_mode, DFlashMLP,
)


class DFlashAttentionMultiFn(nn.Module):
    """Attention with compile-time constant write_pos.

    Cache flow:
    - cache_K_in/cache_V_in shape (1, Hkv, STATE_LEN, D) with content already
      placed by Python: positions [0, write_pos) are the committed history.
    - This layer computes new K/V (T positions, fresh this cycle) and attends
      over the concatenated valid window (write_pos + T for normal, STATE_LEN
      for rotate).
    - Returns the T new K/V (pre-GQA-expand) so Python can splice them into
      its cache buffer.
    """
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
                cache_K_in, cache_V_in, causal_mask,
                write_pos: int, rotate: bool):
        B = 1
        L = self.block_size
        T = self.T
        S = self.state_length

        c = torch.cat([x_ctx, x], dim=1)   # (B, T, H)
        q = self.q_proj(x).view(B, L, self.n_heads, self.head_dim)
        k = self.k_proj(c).view(B, T, self.n_kv_heads, self.head_dim)
        v = self.v_proj(c).view(B, T, self.n_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)       # (1, Hq, L, D)
        k = self.k_norm(k).transpose(1, 2)       # (1, Hkv, T, D)
        v = v.transpose(1, 2)                     # (1, Hkv, T, D)

        q = apply_rope_fixed(q, cos_q.to(q.dtype), sin_q.to(q.dtype), self.half_dim)
        k = apply_rope_fixed(k, cos_k.to(k.dtype), sin_k.to(k.dtype), self.half_dim)

        # Build the attention K/V view. Static slices because write_pos is a
        # Python int constant.
        if rotate:
            # Treat cache as full valid window. Shift-left by T, append new at tail.
            k_tail = torch.narrow(cache_K_in, 2, T, S - T)   # (1, Hkv, S-T, D)
            v_tail = torch.narrow(cache_V_in, 2, T, S - T)
            attend_K = torch.cat([k_tail, k], dim=2)        # (1, Hkv, S, D)
            attend_V = torch.cat([v_tail, v], dim=2)
            attend_len = S
        else:
            wp = write_pos
            if wp == 0:
                attend_K = k
                attend_V = v
            else:
                pre_K = torch.narrow(cache_K_in, 2, 0, wp)
                pre_V = torch.narrow(cache_V_in, 2, 0, wp)
                attend_K = torch.cat([pre_K, k], dim=2)      # (1, Hkv, wp+T, D)
                attend_V = torch.cat([pre_V, v], dim=2)
            attend_len = wp + T

        # GQA expand to n_heads
        attend_K = attend_K.unsqueeze(2).expand(
            1, self.n_kv_heads, self.rep, attend_len, self.head_dim)
        attend_K = attend_K.reshape(1, self.n_heads, attend_len, self.head_dim)
        attend_V = attend_V.unsqueeze(2).expand(
            1, self.n_kv_heads, self.rep, attend_len, self.head_dim)
        attend_V = attend_V.reshape(1, self.n_heads, attend_len, self.head_dim)

        attn = F.scaled_dot_product_attention(
            q, attend_K, attend_V, attn_mask=causal_mask,
            is_causal=False, scale=self.scale,
        )
        attn = attn.transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.o_proj(attn), k, v  # return fresh K/V (T positions, pre-GQA)


class DFlashDecoderLayerMultiFn(nn.Module):
    def __init__(self, config, layer_idx, block_size, ctx_size, state_length):
        super().__init__()
        self.self_attn = DFlashAttentionMultiFn(config, layer_idx, block_size,
                                                 ctx_size, state_length)
        self.mlp = DFlashMLP(config.hidden_size, config.intermediate_size)
        self.input_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                cache_K, cache_V, causal_mask, write_pos: int, rotate: bool):
        attn_out, new_k, new_v = self.self_attn(
            self.input_layernorm(x), x_ctx, cos_q, sin_q, cos_k, sin_k,
            cache_K, cache_V, causal_mask, write_pos, rotate,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x, new_k, new_v


class DFlashDraftModelMultiFn(nn.Module):
    """Multi-function variant: write_pos (and rotate flag) are baked at trace time.

    Python picks the variant each cycle; cache_K/cache_V are always passed in
    shape (N, Hkv, STATE_LEN, D), and new K/V come back shape (N, Hkv, T, D).
    Python places the new K/V at cache[:, :, write_pos:write_pos+T, :] and
    advances write_pos by the committed count. When write_pos + T > STATE_LEN,
    Python shifts cache left by T and uses the `rotate` variant.
    """
    def __init__(self, config: DFlashConfig, block_size: int,
                 ctx_size: int, state_length: int,
                 write_pos: int, rotate: bool):
        super().__init__()
        self.config = config
        self.block_size = block_size
        self.ctx_size = ctx_size
        self.state_length = state_length
        self.write_pos = int(write_pos)
        self.rotate = bool(rotate)
        self.T = ctx_size + block_size

        if not self.rotate:
            assert 0 <= self.write_pos <= state_length - self.T, (
                f"write_pos={self.write_pos} incompatible with STATE_LEN={state_length}, T={self.T}")

        concat_dim = len(config.target_layer_ids) * config.hidden_size
        self.fc = nn.Linear(concat_dim, config.hidden_size, bias=False)
        self.hidden_norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)
        self.layers = nn.ModuleList(
            [DFlashDecoderLayerMultiFn(config, i, block_size, ctx_size, state_length)
             for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNormSwitchable(config.hidden_size, eps=config.rms_norm_eps)

    @property
    def attend_len(self) -> int:
        if self.rotate:
            return self.state_length
        return self.write_pos + self.T

    def forward(self, noise_embedding, target_hidden,
                cos_q, sin_q, cos_k, sin_k,
                cache_K, cache_V, causal_mask):
        """
        cache_K, cache_V: (N, Hkv, STATE_LEN, D) unified across layers.
        causal_mask: (1, 1, BS, attend_len) where attend_len depends on variant.
        Returns: hidden (1, L, H), new_K (N, Hkv, T, D), new_V (same).
        """
        x_ctx = self.hidden_norm(self.fc(target_hidden))
        x = noise_embedding

        new_ks = []
        new_vs = []
        for i, layer in enumerate(self.layers):
            k_in = cache_K[i : i + 1]
            v_in = cache_V[i : i + 1]
            x, nk, nv = layer(x, x_ctx, cos_q, sin_q, cos_k, sin_k,
                                k_in, v_in, causal_mask,
                                self.write_pos, self.rotate)
            new_ks.append(nk)
            new_vs.append(nv)

        new_K = torch.cat(new_ks, dim=0)   # (N, Hkv, T, D)
        new_V = torch.cat(new_vs, dim=0)
        return self.norm(x), new_K, new_V


def copy_weights_to_multifn(src, dst: DFlashDraftModelMultiFn):
    src_sd = src.state_dict()
    dst_sd = dst.state_dict()
    for key in dst_sd.keys():
        if key in src_sd and src_sd[key].shape == dst_sd[key].shape:
            dst_sd[key] = src_sd[key]
    dst.load_state_dict(dst_sd, strict=False)
