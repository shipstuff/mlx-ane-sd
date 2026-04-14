"""Phase F.1 accumulating-cache: DFlash SD with Python-managed persistent cache.

The ANE model is stateless w.r.t. cache:
  inputs:  noise_emb, target_hidden, rope tables, cache_K, cache_V, causal_mask
  outputs: hidden, new_K (one cycle's worth), new_V

Python maintains the cache across cycles:
- Cycles 0..7 (write_pos 0..224): accumulate by writing new K/V at [write_pos, write_pos+T)
- Cycle 8+ (would overflow): shift-left + append (sliding window from here on)

This keeps full acceptance semantics up to STATE_LENGTH / T cycles = 8 cycles for
STATE_LEN=256. At T=32, that's 256 absolute positions. For 100-token generation
with ~12-25 cycles (F.0's range), cycles 8+ slide.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import coremltools as ct

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from dflash.model_mlx import _patch_model
from dflash_ane import _rope_inv_freq
from dflash_torch import DFlashConfig


STATE_LENGTH = 256


class DFlashANEAccumDraft:
    def __init__(self, mlmodelc_path: str, config: DFlashConfig):
        print(f"[ane-draft] loading {mlmodelc_path}...")
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(str(mlmodelc_path),
                                                 ct.ComputeUnit.CPU_AND_NE)
        print(f"[ane-draft] loaded in {time.perf_counter()-t0:.1f}s")
        self.config = config
        self.ctx_size = config.block_size
        self.block_size = config.block_size
        self.head_dim = config.head_dim
        self.state_length = STATE_LENGTH
        self.T = self.ctx_size + self.block_size   # 32
        self.attend_len = self.state_length + self.T  # 288
        self.N = config.num_hidden_layers
        self.Hkv = config.num_key_value_heads
        self.inv_freq = _rope_inv_freq(self.head_dim, config.rope_theta).numpy()

        self.cache_shape = (self.N, self.Hkv, self.state_length, self.head_dim)
        self.cache_K = np.zeros(self.cache_shape, dtype=np.float16)
        self.cache_V = np.zeros(self.cache_shape, dtype=np.float16)
        # write_pos: how many valid positions are in cache, starting from index 0
        # until cache fills; after that, sliding.
        self.write_pos = 0
        self.global_offset = 0  # rope offset counter, never resets
        self.t_predict_total = 0.0
        self.n_predicts = 0

    def reset_cache(self):
        self.cache_K.fill(0)
        self.cache_V.fill(0)
        self.write_pos = 0
        self.global_offset = 0

    def _build_rope(self, s_real: int) -> dict:
        """Rope positions for this cycle. global_offset is the cache-relative base."""
        go = self.global_offset
        k_real = np.arange(s_real, dtype=np.float32) + go
        k_pad = np.zeros(self.ctx_size - s_real, dtype=np.float32)
        k_block = (np.arange(self.block_size, dtype=np.float32) + go + s_real)
        k_positions = np.concatenate([k_real, k_pad, k_block])
        q_positions = np.arange(self.block_size, dtype=np.float32) + go + s_real

        def mktbl(positions):
            freqs = np.outer(positions, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)
        cos_q, sin_q = mktbl(q_positions)
        cos_k, sin_k = mktbl(k_positions)
        return {"cos_q": cos_q, "sin_q": sin_q, "cos_k": cos_k, "sin_k": sin_k}

    def _build_causal_mask(self):
        """Mask: -inf for positions [0, STATE_LEN - write_pos) + for positions
        [STATE_LEN + T, STATE_LEN + T) (none). Valid: [STATE_LEN - write_pos, STATE_LEN + T).

        When write_pos < STATE_LEN: cache positions [0, write_pos) are valid;
            sliding mode: cache positions [0, STATE_LEN) all valid (cache is full).
        new K/V positions always valid.
        """
        L = self.block_size
        M = self.attend_len   # STATE_LEN + T
        mask = np.zeros((1, 1, L, M), dtype=np.float16)
        # During accumulating phase, valid cache = [0, write_pos)
        # During sliding phase (cache full), valid cache = [0, STATE_LEN)
        if self.write_pos < self.state_length:
            invalid_cache_end = self.state_length - self.write_pos
            mask[0, 0, :, :invalid_cache_end] = -np.inf
        # new K/V positions (STATE_LEN..STATE_LEN+T) always valid — no mask needed
        # Wait — I need to flip cache layout: valid cache is at TAIL (right) not head (left)?
        # Actually my Python writes cache[0..write_pos), so left-aligned.
        # So invalid = [write_pos, STATE_LEN). Correct below:
        mask[:] = 0  # reset
        if self.write_pos < self.state_length:
            mask[0, 0, :, self.write_pos:self.state_length] = -np.inf
        return mask

    def forward(self, noise_emb: np.ndarray, target_hidden: np.ndarray, s_real: int):
        """Runs the draft forward. Returns hidden, new_K, new_V.
        Caller must call commit(accepted, s_real) AFTER target verify to
        update the cache with only committed positions."""
        causal_mask = self._build_causal_mask()
        inputs = {
            "noise_embedding": noise_emb,
            "target_hidden": target_hidden,
            "cache_K": self.cache_K,
            "cache_V": self.cache_V,
            "causal_mask": causal_mask,
            **self._build_rope(s_real),
        }
        t0 = time.perf_counter()
        out = self.model.predict(inputs)
        self.t_predict_total += time.perf_counter() - t0
        self.n_predicts += 1
        return out["hidden"], out["new_K"], out["new_V"]

    def commit(self, new_K: np.ndarray, new_V: np.ndarray, s_real: int, accepted: int):
        """Write the COMMITTED K/V to cache (matches MLX trim semantics).

        new_K/new_V are the full T=S+L worth of fresh entries. Of those, only
        the first s_real + accepted + 1 positions are committed (rest were
        rejected draft predictions). Next cycle's write starts where this
        cycle's committed range ends.
        """
        committed = s_real + accepted + 1

        if self.write_pos + self.T <= self.state_length:
            # Accumulating: write the full T entries, but only advance by `committed`
            # so rejected positions get overwritten next cycle.
            self.cache_K[:, :, self.write_pos:self.write_pos + self.T, :] = new_K
            self.cache_V[:, :, self.write_pos:self.write_pos + self.T, :] = new_V
            self.write_pos += committed
        else:
            # Sliding phase: shift-left by `committed`, append committed-worth at tail
            # Here we do a simpler-but-correct variant: shift by T, append T worth.
            # This is the sliding-window approximation (same as ext_cache variant).
            self.cache_K = np.concatenate([self.cache_K[:, :, self.T:, :], new_K], axis=2)
            self.cache_V = np.concatenate([self.cache_V[:, :, self.T:, :], new_V], axis=2)

        # Rope offset advances by committed positions only
        self.global_offset += committed


def pad_or_truncate_ctx(hidden_mx, ctx_size):
    S = hidden_mx.shape[1]
    if S == ctx_size: return hidden_mx
    if S > ctx_size:  return hidden_mx[:, -ctx_size:, :]
    pad = mx.zeros((1, ctx_size - S, hidden_mx.shape[2]), dtype=hidden_mx.dtype)
    return mx.concatenate([hidden_mx, pad], axis=1)


def stream_generate_ane_accum(target, draft: DFlashANEAccumDraft, tok, prompt,
                                max_tokens, sampler=None):
    sampler = sampler or make_sampler(temp=0.0)
    if not isinstance(tok, TokenizerWrapper):
        tok = TokenizerWrapper(tok)
    config = draft.config
    _patch_model(target, config.target_layer_ids)

    add_special = tok.bos_token is None or not prompt.startswith(tok.bos_token)
    prompt_ids = tok.encode(prompt, add_special_tokens=add_special)
    prompt_arr = mx.array(prompt_ids)
    tokens = list(prompt_ids)

    target_cache = make_prompt_cache(target)
    t_prefill = time.perf_counter()
    with mx.stream(mx.default_stream(mx.default_device())):
        logits = target(prompt_arr[None], target_cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
    mx.eval(logits, hidden)
    token = int(mx.argmax(logits[:, -1:], axis=-1).item())
    tokens.append(token)
    n = 1
    t_prefill = time.perf_counter() - t_prefill

    draft.reset_cache()
    bs = config.block_size
    mask_id = config.mask_token_id
    accepted_total = 0
    cycles = 0

    t_decode_start = time.perf_counter()
    while n < max_tokens and not (tok.eos_token_ids and token in tok.eos_token_ids):
        block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])
        noise_emb_mx = target.model.embed_tokens(block)
        mx.eval(noise_emb_mx)
        ctx_mx = pad_or_truncate_ctx(hidden, bs)

        s_real = min(hidden.shape[1], bs)
        noise_np = np.asarray(noise_emb_mx.astype(mx.float32)).astype(np.float16)
        ctx_np = np.asarray(ctx_mx.astype(mx.float32)).astype(np.float16)

        hidden_out_np, new_K, new_V = draft.forward(noise_np, ctx_np, s_real)

        hidden_out_mx = mx.array(hidden_out_np.astype(np.float32)).astype(mx.bfloat16)
        if hasattr(target, "lm_head"):
            draft_logits = target.lm_head(hidden_out_mx)
        else:
            draft_logits = target.model.embed_tokens.as_linear(hidden_out_mx)

        draft_tokens_mx = sampler(draft_logits[:, 1 - bs:])
        mx.eval(draft_tokens_mx)

        verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens_mx], axis=1)
        with mx.stream(mx.default_stream(mx.default_device())):
            logits = target(verify_input, target_cache)
            hidden = mx.concatenate(target._hidden_states, axis=-1)
            target_tokens_mx = sampler(logits)
        mx.eval(target_tokens_mx, hidden)

        d_list = draft_tokens_mx[0].tolist()
        t_list = target_tokens_mx[0].tolist()
        accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                         len(d_list))
        new_tokens = d_list[:accepted] + [t_list[accepted]]
        new_tokens = new_tokens[: max_tokens - n]

        eos_ids = set(tok.eos_token_ids) if tok.eos_token_ids else set()
        if eos_ids:
            for i, t in enumerate(new_tokens):
                if t in eos_ids:
                    new_tokens = new_tokens[: i + 1]
                    break

        tokens.extend(new_tokens)
        n += len(new_tokens)
        accepted_total += accepted + 1
        cycles += 1

        # Commit to draft cache — uses accepted count to advance by committed only
        draft.commit(new_K, new_V, s_real, accepted)

        trim = bs - accepted - 1
        if trim > 0:
            from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
            if can_trim_prompt_cache(target_cache):
                trim_prompt_cache(target_cache, trim)

        hidden = hidden[:, :accepted + 1, :]
        if new_tokens and new_tokens[-1] in eos_ids:
            break
        token = tokens[-1]

    t_decode = time.perf_counter() - t_decode_start
    generated = tokens[len(prompt_ids):]
    return generated, t_decode, accepted_total, cycles, t_prefill


PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("math", "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:"),
    ("story", "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlmodelc",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc")
    ap.add_argument("--max-new", type=int, default=100)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print("[load] draft...")
    draft = DFlashANEAccumDraft(args.mlmodelc, config)

    print("[warmup]...")
    list(stream_generate_ane_accum(target, draft, tok, "The weather", 20))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        gen, t, accepted, cycles, _ = stream_generate_ane_accum(
            target, draft, tok, prompt, args.max_new)
        tps = len(gen) / t
        text = tok.decode(gen)
        print(f"  {len(gen)} tok in {t:.2f}s = {tps:.2f} tok/s "
              f"(cycles={cycles}, accepted={accepted}, avg/cycle={accepted/cycles:.2f})")
        print(f"  text: {text[:90]!r}...")
        rows.append({"name": name, "tps": tps, "cycles": cycles, "accepted": accepted})

    print("\n=== Summary ===")
    for r in rows:
        print(f"{r['name']:<12} {r['tps']:>7.2f}  avg/cycle={r['accepted']/r['cycles']:.2f}")
    tpss = [r['tps'] for r in rows]
    print(f"\nmean: {statistics.mean(tpss):.2f} tok/s")
    print(f"predict: {draft.t_predict_total / draft.n_predicts * 1000:.2f}ms "
          f"({draft.n_predicts} calls)")


if __name__ == "__main__":
    main()
