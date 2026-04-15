"""Phase F.1 multi-function: DFlash SD with compile-time-constant write_pos.

One compiled .mlpackage contains N+1 functions: `write_0`, `write_T`, ...,
`write_((N-1)*T)` baked as fixed write positions, plus a `rotate` function
for the sliding fallback. Python picks `function_name = f"write_{pos}"` each
cycle based on current write_pos; if write_pos + T > STATE_LEN, it shifts
cache left by T and uses `rotate` (which matches the accumcache sliding path).

Differences from `phaseF1_ane_stream_accum.py`:
- Loads one CompiledMLModel per variant (cheap: mlpackage is shared, only
  function selector differs).
- Per-call input `causal_mask` shape is `(1, 1, BS, attend_len)` where
  attend_len = write_pos + T for normal variants, STATE_LEN for rotate.
- No internal shift-by-T on every cycle in sliding mode: each rotate call
  returns fresh K/V pre-GQA, Python still does a small write, but after a
  one-time cache shift.
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


class DFlashANEMultiFnDraft:
    def __init__(self, mlpackage_path: str, config: DFlashConfig,
                 state_length: int = 1024, num_variants: int = 32):
        self.mlpackage_path = str(mlpackage_path)
        self.config = config
        self.ctx_size = config.block_size
        self.block_size = config.block_size
        self.head_dim = config.head_dim
        self.state_length = state_length
        self.T = self.ctx_size + self.block_size
        self.N = config.num_hidden_layers
        self.Hkv = config.num_key_value_heads
        self.inv_freq = _rope_inv_freq(self.head_dim, config.rope_theta).numpy()
        self.num_variants = num_variants

        self.cache_shape = (self.N, self.Hkv, self.state_length, self.head_dim)
        self.cache_K = np.zeros(self.cache_shape, dtype=np.float16)
        self.cache_V = np.zeros(self.cache_shape, dtype=np.float16)
        self.write_pos = 0
        self.global_offset = 0

        # Pre-load all variant handles. Loading each is ~5s (first time) because
        # CoreML fetches/compiles the function.
        print(f"[multifn] loading {num_variants} write_* variants + rotate from {mlpackage_path}")
        t0 = time.perf_counter()
        self.variants = {}
        for i in range(num_variants):
            wp = i * self.T
            if wp + self.T > state_length:
                break
            name = f"write_{wp}"
            self.variants[name] = ct.models.CompiledMLModel(
                self.mlpackage_path, ct.ComputeUnit.CPU_AND_NE, function_name=name)
        # rotate variant (attend_len = STATE_LEN)
        self.variants["rotate"] = ct.models.CompiledMLModel(
            self.mlpackage_path, ct.ComputeUnit.CPU_AND_NE, function_name="rotate")
        print(f"[multifn] loaded {len(self.variants)} variants in {time.perf_counter()-t0:.1f}s")

        self.t_predict_total = 0.0
        self.n_predicts = 0
        self.n_rotate_calls = 0
        self.n_write_calls = 0

    def reset_cache(self):
        self.cache_K.fill(0)
        self.cache_V.fill(0)
        self.write_pos = 0
        self.global_offset = 0

    def _build_rope(self, s_real: int) -> dict:
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

    def _pick_variant(self):
        """Choose which compiled variant to call this cycle."""
        if self.write_pos + self.T <= self.state_length:
            vname = f"write_{self.write_pos}"
            if vname in self.variants:
                return vname, False
        # Fallback: sliding / rotate. Caller will shift cache left by T before
        # passing to the model.
        return "rotate", True

    def forward(self, noise_emb: np.ndarray, target_hidden: np.ndarray, s_real: int):
        """Runs the draft forward. Returns hidden, new_K, new_V.
        Caller must call commit(...) AFTER target verify to advance write_pos."""
        vname, is_rotate = self._pick_variant()
        attend_len = self.state_length if is_rotate else (self.write_pos + self.T)

        # For rotate: don't pre-shift Python-side. Model does the shift via
        # narrow(cache, 2, T, S-T) + cat(new_k). We commit the shifted cache
        # after the call.

        mask = np.zeros((1, 1, self.block_size, attend_len), dtype=np.float16)

        inputs = {
            "noise_embedding": noise_emb,
            "target_hidden": target_hidden,
            "cache_K": self.cache_K,
            "cache_V": self.cache_V,
            "causal_mask": mask,
            **self._build_rope(s_real),
        }
        t0 = time.perf_counter()
        out = self.variants[vname].predict(inputs)
        self.t_predict_total += time.perf_counter() - t0
        self.n_predicts += 1
        if is_rotate:
            self.n_rotate_calls += 1
        else:
            self.n_write_calls += 1
        return out["hidden"], out["new_K"], out["new_V"]

    def commit(self, new_K: np.ndarray, new_V: np.ndarray, s_real: int, accepted: int):
        """Splice the new K/V into Python cache and advance write_pos.

        Multi-function variants are at T-aligned positions only. We advance
        write_pos AND global_offset by T each cycle so RoPE positions align
        with cache slots. This inflates "absolute positions" in RoPE space
        (they outrun real token count) but the relative structure within the
        draft's attention is preserved -- RoPE shift is constant across Q/K.
        """
        vname, is_rotate = self._pick_variant()
        if is_rotate:
            self.cache_K = np.concatenate([self.cache_K[:, :, self.T:, :], new_K], axis=2)
            self.cache_V = np.concatenate([self.cache_V[:, :, self.T:, :], new_V], axis=2)
            self.write_pos = self.state_length - self.T
        else:
            wp = self.write_pos
            self.cache_K[:, :, wp:wp + self.T, :] = new_K
            self.cache_V[:, :, wp:wp + self.T, :] = new_V
            self.write_pos += self.T
        # global_offset advances by T each cycle to match cache slot abs-pos.
        # This means absolute RoPE positions = cycle_idx * T, not commit count.
        self.global_offset += self.T


def pad_or_truncate_ctx(hidden_mx, ctx_size):
    S = hidden_mx.shape[1]
    if S == ctx_size: return hidden_mx
    if S > ctx_size:  return hidden_mx[:, -ctx_size:, :]
    pad = mx.zeros((1, ctx_size - S, hidden_mx.shape[2]), dtype=hidden_mx.dtype)
    return mx.concatenate([hidden_mx, pad], axis=1)


def stream_generate_ane_multifn(target, draft: DFlashANEMultiFnDraft, tok, prompt,
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
    ap.add_argument("--mlpackage", default="/tmp/dflash_ane_multifn.mlpackage")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--state-length", type=int, default=1024)
    ap.add_argument("--num-variants", type=int, default=32)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print("[load] draft (multi-function)...")
    draft = DFlashANEMultiFnDraft(args.mlpackage, config,
                                    state_length=args.state_length,
                                    num_variants=args.num_variants)

    print("[warmup]...")
    list(stream_generate_ane_multifn(target, draft, tok, "The weather", 20))
    draft.t_predict_total = 0.0
    draft.n_predicts = 0
    draft.n_rotate_calls = 0
    draft.n_write_calls = 0

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        gen, t, accepted, cycles, _ = stream_generate_ane_multifn(
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
    print(f"predict: {draft.t_predict_total / max(draft.n_predicts, 1) * 1000:.2f}ms "
          f"({draft.n_predicts} calls; {draft.n_write_calls} write_*, {draft.n_rotate_calls} rotate)")


if __name__ == "__main__":
    main()
