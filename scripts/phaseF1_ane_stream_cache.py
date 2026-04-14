"""Phase F.1 cache: DFlash stream_generate with cache-aware ANE draft.

Variant of phaseF1_ane_stream.py that uses the cache-aware DFlash model.
Cache state persists across cycles via CompiledMLModel.make_state().
On partial-reject trim, we reset the state (could snapshot/restore for
finer control).
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


def make_causal_mask(block_size: int, state_length: int, valid_end: int) -> np.ndarray:
    """Attention mask: 0 for positions [0, valid_end), -inf for [valid_end, state_length).
    All block positions attend to the same "valid range" — there's no within-block causality
    in DFlash (it's bidirectional on the draft side)."""
    mask = np.zeros((1, 1, block_size, state_length), dtype=np.float16)
    if valid_end < state_length:
        mask[0, 0, :, valid_end:] = -np.inf
    return mask


class DFlashANECachedDraft:
    """Cache-aware ANE DFlash draft with persistent KV state."""
    def __init__(self, mlmodelc_path: str, config: DFlashConfig,
                 compute_unit=ct.ComputeUnit.ALL):
        print(f"[ane-draft] loading {mlmodelc_path}...")
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(str(mlmodelc_path), compute_unit)
        print(f"[ane-draft] loaded in {time.perf_counter()-t0:.1f}s")
        self.config = config
        self.ctx_size = config.block_size
        self.block_size = config.block_size
        self.head_dim = config.head_dim
        self.state_length = STATE_LENGTH
        self.inv_freq = _rope_inv_freq(self.head_dim, config.rope_theta).numpy()
        self.state = self.model.make_state()
        self.current_pos = 0  # Python-side tracking
        self.t_predict_total = 0.0
        self.n_predicts = 0

    def reset_state(self):
        self.state = self.model.make_state()
        self.current_pos = 0

    def _build_rope(self, write_pos: int, s_real: int) -> dict:
        """Match MLX cache-based rope semantics.

        The draft was trained with an accumulating KV cache. Rope positions
        are relative to the cache offset at the start of each call:
          new ctx:  [write_pos, write_pos + s_real)
          block:    [write_pos + s_real, write_pos + s_real + L)

        Padding slots (ctx beyond s_real) get rope=0 since their K=0.
        """
        k_real = np.arange(s_real, dtype=np.float32) + write_pos
        k_pad = np.zeros(self.ctx_size - s_real, dtype=np.float32)
        k_block = (np.arange(self.block_size, dtype=np.float32) +
                    write_pos + s_real)
        k_positions = np.concatenate([k_real, k_pad, k_block])
        q_positions = (np.arange(self.block_size, dtype=np.float32) +
                        write_pos + s_real)

        def mktbl(positions):
            freqs = np.outer(positions, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)
        cos_q, sin_q = mktbl(q_positions)
        cos_k, sin_k = mktbl(k_positions)
        return {"cos_q": cos_q, "sin_q": sin_q, "cos_k": cos_k, "sin_k": sin_k}

    def forward(self, noise_emb: np.ndarray, target_hidden: np.ndarray,
                 s_real: int, write_pos: int) -> np.ndarray:
        valid_end = write_pos + self.ctx_size + self.block_size
        causal_mask = make_causal_mask(self.block_size, self.state_length, valid_end)

        inputs = {
            "noise_embedding": noise_emb,
            "target_hidden": target_hidden,
            "current_pos": np.array([write_pos], dtype=np.int32),
            "causal_mask": causal_mask,
            **self._build_rope(write_pos, s_real),
        }
        t0 = time.perf_counter()
        out = self.model.predict(inputs, self.state)
        self.t_predict_total += time.perf_counter() - t0
        self.n_predicts += 1
        return out["hidden"]


def pad_or_truncate_ctx(hidden_mx: mx.array, ctx_size: int) -> mx.array:
    S = hidden_mx.shape[1]
    if S == ctx_size: return hidden_mx
    if S > ctx_size:  return hidden_mx[:, -ctx_size:, :]
    pad = mx.zeros((1, ctx_size - S, hidden_mx.shape[2]), dtype=hidden_mx.dtype)
    return mx.concatenate([hidden_mx, pad], axis=1)


def stream_generate_ane_cache(target, draft: DFlashANECachedDraft, tok, prompt: str,
                                max_tokens: int, sampler=None):
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

    # Reset draft cache state
    draft.reset_state()

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

        # Cache write position (persists across cycles)
        write_pos = draft.current_pos
        if write_pos + bs + bs > draft.state_length:
            # Cache would overflow — reset. Loses memory but keeps correctness.
            draft.reset_state()
            write_pos = 0

        hidden_out_np = draft.forward(noise_np, ctx_np, s_real, write_pos)

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

        # Advance draft cache position by (accepted + 1 + s_real):
        # Rejected positions (accepted+1..bs) are still written but will be
        # OVERWRITTEN next cycle at write_pos, so effectively orphaned.
        # The committed advance is accepted + 1 block positions plus s_real ctx.
        draft.current_pos += s_real + accepted + 1

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
                    default="/tmp/dflash_ane_cache_compiled/dflash_ane_cache.mlmodelc")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--compute-unit", choices=["ALL", "CPU_AND_NE"], default="ALL")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    cu_map = {"ALL": ct.ComputeUnit.ALL, "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE}

    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print(f"[load] draft ({args.compute_unit})...")
    draft = DFlashANECachedDraft(args.mlmodelc, config, compute_unit=cu_map[args.compute_unit])

    print("[warmup]...")
    list(stream_generate_ane_cache(target, draft, tok, "The weather", 20))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        gen, t, accepted, cycles, _ = stream_generate_ane_cache(
            target, draft, tok, prompt, args.max_new
        )
        tps = len(gen) / t
        text = tok.decode(gen)
        print(f"  {len(gen)} tok in {t:.2f}s = {tps:.2f} tok/s "
              f"(cycles={cycles}, accepted={accepted}, avg/cycle={accepted/cycles:.2f})")
        print(f"  text: {text[:90]!r}...")
        rows.append({"name": name, "tps": tps, "cycles": cycles, "accepted": accepted,
                     "tokens": len(gen)})

    print("\n=== Summary ===")
    for r in rows:
        avg = r['accepted'] / r['cycles']
        print(f"{r['name']:<12} {r['tps']:>7.2f}  avg/cycle={avg:.1f}")
    tpss = [r['tps'] for r in rows]
    print(f"\nmean: {statistics.mean(tpss):.2f} tok/s "
          f"(min {min(tpss):.2f}, max {max(tpss):.2f})")
    print(f"mean predict: {draft.t_predict_total / draft.n_predicts * 1000:.2f}ms "
          f"({draft.n_predicts} calls)")


if __name__ == "__main__":
    main()
