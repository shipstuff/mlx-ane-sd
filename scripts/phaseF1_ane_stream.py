"""Phase F.1 step 5: DFlash stream_generate with ANE-hosted draft.

Rebuilds z-lab's stream_generate loop but replaces the MLX draft call with
a CoreML CompiledMLModel running on the ANE. Target remains on MLX/GPU.

Notes:
- Draft is stateless (no KV cache across cycles). Impact on acceptance rate
  will be measured against the F.0 baseline.
- target_hidden is padded/truncated to fixed ctx_size=block_size(16) per call.
- RoPE tables are computed Python-side for each cycle's position offset.
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


class DFlashANEDraft:
    """Wraps the compiled CoreML DFlash draft for stream_generate."""
    def __init__(self, mlmodelc_path: str, config: DFlashConfig):
        print(f"[ane-draft] loading {mlmodelc_path}...")
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(
            str(mlmodelc_path), ct.ComputeUnit.CPU_AND_NE,
        )
        print(f"[ane-draft] loaded in {time.perf_counter()-t0:.1f}s")
        self.config = config
        self.ctx_size = config.block_size  # fixed at 16
        self.block_size = config.block_size
        self.head_dim = config.head_dim
        self.inv_freq = _rope_inv_freq(self.head_dim, config.rope_theta).numpy()
        self.t_predict_total = 0.0
        self.n_predicts = 0

    def _build_rope(self, real_ctx_start: int, s_real: int) -> dict:
        """Build RoPE tables matching MLX's absolute-position semantics.

        Real ctx occupies slots [0..s_real); padding fills [s_real..ctx_size).
        Block occupies slots [ctx_size..ctx_size+L).

        Rope positions:
            real ctx: [real_ctx_start, real_ctx_start + s_real)
            padding: zero (unused, since padded K/V = 0)
            block: [real_ctx_start + s_real, real_ctx_start + s_real + L)
        """
        block_abs_start = real_ctx_start + s_real
        q_positions = np.arange(self.block_size, dtype=np.float32) + block_abs_start
        k_real = np.arange(s_real, dtype=np.float32) + real_ctx_start
        k_pad = np.zeros(self.ctx_size - s_real, dtype=np.float32)
        k_block = np.arange(self.block_size, dtype=np.float32) + block_abs_start
        k_positions = np.concatenate([k_real, k_pad, k_block])

        def mktbl(positions):
            freqs = np.outer(positions, self.inv_freq)
            emb = np.concatenate([freqs, freqs], axis=-1)
            return np.cos(emb).astype(np.float16), np.sin(emb).astype(np.float16)
        cos_q, sin_q = mktbl(q_positions)
        cos_k, sin_k = mktbl(k_positions)
        return {"cos_q": cos_q, "sin_q": sin_q, "cos_k": cos_k, "sin_k": sin_k}

    def forward(self, noise_emb: np.ndarray, target_hidden: np.ndarray,
                 real_ctx_start: int, s_real: int) -> np.ndarray:
        """Inputs in numpy fp16; target_hidden is LEFT-aligned real, right-padded."""
        inputs = {
            "noise_embedding": noise_emb,
            "target_hidden": target_hidden,
            **self._build_rope(real_ctx_start, s_real),
        }
        t0 = time.perf_counter()
        out = self.model.predict(inputs)
        self.t_predict_total += time.perf_counter() - t0
        self.n_predicts += 1
        return out["hidden"]


def pad_or_truncate_ctx(hidden_mx: mx.array, ctx_size: int) -> mx.array:
    """Ensure target_hidden has exactly ctx_size positions on the time axis."""
    S = hidden_mx.shape[1]
    if S == ctx_size:
        return hidden_mx
    if S > ctx_size:
        return hidden_mx[:, -ctx_size:, :]
    pad = mx.zeros((1, ctx_size - S, hidden_mx.shape[2]), dtype=hidden_mx.dtype)
    return mx.concatenate([hidden_mx, pad], axis=1)


def stream_generate_ane(target, draft: DFlashANEDraft, tok, prompt: str,
                         max_tokens: int, sampler=None):
    """DFlash stream generation with ANE-hosted draft."""
    sampler = sampler or make_sampler(temp=0.0)
    if not isinstance(tok, TokenizerWrapper):
        tok = TokenizerWrapper(tok)
    config = draft.config

    # Patch target to capture hidden states
    _patch_model(target, config.target_layer_ids)

    # Tokenize
    add_special = tok.bos_token is None or not prompt.startswith(tok.bos_token)
    prompt_ids = tok.encode(prompt, add_special_tokens=add_special)
    prompt_arr = mx.array(prompt_ids)
    tokens = list(prompt_ids)

    # Prefill
    target_cache = make_prompt_cache(target)
    t_prefill = time.perf_counter()
    with mx.stream(mx.default_stream(mx.default_device())):
        logits = target(prompt_arr[None], target_cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
    mx.eval(logits, hidden)
    # Bonus token from prefill
    token = int(mx.argmax(logits[:, -1:], axis=-1).item())
    tokens.append(token)
    n = 1
    t_prefill = time.perf_counter() - t_prefill

    bs = config.block_size
    mask_id = config.mask_token_id
    accepted_total = 0
    cycles = 0

    # Decode loop
    t_decode_start = time.perf_counter()
    while n < max_tokens and not (tok.eos_token_ids and token in tok.eos_token_ids):
        # Prepare draft inputs
        block = mx.array([[tokens[-1]] + [mask_id] * (bs - 1)])      # (1, bs)
        noise_emb_mx = target.model.embed_tokens(block)              # (1, bs, H)
        mx.eval(noise_emb_mx)
        ctx_mx = pad_or_truncate_ctx(hidden, bs)                      # (1, bs, concat_dim)

        # s_real is the EFFECTIVE number of real ctx positions used:
        # - if hidden was padded (hidden.shape[1] < bs), s_real = original shape
        # - if hidden was truncated (hidden.shape[1] > bs), we took last bs positions,
        #   so effective s_real = bs
        s_real = min(hidden.shape[1], bs)
        noise_np = np.asarray(noise_emb_mx.astype(mx.float32)).astype(np.float16)
        ctx_np = np.asarray(ctx_mx.astype(mx.float32)).astype(np.float16)

        # real_ctx_start = block_abs_start - s_real
        # block_abs_start = position of tokens[-1] in the absolute sequence
        block_abs_start = len(tokens) - 1
        real_ctx_start = block_abs_start - s_real

        # Call ANE draft → hidden (1, bs, H)
        hidden_out_np = draft.forward(noise_np, ctx_np, real_ctx_start, s_real)

        # Apply target's LM head to get draft logits
        hidden_out_mx = mx.array(hidden_out_np.astype(np.float32)).astype(mx.bfloat16)
        if hasattr(target, "lm_head"):
            draft_logits = target.lm_head(hidden_out_mx)
        else:
            draft_logits = target.model.embed_tokens.as_linear(hidden_out_mx)

        # Sample: use positions (1-bs):  (bs-1) draft tokens (drop the first position
        # which corresponds to the known input token)
        draft_tokens_mx = sampler(draft_logits[:, 1 - bs:])          # (1, bs-1)
        mx.eval(draft_tokens_mx)

        # Target verify: [last_token] + draft_tokens, length = bs
        verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens_mx], axis=1)
        with mx.stream(mx.default_stream(mx.default_device())):
            logits = target(verify_input, target_cache)
            hidden = mx.concatenate(target._hidden_states, axis=-1)
            target_tokens_mx = sampler(logits)
        mx.eval(target_tokens_mx, hidden)

        # Compare draft_tokens (bs-1) to target_tokens[:, 1:-1] ... wait,
        # let me match MLX's logic:
        #   d_list = draft_tokens[0].tolist()   (len bs-1)
        #   t_list = target_tokens[0].tolist()  (len bs)
        #   accepted = first i where d_list[i] != t_list[i]
        # Target samples one extra token (bonus), at position i=accepted, t_list[accepted]
        d_list = draft_tokens_mx[0].tolist()
        t_list = target_tokens_mx[0].tolist()
        accepted = next((i for i in range(len(d_list)) if d_list[i] != t_list[i]),
                         len(d_list))
        new_tokens = d_list[:accepted] + [t_list[accepted]]
        new_tokens = new_tokens[: max_tokens - n]

        # Stop on EOS
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

        # Trim target cache to accepted + 1 positions (mirroring stream_generate)
        trim = bs - accepted - 1
        if trim > 0:
            from mlx_lm.models.cache import trim_prompt_cache, can_trim_prompt_cache
            if can_trim_prompt_cache(target_cache):
                trim_prompt_cache(target_cache, trim)

        # Trim hidden to the accepted positions
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
    ap.add_argument("--mlmodelc", default="/tmp/dflash_ane_compiled/dflash_ane.mlmodelc")
    ap.add_argument("--max-new", type=int, default=100)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    from dflash_torch import DFlashConfig as Cfg
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = Cfg.from_hf_json(str(Path(draft_path) / "config.json"))

    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print("[load] ANE draft...")
    draft = DFlashANEDraft(args.mlmodelc, config)

    # Warmup
    print("[warmup] 20 tok...")
    list(stream_generate_ane(target, draft, tok, "The weather", 20))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        gen, t_decode, accepted, cycles, t_prefill = stream_generate_ane(
            target, draft, tok, prompt, args.max_new
        )
        tps = len(gen) / t_decode
        text = tok.decode(gen)
        print(f"  decode: {len(gen)} tok in {t_decode:.2f}s = {tps:.2f} tok/s")
        print(f"  prefill: {t_prefill*1000:.0f}ms, cycles={cycles}, accepted={accepted}")
        print(f"  text: {text[:100]!r}...")
        rows.append({"name": name, "tps": tps, "cycles": cycles, "accepted": accepted,
                     "tokens": len(gen)})

    print("\n=== Summary ===")
    print(f"{'prompt':<12} {'tok/s':>7} {'cycles':>7} {'accept':>7} {'avg/cycle':>10}")
    for r in rows:
        avg = r['accepted'] / r['cycles'] if r['cycles'] else 0
        print(f"{r['name']:<12} {r['tps']:>7.2f} {r['cycles']:>7} {r['accepted']:>7} {avg:>10.1f}")
    tpss = [r['tps'] for r in rows]
    print(f"\nmean tok/s: {statistics.mean(tpss):.2f}  (min {min(tpss):.2f}, max {max(tpss):.2f})")
    print(f"mean predict time per call: {draft.t_predict_total / draft.n_predicts * 1000:.2f}ms "
          f"({draft.n_predicts} calls)")


if __name__ == "__main__":
    main()
