"""Phase F.2 2-chain test: run two candidate draft chains, pick the winner.

Simplest tree speculation: chain_1 uses draft's top-1 at all positions;
chain_2 uses draft's top-2 at position 0, top-1 at the rest. Because
DFlash is bidirectional, positions 1..N-1 are sampled independently of
position 0 — they have the SAME logits in both chains.

For each chain, run target separately (snapshot-restore cache to keep
both chains fair). Pick the chain with longer accepted prefix. Commit
the winning chain.

This tests whether multi-chain speculation offers meaningful gains on
DFlash before investing in proper tree attention.
"""
from __future__ import annotations

import argparse
import copy
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache, trim_prompt_cache, can_trim_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from dflash.model_mlx import _patch_model
from dflash_torch import DFlashConfig
from phaseF1_ane_stream_accum import DFlashANEAccumDraft, pad_or_truncate_ctx


def deep_copy_cache(cache_list):
    """Shallow-copy state to enable rollback."""
    return [c.state if hasattr(c, 'state') else None for c in cache_list]


def stream_generate_2chain(target, draft, tok, prompt, max_tokens, sampler=None):
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
    chain2_wins = 0
    chain2_trials = 0

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

        # Compute top-2 at position 0 only
        draft_logits_slice = draft_logits[:, 1 - bs:, :]
        draft_np = np.asarray(draft_logits_slice.astype(mx.float32))[0]  # (bs-1, vocab)
        top1_per_pos = draft_np.argmax(axis=-1)   # (bs-1,)
        top2_at_pos0 = np.argpartition(-draft_np[0], 1)[:2]  # top-2 at position 0
        top2_at_pos0 = top2_at_pos0[np.argsort(-draft_np[0, top2_at_pos0])]  # sort desc
        assert top2_at_pos0[0] == top1_per_pos[0], \
            f"top-2[0] {top2_at_pos0[0]} != top-1[0] {top1_per_pos[0]}"

        # Chain 1: all top-1
        chain1 = [tokens[-1]] + [int(x) for x in top1_per_pos]
        # Chain 2: top-2 at position 0, top-1 elsewhere
        chain2 = [tokens[-1], int(top2_at_pos0[1])] + [int(x) for x in top1_per_pos[1:]]

        # Snapshot target cache before verify
        # (For MLX caches, we record the offset; after verify we can trim back.)
        # Simpler approach: run chain1, record result. Trim cache back. Run chain2.

        # Chain 1 verify
        verify1 = mx.array([chain1])
        cache_offset_before = target_cache[0].offset if target_cache else 0
        with mx.stream(mx.default_stream(mx.default_device())):
            logits1 = target(verify1, target_cache)
            hidden1 = mx.concatenate(target._hidden_states, axis=-1)
            ttokens1 = sampler(logits1)
        mx.eval(ttokens1, hidden1)
        t1 = ttokens1[0].tolist()
        d1 = chain1[1:]  # draft tokens part of chain1 (sans bonus prefix)
        acc1 = next((i for i in range(len(d1)) if d1[i] != t1[i]), len(d1))

        # Trim cache back to before chain1 verify
        trim_back = target_cache[0].offset - cache_offset_before
        if trim_back > 0:
            trim_prompt_cache(target_cache, trim_back)

        # Chain 2 verify
        verify2 = mx.array([chain2])
        with mx.stream(mx.default_stream(mx.default_device())):
            logits2 = target(verify2, target_cache)
            hidden2 = mx.concatenate(target._hidden_states, axis=-1)
            ttokens2 = sampler(logits2)
        mx.eval(ttokens2, hidden2)
        t2 = ttokens2[0].tolist()
        d2 = chain2[1:]
        acc2 = next((i for i in range(len(d2)) if d2[i] != t2[i]), len(d2))

        # Pick winner
        chain2_trials += 1
        if acc2 > acc1:
            chain2_wins += 1
            accepted = acc2
            t_list = t2
            d_list = d2
            hidden = hidden2
            # cache currently has chain2 state — what we want
        else:
            accepted = acc1
            t_list = t1
            d_list = d1
            # cache has chain2 state — need to re-run chain1 to get chain1 cache state
            # (inefficient but correct)
            # Trim back again
            trim_back = target_cache[0].offset - cache_offset_before
            if trim_back > 0:
                trim_prompt_cache(target_cache, trim_back)
            with mx.stream(mx.default_stream(mx.default_device())):
                logits1b = target(verify1, target_cache)
                hidden1 = mx.concatenate(target._hidden_states, axis=-1)
            mx.eval(hidden1)
            hidden = hidden1

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
        if trim > 0 and can_trim_prompt_cache(target_cache):
            trim_prompt_cache(target_cache, trim)

        hidden = hidden[:, :accepted + 1, :]
        if new_tokens and new_tokens[-1] in eos_ids:
            break
        token = tokens[-1]

    t_decode = time.perf_counter() - t_decode_start
    generated = tokens[len(prompt_ids):]
    return generated, t_decode, accepted_total, cycles, chain2_wins, chain2_trials, t_prefill


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
    ap.add_argument("--state-length", type=int, default=256)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print("[load] draft...")
    draft = DFlashANEAccumDraft(args.mlmodelc, config, state_length=args.state_length)

    print("[warmup]...")
    list(stream_generate_2chain(target, draft, tok, "The weather", 20))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} ===")
        gen, t, accepted, cycles, c2w, c2t, _ = stream_generate_2chain(
            target, draft, tok, prompt, args.max_new)
        tps = len(gen) / t
        print(f"  {len(gen)} tok in {t:.2f}s = {tps:.2f} tok/s "
              f"(cycles={cycles}, accepted={accepted}, chain2 wins {c2w}/{c2t})")
        rows.append({"name": name, "tps": tps, "c2_win_rate": c2w / max(c2t, 1)})

    print("\n=== Summary ===")
    for r in rows:
        print(f"{r['name']:<12} {r['tps']:>7.2f}  chain2 wins {r['c2_win_rate']*100:.0f}%")
    tpss = [r['tps'] for r in rows]
    print(f"\nmean: {statistics.mean(tpss):.2f} tok/s")


if __name__ == "__main__":
    main()
