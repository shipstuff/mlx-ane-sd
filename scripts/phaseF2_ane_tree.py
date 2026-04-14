"""Phase F.2: tree speculation on ANE-hosted DFlash draft.

For DFlash's BIDIRECTIONAL draft, alternative token choices at position i
don't change logits at other positions (block-diffusion attention over
MASK tokens is symmetric). So tree speculation simplifies to:

    "accept target's choice at position i if it's in the draft's top-κ
     at position i"

vs standard SD's top-1-only criterion. Draft forward is unchanged; only
the accept rule changes. No extra target compute. The DDTree paper's
more elaborate best-first tree is equivalent to this for bidirectional
drafts — the tree collapses to per-position top-κ.

Expected gain: direct increase in acceptance rate per cycle, no extra cost.
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler
from mlx_lm.tokenizer_utils import TokenizerWrapper

from dflash.model_mlx import _patch_model
from dflash_torch import DFlashConfig
from phaseF1_ane_stream_accum import DFlashANEAccumDraft, pad_or_truncate_ctx


def stream_generate_tree(target, draft, tok, prompt, max_tokens, kappa: int = 2,
                           sampler=None):
    """Same as accum stream_generate, but uses top-kappa acceptance criterion."""
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

        # TREE CHANGE: get top-1 AND top-κ per position
        draft_logits_slice = draft_logits[:, 1 - bs:, :]   # (1, bs-1, vocab)
        mx.eval(draft_logits_slice)
        draft_np = np.asarray(draft_logits_slice.astype(mx.float32))[0]   # (bs-1, vocab)
        # top-1 (argmax, same as standard SD)
        draft_top1 = draft_np.argmax(axis=-1)                              # (bs-1,)
        # top-κ SET for fallback acceptance check (unordered ok)
        topk_ids = np.argpartition(-draft_np, kappa - 1, axis=-1)[:, :kappa]   # (bs-1, kappa)

        draft_tokens_mx = mx.array(draft_top1)[None]   # (1, bs-1)

        verify_input = mx.concatenate([mx.array([[tokens[-1]]]), draft_tokens_mx], axis=1)
        with mx.stream(mx.default_stream(mx.default_device())):
            logits = target(verify_input, target_cache)
            hidden = mx.concatenate(target._hidden_states, axis=-1)
            target_tokens_mx = sampler(logits)
        mx.eval(target_tokens_mx, hidden)

        # TREE-BASED ACCEPT (correct version):
        # - Top-1 matches: target saw exactly what draft predicted, target's next
        #   prediction is valid, keep going.
        # - Top-K mismatch at i (target's choice != draft's top-1 but IS in draft's top-κ):
        #   target's next prediction (at i+1) was based on a wrong chain, so INVALID.
        #   Commit target's choice at i (as "extra accepted" beyond strict top-1)
        #   and STOP.
        # - Hard mismatch (target's choice not in draft's top-κ): standard reject.
        t_list = target_tokens_mx[0].tolist()
        d_list = draft_top1.tolist()   # draft's actual top-1 (argmax)
        accepted = 0
        hit_topk_boundary = False
        for i in range(bs - 1):
            if t_list[i] == d_list[i]:
                accepted += 1
            elif t_list[i] in topk_ids[i]:
                # extra +1 from top-κ, but target's i+1 prediction is stale
                accepted += 1
                hit_topk_boundary = True
                break
            else:
                break
        if hit_topk_boundary:
            # Commit up to accepted-1 using draft's top-1 + target's top-κ match at accepted-1
            # But we want to commit target's choice, and DON'T include bonus since target's
            # last valid logit was at position accepted-1 (before the divergence)
            new_tokens = [int(t_list[i]) for i in range(accepted)]
        else:
            # Standard SD accept: committed prefix + bonus
            new_tokens = [int(t_list[i]) for i in range(accepted + 1)]
        new_tokens = new_tokens[: max_tokens - n]

        eos_ids = set(tok.eos_token_ids) if tok.eos_token_ids else set()
        if eos_ids:
            for i, t in enumerate(new_tokens):
                if t in eos_ids:
                    new_tokens = new_tokens[: i + 1]
                    break

        tokens.extend(new_tokens)
        n += len(new_tokens)
        accepted_total += len(new_tokens)
        cycles += 1

        # Commit to draft cache. The cache K/V were computed assuming draft's
        # top-1 tokens; in a boundary case we're committing target's top-κ choice
        # (different from top-1). The cache advance should match the number of
        # committed tokens, NOT the tree-accepted counter. Pass effective
        # accepted = len(new_tokens) - 1 (mirroring standard SD's accepted).
        effective_accepted = len(new_tokens) - 1
        draft.commit(new_K, new_V, s_real, effective_accepted)

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
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--kappa", type=int, default=2)
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print("[load] target...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    print(f"[load] draft (kappa={args.kappa})...")
    draft = DFlashANEAccumDraft(args.mlmodelc, config, state_length=args.state_length)

    print("[warmup]...")
    list(stream_generate_tree(target, draft, tok, "The weather", 20, kappa=args.kappa))

    rows = []
    for name, prompt in PROMPTS:
        print(f"\n=== {name} (kappa={args.kappa}) ===")
        gen, t, accepted, cycles, _ = stream_generate_tree(
            target, draft, tok, prompt, args.max_new, kappa=args.kappa)
        tps = len(gen) / t
        text = tok.decode(gen)
        print(f"  {len(gen)} tok in {t:.2f}s = {tps:.2f} tok/s "
              f"(cycles={cycles}, accepted={accepted}, avg/cycle={accepted/cycles:.2f})")
        print(f"  text: {text[:80]!r}...")
        rows.append({"name": name, "tps": tps, "cycles": cycles, "accepted": accepted})

    print(f"\n=== Summary (kappa={args.kappa}) ===")
    for r in rows:
        print(f"{r['name']:<12} {r['tps']:>7.2f}  avg/cycle={r['accepted']/r['cycles']:.2f}")
    tpss = [r['tps'] for r in rows]
    print(f"\nmean: {statistics.mean(tpss):.2f} tok/s")


if __name__ == "__main__":
    main()
