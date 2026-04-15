"""EAGLE-3 baseline on Qwen3-4B bf16 using MLX.

Ports the taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3 draft checkpoint to MLX and
runs chain-style speculative decoding so the numbers are directly comparable to
our F.0 (DFlash GPU) and F.1 (DFlash ANE) baselines.

Approach (chain, not tree — mirrors F.0/F.1 shape):
    1. Prefill target over prompt and capture hidden states at layers [2, 18, 33]
       (low/mid/high per EAGLE-3 convention for 36-layer Qwen3-4B).
    2. Project concat(low, mid, high) -> draft hidden; run 1-layer draft with
       its own KV cache to autoregressively emit ``num_draft`` token ids.
    3. Verify: run target on those ``num_draft`` tokens in parallel; greedy
       accept the longest matching prefix (+1 bonus token from target).
    4. Truncate draft cache / target cache to accepted length, go again.

The EAGLE-3 draft's token embedding is tied to target's embed_tokens; its
lm_head emits ``draft_vocab_size=32000`` logits and ``d2t[arg] + arg`` gives
the full-vocab token id.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import statistics
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten

# EAGLE-3 standard low/mid/high layer selection for the target:
#   indices [2, num_layers//2, num_layers-3]  -- see SpecForge / EAGLE-3 paper.
# For Qwen3-4B (36 layers) that's [2, 18, 33]. An empirical sweep on several
# prompts put (2, 18, 34) and (3, 18, 34) very slightly higher (~61% teacher-
# forced match vs 59%), but we stay with the published convention here.
EAGLE3_CAPTURE_LAYERS = (2, 18, 33)


# --------------------------------------------------------------------------
# Target wrapper: capture hidden states at 3 layers.
# --------------------------------------------------------------------------
class TargetWithHidden:
    """Wraps an mlx-lm Qwen3 Model to expose layer-2/18/33 hidden states.

    We monkeypatch the model's ``__call__`` to capture hidden states before the
    final norm. Returns (logits, captured_hiddens) where captured_hiddens is a
    stacked [B, L, 3*H] tensor.
    """

    def __init__(self, model):
        self.model = model
        self.args = model.args
        self.layers_capture = EAGLE3_CAPTURE_LAYERS
        assert max(self.layers_capture) < len(model.model.layers)

    def __call__(self, inputs: mx.array, cache=None) -> Tuple[mx.array, mx.array]:
        m = self.model.model  # Qwen3Model
        h = m.embed_tokens(inputs)

        from mlx_lm.models.base import create_attention_mask
        if cache is None:
            cache = [None] * len(m.layers)
        mask = create_attention_mask(h, cache[0])

        captured: List[mx.array] = []
        for idx, (layer, c) in enumerate(zip(m.layers, cache)):
            h = layer(h, mask, c)
            if idx in self.layers_capture:
                captured.append(h)

        out = m.norm(h)
        if self.args.tie_word_embeddings:
            logits = m.embed_tokens.as_linear(out)
        else:
            logits = self.model.lm_head(out)

        triple = mx.concatenate(captured, axis=-1)  # [B, L, 3*H]
        return logits, triple


# --------------------------------------------------------------------------
# EAGLE-3 draft in MLX (single midlayer, Llama-style attention).
# --------------------------------------------------------------------------
class _RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        return mx.fast.rms_norm(x, self.weight, self.eps)


class _LlamaRotary(nn.Module):
    def __init__(self, dim: int, base: float = 1_000_000.0, max_pos: int = 8192):
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_pos = max_pos
        self._cos: Optional[mx.array] = None
        self._sin: Optional[mx.array] = None
        self._build(max_pos)

    def _build(self, seq_len: int) -> None:
        inv_freq = 1.0 / (self.base ** (mx.arange(0, self.dim, 2).astype(mx.float32) / self.dim))
        t = mx.arange(seq_len).astype(mx.float32)
        freqs = mx.outer(t, inv_freq)
        emb = mx.concatenate([freqs, freqs], axis=-1)
        self._cos = mx.cos(emb)
        self._sin = mx.sin(emb)
        self.max_pos = seq_len

    def __call__(self, q: mx.array, k: mx.array, positions: mx.array) -> Tuple[mx.array, mx.array]:
        # q, k: [B, H, L, D]; positions: [L] int.
        need = int(mx.max(positions).item()) + 1
        if need > self.max_pos:
            self._build(max(need + 64, self.max_pos * 2))
        cos = self._cos[positions]  # [L, D]
        sin = self._sin[positions]
        cos = cos[None, None, :, :].astype(q.dtype)
        sin = sin[None, None, :, :].astype(q.dtype)
        half = q.shape[-1] // 2
        q1, q2 = q[..., :half], q[..., half:]
        k1, k2 = k[..., :half], k[..., half:]
        q_rot = mx.concatenate([-q2, q1], axis=-1)
        k_rot = mx.concatenate([-k2, k1], axis=-1)
        q_out = q * cos + q_rot * sin
        k_out = k * cos + k_rot * sin
        return q_out, k_out


class _EagleAttention(nn.Module):
    """Llama-style GQA attention where q/k/v inputs are hidden*2 (input_emb || hidden)."""

    def __init__(self, hidden_size: int, num_heads: int, num_kv_heads: int, head_dim: int,
                 rope_theta: float, max_pos: int = 8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        dim = hidden_size * 2
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=False)
        self.rope = _LlamaRotary(head_dim, base=rope_theta, max_pos=max_pos)

    def __call__(self, x: mx.array, positions: mx.array, cache=None) -> mx.array:
        B, L, _ = x.shape
        q = self.q_proj(x).reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        q, k = self.rope(q, k, positions)

        if cache is not None:
            k_past, v_past = cache
            if k_past is not None:
                k = mx.concatenate([k_past, k], axis=2)
                v = mx.concatenate([v_past, v], axis=2)
            new_cache = (k, v)
        else:
            new_cache = (k, v)

        # Repeat kv to match num_heads
        if self.num_groups > 1:
            k = mx.repeat(k, self.num_groups, axis=1)
            v = mx.repeat(v, self.num_groups, axis=1)

        # Causal mask: query indices (positions) vs key indices (0..K-1).
        key_len = k.shape[2]
        key_idx = mx.arange(key_len)
        q_idx = positions[:, None]  # [L, 1]
        mask = (key_idx[None, :] <= q_idx).astype(q.dtype)  # [L, K]
        # Apply as additive mask (0 for allowed, -inf for blocked).
        additive = (1.0 - mask) * -1e9
        additive = additive[None, None, :, :]  # [1,1,L,K]

        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=additive)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(out), new_cache


class _EagleMLP(nn.Module):
    def __init__(self, hidden_size: int, inter_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, inter_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, inter_size, bias=False)
        self.down_proj = nn.Linear(inter_size, hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class _EagleMidLayer(nn.Module):
    def __init__(self, hidden_size: int, inter_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rope_theta: float, rms_eps: float, max_pos: int = 8192):
        super().__init__()
        self.hidden_norm = _RMSNorm(hidden_size, rms_eps)
        self.input_layernorm = _RMSNorm(hidden_size, rms_eps)
        self.post_attention_layernorm = _RMSNorm(hidden_size, rms_eps)
        self.self_attn = _EagleAttention(hidden_size, num_heads, num_kv_heads, head_dim,
                                         rope_theta, max_pos)
        self.mlp = _EagleMLP(hidden_size, inter_size)

    def __call__(self, input_emb: mx.array, hidden: mx.array, positions: mx.array, cache=None):
        residual = hidden
        hidden = self.hidden_norm(hidden)
        input_emb = self.input_layernorm(input_emb)
        combined = mx.concatenate([input_emb, hidden], axis=-1)  # [B, L, 2H]
        attn_out, new_cache = self.self_attn(combined, positions, cache=cache)
        hidden = residual + attn_out
        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        hidden = residual + hidden
        return hidden, new_cache


class Eagle3Draft(nn.Module):
    def __init__(self, hidden_size: int, inter_size: int, num_heads: int, num_kv_heads: int,
                 head_dim: int, rope_theta: float, rms_eps: float, draft_vocab_size: int,
                 max_pos: int = 8192):
        super().__init__()
        self.hidden_size = hidden_size
        self.draft_vocab_size = draft_vocab_size
        self.fc = nn.Linear(hidden_size * 3, hidden_size, bias=False)
        self.midlayer = _EagleMidLayer(hidden_size, inter_size, num_heads, num_kv_heads,
                                       head_dim, rope_theta, rms_eps, max_pos)
        self.norm = _RMSNorm(hidden_size, rms_eps)
        self.lm_head = nn.Linear(hidden_size, draft_vocab_size, bias=False)
        # Buffers not registered as parameters.
        self.d2t: Optional[mx.array] = None  # [draft_vocab_size], int64
        self.t2d: Optional[mx.array] = None  # [vocab_size], bool

    def step(self, input_emb: mx.array, triple_hidden: mx.array, positions: mx.array,
             cache=None) -> Tuple[mx.array, mx.array, Tuple[mx.array, mx.array]]:
        """First-stage forward (fc + midlayer): triple_hidden is [B, L, 3H].

        Returns (logits, pre_norm_hidden, new_cache). pre_norm_hidden is what
        subsequent autoregressive draft steps feed back as ``hidden`` via
        ``step_projected``.
        """
        hidden = self.fc(triple_hidden)
        hidden, new_cache = self.midlayer(input_emb, hidden, positions, cache=cache)
        pre_norm = hidden
        logits = self.lm_head(self.norm(hidden))
        return logits, pre_norm, new_cache

    def step_projected(self, input_emb: mx.array, hidden: mx.array, positions: mx.array,
                        cache=None) -> Tuple[mx.array, mx.array, Tuple[mx.array, mx.array]]:
        """Autoregressive draft step using a previously-produced hidden (skips fc)."""
        hidden, new_cache = self.midlayer(input_emb, hidden, positions, cache=cache)
        pre_norm = hidden
        logits = self.lm_head(self.norm(hidden))
        return logits, pre_norm, new_cache


def load_eagle3_draft(repo_dir: str, *, rope_theta_override: Optional[float] = None) -> Eagle3Draft:
    import json

    cfg_path = Path(repo_dir) / "config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)

    rope_theta = rope_theta_override if rope_theta_override is not None else cfg["rope_theta"]
    draft = Eagle3Draft(
        hidden_size=cfg["hidden_size"],
        inter_size=cfg["intermediate_size"],
        num_heads=cfg["num_attention_heads"],
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        rope_theta=rope_theta,
        rms_eps=cfg["rms_norm_eps"],
        draft_vocab_size=cfg["draft_vocab_size"],
        max_pos=min(cfg.get("max_position_embeddings", 8192), 8192),
    )
    draft.set_dtype(mx.bfloat16)

    # Load safetensors directly via MLX (handles bf16).
    raw = mx.load(str(Path(repo_dir) / "model.safetensors"))

    draft.d2t = raw["d2t"].astype(mx.int64)
    draft.t2d = raw["t2d"].astype(mx.bool_)
    w_params = {k: v.astype(mx.bfloat16) for k, v in raw.items() if k not in ("d2t", "t2d")}

    mlx_weights = list(w_params.items())
    draft.update(tree_unflatten(mlx_weights))
    mx.eval(draft.parameters())
    return draft


# --------------------------------------------------------------------------
# Chain speculative decoding.
# --------------------------------------------------------------------------
def _greedy_argmax(logits: mx.array) -> int:
    return int(mx.argmax(logits, axis=-1).item())


class _KVCache:
    """Simple mutable KV cache per target layer; Qwen3 layers accept an object with
    .offset, .update_and_fetch(keys, values), following mlx-lm's convention."""

    def __init__(self):
        self.k: Optional[mx.array] = None
        self.v: Optional[mx.array] = None
        self.offset = 0

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> Tuple[mx.array, mx.array]:
        if self.k is None:
            self.k = keys
            self.v = values
        else:
            self.k = mx.concatenate([self.k, keys], axis=2)
            self.v = mx.concatenate([self.v, values], axis=2)
        self.offset = self.k.shape[2]
        return self.k, self.v

    def trim(self, new_len: int) -> None:
        if self.k is None:
            return
        self.k = self.k[:, :, :new_len, :]
        self.v = self.v[:, :, :new_len, :]
        self.offset = new_len


def _make_target_caches(n_layers: int) -> List[_KVCache]:
    return [_KVCache() for _ in range(n_layers)]


def _draft_cache_trim(cache: Optional[Tuple[mx.array, mx.array]], new_len: int):
    if cache is None or cache[0] is None:
        return cache
    k, v = cache
    return (k[:, :, :new_len, :], v[:, :, :new_len, :])


def eagle3_generate(
    target,              # mlx_lm.models.qwen3.Model
    target_with_hidden,  # TargetWithHidden wrapper
    draft: Eagle3Draft,
    tokenizer,
    prompt: str,
    *,
    max_new: int = 100,
    num_draft: int = 4,
    verbose: bool = False,
) -> Tuple[List[int], float, int, int]:
    """Chain-style speculative decoding using EAGLE-3.

    Returns (token_ids, elapsed_seconds, cycles, accepted_total_tokens).
    Acceptance counts draft-tokens matched by the target (not the bonus).
    """
    # Tokenize prompt. Qwen3 Instruct expects chat template — but F.0 / F.1 use
    # raw text, so we do the same here for apples-to-apples.
    ids = tokenizer.encode(prompt)
    if isinstance(ids, list):
        prompt_ids = mx.array(ids, dtype=mx.int32)[None, :]
    else:
        prompt_ids = ids[None, :]

    eos_ids = set()
    for name in ("eos_token_id", "bos_token_id"):
        tid = getattr(tokenizer, name, None)
        if isinstance(tid, int):
            eos_ids.add(tid)

    target_cache = _make_target_caches(len(target.model.layers))
    draft_cache: Optional[Tuple[mx.array, mx.array]] = None

    # ---- Prefill target over prompt, capturing hidden states.
    t_start = time.perf_counter()
    logits_all, triples_all = target_with_hidden(prompt_ids, cache=target_cache)
    mx.eval(logits_all, triples_all)

    # First bonus token from target greedy on last prompt position.
    bonus = _greedy_argmax(logits_all[:, -1, :])
    output: List[int] = [bonus]
    accepted_total = 0
    cycles = 0

    # Prefill the draft over the prompt so its KV cache is seeded.
    prompt_embs = target.model.embed_tokens(prompt_ids)
    mx.eval(prompt_embs)
    prompt_positions = mx.arange(prompt_ids.shape[1])
    _, prompt_pre_norm, draft_cache = draft.step(prompt_embs, triples_all,
                                                 prompt_positions, cache=None)
    mx.eval(draft_cache[0], draft_cache[1], prompt_pre_norm)

    # Rolling autoregressive hidden for the draft.
    last_hidden = prompt_pre_norm[:, -1:, :]

    # Keep the triple we should feed for `last_tok` at the start of the next
    # cycle. After prompt prefill the target's triple at position P-1 is what
    # the draft saw for prompt[P-1]; `last_tok` = bonus0 lives at position P,
    # whose triple is unknown. Fallback: use triples_all[:, -1:, :].
    last_tok_triple = triples_all[:, -1:, :]

    # Note on cache semantics. After prompt prefill:
    #   target_cache has P positions (prompt only).
    #   draft_cache  has P positions (prompt only).
    # `bonus` (=output[-1]) is NOT in either cache yet; it will be fed at the
    # start of the first cycle (in verify for target, in the first draft step
    # for draft).

    while len(output) < max_new:
        cycles += 1
        last_tok = output[-1]

        # ---- Draft rollout: K tokens. Step 0 uses draft.step(fc-projected
        # triple at last_tok's position) so the draft sees the target's
        # ground-truth features for last_tok. Subsequent steps use step_projected
        # with the draft's own pre-norm hidden.
        pos_offset = draft_cache[0].shape[2]
        draft_tokens: List[int] = []
        cycle_hiddens: List[mx.array] = []
        cur_tok = last_tok
        cur_hidden = last_hidden
        for d in range(num_draft):
            emb = target.model.embed_tokens(mx.array([[cur_tok]], dtype=mx.int32))
            pos = mx.array([pos_offset + d])
            if d == 0:
                logits, cur_hidden, draft_cache = draft.step(
                    emb, last_tok_triple, pos, cache=draft_cache)
            else:
                logits, cur_hidden, draft_cache = draft.step_projected(
                    emb, cur_hidden, pos, cache=draft_cache)
            cycle_hiddens.append(cur_hidden)
            d_id = int(mx.argmax(logits[0, -1, :], axis=-1).item())
            full_id = d_id + int(draft.d2t[d_id].item())
            draft_tokens.append(full_id)
            cur_tok = full_id

        # ---- Verify: target forward on [last_tok, d_0, ..., d_{K-1}] producing
        # K+1 logits. Target greedy at position i predicts token after input i.
        verify_input = mx.array([[last_tok] + draft_tokens], dtype=mx.int32)
        verify_logits, verify_triples = target_with_hidden(verify_input, cache=target_cache)

        # Target greedy argmax at every position.
        target_preds = mx.argmax(verify_logits, axis=-1)  # [1, K+1]
        mx.eval(target_preds, verify_triples)
        t_list = target_preds[0].tolist()  # length K+1

        # Find first mismatch between d_list[i] and t_list[i] (both of length K
        # and K+1 respectively, compare indices 0..K-1).
        accepted = 0
        for i in range(len(draft_tokens)):
            if t_list[i] == draft_tokens[i]:
                accepted += 1
            else:
                break
        # Bonus token = t_list[accepted].
        bonus_for_next = t_list[accepted]
        new_tokens = draft_tokens[:accepted] + [bonus_for_next]

        # Truncate to max_new; also stop at EOS.
        for tok_id in new_tokens:
            output.append(tok_id)
            if tok_id in eos_ids or len(output) >= max_new:
                break
        accepted_total += accepted

        # ---- Trim target cache: keep last_tok + accepted drafts, drop bonus
        # and any rejected drafts. Bonus will be re-fed at the start of the
        # next cycle's verify batch. This avoids the KV mismatch that would
        # result from "adopting" d_accepted's position as bonus's.
        prev_offset = target_cache[0].offset - (len(draft_tokens) + 1)
        target_keep = prev_offset + accepted + 1
        for c in target_cache:
            c.trim(target_keep)

        # ---- Trim draft cache: drop rejected draft positions.
        draft_keep = pos_offset + accepted
        draft_cache = _draft_cache_trim(draft_cache, draft_keep)

        # Update last_hidden at the last accepted draft position.
        if accepted > 0:
            last_hidden = cycle_hiddens[accepted - 1]

        # Early-out if bonus hit EOS.
        if bonus_for_next in eos_ids:
            break

        # Prepare bonus's triple for the next cycle's first draft step.
        # We use the target's triple at bonus's position = verify_triples at
        # local index (accepted + 1) if it exists, otherwise reuse last_tok's
        # triple (this only happens for accepted == K, where bonus is past the
        # last fed position — we fall back to the last fed triple).
        if accepted + 1 < verify_triples.shape[1]:
            last_tok_triple = verify_triples[:, accepted + 1:accepted + 2, :]
        else:
            last_tok_triple = verify_triples[:, -1:, :]

    elapsed = time.perf_counter() - t_start
    return output[:max_new], elapsed, cycles, accepted_total


# --------------------------------------------------------------------------
# Plain target-only benchmark (for reference).
# --------------------------------------------------------------------------
def bench_target_only(target, tokenizer, prompt: str, max_new: int) -> Tuple[List[int], float]:
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    tokens = []
    for resp in stream_generate(target, tokenizer, prompt, max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


# --------------------------------------------------------------------------
# Background workload (matches phaseF0_contention.py).
# --------------------------------------------------------------------------
def background_load_worker(model_name: str, stop_event, ready_event):
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler

    print(f"[bg] loading {model_name}...", flush=True)
    model, tok = load(model_name)
    sampler = make_sampler(temp=0.0)
    ready_event.set()
    print(f"[bg] ready, generating continuously", flush=True)

    prompts = [
        "Explain neural networks in one paragraph:",
        "Write a haiku about the sea:",
        "The best programming language is",
    ]
    total = 0
    t0 = time.perf_counter()
    while not stop_event.is_set():
        for p in prompts:
            for resp in stream_generate(model, tok, p, max_tokens=80, sampler=sampler):
                if stop_event.is_set():
                    break
                total += 1
            if stop_event.is_set():
                break
    elapsed = time.perf_counter() - t0
    print(f"[bg] generated {total} tokens in {elapsed:.1f}s ({total / elapsed:.2f} tok/s)",
          flush=True)


# --------------------------------------------------------------------------
# Prompts (identical to phaseF0_contention.py for comparability).
# --------------------------------------------------------------------------
PROMPTS = [
    ("capital", "The capital of France is Paris, which is known for"),
    ("fibonacci", "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    print(f'fib({i}) = {fibonacci(i)}')"),
    ("math", "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6. Now solve 3y - 7 = 20:"),
    ("story", "Once upon a time in a small village nestled between two mountains, there lived a young girl named Elara who"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["solo", "parallel"], default="solo")
    ap.add_argument("--approach", choices=["target", "eagle3"], default="eagle3")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--num-draft", type=int, default=4)
    ap.add_argument("--target", default="mlx-community/Qwen3-4B-Instruct-2507-bf16")
    ap.add_argument("--draft", default="taobao-mnn/Qwen3-4B-Instruct-2507-Eagle3")
    ap.add_argument("--bg-model", default="mlx-community/gemma-3-270m-it-bf16")
    ap.add_argument("--warmup", action="store_true", default=True)
    args = ap.parse_args()

    bg_proc = None
    stop_event = mp.Event()
    ready_event = mp.Event()

    if args.mode == "parallel":
        print(f"[main] starting background workload: {args.bg_model}")
        bg_proc = mp.Process(target=background_load_worker,
                              args=(args.bg_model, stop_event, ready_event),
                              daemon=True)
        bg_proc.start()
        if not ready_event.wait(timeout=180):
            print("[main] background failed to load in 180s")
            sys.exit(1)
        time.sleep(3)

    try:
        from mlx_lm import load
        from huggingface_hub import snapshot_download

        target, tok = load(args.target)

        if args.approach == "eagle3":
            draft_dir = snapshot_download(args.draft)
            print(f"[main] loading EAGLE3 draft from {draft_dir}")
            draft = load_eagle3_draft(draft_dir)
            target_hid = TargetWithHidden(target)

            if args.warmup:
                print("[main] warmup (two passes to compile kernels)...")
                for _ in range(2):
                    _ = eagle3_generate(target, target_hid, draft, tok,
                                        "The weather is pleasant. I think",
                                        max_new=30,
                                        num_draft=args.num_draft)

        rows = []
        for name, prompt in PROMPTS:
            print(f"\n=== {name} ({args.mode}, {args.approach}) ===")
            if args.approach == "target":
                toks, elapsed = bench_target_only(target, tok, prompt, args.max_new)
                tps = len(toks) / elapsed
                print(f"  {len(toks)} tok in {elapsed:.2f}s = {tps:.2f} tok/s")
                rows.append({"name": name, "tps": tps, "tokens": len(toks)})
            else:
                toks, elapsed, cycles, accepted = eagle3_generate(
                    target, target_hid, draft, tok, prompt,
                    max_new=args.max_new, num_draft=args.num_draft)
                tps = len(toks) / elapsed
                acc_rate = accepted / max(cycles, 1)
                text = tok.decode(toks)
                print(f"  {len(toks)} tok in {elapsed:.2f}s = {tps:.2f} tok/s  "
                      f"(cycles={cycles}, accepted={accepted}, acc/cycle={acc_rate:.2f})")
                print(f"  preview: {text[:80]!r}...")
                rows.append({"name": name, "tps": tps, "tokens": len(toks),
                             "cycles": cycles, "accepted": accepted,
                             "acc_per_cycle": acc_rate})

        print("\n=== Summary ===")
        tpss = [r['tps'] for r in rows]
        print(f"mean tok/s: {statistics.mean(tpss):.2f}  "
              f"(min {min(tpss):.2f}, max {max(tpss):.2f})")
        if args.approach == "eagle3":
            accs = [r['acc_per_cycle'] for r in rows]
            print(f"mean acc/cycle: {statistics.mean(accs):.2f}")
        print("\nrows:", rows)

    finally:
        if bg_proc is not None:
            stop_event.set()
            bg_proc.join(timeout=10)
            if bg_proc.is_alive():
                bg_proc.terminate()


if __name__ == "__main__":
    main()
