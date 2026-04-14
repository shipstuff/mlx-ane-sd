"""Phase A: sequential heterogeneous speculative decoding.

Draft on ANE (CoreML), target on MLX/GPU. Sequential — no parallelism yet.
Goal is correctness: does the SD loop produce text identical to pure
target-only greedy decode? Once that's proven, Phase B adds concurrency.

Invariant for the ANE draft:
    draft.pos = number of tokens fed so far.
    draft.pos == p means the ANE cache is populated for positions 0..p-1,
    and the next _step(token) will feed `token` at position p, returning
    a prediction for position p+1.

SD cycle:
    committed = prompt + generated so far
    1. Reset draft state; prefill positions 0..len(committed)-2
       (so draft.pos == len(committed) - 1)
    2. Draft produces K tokens starting with committed[-1] as the input
    3. Target verifies via a single forward over [committed | draft]
    4. Accept longest agreeing prefix; commit target's correction for the
       first mismatch (or a bonus token if all accepted)

Step 1 makes the loop O(N*K) in draft work (full re-prefill each cycle) —
not production-efficient, but correct, simple, and enough to validate the
approach. Phase B optimizes.

Env: MLX venv (has both mlx-lm and coremltools 9.0).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np


ANE_MODEL_DIR = Path("/Users/carl/models/gemma3-270m-ane")
ANE_MODEL_FILE = ANE_MODEL_DIR / "gemma3_monolithic_full_lut6.mlmodelc"
CONTEXT_LENGTH = 512
VOCAB_SIZE = 262144
NUM_LMHEAD_CHUNKS = 16
CHUNK_SIZE = VOCAB_SIZE // NUM_LMHEAD_CHUNKS


def make_causal_mask(length: int) -> np.ndarray:
    mask = np.full((length, length), -np.inf, dtype=np.float16)
    for i in range(length):
        mask[i, : i + 1] = 0.0
    return mask[None, None, :, :]


def argmax_from_output(output: dict) -> int:
    argmax_idx = output["argmax_idx"].flatten()
    argmax_val = output["argmax_val"].flatten()
    best_chunk = int(np.argmax(argmax_val))
    local_idx = int(argmax_idx[best_chunk])
    return local_idx + best_chunk * CHUNK_SIZE


class ANEDraft:
    def __init__(self):
        import coremltools as ct
        print(f"[ane-draft] loading {ANE_MODEL_FILE.name}...", flush=True)
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(
            str(ANE_MODEL_FILE), ct.ComputeUnit.CPU_AND_NE, function_name="infer"
        )
        print(f"[ane-draft] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        self.causal_mask = make_causal_mask(CONTEXT_LENGTH)
        self.state = None
        self.pos = 0

    def reset(self):
        self.state = self.model.make_state()
        self.pos = 0

    def step(self, token_id: int) -> int:
        """Feed token at self.pos. Return predicted next token."""
        inp = {
            "input_ids": np.array([[token_id]], dtype=np.int32),
            "position_ids": np.array([self.pos], dtype=np.int32),
            "causal_mask": self.causal_mask[:, :, self.pos : self.pos + 1, :].astype(np.float16),
            "current_pos": np.array([self.pos], dtype=np.int32),
        }
        out = self.model.predict(inp, self.state)
        self.pos += 1
        return argmax_from_output(out)

    def prefill_through(self, committed: List[int]):
        """After this call, draft.pos == len(committed) - 1.

        Reset and feed all but the last token so the next step(committed[-1])
        predicts the token that comes after committed[-1].
        """
        self.reset()
        # Feed positions 0..len-2 (token committed[p] at position p)
        for p in range(len(committed) - 1):
            self.step(committed[p])
        # Now self.pos == len(committed) - 1

    def draft_k(self, start_token: int, k: int) -> List[int]:
        """Feed start_token at current pos, then autoregressively generate k-1 more.

        Returns k predicted tokens.
        """
        tokens = []
        current = start_token
        for _ in range(k):
            nxt = self.step(current)
            tokens.append(nxt)
            current = nxt
        return tokens


class MLXTarget:
    """Thin wrapper around an mlx-lm loaded model for full-sequence verify."""

    def __init__(self, model_path: str):
        from mlx_lm import load
        print(f"[target] loading {model_path}...", flush=True)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        print(f"[target] loaded in {time.perf_counter() - t0:.1f}s", flush=True)

    def verify(self, committed: List[int], draft: List[int]) -> Tuple[int, List[int]]:
        """Run the target on committed+draft, return first_reject_idx and
        the target's argmax prediction at each draft position + bonus.

        The target's logits at seq[i] predict the token at seq[i+1]. We
        feed seq = committed + draft. For draft[i] at absolute position
        c+i (c = len(committed)), the target's prediction came from
        logits at index c+i-1.

        Returns:
            first_reject: index in draft of the first mismatch (K if all match)
            target_preds: K+1 target tokens (at positions c, c+1, ..., c+K)
                          where the last is the bonus token
        """
        import mlx.core as mx
        c = len(committed)
        K = len(draft)
        seq = mx.array(committed + draft).reshape(1, -1)
        logits = self.model(seq)  # [1, c+K, vocab]

        preds = []
        for i in range(K + 1):
            pred_idx = c + i - 1
            # For the first position (committed is empty), predicted from position 0
            # This is an edge case for our use; prompt always nonempty so c >= 1
            pred_idx = max(pred_idx, 0)
            tok = int(mx.argmax(logits[0, pred_idx, :]).item())
            preds.append(tok)

        first_reject = K
        for i in range(K):
            if draft[i] != preds[i]:
                first_reject = i
                break

        return first_reject, preds


def run_sd(draft: ANEDraft, target: MLXTarget, prompt_ids: List[int],
           max_new: int, num_draft: int, verbose: bool = True) -> Tuple[List[int], dict]:
    committed = list(prompt_ids)
    generated: List[int] = []
    stats = {
        "cycles": 0, "draft_tokens_generated": 0, "draft_tokens_accepted": 0,
        "t_draft": 0.0, "t_target": 0.0, "t_total": 0.0,
    }
    t_start = time.perf_counter()

    while len(generated) < max_new:
        if len(committed) >= CONTEXT_LENGTH - num_draft - 2:
            if verbose:
                print("[sd] reached context limit, stopping")
            break

        # --- Draft phase ---
        t0 = time.perf_counter()
        draft.prefill_through(committed)
        start_tok = committed[-1]
        draft_tokens = draft.draft_k(start_tok, num_draft)
        stats["t_draft"] += time.perf_counter() - t0

        # --- Verify phase ---
        t0 = time.perf_counter()
        first_reject, target_preds = target.verify(committed, draft_tokens)
        stats["t_target"] += time.perf_counter() - t0

        # --- Commit ---
        # Accept draft[0..first_reject-1], then target's correction at first_reject
        # (which equals target_preds[first_reject]) as the bonus.
        accepted = draft_tokens[:first_reject]
        # target_preds[first_reject] is the correct token at that position
        # (if first_reject < K: it's the correction; if first_reject == K: it's the bonus)
        correction = target_preds[first_reject]
        committed.extend(accepted)
        committed.append(correction)
        generated.extend(accepted)
        generated.append(correction)

        stats["cycles"] += 1
        stats["draft_tokens_generated"] += num_draft
        stats["draft_tokens_accepted"] += first_reject

        if verbose:
            print(f"  [cycle {stats['cycles']}] draft={num_draft} accept={first_reject}/{num_draft} "
                  f"committed_len={len(committed)}")

    stats["t_total"] = time.perf_counter() - t_start
    return generated, stats


def baseline_target_only(target: MLXTarget, prompt_ids: List[int], max_new: int) -> Tuple[List[int], float]:
    """Pure target-only greedy decode using mlx-lm for reference."""
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    tokens = []
    t0 = time.perf_counter()
    prompt_str = target.tokenizer.decode(prompt_ids)
    for resp in stream_generate(target.model, target.tokenizer, prompt_str,
                                 max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-3-12b-it-bf16")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=30)
    ap.add_argument("--num-draft", type=int, default=4)
    ap.add_argument("--skip-baseline", action="store_true")
    args = ap.parse_args()

    target = MLXTarget(args.target)
    draft = ANEDraft()

    prompt_ids = target.tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"[main] prompt: {args.prompt!r}  tokens: {prompt_ids}")

    # --- Pure target baseline for reference ---
    if not args.skip_baseline:
        print(f"\n=== baseline: target-only (mlx-lm) ===")
        base_tokens, base_elapsed = baseline_target_only(target, prompt_ids, args.max_new_tokens)
        print(f"tokens: {base_tokens}")
        print(f"text: {target.tokenizer.decode(base_tokens)!r}")
        print(f"time: {base_elapsed*1000:.0f} ms  ({len(base_tokens)/base_elapsed:.2f} tok/s)")

    # --- Our heterogeneous SD ---
    print(f"\n=== heterogeneous SD (ANE draft + MLX target) ===")
    gen, stats = run_sd(draft, target, prompt_ids, args.max_new_tokens, args.num_draft)
    print(f"\ntokens: {gen}")
    print(f"text: {target.tokenizer.decode(gen)!r}")
    print(f"time: {stats['t_total']*1000:.0f} ms  ({len(gen)/stats['t_total']:.2f} tok/s)")
    print(f"cycles: {stats['cycles']}  accept: {stats['draft_tokens_accepted']}/"
          f"{stats['draft_tokens_generated']} = "
          f"{stats['draft_tokens_accepted']/max(stats['draft_tokens_generated'],1):.1%}")
    print(f"draft time: {stats['t_draft']*1000:.0f} ms "
          f"({stats['t_draft']/stats['t_total']*100:.0f}%)")
    print(f"target time: {stats['t_target']*1000:.0f} ms "
          f"({stats['t_target']/stats['t_total']*100:.0f}%)")

    # Correctness check: did we produce identical text?
    if not args.skip_baseline:
        print(f"\n=== correctness ===")
        min_len = min(len(base_tokens), len(gen))
        match_len = 0
        for i in range(min_len):
            if base_tokens[i] == gen[i]:
                match_len += 1
            else:
                break
        print(f"matched prefix: {match_len}/{min_len} tokens")
        if match_len < min_len:
            print(f"  base[{match_len}]: {base_tokens[match_len]} = "
                  f"{target.tokenizer.decode([base_tokens[match_len]])!r}")
            print(f"  ours[{match_len}]: {gen[match_len]} = "
                  f"{target.tokenizer.decode([gen[match_len]])!r}")


if __name__ == "__main__":
    main()
