"""Phase B.2: concurrent heterogeneous SD.

Runs cycle N+1's speculative draft on ANE CONCURRENTLY with cycle N's
target verify on GPU. Speculation assumes cycle N is full-accept AND that
the ANE's prediction after d_{K-1} matches the target's.

Per cycle:
    1. Collect primary draft (from previous cycle or cold-start)
    2. Step d_{K-1} into cache → next_guess = ANE's prediction after d_{K-1}
    3. Snapshot state (S_post_primary) and submit spec draft from next_guess
    4. CONCURRENTLY: target verifies cycle N's primary
    5. Decision:
       - Full accept + next_guess == correction: SPEC HIT. Use spec as next primary.
       - Full accept + mismatch: SPEC MISS. Restore S_post_primary, re-draft.
       - Partial: Discard spec. Restore S_committed, advance through accepted.

Env: MLX venv (has mlx-lm and coremltools 9.0).
"""
from __future__ import annotations

import argparse
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ANE_MODEL_DIR = Path("/Users/carl/models/gemma3-270m-ane")
ANE_MODEL_FILE = ANE_MODEL_DIR / "gemma3_monolithic_full_lut6.mlmodelc"
CONTEXT_LENGTH = 512
VOCAB_SIZE = 262144
NUM_LMHEAD_CHUNKS = 16
CHUNK_SIZE = VOCAB_SIZE // NUM_LMHEAD_CHUNKS

STATE_GLOBAL = "model_model_kv_cache_global"
STATE_LOCAL = "model_model_kv_cache_local"


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


class ANEDraftWorker:
    """ANE draft model with a background-thread step loop.

    The worker thread runs multi-step draft jobs. The main thread owns the
    state (read/write snapshots, synchronous steps). We never have the
    worker and main thread touching the state simultaneously — submit_draft
    is non-blocking but callers must collect_draft before doing any sync
    work on the model.
    """

    def __init__(self):
        import coremltools as ct
        print(f"[ane] loading {ANE_MODEL_FILE.name}...", flush=True)
        t0 = time.perf_counter()
        self.model = ct.models.CompiledMLModel(
            str(ANE_MODEL_FILE), ct.ComputeUnit.CPU_AND_NE, function_name="infer"
        )
        print(f"[ane] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        self.causal_mask = make_causal_mask(CONTEXT_LENGTH)
        self.state = self.model.make_state()
        self.pos = 0

        self._request_event = threading.Event()
        self._done_event = threading.Event()
        self._request: Optional[Tuple[int, int]] = None
        self._result: Optional[List[int]] = None
        self._shutdown = False
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

        # Stats
        self.t_step_total = 0.0
        self.n_steps = 0
        self.t_snapshot_total = 0.0
        self.t_restore_total = 0.0

    def _step_raw(self, token_id: int) -> int:
        t0 = time.perf_counter()
        inp = {
            "input_ids": np.array([[token_id]], dtype=np.int32),
            "position_ids": np.array([self.pos], dtype=np.int32),
            "causal_mask": self.causal_mask[:, :, self.pos : self.pos + 1, :].astype(np.float16),
            "current_pos": np.array([self.pos], dtype=np.int32),
        }
        out = self.model.predict(inp, self.state)
        self.pos += 1
        self.t_step_total += time.perf_counter() - t0
        self.n_steps += 1
        return argmax_from_output(out)

    def _worker_loop(self):
        while not self._shutdown:
            self._request_event.wait()
            self._request_event.clear()
            if self._shutdown:
                return
            start_token, k = self._request
            tokens = []
            current = start_token
            for _ in range(k):
                nxt = self._step_raw(current)
                tokens.append(nxt)
                current = nxt
            self._result = tokens
            self._done_event.set()

    def submit_draft(self, start_token: int, k: int):
        assert not self._request_event.is_set()
        assert not self._done_event.is_set()
        self._request = (start_token, k)
        self._request_event.set()

    def collect_draft(self) -> List[int]:
        self._done_event.wait()
        self._done_event.clear()
        r = self._result
        self._result = None
        return r

    def step_sync(self, token_id: int) -> int:
        assert not self._request_event.is_set() and not self._done_event.is_set(), \
            "worker is busy — call collect_draft first"
        return self._step_raw(token_id)

    def snapshot(self) -> Dict[str, np.ndarray]:
        t0 = time.perf_counter()
        snap = {
            "global": self.state.read_state(STATE_GLOBAL).copy(),
            "local": self.state.read_state(STATE_LOCAL).copy(),
        }
        self.t_snapshot_total += time.perf_counter() - t0
        return snap

    def restore(self, snap: Dict[str, np.ndarray], pos: int):
        t0 = time.perf_counter()
        self.state.write_state(STATE_GLOBAL, snap["global"])
        self.state.write_state(STATE_LOCAL, snap["local"])
        self.pos = pos
        self.t_restore_total += time.perf_counter() - t0

    def reset_state(self):
        self.state = self.model.make_state()
        self.pos = 0

    def prefill_initial(self, tokens: List[int]):
        self.reset_state()
        for t in tokens:
            self.step_sync(t)

    def shutdown(self):
        self._shutdown = True
        self._request_event.set()
        self._worker_thread.join(timeout=2.0)


class MLXTarget:
    def __init__(self, model_path: str):
        from mlx_lm import load
        print(f"[target] loading {model_path}...", flush=True)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        print(f"[target] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        self.t_verify_total = 0.0
        self.n_verifies = 0

    def verify(self, committed: List[int], draft: List[int]) -> Tuple[int, List[int]]:
        import mlx.core as mx
        t0 = time.perf_counter()
        c = len(committed)
        K = len(draft)
        seq = mx.array(committed + draft).reshape(1, -1)
        logits = self.model(seq)
        preds = []
        for i in range(K + 1):
            pred_idx = max(c + i - 1, 0)
            tok = int(mx.argmax(logits[0, pred_idx, :]).item())
            preds.append(tok)
        first_reject = K
        for i in range(K):
            if draft[i] != preds[i]:
                first_reject = i
                break
        self.t_verify_total += time.perf_counter() - t0
        self.n_verifies += 1
        return first_reject, preds


def run_sd_concurrent(draft: ANEDraftWorker, target: MLXTarget, prompt_ids: List[int],
                      max_new: int, num_draft: int, verbose: bool = True):
    """Concurrent SD with cross-cycle speculation."""
    committed = list(prompt_ids)
    t0 = time.perf_counter()
    draft.prefill_initial(committed[:-1])
    t_initial_prefill = time.perf_counter() - t0

    # State anchor: S_committed reflects committed[:-1]
    S_committed = draft.snapshot()
    p_committed = draft.pos  # == len(committed) - 1

    # Cycle 0: draft synchronously (no prior spec to reuse)
    draft.submit_draft(committed[-1], num_draft)
    primary: Optional[List[int]] = None
    # We'll collect it in the loop on first iteration

    generated: List[int] = []
    stats = {
        "cycles": 0, "draft_generated": 0, "draft_accepted": 0,
        "full_accept_cycles": 0,
        "spec_hit": 0, "spec_miss_full": 0, "spec_miss_partial": 0,
        "t_total": 0.0, "t_initial_prefill": t_initial_prefill,
        "t_overlap": 0.0,  # time verify ran concurrently with spec draft
    }
    t_start = time.perf_counter()

    # State invariant before each iteration: state anchored at S_committed,
    # a draft job for committed[-1] has been submitted OR primary is non-None
    # (carried over from previous spec hit).

    while len(generated) < max_new:
        if len(committed) >= CONTEXT_LENGTH - num_draft * 2 - 4:
            if verbose:
                print("[sd] ctx limit, stop")
            break

        # --- 1. Get primary draft ---
        if primary is None:
            primary = draft.collect_draft()

        # --- 2. Finalize primary's d_{K-1} → next_guess ---
        next_guess = draft.step_sync(primary[-1])
        # state.pos == p_committed + num_draft + 1

        # --- 3. Snapshot post-primary state + submit spec ---
        S_post_primary = draft.snapshot()
        p_post_primary = draft.pos

        can_speculate = len(generated) + num_draft + 1 < max_new
        if can_speculate:
            draft.submit_draft(next_guess, num_draft)

        # --- 4. CONCURRENT: target verify ---
        t_v0 = time.perf_counter()
        first_reject, target_preds = target.verify(committed, primary)
        t_verify_elapsed = time.perf_counter() - t_v0
        stats["t_overlap"] += t_verify_elapsed

        # --- 5. Collect spec draft ---
        spec_tokens = draft.collect_draft() if can_speculate else None

        # --- 6. Decide + commit ---
        correction = target_preds[first_reject]

        if first_reject == num_draft:
            # Full accept of primary
            stats["full_accept_cycles"] += 1
            committed = committed + primary + [correction]
            generated.extend(primary + [correction])

            if can_speculate and next_guess == correction:
                # SPEC HIT: spec_tokens valid for next cycle as-is
                stats["spec_hit"] += 1
                primary = spec_tokens
                # State currently at p_post_primary + num_draft. It reflects
                # the spec having been drafted from next_guess == correction,
                # which IS committed's new last-token-minus-1 situation.
                # committed[:-1] after this commit = old_committed + primary =
                # len p_committed + 1 + num_draft tokens at positions 0..p_committed+num_draft.
                # So new p_committed should be p_committed + num_draft + 1.
                # State.pos is at p_committed + num_draft + 1 + num_draft after
                # spec (one extra from step d_{K-1}, K for spec draft).
                # Actually state.pos = p_post_primary + num_draft = p_committed + 1 + num_draft + num_draft.
                # But new p_committed wants state.pos = len(new_committed) - 1 = p_committed + num_draft + 1.
                # We'd need to restore S_post_primary (pos = p_committed + num_draft + 1) and THEN run the
                # spec draft — but that's exactly what happened! State IS at p_committed + num_draft + 1 + num_draft.
                # That's "the spec draft has been computed from the next-cycle's committed[:-1] state".
                # For the NEXT iteration, the primary is spec_tokens. That iteration's first step is
                # step_sync(spec_tokens[-1]) which expects state at p_post_primary + num_draft = new_p_committed + num_draft. ✓
                # But we also need S_committed to reflect the NEW committed[:-1].
                # New committed[:-1] = old + primary (ends at d_{K-1}). State at p_post_primary reflects
                # this (cache has old + prompt[-1] + d_0..d_{K-1}). So S_committed should be S_post_primary.
                S_committed = S_post_primary
                p_committed = p_post_primary
                # No need to submit next primary — it's already `primary`
            else:
                # Full accept but spec miss (or no spec)
                if can_speculate:
                    stats["spec_miss_full"] += 1
                # Restore to S_post_primary (pos = p_committed + num_draft + 1)
                # which reflects old committed + primary = new_committed[:-1]
                draft.restore(S_post_primary, p_post_primary)
                S_committed = S_post_primary
                p_committed = p_post_primary
                primary = None
                # Submit next primary draft from the correct start (correction)
                draft.submit_draft(committed[-1], num_draft)
        else:
            # Partial reject
            if can_speculate:
                stats["spec_miss_partial"] += 1
            accepted = primary[:first_reject]
            # Restore to S_committed (pos = p_committed, the OLD committed)
            draft.restore(S_committed, p_committed)
            # Advance through old committed[-1] + accepted (one extra step + j)
            draft.step_sync(committed[-1])
            for t in accepted:
                draft.step_sync(t)
            # Now state at p_committed + 1 + first_reject
            committed = committed + accepted + [correction]
            generated.extend(accepted + [correction])
            S_committed = draft.snapshot()
            p_committed = draft.pos
            primary = None
            draft.submit_draft(committed[-1], num_draft)

        stats["cycles"] += 1
        stats["draft_generated"] += num_draft
        stats["draft_accepted"] += first_reject

        if verbose:
            if first_reject == num_draft:
                if can_speculate and next_guess == correction:
                    tag = "FULL+HIT "
                else:
                    tag = "FULL+miss"
            else:
                tag = "partial  "
            print(f"  [cycle {stats['cycles']:2d}] {tag} accept={first_reject}/{num_draft} "
                  f"committed_len={len(committed)}")

    # Drain any remaining submitted draft
    if primary is None and draft._request_event.is_set() or draft._done_event.is_set():
        try:
            draft.collect_draft()
        except Exception:
            pass

    stats["t_total"] = time.perf_counter() - t_start
    stats["t_step_total"] = draft.t_step_total
    stats["t_snapshot_total"] = draft.t_snapshot_total
    stats["t_restore_total"] = draft.t_restore_total
    stats["n_draft_steps"] = draft.n_steps
    stats["t_verify_total"] = target.t_verify_total
    stats["n_verifies"] = target.n_verifies
    return generated, stats


def baseline_target_only(target: MLXTarget, prompt_ids: List[int], max_new: int):
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    sampler = make_sampler(temp=0.0)
    tokens = []
    t0 = time.perf_counter()
    prompt_str = target.tokenizer.decode(prompt_ids)
    for resp in stream_generate(target.model, target.tokenizer, prompt_str,
                                 max_tokens=max_new, sampler=sampler):
        tokens.append(resp.token)
    return tokens, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-3-12b-it-bf16")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--num-draft", type=int, default=12)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    target = MLXTarget(args.target)
    draft = ANEDraftWorker()

    prompt_ids = target.tokenizer.encode(args.prompt, add_special_tokens=True)
    if not args.quiet:
        print(f"[main] prompt: {args.prompt!r}  tokens: {prompt_ids}")

    if not args.skip_baseline:
        print(f"\n=== baseline: target-only ===", flush=True)
        base_tokens, base_elapsed = baseline_target_only(target, prompt_ids, args.max_new_tokens)
        print(f"time: {base_elapsed*1000:.0f} ms  ({len(base_tokens)/base_elapsed:.2f} tok/s)")

    print(f"\n=== Phase B.2 concurrent (num_draft={args.num_draft}) ===", flush=True)
    gen, stats = run_sd_concurrent(draft, target, prompt_ids, args.max_new_tokens, args.num_draft,
                                    verbose=not args.quiet)

    text = target.tokenizer.decode(gen)
    if not args.quiet:
        print(f"\ntext: {text!r}")
    print(f"time: {stats['t_total']*1000:.0f} ms  ({len(gen)/stats['t_total']:.2f} tok/s)")
    print(f"  cycles: {stats['cycles']}")
    print(f"  full-accept: {stats['full_accept_cycles']}/{stats['cycles']}")
    print(f"  spec hit/miss-full/miss-partial: "
          f"{stats['spec_hit']}/{stats['spec_miss_full']}/{stats['spec_miss_partial']}")
    print(f"  accept: {stats['draft_accepted']}/{stats['draft_generated']} = "
          f"{stats['draft_accepted']/max(stats['draft_generated'],1):.1%}")
    print(f"  draft step: {stats['t_step_total']*1000:.0f} ms  n_steps={stats['n_draft_steps']}")
    print(f"  snapshot: {stats['t_snapshot_total']*1000:.0f} ms")
    print(f"  restore: {stats['t_restore_total']*1000:.0f} ms")
    print(f"  verify: {stats['t_verify_total']*1000:.0f} ms "
          f"({stats['t_verify_total']/stats['t_total']*100:.0f}%)")

    draft.shutdown()

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
        speedup = base_elapsed / stats['t_total']
        print(f"\nSpeedup: {speedup:.2f}×")


if __name__ == "__main__":
    main()
