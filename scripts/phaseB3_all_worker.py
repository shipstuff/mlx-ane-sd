"""Phase B.3: minimize main-thread ANE serial work.

Phase B.2 showed concurrency isn't materializing gains because the main
thread does ~10-15 ms of serial ANE work per cycle (step_sync finalize,
snapshot, submit, collect) that blocks target verify from starting.

Fix: move EVERYTHING possible onto the worker thread. Worker does:
    draft_k → finalize d_{K-1} → snapshot-trigger → spec_draft_k
Main thread only does: target verify, commit logic, and decisions.

The snapshot is still done on main thread (read_state / write_state
require direct state access, and we want it off the worker's critical
path so we can overlap it with verify). But the worker doesn't BLOCK on
the snapshot — it just runs all ANE forward passes back to back.

Protocol:
    Cycle N preparation:
        submit_spec_job(start=next_guess from previous cycle, k=K)
          This is 2K forward passes: K for primary + K for spec.
          No wait — we already know what primary is (from previous spec).
          So just K passes for the NEW spec draft.

Reworking: we need to rethink the cycle boundary.

Simplest clean approach — MERGE primary and spec concepts:
    Maintain a pipeline where the worker ALWAYS has a draft job queued.
    Each draft job is K tokens from some starting point.
    We pipeline by pre-committing to a next-start assumption.

Protocol (revised):

    Initialization:
        prefill committed[:-1]
        snapshot S0 at committed state
        submit_draft(committed[-1], K)   # this is cycle 0's primary

    Each cycle:
        primary = collect_draft()        # wait for current primary
        # state.pos = committed_pos + K (d_0..d_{K-1} predicted, d_{K-2} in cache)

        snapshot Spost_primary = snapshot()    # used on miss/restore
        # Actually we need a snapshot reflecting committed[:-1] + committed[-1] + primary
        # currently cache has: 0..committed_pos-1 + committed[-1] + d_0..d_{K-2}
        # That's committed_pos + K positions, needs ONE more for d_{K-1}.
        # Do that finalize BEFORE snapshot? Adds 5ms serial.

    Hmm, same issue.

Let me try a different simpler tweak: snapshot BEFORE submitting primary.
Then on partial reject, restore and advance from that snapshot. Primary
and spec become the same job: one continuous K-token draft from the
speculative start, snapshotted at the right point.

Actually the original issue is: too many MAIN thread operations between
collect_primary and verify. Let me just remove the step_sync finalize
and snapshot BEFORE verify, and do them AFTER verify instead. The spec
draft will be delayed but concurrency of draft+verify still happens for
the PRIMARY (which is a previous cycle's spec).

Revised protocol:
    Init:
        prefill committed[:-1]
        S_committed = snapshot()
        p_committed = pos
        submit(committed[-1], K)     # primary = cycle 0

    Loop:
        primary = collect()           # get current primary, state pos = p_committed + K (cache has d_0..d_{K-2})
        first_reject, target_preds = target.verify(committed, primary)  # concurrent with... nothing for now
        correction = target_preds[first_reject]

        # Handle commit:
        if first_reject == K:
            # Full accept. Step d_{K-1} to finalize (write it + pos++).
            step_sync(primary[-1])    # pos now p_committed + K + 1
            committed = committed + primary + [correction]
            S_committed = snapshot()   # reflects new committed[:-1]
            p_committed = pos
        else:
            # Partial. Restore + advance.
            restore(S_committed, p_committed)
            step_sync(committed[-1])  # advance pos
            for t in primary[:first_reject]: step_sync(t)
            committed = committed + primary[:first_reject] + [correction]
            S_committed = snapshot()
            p_committed = pos

        # Now state reflects committed[:-1]. Submit next primary.
        submit(committed[-1], K)

This is Phase B.1 with threading. No speculation. Gets ~1.0x speedup over
B.1 since we've just moved the draft_k off-thread but it was already
short enough to not block.

The REAL speedup comes from submitting the NEXT primary BEFORE verify
returns. But that requires knowing new committed[-1] which requires
verify. So we have to speculate.

ALTERNATIVE: submit primary for next cycle while current primary's verify
is running, but START the draft from the CURRENT primary's last token
(speculative start). On full accept, spec start == primary[-1] so it's
consistent. On partial reject, we restore state anyway so the draft
continues from a state we'll throw away.

This was Phase B.2. Its issue is that the draft restore + re-draft happens
on the critical path AFTER verify. For partial reject, that's +60 ms
of serial work that wasn't there in B.1.

OK I'll abandon the cross-cycle spec and try: submit primary BEFORE commit
state restore work, so the work overlaps with draft. This might claw back
some time.

Protocol:
    Loop:
        primary = collect()
        target.verify(committed, primary) → first_reject, target_preds
        correction = target_preds[first_reject]

        # Decide new committed first
        if first_reject == K:
            # Full. Speculate: next cycle's start = correction. But correction = spec_bonus prediction ≈ primary[-1]'s next. If we predicted next_guess during an earlier cycle, use that if it matches.
            ...

Nah, I'm going in circles. Let me just:
1. Write the "submit primary while doing restore/advance work" version.
2. Benchmark.
3. Call it done.
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
        self._lock = threading.Lock()  # protects state access
        self._request_event = threading.Event()
        self._done_event = threading.Event()
        self._request = None
        self._result = None
        self._shutdown = False
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()
        self.t_step_total = 0.0
        self.n_steps = 0
        self.t_snapshot_total = 0.0

    def _step_raw(self, tok: int) -> int:
        t0 = time.perf_counter()
        inp = {
            "input_ids": np.array([[tok]], dtype=np.int32),
            "position_ids": np.array([self.pos], dtype=np.int32),
            "causal_mask": self.causal_mask[:, :, self.pos:self.pos+1, :].astype(np.float16),
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
            req = self._request
            self._request = None
            # Acquire lock while running
            with self._lock:
                tokens = []
                current = req["start"]
                for _ in range(req["k"]):
                    tokens.append(self._step_raw(current))
                    current = tokens[-1]
                # Optional extra: finalize last token (write it to cache)
                if req.get("finalize", False):
                    self._step_raw(tokens[-1])
            self._result = tokens
            self._done_event.set()

    def submit_draft(self, start: int, k: int, finalize: bool = False):
        self._request = {"start": start, "k": k, "finalize": finalize}
        self._request_event.set()

    def collect_draft(self) -> List[int]:
        self._done_event.wait()
        self._done_event.clear()
        r = self._result
        self._result = None
        return r

    def step_sync(self, tok: int) -> int:
        """Wait for worker, then step on main thread."""
        with self._lock:
            return self._step_raw(tok)

    def snapshot(self):
        with self._lock:
            t0 = time.perf_counter()
            s = {
                "global": self.state.read_state(STATE_GLOBAL).copy(),
                "local": self.state.read_state(STATE_LOCAL).copy(),
                "pos": self.pos,
            }
            self.t_snapshot_total += time.perf_counter() - t0
            return s

    def restore(self, snap):
        with self._lock:
            self.state.write_state(STATE_GLOBAL, snap["global"])
            self.state.write_state(STATE_LOCAL, snap["local"])
            self.pos = snap["pos"]

    def reset_state(self):
        with self._lock:
            self.state = self.model.make_state()
            self.pos = 0

    def prefill_initial(self, tokens: List[int]):
        self.reset_state()
        for t in tokens:
            self.step_sync(t)

    def shutdown(self):
        self._shutdown = True
        self._request_event.set()
        self._worker.join(timeout=2.0)


class MLXTarget:
    def __init__(self, model_path: str):
        from mlx_lm import load
        print(f"[target] loading {model_path}...", flush=True)
        t0 = time.perf_counter()
        self.model, self.tokenizer = load(model_path)
        print(f"[target] loaded in {time.perf_counter() - t0:.1f}s", flush=True)
        self.t_verify = 0.0

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
        self.t_verify += time.perf_counter() - t0
        return first_reject, preds


def run(draft: ANEDraftWorker, target: MLXTarget, prompt_ids: List[int],
        max_new: int, num_draft: int, verbose: bool = True):
    """Main loop. Focuses on keeping verify starts as early as possible."""
    committed = list(prompt_ids)
    draft.prefill_initial(committed[:-1])
    S_committed = draft.snapshot()

    # Submit cycle 0 draft (with finalize so cache ends at d_{K-1})
    draft.submit_draft(committed[-1], num_draft, finalize=True)

    generated = []
    stats = {"cycles": 0, "draft_gen": 0, "draft_acc": 0, "full": 0,
             "t_total": 0.0, "t_verify": 0.0}
    t_start = time.perf_counter()

    while len(generated) < max_new:
        if len(committed) >= CONTEXT_LENGTH - num_draft - 4:
            if verbose:
                print("[sd] ctx limit, stop")
            break

        # Collect draft (primary for this cycle)
        primary = draft.collect_draft()
        # State is at p_committed + K + 1 (finalized): cache has committed[-1] + d_0..d_{K-1}.

        # Snapshot post-primary BEFORE verify — this is needed for restore on
        # full-accept to advance without re-prefill
        S_post = draft.snapshot()

        # Kick off next cycle's SPECULATIVE draft. Start = primary[-1]'s
        # prediction (we need to know it).
        # Actually we don't have that precomputed. We have primary = [d_0..d_{K-1}]
        # and the state is finalized (d_{K-1} in cache). The next ANE step
        # would predict the token AFTER d_{K-1}.
        #
        # For concurrency to work, we need to submit draft NOW (before verify).
        # But submitting requires knowing the start token. Options:
        #   a) Do one extra step_sync to get next_guess, then submit. Adds 5ms.
        #   b) Just don't speculate — submit next primary AFTER verify.
        #
        # Let's try (b) — simpler. See if concurrency gain comes from just
        # overlap within the current cycle (draft already done when verify starts).

        # Target verify
        first_reject, preds = target.verify(committed, primary)
        correction = preds[first_reject]

        if first_reject == num_draft:
            # Full accept — state is correct (S_post reflects new_committed[:-1])
            committed += primary + [correction]
            generated += primary + [correction]
            S_committed = S_post
            stats["full"] += 1
            # Submit next draft starting from correction (the new committed[-1])
            # State is already at p + K + 1. Next draft should advance from there.
            draft.submit_draft(correction, num_draft, finalize=True)
        else:
            # Partial: restore to S_committed, advance through accepted + correction
            draft.restore(S_committed)
            draft.step_sync(committed[-1])
            for t in primary[:first_reject]:
                draft.step_sync(t)
            # state pos = old_p + 1 + first_reject
            committed += primary[:first_reject] + [correction]
            generated += primary[:first_reject] + [correction]
            S_committed = draft.snapshot()
            # Submit next primary draft
            draft.submit_draft(correction, num_draft, finalize=True)

        stats["cycles"] += 1
        stats["draft_gen"] += num_draft
        stats["draft_acc"] += first_reject

        if verbose:
            tag = "FULL" if first_reject == num_draft else "part"
            print(f"  [cycle {stats['cycles']:2d}] {tag} accept={first_reject}/{num_draft} "
                  f"committed_len={len(committed)}")

    stats["t_total"] = time.perf_counter() - t_start
    stats["t_verify"] = target.t_verify
    stats["t_step"] = draft.t_step_total
    stats["t_snapshot"] = draft.t_snapshot_total
    stats["n_steps"] = draft.n_steps
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

    if not args.skip_baseline:
        print(f"\n=== baseline ===", flush=True)
        base_tokens, base_elapsed = baseline_target_only(target, prompt_ids, args.max_new_tokens)
        print(f"{base_elapsed*1000:.0f} ms  ({len(base_tokens)/base_elapsed:.2f} tok/s)")

    print(f"\n=== Phase B.3 (num_draft={args.num_draft}) ===", flush=True)
    gen, stats = run(draft, target, prompt_ids, args.max_new_tokens, args.num_draft,
                     verbose=not args.quiet)
    text = target.tokenizer.decode(gen)
    if not args.quiet:
        print(f"\ntext: {text!r}")
    print(f"time: {stats['t_total']*1000:.0f} ms  ({len(gen)/stats['t_total']:.2f} tok/s)")
    print(f"  cycles: {stats['cycles']}  full-accept: {stats['full']}/{stats['cycles']}")
    print(f"  accept: {stats['draft_acc']}/{stats['draft_gen']} = "
          f"{stats['draft_acc']/max(stats['draft_gen'],1):.1%}")
    print(f"  draft step: {stats['t_step']*1000:.0f} ms  n_steps={stats['n_steps']}")
    print(f"  snapshot: {stats['t_snapshot']*1000:.0f} ms")
    print(f"  verify: {stats['t_verify']*1000:.0f} ms "
          f"({stats['t_verify']/stats['t_total']*100:.0f}%)")

    draft.shutdown()

    if not args.skip_baseline:
        ml = min(len(base_tokens), len(gen))
        match = sum(1 for i in range(ml) if base_tokens[i] == gen[i] and i == sum(1 for j in range(i+1) if base_tokens[j] == gen[j]) - 1)
        m = 0
        for i in range(ml):
            if base_tokens[i] == gen[i]:
                m += 1
            else:
                break
        print(f"\nmatched: {m}/{ml} tokens")
        print(f"Speedup: {base_elapsed / stats['t_total']:.2f}×")


if __name__ == "__main__":
    main()
