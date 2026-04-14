"""Phase B.1: sequential heterogeneous SD with smart state handling.

Improvements over Phase A:
  1. Snapshot the ANE draft state before each speculative draft_k
  2. On partial rejection, restore snapshot and advance through accepted
     tokens + correction — no full re-prefill
  3. On full acceptance, commit the speculative state (advance through the
     unprocessed last token) and keep rolling

Still sequential — no concurrent draft+verify yet. That's Phase B.2.

Expected speedup over Phase A: eliminate the O(N) per-cycle re-prefill cost
that ate 35% of time. Draft overhead should drop to near-zero (just the K
speculative forward passes per cycle) or nearly so.

Env: MLX venv (has both mlx-lm and coremltools 9.0).
"""
from __future__ import annotations

import argparse
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


class ANEDraft:
    """Manages the ANE draft model with committed-state snapshotting.

    Invariant:
      - `pos` = number of tokens the ANE cache has processed (cache holds
        KV at positions 0..pos-1)
      - `committed_snapshot` is the last saved state representing the
        authoritative prefix (matches `committed[:-1]` in the SD loop)
      - After each SD cycle, we restore to committed_snapshot, advance
        through the newly-accepted tokens, and re-snapshot
    """

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
        self.committed_snapshot: Optional[Dict[str, np.ndarray]] = None
        self.committed_pos: int = 0

        # Stats
        self.t_step_total = 0.0
        self.t_snapshot_total = 0.0
        self.t_restore_total = 0.0
        self.n_steps = 0

    def reset(self):
        self.state = self.model.make_state()
        self.pos = 0

    def step(self, token_id: int) -> int:
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

    def commit_current(self):
        """Take a snapshot of the current state as the new committed state."""
        self.committed_snapshot = self.snapshot()
        self.committed_pos = self.pos

    def restore_to_committed(self):
        assert self.committed_snapshot is not None, "no committed snapshot yet"
        self.restore(self.committed_snapshot, self.committed_pos)

    def prefill_initial(self, tokens: List[int]):
        """First-time prefill from fresh state. Feeds all tokens through step()."""
        self.reset()
        for t in tokens:
            self.step(t)

    def draft_k(self, start_token: int, k: int) -> List[int]:
        tokens = []
        current = start_token
        for _ in range(k):
            nxt = self.step(current)
            tokens.append(nxt)
            current = nxt
        return tokens


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


def run_sd(draft: ANEDraft, target: MLXTarget, prompt_ids: List[int],
           max_new: int, num_draft: int, verbose: bool = True) -> Tuple[List[int], dict]:
    # --- Initial prefill ---
    # Feed prompt[:-1], then commit the state. Invariant: state matches
    # committed[:-1], draft.pos == len(committed) - 1.
    committed = list(prompt_ids)
    t0 = time.perf_counter()
    draft.prefill_initial(committed[:-1])
    draft.commit_current()
    t_initial_prefill = time.perf_counter() - t0

    generated: List[int] = []
    stats = {
        "cycles": 0, "draft_generated": 0, "draft_accepted": 0,
        "full_accept_cycles": 0, "t_total": 0.0, "t_initial_prefill": t_initial_prefill,
    }
    t_start = time.perf_counter()

    while len(generated) < max_new:
        if len(committed) >= CONTEXT_LENGTH - num_draft - 2:
            if verbose:
                print("[sd] ctx limit, stop")
            break

        # --- Draft phase ---
        # State is at committed[:-1] (pos = len(committed) - 1, committed_snapshot set)
        draft_tokens = draft.draft_k(committed[-1], num_draft)

        # --- Verify phase ---
        first_reject, target_preds = target.verify(committed, draft_tokens)

        # --- Commit phase ---
        accepted = draft_tokens[:first_reject]
        correction = target_preds[first_reject]

        if first_reject == num_draft:
            # Full accept: state has all K draft tokens written, but we ALSO
            # need to write the last draft token's KV. After draft_k:
            #   - step(committed[-1]) wrote committed[-1] at old_pos → pos+1
            #   - step(d0) wrote d0 → pos+2
            #   - ... step(d_{K-2}) wrote d_{K-2} → pos+K (= new_pos)
            #   - d_{K-1} was RETURNED but not written
            # So state has tokens up to d_{K-2} in cache. To commit the full
            # acceptance, we need d_{K-1} in cache.
            # New committed = old + drafts + [bonus]. new_len - 1 = old_len + K.
            # We need cache at positions 0..old_len+K-1, i.e., through d_{K-1}.
            # Do ONE more step feeding d_{K-1} to write it.
            draft.step(draft_tokens[-1])  # write d_{K-1} into cache, advance pos
            # Now pos = old_len + K = new_len - 1. Perfect for next cycle
            # (where committed[-1] will be the bonus token and we'll feed it
            # as the start of next draft_k).
            committed = committed + draft_tokens + [correction]
            generated.extend(draft_tokens + [correction])
            stats["full_accept_cycles"] += 1
        else:
            # Partial reject. State has d_0..d_{j-1} in cache at positions
            # old_len..old_len+j-1 (correctly), plus also d_j..d_{K-1} at
            # positions old_len+j..old_len+K-1 (need to discard).
            # Additionally, committed[-1] was written at position old_len - 1
            # during draft_k's first step — that one we DO want to keep.
            #
            # Cleanest: restore from committed_snapshot, then step through
            # committed[-1] + accepted drafts. Skip the correction because
            # it'll be committed[-1] of the NEW committed and fed in next cycle.
            draft.restore_to_committed()
            # Advance: feed committed[-1] (old last), then each accepted draft
            draft.step(committed[-1])
            for t in accepted:
                draft.step(t)
            # Now pos = old_committed_pos + 1 + j = (old_len - 1) + 1 + j
            #        = old_len + j = new_len - 1. Correct for next cycle.
            committed = committed + accepted + [correction]
            generated.extend(accepted + [correction])

        # Snapshot the new committed state
        draft.commit_current()

        stats["cycles"] += 1
        stats["draft_generated"] += num_draft
        stats["draft_accepted"] += first_reject

        if verbose:
            tag = "FULL" if first_reject == num_draft else "part"
            print(f"  [cycle {stats['cycles']:2d}] {tag} draft={num_draft} accept={first_reject}/{num_draft} "
                  f"committed_len={len(committed)}")

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
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="mlx-community/gemma-3-12b-it-bf16")
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=50)
    ap.add_argument("--num-draft", type=int, default=8)
    ap.add_argument("--skip-baseline", action="store_true")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    target = MLXTarget(args.target)
    draft = ANEDraft()

    prompt_ids = target.tokenizer.encode(args.prompt, add_special_tokens=True)
    if not args.quiet:
        print(f"[main] prompt: {args.prompt!r}  tokens: {prompt_ids}")

    if not args.skip_baseline:
        print(f"\n=== baseline: target-only ===", flush=True)
        base_tokens, base_elapsed = baseline_target_only(target, prompt_ids, args.max_new_tokens)
        print(f"text: {target.tokenizer.decode(base_tokens)!r}")
        print(f"time: {base_elapsed*1000:.0f} ms  ({len(base_tokens)/base_elapsed:.2f} tok/s)")

    print(f"\n=== Phase B.1 heterogeneous SD (num_draft={args.num_draft}) ===", flush=True)
    gen, stats = run_sd(draft, target, prompt_ids, args.max_new_tokens, args.num_draft,
                        verbose=not args.quiet)
    text = target.tokenizer.decode(gen)
    if not args.quiet:
        print(f"\ntext: {text!r}")
    print(f"time: {stats['t_total']*1000:.0f} ms  ({len(gen)/stats['t_total']:.2f} tok/s)")
    print(f"  cycles: {stats['cycles']}  full-accept: {stats['full_accept_cycles']}/{stats['cycles']}")
    print(f"  accept: {stats['draft_accepted']}/{stats['draft_generated']} = "
          f"{stats['draft_accepted']/max(stats['draft_generated'],1):.1%}")
    print(f"  draft step time: {stats['t_step_total']*1000:.0f} ms "
          f"({stats['t_step_total']/stats['t_total']*100:.0f}%)  "
          f"n_steps={stats['n_draft_steps']}")
    print(f"  snapshot time: {stats['t_snapshot_total']*1000:.0f} ms")
    print(f"  restore time: {stats['t_restore_total']*1000:.0f} ms")
    print(f"  verify time: {stats['t_verify_total']*1000:.0f} ms "
          f"({stats['t_verify_total']/stats['t_total']*100:.0f}%)")
    print(f"  initial prefill: {stats['t_initial_prefill']*1000:.0f} ms")

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
