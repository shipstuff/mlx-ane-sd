"""Smoke test: load ANEMLL Gemma-3-270M, prefill a prompt, generate a few tokens.

Before we build the SD loop, prove the ANE model actually produces coherent
tokens from a simple Python wrapper. Validates:
  - Load the monolithic .mlmodelc with CompiledMLModel
  - make_state() works (stateful KV cache baked in)
  - prefill function runs the prompt
  - infer function generates single-token decodes
  - argmax_idx / argmax_val outputs decode to sensible tokens

Expected output: coherent continuation of "The capital of France is"
(e.g., "Paris" or similar). Doesn't need to match any specific output —
we just need tokens that aren't garbage.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

import numpy as np
import coremltools as ct
from transformers import AutoTokenizer


# Model config from meta.yaml
MODEL_DIR = Path("/Users/carl/models/gemma3-270m-ane")
MODEL_FILE = MODEL_DIR / "gemma3_monolithic_full_lut6.mlmodelc"
CONTEXT_LENGTH = 512
BATCH_SIZE = 64
VOCAB_SIZE = 262144
SLIDING_WINDOW = 512
NUM_LMHEAD_CHUNKS = 16
CHUNK_SIZE = VOCAB_SIZE // NUM_LMHEAD_CHUNKS  # 16384 each


def make_causal_mask(length: int) -> np.ndarray:
    """[1, 1, length, length] causal mask with -inf above the diagonal."""
    mask = np.full((length, length), -np.inf, dtype=np.float16)
    for i in range(length):
        mask[i, : i + 1] = 0.0
    return mask[None, None, :, :]


def argmax_output_to_token(output: dict) -> int:
    """Convert split-16 argmax_idx/argmax_val → global token id."""
    argmax_idx = output["argmax_idx"].flatten()  # [16] local indices per chunk
    argmax_val = output["argmax_val"].flatten()  # [16] max values per chunk
    best_chunk = int(np.argmax(argmax_val))
    local_idx = int(argmax_idx[best_chunk])
    return local_idx + best_chunk * CHUNK_SIZE


def prefill_ane(model_infer, state, token_ids: List[int], causal_mask: np.ndarray):
    """Feed prompt tokens one at a time through the infer function.

    The prefill function expects batch=64; for simple single-prompt cases we
    just walk the prompt through infer to populate the cache position-by-
    position. chat.py calls this "single_token_mode".
    """
    for pos in range(len(token_ids)):
        tok = np.array([[token_ids[pos]]], dtype=np.int32)
        pos_ids = np.array([pos], dtype=np.int32)
        single_mask = causal_mask[:, :, pos : pos + 1, :].astype(np.float16)
        inputs = {
            "input_ids": tok,
            "position_ids": pos_ids,
            "causal_mask": single_mask,
            "current_pos": pos_ids,
        }
        model_infer.predict(inputs, state)


def decode_one(model_infer, state, last_token: int, pos: int, causal_mask: np.ndarray) -> int:
    tok = np.array([[last_token]], dtype=np.int32)
    pos_ids = np.array([pos], dtype=np.int32)
    single_mask = causal_mask[:, :, pos : pos + 1, :].astype(np.float16)
    inputs = {
        "input_ids": tok,
        "position_ids": pos_ids,
        "causal_mask": single_mask,
        "current_pos": pos_ids,
    }
    output = model_infer.predict(inputs, state)
    return argmax_output_to_token(output)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--max-new-tokens", type=int, default=20)
    args = ap.parse_args()

    print(f"[ane-smoke] loading model from {MODEL_FILE}...", flush=True)
    t0 = time.perf_counter()

    # Use infer function for single-token mode (prefill expects batch=64)
    model_infer = ct.models.CompiledMLModel(
        str(MODEL_FILE), ct.ComputeUnit.CPU_AND_NE, function_name="infer"
    )
    print(f"[ane-smoke] loaded in {time.perf_counter() - t0:.2f}s", flush=True)

    # The monolithic model carries state (KV cache). One state object is shared
    # across infer/prefill function calls.
    state = model_infer.make_state()

    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    prompt_ids = tokenizer.encode(args.prompt, add_special_tokens=True)
    print(f"[ane-smoke] prompt: {args.prompt!r}")
    print(f"[ane-smoke] prompt token ids ({len(prompt_ids)}): {prompt_ids}")
    print(f"[ane-smoke] decoded: {tokenizer.decode(prompt_ids)!r}")

    # Build causal mask for the full context
    causal_mask = make_causal_mask(CONTEXT_LENGTH)

    # Prefill prompt through positions 0..n-2 (skip last, it's fed in decode)
    print("[ane-smoke] prefilling prompt (positions 0..n-2)...", flush=True)
    t0 = time.perf_counter()
    if len(prompt_ids) > 1:
        prefill_ane(model_infer, state, prompt_ids[:-1], causal_mask)
    t_prefill = time.perf_counter() - t0
    print(f"[ane-smoke] prefill {len(prompt_ids) - 1} tokens in {t_prefill * 1000:.1f} ms "
          f"({(len(prompt_ids) - 1) / max(t_prefill, 1e-6):.1f} tok/s)")

    # Decode loop: feed last prompt token, predict first new token, etc.
    print(f"[ane-smoke] generating {args.max_new_tokens} tokens...", flush=True)
    generated = []
    current_token = prompt_ids[-1]
    current_pos = len(prompt_ids) - 1
    per_tok = []
    for step in range(args.max_new_tokens):
        if current_pos >= CONTEXT_LENGTH - 1:
            break
        t0 = time.perf_counter()
        next_tok = decode_one(model_infer, state, current_token, current_pos, causal_mask)
        per_tok.append(time.perf_counter() - t0)
        generated.append(next_tok)
        current_token = next_tok
        current_pos += 1

    text = tokenizer.decode(generated)
    print()
    print("=" * 60)
    print(f"Generated: {text!r}")
    print(f"Token ids: {generated}")
    print("=" * 60)
    if per_tok:
        import statistics
        median_ms = 1000 * statistics.median(per_tok)
        mean_ms = 1000 * statistics.mean(per_tok)
        print(f"Decode: {len(generated)} tokens, median {median_ms:.1f} ms/tok "
              f"({1000 / median_ms:.1f} tok/s)")


if __name__ == "__main__":
    main()
