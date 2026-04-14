"""Convert DFlashDraftModelANECache to CoreML mlpackage, with KV cache as state.

Fixed shapes:
  B=1, L=16, S=16, STATE_LENGTH=256
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))

from dflash_torch import load_dflash_from_hf
from dflash_ane_cache import (
    DFlashDraftModelANECache, copy_weights_to_cache, set_rmsnorm_mode,
)
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--ctx-size", type=int, default=16)
    args = ap.parse_args()

    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    torch_draft = load_dflash_from_hf(draft_path)
    config = torch_draft.config

    model = DFlashDraftModelANECache(config, block_size=args.block_size,
                                      ctx_size=args.ctx_size,
                                      state_length=args.state_length)
    copy_weights_to_cache(torch_draft, model)
    model = model.to(torch.float32).eval()
    set_rmsnorm_mode("ane")

    # Build example inputs
    B = 1
    L = args.block_size
    S = args.ctx_size
    H = config.hidden_size
    concat_dim = len(config.target_layer_ids) * H
    Dh = config.head_dim
    STATE_LEN = args.state_length

    noise_emb = torch.randn(B, L, H) * 0.01
    target_hidden = torch.randn(B, S, concat_dim) * 0.01
    cos_q = torch.randn(L, Dh)
    sin_q = torch.randn(L, Dh)
    cos_k = torch.randn(S + L, Dh)
    sin_k = torch.randn(S + L, Dh)
    current_pos = torch.tensor(0, dtype=torch.int32)
    causal_mask = torch.zeros(1, 1, L, STATE_LEN, dtype=torch.float32)

    print(f"[trace] noise_emb={noise_emb.shape}, ctx={target_hidden.shape}")
    with torch.no_grad():
        # First eager run to catch obvious bugs
        out = model(noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
                    current_pos, causal_mask)
        print(f"[eager] ok, out shape={out.shape}")

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
             current_pos, causal_mask),
            strict=False,
        )
    print("[trace] ok")

    # Declare the state
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=(2 * config.num_hidden_layers, config.num_key_value_heads,
                       STATE_LEN, Dh),
                dtype=np.float16,
            ),
            name="kv_cache",
        ),
    ]
    inputs = [
        ct.TensorType(name="noise_embedding", shape=(B, L, H), dtype=np.float16),
        ct.TensorType(name="target_hidden", shape=(B, S, concat_dim), dtype=np.float16),
        ct.TensorType(name="cos_q", shape=(L, Dh), dtype=np.float16),
        ct.TensorType(name="sin_q", shape=(L, Dh), dtype=np.float16),
        ct.TensorType(name="cos_k", shape=(S + L, Dh), dtype=np.float16),
        ct.TensorType(name="sin_k", shape=(S + L, Dh), dtype=np.float16),
        ct.TensorType(name="current_pos", shape=(1,), dtype=np.int32),
        ct.TensorType(name="causal_mask", shape=(1, 1, L, STATE_LEN), dtype=np.float16),
    ]
    outputs = [ct.TensorType(name="hidden", dtype=np.float16)]

    print("[convert]...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        states=states,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
