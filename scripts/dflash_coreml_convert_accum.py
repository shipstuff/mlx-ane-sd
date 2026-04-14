"""Convert the accumulating-cache DFlash to CoreML."""
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
from dflash_ane_accumcache import (
    DFlashDraftModelAccum, copy_weights_to_accum, set_rmsnorm_mode,
)
from huggingface_hub import snapshot_download


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--state-length", type=int, default=256)
    args = ap.parse_args()

    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    torch_draft = load_dflash_from_hf(draft_path)
    config = torch_draft.config

    BS = config.block_size
    CS = config.block_size
    STATE_LEN = args.state_length
    H = config.hidden_size
    N = config.num_hidden_layers
    Hkv = config.num_key_value_heads
    concat_dim = len(config.target_layer_ids) * H
    Dh = config.head_dim
    T = CS + BS

    model = DFlashDraftModelAccum(config, block_size=BS, ctx_size=CS,
                                    state_length=STATE_LEN).to(torch.float32).eval()
    copy_weights_to_accum(torch_draft, model)
    set_rmsnorm_mode("ane")

    noise_emb = torch.randn(1, BS, H) * 0.01
    target_hidden = torch.randn(1, CS, concat_dim) * 0.01
    cos_q = torch.randn(BS, Dh)
    sin_q = torch.randn(BS, Dh)
    cos_k = torch.randn(T, Dh)
    sin_k = torch.randn(T, Dh)
    cache_K = torch.zeros(N, Hkv, STATE_LEN, Dh)
    cache_V = torch.zeros(N, Hkv, STATE_LEN, Dh)
    causal_mask = torch.zeros(1, 1, BS, STATE_LEN + T, dtype=torch.float32)

    print("[eager]")
    with torch.no_grad():
        h, nk, nv = model(noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
                            cache_K, cache_V, causal_mask)
    print(f"  h={h.shape}, new_K={nk.shape}, new_V={nv.shape}")

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
             cache_K, cache_V, causal_mask),
            strict=False,
        )
    print("[trace] ok")

    inputs = [
        ct.TensorType(name="noise_embedding", shape=(1, BS, H), dtype=np.float16),
        ct.TensorType(name="target_hidden", shape=(1, CS, concat_dim), dtype=np.float16),
        ct.TensorType(name="cos_q", shape=(BS, Dh), dtype=np.float16),
        ct.TensorType(name="sin_q", shape=(BS, Dh), dtype=np.float16),
        ct.TensorType(name="cos_k", shape=(T, Dh), dtype=np.float16),
        ct.TensorType(name="sin_k", shape=(T, Dh), dtype=np.float16),
        ct.TensorType(name="cache_K", shape=(N, Hkv, STATE_LEN, Dh), dtype=np.float16),
        ct.TensorType(name="cache_V", shape=(N, Hkv, STATE_LEN, Dh), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, BS, STATE_LEN + T), dtype=np.float16),
    ]
    outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="new_K", dtype=np.float16),
        ct.TensorType(name="new_V", dtype=np.float16),
    ]

    print("[convert]...")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,  # iOS18 for per_grouped_channel LUT
        convert_to="mlprogram",
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
