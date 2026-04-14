"""Phase F.1 step 3: convert DFlash draft PyTorch → CoreML via coremltools.

Fixed shapes in the export:
  B = 1, L = block_size (16), S = block_size (16 — padded/truncated by caller)
  H = hidden_size (2560), concat_dim = num_target_layers * H (5 * 2560 = 12800)

Inputs:
  noise_embedding : (1, 16, 2560)           float16
  target_hidden   : (1, 16, 12800)          float16
  cos_q, sin_q    : (16, head_dim=128)      float16
  cos_k, sin_k    : (32, head_dim=128)      float16  (S+L = 32 positions)

Output:
  hidden          : (1, 16, 2560)           float16

LM head is applied externally by the target (shared weights).

Usage:
  python dflash_coreml_convert.py --output /path/to/dflash_ane.mlpackage
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
from dflash_ane import (
    DFlashConfig, DFlashDraftModelANE, copy_weights, set_rmsnorm_mode,
    build_rope_for_offset, _rope_inv_freq,
)

from huggingface_hub import snapshot_download


def load_ane_model_with_weights(draft_id: str = "z-lab/Qwen3-4B-DFlash-b16"):
    draft_path = snapshot_download(draft_id)
    torch_draft = load_dflash_from_hf(draft_path)
    ane_draft = DFlashDraftModelANE(torch_draft.config).to(torch.bfloat16).eval()
    loaded, _ = copy_weights(torch_draft, ane_draft)
    print(f"[load] copied {loaded} weights into ANE variant")
    # coremltools requires float32 for tracing (bf16 isn't supported in all pipelines)
    ane_draft = ane_draft.to(torch.float32).eval()
    return ane_draft, torch_draft.config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True,
                    help="output .mlpackage path")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--compute-units", choices=["ALL", "CPU_AND_NE", "CPU_AND_GPU", "CPU_ONLY"],
                    default="CPU_AND_NE")
    args = ap.parse_args()

    model, config = load_ane_model_with_weights()

    set_rmsnorm_mode("ane")  # use mean-subtract + layer_norm form

    B = 1
    L = args.block_size
    S = args.block_size  # fixed ctx length
    H = config.hidden_size
    concat_dim = len(config.target_layer_ids) * H
    Dh = config.head_dim

    # Build example inputs
    noise_emb = torch.randn(B, L, H, dtype=torch.float32) * 0.01
    target_hidden = torch.randn(B, S, concat_dim, dtype=torch.float32) * 0.01
    cos_q, sin_q, cos_k, sin_k = build_rope_for_offset(config, offset=0,
                                                        ctx_len=S, block_size=L)
    cos_q, sin_q = cos_q.to(torch.float32), sin_q.to(torch.float32)
    cos_k, sin_k = cos_k.to(torch.float32), sin_k.to(torch.float32)

    print(f"[trace] noise_emb={noise_emb.shape}, target_hidden={target_hidden.shape}")
    print(f"[trace] cos_q={cos_q.shape}, cos_k={cos_k.shape}")

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k),
            strict=False,
        )
    print(f"[trace] ok")

    # Verify traced model produces same output
    with torch.no_grad():
        ref = model(noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k)
        out = traced(noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k)
    diff = (ref - out).abs().max().item()
    print(f"[trace] traced vs eager max_abs_diff: {diff:.6f}")
    assert diff < 1e-4, f"trace diverged from eager ({diff})"

    # Convert to CoreML
    cu_map = {
        "ALL": ct.ComputeUnit.ALL,
        "CPU_AND_NE": ct.ComputeUnit.CPU_AND_NE,
        "CPU_AND_GPU": ct.ComputeUnit.CPU_AND_GPU,
        "CPU_ONLY": ct.ComputeUnit.CPU_ONLY,
    }
    inputs = [
        ct.TensorType(name="noise_embedding", shape=(B, L, H), dtype=np.float16),
        ct.TensorType(name="target_hidden", shape=(B, S, concat_dim), dtype=np.float16),
        ct.TensorType(name="cos_q", shape=(L, Dh), dtype=np.float16),
        ct.TensorType(name="sin_q", shape=(L, Dh), dtype=np.float16),
        ct.TensorType(name="cos_k", shape=(S + L, Dh), dtype=np.float16),
        ct.TensorType(name="sin_k", shape=(S + L, Dh), dtype=np.float16),
    ]
    outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
    ]

    print(f"[convert] ct.convert(...) compute_units={args.compute_units}")
    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        compute_units=cu_map[args.compute_units],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS14,
        convert_to="mlprogram",
    )
    print(f"[convert] done in {time.perf_counter()-t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(args.output))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
