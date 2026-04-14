"""Phase F.1 step 4: validate the compiled CoreML DFlash model.

Checks:
1. Model loads via coremltools.
2. Inference produces output of expected shape.
3. Numerical parity vs PyTorch (with the same ANE-mode weights).
4. Profiling: where does it run (ANE vs GPU vs CPU)?
5. Wall-clock latency on ANE.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, "/tmp/dflash")
sys.path.insert(0, str(Path(__file__).parent))

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models.cache import make_prompt_cache
from huggingface_hub import snapshot_download

from dflash_torch import load_dflash_from_hf
from dflash_ane import DFlashDraftModelANE, copy_weights, set_rmsnorm_mode, build_rope_for_offset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mlpackage", type=Path, default="/tmp/dflash_ane.mlpackage")
    args = ap.parse_args()

    print(f"[load] {args.mlpackage}")
    t0 = time.perf_counter()
    mlmodel = ct.models.MLModel(str(args.mlpackage),
                                 compute_units=ct.ComputeUnit.CPU_AND_NE)
    print(f"[load] {time.perf_counter()-t0:.2f}s")

    # Build realistic inputs — same as parity test
    print("[load] target + draft for input prep...")
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    torch_draft = load_dflash_from_hf(draft_path)

    from dflash.model_mlx import _patch_model
    prompt = "The capital of France is Paris, which is known for"
    prompt_ids = tok.encode(prompt, add_special_tokens=True)
    config = torch_draft.config
    _patch_model(target, config.target_layer_ids)
    prompt_arr = mx.array(prompt_ids)[None]
    target_cache = make_prompt_cache(target)
    with mx.stream(mx.default_stream(mx.default_device())):
        _ = target(prompt_arr, target_cache)
        hidden = mx.concatenate(target._hidden_states, axis=-1)
    mx.eval(hidden)

    block_size = config.block_size
    # Pad/truncate ctx to block_size (16)
    if hidden.shape[1] < block_size:
        pad = mx.zeros((1, block_size - hidden.shape[1], hidden.shape[2]), dtype=hidden.dtype)
        ctx = mx.concatenate([hidden, pad], axis=1)
    else:
        ctx = hidden[:, -block_size:, :]
    assert ctx.shape[1] == block_size
    mask_id = config.mask_token_id
    block_tokens = [int(prompt_ids[-1])] + [mask_id] * (block_size - 1)
    block = mx.array([block_tokens])
    embed = target.model.embed_tokens
    noise_emb = embed(block)
    mx.eval(noise_emb)

    cos_q, sin_q, cos_k, sin_k = build_rope_for_offset(config, offset=0,
                                                        ctx_len=block_size, block_size=block_size)

    # Prepare numpy inputs (fp16 per CoreML spec)
    def np16(a):
        return np.asarray(a).astype(np.float16)

    inputs = {
        "noise_embedding": np.asarray(noise_emb.astype(mx.float32)).astype(np.float16),
        "target_hidden":   np.asarray(ctx.astype(mx.float32)).astype(np.float16),
        "cos_q": cos_q.numpy().astype(np.float16),
        "sin_q": sin_q.numpy().astype(np.float16),
        "cos_k": cos_k.numpy().astype(np.float16),
        "sin_k": sin_k.numpy().astype(np.float16),
    }
    for k, v in inputs.items():
        print(f"  {k}: {v.shape} {v.dtype}")

    # ---- CoreML inference ----
    print("\n[run] CoreML inference (CPU_AND_NE)...")
    # Warmup
    _ = mlmodel.predict(inputs)
    # Timed
    N = 10
    t0 = time.perf_counter()
    for _ in range(N):
        out = mlmodel.predict(inputs)
    t = (time.perf_counter() - t0) / N
    print(f"  mean latency: {t*1000:.2f}ms over {N} runs")
    hidden_coreml = out["hidden"]  # (1, 16, 2560)
    print(f"  output shape: {hidden_coreml.shape}, dtype {hidden_coreml.dtype}")

    # ---- PyTorch ANE-mode reference for the same inputs ----
    print("\n[run] PyTorch ANE-mode reference...")
    ane_draft = DFlashDraftModelANE(config, block_size=block_size, ctx_size=block_size)
    ane_draft = ane_draft.to(torch.float32).eval()
    copy_weights(torch_draft, ane_draft)
    set_rmsnorm_mode("ane")

    noise_t = torch.from_numpy(inputs["noise_embedding"].astype(np.float32))
    ctx_t = torch.from_numpy(inputs["target_hidden"].astype(np.float32))
    with torch.no_grad():
        ref = ane_draft(noise_t, ctx_t,
                         torch.from_numpy(inputs["cos_q"].astype(np.float32)),
                         torch.from_numpy(inputs["sin_q"].astype(np.float32)),
                         torch.from_numpy(inputs["cos_k"].astype(np.float32)),
                         torch.from_numpy(inputs["sin_k"].astype(np.float32)))
    ref_np = ref.numpy()

    # Cast both to fp32 for comparison
    a = ref_np.astype(np.float32)[0]
    b = hidden_coreml.astype(np.float32)[0]

    diff = np.abs(a - b)
    print(f"\n[compare] pytorch-ane vs coreml hidden states:")
    print(f"  shapes: pytorch={a.shape}, coreml={b.shape}")
    print(f"  max_abs_diff: {diff.max():.4f}")
    print(f"  mean_abs_diff: {diff.mean():.4f}")
    cos_per_pos = np.array([
        np.dot(a[i], b[i]) / (np.linalg.norm(a[i]) * np.linalg.norm(b[i]) + 1e-8)
        for i in range(a.shape[0])
    ])
    print(f"  cos_sim: mean={cos_per_pos.mean():.6f} min={cos_per_pos.min():.6f}")

    # Apply LM head to both and check top-1 agreement
    def logits_from(h_np):
        h_mx = mx.array(h_np.astype(np.float32)).astype(mx.bfloat16)
        if hasattr(target, "lm_head"):
            logits = target.lm_head(h_mx)
        else:
            logits = target.model.embed_tokens.as_linear(h_mx)
        mx.eval(logits)
        return np.asarray(logits.astype(mx.float32))[0]

    a_logits = logits_from(a[None])
    b_logits = logits_from(b[None])
    top1_a = a_logits.argmax(-1)
    top1_b = b_logits.argmax(-1)
    top1_match = (top1_a == top1_b).mean()
    print(f"  top-1 agreement after LM head: {top1_match:.1%} ({int(top1_match*16)}/16)")

    # Correctness bar: cos > 0.99, top-1 > 80%
    ok = (cos_per_pos.mean() > 0.99 and top1_match > 0.8)
    print(f"\n[{'OK' if ok else 'FAIL'}]")


if __name__ == "__main__":
    main()
