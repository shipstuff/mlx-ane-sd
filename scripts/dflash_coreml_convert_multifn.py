"""Convert DFlash (multi-function variant) to CoreML.

Pipeline:
1. Load reference DFlash checkpoint from HF.
2. For each (write_pos, rotate) spec, trace + convert to a single-function
   mlpackage in a scratch dir.
3. Assemble all specs into one multi-function .mlpackage using
   `ct.utils.MultiFunctionDescriptor` + `ct.utils.save_multifunction`.

Example run (N=32 normal + 1 rotate = 33 functions):
    python scripts/dflash_coreml_convert_multifn.py \
        --output /tmp/dflash_ane_multifn.mlpackage \
        --state-length 1024 \
        --scratch /tmp/dflash_multifn_scratch

The resulting mlpackage holds functions named `write_0`, `write_32`, ...,
`write_992`, and `rotate` (writes at write_pos=992, shift-left first).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import coremltools as ct

sys.path.insert(0, str(Path(__file__).parent))

from dflash_torch import load_dflash_from_hf
from dflash_ane_multifn import (
    DFlashDraftModelMultiFn, copy_weights_to_multifn,
)
from dflash_ane_accumcache import set_rmsnorm_mode
from huggingface_hub import snapshot_download


def build_single_variant(config, BS, CS, STATE_LEN, write_pos, rotate,
                          torch_draft, out_path: Path):
    """Convert one (write_pos, rotate) variant to a single-function mlpackage."""
    H = config.hidden_size
    N = config.num_hidden_layers
    Hkv = config.num_key_value_heads
    concat_dim = len(config.target_layer_ids) * H
    Dh = config.head_dim
    T = CS + BS

    model = DFlashDraftModelMultiFn(
        config, block_size=BS, ctx_size=CS, state_length=STATE_LEN,
        write_pos=write_pos, rotate=rotate,
    ).to(torch.float32).eval()
    copy_weights_to_multifn(torch_draft, model)
    set_rmsnorm_mode("ane")

    attend_len = model.attend_len  # variant-specific

    noise_emb = torch.randn(1, BS, H) * 0.01
    target_hidden = torch.randn(1, CS, concat_dim) * 0.01
    cos_q = torch.randn(BS, Dh)
    sin_q = torch.randn(BS, Dh)
    cos_k = torch.randn(T, Dh)
    sin_k = torch.randn(T, Dh)
    cache_K = torch.zeros(N, Hkv, STATE_LEN, Dh)
    cache_V = torch.zeros(N, Hkv, STATE_LEN, Dh)
    # Mask covers the attention scope for this variant.
    causal_mask = torch.zeros(1, 1, BS, attend_len, dtype=torch.float32)

    # Eager sanity
    with torch.no_grad():
        h, kout, vout = model(noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
                                cache_K, cache_V, causal_mask)
    assert h.shape == (1, BS, H), f"bad hidden shape {h.shape}"
    assert kout.shape == (N, Hkv, T, Dh), f"bad new_K shape {kout.shape} (expected {(N, Hkv, T, Dh)})"

    with torch.no_grad():
        traced = torch.jit.trace(
            model,
            (noise_emb, target_hidden, cos_q, sin_q, cos_k, sin_k,
             cache_K, cache_V, causal_mask),
            strict=False,
        )

    inputs = [
        ct.TensorType(name="noise_embedding", shape=(1, BS, H), dtype=np.float16),
        ct.TensorType(name="target_hidden", shape=(1, CS, concat_dim), dtype=np.float16),
        ct.TensorType(name="cos_q", shape=(BS, Dh), dtype=np.float16),
        ct.TensorType(name="sin_q", shape=(BS, Dh), dtype=np.float16),
        ct.TensorType(name="cos_k", shape=(T, Dh), dtype=np.float16),
        ct.TensorType(name="sin_k", shape=(T, Dh), dtype=np.float16),
        ct.TensorType(name="cache_K", shape=(N, Hkv, STATE_LEN, Dh), dtype=np.float16),
        ct.TensorType(name="cache_V", shape=(N, Hkv, STATE_LEN, Dh), dtype=np.float16),
        ct.TensorType(name="causal_mask", shape=(1, 1, BS, attend_len), dtype=np.float16),
    ]
    outputs = [
        ct.TensorType(name="hidden", dtype=np.float16),
        ct.TensorType(name="new_K", dtype=np.float16),
        ct.TensorType(name="new_V", dtype=np.float16),
    ]

    t0 = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        inputs=inputs,
        outputs=outputs,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
        convert_to="mlprogram",
    )
    print(f"  convert done in {time.perf_counter()-t0:.1f}s")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_path))
    return out_path


def variant_name(write_pos: int, rotate: bool) -> str:
    return "rotate" if rotate else f"write_{write_pos}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True,
                    help="Final multi-function .mlpackage")
    ap.add_argument("--scratch", type=Path, default=Path("/tmp/dflash_multifn_scratch"),
                    help="Dir for per-variant intermediate mlpackages")
    ap.add_argument("--state-length", type=int, default=1024)
    ap.add_argument("--num-variants", type=int, default=32,
                    help="Number of write_pos variants to bake in")
    ap.add_argument("--compile", action="store_true",
                    help="After save, run coremlcompiler to produce mlmodelc")
    ap.add_argument("--compile-dir", type=Path, default=None,
                    help="Dir to compile into (default: <output>.parent)")
    ap.add_argument("--only-variant", type=str, default=None,
                    help="Debug: only convert one variant by name (e.g. 'write_0' or 'rotate')")
    ap.add_argument("--skip-convert", action="store_true",
                    help="Skip per-variant convert; combine existing scratch outputs")
    args = ap.parse_args()

    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    torch_draft = load_dflash_from_hf(draft_path)
    config = torch_draft.config

    BS = config.block_size
    CS = config.block_size
    STATE_LEN = args.state_length
    T = CS + BS
    max_write_pos = STATE_LEN - T
    step = T
    N_variants = args.num_variants
    # write_pos values: 0, T, 2T, ..., up to max_write_pos, capped at N_variants
    candidate_positions = list(range(0, max_write_pos + 1, step))
    if len(candidate_positions) > N_variants:
        candidate_positions = candidate_positions[:N_variants]

    specs = [(p, False) for p in candidate_positions] + [(max_write_pos, True)]
    print(f"[plan] {len(specs)} total variants:")
    for wp, rot in specs[:5]:
        print(f"  {variant_name(wp, rot)}  (write_pos={wp}, rotate={rot})")
    if len(specs) > 5:
        print(f"  ... ({len(specs) - 5} more)")

    args.scratch.mkdir(parents=True, exist_ok=True)
    per_variant_paths = {}

    if not args.skip_convert:
        for idx, (wp, rot) in enumerate(specs):
            vname = variant_name(wp, rot)
            if args.only_variant is not None and vname != args.only_variant:
                continue
            print(f"\n[{idx+1}/{len(specs)}] Converting variant {vname} (write_pos={wp}, rotate={rot})")
            vpath = args.scratch / f"{vname}.mlpackage"
            if vpath.exists():
                shutil.rmtree(vpath, ignore_errors=True)
            build_single_variant(config, BS, CS, STATE_LEN, wp, rot, torch_draft, vpath)
            per_variant_paths[vname] = vpath

    # If we skipped, populate from disk
    if args.skip_convert or args.only_variant is not None:
        for wp, rot in specs:
            vname = variant_name(wp, rot)
            if vname not in per_variant_paths:
                vpath = args.scratch / f"{vname}.mlpackage"
                if vpath.exists():
                    per_variant_paths[vname] = vpath

    if args.only_variant is not None:
        print(f"\n[done] Only built variant {args.only_variant} (skipping combine)")
        return

    # Combine into a single multi-function mlpackage.
    print(f"\n[combine] {len(per_variant_paths)} variants -> {args.output}")
    desc = ct.utils.MultiFunctionDescriptor()
    default_name = None
    for wp, rot in specs:
        vname = variant_name(wp, rot)
        if vname not in per_variant_paths:
            print(f"  WARN: missing {vname}, skipping")
            continue
        desc.add_function(str(per_variant_paths[vname]),
                           src_function_name="main",
                           target_function_name=vname)
        if default_name is None:
            default_name = vname
    desc.default_function_name = default_name
    print(f"[combine] default function = {default_name}")

    if args.output.exists():
        shutil.rmtree(args.output, ignore_errors=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    ct.utils.save_multifunction(desc, str(args.output))
    print(f"[combine] save_multifunction done in {time.perf_counter()-t0:.1f}s")

    # Report sizes
    def du_mb(p: Path):
        try:
            out = subprocess.check_output(["du", "-sm", str(p)], text=True)
            return int(out.split()[0])
        except Exception:
            return -1
    total_scratch = sum(du_mb(p) for p in per_variant_paths.values())
    print(f"[size] combined mlpackage = {du_mb(args.output)} MB")
    print(f"[size] sum of scratch variants = {total_scratch} MB")
    print(f"[size] dedup ratio = {du_mb(args.output) / max(total_scratch, 1):.3f}")

    if args.compile:
        compile_dir = args.compile_dir or args.output.parent
        compile_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[compile] coremlcompiler -> {compile_dir}")
        cmd = ["xcrun", "coremlcompiler", "compile", str(args.output), str(compile_dir)]
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True)
        print(f"[compile] done in {time.perf_counter()-t0:.1f}s rc={r.returncode}")
        if r.returncode != 0:
            print(r.stderr)
            sys.exit(1)
        # mlmodelc has the same stem as mlpackage
        mlmodelc = compile_dir / (args.output.stem + ".mlmodelc")
        print(f"[compile] produced {mlmodelc}")


if __name__ == "__main__":
    main()
