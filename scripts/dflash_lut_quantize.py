"""Apply LUT quantization to the DFlash ANE mlpackage.

LUT (Look-Up Table) palettization groups model weights into K palette
clusters and stores only the cluster index per weight + the palette
values. For LUT4, K=16 palette values, each weight is a 4-bit index →
4× compression vs fp16.

ANEMLL's findings (from anemll-qwen35): LUT4 on qwen3.5-0.8B super-blocks
gave 6.5× speedup per block, same output quality on real prompts. The
speedup comes from ANE bandwidth reduction — the draft's weights are
streamed each call.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import coremltools as ct
import coremltools.optimize.coreml as cto


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True,
                    help="input .mlpackage path")
    ap.add_argument("--output", type=Path, required=True,
                    help="output .mlpackage path")
    ap.add_argument("--bits", type=int, default=4,
                    help="palette bits (4 = LUT4, 6 = LUT6)")
    ap.add_argument("--granularity", choices=["per_tensor", "per_grouped_channel"],
                    default="per_grouped_channel")
    ap.add_argument("--group-size", type=int, default=8)
    args = ap.parse_args()

    print(f"[load] {args.input}")
    mlmodel = ct.models.MLModel(str(args.input), skip_model_load=True)

    config = cto.OptimizationConfig(
        global_config=cto.OpPalettizerConfig(
            nbits=args.bits,
            mode="kmeans",
            granularity=args.granularity,
            group_size=args.group_size if args.granularity == "per_grouped_channel" else None,
        ),
    )
    print(f"[palettize] nbits={args.bits} granularity={args.granularity} "
          f"group_size={args.group_size}")
    import time
    t0 = time.perf_counter()
    mlmodel_q = cto.palettize_weights(mlmodel, config)
    print(f"[palettize] done in {time.perf_counter()-t0:.1f}s")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    mlmodel_q.save(str(args.output))
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
