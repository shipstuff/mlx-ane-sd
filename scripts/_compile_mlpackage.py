"""Compile an .mlpackage into an .mlmodelc using coremltools.

Requires xcode/CommandLineTools to not be available isn't a blocker:
coremltools ships with a bundled compiler and exposes the compiled
artifact via MLModel.get_compiled_model_path().
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import coremltools as ct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    print(f"[compile] loading {args.input}")
    m = ct.models.MLModel(str(args.input), compute_units=ct.ComputeUnit.CPU_AND_NE)
    src = Path(m.get_compiled_model_path())
    print(f"[compile] compiled at {src}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        shutil.rmtree(args.output)
    # src is a temp path that is cleaned up when MLModel is GC'd.
    # Copy its contents into the stable output location.
    shutil.copytree(src, args.output)
    print(f"[compile] copied to {args.output}")


if __name__ == "__main__":
    main()
