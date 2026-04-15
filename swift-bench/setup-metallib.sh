#!/bin/bash
# Copy MLX Metal shader library next to the built Swift binary.
# mlx-swift's SwiftPM build doesn't bundle the metallib, so we source it
# from the Python mlx install. Run after swift build; before running any
# binary that uses mlx-swift (target-load-test, full SD runner).
set -e

SRC="/Users/carl/models/mlx-venv/lib/python3.11/site-packages/mlx/lib/mlx.metallib"
DST_DIR="$(dirname "$0")/.build/release"
DST="$DST_DIR/mlx.metallib"

if [[ ! -f "$SRC" ]]; then
    echo "mlx.metallib not found at $SRC" >&2
    echo "Install Python mlx: pip install mlx" >&2
    exit 1
fi

mkdir -p "$DST_DIR"
cp "$SRC" "$DST"
echo "Copied: $DST"
ls -la "$DST"
