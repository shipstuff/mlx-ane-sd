"""Reproducible grid runner for the (quantization x state_length x gen_length)
matrix on the F.1 accumulating-cache ANE DFlash variant.

Axes (default, fully overridable):
  - Quantization variants: none, lut6_per_tensor, lut4_per_grouped_channel_g8
  - State lengths: 512, 1024, 2048, 4096
  - Generation lengths (max_new): 100, 300, 1000

For each cell the runner:
  1. Ensures the corresponding compiled `.mlmodelc` exists (convert + [lut] +
     compile once, cache on disk).
  2. Runs the 4 canonical prompts through `phaseF1_ane_stream_accum`.
  3. Records mean tok/s + per-prompt detail + per-call draft latency.

Outputs:
  - JSON (`--output-json`): raw per-prompt rows
  - CSV (`--output-csv`): tidy per-(cell, prompt) table
  - Markdown (`--output-md`): three 2-D tables (one per quant variant) with
    rows = state_length and cols = gen_length, values = mean tok/s.

Usage:
  python scripts/bench_lut_cache_grid.py \
      --artifacts-dir /tmp/lut_cache_grid \
      --output-md notes/lut_cache_grid.md \
      --output-json artifacts/lut_cache_grid.json \
      --output-csv artifacts/lut_cache_grid.csv

To run a subset:
  python scripts/bench_lut_cache_grid.py --quant none lut6_pt --states 1024 \
      --gens 100 300 --prompts capital story
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
CONVERT_SCRIPT = SCRIPT_DIR / "dflash_coreml_convert_accum.py"
LUT_SCRIPT = SCRIPT_DIR / "dflash_lut_quantize.py"


# ------------------------------------------------------------ quant variants --

@dataclass(frozen=True)
class QuantSpec:
    key: str
    label: str
    # None => no quantization. Otherwise, args forwarded to dflash_lut_quantize.
    bits: int | None = None
    granularity: str | None = None
    group_size: int | None = None


QUANT_VARIANTS = {
    "none": QuantSpec("none", "Unquantized fp16"),
    "lut6_pt": QuantSpec(
        "lut6_pt", "LUT6 per_tensor",
        bits=6, granularity="per_tensor",
    ),
    "lut4_gc8": QuantSpec(
        "lut4_gc8", "LUT4 per_grouped_channel (group=8)",
        bits=4, granularity="per_grouped_channel", group_size=8,
    ),
}


# ----------------------------------------------------------------- prompts ---

# Reusing the set from phaseF1_ane_stream_accum to keep results comparable.
PROMPTS = [
    ("capital",
     "The capital of France is Paris, which is known for"),
    ("fibonacci",
     "def fibonacci(n):\n    if n <= 1:\n        return n\n    return "
     "fibonacci(n-1) + fibonacci(n-2)\n\n# Test:\nfor i in range(10):\n    "
     "print(f'fib({i}) = {fibonacci(i)}')"),
    ("math",
     "Solve for x: 2x + 5 = 17. Step by step: 2x = 17 - 5 = 12; x = 12/2 = 6."
     " Now solve 3y - 7 = 20:"),
    ("story",
     "Once upon a time in a small village nestled between two mountains, "
     "there lived a young girl named Elara who"),
]


# ------------------------------------------------------------- path layout ---

def paths_for(artifacts: Path, quant: QuantSpec, state: int):
    """Stable paths for every artifact produced per (quant, state)."""
    base = artifacts / f"S{state}"
    base.mkdir(parents=True, exist_ok=True)
    # Base unquantized package is shared across quant variants that start from it.
    base_pkg = base / f"dflash_accum_S{state}.mlpackage"
    if quant.bits is None:
        pkg = base_pkg
    else:
        pkg = base / f"dflash_accum_S{state}_{quant.key}.mlpackage"
    mlmodelc = base / f"{pkg.stem}.mlmodelc"
    return base_pkg, pkg, mlmodelc


# ------------------------------------------------------------- build phases --

def run(cmd: list[str], env: dict | None = None, log_prefix: str = ""):
    print(f"\n{log_prefix}$ {' '.join(cmd)}", flush=True)
    t0 = time.perf_counter()
    res = subprocess.run(cmd, env=env, check=False)
    elapsed = time.perf_counter() - t0
    print(f"{log_prefix}-> exit={res.returncode} in {elapsed:.1f}s", flush=True)
    if res.returncode != 0:
        raise RuntimeError(f"command failed (exit {res.returncode}): {cmd}")
    return elapsed


def ensure_base_mlpackage(base_pkg: Path, state: int, python: str, env: dict):
    if base_pkg.exists():
        print(f"[cache] base pkg exists: {base_pkg}")
        return
    run([python, str(CONVERT_SCRIPT),
         "--output", str(base_pkg),
         "--state-length", str(state)],
        env=env, log_prefix=f"[convert S={state}] ")


def ensure_quant_mlpackage(base_pkg: Path, out_pkg: Path, quant: QuantSpec,
                            python: str, env: dict):
    if quant.bits is None or out_pkg == base_pkg:
        return
    if out_pkg.exists():
        print(f"[cache] quant pkg exists: {out_pkg}")
        return
    cmd = [python, str(LUT_SCRIPT),
           "--input", str(base_pkg), "--output", str(out_pkg),
           "--bits", str(quant.bits),
           "--granularity", quant.granularity]
    if quant.group_size is not None:
        cmd += ["--group-size", str(quant.group_size)]
    run(cmd, env=env, log_prefix=f"[lut {quant.key}] ")


def _compile_via_subprocess(mlpkg: Path, mlmodelc: Path, python: str,
                             env: dict):
    """Compile an .mlpackage into an .mlmodelc using a subprocess to avoid
    holding MLModel objects in the driver process."""
    mlmodelc.parent.mkdir(parents=True, exist_ok=True)
    helper = REPO_ROOT / "scripts" / "_compile_mlpackage.py"
    cmd = [python, str(helper), "--input", str(mlpkg), "--output", str(mlmodelc)]
    run(cmd, env=env, log_prefix=f"[compile {mlpkg.name}] ")


def ensure_compiled(mlpkg: Path, mlmodelc: Path, python: str, env: dict):
    if mlmodelc.exists() and (mlmodelc / "model.mil").exists():
        print(f"[cache] compiled: {mlmodelc}")
        return
    _compile_via_subprocess(mlpkg, mlmodelc, python, env)


# ------------------------------------------------------------- benchmark ----

# Module-level cache of expensive-to-load objects.
_TARGET_CACHE: dict = {}


def _get_target_and_config():
    """Lazily load target model + tokenizer + DFlash config once per process."""
    if "target" in _TARGET_CACHE:
        return _TARGET_CACHE["target"], _TARGET_CACHE["tok"], _TARGET_CACHE["config"]
    from dflash_torch import DFlashConfig
    from huggingface_hub import snapshot_download
    from mlx_lm import load as mlx_load

    draft_path = snapshot_download("z-lab/Qwen3-4B-DFlash-b16")
    config = DFlashConfig.from_hf_json(str(Path(draft_path) / "config.json"))
    print(f"[bench] load target (one-shot)...", flush=True)
    target, tok = mlx_load("mlx-community/Qwen3-4B-bf16")
    _TARGET_CACHE.update({"target": target, "tok": tok, "config": config})
    return target, tok, config


def benchmark_cell(mlmodelc: Path, state: int, max_new: int,
                   prompts: list[tuple[str, str]],
                   warmup: bool = True):
    """Runs the prompts through the stream_generate_ane_accum loop."""
    # Import locally — relies on sys.path being set by caller.
    from phaseF1_ane_stream_accum import (
        DFlashANEAccumDraft, stream_generate_ane_accum,
    )
    target, tok, config = _get_target_and_config()

    print(f"[bench] load draft mlmodelc={mlmodelc}...", flush=True)
    draft = DFlashANEAccumDraft(str(mlmodelc), config, state_length=state)

    if warmup:
        print("[bench] warmup...", flush=True)
        list(stream_generate_ane_accum(target, draft, tok, "The weather", 20))

    rows = []
    for name, prompt in prompts:
        print(f"\n[bench] === {name} @ S={state} max_new={max_new} ===", flush=True)
        t_call0 = draft.t_predict_total
        n_call0 = draft.n_predicts
        gen, t, accepted, cycles, _ = stream_generate_ane_accum(
            target, draft, tok, prompt, max_new)
        tps = len(gen) / t
        per_call_ms = ((draft.t_predict_total - t_call0) /
                        max(1, draft.n_predicts - n_call0)) * 1000
        rows.append({
            "prompt": name,
            "tokens": len(gen),
            "seconds": t,
            "tps": tps,
            "cycles": cycles,
            "accepted": accepted,
            "avg_accept_per_cycle": accepted / max(1, cycles),
            "per_call_ms": per_call_ms,
        })
        print(f"  tokens={len(gen)} in {t:.2f}s -> {tps:.2f} tok/s "
              f"(cycles={cycles}, accepted={accepted}, "
              f"per_call={per_call_ms:.2f}ms)", flush=True)

    mean_tps = statistics.mean(r["tps"] for r in rows)
    print(f"[bench] mean tok/s @ S={state} max_new={max_new}: {mean_tps:.2f}",
          flush=True)
    return {"mean_tps": mean_tps, "prompts": rows}


# ------------------------------------------------------------- reporting ----

def write_markdown(out_md: Path, grid: dict, states: list[int],
                    gens: list[int], prompts_run: list[str]):
    lines = []
    lines.append("# LUT x cache-size x generation-length grid")
    lines.append("")
    lines.append(
        "Solo benchmarks of the F.1 accumulating-cache ANE DFlash draft across "
        "quantization, cache size (state_length S), and generation length "
        "(max_new). Target: `mlx-community/Qwen3-4B-bf16` on GPU. Draft: "
        "`z-lab/Qwen3-4B-DFlash-b16` ported to ANE (100% ANE placement)."
    )
    lines.append("")
    lines.append(
        f"Hardware: Mac mini M4 Pro, 64 GB. Prompts (n={len(prompts_run)}): "
        f"{', '.join(prompts_run)}. Values = mean tok/s across prompts."
    )
    lines.append("")
    for qkey, qspec in QUANT_VARIANTS.items():
        if qkey not in grid:
            continue
        lines.append(f"## {qspec.label}")
        lines.append("")
        header = ["state_length \\ max_new"] + [f"{g}" for g in gens]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---:"] * len(header)) + "|")
        for s in states:
            row = [f"S={s}"]
            for g in gens:
                cell = grid.get(qkey, {}).get((s, g))
                if cell is None:
                    row.append("—")
                else:
                    row.append(f"{cell['mean_tps']:.2f}")
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")
        # Per-prompt detail (fold-out)
        lines.append("<details><summary>Per-prompt detail</summary>")
        lines.append("")
        for s in states:
            for g in gens:
                cell = grid.get(qkey, {}).get((s, g))
                if cell is None:
                    continue
                lines.append(f"**S={s}, max_new={g}** — mean {cell['mean_tps']:.2f} tok/s")
                lines.append("")
                lines.append("| prompt | tokens | seconds | tok/s | cycles | accepted | per_call_ms |")
                lines.append("|---|---:|---:|---:|---:|---:|---:|")
                for r in cell["prompts"]:
                    lines.append(
                        f"| {r['prompt']} | {r['tokens']} | {r['seconds']:.2f} "
                        f"| {r['tps']:.2f} | {r['cycles']} | {r['accepted']} "
                        f"| {r['per_call_ms']:.2f} |"
                    )
                lines.append("")
        lines.append("</details>")
        lines.append("")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines))
    print(f"[report] wrote {out_md}")


def write_json(out_json: Path, grid: dict):
    # Convert tuple keys to strings for JSON.
    serial = {}
    for qkey, cells in grid.items():
        serial[qkey] = []
        for (s, g), cell in cells.items():
            serial[qkey].append({
                "state_length": s, "max_new": g, **cell,
            })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(serial, indent=2))
    print(f"[report] wrote {out_json}")


def write_csv(out_csv: Path, grid: dict):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quant", "state_length", "max_new", "prompt",
                    "tokens", "seconds", "tps", "cycles", "accepted",
                    "avg_accept_per_cycle", "per_call_ms"])
        for qkey, cells in grid.items():
            for (s, g), cell in cells.items():
                for r in cell["prompts"]:
                    w.writerow([
                        qkey, s, g, r["prompt"], r["tokens"], f"{r['seconds']:.3f}",
                        f"{r['tps']:.3f}", r["cycles"], r["accepted"],
                        f"{r['avg_accept_per_cycle']:.3f}", f"{r['per_call_ms']:.3f}",
                    ])
    print(f"[report] wrote {out_csv}")


# ----------------------------------------------------------------- driver ----

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts-dir", type=Path, default=Path("/tmp/lut_cache_grid"),
                    help="where .mlpackage/.mlmodelc artifacts are stored")
    ap.add_argument("--output-md", type=Path, default=REPO_ROOT / "notes/lut_cache_grid.md")
    ap.add_argument("--output-json", type=Path, default=REPO_ROOT / "artifacts/lut_cache_grid.json")
    ap.add_argument("--output-csv", type=Path, default=REPO_ROOT / "artifacts/lut_cache_grid.csv")
    ap.add_argument("--quant", nargs="+", default=list(QUANT_VARIANTS.keys()),
                    help="subset of quant variants to run")
    ap.add_argument("--states", nargs="+", type=int, default=[512, 1024, 2048, 4096])
    ap.add_argument("--gens", nargs="+", type=int, default=[100, 300, 1000])
    ap.add_argument("--prompts", nargs="+", default=[p[0] for p in PROMPTS])
    ap.add_argument("--long-gen-prompts", nargs="+", default=None,
                    help="Restrict to these prompt names ONLY for cells whose "
                         "max_new is >= --long-gen-threshold. Useful when "
                         "gen=1000 cells would take >5 min each.")
    ap.add_argument("--long-gen-threshold", type=int, default=1000,
                    help="max_new at/above which --long-gen-prompts is applied")
    ap.add_argument("--skip-build", action="store_true",
                    help="assume mlpackages/mlmodelcs are already built")
    ap.add_argument("--build-only", action="store_true",
                    help="build all mlpackages/mlmodelcs, then exit")
    ap.add_argument("--python", default=sys.executable,
                    help="python interpreter used for convert/quant subprocesses")
    ap.add_argument("--resume", action="store_true",
                    help="skip cells whose results are already in --output-json")
    args = ap.parse_args()

    # Ensure the local script dir + /tmp/dflash are importable at runtime.
    env = os.environ.copy()
    extra = [str(SCRIPT_DIR), "/tmp/dflash"]
    env["PYTHONPATH"] = os.pathsep.join(extra + [env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    sys.path[:0] = extra

    prompts_run = [(n, p) for n, p in PROMPTS if n in args.prompts]
    if not prompts_run:
        print("[fatal] no prompts selected", file=sys.stderr)
        sys.exit(2)

    long_prompts_run = None
    if args.long_gen_prompts:
        long_prompts_run = [(n, p) for n, p in PROMPTS if n in args.long_gen_prompts]
        print(f"[note] long-gen (max_new>={args.long_gen_threshold}) uses "
              f"reduced prompt set: {[n for n, _ in long_prompts_run]}")

    # ---------------- Build phase: ensure every (quant, state) is compiled ---
    needed = [(q, s) for q in args.quant for s in args.states]
    if not args.skip_build:
        print(f"[build] will ensure {len(needed)} (quant, state) variants are built")
        for qkey, s in needed:
            qspec = QUANT_VARIANTS[qkey]
            base_pkg, pkg, mlmodelc = paths_for(args.artifacts_dir, qspec, s)
            ensure_base_mlpackage(base_pkg, s, args.python, env)
            ensure_quant_mlpackage(base_pkg, pkg, qspec, args.python, env)
            ensure_compiled(pkg, mlmodelc, args.python, env)
    if args.build_only:
        print("[build-only] done")
        return

    # ---------------- Resume support: read existing JSON ---------------------
    grid: dict[str, dict[tuple[int, int], dict]] = {q: {} for q in args.quant}
    if args.resume and args.output_json.exists():
        try:
            existing = json.loads(args.output_json.read_text())
            for qkey, cells in existing.items():
                if qkey not in grid:
                    grid[qkey] = {}
                for c in cells:
                    grid[qkey][(c["state_length"], c["max_new"])] = {
                        "mean_tps": c["mean_tps"], "prompts": c["prompts"],
                    }
            n = sum(len(v) for v in grid.values())
            print(f"[resume] loaded {n} existing cells from {args.output_json}")
        except Exception as exc:
            print(f"[resume] failed to load existing JSON: {exc}")

    # ---------------- Run phase ---------------------------------------------
    t_run0 = time.perf_counter()
    for qkey in args.quant:
        qspec = QUANT_VARIANTS[qkey]
        for s in args.states:
            for g in args.gens:
                if (s, g) in grid[qkey] and args.resume:
                    print(f"[skip] existing result for {qkey} S={s} max_new={g}")
                    continue
                _, _, mlmodelc = paths_for(args.artifacts_dir, qspec, s)
                print(f"\n[run] {qkey} S={s} max_new={g} -> {mlmodelc}", flush=True)
                prompts_for_cell = prompts_run
                if long_prompts_run is not None and g >= args.long_gen_threshold:
                    prompts_for_cell = long_prompts_run
                cell = benchmark_cell(mlmodelc, s, g, prompts_for_cell, warmup=True)
                grid[qkey][(s, g)] = cell
                # Write partial JSON after every cell for crash safety
                write_json(args.output_json, grid)
                write_csv(args.output_csv, grid)
    elapsed = time.perf_counter() - t_run0
    print(f"\n[run] total benchmark wall time: {elapsed/60:.1f} min")

    write_markdown(args.output_md, grid, args.states, args.gens,
                   [p[0] for p in prompts_run])
    write_json(args.output_json, grid)
    write_csv(args.output_csv, grid)


if __name__ == "__main__":
    main()
