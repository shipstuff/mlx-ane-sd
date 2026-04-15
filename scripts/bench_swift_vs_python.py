"""Python vs Swift ANE draft call latency, same inputs.

Runs the same DFlash compiled model from both Python (coremltools) and
the Swift runner at /Users/carl/projects/mlx-ane-sd/swift-bench/. Reports
per-call latency for each, plus the per-100-token extrapolation.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import coremltools as ct


REPO = Path(__file__).parent.parent
SWIFT_BIN = REPO / "swift-bench" / ".build" / "release" / "dflash-swift-runner"


def python_latency(mlmodelc: str, state_length: int, iters: int) -> dict:
    print(f"[python] loading {mlmodelc}...")
    t0 = time.perf_counter()
    model = ct.models.CompiledMLModel(mlmodelc, ct.ComputeUnit.CPU_AND_NE)
    load_ms = (time.perf_counter() - t0) * 1000

    BS, CS, H = 16, 16, 2560
    CONCAT_DIM, DH = 12800, 128
    N_LAYERS, HKV = 5, 8
    T = CS + BS
    attend_len = state_length + T

    # Same deterministic fill as Swift
    def det(shape):
        n = int(np.prod(shape))
        raw = np.arange(n, dtype=np.uint16) & 0xFFFF
        return raw.view(np.float16).reshape(shape)

    inputs = {
        "noise_embedding": det((1, BS, H)),
        "target_hidden": det((1, CS, CONCAT_DIM)),
        "cos_q": det((BS, DH)),
        "sin_q": det((BS, DH)),
        "cos_k": det((T, DH)),
        "sin_k": det((T, DH)),
        "cache_K": det((N_LAYERS, HKV, state_length, DH)),
        "cache_V": det((N_LAYERS, HKV, state_length, DH)),
        "causal_mask": det((1, 1, BS, attend_len)),
    }

    # Warmup
    for _ in range(5):
        model.predict(inputs)

    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        model.predict(inputs)
        samples.append((time.perf_counter() - t0) * 1000)
    samples.sort()
    mean = np.mean(samples)
    return {
        "mean_ms": float(mean),
        "median_ms": float(samples[len(samples) // 2]),
        "p10_ms": float(samples[len(samples) // 10]),
        "p90_ms": float(samples[len(samples) * 9 // 10]),
        "stdev_ms": float(np.std(samples)),
        "min_ms": float(min(samples)),
        "max_ms": float(max(samples)),
        "n_iters": iters,
        "state_length": state_length,
        "load_time_ms": load_ms,
    }


def swift_latency(mlmodelc: str, state_length: int, iters: int) -> dict:
    # Note: the predict-only bench is in a separate binary (ane-latency-bench),
    # while dflash-swift-runner measures full cycle (predict + mask + rope + cache).
    # Use the simpler ane-latency-bench here for predict-only parity.
    simple_bin = SWIFT_BIN.parent / "ane-latency-bench"
    result = subprocess.run(
        [str(simple_bin), mlmodelc, str(state_length), str(iters)],
        check=True, capture_output=True, text=True,
    )
    # ane-latency-bench prints human-readable to stdout; parse the key numbers
    out = result.stdout
    data = {"n_iters": iters, "state_length": state_length}
    for line in out.splitlines():
        line = line.strip()
        if line.startswith("mean:"):
            data["mean_ms"] = float(line.split()[1])
        elif line.startswith("median:"):
            data["median_ms"] = float(line.split()[1])
        elif line.startswith("stdev:"):
            data["stdev_ms"] = float(line.split()[1])
        elif line.startswith("p10:"):
            data["p10_ms"] = float(line.split()[1])
        elif line.startswith("p90:"):
            data["p90_ms"] = float(line.split()[1])
        elif line.startswith("min:"):
            data["min_ms"] = float(line.split()[1])
        elif line.startswith("max:"):
            data["max_ms"] = float(line.split()[1])
    data.setdefault("load_time_ms", 0)
    return data


def swift_full_cycle(mlmodelc: str, state_length: int, cycles: int = 30) -> dict:
    """Measure the full Swift SD draft cycle (predict + mask + rope + cache).
    This is what matters for end-to-end — not just predict latency.
    """
    result = subprocess.run(
        [str(SWIFT_BIN),
         "--draft", mlmodelc,
         "--state-length", str(state_length),
         "--cycles", str(cycles),
         "--accept-rate", "0.5",
         "--json"],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-lengths", type=int, nargs="+", default=[256, 1024])
    ap.add_argument("--iters", type=int, default=100)
    args = ap.parse_args()

    mlmodelc_map = {
        256: "/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc",
        1024: "/tmp/dflash_ane_accum_1024_c/dflash_ane_accum_1024.mlmodelc",
    }

    rows = []
    for s in args.state_lengths:
        path = mlmodelc_map.get(s)
        if not path or not Path(path).exists():
            print(f"[skip] S={s}: mlmodelc not found at {path}")
            continue

        print(f"\n=== S={s} ===")
        py = python_latency(path, s, args.iters)
        print(f"[python predict-only] mean={py['mean_ms']:.3f} stdev={py['stdev_ms']:.3f}")
        sw = swift_latency(path, s, args.iters)
        print(f"[swift predict-only]  mean={sw['mean_ms']:.3f}")
        full = swift_full_cycle(path, s, cycles=30)
        cycle_mean = full["profile"]["phases"]["cycle_total"]["totalMs"] / full["profile"]["phases"]["cycle_total"]["calls"]
        print(f"[swift full cycle]    mean={cycle_mean:.3f} (includes mask/rope/cache)")

        savings = py["mean_ms"] - sw["mean_ms"]
        pct = savings / py["mean_ms"] * 100
        per_100_savings = savings * 25
        rows.append({
            "state_length": s,
            "python_predict_ms": py["mean_ms"],
            "swift_predict_ms": sw["mean_ms"],
            "swift_full_cycle_ms": cycle_mean,
            "savings_ms": savings,
            "savings_pct": pct,
            "per_100_tok_savings_ms": per_100_savings,
        })
        print(f"  predict-only savings: {savings:.2f} ms ({pct:.1f}%)")
        print(f"  per 100 tokens (~25 cycles): {per_100_savings:.0f} ms saved")

    print("\n=== Summary ===")
    print(f"{'S':>6} {'py predict':>12} {'sw predict':>12} {'sw full':>10} {'save %':>8}")
    for r in rows:
        print(f"{r['state_length']:>6} {r['python_predict_ms']:>9.2f}ms "
              f"{r['swift_predict_ms']:>9.2f}ms "
              f"{r['swift_full_cycle_ms']:>7.2f}ms "
              f"{r['savings_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
