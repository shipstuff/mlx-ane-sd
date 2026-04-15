"""Phase G: multi-stream SD serving benchmark.

Launches N subprocess workers, each running one SD stream. All workers
sync start via a file barrier, then race to generate max_new tokens.

Reports:
- Per-stream tok/s (mean, min, max)
- Aggregate tok/s (sum across streams — "total system throughput")
- Per-stream efficiency = per_stream_tps / solo_tps (shows saturation)

Compares three configurations:
- baseline: target-only, no SD, all on GPU
- f0: DFlash, target + draft on GPU (everything on GPU)
- f1: DFlash, target on GPU + draft on ANE (heterogeneous)

The research question: does F.1 scale better with N than F.0?
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).parent.parent
WORKER = Path(__file__).parent / "multistream_worker.py"
PYTHON = "/Users/carl/models/mlx-venv/bin/python"


def run_multistream(mode: str, n_streams: int, max_new: int,
                     mlmodelc: str, state_length: int,
                     results_dir: Path, prompt_id: int = 0) -> list[dict]:
    """Launch n_streams workers (all using the same prompt), sync start,
    collect results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    barrier = results_dir / f"barrier_{mode}_{n_streams}_p{prompt_id}"
    if barrier.exists():
        barrier.unlink()
    # Clean up any stale ready files from previous runs
    for f in results_dir.glob(f"ready_{mode}_*"):
        f.unlink()

    print(f"\n[phaseG] launching {n_streams}×{mode} (prompt_id={prompt_id})...",
          flush=True)
    procs = []
    for i in range(n_streams):
        report = results_dir / f"{mode}_n{n_streams}_p{prompt_id}_s{i}.json"
        cmd = [
            PYTHON, str(WORKER),
            "--mode", mode,
            "--stream-id", str(i),
            "--prompt-id", str(prompt_id),
            "--max-new", str(max_new),
            "--mlmodelc", mlmodelc,
            "--state-length", str(state_length),
            "--report-file", str(report),
            "--start-barrier", str(barrier),
        ]
        procs.append(subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        ))

    # Wait for all workers to post their ready files, then release the barrier.
    print(f"[phaseG] waiting for all {n_streams} workers to finish loading...",
          flush=True)
    ready_timeout = 600.0  # 10 min max to load
    t_wait_start = time.perf_counter()
    while time.perf_counter() - t_wait_start < ready_timeout:
        ready_files = list(results_dir.glob(f"ready_{mode}_*"))
        if len(ready_files) >= n_streams:
            break
        time.sleep(0.2)
    else:
        print(f"[phaseG] TIMEOUT: only {len(ready_files)}/{n_streams} workers ready",
              flush=True)

    elapsed_load = time.perf_counter() - t_wait_start
    print(f"[phaseG] all ready after {elapsed_load:.1f}s, releasing barrier...",
          flush=True)
    barrier.touch()
    t_race_start = time.perf_counter()

    # Wait for all procs
    results = []
    for i, p in enumerate(procs):
        try:
            out, err = p.communicate(timeout=600)
            if p.returncode != 0:
                print(f"[phaseG] stream-{i} exited {p.returncode}", flush=True)
                print(f"  stderr tail: {err.decode()[-500:]}", flush=True)
                continue
            report = results_dir / f"{mode}_n{n_streams}_p{prompt_id}_s{i}.json"
            if report.exists():
                results.append(json.loads(report.read_text()))
        except subprocess.TimeoutExpired:
            print(f"[phaseG] stream-{i} timeout", flush=True)
            p.kill()
    t_race = time.perf_counter() - t_race_start

    # Compute TRUE total throughput: total tokens generated / race wall time.
    total_tokens = sum(r.get("tokens_generated", 0) for r in results)
    true_total_tps = total_tokens / t_race if t_race > 0 else 0
    print(f"[phaseG] {mode}×{n_streams} complete. Race wall: {t_race:.1f}s. "
          f"Total tokens: {total_tokens}. True throughput: {true_total_tps:.2f} tok/s",
          flush=True)
    # Attach race-wall info so summarize can pick it up
    for r in results:
        r["_race_wall_s"] = t_race
        r["_true_total_tps"] = true_total_tps
    return results


def summarize(results: list[dict], mode: str, n_streams: int,
               prompt_id: int = 0) -> dict:
    """Aggregate per-stream results."""
    if not results:
        return {"mode": mode, "n_streams": n_streams, "prompt_id": prompt_id,
                "error": "no_results"}
    tps_list = [r["tok_per_s"] for r in results]
    total_tps_sum = sum(tps_list)  # sum of per-stream rates
    true_total = results[0].get("_true_total_tps", 0)  # tokens / race_wall
    mean_tps = statistics.mean(tps_list)
    return {
        "mode": mode,
        "n_streams": n_streams,
        "prompt_id": prompt_id,
        "n_success": len(results),
        "per_stream_tps_mean": mean_tps,
        "per_stream_tps_min": min(tps_list),
        "per_stream_tps_max": max(tps_list),
        "sum_of_per_stream_tps": total_tps_sum,  # overestimates due to finish skew
        "true_total_tps": true_total,  # honest: tokens / race wall
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["baseline", "f0", "f1"],
                    help="which configurations to test")
    ap.add_argument("--n-streams", nargs="+", type=int, default=[1, 2, 4, 8],
                    help="stream counts to sweep")
    ap.add_argument("--max-new", type=int, default=100)
    ap.add_argument("--mlmodelc",
                    default="/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc")
    ap.add_argument("--state-length", type=int, default=256)
    ap.add_argument("--results-dir", type=Path, default=Path("/tmp/phaseG_results"))
    ap.add_argument("--prompt-id", type=int, default=0,
                    help="Which prompt all streams should use (0-7, see multistream_worker.PROMPT_POOL)")
    args = ap.parse_args()

    all_summaries = []
    for n in args.n_streams:
        for mode in args.modes:
            print(f"\n=== {mode} × N={n} (prompt {args.prompt_id}) ===", flush=True)
            results = run_multistream(mode, n, args.max_new, args.mlmodelc,
                                        args.state_length, args.results_dir,
                                        prompt_id=args.prompt_id)
            summary = summarize(results, mode, n, prompt_id=args.prompt_id)
            all_summaries.append(summary)
            mean = summary.get('per_stream_tps_mean', 0)
            tot = summary.get('true_total_tps', 0)
            print(f"[{mode} N={n}] per-stream mean: {mean:.2f} tok/s, "
                  f"true total: {tot:.2f} tok/s", flush=True)

    # Final table
    print(f"\n\n=== Phase G Summary (prompt {args.prompt_id}) ===")
    print(f"{'mode':<10} {'N':>4} {'per-stream tok/s':<22} {'true total':>12} {'eff':>8}")
    solo = {s["mode"]: s["per_stream_tps_mean"] for s in all_summaries if s["n_streams"] == 1}
    for s in all_summaries:
        if "error" in s:
            print(f"{s['mode']:<10} {s['n_streams']:>4} ERROR")
            continue
        per = s["per_stream_tps_mean"]
        true_tot = s["true_total_tps"]
        eff = per / solo.get(s["mode"], 1) if solo.get(s["mode"], 0) > 0 else 0
        print(f"{s['mode']:<10} {s['n_streams']:>4} {per:>6.2f} "
              f"[{s['per_stream_tps_min']:.1f}-{s['per_stream_tps_max']:.1f}]  "
              f"{true_tot:>8.2f}  {eff:>6.1%}")

    out_file = args.results_dir / "summary.json"
    out_file.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[saved] {out_file}")


if __name__ == "__main__":
    main()
