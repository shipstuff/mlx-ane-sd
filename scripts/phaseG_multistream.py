"""Phase G: multi-stream SD serving benchmark.

Launches N subprocess workers, each running one SD stream. All workers
sync start via a file barrier, then race to generate max_new tokens.
Optionally runs a background MLX workload for contention.

Reports:
- Per-stream tok/s (mean, min, max)
- True total tok/s = total tokens / race wall time
- Per-stream efficiency = per_stream_tps / solo_tps

Compares three configurations:
- baseline: target-only, no SD, all on GPU
- f0: DFlash, target + draft on GPU (everything on GPU)
- f1: DFlash, target on GPU + draft on ANE (heterogeneous)

With --bg-model, runs an additional continuous workload on GPU during
the race — the Phase C contention pattern scaled to multi-stream.
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


def spawn_bg_worker(bg_model: str, bg_log: Path, ready_file: Path) -> subprocess.Popen:
    """Launch a background MLX decoder (writes bg_tokens to log).

    The bg worker writes 'READY' to bg_log once it's loaded, then runs
    continuously until killed.
    """
    bg_script = """
import sys, time
from pathlib import Path
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

log = Path(sys.argv[1])
ready = Path(sys.argv[2])
model_name = sys.argv[3]

model, tok = load(model_name)
sampler = make_sampler(temp=0.0)
ready.touch()

prompts = [
    'Explain neural networks in one paragraph:',
    'Write a haiku about the sea:',
    'The best programming language is',
]
total = 0
t0 = time.perf_counter()
try:
    while True:
        for p in prompts:
            for resp in stream_generate(model, tok, p, max_tokens=50, sampler=sampler):
                total += 1
except KeyboardInterrupt:
    pass
finally:
    elapsed = time.perf_counter() - t0
    log.write_text(f'tokens={total}\\nelapsed={elapsed}\\ntps={total/elapsed if elapsed else 0}\\n')
"""
    # Use a temp script file
    script_path = Path("/tmp/phaseG_bg_worker.py")
    script_path.write_text(bg_script)
    return subprocess.Popen(
        [PYTHON, str(script_path), str(bg_log), str(ready_file), bg_model],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )


def run_multistream(mode: str, n_streams: int, max_new: int,
                     mlmodelc: str, state_length: int,
                     results_dir: Path, prompt_id: int = 0,
                     bg_model: str | None = None) -> list[dict]:
    """Launch n_streams workers (all using the same prompt), sync start,
    collect results."""
    results_dir.mkdir(parents=True, exist_ok=True)
    barrier = results_dir / f"barrier_{mode}_{n_streams}_p{prompt_id}"
    if barrier.exists():
        barrier.unlink()
    # Clean up any stale ready files from previous runs
    for f in results_dir.glob(f"ready_{mode}_*"):
        f.unlink()

    # Spawn background workload if requested
    bg_proc = None
    bg_log = None
    bg_ready = None
    if bg_model:
        bg_log = results_dir / f"bg_{mode}_n{n_streams}.txt"
        bg_ready = results_dir / f"bg_ready_{mode}_n{n_streams}"
        if bg_ready.exists():
            bg_ready.unlink()
        print(f"[phaseG] spawning bg workload: {bg_model}", flush=True)
        bg_proc = spawn_bg_worker(bg_model, bg_log, bg_ready)
        # Wait for bg to finish loading
        t0 = time.perf_counter()
        while not bg_ready.exists() and time.perf_counter() - t0 < 120:
            time.sleep(0.5)
        if not bg_ready.exists():
            print(f"[phaseG] bg worker failed to load", flush=True)
            bg_proc.kill()
            bg_proc = None
        else:
            print(f"[phaseG] bg worker ready, running for contention", flush=True)
            time.sleep(3)  # let bg warm up

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

    # Stop bg worker and capture its throughput
    bg_tps = None
    if bg_proc is not None:
        bg_proc.send_signal(2)  # SIGINT
        try:
            bg_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            bg_proc.kill()
        if bg_log and bg_log.exists():
            for line in bg_log.read_text().splitlines():
                if line.startswith("tps="):
                    bg_tps = float(line.split("=")[1])
        print(f"[phaseG] bg tps: {bg_tps:.2f}" if bg_tps is not None
              else "[phaseG] bg tps: unavailable", flush=True)

    # Attach race-wall info + bg for summarize
    for r in results:
        r["_race_wall_s"] = t_race
        r["_true_total_tps"] = true_total_tps
        r["_bg_tps"] = bg_tps
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
    bg_tps = results[0].get("_bg_tps", None)
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
        "bg_tps": bg_tps,  # None if no bg workload
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
    ap.add_argument("--bg-model", default=None,
                    help="If set, spawn a continuous MLX decoder as background GPU load")
    args = ap.parse_args()

    all_summaries = []
    for n in args.n_streams:
        for mode in args.modes:
            print(f"\n=== {mode} × N={n} (prompt {args.prompt_id}) ===", flush=True)
            results = run_multistream(mode, n, args.max_new, args.mlmodelc,
                                        args.state_length, args.results_dir,
                                        prompt_id=args.prompt_id,
                                        bg_model=args.bg_model)
            summary = summarize(results, mode, n, prompt_id=args.prompt_id)
            all_summaries.append(summary)
            mean = summary.get('per_stream_tps_mean', 0)
            tot = summary.get('true_total_tps', 0)
            print(f"[{mode} N={n}] per-stream mean: {mean:.2f} tok/s, "
                  f"true total: {tot:.2f} tok/s", flush=True)

    # Final table
    contention_tag = f" + bg={args.bg_model.split('/')[-1]}" if args.bg_model else ""
    print(f"\n\n=== Phase G Summary (prompt {args.prompt_id}{contention_tag}) ===")
    print(f"{'mode':<10} {'N':>4} {'per-stream tok/s':<22} {'total':>8} {'bg':>8} {'eff':>8}")
    solo = {s["mode"]: s["per_stream_tps_mean"] for s in all_summaries if s["n_streams"] == 1}
    for s in all_summaries:
        if "error" in s:
            print(f"{s['mode']:<10} {s['n_streams']:>4} ERROR")
            continue
        per = s["per_stream_tps_mean"]
        true_tot = s["true_total_tps"]
        bg = s.get("bg_tps") or 0
        eff = per / solo.get(s["mode"], 1) if solo.get(s["mode"], 0) > 0 else 0
        print(f"{s['mode']:<10} {s['n_streams']:>4} {per:>6.2f} "
              f"[{s['per_stream_tps_min']:.1f}-{s['per_stream_tps_max']:.1f}]  "
              f"{true_tot:>6.2f}  {bg:>6.2f}  {eff:>6.1%}")

    out_file = args.results_dir / "summary.json"
    out_file.write_text(json.dumps(all_summaries, indent=2))
    print(f"\n[saved] {out_file}")


if __name__ == "__main__":
    main()
