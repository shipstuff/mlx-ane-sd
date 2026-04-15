# Swift multi-stream: ANE preservation under GPU contention (confirmed)

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro, 64 GB (mini-02)
**Method:** 2 parallel `dflash-sd` processes (multi-process, not in-process)

## Result: 2-stream aggregate = 1.28× solo (not 2×), confirms GPU serialization

Two Swift dflash-sd processes, different prompts, spawned in parallel:

| metric              | 1 stream solo | 2 streams parallel |
|:--------------------|--------------:|-------------------:|
| per-stream tok/s    |         17.13 |              11.00 |
| aggregate tok/s     |         17.13 |          **19.02** |
| aggregate speedup   |          1.00×|          **1.28×** |

## Per-phase behavior under contention

This is the interesting part — different hardware backends degrade differently:

| phase             | solo ms/cyc | 2-stream ms/cyc | slowdown |
|:------------------|------------:|----------------:|---------:|
| target_verify     |        72.2 |          ~124   |  **1.72×** |
| draft_lmhead      |        19.3 |           ~27   |    1.40× |
| draft_predict     |        10.1 |          ~11.3  |  **1.12×** |

**`draft_predict` on ANE barely degrades (1.12×) while `target_verify` on GPU
slows 1.72×.** This is the Phase C preservation pattern — ANE is a separate
hardware block and its throughput is maintained even when the GPU is fully
contended.

`draft_lmhead` (on GPU) slows 1.40× rather than 1.72× because some of its work
overlaps with the other stream's draft_predict (which runs on ANE), so GPU has
idle windows that get filled.

## Why 1.28× and not 1.11× (my theoretical model predicted)

Per-cycle GPU busy = target_verify (72) + draft_lmhead (19) = 91ms. Per-cycle
ANE busy = 10ms. If GPU fully serializes and ANE fully overlaps:

- 2-stream GPU work = 182ms (serialized)
- 2-stream ANE work = 20ms (overlapped, fits in GPU's serial window)
- 2-stream cycle = max(182, 20 + something) = ~182ms for 2 cycles
- Theoretical aggregate = 2 × 102 / 182 = 1.12×

Measured 1.28× is **better than the theoretical bound**. Best explanation:
some GPU work overlaps between streams through MLX's command-queue scheduling
— the GPU doesn't strictly serialize kernels, it can interleave at smaller
granularity. So the GPU slowdown factor is 1.72× not 2.0×.

## Comparison with Python Phase G (multiprocessing)

Phase G already showed similar multi-stream behavior with Python. The Swift
result confirms:

1. The hardware contention pattern (GPU serializes, ANE preserves) is the
   **physics**, not a Python or Swift implementation artifact.
2. Going to Swift multi-process (vs Python multi-process) doesn't unlock more
   GPU throughput. The ceiling is the GPU.

## What in-process Swift multi-stream could add

Moving from multi-process to in-process (shared weights, shared model loader)
saves **memory** and **startup latency**, not compute:

| per-stream overhead  | multi-process | in-process (projected) |
|:---------------------|:--------------|:-----------------------|
| target weights       | 8 GB × N      | 8 GB × 1               |
| model load time      | 700 ms × N    | 700 ms × 1             |
| compute ceiling      | GPU-bound     | same                   |

For serving N users, in-process is the right architecture. But aggregate tok/s
won't materially exceed 1.28× solo at N=2, and will asymptote as N grows.

**This is the honest story** — the multi-stream value is memory density and
startup amortization, not throughput.

## Next

Option A: build in-process Swift multi-stream to measure memory/latency benefit
for serving, knowing throughput won't materially improve.

Option B: accept the ceiling, focus on pushing **per-stream** performance:
- larger block size (wider speculation window)
- better draft quality (more tokens accepted per cycle)
- target quantization to fit more streams in memory at same quality

Option C: write up characterization paper with the data we have. The thesis
"ANE+GPU heterogeneous SD pushes past single-engine ceiling" needs qualification:

- At **saturation** (N streams, each using GPU fully), ANE+GPU aggregate is
  ~1.28× single-stream on this hardware.
- ANE preservation under GPU load is real (1.12× slowdown vs GPU's 1.72×).
- But the 72ms target_verify dominates per-cycle, so the ANE window is small.

## Reproduction

```bash
cd swift-bench && swift build -c release
bash /tmp/swift_2stream.sh  # script uses dflash-sd at /tmp
```
