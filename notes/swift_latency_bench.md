# Swift native ANE latency bench (Week 2 start)

**Date:** 2026-04-14
**Hardware:** Mac mini M4 Pro (mini-02), 64 GB

## Measurement

Same compiled DFlash draft, same deterministic fp16 inputs, same
compute units (CPU_AND_NE). Python loads via `coremltools.models.
CompiledMLModel`; Swift loads via native `MLModel(contentsOf:)`.

| State length | Python (ms) | Swift (ms) | Savings | Savings % | Per 100 tok |
|---:|---:|---:|---:|---:|---:|
| 256 | 10.74 | **9.71** | 1.02 ms | 9.5% | ~26 ms |
| 1024 | 15.53 | **12.79** | 2.74 ms | 17.6% | ~68 ms |

Swift latency is closer to raw ANE compute time (from anemll-profile:
8.57 ms at S=256, 13 ms at S=1024). The Python+coremltools bridge
adds ~1-3 ms of per-call overhead depending on input size; Swift
eliminates most of it.

## Translation to end-to-end throughput

A 100-token generation in F.1 at S=1024 has ~25 cycles, ~3 sec wall.
Saving 68 ms per generation = **~2.3% solo speedup**. For S=256 with
shorter cycles, 26 ms saved per 100 tok = ~1% speedup.

Net: Swift is a real but modest win on solo throughput. The original
hypothesis ("Swift runner could flip F.1 ≥ F.0 solo") doesn't pan out —
we'd need ~10ms/cycle saved to close a 15-20% gap, and the bridge is
only ~3ms.

## Where the Swift runner would still pay off

1. **Multi-stream serving.** Each Python process carries a full
   interpreter (~100 MB) and mlx-lm import cost (~3-5s startup). Swift
   runner could do N concurrent streams in one process with shared
   target weights (not N × 8 GB duplicates) and no GIL serialization.
   Realistic estimate: 20-30% multi-stream throughput gain vs our
   current Python process-per-stream approach. Plus N=8+ becomes
   feasible on 64 GB.

2. **Deployment artifact.** Ship a standalone CLI or macOS app.
   Python + coremltools + mlx-lm + mlx is a heavy dependency tree;
   Swift binary is self-contained.

3. **iOS / edge use cases** (aspirational): the Python stack doesn't
   ship to iOS. A Swift runner does.

## What's in the repo so far

- `swift-bench/Package.swift` — Swift package targeting macOS 15
- `swift-bench/Sources/ane-latency-bench/main.swift` — minimal ANE
  predict latency measurement
- `swift-bench/Sources/dflash-swift-runner/main.swift` — stub for the
  full SD runner (scope for next iteration: mlx-swift-lm Qwen3 target,
  hidden state capture, SD loop with accept/trim)
- `scripts/bench_swift_vs_python.py` — side-by-side comparison harness
  (what produced the table above)

## Status

Week 2 target (flip F.1 solo ≥ F.0 solo via Swift) is **unlikely to be
achievable** based on these measurements — the Python overhead isn't
large enough. Adjust Week 2 scope:

- **Deprioritize solo throughput race**: it was going to be 2-5% gain
  at best, not 15-20%.
- **Prioritize multi-stream Swift runner**: the shared-process-with-one-
  target-copy pattern is where meaningful gains hide. Also the
  shippable artifact value justifies the work even if individual
  speedup is modest.
- **Target for Week 2**: a Swift CLI that runs N concurrent SD streams
  in one process, sharing target weights and the ANE draft. Benchmark
  vs our current Python process-per-stream (multistream_worker.py).

Full SD loop implementation is ~3-5 days of engineering on top of what's
committed. mlx-swift-lm provides the target loader / tokenizer / forward
pass; we'd need to fork or patch its Qwen3 implementation to expose
hidden states at target_layer_ids (not currently a public API).
