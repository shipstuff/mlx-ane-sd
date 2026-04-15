# mlx-ane-sd

**Speculative decoding research on Apple Silicon — measuring what works and what
doesn't across MLX (GPU) and ANE (Neural Engine).**

Focus: characterize the speedup ceiling for speculative decoding on M-series
Macs, then test whether heterogeneous ANE + GPU parallel execution (per the
Apple [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161) paper)
can push past it.

**Tested on:** Mac mini M4 Pro, 64 GB.

## Current best (Qwen3-4B target)

Native Swift SD runner with ANE-hosted DFlash draft (LUT6) + ANE-hosted LUT6
lm_head + MLX/GPU target verify (`dflash-sd` binary in `swift-bench/`):

| config                                            | mean tok/s | per-cycle | vs MLX-same-precision baseline |
|:--------------------------------------------------|-----------:|----------:|-------------------------------:|
| MLX bf16 baseline (no SD)                         |       28.5 |         — |                          1.00× |
| MLX 8bit baseline (no SD)                         |       53.6 |         — |                          1.00× |
| dflash-mlx (custom MLX draft, bf16 target)        |       43.6 |         — |                          1.53× |
| Python F.1 (ANE draft + bf16 target)              |      34.97 |    102 ms |                          1.23× |
| Swift `dflash-sd` (matches Python F.1)            |      34.32 |    102 ms |                          1.20× |
| + ANE LUT6 lm_head (bf16 target)                  |      41.07 |     87 ms |                          1.44× |
| + LUT6 draft (bf16 target)                        |      43.26 |     82 ms |                          1.52× |
| **+ 8bit target** (current best)                  |  **60.66** |  **62 ms** |                      **1.13×** |

**60.66 tok/s mean** = **2.13× over MLX bf16 baseline** or **1.13× over MLX
8bit baseline**. Prompt-sensitive at 8bit: fibonacci 122 t/s (2.27× over
8bit baseline), story 33 t/s (0.62× — baseline wins). At bf16 precision
all prompts benefit from SD; at 8bit, SD gain compresses. Text byte-identical
to bf16 baseline on math; minor drift on creative prompts (near-tie argmax
flips from quantization noise).

See notes for the journey:
- [ane_lmhead_exploration.md](./notes/ane_lmhead_exploration.md) — ANE LUT6 lm_head, +20%
- [draft_lut6_findings.md](./notes/draft_lut6_findings.md) — DFlash draft LUT6, 1.83× faster predict
- [8bit_target_findings.md](./notes/8bit_target_findings.md) — 8bit target sweet spot
- [swift_multistream_ane_lmhead.md](./notes/swift_multistream_ane_lmhead.md) — multi-stream compounding

## Findings so far

### Day 0 — mlx-lm SD baselines (Gemma 3, MLX-only)

| Config                                    | num_draft | Baseline | SD     | Mean speedup |
|:------------------------------------------|----------:|---------:|-------:|-------------:|
| Gemma-3-12B **bf16** + 270M draft         |        12 |  9.1 t/s | **23.2** |    **2.45×** |
| Gemma-3-12B **4bit** + 270M draft         |         3 | 30.4 t/s |   35.5 |        1.17× |

**4bit targets are too fast on Apple Silicon for SD to help. bf16 wins.**
4bit baseline is bandwidth-bound at ~93 t/s; the draft can't keep up. bf16
baseline at ~9-30 t/s leaves room for SD to amortize. See
[notes/day0_findings.md](./notes/day0_findings.md).

### Day 1 — dflash-mlx (custom block-diffusion draft, MLX-only)

Reproduced [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) on M4 Pro:

| Config         | Baseline | DFlash | Mean speedup | Max (best prompt) |
|:---------------|---------:|-------:|-------------:|------------------:|
| Qwen3-4B bf16  | 28.5 t/s |  43.6  |    **1.53×** | 4.06× (math @ 4028 tok) |
| Qwen3-4B 4bit  | 92.9 t/s |  63.3  |  0.68× (slowdown) | 1.55× (math) |

The bf16 ceiling for *single-engine* SD on this hardware is ~3-4× on ideal
workloads, ~1.5× on average. To push past it, draft time must be hidden —
which is the heterogeneous ANE + GPU pitch.

### Phase F.1 — DFlash port to ANE (Python)

Ported `z-lab/Qwen3-4B-DFlash-b16` (the 5-layer block-diffusion draft) to
CoreML with an accumulating-then-sliding cache pattern. Key data points:

- **Solo ANE draft predict:** 9.8 ms (3.15 TOPS measured, 22.5 MB LUT-compressed
  weights, 0 ANE graph interruptions).
- **Solo F.1 mean tok/s:** 34.97 — ~85% of dflash-mlx (43.6).
- The ANE port pays a ~15% penalty in solo throughput because each draft
  predict is slower than its MLX equivalent. **Win comes under contention.**

### Swift native runner (parity with Python F.1)

Built `swift-bench/Sources/dflash-sd/` — a fully-native Swift SD loop that
loads MLX target via mlx-swift-lm and CoreML draft via the system CoreML
framework. Measured against Python F.1 on 4 prompts at max_new=100:

| prompt    | Python F.1 | Swift dflash-sd | delta |
|:----------|-----------:|----------------:|------:|
| capital   |     17.35  |          17.09  | -1.5% |
| fibonacci |     69.19  |          68.17  | -1.5% |
| math      |     34.66  |          33.35  | -3.8% |
| story     |     18.67  |          18.65  | -0.1% |
| **mean**  |   **34.97** |       **34.32** | **-1.9%** |

Within 2% across all prompts, identical accept rates and cycle counts. Output
matches byte-for-byte. The Swift runner is the deployment artifact and the
foundation for in-process multi-stream serving. See
[notes/swift_runner_bench.md](./notes/swift_runner_bench.md).

### Multi-stream contention (the Phase C pattern, in Swift)

Two parallel `dflash-sd` processes, different prompts:

| metric              | 1 stream solo | 2 streams parallel | aggregate |
|:--------------------|--------------:|-------------------:|----------:|
| per-stream tok/s    |         17.13 |              11.00 |     19.02 |
| target_verify (GPU) |       72 ms   |             124 ms |  **1.72× slowdown** |
| draft_lmhead (GPU)  |       19 ms   |              27 ms |    1.40× slowdown |
| **draft_predict (ANE)** |   **10 ms**   |          **11 ms** |   **1.12× slowdown** |

ANE preserves throughput under GPU load (Phase C pattern confirmed). 2-stream
aggregate is **1.28× solo**, not 2× — GPU is the bottleneck, ANE overlap
only saves ~10% per cycle. See
[notes/swift_multistream_findings.md](./notes/swift_multistream_findings.md).

### ANE lm_head optimization (+20% throughput)

Profile showed ANE was idle 90% of each cycle. Pushed `lm_head` from MLX/GPU
onto ANE via a separate CoreML model, LUT6-palettized to fit ANE compilation
constraints:

| variant                             | weight | placement | latency |
|:------------------------------------|-------:|:----------|--------:|
| MLX bf16 lm_head (was)              | 778 MB | GPU       |  19.5 ms |
| CoreML fp16 (compile-fail → CPU)    | 742 MB | CPU       |   6.8 ms |
| **CoreML LUT6 (per-grouped-channel, gs=16)** | **280 MB** | **100% ANE** | **3.06 ms** |

End-to-end on Qwen3-4B-bf16 with Swift `dflash-sd --ane-lmhead`:
- **Mean tok/s: 34.30 → 41.07 (+20%)**
- Per-cycle: 102 → 87 ms
- Output byte-identical to GPU baseline across all 4 prompts
- ANE utilization 10% → 15% of cycle, GPU 89% → 73%
- LUT6 quality concern (93.3% top-1 agreement vs fp32 on random hidden states)
  did **not** materialize — real Qwen3 hidden states have peaked logits where
  quantization noise rarely shifts argmax

See [notes/ane_lmhead_exploration.md](./notes/ane_lmhead_exploration.md).

## What's next

1. **Bigger draft on ANE** — 5-layer DFlash draft uses only 9.8 ms / cycle
   ANE time. There's headroom for a wider/deeper draft that trades ANE time
   for higher accept rate (better tokens/cycle scales linearly with throughput).
2. **In-process Swift multi-stream** — saves N× target weights memory vs
   multi-process. Aggregate compute won't materially exceed multi-process
   ceiling but unlocks serving density (1× model weights for N users).
3. **Partial target on ANE** — first K Qwen3 transformer layers on ANE, rest
   on GPU. Direct attack on the 72 ms target_verify ceiling. Biggest engineering
   cost; biggest possible upside.

## Quick start

```bash
# 1. Build the Swift runner
cd swift-bench
swift build -c release
cp $(find .build -name mlx.metallib | head -1) .build/release/

# 2. Compile DFlash ANE draft (one-time, ~10 min)
python scripts/dflash_coreml_convert_accum.py

# 3. (Optional) Export Qwen3-4B lm_head as ANE LUT6 (one-time, ~3 min)
python scripts/export_qwen3_lmhead_ane.py

# 4. Run the SD loop
.build/release/dflash-sd \
    --max-new 100 \
    --prompt "The capital of France is Paris, which is known for" \
    --ane-lmhead /tmp/lmhead_qwen3/lmhead_lut6.mlmodelc

# Or benchmark Swift vs Python F.1
python scripts/bench_sd_swift_vs_python.py --max-new 100

# Or benchmark with vs without ANE lm_head
python scripts/bench_ane_lmhead.py --max-new 100
```

## Layout

```
scripts/        # Python: SD sweeps, DFlash convert/quantize, benchmarks
swift-bench/    # Swift: native dflash-sd runner + ANE latency bench
  Sources/
    DFlashCore/        # Shared lib: Qwen3 inspectable, DFlash ANE wrapper, Profiler
    dflash-sd/         # Full SD loop executable
    ane-latency-bench/ # ANE predict-only latency micro-bench
    target-load-test/  # Smoke test for MLX target loading + hidden capture
notes/          # Per-phase findings, comparisons, characterization data
README.md       # this file
CLAUDE.md       # symlinked → README.md (project memory for Claude Code)
AGENTS.md       # symlinked → README.md (agent context)
```

## Environments

Two Python environments are assumed in this repo:

- **MLX venv** — Python 3.11, `mlx 0.31+`, `mlx-lm 0.31+`. All MLX-side work
  (baselines, SD sweeps, F.1 reference, LUT6 quantization).
  At `/Users/carl/models/mlx-venv/`.
- **ANEMLL venv** — Python 3.9.6, `coremltools 9.0`. Used for ANE-side
  bridging and the older anemll-qwen35 work. See
  [`Anemll/Anemll`](https://github.com/Anemll/Anemll) for install.

Swift toolchain: Swift 6.0+ via Xcode 16+. The Swift runner depends on
mlx-swift, mlx-swift-lm, swift-transformers, swift-huggingface — all pulled
via SwiftPM. Note: mlx-swift via SwiftPM does not bundle `mlx.metallib`;
copy it from your Python MLX install (see Quick start step 1).

## Models used

All public on HuggingFace.

| Role                  | Model                                  | Notes |
|:----------------------|:---------------------------------------|:------|
| Target (current)      | `mlx-community/Qwen3-4B-bf16`          | ~8 GB, tied embeddings |
| Draft (current)       | `z-lab/Qwen3-4B-DFlash-b16`            | Block-diffusion, 5 layers, 16-token block |
| Draft compiled (ANE)  | `/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc` | 1 GB, LUT-compressed by convert script |
| lm_head (ANE)         | `/tmp/lmhead_qwen3/lmhead_lut6.mlmodelc` | 280 MB, LUT6 group_size=16 |
| Day-0 target (bf16)   | `mlx-community/gemma-3-12b-it-bf16`    | ~24 GB |
| Day-0 target (4bit)   | `mlx-community/gemma-3-12b-it-4bit`    | ~7.5 GB |
| Day-0 draft           | `mlx-community/gemma-3-270m-it-bf16`   | ~550 MB |

## Related research

- [**Mirror Speculative Decoding** (Apple, 2026)](https://arxiv.org/abs/2510.13161)
  — proposes bidirectional ANE+GPU speculation with early-exit signals. No
  code released. Our ANE+MLX heterogeneous SD is a step toward testing the
  hardware claim.
- [**EAGLE-3**](https://github.com/SafeAILab/EAGLE) — published pre-trained
  draft heads for Llama, Qwen3. Their framework is incompatible with
  mlx-lm's loop.
- [**dflash-mlx**](https://github.com/Aryagm/dflash-mlx) — third-party
  block-diffusion SD on MLX. Used as a baseline + source of the DFlash
  draft architecture.
- [**DDTree**](https://github.com/liranringel/ddtree) — diffusion-style
  tree speculation. Investigated; tree attention required to materialize
  the gain — naive chain-only variant net-loses.

## Related repos in this ecosystem

- [shipstuff/anemll-qwen35](https://github.com/shipstuff/anemll-qwen35) —
  Qwen3.5 hybrid port to ANE. Provides ANE pipeline patterns reused here.
- [shipstuff/anemll-profile](https://github.com/shipstuff/anemll-profile) —
  profiling toolkit (`anemll-profile` + `ane-costplan`). Used to confirm
  ANE placement, weight bandwidth, and graph interruptions throughout
  this project.
- [Anemll/Anemll](https://github.com/Anemll/Anemll) — upstream ANE LLM
  toolchain.

## Technical gotchas

### LUT6 lm_head: real-input quality vs random-input quality

Quality test against an fp32 reference on **random** fp16 hidden states gave
93.3% top-1 argmax agreement — looked alarming. End-to-end with **real**
Qwen3 hidden states gave byte-identical text to GPU baseline. The reason:
real LM hidden states produce peaked logit distributions where the top-1 is
typically ≫ 2nd. LUT6 noise rarely shifts the argmax in practice. **Always
benchmark quality on the actual data distribution, not synthetic noise.**

### mlx-swift SwiftPM doesn't bundle the Metal library

Swift package manager builds of mlx-swift do not include `mlx.metallib`. The
runner errors with `Failed to load the default metallib` at first MLX kernel
call. Fix: copy from a working Python MLX install:

```bash
cp /Users/carl/models/mlx-venv/lib/python3.11/site-packages/mlx/lib/mlx.metallib \
   swift-bench/.build/release/mlx.metallib
```

### Why Qwen3.5 doesn't work for SD

Qwen3.5 / Qwen3-Next / Qwen3.5 MoE use hybrid GatedDeltaNet attention.
`mlx-lm`'s `speculative_generate_step` raises `ValueError: Speculative
decoding requires a trimmable prompt cache` because GatedDeltaNet's recurrent
state isn't trimmable. Use the dense Qwen3 family for SD.

### DFlash multi-function ANE variant fails on acceptance

A multi-function CoreML build (separate functions per cycle to bake position
encodings) compiled and ran but **acceptance rate collapsed to 42% of
baseline** — the baked T-aligned positions don't match DFlash's
committed-based cache advancement. Architectural mismatch with the training,
not a bug we can fix. See `notes/multifn_findings.md`.

## DO NOT

- Don't use Qwen3.5 / Qwen3-Next / Qwen3.5 MoE for SD experiments.
- Don't optimize for 4bit SD on Apple Silicon — structural dead-end.
- Don't trust single-prompt benchmark numbers — variance is huge across
  prompts. Always sweep ≥4 diverse prompts and report mean + max.
- Don't compare to dflash-mlx's headline 4.6× uncritically — that's a
  math@4028-token ideal case; their overall mean is 1.53× at bf16.
- Don't assume LUT6 quantization quality from synthetic-input tests — verify
  on real model hidden states.
- Don't expect 2× aggregate from Swift multi-stream — GPU serializes
  target_verify, ceiling is ~1.3× on this hardware.

## License

MIT
