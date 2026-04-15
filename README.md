# mlx-ane-sd

**Speculative decoding research on Apple Silicon — measuring what works and what
doesn't across MLX (GPU) and ANE (Neural Engine).**

Focus: characterize the speedup ceiling for speculative decoding on M-series
Macs, then test whether heterogeneous ANE + GPU parallel execution (per the
Apple [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161) paper)
can push past it.

**Tested on:** Mac mini M4 Pro, 64 GB.

## Current best (Qwen3-4B-bf16 target)

Native Swift SD runner with **full ANE offload** — draft body + draft
lm_head + full target (2×K=18 chunks) + target lm_head all on ANE; MLX
only does token embedding + final norm (`dflash-sd` binary in
`swift-bench/`):

**Mean tok/s, 4-prompt bench at max_new=100 (all decode-only):**

| config                                            | mean t/s | vs MLX bf16 baseline |
|:--------------------------------------------------|---------:|---------------------:|
| MLX bf16 baseline (no SD)                         |    29.27 |                1.00× |
| dflash-mlx (custom MLX draft)                     |    43.6  |                1.49× |
| Python F.1 (ANE draft + MLX target)               |    34.97 |                1.19× |
| Swift `dflash-sd` (matches Python F.1)            |    34.05 |                1.16× |
| + ANE LUT6 lm_head (draft-side)                   |    40.96 |                1.40× |
| + LUT6 draft body                                 |    43.05 |                1.47× |
| + K=18 partial target (**byte-identical output**) |    52.81 |                1.80× |
| + chunked full target (per_tensor LUT6)           |    55.90 |                1.91× |
| + chunked + ANE target lm_head (per_tensor)       |    62.78 |                2.14× |
| **+ chunked pgc LUT6 + ANE target lm_head** (BEST) | **64.76**|            **2.21×** |
| _(alt: 8bit target option, different quality)_    |    60.66 |                2.07× |

**Per-prompt (current best vs MLX bf16 baseline):**

| prompt    | MLX bf16 | current best | speedup |
|:----------|---------:|-------------:|--------:|
| capital   |    29.39 |        32.99 |   1.12× |
| fibonacci |    29.25 |       140.82 |   4.81× |
| math      |    29.17 |        48.48 |   1.66× |
| story     |    29.26 |        36.74 |   1.26× |

Fibonacci hits 4.8× because its draft acceptance is very high (7.6 tokens
per cycle). Prose prompts gain less (draft accepts 1.8-1.9 tokens/cycle).

**Quality trade-off**: at the byte-identical level, the K=18 partial
config (52.81 t/s, 1.80×) is strictly equivalent to the MLX bf16 target.
Beyond that, LUT6 palettization of all 36 layers introduces minor drift
on open-ended prompts (near-tie argmax flips); text stays coherent and
semantically valid.

See notes for the full journey:
- [ane_lmhead_exploration.md](./notes/ane_lmhead_exploration.md) — ANE LUT6 lm_head, +20%
- [draft_lut6_findings.md](./notes/draft_lut6_findings.md) — DFlash draft LUT6, 1.83× faster predict
- [2c_partial_target_probe.md](./notes/2c_partial_target_probe.md) — per-layer feasibility
- [2c_phase2_hybrid_target.md](./notes/2c_phase2_hybrid_target.md) — K=18 hybrid, +20%
- [2c_phase3_chunked_full_target.md](./notes/2c_phase3_chunked_full_target.md) — chunked, +5%
- [2c_phase4_full_ane.md](./notes/2c_phase4_full_ane.md) — ANE target lm_head + pgc LUT6, 2.21×
- [8bit_target_findings.md](./notes/8bit_target_findings.md) — 8bit-target alternative
- [swift_runner_bench.md](./notes/swift_runner_bench.md) — Python parity
- [swift_multistream_ane_lmhead.md](./notes/swift_multistream_ane_lmhead.md) — multi-stream

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

## Quick start — reproduce the best 2.21× result

```bash
# 0. Ensure MLX env + tools exist
#    MLX venv: /Users/carl/models/mlx-venv (Python 3.11 + mlx-lm + coremltools)
#    anemll-profile: ~/projects/anemll-profile (profiling tool)

# 1. Build the Swift runner
cd swift-bench
swift build -c release
cp $(find .build -name mlx.metallib | head -1) .build/release/

# 2. Convert/compile all ANE artifacts (one-time, ~30 min total)

# 2a. DFlash ANE draft body (accumulating cache)
python scripts/dflash_coreml_convert_accum.py --output /tmp/dflash_ane_accum.mlpackage
python scripts/dflash_lut_quantize.py \
    --input /tmp/dflash_ane_accum.mlpackage \
    --output /tmp/dflash_ane_accum_lut6.mlpackage --bits 6 --granularity per_tensor
xcrun coremlcompiler compile /tmp/dflash_ane_accum_lut6.mlpackage /tmp/dflash_ane_accum_lut6_c/

# 2b. Qwen3-4B lm_head for draft (bs=15) and target (bs=16) — ANE LUT6
python scripts/export_qwen3_lmhead_ane.py --block-size-out 15     # /tmp/lmhead_qwen3/
python scripts/export_qwen3_lmhead_ane.py --block-size-out 16 --skip-extract  # /tmp/lmhead_qwen3/bs16/

# 2c. Qwen3-4B target, 2 × K=18 chunks (per_grouped_channel LUT6)
python scripts/convert_qwen3_layers_ane.py \
    --num-layers 18 --start-layer 0 --capture-indices 1,9,17 \
    --out-dir /tmp/qwen3_klayers_cap_pgc
python scripts/convert_qwen3_layers_ane.py \
    --num-layers 18 --start-layer 18 --capture-indices 7,15 \
    --out-dir /tmp/qwen3_klayers_cap_pgc

# 3. Run the full-ANE SD stack
.build/release/dflash-sd \
    --prompt "The capital of France is Paris, which is known for" \
    --max-new 100 \
    --draft /tmp/dflash_ane_accum_lut6_c/dflash_ane_accum_lut6.mlmodelc \
    --ane-lmhead /tmp/lmhead_qwen3/lmhead_lut6.mlmodelc \
    --ane-target-layers /tmp/qwen3_klayers_cap_pgc/K18/qwen3_K18_lut6.mlmodelc \
    --ane-target-k 18 --ane-target-captures 1,9,17 \
    --ane-target-layers2 /tmp/qwen3_klayers_cap_pgc/K18_s18/qwen3_K18_lut6.mlmodelc \
    --ane-target-k2 18 --ane-target-captures2 7,15 \
    --ane-target-lmhead /tmp/lmhead_qwen3/bs16/lmhead_lut6.mlmodelc

# 4. Reproduce the full comparison bench (~15 min)
python scripts/bench_final_stack.py --max-new 100
#   -> notes/bench_final_stack.json + stdout table
```

For strict byte-identical output at 1.80× speedup, omit the `--ane-target-layers2`
and related chunk-2 flags (K=18 partial config).

### Individual benches

```bash
# Swift vs Python F.1 parity (decode-only)
python scripts/bench_sd_swift_vs_python.py --max-new 100

# Before/after ANE lm_head
python scripts/bench_ane_lmhead.py --max-new 100

# 2-stream contention
bash scripts/bench_swift_2stream.sh
bash scripts/bench_swift_2stream_ane_lmhead.sh
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
