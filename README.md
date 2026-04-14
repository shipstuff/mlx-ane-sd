# mlx-ane-sd

**Speculative decoding research on Apple Silicon — measuring what works and what
doesn't across MLX (GPU) and ANE (Neural Engine).**

Focus: characterize the speedup ceiling for speculative decoding on M-series Macs,
then test whether heterogeneous ANE + GPU parallel execution (per the Apple
[Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161) paper) can push
past it.

**Tested on:** Mac mini M4 Pro, 64 GB (two machines — one for local SD sweeps, one for the dflash-mlx baseline reproduction).

## Current findings (day 0)

### Key result: bf16 is where SD wins; 4bit is a structural dead-end

Using `mlx-lm`'s built-in speculative decoding with a natural same-family draft:

| Config | num_draft | Baseline | SD | Mean speedup | Max |
|---|---:|---:|---:|---:|---:|
| Gemma-3-12B **bf16** + 270M draft | 12 | 9.1 tok/s | **23.2** | **2.45×** | 3.31× |
| Gemma-3-12B **4bit** + 270M draft | 3 | 30.4 tok/s | 35.5 | 1.17× | 1.46× |

### dflash-mlx comparison (M4 Pro, Qwen3-4B, custom block-diffusion draft)

Reproduced [Aryagm/dflash-mlx](https://github.com/Aryagm/dflash-mlx) on M4 Pro:

| Config | Baseline | DFlash | Mean speedup | Max (ideal prompt) |
|---|---:|---:|---:|---:|
| Qwen3-4B bf16 | 28.5 tok/s | 43.6 | **1.53×** | 4.06× (math @ 4028 tok) |
| Qwen3-4B 4bit | 92.9 tok/s | 63.3 | **0.68×** (net slowdown) | 1.55× (math) |

### What we learned

- **4bit targets are too fast on Apple Silicon for SD to help**, across all
  tested approaches. dflash-mlx's custom draft with specialized training gives
  only 1.55× on math (best case) and is net slower on average.
- **bf16 targets benefit meaningfully from SD**, because they're
  memory-bandwidth-bound and the draft's parallel-verify amortizes well.
- **A natural same-family draft (ours: Gemma-3-270M) with standard mlx-lm SD
  beats dflash-mlx's purpose-built draft on average** when the target is
  sufficiently large (12B vs their 4B).
- **Workload matters a lot.** dflash-mlx hits 4.06× on math prompts at 4028
  tokens but 0.35× on creative writing at 512 tokens. Acceptance swings from
  13.55 tokens/window (math) to ~2 (prose). Report mean across diverse
  prompts, not single-point numbers.
- The speedup ceiling at bf16 on a single-engine approach appears to be around
  **3-4× on ideal workloads**. To push past this, draft time has to be hidden
  — which is what heterogeneous ANE + GPU parallel execution is designed to
  provide.

See [notes/day0_findings.md](./notes/day0_findings.md) for per-prompt detail.

## Next: heterogeneous ANE + GPU speculative decoding

The hypothesis:

At `num_draft ≥ 16` in our bf16 sweep, acceptance kept climbing (80-84%) but
speedup declined because serial draft time on the GPU started dominating.
**If the draft runs on ANE in parallel with target verification on GPU, draft
time is hidden entirely.**

Predicted outcome: extend bf16 speedup from 2.45× mean toward 3-4× mean,
potentially matching dflash-mlx's best-case on average.

The plan:

1. Bridge [`anemll/anemll-gemma-3-270m-it-MONO-ctx512-lut6`](https://huggingface.co/anemll/anemll-gemma-3-270m-it-MONO-ctx512-lut6)
   (already ANE-compiled, on local disk) as the draft
2. Keep `mlx-community/gemma-3-12b-it-bf16` on MLX/GPU as the target
3. Build a speculative loop that runs draft on ANE and target verification on
   GPU concurrently
4. Measure end-to-end tok/s, compare to our pure-MLX baseline

See [notes/next_steps.md](./notes/next_steps.md) for the integration plan.

## Related research

- [**Mirror Speculative Decoding** (Apple, 2026)](https://arxiv.org/abs/2510.13161)
  — the Apple paper that motivates this work. Proposes bidirectional speculation
  on GPU + NPU with early-exit signals. No code released. Our work is a step
  toward testing the hardware claim.
- [**EAGLE-3**](https://github.com/SafeAILab/EAGLE) — published pre-trained
  draft heads for Llama, Qwen3. Not used here (requires their framework, not
  mlx-lm).
- [**dflash-mlx**](https://github.com/Aryagm/dflash-mlx) — third-party
  block-diffusion SD on MLX. Used as baseline comparison.

## Related repos in this ecosystem

- [shipstuff/anemll-qwen35](https://github.com/shipstuff/anemll-qwen35) —
  Qwen3.5 hybrid port to ANE + cross-compute experiments. Provides ANE
  pipeline patterns reused here.
- [shipstuff/anemll-profile](https://github.com/shipstuff/anemll-profile) —
  profiling toolkit (`anemll-profile` + `ane-costplan`).
- [Anemll/Anemll](https://github.com/Anemll/Anemll) — upstream ANE LLM
  toolchain.

## Quick start

Requires a Python environment with `mlx-lm 0.31+`. See [Environments](#environments).

```bash
# Gemma 3 12B bf16 + Gemma 3 270M bf16 draft (GPU-only baseline)
python scripts/sweep_num_draft_bf16.py

# 4bit sweep (same pair, confirms 4bit is a dead end for SD)
python scripts/sweep_num_draft_4bit.py

# Higher num_draft values at bf16 to find the ceiling
python scripts/sweep_num_draft_bf16_high.py
```

## Layout

```
scripts/        # benchmarks we ran (mlx-lm SD sweeps, dflash reproduction driver)
notes/          # findings, comparisons, next steps
README.md       # this file
CLAUDE.md       # symlinked to README.md (project memory)
AGENTS.md       # symlinked to README.md (agent context)
```

## Environments

Two Python environments are assumed in this repo:

- **MLX venv** — Python 3.11, `mlx 0.31+`, `mlx-lm 0.31+`. Primary for all
  MLX-side work (baselines, SD sweeps on GPU).
- **ANEMLL venv** — Python 3.9.6, `coremltools 9.0`. For ANE-side bridging.
  See [`Anemll/Anemll`](https://github.com/Anemll/Anemll) for install.

## Models used

All are public on HuggingFace. Download via `huggingface-cli download <repo>`.

| Role | Model | Notes |
|---|---|---|
| Target (bf16) | `mlx-community/gemma-3-12b-it-bf16` | ~24 GB |
| Target (4bit) | `mlx-community/gemma-3-12b-it-4bit` | ~7.5 GB |
| Draft (MLX) | `mlx-community/gemma-3-270m-it-bf16` | ~550 MB |
| Draft (ANE) | `anemll/anemll-gemma-3-270m-it-MONO-ctx512-lut6` | CoreML monolithic |

## Technical gotchas

### Why Qwen3.5 doesn't work here

Qwen3.5 models (what `shipstuff/anemll-qwen35` ports) use hybrid GatedDeltaNet
attention. `mlx-lm`'s `speculative_generate_step` raises `ValueError:
Speculative decoding requires a trimmable prompt cache` because GatedDeltaNet's
recurrent state isn't trimmable. If you want SD with Qwen, use the Qwen3
(non-3.5) dense family.

### Why Gemma 3 is the natural choice

- Dense (standard KV cache, trimmable)
- 262K vocab shared across all Gemma 3 sizes → natural same-family draft
- Gemma-3-270M **already ANE-ported** by ANEMLL as a monolithic CoreML bundle
- Gemma-3-12B available in MLX 4bit (NAS) and bf16 (downloaded)

### The ANE draft is already there

The ANEMLL-provided CoreML bundle (`gemma3_monolithic_full_lut6.mlmodelc`) is the
compiled ANE model. It uses the ANEMLL convention:

- Single monolithic model (embed + FFN + lm_head in one `.mlmodelc`)
- Tensors: `hidden_states`, `position_ids`, `causal_mask`, `current_pos`
- Functions: `infer` (decode) and `prefill`
- LUT6 + LUT4 mixed quantization
- Context length 512, batch size 64

To use it as a draft for mlx-lm's SD, we need a wrapper that:

1. Exposes an `nn.Module`-like `__call__` interface
2. Handles the cache state ourselves (ANE uses static-slice, not trimmable)
3. Rebuilds the SD verify loop outside `mlx-lm`'s `speculative_generate_step`
   since their loop assumes a trimmable MLX cache

**This is where the real engineering sits.** Everything else is plumbing.

## DO NOT

- Don't use Qwen3.5 / Qwen3-Next / Qwen3.5 MoE for SD experiments — they're
  hybrid and mlx-lm rejects the cache as non-trimmable.
- Don't optimize for 4bit SD on Apple Silicon — structural dead-end.
- Don't trust single-prompt benchmark numbers — the variance is huge
  (0.76×–3.31× range for us on the same pair, depending on prompt). Always
  sweep 5+ diverse prompts and report mean + max.
- Don't compare to dflash-mlx's headline 4.6× uncritically — that's a
  math@4028-token ideal case; their overall mean is 1.53× at bf16.

## License

MIT
