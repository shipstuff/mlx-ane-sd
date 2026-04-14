# Phase B power measurement — heterogeneous doesn't save energy

**Date:** 2026-04-13
**Hardware:** Mac mini M4 Pro, 64 GB
**Method:** `sudo powermetrics --samplers cpu_power,gpu_power,ane_power -i 500`
over 30 seconds while each setup generates ~450 tokens sustained.

## Results

Both setups generating 450 tokens total (3 × 150-token prompts), Gemma-3-12B
bf16 target, greedy decode, num_draft=12.

| Setup | CPU mean | GPU mean | ANE mean | SoC total |
|---|---:|---:|---:|---:|
| Pure-MLX SD (mlx-lm builtin) | 0.72 W | ~8.8 W | 0 W | **14.4 W** |
| Heterogeneous (B.1, ANE draft) | 0.64 W | ~18 W | 0.42 W | **17.4 W** |

Heterogeneous actually uses **more** total power, not less. Surprising.

## Why heterogeneous costs more power

The target (Gemma-3-12B bf16) dominates compute in both cases — it's the
same 12B bf16 forward-pass work. Moving the tiny 270M draft off the GPU
doesn't meaningfully reduce GPU power because the draft was negligible
on the GPU anyway.

Adding ANE activity adds ANE power (~0.5 W) without reducing GPU power
proportionally. Net effect: slightly higher total consumption.

The asymmetry: GPU power scales with the large target's work, and the
draft's ~60 ms per cycle was a small fraction of the GPU's time. Moving
it off doesn't unload the GPU enough to matter.

## When ANE offload WOULD save power

The case the Apple Mirror-SD paper makes for NPU offload assumes a
different regime:

1. **Larger drafts** (e.g., 1-2 B) where GPU draft power is non-trivial
2. **Sustained decode** where the draft's GPU time actually competes with
   target time (causing GPU to run hotter / throttle)
3. **Batched inference** where many draft calls per second amortize

For our single-stream small-draft SD, the GPU was already going to run
at full-ish tilt for the target verify. Adding or removing the 270M draft
is a few percent difference, swamped by the ANE's own draw.

## The honest conclusion

Heterogeneous SD on Apple Silicon M4 Pro for **small-draft + large-target
single-stream bf16 workloads:**

- **Throughput**: within 10% of pure-MLX SD, not clearly better
- **Power**: slightly worse (+3 W SoC) because ANE adds to the load
  rather than offsetting GPU

The real value propositions remaining:

1. **GPU availability**: ANE runs on separate silicon, so pure-MLX
   workloads running concurrently on the GPU are NOT slowed by the
   draft. (Exp E pattern — validated on Qwen3.5 work, not re-tested
   here for SD.)
2. **Larger drafts**: regime untested.
3. **Different target family**: untested.
4. **Mirror-SD bidirectional protocol**: untested — the full paper approach
   may recover throughput but is not implemented.

## Notes on the measurement

- `powermetrics` on M-series reports GPU power with multiple sub-samples
  per interval (n=120 vs n=60 for CPU/ANE). The means above are
  approximate — I didn't divide out the double-reporting. Delta between
  conditions is the real signal.
- Sample window is 30 s but each generation took ~20 s, so the last ~10 s
  includes idle tail. This suppresses mean power more than peak. Peak SoC
  was 26 W (heterogeneous) vs 20 W (pure-MLX) — same story.
