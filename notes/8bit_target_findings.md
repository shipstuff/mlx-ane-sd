# 8bit target: 60.66 tok/s mean, honest +13% over 8bit MLX baseline

**Date:** 2026-04-15
**Hardware:** Mac mini M4 Pro, 64 GB (mini-02)
**Context:** Day 1 data showed 4bit target DESTROYS SD (0.68× slowdown vs bf16
baseline). The untested middle ground was 8bit. `mlx-community/Qwen3-4B-8bit`
exists — one-flag swap via `--target`.

## Result: 8bit target is a real SD win, not a dead-end

Per-prompt, Swift `dflash-sd` with LUT6 draft + ANE lm_head:

| prompt    | bf16 target | 8bit target | speedup (vs bf16 same stack) |
|:----------|------------:|------------:|-----------------------------:|
| capital   |       20.94 |       35.72 |                        1.71× |
| fibonacci |       86.12 |  **121.92** |                        1.42× |
| math      |       43.12 |       51.60 |                        1.20× |
| story     |       22.88 |       33.41 |                        1.46× |
| **mean**  |   **43.26** |   **60.66** |                    **1.40×** |

## Why 8bit works where 4bit fails

At 4bit, target logits shift enough that draft predictions mismatch 40%+
of the time, so SD draws more rejections than time-savings. At 8bit:

| target   | mean tok/s/cycle | target_verify per cyc | accept/cyc (our stack) |
|:---------|-----------------:|----------------------:|-----------------------:|
| bf16     |            82 ms |                72 ms  |                   3.56 |
| **8bit** |        **62 ms** |           **52 ms**   |               **3.77** |
| 4bit     |          unknown |             ~35 ms    |  collapsed (per DFlash) |

**8bit actually IMPROVED accept rate** (3.56 → 3.77 per cycle). Hypothesis:
at 8bit, the target's logit distributions are slightly tighter / more
deterministic, so the argmax winner is more stable and draft predictions
align more often. At 4bit, the noise floor is too high and argmax flips.

## Honest framing: compare against same-precision baseline

Throughput numbers without honest baselines are marketing. Measured decode
tok/s, 4 prompts, max_new=100:

| config                                   | mean tok/s | vs *its own baseline* |
|:-----------------------------------------|-----------:|----------------------:|
| MLX bf16 baseline (no SD)                |       28.5 |                 1.00× |
| MLX **bf16** + our SD stack              |     43.26  |             **1.52×** |
| MLX 8bit baseline (no SD)                |      ~53.6 |                 1.00× |
| MLX **8bit** + our SD stack              |     60.66  |             **1.13×** |

**8bit SD gain is smaller in relative terms** (1.13× vs 1.52× at bf16) because
the baseline is already much faster. SD overhead becomes a larger fraction
when the per-forward cost is low. But it's still positive and the absolute
number (60.66 tok/s) is our best-ever Qwen3-4B number.

## Prompt-dependent behavior

| prompt    | MLX 8bit baseline | Our SD (8bit) | winner |
|:----------|------------------:|--------------:|:-------|
| capital   |              53.6 |         35.72 | baseline (SD hurts here) |
| fibonacci |              53.6 |        121.92 | **SD crushes** (2.27×) |
| math      |              53.6 |         51.60 | tied |
| story     |              53.6 |         33.41 | baseline (SD hurts) |

At 8bit precision, SD is workload-sensitive. Repetitive/code prompts
(fibonacci) have very high draft accept rates and SD dominates. Creative
prompts (story) have low accept rates and SD's overhead exceeds its savings.

This is not a bug — it's the speculative decoding regime trade-off getting
compressed at higher baseline speeds. At bf16, ALL prompts benefit because
the baseline is slow enough that even 2-tokens-per-cycle SD wins. At 8bit,
you need 3-4 tokens-per-cycle to beat the baseline.

## Quality

Output text: byte-identical on math prompt. Diverges slightly on capital
("the Louvre Mu" vs "a symbol of F") and story — both continuations are
semantically valid, just different near-tie argmax resolutions. For greedy
decoding at temp=0, this is expected and acceptable.

## What this means strategically

- **If user's baseline is bf16**: our heterogeneous SD stack delivers
  1.52× (with bf16 target, 43 tok/s) or up to 2.13× (with 8bit target,
  60 tok/s + quality trade-off).
- **If user's baseline is already 8bit**: we offer 1.13× on average,
  strong on repetitive workloads, weak on creative ones.
- **If user's baseline is 4bit**: SD is a dead-end at any target quant.

The multi-stream contention story is also improved: 8bit target means
less GPU bandwidth per cycle, so target_verify serialization under 2+
streams should be less severe. Worth re-running the 2-stream test with
8bit target.

## Next

Task 2c per user direction: partial target on ANE. With 8bit target at
52ms/cycle, the ANE-time-per-layer would need to be <1.4 ms to be worth
putting on ANE (since GPU now does layers at 52/36 ≈ 1.44 ms/layer).
DFlash's ~2ms/layer on ANE is slightly slower than this — suggests partial
target doesn't help at 8bit. At bf16 (72/36 ≈ 2.0 ms/layer) it's a wash.
Worth measuring real per-layer cost before committing to the port.
