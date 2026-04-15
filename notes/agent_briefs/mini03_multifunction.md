# Agent brief: multi-function ANE variant (mini-03)

**For:** agent running on mini-03
**Duration:** ~3-5 days
**Converges with:** Week 2 (compared vs S=2048 sweet spot)

## Mission

Build the "proper" multi-function ANE variant that we deferred in F.3.
Compile N versions of the DFlash model with different baked-in cache
write positions, dispatch the right one at runtime. Goal: eliminate
the sliding-window fallback that caps acceptance at long generations
(>1500 tokens), while keeping per-call latency close to S=1024.

## Background

Current state: F.1 accum uses Python-side cache management. When
`write_pos + T > STATE_LENGTH`, we shift-left + append (sliding window).
This loses cache history. At 1000+ tokens, acceptance degrades.

The multi-function fix compiles N variants of the same model, each with
a DIFFERENT baked-in write position (the slice bounds are static per
variant). Python dispatches:
- Variant 0 (write_pos=0): used for cycle 0
- Variant 1 (write_pos=T): used for cycle 1
- ...
- Variant N-1: used for cycle N-1

When write_pos exceeds max baked position, fall back to sliding (but
this happens much later with N=32+ variants).

## Scope

- Build on `scripts/dflash_ane_accumcache.py` — the current 100%-ANE variant.
- Create `scripts/dflash_ane_multifn.py`: parameterize write_pos at trace time.
- Create `scripts/dflash_coreml_convert_multifn.py`: convert N versions,
  bundle into one .mlpackage with multi-function support (see
  coremltools multi-function API).
- Runtime wrapper dispatches based on `self.write_pos % N`.

## Reference materials

- `notes/phaseF3_cache_sizing.md` — explains the sliding-fallback issue
- `~/projects/anemll/anemll/ane_converter/qwen_converter.py` — ANEMLL's
  multi-function pattern (see `convert_part_2` and `convert_part_2_prefill`
  — they're separate compiled functions in one .mlpackage)
- coremltools docs:
  [Multi-Function Models](https://apple.github.io/coremltools/docs-guides/source/mlprogram-multifunction.html)

## Approach

1. **Parameterize write_pos as a compile-time constant.**
   In the model's `__init__`, take a `write_pos` arg. The attention layer
   writes to `cache[:, :, write_pos:write_pos+T, :]` where `write_pos` is
   now a Python int (known at trace time → static slice).

2. **Compile N variants.**
   For N=32 (covering up to 32 cycles before sliding), compile 32 versions:
   `write_pos ∈ {0, 32, 64, ..., 992}` assuming T=32, STATE_LEN=1024.

3. **Multi-function mlpackage.**
   Use coremltools multi-function API to bundle all 32 as separate
   `function_name` entries in one .mlpackage. Naming: `write_0`, `write_32`,
   ..., `write_992`.

4. **Runtime dispatch.**
   At inference, pick `function_name = f"write_{write_pos}"`. All variants
   share the same weights (coremltools handles this), so disk cost is
   still ~1 GB even for 32 variants.

5. **Validate.**
   Benchmark at max_new ∈ {100, 500, 1000, 2000} and compare to S=1024
   accum and S=2048 accum.

## Expected outcome

At max_new=1000, multi-function S=1024 should:
- Match or beat S=2048 accum (25.71 tok/s) — because no sliding
- Have lower per-call latency (16ms like S=1024) than S=2048 (24ms)

If the multi-function variant clearly wins, it becomes the paper's
recommended config.

## Deliverables

1. `scripts/dflash_ane_multifn.py` + `scripts/dflash_coreml_convert_multifn.py`
2. `scripts/phaseF1_ane_stream_multifn.py` — SD runner
3. `notes/phaseF1_multifunction_results.md` — benchmark table vs S=1024/2048
4. Compiled mlpackage saved somewhere durable (e.g., `/tmp/dflash_ane_multifn.mlpackage`)

## Known risks

- coremltools multi-function may have limits on number of functions
  per package (shouldn't be an issue for 32, but check).
- Shared weights across functions may not be supported — if each variant
  requires its own weight copy, disk cost grows. Check with a small test
  (N=2) before going to N=32.
- If multi-function compile fails or has placement issues, fall back to
  N separate compiled mlpackages with Python-side orchestration.

## Stop-and-ask triggers

- Multi-function compilation fails on any variant
- Benchmark numbers are worse than current S=2048 (meaning the
  architecture change isn't paying off)
- Disk cost explodes beyond ~5 GB (probably means weights aren't shared)

## Out of scope

Don't try to also do LUT quantization in the same iteration — combine
those only after base multi-function is validated.
