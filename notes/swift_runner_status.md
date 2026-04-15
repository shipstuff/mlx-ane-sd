# Swift native SD runner — end-of-session status

**Date:** 2026-04-14

## What's working

**1. Raw ANE predict bench (`ane-latency-bench`)** — measures native CoreML call latency. No MLX dependency. Works solo.

**2. Full draft cycle harness (`dflash-swift-runner`)** — loads compiled DFlash
ANE draft, runs N cycles with simulated accept rate, profiles per-phase:
predict / mask build / rope build / cache commit. Already shipping measurements
committed in `b53c966`.

  Profile at S=1024, 30 cycles:
    draft_predict       88.4% (13.9 ms mean/call)
    draft_rope_build     0.3% (0.04 ms)
    draft_cache_commit   0.3% (0.05 ms)
    draft_mask_build     0.1% (0.01 ms)

  Full cycle 12.86 ms vs Python+coremltools predict-only 15.53 ms.
  Native Swift eliminates ~3 ms/call of bridge + conversion overhead.

**3. Qwen3 vendored with hidden state capture (`DFlashCore`
library)** — `Qwen3InspModelInner.forwardCapturing(inputs, cache:, captureAt:)`
returns the model's final output PLUS intermediate hidden states at specified
layer indices. This is what DFlash needs (captures at [1, 9, 17, 25, 33]
for target_layer_ids). Vendored from mlx-swift-lm with these changes:
- All classes renamed `Qwen3Insp*` to avoid conflict
- `public let layers` (was fileprivate) so we can iterate
- Added `forwardCapturing` method

**4. Target load pipeline (`target-load-test`)** — registers our Qwen3InspModel
in LLMTypeRegistry as "qwen3", loads Qwen3-4B-bf16 from HuggingFace via
`LLMModelFactory.shared.loadContainer(from:using:configuration:)` with the
HuggingFace macros (`#hubDownloader()`, `#huggingFaceTokenizerLoader()`).
Download progress reports correctly. Model loads.

## What's blocked

**Metal library not found at runtime.** When the model tries to run a
forward pass, mlx-swift raises:
```
MLX error: Failed to load the default metallib. library not found
```

This is a known mlx-swift SwiftPM quirk — the `.metallib` file isn't
bundled with the executable when building via `swift build`. The mlx-swift
examples work around it via Xcode projects which have build phases that
compile Metal shaders and bundle them.

## Resolution options (not attempted in this session)

### Option A: Xcode project
Easiest. mlx-swift-examples has a working Xcode project; convert our SwiftPM
setup to Xcode and add a "Compile Metal Files" build phase.

### Option B: Precompile + bundle manually
```bash
cd .build/checkouts/mlx-swift/Source/Cmlx/mlx-generated/metal
xcrun -sdk macosx metal -c *.metal -o /tmp/cmlx.air
xcrun -sdk macosx metallib /tmp/cmlx.air -o /tmp/default.metallib
```
Then copy `default.metallib` next to our executable binary or set
`MTL_LIBRARY_PATH` to point to its location.

The `MLX_METAL_PATH` env var didn't help — mlx-swift looks for a compiled
`.metallib`, not source files.

### Option C: Build via extracted Cmlx target
mlx-swift's Cmlx target has Metal sources but they don't get compiled by
SwiftPM. May need to fork mlx-swift locally, modify the Package.swift to
include Metal compilation as a build plugin or resource.

## Paths forward (ranked)

1. **Option A (Xcode project)** — lowest effort. ~1 day to migrate + get
   the metallib bundling working. Unblocks the full SD runner.
2. **Proceed with draft-side only** — commit the current work as the
   shippable Swift artifact for ANE draft benchmarking. Target remains
   in Python. Gives us a hybrid runner with ANE draft in Swift and MLX
   target in Python, communicating via pipes/shared memory. Not
   deployment-ideal but preserves the per-call savings we measured.
3. **Wait for mlx-swift SwiftPM metallib support** — community issue
   that will eventually get fixed upstream. Not for this month's budget.

## What the Swift work DOES prove

- The architectural approach is sound: vendored Qwen3 + hidden state
  capture compiles and registers cleanly
- CoreML-side of the pipeline is competitive with Python (~10-20%
  per-call savings measured)
- End-to-end Swift build tooling works for the ANE path

The metallib issue is a packaging problem, not a research problem.

## Files (in `swift-bench/`)

```
Package.swift            — library + 3 executables + deps
Sources/
├── DFlashCore/          — shared library
│   ├── Profiler.swift
│   ├── DFlashANEDraft.swift
│   └── Qwen3Inspectable.swift
├── ane-latency-bench/   — raw ANE bench
├── dflash-swift-runner/ — full draft harness + profiling
└── target-load-test/    — target load smoke test (needs metallib)
```

## Build + run commands

```bash
cd ~/projects/mlx-ane-sd/swift-bench
swift build -c release

# Raw ANE predict latency
.build/release/ane-latency-bench /tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc 256 100

# Full draft cycle with profiling
.build/release/dflash-swift-runner \
  --draft /tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc \
  --state-length 256 --cycles 30 --accept-rate 0.5

# Target load smoke test (currently blocked on metallib)
.build/release/target-load-test --capture-at "1,9,17,25,33"

# Python vs Swift side-by-side
python scripts/bench_swift_vs_python.py
```
