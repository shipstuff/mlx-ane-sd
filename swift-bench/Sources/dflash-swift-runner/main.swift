// DFlash speculative decoding runner in Swift (scaffold, Week 2 WIP).
//
// Scope for this commit:
//   - Load the CoreML DFlash draft (works, ~10ms/call on ANE)
//   - Run N timed predict calls via the Swift CoreML API (no Python overhead)
//   - Output structured JSON so a Python wrapper can drive this as a subprocess
//
// Next iteration (not yet implemented):
//   - Load Qwen3-4B target via mlx-swift-lm (requires Hub macro integration)
//   - Hook hidden state capture at target_layer_ids (requires patching the
//     mlx-swift-lm Qwen3 model or a custom forward)
//   - Full SD loop with accept/trim logic
//   - Multi-stream orchestration in one process (the potential win for serving)
//
// Why this shape: the ANE call is the measured bottleneck we can improve with
// Swift (3-4 ms saved per call at S=1024 vs Python). Getting this part
// production-ready and measurable gives us a solid foundation; the target
// side is a larger (but mechanically straightforward) port.

import Foundation
import CoreML
import ArgumentParser

let BS = 16
let CS = 16
let H = 2560
let CONCAT_DIM = 12800
let DH = 128
let N_LAYERS = 5
let HKV = 8
let T = CS + BS

func makeArray(_ shape: [Int], dtype: MLMultiArrayDataType = .float16) throws -> MLMultiArray {
    let nsShape = shape.map { NSNumber(value: $0) }
    return try MLMultiArray(shape: nsShape, dataType: dtype)
}

func fillDeterministic(_ arr: MLMultiArray) {
    let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
    for i in 0..<arr.count {
        ptr[i] = UInt16(i & 0xFFFF)
    }
}

struct BenchResult: Codable {
    let mean_ms: Double
    let median_ms: Double
    let p10_ms: Double
    let p90_ms: Double
    let stdev_ms: Double
    let min_ms: Double
    let max_ms: Double
    let n_iters: Int
    let state_length: Int
    let load_time_ms: Double
}

@main
struct DFlashRunner: AsyncParsableCommand {
    @Option(help: "Path to compiled DFlash draft .mlmodelc")
    var draft: String = "/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc"

    @Option(help: "State length matching the compiled draft")
    var stateLength: Int = 256

    @Option(help: "Number of timed iterations after warmup")
    var iters: Int = 100

    @Flag(help: "Output structured JSON instead of human-readable")
    var json: Bool = false

    func run() async throws {
        if !json {
            print("[runner] loading draft \(draft) (S=\(stateLength))...")
        }

        let url = URL(fileURLWithPath: draft)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let loadStart = CFAbsoluteTimeGetCurrent()
        let model = try MLModel(contentsOf: url, configuration: config)
        let loadTimeMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        if !json {
            print("[runner] loaded in \(String(format: "%.1f", loadTimeMs)) ms")
        }

        let attendLen = stateLength + T
        let inputs = try buildInputs(attendLen: attendLen)

        // Warmup
        for _ in 0..<5 {
            _ = try await model.prediction(from: inputs)
        }

        // Timed
        var samples: [Double] = []
        samples.reserveCapacity(iters)
        for _ in 0..<iters {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try await model.prediction(from: inputs)
            samples.append((CFAbsoluteTimeGetCurrent() - t0) * 1000)
        }

        samples.sort()
        let mean = samples.reduce(0, +) / Double(samples.count)
        let variance = samples.map { pow($0 - mean, 2) }.reduce(0, +) / Double(samples.count)
        let stdev = sqrt(variance)
        let result = BenchResult(
            mean_ms: mean,
            median_ms: samples[samples.count / 2],
            p10_ms: samples[samples.count / 10],
            p90_ms: samples[(samples.count * 9) / 10],
            stdev_ms: stdev,
            min_ms: samples.first ?? 0,
            max_ms: samples.last ?? 0,
            n_iters: iters,
            state_length: stateLength,
            load_time_ms: loadTimeMs
        )

        if json {
            let data = try JSONEncoder().encode(result)
            print(String(data: data, encoding: .utf8) ?? "{}")
        } else {
            print("")
            print("=== Swift ANE DFlash draft latency (S=\(stateLength)) ===")
            print(String(format: "  mean:   %.3f ms   stdev: %.3f ms", mean, stdev))
            print(String(format: "  median: %.3f ms", result.median_ms))
            print(String(format: "  p10/p90: %.3f / %.3f ms", result.p10_ms, result.p90_ms))
            print(String(format: "  min/max: %.3f / %.3f ms", result.min_ms, result.max_ms))
        }
    }

    func buildInputs(attendLen: Int) throws -> MLDictionaryFeatureProvider {
        let noise = try makeArray([1, BS, H])
        fillDeterministic(noise)
        let ctx = try makeArray([1, CS, CONCAT_DIM])
        fillDeterministic(ctx)
        let cosQ = try makeArray([BS, DH])
        fillDeterministic(cosQ)
        let sinQ = try makeArray([BS, DH])
        fillDeterministic(sinQ)
        let cosK = try makeArray([T, DH])
        fillDeterministic(cosK)
        let sinK = try makeArray([T, DH])
        fillDeterministic(sinK)
        let cacheK = try makeArray([N_LAYERS, HKV, stateLength, DH])
        fillDeterministic(cacheK)
        let cacheV = try makeArray([N_LAYERS, HKV, stateLength, DH])
        fillDeterministic(cacheV)
        let mask = try makeArray([1, 1, BS, attendLen])
        fillDeterministic(mask)

        return try MLDictionaryFeatureProvider(dictionary: [
            "noise_embedding": MLFeatureValue(multiArray: noise),
            "target_hidden": MLFeatureValue(multiArray: ctx),
            "cos_q": MLFeatureValue(multiArray: cosQ),
            "sin_q": MLFeatureValue(multiArray: sinQ),
            "cos_k": MLFeatureValue(multiArray: cosK),
            "sin_k": MLFeatureValue(multiArray: sinK),
            "cache_K": MLFeatureValue(multiArray: cacheK),
            "cache_V": MLFeatureValue(multiArray: cacheV),
            "causal_mask": MLFeatureValue(multiArray: mask),
        ])
    }
}
