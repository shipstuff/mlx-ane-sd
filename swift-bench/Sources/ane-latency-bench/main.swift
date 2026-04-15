// Measure raw ANE call latency via native CoreML (no Python).
// Compare to Python+coremltools baseline for our DFlash accum model.

import CoreML
import Foundation

// Constants matching our compiled model
let BS = 16       // block_size
let CS = 16       // ctx_size
let H = 2560      // hidden_size
let CONCAT_DIM = 12800  // num_target_layers * hidden = 5 * 2560
let DH = 128      // head_dim
let N_LAYERS = 5
let HKV = 8       // num_key_value_heads
let T = CS + BS   // 32

func makeArray(_ shape: [Int], dtype: MLMultiArrayDataType = .float16) throws -> MLMultiArray {
    let nsShape = shape.map { NSNumber(value: $0) }
    return try MLMultiArray(shape: nsShape, dataType: dtype)
}

func fillRandom(_ arr: MLMultiArray) {
    // Simple deterministic fill — zeros with a tiny bit of noise.
    // Random pattern doesn't matter for latency measurement.
    let count = arr.count
    // MLMultiArray dataPointer is raw. For fp16, each element is 2 bytes.
    let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
    for i in 0..<count {
        ptr[i] = UInt16(i & 0xFFFF)
    }
}

@main
struct Main {
    static func main() async throws {
        guard CommandLine.arguments.count >= 2 else {
            print("Usage: ane-latency-bench <path-to-mlmodelc> [state_length] [N_iters]")
            exit(1)
        }
        let modelPath = CommandLine.arguments[1]
        let stateLen = CommandLine.arguments.count >= 3 ? Int(CommandLine.arguments[2]) ?? 256 : 256
        let iters = CommandLine.arguments.count >= 4 ? Int(CommandLine.arguments[3]) ?? 100 : 100

        print("[bench] loading \(modelPath), state_len=\(stateLen), iters=\(iters)")
        let url = URL(fileURLWithPath: modelPath)
        let config = MLModelConfiguration()
        config.computeUnits = .cpuAndNeuralEngine
        let loadStart = CFAbsoluteTimeGetCurrent()
        let model = try MLModel(contentsOf: url, configuration: config)
        let loadTime = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        print("[bench] loaded in \(String(format: "%.1f", loadTime)) ms")

        // Build inputs matching our DFlash accum model
        let attendLen = stateLen + T  // 288 for S=256
        let noise = try makeArray([1, BS, H])
        fillRandom(noise)
        let ctx = try makeArray([1, CS, CONCAT_DIM])
        fillRandom(ctx)
        let cosQ = try makeArray([BS, DH])
        fillRandom(cosQ)
        let sinQ = try makeArray([BS, DH])
        fillRandom(sinQ)
        let cosK = try makeArray([T, DH])
        fillRandom(cosK)
        let sinK = try makeArray([T, DH])
        fillRandom(sinK)
        let cacheK = try makeArray([N_LAYERS, HKV, stateLen, DH])
        fillRandom(cacheK)
        let cacheV = try makeArray([N_LAYERS, HKV, stateLen, DH])
        fillRandom(cacheV)
        let mask = try makeArray([1, 1, BS, attendLen])
        fillRandom(mask)

        let features: [String: MLFeatureValue] = [
            "noise_embedding": MLFeatureValue(multiArray: noise),
            "target_hidden": MLFeatureValue(multiArray: ctx),
            "cos_q": MLFeatureValue(multiArray: cosQ),
            "sin_q": MLFeatureValue(multiArray: sinQ),
            "cos_k": MLFeatureValue(multiArray: cosK),
            "sin_k": MLFeatureValue(multiArray: sinK),
            "cache_K": MLFeatureValue(multiArray: cacheK),
            "cache_V": MLFeatureValue(multiArray: cacheV),
            "causal_mask": MLFeatureValue(multiArray: mask),
        ]
        let input = try MLDictionaryFeatureProvider(dictionary: features)

        // Warmup
        print("[bench] warmup (5 iters)...")
        for _ in 0..<5 {
            _ = try await model.prediction(from: input)
        }

        // Timed runs
        print("[bench] timing \(iters) iters...")
        var samples: [Double] = []
        samples.reserveCapacity(iters)
        for _ in 0..<iters {
            let t0 = CFAbsoluteTimeGetCurrent()
            _ = try await model.prediction(from: input)
            let elapsed = (CFAbsoluteTimeGetCurrent() - t0) * 1000
            samples.append(elapsed)
        }

        samples.sort()
        let mean = samples.reduce(0, +) / Double(samples.count)
        let median = samples[samples.count / 2]
        let p10 = samples[samples.count / 10]
        let p90 = samples[(samples.count * 9) / 10]
        let stdev: Double = {
            let variance = samples.map { pow($0 - mean, 2) }.reduce(0, +) / Double(samples.count)
            return sqrt(variance)
        }()

        print("")
        print("=== Swift ANE call latency (state_len=\(stateLen)) ===")
        print(String(format: "  mean:   %.3f ms", mean))
        print(String(format: "  median: %.3f ms", median))
        print(String(format: "  stdev:  %.3f ms", stdev))
        print(String(format: "  p10:    %.3f ms", p10))
        print(String(format: "  p90:    %.3f ms", p90))
        print(String(format: "  min:    %.3f ms", samples.first ?? 0))
        print(String(format: "  max:    %.3f ms", samples.last ?? 0))
    }
}
