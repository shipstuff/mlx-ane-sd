// Qwen3 K-layer ANE wrapper.
//
// Loads a CoreML model containing K chained Qwen3 transformer blocks
// (layers 0..K-1 of Qwen3-4B). Manages external KV cache in an
// accumulating-then-sliding pattern — identical to DFlashANEDraft but
// with multi-layer K/V cache shape [K, 1, 8, state_len, 128].
//
// Usage in SD loop:
//   let ane = try Qwen3ANELayers(...)
//   // After prefill, sync cache from MLX target:
//   try ane.loadFromPrefill(mlxCacheKV: [...])
//   // Each target_verify cycle:
//   let (hidden, captures) = try await ane.forward(hidden, positionOffset: pos)
//   // After accept_check:
//   ane.commit(accepted: n)
//   // On rejection trim:
//   ane.trim(by: m)

import Foundation
import CoreML

public struct Qwen3ANEConfig: Sendable {
    public let numLayers: Int
    public let blockSize: Int
    public let stateLength: Int
    public let hiddenSize: Int
    public let numKVHeads: Int
    public let headDim: Int
    public let ropeTheta: Float
    public let captureIndices: [Int]  // 0-based within this K-layer slice

    public var attendLen: Int { stateLength + blockSize }

    public init(numLayers: Int, blockSize: Int = 16, stateLength: Int = 256,
                hiddenSize: Int = 2560, numKVHeads: Int = 8, headDim: Int = 128,
                ropeTheta: Float = 1_000_000,
                captureIndices: [Int] = []) {
        self.numLayers = numLayers
        self.blockSize = blockSize
        self.stateLength = stateLength
        self.hiddenSize = hiddenSize
        self.numKVHeads = numKVHeads
        self.headDim = headDim
        self.ropeTheta = ropeTheta
        self.captureIndices = captureIndices
    }
}

public final class Qwen3ANELayers: @unchecked Sendable {
    public let config: Qwen3ANEConfig
    private let model: MLModel

    // External KV cache: [K, 1, 8, state_len, 128]
    private var cacheK: MLMultiArray
    private var cacheV: MLMultiArray
    public private(set) var writePos: Int = 0
    public private(set) var globalOffset: Int = 0

    private let invFreq: [Float]

    public var profiler: Profiler?

    public init(mlmodelcPath: String, config: Qwen3ANEConfig,
                profiler: Profiler? = nil) throws {
        self.config = config
        self.profiler = profiler

        let url = URL(fileURLWithPath: mlmodelcPath)
        let mlconfig = MLModelConfiguration()
        mlconfig.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: url, configuration: mlconfig)

        let cacheShape: [NSNumber] = [
            NSNumber(value: config.numLayers),
            1,
            NSNumber(value: config.numKVHeads),
            NSNumber(value: config.stateLength),
            NSNumber(value: config.headDim),
        ]
        self.cacheK = try MLMultiArray(shape: cacheShape, dataType: .float16)
        self.cacheV = try MLMultiArray(shape: cacheShape, dataType: .float16)

        // RoPE inverse frequencies
        var freqs = [Float]()
        freqs.reserveCapacity(config.headDim / 2)
        for i in stride(from: 0, to: config.headDim, by: 2) {
            freqs.append(1.0 / pow(config.ropeTheta, Float(i) / Float(config.headDim)))
        }
        self.invFreq = freqs
    }

    public func resetCache() {
        memset(cacheK.dataPointer, 0, cacheK.count * 2)
        memset(cacheV.dataPointer, 0, cacheV.count * 2)
        writePos = 0
        globalOffset = 0
    }

    /// Load pre-populated K/V values from an MLX-style prefill.
    /// `layerKeys[i]` and `layerValues[i]` are fp16 arrays of shape
    /// `[1, numKVHeads, promptLen, headDim]` already having RoPE applied.
    /// Sets writePos = promptLen, globalOffset = promptLen.
    public func loadFromPrefill(layerKeys: [MLMultiArray], layerValues: [MLMultiArray],
                                  promptLen: Int) throws {
        precondition(layerKeys.count == config.numLayers)
        precondition(layerValues.count == config.numLayers)
        precondition(promptLen <= config.stateLength, "prompt longer than state_length not yet supported")

        resetCache()

        let H = config.numKVHeads
        let Dh = config.headDim
        let S = config.stateLength
        let cacheKPtr = cacheK.dataPointer.assumingMemoryBound(to: UInt16.self)
        let cacheVPtr = cacheV.dataPointer.assumingMemoryBound(to: UInt16.self)

        for layer in 0..<config.numLayers {
            let kPtr = layerKeys[layer].dataPointer.assumingMemoryBound(to: UInt16.self)
            let vPtr = layerValues[layer].dataPointer.assumingMemoryBound(to: UInt16.self)
            for head in 0..<H {
                // src: [1, H, promptLen, Dh] indexed [0, head, p, d]
                // dst: [K, 1, H, state_len, Dh] indexed [layer, 0, head, p, d]
                let dstOffset = layer * 1 * H * S * Dh + 0 + head * S * Dh
                let srcOffset = head * promptLen * Dh
                let bytes = promptLen * Dh * 2
                memcpy(cacheKPtr.advanced(by: dstOffset),
                       kPtr.advanced(by: srcOffset),
                       bytes)
                memcpy(cacheVPtr.advanced(by: dstOffset),
                       vPtr.advanced(by: srcOffset),
                       bytes)
            }
        }
        writePos = promptLen
        globalOffset = promptLen
    }

    /// Output from one forward call.
    public struct Output {
        public let hidden: MLMultiArray              // [1, blockSize, hidden]
        public let newK: MLMultiArray                // [K, 1, numKVHeads, blockSize, headDim]
        public let newV: MLMultiArray
        public let captures: MLMultiArray?           // [nCaptures, 1, blockSize, hidden] if any
    }

    public func forward(hidden: MLMultiArray) async throws -> Output {
        let ropeTables = try profiler.measureOrRun("ane_layers_rope_build") {
            try self.buildRopeTables()
        }
        let causalMask = try profiler.measureOrRun("ane_layers_mask_build") {
            try self.buildCausalMask()
        }

        var features: [String: MLFeatureValue] = [
            "x": MLFeatureValue(multiArray: hidden),
            "cos_q": MLFeatureValue(multiArray: ropeTables.cosQ),
            "sin_q": MLFeatureValue(multiArray: ropeTables.sinQ),
            "cos_k_new": MLFeatureValue(multiArray: ropeTables.cosKNew),
            "sin_k_new": MLFeatureValue(multiArray: ropeTables.sinKNew),
            "cache_k_all": MLFeatureValue(multiArray: cacheK),
            "cache_v_all": MLFeatureValue(multiArray: cacheV),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
        ]
        _ = features  // suppress unused warning; features is read via dict init
        let input = try MLDictionaryFeatureProvider(dictionary: features)

        let pred = try await profiler.measureOrRunAsync("ane_layers_predict") {
            try await self.model.prediction(from: input)
        }
        guard let out = pred.featureValue(for: "out")?.multiArrayValue,
              let newK = pred.featureValue(for: "new_k_all")?.multiArrayValue,
              let newV = pred.featureValue(for: "new_v_all")?.multiArrayValue else {
            throw NSError(domain: "Qwen3ANELayers", code: 1,
                           userInfo: [NSLocalizedDescriptionKey: "missing output"])
        }
        let captures = pred.featureValue(for: "captures")?.multiArrayValue
        return Output(hidden: out, newK: newK, newV: newV, captures: captures)
    }

    /// Commit `committed` of the T new K/V rows to cache.
    /// committed = accepted + 1 (number of kept positions in the new block).
    public func commit(newK: MLMultiArray, newV: MLMultiArray, committed: Int) {
        profiler?.begin("ane_layers_cache_commit")
        defer { profiler?.end("ane_layers_cache_commit") }

        let L = config.blockSize
        let K = config.numLayers
        let H = config.numKVHeads
        let Dh = config.headDim
        let S = config.stateLength

        let cacheKPtr = cacheK.dataPointer.assumingMemoryBound(to: UInt16.self)
        let cacheVPtr = cacheV.dataPointer.assumingMemoryBound(to: UInt16.self)
        let newKPtr = newK.dataPointer.assumingMemoryBound(to: UInt16.self)
        let newVPtr = newV.dataPointer.assumingMemoryBound(to: UInt16.self)

        if writePos + L <= S {
            // Accumulating: copy all L new, advance by committed
            for layer in 0..<K {
                for head in 0..<H {
                    let dstOffset = layer * H * S * Dh + head * S * Dh + writePos * Dh
                    let srcOffset = layer * H * L * Dh + head * L * Dh
                    let bytes = L * Dh * 2
                    memcpy(cacheKPtr.advanced(by: dstOffset),
                           newKPtr.advanced(by: srcOffset),
                           bytes)
                    memcpy(cacheVPtr.advanced(by: dstOffset),
                           newVPtr.advanced(by: srcOffset),
                           bytes)
                }
            }
            writePos += committed
        } else {
            // Sliding: shift-left by L, append committed new
            for layer in 0..<K {
                for head in 0..<H {
                    let baseCache = layer * H * S * Dh + head * S * Dh
                    let shiftBytes = (S - L) * Dh * 2
                    memmove(cacheKPtr.advanced(by: baseCache),
                            cacheKPtr.advanced(by: baseCache + L * Dh),
                            shiftBytes)
                    memmove(cacheVPtr.advanced(by: baseCache),
                            cacheVPtr.advanced(by: baseCache + L * Dh),
                            shiftBytes)
                    let tailOffset = baseCache + (S - L) * Dh
                    let srcOffset = layer * H * L * Dh + head * L * Dh
                    let bytes = L * Dh * 2
                    memcpy(cacheKPtr.advanced(by: tailOffset),
                           newKPtr.advanced(by: srcOffset),
                           bytes)
                    memcpy(cacheVPtr.advanced(by: tailOffset),
                           newVPtr.advanced(by: srcOffset),
                           bytes)
                }
            }
        }
        globalOffset += committed
    }

    /// Rewind N positions on rejection (keep cache state, just move writePos back).
    public func trim(by n: Int) {
        writePos = max(0, writePos - n)
        globalOffset = max(0, globalOffset - n)
    }

    // MARK: - RoPE + mask builders

    private func buildRopeTables() throws -> (cosQ: MLMultiArray, sinQ: MLMultiArray,
                                                cosKNew: MLMultiArray, sinKNew: MLMultiArray) {
        let L = config.blockSize
        let dh = config.headDim
        var positions = [Float](repeating: 0, count: L)
        for i in 0..<L { positions[i] = Float(globalOffset + i) }

        // For Qwen3 target: Q positions = K_new positions = current block.
        // We still need two inputs (model expects them separately).
        let cosQ = try makeRoPETable(positions: positions, headDim: dh, sin: false)
        let sinQ = try makeRoPETable(positions: positions, headDim: dh, sin: true)
        // Pass same tensors for K_new since it's the same positions.
        return (cosQ, sinQ, cosQ, sinQ)
    }

    private func makeRoPETable(positions: [Float], headDim: Int, sin useSin: Bool) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [NSNumber(value: positions.count), NSNumber(value: headDim)],
                                    dataType: .float16)
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        for (row, p) in positions.enumerated() {
            for col in 0..<(headDim / 2) {
                let val = useSin ? Foundation.sin(p * invFreq[col]) : Foundation.cos(p * invFreq[col])
                let f16 = floatToHalf(val)
                ptr[row * headDim + col] = f16
                ptr[row * headDim + col + headDim / 2] = f16
            }
        }
        return arr
    }

    /// Causal mask for Qwen3 target_verify: [1, 1, L, S+L]
    /// - cache positions [0, writePos): 0 (valid)
    /// - cache positions [writePos, S): -inf (not filled)
    /// - new positions [S, S+L): causal triangle (row i can attend col S+j if j <= i)
    private func buildCausalMask() throws -> MLMultiArray {
        let L = config.blockSize
        let S = config.stateLength
        let M = config.attendLen  // S + L
        let arr = try MLMultiArray(shape: [1, 1, NSNumber(value: L), NSNumber(value: M)],
                                    dataType: .float16)
        memset(arr.dataPointer, 0, arr.count * 2)
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        let neginf = floatToHalf(-.infinity)

        // Mask invalid cache region
        if writePos < S {
            for row in 0..<L {
                for col in writePos..<S {
                    ptr[row * M + col] = neginf
                }
            }
        }
        // Causal triangle on new positions [S, S+L)
        for row in 0..<L {
            for i in (row + 1)..<L {
                ptr[row * M + (S + i)] = neginf
            }
        }
        return arr
    }
}
