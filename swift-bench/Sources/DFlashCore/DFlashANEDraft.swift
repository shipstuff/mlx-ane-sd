// DFlash ANE draft wrapper.
//
// Loads the compiled .mlmodelc, manages the external cache (accumcache
// pattern: write at current_pos per cycle, advance by committed count,
// slide when overflow), exposes a single forward() call.

import Foundation
import CoreML

public struct DraftConfig: Sendable {
    public let blockSize: Int = 16
    public let ctxSize: Int = 16
    public let hiddenSize: Int = 2560
    public let concatDim: Int = 12800  // 5 layers × 2560
    public let headDim: Int = 128
    public let numLayers: Int = 5
    public let numKVHeads: Int = 8
    public let ropeTheta: Float = 1_000_000
    public let maskTokenId: Int = 151669
    public var stateLength: Int
    public var attendLen: Int { stateLength + blockSize + ctxSize }
    public var T: Int { blockSize + ctxSize }  // 32

    public init(stateLength: Int = 256) {
        self.stateLength = stateLength
    }
}

public final class DFlashANEDraft: @unchecked Sendable {
    public let config: DraftConfig
    private let model: MLModel

    // Python-managed cache (mirror of accumcache)
    private var cacheK: MLMultiArray
    private var cacheV: MLMultiArray
    private var writePos: Int = 0
    private var globalOffset: Int = 0

    // Precomputed rope inverse frequencies
    private let invFreq: [Float]

    public var profiler: Profiler?

    public init(mlmodelcPath: String, config: DraftConfig = DraftConfig(), profiler: Profiler? = nil) throws {
        self.config = config
        self.profiler = profiler

        let url = URL(fileURLWithPath: mlmodelcPath)
        let mlconfig = MLModelConfiguration()
        mlconfig.computeUnits = .cpuAndNeuralEngine
        self.model = try MLModel(contentsOf: url, configuration: mlconfig)

        // Allocate cache buffers
        let cacheShape: [NSNumber] = [
            NSNumber(value: config.numLayers),
            NSNumber(value: config.numKVHeads),
            NSNumber(value: config.stateLength),
            NSNumber(value: config.headDim),
        ]
        self.cacheK = try MLMultiArray(shape: cacheShape, dataType: .float16)
        self.cacheV = try MLMultiArray(shape: cacheShape, dataType: .float16)

        // Precompute inverse frequencies
        var freqs = [Float]()
        freqs.reserveCapacity(config.headDim / 2)
        for i in stride(from: 0, to: config.headDim, by: 2) {
            freqs.append(1.0 / pow(config.ropeTheta, Float(i) / Float(config.headDim)))
        }
        self.invFreq = freqs
    }

    public func resetCache() {
        // Zero out cache K and V
        memset(cacheK.dataPointer, 0, cacheK.count * 2)  // fp16 = 2 bytes
        memset(cacheV.dataPointer, 0, cacheV.count * 2)
        writePos = 0
        globalOffset = 0
    }

    /// Results from a single forward pass.
    public struct Output {
        public let hidden: MLMultiArray   // (1, block_size, hidden)
        public let newK: MLMultiArray     // (num_layers, Hkv, T, head_dim)
        public let newV: MLMultiArray
    }

    /// Build RoPE tables matching the Python accumcache semantics.
    /// - new ctx positions: [globalOffset, globalOffset + sReal)
    /// - padding: at position 0 (arbitrary, k_proj * padding-zero = 0)
    /// - block positions: [globalOffset + sReal, globalOffset + sReal + L)
    /// - query positions: same as block
    private func buildRopeTables(sReal: Int) throws -> (cosQ: MLMultiArray, sinQ: MLMultiArray,
                                                          cosK: MLMultiArray, sinK: MLMultiArray) {
        let L = config.blockSize
        let CS = config.ctxSize
        let T = config.T  // CS + L
        let dh = config.headDim

        // Positions
        let blockStart = globalOffset + sReal
        var qPositions = [Float](repeating: 0, count: L)
        for i in 0..<L { qPositions[i] = Float(blockStart + i) }

        var kPositions = [Float](repeating: 0, count: T)
        for i in 0..<sReal { kPositions[i] = Float(globalOffset + i) }
        // pad with zeros for [sReal, CS)
        for i in 0..<L { kPositions[CS + i] = Float(blockStart + i) }

        let cosQ = try makeRoPETable(positions: qPositions, headDim: dh)
        let sinQ = try makeRoPETable(positions: qPositions, headDim: dh, sin: true)
        let cosK = try makeRoPETable(positions: kPositions, headDim: dh)
        let sinK = try makeRoPETable(positions: kPositions, headDim: dh, sin: true)
        return (cosQ, sinQ, cosK, sinK)
    }

    private func makeRoPETable(positions: [Float], headDim: Int, sin useSin: Bool = false) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: [NSNumber(value: positions.count), NSNumber(value: headDim)],
                                    dataType: .float16)
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        // Each row: [cos(p * inv_freq_0), cos(p * inv_freq_1), ..., cos(p * inv_freq_{dh/2-1}),  (repeated)]
        // Matches numpy: concat([freqs, freqs], axis=-1) where freqs = outer(positions, inv_freq)
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

    private func buildCausalMask() throws -> MLMultiArray {
        let L = config.blockSize
        let M = config.attendLen  // STATE_LEN + T
        let arr = try MLMultiArray(shape: [1, 1, NSNumber(value: L), NSNumber(value: M)],
                                    dataType: .float16)
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        // Fill zeros first
        memset(arr.dataPointer, 0, arr.count * 2)
        if writePos < config.stateLength {
            // mask [writePos, stateLength) as -inf
            let neginf = floatToHalf(-.infinity)
            for row in 0..<L {
                for col in writePos..<config.stateLength {
                    ptr[row * M + col] = neginf
                }
            }
        }
        return arr
    }

    /// Run the draft forward.
    /// - Parameters:
    ///   - noiseEmbedding: MLMultiArray (1, block_size, hidden_size), fp16
    ///   - targetHidden: MLMultiArray (1, ctx_size, concat_dim), fp16
    ///   - sReal: number of real ctx positions (rest is zero-padded)
    public func forward(noiseEmbedding: MLMultiArray,
                         targetHidden: MLMultiArray,
                         sReal: Int) async throws -> Output {
        let ropeTables = try profiler.measureOrRun("draft_rope_build") {
            try self.buildRopeTables(sReal: sReal)
        }
        let causalMask = try profiler.measureOrRun("draft_mask_build") {
            try self.buildCausalMask()
        }

        let features: [String: MLFeatureValue] = [
            "noise_embedding": MLFeatureValue(multiArray: noiseEmbedding),
            "target_hidden": MLFeatureValue(multiArray: targetHidden),
            "cos_q": MLFeatureValue(multiArray: ropeTables.cosQ),
            "sin_q": MLFeatureValue(multiArray: ropeTables.sinQ),
            "cos_k": MLFeatureValue(multiArray: ropeTables.cosK),
            "sin_k": MLFeatureValue(multiArray: ropeTables.sinK),
            "cache_K": MLFeatureValue(multiArray: cacheK),
            "cache_V": MLFeatureValue(multiArray: cacheV),
            "causal_mask": MLFeatureValue(multiArray: causalMask),
        ]
        let input = try MLDictionaryFeatureProvider(dictionary: features)

        let pred = try await profiler.measureOrRunAsync("draft_predict") {
            try await self.model.prediction(from: input)
        }

        guard let hidden = pred.featureValue(for: "hidden")?.multiArrayValue,
              let newK = pred.featureValue(for: "new_K")?.multiArrayValue,
              let newV = pred.featureValue(for: "new_V")?.multiArrayValue else {
            throw NSError(domain: "DFlashANEDraft", code: 1,
                           userInfo: [NSLocalizedDescriptionKey: "missing output"])
        }
        return Output(hidden: hidden, newK: newK, newV: newV)
    }

    /// Commit new K/V to cache (matches Python's commit() — advance by committed count only).
    public func commit(newK: MLMultiArray, newV: MLMultiArray, sReal: Int, accepted: Int) {
        profiler?.begin("draft_cache_commit")
        defer { profiler?.end("draft_cache_commit") }

        let committed = sReal + accepted + 1
        let T = config.T

        if writePos + T <= config.stateLength {
            // Accumulating phase: write the full T entries
            let n = config.numLayers
            let hkv = config.numKVHeads
            let dh = config.headDim
            let cacheRowSize = config.stateLength * dh  // per-Hkv-head slab
            // Flat index: [layer, kv_head, pos, dh_dim]
            // strides for cacheK: [hkv * stateLength * dh, stateLength * dh, dh, 1]
            // For each layer, for each kv_head: copy T rows starting at writePos from newK[layer, kv_head, :, :]
            let cacheKPtr = cacheK.dataPointer.assumingMemoryBound(to: UInt16.self)
            let cacheVPtr = cacheV.dataPointer.assumingMemoryBound(to: UInt16.self)
            let newKPtr = newK.dataPointer.assumingMemoryBound(to: UInt16.self)
            let newVPtr = newV.dataPointer.assumingMemoryBound(to: UInt16.self)

            // new* shape: (num_layers, Hkv, T, head_dim). Strides: [Hkv*T*dh, T*dh, dh, 1]
            // cache shape: (num_layers, Hkv, stateLength, dh). Strides: [Hkv*stateLength*dh, stateLength*dh, dh, 1]
            for layer in 0..<n {
                for head in 0..<hkv {
                    let cacheOffset = layer * hkv * config.stateLength * dh
                                     + head * config.stateLength * dh
                                     + writePos * dh
                    let newOffset = layer * hkv * T * dh
                                   + head * T * dh
                    let bytesPerRow = T * dh * 2
                    memcpy(cacheKPtr.advanced(by: cacheOffset),
                           newKPtr.advanced(by: newOffset),
                           bytesPerRow)
                    memcpy(cacheVPtr.advanced(by: cacheOffset),
                           newVPtr.advanced(by: newOffset),
                           bytesPerRow)
                }
            }
            writePos += committed
        } else {
            // Sliding phase: shift-left by T, append
            let n = config.numLayers
            let hkv = config.numKVHeads
            let dh = config.headDim
            let cacheKPtr = cacheK.dataPointer.assumingMemoryBound(to: UInt16.self)
            let cacheVPtr = cacheV.dataPointer.assumingMemoryBound(to: UInt16.self)
            let newKPtr = newK.dataPointer.assumingMemoryBound(to: UInt16.self)
            let newVPtr = newV.dataPointer.assumingMemoryBound(to: UInt16.self)

            for layer in 0..<n {
                for head in 0..<hkv {
                    let baseCache = layer * hkv * config.stateLength * dh
                                   + head * config.stateLength * dh
                    // Shift: cache[T:] -> cache[:stateLength-T]
                    let shiftBytes = (config.stateLength - T) * dh * 2
                    memmove(cacheKPtr.advanced(by: baseCache),
                            cacheKPtr.advanced(by: baseCache + T * dh),
                            shiftBytes)
                    memmove(cacheVPtr.advanced(by: baseCache),
                            cacheVPtr.advanced(by: baseCache + T * dh),
                            shiftBytes)
                    // Append new at tail
                    let tailOffset = baseCache + (config.stateLength - T) * dh
                    let newOffset = layer * hkv * T * dh + head * T * dh
                    memcpy(cacheKPtr.advanced(by: tailOffset),
                           newKPtr.advanced(by: newOffset),
                           T * dh * 2)
                    memcpy(cacheVPtr.advanced(by: tailOffset),
                           newVPtr.advanced(by: newOffset),
                           T * dh * 2)
                }
            }
            // write_pos stays at stateLength
        }
        globalOffset += committed
    }
}

// MARK: - fp16 helpers

func floatToHalf(_ f: Float) -> UInt16 {
    // IEEE 754 round-to-nearest-even fp32 -> fp16 conversion
    let bits = f.bitPattern
    let sign = UInt16((bits >> 16) & 0x8000)
    let expf = Int32((bits >> 23) & 0xFF) - 127
    let mantissa = bits & 0x007F_FFFF

    if expf == 128 {
        // Inf / NaN
        return sign | 0x7C00 | (mantissa != 0 ? UInt16(0x0200) : 0)
    }
    if expf > 15 {
        return sign | 0x7C00  // +/- inf
    }
    if expf < -14 {
        // Subnormal or zero
        if expf < -24 { return sign }
        let shift = -14 - expf
        let sub = (mantissa | 0x00800000) >> UInt32(14 + shift - 1)
        // round half to even
        let rounded = (sub >> 1) + (sub & 1)
        return sign | UInt16(truncatingIfNeeded: rounded)
    }
    // Normal range
    let exph = UInt16((expf + 15) & 0x1F) << 10
    var mant = mantissa >> 13
    // Round to nearest even using lower 13 bits
    let remainder = mantissa & 0x1FFF
    if remainder > 0x1000 || (remainder == 0x1000 && (mant & 1) != 0) {
        mant += 1
    }
    if mant == 0x0400 {
        // Rounded up into next exponent
        return sign | (UInt16(((expf + 1) + 15) & 0x1F) << 10)
    }
    return sign | exph | UInt16(truncatingIfNeeded: mant)
}

// MARK: - Profiler ergonomics helpers

extension Profiler {
    func measureOrRun<T>(_ name: String, _ block: () throws -> T) rethrows -> T {
        return try self.measure(name, block)
    }
    func measureOrRunAsync<T>(_ name: String, _ block: () async throws -> T) async rethrows -> T {
        return try await self.measureAsync(name, block)
    }
}

// Profiler? ergonomics
extension Optional where Wrapped == Profiler {
    func measureOrRun<T>(_ name: String, _ block: () throws -> T) rethrows -> T {
        if case .some(let p) = self { return try p.measure(name, block) }
        return try block()
    }
    func measureOrRunAsync<T>(_ name: String, _ block: () async throws -> T) async rethrows -> T {
        if case .some(let p) = self { return try await p.measureAsync(name, block) }
        return try await block()
    }
}
