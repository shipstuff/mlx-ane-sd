// Full DFlash speculative decoding runner in Swift.
//
// Pipeline (per cycle):
//   1. target.forwardCapturing([last_tok] ++ draft_tokens, captureAt=target_layer_ids)
//      -> (logits, captured hidden)
//   2. Truncate captured hidden to [:accepted+1] for next cycle's target_hidden
//   3. noise_embedding = target.embedTokens([last_tok] ++ [MASK]*(bs-1))
//   4. draft.forward(noise_embedding, target_hidden, s_real) -> (hidden, new_K, new_V)
//   5. draft_logits = target.lmHead(draft hidden)
//   6. sample draft tokens top-1
//   7. verify: next cycle's target forward with (last_tok ++ draft_tokens)
//   8. compare, accept committed prefix
//   9. draft.commit(new_K, new_V, s_real, accepted)
//  10. target cache trim to (start of cycle + accepted + 1)
//
// Everything profiled via our Profiler.

import Foundation
import HuggingFace
import Tokenizers
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import MLXNN
import CoreML
import ArgumentParser
import Accelerate
import DFlashCore

@main
struct DFlashSDRunner: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "dflash-sd",
        abstract: "Full DFlash speculative decoding with ANE-hosted draft + MLX-hosted Qwen3 target."
    )

    @Option(help: "Target model HF id")
    var target: String = "mlx-community/Qwen3-4B-bf16"

    @Option(help: "Compiled DFlash draft .mlmodelc path")
    var draft: String = "/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc"

    @Option(help: "Draft cache state length")
    var stateLength: Int = 256

    @Option(help: "Comma-separated layer indices (must match DFlash config)")
    var captureAt: String = "1,9,17,25,33"

    @Option(help: "Prompt text")
    var prompt: String = "The capital of France is Paris, which is known for"

    @Option(help: "Max new tokens to generate")
    var maxNew: Int = 100

    @Option(help: "Mask token ID (Qwen3 unused token for DFlash)")
    var maskTokenId: Int = 151669

    @Flag(help: "Verbose per-cycle logging")
    var verbose: Bool = false

    @Option(help: "JSON profile output path (optional)")
    var profileOut: String = ""

    @Flag(help: "Emit a single-line JSON summary on stdout (for benchmark harnesses)")
    var json: Bool = false

    @Option(help: "Optional compiled CoreML .mlmodelc for lm_head (fp16 input [1,15,2560] -> fp16 logits [1,15,vocab]). If set, uses ANE lm_head instead of MLX GPU lm_head.")
    var aneLmhead: String = ""

    @Option(help: "Optional compiled CoreML .mlmodelc with K Qwen3 target layers. Use with --ane-target-k to set K. Enables hybrid target_verify (ANE first K layers + MLX remaining).")
    var aneTargetLayers: String = ""

    @Option(help: "K value for --ane-target-layers: number of target layers on ANE.")
    var aneTargetK: Int = 0

    @Option(help: "Comma-separated 0-based indices within the ANE K layers to expose as captures (for DFlash draft's target_hidden). Must match conversion.")
    var aneTargetCaptures: String = ""

    func run() async throws {
        let profiler = Profiler()
        profiler.begin("total_wall")

        // Parse capture indices
        let captureIndices = captureAt.split(separator: ",").compactMap { Int($0) }
        if !self.json { print("[sd] capture_at=\(captureIndices), block_size=16, state_length=\(stateLength)") }

        // Register our inspectable Qwen3 model
        await LLMTypeRegistry.shared.registerModelType("qwen3") { data in
            let config = try JSONDecoder().decode(Qwen3InspConfiguration.self, from: data)
            return Qwen3InspModel(config)
        }

        // Load target via HF
        if !self.json { print("[sd] loading target \(target)...") }
        let tTargetLoad = CFAbsoluteTimeGetCurrent()
        let downloader: Downloader = #hubDownloader()
        let tokenizerLoader: TokenizerLoader = #huggingFaceTokenizerLoader()
        let container = try await LLMModelFactory.shared.loadContainer(
            from: downloader, using: tokenizerLoader,
            configuration: ModelConfiguration(id: target)
        ) { _ in }
        if !self.json {
            print(String(format: "[sd] target loaded in %.1fs",
                         CFAbsoluteTimeGetCurrent() - tTargetLoad))
        }

        // Load draft
        if !self.json { print("[sd] loading draft \(draft)...") }
        let draftConfig = DraftConfig(stateLength: stateLength)
        let tDraftLoad = CFAbsoluteTimeGetCurrent()
        let draftModel = try DFlashANEDraft(mlmodelcPath: draft,
                                             config: draftConfig,
                                             profiler: profiler)
        if !self.json {
            print(String(format: "[sd] draft loaded in %.2fs",
                         CFAbsoluteTimeGetCurrent() - tDraftLoad))
        }

        // Optional: load ANE lm_head model (wrap for Sendable capture in perform closure)
        let aneLmHeadBox: MLModelBox?
        if !self.aneLmhead.isEmpty {
            if !self.json { print("[sd] loading ANE lm_head \(self.aneLmhead)...") }
            let mlconfig = MLModelConfiguration()
            mlconfig.computeUnits = .cpuAndNeuralEngine
            let tLmHead = CFAbsoluteTimeGetCurrent()
            let m = try MLModel(contentsOf: URL(fileURLWithPath: self.aneLmhead),
                                 configuration: mlconfig)
            aneLmHeadBox = MLModelBox(model: m)
            if !self.json {
                print(String(format: "[sd] ANE lm_head loaded in %.2fs",
                             CFAbsoluteTimeGetCurrent() - tLmHead))
            }
        } else {
            aneLmHeadBox = nil
        }

        // Optional: load K-layer ANE target model (hybrid target_verify)
        let aneLayers: Qwen3ANELayers?
        let aneCaptureIndices: [Int]
        if !self.aneTargetLayers.isEmpty {
            precondition(self.aneTargetK > 0, "--ane-target-k must be > 0 when --ane-target-layers is set")
            aneCaptureIndices = self.aneTargetCaptures.isEmpty ? []
                : self.aneTargetCaptures.split(separator: ",").compactMap { Int($0) }
            if !self.json { print("[sd] loading ANE target layers (K=\(self.aneTargetK)) \(self.aneTargetLayers)...") }
            let tLoad = CFAbsoluteTimeGetCurrent()
            let cfg = Qwen3ANEConfig(numLayers: self.aneTargetK,
                                       stateLength: self.stateLength,
                                       captureIndices: aneCaptureIndices)
            aneLayers = try Qwen3ANELayers(mlmodelcPath: self.aneTargetLayers,
                                             config: cfg, profiler: profiler)
            if !self.json {
                print(String(format: "[sd] ANE target layers loaded in %.2fs",
                             CFAbsoluteTimeGetCurrent() - tLoad))
            }
        } else {
            aneLayers = nil
            aneCaptureIndices = []
        }
        let aneTargetK = self.aneTargetK

        // Run inside the container's context
        try await container.perform { context in
            guard let model = context.model as? Qwen3InspModel else {
                print("[sd] ERROR: not Qwen3InspModel")
                return
            }

            // Tokenize
            let promptTokens = try context.tokenizer.encode(text: self.prompt,
                                                              addSpecialTokens: true)
            if self.verbose { print("[sd] prompt=\(promptTokens.count) tokens") }

            // Create target KV cache
            var targetCache = model.newCache(parameters: nil)

            // Prefill target
            profiler.begin("target_prefill")
            let prefillInput = MLXArray(promptTokens.map { Int32($0) })
                .reshaped([1, promptTokens.count])
            let (prefillLogits, prefillCaptures) = model.forwardCapturing(
                prefillInput, cache: targetCache, captureAt: captureIndices
            )
            prefillLogits.asArray(Float.self)  // eval
            profiler.end("target_prefill")

            // Sync MLX cache -> ANE cache for layers [0..K-1] after prefill.
            if let aneL = aneLayers {
                profiler.begin("ane_cache_sync")
                var layerKeys = [MLMultiArray]()
                var layerValues = [MLMultiArray]()
                for i in 0..<aneTargetK {
                    let state = targetCache[i].state  // [keys, values], each [1, 8, promptLen, 128]
                    precondition(state.count == 2, "unexpected KV cache state for layer \(i)")
                    let k = try self.mlxToMLMultiArray(state[0])
                    let v = try self.mlxToMLMultiArray(state[1])
                    layerKeys.append(k)
                    layerValues.append(v)
                }
                try aneL.loadFromPrefill(layerKeys: layerKeys, layerValues: layerValues,
                                          promptLen: promptTokens.count)
                profiler.end("ane_cache_sync")
                if !self.json {
                    print("[sd] synced \(promptTokens.count) prefill positions to ANE cache")
                }
            }

            // Sample bonus token from prefill
            let bonus = prefillLogits[0..., -1, 0...].argMax(axis: -1).item(Int32.self)
            var tokens = promptTokens
            tokens.append(Int(bonus))

            // Initial target_hidden = prefill captures concat'd along last axis
            var hiddenConcat = MLX.concatenated(prefillCaptures, axis: -1)  // [1, T, concat_dim]
            var generated = [Int(bonus)]
            var cycles = 0
            var acceptedTotal = 0

            // Decode loop
            draftModel.resetCache()
            profiler.begin("decode_total")
            var n = 1
            while n < self.maxNew {
                let bs = draftConfig.blockSize
                let last = tokens.last!

                // Build block [last, MASK, MASK, ...]
                var blockIds: [Int32] = [Int32(last)]
                for _ in 0..<(bs - 1) {
                    blockIds.append(Int32(self.maskTokenId))
                }
                let blockInputs = MLXArray(blockIds).reshaped([1, bs])

                // Get noise_embedding via target.model.embedTokens
                profiler.begin("noise_embed")
                let noiseEmb = model.model.embedTokens(blockInputs)
                noiseEmb.asArray(Float.self)  // eval
                profiler.end("noise_embed")

                // Prepare target_hidden: pad/truncate to ctx_size=16
                let sReal = min(hiddenConcat.dim(1), bs)
                let ctxPadded = padOrTruncateCtx(hiddenConcat, ctxSize: bs)

                // Convert MLX arrays to CoreML MLMultiArrays for the draft
                profiler.begin("mlx_to_coreml")
                let noiseMLArr = try mlxToMLMultiArray(noiseEmb)
                let ctxMLArr = try mlxToMLMultiArray(ctxPadded)
                profiler.end("mlx_to_coreml")

                // Run draft
                let draftOut = try await draftModel.forward(
                    noiseEmbedding: noiseMLArr,
                    targetHidden: ctxMLArr,
                    sReal: sReal
                )

                // Apply lm_head to draft hidden -> logits -> argmax.
                // Two paths: ANE (if aneLmHeadModel loaded) vs MLX GPU.
                profiler.begin("draft_lmhead")
                let draftTokenArray: [Int32]
                if let aneLmHead = aneLmHeadBox?.model {
                    // ANE path: slice draft hidden [1,16,2560] -> [1,15,2560], predict, host argmax.
                    let slicedHidden = try sliceHiddenLastN(draftOut.hidden, n: bs - 1)
                    let features = try MLDictionaryFeatureProvider(
                        dictionary: ["hidden": MLFeatureValue(multiArray: slicedHidden)])
                    let result = try await aneLmHead.prediction(from: features)
                    guard let logitsArr = result.featureValue(for: "logits")?.multiArrayValue else {
                        throw NSError(domain: "dflash-sd", code: 2,
                                       userInfo: [NSLocalizedDescriptionKey: "missing logits"])
                    }
                    draftTokenArray = hostArgmaxFp16(logitsArr, L: bs - 1,
                                                      V: logitsArr.shape[2].intValue)
                } else {
                    // MLX GPU path (default)
                    let draftHiddenMLX = try mlMultiArrayToMLX(draftOut.hidden)
                    let hiddenSlice = draftHiddenMLX[0..., (1 - bs)..., 0...]
                    let draftLogits: MLXArray
                    if let lmHead = model.lmHead {
                        draftLogits = lmHead(hiddenSlice)
                    } else {
                        draftLogits = model.model.embedTokens.asLinear(hiddenSlice)
                    }
                    let t = draftLogits.argMax(axis: -1)
                    draftTokenArray = t.asArray(Int32.self)
                }
                profiler.end("draft_lmhead")

                // Build verify input: [last, draft_tokens...]
                profiler.begin("target_verify")
                let verifyIds = [Int32(last)] + draftTokenArray
                let verifyInput = MLXArray(verifyIds).reshaped([1, verifyIds.count])

                // Two paths: hybrid (ANE K layers + MLX remaining) vs full MLX.
                let verifyLogits: MLXArray
                let verifyCaptures: [MLXArray]
                var aneNewK: MLMultiArray? = nil
                var aneNewV: MLMultiArray? = nil
                if let aneL = aneLayers {
                    // Hybrid: embed -> ANE K layers -> MLX K..35 -> logits.
                    profiler.begin("tv_embed")
                    let embHidden = model.model.embed(verifyInput)
                    embHidden.asArray(Float.self)
                    profiler.end("tv_embed")

                    profiler.begin("tv_mlx_to_coreml")
                    let embMLArr = try self.mlxToMLMultiArray(embHidden)
                    profiler.end("tv_mlx_to_coreml")

                    let aneOut = try await aneL.forward(hidden: embMLArr)
                    aneNewK = aneOut.newK
                    aneNewV = aneOut.newV

                    profiler.begin("tv_coreml_to_mlx")
                    let hiddenAfterKFp16 = try self.mlMultiArrayToMLX(aneOut.hidden)
                    // Cast to bf16 so MLX target (Qwen3-4B-bf16) doesn't re-cast per layer.
                    let hiddenAfterK = hiddenAfterKFp16.asType(.bfloat16)
                    profiler.end("tv_coreml_to_mlx")

                    // Continue with remaining MLX layers + lm_head
                    profiler.begin("tv_mlx_layers")
                    let mlxCaptureAt = captureIndices.filter { $0 >= aneTargetK }
                    let (logits, mlxCaptures) = model.forwardFromLayerCapturing(
                        startIdx: aneTargetK, hidden: hiddenAfterK,
                        cache: targetCache, captureAt: mlxCaptureAt)
                    logits.asArray(Float.self)
                    profiler.end("tv_mlx_layers")

                    // Assemble captures in the order of captureIndices: ANE captures first
                    // (for indices < K), then MLX captures (for indices >= K).
                    var finalCaptures = [MLXArray]()
                    let aneCapLayers = captureIndices.filter { $0 < aneTargetK }
                    if !aneCapLayers.isEmpty {
                        guard let capArr = aneOut.captures else {
                            throw NSError(domain: "dflash-sd", code: 3,
                                           userInfo: [NSLocalizedDescriptionKey: "ANE model did not return captures"])
                        }
                        // capArr shape: [n_captures, 1, 16, 2560]; split into individual MLXArrays
                        let capMLX = try self.mlMultiArrayToMLX(capArr)
                        for i in 0..<aneCapLayers.count {
                            finalCaptures.append(capMLX[i, 0..., 0..., 0...])
                        }
                    }
                    for c in mlxCaptures { finalCaptures.append(c) }
                    // Reorder to match captureIndices
                    var byIndex = [Int: MLXArray]()
                    let aneLayerList = aneCapLayers
                    let mlxLayerList = mlxCaptureAt
                    for (i, layer) in aneLayerList.enumerated() { byIndex[layer] = finalCaptures[i] }
                    for (i, layer) in mlxLayerList.enumerated() {
                        byIndex[layer] = finalCaptures[aneLayerList.count + i]
                    }
                    verifyCaptures = captureIndices.map { byIndex[$0]! }
                    verifyLogits = logits
                } else {
                    // Standard full-MLX path
                    let (logits, caps) = model.forwardCapturing(
                        verifyInput, cache: targetCache, captureAt: captureIndices
                    )
                    logits.asArray(Float.self)
                    verifyLogits = logits
                    verifyCaptures = caps
                }
                profiler.end("target_verify")

                // Sample target tokens
                let targetTokens = verifyLogits.argMax(axis: -1)
                let targetTokenArray = targetTokens.asArray(Int32.self).map { Int($0) }

                // Compare draft_tokens vs target_tokens[0..bs-2] to find first mismatch
                profiler.begin("accept_check")
                let dList = draftTokenArray.map { Int($0) }
                let tList = targetTokenArray  // length bs
                var accepted = dList.count
                for i in 0..<dList.count {
                    if dList[i] != tList[i] {
                        accepted = i
                        break
                    }
                }
                let newTokens = Array(tList[0...accepted])  // first accepted + bonus at accepted position
                profiler.end("accept_check")

                // Append committed tokens
                tokens.append(contentsOf: newTokens.prefix(self.maxNew - n))
                generated.append(contentsOf: newTokens.prefix(self.maxNew - n))
                n += min(newTokens.count, self.maxNew - n)
                cycles += 1
                acceptedTotal += accepted + 1  // match Python F.1: count committed (draft + bonus)

                if self.verbose {
                    print("[sd] cycle \(cycles): accepted \(accepted)/\(bs-1), committed \(newTokens.count), n=\(n)")
                }

                // Commit to draft cache
                draftModel.commit(newK: draftOut.newK, newV: draftOut.newV,
                                   sReal: sReal, accepted: accepted)

                // Trim target cache by unused positions (bs - accepted - 1)
                let trim = bs - accepted - 1
                if trim > 0 {
                    profiler.begin("target_cache_trim")
                    for cache in targetCache {
                        cache.trim(trim)
                    }
                    profiler.end("target_cache_trim")
                }

                // If using ANE target layers: commit committed positions to its cache too.
                // Note: ANE cache was populated with all 16 new K/V during target_verify's
                // ANE forward. Commit needs to advance writePos by (accepted + 1) only.
                // ANE cache trim on rejection happens via negative adjustment in next call.
                // Actually: ANE cache was NOT yet updated this cycle. The forward() returned
                // newK/newV but didn't write them. We write them now with committed count.
                // TODO: if ANE was re-run with a re-tried block after rejection, handle that.
                // For now, assume standard flow: commit (accepted+1) of the 16.
                // Since ANE's commit does "write all T, advance by committed", it correctly
                // handles the rejection case (the rejected slots get overwritten next cycle).
                // However, target_cache_trim on MLX trimmed by `trim`. We already advanced
                // writePos by committed (no trim needed for ANE). But globalOffset must
                // stay synchronized with MLX's cache offset.
                // MLX offset after this cycle = previous_offset + 16 - trim = prev + (accepted+1)
                // ANE globalOffset after commit = prev + (accepted+1). Matches.
                if let aneL = aneLayers, let nK = aneNewK, let nV = aneNewV {
                    profiler.begin("ane_cache_commit")
                    aneL.commit(newK: nK, newV: nV, committed: accepted + 1)
                    profiler.end("ane_cache_commit")
                }

                // Update hiddenConcat for next cycle: take the accepted+1 positions
                hiddenConcat = MLX.concatenated(verifyCaptures, axis: -1)
                let acceptedPlus1 = accepted + 1
                hiddenConcat = hiddenConcat[0..., 0..<acceptedPlus1, 0...]
            }
            profiler.end("decode_total")
            profiler.end("total_wall")

            // Report
            let decoded = context.tokenizer.decode(tokenIds: generated, skipSpecialTokens: true)
            let snap = profiler.snapshot()
            let totalWall = snap.totalMs
            let decodeMs = snap.phases["decode_total"]?.totalMs ?? totalWall
            let tps = Double(generated.count) / (totalWall / 1000)
            let decodeTps = Double(generated.count) / (decodeMs / 1000)

            if self.json {
                // Single-line JSON on stdout for benchmark harnesses.
                var phaseStats: [String: [String: Double]] = [:]
                for (name, s) in snap.phases {
                    phaseStats[name] = [
                        "calls": Double(s.calls),
                        "totalMs": s.totalMs,
                        "meanMs": s.meanMs,
                        "minMs": s.minMs == .infinity ? 0 : s.minMs,
                        "maxMs": s.maxMs,
                    ]
                }
                let payload: [String: Any] = [
                    "tokens": generated.count,
                    "cycles": cycles,
                    "accepted_total": acceptedTotal,
                    "avg_accepted_per_cycle": cycles > 0 ? Double(acceptedTotal) / Double(cycles) : 0,
                    "total_wall_ms": totalWall,
                    "decode_ms": decodeMs,
                    "tok_per_s_wall": tps,
                    "tok_per_s_decode": decodeTps,
                    "text": decoded,
                    "phases": phaseStats,
                ]
                let data = try JSONSerialization.data(withJSONObject: payload, options: [])
                print(String(data: data, encoding: .utf8)!)
            } else {
                print("")
                print("[sd] generated \(generated.count) tokens in \(cycles) cycles "
                    + "(accepted \(acceptedTotal), avg/cycle \(String(format: "%.2f", cycles > 0 ? Double(acceptedTotal) / Double(cycles) : 0)))")
                print("[sd] output: \(decoded.prefix(120))")
                print(String(format: "[sd] wall: %.1f s, %.2f tok/s | decode: %.2f s, %.2f tok/s",
                             totalWall / 1000, tps, decodeMs / 1000, decodeTps))
                profiler.printSummary()
            }

            if !self.profileOut.isEmpty {
                let url = URL(fileURLWithPath: self.profileOut)
                try profiler.writeJSON(to: url)
                if !self.json { print("[sd] profile written to \(self.profileOut)") }
            }
        }
    }

    /// Pad target_hidden on the right with zeros to ctx_size, OR take the last
    /// ctx_size positions if it's longer.
    private func padOrTruncateCtx(_ hidden: MLXArray, ctxSize: Int) -> MLXArray {
        let S = hidden.dim(1)
        if S == ctxSize { return hidden }
        if S > ctxSize {
            return hidden[0..., (S - ctxSize)..<S, 0...]
        }
        // pad at right
        let concatDim = hidden.dim(2)
        let pad = MLXArray.zeros([1, ctxSize - S, concatDim], dtype: hidden.dtype)
        return MLX.concatenated([hidden, pad], axis: 1)
    }

    /// Convert MLXArray (bf16 or fp16 or fp32) to MLMultiArray fp16.
    private func mlxToMLMultiArray(_ arr: MLXArray) throws -> MLMultiArray {
        let shape = arr.shape.map { NSNumber(value: $0) }
        let out = try MLMultiArray(shape: shape, dataType: .float16)
        // Cast to fp16 if not already
        let fp16Arr = arr.dtype == .float16 ? arr : arr.asType(.float16)
        fp16Arr.asArray(Float.self)  // eval
        // Get raw bytes
        let bytes = fp16Arr.asData(access: .noCopyIfContiguous)
        // Copy into MLMultiArray's buffer
        let totalBytes = out.count * 2
        _ = bytes.data.withUnsafeBytes { srcPtr in
            memcpy(out.dataPointer, srcPtr.baseAddress, totalBytes)
        }
        return out
    }

    /// Convert MLMultiArray fp16 -> MLXArray.
    private func mlMultiArrayToMLX(_ arr: MLMultiArray) throws -> MLXArray {
        let shape = arr.shape.map { $0.intValue }
        let totalElements = arr.count
        let bytes = totalElements * 2  // fp16
        let data = Data(bytes: arr.dataPointer, count: bytes)
        return MLXArray(data, shape, type: Float16.self)
    }

    /// Take last n positions of [1, L, H] fp16 MLMultiArray -> new [1, n, H].
    private func sliceHiddenLastN(_ arr: MLMultiArray, n: Int) throws -> MLMultiArray {
        let L = arr.shape[1].intValue
        let H = arr.shape[2].intValue
        precondition(n <= L)
        let out = try MLMultiArray(shape: [1, NSNumber(value: n), NSNumber(value: H)],
                                    dataType: .float16)
        let srcOffsetBytes = (L - n) * H * 2
        let copyBytes = n * H * 2
        memcpy(out.dataPointer,
               arr.dataPointer.advanced(by: srcOffsetBytes),
               copyBytes)
        return out
    }

    /// Sendable wrapper for MLModel so it can be captured in @Sendable closures.
    private final class MLModelBox: @unchecked Sendable {
        let model: MLModel
        init(model: MLModel) { self.model = model }
    }

    /// Host argmax over last dim of fp16 [1, L, V] -> [Int32] of length L.
    /// Uses vImage to convert each row to fp32, then vDSP_maxvi for max+index.
    private func hostArgmaxFp16(_ arr: MLMultiArray, L: Int, V: Int) -> [Int32] {
        let ptr16 = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        var out = [Int32](repeating: 0, count: L)
        var fp32Row = [Float](repeating: 0, count: V)
        let vCount = vImagePixelCount(V)

        for l in 0..<L {
            // fp16 -> fp32 row conversion via Accelerate
            var src = vImage_Buffer(
                data: UnsafeMutableRawPointer(mutating: ptr16.advanced(by: l * V)),
                height: 1, width: vCount, rowBytes: V * 2)
            fp32Row.withUnsafeMutableBufferPointer { dstPtr in
                var dst = vImage_Buffer(
                    data: UnsafeMutableRawPointer(dstPtr.baseAddress!),
                    height: 1, width: vCount, rowBytes: V * 4)
                vImageConvert_Planar16FtoPlanarF(&src, &dst, vImage_Flags(kvImageNoFlags))
            }
            // Find max value + its index in one Accelerate call
            var maxVal: Float = 0
            var maxIdx: vDSP_Length = 0
            vDSP_maxvi(fp32Row, 1, &maxVal, &maxIdx, vDSP_Length(V))
            out[l] = Int32(maxIdx)
        }
        return out
    }
}
