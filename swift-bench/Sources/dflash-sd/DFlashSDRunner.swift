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

                // Apply target.lmHead to draft hidden for draft_logits.
                // Only positions [1-bs:] are the draft's predictions, so slice first
                // to avoid wasting 1 position of lm_head compute (~6% of the matmul).
                profiler.begin("draft_lmhead")
                let draftHiddenMLX = try mlMultiArrayToMLX(draftOut.hidden)
                let hiddenSlice = draftHiddenMLX[0..., (1 - bs)..., 0...]
                let draftLogits: MLXArray
                if let lmHead = model.lmHead {
                    draftLogits = lmHead(hiddenSlice)
                } else {
                    draftLogits = model.model.embedTokens.asLinear(hiddenSlice)
                }
                let draftTokens = draftLogits.argMax(axis: -1)
                draftTokens.asArray(Int32.self)
                profiler.end("draft_lmhead")

                // Build verify input: [last, draft_tokens...]
                profiler.begin("target_verify")
                let draftTokenArray = draftTokens.asArray(Int32.self)
                let verifyIds = [Int32(last)] + draftTokenArray
                let verifyInput = MLXArray(verifyIds).reshaped([1, verifyIds.count])

                // Capture hidden at specified layers
                let (verifyLogits, verifyCaptures) = model.forwardCapturing(
                    verifyInput, cache: targetCache, captureAt: captureIndices
                )
                verifyLogits.asArray(Float.self)
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
}
