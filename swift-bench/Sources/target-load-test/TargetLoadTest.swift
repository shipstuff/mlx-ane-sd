// Smoke test: load Qwen3-4B via mlx-swift-lm's LLMModelFactory
// with our Qwen3InspModel registered as "qwen3". Run a forward pass
// with hidden state capture. Dump shapes.
//
// This is the foundation for the full SD loop — proves the vendored model
// + factory registration works end-to-end.

import Foundation
import HuggingFace
import Tokenizers
import MLX
import MLXLLM
import MLXLMCommon
import MLXHuggingFace
import ArgumentParser
import DFlashCore

@main
struct TargetLoadTest: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "target-load-test",
        abstract: "Load Qwen3 target with Qwen3InspModel and capture hidden states"
    )

    @Option(help: "Target model HF id")
    var target: String = "mlx-community/Qwen3-4B-bf16"

    @Option(help: "Prompt text")
    var prompt: String = "The capital of France is Paris, which is known for"

    @Option(help: "Comma-separated layer indices to capture (0-based)")
    var captureAt: String = "1,9,17,25,33"

    func run() async throws {
        print("[target-load] registering Qwen3InspModel as qwen3...")

        // Register our inspectable model type, overriding the default Qwen3Model.
        await LLMTypeRegistry.shared.registerModelType("qwen3") { data in
            let config = try JSONDecoder().decode(Qwen3InspConfiguration.self, from: data)
            return Qwen3InspModel(config)
        }

        let captureIndices = captureAt.split(separator: ",").compactMap { Int($0) }
        print("[target-load] capture at layers: \(captureIndices)")

        print("[target-load] loading \(target) via LLMModelFactory...")
        let t0 = CFAbsoluteTimeGetCurrent()

        // Use HuggingFace macros for Downloader + TokenizerLoader
        let downloader: Downloader = #hubDownloader()
        let tokenizerLoader: TokenizerLoader = #huggingFaceTokenizerLoader()
        let modelConfig = ModelConfiguration(id: target)
        let container = try await LLMModelFactory.shared.loadContainer(
            from: downloader,
            using: tokenizerLoader,
            configuration: modelConfig
        ) { progress in
            let pct = progress.fractionCompleted * 100
            if Int(pct) % 20 == 0 {
                print("[target-load] download \(String(format: "%.0f", pct))%")
            }
        }
        let loadMs = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        print("[target-load] loaded in \(String(format: "%.1f", loadMs / 1000))s")

        // Run inside the container's serial context
        try await container.perform { context in
            guard let model = context.model as? Qwen3InspModel else {
                print("[target-load] ERROR: loaded model is not Qwen3InspModel (type: \(type(of: context.model)))")
                return
            }
            print("[target-load] ✓ model is Qwen3InspModel")

            // Tokenize
            let prompt = self.prompt
            let tokenIds = try context.tokenizer.encode(text: prompt, addSpecialTokens: true)
            print("[target-load] prompt: \(prompt)")
            print("[target-load] token count: \(tokenIds.count)")

            // Forward pass with hidden state capture
            let inputs = MLXArray(tokenIds.map { Int32($0) }).reshaped([1, tokenIds.count])
            let tFwd = CFAbsoluteTimeGetCurrent()
            let (logits, captures) = model.forwardCapturing(inputs, cache: nil,
                                                              captureAt: captureIndices)
            logits.asArray(Float.self)  // force eval
            let fwdMs = (CFAbsoluteTimeGetCurrent() - tFwd) * 1000

            print("[target-load] forward: \(String(format: "%.1f", fwdMs)) ms")
            print("[target-load] logits shape: \(logits.shape)")
            print("[target-load] captured hidden states:")
            for (idx, cap) in zip(captureIndices, captures) {
                print("  layer \(idx): shape \(cap.shape) dtype \(cap.dtype)")
            }
            let concat = MLX.concatenated(captures, axis: -1)
            print("[target-load] concatenated target_hidden shape: \(concat.shape)")
            print("[target-load] ✓ hidden state capture works")
        }
    }
}
