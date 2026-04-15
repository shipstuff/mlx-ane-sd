// DFlash speculative decoding runner in Swift (WIP).
//
// Current scope: draft-side harness with full profiling.
//   - Loads compiled DFlash ANE draft
//   - Runs N cycles with realistic inputs (pseudo-random representative data)
//   - Commits via accumcache pattern (matches Python reference)
//   - Records per-phase timing via Profiler
//
// Pending (next iteration):
//   - mlx-swift-lm Qwen3 target integration
//   - Hidden state capture at target_layer_ids
//   - Full SD loop (draft + verify + accept)
//   - Multi-stream worker pool

import Foundation
import CoreML
import ArgumentParser
import DFlashCore

@main
struct DFlashRunner: AsyncParsableCommand {
    @Option(help: "Path to compiled DFlash draft .mlmodelc")
    var draft: String = "/tmp/dflash_ane_accum_c/dflash_ane_accum.mlmodelc"

    @Option(help: "State length matching the compiled draft")
    var stateLength: Int = 256

    @Option(help: "Number of draft cycles to simulate")
    var cycles: Int = 30

    @Option(help: "Simulated acceptance rate per cycle (0.0-1.0). Drives committed count.")
    var acceptRate: Double = 0.5

    @Option(help: "JSON output path for profile (omit for stdout only)")
    var profileOut: String = ""

    @Flag(help: "Keep full sample arrays in profile (larger JSON)")
    var keepSamples: Bool = false

    @Flag(help: "Output only JSON (for scripting)")
    var json: Bool = false

    func run() async throws {
        let profiler = Profiler(keepSamples: keepSamples)
        let config = DraftConfig(stateLength: stateLength)

        if !json {
            print("[runner] loading draft \(draft) (S=\(stateLength))...")
        }

        let loadStart = CFAbsoluteTimeGetCurrent()
        let draftModel = try DFlashANEDraft(mlmodelcPath: draft, config: config, profiler: profiler)
        let loadMs = (CFAbsoluteTimeGetCurrent() - loadStart) * 1000
        if !json { print("[runner] loaded in \(String(format: "%.1f", loadMs)) ms") }

        // Build representative inputs (zeros with slight variation)
        let noiseEmbed = try makeInputArray([1, config.blockSize, config.hiddenSize])
        let targetHidden = try makeInputArray([1, config.ctxSize, config.concatDim])

        // Warmup
        if !json { print("[runner] warmup (5 cycles)...") }
        draftModel.resetCache()
        for _ in 0..<5 {
            let _ = try await draftModel.forward(noiseEmbedding: noiseEmbed,
                                                   targetHidden: targetHidden,
                                                   sReal: config.ctxSize)
        }

        // Timed run
        if !json { print("[runner] timed run (\(cycles) cycles, accept_rate=\(acceptRate))...") }
        draftModel.resetCache()
        let runStart = CFAbsoluteTimeGetCurrent()
        for _ in 0..<cycles {
            let output = try await profiler.measureAsync("cycle_total") {
                try await draftModel.forward(noiseEmbedding: noiseEmbed,
                                               targetHidden: targetHidden,
                                               sReal: config.ctxSize)
            }
            let accepted = Int(Double(config.blockSize - 1) * acceptRate)
            draftModel.commit(newK: output.newK, newV: output.newV,
                              sReal: config.ctxSize, accepted: accepted)
        }
        let runMs = (CFAbsoluteTimeGetCurrent() - runStart) * 1000

        if !json {
            profiler.printSummary()
            print("")
            print("[runner] run wall time: \(String(format: "%.1f", runMs)) ms for \(cycles) cycles")
            print(String(format: "[runner] mean cycle wall: %.2f ms", runMs / Double(cycles)))
        } else {
            // Emit compact JSON: { "load_ms": ..., "cycles": N, "run_ms": ..., "profile": {...} }
            struct Result: Codable {
                let loadMs: Double
                let cycles: Int
                let runMs: Double
                let profile: Profiler.Snapshot
            }
            let result = Result(
                loadMs: loadMs,
                cycles: cycles,
                runMs: runMs,
                profile: profiler.snapshot()
            )
            let enc = JSONEncoder()
            enc.outputFormatting = [.sortedKeys]
            let data = try enc.encode(result)
            if let s = String(data: data, encoding: .utf8) { print(s) }
        }

        if !profileOut.isEmpty {
            let url = URL(fileURLWithPath: profileOut)
            try profiler.writeJSON(to: url)
            if !json { print("[runner] profile written to \(profileOut)") }
        }
    }

    private func makeInputArray(_ shape: [Int]) throws -> MLMultiArray {
        let arr = try MLMultiArray(shape: shape.map { NSNumber(value: $0) }, dataType: .float16)
        // Fill with small deterministic values (matches Python's det(shape) pattern)
        let ptr = arr.dataPointer.assumingMemoryBound(to: UInt16.self)
        for i in 0..<arr.count {
            ptr[i] = UInt16(i & 0xFFFF)
        }
        return arr
    }
}
