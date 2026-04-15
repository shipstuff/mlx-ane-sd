// Lightweight per-phase profiler for the SD runner.
//
// Designed to be cheap (single-timestamp bookend per phase) and
// comprehensive (covers draft, verify, sampling, cache ops, data
// marshaling). Dump structured JSON at end of run so Python /
// spreadsheets can process.
//
// Usage:
//   var prof = Profiler()
//   prof.begin("draft_call")
//   // ... work ...
//   prof.end("draft_call")
//   ...
//   prof.printSummary()
//   prof.writeJSON(to: url)

import Foundation

public final class Profiler: @unchecked Sendable {
    public struct PhaseStats: Codable {
        public var calls: Int = 0
        public var totalMs: Double = 0
        public var minMs: Double = .infinity
        public var maxMs: Double = 0
        public var samples: [Double] = []  // kept for pXX if enabled

        public var meanMs: Double { calls > 0 ? totalMs / Double(calls) : 0 }
        public mutating func add(ms: Double, keepSamples: Bool) {
            calls += 1
            totalMs += ms
            minMs = Swift.min(minMs, ms)
            maxMs = Swift.max(maxMs, ms)
            if keepSamples { samples.append(ms) }
        }
    }

    public struct Snapshot: Codable {
        public let phases: [String: PhaseStats]
        public let totalMs: Double
        public let startTime: String
    }

    private var phases: [String: PhaseStats] = [:]
    private var active: [String: CFAbsoluteTime] = [:]
    private let t0 = CFAbsoluteTimeGetCurrent()
    private let keepSamples: Bool

    public init(keepSamples: Bool = false) {
        self.keepSamples = keepSamples
    }

    public func begin(_ name: String) {
        active[name] = CFAbsoluteTimeGetCurrent()
    }

    public func end(_ name: String) {
        guard let start = active.removeValue(forKey: name) else { return }
        let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
        phases[name, default: PhaseStats()].add(ms: ms, keepSamples: keepSamples)
    }

    /// Time a closure and record it.
    public func measure<T>(_ name: String, _ block: () throws -> T) rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            phases[name, default: PhaseStats()].add(ms: ms, keepSamples: keepSamples)
        }
        return try block()
    }

    public func measureAsync<T>(_ name: String, _ block: () async throws -> T) async rethrows -> T {
        let start = CFAbsoluteTimeGetCurrent()
        defer {
            let ms = (CFAbsoluteTimeGetCurrent() - start) * 1000
            phases[name, default: PhaseStats()].add(ms: ms, keepSamples: keepSamples)
        }
        return try await block()
    }

    public func snapshot() -> Snapshot {
        let total = (CFAbsoluteTimeGetCurrent() - t0) * 1000
        let fmt = ISO8601DateFormatter()
        return Snapshot(
            phases: phases,
            totalMs: total,
            startTime: fmt.string(from: Date(timeIntervalSince1970: Date().timeIntervalSince1970 - total / 1000))
        )
    }

    public func printSummary() {
        let snap = snapshot()
        print("")
        print("=== Phase profile (\(String(format: "%.1f", snap.totalMs)) ms total) ===")
        let sortedPhases = phases.sorted { $0.value.totalMs > $1.value.totalMs }
        for (name, s) in sortedPhases {
            let pct = s.totalMs / snap.totalMs * 100
            let minV = s.minMs == .infinity ? 0 : s.minMs
            // Pad name manually (avoid %s format issues with Swift String).
            let padded = name.padding(toLength: 28, withPad: " ", startingAt: 0)
            let line = String(
                format: "  %@  %6d × %6.2f ms = %7.1f ms (%4.1f%%, min %5.2f / max %5.2f)",
                padded as NSString, s.calls, s.meanMs, s.totalMs, pct, minV, s.maxMs
            )
            print(line)
        }
    }

    public func writeJSON(to url: URL) throws {
        let snap = snapshot()
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(snap)
        try data.write(to: url)
    }
}
