// swift-tools-version:6.0
// Swift package for native ANE benchmarking and native SD runner.
import PackageDescription

let package = Package(
    name: "swift-bench",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm", branch: "main"),
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.31.3"),
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.0.0"),
        .package(url: "https://github.com/huggingface/swift-huggingface", branch: "main"),
    ],
    targets: [
        // Shared library: Qwen3 inspectable, profiler, DFlash ANE draft wrapper.
        .target(
            name: "DFlashCore",
            dependencies: [
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
            ],
            path: "Sources/DFlashCore"
        ),
        // Minimal: raw ANE predict latency measurement (no MLX deps).
        .executableTarget(
            name: "ane-latency-bench",
            path: "Sources/ane-latency-bench"
        ),
        // Full SD draft harness with profiling (no target integration yet).
        .executableTarget(
            name: "dflash-swift-runner",
            dependencies: [
                "DFlashCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/dflash-swift-runner"
        ),
        // Smoke test for target model loading + hidden state capture.
        .executableTarget(
            name: "target-load-test",
            dependencies: [
                "DFlashCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/target-load-test"
        ),
        // Full SD loop: target (MLX) + draft (ANE) with accept/trim logic.
        .executableTarget(
            name: "dflash-sd",
            dependencies: [
                "DFlashCore",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "MLXHuggingFace", package: "mlx-swift-lm"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "HuggingFace", package: "swift-huggingface"),
            ],
            path: "Sources/dflash-sd"
        ),
    ]
)
