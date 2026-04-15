// swift-tools-version:6.0
// Swift package for native ANE benchmarking and (WIP) native SD runner.
import PackageDescription

let package = Package(
    name: "swift-bench",
    platforms: [.macOS(.v15)],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .executableTarget(
            name: "ane-latency-bench",
            path: "Sources/ane-latency-bench"
        ),
        .executableTarget(
            name: "dflash-swift-runner",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/dflash-swift-runner"
        ),
    ]
)
