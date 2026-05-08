// swift-tools-version:6.0
import PackageDescription

// Local Swift package containing the Core/* library targets used by the Xephon app.
//
// Layout intentionally mirrors the directory tree described in CLAUDE.md so that
// each subsystem is its own module — `import ASR`, `import Fusion`, etc.
//
// External dependency versions below are loose (`from:`) on first scaffold; they
// will be pinned in `Package.resolved` after the first `swift package resolve`.
//
// `mlx-swift` is intentionally omitted until an LLM path is enabled (per CLAUDE.md).

let package = Package(
    name: "Xephon",
    defaultLocalization: "en",
    platforms: [
        .iOS("26.0"),
        .macOS("26.0"),
    ],
    products: [
        .library(
            name: "XephonCore",
            targets: [
                "XephonLogging",
                "Audio",
                "ASR",
                "Diarization",
                "SERAcoustic",
                "SERText",
                "Fusion",
                "Export",
            ]
        ),
        .library(name: "Audio",         targets: ["Audio"]),
        .library(name: "ASR",           targets: ["ASR"]),
        .library(name: "Diarization",   targets: ["Diarization"]),
        .library(name: "SERAcoustic",   targets: ["SERAcoustic"]),
        .library(name: "SERText",       targets: ["SERText"]),
        .library(name: "Fusion",        targets: ["Fusion"]),
        .library(name: "Export",        targets: ["Export"]),
        .library(name: "XephonLogging", targets: ["XephonLogging"]),
    ],
    dependencies: [
        .package(url: "https://github.com/argmaxinc/WhisperKit", from: "0.10.0"),
        .package(url: "https://github.com/FluidInference/FluidAudio", from: "0.5.0"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0"),
        // For DeBERTa-WRIME tokenization (already a transitive dep via WhisperKit).
        .package(url: "https://github.com/huggingface/swift-transformers.git", from: "1.0.0"),
    ],
    targets: [
        .target(
            name: "XephonLogging",
            path: "Core/Logging"
        ),
        .target(
            name: "Audio",
            dependencies: ["XephonLogging"],
            path: "Core/Audio"
        ),
        .target(
            name: "ASR",
            dependencies: [
                "XephonLogging",
                "Audio",
                .product(name: "WhisperKit", package: "WhisperKit"),
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Core/ASR"
        ),
        .target(
            name: "Diarization",
            dependencies: [
                "XephonLogging",
                "Audio",
                .product(name: "FluidAudio", package: "FluidAudio"),
            ],
            path: "Core/Diarization"
        ),
        .target(
            name: "SERAcoustic",
            dependencies: [
                "XephonLogging",
                "Audio",
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Core/SER/Acoustic"
        ),
        .target(
            name: "SERText",
            dependencies: [
                "XephonLogging",
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Core/SER/Text"
        ),
        .target(
            name: "Fusion",
            dependencies: [
                "XephonLogging",
                "ASR",
                "SERAcoustic",
                "SERText",
            ],
            path: "Core/Fusion"
        ),
        .target(
            name: "Export",
            dependencies: ["XephonLogging", "Fusion"],
            path: "Core/Export"
        ),
    ],
    swiftLanguageModes: [.v6]
)
