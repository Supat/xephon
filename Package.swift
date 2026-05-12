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
// `mlx-swift` + `mlx-swift-examples` are now pulled in — the Summarizer target uses
// them for the Qwen2.5-Instruct on-device session summarizer (Apache 2.0). All
// inference is local; we never reach a cloud API.

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
                "SERRuntime",
                "SERAcoustic",
                "SERText",
                "Fusion",
                "Export",
                "Summarizer",
            ]
        ),
        .library(name: "Audio",         targets: ["Audio"]),
        .library(name: "ASR",           targets: ["ASR"]),
        .library(name: "Diarization",   targets: ["Diarization"]),
        .library(name: "SERRuntime",    targets: ["SERRuntime"]),
        .library(name: "SERAcoustic",   targets: ["SERAcoustic"]),
        .library(name: "SERText",       targets: ["SERText"]),
        .library(name: "Fusion",        targets: ["Fusion"]),
        .library(name: "Export",        targets: ["Export"]),
        .library(name: "Summarizer",    targets: ["Summarizer"]),
        .library(name: "XephonLogging", targets: ["XephonLogging"]),
    ],
    dependencies: [
        .package(url: "https://github.com/FluidInference/FluidAudio", from: "0.5.0"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.20.0"),
        // `swift-transformers` pinned to 1.0.x because mlx-swift-examples
        // hasn't moved past that line yet — see resolver notes when bumping.
        // DeBERTaWRIME uses `AutoTokenizer.from(tokenizerConfig:tokenizerData:)`,
        // which exists in 1.0.x.
        .package(url: "https://github.com/huggingface/swift-transformers.git", "1.0.0"..<"1.1.0"),
        // On-device LLM runtime for the session summarizer (Qwen2.5-Instruct
        // 4-bit MLX). `mlx-swift` is the runtime; `mlx-swift-examples`
        // publishes the higher-level `MLXLLM` + `MLXLMCommon` products that
        // bundle loaders, tokenizer glue, and chat-template handling we'd
        // otherwise have to hand-roll. All inference is on-device; we never
        // reach a cloud endpoint.
        .package(url: "https://github.com/ml-explore/mlx-swift", from: "0.22.0"),
        .package(url: "https://github.com/ml-explore/mlx-swift-examples", from: "2.25.10"),
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
            name: "SERRuntime",
            dependencies: [
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Core/SER/Runtime"
        ),
        .target(
            name: "SERAcoustic",
            dependencies: [
                "XephonLogging",
                "Audio",
                "SERRuntime",
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "Core/SER/Acoustic"
        ),
        .target(
            name: "SERText",
            dependencies: [
                "XephonLogging",
                "SERRuntime",
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
        .target(
            name: "Summarizer",
            dependencies: [
                "XephonLogging",
                "Fusion",
                // `MLX` is the low-level runtime we use directly for
                // `MLX.GPU.set(cacheLimit:)` (memory tuning around the
                // 8B weights). Declared explicitly so SPM stops
                // flagging `mlx-swift` as "not used by any target" —
                // it would otherwise link only transitively through
                // mlx-swift-examples and SPM can't see through that.
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-examples"),
                .product(name: "MLXLMCommon", package: "mlx-swift-examples"),
            ],
            path: "Core/Summarizer"
        ),
    ],
    swiftLanguageModes: [.v6]
)
