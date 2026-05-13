import Foundation

/// Declarative table of the on-device ML models the app needs at runtime.
///
/// Each entry lists the files that comprise the model on disk plus the
/// expected SHA-256 of each — `ModelStore` uses these to decide whether
/// a local copy is valid or needs to be (re-)downloaded from the
/// pinned GitHub Release.
///
/// Workflow when a model is updated:
///   1. Run `scripts/fetch_models.sh` to (re-)hydrate weights locally.
///   2. Run `scripts/upload_models_to_github_release.sh` — uploads
///      assets to a new release tag and prints the Swift constants to
///      paste below.
///   3. Bump `releaseTag` here and replace the SHA-256 placeholders
///      with the printed values.
///   4. Ship. The next launch on each user's device sees a fresh
///      manifest version, wipes the install dir, and re-downloads.
enum ModelManifest {
    /// GitHub Release tag the assets are attached to. Bump when
    /// shipping new model versions.
    static let releaseTag = "models-v1"

    /// Owner/repo for the release. Hard-coded rather than derived from
    /// build settings so the URL is the same on every device regardless
    /// of how the app was signed.
    static let repository = "Supat/xephon"

    /// Subdirectory under `Application Support/` that holds downloaded
    /// model files. Versioned so future manifest bumps don't collide
    /// with old downloads.
    static var installSubdirectory: String { "Models/\(releaseTag)" }

    /// Resolved base URL for release assets:
    ///   https://github.com/<owner>/<repo>/releases/download/<tag>/<asset>
    static var releaseAssetBaseURL: URL {
        URL(string: "https://github.com/\(repository)/releases/download/\(releaseTag)/")!
    }

    /// Approximate total bytes the user will download on first launch.
    /// Surfaced in the setup view so they know what they're committing
    /// to before tapping Start.
    static var approximateTotalBytes: Int64 {
        entries.reduce(0) { $0 + $1.files.reduce(0) { $0 + $1.approximateBytes } }
    }

    /// Identifier for the on-device session-summary LLM. Stable across
    /// `releaseTag` bumps so RecordingController / ModelStore can keep
    /// referencing it without hard-coding the display name. Bumped
    /// 3B → Qwen3-8B once the increased-memory-limit entitlement
    /// lifted the per-app Jetsam ceiling on the 16 GB iPad —
    /// previous 7B attempts crashed at ~5 GB; with the entitlement
    /// the budget is ~10–11 GB, so an 8B at ~5 GB resident fits
    /// with sane headroom and meaningfully better instruction-
    /// following than the 3B.
    static let summarizerID = "qwen3-8b-4bit"

    /// Base URL for the pre-quantized Qwen MLX repo on Hugging Face.
    /// We can't host the safetensors on GitHub Releases (Free tier
    /// caps assets at 2 GB; the 8B weights are 4.6 GB), so the
    /// Qwen files are sourced directly from
    /// `mlx-community/Qwen3-8B-4bit` instead. Public anonymous
    /// read, no auth required, same trust posture as the original
    /// HF mirrors of the acoustic SER weights. Inference is still
    /// strictly on-device — only the one-time install fetch
    /// reaches the network.
    private static let qwenHFBaseURL = URL(
        string: "https://huggingface.co/mlx-community/Qwen3-8B-4bit/resolve/main/"
    )!

    private static func qwenRemote(_ filename: String) -> URL {
        qwenHFBaseURL.appendingPathComponent(filename)
    }

    /// Optional model entries the user has to deliberately opt into
    /// from Settings. Kept out of `entries` so first-launch hydration
    /// doesn't pull 4 GB without consent. `ModelStore.ensureOptional(id:)`
    /// downloads on demand once the user flips the summarizer toggle.
    ///
    /// Sourced from `mlx-community/Qwen2.5-7B-Instruct-4bit` on Hugging
    /// Face — the pre-quantized MLX build, identical in shape to what
    /// `mlx_lm.convert -q --q-bits 4` would produce locally but without
    /// the GPU-timeout cliff that local quantization of 7B hits on
    /// macOS. SHA-256s captured at developer-side fetch time
    /// (`scripts/fetch_models.sh --with-summarizer`) and pasted in.
    static let optionalEntries: [ModelEntry] = [
        ModelEntry(
            id: summarizerID,
            displayName: "Qwen3-8B (4-bit MLX)",
            files: [
                ModelFile(
                    assetName: "qwen3-8b-4bit-config.json",
                    installPath: "qwen3-8b-4bit/config.json",
                    bundleResource: BundleLookup(name: "config", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 939,
                    sha256: "e5485285fd7e289e76e9cffa112f6dc2e3426519082f7db9b69041589f81a218",
                    directRemoteURL: qwenRemote("config.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-tokenizer.json",
                    installPath: "qwen3-8b-4bit/tokenizer.json",
                    bundleResource: BundleLookup(name: "tokenizer", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 11_422_654,
                    sha256: "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
                    directRemoteURL: qwenRemote("tokenizer.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-tokenizer_config.json",
                    installPath: "qwen3-8b-4bit/tokenizer_config.json",
                    bundleResource: BundleLookup(name: "tokenizer_config", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 9_706,
                    sha256: "253153d0738ceb4c668d2eff957714dd2bea0b56de772a9fdccd96cbf517e6a0",
                    directRemoteURL: qwenRemote("tokenizer_config.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-special_tokens_map.json",
                    installPath: "qwen3-8b-4bit/special_tokens_map.json",
                    bundleResource: BundleLookup(name: "special_tokens_map", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 613,
                    sha256: "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
                    directRemoteURL: qwenRemote("special_tokens_map.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-added_tokens.json",
                    installPath: "qwen3-8b-4bit/added_tokens.json",
                    bundleResource: BundleLookup(name: "added_tokens", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 707,
                    sha256: "c0284b582e14987fbd3d5a2cb2bd139084371ed9acbae488829a1c900833c680",
                    directRemoteURL: qwenRemote("added_tokens.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-merges.txt",
                    installPath: "qwen3-8b-4bit/merges.txt",
                    bundleResource: BundleLookup(name: "merges", ext: "txt", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 1_671_853,
                    sha256: "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
                    directRemoteURL: qwenRemote("merges.txt")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-vocab.json",
                    installPath: "qwen3-8b-4bit/vocab.json",
                    bundleResource: BundleLookup(name: "vocab", ext: "json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 2_776_833,
                    sha256: "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
                    directRemoteURL: qwenRemote("vocab.json")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-model.safetensors",
                    installPath: "qwen3-8b-4bit/model.safetensors",
                    bundleResource: BundleLookup(name: "model", ext: "safetensors", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 4_607_835_174,
                    sha256: "f2d29621aab300336ad645567ff38c42aac755513006ef4e8a579cf7ef5256d8",
                    directRemoteURL: qwenRemote("model.safetensors")
                ),
                ModelFile(
                    assetName: "qwen3-8b-4bit-model.safetensors.index.json",
                    installPath: "qwen3-8b-4bit/model.safetensors.index.json",
                    bundleResource: BundleLookup(name: "model.safetensors", ext: "index.json", subdirectory: "qwen3-8b-4bit"),
                    approximateBytes: 64_065,
                    sha256: "3fb25463b4078b1fc27159daa605190029c2e965f533bf0b1b594f96cbfceb8a",
                    directRemoteURL: qwenRemote("model.safetensors.index.json")
                ),
            ]
        )
    ]

    static let entries: [ModelEntry] = [
        ModelEntry(
            id: "w2v2-msp-dim",
            displayName: "audeering W2V2 (V/A/D)",
            files: [
                ModelFile(
                    assetName: "w2v2-model.onnx",
                    installPath: "w2v2-msp-dim/model.onnx",
                    bundleResource: BundleLookup(name: "model", ext: "onnx", subdirectory: "w2v2-msp-dim"),
                    approximateBytes: 330_752_344,
                    sha256: "4306aaabb46cc8e2d0a40caac2b48bf2cbe706573866512fbd05ca1ebb60a8f7"
                )
            ]
        ),
        ModelEntry(
            id: "emotion2vec-plus-large",
            displayName: "emotion2vec+ (categorical)",
            files: [
                ModelFile(
                    assetName: "emotion2vec-model.onnx",
                    installPath: "emotion2vec_onnx/model.onnx",
                    bundleResource: BundleLookup(name: "model", ext: "onnx", subdirectory: "emotion2vec_onnx"),
                    approximateBytes: 418_215,
                    sha256: "b5bb76d2a68d54fb2544dce53c5128611d0e353f8246aaaee6e0f018f1c0a56f"
                ),
                ModelFile(
                    assetName: "emotion2vec-model.data",
                    installPath: "emotion2vec_onnx/model.data",
                    bundleResource: BundleLookup(name: "model", ext: "data", subdirectory: "emotion2vec_onnx"),
                    approximateBytes: 324_280_320,
                    sha256: "ed475684ee8d36a1299b19f827f9f83a9c4b16b7a781f8f374015254a2c225ae"
                ),
            ]
        ),
        ModelEntry(
            id: "wrime-roberta",
            displayName: "WRIME RoBERTa (text)",
            files: [
                ModelFile(
                    assetName: "wrime-model.onnx",
                    installPath: "wrime-roberta/model.onnx",
                    bundleResource: BundleLookup(name: "model", ext: "onnx", subdirectory: "wrime-roberta"),
                    approximateBytes: 221_472_656,
                    sha256: "2d02488dccc856c193e02b4c8759b0cea8285bdf244d90b3d26b3feea223c9e3"
                ),
                ModelFile(
                    assetName: "wrime-tokenizer.json",
                    installPath: "wrime-roberta/tokenizer.json",
                    bundleResource: BundleLookup(name: "tokenizer", ext: "json", subdirectory: "wrime-roberta"),
                    approximateBytes: 2_412_694,
                    sha256: "ea9b5801354f528f3b9073f3142d5e9a21cb919bd1bb8839f9787628df6894f4"
                ),
                ModelFile(
                    assetName: "wrime-tokenizer_config.json",
                    installPath: "wrime-roberta/tokenizer_config.json",
                    bundleResource: BundleLookup(name: "tokenizer_config", ext: "json", subdirectory: "wrime-roberta"),
                    approximateBytes: 1_411,
                    sha256: "8031b2a4f428f0a12089c12a0e8a6059554992607ce9d60033fddba08c306a5f"
                ),
                ModelFile(
                    assetName: "wrime-config.json",
                    installPath: "wrime-roberta/config.json",
                    bundleResource: BundleLookup(name: "config", ext: "json", subdirectory: "wrime-roberta"),
                    approximateBytes: 1_021,
                    sha256: "6da11037a2b57e01bae0f4a6070eac4e1b953e052c27ee5fd21ae3c410c7140f"
                ),
                ModelFile(
                    assetName: "wrime-special_tokens_map.json",
                    installPath: "wrime-roberta/special_tokens_map.json",
                    bundleResource: BundleLookup(name: "special_tokens_map", ext: "json", subdirectory: "wrime-roberta"),
                    approximateBytes: 969,
                    sha256: "f1711429f1addc9015fe127f2f4359c4e5a1e6f3c59a71073467bd552265cd26"
                ),
            ]
        ),
        ModelEntry(
            id: "w2v2-age-gender",
            displayName: "audeering W2V2 (age + gender)",
            files: [
                ModelFile(
                    assetName: "w2v2-age-gender-model.onnx",
                    installPath: "w2v2-age-gender/model.onnx",
                    bundleResource: BundleLookup(name: "model", ext: "onnx", subdirectory: "w2v2-age-gender"),
                    approximateBytes: 1_055_734,
                    sha256: "bf99916cb5bdd90f2d9bdf5c33593f6b3d6ddbab7c1ea7b8d5cb893c6c401854"
                ),
                // The dynamo ONNX exporter writes the weight blob as a
                // `model.onnx.data` sidecar next to `model.onnx`, and the
                // graph proto references it by that exact name — we
                // can't rename to the conventional `model.data` without
                // editing the graph. `BundleLookup` splits on the LAST
                // dot, so name="model.onnx" ext="data" resolves the
                // right file inside the bundle.
                ModelFile(
                    assetName: "w2v2-age-gender-model.data",
                    installPath: "w2v2-age-gender/model.onnx.data",
                    bundleResource: BundleLookup(name: "model.onnx", ext: "data", subdirectory: "w2v2-age-gender"),
                    approximateBytes: 363_266_048,
                    sha256: "680c9bcedc7d8ed065df3b54bb89087bfdbb8a921397d789fa2621b79897e766"
                ),
            ]
        ),
    ]
}

/// One model — possibly composed of several files (e.g. the emotion2vec
/// graph + its external-data sidecar must download as a pair).
struct ModelEntry: Sendable, Identifiable {
    let id: String
    let displayName: String
    let files: [ModelFile]
}

struct ModelFile: Sendable {
    /// Asset filename on the GitHub Release (flat namespace). Used
    /// when `directRemoteURL` is nil — the file is fetched from
    /// `ModelManifest.releaseAssetBaseURL.appendingPathComponent(assetName)`.
    let assetName: String
    /// Where this file should land relative to the install root —
    /// preserves the directory layout the SER inferencers expect
    /// (`emotion2vec_onnx/model.data` lives next to `model.onnx` etc.).
    let installPath: String
    /// Optional dev shortcut: if the same file is present in the app
    /// bundle (because `Models/` was non-empty at build time), use it
    /// directly instead of downloading.
    let bundleResource: BundleLookup
    /// Best-effort byte size for the progress UI before the HTTP
    /// `Content-Length` arrives. Not authoritative.
    let approximateBytes: Int64
    /// Lowercase hex SHA-256 of the released asset. Replace placeholders
    /// after running `scripts/upload_models_to_github_release.sh`.
    let sha256: String
    /// Optional override for the remote download URL. When set,
    /// `ModelStore.resolve` fetches from this URL instead of
    /// composing one off `releaseAssetBaseURL + assetName`. Used by
    /// the Qwen2.5-7B summarizer files, which exceed GitHub
    /// Releases' 2 GB Free-tier asset limit and instead come
    /// straight from `huggingface.co/mlx-community/…/resolve/main/`.
    /// SHA-256 is verified the same way regardless of source.
    let directRemoteURL: URL?

    init(
        assetName: String,
        installPath: String,
        bundleResource: BundleLookup,
        approximateBytes: Int64,
        sha256: String,
        directRemoteURL: URL? = nil
    ) {
        self.assetName = assetName
        self.installPath = installPath
        self.bundleResource = bundleResource
        self.approximateBytes = approximateBytes
        self.sha256 = sha256
        self.directRemoteURL = directRemoteURL
    }
}

struct BundleLookup: Sendable {
    let name: String
    let ext: String
    let subdirectory: String?

    func locate(in bundle: Bundle = .main) -> URL? {
        bundle.url(forResource: name, withExtension: ext, subdirectory: subdirectory)
            ?? bundle.url(forResource: name, withExtension: ext)
    }
}
