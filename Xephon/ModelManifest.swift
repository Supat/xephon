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
    /// Asset filename on the GitHub Release (flat namespace).
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
