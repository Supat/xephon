import Foundation
import XephonLogging
import MLX
import MLXLLM
import MLXLMCommon

/// Centralized container-loading helper for the MLX-backed actors.
/// `MLXQwenSummarizer` and `MLXQwenTranscriptionReviewer` previously
/// duplicated the same five-step sequence (cache-limit tune,
/// `ModelConfiguration` build, `LLMModelFactory.loadContainer`, plus
/// an idempotency check and a log line). Centralizing it here means
/// a future MLX cache-strategy bump or a switch to a different
/// loader is a one-file edit.
///
/// The loader throws the underlying MLX error verbatim; callers are
/// responsible for re-throwing it as their domain error type
/// (`SummarizerError.modelLoadFailed` /
/// `TranscriptionReviewError.modelLoadFailed`), since those carry
/// the user-facing message strings each actor wants to surface.
public enum MLXContainerLoader {

    /// Cap on MLX's Metal buffer cache. The default is bounded by
    /// Metal's `recommendedMaxWorkingSetSize`, which on a 16 GB
    /// iPad sits high enough to push the process over the Jetsam
    /// ceiling once Qwen weights and the SER pipeline coexist.
    /// 32 MB is the value the mlx-swift docs recommend for LLM
    /// evaluation on iOS.
    public static let defaultCacheLimitBytes = 32 * 1024 * 1024

    /// Resolve `directory` into a loaded `ModelContainer`. The
    /// directory must contain `config.json`, `tokenizer.json`, and
    /// the safetensors shards — same shape `mlx-community/*`
    /// publishes on Hugging Face.
    ///
    /// `logLabel` is folded into the info log line so a tail of the
    /// device log identifies which actor pulled the model in.
    public static func load(
        directory: URL,
        logLabel: String,
        cacheLimitBytes: Int = MLXContainerLoader.defaultCacheLimitBytes
    ) async throws -> ModelContainer {
        AppLog.app.info(
            "\(logLabel, privacy: .public) loading from \(directory.path, privacy: .public)"
        )
        MLX.GPU.set(cacheLimit: cacheLimitBytes)
        let configuration = ModelConfiguration(directory: directory)
        let container = try await LLMModelFactory.shared.loadContainer(
            configuration: configuration
        )
        AppLog.app.info("\(logLabel, privacy: .public) loaded")
        return container
    }
}
