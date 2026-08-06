import Foundation

/// Live progress of one MLX generation call, for UI display during
/// the multi-minute summarize/review runs (TurboFieldfare-style:
/// the model's own progress events replace an opaque spinner).
///
/// Emission points sample the generation loop: the cancellable
/// prefill reports once per 128-token chunk, the decode loop is
/// throttled to ~2 Hz. Consumers receive values on the emitting
/// (background) task — hop to the main actor before touching UI
/// state.
///
/// Delivery is via the `handler` task-local rather than plumbed
/// parameters: the app sets it around an actor call
/// (`MLXGenerationProgress.$handler.withValue(...) { ... }`) and it
/// propagates through the actor into `ModelContainer.perform` and
/// the synchronous generate callbacks — no signature changes across
/// the orchestration layers, and backends that never set it
/// (Apple FM, LM Studio) emit nothing.
public struct MLXGenerationProgress: Sendable, Equatable {
    public enum Phase: Sendable, Equatable {
        /// Prompt tokens pushed through the model so far. A warm
        /// prefix-cache start reports its reused tokens as already
        /// processed, so a retry visibly jumps ahead.
        case prefill(processedTokens: Int, totalTokens: Int)
        /// Cumulative generated tokens and the decode rate since
        /// the first token.
        case decoding(generatedTokens: Int, tokensPerSecond: Double)
    }

    public let phase: Phase

    public init(phase: Phase) {
        self.phase = phase
    }

    @TaskLocal public static var handler: (@Sendable (MLXGenerationProgress) -> Void)?
}
