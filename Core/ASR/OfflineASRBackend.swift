import Foundation

/// User-selectable offline ASR backend. Drives the `Transcriber`
/// instance held by `AnalysisPipeline` for the non-streaming paths:
/// file-mode analysis, per-utterance re-evaluation, and the
/// Edit Utterance sheet's Transcribe Range. The live recording
/// path uses Apple's `StreamingTranscriber` and is unaffected
/// by this choice — Qwen3-ASR doesn't conform to the streaming
/// protocol so live can't switch without a chunked wrapper that
/// doesn't exist yet.
///
/// Persisted via `UserDefaults` so the user's pick survives app
/// restarts. Defaults to `.speechAnalyzer` (Apple) so existing
/// installs keep their current behavior.
public enum OfflineASRBackend: String, Sendable, Hashable, CaseIterable, Codable {
    /// Apple's offline `SpeechAnalyzer` + `SpeechTranscriber`.
    /// Same locale gating as the streaming variant (requires the
    /// 16-core ANE on iPad Pro M-series; falls back to whatever
    /// the system ships for the locale otherwise).
    case speechAnalyzer
    /// FluidAudio's Qwen3-ASR (Core ML). Strong on
    /// Japanese/Chinese/Korean/Vietnamese; downloads on first
    /// use. No word-level timestamps — file-mode utterances
    /// arrive as single-sentence blobs.
    case qwen3ASR
}
