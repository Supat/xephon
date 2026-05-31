import Foundation

/// User-selectable ASR backend. Drives both the `Transcriber`
/// instance held by `AnalysisPipeline` for non-streaming paths
/// (file analysis, per-utterance re-evaluation, the Edit
/// Utterance sheet's Transcribe Range) AND the
/// `StreamingTranscriber` instance held by `RecordingController`
/// for live recording. Qwen3 in live mode goes through
/// `StreamingQwen3ASRTranscriber` — an actor that buffers
/// incoming audio into ~8 s chunks and transcribes them
/// serially via Qwen3's one-shot API (with the latency trade-off
/// documented there).
///
/// The type and UserDefaults key keep their legacy
/// `offlineASRBackend` name for backward compatibility (the
/// pick used to only affect offline paths); user-facing labels
/// are language-neutral.
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
