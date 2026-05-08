import Foundation
import Audio
import ASR
import Diarization
import SERAcoustic
import SERText
import Fusion
import XephonLogging

// App-level orchestrator: capture buffer → ASR → SER (acoustic + text) → late fusion.
//
// SER and diarization paths are optional. When their actors are nil — or when
// they throw `.notImplemented` because the model weights haven't been hydrated
// yet via scripts/fetch_models.sh — the pipeline degrades gracefully:
// the corresponding modality is dropped, fusion runs with whatever is left.
/// Per-segment timing snapshot returned by `processSegment`. Used by the
/// pipeline visualization to show last-stage latency.
public struct ProcessingMetrics: Sendable, Hashable {
    public let acousticDuration: TimeInterval?
    public let textDuration: TimeInterval?
    public let totalDuration: TimeInterval
}

// Intentionally NOT @MainActor: heavy SER constructors (e.g. W2V2 ONNX load,
// ~631 MB) and per-segment inference must not block the UI thread. All stored
// references are themselves Sendable (actors / value types).
final class AnalysisPipeline: Sendable {
    private let transcriber: any Transcriber
    private let diarizer: (any Diarizer)?
    private let dimensionalSER: (any DimensionalAcousticSER)?
    private let categoricalSER: (any CategoricalAcousticSER)?
    private let textSER: (any TextSER)?
    private let fuser: any Fuser

    init(
        transcriber: any Transcriber = SpeechAnalyzerTranscriber(),
        diarizer: (any Diarizer)? = nil,
        dimensionalSER: (any DimensionalAcousticSER)? = nil,
        categoricalSER: (any CategoricalAcousticSER)? = nil,
        textSER: (any TextSER)? = nil,
        fuser: any Fuser = LateFusion()
    ) {
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.dimensionalSER = dimensionalSER
        self.categoricalSER = categoricalSER
        self.textSER = textSER
        self.fuser = fuser
    }

    /// Auto-detects bundled models. Modalities whose models aren't found are
    /// skipped — the pipeline still runs with whatever is available.
    /// Heavy SER constructors run here, so always call this off the MainActor.
    /// `enableDiarization` defaults to false because FluidAudio downloads its
    /// segmentation/embedding models on first use (~50 MB) — single-speaker
    /// fallback (S01) is reasonable for the common solo-recording case.
    static func autoConfigured(enableDiarization: Bool = false) async -> AnalysisPipeline {
        AppLog.app.info("AnalysisPipeline.autoConfigured starting…")

        let diarizer: (any Diarizer)? = enableDiarization ? FluidAudioDiarizer() : nil

        let dimensional: (any DimensionalAcousticSER)? = Self.tryInit(
            "W2V2 dimensional SER",
            { try W2V2DimensionalSER() }
        )
        let categorical: (any CategoricalAcousticSER)? = Self.tryInit(
            "emotion2vec categorical SER",
            { try Emotion2VecCategoricalSER() }
        )
        let deberta: (any TextSER)? = await Self.tryInitAsync(
            "DeBERTa WRIME text SER",
            { try await DeBERTaWRIME() }
        )
        let textSER: any TextSER = SwitchingTextSER(
            deberta: deberta,
            foundationModels: FoundationModelsSER()
        )

        AppLog.app.info(
            "AnalysisPipeline ready: dimensional=\(dimensional != nil, privacy: .public), categorical=\(categorical != nil, privacy: .public), text=\(textSER != nil, privacy: .public), diarizer=\(diarizer != nil, privacy: .public)"
        )

        return AnalysisPipeline(
            diarizer: diarizer,
            dimensionalSER: dimensional,
            categoricalSER: categorical,
            textSER: textSER
        )
    }

    private static func tryInit<T>(
        _ name: String,
        _ build: () throws -> T
    ) -> T? {
        do { return try build() }
        catch {
            AppLog.app.warning("\(name, privacy: .public) unavailable: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private static func tryInitAsync<T>(
        _ name: String,
        _ build: () async throws -> T
    ) async -> T? {
        do { return try await build() }
        catch {
            AppLog.app.warning("\(name, privacy: .public) unavailable: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    // MARK: - Text SER backend control (forwards to SwitchingTextSER if present)

    func availableTextSERBackends() async -> [SwitchingTextSER.Backend] {
        await (textSER as? SwitchingTextSER)?.availableBackends ?? []
    }

    func currentTextSERBackend() async -> SwitchingTextSER.Backend? {
        await (textSER as? SwitchingTextSER)?.currentBackend
    }

    func setTextSERBackend(_ backend: SwitchingTextSER.Backend) async {
        await (textSER as? SwitchingTextSER)?.setBackend(backend)
    }

    func analyze(_ buffer: AudioChunk) async throws -> [UtteranceEstimate] {
        AppLog.app.info("analyze: \(buffer.samples.count, privacy: .public) samples (\(Double(buffer.samples.count) / buffer.sampleRate, privacy: .public)s)")

        // 1. ASR over the full buffer.
        let asrStart = Date()
        let asrSegments = try await transcriber.transcribe(buffer)
        AppLog.app.info("ASR produced \(asrSegments.count, privacy: .public) segments in \(Date().timeIntervalSince(asrStart), privacy: .public)s")

        // 2. Diarization (best-effort; falls back to single-speaker S01).
        let diarized = await runDiarization(buffer)

        // 3. Per-segment SER + fuse.
        var estimates: [UtteranceEstimate] = []
        for segment in asrSegments {
            let speakerID = speakerForSegment(segment, in: diarized)
            let segmentBuffer = sliceBuffer(buffer, start: segment.start, end: segment.end)
            let (estimate, _) = try await processSegment(
                asr: segment,
                segmentAudio: segmentBuffer,
                speakerID: speakerID
            )
            estimates.append(estimate)
        }
        return estimates
    }

    /// Per-segment SER + fusion. Used in both batch (`analyze(_:)`) and
    /// streaming flows. Caller is responsible for slicing the audio segment
    /// out of the source buffer.
    func processSegment(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        speakerID: String = "S01"
    ) async throws -> (UtteranceEstimate, ProcessingMetrics) {
        let totalStart = Date()
        let acousticStart = Date()
        async let dimensional = runDimensional(segmentAudio)
        async let categorical = runCategorical(segmentAudio)
        async let plutchik = runText(asr.text)

        // Acoustic timing = wall time until both dimensional + categorical
        // resolve (they're concurrent). Text timing = wall time until plutchik
        // resolves. The two overlap in real time but are reported separately.
        let textStart = Date()
        let txt = await plutchik
        let textDuration = Date().timeIntervalSince(textStart)

        let dim = await dimensional
        let cat = await categorical
        let acousticDuration = Date().timeIntervalSince(acousticStart)

        let baseEstimate = try await fuser.fuse(
            asr: asr,
            speakerID: speakerID,
            dimensional: dim,
            acousticCategorical: cat,
            plutchik: txt
        )
        // Stamp which text backend produced `plutchik`, so the UI can badge it.
        // Nil when text SER was skipped (filler / empty / no model).
        let textBackend: String? = txt == nil
            ? nil
            : await (textSER as? SwitchingTextSER)?.currentBackend.rawValue
        let estimate = baseEstimate.withTextBackend(textBackend)
        let metrics = ProcessingMetrics(
            acousticDuration: dim != nil || cat != nil ? acousticDuration : nil,
            textDuration: txt != nil ? textDuration : nil,
            totalDuration: Date().timeIntervalSince(totalStart)
        )
        return (estimate, metrics)
    }

    // MARK: - Optional stages (return nil on failure → degraded fusion)

    private func runDiarization(_ buffer: AudioChunk) async -> [DiarizedSegment] {
        guard let diarizer else { return [] }
        do {
            return try await diarizer.diarize(buffer)
        } catch {
            AppLog.app.warning("diarization unavailable: \(String(describing: error), privacy: .public)")
            return []
        }
    }

    private func runDimensional(_ buffer: AudioChunk) async -> VADScore? {
        guard let dimensionalSER else { return nil }
        do { return try await dimensionalSER.score(buffer) } catch {
            AppLog.app.debug("dimensional SER skipped: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private func runCategorical(_ buffer: AudioChunk) async -> CategoricalEmotion? {
        guard let categoricalSER else { return nil }
        do { return try await categoricalSER.score(buffer) } catch {
            AppLog.app.debug("categorical SER skipped: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private func runText(_ text: String) async -> PlutchikScore? {
        guard let textSER, !text.isEmpty else { return nil }
        if Self.isFiller(text) {
            AppLog.app.debug("text SER skipped (filler): \(text, privacy: .public)")
            return nil
        }
        do { return try await textSER.classify(text) } catch {
            AppLog.app.debug("text SER skipped: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    /// Backchannels and ultra-short utterances rarely carry useful affect
    /// signal; running a ~1 s LLM round-trip on them is mostly waste.
    /// Conservative list — only obvious fillers, no content words.
    private static let fillers: Set<String> = [
        "あの", "えーと", "えっと", "えと", "うーん", "うんうん",
        "うん", "ええ", "はい", "いえ", "そう", "そうそう",
        "そうですね", "なるほど", "ふむ", "へえ", "ああ", "おお",
    ]

    private static func isFiller(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.count <= 1 { return true }
        return fillers.contains(trimmed)
    }

    // MARK: - Buffer slicing

    private func speakerForSegment(_ asr: ASRSegment, in diarized: [DiarizedSegment]) -> String {
        guard !diarized.isEmpty else { return "S01" }
        let mid = (asr.start + asr.end) / 2
        // Pick the speaker whose segment contains the midpoint; otherwise the closest.
        if let containing = diarized.first(where: { $0.start <= mid && mid <= $0.end }) {
            return containing.speakerID
        }
        let closest = diarized.min(by: { abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid) })
        return closest?.speakerID ?? "S01"
    }

    private func sliceBuffer(_ buffer: AudioChunk, start: TimeInterval, end: TimeInterval) -> AudioChunk {
        Self.slice(buffer, start: start, end: end)
    }

    /// Slice a captured-audio buffer to [start, end] seconds. Public for
    /// streaming callers that already hold the cumulative samples.
    static func slice(_ buffer: AudioChunk, start: TimeInterval, end: TimeInterval) -> AudioChunk {
        let total = Double(buffer.samples.count)
        let startIndex = max(0, min(Int(start * buffer.sampleRate), buffer.samples.count))
        let endIndex = max(startIndex, min(Int(end * buffer.sampleRate), buffer.samples.count))
        guard startIndex < endIndex, total > 0 else { return buffer }
        let slice = Array(buffer.samples[startIndex..<endIndex])
        return AudioChunk(samples: slice, sampleRate: buffer.sampleRate, timestamp: start)
    }
}
