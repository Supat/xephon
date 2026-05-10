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

/// Result of `autoConfigured(modelStore:)`. Carries the pipeline plus
/// any per-modality construction errors so the UI can surface them
/// rather than silently dropping the affected modality. An empty
/// `diagnostics` array means everything loaded.
struct AutoConfiguredPipeline: Sendable {
    let pipeline: AnalysisPipeline
    let diagnostics: [String]
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
    /// Bridges per-call diarizer outputs to session-stable speaker IDs
    /// by time-overlap matching. Reset between sessions via
    /// `resetSpeakerTracking()`.
    private let speakerTracker: StreamingSpeakerTracker

    init(
        transcriber: any Transcriber = SpeechAnalyzerTranscriber(),
        diarizer: (any Diarizer)? = nil,
        dimensionalSER: (any DimensionalAcousticSER)? = nil,
        categoricalSER: (any CategoricalAcousticSER)? = nil,
        textSER: (any TextSER)? = nil,
        fuser: any Fuser = LateFusion(),
        speakerTracker: StreamingSpeakerTracker = StreamingSpeakerTracker()
    ) {
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.dimensionalSER = dimensionalSER
        self.categoricalSER = categoricalSER
        self.textSER = textSER
        self.fuser = fuser
        self.speakerTracker = speakerTracker
    }

    /// Clear cumulative speaker history so the next session starts at S01.
    /// Pipeline instances are pre-warmed and reused across recordings, so
    /// without this the speaker numbering would carry over.
    func resetSpeakerTracking() async {
        await speakerTracker.reset()
    }

    /// Auto-configures from a `ModelStore` that has already resolved each
    /// model's on-disk URL (downloading from the GitHub Release on first
    /// launch if needed). Modalities whose models couldn't be resolved
    /// are skipped — the pipeline still runs with whatever is available.
    /// Heavy SER constructors run here, so always call this off the MainActor.
    /// `enableDiarization` defaults to true. FluidAudio downloads its
    /// segmentation/embedding models on first use (~50 MB), so the first
    /// session pays a one-time download cost; subsequent runs use the
    /// cache. For solo recordings the diarizer just always returns one
    /// speaker, so leaving it on is harmless — for multi-speaker
    /// conversations it's the difference between every utterance being
    /// labeled S01 and getting actual speaker identification.
    static func autoConfigured(
        modelStore: ModelStore,
        enableDiarization: Bool = true
    ) async -> AutoConfiguredPipeline {
        AppLog.app.info("AnalysisPipeline.autoConfigured starting…")

        // Per-modality diagnostics aggregated as we go. Empty on success;
        // non-empty surfaces in the UI as a banner so the user knows
        // *which* component fell back rather than silently losing it.
        var diagnostics: [String] = []

        let diarizer: (any Diarizer)? = enableDiarization ? FluidAudioDiarizer() : nil

        let w2v2URL: URL? = await Self.tryResolveAsync("W2V2", path: "w2v2-msp-dim/model.onnx", store: modelStore, diagnostics: &diagnostics)
        let dimensional: (any DimensionalAcousticSER)? = w2v2URL.flatMap { url in
            Self.tryInit("W2V2 dimensional SER", diagnostics: &diagnostics) {
                try W2V2DimensionalSER(modelURL: url)
            }
        }

        let emotion2vecURL: URL? = await Self.tryResolveAsync("emotion2vec", path: "emotion2vec_onnx/model.onnx", store: modelStore, diagnostics: &diagnostics)
        let categorical: (any CategoricalAcousticSER)? = emotion2vecURL.flatMap { url in
            Self.tryInit("emotion2vec categorical SER", diagnostics: &diagnostics) {
                try Emotion2VecCategoricalSER(modelURL: url)
            }
        }

        // wrime needs both an ONNX file and a tokenizer directory. Resolve
        // both — they live under the same `wrime-roberta/` subdir.
        let wrimeModelURL: URL? = await Self.tryResolveAsync("wrime model", path: "wrime-roberta/model.onnx", store: modelStore, diagnostics: &diagnostics)
        let wrimeTokenizerURL: URL? = await Self.tryResolveAsync("wrime tokenizer", path: "wrime-roberta/tokenizer.json", store: modelStore, diagnostics: &diagnostics)
        let wrimeTokenizerDir = wrimeTokenizerURL?.deletingLastPathComponent()
        let deberta: (any TextSER)?
        if let model = wrimeModelURL, let dir = wrimeTokenizerDir {
            deberta = await Self.tryInitAsync("DeBERTa WRIME text SER", diagnostics: &diagnostics) {
                // CPU-only by design: under fast-pace file analysis the
                // two acoustic SER models on CoreML EP already saturate
                // ANE/GPU memory (CoreML allocates IOSurface-backed
                // tensor buffers per shape per running inference).
                // Adding a third CoreML EP session for DeBERTa was the
                // direct trigger of `IOSurface creation failed:
                // kIOReturnNoMemory` cascades that culminated in a
                // SIGABRT during AudioFileCapture's pump. DeBERTa's
                // matmuls are small (seq 128 × hidden 768) and
                // Accelerate handles them comfortably on CPU; the
                // ~50 ms added latency per segment is far below ASR's
                // per-segment budget.
                try await DeBERTaWRIME(modelURL: model, tokenizerDirectory: dir, useCoreML: false)
            }
        } else {
            deberta = nil
        }
        let textSER: any TextSER = SwitchingTextSER(
            deberta: deberta,
            foundationModels: FoundationModelsSER()
        )

        AppLog.app.info(
            "AnalysisPipeline ready: dimensional=\(dimensional != nil, privacy: .public), categorical=\(categorical != nil, privacy: .public), deberta=\(deberta != nil, privacy: .public), diarizer=\(diarizer != nil, privacy: .public)"
        )

        let pipeline = AnalysisPipeline(
            diarizer: diarizer,
            dimensionalSER: dimensional,
            categoricalSER: categorical,
            textSER: textSER
        )
        return AutoConfiguredPipeline(pipeline: pipeline, diagnostics: diagnostics)
    }

    private static func tryInit<T>(
        _ name: String,
        diagnostics: inout [String],
        _ build: () throws -> T
    ) -> T? {
        do { return try build() }
        catch {
            let detail = "\(name) unavailable: \(error.localizedDescription)"
            AppLog.app.warning("\(name, privacy: .public) unavailable: \(String(describing: error), privacy: .public)")
            diagnostics.append(detail)
            return nil
        }
    }

    private static func tryInitAsync<T>(
        _ name: String,
        diagnostics: inout [String],
        _ build: () async throws -> T
    ) async -> T? {
        do { return try await build() }
        catch {
            let detail = "\(name) unavailable: \(error.localizedDescription)"
            AppLog.app.warning("\(name, privacy: .public) unavailable: \(String(describing: error), privacy: .public)")
            diagnostics.append(detail)
            return nil
        }
    }

    /// `ModelStore.resolvedURL(for:)` throws when a path wasn't resolved
    /// — typically because the user hasn't completed the download yet,
    /// or a file legitimately failed (placeholder hash, network).
    /// Convert to an optional so the pipeline degrades gracefully.
    private static func tryResolveAsync(
        _ name: String,
        path: String,
        store: ModelStore,
        diagnostics: inout [String]
    ) async -> URL? {
        do { return try await store.resolvedURL(for: path) }
        catch {
            let detail = "\(name) URL not resolved: \(error.localizedDescription)"
            AppLog.app.warning("\(name, privacy: .public) URL not resolved: \(String(describing: error), privacy: .public)")
            diagnostics.append(detail)
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

        // 3. Per-segment SER + fuse. We've already diarized the whole
        // buffer above, so resolve the speaker here and pass it as the
        // fallback — `processSegment` will find an empty diarization
        // context (we don't pass one in batch mode) and use the fallback
        // verbatim.
        var estimates: [UtteranceEstimate] = []
        for segment in asrSegments {
            let speakerID = speakerForSegment(segment, in: diarized)
            let segmentBuffer = sliceBuffer(buffer, start: segment.start, end: segment.end)
            let (estimate, _) = try await processSegment(
                asr: segment,
                segmentAudio: segmentBuffer,
                fallbackSpeakerID: speakerID
            )
            estimates.append(estimate)
        }
        return estimates
    }

    /// Per-segment SER + fusion. Used in both batch (`analyze(_:)`) and
    /// streaming flows. Caller is responsible for slicing the audio segment
    /// out of the source buffer.
    ///
    /// `diarizationContext` is the recent cumulative audio buffer (with its
    /// origin timestamp on `AudioChunk.timestamp`) that the streaming caller
    /// passes in for cross-segment speaker identification. When provided
    /// alongside an active diarizer, this method runs diarization on the
    /// context concurrently with SER and looks up the speaker ID for the
    /// segment's midpoint via `speakerForSegment(_:in:)`. When omitted or
    /// unavailable, the result falls back to `fallbackSpeakerID`.
    func processSegment(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        diarizationContext: AudioChunk? = nil,
        fallbackSpeakerID: String = "S01"
    ) async throws -> (UtteranceEstimate, ProcessingMetrics) {
        let totalStart = Date()
        let acousticStart = Date()
        async let dimensional = runDimensional(segmentAudio)
        async let categorical = runCategorical(segmentAudio)
        async let plutchik = runText(asr.text)
        async let diarized: [DiarizedSegment] = {
            guard let diarizationContext, diarizer != nil else { return [] }
            return await self.runDiarization(diarizationContext)
        }()

        // Acoustic timing = wall time until both dimensional + categorical
        // resolve (they're concurrent). Text timing = wall time until plutchik
        // resolves. The two overlap in real time but are reported separately.
        let textStart = Date()
        let txt = await plutchik
        let textDuration = Date().timeIntervalSince(textStart)

        let dim = await dimensional
        let cat = await categorical
        let acousticDuration = Date().timeIntervalSince(acousticStart)

        // Diarization runs in parallel with SER; this await just collects
        // whatever's ready. The local speaker IDs FluidAudio returns
        // ("speaker_0", …) aren't comparable across calls, so we route
        // them through `speakerTracker.ingest` which remaps them to
        // session-stable global IDs (`S01`, `S02`, …) by time-overlap
        // matching against the cumulative history. Empty result →
        // fallback to the supplied id.
        let diarizedSegments = await diarized
        let trackedSegments = diarizedSegments.isEmpty
            ? []
            : await speakerTracker.ingest(diarizedSegments)
        let speakerID = trackedSegments.isEmpty
            ? fallbackSpeakerID
            : speakerForSegment(asr, in: trackedSegments)

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
        let capped = Self.capForSER(buffer)
        do { return try await dimensionalSER.score(capped) } catch {
            AppLog.app.debug("dimensional SER skipped: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    private func runCategorical(_ buffer: AudioChunk) async -> CategoricalEmotion? {
        guard let categoricalSER else { return nil }
        let capped = Self.capForSER(buffer)
        do { return try await categoricalSER.score(capped) } catch {
            AppLog.app.debug("categorical SER skipped: \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    /// Snap audio length to one of a small set of bins before feeding the
    /// acoustic SER models. Five reasons:
    ///   1. audeering's W2V2 dimensional model and emotion2vec+ are trained on
    ///      ≤10 s clips; longer inputs degrade accuracy without helping.
    ///   2. ONNX Runtime's CoreML EP compiles a per-input-shape MLModel for
    ///      the dynamic-time-axis W2V2/emotion2vec graphs and caches the
    ///      result. Without binning, a long session with varied utterance
    ///      lengths grows the cache monotonically — the dominant cause of
    ///      Jetsam-style OOM kills around the 15 min mark of fast-pace
    ///      file analysis. Three bins → at most three cached MLModels.
    ///   3. At long unique shapes the EP has occasionally crashed the ANE
    ///      compiler with EXC_BAD_ACCESS — pinning shape avoids that path.
    ///   4. Inference latency scales linearly with audio length, so binning
    ///      to the smallest fitting bin caps it.
    ///   5. Binning lets us keep accuracy reasonable for short utterances
    ///      (a 1.2 s utterance gets the 2 s bin, padded only ~0.8 s, vs.
    ///      a fixed-8s shape that would dilute the 1.2 s of speech with
    ///      6.8 s of silence).
    /// Center-crop when over the bin; post-pad with zeros when under.
    private static let serBinSeconds: [TimeInterval] = [2.0, 4.0, 8.0]
    private static func capForSER(_ buffer: AudioChunk) -> AudioChunk {
        let durSec = Double(buffer.samples.count) / buffer.sampleRate
        let binSec = Self.serBinSeconds.first(where: { $0 >= durSec }) ?? Self.serBinSeconds.last!
        let target = Int(binSec * buffer.sampleRate)
        if buffer.samples.count == target { return buffer }
        if buffer.samples.count > target {
            let extra = buffer.samples.count - target
            let startIndex = extra / 2
            let slice = Array(buffer.samples[startIndex..<(startIndex + target)])
            return AudioChunk(
                samples: slice,
                sampleRate: buffer.sampleRate,
                timestamp: buffer.timestamp + Double(startIndex) / buffer.sampleRate
            )
        }
        // Repeat-pad rather than zero-pad. The acoustic models are
        // mean-pool classifiers, and silence has a non-trivial bias on
        // both: W2V2 predicts roughly `A=0.61, V=0.39` on silence (the
        // "mild middle"), and emotion2vec predicts `sad≈99%`. Padding a
        // 3 s utterance to a 4 s bin with zeros pulls the mean ~25%
        // toward those silence baselines — enough to flip a borderline
        // categorical label and visibly skew V/A/D. Looping the original
        // samples instead keeps the bin filled with the utterance's own
        // acoustic character, so the mean-pool window represents the
        // speech rather than a speech/silence blend. The micro-clicks
        // at the loop seams are spectrally negligible compared to the
        // 25% silence-mean shift the prior approach introduced.
        guard !buffer.samples.isEmpty else { return buffer }
        var padded = buffer.samples
        padded.reserveCapacity(target)
        while padded.count < target {
            let needed = target - padded.count
            let chunk = min(needed, buffer.samples.count)
            padded.append(contentsOf: buffer.samples.prefix(chunk))
        }
        return AudioChunk(
            samples: padded,
            sampleRate: buffer.sampleRate,
            timestamp: buffer.timestamp
        )
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

    // MARK: - Speaker-change splitting

    /// One sub-segment produced by `splitOnSpeakerChange`. Each carries
    /// its own ASR slice, the audio for that slice, and a pre-resolved
    /// speaker so the caller can pass it as `fallbackSpeakerID` and skip
    /// re-running the diarizer inside `processSegment`.
    struct SegmentSplit: Sendable {
        let asr: ASRSegment
        let audio: AudioChunk
        let speaker: String
    }

    /// Diarize the segment's context once and emit one sub-segment per
    /// contiguous run of same-speaker tokens. When the diarizer reports
    /// only one speaker (or isn't available, or `asr.tokens` is empty),
    /// returns a single-element array equivalent to the original
    /// segment so the caller's loop is uniform either way.
    ///
    /// Splitting happens at *token* boundaries — never mid-token —
    /// because SpeechAnalyzer's per-run timing is the finest grain we
    /// have for the text-side cut. Diarizer accuracy on short windows
    /// is ±200–500 ms, so a hard mid-segment cut at the diarizer's
    /// reported instant could land inside a word; snapping to the
    /// nearest token boundary keeps each sub-segment's text grammatical.
    func splitOnSpeakerChange(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        diarizationContext: AudioChunk?,
        fallbackSpeakerID: String = "S01"
    ) async -> [SegmentSplit] {
        let single = SegmentSplit(asr: asr, audio: segmentAudio, speaker: fallbackSpeakerID)

        // Need both per-token timing AND a diarizer to do the split.
        // Either missing → return the original segment unchanged.
        guard let diarizationContext, diarizer != nil, !asr.tokens.isEmpty else {
            return [single]
        }

        let diarized = await runDiarization(diarizationContext)
        guard !diarized.isEmpty else { return [single] }
        let tracked = await speakerTracker.ingest(diarized)
        guard !tracked.isEmpty else { return [single] }

        // Per-token speaker assignment by midpoint. Falls back to the
        // closest segment when the midpoint falls in a diarizer gap.
        func speakerForToken(_ token: ASRSegment.Token) -> String {
            let mid = (token.start + token.end) / 2
            return Self.speakerAt(midpoint: mid, in: tracked, fallback: fallbackSpeakerID)
        }

        // Single-speaker fast path. If every token resolves to the same
        // speaker, no need to allocate a list of splits — return the
        // original segment with that speaker baked in.
        let assignments = asr.tokens.map(speakerForToken)
        let distinct = Set(assignments)
        if distinct.count <= 1 {
            return [SegmentSplit(
                asr: asr,
                audio: segmentAudio,
                speaker: distinct.first ?? fallbackSpeakerID
            )]
        }

        // Group consecutive tokens by speaker; emit one sub-segment per
        // group. The order of `assignments` matches the order of
        // `asr.tokens`, so a forward scan with a running speaker is
        // enough — no sort required.
        var splits: [SegmentSplit] = []
        var groupStart = 0
        var current = assignments[0]
        for i in 1..<assignments.count {
            if assignments[i] != current {
                splits.append(Self.buildSplit(
                    tokens: Array(asr.tokens[groupStart..<i]),
                    speaker: current,
                    parent: asr,
                    segmentAudio: segmentAudio
                ))
                current = assignments[i]
                groupStart = i
            }
        }
        splits.append(Self.buildSplit(
            tokens: Array(asr.tokens[groupStart..<asr.tokens.count]),
            speaker: current,
            parent: asr,
            segmentAudio: segmentAudio
        ))
        return splits
    }

    private static func buildSplit(
        tokens: [ASRSegment.Token],
        speaker: String,
        parent: ASRSegment,
        segmentAudio: AudioChunk
    ) -> SegmentSplit {
        let text = tokens.map(\.text).joined()
        let start = tokens.first?.start ?? parent.start
        let end = tokens.last?.end ?? parent.end
        let subAsr = ASRSegment(
            text: text,
            start: start,
            end: end,
            // Sub-segments inherit the parent's whole-segment confidence;
            // SpeechAnalyzer doesn't expose per-run confidence we'd need
            // to recompute a per-sub mean against.
            confidence: parent.confidence,
            tokens: tokens
        )
        let subAudio = sliceRelative(segmentAudio, fromAudioTime: start, toAudioTime: end)
        return SegmentSplit(asr: subAsr, audio: subAudio, speaker: speaker)
    }

    // MARK: - Buffer slicing

    private func speakerForSegment(_ asr: ASRSegment, in diarized: [DiarizedSegment]) -> String {
        let mid = (asr.start + asr.end) / 2
        return Self.speakerAt(midpoint: mid, in: diarized, fallback: "S01")
    }

    /// Pick the diarized speaker covering `mid`, falling back to the
    /// closest segment by midpoint when `mid` lands in a diarizer gap.
    /// `fallback` is used only when `tracked` is empty.
    private static func speakerAt(
        midpoint mid: TimeInterval,
        in tracked: [DiarizedSegment],
        fallback: String
    ) -> String {
        guard !tracked.isEmpty else { return fallback }
        if let containing = tracked.first(where: { $0.start <= mid && mid <= $0.end }) {
            return containing.speakerID
        }
        let closest = tracked.min(by: { abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid) })
        return closest?.speakerID ?? fallback
    }

    private func sliceBuffer(_ buffer: AudioChunk, start: TimeInterval, end: TimeInterval) -> AudioChunk {
        Self.slice(buffer, start: start, end: end)
    }

    /// Slice a captured-audio buffer to [start, end] seconds. Public for
    /// streaming callers that already hold the cumulative samples.
    /// Uses absolute audio-time when the buffer's `timestamp` is 0
    /// (i.e. it was captured from session start) — otherwise prefer
    /// `sliceRelative` which respects the buffer's timeline origin.
    static func slice(_ buffer: AudioChunk, start: TimeInterval, end: TimeInterval) -> AudioChunk {
        let total = Double(buffer.samples.count)
        let startIndex = max(0, min(Int(start * buffer.sampleRate), buffer.samples.count))
        let endIndex = max(startIndex, min(Int(end * buffer.sampleRate), buffer.samples.count))
        guard startIndex < endIndex, total > 0 else { return buffer }
        let slice = Array(buffer.samples[startIndex..<endIndex])
        return AudioChunk(samples: slice, sampleRate: buffer.sampleRate, timestamp: start)
    }

    /// Slice an audio chunk by audio time, respecting the chunk's
    /// `timestamp` origin. Used by `splitOnSpeakerChange` where
    /// `segmentAudio` covers `[asr.start, asr.end]` and we need to
    /// extract a sub-range `[subStart, subEnd]` relative to that.
    private static func sliceRelative(
        _ buffer: AudioChunk,
        fromAudioTime start: TimeInterval,
        toAudioTime end: TimeInterval
    ) -> AudioChunk {
        let relStart = max(0, start - buffer.timestamp)
        let relEnd = max(relStart, end - buffer.timestamp)
        let startIndex = min(Int(relStart * buffer.sampleRate), buffer.samples.count)
        let endIndex = min(Int(relEnd * buffer.sampleRate), buffer.samples.count)
        guard startIndex < endIndex else {
            return AudioChunk(samples: [], sampleRate: buffer.sampleRate, timestamp: start)
        }
        let slice = Array(buffer.samples[startIndex..<endIndex])
        return AudioChunk(samples: slice, sampleRate: buffer.sampleRate, timestamp: start)
    }
}
