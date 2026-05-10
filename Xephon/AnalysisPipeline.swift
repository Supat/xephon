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

    /// Run the diarizer on a window of audio and merge the result
    /// into the cumulative speaker timeline via `speakerTracker`.
    /// Called by the continuous-diarization side task in
    /// RecordingController on every stride tick. No-op when the
    /// diarizer is unavailable or returns empty for this window.
    /// The audio's `timestamp` is the absolute audio-time origin of
    /// the window, so the tracker's history stays in the same time
    /// frame as ASRSegment / token timings.
    func ingestDiarizationWindow(_ audio: AudioChunk) async {
        guard diarizer != nil, !audio.samples.isEmpty else { return }
        let diarized = await runDiarization(audio)
        guard !diarized.isEmpty else { return }
        _ = await speakerTracker.ingest(diarized)
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

        // Prefer the streaming timeline (continuously refreshed by the
        // sliding-window task) over a one-shot per-segment diarize.
        // The timeline tracks shorter windows (~10 s) than the
        // per-segment context (~60 s), so its boundary timings are
        // sharper. Fall back to per-segment diarize only when the
        // timeline hasn't been populated yet (e.g. the first segment
        // arrives before the side task has fired).
        let tracked: [DiarizedSegment]
        if await speakerTracker.isPopulated {
            tracked = await speakerTracker.cumulativeSnapshot()
        } else {
            let diarized = await runDiarization(diarizationContext)
            guard !diarized.isEmpty else { return [single] }
            let mapped = await speakerTracker.ingest(diarized)
            guard !mapped.isEmpty else { return [single] }
            tracked = mapped
        }

        // Per-token speaker assignment by midpoint. Falls back to the
        // closest segment when the midpoint falls in a diarizer gap.
        func speakerForToken(_ token: ASRSegment.Token) -> String {
            let mid = (token.start + token.end) / 2
            return Self.speakerAt(midpoint: mid, in: tracked, fallback: fallbackSpeakerID)
        }

        // Single-speaker fast path. If every token resolves to the same
        // speaker, no need to allocate a list of splits — return the
        // original segment with that speaker baked in.
        let rawAssignments = asr.tokens.map(speakerForToken)
        let distinct = Set(rawAssignments)
        if distinct.count <= 1 {
            return [SegmentSplit(
                asr: asr,
                audio: segmentAudio,
                speaker: distinct.first ?? fallbackSpeakerID
            )]
        }

        // VAD-snap: nudge each detected boundary to the nearest local
        // energy minimum within ±2 tokens. The diarizer has ~200–500 ms
        // boundary uncertainty (≈ 1 JP word), so the diarizer-reported
        // change often lands inside a word rather than at the natural
        // silence between speakers. Snapping at silence keeps each
        // sub-segment's text aligned with its actual speaker turn.
        let snapped = Self.snapBoundariesToSilence(
            assignments: rawAssignments,
            tokens: asr.tokens,
            in: segmentAudio
        )

        // Boundary re-diarize: VAD-snap can't help in fast-pace
        // conversation, where speakers cut into each other with no
        // clean pause to snap to. Re-running the diarizer on a tight
        // ±1.5 s window around each remaining boundary gives a
        // sharper local opinion than the 60 s-context call did, and
        // we snap the boundary to the token gap closest to the local
        // change time. No-op when the diarizer can't confirm a
        // change in the local window (overlapping speech, etc.).
        let assignments = await refineBoundariesByReDiarize(
            assignments: snapped,
            tokens: asr.tokens,
            diarizationContext: diarizationContext
        )

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

    // MARK: - VAD-snap

    /// Half-window for the gap-energy probe. 50 ms is wide enough to
    /// average out single-frame outliers (a 16-frame, 1-ms click) but
    /// narrow enough to read a real pause distinct from voiced audio
    /// on either side.
    private static let snapHalfWindowSec: TimeInterval = 0.05
    /// Snap radius in tokens. ±2 covers the "off by a word or two"
    /// failure mode without giving the snap-to-silence heuristic enough
    /// rope to override the diarizer entirely on rapid back-and-forth.
    private static let snapTokenRadius: Int = 2
    /// Snap only when the alternative gap's mean-square energy is
    /// ≤ this fraction of the original. 0.5 = "twice as quiet" — strict
    /// enough that boundaries through voiced audio with marginally
    /// quieter neighbors don't drift, lenient enough that real pauses
    /// pull the boundary in.
    private static let snapEnergyRatio: Float = 0.5

    /// Pull each detected boundary toward the nearest within-radius
    /// gap whose energy is meaningfully lower than the original
    /// boundary's gap. The diarizer assigns one speaker per token by
    /// midpoint; near a boundary, the prediction can drift by ±1–2
    /// tokens because the model's own boundary timing is uncertain.
    /// Snapping at silence captures the actual turn-take location.
    ///
    /// Snap candidates are clamped to the surrounding same-speaker
    /// runs so we never collapse a run to zero tokens — at worst the
    /// boundary moves to the very start (one previous token retained)
    /// or very end (one next token retained) of the snap window.
    private static func snapBoundariesToSilence(
        assignments: [String],
        tokens: [ASRSegment.Token],
        in audio: AudioChunk
    ) -> [String] {
        guard tokens.count >= 2, !audio.samples.isEmpty else { return assignments }
        var result = assignments
        var i = 1
        while i < tokens.count {
            guard result[i - 1] != result[i] else {
                i += 1
                continue
            }
            // Run bounds. The previous run is `tokens[runStart ..< i]`
            // (all `result[i-1]`); the next run starts at `i` and ends
            // at `nextRunEnd` inclusive.
            var runStart = i - 1
            while runStart > 0 && result[runStart - 1] == result[i - 1] {
                runStart -= 1
            }
            var nextRunEnd = i
            while nextRunEnd < tokens.count - 1 && result[nextRunEnd + 1] == result[i] {
                nextRunEnd += 1
            }

            // Candidate boundary positions. `lo = runStart + 1` keeps
            // ≥ 1 previous-speaker token; `hi = nextRunEnd` keeps ≥ 1
            // next-speaker token (the boundary at position `nextRunEnd
            // + 1` would be the *next* boundary in the sequence).
            let lo = max(runStart + 1, i - snapTokenRadius)
            let hi = min(nextRunEnd, i + snapTokenRadius)
            guard lo <= hi else {
                i = nextRunEnd + 1
                continue
            }

            let originalEnergy = gapEnergyAt(position: i, tokens: tokens, audio: audio)
            var bestPos = i
            var bestEnergy = originalEnergy
            for j in lo...hi where j != i {
                let e = gapEnergyAt(position: j, tokens: tokens, audio: audio)
                if e < bestEnergy {
                    bestEnergy = e
                    bestPos = j
                }
            }

            if bestPos != i && bestEnergy <= originalEnergy * snapEnergyRatio {
                let oldSpeaker = result[i - 1]
                let newSpeaker = result[i]
                if bestPos < i {
                    // Boundary moves earlier: tokens [bestPos..i-1]
                    // were the old speaker per the diarizer but the
                    // silence says they belong with the new run.
                    for k in bestPos..<i { result[k] = newSpeaker }
                } else {
                    // Boundary moves later: tokens [i..bestPos-1] flip
                    // back to the old speaker.
                    for k in i..<bestPos { result[k] = oldSpeaker }
                }
                // Skip past the snapped boundary to avoid reprocessing
                // the tokens we just rewrote.
                i = bestPos + 1
            } else {
                // No good snap target; advance past this whole next run
                // so we don't re-evaluate the same boundary.
                i = nextRunEnd + 1
            }
        }
        return result
    }

    /// Mean-square energy in a `±snapHalfWindowSec` window around the
    /// gap before `tokens[position]`. `position == 0` probes audio
    /// start; `position == tokens.count` probes audio end. Indexing
    /// is buffer-relative — respects `audio.timestamp` so callers pass
    /// absolute audio-time positions without pre-shifting.
    private static func gapEnergyAt(
        position: Int,
        tokens: [ASRSegment.Token],
        audio: AudioChunk
    ) -> Float {
        let gapTime: TimeInterval
        if position <= 0 {
            gapTime = tokens.first?.start ?? audio.timestamp
        } else if position >= tokens.count {
            gapTime = tokens.last?.end ?? audio.timestamp
        } else {
            gapTime = (tokens[position - 1].end + tokens[position].start) / 2
        }
        let sr = audio.sampleRate
        let centerFrame = Int((gapTime - audio.timestamp) * sr)
        let halfFrames = Int(snapHalfWindowSec * sr)
        let windowLo = max(0, centerFrame - halfFrames)
        let windowHi = min(audio.samples.count, centerFrame + halfFrames)
        guard windowLo < windowHi else { return 0 }
        var sumSq: Float = 0
        for k in windowLo..<windowHi {
            let s = audio.samples[k]
            sumSq += s * s
        }
        return sumSq / Float(windowHi - windowLo)
    }

    // MARK: - Boundary re-diarize

    /// Half-window for the boundary re-diarize. The diarizer needs
    /// enough acoustic context to extract reliable speaker
    /// embeddings — Sortformer is trained on multi-second windows —
    /// so a sub-second probe would just produce noise. 1.5 s on
    /// each side gives a 3 s window: wide enough for the model to
    /// score speaker membership confidently, narrow enough that the
    /// boundary timing is sharper than the 60 s-context call's.
    private static let boundaryReDiarizeHalfWindowSec: TimeInterval = 1.5

    /// For each detected boundary in `assignments`, re-run the
    /// diarizer on a tight window centered on the boundary and snap
    /// the boundary to the token gap closest to the local
    /// speaker-change time. Bounded by the surrounding same-speaker
    /// runs so we never collapse a run to zero tokens — the
    /// re-diarize can disagree with the main pass about *where* the
    /// boundary is but not *whether* there is one.
    ///
    /// Returns `assignments` unchanged when the diarizer is missing
    /// or `diarizationContext` is nil. Per-boundary calls are
    /// sequential because the underlying diarizer is actor-isolated
    /// — concurrent calls would just queue at the actor anyway.
    private func refineBoundariesByReDiarize(
        assignments: [String],
        tokens: [ASRSegment.Token],
        diarizationContext: AudioChunk?
    ) async -> [String] {
        guard tokens.count >= 2,
              let diarizationContext,
              diarizer != nil
        else { return assignments }

        var result = assignments
        var i = 1
        while i < tokens.count {
            guard result[i - 1] != result[i] else {
                i += 1
                continue
            }

            // Run bounds — same logic as VAD-snap. We never let the
            // boundary cross a same-speaker run boundary, so each
            // run stays non-empty post-snap.
            var runStart = i - 1
            while runStart > 0 && result[runStart - 1] == result[i - 1] {
                runStart -= 1
            }
            var nextRunEnd = i
            while nextRunEnd < tokens.count - 1 && result[nextRunEnd + 1] == result[i] {
                nextRunEnd += 1
            }

            let originalGapTime = (tokens[i - 1].end + tokens[i].start) / 2
            guard let localChangeTime = await localBoundaryTime(
                around: originalGapTime,
                in: diarizationContext
            ) else {
                i = nextRunEnd + 1
                continue
            }

            let lo = max(runStart + 1, i - Self.snapTokenRadius)
            let hi = min(nextRunEnd, i + Self.snapTokenRadius)
            guard lo <= hi else {
                i = nextRunEnd + 1
                continue
            }

            // Pick the token gap whose midpoint is closest to the
            // local re-diarize change time.
            var bestPos = i
            var bestDist = abs(originalGapTime - localChangeTime)
            for j in lo...hi where j != i {
                let gapTime = (tokens[j - 1].end + tokens[j].start) / 2
                let dist = abs(gapTime - localChangeTime)
                if dist < bestDist {
                    bestDist = dist
                    bestPos = j
                }
            }

            if bestPos != i {
                let oldSpeaker = result[i - 1]
                let newSpeaker = result[i]
                if bestPos < i {
                    for k in bestPos..<i { result[k] = newSpeaker }
                } else {
                    for k in i..<bestPos { result[k] = oldSpeaker }
                }
                i = bestPos + 1
            } else {
                i = nextRunEnd + 1
            }
        }
        return result
    }

    /// Local speaker-change time inside a `±boundaryReDiarizeHalfWindowSec`
    /// window around `center`. Slices `context` to the window and runs
    /// the diarizer on it; returns the start time of the first
    /// transition between adjacent local-ID segments. Local speaker
    /// IDs aren't comparable across calls, but within this single
    /// call they're stable, which is all we need to pinpoint a
    /// change time. Returns nil when the local diarizer reports a
    /// single speaker (overlapping speech, marginal window) or the
    /// window can't be sliced (boundary off the context's timeline).
    private func localBoundaryTime(
        around center: TimeInterval,
        in context: AudioChunk
    ) async -> TimeInterval? {
        let halfWindow = Self.boundaryReDiarizeHalfWindowSec
        let lo = max(context.timestamp, center - halfWindow)
        let contextEnd = context.timestamp + Double(context.samples.count) / context.sampleRate
        let hi = min(contextEnd, center + halfWindow)
        guard lo < hi else { return nil }

        let chunk = Self.sliceRelative(context, fromAudioTime: lo, toAudioTime: hi)
        guard !chunk.samples.isEmpty else { return nil }

        let local = await runDiarization(chunk)
        guard local.count >= 2 else { return nil }

        let sorted = local.sorted { $0.start < $1.start }
        for k in 1..<sorted.count where sorted[k].speakerID != sorted[k - 1].speakerID {
            return sorted[k].start
        }
        return nil
    }

    // MARK: - Buffer slicing

    private func speakerForSegment(_ asr: ASRSegment, in diarized: [DiarizedSegment]) -> String {
        let mid = (asr.start + asr.end) / 2
        return Self.speakerAt(midpoint: mid, in: diarized, fallback: "S01")
    }

    /// Pick the diarized speaker covering `mid` by majority vote
    /// across every segment in `tracked` that contains `mid`. With a
    /// single diarizer call's output (e.g. inside `processSegment`'s
    /// in-pipeline diarization path), each audio time is covered by
    /// at most one segment, so voting reduces to "pick the
    /// containing segment". With the streaming timeline's
    /// multi-observation snapshot (~5 overlapping windows per
    /// moment), voting picks the speaker the majority of recent
    /// diarize calls agreed on — robust to single-call
    /// misclassifications. Falls back to the closest segment by
    /// midpoint distance when no segment contains `mid`; `fallback`
    /// applies only when `tracked` is entirely empty.
    private static func speakerAt(
        midpoint mid: TimeInterval,
        in tracked: [DiarizedSegment],
        fallback: String
    ) -> String {
        guard !tracked.isEmpty else { return fallback }
        var votes: [String: Int] = [:]
        for s in tracked where s.start <= mid && mid <= s.end {
            votes[s.speakerID, default: 0] += 1
        }
        if let best = votes.max(by: { $0.value < $1.value }) {
            return best.key
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
