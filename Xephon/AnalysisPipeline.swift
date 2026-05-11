import Foundation
import NaturalLanguage
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
        // FluidAudio's SpeakerManager has its own embedding-based
        // database that's independent of `speakerTracker.cumulative`.
        // Without this reset, an earlier session's centroid for "S01"
        // would attract embeddings from the new session's first
        // speaker via cosine similarity, conflating two different
        // people under the same ID.
        await diarizer?.resetSpeakers()
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

    /// Re-run offline ASR on `audio` and fuse the result with fresh SER
    /// estimates. Used by per-utterance "re-evaluate" — the caller has
    /// already pulled the relevant slice from the source file (padded
    /// front and back so ASR has prosodic context the streaming pass
    /// didn't see at the segment boundary). The original utterance's
    /// time range and speaker are preserved on the result; only
    /// transcript / SER / fusion outputs come from the new run.
    ///
    /// Returns nil when offline ASR finds no usable transcript in the
    /// padded audio (silent gap, drowned by noise) — callers should
    /// leave the original utterance in place rather than collapsing it
    /// to empty text. Multiple sub-segments from offline ASR are
    /// concatenated into one synthesized ASRSegment whose start/end
    /// match the caller's preserved range; per-token timing is dropped
    /// because we don't have a consistent timeline to slot it into.
    func reevaluate(
        audio: AudioChunk,
        originalStart: TimeInterval,
        originalEnd: TimeInterval,
        speakerID: String,
        onVolatileText: (@Sendable @MainActor (String) -> Void)? = nil
    ) async throws -> (UtteranceEstimate, ProcessingMetrics)? {
        // Forward the volatile-text callback only when the configured
        // transcriber is the Apple SpeechAnalyzer variant — it's the
        // only one we know exposes rolling hypotheses. WhisperKit /
        // Qwen3 fall back to final-only output. The Transcriber
        // protocol stays unchanged.
        let segments: [ASRSegment]
        if let speech = transcriber as? SpeechAnalyzerTranscriber, onVolatileText != nil {
            segments = try await speech.transcribe(audio, onVolatileText: onVolatileText)
        } else {
            segments = try await transcriber.transcribe(audio)
        }
        guard !segments.isEmpty else { return nil }
        let combinedText = segments.map(\.text).joined()
        guard !combinedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        let confidences = segments.compactMap(\.confidence)
        let combinedConfidence: Float? = confidences.isEmpty
            ? nil
            : confidences.reduce(0, +) / Float(confidences.count)
        let synthesized = ASRSegment(
            text: combinedText,
            start: originalStart,
            end: originalEnd,
            confidence: combinedConfidence,
            tokens: []
        )
        return try await processSegment(
            asr: synthesized,
            segmentAudio: audio,
            fallbackSpeakerID: speakerID
        )
    }

    /// Per-segment SER + fusion. Caller is responsible for slicing the
    /// audio segment out of the source buffer and for resolving the
    /// speaker ID upstream — this method just stamps `fallbackSpeakerID`
    /// onto the result. The streaming pipeline pre-resolves the speaker
    /// in `splitForProcessing` (via the cumulative diarizer timeline)
    /// and passes it here as `fallbackSpeakerID`.
    func processSegment(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        fallbackSpeakerID: String = "S01"
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
            speakerID: fallbackSpeakerID,
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

    // MARK: - Sentence splitting (punctuation + long-pause fallback)

    /// Minimum inter-token gap (seconds) that forces a sub-segment
    /// split even when no punctuation is involved. 700 ms is well
    /// above breath pauses and below most mid-sentence hesitations
    /// — long enough that "this is a new sentence" is unambiguous
    /// even without Apple emitting `。`. Apple's SpeechTranscriber
    /// usually punctuates between sentences, but it sometimes
    /// omits the mark on incomplete/elliptical speech; this catches
    /// those.
    private static let sentenceSplitMinPauseSec: TimeInterval = 0.7
    /// Sentence-ending characters in JP and EN. SpeechTranscriber
    /// emits punctuation as a tail on the preceding token, so a
    /// `text.last`-only check catches them.
    private static let sentenceEndChars: Set<Character> = [
        "。", "！", "？", "．", ".", "!", "?",
    ]

    /// Split an `ASRSegment` so each sub-segment contains at most
    /// one sentence. Splits after every token whose `text.last` is
    /// in `sentenceEndChars` (the primary rule — Apple's JP
    /// recognizer punctuates sentence boundaries reliably), and
    /// also after any inter-token gap ≥ `sentenceSplitMinPauseSec`
    /// (the fallback for un-punctuated long pauses).
    ///
    /// Returns the segment unchanged when no boundary qualifies,
    /// when the segment has fewer than two tokens, or when the
    /// transcriber didn't supply per-token timing. Independent of
    /// speaker detection — speaker assignment in `splitForProcessing`
    /// runs after this and operates on per-sentence inputs.
    static func splitIntoSentences(_ asr: ASRSegment) -> [ASRSegment] {
        guard asr.tokens.count >= 2 else { return [asr] }

        var splitAfter: [Int] = []
        for i in 0..<(asr.tokens.count - 1) {
            if let lastChar = asr.tokens[i].text.last,
               sentenceEndChars.contains(lastChar) {
                splitAfter.append(i)
                continue
            }
            let gap = asr.tokens[i + 1].start - asr.tokens[i].end
            if gap >= sentenceSplitMinPauseSec {
                splitAfter.append(i)
            }
        }

        guard !splitAfter.isEmpty else { return [asr] }

        var result: [ASRSegment] = []
        var start = 0
        let endpoints = splitAfter + [asr.tokens.count - 1]
        for end in endpoints {
            let subTokens = Array(asr.tokens[start...end])
            let subText = subTokens.map(\.text).joined()
            let subStart = subTokens.first?.start ?? asr.start
            let subEnd = subTokens.last?.end ?? asr.end
            result.append(ASRSegment(
                text: subText,
                start: subStart,
                end: subEnd,
                // Sub-segments inherit the parent's whole-segment
                // confidence — we don't compute a per-sub mean
                // because it'd require averaging per-token confidence
                // that SpeechAttributes already discards in
                // `averageConfidence`.
                confidence: asr.confidence,
                tokens: subTokens
            ))
            start = end + 1
        }
        return result
    }

    // MARK: - Sentence splitting

    /// One sub-segment produced by `splitForProcessing`. Each carries
    /// its own ASR slice, the audio for that slice, and a pre-resolved
    /// speaker so the caller passes it as `fallbackSpeakerID` to
    /// `processSegment`.
    struct SegmentSplit: Sendable {
        let asr: ASRSegment
        let audio: AudioChunk
        let speaker: String
    }

    /// Sentence-level split with one dominant speaker per sentence.
    ///
    /// Runs `splitIntoSentences` and emits one `SegmentSplit` per
    /// resulting sub-segment, tagged with whichever speaker the
    /// cumulative diarizer timeline reports as dominant inside
    /// `[sentence.start, sentence.end]`. A sentence is never
    /// sub-divided by mid-sentence speaker changes — sentence
    /// integrity wins over speaker boundary precision. When the
    /// diarizer occasionally mis-classifies a few tokens inside a
    /// sentence, the per-instant majority vote in `dominantSpeaker`
    /// absorbs that noise instead of cutting the sentence into
    /// fragments. Speaker resolution reads `speakerTracker` directly;
    /// the continuous diarize task in RecordingController is what
    /// keeps that timeline fresh.
    func splitForProcessing(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        fallbackSpeakerID: String = "S01"
    ) async -> [SegmentSplit] {
        let sentenceSplits = Self.splitIntoSentences(asr)
        var result: [SegmentSplit] = []
        for sentence in sentenceSplits {
            let sentenceAudio: AudioChunk = sentenceSplits.count == 1
                ? segmentAudio
                : Self.sliceRelative(
                    segmentAudio,
                    fromAudioTime: sentence.start,
                    toAudioTime: sentence.end
                )
            let speaker = await dominantSpeaker(
                from: sentence.start,
                to: sentence.end,
                fallback: fallbackSpeakerID
            )
            result.append(SegmentSplit(
                asr: sentence,
                audio: sentenceAudio,
                speaker: speaker
            ))
        }
        return result
    }

    /// Speaker the cumulative diarizer timeline reports as dominant
    /// in `[start, end]`. Two-stage vote:
    ///
    /// 1. **Per-instant majority.** At each sample point, tally
    ///    overlapping observations and pick the per-instant winner.
    ///    The continuous diarize task accumulates ~5 overlapping
    ///    observations per audio moment, so a single noisy window
    ///    can't outvote the consensus at that instant.
    /// 2. **Mode across instants.** Aggregate per-instant winners
    ///    across the sentence range and return the most common one.
    ///    This is duration-weighted (every sample point contributes
    ///    one vote) but immune to the duration × observation-count
    ///    skew of a flat overlap-sum.
    ///
    /// Sample step is 50 ms — well below the diarizer's segment
    /// granularity (~200–500 ms) — and capped so a long sentence
    /// doesn't blow up the inner loop. When the timeline is empty
    /// or no entry overlaps any sample, falls back to the timeline
    /// entry whose midpoint is closest to the sentence's midpoint
    /// (matching `speakerTracker.speakerAt`'s fallback) and finally
    /// to the supplied default.
    ///
    /// The timeline is sorted by `start` ascending (invariant of
    /// `StreamingSpeakerTracker.ingest`), and sample points advance
    /// monotonically — so we maintain a moving upper-bound index
    /// across samples instead of binary-searching each time. This
    /// keeps the per-call work to roughly O(N + samples × active),
    /// where `active` is the count of timeline entries that started
    /// before `t`.
    private func dominantSpeaker(
        from start: TimeInterval,
        to end: TimeInterval,
        fallback: String
    ) async -> String {
        let timeline = await speakerTracker.cumulativeSnapshot()
        guard !timeline.isEmpty, end > start else { return fallback }

        let dt: TimeInterval = 0.05
        let sampleCount = min(256, max(8, Int(((end - start) / dt).rounded(.up))))
        let step = (end - start) / TimeInterval(sampleCount)

        var votes: [String: Int] = [:]
        var upperBound = 0
        for i in 0..<sampleCount {
            let t = start + (TimeInterval(i) + 0.5) * step
            while upperBound < timeline.count && timeline[upperBound].start <= t {
                upperBound += 1
            }
            var instant: [String: Int] = [:]
            for j in 0..<upperBound where t <= timeline[j].end {
                instant[timeline[j].speakerID, default: 0] += 1
            }
            if let winner = instant.max(by: { $0.value < $1.value })?.key {
                votes[winner, default: 0] += 1
            }
        }
        if let mode = votes.max(by: { $0.value < $1.value })?.key {
            return mode
        }

        let mid = (start + end) / 2
        return timeline.min(by: {
            abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid)
        })?.speakerID ?? fallback
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
    /// `timestamp` origin. Used by `splitForProcessing` where
    /// `segmentAudio` covers `[asr.start, asr.end]` and we need to
    /// extract a sub-sentence range relative to that.
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
