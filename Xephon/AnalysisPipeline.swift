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
    /// speaker detection — runs upstream of `splitOnSpeakerChange`
    /// so the speaker pipeline operates on per-sentence inputs.
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

    /// Sentence-level split with one dominant speaker per sentence.
    ///
    /// Runs `splitIntoSentences` and emits one `SegmentSplit` per
    /// resulting sub-segment, tagged with whichever speaker the
    /// cumulative diarizer timeline says occupied the most audio
    /// time inside `[sentence.start, sentence.end]`. A sentence is
    /// never sub-divided by mid-sentence speaker changes — sentence
    /// integrity wins over speaker boundary precision. When the
    /// diarizer occasionally mis-classifies a few tokens inside a
    /// sentence, the majority-overlap tally absorbs that noise
    /// instead of cutting the sentence into fragments.
    ///
    /// `diarizationContext` is accepted for signature compatibility
    /// but no longer used here — per-sentence speaker resolution
    /// reads `speakerTracker` directly. The continuous diarize task
    /// in RecordingController is what keeps that timeline fresh.
    func splitForProcessing(
        asr: ASRSegment,
        segmentAudio: AudioChunk,
        diarizationContext: AudioChunk?,
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
        for i in 0..<sampleCount {
            let t = start + (TimeInterval(i) + 0.5) * step
            var instant: [String: Int] = [:]
            for entry in timeline where entry.start <= t && t <= entry.end {
                instant[entry.speakerID, default: 0] += 1
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
        let refined = await refineBoundariesByReDiarize(
            assignments: snapped,
            tokens: asr.tokens,
            diarizationContext: diarizationContext
        )

        // Short-run smoothing: absorb 1–3-token runs (~< 700 ms
        // total) whose neighbors share the same speaker. The
        // diarizer occasionally flips one short word — sometimes
        // emitted as several character-tokens — to a different
        // speaker because the embedding signal in <1 s of audio is
        // too weak to assign reliably. Smoothing those reduces the
        // visible "one wrong word" misclassifications without
        // erasing legitimate single-word interjections, which tend
        // to be longer / more emphatic.
        let smoothed = Self.absorbShortRuns(
            assignments: refined,
            tokens: asr.tokens
        )

        // Voice-based boundary confirmation: for 2-speaker
        // sentences, score each candidate gap by extracting a
        // speaker embedding from the audio on each side and
        // computing cosine distance. The gap where the two-side
        // distance is highest is where the speaker change actually
        // is — independent of which gap is largest, which token
        // the diarizer flagged, or where the energy minimum sits.
        // Falls back to the gap-only largest-pause snap when the
        // diarizer / embeddings aren't available or no candidate
        // hit the distance threshold.
        let voiceOutcome = await snapBoundaryByVoiceComparison(
            assignments: smoothed,
            tokens: asr.tokens,
            segmentAudio: segmentAudio
        )
        let acousticallySnapped: [String]
        switch voiceOutcome {
        case .fired(let confirmed):
            // Voice authoritatively confirmed the boundary (with or
            // without movement). Skip largest-pause — letting it
            // run could move a voice-confirmed boundary onto a
            // different gap that just happens to be larger but is
            // a same-speaker mid-sentence pause.
            acousticallySnapped = confirmed
        case .notFired:
            acousticallySnapped = Self.snapBoundaryToLargestPause(
                assignments: smoothed,
                tokens: asr.tokens
            )
        }

        // Grammatical-boundary snap: nudge the boundary to the
        // nearest natural cut point in the text using NLTokenizer's
        // JP word segmenter plus a particle/punctuation check on
        // each word's tail. Catches "split mid-word" and "split
        // mid-phrase" failures that the audio-based passes can't
        // see — they reason about energy, embeddings, and gaps,
        // not about whether the text reads naturally on each side.
        // Phrase ends (after particle, polite verb ending, or
        // punctuation) score higher than plain word boundaries,
        // with a small distance penalty so we don't drift far for
        // marginal grammatical gains.
        let assignments = Self.snapBoundaryToGrammaticalCut(
            assignments: acousticallySnapped,
            tokens: asr.tokens
        )

        // Note: we previously had a per-sub-segment embedding-based
        // override that re-queried FluidAudio's SpeakerManager via
        // cosine similarity. It was removed because (1) FluidAudio's
        // `performCompleteDiarization` already does embedding-based
        // clustering internally — assignSpeaker is called per
        // diarized segment with the WeSpeaker embedding, returning
        // session-stable IDs, so the override was duplicative; and
        // (2) running an extra extractSpeakerEmbedding + findSpeaker
        // per sub-segment serialized at the diarizer actor on top of
        // the continuous diarize task and boundary re-diarize, which
        // backed up the segment processing queue enough to drop
        // utterances under conversational pace.

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
        // Pick the transition flanked by the two longest-duration
        // adjacent segments (max of `min(prev.duration, next.duration)`).
        // The first transition in time order can be a transient — a
        // 0.3 s blip of speaker B before A's main turn — which would
        // pull the boundary onto noise. Weighting by min adjacent
        // duration favors transitions where both sides have enough
        // signal to be a real turn-take.
        var bestTime: TimeInterval? = nil
        var bestSupport: TimeInterval = 0
        for k in 1..<sorted.count where sorted[k].speakerID != sorted[k - 1].speakerID {
            let prevDur = sorted[k - 1].end - sorted[k - 1].start
            let nextDur = sorted[k].end - sorted[k].start
            let support = min(prevDur, nextDur)
            if support > bestSupport {
                bestSupport = support
                bestTime = sorted[k].start
            }
        }
        return bestTime
    }

    // MARK: - Short-run smoothing

    /// Maximum total duration for a same-speaker run to be eligible
    /// for absorption. Up to ~700 ms covers a typical short
    /// Japanese word; real one-word turns longer than that pass
    /// through. Tightening this threshold biased earlier versions
    /// toward keeping diarizer mistakes intact when a word
    /// happened to be 400–700 ms long.
    private static let shortRunAbsorbDurationSec: TimeInterval = 0.7
    /// Maximum length (in tokens) of a run eligible for
    /// absorption. SpeechAnalyzer sometimes emits one Japanese
    /// word as 2–3 character / syllable runs, so capping at 3
    /// catches "one wrong word" misclassifications even when they
    /// span multiple tokens. Longer runs are likely real turns
    /// even if total duration is short.
    private static let shortRunAbsorbMaxLength: Int = 3

    /// Absorb same-speaker runs whose neighboring runs agree on a
    /// single different speaker and whose total duration is below
    /// `shortRunAbsorbDurationSec` and length below
    /// `shortRunAbsorbMaxLength`. Targets the failure pattern
    /// where the diarizer assigns a few consecutive tokens
    /// (often one word's worth) to the wrong speaker even though
    /// both surrounding runs agree on the real speaker. Requires
    /// both the same-speaker sandwich AND the duration / length
    /// caps so legitimate brief interjections that happen to be
    /// longer or sit at run boundaries pass through.
    ///
    /// Runs touching the segment's first or last token are
    /// intentionally not smoothed — the surrounding context (the
    /// previous / next ASR segment) lives outside this function's
    /// view, so we can't safely judge whether they're spurious.
    private static func absorbShortRuns(
        assignments: [String],
        tokens: [ASRSegment.Token]
    ) -> [String] {
        guard assignments.count >= 3 else { return assignments }
        var result = assignments
        var i = 1
        while i < result.count - 1 {
            // Skip if no run boundary at i.
            if result[i - 1] == result[i] {
                i += 1
                continue
            }
            // Walk forward to the end of this run.
            var runEnd = i
            while runEnd + 1 < result.count && result[runEnd + 1] == result[i] {
                runEnd += 1
            }
            // Skip if the run extends to the segment's last token —
            // we don't have an "after" speaker to validate against.
            guard runEnd < result.count - 1 else {
                i = runEnd + 1
                continue
            }
            // Same speaker on both sides?
            guard result[i - 1] == result[runEnd + 1] else {
                i = runEnd + 1
                continue
            }
            let runLength = runEnd - i + 1
            let duration = tokens[runEnd].end - tokens[i].start
            if runLength <= shortRunAbsorbMaxLength,
               duration < shortRunAbsorbDurationSec {
                let surrounding = result[i - 1]
                for k in i...runEnd {
                    result[k] = surrounding
                }
            }
            i = runEnd + 1
        }
        return result
    }

    // MARK: - Voice-based boundary comparison

    /// Minimum audio duration (seconds) required on each side of a
    /// candidate boundary for voice comparison to be meaningful.
    /// FluidAudio's WeSpeaker model is trained on multi-second
    /// windows; under ~1 s the embedding has too little acoustic
    /// content to be a confident speaker fingerprint, so candidates
    /// where either side is too short are skipped.
    private static let voiceComparisonMinSideDurationSec: TimeInterval = 1.0
    /// Minimum cosine distance between the two halves' embeddings
    /// for the boundary to count as "voice-confirmed". 0.6 is well
    /// above same-speaker distances (typically < 0.4 even across
    /// prosody shifts) and below the diarizer's own clusteringThreshold,
    /// so it requires clear acoustic separation without being so
    /// strict that legitimate boundaries fail to confirm.
    private static let voiceComparisonMinDistance: Float = 0.6
    /// Minimum gap to even consider as a candidate. Below this,
    /// gaps within a sentence are too short to be a meaningful
    /// speaker-change marker.
    private static let voiceComparisonMinGapSec: TimeInterval = 0.15
    /// Required improvement over the original boundary's distance
    /// before voice comparison MOVES the boundary. Below this, the
    /// improvement is within embedding noise and the original
    /// position stays put — but voice still counts as "fired",
    /// telling the chain to skip the largest-pause fallback.
    private static let voiceComparisonMinMargin: Float = 0.05

    /// Outcome of `snapBoundaryByVoiceComparison`. Distinguishes
    /// "voice authoritatively confirmed a boundary (whether or not
    /// it moved)" from "voice didn't reach a confident decision",
    /// so the caller can skip the largest-pause fallback in the
    /// former case. Without this distinction, voice-confirmed
    /// boundaries that didn't need to move would be silently
    /// overridden by the fallback's largest-pause heuristic.
    private enum VoiceSnapOutcome {
        case fired([String])
        case notFired
    }

    /// For sentences with exactly 2 speakers and a single
    /// diarizer-detected boundary, find the candidate gap (≥ 150 ms,
    /// each side ≥ 1 s) where the two-side speaker embeddings are
    /// MOST distinct, and snap the boundary there if the cosine
    /// distance ≥ voiceComparisonMinDistance.
    ///
    /// Different from largest-pause snap (which assumes the largest
    /// gap is the boundary) — voice comparison directly answers
    /// "are the speakers actually different on each side?" so it
    /// won't mis-snap when the largest gap happens to be one
    /// speaker's mid-sentence pause. Falls back to returning the
    /// input unchanged when no candidate confirms; the caller can
    /// then chain the largest-pause snap as a fallback.
    private func snapBoundaryByVoiceComparison(
        assignments: [String],
        tokens: [ASRSegment.Token],
        segmentAudio: AudioChunk
    ) async -> VoiceSnapOutcome {
        guard let diarizer else { return .notFired }
        guard tokens.count >= 2, Set(assignments).count == 2 else { return .notFired }

        var currentBoundaryOpt: Int? = nil
        for i in 1..<assignments.count where assignments[i] != assignments[i - 1] {
            if currentBoundaryOpt != nil { return .notFired }
            currentBoundaryOpt = i
        }
        guard let currentBoundary = currentBoundaryOpt else { return .notFired }

        let firstStart = tokens[0].start
        let lastEnd = tokens[tokens.count - 1].end

        // Collect candidate gap positions. Skipping ones too short
        // for confident embedding extraction on either side keeps
        // cost bounded — typical sentence has 1–3 candidates.
        var candidates: [Int] = []
        for i in 1..<tokens.count {
            let gap = tokens[i].start - tokens[i - 1].end
            guard gap >= Self.voiceComparisonMinGapSec else { continue }
            let beforeDuration = tokens[i - 1].end - firstStart
            let afterDuration = lastEnd - tokens[i].start
            guard beforeDuration >= Self.voiceComparisonMinSideDurationSec,
                  afterDuration >= Self.voiceComparisonMinSideDurationSec
            else { continue }
            candidates.append(i)
        }
        guard !candidates.isEmpty else { return .notFired }

        // For each candidate, slice audio on both sides, extract
        // embeddings, compute cosine distance. Track both the best
        // candidate's distance and the current boundary's distance
        // (when current qualifies as a candidate) so we can require
        // a margin over the current position before moving.
        var bestCandidate: Int? = nil
        var bestDistance: Float = -.infinity
        var originalDistance: Float = -.infinity
        for candidate in candidates {
            let beforeAudio = Self.sliceRelative(
                segmentAudio,
                fromAudioTime: firstStart,
                toAudioTime: tokens[candidate - 1].end
            )
            let afterAudio = Self.sliceRelative(
                segmentAudio,
                fromAudioTime: tokens[candidate].start,
                toAudioTime: lastEnd
            )
            do {
                guard let beforeEmb = try await diarizer.embedding(for: beforeAudio.samples),
                      let afterEmb = try await diarizer.embedding(for: afterAudio.samples),
                      beforeEmb.count == afterEmb.count
                else { continue }
                // Embeddings are L2-normalized, so cosine similarity
                // = dot product. Distance = 1 - similarity.
                var dot: Float = 0
                for k in 0..<beforeEmb.count {
                    dot += beforeEmb[k] * afterEmb[k]
                }
                let distance = 1.0 - dot
                if candidate == currentBoundary {
                    originalDistance = distance
                }
                if distance > bestDistance {
                    bestDistance = distance
                    bestCandidate = candidate
                }
            } catch {
                continue
            }
        }

        guard let best = bestCandidate,
              bestDistance >= Self.voiceComparisonMinDistance
        else { return .notFired }

        // Margin gate: require a meaningful improvement over the
        // current position before moving. Same threshold-passed
        // candidates with bestDistance ≈ originalDistance are
        // within embedding noise — moving on noise causes drift.
        // When the current boundary doesn't qualify as a candidate
        // (originalDistance stayed -.infinity), any threshold-passed
        // candidate is automatically a sufficient improvement.
        let beatsMarginOverOriginal =
            originalDistance == -.infinity ||
            bestDistance >= originalDistance + Self.voiceComparisonMinMargin
        let shouldMove = best != currentBoundary && beatsMarginOverOriginal

        guard shouldMove else {
            // Voice fired (a candidate cleared the threshold) but
            // the result doesn't justify movement. Return the
            // original assignments tagged as fired so the caller
            // skips the largest-pause fallback — voice's verdict
            // (this boundary is correct as-is) stands.
            AppLog.diarization.info(
                "voice-confirmed boundary at token \(currentBoundary, privacy: .public) (d=\(bestDistance, privacy: .public), original d=\(originalDistance, privacy: .public))"
            )
            return .fired(assignments)
        }

        let speakerBefore = assignments[0]
        let speakerAfter = assignments[assignments.count - 1]
        var result = assignments
        for k in 0..<result.count {
            result[k] = k < best ? speakerBefore : speakerAfter
        }
        AppLog.diarization.info(
            "voice-snapped boundary \(currentBoundary, privacy: .public) → \(best, privacy: .public) (d=\(bestDistance, privacy: .public), original d=\(originalDistance, privacy: .public))"
        )
        return .fired(result)
    }

    // MARK: - Boundary-to-largest-pause snap

    /// Minimum inter-token gap (seconds) for the largest-pause snap
    /// to apply. Below this, gaps within a sentence are too short
    /// to be a meaningful speaker-change marker — they're often
    /// just recognizer-internal token boundaries with no real
    /// pause. Set well below `sentenceSplitMinPauseSec` (700 ms)
    /// so this catches within-sentence turn-takes that the
    /// pause-split intentionally let through.
    private static let largestPauseSnapMinGapSec: TimeInterval = 0.15

    /// For each detected boundary in `assignments`, snap it to the
    /// largest token gap inside the surrounding same-speaker runs —
    /// the natural silence where the speaker change actually
    /// happened. Targets the failure mode where Apple finalized
    /// one ASR segment spanning a speaker change and the diarizer's
    /// boundary lands a token or two off, so one speaker's last
    /// word "bleeds" into the other's run.
    ///
    /// Per-boundary local search (clamped to surrounding same-
    /// speaker runs) generalizes to multi-speaker / multi-
    /// boundary segments — each boundary picks the largest gap in
    /// its own neighborhood without affecting others. For the
    /// 1-boundary case the surrounding window covers the whole
    /// segment, matching the previous global-max behavior.
    private static func snapBoundaryToLargestPause(
        assignments: [String],
        tokens: [ASRSegment.Token]
    ) -> [String] {
        guard tokens.count >= 2, assignments.count == tokens.count else { return assignments }

        var result = assignments
        var i = 1
        while i < tokens.count {
            guard result[i - 1] != result[i] else {
                i += 1
                continue
            }

            // Run bounds — same logic as VAD-snap and re-diarize.
            // Clamp candidate range so a snap can't collapse a
            // neighboring same-speaker run to zero tokens or cross
            // an adjacent boundary.
            var runStart = i - 1
            while runStart > 0 && result[runStart - 1] == result[i - 1] {
                runStart -= 1
            }
            var nextRunEnd = i
            while nextRunEnd < tokens.count - 1 && result[nextRunEnd + 1] == result[i] {
                nextRunEnd += 1
            }

            let lo = runStart + 1
            let hi = nextRunEnd
            guard lo <= hi else {
                i = nextRunEnd + 1
                continue
            }

            var maxGap: TimeInterval = 0
            var maxGapAt = i
            for j in lo...hi {
                let gap = tokens[j].start - tokens[j - 1].end
                if gap > maxGap {
                    maxGap = gap
                    maxGapAt = j
                }
            }

            if maxGap >= largestPauseSnapMinGapSec, maxGapAt != i {
                let oldSpeaker = result[i - 1]
                let newSpeaker = result[i]
                if maxGapAt < i {
                    for k in maxGapAt..<i { result[k] = newSpeaker }
                } else {
                    for k in i..<maxGapAt { result[k] = oldSpeaker }
                }
                i = maxGapAt + 1
            } else {
                i = nextRunEnd + 1
            }
        }
        return result
    }

    // MARK: - Grammatical-boundary snap (NLTokenizer)

    /// Search radius in tokens for the grammatical-boundary snap.
    /// Same scale as the other snap passes — keeps the boundary
    /// in the local neighborhood while letting it move to the
    /// nearest grammatical seam.
    private static let grammaticalSnapRadius: Int = 3
    /// Per-character distance penalty in the snap scoring. Small
    /// enough that a clear phrase end ~5 chars away beats a
    /// mid-word at the original position; large enough that we
    /// don't drift across a long stretch for marginal grammatical
    /// improvement.
    private static let grammaticalDistancePenalty: Double = 0.05

    private struct CutInfo {
        let isWordBoundary: Bool
        let isPhraseEnd: Bool
    }

    /// For each detected boundary, snap it to the nearest natural
    /// cut point in the text. Phrase ends (after particle, polite
    /// verb ending, or punctuation) score highest; plain word
    /// boundaries are next; mid-word positions score worst.
    /// Distance penalty prefers candidates close to the original
    /// boundary.
    ///
    /// Per-boundary local search clamped to surrounding same-
    /// speaker runs, so a snap can't collapse a neighboring run
    /// to zero tokens or cross an adjacent boundary. NLTokenizer's
    /// word boundaries are good but not perfect on conversational
    /// JP — uncommon names and code-switched terms may not snap
    /// cleanly, in which case we leave the boundary alone.
    private static func snapBoundaryToGrammaticalCut(
        assignments: [String],
        tokens: [ASRSegment.Token]
    ) -> [String] {
        guard tokens.count >= 2, assignments.count == tokens.count else { return assignments }

        // Build text + per-token character offset map. tokenStart[i]
        // is the character index (in the concatenated text) where
        // token i begins; the gap between tokens i-1 and i lands
        // at character index tokenStart[i].
        var tokenStart: [Int] = []
        var totalText = ""
        for token in tokens {
            tokenStart.append(totalText.count)
            totalText += token.text
        }

        let cuts = grammaticalCutPoints(in: totalText)

        var result = assignments
        var i = 1
        while i < tokens.count {
            guard result[i - 1] != result[i] else {
                i += 1
                continue
            }

            // Run bounds — same logic as VAD-snap and re-diarize.
            var runStart = i - 1
            while runStart > 0 && result[runStart - 1] == result[i - 1] {
                runStart -= 1
            }
            var nextRunEnd = i
            while nextRunEnd < tokens.count - 1 && result[nextRunEnd + 1] == result[i] {
                nextRunEnd += 1
            }

            let lo = max(runStart + 1, i - Self.grammaticalSnapRadius)
            let hi = min(nextRunEnd, i + Self.grammaticalSnapRadius)
            guard lo <= hi else {
                i = nextRunEnd + 1
                continue
            }
            let originCharPos = tokenStart[i]

            func score(at tokenIdx: Int) -> Double {
                let charPos = tokenStart[tokenIdx]
                let distance = Double(abs(charPos - originCharPos))
                let distancePenalty = distance * Self.grammaticalDistancePenalty
                if let info = cuts[charPos] {
                    if info.isPhraseEnd { return 2.0 - distancePenalty }
                    if info.isWordBoundary { return 1.0 - distancePenalty }
                }
                // Mid-word — no entry in `cuts` for this char index.
                return -distancePenalty
            }

            var bestPos = i
            var bestScore = score(at: i)
            for j in lo...hi where j != i {
                let s = score(at: j)
                if s > bestScore {
                    bestScore = s
                    bestPos = j
                }
            }

            if bestPos != i {
                AppLog.diarization.info(
                    "grammatical snap: token boundary \(i, privacy: .public) → \(bestPos, privacy: .public) (score \(bestScore, privacy: .public))"
                )
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

    /// Use NLTokenizer's JP word segmenter to map each word's
    /// start character index to a `CutInfo`. The "phrase end" flag
    /// is set when the *previous* word is a particle or has a
    /// polite-form verb ending, OR the position immediately
    /// follows a punctuation character in the source text — those
    /// are the natural seams in JP where a sentence can be cut
    /// without leaving a sub-segment that reads as a sentence
    /// fragment. Indices not in the dictionary are inside a word.
    ///
    /// Two limitations the implementation works around:
    /// 1. NLTagger's `.lexicalClass` scheme does not tag JP
    ///    particles/verbs/punctuation reliably even with an
    ///    explicit `.japanese` language hint (observed on iOS 26:
    ///    27 word starts → 0 POS-tagged tokens), so the
    ///    phrase-end check works directly off each word's text.
    /// 2. NLTokenizer's `.word` unit silently omits punctuation
    ///    tokens from JP enumeration — 「だな。ちょっと」 yields
    ///    [だ, な, ちょっと] with 「。」 missing, so detecting
    ///    sentence boundaries via "previous word ended in 。"
    ///    fails for nominal endings (e.g., 「親父殿。」). We patch
    ///    that by scanning the text for punctuation directly and
    ///    marking the position after each as a phrase-end.
    private static func grammaticalCutPoints(in text: String) -> [Int: CutInfo] {
        guard !text.isEmpty else { return [:] }
        var info: [Int: CutInfo] = [:]
        let tokenizer = NLTokenizer(unit: .word)
        tokenizer.string = text
        // Force JP segmentation. Auto-detection misclassifies
        // short fragments as undetermined, which falls back to a
        // whitespace tokenizer that produces one giant token for
        // unspaced JP text.
        tokenizer.setLanguage(.japanese)
        var lastWordText: Substring? = nil
        tokenizer.enumerateTokens(
            in: text.startIndex..<text.endIndex
        ) { tokenRange, _ in
            let charIdx = text.distance(from: text.startIndex, to: tokenRange.lowerBound)
            let isPhraseEnd = lastWordText.map { Self.isPhraseEndingWord($0) } ?? false
            info[charIdx] = CutInfo(isWordBoundary: true, isPhraseEnd: isPhraseEnd)
            lastWordText = text[tokenRange]
            return true
        }
        // Punctuation augmentation — see (2) in the doc comment.
        let totalChars = text.count
        for (i, ch) in text.enumerated() {
            guard Self.phraseEndPunctuation.contains(ch) else { continue }
            let nextIdx = i + 1
            guard nextIdx < totalChars else { continue }
            info[nextIdx] = CutInfo(isWordBoundary: true, isPhraseEnd: true)
        }
        AppLog.diarization.debug(
            "NLTokenizer cuts in \"\(text, privacy: .public)\": \(info.count, privacy: .public) word starts, \(info.values.filter { $0.isPhraseEnd }.count, privacy: .public) phrase-ends"
        )
        return info
    }

    /// Phrase-end heuristic for a JP word. Fires when the word
    /// ends a natural clause: punctuation tail, a single-char
    /// case/sentence-final particle, a known multi-char particle
    /// or conjunction, or a polite-form verb ending
    /// (です／ます／でした／ました).
    private static func isPhraseEndingWord(_ word: Substring) -> Bool {
        guard let last = word.last else { return false }
        if Self.phraseEndPunctuation.contains(last) { return true }
        if word.count == 1, Self.singleCharParticles.contains(last) {
            return true
        }
        if Self.multiCharParticles.contains(String(word)) { return true }
        if word.hasSuffix("です") || word.hasSuffix("ます")
            || word.hasSuffix("でした") || word.hasSuffix("ました") {
            return true
        }
        return false
    }

    private static let phraseEndPunctuation: Set<Character> = [
        "。", "、", "！", "？", "…",
        "!", "?", ".", ",",
        "」", "』", ")", "）",
    ]
    /// Hiragana that, as a standalone one-character word, signal
    /// a clause boundary: case particles (は・が・を・に・で・と・へ),
    /// inclusive/listing markers (も・や), and question /
    /// sentence-final particles (か・ね・よ・な・わ・ぞ・ぜ・さ).
    private static let singleCharParticles: Set<Character> = [
        "は", "が", "を", "に", "で", "と", "へ", "も", "や",
        "か", "ね", "よ", "な", "わ", "ぞ", "ぜ", "さ",
    ]
    private static let multiCharParticles: Set<String> = [
        "から", "まで", "より", "には", "では", "とは",
        "って", "けど", "けれど", "のに", "ので", "なら",
        "じゃ", "だけ", "しか", "ばかり", "ほど",
    ]

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
