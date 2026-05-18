import Foundation
import NaturalLanguage
import Audio
import ASR
import Diarization
import SERAcoustic
import SERRuntime
import SERText
import Fusion
import XephonLogging
import XephonUtilities

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

/// Outcome of a `runText` call. Distinguishes a clean score from the
/// two skip reasons we want the UI to render differently: filler /
/// empty / no-backend (no chip) vs. Apple FM guardrail decline
/// (dedicated "Apple FM ✕" chip).
struct TextSEROutcome: Sendable {
    let score: PlutchikScore?
    let guardrailViolation: Bool

    static let empty = TextSEROutcome(score: nil, guardrailViolation: false)
}

/// Result of `autoConfigured(modelStore:)`. Carries the pipeline plus
/// any per-modality construction errors so the UI can surface them
/// rather than silently dropping the affected modality. An empty
/// `diagnostics` array means everything loaded.
struct AutoConfiguredPipeline: Sendable {
    let pipeline: AnalysisPipeline
    let diagnostics: [String]
}

// `@unchecked Sendable` is a deliberate concession to the
// `transcriber` swap on `setLocale(_:)`. The class is otherwise
// composed of `let` actor/value-type references which are Sendable
// by construction; only `transcriber` is mutable. All mutation
// happens from `RecordingController`'s @MainActor surface, and
// readers reach it via `await` on actor methods, so cross-isolation
// concurrent access doesn't occur in practice. Marked unchecked
// rather than wrapping in a lock because each call site already
// goes through `await`/`async` boundaries that serialize naturally.
// Intentionally NOT @MainActor: heavy SER constructors (e.g. W2V2 ONNX load,
// ~631 MB) and per-segment inference must not block the UI thread. All stored
// references are themselves Sendable (actors / value types).
final class AnalysisPipeline: @unchecked Sendable {
    /// Offline (re-evaluate / file-mode batch) transcriber. `var`
    /// rather than `let` because the user-facing language picker can
    /// change between sessions while the pipeline instance lives on
    /// across them — `setLocale(_:)` swaps in a fresh
    /// `SpeechAnalyzerTranscriber` rather than recreating the whole
    /// pipeline. Tests inject a different `Transcriber` and bypass
    /// `setLocale`, so their injection survives.
    private var transcriber: any Transcriber
    private let diarizer: (any Diarizer)?
    private let dimensionalSER: (any DimensionalAcousticSER)?
    private let categoricalSER: (any CategoricalAcousticSER)?
    /// Optional speaker-demographics inferencer (W2V2 age-gender).
    /// Skipped when the model isn't installed; pipeline degrades
    /// gracefully and rows just carry `ageGender == nil`.
    private let ageGenderSER: (any AgeGenderSER)?
    private let textSER: (any TextSER)?

    /// Snapshot booleans for surfacing per-component readiness in
    /// the UI ("Models" card) without exposing the pipeline's
    /// private component references. All are `let` so they're safe
    /// to access `nonisolated` — the values are fixed at
    /// construction by `autoConfigured(modelStore:)`.
    public nonisolated var hasDiarizer: Bool { diarizer != nil }
    public nonisolated var hasDimensionalSER: Bool { dimensionalSER != nil }
    public nonisolated var hasCategoricalSER: Bool { categoricalSER != nil }
    public nonisolated var hasAgeGenderSER: Bool { ageGenderSER != nil }
    /// True when the bundled DeBERTa-WRIME text-SER model was
    /// successfully loaded into the pipeline's `SwitchingTextSER`.
    /// `false` doesn't mean the pipeline can't do text SER — Apple
    /// Foundation Models stays available as the fallback backend —
    /// but the Japanese-tuned Plutchik head is gone.
    public nonisolated let hasDeBERTaTextSER: Bool
    /// Mutable so the controller can swap in a `LateFusion` with
    /// fresh weights when the user adjusts the fusion-control
    /// sliders. New utterances fuse under the new weights; old
    /// utterances keep their cached fused V/A/D until manually
    /// re-evaluated.
    private var fuser: any Fuser
    /// Bridges per-call diarizer outputs to session-stable speaker IDs
    /// by time-overlap matching. Reset between sessions via
    /// `resetSpeakerTracking()`.
    private let speakerTracker: StreamingSpeakerTracker

    init(
        transcriber: any Transcriber = SpeechAnalyzerTranscriber(),
        diarizer: (any Diarizer)? = nil,
        dimensionalSER: (any DimensionalAcousticSER)? = nil,
        categoricalSER: (any CategoricalAcousticSER)? = nil,
        ageGenderSER: (any AgeGenderSER)? = nil,
        textSER: (any TextSER)? = nil,
        hasDeBERTaTextSER: Bool = false,
        fuser: any Fuser = LateFusion(),
        speakerTracker: StreamingSpeakerTracker = StreamingSpeakerTracker()
    ) {
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.dimensionalSER = dimensionalSER
        self.categoricalSER = categoricalSER
        self.ageGenderSER = ageGenderSER
        self.textSER = textSER
        self.hasDeBERTaTextSER = hasDeBERTaTextSER
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

    /// Swap in a fresh `LateFusion` with the supplied weights so
    /// subsequent `fuse(...)` calls use them. Old already-fused
    /// utterances are not retroactively updated — the controller
    /// re-evaluates them out-of-band if the user wants the change
    /// to apply to historical rows.
    func setFusionWeights(
        acousticWeight: Float,
        textWeightFloor: Float
    ) {
        fuser = LateFusion(
            textWeightFloor: textWeightFloor,
            acousticWeight: acousticWeight
        )
    }

    /// No-op when diarization isn't configured.
    func setDiarizerClusteringThreshold(_ value: Float) async {
        await diarizer?.setClusteringThreshold(value)
    }

    /// Run the diarizer on a window of audio and merge the result
    /// into the cumulative speaker timeline via `speakerTracker`.
    /// Called by the continuous-diarization side task in
    /// RecordingController on every stride tick. No-op when the
    /// diarizer is unavailable or returns empty for this window.
    /// The audio's `timestamp` is the absolute audio-time origin of
    /// the window, so the tracker's history stays in the same time
    /// frame as ASRSegment / token timings.
    /// Snapshot the cumulative diarizer timeline (every observation
    /// the continuous-diarize task has ingested since the session
    /// reset) as a value-typed copy, sorted by `start`. Used by the
    /// timeline-strip visualization in the transcript pane.
    func diarizationTimelineSnapshot() async -> [DiarizedSegment] {
        await speakerTracker.cumulativeSnapshot()
    }

    /// Snapshot the diarizer's speaker database for persistence in
    /// the session bundle. Returns nil when diarization isn't
    /// configured or there's nothing to save.
    func exportSpeakerDatabase() async -> Data? {
        await diarizer?.exportSpeakerDatabase()
    }

    /// Snapshot the in-memory speaker cluster for the visualization
    /// panels. Empty when no diarizer is configured. Forwards
    /// `maxObservationsPerSpeaker` so the caller can bound the PCA
    /// observation cloud size.
    func clusterSnapshot(maxObservationsPerSpeaker: Int) async -> SpeakerClusterSnapshot {
        guard let diarizer else {
            return SpeakerClusterSnapshot(speakers: [])
        }
        return await diarizer.clusterSnapshot(
            maxObservationsPerSpeaker: maxObservationsPerSpeaker
        )
    }

    /// Restore a previously-saved speaker database. No-op when the
    /// diarizer isn't configured. Throws on malformed blobs;
    /// callers handle/log.
    func importSpeakerDatabase(_ data: Data) async throws {
        try await diarizer?.importSpeakerDatabase(data)
    }

    /// Extract a session-stable speaker embedding from a clip of
    /// audio. Used by the "Promote New Speaker" flow: pull a
    /// 256-D embedding from the user-vetted utterance's audio
    /// slice, then hand it to `promoteSpeaker(id:embedding:)`.
    /// Returns nil when no diarizer is configured or when the
    /// underlying embedding extractor isn't loaded yet.
    func extractSpeakerEmbedding(audio: [Float]) async -> [Float]? {
        guard let diarizer else { return nil }
        do {
            return try await diarizer.embedding(for: audio)
        } catch {
            AppLog.app.warning(
                "embedding extraction failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Argmin cosine distance over the diarizer's currently-resident
    /// raw observations for `speakerID`, returning the matching
    /// observation's stable `segmentId`. Captured at utterance-
    /// finalization time so the speaker-cluster scatter's tap-to-
    /// scroll can resolve back from observation → utterance by id
    /// rather than another embedding-distance pass at tap time.
    /// Nil when the diarizer hasn't loaded, the speaker isn't in
    /// the database yet, or the speaker has no observations.
    func bestMatchingObservationID(
        forEmbedding embedding: [Float],
        speakerID: String
    ) async -> UUID? {
        await diarizer?.bestMatchingObservationID(
            forEmbedding: embedding,
            speakerID: speakerID
        )
    }

    /// Register a user-promoted speaker (id + embedding) into the
    /// diarizer's session database. Throws on diarizer failure
    /// (e.g. models not loaded and download fails).
    func promoteSpeaker(id: String, embedding: [Float]) async throws {
        try await diarizer?.promoteSpeaker(id: id, embedding: embedding)
    }

    /// Fold a new audio observation into an existing speaker's
    /// centroid in the diarizer DB. Used by the "Teach diarizer"
    /// corrective reassignment flow.
    func correctSpeaker(id: String, embedding: [Float], duration: Float) async throws {
        try await diarizer?.correctSpeaker(id: id, embedding: embedding, duration: duration)
    }

    /// Remove a speaker from the diarizer DB. Used by the auto-
    /// demote pass when a speaker id falls out of every
    /// utterance after a reassignment / correction / promotion /
    /// re-evaluation. `keepIfPermanent: true` skips user-
    /// promoted entries.
    func removeSpeakerFromDB(id: String, keepIfPermanent: Bool) async throws {
        try await diarizer?.removeSpeakerFromDB(id: id, keepIfPermanent: keepIfPermanent)
    }

    /// Propagate a foreground/background lifecycle transition to
    /// every SER actor that knows how to swap its inference backend
    /// across the boundary. Forwarded from
    /// `RecordingController.setBackgroundMode`, which itself is
    /// driven by the app's scene-phase observer in `ContentView`.
    ///
    /// `inBackground == true` → actors rebuild on CPU before iOS
    /// revokes ANE/GPU privileges. `false` → actors rebuild on
    /// their preferred backend (CoreML EP if they were constructed
    /// with it allowed).
    func setBackgroundMode(_ inBackground: Bool) async {
        if let m = dimensionalSER as? any BackgroundAwareSER {
            await m.setBackgroundMode(inBackground)
        }
        if let m = categoricalSER as? any BackgroundAwareSER {
            await m.setBackgroundMode(inBackground)
        }
        if let m = ageGenderSER as? any BackgroundAwareSER {
            await m.setBackgroundMode(inBackground)
        }
        if let m = textSER as? any BackgroundAwareSER {
            await m.setBackgroundMode(inBackground)
        }
    }

    func ingestDiarizationWindow(_ audio: AudioChunk) async {
        guard diarizer != nil, !audio.samples.isEmpty else { return }
        let diarized = await runDiarization(audio)
        guard !diarized.isEmpty else { return }
        _ = await speakerTracker.ingest(diarized)
    }

    /// Run the diarizer on `audio` in isolation and resolve a
    /// speaker for each requested sub-range from **only the fresh
    /// segments** — i.e. without consulting the cumulative timeline
    /// and without mutating it. Used by the hand-edit path so the
    /// re-diarized verdict actually drives the resulting speaker
    /// assignment instead of being drowned out by the ~5 overlapping
    /// observations the continuous-diarize task already deposited
    /// for that window during the original streaming pass.
    ///
    /// Sub-ranges are in absolute audio time. Each range's speaker
    /// is the duration-weighted mode of fresh segments that overlap
    /// it. When the fresh diarization is empty (e.g. the window is
    /// shorter than Sortformer's effective minimum, or the model
    /// found no speech) or no fresh segment overlaps the range, that
    /// entry falls back to `fallback`.
    ///
    /// The cumulative timeline is intentionally untouched here. A
    /// snapshot-revert via `revertReevaluation` therefore restores
    /// the row to its pre-hand-edit state without needing to also
    /// roll back the global speaker timeline.
    func resolveSpeakersForRanges(
        audio: AudioChunk,
        ranges: [(start: TimeInterval, end: TimeInterval)],
        fallback: String,
        preserveSpeakerDatabase: Bool = false
    ) async -> [String] {
        guard !ranges.isEmpty else { return [] }
        let durationSec = Double(audio.samples.count) / audio.sampleRate
        AppLog.diarization.info(
            "resolveSpeakersForRanges: input audio=\(audio.samples.count, privacy: .public) samples (\(durationSec, privacy: .public)s) timestamp=\(audio.timestamp, privacy: .public)s ranges=\(ranges.count, privacy: .public) fallback=\(fallback, privacy: .public) preserveDB=\(preserveSpeakerDatabase, privacy: .public)"
        )
        // When the caller asks us to leave the speaker database
        // untouched (utterance-split path: we want the diarizer's
        // verdict for the split sub-ranges, but we DON'T want each
        // sub-range's audio EMA-blended into the speakers' centroids
        // — that would let one wrongly-split utterance silently
        // poison the database for the rest of the session), snapshot
        // before the diarization run and restore after. The
        // snapshot/restore round-trip is the same machinery used by
        // Save/Load, so it's already exercised in production.
        let snapshot: Data? = preserveSpeakerDatabase
            ? await exportSpeakerDatabase()
            : nil
        let fresh = await runDiarization(audio)
        if let snapshot {
            do {
                try await importSpeakerDatabase(snapshot)
            } catch {
                AppLog.diarization.warning(
                    "resolveSpeakersForRanges: speaker-DB restore failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
        AppLog.diarization.info(
            "resolveSpeakersForRanges: fresh segments count=\(fresh.count, privacy: .public)"
        )
        for (i, seg) in fresh.enumerated() {
            AppLog.diarization.info(
                "  seg[\(i, privacy: .public)] speaker=\(seg.speakerID, privacy: .public) start=\(seg.start, privacy: .public)s end=\(seg.end, privacy: .public)s"
            )
        }
        guard !fresh.isEmpty else {
            AppLog.diarization.warning(
                "resolveSpeakersForRanges: empty fresh diarization → falling back to \(fallback, privacy: .public) for all \(ranges.count, privacy: .public) ranges"
            )
            return Array(repeating: fallback, count: ranges.count)
        }
        let resolved = ranges.map { range in
            Self.dominantSpeakerInSegments(
                fresh,
                from: range.start,
                to: range.end,
                fallback: fallback
            )
        }
        for (i, range) in ranges.enumerated() {
            AppLog.diarization.info(
                "  range[\(i, privacy: .public)] [\(range.start, privacy: .public)..\(range.end, privacy: .public)] → speaker=\(resolved[i], privacy: .public)"
            )
        }
        return resolved
    }

    /// Pure helper: per-instant majority vote over `segments` for the
    /// `[start, end]` window, with a nearest-midpoint fallback when
    /// nothing overlaps. Factored out of `dominantSpeaker` so the
    /// hand-edit path can apply the same voting rule to a *fresh*
    /// segment list without touching `speakerTracker`.
    /// Vote-density tuning shared between `dominantSpeaker` (queries
    /// the cumulative timeline) and `dominantSpeakerInSegments`
    /// (queries an in-isolation fresh diarization). 50 ms sample
    /// step sits well below the diarizer's ~200–500 ms segment
    /// granularity, and the (8, 256) clamp keeps the per-call work
    /// bounded — very long ranges don't grow the inner loop
    /// unboundedly, very short ones still get a meaningful number
    /// of votes.
    static let speakerVoteSampleStepSec: TimeInterval = 0.05
    static let speakerVoteSampleCountMin: Int = 8
    static let speakerVoteSampleCountMax: Int = 256

    static func dominantSpeakerInSegments(
        _ segments: [DiarizedSegment],
        from start: TimeInterval,
        to end: TimeInterval,
        fallback: String
    ) -> String {
        guard !segments.isEmpty, end > start else { return fallback }
        let sorted = segments.sorted { $0.start < $1.start }

        let dt: TimeInterval = speakerVoteSampleStepSec
        let sampleCount = min(
            speakerVoteSampleCountMax,
            max(speakerVoteSampleCountMin, Int(((end - start) / dt).rounded(.up)))
        )
        let step = (end - start) / TimeInterval(sampleCount)

        var votes: [String: Int] = [:]
        var upperBound = 0
        for i in 0..<sampleCount {
            let t = start + (TimeInterval(i) + 0.5) * step
            while upperBound < sorted.count && sorted[upperBound].start <= t {
                upperBound += 1
            }
            var instant: [String: Int] = [:]
            for j in 0..<upperBound where t <= sorted[j].end {
                instant[sorted[j].speakerID, default: 0] += 1
            }
            if let winner = instant.max(by: { $0.value < $1.value })?.key {
                votes[winner, default: 0] += 1
            }
        }
        if let mode = votes.max(by: { $0.value < $1.value })?.key {
            return mode
        }
        let mid = (start + end) / 2
        return sorted.min(by: {
            abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid)
        })?.speakerID ?? fallback
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
                // CPU-only at construction. The post-FP16 graph
                // CoreML EP previously consumed worked fine, but the
                // dynamo-re-exported FP32 graph trips the EP with
                // "broken/unsupported model (error code: -1)" on
                // every first call — the per-instance fallback in
                // `runInference` already rebuilds on CPU, but skipping
                // the doomed first attempt avoids the noisy log line
                // and the wasted session-init at every app launch.
                try W2V2DimensionalSER(modelURL: url, useCoreML: false)
            }
        }

        let emotion2vecURL: URL? = await Self.tryResolveAsync("emotion2vec", path: "emotion2vec_onnx/model.onnx", store: modelStore, diagnostics: &diagnostics)
        let categorical: (any CategoricalAcousticSER)? = emotion2vecURL.flatMap { url in
            Self.tryInit("emotion2vec categorical SER", diagnostics: &diagnostics) {
                try Emotion2VecCategoricalSER(modelURL: url)
            }
        }

        let ageGenderURL: URL? = await Self.tryResolveAsync("W2V2 age-gender", path: "w2v2-age-gender/model.onnx", store: modelStore, diagnostics: &diagnostics)
        let ageGender: (any AgeGenderSER)? = ageGenderURL.flatMap { url in
            Self.tryInit("W2V2 age-gender SER", diagnostics: &diagnostics) {
                // CPU-only. Tried CoreML EP after the V/A/D move to
                // CPU eased the historical IOSurface cascade, but
                // the age-gender `.onnx` references a separate
                // `.data` weights file (external-data layout) and
                // the CoreML EP's subgraph partitioner doesn't
                // propagate `model_path` into its compilation
                // pass — ORT crashes session-init with
                // `!model_path.empty() was false` in
                // `onnxruntime::Initializer::Initializer`. Fix
                // requires re-exporting age-gender with weights
                // inlined into a single self-contained `.onnx`,
                // or upstream ORT patching the EP path resolution.
                // emotion2vec uses external data too but happens
                // to take a different ORT code path that doesn't
                // trip this — fragile, but it works for now.
                try W2V2AgeGenderSER(modelURL: url, useCoreML: false)
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
                // CoreML EP. Same historical caveat as
                // W2V2 age-gender above: the IOSurface cascade was
                // a 3-session phenomenon. With V/A/D now on CPU
                // permanently, the EP user count tops out at 3
                // (emotion2vec + age-gender + DeBERTa) — back at
                // the historical threshold, but now mitigated by
                // (a) the proactive foreground/background swap so
                // EP sessions don't pile up rebuilding during
                // backgrounding, and (b) the per-actor reactive
                // CPU-rebuild shim that catches anything the swap
                // misses. DeBERTa's CPU latency is small (~50 ms)
                // so reverting to CPU here is the cheapest rollback
                // if the cascade re-appears in practice.
                try await DeBERTaWRIME(modelURL: model, tokenizerDirectory: dir, useCoreML: true)
            }
        } else {
            deberta = nil
        }
        let textSER: any TextSER = SwitchingTextSER(
            deberta: deberta,
            foundationModels: FoundationModelsSER()
        )

        AppLog.app.info(
            "AnalysisPipeline ready: dimensional=\(dimensional != nil, privacy: .public), categorical=\(categorical != nil, privacy: .public), ageGender=\(ageGender != nil, privacy: .public), deberta=\(deberta != nil, privacy: .public), diarizer=\(diarizer != nil, privacy: .public)"
        )

        // Eagerly load the diarizer models so the first-install
        // compile cost (~5-6 s for wespeaker_v2 + pyannote on
        // M-class) is paid here in the pre-warm rather than
        // stalling the continuous-diarize loop's first call mid-
        // recording. Without this, the first ~6 s of audio produces
        // utterances with no cumulative-strip entries and no
        // cluster-plot nodes (the diarizer lazy-loads on its first
        // `diarize` call, blocking that call). On subsequent
        // launches the models are cached and `preload` returns
        // promptly.
        if let diarizer {
            do {
                try await diarizer.preload()
            } catch {
                diagnostics.append("diarizer preload failed: \(String(describing: error))")
                AppLog.diarization.warning(
                    "diarizer preload failed: \(String(describing: error), privacy: .public)"
                )
            }
        }

        let pipeline = AnalysisPipeline(
            diarizer: diarizer,
            dimensionalSER: dimensional,
            categoricalSER: categorical,
            ageGenderSER: ageGender,
            textSER: textSER,
            hasDeBERTaTextSER: deberta != nil
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

    /// Re-target the pipeline at a new session language. Swaps in a
    /// fresh offline `SpeechAnalyzerTranscriber` for the new locale
    /// (the streaming transcriber lives on `RecordingController`
    /// and gets recreated alongside) and tells `SwitchingTextSER`
    /// about the language so its DeBERTa-WRIME gating and its
    /// Foundation Models prompt language update in lockstep.
    /// `languageLabel` is the human-readable string baked into the
    /// FM prompt opener — pass nil to leave the prompt
    /// language-agnostic.
    func setLocale(_ locale: Locale, languageLabel: String?) async {
        transcriber = SpeechAnalyzerTranscriber(locale: locale)
        let code = locale.language.languageCode?.identifier
        await (textSER as? SwitchingTextSER)?.setLanguage(
            code: code,
            label: languageLabel
        )
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
        let segments = try await transcribeForReevaluation(
            audio: audio,
            onVolatileText: onVolatileText
        )
        return try await reevaluateFromSegments(
            segments: segments,
            audio: audio,
            originalStart: originalStart,
            originalEnd: originalEnd,
            speakerID: speakerID
        )
    }

    /// Run the offline transcriber for a re-evaluate pass and return
    /// its raw segments. Split out from `reevaluate` so callers can
    /// drive their own retry / back-pad-expansion loops (e.g.
    /// short-utterance retry) without paying the SER + fusion cost
    /// on each iteration. Forwards the volatile callback only when
    /// the configured transcriber is the Apple variant — others
    /// fall back to final-only.
    func transcribeForReevaluation(
        audio: AudioChunk,
        onVolatileText: (@Sendable @MainActor (String) -> Void)? = nil
    ) async throws -> [ASRSegment] {
        if let speech = transcriber as? SpeechAnalyzerTranscriber, onVolatileText != nil {
            return try await speech.transcribe(audio, onVolatileText: onVolatileText)
        }
        return try await transcriber.transcribe(audio)
    }

    /// Trim already-collected segments to "every full sentence", slice
    /// the audio in lockstep, and run SER + fusion. Same flow as
    /// `reevaluate`; just lets the caller supply the segments. The
    /// short-utterance retry path uses this to avoid re-running ASR
    /// after it already has the segments that contain a terminator.
    func reevaluateFromSegments(
        segments: [ASRSegment],
        audio: AudioChunk,
        originalStart: TimeInterval,
        originalEnd: TimeInterval,
        speakerID: String
    ) async throws -> (UtteranceEstimate, ProcessingMetrics)? {
        guard !segments.isEmpty else { return nil }
        let combinedText = segments.map(\.text).joined()
        guard !combinedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return nil
        }
        // Keep every full sentence the offline ASR produced; drop any
        // trailing fragment that didn't terminate. The pad on each
        // side of the original utterance often picks up the leading
        // phoneme of the next sentence (or the tail of the previous
        // one), and SpeechAnalyzer happily transcribes whatever's
        // there — but those fragments rarely punctuate cleanly.
        // Trimming at the LAST terminator gets us every committed
        // sentence while pruning the unfinished tail.
        //
        // When the offline ASR emitted per-token timing we trim the
        // audio chunk in lockstep so acoustic SER sees only the
        // committed sentences' prosody — the unfinished tail no
        // longer colours the V/A/D. When tokens are missing (e.g. a
        // fallback path that didn't capture them), text trims but
        // audio passes through unchanged.
        let allTokens = segments.flatMap(\.tokens)
        let trimmedText: String
        let trimmedAudio: AudioChunk
        // Corrected absolute time range for the result. When token
        // timing is available, the first / last chosen token's audio
        // times tell us where the kept sentences actually begin and
        // end in the source file — the row's timestamp updates to
        // match. When token timing is missing, we fall back to the
        // caller's originalStart/originalEnd anchors.
        let correctedStart: TimeInterval
        let correctedEnd: TimeInterval
        if let endTokenIdx = Self.lastSentenceEndTokenIndex(in: allTokens) {
            let chosen = Array(allTokens[0...endTokenIdx])
            trimmedText = chosen.map(\.text).joined()
            let firstBufferLocal = chosen.first?.start ?? 0
            let lastBufferLocal = chosen.last?.end ?? firstBufferLocal
            trimmedAudio = Self.sliceAudioFromStart(
                audio,
                upToBufferLocalEnd: lastBufferLocal
            )
            correctedStart = audio.timestamp + firstBufferLocal
            correctedEnd = audio.timestamp + lastBufferLocal
        } else if let terminatedPrefix = Self.allFullSentences(in: combinedText),
                  terminatedPrefix.count < combinedText.count {
            // No tokens, but text-level terminator(s) found. Trim
            // text at the last terminator, keep audio.
            trimmedText = terminatedPrefix
            trimmedAudio = audio
            correctedStart = originalStart
            correctedEnd = originalEnd
        } else {
            // No terminator at all (single-clause utterance or
            // transcriber didn't punctuate). Keep everything.
            trimmedText = combinedText
            trimmedAudio = audio
            correctedStart = originalStart
            correctedEnd = originalEnd
        }
        guard !trimmedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              !trimmedAudio.samples.isEmpty else {
            return nil
        }
        let confidences = segments.compactMap(\.confidence)
        let combinedConfidence: Float? = confidences.isEmpty
            ? nil
            : confidences.reduce(0, +) / Float(confidences.count)
        let synthesized = ASRSegment(
            text: trimmedText,
            start: correctedStart,
            end: correctedEnd,
            confidence: combinedConfidence,
            tokens: []
        )
        return try await processSegment(
            asr: synthesized,
            segmentAudio: trimmedAudio,
            fallbackSpeakerID: speakerID
        )
    }

    /// True when `segments` contain at least one sentence-ending
    /// character anywhere in their concatenated text. Used by the
    /// short-utterance retry path to decide whether the current
    /// back-pad is enough to capture a full sentence boundary, or
    /// whether to grow the pad and try offline ASR again.
    static func segmentsContainFullSentence(_ segments: [ASRSegment]) -> Bool {
        for segment in segments {
            if segment.text.contains(where: { sentenceEndChars.contains($0) }) {
                return true
            }
        }
        return false
    }

    /// Index of the LAST token whose text ends with a sentence-end
    /// character — the cut point for "keep every full sentence,
    /// drop the trailing fragment". Returns nil when no token has
    /// one (so the caller can fall back to a text-only or no-op
    /// trim).
    private static func lastSentenceEndTokenIndex(in tokens: [ASRSegment.Token]) -> Int? {
        for i in stride(from: tokens.count - 1, through: 0, by: -1) {
            if let last = tokens[i].text.last, sentenceEndChars.contains(last) {
                return i
            }
        }
        return nil
    }

    /// Returns the prefix of `text` up to and including the LAST
    /// sentence-ending character — i.e. every completed sentence
    /// concatenated, with any unterminated trailing fragment cut.
    /// Returns nil when no terminator appears so the caller can
    /// distinguish "trimmed" from "single-clause keep-everything".
    private static func allFullSentences(in text: String) -> String? {
        var lastTerminatorIndex: String.Index?
        for index in text.indices where sentenceEndChars.contains(text[index]) {
            lastTerminatorIndex = index
        }
        guard let last = lastTerminatorIndex else { return nil }
        return String(text[...last])
    }

    /// Slice an audio chunk from sample 0 up to `upToBufferLocalEnd`
    /// seconds (where the chunk's local timeline begins at sample 0,
    /// not at its absolute `timestamp`). Clamps to the available
    /// sample count so a slightly-overshooting token end doesn't
    /// crash. Preserves `timestamp` so downstream code interpreting
    /// the chunk as "starts at absolute time X" still works.
    private static func sliceAudioFromStart(
        _ audio: AudioChunk,
        upToBufferLocalEnd end: TimeInterval
    ) -> AudioChunk {
        let endSample = min(
            audio.samples.count,
            max(0, Int((end * audio.sampleRate).rounded()))
        )
        return AudioChunk(
            samples: Array(audio.samples[0..<endSample]),
            sampleRate: audio.sampleRate,
            timestamp: audio.timestamp
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
        async let demographics = runAgeGender(segmentAudio)
        async let plutchik = runText(asr.text)

        // Acoustic timing = wall time until both dimensional + categorical
        // resolve (they're concurrent). Text timing = wall time until plutchik
        // resolves. The two overlap in real time but are reported separately.
        let textStart = Date()
        let textResult = await plutchik
        let textDuration = Date().timeIntervalSince(textStart)
        let txt = textResult.score

        let dim = await dimensional
        let cat = await categorical
        let ageGender = await demographics
        let acousticDuration = Date().timeIntervalSince(acousticStart)

        let baseEstimate = try await fuser.fuse(
            asr: asr,
            speakerID: fallbackSpeakerID,
            dimensional: dim,
            acousticCategorical: cat,
            plutchik: txt
        )
        // Stamp which text backend produced `plutchik`, so the UI can badge it.
        // Nil when text SER was skipped (filler / empty / no model);
        // sentinel string when Apple FM declined via its safety guardrail.
        let textBackend: String? = await resolveTextBackend(for: textResult)
        let estimate = baseEstimate
            .withTextBackend(textBackend)
            .withAgeGender(ageGender)
        let metrics = ProcessingMetrics(
            acousticDuration: dim != nil || cat != nil ? acousticDuration : nil,
            textDuration: txt != nil ? textDuration : nil,
            totalDuration: Date().timeIntervalSince(totalStart)
        )
        return (estimate, metrics)
    }

    private func resolveTextBackend(for result: TextSEROutcome) async -> String? {
        if result.guardrailViolation {
            return SwitchingTextSER.foundationModelsGuardrailBackend
        }
        guard result.score != nil else { return nil }
        return await (textSER as? SwitchingTextSER)?.currentBackend.rawValue
    }

    /// Text-only re-analysis path for hand-edits on a session
    /// that has no source audio (a still-running or just-stopped
    /// mic-mode session, or an imported mic-mode `.xph`). We
    /// can't re-slice audio or re-run acoustic SER, so the
    /// dimensional V/A/D and acoustic categorical scores get
    /// **inherited** from the parent utterance unchanged; only
    /// the text-SER + late-fusion step re-runs against the new
    /// transcript. The resulting estimate carries the new
    /// transcript + new plutchik + new fused V/A/D + new fused
    /// top label; everything else is taken from `original`.
    ///
    /// Used by `RecordingController.commitHandEdit` when
    /// `playbackSourceURL == nil`. Skipped on the file-mode path
    /// which still routes through `processSegment`.
    func reanalyzeTextOnly(
        text: String,
        inheriting original: UtteranceEstimate
    ) async throws -> UtteranceEstimate {
        let textResult = await runText(text)
        let plutchik = textResult.score
        let textBackend: String? = await resolveTextBackend(for: textResult)
        let asr = ASRSegment(
            text: text,
            start: original.start,
            end: original.end,
            // User-verified transcript; trust it at full
            // confidence for fusion weighting.
            confidence: 1.0,
            tokens: []
        )
        let baseEstimate = try await fuser.fuse(
            asr: asr,
            speakerID: original.speakerID,
            dimensional: original.dimensional,
            acousticCategorical: original.acousticCategorical,
            plutchik: plutchik
        )
        // Inherit age-gender from the parent: it's audio-derived and
        // the audio didn't change on the mic-mode hand-edit path,
        // same reason we inherit dimensional + categorical above.
        return baseEstimate
            .withTextBackend(textBackend)
            .withAgeGender(original.ageGender)
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

    private func runAgeGender(_ buffer: AudioChunk) async -> AgeGenderEstimate? {
        guard let ageGenderSER else { return nil }
        let capped = Self.capForSER(buffer)
        do { return try await ageGenderSER.estimate(capped) } catch {
            AppLog.app.debug("age-gender SER skipped: \(String(describing: error), privacy: .public)")
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
    ///      Jetsam-style OOM kills around the 15 min mark of non-realtime
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

    private func runText(_ text: String) async -> TextSEROutcome {
        guard let textSER else {
            AppLog.serText.warning("text SER skipped: no backend wired")
            return .empty
        }
        guard !text.isEmpty else { return .empty }
        if Self.isFiller(text) {
            AppLog.serText.debug("text SER skipped (filler): \(text, privacy: .public)")
            return .empty
        }
        do {
            let score = try await textSER.classify(text)
            return TextSEROutcome(score: score, guardrailViolation: false)
        } catch TextSERError.guardrailViolation {
            AppLog.serText.info("text SER declined (Apple FM guardrail): \(text, privacy: .public)")
            return TextSEROutcome(score: nil, guardrailViolation: true)
        } catch {
            // Promoted to .warning so non-guardrail failures aren't
            // silently swallowed — debugging "text SER never runs"
            // takes priority over log noise here.
            AppLog.serText.warning("text SER threw: \(String(describing: error), privacy: .public) — text=\(text, privacy: .public)")
            return .empty
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

        let dt: TimeInterval = Self.speakerVoteSampleStepSec
        let sampleCount = min(
            Self.speakerVoteSampleCountMax,
            max(Self.speakerVoteSampleCountMin, Int(((end - start) / dt).rounded(.up)))
        )
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
        let startIndex = Int(start * buffer.sampleRate).clamped(to: 0...buffer.samples.count)
        let endIndex = Int(end * buffer.sampleRate).clamped(to: startIndex...buffer.samples.count)
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
