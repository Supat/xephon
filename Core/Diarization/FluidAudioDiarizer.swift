import Foundation
import CoreML
import FluidAudio
import Audio
import XephonLogging

// FluidAudio Sortformer (≤4 speakers, very stable) or LS-EEND (≤10, lighter), ANE-targeted.
// Models are downloaded by FluidAudio on first use into its cache directory.
public enum FluidDiarizerKind: Sendable {
    case sortformer
    case lseend
}

public actor FluidAudioDiarizer: Diarizer {
    private let kind: FluidDiarizerKind
    // FluidAudio's DiarizerManager isn't formally Sendable; it's an internal
    // class with its own thread safety. Mark unsafe so Swift 6 lets us hold it
    // inside an actor and call its async methods.
    private nonisolated(unsafe) let manager: DiarizerManager

    /// Default config tuned for conversational speech. The two
    /// values that differ from `DiarizerConfig.default` are:
    ///
    /// - `clusteringThreshold = 0.6` (default 0.7). Lower value =
    ///   stricter matching, so similar-but-distinct voices are more
    ///   likely to get separate IDs. FluidAudio derives
    ///   `speakerThreshold = clusteringThreshold × 1.2` from this
    ///   and `embeddingThreshold = clusteringThreshold × 0.8`.
    ///   Progression on this codebase has been 0.78 → 0.72 → 0.6,
    ///   each step trading more separation for more observation
    ///   noise. The mode-vote in `dominantSpeaker` (per-instant
    ///   majority across overlapping observations) absorbs the
    ///   noise; what we get back is real speakers that the looser
    ///   thresholds were merging into one ID.
    ///
    /// - `minSpeechDuration = 0.5` s (FluidAudio default 1.0). Lets
    ///   a real new speaker who only says a short turn (a brief
    ///   backchannel, a one-word interjection) first still get
    ///   their own ID. The per-instant majority vote in
    ///   `dominantSpeaker` filters out spurious one-window
    ///   creations from background noise, so the cost of a lower
    ///   floor is small.
    ///
    /// Other fields stay at FluidAudio's defaults.
    public static let conversationalConfig = DiarizerConfig(
        clusteringThreshold: 0.6,
        minSpeechDuration: 0.5
    )

    public init(
        kind: FluidDiarizerKind = .sortformer,
        config: DiarizerConfig = FluidAudioDiarizer.conversationalConfig
    ) {
        self.kind = kind
        self.manager = DiarizerManager(config: config)
    }

    public func diarize(_ buffer: AudioChunk) async throws -> [DiarizedSegment] {
        if !manager.isAvailable {
            try await loadModels()
        }
        let result = try await manager.performCompleteDiarization(
            buffer.samples,
            sampleRate: Int(buffer.sampleRate),
            atTime: buffer.timestamp
        )
        AppLog.diarization.info(
            "Diarized \(result.segments.count, privacy: .public) segments (kind=\(String(describing: self.kind), privacy: .public))"
        )
        return result.segments.map {
            DiarizedSegment(
                speakerID: Self.formatGlobalID($0.speakerId),
                start: TimeInterval($0.startTimeSeconds),
                end: TimeInterval($0.endTimeSeconds)
            )
        }
    }

    /// Reset FluidAudio's session-wide speaker database. Without this,
    /// speakers from a prior recording carry over — the next session's
    /// "S01" might unify with a previous "S01" purely because the
    /// embedding centroids happen to be close, conflating two
    /// unrelated people.
    public func resetSpeakers() async {
        guard manager.isAvailable else { return }
        await manager.speakerManager.reset()
        AppLog.diarization.info("FluidAudio speaker database reset")
    }

    /// Extract a 256-dimensional L2-normalized speaker embedding for
    /// `audio`. Returns nil before models load. Callers should pass
    /// audio of a single speaker for meaningful results — the
    /// embedding extractor uses an all-ones mask, so mixed-speaker
    /// input averages embeddings together.
    public func embedding(for audio: [Float]) async throws -> [Float]? {
        guard manager.isAvailable else { return nil }
        return try manager.extractSpeakerEmbedding(from: audio)
    }

    /// Look up the closest known speaker in FluidAudio's database via
    /// cosine similarity. Returns nil before models load.
    public func findSpeaker(byEmbedding embedding: [Float]) async -> SpeakerMatch? {
        guard manager.isAvailable else { return nil }
        let result = await manager.speakerManager.findSpeaker(with: embedding)
        let mapped = result.id.map { Self.formatGlobalID($0) }
        return SpeakerMatch(id: mapped, distance: result.distance)
    }

    /// Bridge FluidAudio's numeric speaker IDs ("1", "2", …) to our
    /// display convention ("S01", "S02", …). Keeps everything
    /// downstream — UI labels, exports, JSON schema — agnostic of
    /// the upstream library's representation.
    private static func formatGlobalID(_ raw: String) -> String {
        if let n = Int(raw) {
            return String(format: "S%02d", n)
        }
        return raw
    }

    /// JSON-encode the FluidAudio `SpeakerManager`'s current speaker
    /// database (the per-id `Speaker` records with their averaged
    /// 256-D embeddings and accumulated raw observations). Returns
    /// nil when models haven't loaded yet, when the database is
    /// empty, or when encoding fails. Each `Speaker` is ~few KB, so
    /// a realistic conversational session (≤4 speakers) costs <20 KB
    /// — negligible against the embedded audio bytes.
    public func exportSpeakerDatabase() async -> Data? {
        guard manager.isAvailable else { return nil }
        let speakers = await manager.speakerManager.getSpeakerList()
        guard !speakers.isEmpty else { return nil }
        do {
            return try JSONEncoder().encode(speakers)
        } catch {
            AppLog.diarization.warning(
                "speaker DB export failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Decode a previously-exported speaker database blob and load
    /// it into the FluidAudio `SpeakerManager`, replacing any
    /// in-memory state. Loads diarizer models on demand because
    /// callers may want to import a saved DB before any audio has
    /// flowed through `diarize`. Uses `.reset` initialization mode
    /// so a partial-overlap conflict doesn't silently keep stale
    /// embeddings from a prior session.
    public func importSpeakerDatabase(_ data: Data) async throws {
        if !manager.isAvailable {
            try await loadModels()
        }
        let speakers = try JSONDecoder().decode([Speaker].self, from: data)
        await manager.speakerManager.initializeKnownSpeakers(speakers, mode: .reset)
        AppLog.diarization.info(
            "FluidAudio speaker DB restored: \(speakers.count, privacy: .public) speakers"
        )
    }

    /// Register `id` in FluidAudio's session-wide `SpeakerManager`
    /// with the supplied embedding so subsequent `diarize` passes
    /// can match similar audio to this entry. Speakers are added
    /// with `isPermanent: true` so FluidAudio's inactive-pruning
    /// can't drop a user-promoted voice. Skips silently when the
    /// id already exists (caller is responsible for fresh ids).
    public func promoteSpeaker(id: String, embedding: [Float]) async throws {
        if !manager.isAvailable {
            try await loadModels()
        }
        let speaker = Speaker(
            id: id,
            currentEmbedding: embedding,
            isPermanent: true
        )
        await manager.speakerManager.initializeKnownSpeakers(
            [speaker],
            mode: .skip,
            preserveIfPermanent: true
        )
        AppLog.diarization.info(
            "speaker promoted: id=\(id, privacy: .public) (embedding dim=\(embedding.count, privacy: .public))"
        )
    }

    /// EMA-blend a new observation into an existing speaker's
    /// centroid. Uses FluidAudio's mutating `updateMainEmbedding`,
    /// then writes the updated record back via `upsertSpeaker`.
    /// Silently no-ops when the id isn't already in the database
    /// (caller should validate first).
    public func correctSpeaker(id: String, embedding: [Float], duration: Float) async throws {
        if !manager.isAvailable {
            try await loadModels()
        }
        guard var speaker = await manager.speakerManager.getSpeaker(for: id) else {
            AppLog.diarization.warning(
                "correctSpeaker: id \(id, privacy: .public) not in database — no-op"
            )
            return
        }
        speaker.updateMainEmbedding(
            duration: duration,
            embedding: embedding,
            segmentId: UUID()
        )
        await manager.speakerManager.upsertSpeaker(speaker)
        AppLog.diarization.info(
            "speaker corrected: id=\(id, privacy: .public) (duration=\(duration, privacy: .public)s, embedding dim=\(embedding.count, privacy: .public))"
        )
    }

    /// Forward to FluidAudio's `SpeakerManager.removeSpeaker(_:,
    /// keepIfPermanent:)`. Silent no-op when the id isn't in the
    /// DB or models haven't loaded yet (auto-demote shouldn't
    /// fail in either case).
    public func removeSpeakerFromDB(id: String, keepIfPermanent: Bool) async throws {
        guard manager.isAvailable else { return }
        await manager.speakerManager.removeSpeaker(id, keepIfPermanent: keepIfPermanent)
        AppLog.diarization.info(
            "speaker removed from DB: id=\(id, privacy: .public) (keepIfPermanent=\(keepIfPermanent, privacy: .public))"
        )
    }

    private func loadModels() async throws {
        AppLog.diarization.info("Downloading FluidAudio diarizer models (first run)…")
        do {
            // Pin to `.cpuAndNeuralEngine` so the diarizer never
            // submits Metal work. iOS revokes background GPU access
            // even with the `audio` background mode declared
            // (E5RT / kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted),
            // which crashed the continuous-diarize loop the moment
            // the user switched to another app mid-recording. The
            // ANE doesn't share that restriction and runs the
            // Sortformer + embedding models comfortably; in
            // foreground there's no measurable latency difference
            // either, so this is a strict win.
            let configuration = MLModelConfiguration()
            configuration.computeUnits = .cpuAndNeuralEngine
            let models = try await DiarizerModels.download(
                configuration: configuration
            )
            manager.initialize(models: models)
        } catch {
            throw DiarizationError.modelUnavailable(reason: String(describing: error))
        }
    }
}
