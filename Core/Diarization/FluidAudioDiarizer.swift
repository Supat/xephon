import Foundation
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

    public init(kind: FluidDiarizerKind = .sortformer, config: DiarizerConfig = .default) {
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

    private func loadModels() async throws {
        AppLog.diarization.info("Downloading FluidAudio diarizer models (first run)…")
        do {
            let models = try await DiarizerModels.download()
            manager.initialize(models: models)
        } catch {
            throw DiarizationError.modelUnavailable(reason: String(describing: error))
        }
    }
}
