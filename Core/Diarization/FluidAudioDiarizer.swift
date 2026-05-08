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
                speakerID: $0.speakerId,
                start: TimeInterval($0.startTimeSeconds),
                end: TimeInterval($0.endTimeSeconds)
            )
        }
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
