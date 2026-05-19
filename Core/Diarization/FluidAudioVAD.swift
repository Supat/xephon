import Foundation
import FluidAudio
import Audio
import XephonLogging

/// Voice activity detection over 16 kHz mono PCM, wrapping
/// FluidAudio's `VadManager` (Silero-style unified model). Why a
/// separate VAD pass when the diarizer already runs on the same
/// audio? FluidAudio's diarizer uses speech segmentation internally
/// but emits its results at "speaker-track" granularity: per-frame
/// speaker labels get clustered + smoothed, so a single `S01`
/// segment can span 4 s of "speaker presence" that included a
/// 1 s mid-utterance pause. The standalone VAD skips the
/// smoothing/clustering and answers "where is speech?" at the
/// segmenter's native ~30 ms resolution — exactly the signal the
/// live-mode acoustic-SER trim needs to AND with the diarizer's
/// per-speaker regions.
///
/// Model is loaded lazily on first `segment` call (matches the
/// diarizer adapter's pattern). FluidAudio handles the download
/// into its app-support cache.
public actor FluidAudioVAD {
    private nonisolated(unsafe) var manager: VadManager?

    public init() {}

    /// Segment `audio` into speech regions. Returned `SpeechSegment`
    /// times are in **absolute audio time** — the caller's `audio`
    /// buffer carries its own `timestamp` (file-time of the first
    /// sample), and we offset VadManager's input-relative times by
    /// that so cumulative-timeline consumers can intersect VAD
    /// regions with the diarizer's absolute-time segments directly.
    public func segment(_ audio: AudioChunk) async throws -> [SpeechSegment] {
        if manager == nil {
            manager = try await VadManager()
        }
        guard let manager else { return [] }
        let raw = try await manager.segmentSpeech(audio.samples)
        let offset = audio.timestamp
        return raw.map {
            SpeechSegment(
                start: $0.startTime + offset,
                end: $0.endTime + offset
            )
        }
    }
}
