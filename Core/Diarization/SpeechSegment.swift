import Foundation

/// A frame-level "speech is present here" interval, output by VAD.
/// Distinct from `DiarizedSegment` — VAD only answers "is this
/// speech?" and not "whose speech?" — so this type intentionally
/// carries no speaker ID. The combination of (per-speaker
/// `DiarizedSegment`s) AND-ed with (speech-only `SpeechSegment`s)
/// is what the live-mode acoustic-SER trim needs: the diarizer's
/// segment boundaries are aggregated to speaker-track granularity
/// and absorb sub-second silences inside a speaker's region; the
/// VAD's boundaries are pre-aggregation and catch those silences.
public struct SpeechSegment: Sendable, Hashable, Codable {
    public let start: TimeInterval
    public let end: TimeInterval

    public init(start: TimeInterval, end: TimeInterval) {
        self.start = start
        self.end = end
    }
}
