import Foundation
import Audio

public struct DiarizedSegment: Sendable, Hashable {
    public let speakerID: String
    public let start: TimeInterval
    public let end: TimeInterval

    public init(speakerID: String, start: TimeInterval, end: TimeInterval) {
        self.speakerID = speakerID
        self.start = start
        self.end = end
    }
}

/// Result of an embedding-based speaker lookup. `id` is nil when the
/// closest known speaker is farther than the diarizer's similarity
/// threshold (i.e. the audio doesn't match anyone we've seen yet);
/// `distance` is still reported so callers can reason about
/// confidence regardless. Cosine distance, in [0, 2].
public struct SpeakerMatch: Sendable {
    public let id: String?
    public let distance: Float

    public init(id: String?, distance: Float) {
        self.id = id
        self.distance = distance
    }
}

public protocol Diarizer: Actor {
    func diarize(_ buffer: AudioChunk) async throws -> [DiarizedSegment]
    /// Reset the diarizer's session-wide speaker database. Called at
    /// the start of each recording so speaker IDs from the previous
    /// session don't carry over and confuse the new one.
    func resetSpeakers() async
    /// Extract a session-stable speaker embedding from a clip of
    /// audio. Used for per-sub-segment refinement on top of
    /// time-based assignment. Returns nil for diarizers that don't
    /// expose an embedding API (the default protocol extension does).
    func embedding(for audio: [Float]) async throws -> [Float]?
    /// Look up the closest known speaker to `embedding` in the
    /// diarizer's session-wide database. Returns nil if the
    /// diarizer doesn't expose this. The returned `id` is the
    /// session-stable speaker ID in the same format `diarize`
    /// produces; `distance` is cosine distance to that speaker.
    func findSpeaker(byEmbedding embedding: [Float]) async -> SpeakerMatch?
}

public extension Diarizer {
    func resetSpeakers() async {}
    func embedding(for audio: [Float]) async throws -> [Float]? { nil }
    func findSpeaker(byEmbedding embedding: [Float]) async -> SpeakerMatch? { nil }
}
