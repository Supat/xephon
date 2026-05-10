import Foundation
import Audio

public struct ASRSegment: Sendable, Hashable, Codable {
    /// Per-run timing inside the segment. Populated when the underlying
    /// transcriber exposes per-token / per-character `audioTimeRange`
    /// (Apple SpeechTranscriber does; WhisperKit currently doesn't).
    /// Empty when unavailable — downstream code (e.g. speaker-change
    /// splitting) falls back to whole-segment behavior.
    public struct Token: Sendable, Hashable, Codable {
        public let text: String
        public let start: TimeInterval
        public let end: TimeInterval

        public init(text: String, start: TimeInterval, end: TimeInterval) {
            self.text = text
            self.start = start
            self.end = end
        }
    }

    public let text: String
    public let start: TimeInterval
    public let end: TimeInterval
    // 0...1; propagated into fusion weights per CLAUDE.md.
    public let confidence: Float?
    public let tokens: [Token]

    public init(
        text: String,
        start: TimeInterval,
        end: TimeInterval,
        confidence: Float?,
        tokens: [Token] = []
    ) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
        self.tokens = tokens
    }
}

public protocol Transcriber: Actor {
    var locale: Locale { get }
    func transcribe(_ buffer: AudioChunk) async throws -> [ASRSegment]
}
