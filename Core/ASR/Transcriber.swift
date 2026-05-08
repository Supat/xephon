import Foundation
import Audio

public struct ASRSegment: Sendable, Hashable, Codable {
    public let text: String
    public let start: TimeInterval
    public let end: TimeInterval
    // 0...1; propagated into fusion weights per CLAUDE.md.
    public let confidence: Float?

    public init(text: String, start: TimeInterval, end: TimeInterval, confidence: Float?) {
        self.text = text
        self.start = start
        self.end = end
        self.confidence = confidence
    }
}

public protocol Transcriber: Actor {
    var locale: Locale { get }
    func transcribe(_ buffer: AudioChunk) async throws -> [ASRSegment]
}
