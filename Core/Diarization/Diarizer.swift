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

public protocol Diarizer: Actor {
    func diarize(_ buffer: AudioChunk) async throws -> [DiarizedSegment]
}
