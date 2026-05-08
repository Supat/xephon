import Foundation

public enum DiarizationError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case tooManySpeakers(limit: Int)
    case underlying(any Error)
}
