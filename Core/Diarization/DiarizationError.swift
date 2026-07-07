import Foundation

public enum DiarizationError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case underlying(any Error)
}
