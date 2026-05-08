import Foundation

public enum AudioError: Error, Sendable {
    case notImplemented
    case engineUnavailable(reason: String)
    case unsupportedFormat(expected: String, got: String)
    case permissionDenied
    case underlying(any Error)
}
