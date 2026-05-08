import Foundation

public enum ExportError: Error, Sendable {
    case notImplemented
    case ioFailure(reason: String)
    case underlying(any Error)
}
