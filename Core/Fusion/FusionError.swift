import Foundation

public enum FusionError: Error, Sendable {
    case notImplemented
    case missingModality(name: String)
    case calibrationUnavailable
    case underlying(any Error)
}
