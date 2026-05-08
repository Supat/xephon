import Foundation

public enum TextSERError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case foundationModelsUnavailable
    case underlying(any Error)
}
