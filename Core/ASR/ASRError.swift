import Foundation

public enum ASRError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case unsupportedLocale(String)
    case audioFormatMismatch(expected: String, got: String)
    case neuralEngineUnavailable
    case underlying(any Error)
}
