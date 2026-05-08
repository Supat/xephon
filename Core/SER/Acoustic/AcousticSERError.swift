import Foundation

public enum AcousticSERError: Error, Sendable {
    case notImplemented
    case modelUnavailable(reason: String)
    case onnxRuntimeFailure(reason: String)
    case underlying(any Error)
}
