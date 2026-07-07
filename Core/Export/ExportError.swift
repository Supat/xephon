import Foundation

public enum ExportError: Error, Sendable {
    case ioFailure(reason: String)
}
