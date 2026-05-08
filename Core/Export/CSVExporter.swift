import Foundation
import XephonLogging
import Fusion

public actor CSVExporter {
    public init() {}

    public func write(_ utterances: [UtteranceEstimate], to url: URL) async throws {
        AppLog.export.debug("CSVExporter.write stub (\(utterances.count) utterances → \(url.path))")
        throw ExportError.notImplemented
    }
}
