import Foundation
import XephonLogging
import Fusion

public actor JSONExporter {
    private let encoder: JSONEncoder

    public init() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    /// Encodes utterances to a JSON array on disk.
    public func write(_ utterances: [UtteranceEstimate], to url: URL) async throws {
        do {
            let data = try encoder.encode(utterances)
            try data.write(to: url, options: [.atomic])
            AppLog.export.info(
                "Wrote \(utterances.count, privacy: .public) utterances → \(url.lastPathComponent, privacy: .public)"
            )
        } catch let error as ExportError {
            throw error
        } catch {
            throw ExportError.ioFailure(reason: String(describing: error))
        }
    }

    /// Encodes utterances to an in-memory `Data` blob — useful for share sheets.
    public func encode(_ utterances: [UtteranceEstimate]) async throws -> Data {
        do {
            return try encoder.encode(utterances)
        } catch {
            throw ExportError.ioFailure(reason: String(describing: error))
        }
    }
}
