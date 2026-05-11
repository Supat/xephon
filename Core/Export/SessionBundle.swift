import Foundation
import Fusion

/// Single-file save/load format for a Xephon analysis session.
///
/// One `.xph` file contains the full utterance list, a manifest with
/// session metadata, and (when the source was a file analysis) the
/// raw audio bytes inlined so per-utterance playback round-trips
/// without needing the original media. Mic-mode sessions omit the
/// audio block — the schema doc explicitly does not promise playback
/// support for live recordings.
///
/// Serialized as a binary property list because:
///   - it's a single self-contained file (no zip dependency)
///   - `PropertyListEncoder` already understands `Data`, so the audio
///     bytes get embedded without a custom container
///   - everything else is `Codable` so the manifest + utterances are
///     just struct serialization
public struct SessionDocument: Codable, Sendable {
    /// Format generation counter. Bump on incompatible schema changes.
    /// The reader rejects unknown versions rather than guessing.
    public let formatVersion: Int

    /// Local wall-clock time when the bundle was written. Surfaces
    /// in any future "Recent" UI; not used for ordering or anything
    /// load-bearing.
    public let createdAt: Date

    /// Source modality the session was captured under. `.microphone`
    /// means the bundle never contained `audio`; `.file` means it
    /// did (and `audioFilename` carries the original name).
    public let sourceKind: SourceKind

    /// Original file's `lastPathComponent`, kept so the extracted
    /// temp file can recover the right extension/container hint for
    /// AVFoundation. Nil for mic sessions.
    public let audioFilename: String?

    /// Embedded audio bytes — present only for file-mode sessions.
    /// Loaded eagerly into memory on import; for typical research
    /// clips (≤30 min MP3, ~30 MB) this is fine on M-class iPads.
    public let audio: Data?

    /// The exported utterances. Same Codable representation
    /// `JSONExporter` produces, so a `.xph` always implies an
    /// equivalent `.json` could be extracted.
    public let utterances: [UtteranceEstimate]

    public enum SourceKind: String, Codable, Sendable {
        case microphone, file
    }

    /// Current schema version. Bumped here when the layout changes;
    /// readers compare against this to decide whether to accept the
    /// file.
    public static let currentFormatVersion: Int = 1

    public init(
        formatVersion: Int = SessionDocument.currentFormatVersion,
        createdAt: Date = Date(),
        sourceKind: SourceKind,
        audioFilename: String?,
        audio: Data?,
        utterances: [UtteranceEstimate]
    ) {
        self.formatVersion = formatVersion
        self.createdAt = createdAt
        self.sourceKind = sourceKind
        self.audioFilename = audioFilename
        self.audio = audio
        self.utterances = utterances
    }
}

/// Read/write helpers for the binary plist representation.
public enum SessionBundle {
    public enum BundleError: Error, Sendable, CustomStringConvertible {
        case unsupportedFormatVersion(Int)
        case decodeFailed(String)
        case encodeFailed(String)
        case ioFailure(String)

        public var description: String {
            switch self {
            case .unsupportedFormatVersion(let v):
                return "Unsupported session format version \(v). Update the app."
            case .decodeFailed(let r):
                return "Couldn't read session: \(r)"
            case .encodeFailed(let r):
                return "Couldn't write session: \(r)"
            case .ioFailure(let r):
                return "Session I/O failed: \(r)"
            }
        }
    }

    /// Serialize `document` to a binary plist blob.
    public static func encode(_ document: SessionDocument) throws -> Data {
        let encoder = PropertyListEncoder()
        encoder.outputFormat = .binary
        do {
            return try encoder.encode(document)
        } catch {
            throw BundleError.encodeFailed(String(describing: error))
        }
    }

    /// Parse `data` back into a `SessionDocument`. Rejects future
    /// `formatVersion` values rather than risking a partial decode.
    public static func decode(_ data: Data) throws -> SessionDocument {
        let decoder = PropertyListDecoder()
        let document: SessionDocument
        do {
            document = try decoder.decode(SessionDocument.self, from: data)
        } catch {
            throw BundleError.decodeFailed(String(describing: error))
        }
        guard document.formatVersion == SessionDocument.currentFormatVersion else {
            throw BundleError.unsupportedFormatVersion(document.formatVersion)
        }
        return document
    }
}
