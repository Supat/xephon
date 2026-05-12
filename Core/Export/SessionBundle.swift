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

    /// User-supplied display names keyed by stored speaker id
    /// (e.g. `"S01" → "Alice"`). Optional so v1 bundles without
    /// this field continue to decode cleanly — Swift's Codable
    /// treats a missing key on an optional as nil rather than a
    /// decode error, so no `formatVersion` bump is needed.
    public let speakerNames: [String: String]?

    /// Opaque blob carrying the diarizer's session-wide speaker
    /// database at save time (e.g. FluidAudio's `[Speaker]` JSON,
    /// embeddings included). Decoded by the same diarizer type on
    /// load so re-diarizing a hand-edited slice resolves to the
    /// same session-stable IDs the original recording assigned —
    /// instead of clustering from scratch on an empty DB. Optional
    /// because (a) v1 bundles predate this field and (b) mic-mode
    /// sessions that never engaged the diarizer have nothing to
    /// persist. Encoding is the diarizer's choice and intentionally
    /// not interpreted at this layer; a future diarizer swap is
    /// free to write a different format here.
    public let speakerDatabase: Data?

    /// Pre-first-reeval / pre-hand-edit snapshots, keyed by the
    /// utterance id whose row was edited. Restored into the
    /// controller's in-memory `preReevaluationSnapshots` map on
    /// load so the long-press-to-revert affordance survives Save.
    /// Optional + missing-key tolerant; v1 bundles read back as
    /// nil and the revert path no-ops as it did before.
    public let originalSnapshots: [UUID: UtteranceEstimate]?

    /// Sibling-row ids created when a multi-sentence hand-edit
    /// split one utterance into several. Keyed by the parent's id
    /// (the row that retains the original id post-split). Restored
    /// into `handEditChildren` on load so reverting the parent
    /// also removes the siblings — without this, a Save/Load cycle
    /// would leave the siblings as orphans after a post-load
    /// revert. Optional for v1 compat.
    public let handEditChildren: [UUID: [UUID]]?

    /// Opaque blob carrying the diarizer's cumulative timeline at
    /// save time (JSON-encoded `[DiarizedSegment]`). Decoded by the
    /// controller on load so the per-session diarizer-timeline
    /// strip in the transcript pane survives Save → Open. Kept as
    /// `Data?` rather than a typed field so this layer doesn't
    /// take a direct dependency on the diarizer module's segment
    /// type. Optional for v1 compat.
    public let diarizationTimeline: Data?

    /// Opaque blob carrying the cached LLM session summary at save
    /// time (JSON-encoded `SessionSummary` from the Summarizer
    /// module). Decoded by the controller on load so the user
    /// doesn't have to re-run the multi-second summarize pass just
    /// to re-read what they already generated. Same `Data?` trick
    /// used for the diarizer fields — keeps Export free of an
    /// upward dependency on Summarizer (which would drag MLX into
    /// the export module). Optional for v1 compat.
    public let sessionSummary: Data?

    public enum SourceKind: String, Codable, Sendable {
        case microphone, file
    }

    /// Current schema version. Bumped here when the layout changes;
    /// readers compare against this to accept the file. Adding a
    /// new optional field doesn't qualify as an incompatible
    /// change.
    public static let currentFormatVersion: Int = 1

    public init(
        formatVersion: Int = SessionDocument.currentFormatVersion,
        createdAt: Date = Date(),
        sourceKind: SourceKind,
        audioFilename: String?,
        audio: Data?,
        utterances: [UtteranceEstimate],
        speakerNames: [String: String]? = nil,
        speakerDatabase: Data? = nil,
        originalSnapshots: [UUID: UtteranceEstimate]? = nil,
        handEditChildren: [UUID: [UUID]]? = nil,
        diarizationTimeline: Data? = nil,
        sessionSummary: Data? = nil
    ) {
        self.formatVersion = formatVersion
        self.createdAt = createdAt
        self.sourceKind = sourceKind
        self.audioFilename = audioFilename
        self.audio = audio
        self.utterances = utterances
        self.speakerNames = speakerNames
        self.speakerDatabase = speakerDatabase
        self.originalSnapshots = originalSnapshots
        self.handEditChildren = handEditChildren
        self.diarizationTimeline = diarizationTimeline
        self.sessionSummary = sessionSummary
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
