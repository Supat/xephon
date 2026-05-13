import Foundation
import Audio

/// Knobs the controller / UI can tune on a diarizer mid-session.
/// Deliberately narrower than FluidAudio's full `DiarizerConfig` —
/// only the two thresholds the user has a mental model for and that
/// produce visibly different speaker assignments. Adding a third
/// knob here forces a conscious choice rather than spraying every
/// FluidAudio field into the UI.
///
/// Kept inside the Diarization module so the Xephon target and the
/// pipeline don't have to take a hard dependency on FluidAudio just
/// to talk about tuning.
public struct DiarizationTuning: Sendable, Hashable {
    /// Clustering threshold for speaker embeddings. Lower values
    /// split more aggressively (more speakers); higher values
    /// merge similar voices into one ID. Reasonable range 0.5–0.9.
    public var clusteringThreshold: Float
    /// Minimum speech segment duration in seconds. Shorter segments
    /// are discarded by the diarizer rather than creating a fresh
    /// speaker. Reasonable range 0.2–2.0.
    public var minSpeechDuration: Float

    public init(clusteringThreshold: Float, minSpeechDuration: Float) {
        self.clusteringThreshold = clusteringThreshold
        self.minSpeechDuration = minSpeechDuration
    }
}

public struct DiarizedSegment: Sendable, Hashable, Codable {
    public let speakerID: String
    public let start: TimeInterval
    public let end: TimeInterval

    public init(speakerID: String, start: TimeInterval, end: TimeInterval) {
        self.speakerID = speakerID
        self.start = start
        self.end = end
    }
}

/// Result of an embedding-based speaker lookup. `id` is nil when the
/// closest known speaker is farther than the diarizer's similarity
/// threshold (i.e. the audio doesn't match anyone we've seen yet);
/// `distance` is still reported so callers can reason about
/// confidence regardless. Cosine distance, in [0, 2].
public struct SpeakerMatch: Sendable {
    public let id: String?
    public let distance: Float

    public init(id: String?, distance: Float) {
        self.id = id
        self.distance = distance
    }
}

public protocol Diarizer: Actor {
    func diarize(_ buffer: AudioChunk) async throws -> [DiarizedSegment]
    /// Reset the diarizer's session-wide speaker database. Called at
    /// the start of each recording so speaker IDs from the previous
    /// session don't carry over and confuse the new one.
    func resetSpeakers() async
    /// Extract a session-stable speaker embedding from a clip of
    /// audio. Used for per-sub-segment refinement on top of
    /// time-based assignment. Returns nil for diarizers that don't
    /// expose an embedding API (the default protocol extension does).
    func embedding(for audio: [Float]) async throws -> [Float]?
    /// Look up the closest known speaker to `embedding` in the
    /// diarizer's session-wide database. Returns nil if the
    /// diarizer doesn't expose this. The returned `id` is the
    /// session-stable speaker ID in the same format `diarize`
    /// produces; `distance` is cosine distance to that speaker.
    func findSpeaker(byEmbedding embedding: [Float]) async -> SpeakerMatch?
    /// Serialize the diarizer's session-wide speaker database to an
    /// opaque blob suitable for stashing in a `.xph` session file.
    /// Returns nil when the diarizer has nothing to persist (e.g.
    /// models not yet loaded, no speakers seen, or the encoding
    /// failed). Specific encoding is the diarizer's choice; the
    /// blob is treated as opaque by callers.
    func exportSpeakerDatabase() async -> Data?
    /// Restore a previously-exported speaker database. Treat `data`
    /// as authoritative and replace any in-memory state. Throws on
    /// malformed input; callers should swallow throws and log
    /// (loading a session shouldn't fail just because the diarizer
    /// state went stale across an app version).
    func importSpeakerDatabase(_ data: Data) async throws
    /// Register a speaker with the supplied id and embedding in the
    /// diarizer's session-wide database, so future `diarize` calls
    /// can match similar audio to this entry. Used by the
    /// "Promote New Speaker" action in the speaker chip popover:
    /// take a user-vetted utterance's audio embedding and teach
    /// the diarizer to recognize this voice going forward.
    /// Implementations should mark the entry permanent (so it
    /// doesn't get pruned by inactivity) and silently no-op when
    /// the id already exists in the database — the controller is
    /// responsible for picking a fresh id.
    func promoteSpeaker(id: String, embedding: [Float]) async throws
    /// Fold a new observation (embedding + speech duration) into
    /// an *existing* speaker entry's centroid via EMA blending.
    /// Used by the "Teach diarizer" toggle in the chip popover:
    /// a user-corrected reassignment to an existing speaker
    /// updates that speaker's record so future `diarize` calls
    /// match similar audio to the same id. No-op when `id` isn't
    /// in the database — the caller is expected to confirm the
    /// target exists (e.g. by picking it from a list derived from
    /// `getAllSpeakers`).
    func correctSpeaker(id: String, embedding: [Float], duration: Float) async throws
    /// Remove a speaker entry from the diarizer's session-wide
    /// database. Used by the auto-demote pass that fires whenever
    /// a reassign / correct / promote / re-eval / hand-edit leaves
    /// a speaker id with no utterance referencing it. Pass
    /// `keepIfPermanent: true` (the conservative default) to skip
    /// user-promoted entries — the explicit promotion was a
    /// deliberate act and shouldn't be silently undone by an
    /// unrelated reassignment elsewhere.
    func removeSpeakerFromDB(id: String, keepIfPermanent: Bool) async throws
    /// Surface the diarizer's current tuning so the UI can render
    /// the live values. Implementations that don't expose
    /// configuration (mocks, alternate backends) can leave the
    /// default — the controller treats the values as advisory and
    /// re-reads after every `applyTuning(_:)` call.
    var currentTuning: DiarizationTuning { get async }
    /// Reconfigure the underlying model with new thresholds. The
    /// speaker DB should be preserved across the swap;
    /// implementations that can't honor that should reset cleanly.
    /// No-op default for diarizers that don't support tuning.
    func applyTuning(_ tuning: DiarizationTuning) async throws
}

public extension Diarizer {
    func resetSpeakers() async {}
    func embedding(for audio: [Float]) async throws -> [Float]? { nil }
    func findSpeaker(byEmbedding embedding: [Float]) async -> SpeakerMatch? { nil }
    func exportSpeakerDatabase() async -> Data? { nil }
    func importSpeakerDatabase(_ data: Data) async throws {}
    func promoteSpeaker(id: String, embedding: [Float]) async throws {}
    func correctSpeaker(id: String, embedding: [Float], duration: Float) async throws {}
    func removeSpeakerFromDB(id: String, keepIfPermanent: Bool) async throws {}
    var currentTuning: DiarizationTuning {
        get async { DiarizationTuning(clusteringThreshold: 0.6, minSpeechDuration: 0.5) }
    }
    func applyTuning(_ tuning: DiarizationTuning) async throws {}
}
