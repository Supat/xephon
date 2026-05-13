import Foundation
import Audio

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

/// Live snapshot of the diarizer's internal speaker cluster — the
/// per-id averaged centroid plus every raw observation embedding
/// retained for that speaker. Surfaced for the cluster-visualization
/// panels (pairwise cosine-distance heatmap + 2D PCA scatter); these
/// reads happen at ~1 Hz so the snapshot is value-typed and self-
/// contained rather than handing out actor-internal references.
public struct SpeakerClusterSnapshot: Sendable {
    public struct Speaker: Sendable {
        /// Display-format id (`S01`, `S02`, …) — already bridged from
        /// FluidAudio's raw numeric ids so consumers don't need to
        /// reformat.
        public let id: String
        /// L2-normalized averaged embedding (256-D for the FluidAudio
        /// extractor).
        public let centroid: [Float]
        /// Raw per-observation embeddings, capped to the most recent
        /// few dozen at the source so PCA over the cloud stays cheap.
        public let observations: [[Float]]

        public init(id: String, centroid: [Float], observations: [[Float]]) {
            self.id = id
            self.centroid = centroid
            self.observations = observations
        }
    }
    public let speakers: [Speaker]

    public init(speakers: [Speaker]) {
        self.speakers = speakers
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
    /// Snapshot of the diarizer's current speaker cluster — every
    /// known speaker's averaged centroid plus a cap-bounded slice of
    /// raw observations. Read by the cluster-visualization panels at
    /// ~1 Hz; cheap enough to call inline because the implementation
    /// just hands back already-resident `[Float]` arrays.
    func clusterSnapshot(maxObservationsPerSpeaker: Int) async -> SpeakerClusterSnapshot
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
    func clusterSnapshot(maxObservationsPerSpeaker: Int) async -> SpeakerClusterSnapshot {
        SpeakerClusterSnapshot(speakers: [])
    }
}
