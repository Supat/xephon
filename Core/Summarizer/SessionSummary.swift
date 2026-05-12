import Foundation

/// Structured output produced by `SessionSummarizer.summarize(_:)`.
/// Codable so it round-trips through `JSONExporter` alongside the
/// per-utterance estimates — external tooling can read the human
/// summary next to the raw affect data.
///
/// Fields are intentionally a flat shape: a topical paragraph, an
/// overall-mood string, and a per-speaker list. The LLM fills them
/// via JSON-schema-constrained generation; the schema lives on the
/// `Generable` wrapper in the MLX-backed summarizer implementation
/// (kept out of this type so the data shape can be consumed by
/// non-MLX tests / mocks without dragging the runtime in).
public struct SessionSummary: Sendable, Hashable, Codable {
    public struct SpeakerSummary: Sendable, Hashable, Codable {
        public let speakerID: String
        /// Optional rename (`speakerNameOverrides[speakerID]`).
        public let speakerName: String?
        /// One-paragraph emotional arc for this speaker — what they
        /// were saying, how their mood evolved, anything notable
        /// about their stance vs. other speakers.
        public let summary: String
        /// The single emotion label the speaker spent the most time
        /// in across the session (or a short phrase like
        /// "predominantly neutral with a sad turn at the end").
        public let dominantMood: String

        public init(
            speakerID: String,
            speakerName: String?,
            summary: String,
            dominantMood: String
        ) {
            self.speakerID = speakerID
            self.speakerName = speakerName
            self.summary = summary
            self.dominantMood = dominantMood
        }
    }

    /// One- or two-sentence topical summary — "what is the
    /// conversation about." Independent of mood / affect.
    public let topic: String
    /// One-paragraph overall-mood description for the session as
    /// a whole, factoring in the V/A/D and fused top labels across
    /// every utterance.
    public let overallMood: String
    /// Per-speaker arcs, one entry per distinct `speakerID` in the
    /// input. Order matches the input's first-appearance ordering.
    public let perSpeaker: [SpeakerSummary]
    /// Identifier for the model that produced this summary
    /// (e.g. `"qwen2.5-7b-instruct-4bit"`). Persisted so a later
    /// re-summarize knows whether the existing summary is stale
    /// against a newer model, and so external consumers can
    /// attribute the output.
    public let model: String
    /// Wall-clock time the summary was generated. ISO-8601 in JSON.
    public let generatedAt: Date

    public init(
        topic: String,
        overallMood: String,
        perSpeaker: [SpeakerSummary],
        model: String,
        generatedAt: Date
    ) {
        self.topic = topic
        self.overallMood = overallMood
        self.perSpeaker = perSpeaker
        self.model = model
        self.generatedAt = generatedAt
    }
}
