import Foundation

/// One LLM-flagged transcription issue. The reviewer ranges over an
/// existing utterance list and emits zero or more of these per row;
/// the UI surfaces them as accept/reject affordances and the accept
/// path routes through the controller's existing hand-edit pipeline
/// so SER + fusion get re-run on the corrected transcript.
///
/// Stable identity (`id`) lets the SwiftUI list track rows across
/// accept/reject churn. Locality is by `utteranceID` — the suggestion
/// is meaningless outside the context of a specific row.
public struct TranscriptionSuggestion: Sendable, Hashable, Codable, Identifiable {
    public enum Kind: String, Sendable, Hashable, Codable, CaseIterable {
        /// Likely wrong homophone or near-homophone — Japanese ASR's
        /// most common error class (橋/箸/端, 紙/神/髪, 雨/飴, …).
        case homophone
        /// Sentence doesn't follow from session context — speaker
        /// arc, topic, prior turns. Often a downstream symptom of an
        /// upstream ASR error the model could only locate via
        /// context, not via the row itself.
        case contextual
        /// Surface-level grammar / agreement / particle issue that
        /// reads as a clear ASR slip rather than a stylistic choice.
        case grammar
        /// Anything the model wanted to flag that doesn't fit the
        /// above. Kept open-ended so the prompt can stay short and
        /// the model isn't penalized for borderline cases.
        case other
    }

    public let id: UUID
    /// `UtteranceEstimate.id` this suggestion targets.
    public let utteranceID: UUID
    public let kind: Kind
    /// The transcript text as it appears on the row right now. Used
    /// for diff rendering and as a sanity check at accept time —
    /// if the row's transcript has been edited since the review ran,
    /// we can flag the staleness rather than blindly overwrite.
    public let originalText: String
    /// The model's proposed replacement. Empty string is allowed
    /// and means "the model couldn't propose a fix, just calling
    /// attention to the row" — the UI shows the reason but disables
    /// the accept button.
    public let suggestedText: String
    /// One-sentence explanation of why this is flagged. Drives the
    /// trust the user has in the suggestion — without a reason the
    /// row is just noise.
    public let reason: String
    /// 0.0–1.0 self-reported confidence. Optional because not every
    /// backend emits one; the UI can sort or filter by it when
    /// present.
    public let confidence: Float?

    public init(
        id: UUID = UUID(),
        utteranceID: UUID,
        kind: Kind,
        originalText: String,
        suggestedText: String,
        reason: String,
        confidence: Float?
    ) {
        self.id = id
        self.utteranceID = utteranceID
        self.kind = kind
        self.originalText = originalText
        self.suggestedText = suggestedText
        self.reason = reason
        self.confidence = confidence
    }
}
