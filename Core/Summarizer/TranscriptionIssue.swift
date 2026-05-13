import Foundation

/// One LLM-flagged transcription issue. The reviewer ranges over an
/// existing utterance list and emits zero or more of these per row;
/// the UI surfaces each as an "Edit" affordance that opens the same
/// hand-edit panel a long-press would raise, leaving the actual
/// correction to the user.
///
/// We deliberately don't carry a `suggestedText` field — small LLMs
/// produce unreliable proposed fixes for Japanese conversational
/// speech (echoing the original verbatim, swapping in plausible-
/// looking but wrong kanji), and showing them risks the user
/// accepting a fluent-sounding mistake. Flagging where to look is
/// where the model adds value; the correction is a human decision.
///
/// Stable identity (`id`) lets the SwiftUI list track rows across
/// dismiss churn. Locality is by `utteranceID` — the issue is
/// meaningless outside the context of a specific row.
public struct TranscriptionIssue: Sendable, Hashable, Codable, Identifiable {
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
    /// `UtteranceEstimate.id` this issue targets.
    public let utteranceID: UUID
    public let kind: Kind
    /// One-sentence explanation of why this is flagged. Drives the
    /// trust the user has in the issue — without a reason the row
    /// is just noise.
    public let reason: String
    /// 0.0–1.0 self-reported confidence. Optional because not every
    /// backend emits one; the UI can sort or filter by it when
    /// present.
    public let confidence: Float?

    public init(
        id: UUID = UUID(),
        utteranceID: UUID,
        kind: Kind,
        reason: String,
        confidence: Float?
    ) {
        self.id = id
        self.utteranceID = utteranceID
        self.kind = kind
        self.reason = reason
        self.confidence = confidence
    }
}
