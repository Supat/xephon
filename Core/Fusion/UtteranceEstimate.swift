import Foundation
import ASR
import SERAcoustic
import SERText

// One row of the canonical per-utterance JSON output (see docs/output_schema.md).
public struct UtteranceEstimate: Sendable, Hashable, Codable, Identifiable {
    /// Stable per-utterance identifier. Default-initialized to a fresh UUID
    /// so call sites that don't supply one keep working; survives JSON
    /// export so external tooling can cross-reference rows.
    public let id: UUID
    public let speakerID: String
    /// User-supplied display name for `speakerID` at export time
    /// (e.g. `"Alice"` for `"S01"`). Stamped by
    /// `RecordingController.exportJSON` from the active rename map
    /// so external tooling can read the human name without losing
    /// the canonical id. Nil for every code path that constructs an
    /// estimate from the pipeline (LateFusion, re-eval, hand-edit) —
    /// the rename layer lives one level above the fusion stage.
    /// Optional + nil by default keeps the JSON / `.xph` schema
    /// backward-compatible: pre-rename files decode cleanly with a
    /// nil here.
    public let speakerName: String?
    public let start: TimeInterval
    public let end: TimeInterval
    public let transcript: String
    public let asrConfidence: Float?

    // Acoustic side
    public let dimensional: VADScore?
    public let acousticCategorical: CategoricalEmotion?

    // Text side
    public let plutchik: PlutchikScore?
    /// Identifier for the text-SER backend that produced `plutchik`
    /// (e.g. "deberta", "foundationModels"). Nil when text SER was skipped.
    public let textBackend: String?

    /// Whether the speech-boost EQ was enabled when this utterance was
    /// captured. Nil when unknown (e.g. batch processing of imported audio).
    public let speechBoost: Bool?

    /// Whether this utterance was produced (or refreshed) by a manual
    /// re-evaluate pass — offline ASR re-run with padded boundaries,
    /// then SER + fusion redone. `true` after at least one successful
    /// re-evaluation; nil for utterances that came straight from the
    /// streaming pipeline. Persisted in both the `.xph` bundle and
    /// the JSON export, so the marker survives a Save/Load round-trip
    /// and shows up alongside the affect data in external tooling.
    public let wasReevaluated: Bool?

    /// Whether the user manually edited the transcript / time range
    /// via the Edit Utterance dialog. `true` after at least one
    /// successful hand-edit; nil otherwise. A later re-evaluation
    /// clears this back to nil (the row reverts to a model-driven
    /// estimate). Persists in `.xph` and JSON so external tooling
    /// can flag rows whose transcript came from human review.
    public let wasHandEdited: Bool?

    // Fused
    public let fusedValence: Float?
    public let fusedArousal: Float?
    public let fusedDominance: Float?
    public let fusedTopLabel: String?

    public init(
        id: UUID = UUID(),
        speakerID: String,
        speakerName: String? = nil,
        start: TimeInterval,
        end: TimeInterval,
        transcript: String,
        asrConfidence: Float?,
        dimensional: VADScore?,
        acousticCategorical: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        textBackend: String? = nil,
        speechBoost: Bool? = nil,
        wasReevaluated: Bool? = nil,
        wasHandEdited: Bool? = nil,
        fusedValence: Float?,
        fusedArousal: Float?,
        fusedDominance: Float?,
        fusedTopLabel: String?
    ) {
        self.id = id
        self.speakerID = speakerID
        self.speakerName = speakerName
        self.start = start
        self.end = end
        self.transcript = transcript
        self.asrConfidence = asrConfidence
        self.dimensional = dimensional
        self.acousticCategorical = acousticCategorical
        self.plutchik = plutchik
        self.textBackend = textBackend
        self.speechBoost = speechBoost
        self.wasReevaluated = wasReevaluated
        self.wasHandEdited = wasHandEdited
        self.fusedValence = fusedValence
        self.fusedArousal = fusedArousal
        self.fusedDominance = fusedDominance
        self.fusedTopLabel = fusedTopLabel
    }

    /// Return a copy with `transcript` replaced. Used by the
    /// transcription-review sheet to hand the in-progress inline
    /// edit off to the full Edit Utterance panel without losing
    /// what the user has already typed.
    public func withTranscript(_ newTranscript: String) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: newTranscript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withTextBackend(_ backend: String?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: backend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withSpeakerID(_ id: String) -> UtteranceEstimate {
        UtteranceEstimate(
            id: self.id,
            speakerID: id,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    /// Return a copy with `speakerName` set to `name`. Used by
    /// `RecordingController.exportJSON` to stamp the active rename
    /// (`speakerNameOverrides[speakerID]`) onto each row right
    /// before writing the JSON, so external tooling sees the human
    /// name alongside the canonical `speakerID`. Pass nil to clear.
    public func withSpeakerName(_ name: String?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: name,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withSpeechBoost(_ enabled: Bool?) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            speakerName: speakerName,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: textBackend,
            speechBoost: enabled,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }
}
