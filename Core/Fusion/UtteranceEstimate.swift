import Foundation
import ASR
import SERAcoustic
import SERText

// One row of the canonical per-utterance JSON output (see docs/output_schema.md).
public struct UtteranceEstimate: Sendable, Hashable, Codable {
    public let speakerID: String
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

    // Fused
    public let fusedValence: Float?
    public let fusedArousal: Float?
    public let fusedDominance: Float?
    public let fusedTopLabel: String?

    public init(
        speakerID: String,
        start: TimeInterval,
        end: TimeInterval,
        transcript: String,
        asrConfidence: Float?,
        dimensional: VADScore?,
        acousticCategorical: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        textBackend: String? = nil,
        speechBoost: Bool? = nil,
        fusedValence: Float?,
        fusedArousal: Float?,
        fusedDominance: Float?,
        fusedTopLabel: String?
    ) {
        self.speakerID = speakerID
        self.start = start
        self.end = end
        self.transcript = transcript
        self.asrConfidence = asrConfidence
        self.dimensional = dimensional
        self.acousticCategorical = acousticCategorical
        self.plutchik = plutchik
        self.textBackend = textBackend
        self.speechBoost = speechBoost
        self.fusedValence = fusedValence
        self.fusedArousal = fusedArousal
        self.fusedDominance = fusedDominance
        self.fusedTopLabel = fusedTopLabel
    }

    public func withTextBackend(_ backend: String?) -> UtteranceEstimate {
        UtteranceEstimate(
            speakerID: speakerID,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: backend,
            speechBoost: speechBoost,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }

    public func withSpeechBoost(_ enabled: Bool?) -> UtteranceEstimate {
        UtteranceEstimate(
            speakerID: speakerID,
            start: start,
            end: end,
            transcript: transcript,
            asrConfidence: asrConfidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            textBackend: textBackend,
            speechBoost: enabled,
            fusedValence: fusedValence,
            fusedArousal: fusedArousal,
            fusedDominance: fusedDominance,
            fusedTopLabel: fusedTopLabel
        )
    }
}
