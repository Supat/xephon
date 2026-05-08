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
        self.fusedValence = fusedValence
        self.fusedArousal = fusedArousal
        self.fusedDominance = fusedDominance
        self.fusedTopLabel = fusedTopLabel
    }
}
