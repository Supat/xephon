import Foundation
import XephonLogging
import ASR
import SERAcoustic
import SERText

public protocol Fuser: Actor {
    func fuse(
        asr: ASRSegment,
        speakerID: String,
        dimensional: VADScore?,
        acousticCategorical: CategoricalEmotion?,
        plutchik: PlutchikScore?
    ) async throws -> UtteranceEstimate
}

// Default fusion: ASR-confidence-aware weighted late fusion.
// Per CLAUDE.md: do NOT introduce a trained cross-modal head without an
// in-domain calibration dataset and explicit go-ahead.
//
// Strategy:
//   - V/A: weighted average of acoustic dimensional and a Plutchik-derived
//     valence/arousal estimate. Text weight is scaled by ASR confidence.
//   - D (dominance): acoustic only — text-only models don't estimate dominance.
//   - top_label: argmax over the union of acoustic categorical and Plutchik
//     after a coarse mapping onto a shared label space.
//   - Missing modalities are skipped, not zero-imputed.
public actor LateFusion: Fuser {
    private let textWeightFloor: Float
    private let acousticWeight: Float

    public init(textWeightFloor: Float = 0.2, acousticWeight: Float = 1.0) {
        self.textWeightFloor = textWeightFloor
        self.acousticWeight = acousticWeight
    }

    public func fuse(
        asr: ASRSegment,
        speakerID: String,
        dimensional: VADScore?,
        acousticCategorical: CategoricalEmotion?,
        plutchik: PlutchikScore?
    ) async throws -> UtteranceEstimate {
        let asrConfidence = asr.confidence ?? 0.5
        let textWeight = max(textWeightFloor, asrConfidence)

        let plutchikValence = plutchik.map(Self.plutchikToValence)
        let plutchikArousal = plutchik.map(Self.plutchikToArousal)

        let fusedV = Self.weightedAverage(
            (dimensional?.valence, acousticWeight),
            (plutchikValence, textWeight)
        )
        let fusedA = Self.weightedAverage(
            (dimensional?.arousal, acousticWeight),
            (plutchikArousal, textWeight)
        )
        let fusedD = dimensional?.dominance

        let topLabel = Self.topLabel(
            acoustic: acousticCategorical,
            plutchik: plutchik,
            asrConfidence: asrConfidence
        )

        AppLog.fusion.debug(
            "fused [\(asr.start, privacy: .public)–\(asr.end, privacy: .public)s] V=\(fusedV ?? -1) A=\(fusedA ?? -1) label=\(topLabel ?? "nil", privacy: .public)"
        )

        return UtteranceEstimate(
            speakerID: speakerID,
            start: asr.start,
            end: asr.end,
            transcript: asr.text,
            asrConfidence: asr.confidence,
            dimensional: dimensional,
            acousticCategorical: acousticCategorical,
            plutchik: plutchik,
            fusedValence: fusedV,
            fusedArousal: fusedA,
            fusedDominance: fusedD,
            fusedTopLabel: topLabel
        )
    }

    // MARK: - Helpers

    private static func weightedAverage(
        _ a: (Float?, Float),
        _ b: (Float?, Float)
    ) -> Float? {
        switch (a.0, b.0) {
        case (nil, nil): return nil
        case (let x?, nil): return x
        case (nil, let y?): return y
        case (let x?, let y?):
            let total = a.1 + b.1
            return total > 0 ? (x * a.1 + y * b.1) / total : nil
        }
    }

    private static func plutchikToValence(_ p: PlutchikScore) -> Float {
        // Russell-style polar mapping. Coefficients are conservative defaults;
        // tune from a calibration set per docs/eval_log.md.
        let pos = (p.probabilities[.joy] ?? 0)
                + (p.probabilities[.trust] ?? 0) * 0.6
                + (p.probabilities[.anticipation] ?? 0) * 0.3
        let neg = (p.probabilities[.sadness] ?? 0)
                + (p.probabilities[.fear] ?? 0)
                + (p.probabilities[.disgust] ?? 0)
                + (p.probabilities[.anger] ?? 0) * 0.7
        let raw = pos - neg
        // Map [-2, +2] → [0, 1] with a soft clamp.
        return max(0, min(1, 0.5 + raw * 0.25))
    }

    private static func plutchikToArousal(_ p: PlutchikScore) -> Float {
        // Anger / fear / surprise / joy are high-arousal; sadness / trust low.
        let high = (p.probabilities[.anger] ?? 0)
                 + (p.probabilities[.fear] ?? 0)
                 + (p.probabilities[.surprise] ?? 0)
                 + (p.probabilities[.joy] ?? 0) * 0.6
        let low  = (p.probabilities[.sadness] ?? 0)
                 + (p.probabilities[.trust] ?? 0) * 0.5
        let raw = high - low
        return max(0, min(1, 0.5 + raw * 0.3))
    }

    private static func topLabel(
        acoustic: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        asrConfidence: Float
    ) -> String? {
        var scores: [String: Float] = [:]
        if let a = acoustic {
            for (k, v) in a.probabilities where k != .unknown && k != .other {
                scores[k.rawValue, default: 0] += v
            }
        }
        if let p = plutchik {
            // Lift Plutchik 8 to overlapping rough labels with the acoustic 9.
            let mapping: [PlutchikScore.Label: String] = [
                .joy: "happy", .sadness: "sad", .anger: "angry",
                .fear: "fearful", .disgust: "disgusted", .surprise: "surprised",
                .trust: "happy", .anticipation: "happy"
            ]
            for (k, v) in p.probabilities {
                guard let mapped = mapping[k] else { continue }
                scores[mapped, default: 0] += v * asrConfidence
            }
        }
        return scores.max(by: { $0.value < $1.value })?.key
    }
}
