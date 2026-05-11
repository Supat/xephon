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
    /// Floor applied to the text weight when ASR confidence is low,
    /// so a noisy transcript still contributes something instead of
    /// being silently dropped. Also re-exposed as
    /// `defaultTextWeightFloor` for view-layer reads.
    public static let defaultTextWeightFloor: Float = 0.2
    /// Constant weight applied to the acoustic modality. Not scaled
    /// by ASR confidence — the acoustic side doesn't depend on the
    /// transcript being correct.
    public static let defaultAcousticWeight: Float = 1.0

    private let textWeightFloor: Float
    private let acousticWeight: Float

    public init(
        textWeightFloor: Float = LateFusion.defaultTextWeightFloor,
        acousticWeight: Float = LateFusion.defaultAcousticWeight
    ) {
        self.textWeightFloor = textWeightFloor
        self.acousticWeight = acousticWeight
    }

    /// Normalized (acoustic, text) fraction the V/A weighted average
    /// would apply for an utterance with the given ASR confidence,
    /// using the default weights. Fractions sum to 1.0.
    public static func defaultVAFusionShare(
        asrConfidence: Float
    ) -> (acoustic: Float, text: Float) {
        let text = max(defaultTextWeightFloor, asrConfidence)
        let total = defaultAcousticWeight + text
        return (acoustic: defaultAcousticWeight / total, text: text / total)
    }

    /// Coarse mapping from Plutchik labels to the overlapping
    /// acoustic-categorical names used by `topLabel`. Public so the
    /// view layer can attribute per-label contributions back to each
    /// modality without re-stating the mapping.
    public static let plutchikToAcousticLabelMapping: [PlutchikScore.Label: String] = [
        .joy: "happy", .sadness: "sad", .anger: "angry",
        .fear: "fearful", .disgust: "disgusted", .surprise: "surprised",
        .trust: "happy", .anticipation: "happy",
    ]

    /// Normalized (acoustic, text) fraction of the merged score that
    /// went to `label` during `topLabel` argmax. Returns nil when
    /// neither side contributed (e.g. a label that no modality
    /// produced, or both inputs missing). Acoustic side contributes
    /// `prob`; text side contributes `prob × asrConfidence`, summed
    /// across all Plutchik labels that map to `label`.
    public static func defaultLabelFusionShare(
        forLabel label: String,
        acoustic: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        asrConfidence: Float
    ) -> (acoustic: Float, text: Float)? {
        var acousticContrib: Float = 0
        if let a = acoustic,
           let key = CategoricalEmotion.Label(rawValue: label),
           key != .unknown, key != .other {
            acousticContrib = a.probabilities[key] ?? 0
        }
        var textContrib: Float = 0
        if let p = plutchik {
            for (k, v) in p.probabilities
                where plutchikToAcousticLabelMapping[k] == label {
                textContrib += v
            }
            textContrib *= asrConfidence
        }
        let total = acousticContrib + textContrib
        guard total > 0 else { return nil }
        return (acoustic: acousticContrib / total, text: textContrib / total)
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
            for (k, v) in p.probabilities {
                guard let mapped = plutchikToAcousticLabelMapping[k] else { continue }
                scores[mapped, default: 0] += v * asrConfidence
            }
        }
        return scores.max(by: { $0.value < $1.value })?.key
    }
}
