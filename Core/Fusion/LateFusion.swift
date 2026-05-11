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
    ///
    /// Lowered from 1.0 → 0.7 → 0.35 across iterations to give the
    /// text SER more pull on the fused V/A. At 0.35, with ASR
    /// confidence 0.9: text contributes ~72%, acoustic ~28%. At
    /// confidence 0.5: text ~59%, acoustic ~41%. Even at the
    /// textWeightFloor (0.2), text contributes ~36% so acoustic
    /// can't fully outvote a low-confidence transcript.
    public static let defaultAcousticWeight: Float = 0.35

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
    ///
    /// `.trust` and `.anticipation` route to `"other"` rather than
    /// `"happy"` — they share Russell-quadrant overlap with `.joy`
    /// but the 3:1-into-happy accumulation under the previous
    /// mapping was the dominant contributor to the Happy-skewed
    /// fused label (see `docs/acoustic_ser_bias.md`). `"other"` is
    /// emotion2vec's sink class; `topLabel` already filters acoustic
    /// `"other"`/`"unknown"` from contributing, so trust /
    /// anticipation text votes effectively drop out of the label
    /// argmax under this mapping while still showing up in the
    /// inspector's Text-SER probability bars.
    public static let plutchikToAcousticLabelMapping: [PlutchikScore.Label: String] = [
        .joy: "happy", .sadness: "sad", .anger: "angry",
        .fear: "fearful", .disgust: "disgusted", .surprise: "surprised",
        .trust: "other", .anticipation: "other",
    ]

    /// Normalized (acoustic, text) fraction of the **total** score
    /// each modality contributed to the `topLabel` argmax, summed
    /// across every label bucket — not the share within the
    /// winning bucket alone. The "winning bucket only" view that
    /// this method used to compute frequently produced 100% / 0%
    /// splits when the strongest Plutchik categories happened to
    /// map to a different acoustic label than the one acoustic was
    /// most confident about, which was misleading because text
    /// often contributed plenty of score — just to non-winning
    /// buckets.
    ///
    /// Both sides use the same weighting curve as the V/A weighted
    /// average:
    ///
    /// - Acoustic total = Σ over valid (non-unknown/other) labels
    ///   of `prob × defaultAcousticWeight`.
    /// - Text total = (Σ over Plutchik labels that have a mapping
    ///   into the acoustic label space) of `prob`, then multiplied
    ///   by `max(defaultTextWeightFloor, asrConfidence)`.
    ///
    /// Returns nil only when neither side contributed any score
    /// (both inputs missing or all probabilities zero).
    public static func defaultLabelFusionShare(
        acoustic: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        asrConfidence: Float
    ) -> (acoustic: Float, text: Float)? {
        let textWeight = max(defaultTextWeightFloor, asrConfidence)
        var acousticTotal: Float = 0
        if let a = acoustic {
            for (k, v) in a.probabilities where k != .unknown && k != .other {
                acousticTotal += v * defaultAcousticWeight
            }
        }
        var textTotal: Float = 0
        if let p = plutchik {
            for (k, v) in p.probabilities
                where plutchikToAcousticLabelMapping[k] != nil {
                textTotal += v
            }
            textTotal *= textWeight
        }
        let total = acousticTotal + textTotal
        guard total > 0 else { return nil }
        return (acoustic: acousticTotal / total, text: textTotal / total)
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

    /// Argmax over the per-label score combining acoustic-categorical
    /// (emotion2vec) and Plutchik (text SER) contributions.
    ///
    /// Acoustic side scales by `defaultAcousticWeight`; text side
    /// scales by `max(defaultTextWeightFloor, asrConfidence)` — same
    /// weighting curve as the V/A weighted average in `fuse`, so the
    /// green chip and the V/A axis line up under one consistent
    /// "text weight vs acoustic weight" knob.
    ///
    /// When the raw argmax winner lands on a sink bucket
    /// (`"other"` / `"unknown"`) — which under the current Plutchik
    /// mapping happens when `trust` + `anticipation` votes outweigh
    /// every emotion-bearing bucket — the function falls back to
    /// the text SER's own top Plutchik label so the chip stays
    /// meaningful instead of grey-uncategorized.
    private static func topLabel(
        acoustic: CategoricalEmotion?,
        plutchik: PlutchikScore?,
        asrConfidence: Float
    ) -> String? {
        let textWeight = max(defaultTextWeightFloor, asrConfidence)
        var scores: [String: Float] = [:]
        if let a = acoustic {
            for (k, v) in a.probabilities where k != .unknown && k != .other {
                scores[k.rawValue, default: 0] += v * defaultAcousticWeight
            }
        }
        if let p = plutchik {
            for (k, v) in p.probabilities {
                guard let mapped = plutchikToAcousticLabelMapping[k] else { continue }
                scores[mapped, default: 0] += v * textWeight
            }
        }
        let winner = scores.max(by: { $0.value < $1.value })?.key
        if winner == "other" || winner == "unknown" {
            if let p = plutchik,
               let topPlutchik = p.probabilities.max(by: { $0.value < $1.value })?.key {
                return topPlutchik.rawValue
            }
        }
        return winner
    }
}
