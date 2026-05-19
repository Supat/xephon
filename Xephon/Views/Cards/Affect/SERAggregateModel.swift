import Foundation
import Fusion
import SERAcoustic
import SERText

/// Session-wide aggregate of the per-modality SER outputs that
/// `SERAggregateCard` renders: per-Plutchik-label means, per-
/// acoustic-class means, and one V×A point per utterance.
///
/// Owned by the card via `@State`; recomputed via `.task(id:
/// recorder.utterancesVersion)` so the three folds only run when
/// the utterance list actually shifts — not on every render
/// (focus-id changes, scroll ticks, unrelated card mutations).
@MainActor
@Observable
final class SERAggregateModel {
    public struct VAPoint: Sendable {
        public let id: UUID
        public let valence: Float
        public let arousal: Float
        public let speakerID: String
    }

    private(set) var plutchikMeans: [PlutchikScore.Label: Float] = [:]
    private(set) var acousticMeans: [CategoricalEmotion.Label: Float] = [:]
    private(set) var vaPoints: [VAPoint] = []

    /// True when neither the Plutchik nor the acoustic means
    /// contain any non-zero entry — the card's empty-state copy
    /// reads this for both modality panels independently, so we
    /// expose the per-modality emptiness directly.
    var hasAnyPlutchik: Bool {
        plutchikMeans.values.contains(where: { $0 != 0 })
    }

    var hasAnyAcoustic: Bool {
        acousticMeans.values.contains(where: { $0 != 0 })
    }

    /// Recompute all three aggregates from the supplied utterance
    /// list. Cheap: O(N × labels) per pass with no allocations
    /// besides the output dictionaries. Each modality's mean
    /// follows the per-utterance schema's missing-label-means-zero
    /// rule (not "missing"-as-NaN).
    func recompute(from utterances: [UtteranceEstimate]) {
        plutchikMeans = Self.foldPlutchikMeans(utterances)
        acousticMeans = Self.foldAcousticMeans(utterances)
        vaPoints = Self.collectVAPoints(utterances)
    }

    private static func foldPlutchikMeans(
        _ utterances: [UtteranceEstimate]
    ) -> [PlutchikScore.Label: Float] {
        var sums: [PlutchikScore.Label: Float] = [:]
        var count: Int = 0
        for utt in utterances {
            guard let probs = utt.plutchik?.probabilities else { continue }
            count += 1
            for label in PlutchikScore.Label.allCases {
                sums[label, default: 0] += probs[label] ?? 0
            }
        }
        guard count > 0 else { return [:] }
        var out: [PlutchikScore.Label: Float] = [:]
        for label in PlutchikScore.Label.allCases {
            out[label] = (sums[label] ?? 0) / Float(count)
        }
        return out
    }

    private static func foldAcousticMeans(
        _ utterances: [UtteranceEstimate]
    ) -> [CategoricalEmotion.Label: Float] {
        var sums: [CategoricalEmotion.Label: Float] = [:]
        var count: Int = 0
        for utt in utterances {
            guard let probs = utt.acousticCategorical?.probabilities else { continue }
            count += 1
            for label in CategoricalEmotion.Label.allCases {
                sums[label, default: 0] += probs[label] ?? 0
            }
        }
        guard count > 0 else { return [:] }
        var out: [CategoricalEmotion.Label: Float] = [:]
        for label in CategoricalEmotion.Label.allCases {
            out[label] = (sums[label] ?? 0) / Float(count)
        }
        return out
    }

    /// One dot per utterance with both fused V and fused A
    /// present. Rows missing either coordinate are dropped —
    /// there's no meaningful position to plot otherwise. The V/A
    /// space is canonically 0..1 in this codebase; we don't clamp
    /// here so out-of-range values would render at the canvas
    /// edges and flag the anomaly visually.
    private static func collectVAPoints(
        _ utterances: [UtteranceEstimate]
    ) -> [VAPoint] {
        utterances.compactMap { utt in
            guard let v = utt.fusedValence,
                  let a = utt.fusedArousal else { return nil }
            return VAPoint(
                id: utt.id,
                valence: v,
                arousal: a,
                speakerID: utt.speakerID
            )
        }
    }
}
