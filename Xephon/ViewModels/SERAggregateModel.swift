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
        plutchikMeans = Self.foldMeans(utterances) { $0.plutchik?.probabilities }
        acousticMeans = Self.foldMeans(utterances) { $0.acousticCategorical?.probabilities }
        vaPoints = Self.collectVAPoints(utterances)
    }

    /// Label-generic per-label mean over the utterances that carry
    /// the modality at all (missing label within a carried modality
    /// means zero, per the output schema).
    private static func foldMeans<Label: CaseIterable & Hashable>(
        _ utterances: [UtteranceEstimate],
        probs: (UtteranceEstimate) -> [Label: Float]?
    ) -> [Label: Float] {
        var sums: [Label: Float] = [:]
        var count: Int = 0
        for utt in utterances {
            guard let probs = probs(utt) else { continue }
            count += 1
            for label in Label.allCases {
                sums[label, default: 0] += probs[label] ?? 0
            }
        }
        guard count > 0 else { return [:] }
        return sums.mapValues { $0 / Float(count) }
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
