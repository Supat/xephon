import Foundation
import SERAcoustic
import SERText

/// Per-utterance "modality disagreement" score: how much the
/// acoustic 9-class and text 8-class affect distributions point at
/// **different** emotions on the shared label subspace. Surfaces
/// candidate sarcasm / mixed-affect / irony rows — the ones where
/// the voice and the words are doing different jobs.
///
/// Backlog item #11 in `docs/social_dynamics_backlog.md`. The data
/// is already on every `UtteranceEstimate`; this is pure derivation
/// — no extra ML, no actor isolation.
public enum ModalityDisagreement {

    /// Acoustic ↔ Plutchik label pairs that have a one-to-one
    /// correspondence (acoustic's `.angry` is Plutchik's `.anger`,
    /// etc.). Acoustic-only labels (`.neutral`, `.other`, `.unknown`)
    /// and Plutchik-only labels (`.anticipation`, `.trust`) are
    /// dropped from the comparison — there's no honest projection
    /// from one to the other, so including them would just add
    /// noise. The remaining six axes cover the bulk of affect-
    /// bearing speech and form a fair shared subspace.
    public static let sharedAxes: [(acoustic: CategoricalEmotion.Label, plutchik: PlutchikScore.Label)] = [
        (.angry,     .anger),
        (.disgusted, .disgust),
        (.fearful,   .fear),
        (.happy,     .joy),
        (.sad,       .sadness),
        (.surprised, .surprise),
    ]

    /// A row's disagreement reading.
    public struct Score: Sendable, Hashable {
        /// Total Variation Distance between the renormalized
        /// acoustic and Plutchik distributions over the shared
        /// six-axis space. Range `[0, 1]`: 0 = perfect agreement;
        /// 1 = entirely disjoint.
        public let tvd: Double
        /// Acoustic's top label inside the shared subspace, or nil
        /// if acoustic had no shared mass.
        public let acousticTop: CategoricalEmotion.Label?
        /// Plutchik's top label inside the shared subspace, or nil
        /// if Plutchik had no shared mass.
        public let plutchikTop: PlutchikScore.Label?
        /// True when the two top labels point at Plutchik **opposites**
        /// (joy↔sadness, anger↔fear). A weaker form of disagreement
        /// than TVD alone — the top-vs-top axis answers "is the row
        /// mixed-affect, or is it specifically *flipped*?"
        public let topsAreOpposites: Bool

        public init(
            tvd: Double,
            acousticTop: CategoricalEmotion.Label?,
            plutchikTop: PlutchikScore.Label?,
            topsAreOpposites: Bool
        ) {
            self.tvd = tvd
            self.acousticTop = acousticTop
            self.plutchikTop = plutchikTop
            self.topsAreOpposites = topsAreOpposites
        }
    }

    /// Default threshold above which a row is flagged as
    /// "modality-disagreeing." TVD ≥ 0.50 means the two
    /// distributions overlap on at most half their mass — a
    /// substantively different reading, not a small drift.
    public static let flagThreshold: Double = 0.50

    /// Compute the disagreement score from a row's acoustic +
    /// Plutchik outputs. Returns nil when either modality is
    /// missing — there's nothing to compare, not "agreement."
    public static func score(
        acoustic: CategoricalEmotion?,
        plutchik: PlutchikScore?
    ) -> Score? {
        guard let acoustic, let plutchik else { return nil }

        var a = [Double](repeating: 0, count: sharedAxes.count)
        var p = [Double](repeating: 0, count: sharedAxes.count)
        for (idx, axis) in sharedAxes.enumerated() {
            a[idx] = Double(acoustic.probabilities[axis.acoustic] ?? 0)
            p[idx] = Double(plutchik.probabilities[axis.plutchik] ?? 0)
        }
        let aSum = a.reduce(0, +)
        let pSum = p.reduce(0, +)
        guard aSum > 0, pSum > 0 else { return nil }
        for i in 0..<a.count {
            a[i] /= aSum
            p[i] /= pSum
        }
        // Total Variation Distance.
        var tvd = 0.0
        for i in 0..<a.count {
            tvd += abs(a[i] - p[i])
        }
        tvd *= 0.5

        let acousticTopIdx = a.indices.max(by: { a[$0] < a[$1] })
        let plutchikTopIdx = p.indices.max(by: { p[$0] < p[$1] })
        let aTop = acousticTopIdx.map { sharedAxes[$0].acoustic }
        let pTop = plutchikTopIdx.map { sharedAxes[$0].plutchik }

        let opposites: Bool = {
            guard let aTop, let pTop else { return false }
            return isPlutchikOpposite(acoustic: aTop, plutchik: pTop)
        }()

        return Score(
            tvd: tvd,
            acousticTop: aTop,
            plutchikTop: pTop,
            topsAreOpposites: opposites
        )
    }

    /// Plutchik wheel opposites that exist in the shared subspace.
    /// Plutchik's canonical opposites are joy↔sadness, anger↔fear,
    /// trust↔disgust, anticipation↔surprise — but the latter two
    /// don't fully exist on the acoustic side (no `.trust`,
    /// `.anticipation`), so disgust and surprise have no opposites
    /// in the shared subspace and never trip `topsAreOpposites`.
    private static func isPlutchikOpposite(
        acoustic: CategoricalEmotion.Label,
        plutchik: PlutchikScore.Label
    ) -> Bool {
        switch (acoustic, plutchik) {
        case (.happy, .sadness), (.sad, .joy):
            return true
        case (.angry, .fear), (.fearful, .anger):
            return true
        default:
            return false
        }
    }

    public struct SpeakerTally: Sendable, Hashable {
        public let speakerID: String
        /// Rows flagged above `flagThreshold`.
        public let flaggedCount: Int
        /// Mean TVD across this speaker's rows that had both
        /// modalities present. Nil when no qualifying rows.
        public let meanTVD: Double?
        /// Rows where the top acoustic and top Plutchik labels were
        /// Plutchik opposites (e.g. happy voice but sad text).
        /// Subset of `flaggedCount`.
        public let oppositeCount: Int

        public init(
            speakerID: String,
            flaggedCount: Int,
            meanTVD: Double?,
            oppositeCount: Int
        ) {
            self.speakerID = speakerID
            self.flaggedCount = flaggedCount
            self.meanTVD = meanTVD
            self.oppositeCount = oppositeCount
        }
    }

    /// Per-speaker tally of disagreement events. `utterances` is
    /// scanned once; rows missing either modality contribute
    /// nothing. Sorted by `flaggedCount` descending (then by
    /// `speakerID` for stability).
    public static func tallies(
        utterances: [UtteranceEstimate],
        threshold: Double = flagThreshold
    ) -> [SpeakerTally] {
        struct Accum {
            var flagged: Int = 0
            var opposites: Int = 0
            var tvdSum: Double = 0
            var tvdCount: Int = 0
        }
        var bySpeaker: [String: Accum] = [:]
        for u in utterances {
            guard let s = score(
                acoustic: u.acousticCategorical,
                plutchik: u.plutchik
            ) else { continue }
            var acc = bySpeaker[u.speakerID] ?? Accum()
            acc.tvdSum += s.tvd
            acc.tvdCount += 1
            if s.tvd >= threshold {
                acc.flagged += 1
                if s.topsAreOpposites { acc.opposites += 1 }
            }
            bySpeaker[u.speakerID] = acc
        }
        return bySpeaker.map { spk, acc in
            SpeakerTally(
                speakerID: spk,
                flaggedCount: acc.flagged,
                meanTVD: acc.tvdCount > 0 ? acc.tvdSum / Double(acc.tvdCount) : nil,
                oppositeCount: acc.opposites
            )
        }
        .sorted {
            if $0.flaggedCount != $1.flaggedCount {
                return $0.flaggedCount > $1.flaggedCount
            }
            return $0.speakerID < $1.speakerID
        }
    }
}
