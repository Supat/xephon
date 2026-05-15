import Foundation

/// Accommodation + cohesion analyses over a session's V/A trajectory.
/// Backlog items #8–#10 in `docs/social_dynamics_backlog.md`:
///
///   8. **Accommodation over time** — mean cross-speaker V/A distance
///      between consecutive turns, time-binned. Falling curve =
///      speakers warming up; rising = splitting. A linear-fit slope
///      across the curve compresses this into one number per session.
///   9. **Group cohesion index** — within each bin, variance of all
///      utterances' fused V across speakers. Low = cohesive moment;
///      high = fragmenting. Surfaces phase transitions.
///  10. **Drift toward group mean** — per-speaker per-bin deviation
///      from the session mean V. Speakers whose deviation flattens
///      to 0 are accommodating; deviation that grows = anchoring.
///
/// One pass over the utterance list produces all three series. Pure
/// derivation; no actor isolation. Bins are evenly spaced across
/// `[min start, max end]`; rows count toward the bin their `start`
/// falls in. Identical binning semantics to
/// `AffectiveSynchrony.sessionArc` so the time axes line up if both
/// are rendered next to each other.
public enum AccommodationCohesion {

    public struct AccommodationPoint: Sendable, Hashable {
        public let midTime: TimeInterval
        /// Mean Euclidean distance in (V, A) space between
        /// consecutive cross-speaker pairs whose first turn started
        /// inside this bin. Nil when no qualifying pair landed here.
        public let meanDistance: Double?
        public let pairCount: Int

        public init(midTime: TimeInterval, meanDistance: Double?, pairCount: Int) {
            self.midTime = midTime
            self.meanDistance = meanDistance
            self.pairCount = pairCount
        }
    }

    public struct CohesionPoint: Sendable, Hashable {
        public let midTime: TimeInterval
        /// Variance of fused V across distinct speakers in this bin.
        /// Population form (Σ(x−μ)²/n), not Bessel's — n is tiny.
        /// Nil when fewer than 2 speakers had V data in the bin.
        public let valenceVariance: Double?
        public let arousalVariance: Double?
        public let activeSpeakerCount: Int

        public init(
            midTime: TimeInterval,
            valenceVariance: Double?,
            arousalVariance: Double?,
            activeSpeakerCount: Int
        ) {
            self.midTime = midTime
            self.valenceVariance = valenceVariance
            self.arousalVariance = arousalVariance
            self.activeSpeakerCount = activeSpeakerCount
        }
    }

    public struct DriftPoint: Sendable, Hashable {
        public let midTime: TimeInterval
        /// (speaker's bin-mean V) − (session-mean V). Sparse:
        /// only speakers with ≥ 1 V utterance in this bin.
        public let perSpeakerValenceDeviation: [String: Double]
        /// Same for arousal.
        public let perSpeakerArousalDeviation: [String: Double]

        public init(
            midTime: TimeInterval,
            perSpeakerValenceDeviation: [String: Double],
            perSpeakerArousalDeviation: [String: Double]
        ) {
            self.midTime = midTime
            self.perSpeakerValenceDeviation = perSpeakerValenceDeviation
            self.perSpeakerArousalDeviation = perSpeakerArousalDeviation
        }
    }

    public struct Result: Sendable {
        public let accommodation: [AccommodationPoint]
        public let cohesion: [CohesionPoint]
        public let drift: [DriftPoint]
        /// First-appearance speaker order. Used by the UI for stable
        /// per-speaker line color + legend ordering.
        public let speakerIDs: [String]
        public let sessionStart: TimeInterval
        public let sessionEnd: TimeInterval
        public let sessionMeanValence: Double?
        public let sessionMeanArousal: Double?
        /// Linear-fit slope (units / second) of accommodation
        /// distance over the session. Negative = warming up;
        /// positive = drifting apart. Nil when too few bins have
        /// data for a fit.
        public let accommodationSlope: Double?

        public init(
            accommodation: [AccommodationPoint],
            cohesion: [CohesionPoint],
            drift: [DriftPoint],
            speakerIDs: [String],
            sessionStart: TimeInterval,
            sessionEnd: TimeInterval,
            sessionMeanValence: Double?,
            sessionMeanArousal: Double?,
            accommodationSlope: Double?
        ) {
            self.accommodation = accommodation
            self.cohesion = cohesion
            self.drift = drift
            self.speakerIDs = speakerIDs
            self.sessionStart = sessionStart
            self.sessionEnd = sessionEnd
            self.sessionMeanValence = sessionMeanValence
            self.sessionMeanArousal = sessionMeanArousal
            self.accommodationSlope = accommodationSlope
        }
    }

    /// Default bin count. Matches `AffectiveSynchrony.sessionArc` so
    /// the cohesion / drift time axes line up across cards in the
    /// same session.
    public static let defaultBinCount: Int = 16

    public static func compute(
        utterances: [UtteranceEstimate],
        binCount: Int = defaultBinCount
    ) -> Result {
        guard !utterances.isEmpty, binCount > 0 else {
            return Result(
                accommodation: [], cohesion: [], drift: [],
                speakerIDs: [],
                sessionStart: 0, sessionEnd: 0,
                sessionMeanValence: nil, sessionMeanArousal: nil,
                accommodationSlope: nil
            )
        }
        let sorted = utterances.sorted { $0.start < $1.start }
        let sessionStart = sorted.first!.start
        let sessionEnd = max(sessionStart, sorted.map(\.end).max() ?? sessionStart)
        let span = max(sessionEnd - sessionStart, 1.0)
        let binWidth = span / Double(binCount)

        var speakerOrder: [String] = []
        var seenSpeakers: Set<String> = []
        for u in sorted where seenSpeakers.insert(u.speakerID).inserted {
            speakerOrder.append(u.speakerID)
        }

        // Session-level means (utterance-weighted — speakers with
        // more turns dominate, matching how the rest of the codebase
        // computes session means).
        var vSumAll = 0.0, vCountAll = 0
        var aSumAll = 0.0, aCountAll = 0
        for u in sorted {
            if let v = u.fusedValence { vSumAll += Double(v); vCountAll += 1 }
            if let a = u.fusedArousal { aSumAll += Double(a); aCountAll += 1 }
        }
        let sessionMeanV: Double? = vCountAll > 0 ? vSumAll / Double(vCountAll) : nil
        let sessionMeanA: Double? = aCountAll > 0 ? aSumAll / Double(aCountAll) : nil

        // #8 — accommodation: walk consecutive cross-speaker pairs.
        // Attribute each pair's distance to the bin its leader's
        // `start` falls in, mirroring the convention `sessionArc`
        // uses for V/A.
        struct AccommodationAccum {
            var sum: Double = 0
            var count: Int = 0
        }
        var accommodationBins = [AccommodationAccum](repeating: .init(), count: binCount)
        for i in 0..<(sorted.count - 1) {
            let leader = sorted[i]
            let follower = sorted[i + 1]
            guard leader.speakerID != follower.speakerID,
                  let lv = leader.fusedValence,
                  let fv = follower.fusedValence,
                  let la = leader.fusedArousal,
                  let fa = follower.fusedArousal else { continue }
            let dv = Double(lv - fv)
            let da = Double(la - fa)
            let dist = (dv * dv + da * da).squareRoot()
            let idx = Self.clampedBinIndex(
                for: leader.start,
                start: sessionStart,
                width: binWidth,
                count: binCount
            )
            accommodationBins[idx].sum += dist
            accommodationBins[idx].count += 1
        }

        // #9 + #10 — cohesion + drift: collect per-speaker per-bin
        // V/A means, then compute variance across speakers and
        // deviation from session mean.
        struct BinAccum {
            var vSums: [String: Double] = [:]
            var vCounts: [String: Int] = [:]
            var aSums: [String: Double] = [:]
            var aCounts: [String: Int] = [:]
        }
        var binAccums = [BinAccum](repeating: .init(), count: binCount)
        for u in sorted {
            let idx = Self.clampedBinIndex(
                for: u.start,
                start: sessionStart,
                width: binWidth,
                count: binCount
            )
            if let v = u.fusedValence {
                binAccums[idx].vSums[u.speakerID, default: 0] += Double(v)
                binAccums[idx].vCounts[u.speakerID, default: 0] += 1
            }
            if let a = u.fusedArousal {
                binAccums[idx].aSums[u.speakerID, default: 0] += Double(a)
                binAccums[idx].aCounts[u.speakerID, default: 0] += 1
            }
        }

        var accommodation: [AccommodationPoint] = []
        var cohesion: [CohesionPoint] = []
        var drift: [DriftPoint] = []
        accommodation.reserveCapacity(binCount)
        cohesion.reserveCapacity(binCount)
        drift.reserveCapacity(binCount)

        for i in 0..<binCount {
            let mid = sessionStart + (Double(i) + 0.5) * binWidth

            // #8
            let acc = accommodationBins[i]
            accommodation.append(AccommodationPoint(
                midTime: mid,
                meanDistance: acc.count > 0 ? acc.sum / Double(acc.count) : nil,
                pairCount: acc.count
            ))

            // #9 + #10
            let bin = binAccums[i]
            var perSpkV: [String: Double] = [:]
            var perSpkA: [String: Double] = [:]
            for (spk, sum) in bin.vSums {
                let c = bin.vCounts[spk] ?? 1
                perSpkV[spk] = sum / Double(c)
            }
            for (spk, sum) in bin.aSums {
                let c = bin.aCounts[spk] ?? 1
                perSpkA[spk] = sum / Double(c)
            }

            // Variance across speakers.
            let vMeans = Array(perSpkV.values)
            let aMeans = Array(perSpkA.values)
            let vVar: Double? = vMeans.count >= 2 ? Self.populationVariance(vMeans) : nil
            let aVar: Double? = aMeans.count >= 2 ? Self.populationVariance(aMeans) : nil
            cohesion.append(CohesionPoint(
                midTime: mid,
                valenceVariance: vVar,
                arousalVariance: aVar,
                activeSpeakerCount: max(perSpkV.count, perSpkA.count)
            ))

            // Drift = (speaker bin mean) − (session mean).
            var vDev: [String: Double] = [:]
            var aDev: [String: Double] = [:]
            if let m = sessionMeanV {
                for (spk, mean) in perSpkV { vDev[spk] = mean - m }
            }
            if let m = sessionMeanA {
                for (spk, mean) in perSpkA { aDev[spk] = mean - m }
            }
            drift.append(DriftPoint(
                midTime: mid,
                perSpeakerValenceDeviation: vDev,
                perSpeakerArousalDeviation: aDev
            ))
        }

        // Accommodation slope: linear fit of bin index → mean
        // distance, in distance-units / second. Only bins with
        // data contribute (no imputation). Need ≥ 3 non-nil bins
        // before the slope is meaningful.
        let slope: Double? = Self.linearSlope(
            points: accommodation.enumerated().compactMap { idx, p -> (Double, Double)? in
                guard let d = p.meanDistance else { return nil }
                return (p.midTime, d)
            }
        )

        return Result(
            accommodation: accommodation,
            cohesion: cohesion,
            drift: drift,
            speakerIDs: speakerOrder,
            sessionStart: sessionStart,
            sessionEnd: sessionEnd,
            sessionMeanValence: sessionMeanV,
            sessionMeanArousal: sessionMeanA,
            accommodationSlope: slope
        )
    }

    private static func clampedBinIndex(
        for time: TimeInterval,
        start: TimeInterval,
        width: TimeInterval,
        count: Int
    ) -> Int {
        let offset = time - start
        var idx = Int((offset / width).rounded(.down))
        if idx >= count { idx = count - 1 }
        if idx < 0 { idx = 0 }
        return idx
    }

    private static func populationVariance(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let mean = values.reduce(0, +) / Double(values.count)
        var sumSq = 0.0
        for v in values {
            let d = v - mean
            sumSq += d * d
        }
        return sumSq / Double(values.count)
    }

    /// Ordinary-least-squares slope of `y` vs. `x` across `points`.
    /// Returns nil when fewer than 3 points or zero variance in x.
    private static func linearSlope(points: [(Double, Double)]) -> Double? {
        guard points.count >= 3 else { return nil }
        let n = Double(points.count)
        let meanX = points.reduce(0) { $0 + $1.0 } / n
        let meanY = points.reduce(0) { $0 + $1.1 } / n
        var num = 0.0
        var den = 0.0
        for (x, y) in points {
            let dx = x - meanX
            num += dx * (y - meanY)
            den += dx * dx
        }
        guard den > 1e-9 else { return nil }
        return num / den
    }
}
