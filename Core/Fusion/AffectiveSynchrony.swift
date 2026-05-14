import Foundation
import SERText

/// Pairwise affective-synchrony analysis over a session's utterance
/// list, plus three derived views:
///
///   1. **Lag profile** per directed speaker pair — correlations at
///      lags 0…N where lag k means "leader speaks, then k other
///      turns by any speaker, then follower speaks." Lag 0 is the
///      immediate-response signal; non-zero lags surface multi-turn
///      delayed echo patterns.
///   2. **Leadership scores** per speaker — average correlation at a
///      chosen lag, weighted by sample count, across all pairs where
///      this speaker is the leader. High lag-1 leadership = "others
///      follow this speaker's emotion one turn later."
///   3. **Plutchik dyads** — across-speaker pairings of dominant
///      Plutchik labels mapped to Plutchik's primary-dyad table
///      (joy+trust=love, anger+disgust=contempt, …). Counts the
///      distribution over consecutive turn-pairs.
///   4. **Session-level arc** — time-binned per-speaker mean V/A
///      plus the session aggregate, used to plot convergence /
///      divergence over the conversation.
///
/// All operations are pure functions over the utterance list — the
/// caller passes whatever subset they want analyzed and gets back a
/// `Sendable` value type. No actor isolation, no persistence; results
/// are cheap enough to recompute on each render.
public enum AffectiveSynchrony {
    // MARK: - Pair correlations

    /// Directed speaker pair "leader → follower" — the leader spoke,
    /// the follower responded. Asymmetric: `(A, B)` and `(B, A)` are
    /// distinct correlations because the conversational roles differ.
    public struct DirectedPair: Sendable, Hashable {
        public let leader: String
        public let follower: String
        public init(leader: String, follower: String) {
            self.leader = leader
            self.follower = follower
        }
    }

    public struct PairResult: Sendable, Hashable {
        public let pair: DirectedPair
        /// Correlations at lags 0…maxLag. Index = lag. Nil entries
        /// = below `minSamples` at that lag — still listed (not
        /// trimmed) so the sparkline maintains its x-axis alignment
        /// across pairs.
        public let valenceProfile: [Double?]
        public let arousalProfile: [Double?]
        public let sampleCountsByLag: [Int]

        public init(
            pair: DirectedPair,
            valenceProfile: [Double?],
            arousalProfile: [Double?],
            sampleCountsByLag: [Int]
        ) {
            self.pair = pair
            self.valenceProfile = valenceProfile
            self.arousalProfile = arousalProfile
            self.sampleCountsByLag = sampleCountsByLag
        }

        /// Headline correlation for sort + headline display:
        /// lag 0 valence. Convenience for callers that don't need
        /// the full profile.
        public var valenceCorrelation: Double? {
            valenceProfile.first ?? nil
        }
        public var arousalCorrelation: Double? {
            arousalProfile.first ?? nil
        }
        /// Headline sample count (lag 0). Convenience.
        public var sampleCount: Int {
            sampleCountsByLag.first ?? 0
        }
    }

    public struct Result: Sendable {
        public let pairs: [PairResult]
        public let maxLag: Int
        public init(pairs: [PairResult], maxLag: Int) {
            self.pairs = pairs
            self.maxLag = maxLag
        }
    }

    /// Minimum response-pair sample count before we report a
    /// correlation. Pearson over 1 or 2 points is mathematically
    /// degenerate / trivially ±1; 3 is the smallest sample where
    /// the sign actually reflects covariance.
    public static let minSamples: Int = 3

    /// Default lag profile depth. Lag 0 covers the "immediate
    /// response" case that's all most 2-speaker dialogues will
    /// have; lags 1–3 only get data when ≥3 speakers create
    /// intervening turns between leader and follower. Going beyond
    /// 3 mostly yields empty buckets even on busy conversations.
    public static let defaultMaxLag: Int = 3

    public static func compute(
        utterances: [UtteranceEstimate],
        maxLag: Int = defaultMaxLag
    ) -> Result {
        guard utterances.count >= 2, maxLag >= 0 else {
            return Result(pairs: [], maxLag: max(0, maxLag))
        }
        let sorted = utterances.sorted { $0.start < $1.start }
        let pairCount = maxLag + 1

        // Lag k pairs leader sorted[i] with follower sorted[i+k+1]:
        // k intervening turns between them. For k=0 that reduces to
        // the original consecutive-turn definition.
        struct Bucket {
            var valencePairsByLag: [[(Double, Double)]]
            var arousalPairsByLag: [[(Double, Double)]]
            init(lagCount: Int) {
                valencePairsByLag = Array(repeating: [], count: lagCount)
                arousalPairsByLag = Array(repeating: [], count: lagCount)
            }
        }
        var buckets: [DirectedPair: Bucket] = [:]

        for k in 0...maxLag {
            let offset = k + 1
            guard sorted.count > offset else { break }
            for i in 0..<(sorted.count - offset) {
                let leader = sorted[i]
                let follower = sorted[i + offset]
                guard leader.speakerID != follower.speakerID else { continue }
                let pair = DirectedPair(
                    leader: leader.speakerID,
                    follower: follower.speakerID
                )
                var bucket = buckets[pair] ?? Bucket(lagCount: pairCount)
                if let lv = leader.fusedValence, let fv = follower.fusedValence {
                    bucket.valencePairsByLag[k]
                        .append((Double(lv), Double(fv)))
                }
                if let la = leader.fusedArousal, let fa = follower.fusedArousal {
                    bucket.arousalPairsByLag[k]
                        .append((Double(la), Double(fa)))
                }
                buckets[pair] = bucket
            }
        }

        var results: [PairResult] = []
        results.reserveCapacity(buckets.count)
        for (pair, bucket) in buckets {
            let vProfile: [Double?] = bucket.valencePairsByLag.map { samples in
                samples.count >= minSamples ? pearson(samples) : nil
            }
            let aProfile: [Double?] = bucket.arousalPairsByLag.map { samples in
                samples.count >= minSamples ? pearson(samples) : nil
            }
            let counts: [Int] = (0..<pairCount).map { k in
                max(
                    bucket.valencePairsByLag[k].count,
                    bucket.arousalPairsByLag[k].count
                )
            }
            // Skip pairs with no data on any lag.
            guard vProfile.contains(where: { $0 != nil })
                || aProfile.contains(where: { $0 != nil }) else { continue }
            results.append(PairResult(
                pair: pair,
                valenceProfile: vProfile,
                arousalProfile: aProfile,
                sampleCountsByLag: counts
            ))
        }
        results.sort { lhs, rhs in
            if lhs.sampleCount != rhs.sampleCount {
                return lhs.sampleCount > rhs.sampleCount
            }
            if lhs.pair.leader != rhs.pair.leader {
                return lhs.pair.leader < rhs.pair.leader
            }
            return lhs.pair.follower < rhs.pair.follower
        }
        return Result(pairs: results, maxLag: maxLag)
    }

    // MARK: - Leadership scoring

    public struct LeadershipScore: Sendable, Hashable {
        public let speakerID: String
        /// Sample-weighted mean of `(this speaker → *)` valence
        /// correlations at the chosen lag. Nil when this speaker
        /// doesn't lead any sufficiently-sampled pair at that lag.
        public let valenceLeadership: Double?
        public let arousalLeadership: Double?
        /// Total sample count summed across all `(this → *)` pairs
        /// at the chosen lag.
        public let sampleCount: Int

        public init(
            speakerID: String,
            valenceLeadership: Double?,
            arousalLeadership: Double?,
            sampleCount: Int
        ) {
            self.speakerID = speakerID
            self.valenceLeadership = valenceLeadership
            self.arousalLeadership = arousalLeadership
            self.sampleCount = sampleCount
        }
    }

    /// Derive a per-speaker leadership score at the given lag — for
    /// each speaker A, the sample-weighted mean of every PairResult
    /// where `pair.leader == A`. Lag 1 is the canonical "emotional
    /// leadership" reading: A's emotion at turn t is echoed by
    /// others at turn t+2 (one intervening turn). Lag 0 collapses
    /// to "immediate-response synchrony" which is more about adjacency.
    public static func leadershipScores(
        from result: Result,
        atLag lag: Int = 1
    ) -> [LeadershipScore] {
        guard lag >= 0, lag <= result.maxLag else { return [] }
        var bySpeaker: [String: (vNum: Double, vDen: Int, aNum: Double, aDen: Int)] = [:]
        for entry in result.pairs {
            let n = entry.sampleCountsByLag[lag]
            guard n > 0 else { continue }
            let vCorr = entry.valenceProfile[lag]
            let aCorr = entry.arousalProfile[lag]
            var existing = bySpeaker[entry.pair.leader]
                ?? (vNum: 0, vDen: 0, aNum: 0, aDen: 0)
            if let v = vCorr {
                existing.vNum += v * Double(n)
                existing.vDen += n
            }
            if let a = aCorr {
                existing.aNum += a * Double(n)
                existing.aDen += n
            }
            bySpeaker[entry.pair.leader] = existing
        }
        var out: [LeadershipScore] = []
        for (speaker, sums) in bySpeaker {
            let vScore = sums.vDen >= minSamples ? sums.vNum / Double(sums.vDen) : nil
            let aScore = sums.aDen >= minSamples ? sums.aNum / Double(sums.aDen) : nil
            guard vScore != nil || aScore != nil else { continue }
            out.append(LeadershipScore(
                speakerID: speaker,
                valenceLeadership: vScore,
                arousalLeadership: aScore,
                sampleCount: max(sums.vDen, sums.aDen)
            ))
        }
        // Stable sort: descending V leadership when present, else
        // descending A leadership, else speaker id ascending.
        out.sort { lhs, rhs in
            let lv = lhs.valenceLeadership ?? lhs.arousalLeadership ?? -.infinity
            let rv = rhs.valenceLeadership ?? rhs.arousalLeadership ?? -.infinity
            if lv != rv { return lv > rv }
            return lhs.speakerID < rhs.speakerID
        }
        return out
    }

    // MARK: - Plutchik dyads

    /// Plutchik's primary dyad table — adjacent emotions on the
    /// wheel pair into a named compound emotion. Order-independent
    /// (joy+trust == trust+joy = love).
    public enum PlutchikDyad: String, Sendable, Hashable, CaseIterable {
        case love         // joy + trust
        case submission   // trust + fear
        case awe          // fear + surprise
        case disapproval  // surprise + sadness
        case remorse      // sadness + disgust
        case contempt     // disgust + anger
        case aggression   // anger + anticipation
        case optimism     // anticipation + joy

        /// Look up the dyad for an unordered pair of Plutchik
        /// labels. Returns nil for same-emotion pairs (no compound
        /// emotion is defined) and for non-adjacent pairs (those
        /// would land in Plutchik's secondary / tertiary tables,
        /// which we're not surfacing in the first cut).
        public static func from(
            _ a: PlutchikScore.Label,
            _ b: PlutchikScore.Label
        ) -> PlutchikDyad? {
            guard a != b else { return nil }
            let set: Set<PlutchikScore.Label> = [a, b]
            switch set {
            case [.joy, .trust]:         return .love
            case [.trust, .fear]:        return .submission
            case [.fear, .surprise]:     return .awe
            case [.surprise, .sadness]:  return .disapproval
            case [.sadness, .disgust]:   return .remorse
            case [.disgust, .anger]:     return .contempt
            case [.anger, .anticipation]: return .aggression
            case [.anticipation, .joy]:  return .optimism
            default: return nil
            }
        }
    }

    public struct DyadTally: Sendable, Hashable {
        public let dyad: PlutchikDyad
        public let count: Int
        public init(dyad: PlutchikDyad, count: Int) {
            self.dyad = dyad
            self.count = count
        }
    }

    /// Tally Plutchik primary dyads across consecutive speaker-
    /// change turn-pairs. For each pair (leader, follower) the
    /// dominant Plutchik label on each side is fed into the dyad
    /// table; matches contribute to the count. Same-label pairs
    /// (joy↔joy) and non-adjacent pairs (joy↔sadness) are skipped.
    public static func plutchikDyadTallies(
        utterances: [UtteranceEstimate]
    ) -> [DyadTally] {
        guard utterances.count >= 2 else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }
        var counts: [PlutchikDyad: Int] = [:]
        for i in 0..<(sorted.count - 1) {
            let a = sorted[i], b = sorted[i + 1]
            guard a.speakerID != b.speakerID else { continue }
            guard let topA = dominantPlutchikLabel(a),
                  let topB = dominantPlutchikLabel(b) else { continue }
            guard let dyad = PlutchikDyad.from(topA, topB) else { continue }
            counts[dyad, default: 0] += 1
        }
        return counts
            .map { DyadTally(dyad: $0.key, count: $0.value) }
            .sorted { $0.count > $1.count }
    }

    private static func dominantPlutchikLabel(
        _ utterance: UtteranceEstimate
    ) -> PlutchikScore.Label? {
        guard let plutchik = utterance.plutchik else { return nil }
        return plutchik.probabilities.max { lhs, rhs in lhs.value < rhs.value }?.key
    }

    // MARK: - Session arc

    public struct ArcBin: Sendable, Hashable {
        public let midTime: TimeInterval
        public let sessionMeanValence: Double?
        public let sessionMeanArousal: Double?
        /// Per-speaker mean V inside this bin. Sparse — only
        /// entries for speakers who had ≥1 utterance in the bin.
        public let perSpeakerValence: [String: Double]
        public let perSpeakerArousal: [String: Double]

        public init(
            midTime: TimeInterval,
            sessionMeanValence: Double?,
            sessionMeanArousal: Double?,
            perSpeakerValence: [String: Double],
            perSpeakerArousal: [String: Double]
        ) {
            self.midTime = midTime
            self.sessionMeanValence = sessionMeanValence
            self.sessionMeanArousal = sessionMeanArousal
            self.perSpeakerValence = perSpeakerValence
            self.perSpeakerArousal = perSpeakerArousal
        }
    }

    public struct ArcResult: Sendable {
        public let bins: [ArcBin]
        /// Distinct speaker ids in the order they first appear —
        /// used by the UI to assign consistent line colors and to
        /// order the legend.
        public let speakerIDs: [String]
        public let sessionStart: TimeInterval
        public let sessionEnd: TimeInterval

        public init(
            bins: [ArcBin],
            speakerIDs: [String],
            sessionStart: TimeInterval,
            sessionEnd: TimeInterval
        ) {
            self.bins = bins
            self.speakerIDs = speakerIDs
            self.sessionStart = sessionStart
            self.sessionEnd = sessionEnd
        }
    }

    /// Time-binned per-speaker mean V/A plus the session aggregate.
    /// Bins are evenly spaced across `[min start, max end]`. Each
    /// utterance contributes to whichever bin its `start` falls in.
    /// A bin with no utterances has nil session means and empty
    /// per-speaker maps — the UI renders those as gaps in the line.
    public static func sessionArc(
        utterances: [UtteranceEstimate],
        binCount: Int = 16
    ) -> ArcResult {
        guard !utterances.isEmpty, binCount > 0 else {
            return ArcResult(
                bins: [], speakerIDs: [],
                sessionStart: 0, sessionEnd: 0
            )
        }
        let sorted = utterances.sorted { $0.start < $1.start }
        let sessionStart = sorted.first!.start
        let sessionEnd = max(sessionStart, sorted.map(\.end).max() ?? sessionStart)
        let span = max(sessionEnd - sessionStart, 1.0)
        let binWidth = span / Double(binCount)

        var speakerOrder: [String] = []
        var seen: Set<String> = []
        for u in sorted where seen.insert(u.speakerID).inserted {
            speakerOrder.append(u.speakerID)
        }

        // Accumulator per bin: per-speaker running sums + counts.
        struct Accum {
            var vSums: [String: Double] = [:]
            var vCounts: [String: Int] = [:]
            var aSums: [String: Double] = [:]
            var aCounts: [String: Int] = [:]
        }
        var accums = Array(repeating: Accum(), count: binCount)
        for u in sorted {
            let offset = u.start - sessionStart
            var idx = Int((offset / binWidth).rounded(.down))
            // Trailing-edge utterance: clamp the last-bin boundary
            // case so we don't index out of range.
            if idx >= binCount { idx = binCount - 1 }
            if idx < 0 { idx = 0 }
            if let v = u.fusedValence {
                accums[idx].vSums[u.speakerID, default: 0] += Double(v)
                accums[idx].vCounts[u.speakerID, default: 0] += 1
            }
            if let a = u.fusedArousal {
                accums[idx].aSums[u.speakerID, default: 0] += Double(a)
                accums[idx].aCounts[u.speakerID, default: 0] += 1
            }
        }

        var bins: [ArcBin] = []
        bins.reserveCapacity(binCount)
        for i in 0..<binCount {
            let acc = accums[i]
            let mid = sessionStart + (Double(i) + 0.5) * binWidth
            var perSpkV: [String: Double] = [:]
            var perSpkA: [String: Double] = [:]
            var vTotal = 0.0, vN = 0, aTotal = 0.0, aN = 0
            for (spk, sum) in acc.vSums {
                let cnt = acc.vCounts[spk] ?? 1
                let mean = sum / Double(cnt)
                perSpkV[spk] = mean
                vTotal += mean
                vN += 1
            }
            for (spk, sum) in acc.aSums {
                let cnt = acc.aCounts[spk] ?? 1
                let mean = sum / Double(cnt)
                perSpkA[spk] = mean
                aTotal += mean
                aN += 1
            }
            // Session mean is the mean of per-speaker means in this
            // bin — equal-weight across speakers rather than
            // equal-weight across utterances, so a chatty speaker
            // doesn't dominate the line.
            let sessV = vN > 0 ? vTotal / Double(vN) : nil
            let sessA = aN > 0 ? aTotal / Double(aN) : nil
            bins.append(ArcBin(
                midTime: mid,
                sessionMeanValence: sessV,
                sessionMeanArousal: sessA,
                perSpeakerValence: perSpkV,
                perSpeakerArousal: perSpkA
            ))
        }
        return ArcResult(
            bins: bins,
            speakerIDs: speakerOrder,
            sessionStart: sessionStart,
            sessionEnd: sessionEnd
        )
    }

    // MARK: - Pearson helper

    /// Two-pass Pearson for numerical stability. Returns 0 (not nil)
    /// when the denominator collapses — correlation with a constant
    /// signal is mathematically undefined, but "no relationship" is
    /// the right UI read.
    private static func pearson(_ samples: [(Double, Double)]) -> Double {
        let n = Double(samples.count)
        guard n >= 2 else { return 0 }
        var meanX = 0.0, meanY = 0.0
        for (x, y) in samples { meanX += x; meanY += y }
        meanX /= n
        meanY /= n
        var cov = 0.0, varX = 0.0, varY = 0.0
        for (x, y) in samples {
            let dx = x - meanX
            let dy = y - meanY
            cov += dx * dy
            varX += dx * dx
            varY += dy * dy
        }
        let denom = (varX * varY).squareRoot()
        guard denom > 1e-12 else { return 0 }
        return cov / denom
    }
}
