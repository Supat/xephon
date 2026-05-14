import Foundation

/// Per-speaker behavioral fingerprint over a session's utterance
/// list. Five dimensions chosen to span distinct behavioral axes
/// (volume, valence, volatility, leadership, responsiveness)
/// without overlap:
///
///   1. **Talk share** — fraction of total speaking time spent by
///      this speaker. Pure timeline metric.
///   2. **Mean valence** — average fused V across this speaker's
///      utterances. "Is this person typically positive?"
///   3. **Lability** — `sqrt(varV + varA)`, a single scalar for
///      emotional volatility. High = swings widely; low = even keel.
///   4. **Leadership** — sample-weighted lag-1 valence correlation
///      from `AffectiveSynchrony.leadershipScores`. High = others
///      tend to echo this speaker's emotion one turn later.
///   5. **Response quickness** — inverted median response latency
///      (gap between when another speaker finished and this one
///      started). High = fast responder.
///
/// All five are returned in two forms:
///   - `raw`: the actual value (seconds, [0..1] valence, etc.) for
///     popover / inspector display.
///   - `normalized`: rank-normalized to [0,1] within the session
///     for bar display. Comparing speakers is what matters here —
///     absolute V/A numbers from cross-lingual SER models aren't
///     trustworthy enough to put on a bar (see CLAUDE.md: "treat
///     zero-shot outputs as relative, not absolute").
public enum SpeakerBehavior {
    public struct RawMetrics: Sendable, Hashable {
        /// Fraction of total speaking time in [0, 1].
        public let talkTimeShare: Double
        /// Mean fused valence in [0, 1]. Nil when no utterance from
        /// this speaker carried V.
        public let meanValence: Double?
        /// `sqrt(varV + varA)` — combined V/A standard deviation
        /// scalar. Nil when fewer than 2 utterances had V or A.
        public let lability: Double?
        /// Lag-1 valence leadership correlation in [-1, 1]. Nil
        /// when the speaker has no qualifying pairs.
        public let leadership: Double?
        /// Median response-latency in seconds. Nil when this
        /// speaker never responded to another speaker's turn.
        public let medianResponseLatency: Double?
    }

    public struct NormalizedMetrics: Sendable, Hashable {
        public let talkTimeShare: Double
        public let meanValence: Double?
        public let lability: Double?
        public let leadership: Double?
        /// Inverted from `medianResponseLatency` — high = fast.
        public let respondQuickness: Double?
    }

    public struct Profile: Sendable, Hashable {
        public let speakerID: String
        public let raw: RawMetrics
        public let normalized: NormalizedMetrics
        public let utteranceCount: Int

        public init(
            speakerID: String,
            raw: RawMetrics,
            normalized: NormalizedMetrics,
            utteranceCount: Int
        ) {
            self.speakerID = speakerID
            self.raw = raw
            self.normalized = normalized
            self.utteranceCount = utteranceCount
        }
    }

    public static func computeProfiles(
        utterances: [UtteranceEstimate]
    ) -> [Profile] {
        guard !utterances.isEmpty else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }

        // Single pass over utterances builds per-speaker accumulators
        // for talk time, count, and sufficient statistics for V/A
        // variance. Two-pass mean-then-variance would be more
        // numerically stable, but session lengths cap in the hundreds
        // of utterances — well inside Float64's precision budget.
        struct Accum {
            var talkTime: Double = 0
            var utteranceCount: Int = 0
            var vSum: Double = 0
            var vSqSum: Double = 0
            var vCount: Int = 0
            var aSum: Double = 0
            var aSqSum: Double = 0
            var aCount: Int = 0
        }
        var accs: [String: Accum] = [:]
        for u in sorted {
            var a = accs[u.speakerID, default: Accum()]
            a.talkTime += max(0, u.end - u.start)
            a.utteranceCount += 1
            if let v = u.fusedValence {
                let dv = Double(v)
                a.vSum += dv
                a.vSqSum += dv * dv
                a.vCount += 1
            }
            if let av = u.fusedArousal {
                let da = Double(av)
                a.aSum += da
                a.aSqSum += da * da
                a.aCount += 1
            }
            accs[u.speakerID] = a
        }
        let totalTalkTime = accs.values.reduce(0) { $0 + $1.talkTime }

        // Reuse the existing synchrony compute for leadership — it's
        // O(turns × maxLag) and we'd otherwise be duplicating the
        // turn-pair scan. Lag 1 is the canonical "emotional
        // leadership" reading (one intervening turn).
        let sync = AffectiveSynchrony.compute(utterances: utterances)
        let leaders = AffectiveSynchrony.leadershipScores(from: sync, atLag: 1)
        var leadershipByID: [String: Double] = [:]
        for s in leaders {
            // Prefer V leadership; fall back to A when V is nil so
            // the column doesn't go blank for a speaker who has
            // arousal-side influence without a valence-side signal.
            if let v = s.valenceLeadership {
                leadershipByID[s.speakerID] = v
            } else if let a = s.arousalLeadership {
                leadershipByID[s.speakerID] = a
            }
        }

        // Per-speaker response latencies: for each consecutive
        // turn-pair (prev, curr) with different speakers, attribute
        // the gap to `curr.speakerID` — they're the one responding.
        // Negative gaps (overlap from diarization noise) clamp to 0.
        var latencies: [String: [Double]] = [:]
        for i in 1..<sorted.count {
            let prev = sorted[i - 1]
            let curr = sorted[i]
            guard prev.speakerID != curr.speakerID else { continue }
            let gap = max(0, curr.start - prev.end)
            latencies[curr.speakerID, default: []].append(gap)
        }
        var medianLatencyByID: [String: Double] = [:]
        for (spk, lats) in latencies where !lats.isEmpty {
            medianLatencyByID[spk] = median(lats)
        }

        // Build raw metrics. Variance: E[X²] - E[X]² with a
        // Bessel-style correction is overkill for n ≥ 10 utterances
        // and underflows numerically for n ≤ 2; the population form
        // is honest for our normalization needs.
        var rawByID: [String: RawMetrics] = [:]
        for (spk, acc) in accs {
            let talkShare = totalTalkTime > 0
                ? acc.talkTime / totalTalkTime
                : 0
            let meanV: Double? = acc.vCount > 0 ? acc.vSum / Double(acc.vCount) : nil
            let meanA: Double? = acc.aCount > 0 ? acc.aSum / Double(acc.aCount) : nil
            let varV: Double? = {
                guard acc.vCount >= 2, let m = meanV else { return nil }
                let v = acc.vSqSum / Double(acc.vCount) - m * m
                return max(0, v)
            }()
            let varA: Double? = {
                guard acc.aCount >= 2, let m = meanA else { return nil }
                let v = acc.aSqSum / Double(acc.aCount) - m * m
                return max(0, v)
            }()
            let lability: Double? = {
                guard varV != nil || varA != nil else { return nil }
                return ((varV ?? 0) + (varA ?? 0)).squareRoot()
            }()
            rawByID[spk] = RawMetrics(
                talkTimeShare: talkShare,
                meanValence: meanV,
                lability: lability,
                leadership: leadershipByID[spk],
                medianResponseLatency: medianLatencyByID[spk]
            )
        }

        // Rank-normalize each metric independently across speakers.
        // Latency inverts: fastest responder gets the full bar.
        let normTalk = rankNormalize(rawByID.mapValues { $0.talkTimeShare })
        let normV = rankNormalize(rawByID.compactMapValues { $0.meanValence })
        let normLab = rankNormalize(rawByID.compactMapValues { $0.lability })
        let normLead = rankNormalize(rawByID.compactMapValues { $0.leadership })
        let normLatency = rankNormalize(rawByID.compactMapValues { $0.medianResponseLatency })
        let normResp = normLatency.mapValues { 1.0 - $0 }

        // Output order = first-appearance in the conversation so the
        // card lists speakers as the user encountered them, matching
        // the roster and arc cards' conventions.
        var seen: Set<String> = []
        var order: [String] = []
        for u in sorted where seen.insert(u.speakerID).inserted {
            order.append(u.speakerID)
        }

        var profiles: [Profile] = []
        profiles.reserveCapacity(order.count)
        for spk in order {
            guard let raw = rawByID[spk], let acc = accs[spk] else { continue }
            profiles.append(Profile(
                speakerID: spk,
                raw: raw,
                normalized: NormalizedMetrics(
                    talkTimeShare: normTalk[spk] ?? 0,
                    meanValence: normV[spk],
                    lability: normLab[spk],
                    leadership: normLead[spk],
                    respondQuickness: normResp[spk]
                ),
                utteranceCount: acc.utteranceCount
            ))
        }
        return profiles
    }

    /// Rank-normalize values to [0, 1] across the keyed map. Single-
    /// entry maps return 0.5 (the bar reads as "no signal, middle"
    /// rather than "full bar misleadingly"). Ties get distinct
    /// neighboring ranks rather than averaged — the bar visualization
    /// reads better when each speaker has a unique height even on a
    /// 0.001-apart tie.
    private static func rankNormalize(
        _ values: [String: Double]
    ) -> [String: Double] {
        guard !values.isEmpty else { return [:] }
        if values.count == 1, let key = values.keys.first {
            return [key: 0.5]
        }
        let sorted = values.sorted { $0.value < $1.value }
        var out: [String: Double] = [:]
        for (i, entry) in sorted.enumerated() {
            out[entry.key] = Double(i) / Double(sorted.count - 1)
        }
        return out
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let s = values.sorted()
        let n = s.count
        return n % 2 == 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[n / 2]
    }
}
