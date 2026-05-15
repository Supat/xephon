import Foundation

/// "How does each speaker react to the session around them?" —
/// backlog items #12–#14 in `docs/social_dynamics_backlog.md`:
///
///  12. **Response vs. initiative.** Each turn is classified as a
///      response (started within `responseGapThresholdSec` of
///      another speaker's turn end) or an initiative (after a long
///      silence, or the very first turn). Per-speaker ratio
///      reveals reactive vs. agenda-setting role.
///  13. **Recovery time after negative valence.** For each
///      speaker, the median time from a sub-threshold V turn to
///      their next turn at-or-above their session-baseline V.
///      Resilience metric.
///  14. **Reaction to interruption.** When speaker A is interrupted
///      by speaker B (B starts before A finishes), look at A's
///      *next* turn after the interruption: how does A's V compare
///      to A's V on the interrupted turn? Negative delta = A's
///      affect drops after being talked over.
///
/// Pure derivation over the utterance list. No actor isolation.
public enum ReactivityDynamics {

    // MARK: - #12 Response vs. initiative

    public enum Role: String, Sendable, Hashable {
        case response, initiative
    }

    /// Maximum gap (seconds) from another speaker's turn end to this
    /// turn's start for the row to count as a "response." Longer
    /// gaps read as a new initiative. Same window as
    /// `TurnTakingDynamics.maxResponseWindowSec`'s spirit but
    /// tighter — turn-taking is about "did they reply at all,"
    /// while role classification asks "was it tight enough to feel
    /// like a reply?"
    public static let responseGapThresholdSec: Double = 3.0

    /// Per-utterance role lookup. Keyed by `utterance.id` so the
    /// row chip can resolve in O(1).
    public static func roleClassifications(
        utterances: [UtteranceEstimate]
    ) -> [UUID: Role] {
        guard !utterances.isEmpty else { return [:] }
        let sorted = utterances.sorted { $0.start < $1.start }
        var out: [UUID: Role] = [:]
        // Track the most recent end-of-turn for each OTHER speaker —
        // a turn is a response to whichever cross-speaker turn ended
        // closest before its start, regardless of who else might
        // have spoken in between.
        var lastEndByOther: [String: TimeInterval] = [:]
        for (i, u) in sorted.enumerated() {
            // First turn is always an initiative — no preceding
            // cross-speaker turn to respond to.
            if i == 0 {
                out[u.id] = .initiative
                lastEndByOther[u.speakerID] = u.end
                continue
            }
            // Look at the most recent end across speakers other
            // than this one.
            var bestEnd: TimeInterval = -.infinity
            for (spk, end) in lastEndByOther where spk != u.speakerID {
                if end > bestEnd { bestEnd = end }
            }
            let role: Role
            if bestEnd.isFinite,
               u.start - bestEnd <= Self.responseGapThresholdSec,
               u.start - bestEnd >= -Self.responseGapThresholdSec {
                role = .response
            } else {
                role = .initiative
            }
            out[u.id] = role
            // Update the lookup. We update for THIS speaker even
            // though the next turn from them isn't measured against
            // themselves — keeps the data structure simple and the
            // filter above already skips same-speaker entries.
            lastEndByOther[u.speakerID] = max(
                lastEndByOther[u.speakerID] ?? -.infinity,
                u.end
            )
        }
        return out
    }

    public struct RoleRatio: Sendable, Hashable {
        public let speakerID: String
        public let responses: Int
        public let initiatives: Int

        public init(speakerID: String, responses: Int, initiatives: Int) {
            self.speakerID = speakerID
            self.responses = responses
            self.initiatives = initiatives
        }

        /// `responses / (responses + initiatives)`. Convenience.
        public var responseShare: Double {
            let total = responses + initiatives
            return total > 0 ? Double(responses) / Double(total) : 0
        }
    }

    public static func roleRatios(
        utterances: [UtteranceEstimate]
    ) -> [RoleRatio] {
        let roles = roleClassifications(utterances: utterances)
        var responses: [String: Int] = [:]
        var initiatives: [String: Int] = [:]
        for u in utterances {
            switch roles[u.id] {
            case .response: responses[u.speakerID, default: 0] += 1
            case .initiative: initiatives[u.speakerID, default: 0] += 1
            case nil: break
            }
        }
        let speakers = Set(responses.keys).union(initiatives.keys)
        return speakers.map { spk in
            RoleRatio(
                speakerID: spk,
                responses: responses[spk] ?? 0,
                initiatives: initiatives[spk] ?? 0
            )
        }
        .sorted {
            if $0.responseShare != $1.responseShare {
                return $0.responseShare > $1.responseShare
            }
            return $0.speakerID < $1.speakerID
        }
    }

    // MARK: - #13 Recovery time after negative valence

    /// Threshold below which a turn counts as a "low" moment whose
    /// recovery time we measure. Same value `InfluenceDynamics`
    /// uses for the mood-rescue subject — both analyses describe
    /// the same regime of speaker valence.
    public static let lowValenceThreshold: Double = 0.40

    /// Maximum recovery window (seconds) — if the speaker doesn't
    /// recover to baseline within this, the event is dropped from
    /// the recovery-time sample (it's too detached to read as
    /// recovery of *that* moment).
    public static let recoveryMaxWindowSec: Double = 180.0

    public struct RecoveryTally: Sendable, Hashable {
        public let speakerID: String
        public let medianRecoverySec: Double?
        /// Number of low-V events that recovered within the window.
        public let recoveredCount: Int
        /// Number of low-V events that did NOT recover within the
        /// window (speaker either stayed below baseline or stopped
        /// speaking before recovering).
        public let unresolvedCount: Int

        public init(
            speakerID: String,
            medianRecoverySec: Double?,
            recoveredCount: Int,
            unresolvedCount: Int
        ) {
            self.speakerID = speakerID
            self.medianRecoverySec = medianRecoverySec
            self.recoveredCount = recoveredCount
            self.unresolvedCount = unresolvedCount
        }
    }

    /// For each speaker, find their session baseline V (mean V across
    /// their utterances). Then for each of their sub-threshold turns,
    /// scan forward through THEIR own turns until V ≥ baseline (or
    /// the window expires). Sample the recovery latencies and
    /// report the median.
    public static func recoveryTimes(
        utterances: [UtteranceEstimate]
    ) -> [RecoveryTally] {
        guard !utterances.isEmpty else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }

        // Speaker baseline V = mean V across the speaker's turns.
        var vSums: [String: Double] = [:]
        var vCounts: [String: Int] = [:]
        for u in sorted {
            guard let v = u.fusedValence else { continue }
            vSums[u.speakerID, default: 0] += Double(v)
            vCounts[u.speakerID, default: 0] += 1
        }
        var baselineBySpeaker: [String: Double] = [:]
        for (spk, sum) in vSums {
            let n = vCounts[spk] ?? 1
            baselineBySpeaker[spk] = sum / Double(n)
        }

        // Index utterances by speaker so we can walk each speaker's
        // own timeline.
        var bySpeaker: [String: [UtteranceEstimate]] = [:]
        for u in sorted {
            bySpeaker[u.speakerID, default: []].append(u)
        }

        var out: [RecoveryTally] = []
        for (spk, turns) in bySpeaker {
            guard let baseline = baselineBySpeaker[spk] else { continue }
            var latencies: [Double] = []
            var unresolved = 0
            for i in 0..<turns.count {
                let lo = turns[i]
                guard let lv = lo.fusedValence,
                      Double(lv) < Self.lowValenceThreshold else { continue }
                var recovered = false
                for j in (i + 1)..<turns.count {
                    let candidate = turns[j]
                    let gap = candidate.start - lo.end
                    if gap > Self.recoveryMaxWindowSec { break }
                    guard let cv = candidate.fusedValence else { continue }
                    if Double(cv) >= baseline {
                        latencies.append(max(0, gap))
                        recovered = true
                        break
                    }
                }
                if !recovered { unresolved += 1 }
            }
            guard !latencies.isEmpty || unresolved > 0 else { continue }
            out.append(RecoveryTally(
                speakerID: spk,
                medianRecoverySec: latencies.isEmpty ? nil : Self.median(latencies),
                recoveredCount: latencies.count,
                unresolvedCount: unresolved
            ))
        }
        return out.sorted {
            let l = $0.medianRecoverySec ?? .infinity
            let r = $1.medianRecoverySec ?? .infinity
            if l != r { return l < r }
            return $0.speakerID < $1.speakerID
        }
    }

    // MARK: - #14 Reaction to interruption

    public struct InterruptionReaction: Sendable, Hashable {
        public let speakerID: String  // the one who was interrupted
        /// Mean V on the speaker's next turn after being
        /// interrupted, minus their V on the interrupted turn.
        /// Negative = drop after being talked over.
        public let meanValenceDelta: Double
        /// Number of (interrupted turn → next turn) pairs sampled.
        public let sampleCount: Int

        public init(speakerID: String, meanValenceDelta: Double, sampleCount: Int) {
            self.speakerID = speakerID
            self.meanValenceDelta = meanValenceDelta
            self.sampleCount = sampleCount
        }
    }

    /// Detect interruptions (same rule as
    /// `TurnTakingDynamics.compute`'s interruption pass: a cross-
    /// speaker turn starting before the previous turn ends, and the
    /// interrupter's turn isn't itself a backchannel) and for each
    /// such event measure the *victim's* next-turn V minus their
    /// interrupted-turn V.
    public static func interruptionReactions(
        utterances: [UtteranceEstimate]
    ) -> [InterruptionReaction] {
        guard utterances.count >= 2 else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }

        // Pre-build a per-speaker index → original-sorted-index map
        // so we can find a speaker's next turn after a given index
        // without a linear scan each time.
        var nextTurnByID: [UUID: UtteranceEstimate] = [:]
        var lastSeenIndexBySpeaker: [String: Int] = [:]
        for i in stride(from: sorted.count - 1, through: 0, by: -1) {
            let u = sorted[i]
            if let _ = lastSeenIndexBySpeaker[u.speakerID] {
                // For *this* utterance, the next-by-speaker is the
                // one we last saw walking backwards from the end.
                // We'll patch it in below.
            }
            // Defer the binding — easiest single pass: just lookup
            // forward from i+1 the first matching speakerID.
            for j in (i + 1)..<sorted.count {
                if sorted[j].speakerID == u.speakerID {
                    nextTurnByID[u.id] = sorted[j]
                    break
                }
            }
            lastSeenIndexBySpeaker[u.speakerID] = i
        }

        // Walk consecutive pairs, detect interruptions, and aggregate
        // the victim's V deltas.
        var deltasBySpeaker: [String: [Double]] = [:]
        for i in 1..<sorted.count {
            let prev = sorted[i - 1]
            let curr = sorted[i]
            guard prev.speakerID != curr.speakerID,
                  curr.start < prev.end,
                  !TurnTakingDynamics.isBackchannel(curr.transcript) else { continue }
            // `prev` is the victim. Find their next turn.
            guard let next = nextTurnByID[prev.id],
                  let pv = prev.fusedValence,
                  let nv = next.fusedValence else { continue }
            deltasBySpeaker[prev.speakerID, default: []].append(
                Double(nv) - Double(pv)
            )
        }
        return deltasBySpeaker.map { spk, deltas in
            let mean = deltas.reduce(0, +) / Double(deltas.count)
            return InterruptionReaction(
                speakerID: spk,
                meanValenceDelta: mean,
                sampleCount: deltas.count
            )
        }
        .sorted {
            if $0.meanValenceDelta != $1.meanValenceDelta {
                return $0.meanValenceDelta < $1.meanValenceDelta
            }
            return $0.speakerID < $1.speakerID
        }
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let s = values.sorted()
        let n = s.count
        return n % 2 == 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[n / 2]
    }
}
