import Foundation

/// Directed-influence and emotional-contagion analyses over a
/// session's utterance list. Three signals corresponding to backlog
/// items #5–#7 in `docs/social_dynamics_backlog.md`:
///
///   5. **Multi-lag leadership graph** — per directed pair `(A → B)`,
///      a single "leadership" correlation aggregated across lags
///      1…N from the existing `AffectiveSynchrony.Result`. Surfaces
///      who shapes the room's emotional weather across multiple
///      delayed-echo windows, not just the lag-1 snapshot.
///   6. **Mood-rescue events** — turns where a speaker's V rose
///      above a recent low-V turn from another speaker. Per-speaker
///      tally reads as "emotional caretaker" tendency.
///   7. **Contagion windows** — periods after a strong-valence seed
///      turn where ≥2 other speakers' V converges toward the seed's
///      V inside a fixed time window.
///
/// Pure functions over the utterance list (and, for #5, the existing
/// synchrony result); no extra ML, no actor isolation.
public enum InfluenceDynamics {

    // MARK: - Multi-lag leadership

    /// One directed-pair entry in the multi-lag leadership graph.
    /// `valenceLeadership` and `arousalLeadership` are
    /// sample-weighted means across the included lags.
    public struct DirectedLeadership: Sendable, Hashable {
        public let leader: String
        public let follower: String
        public let valenceLeadership: Double?
        public let arousalLeadership: Double?
        /// Total sample count summed across the included lags. Pairs
        /// below `AffectiveSynchrony.minSamples` total are excluded.
        public let sampleCount: Int

        public init(
            leader: String,
            follower: String,
            valenceLeadership: Double?,
            arousalLeadership: Double?,
            sampleCount: Int
        ) {
            self.leader = leader
            self.follower = follower
            self.valenceLeadership = valenceLeadership
            self.arousalLeadership = arousalLeadership
            self.sampleCount = sampleCount
        }
    }

    /// Aggregate the per-pair lag profile from `AffectiveSynchrony.compute`
    /// into a single across-lags leadership value per directed pair.
    /// Lag 0 is intentionally excluded — that's immediate-response
    /// synchrony, not leadership in the "others echo this speaker"
    /// sense.
    public static func multiLagLeadership(
        from result: AffectiveSynchrony.Result,
        lags: ClosedRange<Int> = 1...3
    ) -> [DirectedLeadership] {
        let clamped = max(0, lags.lowerBound)...min(result.maxLag, lags.upperBound)
        guard clamped.lowerBound <= clamped.upperBound else { return [] }

        var out: [DirectedLeadership] = []
        out.reserveCapacity(result.pairs.count)
        for entry in result.pairs {
            var vNum = 0.0, vDen = 0
            var aNum = 0.0, aDen = 0
            for k in clamped {
                let n = entry.sampleCountsByLag[k]
                guard n > 0 else { continue }
                if let v = entry.valenceProfile[k] {
                    vNum += v * Double(n)
                    vDen += n
                }
                if let a = entry.arousalProfile[k] {
                    aNum += a * Double(n)
                    aDen += n
                }
            }
            let v: Double? = vDen >= AffectiveSynchrony.minSamples
                ? vNum / Double(vDen) : nil
            let a: Double? = aDen >= AffectiveSynchrony.minSamples
                ? aNum / Double(aDen) : nil
            guard v != nil || a != nil else { continue }
            out.append(DirectedLeadership(
                leader: entry.pair.leader,
                follower: entry.pair.follower,
                valenceLeadership: v,
                arousalLeadership: a,
                sampleCount: max(vDen, aDen)
            ))
        }
        // Strongest positive V leadership first.
        out.sort { lhs, rhs in
            let lv = lhs.valenceLeadership ?? -.infinity
            let rv = rhs.valenceLeadership ?? -.infinity
            if lv != rv { return lv > rv }
            return lhs.leader < rhs.leader
        }
        return out
    }

    // MARK: - Mood rescue

    /// Below this fused valence a turn counts as "low" — eligible
    /// to be the subject of a rescue. Slightly below the V=0.5
    /// midpoint so neutral content doesn't fire the detector.
    public static let rescueLowThreshold: Double = 0.40

    /// Minimum V uplift between the low turn and the rescuer's
    /// next turn. Filters out trivial fluctuations.
    public static let rescueMinUplift: Double = 0.10

    /// Maximum time gap (seconds) from the low-V turn end to the
    /// rescuer's turn start. Past this the response feels too
    /// detached to read as a rescue of *that* moment.
    public static let rescueWindowSec: Double = 30.0

    public struct MoodRescueEvent: Sendable, Hashable {
        public let downedSpeaker: String
        public let downedValence: Double
        public let downedTime: TimeInterval
        public let rescuer: String
        public let rescuerValence: Double
        public let rescuerTime: TimeInterval
    }

    public struct MoodRescueTally: Sendable, Hashable {
        public let speakerID: String
        public let asRescuer: Int
        public let asDowned: Int

        public init(speakerID: String, asRescuer: Int, asDowned: Int) {
            self.speakerID = speakerID
            self.asRescuer = asRescuer
            self.asDowned = asDowned
        }
    }

    /// Detect mood-rescue events. For each low-V turn (V <
    /// `rescueLowThreshold`), scan forward up to `rescueWindowSec`
    /// for the next different-speaker turn whose V exceeds the low
    /// V by at least `rescueMinUplift`. First qualifying turn is the
    /// rescuer; one rescue per low moment.
    public static func moodRescues(
        utterances: [UtteranceEstimate]
    ) -> [MoodRescueEvent] {
        guard !utterances.isEmpty else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }
        var out: [MoodRescueEvent] = []
        for i in 0..<sorted.count {
            let downed = sorted[i]
            guard let dv = downed.fusedValence,
                  Double(dv) < Self.rescueLowThreshold else { continue }
            let downedV = Double(dv)
            // Scan forward until window closes or speaker changes
            // back to the downed speaker (their own next turn isn't
            // a rescue of themselves).
            for j in (i + 1)..<sorted.count {
                let candidate = sorted[j]
                if candidate.start - downed.end > Self.rescueWindowSec {
                    break
                }
                guard candidate.speakerID != downed.speakerID,
                      let cv = candidate.fusedValence else { continue }
                let candidateV = Double(cv)
                if candidateV - downedV >= Self.rescueMinUplift {
                    out.append(MoodRescueEvent(
                        downedSpeaker: downed.speakerID,
                        downedValence: downedV,
                        downedTime: downed.start,
                        rescuer: candidate.speakerID,
                        rescuerValence: candidateV,
                        rescuerTime: candidate.start
                    ))
                    break
                }
            }
        }
        return out
    }

    public static func moodRescueTallies(
        from events: [MoodRescueEvent],
        in utterances: [UtteranceEstimate]
    ) -> [MoodRescueTally] {
        var rescuers: [String: Int] = [:]
        var downed: [String: Int] = [:]
        for e in events {
            rescuers[e.rescuer, default: 0] += 1
            downed[e.downedSpeaker, default: 0] += 1
        }
        let speakers = Set(utterances.map(\.speakerID))
            .union(rescuers.keys).union(downed.keys)
        return speakers.map { spk in
            MoodRescueTally(
                speakerID: spk,
                asRescuer: rescuers[spk] ?? 0,
                asDowned: downed[spk] ?? 0
            )
        }
        .filter { $0.asRescuer > 0 || $0.asDowned > 0 }
        .sorted {
            if $0.asRescuer != $1.asRescuer { return $0.asRescuer > $1.asRescuer }
            return $0.speakerID < $1.speakerID
        }
    }

    // MARK: - Contagion windows

    /// Seed-valence strength threshold. A turn with `|V - 0.5| >`
    /// this counts as a strong-valence moment eligible to seed a
    /// contagion window.
    public static let contagionSeedStrength: Double = 0.20

    /// Window length (seconds) after the seed turn to look for
    /// converging speakers.
    public static let contagionWindowSec: Double = 60.0

    /// Maximum V distance from the seed at which a follower's first
    /// in-window turn counts as "aligned." 0.20 ≈ a quarter of the V
    /// scale [0, 1] minus the half-width above and below the seed,
    /// so a seed of 0.85 catches followers in [0.65, 1.0].
    public static let contagionAlignDistance: Double = 0.20

    /// Distinct other speakers (beyond the seed) required for the
    /// seed to register as a contagion event. Backlog spec says
    /// "≥3 speakers converge" total → 2 followers + seed.
    public static let contagionMinFollowers: Int = 2

    public enum ContagionDirection: String, Sendable, Hashable {
        case positive  // seed V above 0.5; followers also above
        case negative  // seed V below 0.5; followers also below
    }

    public struct ContagionWindow: Sendable, Hashable {
        public let seedSpeaker: String
        public let seedTime: TimeInterval
        public let seedValence: Double
        public let direction: ContagionDirection
        public let windowEnd: TimeInterval
        /// Distinct other-speaker IDs whose first in-window turn
        /// landed within `contagionAlignDistance` of the seed.
        public let followers: [String]
    }

    /// Detect contagion windows. Each seed turn (a strong-valence
    /// utterance) opens a `contagionWindowSec` window; we look at the
    /// first turn each other speaker takes inside that window and
    /// count it if it's within `contagionAlignDistance` of the seed.
    /// Seeds that recruit ≥`contagionMinFollowers` distinct others
    /// register as a contagion window.
    public static func contagionWindows(
        utterances: [UtteranceEstimate]
    ) -> [ContagionWindow] {
        guard utterances.count >= 3 else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }
        var out: [ContagionWindow] = []

        for i in 0..<sorted.count {
            let seed = sorted[i]
            guard let sv = seed.fusedValence else { continue }
            let seedV = Double(sv)
            guard abs(seedV - 0.5) > Self.contagionSeedStrength else { continue }
            let windowEnd = seed.start + Self.contagionWindowSec
            var seenFollowers: Set<String> = []
            var alignedFollowers: [String] = []
            for j in (i + 1)..<sorted.count {
                let candidate = sorted[j]
                if candidate.start > windowEnd { break }
                guard candidate.speakerID != seed.speakerID,
                      !seenFollowers.contains(candidate.speakerID),
                      let cv = candidate.fusedValence else { continue }
                seenFollowers.insert(candidate.speakerID)
                let candidateV = Double(cv)
                if abs(candidateV - seedV) <= Self.contagionAlignDistance {
                    alignedFollowers.append(candidate.speakerID)
                }
            }
            guard alignedFollowers.count >= Self.contagionMinFollowers else { continue }
            out.append(ContagionWindow(
                seedSpeaker: seed.speakerID,
                seedTime: seed.start,
                seedValence: seedV,
                direction: seedV >= 0.5 ? .positive : .negative,
                windowEnd: windowEnd,
                followers: alignedFollowers
            ))
        }

        // Suppress overlapping seeds. When two seed turns sit close
        // together and would describe the same recruitment event,
        // keep the earlier one — the second seed inherits the
        // already-aligned room rather than driving fresh contagion.
        var deduped: [ContagionWindow] = []
        deduped.reserveCapacity(out.count)
        var lastEnd: TimeInterval = -.infinity
        for w in out where w.seedTime >= lastEnd {
            deduped.append(w)
            lastEnd = w.windowEnd
        }
        return deduped
    }
}
