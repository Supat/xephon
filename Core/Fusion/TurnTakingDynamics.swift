import Foundation

/// Turn-taking analyses over a session's utterance list. Four
/// per-session signals chosen to surface distinct social-dynamics
/// axes, mirroring the backlog items #1–#4 in
/// `docs/social_dynamics_backlog.md`:
///
///   1. **Interruptions** — directed `(A → B)` count: A started
///      speaking while B still held the floor and A's utterance is
///      not a backchannel.
///   2. **Floor-holding** — per-speaker distribution of consecutive
///      run lengths (in seconds), where a run = a maximal stretch
///      of utterances by one speaker not broken by another speaker's
///      non-backchannel turn. Backchannels from other speakers do
///      not break a run.
///   3. **Backchannels** — per-speaker count + rate of utterances
///      whose transcript matches the canonical filler set.
///   4. **Response latency** — directed `(responder → partner)`
///      median gap (seconds) between a partner's turn end and the
///      responder's next turn start. Bounded by
///      `maxResponseWindowSec`; longer gaps are new initiatives,
///      not responses.
///
/// All four are pure functions of the utterance list — no audio
/// re-decoding, no extra ML — so they're cheap enough to recompute
/// per render. View layer can memoize on `utterancesVersion` if
/// per-frame cost ever shows up.
public enum TurnTakingDynamics {

    public struct InterruptionPair: Sendable, Hashable {
        /// The speaker who cut in.
        public let interrupter: String
        /// The speaker who was holding the floor.
        public let victim: String
        public let count: Int

        public init(interrupter: String, victim: String, count: Int) {
            self.interrupter = interrupter
            self.victim = victim
            self.count = count
        }
    }

    public struct PairLatency: Sendable, Hashable {
        /// Speaker whose utterance start time we measured.
        public let responder: String
        /// Speaker who immediately preceded the responder.
        public let partner: String
        public let medianSeconds: Double
        public let sampleCount: Int

        public init(
            responder: String,
            partner: String,
            medianSeconds: Double,
            sampleCount: Int
        ) {
            self.responder = responder
            self.partner = partner
            self.medianSeconds = medianSeconds
            self.sampleCount = sampleCount
        }
    }

    public struct FloorHolding: Sendable, Hashable {
        public let speakerID: String
        public let runCount: Int
        public let totalSeconds: Double
        public let medianSeconds: Double
        public let maxSeconds: Double

        public init(
            speakerID: String,
            runCount: Int,
            totalSeconds: Double,
            medianSeconds: Double,
            maxSeconds: Double
        ) {
            self.speakerID = speakerID
            self.runCount = runCount
            self.totalSeconds = totalSeconds
            self.medianSeconds = medianSeconds
            self.maxSeconds = maxSeconds
        }
    }

    public struct Backchannel: Sendable, Hashable {
        public let speakerID: String
        public let backchannelCount: Int
        public let totalUtterances: Int

        public init(
            speakerID: String,
            backchannelCount: Int,
            totalUtterances: Int
        ) {
            self.speakerID = speakerID
            self.backchannelCount = backchannelCount
            self.totalUtterances = totalUtterances
        }

        public var rate: Double {
            totalUtterances > 0
                ? Double(backchannelCount) / Double(totalUtterances)
                : 0
        }
    }

    public struct Profile: Sendable, Hashable {
        public let interruptions: [InterruptionPair]
        public let floorHolding: [FloorHolding]
        public let backchannels: [Backchannel]
        public let responseLatencies: [PairLatency]

        public init(
            interruptions: [InterruptionPair],
            floorHolding: [FloorHolding],
            backchannels: [Backchannel],
            responseLatencies: [PairLatency]
        ) {
            self.interruptions = interruptions
            self.floorHolding = floorHolding
            self.backchannels = backchannels
            self.responseLatencies = responseLatencies
        }
    }

    /// Maximum gap (seconds) between two cross-speaker utterances
    /// still classified as a response. Longer gaps would dilute
    /// per-pair medians with what are effectively new conversation
    /// initiatives. 10 s is generous for a "thinking" beat in
    /// Japanese conversation while still excluding genuine
    /// topic-shift silences.
    public static let maxResponseWindowSec: Double = 10

    /// Canonical backchannel set. Same tokens
    /// `AnalysisPipeline.fillers` uses to skip text SER on these
    /// rows; surfaced publicly here so the dynamics module owns
    /// its own copy without depending on the app target. Match is
    /// trimmed + exact (no substring) — these are short, distinct
    /// forms whose acoustic content is conventionalized as a
    /// continuer rather than a turn move.
    public static let backchannelTokens: Set<String> = [
        "あの", "えーと", "えっと", "えと", "うーん", "うんうん",
        "うん", "ええ", "はい", "いえ", "そう", "そうそう",
        "そうですね", "なるほど", "ふむ", "へえ", "ああ", "おお",
    ]

    public static func isBackchannel(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return backchannelTokens.contains(trimmed)
    }

    public static func compute(utterances: [UtteranceEstimate]) -> Profile {
        guard !utterances.isEmpty else {
            return Profile(
                interruptions: [],
                floorHolding: [],
                backchannels: [],
                responseLatencies: []
            )
        }
        let sorted = utterances.sorted { $0.start < $1.start }

        // (#1) interruption counts, keyed (interrupter, victim).
        var interruptionCounts: [InterruptionKey: Int] = [:]
        // (#2) floor-run lengths per speaker, in seconds. Run state
        // carries across iterations.
        var runsBySpeaker: [String: [Double]] = [:]
        var currentSpeaker: String? = nil
        var currentStart: TimeInterval = 0
        var currentLastEnd: TimeInterval = 0
        // (#3) backchannel + total counts per speaker.
        var backchannelCounts: [String: Int] = [:]
        var totalCounts: [String: Int] = [:]
        // (#4) latency samples per directed pair.
        var latenciesByPair: [PairKey: [Double]] = [:]

        for (i, u) in sorted.enumerated() {
            totalCounts[u.speakerID, default: 0] += 1
            let bc = isBackchannel(u.transcript)
            if bc { backchannelCounts[u.speakerID, default: 0] += 1 }

            if i > 0 {
                let prev = sorted[i - 1]
                let differentSpeaker = prev.speakerID != u.speakerID
                let overlaps = u.start < prev.end
                let gap = u.start - prev.end
                if differentSpeaker {
                    if overlaps, !bc {
                        // (#1) Real interruption — A cuts in on B
                        // while B is still talking, and A isn't
                        // just a backchannel "うん."
                        let key = InterruptionKey(
                            interrupter: u.speakerID,
                            victim: prev.speakerID
                        )
                        interruptionCounts[key, default: 0] += 1
                    } else if !overlaps, gap >= 0, gap <= Self.maxResponseWindowSec {
                        // (#4) Response latency — A speaks within a
                        // window of B finishing.
                        let key = PairKey(
                            responder: u.speakerID,
                            partner: prev.speakerID
                        )
                        latenciesByPair[key, default: []].append(gap)
                    }
                }
            }

            // (#2) Floor-run accumulation. Backchannels don't take
            // the floor — they're acknowledgments while someone else
            // is still the floor-holder — so we skip them entirely
            // for run tracking.
            if bc { continue }
            if let cur = currentSpeaker {
                if u.speakerID == cur {
                    currentLastEnd = max(currentLastEnd, u.end)
                } else {
                    runsBySpeaker[cur, default: []].append(
                        max(0, currentLastEnd - currentStart)
                    )
                    currentSpeaker = u.speakerID
                    currentStart = u.start
                    currentLastEnd = u.end
                }
            } else {
                currentSpeaker = u.speakerID
                currentStart = u.start
                currentLastEnd = u.end
            }
        }
        // Trailing-run flush.
        if let cur = currentSpeaker {
            runsBySpeaker[cur, default: []].append(
                max(0, currentLastEnd - currentStart)
            )
        }

        let interruptions = interruptionCounts.map { key, count in
            InterruptionPair(
                interrupter: key.interrupter,
                victim: key.victim,
                count: count
            )
        }.sorted { $0.count > $1.count }

        let floorHolding = runsBySpeaker.map { spk, runs in
            FloorHolding(
                speakerID: spk,
                runCount: runs.count,
                totalSeconds: runs.reduce(0, +),
                medianSeconds: Self.median(runs),
                maxSeconds: runs.max() ?? 0
            )
        }.sorted { $0.totalSeconds > $1.totalSeconds }

        let backchannels = totalCounts.map { spk, total in
            Backchannel(
                speakerID: spk,
                backchannelCount: backchannelCounts[spk] ?? 0,
                totalUtterances: total
            )
        }.sorted { $0.rate > $1.rate }

        let responseLatencies = latenciesByPair.map { key, lats in
            PairLatency(
                responder: key.responder,
                partner: key.partner,
                medianSeconds: Self.median(lats),
                sampleCount: lats.count
            )
        }.sorted { $0.medianSeconds < $1.medianSeconds }

        return Profile(
            interruptions: interruptions,
            floorHolding: floorHolding,
            backchannels: backchannels,
            responseLatencies: responseLatencies
        )
    }

    private struct InterruptionKey: Hashable {
        let interrupter: String
        let victim: String
    }

    private struct PairKey: Hashable {
        let responder: String
        let partner: String
    }

    private static func median(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        let s = values.sorted()
        let n = s.count
        return n % 2 == 0 ? (s[n / 2 - 1] + s[n / 2]) / 2 : s[n / 2]
    }
}
