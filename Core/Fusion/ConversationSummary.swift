import Foundation

/// Running summary of a session's utterances, updated incrementally as
/// each `UtteranceEstimate` arrives (O(1) per fold). Tracks
/// duration-weighted mean V/A/D, an ASR-confidence-weighted dominant
/// label, and a bounded recent trajectory for the V/A sparkline.
///
/// Std-dev is computed on demand from the bounded trajectory ring
/// (≤ `trajectoryCap` entries) so the summary itself stays compact even
/// for hour-long sessions.
public struct ConversationSummary: Sendable, Hashable {
    public private(set) var utteranceCount: Int = 0
    public private(set) var totalDuration: TimeInterval = 0
    public private(set) var trajectory: [TrajectoryPoint] = []
    public private(set) var labelScores: [String: Float] = [:]
    /// Raw count of utterances per fused top label. Distinct from
    /// `labelScores` (which is confidence-weighted) — counts are useful
    /// for the Statistics panel where the user wants "how many times
    /// did the model produce this label" without ASR-confidence
    /// dilution.
    public private(set) var labelCounts: [String: Int] = [:]

    // Per-modality weighted accumulators. V/A/D can independently be nil
    // when their producing modality fails, so each carries its own
    // weight-sum so the means stay correct under partial coverage.
    private var sumV: Double = 0
    private var weightV: Double = 0
    private var sumA: Double = 0
    private var weightA: Double = 0
    private var sumD: Double = 0
    private var weightD: Double = 0

    public struct TrajectoryPoint: Sendable, Hashable, Codable {
        public let time: TimeInterval   // utterance start (audio-time seconds)
        public let valence: Float       // [0, 1], 0.5 = neutral
        public let arousal: Float       // [0, 1], 0.5 = neutral
    }

    /// Last ~60 utterances, ~5 min of typical conversation. Bounded so
    /// memory stays constant for long sessions.
    public static let trajectoryCap = 60
    /// Below this count, the running means are noisy enough that we want
    /// the UI to flag "calibrating" rather than display them with
    /// false precision.
    public static let calibrationThreshold = 3

    public var meanValence:   Float? { weightV > 0 ? Float(sumV / weightV) : nil }
    public var meanArousal:   Float? { weightA > 0 ? Float(sumA / weightA) : nil }
    public var meanDominance: Float? { weightD > 0 ? Float(sumD / weightD) : nil }

    /// Argmax over the confidence-weighted label scores. Nil until at
    /// least one utterance has produced a fused top label.
    public var topLabel: String? {
        labelScores.max(by: { $0.value < $1.value })?.key
    }

    /// Population standard deviation of the trajectory's valence values
    /// (recent window only — the bounded ring is the right scale for
    /// "how variable has the mood been *lately*"). Nil when we don't
    /// have at least two points to compare.
    public var valenceStdDev: Float? { Self.populationStdDev(trajectory.map { Double($0.valence) }) }
    public var arousalStdDev: Float? { Self.populationStdDev(trajectory.map { Double($0.arousal) }) }

    public init() {}

    /// Fold one utterance into the running summary. Idempotent across
    /// calls with the same input only because each call mutates the
    /// accumulators — callers should ensure each utterance is folded
    /// exactly once (RecordingController.applySegmentResult does).
    public mutating func update(with u: UtteranceEstimate) {
        // Floor the per-utterance duration to avoid divide-by-zero from
        // pathological zero-length finals (Apple has been seen to emit
        // them at session boundaries). 0.1 s minimum keeps the weight
        // contribution small but non-zero.
        let duration = max(u.end - u.start, 0.1)
        utteranceCount += 1
        totalDuration += duration

        if let v = u.fusedValence {
            sumV += Double(v) * duration
            weightV += duration
        }
        if let a = u.fusedArousal {
            sumA += Double(a) * duration
            weightA += duration
        }
        if let d = u.fusedDominance {
            sumD += Double(d) * duration
            weightD += duration
        }
        if let v = u.fusedValence, let a = u.fusedArousal {
            trajectory.append(TrajectoryPoint(time: u.start, valence: v, arousal: a))
            if trajectory.count > Self.trajectoryCap {
                trajectory.removeFirst()
            }
        }

        // Confidence-weighted vote for the top-label histogram. Falls back
        // to 0.5 when ASR didn't expose a per-utterance confidence (text
        // SER skipped, etc.) so the utterance still contributes something.
        if let label = u.fusedTopLabel, !label.isEmpty {
            let weight = u.asrConfidence ?? 0.5
            labelScores[label, default: 0] += weight
            labelCounts[label, default: 0] += 1
        }
    }

    public mutating func reset() {
        self = ConversationSummary()
    }

    private static func populationStdDev(_ xs: [Double]) -> Float? {
        guard xs.count > 1 else { return nil }
        let mean = xs.reduce(0, +) / Double(xs.count)
        let variance = xs.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / Double(xs.count)
        return Float(variance.squareRoot())
    }
}
