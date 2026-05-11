import Foundation
import XephonLogging

/// Append-only log of session-stable diarized segments. The diarizer
/// (FluidAudio's `SpeakerManager`) already produces global, session-
/// stable speaker IDs via embedding-based clustering, so we don't
/// need to remap local→global ourselves — earlier versions of this
/// file did, but that was a workaround for an outdated assumption
/// about FluidAudio's API. Now we just accumulate observations and
/// expose a vote-based query path for per-token assignment.
///
/// Why keep the log at all when each diarize call's IDs are already
/// session-stable? Two reasons:
///
/// 1. **Voting smooths single-call noise.** The continuous diarize
///    task fires every 2 s on a 10 s window, so each audio moment
///    is covered by ~5 overlapping observations. A single window's
///    verdict can be wrong (short turn at the edge, marginal
///    embedding quality); a majority across the overlapping
///    observations is more stable.
///
/// 2. **Cross-call timeline queries.** `dominantSpeaker` reads a
///    single snapshot and votes per-instant across the sentence's
///    range. Doing that requires a unified timeline rather than
///    per-call results.
public actor StreamingSpeakerTracker {
    private var cumulative: [DiarizedSegment] = []

    /// Cap on `cumulative` so very long sessions don't grow the log
    /// indefinitely. With ~5 overlapping observations per audio
    /// moment from continuous diarize, 8192 entries covers roughly
    /// the same wall-clock as the prior single-observation 2048 cap.
    public static let cumulativeCap: Int = 8192

    public init() {}

    /// Append every diarize call's verdict as an independent
    /// observation. No remapping — incoming IDs are trusted as
    /// session-stable. Sort and cap.
    public func ingest(_ incoming: [DiarizedSegment]) -> [DiarizedSegment] {
        guard !incoming.isEmpty else { return [] }
        cumulative.append(contentsOf: incoming)
        cumulative.sort { $0.start < $1.start }
        if cumulative.count > Self.cumulativeCap {
            cumulative.removeFirst(cumulative.count - Self.cumulativeCap)
        }
        return incoming
    }

    public func reset() {
        cumulative.removeAll()
    }

    /// True once at least one diarization call has populated the
    /// cumulative timeline. Callers querying by audio time should
    /// gate on this — the empty-timeline path falls back to a fresh
    /// per-segment diarize.
    public var isPopulated: Bool { !cumulative.isEmpty }

    /// Snapshot of the cumulative speaker timeline, sorted by start
    /// time. Returned by value so callers can do many lookups
    /// against it without paying actor-await cost per query —
    /// per-token assignment at sub-segment splitting needs O(N
    /// tokens) lookups, and waking the actor for each one would
    /// dominate split latency. The copy itself is a few KB on
    /// realistic sessions (≤ `cumulativeCap` entries × ~40 B).
    public func cumulativeSnapshot() -> [DiarizedSegment] { cumulative }

    /// Speaker ID covering `audioTime`. Tallies votes across every
    /// observation containing `audioTime` and returns the majority
    /// global ID. Returns nil only when the timeline is empty;
    /// falls back to the closest segment (by midpoint) when no
    /// observation contains `audioTime`.
    public func speakerAt(_ audioTime: TimeInterval) -> String? {
        guard !cumulative.isEmpty else { return nil }
        var votes: [String: Int] = [:]
        for s in cumulative where s.start <= audioTime && audioTime <= s.end {
            votes[s.speakerID, default: 0] += 1
        }
        if let best = votes.max(by: { $0.value < $1.value }) {
            return best.key
        }
        return cumulative.min(by: {
            abs(($0.start + $0.end) / 2 - audioTime) < abs(($1.start + $1.end) / 2 - audioTime)
        })?.speakerID
    }
}
