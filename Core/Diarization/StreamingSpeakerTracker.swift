import Foundation
import XephonLogging

/// Maintains session-stable speaker IDs across overlapping diarization
/// calls. FluidAudio's `performCompleteDiarization` is per-call stateless
/// w.r.t. embeddings, so the local IDs it returns ("speaker_0",
/// "speaker_1", …) aren't comparable across calls — even on overlapping
/// audio.
///
/// We bridge calls by **time-overlap matching**: when a new diarization
/// result arrives, we group its segments by local speaker, then for each
/// group we look at which already-known global speaker has spoken at
/// overlapping audio-time ranges. The local→global mapping is whichever
/// global accumulates the most overlap above a small threshold; new
/// locals with no overlap get a fresh global ID.
///
/// Overlap matching works because our streaming caller hands the
/// diarizer a sliding window that includes prior audio; segments inside
/// the overlap region carry the same audio across calls, so the
/// diarizer assigns them the same speaker — even if its per-call local
/// labels rotate. Beyond the overlap region (a speaker who's been
/// silent longer than the window) we genuinely don't know if it's the
/// same person, so we allocate a new ID — the conservative choice.
public actor StreamingSpeakerTracker {
    private var cumulative: [DiarizedSegment] = []
    private var nextSpeakerNumber: Int = 1
    private let overlapThreshold: TimeInterval

    /// Cap on `cumulative` so very long sessions don't grow the matching
    /// arena indefinitely. Old entries fall off the head — speakers
    /// silent for longer than the cap's audio-time coverage will be
    /// re-allocated as new global IDs if they reappear. Bumped from
    /// 2048 once the tracker switched to multi-observation voting:
    /// the continuous diarizer (10 s window, 2 s stride) covers each
    /// audio moment with ~5 overlapping calls, so 8192 entries
    /// represents roughly the same wall-clock coverage 2048 used to
    /// give in single-observation mode.
    public static let cumulativeCap: Int = 8192

    /// Sum of overlap (seconds) required to consider a new local
    /// speaker the "same" as an existing global. Bumped from 0.5 to
    /// 1.5 because voting mode keeps every overlapping observation,
    /// so `bestOverlapMatch` sums across ~5x as many entries per
    /// audio moment — the original 0.5 s threshold became too easy
    /// to clear and would conflate distinct speakers in adjacent
    /// windows.
    public init(overlapThreshold: TimeInterval = 1.5) {
        self.overlapThreshold = overlapThreshold
    }

    /// Ingest a diarization result and return the same segments with
    /// session-stable speaker IDs.
    public func ingest(_ incoming: [DiarizedSegment]) -> [DiarizedSegment] {
        guard !incoming.isEmpty else { return [] }

        let groups = Dictionary(grouping: incoming) { $0.speakerID }
        var localToGlobal: [String: String] = [:]
        for (localId, segments) in groups {
            if let match = bestOverlapMatch(for: segments) {
                localToGlobal[localId] = match
            } else {
                let id = String(format: "S%02d", nextSpeakerNumber)
                nextSpeakerNumber += 1
                localToGlobal[localId] = id
                AppLog.diarization.info("new speaker \(id, privacy: .public)")
            }
        }

        let mapped = incoming.map {
            DiarizedSegment(
                speakerID: localToGlobal[$0.speakerID] ?? $0.speakerID,
                start: $0.start,
                end: $0.end
            )
        }

        // Append-only: every diarize call's verdict is preserved as
        // an independent observation. The query path
        // (`speakerAt` / `cumulativeSnapshot` consumers) tallies
        // votes across overlapping observations to decide who
        // covered a given audio time. Earlier behavior — replacing
        // overlapping entries with the latest call — was effectively
        // "newest wins" and let single-call misclassifications stay
        // permanent until evicted by the cap. Voting smooths over
        // those by requiring a majority across the ~5 overlapping
        // windows that cover each audio moment under continuous
        // diarization.
        cumulative.append(contentsOf: mapped)
        cumulative.sort { $0.start < $1.start }

        if cumulative.count > Self.cumulativeCap {
            cumulative.removeFirst(cumulative.count - Self.cumulativeCap)
        }

        return mapped
    }

    public func reset() {
        cumulative.removeAll()
        nextSpeakerNumber = 1
    }

    /// True once at least one diarization call has populated the
    /// cumulative timeline. Callers querying by audio time should
    /// gate on this — `speakerAt` returns nil when empty, which the
    /// caller would otherwise have to treat as a fallback path.
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
    /// global ID — under continuous diarization, ~5 overlapping
    /// windows cover each moment, and a majority across them is more
    /// stable than any single window's verdict. Returns nil only
    /// when the timeline is empty; falls back to the closest segment
    /// (by midpoint) when no observations contain `audioTime`
    /// (diarizer gap, very early in the session).
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

    /// Total time-overlap (seconds) between the candidate local segments
    /// and each known global speaker's history. Returns the global ID
    /// with the most overlap, provided it exceeds `overlapThreshold`.
    private func bestOverlapMatch(for localSegments: [DiarizedSegment]) -> String? {
        guard !cumulative.isEmpty else { return nil }
        var totals: [String: TimeInterval] = [:]
        for local in localSegments {
            for global in cumulative {
                if global.end <= local.start || global.start >= local.end { continue }
                let overlap = min(local.end, global.end) - max(local.start, global.start)
                if overlap > 0 {
                    totals[global.speakerID, default: 0] += overlap
                }
            }
        }
        guard let best = totals.max(by: { $0.value < $1.value }),
              best.value >= overlapThreshold else {
            return nil
        }
        return best.key
    }
}
