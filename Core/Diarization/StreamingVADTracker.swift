import Foundation

/// Cumulative VAD timeline maintained across continuous-VAD ticks.
/// Mirrors `StreamingSpeakerTracker` structurally, but stores
/// pre-merged speech intervals (no speaker dimension to vote on).
///
/// Each ingest unions the incoming segments with the existing
/// timeline and re-merges overlaps/adjacencies. The continuous-VAD
/// task fires on the same audio window as the continuous-diarize
/// task (10 s sliding, 2 s stride), so each audio moment is covered
/// by ~5 overlapping observations — overlap-union here gives us the
/// generous "any VAD call said speech here" set, which is what we
/// want for an AND with the diarizer's per-speaker regions in
/// `trimToSpeakerActive`.
public actor StreamingVADTracker {
    private var cumulative: [SpeechSegment] = []

    /// Cap on `cumulative` so very long sessions don't grow the
    /// timeline indefinitely. Speech segments post-union are far
    /// fewer than per-call observations — a 1 h conversational
    /// session typically settles at <2000 entries — so the cap
    /// matches `StreamingSpeakerTracker.cumulativeCap` even though
    /// the per-segment overhead is lower (no speaker field).
    public static let cumulativeCap: Int = 8192

    public init() {}

    /// Merge `incoming` into the cumulative timeline. Returns the
    /// incoming segments unchanged so the caller can chain logging
    /// off the same value the ingester saw.
    @discardableResult
    public func ingest(_ incoming: [SpeechSegment]) -> [SpeechSegment] {
        guard !incoming.isEmpty else { return [] }
        var combined = cumulative
        combined.append(contentsOf: incoming)
        combined.sort { $0.start < $1.start }
        var merged: [SpeechSegment] = []
        merged.reserveCapacity(combined.count)
        for s in combined {
            if let last = merged.last, s.start <= last.end {
                merged[merged.count - 1] = SpeechSegment(
                    start: last.start,
                    end: max(last.end, s.end)
                )
            } else {
                merged.append(s)
            }
        }
        if merged.count > Self.cumulativeCap {
            merged.removeFirst(merged.count - Self.cumulativeCap)
        }
        cumulative = merged
        return incoming
    }

    public func reset() {
        cumulative.removeAll()
    }

    /// Snapshot the cumulative speech timeline, sorted by start
    /// time. Returned by value so callers (e.g. `trimToSpeakerActive`)
    /// can do many lookups against it without paying actor-await
    /// cost per query.
    public func cumulativeSnapshot() -> [SpeechSegment] { cumulative }
}
