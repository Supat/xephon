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
    /// arena indefinitely. Old entries fall off the head — speakers from
    /// >~1 hour ago will be re-allocated as new global IDs if they
    /// reappear, which is acceptable.
    public static let cumulativeCap: Int = 2048

    public init(overlapThreshold: TimeInterval = 0.5) {
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

        // Replace any cumulative entries entirely within the new
        // call's time range — they're now superseded by the latest
        // result. Append the new mapped segments and re-sort. Outside
        // the new range we keep the previous history intact.
        let lo = incoming.lazy.map { $0.start }.min() ?? 0
        let hi = incoming.lazy.map { $0.end }.max() ?? 0
        cumulative.removeAll { $0.start >= lo && $0.end <= hi }
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
