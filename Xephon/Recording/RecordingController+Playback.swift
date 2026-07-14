import Foundation
@preconcurrency import AVFoundation
import Audio
import Fusion
import XephonLogging
import XephonPluginKit

// Utterance lookup helpers + shared post-edit commit pass. The
// playback machinery that used to live here (setPlaybackSourceURL /
// togglePlayback / playRange / stopPlayback and the session latch)
// moved to PlaybackCoordinator; RecordingController keeps thin
// forwarders so call sites are unchanged.
extension RecordingController {

    /// Replace the utterance whose `id == utteranceID` with a merge
    /// of the original's identity (`id`, `speakerID`, `speechBoost`)
    /// and the freshly-computed content. `start` and `end` now come
    /// from `fresh` so the row's displayed timestamp updates to the
    /// re-evaluated audio range — typically a tighter span than the
    /// streaming pass produced, since the sentence-aware trim
    /// landed on per-token anchors. Stamps `wasReevaluated = true`
    /// so the marker rides along with the utterance through
    /// Save/Load and JSON export. Rebuilds the running conversation
    /// summary from scratch — ConversationSummary is an incremental
    /// fold with no "replace" path, and N is small enough that
    /// re-folding is cheap.
    ///
    /// After the replacement, re-sorts `utterances` if the corrected
    /// `start` moved the row out of chronological order — without
    /// this, a substantially-shifted re-eval could leave the list
    /// out of order, and List's identity-stable rendering would
    /// keep it at its old position visually.
    /// Bump `utterancesVersion` (so ContentView's filter memo
    /// invalidates after in-place mutations that leave
    /// `utterances.count` unchanged) and rebuild the conversation
    /// summary from the current `utterances`. Called after every
    /// path that mutates a row's content or replaces an entry.
    ///
    /// Also runs the auto-demote pass: any speaker id that was
    /// present in `lastKnownSpeakerIDs` but no longer appears in
    /// any utterance has been fully reassigned away and gets
    /// cleaned up — drop the speaker name override, scrub
    /// cumulative-timeline observations with that id, and remove
    /// the entry from the diarizer DB (keeping permanent
    /// user-promoted entries). Gated on `phase == .idle` so
    /// mid-streaming flux doesn't trigger premature deletion of
    /// a speaker the streaming pass is still about to use.
    func commitUtteranceChanges() {
        utterancesVersion &+= 1
        conversationSummary.reset()
        for u in utterances { conversationSummary.update(with: u) }
        let current = Set(utterances.map(\.speakerID))
        let removed = lastKnownSpeakerIDs.subtracting(current)
        if !removed.isEmpty, phase == .idle {
            sweepUnreferencedSpeakers(removed)
        }
        lastKnownSpeakerIDs = current
        cachedKnownSpeakerIDs = current.sorted()
        pluginEventSink?(.utterancesChanged(version: utterancesVersion))
    }

    /// Drop every trace of speakers that no row references
    /// anymore. Synchronous parts run inline (overrides, timeline
    /// observations); the diarizer-DB removal is fired off in a
    /// detached MainActor task so it doesn't block the commit
    /// pass. Failure of the async removal is non-fatal — the
    /// embedding stays in the DB but nothing else in the app
    /// references the id, so it's only memory waste.
    private func sweepUnreferencedSpeakers(_ ids: Set<String>) {
        for id in ids {
            speakerNameOverrides.removeValue(forKey: id)
            diarizationTimeline.removeAll { $0.speakerID == id }
        }
        AppLog.app.info(
            "auto-demoting unreferenced speakers: \(Array(ids).sorted().joined(separator: ", "), privacy: .public)"
        )
        Task { @MainActor [weak self] in
            guard let self else { return }
            let pipeline = await self.ensurePipeline()
            for id in ids {
                try? await pipeline.removeSpeakerFromDB(id: id, keepIfPermanent: true)
            }
        }
    }

    /// Build a merged `UtteranceEstimate` from the fresh SER/fusion
    /// output combined with stable origin fields (id, speakerID,
    /// speechBoost) and the per-flow edit flags. Used by the three
    /// "replace this row with new SER results" paths
    /// (`applyReevaluation`, `applyHandEdit`, `applyHandEditSplit`)
    /// to keep their constructor footprints small and consistent.
    static func mergedEstimate(
        id: UUID,
        speakerID: String,
        speechBoost: Bool?,
        fresh: UtteranceEstimate,
        wasReevaluated: Bool?,
        wasHandEdited: Bool?
    ) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            start: fresh.start,
            end: fresh.end,
            transcript: fresh.transcript,
            asrConfidence: fresh.asrConfidence,
            dimensional: fresh.dimensional,
            acousticCategorical: fresh.acousticCategorical,
            ageGender: fresh.ageGender,
            plutchik: fresh.plutchik,
            textBackend: fresh.textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fresh.fusedValence,
            fusedArousal: fresh.fusedArousal,
            fusedDominance: fresh.fusedDominance,
            fusedTopLabel: fresh.fusedTopLabel
        )
    }


    /// Utterance whose `[start, end)` contains `t`, or the one
    /// whose midpoint is closest if no row contains `t`. Used by
    /// the timeline strips' tap-to-scroll.
    func nearestUtterance(toTime t: TimeInterval) -> UtteranceEstimate? {
        if let containing = utterances.first(where: { $0.start <= t && t < $0.end }) {
            return containing
        }
        return utterances.min {
            abs(($0.start + $0.end) / 2 - t) < abs(($1.start + $1.end) / 2 - t)
        }
    }

    /// Exact-id resolution from a diarizer observation segment id
    /// back to its emitting utterance. Nil for sessions that
    /// pre-date observation pinning, and for centroid taps which
    /// pass `nil` as the segment id by convention.
    func utterance(forSegmentID sid: UUID) -> UtteranceEstimate? {
        guard let uid = utteranceObservationSegmentIDs.first(
            where: { $0.value == sid }
        )?.key else { return nil }
        return utterances.first(where: { $0.id == uid })
    }

    /// Argmin Euclidean distance from `query` over utterances
    /// belonging to `speakerID`. Constraining to one speaker
    /// matters because overlapping clouds in the cluster scatter
    /// otherwise let a global argmin land in a neighbor's cloud.
    /// Nil when the speaker has no utterance with a stored
    /// embedding (older session, mic-mode without diarizer, or
    /// auto-demoted centroid).
    func nearestUtterance(
        toEmbedding query: [Float],
        speakerID: String
    ) -> UtteranceEstimate? {
        guard !utteranceEmbeddings.isEmpty else { return nil }
        let speakerByID = utterances.reduce(into: [UUID: String]()) {
            $0[$1.id] = $1.speakerID
        }
        var bestID: UUID?
        var bestDist: Float = .infinity
        for (id, e) in utteranceEmbeddings {
            guard speakerByID[id] == speakerID else { continue }
            let n = min(query.count, e.count)
            guard n > 0 else { continue }
            var sum: Float = 0
            for j in 0..<n {
                let d = e[j] - query[j]
                sum += d * d
            }
            if sum < bestDist {
                bestDist = sum
                bestID = id
            }
        }
        guard let id = bestID else { return nil }
        return utterances.first(where: { $0.id == id })
    }



}
