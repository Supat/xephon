import Foundation
@preconcurrency import AVFoundation
import Audio
import Fusion
import XephonLogging

// Per-utterance / arbitrary-range playback split out of
// RecordingController.swift to keep the controller's source file
// readable. Same cross-file extension pattern as the other slices.
// Covers `setPlaybackSourceURL` (with the security-scoped access
// ref balancing), `togglePlayback`, `playRange`, and `stopPlayback`.
//
// The unrelated `applySegmentResult` callback that historically
// lived under the same MARK section stays in the main file —
// it's recording-pipeline state, not playback.
extension RecordingController {



    /// Assign `playbackSourceURL` while keeping the security-scoped
    /// access ref balanced. File-picker URLs only stay readable while
    /// some part of the app holds a ref via
    /// `startAccessingSecurityScopedResource()`; AudioFileCapture's
    /// ref drops at the end of analysis, so we hold our own ref here
    /// for the duration that the URL is exposed for playback. The
    /// `start` can fail for already-accessible URLs (e.g. in tests),
    /// which is fine — we just don't stash a stop counterpart.
    func setPlaybackSourceURL(_ newURL: URL?) {
        if let scoped = scopedPlaybackURL {
            scoped.stopAccessingSecurityScopedResource()
            scopedPlaybackURL = nil
        }
        playbackSourceURL = newURL
        if let url = newURL {
            let ok = url.startAccessingSecurityScopedResource()
            AppLog.app.info(
                "playback scope start: \(ok ? "ok" : "skipped", privacy: .public) for \(url.lastPathComponent, privacy: .public)"
            )
            if ok { scopedPlaybackURL = url }
        }
    }

    /// Toggle playback of the audio range `[utterance.start, utterance.end]`
    /// from `playbackSourceURL`. No-op when there's no source URL (mic
    /// session) or when analysis is still running — the row gates the
    /// button so this is defense-in-depth. Tapping the row that's
    /// currently playing stops it; tapping a different row stops the
    /// previous playback and starts the new one.
    func togglePlayback(for utterance: UtteranceEstimate) {
        AppLog.app.info(
            "togglePlayback called: utt=\(utterance.id, privacy: .public) src=\(self.playbackSourceURL?.lastPathComponent ?? "nil", privacy: .public) phase=\(String(describing: self.phase), privacy: .public) scoped=\(self.scopedPlaybackURL?.lastPathComponent ?? "nil", privacy: .public)"
        )
        guard let url = playbackSourceURL else {
            AppLog.app.warning("togglePlayback: no playbackSourceURL")
            return
        }
        guard phase == .idle else {
            AppLog.app.warning("togglePlayback: phase not idle: \(String(describing: self.phase), privacy: .public)")
            return
        }
        if playingUtteranceID == utterance.id {
            stopPlayback()
            return
        }
        stopPlayback()
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            AppLog.app.info(
                "playback session BEFORE: category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
            AppLog.app.info(
                "playback session AFTER:  category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
        } catch {
            AppLog.app.warning("playback session setup failed: \(String(describing: error), privacy: .public)")
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            // Retain the player BEFORE calling play(). If the local
            // reference is the only retain at the moment of play(),
            // the optimizer is free to release it on the next line —
            // not normally an issue, but we've seen the case where
            // state doesn't progress past idle on the second session.
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, utterance.start)
            let didStart = player.play()
            AppLog.app.info(
                "togglePlayback: play() returned \(didStart ? "true" : "false", privacy: .public), duration=\(player.duration, privacy: .public)s, seek=\(player.currentTime, privacy: .public)s"
            )
            guard didStart else {
                AppLog.app.warning("playback failed to start for \(utterance.id, privacy: .public)")
                playbackPlayer = nil
                return
            }
            playingUtteranceID = utterance.id
            AppLog.app.info("togglePlayback: playingUtteranceID set to \(utterance.id, privacy: .public)")
            let duration = max(0, utterance.end - utterance.start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(duration))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error("playback open failed: \(String(describing: error), privacy: .public)")
        }
    }


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


    /// Play `[start, end]` from the source file. Used by the Edit
    /// Utterance dialog's preview button — the dialog's spinners
    /// hold arbitrary times that don't correspond to an existing
    /// utterance row, so `togglePlayback(for:)` doesn't apply.
    ///
    /// `owner` ties the playback session to a specific utterance id
    /// when one applies — the review sheet's inline play buttons use
    /// this so each card knows whether it's the active one. Pass
    /// nil for arbitrary-range previews (the edit dialog's preview
    /// button, where the spinners may not correspond to any row).
    func playRange(
        start: TimeInterval,
        end: TimeInterval,
        owner: UUID? = nil
    ) {
        guard let url = playbackSourceURL else { return }
        guard phase == .idle else { return }
        guard end > start else { return }
        stopPlayback()
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
        } catch {
            AppLog.app.warning(
                "playRange: session setup failed: \(String(describing: error), privacy: .public)"
            )
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, start)
            guard player.play() else { return }
            isPreviewPlaying = true
            playingUtteranceID = owner
            let duration = max(0, end - start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(duration))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error(
                "playRange failed: \(String(describing: error), privacy: .public)"
            )
        }
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



    /// Stop any in-flight playback. Safe to call when nothing is
    /// playing — it just clears the latch.
    func stopPlayback(caller: String = #function) {
        if playingUtteranceID != nil || playbackPlayer != nil {
            AppLog.app.info("stopPlayback called by \(caller, privacy: .public), playingID=\(self.playingUtteranceID?.uuidString ?? "nil", privacy: .public)")
        }
        playbackStopTask?.cancel()
        playbackStopTask = nil
        playbackPlayer?.stop()
        playbackPlayer = nil
        playingUtteranceID = nil
        isPreviewPlaying = false
    }

}
