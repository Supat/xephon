import Foundation
import Audio
import ASR
import Diarization
import Fusion
import XephonLogging
import XephonUtilities

// Speaker-editing flow split out of RecordingController.swift to
// keep the controller's source file readable. Covers the row-
// level speaker mutations (rename, reassign, correct, affirm,
// promote-new), the speaker-id allocator, and the cumulative-
// timeline rewrite helpers those mutations need. Same cross-file
// `extension RecordingController` pattern as +HandEdit /
// +Reevaluation / +SessionPersistence.
extension RecordingController {

    /// Set a custom display name for `stored` (e.g. `S01`). Pass an
    /// empty / whitespace-only `name` to clear the override (revert
    /// to the default `S01`-style label). Bumps
    /// `utterancesVersion` so the ContentView filter memo
    /// invalidates and every row re-renders with the new label.
    func renameSpeaker(stored: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            guard speakerNameOverrides.removeValue(forKey: stored) != nil else { return }
            AppLog.app.info(
                "speaker rename: cleared override for \(stored, privacy: .public)"
            )
        } else {
            if speakerNameOverrides[stored] == trimmed { return }
            speakerNameOverrides[stored] = trimmed
            AppLog.app.info(
                "speaker rename: \(stored, privacy: .public) → \(trimmed, privacy: .public)"
            )
        }
        utterancesVersion &+= 1
    }

    /// Custom display name for `stored` if the user has renamed it,
    /// nil otherwise. Convenience for the view layer.
    func speakerDisplayName(forStored stored: String) -> String? {
        speakerNameOverrides[stored]
    }

    /// Walk every utterance once and reassign its `speakerID` to
    /// the cumulative timeline's per-instant majority verdict for
    /// its `[start, end]` window. Streaming-pass assignment uses
    /// the timeline as it stood at finalize-time; by end-of-file,
    /// later observations may have shifted the majority. This pass
    /// brings the labels into sync with the strip so the two
    /// visualizations agree before the session goes idle.
    ///
    /// Mutates in place + a single `commitUtteranceChanges()` at
    /// the end, so the auto-demote sweep and version bump fire
    /// once for the whole batch instead of N times.
    func reconcileSpeakersWithTimeline() {
        guard !diarizationTimeline.isEmpty, !utterances.isEmpty else { return }
        var changed = 0
        for i in utterances.indices {
            let utt = utterances[i]
            let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                diarizationTimeline,
                from: utt.start,
                to: utt.end,
                fallback: utt.speakerID
            )
            if dominant != utt.speakerID {
                utterances[i] = utt.withSpeakerID(dominant)
                changed += 1
            }
        }
        guard changed > 0 else { return }
        AppLog.app.info(
            "finalize: reconciled \(changed, privacy: .public) utterance speaker assignments against cumulative timeline"
        )
        commitUtteranceChanges()
    }

    /// Manually reassign one utterance's `speakerID`. Used by the
    /// speaker chip's action sheet to override the diarizer's
    /// verdict on a row whose voice the user knows belongs to a
    /// different speaker. The mutation is in-place — `id`, times,
    /// transcript, SER/fusion outputs all carry over — so list
    /// position and selection are stable. Bumps
    /// `utterancesVersion` so the filter memo invalidates and the
    /// chip re-renders with the new tint/label. No-op when the row
    /// is already on that speaker, when no row matches `utteranceID`,
    /// or when `newSpeakerID` is empty / whitespace-only.
    func reassignSpeaker(utteranceID: UUID, to newSpeakerID: String) {
        let trimmed = newSpeakerID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]
        guard original.speakerID != trimmed else { return }
        utterances[index] = original.withSpeakerID(trimmed)
        AppLog.app.info(
            "speaker reassigned: utt=\(original.id, privacy: .public) \(original.speakerID, privacy: .public) → \(trimmed, privacy: .public)"
        )
        commitUtteranceChanges()
    }

    /// Sorted, deduplicated list of speaker IDs currently appearing
    /// in the session — surface for the reassignment menu. Includes
    /// the currently-assigned speaker so the menu reads as a Picker
    /// (with the active row visually marked, not removed). Backed
    /// by a cache maintained at every utterance-mutation boundary
    /// so this is O(1) per call — it's invoked from
    /// `TranscriptList.row(for:)` per visible row per render.
    func knownSpeakerIDs() -> [String] {
        cachedKnownSpeakerIDs
    }

    /// Corrective reassignment: extract the row's audio embedding
    /// and fold it into `targetSpeakerID`'s centroid in the
    /// diarizer DB via EMA, reassign the row, and rewrite the
    /// timeline range so the per-instant majority for the window
    /// reflects the corrected speaker. Future re-eval / hand-edit
    /// on similar audio will be more likely to match the target,
    /// not just for this row but anywhere in the session.
    ///
    /// Distinct from the pure-annotation `reassignSpeaker(...)`,
    /// which only swaps the row's label and leaves the diarizer
    /// state untouched.
    ///
    /// No-op when no source audio, the row doesn't exist, the
    /// audio slice is empty, embedding extraction is unavailable,
    /// the target id is the same as the current id, or the
    /// underlying diarizer call throws. Returns true on success.
    @discardableResult
    func correctUtteranceSpeaker(
        utteranceID: UUID,
        to targetSpeakerID: String
    ) async -> Bool {
        let trimmed = targetSpeakerID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let url = playbackSourceURL else { return false }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return false }
        let utt = utterances[index]
        guard utt.speakerID != trimmed else { return false }
        let chunk: AudioChunk
        do {
            chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: utt.start,
                    end: utt.end
                )
            }.value
        } catch {
            AppLog.app.error(
                "correct: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("correct: empty audio slice")
            return false
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("correct: embedding extractor unavailable")
            return false
        }
        let duration = Float(max(0, utt.end - utt.start))
        do {
            try await pipeline.correctSpeaker(
                id: trimmed,
                embedding: embedding,
                duration: duration
            )
        } catch {
            AppLog.app.error(
                "correct: DB update failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        utterances[index] = utt.withSpeakerID(trimmed)
        rewriteTimelineRange(speakerID: trimmed, start: utt.start, end: utt.end)
        AppLog.app.info(
            "speaker corrected: utt=\(utt.id, privacy: .public) \(utt.speakerID, privacy: .public) → \(trimmed, privacy: .public) (taught diarizer)"
        )
        commitUtteranceChanges()
        return true
    }

    /// "Affirm": tell the diarizer that the row's currently-assigned
    /// speaker is correct. Extracts the row's audio embedding and
    /// folds it into the **current** speaker's centroid via EMA, the
    /// same teaching path `correctUtteranceSpeaker` uses for a
    /// different target — so subsequent re-eval / hand-edit on
    /// acoustically similar audio is more likely to match this
    /// speaker. No row reassignment, no cumulative-timeline rewrite,
    /// no utterance mutation: the assignment is already what the
    /// user wants; this just reinforces it in the diarizer DB.
    ///
    /// No-op when source audio is absent (mic mode), the row doesn't
    /// exist, the audio slice is empty, embedding extraction is
    /// unavailable, or the underlying diarizer call throws. Returns
    /// true on success.
    @discardableResult
    func affirmUtteranceSpeaker(utteranceID: UUID) async -> Bool {
        guard let url = playbackSourceURL else { return false }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return false }
        let utt = utterances[index]
        let chunk: AudioChunk
        do {
            chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: utt.start,
                    end: utt.end
                )
            }.value
        } catch {
            AppLog.app.error(
                "affirm: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("affirm: empty audio slice")
            return false
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("affirm: embedding extractor unavailable")
            return false
        }
        let duration = Float(max(0, utt.end - utt.start))
        do {
            try await pipeline.correctSpeaker(
                id: utt.speakerID,
                embedding: embedding,
                duration: duration
            )
        } catch {
            AppLog.app.error(
                "affirm: DB update failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        // Declare the affirmed speaker as authoritative over the
        // row's audio range, the same way `correctUtteranceSpeaker`
        // does for a different target. Without this the cumulative
        // timeline may still hold competing observations for
        // `[utt.start, utt.end]`, so the mismatch detector keeps
        // flagging the row and the orange caution glyph stays put
        // — even though the user just confirmed the assignment.
        rewriteTimelineRange(
            speakerID: utt.speakerID,
            start: utt.start,
            end: utt.end
        )
        AppLog.app.info(
            "speaker affirmed: utt=\(utt.id, privacy: .public) \(utt.speakerID, privacy: .public) (reinforced diarizer, timeline range rewritten)"
        )
        // The timeline rewrite already bumped `diarizationTimelineVersion`
        // via the didSet, which is what the mismatch / filter memos
        // key on — no need for `commitUtteranceChanges()` here since
        // utterance content is unchanged.
        return true
    }

    /// Extract an embedding from `utteranceID`'s audio, register it
    /// in the diarizer's SpeakerManager DB under a freshly-minted
    /// `S0N` id, reassign the row to that id, and re-write the
    /// cumulative timeline so the new id wins the majority vote
    /// for the utterance's window. Future re-eval / hand-edit
    /// passes on similar audio will match this entry.
    ///
    /// No-op when: there's no source audio (mic mode without a
    /// playback URL), no row matches `utteranceID`, the audio
    /// slice came back empty, the diarizer's embedding extractor
    /// isn't available, or promotion throws. Returns the new id
    /// on success, nil on any failure path — caller can surface
    /// a toast on nil.
    func promoteUtteranceToNewSpeaker(utteranceID: UUID) async -> String? {
        guard let url = playbackSourceURL else { return nil }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return nil }
        let utt = utterances[index]
        let chunk: AudioChunk
        do {
            chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: utt.start,
                    end: utt.end
                )
            }.value
        } catch {
            AppLog.app.error(
                "promote: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("promote: empty audio slice")
            return nil
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("promote: embedding extractor unavailable")
            return nil
        }
        let newID = nextAvailableSpeakerID()
        do {
            try await pipeline.promoteSpeaker(id: newID, embedding: embedding)
        } catch {
            AppLog.app.error(
                "promote: register failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        utterances[index] = utt.withSpeakerID(newID)
        rewriteTimelineRange(speakerID: newID, start: utt.start, end: utt.end)
        AppLog.app.info(
            "speaker promoted: utt=\(utt.id, privacy: .public) → \(newID, privacy: .public) [\(utt.start, privacy: .public)..\(utt.end, privacy: .public)]"
        )
        commitUtteranceChanges()
        return newID
    }

    /// Smallest `S0N` id not in use anywhere the user might see it:
    /// neither in the current utterance roster nor in the diarizer's
    /// internal speaker DB. The latter matters because FluidAudio's
    /// streaming `assignSpeaker` mints raw numeric ids ("4") that
    /// our snapshot formats to "S04" for display — picking that same
    /// "S04" for a promote would leave two distinct DB entries
    /// rendering as the same chip and trigger SwiftUI's "id occurs
    /// multiple times" warning on the cluster panels. Used by every
    /// "new speaker" allocation path (chip-menu reassign, promote).
    func nextAvailableSpeakerID() -> String {
        var used = cachedKnownSpeakerIDs
        used.append(contentsOf: speakerCluster.speakers.map(\.id))
        return Self.nextNewSpeakerID(in: used)
    }

    /// Smallest unused `S0N` id given the supplied existing ids.
    /// Backs `nextAvailableSpeakerID` — kept static + pure so callers
    /// that already have a specific id set in hand can use it
    /// directly. Both "New Speaker" and "Promote New Speaker" route
    /// through `nextAvailableSpeakerID` so they agree on which slot
    /// to fill.
    private static func nextNewSpeakerID(in existing: [String]) -> String {
        var used: Set<Int> = []
        for id in existing {
            guard id.hasPrefix("S") else { continue }
            if let n = Int(id.dropFirst()) { used.insert(n) }
        }
        var candidate = 1
        while used.contains(candidate) { candidate += 1 }
        return String(format: "S%02d", candidate)
    }

    /// Replace every cumulative-timeline observation that covers
    /// any part of `[start, end]` so the result has exactly one
    /// segment for `speakerID` over that window. Overlapping
    /// segments get split at the boundary (the portions outside
    /// `[start, end]` survive intact, the portion inside is
    /// dropped). Used after `promoteUtteranceToNewSpeaker` so the
    /// per-instant majority for the promoted range cleanly
    /// reflects the new speaker — without this, the streaming
    /// pass's ~5 votes for the old id would still win the
    /// `dominantSpeakerInSegments` tally and the mismatch
    /// warning would persist.
    func rewriteTimelineRange(
        speakerID: String,
        start: TimeInterval,
        end: TimeInterval
    ) {
        guard end > start else { return }
        var rewritten: [DiarizedSegment] = []
        rewritten.reserveCapacity(diarizationTimeline.count + 1)
        for seg in diarizationTimeline {
            if seg.end <= start || seg.start >= end {
                rewritten.append(seg)
            } else if seg.start < start && seg.end > end {
                // Segment surrounds the promoted range — split.
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: start))
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: end, end: seg.end))
            } else if seg.start < start {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: start))
            } else if seg.end > end {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: end, end: seg.end))
            }
            // Fully inside → drop.
        }
        rewritten.append(DiarizedSegment(speakerID: speakerID, start: start, end: end))
        rewritten.sort { $0.start < $1.start }
        diarizationTimeline = rewritten
    }

    /// Replace every cumulative-timeline observation overlapping
    /// `[coveringStart, coveringEnd]` with the per-slice segments
    /// in `slices`. Segments outside the covered range survive
    /// intact; segments straddling either boundary get trimmed at
    /// the boundary; segments fully inside are dropped. Used after
    /// a multi-sentence hand-edit re-diarizes the parent window —
    /// without this, the streaming pass's older per-instant
    /// majority would still drive the timeline strip and disagree
    /// with the post-edit per-row speaker labels.
    func rewriteTimelineWithSlices(
        coveringStart: TimeInterval,
        coveringEnd: TimeInterval,
        slices: [DiarizedSegment]
    ) {
        guard coveringEnd > coveringStart, !slices.isEmpty else { return }
        var rewritten: [DiarizedSegment] = []
        rewritten.reserveCapacity(diarizationTimeline.count + slices.count)
        for seg in diarizationTimeline {
            if seg.end <= coveringStart || seg.start >= coveringEnd {
                rewritten.append(seg)
            } else if seg.start < coveringStart && seg.end > coveringEnd {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: coveringStart))
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: coveringEnd, end: seg.end))
            } else if seg.start < coveringStart {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: coveringStart))
            } else if seg.end > coveringEnd {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: coveringEnd, end: seg.end))
            }
            // Fully inside → drop.
        }
        for slice in slices where slice.end > slice.start {
            rewritten.append(slice)
        }
        rewritten.sort { $0.start < $1.start }
        diarizationTimeline = rewritten
    }

}
