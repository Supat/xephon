import Foundation
import Audio
import ASR
import Diarization
import Fusion
import XephonLogging
import XephonUtilities

// Hand-edit flow split out of RecordingController.swift to keep the
// controller's source file readable. These methods are still members
// of `RecordingController` — Swift extensions across files share
// the same type — so they get the controller's full state surface
// without any explicit forwarding layer. The downside is that a few
// properties + helpers had to drop their `private(set)` / `private`
// modifiers (each annotated at its declaration site) so the
// extension can mutate them; for an app target with a single
// controller this is an acceptable encapsulation cost.
extension RecordingController {



    /// Commit a hand-edited utterance: new transcript text and new
    /// time range, with SER + fusion re-run on the audio slice
    /// `[newStart, newEnd]`. Identity is preserved (`id`,
    /// `speakerID`, `speechBoost`), `wasHandEdited` is stamped
    /// true, `wasReevaluated` is cleared. Captures the pre-edit
    /// snapshot first (sharing the same `preReevaluationSnapshots`
    /// store used by the re-evaluate revert path) so the long-press
    /// revert restores the truly-original streaming row.
    ///
    /// No-op when mic-mode (no source audio), when something else
    /// is already running, when the range is empty, or when the
    /// trimmed transcript is empty.
    func commitHandEdit(
        utteranceID: UUID,
        newText: String,
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) async {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let trimmedText = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }
        let sentences = Self.splitTranscriptIntoSentences(trimmedText)
        guard !sentences.isEmpty else { return }

        let original = utterances[index]
        // Two flows split on whether the session has source audio.
        // File-mode (`playbackSourceURL != nil`) follows the full
        // pipeline: re-read the audio, re-diarize, re-run SER +
        // fusion. Mic-mode (live or imported) has no audio to
        // re-slice, so we inherit the parent's speaker, time
        // range, dimensional + acoustic SER, and only re-run text
        // SER + fusion. The Edit Utterance dialog hides time
        // controls + play button on the mic-mode path; the
        // controller treats `newStart`/`newEnd` as advisory and
        // overrides them with `original.start`/`original.end`.
        if playbackSourceURL == nil {
            reevaluatingUtteranceID = utteranceID
            defer { reevaluatingUtteranceID = nil }
            if preReevaluationSnapshots[utteranceID] == nil {
                preReevaluationSnapshots[utteranceID] = original
            }
            let plans = Self.planHandEditSentences(
                sentences,
                newStart: original.start,
                newEnd: original.end
            )
            guard !plans.isEmpty else { return }
            let pipeline = await ensurePipeline()
            await runTextOnlyHandEdit(
                utteranceID: utteranceID,
                plans: plans,
                original: original,
                pipeline: pipeline
            )
            return
        }

        guard let url = playbackSourceURL else { return }
        guard newEnd > newStart else { return }
        let plans = Self.planHandEditSentences(
            sentences,
            newStart: newStart,
            newEnd: newEnd
        )
        guard !plans.isEmpty else { return }

        reevaluatingUtteranceID = utteranceID
        defer { reevaluatingUtteranceID = nil }

        if preReevaluationSnapshots[utteranceID] == nil {
            preReevaluationSnapshots[utteranceID] = original
        }

        let fallbackSpeaker = original.speakerID
        let pipeline = await ensurePipeline()

        guard let chunks = await readHandEditAudio(
            url: url,
            newStart: newStart,
            newEnd: newEnd
        ) else { return }
        let (diarChunk, serChunk) = chunks

        if plans.count == 1 {
            await runSingleSentenceHandEdit(
                utteranceID: utteranceID,
                plan: plans[0],
                diarChunk: diarChunk,
                serChunk: serChunk,
                fallbackSpeaker: fallbackSpeaker,
                pipeline: pipeline
            )
            return
        }

        let freshResults = await runMultiSentenceHandEdit(
            plans: plans,
            diarChunk: diarChunk,
            url: url,
            fallbackSpeaker: fallbackSpeaker,
            pipeline: pipeline
        )
        guard !freshResults.isEmpty else { return }
        await applyHandEditSplit(utteranceID: utteranceID, sentences: freshResults)
    }

    /// Mic-mode (no source audio) hand-edit dispatcher. Inherits
    /// the parent's dimensional / acoustic-categorical / speech-
    /// boost; the speaker is *re-voted* against the cumulative
    /// diarizer timeline for each plan's sub-range, so a commit
    /// can shift the row's speaker assignment when the timeline
    /// has accumulated more observations since the row was
    /// originally finalized. Falls back to the parent's
    /// `speakerID` when the timeline has no overlap with the
    /// plan's window. Routes to `applyHandEdit` for one sentence
    /// and `applyHandEditSplit` for many — proportional
    /// per-character time allocation keeps the overall duration
    /// `[original.start, original.end]` consistent across the
    /// resulting rows.
    private func runTextOnlyHandEdit(
        utteranceID: UUID,
        plans: [HandEditPlan],
        original: UtteranceEstimate,
        pipeline: AnalysisPipeline
    ) async {
        if plans.count == 1 {
            do {
                let speaker = revoteSpeakerFromTimeline(
                    plan: plans[0],
                    fallback: original.speakerID
                )
                let parent = inheritedTimeStub(
                    from: original,
                    plan: plans[0],
                    speakerID: speaker
                )
                let fresh = try await pipeline.reanalyzeTextOnly(
                    text: plans[0].text,
                    inheriting: parent
                )
                applyHandEdit(utteranceID: utteranceID, fresh: fresh)
            } catch {
                AppLog.app.error(
                    "commitHandEdit (text-only) failed: \(String(describing: error), privacy: .public)"
                )
            }
            return
        }
        var freshResults: [HandEditSplitResult] = []
        freshResults.reserveCapacity(plans.count)
        for (i, plan) in plans.enumerated() {
            do {
                let speaker = revoteSpeakerFromTimeline(
                    plan: plan,
                    fallback: original.speakerID
                )
                let parent = inheritedTimeStub(
                    from: original,
                    plan: plan,
                    speakerID: speaker
                )
                let fresh = try await pipeline.reanalyzeTextOnly(
                    text: plan.text,
                    inheriting: parent
                )
                // Mic-mode has no audio to re-extract from, so no
                // embedding is captured for these rows. Cluster
                // scatter falls back to centroid arrow on those,
                // consistent with the original mic-mode behavior.
                freshResults.append(HandEditSplitResult(
                    estimate: fresh,
                    embedding: nil
                ))
            } catch {
                AppLog.app.error(
                    "commitHandEdit (text-only) split[\(i, privacy: .public)] failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
        guard !freshResults.isEmpty else { return }
        await applyHandEditSplit(utteranceID: utteranceID, sentences: freshResults)
    }

    /// Per-instant majority over the cumulative diarizer timeline
    /// for `plan.[start, end]`. Used by the mic-mode (no-audio)
    /// hand-edit path to re-attribute a row's speaker without
    /// running the diarizer model — the timeline still holds the
    /// streaming-pass observations (and is persisted across Save),
    /// so re-tallying with the now-fuller record can move the
    /// vote even though no fresh audio is processed.
    private func revoteSpeakerFromTimeline(
        plan: HandEditPlan,
        fallback: String
    ) -> String {
        AnalysisPipeline.dominantSpeakerInSegments(
            diarizationTimeline,
            from: plan.start,
            to: plan.end,
            fallback: fallback
        )
    }

    /// Build a synthetic "parent" utterance that carries the
    /// inherited dimensional / acoustic-categorical / speech-
    /// boost fields, the plan's time range, and the supplied
    /// `speakerID` (either the inherited parent's or a fresh
    /// timeline re-vote). Fed to `reanalyzeTextOnly` so the
    /// fusion step sees the right inputs for this slice.
    private func inheritedTimeStub(
        from original: UtteranceEstimate,
        plan: HandEditPlan,
        speakerID: String
    ) -> UtteranceEstimate {
        UtteranceEstimate(
            id: original.id,
            speakerID: speakerID,
            start: plan.start,
            end: plan.end,
            transcript: plan.text,
            asrConfidence: 1.0,
            dimensional: original.dimensional,
            acousticCategorical: original.acousticCategorical,
            plutchik: nil,
            textBackend: nil,
            speechBoost: original.speechBoost,
            wasReevaluated: nil,
            wasHandEdited: nil,
            fusedValence: nil,
            fusedArousal: nil,
            fusedDominance: nil,
            fusedTopLabel: nil
        )
    }

    /// Read the two audio chunks `commitHandEdit` needs: a wide
    /// diarizer window (slice ± `rediarizePadSec`, clamped to
    /// file bounds) so Sortformer has real surrounding audio to
    /// work with, and the unpadded SER window the user committed.
    /// Both reads happen on the same detached task so the file
    /// I/O + AVAudioConverter pass don't hitch the MainActor.
    /// Returns nil on read failure or when the SER chunk came back
    /// empty (e.g. the user dialed past EOF).
    private func readHandEditAudio(
        url: URL,
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) async -> (diar: AudioChunk, ser: AudioChunk)? {
        let totalDur: TimeInterval = fileTotalAudioDuration
            ?? max(newEnd + Self.rediarizePadSec, newEnd)
        let diarStart = max(0, newStart - Self.rediarizePadSec)
        let diarEnd = min(totalDur, newEnd + Self.rediarizePadSec)
        let diarChunk: AudioChunk
        let serChunk: AudioChunk
        do {
            (diarChunk, serChunk) = try await Task.detached(priority: .userInitiated) {
                let dc = try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: diarStart,
                    end: diarEnd
                )
                let sc = try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: newStart,
                    end: newEnd
                )
                return (dc, sc)
            }.value
        } catch {
            AppLog.app.error(
                "commitHandEdit: parent slice read failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        guard !serChunk.samples.isEmpty else {
            AppLog.app.warning("commitHandEdit: empty audio slice")
            return nil
        }
        AppLog.app.info(
            "commitHandEdit: SER window [\(newStart, privacy: .public)..\(newEnd, privacy: .public)] s, diarizer window [\(diarStart, privacy: .public)..\(diarEnd, privacy: .public)] s (pad=\(Self.rediarizePadSec, privacy: .public)s)"
        )
        return (diarChunk, serChunk)
    }

    /// Single-sentence hand-edit: resolve the speaker for the
    /// whole window from the diarizer, then run SER + fusion on
    /// the unpadded SER chunk and stamp the result onto the row
    /// via `applyHandEdit`. No-op on pipeline error.
    private func runSingleSentenceHandEdit(
        utteranceID: UUID,
        plan: HandEditPlan,
        diarChunk: AudioChunk,
        serChunk: AudioChunk,
        fallbackSpeaker: String,
        pipeline: AnalysisPipeline
    ) async {
        let speakers = await pipeline.resolveSpeakersForRanges(
            audio: diarChunk,
            ranges: [(start: plan.start, end: plan.end)],
            fallback: fallbackSpeaker
        )
        let speaker = speakers.first ?? fallbackSpeaker
        do {
            let asr = ASRSegment(
                text: plan.text,
                start: plan.start,
                end: plan.end,
                // User-verified text is presumed correct, so weight
                // the text side at full confidence in fusion.
                confidence: 1.0,
                tokens: []
            )
            let (fresh, _) = try await pipeline.processSegment(
                asr: asr,
                segmentAudio: serChunk,
                fallbackSpeakerID: speaker
            )
            applyHandEdit(utteranceID: utteranceID, fresh: fresh)
            if let embedding = await pipeline.extractSpeakerEmbedding(
                audio: serChunk.samples
            ) {
                utteranceEmbeddings[utteranceID] = embedding
                await pinObservationSegmentID(
                    utteranceID: utteranceID,
                    embedding: embedding,
                    speakerID: speaker
                )
            }
        } catch {
            AppLog.app.error("commitHandEdit failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Multi-sentence hand-edit: diarize ONCE on the full parent
    /// window (so Sortformer has speaker-boundary context the per-
    /// slice reads couldn't provide), then loop the SER pipeline
    /// per sentence-slice with the resolved speaker as fallback.
    /// Returns the freshly-fused per-sentence estimates in order
    /// (skipping any slice whose audio came back empty or whose
    /// pipeline call threw). Caller hands the result to
    /// `applyHandEditSplit`.
    /// Per-slice result of a multi-sentence hand-edit. Pairs the
    /// freshly-fused estimate with the speaker embedding extracted
    /// from that slice's audio so the controller can stamp
    /// `utteranceEmbeddings` for each split row — without this,
    /// only the first slice (which inherits the parent's id and
    /// therefore the parent's pre-edit embedding) ends up with a
    /// cluster-scatter node, while the sibling rows fall back to
    /// the centroid arrow.
    private struct HandEditSplitResult {
        let estimate: UtteranceEstimate
        let embedding: [Float]?
    }

    private func runMultiSentenceHandEdit(
        plans: [HandEditPlan],
        diarChunk: AudioChunk,
        url: URL,
        fallbackSpeaker: String,
        pipeline: AnalysisPipeline
    ) async -> [HandEditSplitResult] {
        // Split path: ask the diarizer for verdicts on the
        // sub-ranges WITHOUT enrolling the audio into the speaker
        // database. A hand-edit split is the user carving up a
        // single original utterance into N pieces — they want the
        // dominant-speaker call per piece, not for each piece's
        // audio to fold itself into the centroids and skew future
        // decisions. The other two `resolveSpeakersForRanges` call
        // sites (re-evaluation, single-sentence hand-edit) keep the
        // default behaviour of enrolling.
        let sliceSpeakers = await pipeline.resolveSpeakersForRanges(
            audio: diarChunk,
            ranges: plans.map { ($0.start, $0.end) },
            fallback: fallbackSpeaker,
            preserveSpeakerDatabase: true
        )
        // Push the per-slice speakers back onto the cumulative
        // timeline so the strip reflects the same per-instant
        // labels the hand-edit's split rows will show. Plans are
        // contiguous and cover `[plans.first.start, plans.last.end]`
        // by construction (see `planHandEditSentences`).
        if let firstStart = plans.first?.start,
           let lastEnd = plans.last?.end {
            let timelineSlices: [DiarizedSegment] = plans.enumerated().map { i, plan in
                let speaker = i < sliceSpeakers.count ? sliceSpeakers[i] : fallbackSpeaker
                return DiarizedSegment(
                    speakerID: speaker,
                    start: plan.start,
                    end: plan.end
                )
            }
            rewriteTimelineWithSlices(
                coveringStart: firstStart,
                coveringEnd: lastEnd,
                slices: timelineSlices
            )
        }
        var freshResults: [HandEditSplitResult] = []
        freshResults.reserveCapacity(plans.count)
        for (i, plan) in plans.enumerated() {
            let speaker = i < sliceSpeakers.count ? sliceSpeakers[i] : fallbackSpeaker
            do {
                let chunk = try await Task.detached(priority: .userInitiated) {
                    try Self.readAudioChunkForReevaluation(
                        fileURL: url,
                        start: plan.start,
                        end: plan.end
                    )
                }.value
                guard !chunk.samples.isEmpty else { continue }
                let asr = ASRSegment(
                    text: plan.text,
                    start: plan.start,
                    end: plan.end,
                    confidence: 1.0,
                    tokens: []
                )
                let (fresh, _) = try await pipeline.processSegment(
                    asr: asr,
                    segmentAudio: chunk,
                    fallbackSpeakerID: speaker
                )
                let embedding = await pipeline.extractSpeakerEmbedding(
                    audio: chunk.samples
                )
                freshResults.append(HandEditSplitResult(
                    estimate: fresh,
                    embedding: embedding
                ))
            } catch {
                AppLog.app.error(
                    "commitHandEdit split[\(i, privacy: .public)] failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
        return freshResults
    }

    /// One sentence's slot in a multi-sentence hand-edit commit:
    /// the trimmed sentence text plus the time range carved out of
    /// the user-supplied `[newStart, newEnd]` window for it.
    private struct HandEditPlan {
        let text: String
        let start: TimeInterval
        let end: TimeInterval
    }

    /// Allocate per-sentence time slots inside the user's edit
    /// window proportionally to character count. The last slot
    /// pins to `newEnd` exactly so floating-point drift doesn't
    /// leave a sub-millisecond gap. Returns `[]` when the input
    /// can't be sliced sensibly (empty sentences, zero total
    /// chars, or the slot collapsing to zero width).
    private static func planHandEditSentences(
        _ sentences: [String],
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) -> [HandEditPlan] {
        let totalChars = sentences.reduce(0) { $0 + $1.count }
        guard totalChars > 0, newEnd > newStart else { return [] }
        let totalDuration = newEnd - newStart
        var plans: [HandEditPlan] = []
        plans.reserveCapacity(sentences.count)
        var charsConsumed = 0
        for (i, sentence) in sentences.enumerated() {
            let sliceStart = newStart
                + Double(charsConsumed) / Double(totalChars) * totalDuration
            charsConsumed += sentence.count
            let sliceEnd = i == sentences.count - 1
                ? newEnd
                : newStart
                    + Double(charsConsumed) / Double(totalChars) * totalDuration
            guard sliceEnd > sliceStart else { continue }
            plans.append(HandEditPlan(text: sentence, start: sliceStart, end: sliceEnd))
        }
        return plans
    }

    /// Mirror of `applyReevaluation` for the hand-edit path.
    /// Stamps `wasHandEdited = true` and clears `wasReevaluated`
    /// (the row is no longer a model-driven re-eval, it's a user
    /// correction).
    private func applyHandEdit(utteranceID: UUID, fresh: UtteranceEstimate) {
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]
        // Carry `fresh.speakerID` (resolved against the cumulative
        // diarizer timeline in `commitHandEdit`) rather than the
        // pre-edit `original.speakerID`, so a hand-edit that moved
        // the range across a speaker boundary actually re-labels.
        utterances[index] = Self.mergedEstimate(
            id: original.id,
            speakerID: fresh.speakerID,
            speechBoost: original.speechBoost,
            fresh: fresh,
            wasReevaluated: nil,
            wasHandEdited: true
        )
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    /// Multi-sentence hand-edit apply path. Replaces the parent row
    /// in place with the first sentence (keeping the original `id`,
    /// `speakerID`, and `speechBoost`) and inserts the remaining
    /// sentences as fresh-id siblings immediately after it. Every
    /// row gets `wasHandEdited = true` and `wasReevaluated = nil`.
    /// Sibling ids are tracked in `handEditChildren` so a revert on
    /// the parent can also remove the siblings the split spawned.
    private func applyHandEditSplit(
        utteranceID: UUID,
        sentences: [HandEditSplitResult]
    ) async {
        guard !sentences.isEmpty else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]

        var newRows: [UtteranceEstimate] = []
        newRows.reserveCapacity(sentences.count)
        var siblingIDs: [UUID] = []
        for (i, result) in sentences.enumerated() {
            // First row keeps the original id (so existing
            // references stay valid); siblings get fresh ids and
            // are tracked for revert-time cleanup. Speaker is the
            // per-sentence diarizer verdict — different sentences
            // in a split can legitimately differ in speaker.
            let rowID = i == 0 ? original.id : UUID()
            if i > 0 { siblingIDs.append(rowID) }
            newRows.append(Self.mergedEstimate(
                id: rowID,
                speakerID: result.estimate.speakerID,
                speechBoost: original.speechBoost,
                fresh: result.estimate,
                wasReevaluated: nil,
                wasHandEdited: true
            ))
            // Stamp the per-slice speaker embedding under the
            // assigned row id so the cluster scatter can pin each
            // split row to its own observation. The pre-edit
            // parent's embedding stays at `original.id` for the
            // first slice; nil-valued results (mic-mode / failed
            // extraction) leave any existing entry untouched.
            if let embedding = result.embedding {
                utteranceEmbeddings[rowID] = embedding
                await pinObservationSegmentID(
                    utteranceID: rowID,
                    embedding: embedding,
                    speakerID: result.estimate.speakerID
                )
            }
        }

        utterances.remove(at: index)
        utterances.insert(contentsOf: newRows, at: index)
        if !siblingIDs.isEmpty {
            handEditChildren[utteranceID] = siblingIDs
        }
        // Sentences carve up the original `[newStart, newEnd]` window
        // in order, so their starts are already monotonic among
        // themselves. Resort once to be safe in case a neighbouring
        // row overlaps the window (e.g. overlapping speakers).
        utterances.sort { $0.start < $1.start }

        commitUtteranceChanges()
    }

    /// Split a user-typed transcript into sentences. Terminators
    /// (`。！？．.!?` and `\n`) stay with the preceding sentence;
    /// trailing whitespace is trimmed; empty fragments are dropped.
    /// Returns `[trimmed]` when no terminator appears so the
    /// single-sentence path stays a no-op refactor of the previous
    /// behaviour.
    private static func splitTranscriptIntoSentences(_ text: String) -> [String] {
        let terminators: Set<Character> = ["。", "！", "？", "．", ".", "!", "?", "\n"]
        var sentences: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if terminators.contains(ch) {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { sentences.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { sentences.append(tail) }
        return sentences
    }


}
