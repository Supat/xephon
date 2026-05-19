import SwiftUI
import Diarization
import Fusion

/// Right two-thirds of the main split: the filter bar, the three
/// timeline strips (diarizer, emotion labels, fusion contribution),
/// and the transcript list (or a "no matches" placeholder when the
/// filter is too narrow). Falls back to an empty-state placeholder
/// when the recorder has no utterances at all.
///
/// Hands speaker-rename / utterance-edit / mismatch-correction
/// gestures back to the parent via callbacks so the alert / sheet
/// presentations stay at the NavigationStack root.
struct TranscriptPaneView: View {
    let recorder: RecordingController
    @Bindable var filterModel: TranscriptFilterModel

    @Binding var selectedUtteranceID: UUID?
    @Binding var scrollRequestUtteranceID: UUID?
    @Binding var expandedUtteranceIDs: Set<UUID>
    @Binding var visibleUtteranceIDs: Set<UUID>
    @Binding var hasUnreadUtterance: Bool

    var searchFieldFocused: FocusState<Bool>.Binding

    let onRenameSpeaker: (UtteranceEstimate) -> Void
    let onEditTranscript: (UtteranceEstimate) -> Void

    @ViewBuilder
    var body: some View {
        if recorder.utterances.isEmpty {
            ContentUnavailableView(
                String(localized: "transcript.empty.title"),
                systemImage: "waveform",
                description: Text(String(localized: "transcript.empty.subtitle"))
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            VStack(spacing: 0) {
                TranscriptFilterBar(
                    recorder: recorder,
                    model: filterModel,
                    searchFieldFocused: searchFieldFocused
                )
                diarizationTimelineStrip
                emotionTimelineStrip
                fusionContributionStrip
                if filterModel.filteredIndexedUtterances(in: recorder).isEmpty {
                    TranscriptNoMatchesView(model: filterModel)
                } else {
                    transcriptList
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    // MARK: - Timeline strips

    /// Per-session diarizer-timeline visualization. Hidden until
    /// the cumulative timeline has at least one observation and we
    /// can derive a positive total duration. Selecting an utterance
    /// in the list outlines the strip region for that row's audio
    /// range.
    @ViewBuilder
    private var diarizationTimelineStrip: some View {
        let timeline = recorder.diarizationTimeline
        let total = transcriptTotalDuration
        if !timeline.isEmpty, total > 0 {
            DiarizationTimelineStrip(
                segments: timeline,
                totalDuration: total,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = recorder.nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }

    /// Companion emotion-label strip rendered immediately below the
    /// diarizer timeline. Shares X axis + tap behavior with the
    /// speaker strip so a glance across both reveals "this speaker,
    /// this emotion, at this audio time" in one read. Hidden until
    /// at least one utterance has a fused top label — before the
    /// first analysis finishes there's nothing to colour.
    @ViewBuilder
    private var emotionTimelineStrip: some View {
        let total = transcriptTotalDuration
        let hasAnyLabel = recorder.utterances.contains { $0.fusedTopLabel != nil }
        if hasAnyLabel, total > 0 {
            EmotionTimelineStrip(
                utterances: recorder.utterances,
                totalDuration: total,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = recorder.nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            // Tighter vertical pad than the speaker strip so the
            // two strips read as a stacked pair, not two unrelated
            // bars with whitespace between them.
            .padding(.bottom, 6)
        }
    }

    /// Per-utterance modality-balance strip — for each row, a
    /// horizontal bar split into acoustic (blue) / text (orange)
    /// segments sized by `LateFusion.defaultLabelFusionShare`. Lets
    /// the user scan the conversation and spot stretches where one
    /// modality dominated the fused label. Hidden until at least
    /// one utterance carries one of the two modality outputs.
    @ViewBuilder
    private var fusionContributionStrip: some View {
        let total = transcriptTotalDuration
        let hasAnyModality = recorder.utterances.contains {
            $0.acousticCategorical != nil || $0.plutchik != nil
        }
        if hasAnyModality, total > 0 {
            FusionContributionStrip(
                utterances: recorder.utterances,
                totalDuration: total,
                acousticWeight: recorder.fusionAcousticWeight,
                textWeightFloor: recorder.fusionTextWeightFloor,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = recorder.nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            .padding(.bottom, 6)
        }
    }

    /// Conversation duration for the timeline strip's X axis.
    /// Always derived from the latest processed timestamp — the
    /// max of any finalized utterance's end and the diarizer's
    /// latest observation — so the strip grows in real time as
    /// analysis progresses. File mode previously preferred the
    /// source-file's full length here, which made the strip
    /// snap to its final width the moment the file opened and
    /// left the highlighter inching along the leftmost few
    /// percent during analysis. That was jarring — the user
    /// wants the same "depict what we've actually processed so
    /// far" semantics they get from a live mic recording, in both
    /// modes.
    private var transcriptTotalDuration: TimeInterval {
        let utteranceMax = recorder.utterances.map(\.end).max() ?? 0
        let timelineMax = recorder.diarizationTimeline.map(\.end).max() ?? 0
        return max(utteranceMax, timelineMax)
    }

    /// What the timeline strip should outline. Two cases:
    ///
    /// 1. A row is explicitly selected → outline that row's
    ///    `[start, end]` so the user can see where on the
    ///    timeline they tapped.
    /// 2. Nothing is selected → outline the range that spans
    ///    every utterance currently on screen. This keeps the
    ///    strip's highlight tied to the user's reading focus
    ///    even without an explicit selection: scrolling the list
    ///    moves the highlight along with what they're looking at.
    ///
    /// Returns nil when neither case has data (empty list, or
    /// nothing visible yet on first layout). Visibility tracking
    /// uses `.onScrollVisibilityChange` in `TranscriptList` (not
    /// `.onAppear` / `.onDisappear`) so the set stays in sync
    /// even across the layout-flux window of a row expansion;
    /// `.onDisappear` was firing unreliably in that case and
    /// leaving phantom entries that stretched the highlighter.
    private var selectedUtteranceRange: (start: TimeInterval, end: TimeInterval)? {
        if let id = selectedUtteranceID,
           let u = recorder.utterances.first(where: { $0.id == id }) {
            return (start: u.start, end: u.end)
        }
        // Single-pass min/max instead of filter + map + min/map +
        // max — three allocations and three iterations collapse to
        // one. Body re-fires per scroll because the per-row
        // visibility tracker writes to `visibleUtteranceIDs`, so
        // this getter runs on every flick.
        //
        // Expanded rows are excluded from the visible-range vote.
        // An expansion is tall enough to push its neighbors off-
        // screen — without this filter, `visibleUtteranceIDs`
        // collapses to just the expanded id and the timeline
        // highlight gets visually anchored to that one row even
        // though the user only opened it to read its detail. With
        // the filter, scrolling still drives the highlight; an
        // expansion that dominates the viewport simply lets the
        // highlight fade out (returning nil here) instead of
        // pinning itself.
        var minStart: TimeInterval = .infinity
        var maxEnd: TimeInterval = -.infinity
        for u in recorder.utterances
            where visibleUtteranceIDs.contains(u.id)
                && !expandedUtteranceIDs.contains(u.id) {
            if u.start < minStart { minStart = u.start }
            if u.end > maxEnd { maxEnd = u.end }
        }
        guard minStart.isFinite else { return nil }
        return (start: minStart, end: maxEnd)
    }

    // MARK: - Transcript list

    private var transcriptList: some View {
        TranscriptList(
            recorder: recorder,
            filterModel: filterModel,
            items: filterModel.filteredIndexedUtterances(in: recorder),
            selectedUtteranceID: $selectedUtteranceID,
            scrollRequestUtteranceID: $scrollRequestUtteranceID,
            expandedUtteranceIDs: $expandedUtteranceIDs,
            visibleUtteranceIDs: $visibleUtteranceIDs,
            hasUnreadUtterance: $hasUnreadUtterance,
            searchFieldFocused: searchFieldFocused,
            onToggleExpansion: toggleExpansion,
            onRenameSpeaker: onRenameSpeaker,
            onPromoteNewSpeaker: { u in
                Task { _ = await recorder.promoteUtteranceToNewSpeaker(utteranceID: u.id) }
            },
            onAffirmSpeaker: { u in
                Task { _ = await recorder.affirmUtteranceSpeaker(utteranceID: u.id) }
            },
            onCorrectSpeaker: { u, target in
                Task {
                    _ = await recorder.correctUtteranceSpeaker(
                        utteranceID: u.id,
                        to: target
                    )
                }
            },
            onCorrectMismatch: { u in
                // Long-press on the orange mismatch glyph accepts
                // the cumulative timeline's verdict for this row.
                // Recompute the dominant speaker on demand (it's
                // ~256 samples × a handful of segments, well under
                // a millisecond) so we always read the freshest
                // timeline state instead of caching it alongside
                // the mismatch flag. Falls back to a no-op when
                // the timeline has no overlap with the row or the
                // verdict no longer disagrees (the glyph went
                // stale between render and tap).
                let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                    recorder.diarizationTimeline,
                    from: u.start,
                    to: u.end,
                    fallback: u.speakerID
                )
                guard dominant != u.speakerID else { return }
                recorder.reassignSpeaker(utteranceID: u.id, to: dominant)
            },
            onEditTranscript: onEditTranscript
        )
    }

    // MARK: - Per-row expansion

    /// Flip the per-utterance expansion state. Invoked by long-press
    /// on a row and by Space-key while a row is the list selection.
    private func toggleExpansion(_ id: UUID) {
        if expandedUtteranceIDs.contains(id) {
            expandedUtteranceIDs.remove(id)
        } else {
            expandedUtteranceIDs.insert(id)
        }
    }
}
