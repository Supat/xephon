import SwiftUI
import Diarization
import Fusion

/// The scrollable list of utterances on the right pane, plus the
/// auto-scroll / unread-capsule logic that depends on
/// `ScrollViewReader`. Extracted from ContentView for readability —
/// the list itself plus its `.onChange`/`.onKeyPress`/`.overlay`
/// modifiers were ~180 lines that crowded everything else.
///
/// Filter state lives on the parent because `displayedSummary` on
/// the left pane reads it too; we take the resolved/filtered
/// `items` as input and the action surfaces (renaming, editing) as
/// closures. The List's selection and the per-row mutations on the
/// parent's @State come back through @Binding.
/// Memo cache for `speakerMismatchedIDs`. Held as a `final class`
/// via `@State` so internal mutation from inside the body computed
/// property doesn't trip SwiftUI invalidation — the reference is
/// stable across renders, only the inner fields change. Without
/// this, scrolling a 500-row list fired O(N × 256) vote loops on
/// every body re-eval (visibility changes invalidate body too), so
/// every flick stutters. Key includes `utterancesVersion` (any row
/// edit) and `timelineCount` (cumulative timeline growth) so a
/// re-vote happens iff one of the inputs to the mismatch verdict
/// could have changed.
private final class MismatchMemo {
    struct Key: Equatable {
        let utterancesVersion: Int
        let timelineCount: Int
        let utteranceCount: Int
    }
    var key: Key?
    var set: Set<UUID> = []
}

struct TranscriptList: View {
    let recorder: RecordingController
    let items: [(idx: Int, u: UtteranceEstimate)]

    @Binding var selectedUtteranceID: UUID?
    /// Set by the diarizer-timeline-strip tap handler in
    /// ContentView. Always scrolls to the requested id, even when
    /// it's already selected or already on screen — so a tap on
    /// the strip jumps to that row regardless of current state.
    /// Cleared back to nil after handling so a second tap on the
    /// same time re-fires.
    @Binding var scrollRequestUtteranceID: UUID?
    @Binding var expandedUtteranceIDs: Set<UUID>
    @Binding var visibleUtteranceIDs: Set<UUID>
    @Binding var hasUnreadUtterance: Bool
    @Binding var normalizedTranscriptCache: [UUID: String]
    @Binding var searchText: String
    @Binding var selectedLabelFilter: String?
    @Binding var selectedSpeakerFilter: String?
    /// Bound through so a tap on dead list space can drop search
    /// focus without TranscriptList owning the field itself.
    var searchFieldFocused: FocusState<Bool>.Binding

    let playbackAvailability: (UtteranceEstimate) -> UtteranceRow.PlaybackAvailability
    let reevaluateAvailability: (UtteranceEstimate) -> UtteranceRow.ReevaluateAvailability
    let onToggleExpansion: (UUID) -> Void
    let onRenameSpeaker: (UtteranceEstimate) -> Void
    let onPromoteNewSpeaker: (UtteranceEstimate) -> Void
    let onCorrectSpeaker: (UtteranceEstimate, String) -> Void
    /// Fires when the user long-presses a row's mismatch warning
    /// glyph. ContentView resolves the timeline's dominant speaker
    /// for the row's `[start, end]` window and calls
    /// `recorder.reassignSpeaker` so the row label aligns with what
    /// the diarizer has been collectively observing.
    let onCorrectMismatch: (UtteranceEstimate) -> Void
    let onEditTranscript: (UtteranceEstimate) -> Void
    let refreshSearchCache: () -> Void

    @State private var mismatchMemo = MismatchMemo()

    var body: some View {
        let mismatched = speakerMismatchedIDs
        ScrollViewReader { proxy in
            List(items, id: \.u.id, selection: $selectedUtteranceID) { item in
                row(for: item, hasSpeakerMismatch: mismatched.contains(item.u.id))
            }
            .listStyle(.plain)
            // Any tap on the list dismisses keyboard focus on the
            // search field. Selection is now owned by each row's
            // own `.onTapGesture` (tap-to-focus / re-tap-to-unfocus
            // toggle), so this handler intentionally does NOT clear
            // `selectedUtteranceID` — doing so would race with the
            // row's own focus toggle and intermittently strip the
            // focus the user just established.
            .simultaneousGesture(
                TapGesture().onEnded {
                    if searchFieldFocused.wrappedValue { searchFieldFocused.wrappedValue = false }
                }
            )
            // Keep the selected row scrolled into view when the user
            // arrow-keys past the visible window. The system handles this
            // for tap-driven selection automatically; for keyboard-driven
            // selection on a plain list it's not always automatic.
            .onChange(of: selectedUtteranceID) { _, newID in
                guard let newID else { return }
                if !visibleUtteranceIDs.contains(newID) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(newID, anchor: .center)
                    }
                }
            }
            // Explicit scroll request — e.g. a tap on the
            // diarizer-timeline strip. Scrolls unconditionally
            // (visibility-gated path above wouldn't re-scroll an
            // already-onscreen row) and resets the binding so a
            // subsequent tap on the same time fires again.
            .onChange(of: scrollRequestUtteranceID) { _, newID in
                guard let newID else { return }
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(newID, anchor: .center)
                }
                scrollRequestUtteranceID = nil
            }
            // Esc clears the selection. Returning `.ignored` when nothing
            // is selected lets the system route Esc to its default
            // handlers (e.g. dismissing a sheet or popover) instead of
            // silently swallowing the key.
            .onKeyPress(.escape) {
                if selectedUtteranceID != nil {
                    selectedUtteranceID = nil
                    return .handled
                }
                return .ignored
            }
            // Space toggles the expanded detail panel on the currently
            // selected row. Long-press on a row does the same thing.
            // Ignored when nothing is selected so the keystroke can fall
            // through to other handlers.
            .onKeyPress(.space) {
                guard let id = selectedUtteranceID else { return .ignored }
                onToggleExpansion(id)
                return .handled
            }
            // Observe the FILTERED last so adding an utterance that
            // doesn't match the active filter doesn't auto-scroll, and
            // adding one that does match scrolls the (possibly shorter)
            // filtered view to the right place.
            .onChange(of: items.last?.u.id) { oldLastID, newLastID in
                guard let newLastID else {
                    hasUnreadUtterance = false
                    return
                }
                // "Was the user following along?" must be answered from
                // the PREVIOUS last utterance — by the time this closure
                // fires, the array's `last` is already the new entry,
                // whose row hasn't been laid out yet, so its id can't
                // be in `visibleUtteranceIDs`.
                let wasFollowing = oldLastID.map { visibleUtteranceIDs.contains($0) } ?? false
                if oldLastID == nil || wasFollowing {
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(newLastID, anchor: .bottom)
                    }
                    hasUnreadUtterance = false
                } else {
                    hasUnreadUtterance = true
                }
            }
            .onChange(of: recorder.utterances.isEmpty) { _, isEmpty in
                if isEmpty {
                    hasUnreadUtterance = false
                    selectedUtteranceID = nil
                    normalizedTranscriptCache.removeAll(keepingCapacity: true)
                    // New session (mic record or file analysis) just
                    // cleared the utterance list — reset the filter
                    // controls so the empty list isn't shown through
                    // a stale filter the user no longer remembers
                    // setting.
                    searchText = ""
                    selectedLabelFilter = nil
                    selectedSpeakerFilter = nil
                }
            }
            .onChange(of: recorder.utterances.count, initial: true) { _, _ in
                refreshSearchCache()
            }
            .onChange(of: isLastUtteranceVisible) { _, visible in
                if visible { hasUnreadUtterance = false }
            }
            .overlay(alignment: .bottom) {
                if hasUnreadUtterance {
                    NewUtteranceCapsule {
                        guard let lastID = items.last?.u.id else { return }
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(lastID, anchor: .bottom)
                        }
                        hasUnreadUtterance = false
                    }
                    .padding(.bottom, 16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.easeOut(duration: 0.2), value: hasUnreadUtterance)
        }
    }

    /// One row, with all of UtteranceRow's bindings wired to the
    /// parent's actions. Factored out of `body` purely for
    /// readability — the row construction is 30+ lines on its own.
    @ViewBuilder
    private func row(
        for item: (idx: Int, u: UtteranceEstimate),
        hasSpeakerMismatch: Bool
    ) -> some View {
        UtteranceRow(
            number: item.idx + 1,
            utterance: item.u,
            isExpanded: expandedUtteranceIDs.contains(item.u.id),
            onToggleExpanded: { onToggleExpansion(item.u.id) },
            playback: playbackAvailability(item.u),
            onPlaybackToggle: { recorder.togglePlayback(for: item.u) },
            reevaluate: reevaluateAvailability(item.u),
            onReevaluate: {
                Task { await recorder.reevaluate(item.u) }
            },
            onRevert: { recorder.revertReevaluation(item.u) },
            speakerCustomName: recorder.speakerDisplayName(forStored: item.u.speakerID),
            hasSpeakerMismatch: hasSpeakerMismatch,
            knownSpeakerIDs: recorder.knownSpeakerIDs(),
            speakerDisplayName: { recorder.speakerDisplayName(forStored: $0) },
            fusionAcousticWeight: recorder.fusionAcousticWeight,
            fusionTextWeightFloor: recorder.fusionTextWeightFloor,
            onReassignSpeaker: { newSpeakerID in
                recorder.reassignSpeaker(
                    utteranceID: item.u.id,
                    to: newSpeakerID
                )
            },
            onCorrectSpeaker: { targetSpeakerID in
                onCorrectSpeaker(item.u, targetSpeakerID)
            },
            onPromoteNewSpeaker: { onPromoteNewSpeaker(item.u) },
            onRenameSpeaker: { onRenameSpeaker(item.u) },
            onEditTranscript: { onEditTranscript(item.u) },
            // Tap toggles focus: re-tapping the already-focused
            // row clears the selection, tapping any other row
            // moves focus there. Writes the same `selectedUtter-
            // anceID` binding `List(selection:)` reads, so the
            // visual highlight + keyboard arrow nav follow.
            onTap: {
                if selectedUtteranceID == item.u.id {
                    selectedUtteranceID = nil
                } else {
                    selectedUtteranceID = item.u.id
                }
            },
            onCorrectMismatch: { onCorrectMismatch(item.u) },
            teachingDiarizer: Binding(
                get: { recorder.teachingDiarizer },
                set: { recorder.teachingDiarizer = $0 }
            )
            // Note: the gate "only when source audio is present"
            // moved to the dialog itself, which hides the time
            // spinners + play button when the session is mic-mode.
            // The transcript-text long-press now opens the editor
            // unconditionally (as long as recording isn't running,
            // which is gated upstream in ContentView).
        )
        .id(item.u.id)
        // Visibility tracking is belt-and-suspenders: neither
        // signal on its own is reliable in a `.plain` List, so we
        // combine both and union their verdicts.
        //
        // `.onScrollVisibilityChange` (iOS 18+) is frame-based —
        // catches the visibility transition while the row stays
        // mounted, including the layout-flux window of a row
        // expansion where `.onDisappear` was misfiring and
        // leaving phantom entries that stretched the timeline
        // highlighter unboundedly.
        //
        // `.onDisappear` catches the lazy-unmount case: the List
        // detaches rows that scroll past its prefetch window
        // entirely, and once a view leaves the hierarchy
        // `.onScrollVisibilityChange` has no frame to compare
        // and never fires `false`. Without this signal, IDs
        // accumulate forever and the highlighter grows from the
        // start.
        //
        // `threshold: 0.05` keeps the scroll-visibility semantics
        // matched to the old `.onAppear` (any sliver counts as
        // visible) instead of the default 0.5 (half-visible)
        // which would visibly lag the highlighter at the
        // viewport's top and bottom edges.
        //
        // The contains-guards skip redundant binding writes that
        // would otherwise invalidate ContentView's body on every
        // fling-scroll frame.
        .onAppear {
            if !visibleUtteranceIDs.contains(item.u.id) {
                visibleUtteranceIDs.insert(item.u.id)
            }
        }
        .onDisappear {
            if visibleUtteranceIDs.contains(item.u.id) {
                visibleUtteranceIDs.remove(item.u.id)
            }
        }
        .onScrollVisibilityChange(threshold: 0.05) { visible in
            if visible {
                if !visibleUtteranceIDs.contains(item.u.id) {
                    visibleUtteranceIDs.insert(item.u.id)
                }
            } else {
                if visibleUtteranceIDs.contains(item.u.id) {
                    visibleUtteranceIDs.remove(item.u.id)
                }
            }
        }
        .tag(item.u.id)
        // Replace the default pale-tint selection highlight with
        // a darker accent-tinted background. `.listRowBackground`
        // overrides the system selection paint on `.plain` lists.
        .listRowBackground(
            selectedUtteranceID == item.u.id
                ? Color.accentColor.opacity(0.28)
                : Color.clear
        )
    }

    /// Utterance ids whose stored `speakerID` disagrees with the
    /// cumulative diarizer timeline's majority verdict for that
    /// row's `[start, end]` window. Drives the warning glyph
    /// after the speaker chip — purely informational, the user
    /// can reconcile by re-evaluating the row or reassigning the
    /// speaker via the chip menu.
    ///
    /// Memoized on `(utterancesVersion, timelineCount,
    /// utteranceCount)` because body re-eval fires on every
    /// scroll (visibility tracker writes to a binding), and the
    /// underlying vote is O(samples × segments) per row —
    /// recomputing on every scroll tick on a 500-row session
    /// noticeably stuttered fling-scrolling. Re-runs only when
    /// one of the verdict inputs actually changed. Skipped
    /// entirely when the timeline is empty so freshly-loaded
    /// sessions (where the timeline blob hadn't been persisted
    /// yet) don't pay for an all-empty pass.
    private var speakerMismatchedIDs: Set<UUID> {
        let key = MismatchMemo.Key(
            utterancesVersion: recorder.utterancesVersion,
            timelineCount: recorder.diarizationTimeline.count,
            utteranceCount: recorder.utterances.count
        )
        if mismatchMemo.key == key { return mismatchMemo.set }
        let timeline = recorder.diarizationTimeline
        var result: Set<UUID> = []
        if !timeline.isEmpty {
            for u in recorder.utterances {
                let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                    timeline,
                    from: u.start,
                    to: u.end,
                    fallback: u.speakerID
                )
                if dominant != u.speakerID {
                    result.insert(u.id)
                }
            }
        }
        mismatchMemo.key = key
        mismatchMemo.set = result
        return result
    }

    /// True when the bottom-most filtered row is currently on
    /// screen. Used by the unread-capsule logic to clear the
    /// "unread" latch the moment the user scrolls (or the layout
    /// catches up) so the latest utterance is in view.
    private var isLastUtteranceVisible: Bool {
        guard let last = items.last?.u.id else { return true }
        return visibleUtteranceIDs.contains(last)
    }
}
