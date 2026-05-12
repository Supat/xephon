import SwiftUI
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
struct TranscriptList: View {
    let recorder: RecordingController
    let items: [(idx: Int, u: UtteranceEstimate)]
    let distinctSpeakerCount: Int

    @Binding var selectedUtteranceID: UUID?
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
    let onEditTranscript: (UtteranceEstimate) -> Void
    let refreshSearchCache: () -> Void

    var body: some View {
        ScrollViewReader { proxy in
            List(items, id: \.u.id, selection: $selectedUtteranceID) { item in
                row(for: item)
            }
            .listStyle(.plain)
            // Any tap on the list — row, dead-space between rows,
            // header gap, anywhere — clears the previously selected
            // utterance's focus highlight and dismisses any
            // keyboard focus on the search field. When the tap
            // lands on a row, the List's own tap-to-select handler
            // runs after this gesture and writes the row's id back
            // into `selectedUtteranceID`, so legitimate row
            // selection still works; only "stale" focus that the
            // user has moved past gets cleared.
            .simultaneousGesture(
                TapGesture().onEnded {
                    if searchFieldFocused.wrappedValue { searchFieldFocused.wrappedValue = false }
                    if selectedUtteranceID != nil { selectedUtteranceID = nil }
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
    private func row(for item: (idx: Int, u: UtteranceEstimate)) -> some View {
        UtteranceRow(
            number: item.idx + 1,
            utterance: item.u,
            isMultiSpeaker: distinctSpeakerCount > 1,
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
            knownSpeakerIDs: recorder.knownSpeakerIDs(),
            speakerDisplayName: { recorder.speakerDisplayName(forStored: $0) },
            onReassignSpeaker: { newSpeakerID in
                recorder.reassignSpeaker(
                    utteranceID: item.u.id,
                    to: newSpeakerID
                )
            },
            onRenameSpeaker: { onRenameSpeaker(item.u) },
            onEditTranscript: { onEditTranscript(item.u) }
        )
        .id(item.u.id)
        .onAppear { visibleUtteranceIDs.insert(item.u.id) }
        .onDisappear { visibleUtteranceIDs.remove(item.u.id) }
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

    /// True when the bottom-most filtered row is currently on
    /// screen. Used by the unread-capsule logic to clear the
    /// "unread" latch the moment the user scrolls (or the layout
    /// catches up) so the latest utterance is in view.
    private var isLastUtteranceVisible: Bool {
        guard let last = items.last?.u.id else { return true }
        return visibleUtteranceIDs.contains(last)
    }
}
