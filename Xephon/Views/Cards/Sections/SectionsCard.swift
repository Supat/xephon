import SwiftUI
import Fusion

/// Card hosting the user-defined `ConversationSection` list.
/// Lives on the Sections page in `ControlPaneView`'s TabView
/// between Speakers and Keywords. Each row shows the
/// section's title + start-end time range; a tap opens the
/// editor sheet pre-filled, a trash button removes the row.
/// "Add Section" at the bottom raises the editor for a fresh
/// section.
///
/// Sections live in memory only (`SectionStore`) — they
/// reference per-session utterance IDs, so the card disables
/// the Add button when the session is empty and surfaces a
/// "no sections yet" empty state otherwise.
struct SectionsCard: View {
    let recorder: RecordingController
    let store: SectionStore
    /// ID of the utterance currently focused in the
    /// transcript pane, or nil when nothing is selected.
    /// When non-nil and the id resolves to a live utterance,
    /// the card surfaces two quick-add buttons that create
    /// an incomplete section anchored at that utterance —
    /// fastest way to bookmark "this is where the section
    /// starts" (or ends) while listening.
    let selectedUtteranceID: UUID?
    /// Raise the per-section summary sheet for the given
    /// section. Driven from the parent so the card itself
    /// doesn't need to know about `LLMSheetCoordinator` /
    /// inflight Task plumbing — it just signals intent.
    let onSummarize: (ConversationSection) -> Void

    @State private var editing: ConversationSection? = nil
    @State private var showingNewSection: Bool = false

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(String(localized: "sections.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if !store.sections.isEmpty {
                    Text("\(store.sections.count)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            if recorder.utterances.isEmpty {
                emptyStateNoUtterances
            } else if store.sections.isEmpty {
                emptyStateNoSections
            } else {
                ForEach(store.sections) { section in
                    SectionsCardRow(
                        section: section,
                        utterances: recorder.utterances,
                        focusedID: validFocusedID,
                        // The summary button on each complete
                        // row is gated on the parent's view of
                        // global summarize availability so a row
                        // can't kick off a second pass while the
                        // overall summary (or another section)
                        // is mid-flight. The row separately
                        // shows a spinner-on-self when it's the
                        // current target — see `isSummarizing`.
                        canSummarize: recorder.summarizerEnabled
                            && recorder.summarizerReady
                            && !recorder.summarizerInferenceRunning,
                        isSummarizing: recorder.summarizingSectionID == section.id,
                        hasCachedSummary: section.cachedSummary != nil,
                        onCompleteWithFocus: {
                            completeSection(section, withFocusedID: validFocusedID)
                        },
                        onSummary: { onSummarize(section) },
                        onEdit: { editing = section },
                        onDelete: { store.remove(id: section.id) }
                    )
                    if section.id != store.sections.last?.id {
                        Divider()
                    }
                }
            }

            Button {
                showingNewSection = true
            } label: {
                Label(
                    String(localized: "sections.add"),
                    systemImage: "plus.circle.fill"
                )
                .font(.callout)
            }
            .buttonStyle(.borderedProminent)
            .disabled(recorder.utterances.isEmpty)
            .frame(maxWidth: .infinity, alignment: .center)
            .padding(.top, 4)

            // Quick-add for the focused utterance — always
            // visible so the affordance is discoverable, but
            // disabled until an utterance is focused in the
            // transcript pane (and that id still resolves to
            // a live row; defensive against stale selection
            // across session swaps). Each button creates an
            // INCOMPLETE section immediately (no editor) so
            // the user can keep listening; the other bound
            // can be filled in later via the row's edit
            // button or the row-level complete-with-focus
            // button (also driven off the same `focusedID`).
            HStack(spacing: 8) {
                Button {
                    guard let focusedID = validFocusedID else { return }
                    store.add(ConversationSection(
                        title: "",
                        startUtteranceID: focusedID,
                        endUtteranceID: nil
                    ))
                } label: {
                    Label(
                        String(localized: "sections.addStart"),
                        systemImage: "arrow.forward.to.line"
                    )
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(validFocusedID == nil)

                Button {
                    guard let focusedID = validFocusedID else { return }
                    store.add(ConversationSection(
                        title: "",
                        startUtteranceID: nil,
                        endUtteranceID: focusedID
                    ))
                } label: {
                    Label(
                        String(localized: "sections.addEnd"),
                        systemImage: "arrow.backward.to.line"
                    )
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)
                .disabled(validFocusedID == nil)
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .sheet(isPresented: $showingNewSection) {
            SectionEditorSheet(
                utterances: recorder.utterances,
                speakerNames: recorder.speakerNameOverrides,
                section: nil,
                onSave: { newSection in
                    store.add(newSection)
                    showingNewSection = false
                },
                onCancel: { showingNewSection = false }
            )
        }
        .sheet(item: $editing) { section in
            SectionEditorSheet(
                utterances: recorder.utterances,
                speakerNames: recorder.speakerNameOverrides,
                section: section,
                onSave: { updated in
                    store.update(updated)
                    editing = nil
                },
                onCancel: { editing = nil }
            )
        }
    }

    /// The focused utterance's id, validated against the
    /// current session's utterance list. Returns nil when
    /// nothing is focused OR the focused id is stale (left
    /// over from a session that's since been replaced).
    /// Drives every focus-dependent affordance in the card
    /// (both the global Add-Start / Add-End buttons and the
    /// per-row complete-with-focus button) off the same
    /// value so they enable / disable in lockstep.
    private var validFocusedID: UUID? {
        guard let id = selectedUtteranceID,
              recorder.utterances.contains(where: { $0.id == id })
        else { return nil }
        return id
    }

    /// Stamp `focusedID` onto whichever bound of `section` is
    /// missing, then `store.update` it. Called by an
    /// incomplete row's complete-with-focus button; no-op
    /// when there's no focused utterance or the section is
    /// already complete. Validates that adding the bound
    /// preserves start ≤ end ordering (the row's button is
    /// already disabled when this would be violated, but
    /// the guard is a belt-and-braces check).
    private func completeSection(
        _ section: ConversationSection,
        withFocusedID focusedID: UUID?
    ) {
        guard let focusedID else { return }
        guard !section.isComplete else { return }
        var updated = section
        if updated.startUtteranceID == nil {
            // Setting start; verify it lands at or before
            // the existing end in chronological order.
            if let endID = updated.endUtteranceID,
               let endIdx = recorder.utterances.firstIndex(where: { $0.id == endID }),
               let focusedIdx = recorder.utterances.firstIndex(where: { $0.id == focusedID }),
               focusedIdx > endIdx {
                return
            }
            updated.startUtteranceID = focusedID
        } else if updated.endUtteranceID == nil {
            // Setting end; verify it lands at or after the
            // existing start in chronological order.
            if let startID = updated.startUtteranceID,
               let startIdx = recorder.utterances.firstIndex(where: { $0.id == startID }),
               let focusedIdx = recorder.utterances.firstIndex(where: { $0.id == focusedID }),
               focusedIdx < startIdx {
                return
            }
            updated.endUtteranceID = focusedID
        }
        store.update(updated)
    }

    @ViewBuilder
    private var emptyStateNoUtterances: some View {
        Text(String(localized: "sections.emptyNoUtterances"))
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
    }

    @ViewBuilder
    private var emptyStateNoSections: some View {
        Text(String(localized: "sections.empty"))
            .font(.caption)
            .foregroundStyle(.secondary)
            .multilineTextAlignment(.center)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
    }
}

/// One row in the sections list. Title + range on the left
/// (read-only label, not tappable — explicit affordances
/// only), then an Edit button (pencil) that opens the
/// editor, then a Delete button (trash) that raises a
/// confirmation. Two visible buttons makes the row's
/// affordances obvious; the prior "tap row to edit"
/// gesture was ambiguous against the trash target right
/// next to the same hit area.
private struct SectionsCardRow: View {
    let section: ConversationSection
    let utterances: [UtteranceEstimate]
    /// The focused utterance's id, already validated against
    /// `utterances` by the parent. nil disables the complete-
    /// with-focus button.
    let focusedID: UUID?
    /// Globally OK to kick off a summary for this row right
    /// now (summarizer enabled, ready, and no other run in
    /// flight). The row separately renders an in-progress
    /// indicator when `isSummarizing` is true so the user
    /// knows which section the active pass belongs to.
    let canSummarize: Bool
    /// True iff THIS section is the in-flight section-
    /// summary target. Drives the row's per-section spinner
    /// state (separate from the broader `canSummarize`
    /// gating so the spinner sits on the right row).
    let isSummarizing: Bool
    /// True iff the section already has a cached summary
    /// stamped on it. Distinguishes the "open existing
    /// summary" tap target from the "generate first time"
    /// one by switching the button glyph between a filled
    /// sparkles (cached) and an outline sparkles (empty);
    /// the action itself is the same in both cases — open
    /// the sheet, which auto-fires generation when empty.
    let hasCachedSummary: Bool
    /// Stamp the focused id onto this section's missing
    /// bound. Driven by the parent so the mutation routes
    /// through `store.update`; the row just signals intent.
    let onCompleteWithFocus: () -> Void
    /// Open the per-section summary sheet. Only attached to
    /// the button on COMPLETE rows (incomplete rows can't be
    /// summarized — no bounded utterance range).
    let onSummary: () -> Void
    let onEdit: () -> Void
    let onDelete: () -> Void

    @State private var showingDeleteConfirm: Bool = false

    var body: some View {
        HStack(spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(displayTitle)
                        .font(.callout)
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    if !section.isComplete {
                        // Distinctive glyph for the
                        // incomplete state so the user can
                        // tell at a glance which sections
                        // still need their other bound
                        // filled in.
                        Image(systemName: "exclamationmark.circle")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                            .accessibilityLabel(Text(
                                String(localized: "sections.incomplete.a11y")
                            ))
                    }
                }
                Text(rangeDescription)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)

            // Complete-with-focus button — only renders on
            // INCOMPLETE rows so complete sections aren't
            // crowded with a non-applicable affordance.
            // Disabled in three cases: (a) no utterance is
            // focused in the transcript pane, (b) the
            // focused id is stale (parent's `validFocusedID`
            // already filters this — arrives here as nil),
            // (c) applying the focus would violate start ≤
            // end ordering. The explicit `.opacity` dim
            // overrides any interaction between `.tint` +
            // `.disabled()` so the greyed state is visually
            // unambiguous regardless of how SwiftUI resolves
            // the foreground style on disabled controls.
            // Icon mirrors the global add-start / add-end
            // glyphs so the relationship is visible.
            if !section.isComplete {
                Button(action: onCompleteWithFocus) {
                    Image(systemName: completeWithFocusGlyph)
                        .font(.caption)
                        .foregroundStyle(.tint)
                        .opacity(canCompleteWithFocus ? 1.0 : 0.35)
                }
                .buttonStyle(.borderless)
                .disabled(!canCompleteWithFocus)
                .accessibilityLabel(Text(completeWithFocusA11y))
            }

            // Per-section summary button — only on COMPLETE
            // rows (incomplete sections have no bounded
            // range to summarize). Tap opens the section's
            // summary sheet; if the section has a cached
            // result it re-presents immediately, otherwise
            // the sheet auto-fires generation (matching the
            // overall-summary auto-fire policy). The glyph
            // switches between filled / outline sparkles so
            // a glance tells the user whether tapping will
            // open a cached read or start fresh inference.
            // Disabled when the summarizer isn't configured,
            // when another inference pass is already in
            // flight, or while THIS section is mid-run (the
            // spinner state). The `.opacity` dim matches the
            // complete-with-focus button's defensive pattern
            // so disabled state reads unambiguously regardless
            // of how `.tint` + `.disabled()` interpolate.
            if section.isComplete {
                Button(action: onSummary) {
                    if isSummarizing {
                        ProgressView()
                            .controlSize(.mini)
                    } else {
                        Image(systemName: summaryGlyph)
                            .font(.caption)
                            .foregroundStyle(.tint)
                            .opacity(canSummarize || hasCachedSummary ? 1.0 : 0.35)
                    }
                }
                .buttonStyle(.borderless)
                // Cached-summary rows always allow re-opening
                // the sheet even when the summarizer isn't
                // currently ready (the user can still SEE the
                // prior result, they just can't regenerate);
                // first-time rows need a ready summarizer to
                // be useful since the auto-fire is the entire
                // point of the tap.
                .disabled(isSummarizing || (!hasCachedSummary && !canSummarize))
                .accessibilityLabel(Text(summaryA11y))
            }

            Button(action: onEdit) {
                Image(systemName: "pencil")
                    .font(.caption)
                    .foregroundStyle(.tint)
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(Text(String(localized: "sections.edit")))

            Button {
                showingDeleteConfirm = true
            } label: {
                Image(systemName: "trash")
                    .font(.caption)
                    .foregroundStyle(.red.opacity(0.7))
            }
            .buttonStyle(.borderless)
            .accessibilityLabel(Text(String(localized: "sections.delete")))
            .confirmationDialog(
                String(format: String(localized: "sections.deleteConfirm.title"), displayTitle),
                isPresented: $showingDeleteConfirm,
                titleVisibility: .visible
            ) {
                Button(
                    String(localized: "sections.deleteConfirm.confirm"),
                    role: .destructive,
                    action: onDelete
                )
                Button(
                    String(localized: "sections.deleteConfirm.cancel"),
                    role: .cancel
                ) {}
            } message: {
                Text(String(localized: "sections.deleteConfirm.message"))
            }
        }
        .padding(.vertical, 4)
    }

    private var displayTitle: String {
        let trimmed = section.title.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty
            ? String(localized: "sections.untitled")
            : trimmed
    }

    /// Three rendering paths:
    /// - Both bounds set → "0:12 → 1:45   1:33 total"
    /// - Start only → "Starts at 0:12 — end not set"
    /// - End only → "Ends at 1:45 — start not set"
    /// (The neither-bound-set case can't occur — the editor
    /// rejects it — but `rangeMissing` is the safe fallback.)
    private var rangeDescription: String {
        let startTime = section.startUtteranceID.flatMap { id in
            utterances.first(where: { $0.id == id })?.start
        }
        let endTime = section.endUtteranceID.flatMap { id in
            utterances.first(where: { $0.id == id })?.end
        }
        switch (startTime, endTime) {
        case (let s?, let e?):
            return String(
                format: String(localized: "sections.rangeFormat"),
                formatTime(s),
                formatTime(e),
                formatTime(max(0, e - s))
            )
        case (let s?, nil):
            return String(
                format: String(localized: "sections.rangeStartOnly"),
                formatTime(s)
            )
        case (nil, let e?):
            return String(
                format: String(localized: "sections.rangeEndOnly"),
                formatTime(e)
            )
        case (nil, nil):
            return String(localized: "sections.rangeMissing")
        }
    }

    private func formatTime(_ t: Double) -> String {
        let minutes = Int(t) / 60
        let seconds = Int(t) % 60
        return String(format: "%d:%02d", minutes, seconds)
    }

    /// SF Symbol for the complete-with-focus button — matches
    /// the glyph the global Add-Start / Add-End buttons use
    /// so the row-level button reads as "the same action,
    /// applied to this section." Defaults to the add-start
    /// glyph for the degenerate neither-bound-set case
    /// (which the editor doesn't allow saving, but the row
    /// can encounter via a hand-mutated store).
    private var completeWithFocusGlyph: String {
        if section.startUtteranceID == nil { return "arrow.forward.to.line" }
        return "arrow.backward.to.line"
    }

    /// Accessibility label for the complete-with-focus
    /// button — describes which bound gets filled in.
    private var completeWithFocusA11y: String {
        if section.startUtteranceID == nil {
            return String(localized: "sections.row.setStart")
        }
        return String(localized: "sections.row.setEnd")
    }

    /// SF Symbol for the summary button. Matches the
    /// toolbar's overall-summary glyph (`text.book.closed`)
    /// so the affordance is recognizable as "the summary
    /// thing." Filled variant when a cached summary is on
    /// the section so the row reads as "this book has been
    /// written" vs. an empty outline for "tap to generate."
    private var summaryGlyph: String {
        hasCachedSummary ? "text.book.closed.fill" : "text.book.closed"
    }

    /// Accessibility label for the summary button —
    /// describes the action the tap will trigger so VoiceOver
    /// can distinguish open-cached from generate-new.
    private var summaryA11y: String {
        hasCachedSummary
            ? String(localized: "sections.row.openSummary")
            : String(localized: "sections.row.generateSummary")
    }

    /// True when there's a focused utterance AND assigning
    /// it to the section's missing bound would keep start ≤
    /// end. False on complete sections (no missing bound),
    /// when nothing is focused, when the focused id is stale,
    /// or when adding it would invert the range.
    private var canCompleteWithFocus: Bool {
        guard let focusedID else { return false }
        guard !section.isComplete else { return false }
        guard let focusedIdx = utterances.firstIndex(where: { $0.id == focusedID }) else {
            return false
        }
        if section.startUtteranceID == nil,
           let endID = section.endUtteranceID {
            guard let endIdx = utterances.firstIndex(where: { $0.id == endID }) else {
                return false
            }
            return focusedIdx <= endIdx
        }
        if section.endUtteranceID == nil,
           let startID = section.startUtteranceID {
            guard let startIdx = utterances.firstIndex(where: { $0.id == startID }) else {
                return false
            }
            return focusedIdx >= startIdx
        }
        return false
    }
}
