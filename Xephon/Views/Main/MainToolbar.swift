import SwiftUI

/// Top-bar trailing toolbar: Summarize / Review / Search-Replace
/// / Export. Each button was previously a `@ViewBuilder` property
/// on ContentView; pulling them out as a `ToolbarContent` struct
/// makes them composable and isolates each one's `.disabled(...)`
/// gate from the others (the four chained `.disabled` clauses had
/// pushed the body's type-checker past its budget when they lived
/// inline).
struct MainToolbar: ToolbarContent {
    @Bindable var recorder: RecordingController
    let llmCoord: LLMSheetCoordinator
    let fileCoord: SessionFileCoordinator
    let filePicker: FilePickerCoordinator

    var body: some ToolbarContent {
        ToolbarItem(placement: .topBarLeading) { openSession }
        ToolbarItem(placement: .topBarLeading) { saveSession }
        ToolbarItem(placement: .principal) {
            SessionTitleField(recorder: recorder)
        }
        ToolbarItem(placement: .topBarTrailing) { summarize }
        ToolbarItem(placement: .topBarTrailing) { review }
        ToolbarItem(placement: .topBarTrailing) { TimelineStripMenu() }
        ToolbarItem(placement: .topBarTrailing) { searchReplace }
        ToolbarItem(placement: .topBarTrailing) { export }
    }

    // Open / Save Session — same actions as the File menu's
    // Open Session (⇧⌘O) / Save Session (⇧⌘S) items, surfaced on the
    // chrome's leading edge. Glyphs and the save gate
    // (idleWithTranscript, == MenuCommands.canSaveSession) match the
    // menu so the two entry points stay in sync.
    @ViewBuilder
    private var openSession: some View {
        Button {
            fileCoord.presentSessionPicker(recorder: recorder, filePicker: filePicker)
        } label: {
            Label(String(localized: "menu.importSession"), systemImage: "folder")
        }
    }

    @ViewBuilder
    private var saveSession: some View {
        Button {
            Task { await fileCoord.saveSession(recorder: recorder, filePicker: filePicker) }
        } label: {
            Label(String(localized: "menu.saveSession"), systemImage: "square.and.arrow.down")
        }
        .disabled(!recorder.isIdleWithTranscript)
    }

    @ViewBuilder
    private var summarize: some View {
        // Mirrors the per-section summary button's two-stage
        // visual: outline `text.book.closed` when no cached
        // session summary yet (tap to generate), filled
        // `text.book.closed.fill` when one exists (tap to open
        // the cached result). The action is identical in both
        // states — `presentSummary` opens the sheet and auto-
        // fires generation only when there's nothing cached
        // and the summarizer is ready — but the glyph + a11y
        // label tell the user which path the tap will take.
        let hasCachedSummary = recorder.lastSessionSummary != nil
        Button {
            llmCoord.presentSummary(recorder: recorder)
        } label: {
            Label(
                String(localized: hasCachedSummary
                    ? "summary.openSummary"
                    : "summary.summarize"),
                systemImage: hasCachedSummary
                    ? "text.book.closed.fill"
                    : "text.book.closed"
            )
        }
        .disabled(!recorder.isIdleWithTranscript || recorder.summarizerInferenceRunning)
    }

    @ViewBuilder
    private var review: some View {
        // Two-stage visual mirroring the Summarize button:
        // outline `exclamationmark.bubble` when the issue list
        // is empty (tap to run review), filled
        // `exclamationmark.bubble.fill` when issues exist (tap
        // to open the cached list). The exclamation-bubble pair
        // reads as "alerts about the text" — outline = no alert
        // pending, fill = alerts exist — which matches the
        // semantics of the review flow more directly than a
        // generic magnifying glass.
        //
        // Reset is wired through the underlying data: a new
        // session calls `resetSessionState` → `clearIssues`,
        // which flips this back to the outline state.
        let hasIssues = !recorder.transcriptionIssues.isEmpty
        Button {
            llmCoord.presentReview(recorder: recorder)
        } label: {
            Label(
                String(localized: hasIssues
                    ? "review.openReview"
                    : "review.toolbar"),
                systemImage: hasIssues
                    ? "exclamationmark.bubble.fill"
                    : "exclamationmark.bubble"
            )
        }
        .disabled(
            !recorder.isIdleWithTranscript
                || recorder.transcriptionReviewRunning
                || recorder.summarizerInferenceRunning
        )
    }

    @ViewBuilder
    private var searchReplace: some View {
        Button {
            llmCoord.presentSearchReplace()
        } label: {
            Label(
                String(localized: "searchReplace.toolbar"),
                systemImage: "magnifyingglass.circle"
            )
        }
        // Disable during recording / analysis: commitHandEdit
        // requires the controller to be idle, and surfacing the
        // sheet earlier would let the user queue work that
        // silently fails.
        .disabled(!recorder.isIdleWithTranscript)
    }

    @ViewBuilder
    private var export: some View {
        Button {
            Task { await fileCoord.exportJSONFromToolbar(recorder: recorder) }
        } label: {
            Label(
                String(localized: "export.json"),
                systemImage: "square.and.arrow.up"
            )
        }
        .disabled(!recorder.isIdleWithTranscript)
    }
}

/// Timeline-strip visibility menu — checkmark toggles for the three
/// strips stacked above the transcript list (speaker/diarization,
/// emotion, fusion contribution). State lives in UserDefaults via
/// `@AppStorage` (keys in `TimelineStripPrefs`); TranscriptPaneView
/// reads the same keys to gate each strip, so a toggle here
/// shows/hides the strip immediately with no plumbing through
/// ContentView.
///
/// A plain `View` struct rather than `@AppStorage` directly on the
/// `ToolbarContent` — dynamic properties on plain Views are
/// unconditionally reliable, same reasoning as `SessionTitleField`
/// below. Always enabled: strip visibility is a view preference,
/// meaningful while recording as much as while idle. A strip whose
/// toggle is ON still hides itself when it has no data yet (the
/// pre-existing per-strip data gates).
private struct TimelineStripMenu: View {
    @AppStorage(TimelineStripPrefs.showDiarizationKey)
    private var showDiarization = true
    @AppStorage(TimelineStripPrefs.showEmotionKey)
    private var showEmotion = true
    @AppStorage(TimelineStripPrefs.showFusionKey)
    private var showFusion = true

    var body: some View {
        Menu {
            Toggle(isOn: $showDiarization) {
                Label(
                    String(localized: "timeline.toggle.speakers"),
                    systemImage: "person.2"
                )
            }
            Toggle(isOn: $showEmotion) {
                Label(
                    String(localized: "timeline.toggle.emotion"),
                    systemImage: "face.smiling"
                )
            }
            Toggle(isOn: $showFusion) {
                Label(
                    String(localized: "timeline.toggle.fusion"),
                    systemImage: "chart.bar"
                )
            }
        } label: {
            Label(
                String(localized: "timeline.menu.toolbar"),
                systemImage: "list.and.film"
            )
        }
    }
}

/// Editable session-title field. Locally-buffered: the TextField
/// binds to a `@State` shadow, not to `recorder.sessionTitle`
/// directly. This avoids two hazards:
///
/// 1. Per-keystroke writes into the `@Observable` controller
///    invalidate every view observing `sessionTitle` on every
///    character — visible on long sessions as keystroke lag.
/// 2. A direct binding pushes one undo step per character. Instead
///    we capture `previousCommitted` when editing begins and push a
///    single `.sessionTitle` undo step on `.onSubmit` / end-editing
///    if the committed value differs from where we started.
///
/// Per-character undo inside the field continues to be handled by
/// UIKit's built-in text-edit undo manager (the standard system
/// behavior is independent of our app-level stack).
private struct SessionTitleField: View {
    @Bindable var recorder: RecordingController
    @State private var draft: String = ""
    @State private var previousCommitted: String = ""
    // Control-driven editing flag, set by the TextField's
    // `onEditingChanged` (UIKit begin/end-editing events). NOT
    // @FocusState — see tombstone (3) below.
    @State private var isEditing = false

    var body: some View {
        // A bare TextField clips/scrolls a too-long title (truncating
        // the END). To middle-truncate the displayed title we overlay a
        // Text with `.truncationMode(.middle)` while unfocused; the
        // TextField stays in the tree so a NATIVE tap moves first
        // responder into it — the path that always worked. While
        // editing, the real TextField shows (no truncation — the user
        // needs to see what they type).
        //
        // The TextField sits ON TOP at FULL OPACITY always; while
        // unfocused only its TEXT COLOR is `.clear`, which hides the
        // glyphs without touching hit-testing. Its native placeholder
        // keeps its own color regardless of `foregroundStyle`, so the
        // empty state needs no Text underlay at all.
        //
        // Tombstones — three failed shapes, don't re-walk any of them:
        // (1) Text on top + `.onTapGesture { focused = true }`:
        //     programmatic focus is rejected while the target field
        //     renders at alpha 0 (an invisible UIKit responder refuses
        //     first-responder status).
        // (2) TextField on top with `.opacity(focused ? 1 : 0)`:
        //     alpha < 0.01 removes a UIKit-backed view from hit
        //     testing entirely, so the "transparent but hit-testable"
        //     premise was false — the tap never reached the field and
        //     `focused` could never flip true. Hiding via clear TEXT
        //     COLOR (not view opacity) is what keeps the field
        //     tappable.
        // (3) Clear text color driven by @FocusState: @FocusState
        //     never OBSERVES focus inside a toolbar *principal* item
        //     on iPadOS 26 (setting was already known-broken; reading
        //     is too). UIKit editing went live while `focused` stayed
        //     false, so the view never left display mode — keystrokes
        //     were invisible and the user watched the middle-truncated
        //     underlay update instead ("editing the truncated title").
        //     Editing state must come from the CONTROL's own
        //     begin/end-editing events (`onEditingChanged`), which are
        //     hosting-agnostic.
        ZStack {
            Text(draft)
                .lineLimit(1)
                .truncationMode(.middle)
                .frame(maxWidth: .infinity)
                .opacity(isEditing ? 0 : 1)
                .allowsHitTesting(false)

            TextField(
                String(localized: "chrome.sessionTitle.placeholder"),
                text: $draft,
                onEditingChanged: { editing in
                    isEditing = editing
                    if editing {
                        previousCommitted = recorder.sessionTitle
                    } else {
                        commitIfChanged()
                    }
                }
            )
            .multilineTextAlignment(.center)
            .textFieldStyle(.plain)
            .foregroundStyle(isEditing ? Color.primary : Color.clear)
        }
        .font(.headline)
        .frame(maxWidth: 320)
        .submitLabel(.done)
        .onAppear {
            draft = recorder.sessionTitle
            previousCommitted = recorder.sessionTitle
        }
        .onChange(of: recorder.sessionTitle) { _, newValue in
            // External updates (session load, undo restore) must
            // flow into the shadow while the field is not being
            // edited. Skip mid-edit so we don't stomp the user's
            // typing.
            if !isEditing {
                draft = newValue
                previousCommitted = newValue
            }
        }
        // Return key: commit immediately. The end-editing event that
        // follows resignation calls `commitIfChanged` again — the
        // trimmed-vs-previousCommitted guard makes the second call a
        // no-op.
        .onSubmit {
            commitIfChanged()
        }
    }

    private func commitIfChanged() {
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        // Normalize the visible draft even on a no-op commit, so
        // "  same title  " doesn't linger untrimmed in the field.
        draft = trimmed
        guard trimmed != previousCommitted else { return }
        recorder.registerUndoStep(
            .sessionTitle(previous: previousCommitted),
            actionName: String(localized: "undo.sessionTitle")
        )
        recorder.sessionTitle = trimmed
        previousCommitted = trimmed
    }
}
