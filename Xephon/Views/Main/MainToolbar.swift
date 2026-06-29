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

/// Editable session-title field. Locally-buffered: the TextField
/// binds to a `@State` shadow, not to `recorder.sessionTitle`
/// directly. This avoids two hazards:
///
/// 1. Per-keystroke writes into the `@Observable` controller
///    invalidate every view observing `sessionTitle` on every
///    character — visible on long sessions as keystroke lag.
/// 2. A direct binding pushes one undo step per character. Instead
///    we capture `previousCommitted` on focus gain and push a single
///    `.sessionTitle` undo step on `.onSubmit` / focus loss if the
///    committed value differs from where we started.
///
/// Per-character undo inside the field continues to be handled by
/// UIKit's built-in text-edit undo manager (the standard system
/// behavior is independent of our app-level stack).
private struct SessionTitleField: View {
    @Bindable var recorder: RecordingController
    @State private var draft: String = ""
    @State private var previousCommitted: String = ""
    @FocusState private var focused: Bool

    var body: some View {
        TextField(
            String(localized: "chrome.sessionTitle.placeholder"),
            text: $draft
        )
        .font(.headline)
        .multilineTextAlignment(.center)
        .textFieldStyle(.plain)
        .frame(maxWidth: 320)
        .submitLabel(.done)
        .focused($focused)
        .onAppear {
            draft = recorder.sessionTitle
            previousCommitted = recorder.sessionTitle
        }
        .onChange(of: recorder.sessionTitle) { _, newValue in
            // External updates (session load, undo restore) must
            // flow into the shadow while the field is unfocused.
            // Skip when focused so we don't stomp the user's typing.
            if !focused {
                draft = newValue
                previousCommitted = newValue
            }
        }
        .onChange(of: focused) { _, isFocused in
            if isFocused {
                previousCommitted = recorder.sessionTitle
            } else {
                commitIfChanged()
            }
        }
        .onSubmit {
            commitIfChanged()
            focused = false
        }
    }

    private func commitIfChanged() {
        let trimmed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard trimmed != previousCommitted else { return }
        recorder.registerUndoStep(
            .sessionTitle(previous: previousCommitted),
            actionName: String(localized: "undo.sessionTitle")
        )
        recorder.sessionTitle = trimmed
        previousCommitted = trimmed
        draft = trimmed
    }
}
