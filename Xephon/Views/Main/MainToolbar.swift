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

    var body: some ToolbarContent {
        // Skip the principal TextField when running as
        // "Designed for iPad" on Apple Silicon Mac. The macOS
        // host always renders its own title bar above the iPad
        // chrome (Mac Catalyst is disabled — see project.yml —
        // so we can't suppress the macOS bar from iPad code).
        // The session title surfaces through
        // `.navigationTitle(Binding<String>)` in ContentView,
        // which maps to the macOS window title and lets the
        // user rename via the proxy-icon menu. On iPad the
        // principal item is the only title surface.
        if !ProcessInfo.processInfo.isiOSAppOnMac {
            ToolbarItem(placement: .principal) { sessionTitleField }
        }
        ToolbarItem(placement: .topBarTrailing) { summarize }
        ToolbarItem(placement: .topBarTrailing) { review }
        ToolbarItem(placement: .topBarTrailing) { searchReplace }
        ToolbarItem(placement: .topBarTrailing) { export }
    }

    /// Editable session-title field that replaces the static
    /// "Xephon" nav title. Bound to `recorder.sessionTitle`;
    /// empty value renders the placeholder. Centered, headline
    /// font, plain field style so it visually matches the nav
    /// title look. `.frame(maxWidth: 320)` caps the editor width
    /// so it doesn't grow past the available principal-slot space
    /// on landscape iPad — without the cap the field stretches and
    /// crowds out the trailing toolbar buttons.
    @ViewBuilder
    private var sessionTitleField: some View {
        TextField("Xephon", text: $recorder.sessionTitle)
            .font(.headline)
            .multilineTextAlignment(.center)
            .textFieldStyle(.plain)
            .frame(maxWidth: 320)
            .submitLabel(.done)
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
                systemImage: "magnifyingglass"
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
