import SwiftUI

/// Top-bar trailing toolbar: Summarize / Review / Search-Replace
/// / Export. Each button was previously a `@ViewBuilder` property
/// on ContentView; pulling them out as a `ToolbarContent` struct
/// makes them composable and isolates each one's `.disabled(...)`
/// gate from the others (the four chained `.disabled` clauses had
/// pushed the body's type-checker past its budget when they lived
/// inline).
struct MainToolbar: ToolbarContent {
    let recorder: RecordingController
    let llmCoord: LLMSheetCoordinator
    let fileCoord: SessionFileCoordinator

    var body: some ToolbarContent {
        ToolbarItem(placement: .topBarTrailing) { summarize }
        ToolbarItem(placement: .topBarTrailing) { review }
        ToolbarItem(placement: .topBarTrailing) { searchReplace }
        ToolbarItem(placement: .topBarTrailing) { export }
    }

    @ViewBuilder
    private var summarize: some View {
        Button {
            llmCoord.presentSummary(recorder: recorder)
        } label: {
            Label(
                String(localized: "summary.summarize"),
                systemImage: "text.book.closed"
            )
        }
        .disabled(!recorder.isIdleWithTranscript || recorder.summarizerInferenceRunning)
    }

    @ViewBuilder
    private var review: some View {
        Button {
            llmCoord.presentReview(recorder: recorder)
        } label: {
            Label(
                String(localized: "review.toolbar"),
                systemImage: "text.magnifyingglass"
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
