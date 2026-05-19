import SwiftUI

/// Attaches the three LLM-adjacent sheets driven by
/// `LLMSheetCoordinator`: session summary, transcription review,
/// and search-and-replace.
///
/// Pulled out of `ContentView.mainBody` for the same reason its
/// siblings were — chaining a fourth `.sheet` directly on the body
/// tipped the Swift type-checker past its "type-check in
/// reasonable time" budget, so each modifier gets a discrete
/// inference boundary.
struct LLMSheetBridge: ViewModifier {
    let recorder: RecordingController
    @Bindable var coord: LLMSheetCoordinator

    func body(content: Content) -> some View {
        content
            .sheet(isPresented: $coord.showingSummary) {
                SessionSummarySheet(
                    recorder: recorder,
                    summary: recorder.lastSessionSummary,
                    isGenerating: recorder.summarizerInferenceRunning,
                    onRegenerate: { coord.startSummarization(recorder: recorder) },
                    onDismiss: { coord.dismissSummary() }
                )
            }
            .sheet(isPresented: $coord.showingReview) {
                TranscriptionReviewSheet(
                    recorder: recorder,
                    issues: recorder.transcriptionIssues,
                    isReviewing: recorder.transcriptionReviewRunning,
                    onReview: { coord.startReview(recorder: recorder) },
                    onDismiss: { coord.dismissReview() }
                )
            }
            .sheet(isPresented: $coord.showingSearchReplace) {
                SearchReplaceSheet(
                    recorder: recorder,
                    onDismiss: { coord.dismissSearchReplace() }
                )
            }
    }
}
