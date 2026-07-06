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
                    // Exclude section passes: they share the
                    // inferenceRunning gate, and without the
                    // sectionID check this sheet showed a
                    // "generating" spinner (and a Cancel button
                    // that detached the wrong run) while a SECTION
                    // summary was what was actually running.
                    isGenerating: recorder.summarizerInferenceRunning
                        && recorder.summarizingSectionID == nil,
                    onRegenerate: { coord.startSummarization(recorder: recorder) },
                    onCancel: { coord.cancelSummarization(recorder: recorder) },
                    onDismiss: { coord.closeSummary() }
                )
            }
            .sheet(isPresented: $coord.showingReview) {
                TranscriptionReviewSheet(
                    recorder: recorder,
                    issues: recorder.transcriptionIssues,
                    isReviewing: recorder.transcriptionReviewRunning,
                    onReview: { coord.startReview(recorder: recorder) },
                    onCancel: { coord.cancelReview(recorder: recorder) },
                    onDismiss: { coord.closeReview() }
                )
            }
            .sheet(isPresented: $coord.showingSearchReplace) {
                SearchReplaceSheet(
                    recorder: recorder,
                    onDismiss: { coord.dismissSearchReplace() }
                )
            }
            // Per-section summary sheet. Bound via `.sheet(item:)`
            // through a synthetic Identifiable wrapper around the
            // resolved section so the sheet auto-dismisses if the
            // section is deleted out from under it (e.g. the user
            // taps trash on the row mid-generation). Dismissal here
            // (swipe-down, or the section vanishing) only CLOSES —
            // a running pass finishes in the background; if its
            // section was deleted meanwhile, the writeback lands on
            // a missing id and no-ops. Explicit cancellation is the
            // sheet's Cancel button.
            .sheet(
                item: Binding<SectionSummaryPresentation?>(
                    get: {
                        guard let id = coord.presentingSectionSummaryID,
                              let section = recorder.sections.section(id: id)
                        else { return nil }
                        return SectionSummaryPresentation(section: section)
                    },
                    set: { newValue in
                        if newValue == nil {
                            coord.closeSectionSummary()
                        }
                    }
                )
            ) { presentation in
                SectionSummarySheet(
                    recorder: recorder,
                    section: presentation.section,
                    summary: presentation.section.cachedSummary,
                    isGenerating: recorder.summarizingSectionID == presentation.section.id
                        && recorder.summarizerInferenceRunning,
                    onRegenerate: {
                        coord.startSectionSummarization(
                            sectionID: presentation.section.id,
                            recorder: recorder
                        )
                    },
                    onCancel: {
                        coord.cancelSectionSummarization(recorder: recorder)
                    },
                    onDismiss: { coord.closeSectionSummary() }
                )
            }
    }
}

/// Identifiable wrapper around the section being summarized so
/// `.sheet(item:)` can drive presentation off `coord
/// .presentingSectionSummaryID`. The id is forwarded from the
/// underlying section so SwiftUI treats the same section as the
/// same presentation across re-evaluations (a cached-summary
/// stamp on the section, for example, just refreshes the sheet
/// body rather than tearing it down and re-presenting).
private struct SectionSummaryPresentation: Identifiable {
    let section: ConversationSection
    var id: UUID { section.id }
}
