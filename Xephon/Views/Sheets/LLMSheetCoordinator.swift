import Foundation

/// Owns the presentation flags and in-flight `Task` handles for the
/// three on-device LLM-adjacent sheets in the main toolbar:
/// session summarization, transcription review, and search-and-
/// replace. Pulled out of `ContentView` so the toolbar buttons and
/// the sheets that observe them can be wired through a single
/// sibling instead of through four separate `@State` properties
/// plus two helper methods.
///
/// The two `Task` handles are kept `@ObservationIgnored` because
/// nothing in the view tree binds against them — they're retained
/// purely so the dismiss path (and the regenerate-while-running
/// path) can `.cancel()` the prior pass. The MLX `generate`
/// closures observe `Task.isCancelled` and return `.stop`; the
/// Apple Foundation Models `session.respond(...)` throws
/// `CancellationError`. Both stop draining power / GPU as soon as
/// cancellation lands.
@MainActor
@Observable
final class LLMSheetCoordinator {
    var showingSummary = false
    var showingReview = false
    var showingSearchReplace = false
    /// ID of the section whose per-section summary sheet is
    /// currently presented, or nil when no section sheet is
    /// open. Driven by `presentSectionSummary(id:)` / dismissed
    /// via `dismissSectionSummary()` — never set directly from
    /// the view layer so the inflight Task cancellation stays
    /// paired with the presentation flag.
    var presentingSectionSummaryID: UUID?

    @ObservationIgnored
    private var inflightSummarization: Task<Void, Never>?
    @ObservationIgnored
    private var inflightReview: Task<Void, Never>?
    /// In-flight per-section summarization Task. Same role as
    /// `inflightSummarization` for the overall summary — kept
    /// so the dismiss-while-running path can `.cancel()` and
    /// MLX stops spending tokens on a result the user has
    /// already walked away from. Only one section sheet can
    /// be presented at a time, so a single slot is enough.
    @ObservationIgnored
    private var inflightSectionSummarization: Task<Void, Never>?

    // MARK: - Summary

    /// Toolbar Summarize-button entry. Raises the sheet and
    /// auto-fires generation on first open if the summarizer is
    /// fully configured. Otherwise just opens the sheet so the
    /// user can reach the bottom controls and enable / pick a
    /// backend / download the model.
    func presentSummary(recorder: RecordingController) {
        showingSummary = true
        if recorder.lastSessionSummary == nil
            && recorder.summarizerEnabled
            && recorder.summarizerReady {
            startSummarization(recorder: recorder)
        }
    }

    /// Start (or re-start) the session summarization. Cancels any
    /// prior in-flight task first so re-tapping Regenerate while a
    /// pass is still running supersedes it cleanly.
    func startSummarization(recorder: RecordingController) {
        inflightSummarization?.cancel()
        inflightSummarization = Task {
            _ = await recorder.summarizeSession()
        }
    }

    /// Sheet-dismiss path. Cancel in-flight generation — no point
    /// spending tokens on a result the user has already walked
    /// away from.
    func dismissSummary() {
        inflightSummarization?.cancel()
        inflightSummarization = nil
        showingSummary = false
    }

    // MARK: - Section Summary

    /// Open the per-section summary sheet for `section`. Same
    /// auto-fire policy as the overall summarize entry: kick
    /// off generation immediately if the section has no cached
    /// summary AND the summarizer is fully configured;
    /// otherwise just raise the sheet so the user can see the
    /// cached result (or the empty / not-configured state) and
    /// decide whether to regenerate.
    func presentSectionSummary(
        section: ConversationSection,
        recorder: RecordingController
    ) {
        presentingSectionSummaryID = section.id
        if section.cachedSummary == nil
            && recorder.summarizerEnabled
            && recorder.summarizerReady {
            startSectionSummarization(sectionID: section.id, recorder: recorder)
        }
    }

    /// Start (or re-start) the per-section summarization.
    /// Cancels the prior in-flight section task first so re-
    /// tapping Regenerate while a pass is still running
    /// supersedes it cleanly.
    func startSectionSummarization(
        sectionID: UUID,
        recorder: RecordingController
    ) {
        inflightSectionSummarization?.cancel()
        inflightSectionSummarization = Task {
            _ = await recorder.summarizeSection(id: sectionID)
        }
    }

    /// Sheet-dismiss path. Cancel in-flight generation — same
    /// reasoning as `dismissSummary`.
    func dismissSectionSummary() {
        inflightSectionSummarization?.cancel()
        inflightSectionSummarization = nil
        presentingSectionSummaryID = nil
    }

    // MARK: - Review

    func presentReview(recorder: RecordingController) {
        showingReview = true
        // Same auto-fire policy as the summarize button: kick off
        // only when we have nothing cached AND the backend is
        // fully configured.
        if recorder.transcriptionIssues.isEmpty
            && recorder.summarizerEnabled
            && recorder.summarizerReady {
            startReview(recorder: recorder)
        }
    }

    /// Mirror of `startSummarization` for the transcription
    /// reviewer. Same cancel-prior-task discipline.
    func startReview(recorder: RecordingController) {
        inflightReview?.cancel()
        inflightReview = Task {
            _ = await recorder.reviewSession()
        }
    }

    func dismissReview() {
        inflightReview?.cancel()
        inflightReview = nil
        showingReview = false
    }

    // MARK: - Search & Replace

    func presentSearchReplace() {
        showingSearchReplace = true
    }

    func dismissSearchReplace() {
        showingSearchReplace = false
    }

    // MARK: - Backgrounding

    /// Called from the scenePhase observer when the app moves to
    /// `.background`. Backgrounding while MLX is mid-`generate`
    /// crashes the process — iOS revokes GPU access and the next
    /// Metal command buffer comes back as
    /// `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`,
    /// surfacing as an uncaught C++ exception that Swift can't
    /// catch. Cancellation propagates into MLX's `didGenerate`
    /// hook, which returns `.stop` and exits the loop before the
    /// next forward pass submits to Metal. We intentionally do
    /// NOT dismiss the sheets here: the user comes back to the
    /// empty / partial state with the Regenerate button live,
    /// which is the right resume behaviour.
    func cancelInflightTasks() {
        inflightSummarization?.cancel()
        inflightSummarization = nil
        inflightReview?.cancel()
        inflightReview = nil
        inflightSectionSummarization?.cancel()
        inflightSectionSummarization = nil
    }
}
