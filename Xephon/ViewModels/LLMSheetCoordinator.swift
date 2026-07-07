import Foundation
import XephonLogging

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
        // !inferenceRunning: the toolbar button is tappable
        // mid-run (spinner state) so the user can watch progress —
        // auto-firing here would cancel-and-restart the very pass
        // they came to see.
        if recorder.lastSessionSummary == nil
            && recorder.summarizerEnabled
            && recorder.summarizerReady
            && !recorder.summarizerInferenceRunning {
            startSummarization(recorder: recorder)
        }
    }

    /// Start (or re-start) the session summarization. Cancels any
    /// prior in-flight task first so re-tapping Regenerate while a
    /// pass is still running supersedes it cleanly.
    func startSummarization(recorder: RecordingController) {
        // Any pending auto-run is superseded by this start (whether
        // the user tapped or the auto path itself requested it — in
        // the latter case the timers have already fired and the
        // cancel is a no-op).
        recorder.summarizer.cancelAutoRuns(reason: "summarization starting")
        inflightSummarization?.cancel()
        // LAST TAP WINS: a still-unwinding cancelled run (this
        // slot's, or a section pass in the sibling slot) holds the
        // shared `inferenceRunning` gate for however long its MLX
        // cancellation takes to propagate — without the eager
        // gate-clear the fresh run's `guard !inferenceRunning`
        // fails and the tap silently does nothing (worst case:
        // old run cancelled AND new run never starts). Same eager
        // clear the auto-supersede path uses; the generation token
        // keeps the dying run's defer from stomping the new gate.
        if recorder.summarizerInferenceRunning {
            inflightSectionSummarization?.cancel()
            inflightSectionSummarization = nil
            recorder.summarizer.userCancelledSummary()
        }
        inflightSummarization = Task {
            _ = await recorder.summarizeSession()
        }
    }

    /// Sheet-close path (Done button / programmatic). Deliberately
    /// does NOT cancel a running generation: closing the sheet is
    /// "let it finish in the background" — the toolbar button keeps
    /// its spinner and the result lands via the normal writeback.
    /// This also matches what swipe-down dismissal always did (the
    /// presentation binding just flips; no cancel ever ran there).
    /// Explicit cancellation is the sheet's Cancel button →
    /// `cancelSummarization`.
    func closeSummary() {
        showingSummary = false
    }

    /// Explicit-cancel path (the sheet's Cancel button). Cancels
    /// the in-flight Task and flips the coordinator's running flags
    /// immediately — otherwise the toolbar Summary / Review buttons
    /// would stay in their spinner state until the underlying
    /// cancel chain (URLSession cancellation for LM Studio, MLX
    /// `didGenerate` returning `.stop`, Apple FM's
    /// CancellationError throw) fully propagates and the
    /// summarizer's `withInferenceGate` defer fires. The sheet
    /// stays open showing the regenerate bar.
    func cancelSummarization(recorder: RecordingController) {
        inflightSummarization?.cancel()
        inflightSummarization = nil
        recorder.summarizer.userCancelledSummary()
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
        // LAST TAP WINS — see startSummarization. Also covers a
        // running OVERALL pass being superseded by a section tap.
        if recorder.summarizerInferenceRunning {
            inflightSummarization?.cancel()
            inflightSummarization = nil
            recorder.summarizer.userCancelledSummary()
        }
        inflightSectionSummarization = Task {
            _ = await recorder.summarizeSection(id: sectionID)
        }
    }

    /// Close-vs-cancel split — same reasoning as `closeSummary` /
    /// `cancelSummarization`: Done (and swipe-down) only closes;
    /// a running section pass finishes in the background and its
    /// result lands on the section via the normal writeback.
    func closeSectionSummary() {
        presentingSectionSummaryID = nil
    }

    func cancelSectionSummarization(recorder: RecordingController) {
        inflightSectionSummarization?.cancel()
        inflightSectionSummarization = nil
        recorder.summarizer.userCancelledSummary()
    }

    // MARK: - Review

    func presentReview(recorder: RecordingController) {
        showingReview = true
        // Same auto-fire policy as the summarize button: kick off
        // only when we have nothing cached AND the backend is
        // fully configured. !reviewRunning mirrors presentSummary —
        // opening the sheet mid-run must not restart the pass.
        if recorder.transcriptionIssues.isEmpty
            && recorder.summarizerEnabled
            && recorder.summarizerReady
            && !recorder.transcriptionReviewRunning {
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

    /// Close-vs-cancel split — same reasoning as `closeSummary` /
    /// `cancelSummarization`.
    func closeReview() {
        showingReview = false
    }

    func cancelReview(recorder: RecordingController) {
        inflightReview?.cancel()
        inflightReview = nil
        recorder.summarizer.userCancelledReview()
    }

    // MARK: - Search & Replace

    func presentSearchReplace() {
        showingSearchReplace = true
    }

    func dismissSearchReplace() {
        showingSearchReplace = false
    }

    // MARK: - Backgrounding

    /// What the backgrounding cancel killed, so foreground return
    /// can re-fire it. Cleared on consumption AND by the plain
    /// `cancelInflightTasks` (the recording-start supersede path) —
    /// a run superseded by a new recording must stay dead.
    @ObservationIgnored private var refireSummaryOnForeground = false
    @ObservationIgnored private var refireReviewOnForeground = false

    /// scenePhase → `.background` entry. Backgrounding while MLX is
    /// mid-`generate` crashes the process — iOS revokes GPU access
    /// and the next Metal command buffer aborts as
    /// `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`,
    /// an uncaught C++ exception Swift can't catch. Cancellation
    /// propagates into the cancellable prefill / `didGenerate`
    /// checks before the next forward pass submits. Records what
    /// was in flight so `refireAfterForeground` can restart it.
    /// Section passes are deliberately NOT recorded — they're
    /// short, and their sheet's Regenerate covers the rare loss.
    func cancelForBackground() {
        let hadSummary = inflightSummarization != nil
        let hadReview = inflightReview != nil
        cancelInflightTasks()
        refireSummaryOnForeground = hadSummary
        refireReviewOnForeground = hadReview
    }

    /// scenePhase → `.active` entry: restart whatever the
    /// backgrounding cancel killed, through the same start paths
    /// the toolbar uses. Re-checks the world at fire time — the
    /// summarizer may have been disabled, or the session replaced,
    /// while backgrounded. If an auto-summarize deferral fires on
    /// the same return (rare — it only exists for runs that hadn't
    /// STARTED at background time), its supersede path wins; one
    /// redundant cancel/start, no double run.
    func refireAfterForeground(recorder: RecordingController) {
        let summary = refireSummaryOnForeground
        let review = refireReviewOnForeground
        refireSummaryOnForeground = false
        refireReviewOnForeground = false
        guard summary || review else { return }
        guard recorder.summarizerEnabled, recorder.summarizerReady,
              !recorder.utterances.isEmpty,
              !recorder.isRecording, !recorder.isAnalyzing else {
            AppLog.app.info("foreground re-fire skipped: conditions no longer hold")
            return
        }
        // The two can't have been running concurrently (shared
        // inference gate), so at most one branch fires.
        if summary {
            AppLog.app.info("re-firing summarization cancelled by backgrounding")
            startSummarization(recorder: recorder)
        } else if review {
            AppLog.app.info("re-firing transcription review cancelled by backgrounding")
            startReview(recorder: recorder)
        }
    }

    /// Cancel everything without recording a re-fire — the
    /// recording-start supersede hook. We intentionally do NOT
    /// dismiss the sheets here: the user comes back to the empty /
    /// partial state with the Regenerate button live.
    func cancelInflightTasks() {
        refireSummaryOnForeground = false
        refireReviewOnForeground = false
        inflightSummarization?.cancel()
        inflightSummarization = nil
        inflightReview?.cancel()
        inflightReview = nil
        inflightSectionSummarization?.cancel()
        inflightSectionSummarization = nil
    }
}
