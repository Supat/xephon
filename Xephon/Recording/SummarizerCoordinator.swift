import Foundation
import os
import FoundationModels
import Summarizer
import XephonLogging

/// Owns every piece of the on-device summarization + transcription-review
/// flow that used to live as ~430 lines on `RecordingController`. The
/// controller now holds one `let summarizer: SummarizerCoordinator` and
/// keeps thin API-preserving forwarders so existing view-layer reads
/// (`recorder.summarizerEnabled`, `recorder.lastSessionSummary`, …)
/// don't change. Cross-cutting reads (utterances, speakerNameOverrides,
/// pipeline lifecycle, ModelStore) come through an `unowned` ref to
/// the parent — the coordinator never outlives the controller, and the
/// shared `@MainActor` isolation makes the access pattern safe.
@MainActor
@Observable
final class SummarizerCoordinator {
    private unowned let parent: RecordingController

    private(set) var enabled: Bool
    private(set) var backend: SummarizerBackend
    /// Apple's `SystemLanguageModel.default` availability snapshot.
    /// Refreshed at init and on backend change. Folded into `ready`.
    private(set) var appleFMAvailable: Bool = false
    /// True iff every file declared by the summarizer's optional
    /// manifest entry is present on disk.
    private(set) var modelInstalled: Bool = false
    /// True while `ModelStore.ensureOptional` is in flight.
    private(set) var downloading: Bool = false
    /// True while `summarize` is generating tokens. Disables the
    /// "Summarize session" toolbar button mid-run.
    private(set) var inferenceRunning: Bool = false
    /// Wall-clock instant the most recent summarization started;
    /// drives the live elapsed-time readout in `SessionSummarySheet`.
    private(set) var inferenceStart: Date?
    /// Last successful summary, cached so the result sheet survives
    /// re-presentation. Cleared on session start.
    private(set) var lastSessionSummary: SessionSummary?

    /// True while `review` is in flight.
    private(set) var reviewRunning: Bool = false
    private(set) var reviewStart: Date?
    /// Last successful issue list. Issues are removed as the user
    /// edits or dismisses them. Cleared on session start.
    private(set) var issues: [TranscriptionIssue] = []

    /// Resident Qwen weights. Lazy-created on first `summarize` and
    /// dropped when the user disables the summarizer or starts a new
    /// session, so the ~4 GB working set doesn't linger.
    private var summarizerActor: MLXQwenSummarizer?
    /// Qwen reviewer's actor — separate `ModelContainer`. The
    /// coordinator ensures only one of the two is loaded at a time
    /// (both = ~9 GB resident, well over the per-app ceiling).
    private var reviewerActor: MLXQwenTranscriptionReviewer?
    /// Snapshot of the FluidAudio diarizer's speaker DB captured
    /// right before the pipeline is released for summarization, so
    /// embedding-based matching survives the rebuild.
    private var savedSpeakerDB: Data?

    private static let enabledKey = "xephon.summarizerEnabled"
    private static let backendKey = "xephon.summarizerBackend"

    init(parent: RecordingController) {
        self.parent = parent
        self.enabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
        let rawBackend = UserDefaults.standard.string(forKey: Self.backendKey) ?? ""
        self.backend = SummarizerBackend(rawValue: rawBackend) ?? .appleFM
        self.appleFMAvailable = SystemLanguageModel.default.isAvailable
    }

    /// True iff the chosen backend is ready to summarize. Apple FM
    /// is "ready" when the system model is available on this device;
    /// Qwen is "ready" when its 4.3 GB on-disk install is complete.
    var ready: Bool {
        switch backend {
        case .appleFM: return appleFMAvailable
        case .qwen:    return modelInstalled
        }
    }

    /// Flip the enabled flag. Persist + refresh backend-specific
    /// readiness. Turning Qwen on with weights missing kicks off the
    /// download; turning off unloads the resident Qwen actor.
    func setEnabled(_ value: Bool) async {
        guard enabled != value else { return }
        enabled = value
        UserDefaults.standard.set(value, forKey: Self.enabledKey)
        AppLog.app.info("summarizer enabled → \(value, privacy: .public)")
        if !value {
            await summarizerActor?.unload()
            summarizerActor = nil
            return
        }
        syncAppleFMAvailability()
        syncInstallState()
        if backend == .qwen, !modelInstalled, !downloading {
            await triggerDownload()
        }
    }

    /// Switch backend. Apple FM has no install step; Qwen kicks off
    /// the download when weights are missing.
    func setBackend(_ value: SummarizerBackend) async {
        guard backend != value else { return }
        backend = value
        UserDefaults.standard.set(value.rawValue, forKey: Self.backendKey)
        AppLog.app.info("summarizer backend → \(value.rawValue, privacy: .public)")
        if value == .qwen, enabled, !modelInstalled, !downloading {
            await triggerDownload()
        }
        if value == .appleFM {
            // Reclaim Qwen's RAM if it was loaded.
            await summarizerActor?.unload()
            summarizerActor = nil
        }
        syncAppleFMAvailability()
    }

    func syncAppleFMAvailability() {
        appleFMAvailable = SystemLanguageModel.default.isAvailable
    }

    /// Recheck `modelInstalled` against the filesystem. Cheap — just
    /// an existence check per declared file.
    func syncInstallState() {
        guard let modelStore = parent.modelStore else {
            modelInstalled = false
            return
        }
        Task {
            let installed = await modelStore.isOptionalInstalled(
                id: ModelManifest.summarizerID
            )
            await MainActor.run {
                self.modelInstalled = installed
            }
        }
    }

    /// Drive the on-demand download via `ModelStore.ensureOptional`.
    /// Wraps the call in `downloading` so the Settings card can
    /// render an inline progress indicator.
    private func triggerDownload() async {
        guard let modelStore = parent.modelStore else { return }
        downloading = true
        defer { downloading = false }
        do {
            try await modelStore.ensureOptional(id: ModelManifest.summarizerID)
            syncInstallState()
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "Summarizer download failed: \(String(describing: error), privacy: .public)"
            )
        }
    }

    /// Run the summarizer over the current session. Nil when the
    /// model isn't installed, the session is empty, or inference is
    /// already in flight. Memory orchestration: the 4.3 GB Qwen
    /// weights plus the AnalysisPipeline's already-loaded inference
    /// actors (~1.5 GB) trip iOS Jetsam on a 16 GB iPad once Qwen
    /// prefill kicks off — we release the pipeline first, then
    /// unload Qwen and re-warm the pipeline in the background after
    /// the user has their result.
    func summarize() async -> SessionSummary? {
        guard !inferenceRunning else { return nil }
        guard !parent.utterances.isEmpty else { return nil }
        // Both backends benefit from releasing the analysis pipeline
        // before invoking — even Apple FM, light on RAM in our
        // process, can trip Jetsam under device pressure (2-3 GB of
        // resident ONNX models + fat speaker DB before we allocate
        // anything for the summary). The pipeline lazy-rewarms in
        // the deferred cleanup.
        logAvailableMemory(label: "summarize start (before pipeline release)")
        await releasePipelineForSummarization()
        logAvailableMemory(label: "summarize start (after pipeline release)")
        switch backend {
        case .appleFM: return await summarizeWithAppleFM()
        case .qwen:    return await summarizeWithQwen()
        }
    }

    private func summarizeWithAppleFM() async -> SessionSummary? {
        guard SystemLanguageModel.default.isAvailable else {
            parent.errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMSummarizer()
        inferenceRunning = true
        inferenceStart = Date()
        defer {
            inferenceRunning = false
            inferenceStart = nil
            scheduleUnloadAndPipelineRewarm()
        }
        logAvailableMemory(label: "summarize Apple FM (before respond)")
        do {
            let summary = try await backend.summarize(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides
            )
            logAvailableMemory(label: "summarize Apple FM (after respond)")
            lastSessionSummary = summary
            return summary
        } catch is CancellationError {
            AppLog.app.info("summarizeWithAppleFM cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithAppleFM failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    private func summarizeWithQwen() async -> SessionSummary? {
        guard let modelStore = parent.modelStore else {
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(
            id: ModelManifest.summarizerID
        ) else {
            parent.errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        let actor: MLXQwenSummarizer
        if let existing = summarizerActor {
            actor = existing
        } else {
            actor = MLXQwenSummarizer(
                modelIdentifier: ModelManifest.summarizerID,
                modelDirectory: directory
            )
            summarizerActor = actor
        }
        inferenceRunning = true
        inferenceStart = Date()
        defer {
            inferenceRunning = false
            inferenceStart = nil
            scheduleUnloadAndPipelineRewarm()
        }
        do {
            let summary = try await actor.summarize(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides
            )
            lastSessionSummary = summary
            return summary
        } catch is CancellationError {
            AppLog.app.info("summarizeWithQwen cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithQwen failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Walk the current utterance list through the on-device LLM and
    /// collect transcription issues. Same orchestration as
    /// `summarize`: release pipeline, run, unload + rewarm.
    func review() async -> [TranscriptionIssue]? {
        guard !reviewRunning else { return nil }
        guard !inferenceRunning else { return nil }
        guard !parent.utterances.isEmpty else { return nil }
        logAvailableMemory(label: "review start (before pipeline release)")
        await releasePipelineForSummarization()
        logAvailableMemory(label: "review start (after pipeline release)")
        switch backend {
        case .appleFM: return await reviewWithAppleFM()
        case .qwen:    return await reviewWithQwen()
        }
    }

    private func reviewWithAppleFM() async -> [TranscriptionIssue]? {
        guard SystemLanguageModel.default.isAvailable else {
            parent.errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMTranscriptionReviewer()
        reviewRunning = true
        reviewStart = Date()
        defer {
            reviewRunning = false
            reviewStart = nil
            scheduleUnloadAndPipelineRewarm()
        }
        logAvailableMemory(label: "review Apple FM (before respond)")
        do {
            let issues = try await backend.review(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides,
                language: reviewLanguage()
            )
            logAvailableMemory(label: "review Apple FM (after respond)")
            self.issues = issues
            return issues
        } catch is CancellationError {
            AppLog.app.info("reviewWithAppleFM cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithAppleFM failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    private func reviewWithQwen() async -> [TranscriptionIssue]? {
        guard let modelStore = parent.modelStore else {
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(
            id: ModelManifest.summarizerID
        ) else {
            parent.errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        // Belt-and-braces: drop the summarizer actor before the
        // reviewer comes up. Both share Qwen3-8B's 4.6 GB weights;
        // holding both = ~9 GB resident and a guaranteed Jetsam.
        await summarizerActor?.unload()
        summarizerActor = nil

        let actor: MLXQwenTranscriptionReviewer
        if let existing = reviewerActor {
            actor = existing
        } else {
            actor = MLXQwenTranscriptionReviewer(
                modelIdentifier: ModelManifest.summarizerID,
                modelDirectory: directory
            )
            reviewerActor = actor
        }
        reviewRunning = true
        reviewStart = Date()
        defer {
            reviewRunning = false
            reviewStart = nil
            scheduleUnloadAndPipelineRewarm()
        }
        do {
            let issues = try await actor.review(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides,
                language: reviewLanguage()
            )
            self.issues = issues
            return issues
        } catch is CancellationError {
            AppLog.app.info("reviewWithQwen cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithQwen failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Map the controller's `SessionLanguage` onto the Summarizer
    /// module's `ReviewLanguage`. Lives here (not on
    /// `SessionLanguage`) so the latter type doesn't grow a
    /// dependency on Summarizer just for one bridge.
    private func reviewLanguage() -> ReviewLanguage {
        switch parent.sessionLanguage {
        case .japanese: return .japanese
        case .english:  return .english
        }
    }

    /// Dismiss an issue without acting on it.
    func dismissIssue(id: UUID) {
        issues.removeAll { $0.id == id }
    }

    /// Clear cached issues. Called by `RecordingController.start()`
    /// when the user begins a new session.
    func clearIssues() {
        issues = []
    }

    /// Clear the cached summary. Called on session start.
    func clearLastSummary() {
        lastSessionSummary = nil
    }

    /// Restore a previously-saved summary from a `.xph` bundle so
    /// the result sheet re-presents the persisted summary instead
    /// of forcing the user to regenerate.
    func restore(summary: SessionSummary?) {
        lastSessionSummary = summary
    }

    /// Restore a previously-saved issue list from a `.xph` bundle.
    func restore(issues: [TranscriptionIssue]) {
        self.issues = issues
    }

    /// Remove the on-disk model. Toggle state is preserved so the
    /// user's preference survives.
    func removeModel() async {
        await summarizerActor?.unload()
        summarizerActor = nil
        await reviewerActor?.unload()
        reviewerActor = nil
        do {
            try await parent.modelStore?.removeOptional(id: ModelManifest.summarizerID)
        } catch {
            AppLog.app.warning(
                "removeOptional failed: \(String(describing: error), privacy: .public)"
            )
        }
        syncInstallState()
    }

    /// Drop strong refs to the analysis pipeline so ARC can reclaim
    /// the ~1.5 GB of resident ONNX session memory before Qwen
    /// claims its 4.3 GB. Yields cooperatively after nilling so the
    /// runtime gets a tick to release before MLX starts allocating
    /// prefill memory. Snapshots the FluidAudio speaker DB first so
    /// embedding-based matching survives the rebuild.
    private func releasePipelineForSummarization() async {
        AppLog.app.info(
            "releasing analysis pipeline before summarization (free ~1.5 GB)"
        )
        if let pipeline = parent.pipeline {
            savedSpeakerDB = await pipeline.exportSpeakerDatabase()
            if let blob = savedSpeakerDB {
                AppLog.app.info(
                    "snapshotted speaker DB before summarize (\(blob.count, privacy: .public) bytes)"
                )
            }
        }
        parent.pipelineTask?.cancel()
        parent.pipelineTask = nil
        parent.pipeline = nil
        await Task.yield()
    }

    /// Unload Qwen and re-warm the pipeline in the background.
    /// Runs in `defer` so it fires whether summarization succeeded
    /// or failed. Restores the pre-summarize speaker DB snapshot
    /// after the fresh diarizer is warm.
    private func scheduleUnloadAndPipelineRewarm() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            // Unload BOTH Qwen actors. Only one is loaded at a time
            // by construction, but a stale reference here would keep
            // ~4.6 GB of weights resident across the pipeline rewarm.
            await self.summarizerActor?.unload()
            self.summarizerActor = nil
            await self.reviewerActor?.unload()
            self.reviewerActor = nil
            AppLog.app.info("Qwen unloaded; re-warming analysis pipeline")
            let pipeline = await self.parent.ensurePipeline()
            if let saved = self.savedSpeakerDB {
                do {
                    try await pipeline.importSpeakerDatabase(saved)
                    AppLog.app.info("restored speaker DB after pipeline re-warm")
                } catch {
                    AppLog.app.warning(
                        "speaker DB restore after summarize failed: \(String(describing: error), privacy: .public)"
                    )
                }
                self.savedSpeakerDB = nil
            }
        }
    }

    /// Log how much memory the process can still allocate before
    /// iOS Jetsam will start culling. `os_proc_available_memory()`
    /// is the canonical sentinel; surfaced around the summarize call
    /// so we can see exactly how much headroom we have at each stage.
    private func logAvailableMemory(label: String) {
        let bytes = os_proc_available_memory()
        let mb = bytes / (1024 * 1024)
        AppLog.app.info(
            "memory available [\(label, privacy: .public)]: \(mb, privacy: .public) MB"
        )
    }
}
