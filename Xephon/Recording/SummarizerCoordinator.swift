import Foundation
import os
import FoundationModels
import Fusion
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
    /// User's pick of `SummarizeMode`. `.trailing` truncates to
    /// a trailing window, `.heuristic` picks the top-N most
    /// distinctive utterances by TF-IDF, `.deep` runs map-reduce
    /// over every utterance. Persisted via
    /// `xephon.summarizerMode`. Defaults to `.trailing` for fresh
    /// installs and for the historical `deepMode = false` users
    /// (we don't migrate from the legacy boolean — opting back
    /// into deep is a one-tap action in the picker).
    private(set) var mode: SummarizeMode
    /// Apple's `SystemLanguageModel.default` availability snapshot.
    /// Refreshed at init and on backend change. Folded into `ready`.
    private(set) var appleFMAvailable: Bool = false
    /// True iff every file declared by the CURRENTLY-SELECTED
    /// MLX backend's manifest entry is present on disk. Apple FM
    /// reads as `true` (no install needed). Drives the picker's
    /// `ready` check and the SummarizerCard's "Downloading / Ready"
    /// status line.
    private(set) var modelInstalled: Bool = false
    /// Per-MLX-backend install flags, refreshed alongside
    /// `modelInstalled` whenever `syncInstallState` runs. Lets the
    /// ModelsCard show both rows (Qwen + Llama) with their own
    /// status independent of which one is the active backend.
    private(set) var qwenInstalled: Bool = false
    private(set) var llamaSwallowInstalled: Bool = false
    /// True while `ModelStore.ensureOptional` is in flight.
    private(set) var downloading: Bool = false
    /// Which backend's weights are being fetched right now (nil when
    /// idle). Lets the Models card paint the correct row as
    /// downloading even when it isn't the active backend.
    private(set) var downloadingBackend: SummarizerBackend?
    /// True while `summarize` is generating tokens. Disables the
    /// "Summarize session" toolbar button mid-run.
    private(set) var inferenceRunning: Bool = false
    /// Wall-clock instant the most recent summarization started;
    /// drives the live elapsed-time readout in `SessionSummarySheet`.
    private(set) var inferenceStart: Date?
    /// Last successful summary, cached so the result sheet survives
    /// re-presentation. Cleared on session start.
    private(set) var lastSessionSummary: SessionSummary?
    /// ID of the section currently being summarized, or nil
    /// when no per-section pass is in flight. Distinct from
    /// `inferenceRunning` (which also covers the overall
    /// session summary and the reviewer) so the Sections card
    /// can pick out exactly which row is mid-run and render
    /// the spinner on that row only. The wider
    /// `inferenceRunning` gate still applies — only one
    /// summary (overall OR section) can run at a time because
    /// they share the same MLX actor + GPU.
    private(set) var summarizingSectionID: UUID?

    /// True while `review` is in flight.
    private(set) var reviewRunning: Bool = false
    private(set) var reviewStart: Date?
    /// Last successful issue list. Issues are removed as the user
    /// edits or dismisses them. Cleared on session start.
    private(set) var issues: [TranscriptionIssue] = []

    /// Resident MLX summarizer (Qwen or Llama, depending on
    /// `backend`). Lazy-created on first `summarize` and dropped
    /// when the user disables the summarizer or starts a new
    /// session so the ~4 GB working set doesn't linger.
    /// `MLXLLMSummarizerActor` is the small protocol both
    /// `MLXQwenSummarizer` and `MLXLlamaSummarizer` conform to
    /// (lifecycle + `summarize`), letting one slot hold either.
    private var summarizerActor: (any MLXLLMSummarizerActor)?
    /// Resident MLX reviewer (Qwen or Llama, depending on
    /// `backend`). Separate `ModelContainer` from the
    /// summarizer — the coordinator ensures only one of the
    /// two is loaded at a time (both = ~9 GB resident, well
    /// over the per-app ceiling). `MLXLLMReviewerActor` is
    /// the small protocol both `MLXQwenTranscriptionReviewer`
    /// and `MLXLlamaTranscriptionReviewer` conform to so one
    /// slot holds either.
    private var reviewerActor: (any MLXLLMReviewerActor)?
    /// Snapshot of the FluidAudio diarizer's speaker DB captured
    /// right before the pipeline is released for summarization, so
    /// embedding-based matching survives the rebuild.
    private var savedSpeakerDB: Data?

    private static let enabledKey = "xephon.summarizerEnabled"
    private static let backendKey = "xephon.summarizerBackend"
    private static let modeKey    = "xephon.summarizerMode"

    init(parent: RecordingController) {
        self.parent = parent
        self.enabled = UserDefaults.standard.bool(forKey: Self.enabledKey)
        let rawBackend = UserDefaults.standard.string(forKey: Self.backendKey) ?? ""
        self.backend = SummarizerBackend(rawValue: rawBackend) ?? .appleFM
        let rawMode = UserDefaults.standard.string(forKey: Self.modeKey) ?? ""
        self.mode = SummarizeMode(rawValue: rawMode) ?? .trailing
        self.appleFMAvailable = SystemLanguageModel.default.isAvailable
    }

    /// Persist + apply a new mode preference. No side effects
    /// beyond the persist + state update — the next `summarize`
    /// call reads `mode` and dispatches to the matching path.
    func setMode(_ newMode: SummarizeMode) {
        guard mode != newMode else { return }
        mode = newMode
        UserDefaults.standard.set(newMode.rawValue, forKey: Self.modeKey)
    }

    /// True iff the chosen backend is ready to summarize. Apple FM
    /// is "ready" when the system model is available on this device;
    /// the MLX backends are "ready" when their on-disk install is
    /// complete.
    var ready: Bool {
        switch backend {
        case .appleFM:      return appleFMAvailable
        case .qwen:         return modelInstalled
        case .llamaSwallow: return modelInstalled
        case .lmStudio:
            // Settings + reachable URL gate the picker; we don't
            // verify connectivity here (would fire every body
            // re-render). The actual chat call surfaces an
            // error if the server is down.
            return parent.lmStudioSettings.enabled
                && parent.lmStudioSettings.baseURL != nil
        }
    }

    /// Model id of the MLX backend selected (Qwen or Llama). Nil
    /// for Apple FM (no install needed). Drives `ModelStore` calls
    /// — install / directory lookup / removal — so a single switch
    /// statement here centralizes the per-backend manifest mapping.
    private var mlxModelID: String? { modelID(for: backend) }

    /// Per-backend manifest mapping (active or not), so an explicit
    /// download can target a backend that isn't the current pick.
    private func modelID(for backend: SummarizerBackend) -> String? {
        switch backend {
        case .appleFM:      return nil
        case .qwen:         return ModelManifest.summarizerID
        case .llamaSwallow: return ModelManifest.summarizerLlamaID
        case .lmStudio:     return nil
        }
    }

    /// Flip the enabled flag. Persist + refresh backend-specific
    /// readiness. Turning an MLX backend on with weights missing
    /// kicks off the download; turning off unloads the resident
    /// MLX actor.
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
        await syncInstallState()
        if mlxModelID != nil, !modelInstalled, !downloading {
            await triggerDownload()
        }
    }

    /// Switch backend. Apple FM has no install step; MLX backends
    /// kick off the download when weights are missing AND drop the
    /// previously-resident MLX actor so the new family's weights
    /// can claim the memory.
    func setBackend(_ value: SummarizerBackend) async {
        guard backend != value else { return }
        backend = value
        UserDefaults.standard.set(value.rawValue, forKey: Self.backendKey)
        AppLog.app.info("summarizer backend → \(value.rawValue, privacy: .public)")
        // Switching backend always tears down the prior MLX actor.
        // Even Qwen → Llama (or vice versa) requires this because
        // the two would otherwise co-exist at ~9 GB resident, and
        // both reviewer + summarizer of the prior family would
        // also drift out of sync with the picker.
        await summarizerActor?.unload()
        summarizerActor = nil
        await reviewerActor?.unload()
        reviewerActor = nil
        // Re-sync install state against the new backend's model id
        // before deciding whether to download. `await` is load-
        // bearing: without it, `modelInstalled` still reflects the
        // PRIOR backend's state and the download trigger below
        // would skip when switching from Qwen (installed) → Llama
        // (not installed).
        await syncInstallState()
        if mlxModelID != nil, enabled, !modelInstalled, !downloading {
            await triggerDownload()
        }
        syncAppleFMAvailability()
    }

    func syncAppleFMAvailability() {
        appleFMAvailable = SystemLanguageModel.default.isAvailable
    }

    /// Recheck install state against the filesystem for BOTH MLX
    /// backends (so the ModelsCard's per-row badges stay accurate
    /// regardless of which one is the active picker selection)
    /// AND for the currently-selected backend (so the SummarizerCard
    /// status line + the `ready` check stay in sync). Apple FM has
    /// no on-disk install so its `modelInstalled` reads as `true`.
    /// Cheap — just an existence check per declared file.
    ///
    /// `async` so callers that immediately read `modelInstalled` /
    /// `qwenInstalled` / `llamaSwallowInstalled` see the refreshed
    /// values (the pre-async fire-and-forget version had setBackend
    /// reading stale state on every backend switch and skipping
    /// the auto-download trigger).
    func syncInstallState() async {
        guard let modelStore = parent.modelStore else {
            modelInstalled = mlxModelID == nil
            qwenInstalled = false
            llamaSwallowInstalled = false
            return
        }
        let qwen = await modelStore.isOptionalInstalled(
            id: ModelManifest.summarizerID
        )
        let llama = await modelStore.isOptionalInstalled(
            id: ModelManifest.summarizerLlamaID
        )
        qwenInstalled = qwen
        llamaSwallowInstalled = llama
        switch backend {
        case .appleFM:      modelInstalled = true
        case .qwen:         modelInstalled = qwen
        case .llamaSwallow: modelInstalled = llama
        case .lmStudio:     modelInstalled = true
        }
    }

    /// Auto-download the current backend's weights (fired on
    /// enable / backend switch when missing).
    private func triggerDownload() async {
        await downloadModel(for: backend)
    }

    /// Explicitly download a SPECIFIC MLX backend's weights —
    /// drives the Models card's per-row Download button, so a user
    /// can fetch Qwen or Llama-Swallow without first switching the
    /// active backend to it. Apple FM / LM Studio have no on-disk
    /// install and are no-ops. `downloadingBackend` names the model
    /// in flight so the card paints the right row as downloading
    /// (only one download runs at a time — guarded).
    func downloadModel(for target: SummarizerBackend) async {
        guard let id = modelID(for: target),
              let modelStore = parent.modelStore,
              downloadingBackend == nil else { return }
        downloadingBackend = target
        downloading = true
        defer {
            downloading = false
            downloadingBackend = nil
        }
        do {
            try await modelStore.ensureOptional(id: id)
            await syncInstallState()
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "Summarizer download failed (\(target.rawValue, privacy: .public)): \(String(describing: error), privacy: .public)"
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
        guard !reviewRunning else { return nil }
        guard !parent.utterances.isEmpty else { return nil }
        return await withInferenceGate {
            await runSummarize(
                utterances: parent.utterances,
                logLabelPrefix: "summarize",
                writeback: { summary in self.lastSessionSummary = summary }
            )
        }
    }

    /// Run the summarizer over a single user-defined section's
    /// utterance range. Returns the same `SessionSummary` shape as
    /// the overall summary — the LLM is told the conversation IS
    /// the slice, not a sub-clip of a larger session, so the
    /// "topic" / "overall mood" etc. read as a focused snapshot
    /// rather than "this section of the larger conversation."
    /// Caches the result on the section itself (via
    /// `SectionStore.setSummary(forSectionID:summary:)`) so the
    /// per-section sheet reopens with the same result, and the
    /// summary persists into the `.xph` bundle as part of the
    /// section's `Codable` payload.
    func summarizeSection(id: UUID) async -> SessionSummary? {
        guard !inferenceRunning else { return nil }
        guard !reviewRunning else { return nil }
        guard let section = parent.sections.section(id: id),
              section.isComplete,
              let startID = section.startUtteranceID,
              let endID = section.endUtteranceID,
              let startIdx = parent.utterances.firstIndex(where: { $0.id == startID }),
              let endIdx = parent.utterances.firstIndex(where: { $0.id == endID }),
              startIdx <= endIdx
        else { return nil }
        let slice = Array(parent.utterances[startIdx...endIdx])
        guard !slice.isEmpty else { return nil }
        return await withInferenceGate(sectionID: id) {
            await runSummarize(
                utterances: slice,
                logLabelPrefix: "summarize section",
                writeback: { summary in
                    self.parent.sections.setSummary(forSectionID: id, summary: summary)
                }
            )
        }
    }

    /// Set the inference gate eagerly (before any await inside
    /// `body` yields), run the body, and clear the gate on
    /// return. Bundles `inferenceRunning` + `inferenceStart` +
    /// `summarizingSectionID` so all three flip together; the
    /// summarizer card's "in flight" UI then can't observe one
    /// without the others. `sectionID` non-nil marks a per-
    /// section run so the Sections card knows which row owns
    /// the active pass. Closing the gate eagerly closes the
    /// race window where a second tap could pass the
    /// `!inferenceRunning` precondition during the pipeline-
    /// release yield.
    private func withInferenceGate(
        sectionID: UUID? = nil,
        body: () async -> SessionSummary?
    ) async -> SessionSummary? {
        inferenceGenerationToken &+= 1
        let myToken = inferenceGenerationToken
        inferenceRunning = true
        inferenceStart = Date()
        summarizingSectionID = sectionID
        defer {
            // Only clear flags if no later cancel/summarize call
            // has bumped the generation token past us. The
            // `userCancelledSummary()` path clears flags eagerly
            // (so buttons re-enable instantly on Done) and bumps
            // the token; if a fresh summarize started in the
            // meantime, our defer must not stomp its in-flight
            // state.
            if inferenceGenerationToken == myToken {
                inferenceRunning = false
                inferenceStart = nil
                summarizingSectionID = nil
            }
        }
        return await body()
    }

    /// Monotonically-incrementing token used by `withInferenceGate`
    /// and `userCancelledSummary` to ensure a stale Task's defer
    /// can't clobber state owned by a newer Task. Bumped on every
    /// gate entry and on every explicit user-cancellation.
    private var inferenceGenerationToken: UInt64 = 0
    /// Same role as `inferenceGenerationToken` for the reviewer
    /// path. Separate counter because summarize + review have
    /// independent gates.
    private var reviewGenerationToken: UInt64 = 0

    /// Sheet-dismiss entry point — clears `inferenceRunning`
    /// immediately so the toolbar buttons re-enable without
    /// having to wait for the underlying inference task's cancel
    /// + cleanup chain to propagate (LM Studio's URLSession
    /// cancellation, MLX's didGenerate `.stop`, Apple FM's
    /// CancellationError throw). The actual Task cancellation is
    /// driven by `LLMSheetCoordinator.dismissSummary`; this just
    /// makes the UI feel responsive.
    func userCancelledSummary() {
        inferenceGenerationToken &+= 1
        inferenceRunning = false
        inferenceStart = nil
        summarizingSectionID = nil
    }

    /// Reviewer counterpart to `userCancelledSummary`.
    func userCancelledReview() {
        reviewGenerationToken &+= 1
        reviewRunning = false
        reviewStart = nil
    }

    /// Shared dispatch entry for both overall-session and per-
    /// section summarization. Handles the pipeline release /
    /// memory log envelope, then routes to the per-backend
    /// runner. `writeback` is invoked synchronously on the
    /// MainActor before the runner returns the success result,
    /// so callers can cache the summary wherever they want
    /// (controller-level `lastSessionSummary`, per-section
    /// `setSummary`, etc.) without the runner needing to know
    /// the destination.
    private func runSummarize(
        utterances: [UtteranceEstimate],
        logLabelPrefix: String,
        writeback: @MainActor (SessionSummary) -> Void
    ) async -> SessionSummary? {
        // Both backends benefit from releasing the analysis pipeline
        // before invoking — even Apple FM, light on RAM in our
        // process, can trip Jetsam under device pressure (2-3 GB of
        // resident ONNX models + fat speaker DB before we allocate
        // anything for the summary). The pipeline lazy-rewarms in
        // the deferred cleanup.
        logAvailableMemory(label: "\(logLabelPrefix) start (before pipeline release)")
        await releasePipelineForSummarization()
        logAvailableMemory(label: "\(logLabelPrefix) start (after pipeline release)")
        switch backend {
        case .appleFM:
            return await summarizeWithAppleFM(
                utterances: utterances,
                logLabelPrefix: logLabelPrefix,
                writeback: writeback
            )
        case .qwen, .llamaSwallow:
            return await summarizeWithMLX(
                utterances: utterances,
                logLabelPrefix: logLabelPrefix,
                writeback: writeback
            )
        case .lmStudio:
            return await summarizeWithLMStudio(
                utterances: utterances,
                logLabelPrefix: logLabelPrefix,
                writeback: writeback
            )
        }
    }

    private func summarizeWithAppleFM(
        utterances: [UtteranceEstimate],
        logLabelPrefix: String,
        writeback: @MainActor (SessionSummary) -> Void
    ) async -> SessionSummary? {
        guard SystemLanguageModel.default.isAvailable else {
            parent.errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMSummarizer()
        // Inference gating (`inferenceRunning` / `inferenceStart`)
        // is owned by the public entry method (`summarize` /
        // `summarizeSection`) so a section pass and an overall
        // pass can't slip past each other's checks during the
        // pipeline-release yield. This method only schedules
        // the post-run pipeline rewarm.
        defer { scheduleUnloadAndPipelineRewarm() }
        logAvailableMemory(label: "\(logLabelPrefix) Apple FM (before respond)")
        let mode: SummarizeMode = self.mode
        let boostedIDs = (mode == .heuristic || mode == .meeting)
            ? Self.keywordBoostedIDs(keywords: parent.keywords.keywords, in: utterances)
            : Set<UUID>()
        do {
            let summary = try await backend.summarize(
                utterances: utterances,
                speakerNames: parent.speakerNameOverrides,
                mode: mode,
                boostedUtteranceIDs: boostedIDs
            )
            logAvailableMemory(label: "\(logLabelPrefix) Apple FM (after respond)")
            writeback(summary)
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

    private func summarizeWithMLX(
        utterances: [UtteranceEstimate],
        logLabelPrefix: String,
        writeback: @MainActor (SessionSummary) -> Void
    ) async -> SessionSummary? {
        guard let modelStore = parent.modelStore,
              let modelID = mlxModelID else {
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(id: modelID) else {
            parent.errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        // Pick the right per-family actor type. `backend` is
        // checked at the call site so the switch is exhaustive
        // over the MLX backends (Apple FM is routed through
        // `summarizeWithAppleFM` in `runSummarize`).
        let actor: any MLXLLMSummarizerActor
        if let existing = summarizerActor {
            actor = existing
        } else {
            switch backend {
            case .qwen:
                actor = MLXQwenSummarizer(
                    modelIdentifier: modelID,
                    modelDirectory: directory
                )
            case .llamaSwallow:
                actor = MLXLlamaSummarizer(
                    modelIdentifier: modelID,
                    modelDirectory: directory
                )
            case .appleFM, .lmStudio:
                // Unreachable — Apple FM is routed via
                // `summarizeWithAppleFM` and LM Studio via
                // `summarizeWithLMStudio` from `runSummarize`.
                scheduleUnloadAndPipelineRewarm()
                return nil
            }
            summarizerActor = actor
        }
        // Inference gating owned by the public entry method
        // (see comment in `summarizeWithAppleFM`).
        defer { scheduleUnloadAndPipelineRewarm() }
        let mode: SummarizeMode = self.mode
        let boostedIDs = (mode == .heuristic || mode == .meeting)
            ? Self.keywordBoostedIDs(keywords: parent.keywords.keywords, in: utterances)
            : Set<UUID>()
        do {
            let summary = try await actor.summarize(
                utterances: utterances,
                speakerNames: parent.speakerNameOverrides,
                mode: mode,
                boostedUtteranceIDs: boostedIDs
            )
            writeback(summary)
            return summary
        } catch is CancellationError {
            AppLog.app.info("summarizeWithMLX cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithMLX failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Dispatch to the LM Studio remote backend. Same shape as
    /// `summarizeWithAppleFM` — no on-disk model install + no
    /// MLX actor lifecycle to orchestrate, just build a client
    /// from the live settings, run, surface errors. Pipeline
    /// release still happens upstream in `runSummarize` so the
    /// ANE + SER actors come down before the network round-trip
    /// (negligible memory help here vs. MLX, but keeps every
    /// summarizer path observing the same lifecycle envelope —
    /// cheaper to keep the pattern uniform than to special-case
    /// the lighter backends).
    private func summarizeWithLMStudio(
        utterances: [UtteranceEstimate],
        logLabelPrefix: String,
        writeback: @MainActor (SessionSummary) -> Void
    ) async -> SessionSummary? {
        defer { scheduleUnloadAndPipelineRewarm() }
        guard parent.lmStudioSettings.enabled,
              let baseURL = parent.lmStudioSettings.baseURL else {
            parent.errorMessage = String(
                describing: LMStudioError.notConfigured(reason: "host or port missing")
            )
            return nil
        }
        let config = LMStudioClient.Configuration(
            baseURL: baseURL,
            modelID: parent.lmStudioSettings.modelID,
            requestTimeoutSeconds: parent.lmStudioSettings.requestTimeoutSeconds
        )
        let client = LMStudioClient(configuration: config)
        let backend = LMStudioSummarizer(
            modelIdentifier: parent.lmStudioSettings.modelID,
            client: client,
            useStructuredOutput: parent.lmStudioSettings.useStructuredOutput
        )
        logAvailableMemory(label: "\(logLabelPrefix) LM Studio (before request)")
        let mode: SummarizeMode = self.mode
        let boostedIDs = (mode == .heuristic || mode == .meeting)
            ? Self.keywordBoostedIDs(keywords: parent.keywords.keywords, in: utterances)
            : Set<UUID>()
        do {
            let summary = try await backend.summarize(
                utterances: utterances,
                speakerNames: parent.speakerNameOverrides,
                mode: mode,
                boostedUtteranceIDs: boostedIDs
            )
            logAvailableMemory(label: "\(logLabelPrefix) LM Studio (after request)")
            writeback(summary)
            return summary
        } catch is CancellationError {
            AppLog.app.info("summarizeWithLMStudio cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithLMStudio failed: \(String(describing: error), privacy: .public)"
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
        return await withReviewGate {
            logAvailableMemory(label: "review start (before pipeline release)")
            await releasePipelineForSummarization()
            logAvailableMemory(label: "review start (after pipeline release)")
            switch backend {
            case .appleFM:              return await reviewWithAppleFM()
            case .qwen, .llamaSwallow:  return await reviewWithMLX()
            case .lmStudio:             return await reviewWithLMStudio()
            }
        }
    }

    /// Reviewer-side counterpart of `withInferenceGate`. Sets
    /// `reviewRunning` + `reviewStart` eagerly so a second
    /// review tap (or, with the summarize check we ALSO want
    /// in place, a summarize tap) can't slip past the
    /// precondition during the pipeline-release yield, then
    /// clears them on return.
    private func withReviewGate(
        body: () async -> [TranscriptionIssue]?
    ) async -> [TranscriptionIssue]? {
        reviewGenerationToken &+= 1
        let myToken = reviewGenerationToken
        reviewRunning = true
        reviewStart = Date()
        defer {
            // See `withInferenceGate`'s defer comment — same
            // staleness check so `userCancelledReview()`'s eager
            // clear isn't undone by a later defer fire.
            if reviewGenerationToken == myToken {
                reviewRunning = false
                reviewStart = nil
            }
        }
        return await body()
    }

    private func reviewWithAppleFM() async -> [TranscriptionIssue]? {
        guard SystemLanguageModel.default.isAvailable else {
            parent.errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMTranscriptionReviewer()
        // Review gating (`reviewRunning` / `reviewStart`) is
        // owned by `review()` via `withReviewGate` so it can't
        // race with itself during the pipeline-release yield.
        // This method only schedules the post-run pipeline
        // rewarm.
        defer { scheduleUnloadAndPipelineRewarm() }
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

    private func reviewWithMLX() async -> [TranscriptionIssue]? {
        guard let modelStore = parent.modelStore,
              let modelID = mlxModelID else {
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(id: modelID) else {
            parent.errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleUnloadAndPipelineRewarm()
            return nil
        }
        // Belt-and-braces: drop the summarizer actor before the
        // reviewer comes up. Both share an MLX model's ~4.6 GB
        // weights (Qwen3 or Llama-3-Swallow); holding both =
        // ~9 GB resident and a guaranteed Jetsam.
        await summarizerActor?.unload()
        summarizerActor = nil

        // Pick the right per-family reviewer actor. Same
        // pattern as `summarizeWithMLX` — Apple FM is routed
        // via `reviewWithAppleFM` so the switch is exhaustive
        // over the MLX backends.
        let actor: any MLXLLMReviewerActor
        if let existing = reviewerActor {
            actor = existing
        } else {
            switch backend {
            case .qwen:
                actor = MLXQwenTranscriptionReviewer(
                    modelIdentifier: modelID,
                    modelDirectory: directory
                )
            case .llamaSwallow:
                actor = MLXLlamaTranscriptionReviewer(
                    modelIdentifier: modelID,
                    modelDirectory: directory
                )
            case .appleFM, .lmStudio:
                scheduleUnloadAndPipelineRewarm()
                return nil
            }
            reviewerActor = actor
        }
        // Review gating owned by `review()` via `withReviewGate`
        // (see comment in `reviewWithAppleFM`).
        defer { scheduleUnloadAndPipelineRewarm() }
        do {
            let issues = try await actor.review(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides,
                language: reviewLanguage()
            )
            self.issues = issues
            return issues
        } catch is CancellationError {
            AppLog.app.info("reviewWithMLX cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithMLX failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Dispatch the reviewer to LM Studio. Same shape as
    /// `reviewWithAppleFM` — no MLX teardown, no on-disk model
    /// install, just build a client + run. Pipeline release
    /// still happens upstream in `review()` for the same
    /// "uniform envelope" reasoning as `summarizeWithLMStudio`.
    private func reviewWithLMStudio() async -> [TranscriptionIssue]? {
        defer { scheduleUnloadAndPipelineRewarm() }
        guard parent.lmStudioSettings.enabled,
              let baseURL = parent.lmStudioSettings.baseURL else {
            parent.errorMessage = String(
                describing: LMStudioError.notConfigured(reason: "host or port missing")
            )
            return nil
        }
        let config = LMStudioClient.Configuration(
            baseURL: baseURL,
            modelID: parent.lmStudioSettings.modelID,
            requestTimeoutSeconds: parent.lmStudioSettings.requestTimeoutSeconds
        )
        let client = LMStudioClient(configuration: config)
        let backend = LMStudioTranscriptionReviewer(
            modelIdentifier: parent.lmStudioSettings.modelID,
            client: client,
            useStructuredOutput: parent.lmStudioSettings.useStructuredOutput
        )
        logAvailableMemory(label: "review LM Studio (before request)")
        do {
            let issues = try await backend.review(
                utterances: parent.utterances,
                speakerNames: parent.speakerNameOverrides,
                language: reviewLanguage()
            )
            logAvailableMemory(label: "review LM Studio (after request)")
            self.issues = issues
            return issues
        } catch is CancellationError {
            AppLog.app.info("reviewWithLMStudio cancelled by user")
            return nil
        } catch {
            parent.errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithLMStudio failed: \(String(describing: error), privacy: .public)"
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

    /// Remove the on-disk model for the currently-selected backend.
    /// Toggle state is preserved so the user's preference survives.
    /// No-op for Apple FM (no on-disk install).
    func removeModel() async {
        await summarizerActor?.unload()
        summarizerActor = nil
        await reviewerActor?.unload()
        reviewerActor = nil
        guard let id = mlxModelID else { return }
        do {
            try await parent.modelStore?.removeOptional(id: id)
        } catch {
            AppLog.app.warning(
                "removeOptional failed: \(String(describing: error), privacy: .public)"
            )
        }
        await syncInstallState()
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

    /// Build the set of utterance IDs whose normalized
    /// transcript contains at least one normalized user-
    /// keyword. Drives the heuristic summarizer's
    /// keyword-boost so user-curated terms reliably surface in
    /// the prompt window. Returns empty when the keyword list
    /// is empty (most users won't have one). Uses the same
    /// `JapaneseSearchNormalizer` the transcript-filter +
    /// keyword-occurrence counters use, so cross-script
    /// matching (kanji ↔ kana ↔ romaji) behaves consistently
    /// with what the user sees in the transcript pane.
    ///
    /// Pure-data overload: callers pass in the utterance slice
    /// they care about so the helper works for both the whole
    /// session (overall summary) and a single section's range
    /// (per-section summary).
    static func keywordBoostedIDs(
        keywords: [Keyword],
        in utterances: [UtteranceEstimate]
    ) -> Set<UUID> {
        guard !keywords.isEmpty else { return [] }
        let normalizedKeywords: [String] = keywords.compactMap {
            let n = JapaneseSearchNormalizer.normalize($0.text)
            return n.isEmpty ? nil : n
        }
        guard !normalizedKeywords.isEmpty else { return [] }
        var hits: Set<UUID> = []
        for u in utterances {
            let normalizedText = JapaneseSearchNormalizer.normalize(u.transcript)
            if normalizedKeywords.contains(where: { normalizedText.contains($0) }) {
                hits.insert(u.id)
            }
        }
        AppLog.app.info(
            "keyword boost: \(hits.count, privacy: .public) of \(utterances.count, privacy: .public) utterances match \(normalizedKeywords.count, privacy: .public) keyword(s)"
        )
        return hits
    }
}
