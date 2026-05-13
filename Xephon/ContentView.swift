import SwiftUI
import UniformTypeIdentifiers
import Audio
import Export
import Fusion
import SERText
import Summarizer
import XephonLogging

struct ContentView: View {
    @Environment(MenuCommands.self) private var menuCommands
    /// Backgrounding the app while an MLX summarize/review is in
    /// flight crashes the process: iOS revokes GPU access for
    /// background apps and MLX's next Metal command buffer comes
    /// back as `kIOGPUCommandBufferCallbackErrorBackgroundExecution-
    /// NotPermitted`, surfacing as an uncaught C++ exception that
    /// Swift can't catch. We watch `scenePhase` and cancel both
    /// in-flight tasks on `.background` so the generate loop's
    /// `Task.isCancelled` check bails before submitting the next
    /// forward pass. (The diarizer doesn't share this hazard —
    /// `FluidAudioDiarizer.loadModels` pins to `.cpuAndNeuralEngine`
    /// so it never touches Metal.)
    @Environment(\.scenePhase) private var scenePhase
    @State private var recorder = RecordingController()
    @State private var shareURL: URL?
    @State private var showingDiscardConfirm: Bool = false
    @State private var showingFilePicker: Bool = false
    /// What the next `.fileImporter` presentation should accept and
    /// what its result-handler should do with the picked URL. Set
    /// before flipping `showingFilePicker` to true. Two SwiftUI
    /// `.fileImporter` modifiers attached to the same view chain
    /// quietly collide on iPadOS 26 — the menu fires, the binding
    /// flips, the document picker initializes, but nothing presents.
    /// One fileImporter switching its content type and handler by
    /// mode is the reliable shape.
    @State private var filePickerMode: FilePickerMode = .audio
    enum FilePickerMode { case audio, session }
    /// True while the session-save panel is up. Driven by the
    /// File → Save Session… menu command.
    @State private var showingSaveSession: Bool = false
    /// Snapshot bundled when the user invokes Save Session. Captured
    /// at command time (synchronously) so the file-exporter sheet
    /// writes a stable copy even if the user keeps interacting with
    /// the app while it's open. Nil = no save in progress.
    @State private var pendingSaveDocument: SessionFileDocument?
    /// Last error from a save/load attempt; surfaces as an alert.
    @State private var sessionIOError: String?
    @State private var pendingFileURL: URL?
    /// True when we successfully called
    /// `startAccessingSecurityScopedResource()` on `pendingFileURL`
    /// inside the picker callback. The picker's implicit grant can
    /// expire over the multi-dialog hop to `startFromFile`, so we
    /// pin scope as soon as the URL arrives and release it once the
    /// recorder has taken its own ref (or the user has cancelled).
    @State private var pendingFileScopeAcquired: Bool = false
    @State private var showingFileDiscardConfirm: Bool = false
    @State private var showingPacingDialog: Bool = false
    /// Currently-visible utterance row IDs. Maintained via per-row
    /// `.onAppear`/`.onDisappear` so we can tell whether the most recent
    /// utterance is in frame and decide between auto-scroll vs. surfacing
    /// the "New utterance" capsule.
    @State private var visibleUtteranceIDs: Set<UUID> = []
    /// True when a new utterance has arrived while the user has scrolled
    /// the most recent off-screen. Cleared when the most recent comes
    /// back into view (either by user scroll or capsule tap).
    @State private var hasUnreadUtterance: Bool = false
    /// Currently selected utterance for hardware-keyboard navigation.
    /// SwiftUI's `List(selection:)` natively responds to ↑/↓ arrow keys
    /// once the user has tapped into the list (or focus has otherwise
    /// landed on it). Bound nil = no selection.
    @State private var selectedUtteranceID: UUID?
    /// Free-text filter applied to each utterance's `text`. Empty
    /// string disables filtering.
    @State private var searchText: String = ""
    /// Label filter. Nil = "All labels"; non-nil only shows utterances
    /// whose fused top label matches.
    @State private var selectedLabelFilter: String?
    /// Speaker filter. Nil = all speakers; non-nil only shows
    /// utterances stamped with the matching `speakerID`. Toggled by
    /// tapping the chip row directly under the filter bar.
    @State private var selectedSpeakerFilter: String?
    /// Keyboard focus for the search field. Driven by ⌘F (sets it
    /// true) and Esc (sets it false). `@FocusState` is the only
    /// mechanism that programmatically moves focus into a TextField.
    @FocusState private var searchFieldFocused: Bool
    /// IDs of utterances whose detail panel is expanded. Toggled by
    /// long-press on the row or by pressing Space while the row is
    /// the selected list item. Kept here (not on the row) so a
    /// rebuild of the row doesn't drop the expansion state.
    @State private var expandedUtteranceIDs: Set<UUID> = []
    /// Normalized (Hepburn romaji, lowercased) form of each utterance's
    /// transcript, keyed by utterance ID. Populated off-MainActor in
    /// `refreshSearchCache()` so the filter loop is a dictionary
    /// lookup per row instead of an N-per-keystroke CFStringTokenizer
    /// pass. Survives utterance additions because we only normalize
    /// the new arrivals on each utterance-count change.
    @State private var normalizedTranscriptCache: [UUID: String] = [:]
    /// Background task that's currently rebuilding `normalizedTranscriptCache`.
    /// Cancelled and replaced on every utterance-count change so a
    /// fast-arriving stream of utterances doesn't spawn unbounded work.
    @State private var searchCacheTask: Task<Void, Never>?
    /// Stored speaker id whose Rename alert is currently presented
    /// (or `nil` when no alert is up). Set by the row's
    /// context-menu Rename action, cleared on alert dismiss.
    @State private var editingSpeakerStored: String?
    /// Pending text in the Rename TextField, bound to the alert.
    @State private var editingSpeakerName: String = ""
    /// Snapshot of the utterance whose Edit Utterance sheet is
    /// currently presented. Non-nil drives the `.sheet(item:)`
    /// presentation; cleared on dismiss / commit / cancel.
    /// We hold the full struct (not just the id) so the sheet's
    /// initial state populates from a stable snapshot even if
    /// the underlying row gets mutated by an in-flight re-eval.
    @State private var editingUtterance: UtteranceEstimate?
    /// Set when a tap on the diarizer timeline strip should scroll
    /// the transcript list to a specific row, regardless of whether
    /// that row is already selected or already on screen. The
    /// `TranscriptList` consumes this and clears it back to nil
    /// after one scroll, so a subsequent tap on the same time
    /// re-fires.
    @State private var scrollRequestUtteranceID: UUID?
    /// Drives the `SessionSummarySheet` presentation. Set true from
    /// the toolbar "Summarize" button just before kicking off the
    /// async inference call; the sheet observes
    /// `recorder.summarizerInferenceRunning` and
    /// `recorder.lastSessionSummary` to switch between progress and
    /// result UI without needing a separate "result is ready" flag.
    @State private var showingSummaryView: Bool = false
    /// Drives the `TranscriptionReviewSheet` presentation. Same
    /// pattern as `showingSummaryView` — sheet observes the
    /// controller's `transcriptionReviewRunning` + `transcriptionIssues`
    /// to render its three states.
    @State private var showingReviewView: Bool = false
    /// Drives the `SearchReplaceSheet` presentation. Unlike review,
    /// nothing is LLM-backed here — the sheet is just a filter +
    /// substitute pass over the utterance list, so no
    /// `isRunning`-flavored gating is needed.
    @State private var showingSearchReplaceView: Bool = false
    /// Active in-flight LLM tasks, owned by the view so the dismiss
    /// path (and the regenerate-while-running path) can `.cancel()`
    /// them. The MLX inference closures observe `Task.isCancelled`
    /// and return `.stop`; the Apple FM session call throws
    /// `CancellationError` from `respond(...)`. Both stop draining
    /// power / GPU as soon as the user closes the sheet.
    @State private var inflightSummarizationTask: Task<Void, Never>?
    @State private var inflightReviewTask: Task<Void, Never>?
    /// Memoization layer for `filteredIndexedUtterances` and
    /// `displayedSummary`. SwiftUI re-evaluates the view body on
    /// every observed-state change (playback state, pipeline metrics,
    /// scroll position…), and naïvely those computed properties
    /// re-ran the full O(N) filter + summary fold per render. Stored
    /// as a `final class` so we can mutate it from inside the getters
    /// without triggering a SwiftUI re-render — `@State` retains it
    /// across renders but doesn't observe its internal properties.
    @State private var filterMemo = FilterMemo()

    var body: some View {
        if !recorder.modelsReady {
            SetupView(controller: recorder)
        } else {
            mainBody
        }
    }

    private var mainBody: some View {
        NavigationStack {
            VStack(spacing: 0) {
                if !recorder.pipelineDiagnostics.isEmpty {
                    PipelineDiagnosticsBanner(messages: recorder.pipelineDiagnostics)
                }
                GeometryReader { geo in
                    HStack(spacing: 0) {
                        controlPane
                            .frame(width: geo.size.width / 3)
                        Divider()
                        transcriptPane
                            .frame(maxWidth: .infinity)
                    }
                }
            }
            .navigationTitle("Xephon")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    summarizeToolbarButton
                }
                ToolbarItem(placement: .topBarTrailing) {
                    reviewToolbarButton
                }
                ToolbarItem(placement: .topBarTrailing) {
                    searchReplaceToolbarButton
                }
                ToolbarItem(placement: .topBarTrailing) {
                    exportToolbarButton
                }
            }
            .sheet(item: $shareURL) { url in
                ShareSheet(items: [url])
            }
            .sheet(isPresented: $showingSummaryView) {
                SessionSummarySheet(
                    recorder: recorder,
                    summary: recorder.lastSessionSummary,
                    isGenerating: recorder.summarizerInferenceRunning,
                    onRegenerate: { startSummarizationTask() },
                    onDismiss: {
                        // Cancel in-flight generation when the user
                        // dismisses — no point spending tokens on a
                        // result they've already walked away from.
                        inflightSummarizationTask?.cancel()
                        inflightSummarizationTask = nil
                        showingSummaryView = false
                    }
                )
            }
            .sheet(isPresented: $showingReviewView) {
                TranscriptionReviewSheet(
                    recorder: recorder,
                    issues: recorder.transcriptionIssues,
                    isReviewing: recorder.transcriptionReviewRunning,
                    onReview: { startReviewTask() },
                    onDismiss: {
                        inflightReviewTask?.cancel()
                        inflightReviewTask = nil
                        showingReviewView = false
                    }
                )
            }
            .modifier(SearchReplaceSheetPresenter(
                isPresented: $showingSearchReplaceView,
                recorder: recorder
            ))
            // Cancel any in-flight MLX summarize/review the moment
            // the app backgrounds — see the `scenePhase` env
            // declaration for why this isn't optional. Cancellation
            // propagates into MLXLMCommon.generate's didGenerate
            // hook, which returns `.stop` and exits the loop before
            // the next forward pass submits to Metal. We don't
            // dismiss the sheets here: the user comes back to the
            // empty / partial state with the Regenerate button live,
            // which is the right resume behaviour.
            .onChange(of: scenePhase) { _, newPhase in
                guard newPhase == .background else { return }
                inflightSummarizationTask?.cancel()
                inflightSummarizationTask = nil
                inflightReviewTask?.cancel()
                inflightReviewTask = nil
            }
            // File → Open… (⌘O) command pipe. The menu writes a fresh
            // UUID into `menuCommands.openAudioFileToken`; we observe
            // the change and raise the same `.fileImporter` the
            // on-screen button does. We gate on busy state here rather
            // than at the menu level because the menu can't easily
            // observe `recorder` from app-level commands.
            .onChange(of: menuCommands.openAudioFileToken) { _, _ in
                guard !recorder.isRecording, !recorder.isAnalyzing else { return }
                filePickerMode = .audio
                showingFilePicker = true
            }
            // File → Export to JSON (⌘S) command pipe. Same gating as the
            // toolbar button so cmd-S during recording / analyzing /
            // empty-utterances no-ops cleanly. Additionally guards
            // `shareURL == nil` so repeated ⌘S while the share sheet is
            // already up doesn't write a fresh file + reassign shareURL —
            // `.sheet(item:)` interprets a new URL as "dismiss + represent",
            // which under rapid presses appears as a stacking sheet.
            .onChange(of: menuCommands.exportJSONToken) { _, _ in
                guard !recorder.utterances.isEmpty,
                      !recorder.isRecording,
                      !recorder.isAnalyzing,
                      shareURL == nil else { return }
                Task {
                    if let url = await recorder.exportJSON() {
                        shareURL = url
                    }
                }
            }
            // File → Save Session… (⌘S). Snapshot the recorder's
            // state into a `SessionDocument` synchronously, stash it
            // in `pendingSaveDocument`, and present the
            // `.fileExporter`. The exporter dismisses by clearing the
            // pending doc so re-triggering works.
            .onChange(of: menuCommands.saveSessionToken) { _, _ in
                guard !recorder.utterances.isEmpty,
                      !recorder.isRecording,
                      !recorder.isAnalyzing else { return }
                // makeSessionDocument turned async when it grew the
                // FluidAudio speaker-DB snapshot step — it now awaits
                // the diarizer actor. Wrap in Task so the SwiftUI
                // onChange closure stays synchronous.
                Task {
                    do {
                        let doc = try await recorder.makeSessionDocument()
                        pendingSaveDocument = SessionFileDocument(session: doc)
                        showingSaveSession = true
                    } catch {
                        sessionIOError = String(describing: error)
                    }
                }
            }
            // File → Import Session… (⇧⌘O). Reuses the single
            // fileImporter by switching its mode to .session before
            // raising it.
            .onChange(of: menuCommands.importSessionToken) { _, _ in
                guard !recorder.isRecording, !recorder.isAnalyzing else { return }
                filePickerMode = .session
                showingFilePicker = true
            }
            // Edit → Find (⌘F): move keyboard focus into the search
            // field. Setting `@FocusState` to true is the only way to
            // programmatically focus a SwiftUI TextField.
            .onChange(of: menuCommands.findToken) { _, _ in
                searchFieldFocused = true
            }
            // Recorder rotates `sessionToken` whenever its utterance
            // list changes identity (new recording, new file
            // analysis, imported `.xph`). Drop every view-side
            // `@State` keyed by the prior session's UUIDs so the
            // next session's renders aren't poisoned by stale row
            // ids — `selectedUtteranceRange` (and the timeline
            // strips that read it) mixes `recorder.utterances` with
            // `visibleUtteranceIDs`, so a leftover ID is benign in
            // theory but every set tracked here has its own way of
            // going wrong (a stale `selectedUtteranceID` carrying
            // a phantom selection; a stale `expandedUtteranceIDs`
            // member silently expanding the wrong row if a UUID
            // ever collides on import; etc). Cheap to wipe; keeps
            // the surface predictable across session boundaries.
            .onChange(of: recorder.sessionToken) { _, _ in
                visibleUtteranceIDs.removeAll()
                expandedUtteranceIDs.removeAll()
                normalizedTranscriptCache.removeAll()
                selectedUtteranceID = nil
                scrollRequestUtteranceID = nil
                hasUnreadUtterance = false
            }
            .modifier(SessionIOModifier(
                showingSaveSession: $showingSaveSession,
                pendingSaveDocument: $pendingSaveDocument,
                sessionIOError: $sessionIOError,
                defaultFilename: defaultSessionFilename()
            ))
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: filePickerAllowedTypes,
                allowsMultipleSelection: false,
                onCompletion: handleFilePickerResult
            )
            .alert(
                String(localized: "record.discardConfirm.title"),
                isPresented: $showingDiscardConfirm
            ) {
                Button(String(localized: "record.discardConfirm.confirm"), role: .destructive) {
                    Task { await recorder.toggle() }
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {}
            } message: {
                Text(
                    String(
                        format: String(localized: "record.discardConfirm.message"),
                        recorder.utterances.count
                    )
                )
            }
            .alert(
                String(localized: "record.discardConfirm.title"),
                isPresented: $showingFileDiscardConfirm
            ) {
                Button(String(localized: "record.discardConfirm.confirm"), role: .destructive) {
                    // Discard accepted — proceed to pacing choice. The
                    // pendingFileURL + scope ref are preserved across
                    // alerts so they survive the second hop.
                    showingPacingDialog = true
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {
                    releasePendingFileScope()
                }
            } message: {
                Text(
                    String(
                        format: String(localized: "record.discardConfirm.message"),
                        recorder.utterances.count
                    )
                )
            }
            .alert(
                String(localized: "pacing.title"),
                isPresented: $showingPacingDialog
            ) {
                Button(String(localized: "pacing.realtime")) {
                    if let url = pendingFileURL {
                        startFromFileAndReleaseScope(url, realTimePacing: true)
                    }
                }
                Button(String(localized: "pacing.realtime.audio")) {
                    if let url = pendingFileURL {
                        startFromFileAndReleaseScope(url, realTimePacing: true, audioOutputEnabled: true)
                    }
                }
                Button(String(localized: "pacing.fast8x")) {
                    if let url = pendingFileURL {
                        startFromFileAndReleaseScope(url, realTimePacing: false, fastPaceMultiplier: 8)
                    }
                }
                Button(String(localized: "pacing.fast4x")) {
                    if let url = pendingFileURL {
                        startFromFileAndReleaseScope(url, realTimePacing: false, fastPaceMultiplier: 4)
                    }
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {
                    releasePendingFileScope()
                }
            } message: {
                Text(String(localized: "pacing.message"))
            }
            // Edit Utterance sheet — raised by long-press on the
            // transcript Text of a row. Carries an
            // `UtteranceEstimate` snapshot so the sheet's initial
            // state is stable even if a parallel re-eval mutates
            // the row's underlying record.
            .sheet(item: $editingUtterance) { snapshot in
                EditUtteranceSheet(
                    utterance: snapshot,
                    maxDuration: recorder.fileTotalAudioDuration,
                    // Source-audio-backed sessions (file mode or an
                    // imported file-mode bundle) keep the play
                    // button + time spinners; mic-mode sessions hide
                    // them and only re-run text SER on commit.
                    audioEditingEnabled: recorder.playbackSourceURL != nil,
                    onPlayRange: { start, end in
                        recorder.playRange(start: start, end: end)
                    },
                    onStopRange: { recorder.stopPlayback() },
                    isPreviewPlaying: recorder.isPreviewPlaying,
                    onCommit: { newText, newStart, newEnd in
                        recorder.stopPlayback()
                        editingUtterance = nil
                        Task {
                            await recorder.commitHandEdit(
                                utteranceID: snapshot.id,
                                newText: newText,
                                newStart: newStart,
                                newEnd: newEnd
                            )
                        }
                    },
                    onCancel: {
                        recorder.stopPlayback()
                        editingUtterance = nil
                    }
                )
            }
            // Speaker rename alert — raised by the row's context
            // menu "Rename Speaker…" action. `editingSpeakerStored`
            // is the stored speaker id (e.g. `S01`); the bound text
            // is pre-filled with the current override if any.
            // Confirming with a blank field clears the override
            // (reverts to the default `S01`-style label).
            .alert(
                String(localized: "speaker.rename.title"),
                isPresented: Binding(
                    get: { editingSpeakerStored != nil },
                    set: { presented in if !presented { editingSpeakerStored = nil } }
                )
            ) {
                TextField(
                    String(localized: "speaker.rename.placeholder"),
                    text: $editingSpeakerName
                )
                Button(String(localized: "speaker.rename.save")) {
                    if let stored = editingSpeakerStored {
                        recorder.renameSpeaker(stored: stored, to: editingSpeakerName)
                    }
                    editingSpeakerStored = nil
                }
                Button(String(localized: "speaker.rename.cancel"), role: .cancel) {
                    editingSpeakerStored = nil
                }
            } message: {
                if let stored = editingSpeakerStored {
                    Text(String(format: String(localized: "speaker.rename.message"), stored))
                }
            }
        }
    }

    // MARK: - In-flight LLM task lifecycle

    /// Start (or re-start) the session summarization. Cancels any
    /// prior in-flight task first so re-tapping Regenerate while a
    /// pass is still running supersedes it cleanly. The Task is
    /// retained in `inflightSummarizationTask` so dismissing the
    /// sheet can cancel it.
    private func startSummarizationTask() {
        inflightSummarizationTask?.cancel()
        inflightSummarizationTask = Task {
            _ = await recorder.summarizeSession()
        }
    }

    /// Mirror of `startSummarizationTask` for the transcription
    /// reviewer. Same cancel-prior-task discipline.
    private func startReviewTask() {
        inflightReviewTask?.cancel()
        inflightReviewTask = Task {
            _ = await recorder.reviewSession()
        }
    }

    // MARK: - Toolbar buttons
    //
    // Split out as separate `@ViewBuilder` properties because three
    // inline `ToolbarItem` blocks with multi-clause `.disabled(...)`
    // gates plus auto-fire `Task` bodies tipped the SwiftUI body's
    // type-checker past its budget ("unable to type-check this
    // expression in reasonable time"). Each button now type-checks
    // in isolation.

    @ViewBuilder
    private var summarizeToolbarButton: some View {
        Button {
            showingSummaryView = true
            // Auto-generate on first open only, and only when the
            // summarizer is fully configured. Otherwise just open
            // the sheet so the user can reach the bottom controls
            // and enable / pick a backend / download the model.
            if recorder.lastSessionSummary == nil
                && recorder.summarizerEnabled
                && recorder.summarizerReady {
                startSummarizationTask()
            }
        } label: {
            Label(
                String(localized: "summary.summarize"),
                systemImage: "text.book.closed"
            )
        }
        .disabled(
            recorder.utterances.isEmpty
                || recorder.isRecording
                || recorder.isAnalyzing
                || recorder.summarizerInferenceRunning
        )
    }

    @ViewBuilder
    private var reviewToolbarButton: some View {
        Button {
            showingReviewView = true
            // Same auto-fire policy as the summarize button: kick
            // off only when we have nothing cached AND the backend
            // is fully configured.
            if recorder.transcriptionIssues.isEmpty
                && recorder.summarizerEnabled
                && recorder.summarizerReady {
                startReviewTask()
            }
        } label: {
            Label(
                String(localized: "review.toolbar"),
                systemImage: "text.magnifyingglass"
            )
        }
        .disabled(
            recorder.utterances.isEmpty
                || recorder.isRecording
                || recorder.isAnalyzing
                || recorder.transcriptionReviewRunning
                || recorder.summarizerInferenceRunning
        )
    }

    @ViewBuilder
    private var searchReplaceToolbarButton: some View {
        Button {
            showingSearchReplaceView = true
        } label: {
            Label(
                String(localized: "searchReplace.toolbar"),
                systemImage: "magnifyingglass"
            )
        }
        // Disable during recording / analysis: commitHandEdit requires
        // the controller to be idle, and surfacing the sheet earlier
        // would let the user queue work that silently fails.
        .disabled(
            recorder.utterances.isEmpty
                || recorder.isRecording
                || recorder.isAnalyzing
        )
    }

    @ViewBuilder
    private var exportToolbarButton: some View {
        Button {
            Task {
                if let url = await recorder.exportJSON() {
                    shareURL = url
                }
            }
        } label: {
            Label(
                String(localized: "export.json"),
                systemImage: "square.and.arrow.up"
            )
        }
        .disabled(
            recorder.utterances.isEmpty
                || recorder.isRecording
                || recorder.isAnalyzing
        )
    }

    // MARK: - Left pane (1/3): controls

    private var controlPane: some View {
        // Two-region layout: a fixed header that pins the controls at
        // the top (input picker, record/open, level meter, status,
        // error) plus a scrollable region below that holds the cards
        // (Settings + Pipeline + Summary + Statistics). The header
        // never scrolls off — the user can always reach Start/Stop
        // even with every card expanded.
        VStack(spacing: 16) {
            inputPicker

            HStack(spacing: 12) {
                recordButton
                openFileButton
            }

            if recorder.isRecording {
                LevelMeterView(level: recorder.inputLevel)
                    .frame(maxWidth: 280)
            }

            statusLine

            if let error = recorder.errorMessage {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .padding(.horizontal)
                    .multilineTextAlignment(.center)
            }

            // Card section split across four swipeable pages so
            // the left pane doesn't grow into a long single scroll
            // (the cluster + heatmap especially want vertical room
            // to render their data legibly). Page 1: session
            // controls — Settings + Pipeline. Page 2: read-only
            // affect output — Summary + Statistics. Page 3:
            // diarizer cluster diagnostics — PCA scatter + pairwise
            // heatmap. Page 4: summarizer configuration — toggle,
            // backend picker, install / Remove-model. The page-
            // style indicator dots render at the bottom of the
            // TabView; we force `backgroundDisplayMode: .always`
            // so they stay visible against the glass cards on
            // iPadOS 26.
            TabView {
                ScrollView(.vertical, showsIndicators: true) {
                    VStack(spacing: 16) {
                        SettingsCard(
                            languagePicker: { languagePicker },
                            speechBoostToggle: { speechBoostToggle },
                            textSERPicker: { textSERPicker }
                        )
                        PipelineCard(recorder: recorder)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 32)
                }
                .clipped()

                ScrollView(.vertical, showsIndicators: true) {
                    VStack(spacing: 16) {
                        SummaryCard(
                            summary: displayedSummary,
                            totalDuration: displayedSummary.totalDuration
                        )
                        StatisticsCard(summary: displayedSummary)
                        SERAggregateCard(
                            recorder: recorder,
                            focusedUtteranceID: selectedUtteranceID
                        )
                        FusionLegendCard(recorder: recorder)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 32)
                }
                .clipped()

                ScrollView(.vertical, showsIndicators: true) {
                    VStack(spacing: 16) {
                        SpeakerRosterCard(
                            recorder: recorder,
                            cluster: recorder.speakerCluster,
                            highlightedSpeakerID: focusedUtteranceSpeakerID
                        )
                        SpeakerClusterCard(
                            cluster: recorder.speakerCluster,
                            highlightedSpeakerID: focusedUtteranceSpeakerID,
                            focusedEmbedding: focusedUtteranceEmbedding
                        )
                        SpeakerHeatmapCard(
                            cluster: recorder.speakerCluster,
                            highlightedSpeakerID: focusedUtteranceSpeakerID
                        )
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 32)
                }
                .clipped()

                ScrollView(.vertical, showsIndicators: true) {
                    VStack(spacing: 16) {
                        SummarizerCard(recorder: recorder)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.horizontal, 8)
                    .padding(.bottom, 32)
                }
                .clipped()
            }
            .tabViewStyle(.page(indexDisplayMode: .always))
            .indexViewStyle(.page(backgroundDisplayMode: .always))
            // While idle (no recording in flight) the controller's
            // continuous-diarize tick isn't refreshing the cluster
            // snapshot — pull at 1 Hz so the heatmap + scatter stay
            // live after a file analysis completes or a session is
            // loaded. Cheap (just hands back resident `[Float]`
            // arrays), no-op when no pipeline is up. Lives on the
            // TabView (not the cluster page) so swiping to that
            // page shows the latest snapshot immediately rather
            // than blinking through a stale state for one second.
            .task {
                while !Task.isCancelled {
                    if !recorder.isRecording {
                        await recorder.refreshClusterSnapshot()
                    }
                    try? await Task.sleep(nanoseconds: 1_000_000_000)
                }
            }
        }
        .padding()
        .frame(maxHeight: .infinity, alignment: .top)
    }

    // MARK: - Right pane (2/3): transcript

    @ViewBuilder
    private var transcriptPane: some View {
        if recorder.utterances.isEmpty {
            ContentUnavailableView(
                String(localized: "transcript.empty.title"),
                systemImage: "waveform",
                description: Text(String(localized: "transcript.empty.subtitle"))
            )
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else {
            VStack(spacing: 0) {
                filterBar
                speakerChipBar
                diarizationTimelineStrip
                emotionTimelineStrip
                fusionContributionStrip
                if filteredIndexedUtterances.isEmpty {
                    noMatchesView
                } else {
                    transcriptList
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
    }

    /// Per-session diarizer-timeline visualization. Hidden until
    /// the cumulative timeline has at least one observation and we
    /// can derive a positive total duration. Selecting an utterance
    /// in the list outlines the strip region for that row's audio
    /// range.
    @ViewBuilder
    private var diarizationTimelineStrip: some View {
        let timeline = recorder.diarizationTimeline
        let total = transcriptTotalDuration
        if !timeline.isEmpty, total > 0 {
            DiarizationTimelineStrip(
                segments: timeline,
                totalDuration: total,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
        }
    }

    /// Companion emotion-label strip rendered immediately below the
    /// diarizer timeline. Shares X axis + tap behavior with the
    /// speaker strip so a glance across both reveals "this speaker,
    /// this emotion, at this audio time" in one read. Hidden until
    /// at least one utterance has a fused top label — before the
    /// first analysis finishes there's nothing to colour.
    @ViewBuilder
    private var emotionTimelineStrip: some View {
        let total = transcriptTotalDuration
        let hasAnyLabel = recorder.utterances.contains { $0.fusedTopLabel != nil }
        if hasAnyLabel, total > 0 {
            EmotionTimelineStrip(
                utterances: recorder.utterances,
                totalDuration: total,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            // Tighter vertical pad than the speaker strip so the
            // two strips read as a stacked pair, not two unrelated
            // bars with whitespace between them.
            .padding(.bottom, 6)
        }
    }

    /// Per-utterance modality-balance strip — for each row, a
    /// horizontal bar split into acoustic (blue) / text (orange)
    /// segments sized by `LateFusion.defaultLabelFusionShare`. Lets
    /// the user scan the conversation and spot stretches where one
    /// modality dominated the fused label. Hidden until at least
    /// one utterance carries one of the two modality outputs.
    @ViewBuilder
    private var fusionContributionStrip: some View {
        let total = transcriptTotalDuration
        let hasAnyModality = recorder.utterances.contains {
            $0.acousticCategorical != nil || $0.plutchik != nil
        }
        if hasAnyModality, total > 0 {
            FusionContributionStrip(
                utterances: recorder.utterances,
                totalDuration: total,
                acousticWeight: recorder.fusionAcousticWeight,
                textWeightFloor: recorder.fusionTextWeightFloor,
                selectedRange: selectedUtteranceRange,
                onTapAtTime: { t in
                    guard let target = nearestUtterance(toTime: t) else { return }
                    selectedUtteranceID = target.id
                    scrollRequestUtteranceID = target.id
                }
            )
            .padding(.horizontal, 12)
            .padding(.bottom, 6)
        }
    }

    /// Resolve an audio-time to the nearest utterance: prefer the
    /// row whose `[start, end]` contains the time, fall back to
    /// the row whose midpoint is closest. Returns nil only when
    /// `utterances` is empty.
    private func nearestUtterance(toTime t: TimeInterval) -> UtteranceEstimate? {
        let utts = recorder.utterances
        if let containing = utts.first(where: { $0.start <= t && t < $0.end }) {
            return containing
        }
        return utts.min {
            abs(($0.start + $0.end) / 2 - t) < abs(($1.start + $1.end) / 2 - t)
        }
    }

    /// Conversation duration for the timeline strip's X axis.
    /// Always derived from the latest processed timestamp — the
    /// max of any finalized utterance's end and the diarizer's
    /// latest observation — so the strip grows in real time as
    /// analysis progresses. File mode previously preferred the
    /// source-file's full length here, which made the strip
    /// snap to its final width the moment the file opened and
    /// left the highlighter inching along the leftmost few
    /// percent during analysis. That was jarring — the user
    /// wants the same "depict what we've actually processed so
    /// far" semantics they get from a live mic recording, in both
    /// modes.
    private var transcriptTotalDuration: TimeInterval {
        let utteranceMax = recorder.utterances.map(\.end).max() ?? 0
        let timelineMax = recorder.diarizationTimeline.map(\.end).max() ?? 0
        return max(utteranceMax, timelineMax)
    }

    /// What the timeline strip should outline. Two cases:
    ///
    /// 1. A row is explicitly selected → outline that row's
    ///    `[start, end]` so the user can see where on the
    ///    timeline they tapped.
    /// 2. Nothing is selected → outline the range that spans
    ///    every utterance currently on screen. This keeps the
    ///    strip's highlight tied to the user's reading focus
    ///    even without an explicit selection: scrolling the list
    ///    moves the highlight along with what they're looking at.
    ///
    /// Returns nil when neither case has data (empty list, or
    /// nothing visible yet on first layout). Visibility tracking
    /// uses `.onScrollVisibilityChange` in `TranscriptList` (not
    /// `.onAppear` / `.onDisappear`) so the set stays in sync
    /// even across the layout-flux window of a row expansion;
    /// `.onDisappear` was firing unreliably in that case and
    /// leaving phantom entries that stretched the highlighter.
    /// Speaker id of the currently-focused utterance, or nil when
    /// no row is focused or the focused row was just deleted.
    /// Feeds the cluster-scatter highlight ring so the user can
    /// see which centroid corresponds to the row they're inspecting.
    private var focusedUtteranceSpeakerID: String? {
        guard let id = selectedUtteranceID else { return nil }
        return recorder.utterances.first(where: { $0.id == id })?.speakerID
    }

    /// Raw speaker embedding of the focused utterance, captured by
    /// the pipeline at analysis time. Drives the cluster scatter's
    /// per-observation focus arrow so it points at the *specific*
    /// node for the focused row instead of falling back to the
    /// speaker's centroid. Nil when the row predates this capture
    /// (older session) or the diarizer was unavailable.
    private var focusedUtteranceEmbedding: [Float]? {
        guard let id = selectedUtteranceID else { return nil }
        return recorder.utteranceEmbeddings[id]
    }

    private var selectedUtteranceRange: (start: TimeInterval, end: TimeInterval)? {
        if let id = selectedUtteranceID,
           let u = recorder.utterances.first(where: { $0.id == id }) {
            return (start: u.start, end: u.end)
        }
        // Single-pass min/max instead of filter + map + min/map +
        // max — three allocations and three iterations collapse to
        // one. Body re-fires per scroll because the per-row
        // visibility tracker writes to `visibleUtteranceIDs`, so
        // this getter runs on every flick.
        var minStart: TimeInterval = .infinity
        var maxEnd: TimeInterval = -.infinity
        for u in recorder.utterances where visibleUtteranceIDs.contains(u.id) {
            if u.start < minStart { minStart = u.start }
            if u.end > maxEnd { maxEnd = u.end }
        }
        guard minStart.isFinite else { return nil }
        return (start: minStart, end: maxEnd)
    }

    /// Inline filter row: a free-text search field plus a label
    /// dropdown. Both filters AND together so the user can scope by
    /// "happy utterances containing 楽しい" etc. The bar always shows
    /// when the full list is non-empty so the controls don't appear
    /// then disappear as utterances arrive.
    @ViewBuilder
    private var filterBar: some View {
        HStack(spacing: 8) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField(
                    String(localized: "filter.search.placeholder"),
                    text: $searchText
                )
                .textFieldStyle(.plain)
                .font(.callout)
                .submitLabel(.search)
                .focused($searchFieldFocused)
                // Esc inside the search field returns focus to the
                // surrounding view so subsequent ⌘ shortcuts and
                // arrow-key list navigation work without an extra tap.
                // Returning `.handled` keeps the keystroke from
                // bubbling to the List's own Esc handler (which would
                // otherwise also clear the row selection).
                .onKeyPress(.escape) {
                    if searchFieldFocused {
                        searchFieldFocused = false
                        return .handled
                    }
                    return .ignored
                }
                if !searchText.isEmpty {
                    Button {
                        searchText = ""
                    } label: {
                        Image(systemName: "xmark.circle.fill")
                            .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                }
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 7)
            .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))

            // Label picker. Pulled into a Menu so the chevron looks
            // native and tinted-by-label rows read at a glance. Tap
            // expands to a list of labels with their per-label counts;
            // "All labels" clears the filter.
            Menu {
                Button {
                    selectedLabelFilter = nil
                } label: {
                    if selectedLabelFilter == nil {
                        Label(String(localized: "filter.label.all"), systemImage: "checkmark")
                    } else {
                        Text(String(localized: "filter.label.all"))
                    }
                }
                if !availableLabels.isEmpty {
                    Divider()
                }
                ForEach(availableLabels, id: \.self) { label in
                    Button {
                        selectedLabelFilter = label
                    } label: {
                        let displayed = label.capitalized(with: Locale(identifier: "en_US"))
                        let count = recorder.conversationSummary.labelCounts[label] ?? 0
                        if selectedLabelFilter == label {
                            Label("\(displayed) (\(count))", systemImage: "checkmark")
                        } else {
                            Text("\(displayed) (\(count))")
                        }
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: "line.3.horizontal.decrease.circle")
                    Text(filterLabelDisplay)
                        .lineLimit(1)
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2)
                }
                .font(.callout)
                .padding(.horizontal, 10)
                .padding(.vertical, 7)
                .background(.quaternary, in: RoundedRectangle(cornerRadius: 8, style: .continuous))
            }
            .foregroundStyle(selectedLabelFilter.map(emotionTint(for:)) ?? Color.primary)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
    }

    private var filterLabelDisplay: String {
        if let label = selectedLabelFilter {
            return label.capitalized(with: Locale(identifier: "en_US"))
        }
        return String(localized: "filter.label.all")
    }

    /// Horizontal chip row of every speaker seen this session, plus
    /// an "All" chip on the left that clears the speaker filter.
    /// Tapping a speaker chip toggles it as the active speaker
    /// filter — tapping the same chip again returns to "All". Hidden
    /// when only one speaker has been detected (no point filtering
    /// when there's nothing to choose between).
    @ViewBuilder
    private var speakerChipBar: some View {
        let speakers = availableSpeakers
        if speakers.count >= 2 {
            HStack(spacing: 8) {
                ScrollView(.horizontal, showsIndicators: false) {
                    // Wrap the chip strip in a `GlassEffectContainer`
                    // so adjacent glass capsules merge their blurs
                    // and the morph feels fluid when a speaker is
                    // added/removed.
                    GlassEffectContainer(spacing: 8) {
                        HStack(spacing: 8) {
                            speakerChip(
                                text: String(localized: "filter.speaker.all"),
                                tint: .secondary,
                                isSelected: selectedSpeakerFilter == nil
                            ) {
                                selectedSpeakerFilter = nil
                            }
                            ForEach(speakers, id: \.self) { id in
                                let label = formatSpeakerLabel(
                                    id,
                                    customName: recorder.speakerDisplayName(forStored: id)
                                )
                                let tint = speakerTint(for: id)
                                speakerChip(
                                    text: label,
                                    tint: tint,
                                    isSelected: selectedSpeakerFilter == id
                                ) {
                                    selectedSpeakerFilter = (selectedSpeakerFilter == id) ? nil : id
                                }
                            }
                        }
                    }
                    .padding(.leading, 12)
                    .padding(.vertical, 6)
                }
                // Distinct speaker count, pinned to the trailing edge
                // so the chip strip can scroll horizontally beneath it
                // without pushing the count out of view.
                Label("\(speakers.count)", systemImage: "person.2.fill")
                    .font(.caption.monospacedDigit())
                    .labelStyle(.titleAndIcon)
                    .foregroundStyle(.secondary)
                    .padding(.trailing, 12)
            }
        }
    }

    @ViewBuilder
    private func speakerChip(
        text: String,
        tint: Color,
        isSelected: Bool,
        action: @escaping () -> Void
    ) -> some View {
        Button(action: action) {
            Text(text)
                .font(.caption.bold())
                .padding(.horizontal, 10)
                .padding(.vertical, 5)
                .foregroundStyle(tint)
                .glassEffect(
                    .regular
                        .tint(tint.opacity(isSelected ? 0.55 : 0.2))
                        .interactive(),
                    in: Capsule()
                )
        }
        .buttonStyle(.plain)
    }

    @ViewBuilder
    private var noMatchesView: some View {
        ContentUnavailableView {
            Label(
                String(localized: "filter.noMatches.title"),
                systemImage: "line.3.horizontal.decrease.circle"
            )
        } description: {
            Text(String(localized: "filter.noMatches.subtitle"))
        } actions: {
            Button(String(localized: "filter.noMatches.clear")) {
                searchText = ""
                selectedLabelFilter = nil
                selectedSpeakerFilter = nil
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// Session-language picker. Drives the ASR locale (Apple
    /// SpeechTranscriber) and the text-SER gating (DeBERTa-WRIME is
    /// Japanese-only and hides for non-Japanese sessions). Disabled
    /// while a session is active because the streaming transcriber
    /// is locked to its start-time locale — the user can still see
    /// which language is in effect for the running session.
    @ViewBuilder
    private var languagePicker: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(String(localized: "settings.language"))
                .font(.caption)
                .foregroundStyle(.secondary)
            Picker(
                String(localized: "settings.language"),
                selection: Binding(
                    get: { recorder.sessionLanguage },
                    set: { newValue in
                        Task { await recorder.setSessionLanguage(newValue) }
                    }
                )
            ) {
                ForEach(SessionLanguage.allCases, id: \.self) { lang in
                    Text("\(lang.flag) \(lang.displayName)").tag(lang)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .disabled(recorder.isRecording || recorder.isAnalyzing)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private var textSERPicker: some View {
        if recorder.availableTextSERBackends.count > 1 {
            VStack(alignment: .leading, spacing: 4) {
                Text(String(localized: "settings.textSER"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Picker(
                    String(localized: "settings.textSER"),
                    selection: Binding(
                        get: { recorder.currentTextSERBackend ?? .foundationModels },
                        set: { newValue in
                            Task { await recorder.setTextSERBackend(newValue) }
                        }
                    )
                ) {
                    ForEach(recorder.availableTextSERBackends, id: \.self) { backend in
                        Text(Self.label(for: backend)).tag(backend)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private static func label(for backend: SwitchingTextSER.Backend) -> String {
        switch backend {
        case .deberta:          return String(localized: "settings.textSER.deberta")
        case .foundationModels: return String(localized: "settings.textSER.foundationModels")
        }
    }

    @ViewBuilder
    private var speechBoostToggle: some View {
        // Hide the speech-boost EQ control while the active source is a
        // file — the toggle wouldn't affect file content.
        if case .microphone = recorder.sourceMode {
            speechBoostToggleBody
        }
    }

    private var speechBoostToggleBody: some View {
        Toggle(
            isOn: Binding(
                get: { recorder.isSpeechBoostEnabled },
                set: { newValue in
                    Task { await recorder.setSpeechBoostEnabled(newValue) }
                }
            )
        ) {
            Label(
                String(localized: "settings.speechBoost"),
                systemImage: "waveform.badge.plus"
            )
        }
        .toggleStyle(.switch)
        .padding(.horizontal)
    }

    @ViewBuilder
    private var inputPicker: some View {
        // Always render, even when the inputs list is empty (file
        // mode, or before the first refresh) or contains only the
        // built-in mic. Keeping the picker visible keeps the
        // toolbar layout stable across state transitions and gives
        // the user a permanent at-a-glance indicator of which input
        // would be used if recording started now.
        Menu {
            if recorder.availableInputs.isEmpty {
                Text(String(localized: "input.default"))
            } else {
                ForEach(recorder.availableInputs) { input in
                    Button {
                        Task { await recorder.selectInput(uid: input.uid) }
                    } label: {
                        if input.uid == recorder.currentInputUID {
                            Label(input.displayName, systemImage: "checkmark")
                        } else {
                            Text(input.displayName)
                        }
                    }
                }
            }
        } label: {
            let current = recorder.availableInputs.first(where: { $0.uid == recorder.currentInputUID })
            HStack(spacing: 6) {
                Image(systemName: Self.symbol(for: current?.kind ?? .builtInMic))
                Text(current?.displayName ?? String(localized: "input.default"))
                    .lineLimit(1)
                Image(systemName: "chevron.up.chevron.down")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .font(.footnote)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .background(.tint.opacity(0.12), in: Capsule())
        }
        .disabled(inputPickerDisabled)
    }

    /// True when there's nothing actionable for the input picker:
    /// a session is in flight, the source is a file (mic isn't
    /// used), or there's at most one input to choose from.
    private var inputPickerDisabled: Bool {
        if recorder.isRecording { return true }
        if recorder.isAnalyzing { return true }
        if case .file = recorder.sourceMode { return true }
        return recorder.availableInputs.count <= 1
    }

    private static func symbol(for kind: AudioInputDescription.Kind) -> String {
        switch kind {
        case .builtInMic:   return "mic"
        case .wiredHeadset: return "headphones"
        case .bluetooth:    return "airpods"
        case .usb:          return "cable.connector"
        case .airPlay:      return "airplayaudio"
        case .carPlay:      return "car"
        case .other:        return "mic"
        }
    }

    @ViewBuilder
    private var recordButton: some View {
        Button {
            if !recorder.isRecording && !recorder.utterances.isEmpty {
                showingDiscardConfirm = true
            } else {
                Task { await recorder.toggle() }
            }
        } label: {
            Label(
                recordButtonTitle,
                systemImage: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill"
            )
            .font(.title3)
        }
        .buttonStyle(.borderedProminent)
        .tint(recorder.isRecording ? .red : .accentColor)
        .disabled(recorder.isAnalyzing)
    }

    private var recordButtonTitle: String {
        guard recorder.isRecording else {
            return String(localized: "record.start")
        }
        if case .file = recorder.sourceMode {
            return String(localized: "record.stop.file")
        }
        return String(localized: "record.stop")
    }

    private var openFileButton: some View {
        Button {
            // Strict audio entry point: pin the picker mode to
            // `.audio` so a stale mode left over from an earlier
            // Import Session… invocation can't leak through and
            // make this button accept `.xph` files.
            filePickerMode = .audio
            showingFilePicker = true
        } label: {
            Label(String(localized: "file.open"), systemImage: "doc.badge.arrow.up")
                .font(.title3)
                .labelStyle(.iconOnly)
        }
        .buttonStyle(.bordered)
        .disabled(recorder.isRecording || recorder.isAnalyzing)
    }

    @ViewBuilder
    private var statusLine: some View {
        if recorder.isRecording {
            VStack(spacing: 4) {
                if case .file(let url) = recorder.sourceMode {
                    Text(String(format: String(localized: "file.analyzing"), url.lastPathComponent))
                        .font(.caption)
                        .foregroundStyle(.tint)
                        .lineLimit(1)
                        .truncationMode(.middle)
                }
                if case .file = recorder.sourceMode,
                   let frac = recorder.fileCompletionFraction {
                    ProgressView(value: frac)
                        .progressViewStyle(.linear)
                        .tint(.accentColor)
                }
                Text(statusLineText)
                    .font(.system(.body, design: .monospaced))
                    .foregroundStyle(.secondary)
            }
        } else if recorder.isAnalyzing {
            HStack(spacing: 8) {
                ProgressView()
                Text(String(localized: "analyze.inProgress"))
                    .foregroundStyle(.secondary)
            }
        } else if recorder.isWarmingUp {
            HStack(spacing: 8) {
                ProgressView()
                Text(String(localized: "warmup.inProgress"))
                    .foregroundStyle(.secondary)
            }
        }
    }

    /// Summary derived from whatever's currently displayed in the
    /// transcript list. When all filters are off this matches the
    /// recorder's running `conversationSummary` exactly (same input
    /// utterances, same fold). When any filter is active — search
    /// text, label, speaker — this re-folds only the visible
    /// utterances, so the Summary and Statistics panels read the
    /// filtered slice instead of the lifetime totals.
    ///
    /// O(n) per render. Negligible for typical session sizes (a few
    /// hundred utterances); SwiftUI's view-diffing means it only
    /// runs when the filter or utterance state actually changes.
    private var displayedSummary: ConversationSummary {
        refreshFilterMemoIfNeeded()
        return filterMemo.summary
    }

    /// Distinct top labels seen so far this session, used to populate
    /// the label-filter menu. Sorted alphabetically for stable order.
    private var availableLabels: [String] {
        Set(recorder.utterances.compactMap { $0.fusedTopLabel }).sorted()
    }

    /// Distinct speaker IDs in the order they first appear in the
    /// session. First-appearance order (rather than alphabetical)
    /// keeps the chip row stable as new utterances arrive — a new
    /// speaker is appended at the end instead of reshuffling
    /// existing chips, and the row reads in the same order as the
    /// utterance list.
    private var availableSpeakers: [String] {
        var seen: Set<String> = []
        var ordered: [String] = []
        for u in recorder.utterances where !seen.contains(u.speakerID) {
            seen.insert(u.speakerID)
            ordered.append(u.speakerID)
        }
        return ordered
    }

    /// `(originalIndex, utterance)` pairs surviving both filters. The
    /// original index is preserved so each row's "#N" badge keeps the
    /// utterance's stable session number even when the list is
    /// filtered (a row labeled #15 always points at the 15th utterance
    /// of the recording).
    private var filteredIndexedUtterances: [(idx: Int, u: UtteranceEstimate)] {
        refreshFilterMemoIfNeeded()
        return filterMemo.results
    }

    /// Recompute the filter + summary memo when any input dependency
    /// has changed since the last render. No-op when nothing changed
    /// — the cached `results` and `summary` are returned as-is. The
    /// dependency key intentionally only fingerprints inputs that
    /// affect the filter outcome; render-only state (selection,
    /// playback, scroll) doesn't invalidate the memo.
    private func refreshFilterMemoIfNeeded() {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let key = FilterDepsKey(
            normalizedQuery: trimmed.isEmpty
                ? ""
                : JapaneseSearchNormalizer.normalize(trimmed),
            labelFilter: selectedLabelFilter,
            speakerFilter: selectedSpeakerFilter,
            utteranceCount: recorder.utterances.count,
            utterancesVersion: recorder.utterancesVersion
        )
        if filterMemo.lastKey == key { return }

        let results: [(idx: Int, u: UtteranceEstimate)] = recorder
            .utterances
            .enumerated()
            .compactMap { idx, u in
                if let filterLabel = key.labelFilter,
                   u.fusedTopLabel != filterLabel {
                    return nil
                }
                if let filterSpeaker = key.speakerFilter,
                   u.speakerID != filterSpeaker {
                    return nil
                }
                if !key.normalizedQuery.isEmpty {
                    // Fall back to inline normalization when the
                    // async cache hasn't caught up yet. The async
                    // refresher will populate it momentarily; the
                    // per-row inline call is rare.
                    let normalizedText = normalizedTranscriptCache[u.id]
                        ?? JapaneseSearchNormalizer.normalize(u.transcript)
                    if !normalizedText.contains(key.normalizedQuery) {
                        return nil
                    }
                }
                return (idx, u)
            }

        var summary = ConversationSummary()
        for (_, u) in results {
            summary.update(with: u)
        }
        filterMemo.lastKey = key
        filterMemo.results = results
        filterMemo.summary = summary
    }

    /// Bring `normalizedTranscriptCache` up to date with the recorder's
    /// current utterance list. Only normalizes utterances that aren't
    /// already in the cache, so steady-state utterance arrivals each
    /// pay one normalize call (not N). Normalization runs concurrently
    /// across the missing entries via `TaskGroup`, off the MainActor;
    /// completed results are merged back into `@State` in one hop so
    /// the view body doesn't re-evaluate per row.
    private func refreshSearchCache() {
        let utterances = recorder.utterances
        let cached = normalizedTranscriptCache
        let missing = utterances.filter { cached[$0.id] == nil }
        guard !missing.isEmpty else { return }

        searchCacheTask?.cancel()
        searchCacheTask = Task.detached(priority: .userInitiated) {
            // Hand each utterance to its own child task; the work is
            // CPU-bound CFStringTokenizer calls, so the cooperative
            // pool will parallelize across the device's cores.
            let normalized = await withTaskGroup(
                of: (UUID, String).self
            ) { group -> [(UUID, String)] in
                for u in missing {
                    if Task.isCancelled { break }
                    group.addTask {
                        (u.id, JapaneseSearchNormalizer.normalize(u.transcript))
                    }
                }
                var out: [(UUID, String)] = []
                for await pair in group {
                    out.append(pair)
                }
                return out
            }
            if Task.isCancelled { return }
            await MainActor.run {
                for (id, text) in normalized {
                    normalizedTranscriptCache[id] = text
                }
            }
        }
    }

    /// Wall-time + sample-count line for the active session.
    /// File-mode shows the completion bar separately above this line,
    /// so the text content is identical for both modes.
    private var statusLineText: String {
        String(
            format: String(localized: "record.status.format"),
            formatClock(recorder.elapsedSeconds),
            formatCount(recorder.samplesCaptured)
        )
    }

    /// Content types the single shared fileImporter advertises, based
    /// on which menu command opened it. `xephonSession` is registered
    /// via project.yml's `UTExportedTypeDeclarations`, so the picker
    /// greys out non-`.xph` files when in session mode.
    private var filePickerAllowedTypes: [UTType] {
        switch filePickerMode {
        case .audio:
            return [.audio, .mp3, .wav, .mpeg4Audio, .aiff]
        case .session:
            return [.xephonSession]
        }
    }

    /// Single dispatcher for the shared fileImporter. The audio path
    /// pins security scope and hands off to the pacing dialog; the
    /// session path reads + decodes synchronously off MainActor.
    private func handleFilePickerResult(_ result: Result<[URL], any Error>) {
        switch result {
        case .success(let urls):
            guard let url = urls.first else { return }
            switch filePickerMode {
            case .audio:
                pendingFileURL = url
                pendingFileScopeAcquired = url.startAccessingSecurityScopedResource()
                if !recorder.utterances.isEmpty {
                    showingFileDiscardConfirm = true
                } else {
                    showingPacingDialog = true
                }
            case .session:
                Task { await loadSessionFromPickedFile(url) }
            }
        case .failure(let error):
            AppLog.app.error("file picker: \(String(describing: error), privacy: .public)")
            if filePickerMode == .session {
                sessionIOError = String(describing: error)
            }
        }
    }

    /// Filename suggestion for the Save Session… panel. Uses an
    /// ISO-8601-ish stamp so successive saves don't collide and so
    /// the user can scan the file list chronologically.
    private func defaultSessionFilename() -> String {
        let fmt = DateFormatter()
        fmt.dateFormat = "yyyy-MM-dd_HHmm"
        return "xephon-\(fmt.string(from: Date())).xph"
    }

    /// Read a picked `.xph` URL into the recorder. Security-scoped:
    /// the picker hands us a scoped URL; we hold it just long enough
    /// to read the bytes and let `loadSession` extract any audio
    /// into the app's sandbox.
    private func loadSessionFromPickedFile(_ url: URL) async {
        let scoped = url.startAccessingSecurityScopedResource()
        defer { if scoped { url.stopAccessingSecurityScopedResource() } }
        do {
            let data = try Data(contentsOf: url)
            let document = try SessionBundle.decode(data)
            try await recorder.loadSession(document)
        } catch {
            sessionIOError = String(describing: error)
        }
    }

    /// Release the picker's scope ref we grabbed in the fileImporter
    /// callback and clear the pending URL. Used by all cancel paths
    /// in the discard / pacing dialogs.
    private func releasePendingFileScope() {
        if pendingFileScopeAcquired, let url = pendingFileURL {
            url.stopAccessingSecurityScopedResource()
        }
        pendingFileScopeAcquired = false
        pendingFileURL = nil
    }

    /// Hand the URL to the recorder (which acquires its own scope ref
    /// synchronously in `startFromFile`), then release the picker's
    /// ref we've been holding through the dialog hop.
    private func startFromFileAndReleaseScope(
        _ url: URL,
        realTimePacing: Bool = false,
        fastPaceMultiplier: Int = 8,
        audioOutputEnabled: Bool = false
    ) {
        let acquired = pendingFileScopeAcquired
        pendingFileScopeAcquired = false
        pendingFileURL = nil
        Task {
            await recorder.startFromFile(
                url,
                realTimePacing: realTimePacing,
                fastPaceMultiplier: fastPaceMultiplier,
                audioOutputEnabled: audioOutputEnabled
            )
            if acquired {
                url.stopAccessingSecurityScopedResource()
            }
        }
    }

    /// Map the recorder's source/phase state onto the per-row playback
    /// availability. Hidden for mic sessions, disabled while a file
    /// analysis is still running (so the user knows playback is on
    /// the way), idle when ready, and playing for the one row that's
    /// currently mid-playback.
    private func playbackAvailability(for u: UtteranceEstimate) -> UtteranceRow.PlaybackAvailability {
        guard recorder.playbackSourceURL != nil else { return .unavailable }
        if recorder.isRecording || recorder.isAnalyzing { return .disabled }
        // Disable playback across the list while a re-evaluation is
        // in flight — offline ASR + SER serialize naturally and the
        // user shouldn't be racing audio reads against the
        // re-analysis pass.
        if recorder.reevaluatingUtteranceID != nil { return .disabled }
        if recorder.playingUtteranceID == u.id { return .playing }
        return .idle
    }

    /// Re-evaluate availability matches playback's gating (no source
    /// audio → unavailable, recording/analyzing → disabled), plus a
    /// dedicated `.running` for the one row whose re-evaluation is in
    /// flight and `.completed` for rows whose `wasReevaluated` flag
    /// is set. Other rows get `.disabled` during a re-evaluation so
    /// the user can't queue overlapping passes. Reading the flag off
    /// the utterance itself means the green marker survives Save/Load
    /// and shows up in the JSON export alongside the affect data.
    ///
    /// `.completed` is checked **before** the session-busy `.disabled`
    /// branches so the green marker stays put while another row's
    /// re-evaluation runs. The controller's own guards
    /// (`reevaluatingUtteranceID == nil`, `phase == .idle`) keep the
    /// no-op safety net intact for any tap that arrives during the
    /// busy window.
    private func reevaluateAvailability(for u: UtteranceEstimate) -> UtteranceRow.ReevaluateAvailability {
        guard recorder.playbackSourceURL != nil else { return .unavailable }
        if recorder.reevaluatingUtteranceID == u.id { return .running }
        if u.wasReevaluated == true { return .completed }
        if recorder.isRecording || recorder.isAnalyzing { return .disabled }
        if recorder.reevaluatingUtteranceID != nil { return .disabled }
        return .idle
    }

    /// Flip the per-utterance expansion state. Invoked by long-press
    /// on a row and by Space-key while a row is the list selection.
    private func toggleExpansion(_ id: UUID) {
        if expandedUtteranceIDs.contains(id) {
            expandedUtteranceIDs.remove(id)
        } else {
            expandedUtteranceIDs.insert(id)
        }
    }

    /// True iff the chronologically last utterance *in the filtered
    /// view* has been laid out and is currently on-screen. Empty list
    /// counts as "visible" (no last to track) so the capsule never
    /// appears in the empty state. Tracking the filtered last (not the
    /// full-list last) keeps the auto-scroll behavior correct under
    /// active filters: a new utterance that doesn't match the filter
    /// shouldn't scroll the list, and one that does shouldn't be
    /// labeled "unread" if the user is already at the filtered bottom.
    private var transcriptList: some View {
        TranscriptList(
            recorder: recorder,
            items: filteredIndexedUtterances,
            selectedUtteranceID: $selectedUtteranceID,
            scrollRequestUtteranceID: $scrollRequestUtteranceID,
            expandedUtteranceIDs: $expandedUtteranceIDs,
            visibleUtteranceIDs: $visibleUtteranceIDs,
            hasUnreadUtterance: $hasUnreadUtterance,
            normalizedTranscriptCache: $normalizedTranscriptCache,
            searchText: $searchText,
            selectedLabelFilter: $selectedLabelFilter,
            selectedSpeakerFilter: $selectedSpeakerFilter,
            searchFieldFocused: $searchFieldFocused,
            playbackAvailability: playbackAvailability(for:),
            reevaluateAvailability: reevaluateAvailability(for:),
            onToggleExpansion: toggleExpansion,
            onRenameSpeaker: { u in
                editingSpeakerStored = u.speakerID
                editingSpeakerName = recorder
                    .speakerDisplayName(forStored: u.speakerID) ?? ""
            },
            onPromoteNewSpeaker: { u in
                Task { _ = await recorder.promoteUtteranceToNewSpeaker(utteranceID: u.id) }
            },
            onCorrectSpeaker: { u, target in
                Task {
                    _ = await recorder.correctUtteranceSpeaker(
                        utteranceID: u.id,
                        to: target
                    )
                }
            },
            onCorrectMismatch: { u in
                // Long-press on the orange mismatch glyph accepts
                // the cumulative timeline's verdict for this row.
                // Recompute the dominant speaker on demand (it's
                // ~256 samples × a handful of segments, well under
                // a millisecond) so we always read the freshest
                // timeline state instead of caching it alongside
                // the mismatch flag. Falls back to a no-op when
                // the timeline has no overlap with the row or the
                // verdict no longer disagrees (the glyph went
                // stale between render and tap).
                let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                    recorder.diarizationTimeline,
                    from: u.start,
                    to: u.end,
                    fallback: u.speakerID
                )
                guard dominant != u.speakerID else { return }
                recorder.reassignSpeaker(utteranceID: u.id, to: dominant)
            },
            onEditTranscript: { u in
                // Editing is allowed in both file mode (full
                // pipeline re-run) and mic mode (text-only
                // re-run inheriting time + acoustic from the
                // parent). The only blocker is an active
                // recording / analysis pass — gated by `phase`.
                guard !recorder.isRecording, !recorder.isAnalyzing else { return }
                editingUtterance = u
            },
            refreshSearchCache: refreshSearchCache
        )
    }

}

/// Hosts the find-and-replace sheet. Lifted into a dedicated
/// `ViewModifier` because chaining a fourth `.sheet` directly onto
/// `ContentView.mainBody` pushed the Swift type-checker past its
/// "type-check in reasonable time" budget — extracting the sheet
/// gives the type-checker a discrete boundary and the chain stays
/// inferrable.
private struct SearchReplaceSheetPresenter: ViewModifier {
    @Binding var isPresented: Bool
    let recorder: RecordingController

    func body(content: Content) -> some View {
        content.sheet(isPresented: $isPresented) {
            SearchReplaceSheet(
                recorder: recorder,
                onDismiss: { isPresented = false }
            )
        }
    }
}

#Preview {
    ContentView()
}
