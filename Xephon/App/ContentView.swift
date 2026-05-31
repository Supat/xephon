import SwiftUI
import Audio
import Diarization
import Fusion
import SERText
import Summarizer
import XephonLogging
import XephonUtilities

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
    /// Owns every session-file UI surface: the shared
    /// `.fileImporter` mode + flag, the Save Session export panel,
    /// the discard-before-import alert, the JSON-export share URL,
    /// and the I/O error string. Wired into the view tree via
    /// `SessionFileBridge`.
    @State private var fileCoord = SessionFileCoordinator()
    @State private var showingDiscardConfirm: Bool = false
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
    /// All transcript-filter state (search text, label / speaker
    /// chips, mismatch toggle) plus the normalized-transcript cache
    /// and filter / mismatch memos live on this `@Observable`
    /// sibling. ContentView still owns the @FocusState because the
    /// ⌘F menu command writes from here, but the value-typed knobs
    /// and their derived slices route through the model.
    @State private var filterModel = TranscriptFilterModel()
    /// Keyboard focus for the search field. Driven by ⌘F (sets it
    /// true) and Esc (sets it false). `@FocusState` is the only
    /// mechanism that programmatically moves focus into a TextField.
    @FocusState private var searchFieldFocused: Bool
    /// IDs of utterances whose detail panel is expanded. Toggled by
    /// long-press on the row or by pressing Space while the row is
    /// the selected list item. Kept here (not on the row) so a
    /// rebuild of the row doesn't drop the expansion state.
    @State private var expandedUtteranceIDs: Set<UUID> = []
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
    /// Owns the three LLM-adjacent sheets' presentation flags
    /// (summary, review, search-replace) and the two in-flight
    /// `Task` handles that the dismiss / scenePhase-background
    /// paths cancel. Wired into the view tree via `LLMSheetBridge`.
    @State private var llmCoord = LLMSheetCoordinator()
    /// App-level file pickup. Every card that needs `.fileImporter`
    /// or `.fileExporter` routes through this rather than attaching
    /// its own modifier — see `FilePickerCoordinator`'s doc comment
    /// for why one pair of modifiers at the app root sidesteps the
    /// SwiftUI multi-modifier presentation hazard.
    @State private var filePicker = FilePickerCoordinator()

    var body: some View {
        if !recorder.modelsReady {
            SetupView(controller: recorder)
        } else {
            mainBody
        }
    }

    private var mainBody: some View {
        // Local @Bindable shadow so the centralized file picker
        // modifiers below can bind to `$filePicker.is*Presented`
        // with proper two-way semantics. @State alone doesn't
        // expose `$` syntax for @Observable class properties.
        @Bindable var filePicker = filePicker
        return NavigationStack {
            VStack(spacing: 0) {
                if !recorder.pipelineDiagnostics.isEmpty {
                    PipelineDiagnosticsBanner(messages: recorder.pipelineDiagnostics)
                }
                GeometryReader { geo in
                    HStack(spacing: 0) {
                        ControlPaneView(
                            recorder: recorder,
                            filterModel: filterModel,
                            fileCoord: fileCoord,
                            filePicker: filePicker,
                            selectedUtteranceID: $selectedUtteranceID,
                            scrollRequestUtteranceID: $scrollRequestUtteranceID,
                            showingDiscardConfirm: $showingDiscardConfirm
                        )
                        .frame(width: geo.size.width / 3)
                        Divider()
                        TranscriptPaneView(
                            recorder: recorder,
                            filterModel: filterModel,
                            selectedUtteranceID: $selectedUtteranceID,
                            scrollRequestUtteranceID: $scrollRequestUtteranceID,
                            expandedUtteranceIDs: $expandedUtteranceIDs,
                            visibleUtteranceIDs: $visibleUtteranceIDs,
                            hasUnreadUtterance: $hasUnreadUtterance,
                            searchFieldFocused: $searchFieldFocused,
                            onRenameSpeaker: { u in
                                editingSpeakerStored = u.speakerID
                                editingSpeakerName = recorder
                                    .speakerDisplayName(forStored: u.speakerID) ?? ""
                            },
                            onEditTranscript: { u in
                                // Editing is allowed in both file
                                // mode (full pipeline re-run) and
                                // mic mode (text-only re-run
                                // inheriting time + acoustic from
                                // the parent). The only blocker is
                                // an active recording / analysis
                                // pass — gated by `phase`.
                                guard !recorder.isRecording, !recorder.isAnalyzing else { return }
                                editingUtterance = u
                            }
                        )
                        .frame(maxWidth: .infinity)
                    }
                }
            }
            .navigationTitle("Xephon")
            .toolbarTitleDisplayMode(.inline)
            .toolbar {
                MainToolbar(
                    recorder: recorder,
                    llmCoord: llmCoord,
                    fileCoord: fileCoord
                )
            }
            .modifier(LLMSheetBridge(recorder: recorder, coord: llmCoord))
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
                // Acoustic SER actors swap their ORT session between
                // CoreML EP and CPU across the foreground/background
                // boundary. `.inactive` (notification center pulled,
                // system alert) is treated as background-ish — iOS
                // restricts GPU work in that state too, and the swap
                // is cheap enough that the conservative call is
                // fine.
                let inBackground = newPhase != .active
                Task { await recorder.setBackgroundMode(inBackground) }
                guard newPhase == .background else { return }
                llmCoord.cancelInflightTasks()
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
                filterModel.resetForNewSession()
                selectedUtteranceID = nil
                scrollRequestUtteranceID = nil
                hasUnreadUtterance = false
            }
            .modifier(SessionFileBridge(
                recorder: recorder,
                coord: fileCoord,
                filePicker: filePicker,
                menuCommands: menuCommands
            ))
            // Centralized file pickers. Every feature that wants to
            // present an import or export goes through
            // `filePicker.presentImport` / `presentExport` rather
            // than attaching its own modifier. Two modifiers,
            // attached at the navigation root, drive the actual
            // system pickers from the coordinator's flags.
            //
            // Bindings target the explicit `is*Presented` bools
            // (not a derived `request != nil` closure) so SwiftUI's
            // standard two-way dismissal flow doesn't race with
            // request cleanup. `finishImport` / `finishExport`
            // capture the request first, then clear, then dispatch
            // — see the coordinator's doc-comments for the race
            // that prompted this split.
            .fileImporter(
                isPresented: $filePicker.isImporterPresented,
                allowedContentTypes: filePicker.importRequest?.allowedTypes ?? []
            ) { result in
                filePicker.finishImport(result)
            }
            .fileExporter(
                isPresented: $filePicker.isExporterPresented,
                document: filePicker.exportRequest?.document,
                contentType: filePicker.exportRequest?.contentType ?? .data,
                defaultFilename: filePicker.exportRequest?.defaultFilename ?? "export"
            ) { result in
                filePicker.finishExport(result)
            }
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
                    onTranscribeRange: { start, end in
                        await recorder.transcribeRange(start: start, end: end)
                    },
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

}

#Preview {
    ContentView()
}
