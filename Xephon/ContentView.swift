import SwiftUI
import UniformTypeIdentifiers
import Audio
import Export
import Fusion
import SERText
import XephonLogging

struct ContentView: View {
    @Environment(MenuCommands.self) private var menuCommands
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
                    Button {
                        Task {
                            if let url = await recorder.exportJSON() {
                                shareURL = url
                            }
                        }
                    } label: {
                        Label(String(localized: "export.json"), systemImage: "square.and.arrow.up")
                    }
                    .disabled(
                        recorder.utterances.isEmpty
                            || recorder.isRecording
                            || recorder.isAnalyzing
                    )
                }
            }
            .sheet(item: $shareURL) { url in
                ShareSheet(items: [url])
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
                do {
                    let doc = try recorder.makeSessionDocument()
                    pendingSaveDocument = SessionFileDocument(session: doc)
                    showingSaveSession = true
                } catch {
                    sessionIOError = String(describing: error)
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
        }
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

            ScrollView(.vertical, showsIndicators: true) {
                VStack(spacing: 16) {
                    SettingsCard(
                        speechBoostToggle: { speechBoostToggle },
                        textSERPicker: { textSERPicker }
                    )

                    PipelineCard(recorder: recorder)

                    SummaryCard(
                        summary: displayedSummary,
                        totalDuration: displayedSummary.totalDuration
                    )

                    StatisticsCard(summary: displayedSummary)
                }
                .frame(maxWidth: .infinity)
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
                if filteredIndexedUtterances.isEmpty {
                    noMatchesView
                } else {
                    transcriptList
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
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
                    HStack(spacing: 8) {
                        speakerChip(
                            text: String(localized: "filter.speaker.all"),
                            tint: .secondary,
                            isSelected: selectedSpeakerFilter == nil
                        ) {
                            selectedSpeakerFilter = nil
                        }
                        ForEach(speakers, id: \.self) { id in
                            let label = formatSpeakerLabel(id, multiSpeaker: true)
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
                .background(
                    tint.opacity(isSelected ? 0.25 : 0.08),
                    in: Capsule()
                )
                .overlay(
                    Capsule().strokeBorder(
                        tint.opacity(isSelected ? 0.6 : 0.2),
                        lineWidth: isSelected ? 1.0 : 0.5
                    )
                )
                .foregroundStyle(tint)
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
                .pickerStyle(.segmented)
                .labelsHidden()
            }
            .padding(.horizontal)
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
            String(localized: "settings.speechBoost"),
            isOn: Binding(
                get: { recorder.isSpeechBoostEnabled },
                set: { newValue in
                    Task { await recorder.setSpeechBoostEnabled(newValue) }
                }
            )
        )
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

    private var distinctSpeakerCount: Int {
        Set(recorder.utterances.map { $0.speakerID }).count
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
    private var isLastUtteranceVisible: Bool {
        guard let lastID = filteredIndexedUtterances.last?.u.id else { return true }
        return visibleUtteranceIDs.contains(lastID)
    }

    @ViewBuilder
    private var transcriptList: some View {
        ScrollViewReader { proxy in
            List(
                filteredIndexedUtterances,
                id: \.u.id,
                selection: $selectedUtteranceID
            ) { item in
                UtteranceRow(
                    number: item.idx + 1,
                    utterance: item.u,
                    isMultiSpeaker: distinctSpeakerCount > 1,
                    isExpanded: expandedUtteranceIDs.contains(item.u.id),
                    onToggleExpanded: { toggleExpansion(item.u.id) },
                    playback: playbackAvailability(for: item.u),
                    onPlaybackToggle: { recorder.togglePlayback(for: item.u) },
                    reevaluate: reevaluateAvailability(for: item.u),
                    onReevaluate: {
                        Task { await recorder.reevaluate(item.u) }
                    },
                    onRevert: { recorder.revertReevaluation(item.u) }
                )
                .id(item.u.id)
                .onAppear { visibleUtteranceIDs.insert(item.u.id) }
                .onDisappear { visibleUtteranceIDs.remove(item.u.id) }
                // Tag each row so List knows what UUID to write into the
                // `selectedUtteranceID` binding when it changes selection
                // (via tap, ↑/↓ arrow, or Home/End).
                .tag(item.u.id)
                // Replace the default pale-tint selection highlight with
                // a darker accent-tinted background. `.listRowBackground`
                // overrides the system selection paint on `.plain` lists.
                .listRowBackground(
                    selectedUtteranceID == item.u.id
                        ? Color.accentColor.opacity(0.28)
                        : Color.clear
                )
            }
            .listStyle(.plain)
            // Any tap on the list — row, dead-space between rows,
            // header gap, anywhere — clears the previously selected
            // utterance's focus highlight and dismisses any
            // keyboard focus on the search field. When the tap
            // lands on a row, the List's own tap-to-select handler
            // runs after this gesture and writes the row's id back
            // into `selectedUtteranceID`, so legitimate row
            // selection still works; only "stale" focus that the
            // user has moved past gets cleared.
            .simultaneousGesture(
                TapGesture().onEnded {
                    if searchFieldFocused { searchFieldFocused = false }
                    if selectedUtteranceID != nil { selectedUtteranceID = nil }
                }
            )
            // Keep the selected row scrolled into view when the user
            // arrow-keys past the visible window. The system handles this
            // for tap-driven selection automatically; for keyboard-driven
            // selection on a plain list it's not always automatic.
            .onChange(of: selectedUtteranceID) { _, newID in
                guard let newID else { return }
                // Scroll only if the row isn't already on screen, so we
                // don't fight the system's default keep-visible behavior
                // when the user is paging row-by-row near the edge.
                if !visibleUtteranceIDs.contains(newID) {
                    withAnimation(.easeOut(duration: 0.15)) {
                        proxy.scrollTo(newID, anchor: .center)
                    }
                }
            }
            // Esc clears the selection. Returning `.ignored` when nothing
            // is selected lets the system route Esc to its default
            // handlers (e.g. dismissing a sheet or popover) instead of
            // silently swallowing the key.
            .onKeyPress(.escape) {
                if selectedUtteranceID != nil {
                    selectedUtteranceID = nil
                    return .handled
                }
                return .ignored
            }
            // Space toggles the expanded detail panel on the currently
            // selected row. Long-press on a row does the same thing.
            // Ignored when nothing is selected so the keystroke can fall
            // through to other handlers.
            .onKeyPress(.space) {
                guard let id = selectedUtteranceID else { return .ignored }
                toggleExpansion(id)
                return .handled
            }
            // Observe the FILTERED last so adding an utterance that
            // doesn't match the active filter doesn't auto-scroll, and
            // adding one that does match scrolls the (possibly shorter)
            // filtered view to the right place.
            .onChange(of: filteredIndexedUtterances.last?.u.id) { oldLastID, newLastID in
                // Session reset cleared the array.
                guard let newLastID else {
                    hasUnreadUtterance = false
                    return
                }
                // "Was the user following along?" must be answered from
                // the PREVIOUS last utterance — by the time this closure
                // fires, the array's `last` is already the new entry,
                // whose row hasn't been laid out yet, so its id can't
                // be in `visibleUtteranceIDs`. Checking the new id was
                // the bug behind "auto-scroll never fires when the list
                // is at the bottom"; the old id, on the other hand,
                // is still in `visibleUtteranceIDs` for as long as that
                // row remains on screen.
                let wasFollowing = oldLastID.map { visibleUtteranceIDs.contains($0) } ?? false
                if oldLastID == nil || wasFollowing {
                    // First utterance of a session, OR user was at the
                    // bottom: jump to the new entry.
                    withAnimation(.easeOut(duration: 0.2)) {
                        proxy.scrollTo(newLastID, anchor: .bottom)
                    }
                    hasUnreadUtterance = false
                } else {
                    // User has scrolled the most-recent off-screen.
                    // Don't yank them; surface the capsule instead.
                    hasUnreadUtterance = true
                }
            }
            .onChange(of: recorder.utterances.isEmpty) { _, isEmpty in
                if isEmpty {
                    hasUnreadUtterance = false
                    selectedUtteranceID = nil
                    normalizedTranscriptCache.removeAll(keepingCapacity: true)
                    // New session (mic record or file analysis) just
                    // cleared the utterance list — reset the filter
                    // controls so the empty list isn't shown through
                    // a stale filter the user no longer remembers
                    // setting. Both filter chips and the search field
                    // return to "All" / blank.
                    searchText = ""
                    selectedLabelFilter = nil
                    selectedSpeakerFilter = nil
                }
            }
            .onChange(of: recorder.utterances.count, initial: true) { _, _ in
                refreshSearchCache()
            }
            .onChange(of: isLastUtteranceVisible) { _, visible in
                // User scrolled (or list re-laid out) to bring the most
                // recent utterance into view → no longer "unread".
                if visible { hasUnreadUtterance = false }
            }
            .overlay(alignment: .bottom) {
                if hasUnreadUtterance {
                    NewUtteranceCapsule {
                        guard let lastID = filteredIndexedUtterances.last?.u.id else { return }
                        withAnimation(.easeOut(duration: 0.25)) {
                            proxy.scrollTo(lastID, anchor: .bottom)
                        }
                        hasUnreadUtterance = false
                    }
                    .padding(.bottom, 16)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }
            }
            .animation(.easeOut(duration: 0.2), value: hasUnreadUtterance)
        }
    }

}

/// Fingerprint of the inputs that affect `filteredIndexedUtterances`
/// and `displayedSummary`. `Equatable` so the memo can early-exit on
/// no-change renders.
private struct FilterDepsKey: Equatable {
    /// Already-normalized search query, so we don't re-tokenize the
    /// query string on every change-check.
    let normalizedQuery: String
    let labelFilter: String?
    let speakerFilter: String?
    /// Utterance count handles appends (live streaming) and most
    /// resets. Paired with `utterancesVersion` for in-place mutations
    /// (re-evaluate) where the count doesn't change but the content
    /// did. Both are O(1) reads off the controller.
    let utteranceCount: Int
    let utterancesVersion: Int
}

/// Reference-typed memo for the filter + summary derivation in
/// ContentView. Held via `@State` so it survives view-body
/// re-evaluations; mutating its stored properties does NOT
/// re-trigger the body (which is what we want — the memo is read
/// during the current render pass).
@MainActor
private final class FilterMemo {
    var lastKey: FilterDepsKey?
    var results: [(idx: Int, u: UtteranceEstimate)] = []
    var summary: ConversationSummary = ConversationSummary()
}


#Preview {
    ContentView()
}
