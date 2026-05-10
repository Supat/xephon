import SwiftUI
import UniformTypeIdentifiers
import Audio
import Fusion
import SERText
import XephonLogging

struct ContentView: View {
    @Environment(MenuCommands.self) private var menuCommands
    @State private var recorder = RecordingController()
    @State private var shareURL: URL?
    @State private var showingDiscardConfirm: Bool = false
    @State private var showingFilePicker: Bool = false
    @State private var pendingFileURL: URL?
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
    /// Keyboard focus for the search field. Driven by ⌘F (sets it
    /// true) and Esc (sets it false). `@FocusState` is the only
    /// mechanism that programmatically moves focus into a TextField.
    @FocusState private var searchFieldFocused: Bool

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
            // Edit → Find (⌘F): move keyboard focus into the search
            // field. Setting `@FocusState` to true is the only way to
            // programmatically focus a SwiftUI TextField.
            .onChange(of: menuCommands.findToken) { _, _ in
                searchFieldFocused = true
            }
            .fileImporter(
                isPresented: $showingFilePicker,
                allowedContentTypes: [.audio, .mp3, .wav, .mpeg4Audio, .aiff],
                allowsMultipleSelection: false
            ) { result in
                switch result {
                case .success(let urls):
                    guard let url = urls.first else { return }
                    pendingFileURL = url
                    if !recorder.utterances.isEmpty {
                        showingFileDiscardConfirm = true
                    } else {
                        showingPacingDialog = true
                    }
                case .failure(let error):
                    AppLog.app.error("file picker: \(String(describing: error), privacy: .public)")
                }
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
            .alert(
                String(localized: "record.discardConfirm.title"),
                isPresented: $showingFileDiscardConfirm
            ) {
                Button(String(localized: "record.discardConfirm.confirm"), role: .destructive) {
                    // Discard accepted — proceed to pacing choice. The
                    // pendingFileURL is preserved across alerts.
                    showingPacingDialog = true
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {
                    pendingFileURL = nil
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
                        Task { await recorder.startFromFile(url, realTimePacing: true) }
                    }
                    pendingFileURL = nil
                }
                Button(String(localized: "pacing.realtime.audio")) {
                    if let url = pendingFileURL {
                        Task { await recorder.startFromFile(url, realTimePacing: true, audioOutputEnabled: true) }
                    }
                    pendingFileURL = nil
                }
                Button(String(localized: "pacing.fast")) {
                    if let url = pendingFileURL {
                        Task { await recorder.startFromFile(url, realTimePacing: false) }
                    }
                    pendingFileURL = nil
                }
                Button(String(localized: "record.discardConfirm.cancel"), role: .cancel) {
                    pendingFileURL = nil
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
                        summary: recorder.conversationSummary,
                        totalDuration: recorder.conversationSummary.totalDuration
                    )

                    StatisticsCard(summary: recorder.conversationSummary)
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
        if !recorder.availableInputs.isEmpty {
            Menu {
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
            .disabled(recorder.isRecording)
        }
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
                recorder.isRecording
                    ? String(localized: "record.stop")
                    : String(localized: "record.start"),
                systemImage: recorder.isRecording ? "stop.circle.fill" : "mic.circle.fill"
            )
            .font(.title3)
        }
        .buttonStyle(.borderedProminent)
        .tint(recorder.isRecording ? .red : .accentColor)
        .disabled(recorder.isAnalyzing)
    }

    private var openFileButton: some View {
        Button {
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
                Text(String(
                    format: String(localized: "record.status.format"),
                    formatClock(recorder.elapsedSeconds),
                    formatCount(recorder.samplesCaptured)
                ))
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

    /// Distinct top labels seen so far this session, used to populate
    /// the label-filter menu. Sorted alphabetically for stable order.
    private var availableLabels: [String] {
        Set(recorder.utterances.compactMap { $0.fusedTopLabel }).sorted()
    }

    /// `(originalIndex, utterance)` pairs surviving both filters. The
    /// original index is preserved so each row's "#N" badge keeps the
    /// utterance's stable session number even when the list is
    /// filtered (a row labeled #15 always points at the 15th utterance
    /// of the recording).
    private var filteredIndexedUtterances: [(idx: Int, u: UtteranceEstimate)] {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        return recorder.utterances.enumerated().compactMap {
            (idx, u) -> (idx: Int, u: UtteranceEstimate)? in
            if let filterLabel = selectedLabelFilter,
               u.fusedTopLabel != filterLabel {
                return nil
            }
            if !trimmed.isEmpty,
               !u.transcript.localizedCaseInsensitiveContains(trimmed) {
                return nil
            }
            return (idx, u)
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
                    isMultiSpeaker: distinctSpeakerCount > 1
                )
                .id(item.u.id)
                .onAppear { visibleUtteranceIDs.insert(item.u.id) }
                .onDisappear { visibleUtteranceIDs.remove(item.u.id) }
                // Tag each row so List knows what UUID to write into the
                // `selectedUtteranceID` binding when it changes selection
                // (via tap, ↑/↓ arrow, or Home/End).
                .tag(item.u.id)
            }
            .listStyle(.plain)
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
                }
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


#Preview {
    ContentView()
}
