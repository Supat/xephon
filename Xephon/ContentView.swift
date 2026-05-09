import SwiftUI
import UniformTypeIdentifiers
import Audio
import Fusion
import SERText
import XephonLogging

private extension SwitchingTextSER.Backend {
    /// Short tag rendered on each utterance row's text-backend badge.
    var badgeLabel: String {
        switch self {
        case .deberta:          return "DeBERTa"
        case .foundationModels: return "Apple FM"
        }
    }
}

/// Format an audio-time offset (seconds) as a clock string. Adapts to length:
///   < 1 min  → `M:SS.s`     (e.g. `0:05.2`)
///   < 1 hour → `MM:SS.s`    (e.g. `12:34.5`)
///   ≥ 1 hour → `H:MM:SS`    (e.g. `1:23:45`)
/// Hour-or-greater drops the fractional part to keep the row compact;
/// sub-second precision rarely matters at that scale.
func formatClock(_ seconds: TimeInterval) -> String {
    let clamped = max(0, seconds)
    let total = Int(clamped)
    let h = total / 3600
    let m = (total % 3600) / 60
    let s = total % 60
    if h > 0 {
        return String(format: "%d:%02d:%02d", h, m, s)
    }
    let frac = clamped - Double(total)
    return String(format: "%d:%02d.%d", m, s, Int(frac * 10))
}

/// SI suffixes for raw sample counts. 16 kHz audio crosses millions in
/// minutes, so plain integers fill the status line — `123K` / `1.23M` is
/// easier to scan. Decimal precision tapers as magnitude grows.
func formatCount(_ n: Int) -> String {
    let v = Double(n)
    if abs(v) < 1_000          { return "\(n)" }
    if abs(v) < 1_000_000      { return formatSI(v / 1_000,         "K") }
    if abs(v) < 1_000_000_000  { return formatSI(v / 1_000_000,     "M") }
    return                       formatSI(v / 1_000_000_000, "G")
}

private func formatSI(_ v: Double, _ suffix: String) -> String {
    if v >= 100 { return String(format: "%.0f%@", v, suffix) }
    if v >= 10  { return String(format: "%.1f%@", v, suffix) }
    return        String(format: "%.2f%@", v, suffix)
}

/// Render a stored speaker ID for display. The diarizer-tracker writes
/// `S01`, `S02`, … internally; the row label uses an `M` prefix
/// ("multi") when there are two or more distinct speakers in the
/// session, leaving `S` ("single") for the simple case. Number is
/// preserved either way.
func formatSpeakerLabel(_ stored: String, multiSpeaker: Bool) -> String {
    let prefix = multiSpeaker ? "M" : "S"
    if let trail = stored.dropFirst().first, trail.isNumber {
        return "\(prefix)\(stored.dropFirst())"
    }
    return stored
}

/// Per-speaker tint pulled from a small palette indexed by the speaker
/// number embedded in the stored ID (`S01` → 1). The palette is meant
/// to read as a row of colored chips at a glance; ordering matches the
/// rough warmth gradient (cool → warm → neutral) so consecutive
/// speakers stay distinguishable.
func speakerTint(for stored: String) -> Color {
    let palette: [Color] = [
        .blue, .orange, .green, .pink, .purple,
        .teal, .indigo, .brown, .mint, .red,
    ]
    let digits = stored.drop(while: { !$0.isNumber })
    let n = Int(digits) ?? 1
    let idx = max(0, n - 1) % palette.count
    return palette[idx]
}

/// Color tint for a fused emotion label. Tracks the conventional Plutchik
/// wheel where it overlaps; falls back to grey for unknown/neutral labels.
/// File-scope so both `UtteranceRow` and `SummaryCard` reuse the same
/// color mapping. Named `emotionTint` (not `color`) to avoid clashing
/// with implicit `color`-named members SwiftUI exposes inside View
/// closures.
func emotionTint(for raw: String) -> Color {
    switch raw.lowercased() {
    case "happy", "joy", "joyful":               return .yellow
    case "sad", "sadness":                       return .blue
    case "angry", "anger":                       return .red
    case "fear", "fearful", "afraid":            return .purple
    case "disgust", "disgusted":                 return Color(red: 0.45, green: 0.55, blue: 0.20)
    case "surprise", "surprised":                return .orange
    case "trust":                                return .green
    case "anticipation":                         return Color(red: 0.95, green: 0.55, blue: 0.10)
    case "neutral", "calm":                      return .gray
    default:                                     return Color.secondary
    }
}

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

            Spacer(minLength: 0)
        }
        .padding()
        .frame(maxHeight: .infinity)
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

private struct UtteranceRow: View {
    let number: Int
    let utterance: UtteranceEstimate
    let isMultiSpeaker: Bool

    // V/A from fusion are in [0, 1] with 0.5 = neutral. Re-center to [-1, +1]
    // so positive vs negative read naturally and 0 maps to "neutral grey".
    private static let neutralEpsilon: Float = 0.05

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text("#\(number)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                    Text(formatSpeakerLabel(utterance.speakerID, multiSpeaker: isMultiSpeaker))
                        .font(.caption.bold())
                        .foregroundStyle(speakerTint(for: utterance.speakerID))
                    if utterance.speechBoost == true {
                        Label("Boost", systemImage: "waveform.badge.plus")
                            .labelStyle(.titleAndIcon)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .overlay(
                                Capsule().strokeBorder(.orange.opacity(0.5), lineWidth: 0.5)
                            )
                            .foregroundStyle(.orange)
                    }
                }
                Text(utterance.transcript.isEmpty ? "—" : utterance.transcript)
                    .font(.body)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 4) {
                HStack(spacing: 6) {
                    if let backendBadge {
                        Text(backendBadge)
                            .font(.caption2)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .overlay(
                                Capsule().strokeBorder(.secondary.opacity(0.4), lineWidth: 0.5)
                            )
                            .foregroundStyle(.secondary)
                    }
                    Text("\(formatClock(utterance.start))–\(formatClock(utterance.end))")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 8) {
                    if let label = utterance.fusedTopLabel {
                        let tint = emotionTint(for: label)
                        Text(label.capitalized(with: Locale(identifier: "en_US")))
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(tint.opacity(0.18), in: Capsule())
                            .foregroundStyle(tint)
                    }
                    if let v = utterance.fusedValence {
                        vaLabel("V", value: v)
                    }
                    if let a = utterance.fusedArousal {
                        vaLabel("A", value: a)
                    }
                }
            }
        }
        .padding(.vertical, 4)
    }

    private var backendBadge: String? {
        guard let raw = utterance.textBackend,
              let backend = SwitchingTextSER.Backend(rawValue: raw) else { return nil }
        return backend.badgeLabel
    }

    @ViewBuilder
    private func vaLabel(_ axis: String, value: Float) -> some View {
        let centered = value * 2 - 1
        Text(String(format: "%@ %+.2f", axis, centered))
            .font(.caption.monospacedDigit())
            .foregroundStyle(color(for: centered))
    }

    private func color(for centered: Float) -> Color {
        if centered > Self.neutralEpsilon { return .green }
        if centered < -Self.neutralEpsilon { return .red }
        return .gray
    }
}

extension URL: @retroactive Identifiable {
    public var id: String { absoluteString }
}

// MARK: - Pipeline visualization

/// Settings card sitting above `PipelineCard`. Hosts the speech-boost
/// toggle (Capture's EQ) and the text-SER backend picker — controls that
/// configure how the pipeline runs but aren't part of the live stage
/// visualization itself.
private struct SettingsCard<SpeechBoost: View, TextSER: View>: View {
    @ViewBuilder let speechBoostToggle: () -> SpeechBoost
    @ViewBuilder let textSERPicker: () -> TextSER

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            speechBoostToggle()
            textSERPicker()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct PipelineCard: View {
    let recorder: RecordingController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(String(localized: "pipeline.header"))
                .font(.caption.bold())
                .foregroundStyle(.secondary)

            StageRow(
                icon: "mic.fill",
                name: String(localized: "pipeline.capture"),
                state: captureState,
                metric: captureMetric
            )
            VStack(alignment: .leading, spacing: 2) {
                StageRow(
                    icon: "waveform.and.mic",
                    name: String(localized: "pipeline.asr"),
                    state: asrState,
                    metric: asrMetric
                )
                // Always render the volatile-preview region, even when empty,
                // so the pipeline panel keeps a fixed height and doesn't
                // shift the rows below as text streams in. `reservesSpace`
                // pads the Text to its line limit regardless of content;
                // `.head` truncation keeps the most recent words visible
                // when the preview overflows the 3-line budget.
                Text(recorder.volatileText.isEmpty ? " " : "“\(recorder.volatileText)…”")
                    .font(.caption2.italic())
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 24)
                    .lineLimit(3, reservesSpace: true)
                    .truncationMode(.head)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            StageRow(
                icon: "waveform",
                name: String(localized: "pipeline.acousticSER"),
                state: perSegmentState(latency: recorder.lastAcousticDuration),
                metric: latencyMetric(recorder.lastAcousticDuration)
            )
            StageRow(
                icon: "text.bubble.fill",
                name: String(localized: "pipeline.textSER"),
                state: perSegmentState(latency: recorder.lastTextDuration),
                metric: latencyMetric(recorder.lastTextDuration)
            )
            StageRow(
                icon: "circle.hexagongrid.fill",
                name: String(localized: "pipeline.fusion"),
                state: fusionState,
                metric: Text(recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count) utts")
            )
            StageRow(
                icon: "square.and.arrow.up",
                name: String(localized: "pipeline.export"),
                state: recorder.lastExportAt == nil ? .idle : .ready,
                metric: exportMetric
            )
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    // MARK: derived state

    private var captureState: StageRow.State {
        recorder.isRecording ? .active(recorder.inputLevel) : .idle
    }
    /// Rolling buffer size, not session elapsed time. Once the trim
    /// kicks in (after ~60 s of audio = the diarization context window)
    /// this metric stops growing and oscillates near the cap, which is
    /// the more useful signal — it shows the buffer's actual memory
    /// footprint instead of restating the wall clock that's already
    /// visible in the status row above.
    private var captureMetric: Text {
        Text(recorder.isRecording
            ? "\(formatCount(recorder.bufferedSamples)) buf"
            : "—")
    }

    private var asrState: StageRow.State {
        if recorder.isRecording { return .pending }
        if !recorder.utterances.isEmpty { return .ready }
        return .idle
    }
    /// Real-time / mic mode shows the most recent finalize latency
    /// (wall clock from end-of-utterance to ASR-final-emitted). 200–800 ms
    /// is the typical range on M-class. Fast-pace mode falls through to
    /// the utterance count since audio time isn't wall-clock-aligned
    /// there and the latency number would be meaningless.
    /// The leading `backward.frame` glyph reads as "look back to the
    /// most recent finalize" — replaces the verbose "last:" text.
    private var asrMetric: Text {
        if let latency = recorder.lastASRFinalizeLatency {
            let ms = Int((latency * 1000).rounded())
            return Text("\(Image(systemName: "backward.frame")) \(ms) ms")
        }
        return Text(recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count)")
    }

    /// Per-segment stages (Acoustic SER, Text SER) flip to active while a
    /// segment is in flight, otherwise idle. Latency value is shown in the
    /// metric column, but the glyph state itself doesn't latch to .ready —
    /// otherwise the stage looks "done" forever even though it'll fire again
    /// on the next segment.
    private func perSegmentState(latency _: TimeInterval?) -> StageRow.State {
        recorder.inflightSegments > 0 ? .active(0) : .idle
    }

    private var fusionState: StageRow.State {
        if recorder.inflightSegments > 0 { return .active(0) }
        return recorder.utterances.isEmpty ? .idle : .ready
    }

    private func latencyMetric(_ value: TimeInterval?) -> Text {
        guard let value else { return Text("—") }
        if value >= 1 { return Text(String(format: "%.2f s", value)) }
        return Text(String(format: "%.0f ms", value * 1000))
    }

    private var exportMetric: Text {
        guard let date = recorder.lastExportAt else { return Text("—") }
        let interval = -date.timeIntervalSinceNow
        if interval < 60 { return Text(String(format: "%.0fs ago", interval)) }
        return Text(String(format: "%.0fm ago", interval / 60))
    }
}

/// Real-time conversation mood summary. Sits below `PipelineCard` and
/// updates incrementally as each utterance lands. Below
/// `ConversationSummary.calibrationThreshold` we suppress the V/A/D
/// numbers and show a "calibrating" placeholder — the means are noisy
/// when only one or two utterances exist and would mislead.
private struct SummaryCard: View {
    let summary: ConversationSummary
    let totalDuration: TimeInterval

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "summary.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if summary.utteranceCount > 0 {
                    Text(String(
                        format: String(localized: "summary.utterances"),
                        summary.utteranceCount,
                        formatClock(totalDuration)
                    ))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.tertiary)
                }
            }

            if summary.utteranceCount < ConversationSummary.calibrationThreshold {
                HStack(spacing: 6) {
                    Image(systemName: "chart.line.uptrend.xyaxis")
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                    Text(String(localized: "summary.calibrating"))
                        .font(.caption)
                        .foregroundStyle(.tertiary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                topLabelRow
                vadLines
                if summary.trajectory.count > 1 {
                    TrajectorySparkline(points: summary.trajectory)
                        .frame(height: 32)
                        .padding(.top, 2)
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private var topLabelRow: some View {
        if let label = summary.topLabel {
            let tint = emotionTint(for: label)
            Text(label.capitalized(with: Locale(identifier: "en_US")))
                .font(.body.bold())
                .padding(.horizontal, 8)
                .padding(.vertical, 3)
                .background(tint.opacity(0.18), in: Capsule())
                .foregroundStyle(tint)
        }
    }

    @ViewBuilder
    private var vadLines: some View {
        // Match the per-utterance row, which displays V and A only.
        // Dominance is unstable across modalities (text SER doesn't
        // produce it) and would mislead in a session-long aggregate.
        VStack(alignment: .leading, spacing: 2) {
            vaLine("V", mean: summary.meanValence, stdDev: summary.valenceStdDev)
            vaLine("A", mean: summary.meanArousal, stdDev: summary.arousalStdDev)
        }
    }

    @ViewBuilder
    private func vaLine(_ axis: String, mean: Float?, stdDev: Float?) -> some View {
        if let mean {
            let centered = mean * 2 - 1
            HStack(spacing: 6) {
                Text(axis)
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                    .frame(width: 12, alignment: .leading)
                Text(String(format: "%+.2f", centered))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(vadColor(centered: centered))
                if let stdDev {
                    Text(String(format: String(localized: "summary.dispersion"), stdDev))
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }
        }
    }

    private func vadColor(centered: Float) -> Color {
        let eps: Float = 0.05
        if centered >  eps { return .green }
        if centered < -eps { return .red }
        return .gray
    }
}

/// Per-label utterance count histogram. Sits below `SummaryCard` and
/// reads `ConversationSummary.labelCounts` (raw counts, not the
/// confidence-weighted scores `topLabel` uses). Sorted by count
/// descending so the dominant labels read top-down. Empty until the
/// first labeled utterance arrives.
private struct StatisticsCard: View {
    let summary: ConversationSummary

    private var sortedRows: [(label: String, count: Int)] {
        summary.labelCounts
            .map { (label: $0.key, count: $0.value) }
            // Tiebreak on the label string so the order is deterministic
            // for equal counts — otherwise dictionary iteration order
            // makes the panel jitter as new utterances arrive.
            .sorted { ($0.count, $1.label) > ($1.count, $0.label) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "statistics.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if summary.utteranceCount > 0 {
                    Text("\(summary.utteranceCount)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }

            if sortedRows.isEmpty {
                Text(String(localized: "statistics.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(sortedRows, id: \.label) { row in
                        StatisticsRow(
                            label: row.label,
                            count: row.count,
                            total: summary.utteranceCount
                        )
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

private struct StatisticsRow: View {
    let label: String
    let count: Int
    let total: Int

    var body: some View {
        let tint = emotionTint(for: label)
        let fraction: Double = total > 0 ? Double(count) / Double(total) : 0
        HStack(spacing: 8) {
            Text(label.capitalized(with: Locale(identifier: "en_US")))
                .font(.caption.monospaced())
                .foregroundStyle(tint)
                .frame(minWidth: 80, alignment: .leading)
            // Inline bar so the relative weight of each label is legible
            // at a glance — the count alone reads as a flat list.
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(tint.opacity(0.12))
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: geo.size.width * fraction)
                }
            }
            .frame(height: 6)
            Text("\(count)")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .frame(minWidth: 28, alignment: .trailing)
        }
    }
}

/// Sparkline for the bounded valence trajectory. Center line at 0.5
/// (neutral), points above are positive valence (green tint), below are
/// negative (red tint). Drawn with a single Path so it scales cheaply
/// even at the trajectory cap.
private struct TrajectorySparkline: View {
    let points: [ConversationSummary.TrajectoryPoint]

    var body: some View {
        GeometryReader { geo in
            let w = geo.size.width
            let h = geo.size.height
            let mid = h * 0.5
            let count = points.count

            ZStack {
                // Neutral center line.
                Path { p in
                    p.move(to: CGPoint(x: 0, y: mid))
                    p.addLine(to: CGPoint(x: w, y: mid))
                }
                .stroke(Color.secondary.opacity(0.25), style: .init(lineWidth: 0.5, dash: [2, 3]))

                // Valence trace.
                Path { p in
                    guard count > 1 else { return }
                    for (i, point) in points.enumerated() {
                        let x = CGFloat(i) / CGFloat(count - 1) * w
                        let y = h * (1 - CGFloat(point.valence))
                        if i == 0 { p.move(to: CGPoint(x: x, y: y)) }
                        else      { p.addLine(to: CGPoint(x: x, y: y)) }
                    }
                }
                .stroke(Color.accentColor, lineWidth: 1.2)
            }
        }
    }
}

private struct StageRow: View {
    enum State {
        case idle
        case pending
        case active(Float)   // 0...1 intensity
        case ready
    }

    let icon: String
    let name: String
    let state: State
    /// Right-aligned per-stage value. `Text` (not `String`) so callers
    /// can embed SF Symbols inline via `Text("\(Image(systemName:)) …")`.
    let metric: Text

    var body: some View {
        HStack(spacing: 8) {
            Image(systemName: icon)
                .font(.caption)
                .frame(width: 16)
                .foregroundStyle(iconColor)
            Text(name)
                .font(.caption)
                .lineLimit(1)
            Spacer(minLength: 4)
            metric
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .lineLimit(1)
            stateGlyph
                .font(.caption2)
                .frame(width: 14, alignment: .center)
        }
    }

    private var iconColor: Color {
        switch state {
        case .idle: return .secondary
        case .pending: return .blue
        case .active: return .green
        case .ready: return .accentColor
        }
    }

    @ViewBuilder
    private var stateGlyph: some View {
        switch state {
        case .idle:
            Image(systemName: "circle")
                .foregroundStyle(.tertiary)
        case .pending:
            Text("⋯")
                .foregroundStyle(.blue)
        case .active(let intensity):
            Image(systemName: "circle.fill")
                .foregroundStyle(.green.opacity(Double(0.4 + intensity * 0.6)))
        case .ready:
            Image(systemName: "checkmark.circle.fill")
                .foregroundStyle(.green)
        }
    }
}

private struct LevelMeterView: View {
    let level: Float
    private let segmentCount = 24

    var body: some View {
        HStack(spacing: 3) {
            ForEach(0..<segmentCount, id: \.self) { i in
                let threshold = Float(i) / Float(segmentCount - 1)
                let isLit = level >= threshold
                RoundedRectangle(cornerRadius: 2, style: .continuous)
                    .fill(isLit ? color(for: threshold) : color(for: threshold).opacity(0.15))
                    .frame(height: 14)
            }
        }
        .animation(.linear(duration: 0.05), value: level)
        .accessibilityElement()
        .accessibilityLabel("Microphone level")
        .accessibilityValue("\(Int(level * 100)) percent")
    }

    private func color(for ratio: Float) -> Color {
        if ratio < 0.65 { return .green }
        if ratio < 0.88 { return .yellow }
        return .red
    }
}

/// Floating "new utterance available" affordance shown over the bottom
/// of the transcript when a new entry has arrived but the user has
/// scrolled the list away from the latest row. Tapping scrolls back
/// to the most recent utterance. Hidden when the latest is in view.
private struct NewUtteranceCapsule: View {
    let action: () -> Void

    var body: some View {
        Button(action: action) {
            HStack(spacing: 6) {
                Image(systemName: "arrow.down")
                    .font(.caption.bold())
                Text(String(localized: "transcript.newUtterance"))
                    .font(.subheadline.weight(.semibold))
            }
            .foregroundStyle(.primary)
            .padding(.horizontal, 14)
            .padding(.vertical, 8)
            .background(.ultraThinMaterial, in: Capsule())
            .overlay(Capsule().strokeBorder(.tertiary, lineWidth: 0.5))
            .shadow(color: .black.opacity(0.15), radius: 6, y: 2)
        }
        .buttonStyle(.plain)
    }
}

/// Surfaces per-modality construction failures captured during pipeline
/// pre-warm. Without this banner, a model that fails to load (e.g. an
/// FP16 ONNX graph that ORT rejects) silently disappears from the
/// available SER backends — the user sees a degraded picker with no
/// hint as to why.
private struct PipelineDiagnosticsBanner: View {
    let messages: [String]

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: "exclamationmark.triangle.fill")
                .foregroundStyle(.yellow)
            VStack(alignment: .leading, spacing: 2) {
                Text("Some models didn't load")
                    .font(.caption.bold())
                ForEach(messages, id: \.self) { msg in
                    Text(msg)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                        .truncationMode(.middle)
                }
            }
            Spacer(minLength: 0)
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 8)
        .background(Color.yellow.opacity(0.12))
    }
}

private struct ShareSheet: UIViewControllerRepresentable {
    let items: [Any]
    func makeUIViewController(context: Context) -> UIActivityViewController {
        UIActivityViewController(activityItems: items, applicationActivities: nil)
    }
    func updateUIViewController(_ controller: UIActivityViewController, context: Context) {}
}

#Preview {
    ContentView()
}
