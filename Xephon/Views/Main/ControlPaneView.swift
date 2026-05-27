import SwiftUI
import Audio
import Fusion

/// Left third of the main split: input picker, record / open
/// buttons, level meter (during capture), status line, error
/// surface, then a 4-page TabView of cards.
///
/// Pages: 1) Settings + Pipeline. 2) Read-only affect output
/// (Summary, Statistics, SER aggregate, Fusion legend). 3) Diarizer
/// cluster + speaker-behavior cards. 4) Summarizer configuration.
struct ControlPaneView: View {
    let recorder: RecordingController
    let filterModel: TranscriptFilterModel
    let fileCoord: SessionFileCoordinator
    @Binding var selectedUtteranceID: UUID?
    @Binding var scrollRequestUtteranceID: UUID?
    @Binding var showingDiscardConfirm: Bool

    var body: some View {
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
                LevelMeterView(channelLevels: recorder.inputChannelLevels)
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
                settingsPage
                summaryPage
                speakerAnalysisPage
                keywordsPage
                summarizerPage
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

    // MARK: - Tab pages

    @ViewBuilder
    private var settingsPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                SettingsCard(recorder: recorder)
                PipelineCard(recorder: recorder)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
    }

    @ViewBuilder
    private var summaryPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            let summary = filterModel.displayedSummary(in: recorder)
            VStack(spacing: 16) {
                SummaryCard(
                    summary: summary,
                    totalDuration: summary.totalDuration
                )
                StatisticsCard(summary: summary)
                SERAggregateCard(
                    recorder: recorder,
                    focusedUtteranceID: selectedUtteranceID,
                    onTapUtterance: { id in
                        selectedUtteranceID = id
                        scrollRequestUtteranceID = id
                    }
                )
                FusionLegendCard(recorder: recorder)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
    }

    @ViewBuilder
    private var speakerAnalysisPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                SpeakerRosterCard(
                    recorder: recorder,
                    cluster: recorder.speakerCluster,
                    highlightedSpeakerID: focusedUtteranceSpeakerID,
                    linkedSpeakerIDs: linkedSpeakerIDs
                )
                SpeakerClusterCard(
                    cluster: recorder.speakerCluster,
                    highlightedSpeakerID: focusedUtteranceSpeakerID,
                    focusedEmbedding: focusedUtteranceEmbedding,
                    onTapNode: { speakerID, segmentID, embedding in
                        // Prefer the exact id pin — every utterance
                        // captured under this app version has its
                        // emitting observation's segmentId stashed
                        // at finalize time, so this is the right
                        // utterance by construction. The embedding
                        // fallback only kicks in for legacy sessions
                        // or centroid taps.
                        if let sid = segmentID,
                           let target = recorder.utterance(forSegmentID: sid) {
                            selectedUtteranceID = target.id
                            scrollRequestUtteranceID = target.id
                            return
                        }
                        guard let target = recorder.nearestUtterance(
                            toEmbedding: embedding,
                            speakerID: speakerID
                        ) else { return }
                        selectedUtteranceID = target.id
                        scrollRequestUtteranceID = target.id
                    },
                    linkedObservationIDs: Set(
                        recorder.utteranceObservationSegmentIDs.values
                    ),
                    linkedSpeakerIDs: linkedSpeakerIDs
                )
                SpeakerHeatmapCard(
                    cluster: recorder.speakerCluster,
                    highlightedSpeakerID: focusedUtteranceSpeakerID,
                    linkedSpeakerIDs: linkedSpeakerIDs
                )
                SpeakerBehaviorCard(
                    utterances: recorder.utterances
                )
                TurnTakingCard(
                    utterances: recorder.utterances
                )
                AffectiveSynchronyCard(
                    utterances: recorder.utterances
                )
                InfluenceContagionCard(
                    utterances: recorder.utterances
                )
                AccommodationCohesionCard(
                    utterances: recorder.utterances
                )
                ReactivityCard(
                    utterances: recorder.utterances
                )
                SynchronyArcCard(
                    utterances: recorder.utterances
                )
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
    }

    @ViewBuilder
    private var keywordsPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                KeywordsCard(store: recorder.keywords)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
    }

    @ViewBuilder
    private var summarizerPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                SummarizerCard(recorder: recorder)
                ModelsCard(recorder: recorder)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
    }

    // MARK: - Header pieces

    @ViewBuilder
    private var inputPicker: some View {
        // Always render, even when the inputs list is empty (file
        // mode, or before the first refresh) or contains only the
        // built-in mic. Keeping the picker visible keeps the
        // toolbar layout stable across state transitions and gives
        // the user a permanent at-a-glance indicator of which input
        // would be used if recording started now.
        // Picker label and menu check binds to `effectiveInputUID`
        // (user's pick falling back to built-in), NOT `currentInputUID`
        // (the OS's live currentRoute, which on iPadOS 26 flickers
        // between built-in and USB during recording even when the
        // engine's input bind is stable). Without this, the label
        // would silently jump to "USB Mic" mid-session and read as
        // "the recording switched to USB" even though it didn't.
        let effectiveUID = recorder.effectiveInputUID
        Menu {
            if recorder.availableInputs.isEmpty {
                Text(String(localized: "input.default"))
            } else {
                ForEach(recorder.availableInputs) { input in
                    Button {
                        Task { await recorder.selectInput(uid: input.uid) }
                    } label: {
                        if input.uid == effectiveUID {
                            Label(input.displayName, systemImage: "checkmark")
                        } else {
                            Text(input.displayName)
                        }
                    }
                }
            }
        } label: {
            let current = recorder.availableInputs.first(where: { $0.uid == effectiveUID })
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
        // Force a fresh enumeration whenever the user reaches for the
        // picker. Route-change notifications mostly cover plug/unplug
        // events but aren't 100% reliable on iPadOS — we've observed
        // cases where AVAudioSession.routeChangeNotification doesn't
        // fire for USB-C audio devices, especially when the session
        // is inactive between recordings. Refreshing on tap guarantees
        // the menu's contents reflect what's actually connected the
        // moment the user opens it.
        .simultaneousGesture(
            TapGesture().onEnded {
                Task { @MainActor in await recorder.refreshInputs() }
            }
        )
    }

    /// True when there's nothing actionable for the input picker:
    /// a session is in flight or the source is a file (mic isn't
    /// used). We deliberately don't disable on `availableInputs.count
    /// <= 1` anymore — that case (only the built-in mic visible)
    /// is exactly when the user would want to tap the picker after
    /// plugging in a USB mic, and the tap drives the refresh that
    /// makes the new device appear. Leaving it tappable means the
    /// menu shows just one entry briefly, the refresh runs, and the
    /// USB device appears on the next render.
    private var inputPickerDisabled: Bool {
        if recorder.isRecording { return true }
        if recorder.isAnalyzing { return true }
        if case .file = recorder.sourceMode { return true }
        return false
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
            fileCoord.presentAudioPicker(recorder: recorder)
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

    // MARK: - Derived speaker focus

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

    /// Speaker ids referenced by at least one utterance in the live
    /// list. Fed to the roster + heatmap cards as the seed set
    /// their "Linked only" toggles filter against — same pattern
    /// the cluster scatter card uses for observation ids.
    private var linkedSpeakerIDs: Set<String> {
        Set(recorder.utterances.map(\.speakerID))
    }
}
