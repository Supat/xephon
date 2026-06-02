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
    let filePicker: FilePickerCoordinator
    /// Drives the per-section summary sheet presentation —
    /// the only LLM-adjacent sheet that originates from a
    /// card rather than from the top-of-window toolbar. The
    /// existing overall-summary / review / search-replace
    /// sheets are still raised by `LLMToolbar` in the main
    /// chrome, so this prop only carries the section
    /// presentation path through to `SectionsCard`.
    let llmCoord: LLMSheetCoordinator
    @Binding var selectedUtteranceID: UUID?
    @Binding var scrollRequestUtteranceID: UUID?
    @Binding var showingDiscardConfirm: Bool

    /// Index of the currently-visible TabView page. Bound to the
    /// TabView's `selection` purely so swipes trigger an
    /// `onChange` we can use to re-show the page-indicator dots.
    @State private var selectedTab: Int = 0
    /// Visibility of the system page-indicator. Flipped to
    /// `.always` for a short window after the user swipes, then
    /// back to `.never` so the dots fully disappear (not just
    /// their capsule background, which is all `.automatic` mode
    /// hides).
    @State private var dotsVisible: Bool = false
    /// Pending hide work. Cancelled and rescheduled every time
    /// the user swipes so a rapid sequence of pages keeps the
    /// dots up until they pause.
    @State private var dotsHideTask: Task<Void, Never>? = nil

    /// Measured height of the top header VStack (input picker,
    /// record/open, optional level meter, status, optional
    /// error). Pushed up from a `GeometryReader` in the
    /// header's background; consumed by the outer `.id(...)`
    /// so the page-controller-backed TabView is forced to
    /// rebuild when the header expands or contracts (e.g.
    /// LevelMeterView appearing on recording start, or an
    /// error banner showing up). Without this, the
    /// UIPageViewController inside SwiftUI's `.page`-style
    /// TabView keeps its prior internal layout and the page
    /// ScrollView's top edge slides under the expanded
    /// header, hiding the first few px of card content.
    @State private var headerHeight: CGFloat = 0

    private static let dotsHideDelayNanos: UInt64 = 1_500_000_000

    var body: some View {
        // Two-region layout: a fixed header that pins the controls
        // at the top (input picker, record/open, level meter,
        // status, error) plus a card region below that holds the
        // swipeable TabView. The header never scrolls off — the
        // user can always reach Start/Stop even with every card
        // expanded.
        //
        // The two regions live in separate VStacks so the outer
        // VStack has a clear intrinsic-vs-flex split: header is
        // intrinsic-sized, TabView region is `.frame(maxHeight:
        // .infinity)` and absorbs the slack. The outer VStack is
        // re-identified via `.id(geometry.size.height)` so a
        // rotation (portrait → landscape, or any size-class
        // change that shifts the column height) forces SwiftUI to
        // rebuild the TabView's internal UIPageViewController —
        // without this, the page content kept its pre-rotation
        // content offset and overlapped the header. `safeAreaInset`
        // was tried as an alternative but TabView's `.page` style
        // doesn't respect SwiftUI safe-area insets (the page
        // controller draws through the inset region), which made
        // the overlap permanent rather than rotation-conditional.
        GeometryReader { geo in
            VStack(spacing: 0) {
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
                        // Inline clear button on the trailing
                        // edge — only present while a message
                        // is showing, so the trailing column
                        // doesn't reserve dead space in the
                        // happy path. Setting the error to
                        // nil triggers the @Observable re-eval
                        // and this whole block disappears.
                        HStack(alignment: .top, spacing: 8) {
                            Text(error)
                                .font(.footnote)
                                .foregroundStyle(.red)
                                .multilineTextAlignment(.center)
                                .frame(maxWidth: .infinity)
                            Button {
                                recorder.errorMessage = nil
                            } label: {
                                Image(systemName: "xmark.circle.fill")
                                    .font(.footnote)
                                    .foregroundStyle(.secondary)
                            }
                            .buttonStyle(.plain)
                            .accessibilityLabel(Text(String(localized: "errorBanner.clear")))
                        }
                        .padding(.horizontal)
                    }
                }
                .padding()
                // Track the header's measured height directly via
                // `.onGeometryChange` so the outer `.id(...)` can
                // rebuild the page-controller-backed TabView when
                // the header grows or shrinks (level meter
                // appearing, error banner showing, status line
                // wrapping, level meter row count changing from
                // mono to stereo mid-recording). `.onGeometryChange`
                // (iOS 17+) fires on every layout pass for this
                // view, more reliably than a
                // `.background(GeometryReader)` + `PreferenceKey`
                // chain that can miss grandchild size changes.
                .onGeometryChange(for: CGFloat.self) { proxy in
                    proxy.size.height
                } action: { newHeight in
                    headerHeight = newHeight
                }

                // Card section split across six swipeable pages so
                // the left pane doesn't grow into a long single scroll
                // (the cluster + heatmap especially want vertical room
                // to render their data legibly). Page 1: session
                // controls — Settings + Pipeline. Page 2: read-only
                // affect output — Summary + Statistics. Page 3:
                // diarizer cluster + speaker-behavior cards. Page 4:
                // user-defined sections. Page 5: keywords. Page 6:
                // summarizer configuration.
                //
                // The selection binding exists only so swipes fire
                // `onChange` and we can re-show the page indicator.
                // The standard `.page(indexDisplayMode: .automatic)`
                // mode only hides the dots' capsule background, not
                // the dots themselves, so we flip the index display
                // mode between `.always` and `.never` ourselves —
                // still the system indicator, just with timed
                // visibility.
                TabView(selection: $selectedTab) {
                    settingsPage.tag(0)
                    summaryPage.tag(1)
                    speakerAnalysisPage.tag(2)
                    sectionsPage.tag(3)
                    keywordsPage.tag(4)
                    summarizerPage.tag(5)
                }
                .tabViewStyle(.page(indexDisplayMode: dotsVisible ? .always : .never))
                .indexViewStyle(.page(backgroundDisplayMode: .always))
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                // Let the TabView visually extend through the home-
                // indicator safe-area zone so the column reaches the
                // screen's bottom edge instead of stopping at the
                // safe-area top. The system page indicator positions
                // itself with the safe-area inset internally so it
                // doesn't disappear under the home indicator.
                .ignoresSafeArea(.container, edges: .bottom)
                // Swallow taps over the indicator-capsule area. The
                // system `UIPageControl` advances ±1 page on tap
                // depending on which half of the bar got touched —
                // since we're not exposing per-dot jump and tapping
                // an indicator dot mid-fade reads as random, absorb
                // the tap before it reaches the page control.
                // `onTapGesture` only consumes taps, so swipes still
                // pass through to the TabView's pan gesture.
                .overlay(alignment: .bottom) {
                    Color.clear
                        .contentShape(Rectangle())
                        .frame(maxWidth: 240, maxHeight: 36)
                        .padding(.bottom, 16)
                        .onTapGesture { }
                        .allowsHitTesting(dotsVisible)
                }
                .onChange(of: selectedTab) { _, _ in
                    showDotsAndScheduleHide()
                }
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
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            // Force SwiftUI to rebuild the page-controller-backed
            // TabView on column-height changes — without this the
            // post-rotation page content kept its pre-rotation
            // content offset and overlapped the header.
            //
            // The id also folds in `headerHeight` and the
            // explicit recording-state / error-presence /
            // channel-count flags so any header expansion
            // triggers the same rebuild. The measured height
            // alone can miss transitions that don't reach
            // `.onGeometryChange` in time (e.g. the level
            // meter's row count flipping from 1 to 2 the moment
            // capture reports stereo). Including the underlying
            // state directly is a belt-and-braces signal: even
            // if the geometry callback hasn't fired yet, the
            // state flip forces the rebuild. SwiftUI's flex
            // layout shrinks the TabView's frame correctly, but
            // UIPageViewController doesn't reliably propagate
            // that to the page content — the inner ScrollView
            // keeps its prior layout and its top edge ends up
            // obscured by the now-larger header. Rebuilding
            // gives the page controller a fresh frame to lay
            // out against.
            .id(
                "\(Int(geo.size.height.rounded()))"
                + "-\(Int(headerHeight.rounded()))"
                + "-\(recorder.isRecording ? 1 : 0)"
                + "-\(recorder.inputChannelLevels.count)"
                + "-\(recorder.errorMessage != nil ? 1 : 0)"
            )
            // View menu (⌘1–⌘6) dispatch lives in a sibling
            // ViewModifier so the chained `.onChange` handlers
            // don't pile onto the same type-check expression as
            // the `.id` / `.onPreferenceChange` chain (which
            // already pushed the body past SwiftUI's type-check
            // budget when they lived inline).
            .modifier(ViewMenuTabDispatch(selectedTab: $selectedTab))
        }
    }

    // MARK: - Page-indicator auto-hide

    /// Flip `dotsVisible` true and schedule it back to false
    /// after a short pause. Cancels any pending hide so rapid
    /// swipes keep the indicator up until the user pauses.
    /// Both transitions go through `withAnimation` so the
    /// indicator's appearance is driven by SwiftUI's animation
    /// context rather than snapping on a state flip.
    private func showDotsAndScheduleHide() {
        dotsHideTask?.cancel()
        withAnimation(.easeInOut(duration: 0.25)) {
            dotsVisible = true
        }
        dotsHideTask = Task { @MainActor in
            try? await Task.sleep(nanoseconds: Self.dotsHideDelayNanos)
            guard !Task.isCancelled else { return }
            withAnimation(.easeInOut(duration: 0.6)) {
                dotsVisible = false
            }
        }
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
        .ignoresSafeArea(.container, edges: .bottom)
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
        .ignoresSafeArea(.container, edges: .bottom)
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
        .ignoresSafeArea(.container, edges: .bottom)
    }

    @ViewBuilder
    private var sectionsPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                SectionsCard(
                    recorder: recorder,
                    store: recorder.sections,
                    selectedUtteranceID: selectedUtteranceID,
                    onSummarize: { section in
                        llmCoord.presentSectionSummary(
                            section: section,
                            recorder: recorder
                        )
                    }
                )
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
        .ignoresSafeArea(.container, edges: .bottom)
    }

    @ViewBuilder
    private var keywordsPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                KeywordsCard(
                    store: recorder.keywords,
                    filePicker: filePicker,
                    keywordCounts: filterModel.keywordOccurrenceCounts(in: recorder)
                )
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
        .ignoresSafeArea(.container, edges: .bottom)
    }

    @ViewBuilder
    private var summarizerPage: some View {
        ScrollView(.vertical, showsIndicators: true) {
            VStack(spacing: 16) {
                SummarizerCard(recorder: recorder)
                ModelsCard(recorder: recorder)
                PromptsCard(recorder: recorder)
            }
            .frame(maxWidth: .infinity)
            .padding(.horizontal, 8)
            .padding(.bottom, 32)
        }
        .clipped()
        .ignoresSafeArea(.container, edges: .bottom)
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
            fileCoord.presentAudioPicker(recorder: recorder, filePicker: filePicker)
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

/// Listens for the six View-menu page tokens on
/// `MenuCommands` and writes the matching index to a bound
/// `selectedTab`. Extracted as a ViewModifier so the chain of
/// `.onChange` handlers doesn't collide with the parent body's
/// already-heavy `.id` / `.onPreferenceChange` chain inside
/// SwiftUI's type-checker budget.
private struct ViewMenuTabDispatch: ViewModifier {
    @Environment(MenuCommands.self) private var menuCommands
    @Binding var selectedTab: Int

    func body(content: Content) -> some View {
        content
            .onChange(of: menuCommands.viewSettingsToken)   { _, _ in selectedTab = 0 }
            .onChange(of: menuCommands.viewAffectToken)     { _, _ in selectedTab = 1 }
            .onChange(of: menuCommands.viewSpeakersToken)   { _, _ in selectedTab = 2 }
            .onChange(of: menuCommands.viewSectionsToken)   { _, _ in selectedTab = 3 }
            .onChange(of: menuCommands.viewKeywordsToken)   { _, _ in selectedTab = 4 }
            .onChange(of: menuCommands.viewSummarizerToken) { _, _ in selectedTab = 5 }
    }
}

