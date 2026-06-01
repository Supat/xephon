import SwiftUI
import ASR
import Diarization
import SERText

/// Settings card sitting above `PipelineCard`. Hosts the session-
/// language picker, the text-SER backend picker, the speech-boost
/// toggle, and the diarizer-sensitivity slider — controls that
/// configure how the pipeline runs but aren't part of the live stage
/// visualization itself. The session-summarizer controls live on
/// `SessionSummarySheet` so they sit next to the artifact they affect.
///
/// Language and Text SER share a row when there's enough horizontal
/// space (landscape, regular iPad layout). In portrait — where the
/// left pane shrinks to ~1/3 of screen width — they stack vertically
/// and switch to an *inline* row layout (label on the leading edge,
/// menu pinned to the trailing edge) so each row reads "Language →
/// Japanese", "Text SER → WRIME" without truncation. `ViewThatFits`
/// picks the first variant whose horizontal extent fits.
///
/// Speech-boost (a toggle, distinct affordance) sits below on its
/// own line, followed by the diarizer-sensitivity slider — both are
/// full-width controls that don't share a row with the pickers.
struct SettingsCard: View {
    let recorder: RecordingController

    @State private var showingGlossary = false

    /// Picker layout style. Landscape gets `.stacked` (label above
    /// control). Portrait gets `.inline` so the label hugs the leading
    /// edge and the control hugs the trailing edge.
    enum PickerLayout {
        case stacked, inline
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Row 1: language + offline ASR live together —
            // they're both "what audio gets transcribed as" knobs
            // and benefit from sitting side-by-side. Text SER
            // moves to its own row (different concern: emotion
            // classifier choice, independent of ASR pipeline).
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) {
                    languagePicker(layout: .stacked)
                    offlineASRPicker(layout: .stacked)
                }
                VStack(spacing: 12) {
                    languagePicker(layout: .inline)
                    offlineASRPicker(layout: .inline)
                }
            }
            textSERPicker
            speechBoostToggle
            diarizerSensitivitySlider
            customGlossaryButton
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
        .sheet(isPresented: $showingGlossary) {
            CustomGlossarySheet(
                store: recorder.glossary,
                onDismiss: {
                    showingGlossary = false
                    Task { await recorder.reapplyGlossaryBias() }
                }
            )
        }
    }

    /// Settings row that raises the Custom Glossary sheet. Trailing
    /// count chip so the user sees at a glance how loaded their
    /// glossary is, and whether the bias is currently armed (the
    /// chip dims when `isEnabled` is false).
    @ViewBuilder
    private var customGlossaryButton: some View {
        Button {
            showingGlossary = true
        } label: {
            HStack(spacing: 8) {
                Label(
                    String(localized: "glossary.title"),
                    systemImage: "book.closed"
                )
                Spacer(minLength: 8)
                if !recorder.glossary.entries.isEmpty {
                    Text("\(recorder.glossary.entries.count)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                        .opacity(recorder.glossary.isEnabled ? 1.0 : 0.4)
                }
                Image(systemName: "chevron.right")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .padding(.horizontal)
    }

    /// Session-language picker. Drives the ASR locale (Apple
    /// SpeechTranscriber) and the text-SER gating (the WRIME-tuned text SER is
    /// Japanese-only and hides for non-Japanese sessions). Disabled
    /// while a session is active because the streaming transcriber
    /// is locked to its start-time locale — the user can still see
    /// which language is in effect for the running session.
    @ViewBuilder
    private func languagePicker(layout: PickerLayout) -> some View {
        let label = Text(String(localized: "settings.language"))
            .font(.caption)
            .foregroundStyle(.secondary)
        let control = Picker(
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
        layoutPair(label: label, control: control, layout: layout)
    }

    /// Text SER picker lives on its own row — distinct concern
    /// from Language / Offline ASR (it's the emotion classifier,
    /// not the transcriber).
    ///
    /// Layout depends on pane width (proxy for orientation):
    /// - Landscape (pane wide enough): `.stacked` — label on top,
    ///   picker on the row below, LEFT-aligned filling the row.
    /// - Portrait (pane narrower): `.inline` — label on top,
    ///   picker on the row below, RIGHT-aligned (matches the
    ///   Language / Offline ASR portrait layout).
    ///
    /// The 340pt threshold roughly tracks where the
    /// Language/Offline ASR `ViewThatFits` flips between its
    /// two-column HStack and the stacked VStack, so all three
    /// pickers in the card switch alignment together as the user
    /// rotates the device.
    @ViewBuilder
    private var textSERPicker: some View {
        if recorder.availableTextSERBackends.count > 1 {
            let label = Text(String(localized: "settings.textSER"))
                .font(.caption)
                .foregroundStyle(.secondary)
            let control = Picker(
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
            ViewThatFits(in: .horizontal) {
                layoutPair(label: label, control: control, layout: .stacked)
                    .frame(minWidth: 340)
                layoutPair(label: label, control: control, layout: .inline)
            }
        }
    }

    /// Shared label-above-control layout. Landscape places both in a
    /// `VStack` that fills available width. Portrait — where the
    /// left pane shrinks to ~1/3 — pushes the control to the trailing
    /// edge so the menu button lines up with the other right-aligned
    /// values on this card.
    @ViewBuilder
    private func layoutPair<L: View, C: View>(
        label: L, control: C, layout: PickerLayout
    ) -> some View {
        switch layout {
        case .stacked:
            VStack(alignment: .leading, spacing: 4) {
                label
                control
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        case .inline:
            VStack(alignment: .leading, spacing: 4) {
                label
                control
                    .frame(maxWidth: .infinity, alignment: .trailing)
            }
            .frame(maxWidth: .infinity)
        }
    }

    private static func label(for backend: SwitchingTextSER.Backend) -> String {
        switch backend {
        case .deberta:          return String(localized: "settings.textSER.deberta")
        case .foundationModels: return String(localized: "settings.textSER.foundationModels")
        }
    }

    /// Picker for the ASR backend used by both live recording and
    /// the non-streaming paths (file analysis, re-evaluation,
    /// Transcribe Range). Qwen3-ASR rows carry a yellow warning
    /// glyph as a discoverable hint that the backend's
    /// performance is subpar today: it sometimes drifts to its
    /// training-dominant language despite the hint (post-filter
    /// keeps the picker's language but can produce empty
    /// transcripts) and the streaming wrapper's chunk latency
    /// lags live audio by 10–20 s. Disabled while a session is
    /// in flight to avoid swapping the transcriber mid-analysis.
    @ViewBuilder
    private func offlineASRPicker(layout: PickerLayout) -> some View {
        if recorder.availableOfflineASRBackends.count > 1 {
            let label = Text(String(localized: "settings.offlineASR"))
                .font(.caption)
                .foregroundStyle(.secondary)
            let control = Picker(
                String(localized: "settings.offlineASR"),
                selection: Binding(
                    get: { recorder.currentOfflineASRBackend },
                    set: { newValue in
                        Task { await recorder.setOfflineASRBackend(newValue) }
                    }
                )
            ) {
                ForEach(recorder.availableOfflineASRBackends, id: \.self) { backend in
                    Self.pickerRow(for: backend).tag(backend)
                }
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .disabled(recorder.isRecording || recorder.isAnalyzing)
            layoutPair(label: label, control: control, layout: layout)
        }
    }

    /// Row content rendered inside each Picker item. Most backends
    /// are a plain `Text`; `.qwen3ASR` prefixes the warning emoji
    /// (U+26A0 U+FE0F) to flag that the backend's performance is
    /// subpar today.
    ///
    /// The emoji ("⚠️") is used directly in the Text rather than a
    /// `Label(systemImage:)` with `.foregroundStyle(.yellow)`
    /// because the UIPickerMenu rendering routes SF Symbol icons
    /// through `UIImage.withRenderingMode(.alwaysTemplate)`,
    /// which strips any SwiftUI color modifier and tints the
    /// glyph with the menu's text style. Emoji glyphs are text
    /// content, not template images, so their native colors
    /// survive.
    @ViewBuilder
    private static func pickerRow(for backend: OfflineASRBackend) -> some View {
        let baseLabel = Self.offlineASRLabel(for: backend)
        switch backend {
        case .qwen3ASR:
            Text("⚠️ \(baseLabel)")
        default:
            Text(baseLabel)
        }
    }

    private static func offlineASRLabel(for backend: OfflineASRBackend) -> String {
        switch backend {
        case .speechAnalyzer: return String(localized: "settings.offlineASR.speechAnalyzer")
        case .qwen3ASR:       return String(localized: "settings.offlineASR.qwen3ASR")
        }
    }

    @ViewBuilder
    private var speechBoostToggle: some View {
        // Hidden in file mode — the toggle wouldn't affect file content.
        if case .microphone = recorder.sourceMode {
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
    }

    /// "Sensitivity" inverts the underlying clustering threshold
    /// (lower threshold = more distinct speakers) so dragging right
    /// reads as "more speakers." Double-tap the label to restore
    /// the default. `step:` discretizes the drag so a smooth gesture
    /// doesn't queue a Task per frame against the diarizer actor.
    @ViewBuilder
    private var diarizerSensitivitySlider: some View {
        let bounds = FluidAudioDiarizer.displayClusteringThresholdRange
        let lower = bounds.lowerBound
        let upper = bounds.upperBound
        let current = recorder.diarizerClusteringThreshold
        let sensitivityBinding = Binding<Double>(
            get: {
                // Clamp into the displayed band in case a stored
                // value sits outside it (older builds, manual
                // UserDefaults edits).
                let clamped = current.clamped(to: lower...upper)
                return Double(1.0 - (clamped - lower) / (upper - lower))
            },
            set: { newValue in
                let clampedSensitivity = Float(newValue.clamped(to: 0.0...1.0))
                let newThreshold = upper - clampedSensitivity * (upper - lower)
                Task { await recorder.setDiarizerClusteringThreshold(newThreshold) }
            }
        )
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Label(
                    String(localized: "settings.diarizerSensitivity"),
                    systemImage: "person.2.wave.2"
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Spacer(minLength: 8)
                Text(String(format: "%.2f", current))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
            }
            Slider(
                value: sensitivityBinding,
                in: 0.0...1.0,
                step: 0.025
            ) {
                Text(String(localized: "settings.diarizerSensitivity"))
            } minimumValueLabel: {
                Text(String(localized: "settings.diarizerSensitivity.min"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            } maximumValueLabel: {
                Text(String(localized: "settings.diarizerSensitivity.max"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(String(localized: "settings.diarizerSensitivity.hint"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
        }
        .padding(.horizontal)
        .contentShape(Rectangle())
        .onTapGesture(count: 2) {
            Task { await recorder.resetDiarizerClusteringThreshold() }
        }
    }
}
