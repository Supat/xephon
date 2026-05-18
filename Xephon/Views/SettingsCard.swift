import SwiftUI
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
/// Japanese", "Text SER → DeBERTa" without truncation. `ViewThatFits`
/// picks the first variant whose horizontal extent fits.
///
/// Speech-boost (a toggle, distinct affordance) sits below on its
/// own line, followed by the diarizer-sensitivity slider — both are
/// full-width controls that don't share a row with the pickers.
struct SettingsCard: View {
    let recorder: RecordingController

    /// Picker layout style. Landscape gets `.stacked` (label above
    /// control). Portrait gets `.inline` so the label hugs the leading
    /// edge and the control hugs the trailing edge.
    enum PickerLayout {
        case stacked, inline
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            ViewThatFits(in: .horizontal) {
                HStack(alignment: .top, spacing: 16) {
                    languagePicker(layout: .stacked)
                    textSERPicker(layout: .stacked)
                }
                VStack(spacing: 12) {
                    languagePicker(layout: .inline)
                    textSERPicker(layout: .inline)
                }
            }
            speechBoostToggle
            diarizerSensitivitySlider
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    /// Session-language picker. Drives the ASR locale (Apple
    /// SpeechTranscriber) and the text-SER gating (DeBERTa-WRIME is
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

    @ViewBuilder
    private func textSERPicker(layout: PickerLayout) -> some View {
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
            layoutPair(label: label, control: control, layout: layout)
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
