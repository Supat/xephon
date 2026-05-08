import SwiftUI
import Audio
import Fusion
import SERText

private extension SwitchingTextSER.Backend {
    /// Short tag rendered on each utterance row's text-backend badge.
    var badgeLabel: String {
        switch self {
        case .deberta:          return "DeBERTa"
        case .foundationModels: return "Apple FM"
        }
    }
}

struct ContentView: View {
    @State private var recorder = RecordingController()
    @State private var shareURL: URL?
    @State private var showingDiscardConfirm: Bool = false

    var body: some View {
        NavigationStack {
            GeometryReader { geo in
                HStack(spacing: 0) {
                    controlPane
                        .frame(width: geo.size.width / 3)
                    Divider()
                    transcriptPane
                        .frame(maxWidth: .infinity)
                }
            }
            .navigationTitle("Xephon")
            .navigationBarTitleDisplayMode(.inline)
            .toolbarTitleDisplayMode(.inline)
            .sheet(item: $shareURL) { url in
                ShareSheet(items: [url])
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
        }
    }

    // MARK: - Left pane (1/3): controls

    private var controlPane: some View {
        VStack(spacing: 16) {
            speechBoostToggle

            textSERPicker

            inputPicker

            recordButton

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

            PipelineCard(recorder: recorder)
                .padding(.top, 4)

            Spacer(minLength: 0)

            if !recorder.utterances.isEmpty {
                exportButton
            }
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
            transcriptList
                .frame(maxWidth: .infinity, maxHeight: .infinity)
        }
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

    private var speechBoostToggle: some View {
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

    @ViewBuilder
    private var statusLine: some View {
        if recorder.isRecording {
            Text(String(
                format: String(localized: "record.status.format"),
                recorder.elapsedSeconds,
                recorder.samplesCaptured
            ))
            .font(.system(.body, design: .monospaced))
            .foregroundStyle(.secondary)
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

    @ViewBuilder
    private var transcriptList: some View {
        ScrollViewReader { proxy in
            List(Array(recorder.utterances.enumerated()), id: \.offset) { idx, u in
                UtteranceRow(utterance: u)
                    .id(idx)
            }
            .listStyle(.plain)
            .onChange(of: recorder.utterances.count) { _, count in
                guard count > 0 else { return }
                withAnimation(.easeOut(duration: 0.2)) {
                    proxy.scrollTo(count - 1, anchor: .bottom)
                }
            }
        }
    }

    private var exportButton: some View {
        Button {
            Task {
                if let url = await recorder.exportJSON() {
                    shareURL = url
                }
            }
        } label: {
            Label(String(localized: "export.json"), systemImage: "square.and.arrow.up")
        }
        .buttonStyle(.bordered)
    }
}

private struct UtteranceRow: View {
    let utterance: UtteranceEstimate

    // V/A from fusion are in [0, 1] with 0.5 = neutral. Re-center to [-1, +1]
    // so positive vs negative read naturally and 0 maps to "neutral grey".
    private static let neutralEpsilon: Float = 0.05

    var body: some View {
        HStack(alignment: .top) {
            VStack(alignment: .leading, spacing: 6) {
                HStack(spacing: 6) {
                    Text(utterance.speakerID)
                        .font(.caption.bold())
                        .foregroundStyle(.tint)
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
                    Text(String(format: "%.1f–%.1f s", utterance.start, utterance.end))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                HStack(spacing: 8) {
                    if let label = utterance.fusedTopLabel {
                        let tint = Self.color(forLabel: label)
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

    /// Color tint for a fused emotion label. Tracks the conventional Plutchik
    /// wheel where it overlaps; falls back to grey for unknown/neutral labels.
    private static func color(forLabel raw: String) -> Color {
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
                if !recorder.volatileText.isEmpty {
                    Text("“\(recorder.volatileText)…”")
                        .font(.caption2.italic())
                        .foregroundStyle(.tertiary)
                        .padding(.leading, 24)
                        .lineLimit(2)
                        .truncationMode(.head)
                }
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
                metric: recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count) utts"
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
    private var captureMetric: String {
        recorder.isRecording
            ? String(format: "%.1f s", recorder.elapsedSeconds)
            : "—"
    }

    private var asrState: StageRow.State {
        if recorder.isRecording { return .pending }
        if !recorder.utterances.isEmpty { return .ready }
        return .idle
    }
    private var asrMetric: String {
        recorder.utterances.isEmpty ? "—" : "\(recorder.utterances.count)"
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

    private func latencyMetric(_ value: TimeInterval?) -> String {
        guard let value else { return "—" }
        if value >= 1 { return String(format: "%.2f s", value) }
        return String(format: "%.0f ms", value * 1000)
    }

    private var exportMetric: String {
        guard let date = recorder.lastExportAt else { return "—" }
        let interval = -date.timeIntervalSinceNow
        if interval < 60 { return String(format: "%.0fs ago", interval) }
        return String(format: "%.0fm ago", interval / 60)
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
    let metric: String

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
            Text(metric)
                .font(.caption.monospacedDigit())
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
