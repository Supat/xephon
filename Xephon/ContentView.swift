import SwiftUI
import Audio
import Fusion

struct ContentView: View {
    @State private var allowsCloudASRFallback: Bool = false
    @State private var recorder = RecordingController()
    @State private var shareURL: URL?

    var body: some View {
        NavigationStack {
            VStack(spacing: 16) {
                // Per CLAUDE.md: cloud-fallback toggle state must be visible while recording.
                Toggle(String(localized: "settings.cloudASRFallback"),
                       isOn: $allowsCloudASRFallback)
                    .toggleStyle(.switch)
                    .padding(.horizontal)

                inputPicker

                recordButton

                if recorder.isRecording {
                    LevelMeterView(level: recorder.inputLevel)
                        .frame(maxWidth: 320)
                }

                statusLine

                if let error = recorder.errorMessage {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .padding(.horizontal)
                        .multilineTextAlignment(.center)
                }

                if !recorder.utterances.isEmpty {
                    Divider()
                    transcriptList
                    exportButton
                }

                Spacer(minLength: 0)
            }
            .padding()
            .navigationTitle("Xephon")
            .sheet(item: $shareURL) { url in
                ShareSheet(items: [url])
            }
        }
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
            Task { await recorder.toggle() }
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
                Text(utterance.speakerID)
                    .font(.caption.bold())
                    .foregroundStyle(.tint)
                Text(utterance.transcript.isEmpty ? "—" : utterance.transcript)
                    .font(.body)
            }
            Spacer(minLength: 8)
            VStack(alignment: .trailing, spacing: 4) {
                Text(String(format: "%.1f–%.1f s", utterance.start, utterance.end))
                    .font(.caption.monospacedDigit())
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    if let label = utterance.fusedTopLabel {
                        Text(label)
                            .font(.caption)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(.tint.opacity(0.15), in: Capsule())
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
