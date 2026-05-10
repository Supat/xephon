import SwiftUI
import Fusion
import SERText

struct UtteranceRow: View {
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
