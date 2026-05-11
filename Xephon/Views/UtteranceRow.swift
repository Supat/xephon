import SwiftUI
import Fusion
import SERAcoustic
import SERText

struct UtteranceRow: View {
    let number: Int
    let utterance: UtteranceEstimate
    let isMultiSpeaker: Bool
    let isExpanded: Bool
    let onToggleExpanded: () -> Void

    // V/A from fusion are in [0, 1] with 0.5 = neutral. Re-center to [-1, +1]
    // so positive vs negative read naturally and 0 maps to "neutral grey".
    private static let neutralEpsilon: Float = 0.05

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
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
            if isExpanded {
                detailSection
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .padding(.vertical, 4)
        .contentShape(Rectangle())
        .onLongPressGesture(minimumDuration: 0.4, perform: onToggleExpanded)
        .animation(.easeInOut(duration: 0.18), value: isExpanded)
    }

    @ViewBuilder
    private var detailSection: some View {
        VStack(alignment: .leading, spacing: 6) {
            Divider()
            HStack(spacing: 12) {
                if let conf = utterance.asrConfidence {
                    metaLine("ASR conf", String(format: "%.2f", conf))
                }
                if let vad = utterance.dimensional {
                    metaLine(
                        "Acoustic V/A/D",
                        String(
                            format: "%.2f · %.2f · %.2f",
                            vad.valence, vad.arousal, vad.dominance
                        )
                    )
                }
                if let weightText = fusionWeightSummary {
                    metaLine("Fusion V/A", weightText)
                }
                if let labelText = labelFusionSummary {
                    metaLine("Fusion label", labelText)
                }
            }

            HStack(alignment: .top, spacing: 16) {
                if let cat = utterance.acousticCategorical, !cat.probabilities.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("Acoustic SER (emotion2vec)")
                        ForEach(acousticEntries(cat.probabilities), id: \.label) { entry in
                            ProbabilityBar(label: entry.label, value: entry.value)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }

                if let pl = utterance.plutchik, !pl.probabilities.isEmpty {
                    VStack(alignment: .leading, spacing: 3) {
                        sectionHeader("Text SER (\(textBackendName))")
                        ForEach(plutchikEntries(pl.probabilities), id: \.label) { entry in
                            ProbabilityBar(label: entry.label, value: entry.value)
                        }
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
    }

    /// Compact summary of how V/A fusion weighted the two sides for
    /// this utterance. Nil when neither modality contributed (the
    /// fused V/A would be nil too, so there's nothing to attribute).
    /// When only one modality was present, reports that side as 100%
    /// so the user can see which side carried the result.
    private var fusionWeightSummary: String? {
        let hasAcoustic = utterance.dimensional != nil
        let hasText = utterance.plutchik != nil
        switch (hasAcoustic, hasText) {
        case (false, false):
            return nil
        case (true, false):
            return "Acoustic 100% (no text)"
        case (false, true):
            return "Text 100% (no acoustic)"
        case (true, true):
            let share = LateFusion.defaultVAFusionShare(
                asrConfidence: utterance.asrConfidence ?? 0.5
            )
            return String(
                format: "Acoustic %.0f%% · Text %.0f%%",
                share.acoustic * 100,
                share.text * 100
            )
        }
    }

    /// Compact summary of how each modality contributed to the
    /// winning top label. Nil when there's no top label or when the
    /// winning label wasn't reachable from either modality (which
    /// shouldn't happen for a well-formed estimate but we guard
    /// anyway).
    private var labelFusionSummary: String? {
        guard let label = utterance.fusedTopLabel else { return nil }
        guard let share = LateFusion.defaultLabelFusionShare(
            forLabel: label,
            acoustic: utterance.acousticCategorical,
            plutchik: utterance.plutchik,
            asrConfidence: utterance.asrConfidence ?? 0.5
        ) else { return nil }
        return String(
            format: "Acoustic %.0f%% · Text %.0f%%",
            share.acoustic * 100,
            share.text * 100
        )
    }

    private var textBackendName: String {
        guard let raw = utterance.textBackend,
              let backend = SwitchingTextSER.Backend(rawValue: raw) else {
            return "Plutchik"
        }
        return backend.badgeLabel
    }

    private func metaLine(_ key: String, _ value: String) -> some View {
        HStack(spacing: 4) {
            Text(key)
                .font(.caption2)
                .foregroundStyle(.secondary)
            Text(value)
                .font(.caption.monospacedDigit())
                .foregroundStyle(.primary)
        }
    }

    private func sectionHeader(_ text: String) -> some View {
        Text(text)
            .font(.caption2.bold())
            .foregroundStyle(.secondary)
            .padding(.top, 2)
    }

    private func acousticEntries(
        _ probs: [CategoricalEmotion.Label: Float]
    ) -> [(label: String, value: Float)] {
        probs
            .map { (label: $0.key.rawValue, value: $0.value) }
            .sorted { $0.value > $1.value }
    }

    private func plutchikEntries(
        _ probs: [PlutchikScore.Label: Float]
    ) -> [(label: String, value: Float)] {
        probs
            .map { (label: $0.key.rawValue, value: $0.value) }
            .sorted { $0.value > $1.value }
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

private struct ProbabilityBar: View {
    let label: String
    let value: Float

    var body: some View {
        let tint = emotionTint(for: label)
        let fraction = Double(max(0, min(1, value)))
        HStack(spacing: 8) {
            Text(label.capitalized(with: Locale(identifier: "en_US")))
                .font(.caption2.monospaced())
                .foregroundStyle(tint)
                .frame(minWidth: 80, alignment: .leading)
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule().fill(tint.opacity(0.12))
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: geo.size.width * fraction)
                }
            }
            .frame(height: 5)
            Text(String(format: "%.2f", value))
                .font(.caption2.monospacedDigit())
                .foregroundStyle(.secondary)
                .frame(minWidth: 36, alignment: .trailing)
        }
    }
}
