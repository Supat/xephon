import SwiftUI
import Fusion

/// Real-time conversation mood summary. Sits below `PipelineCard` and
/// updates incrementally as each utterance lands. Below
/// `ConversationSummary.calibrationThreshold` we suppress the V/A/D
/// numbers and show a "calibrating" placeholder — the means are noisy
/// when only one or two utterances exist and would mislead.
struct SummaryCard: View {
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

/// Sparkline for the bounded valence trajectory. Center line at 0.5
/// (neutral), points above are positive valence (green tint), below are
/// negative (red tint). Drawn with a single Path so it scales cheaply
/// even at the trajectory cap.
struct TrajectorySparkline: View {
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
