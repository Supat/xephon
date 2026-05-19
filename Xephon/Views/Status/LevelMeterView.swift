import SwiftUI

struct LevelMeterView: View {
    /// One entry per source channel. A stereo USB mic shows two
    /// rows (L/R) so per-channel balance is visible; a built-in
    /// mono mic shows one. Falls back to a single bar at 0 when
    /// empty (capture hasn't produced a chunk yet).
    let channelLevels: [Float]
    private let segmentCount = 24

    /// Backwards-compat convenience for callers that still pass a
    /// single `level:` value (e.g. PipelineCard's binding).
    init(level: Float) {
        self.channelLevels = [level]
    }

    init(channelLevels: [Float]) {
        self.channelLevels = channelLevels.isEmpty ? [0] : channelLevels
    }

    var body: some View {
        VStack(spacing: 3) {
            ForEach(Array(channelLevels.enumerated()), id: \.offset) { _, level in
                channelRow(level: level)
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Microphone level")
        .accessibilityValue(accessibilitySummary)
    }

    @ViewBuilder
    private func channelRow(level: Float) -> some View {
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
    }

    private var accessibilitySummary: String {
        if channelLevels.count == 1 {
            return "\(Int(channelLevels[0] * 100)) percent"
        }
        return channelLevels.enumerated().map { i, l in
            "channel \(i + 1) \(Int(l * 100)) percent"
        }.joined(separator: ", ")
    }

    private func color(for ratio: Float) -> Color {
        if ratio < 0.65 { return .green }
        if ratio < 0.88 { return .yellow }
        return .red
    }
}
