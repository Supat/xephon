import SwiftUI

struct LevelMeterView: View {
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
