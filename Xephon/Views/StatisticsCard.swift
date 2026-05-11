import SwiftUI
import Fusion

/// Per-label utterance count histogram. Sits below `SummaryCard` and
/// reads `ConversationSummary.labelCounts` (raw counts, not the
/// confidence-weighted scores `topLabel` uses). Sorted by count
/// descending so the dominant labels read top-down. Empty until the
/// first labeled utterance arrives.
struct StatisticsCard: View {
    let summary: ConversationSummary

    private var sortedRows: [(label: String, count: Int)] {
        summary.labelCounts
            .map { (label: $0.key, count: $0.value) }
            // Tiebreak on the label string so the order is deterministic
            // for equal counts — otherwise dictionary iteration order
            // makes the panel jitter as new utterances arrive.
            .sorted { ($0.count, $1.label) > ($1.count, $0.label) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "statistics.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
                if summary.utteranceCount > 0 {
                    Text("\(summary.utteranceCount)")
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.tertiary)
                }
            }

            if sortedRows.isEmpty {
                Text(String(localized: "statistics.empty"))
                    .font(.caption)
                    .foregroundStyle(.tertiary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            } else {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(sortedRows, id: \.label) { row in
                        StatisticsRow(
                            label: row.label,
                            count: row.count,
                            total: summary.utteranceCount
                        )
                    }
                }
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }
}

struct StatisticsRow: View {
    let label: String
    let count: Int
    let total: Int

    var body: some View {
        let tint = emotionTint(for: label)
        let fraction: Double = total > 0 ? Double(count) / Double(total) : 0
        HStack(spacing: 8) {
            Text(label.capitalized(with: Locale(identifier: "en_US")))
                .font(.caption.monospaced())
                .foregroundStyle(tint)
                .frame(minWidth: 80, alignment: .leading)
            // Inline bar so the relative weight of each label is legible
            // at a glance — the count alone reads as a flat list.
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(tint.opacity(0.12))
                    Capsule()
                        .fill(tint.opacity(0.55))
                        .frame(width: geo.size.width * fraction)
                }
            }
            .frame(height: 6)
            Text("\(count)")
                .font(.caption.monospaced())
                .foregroundStyle(.secondary)
                .frame(minWidth: 28, alignment: .trailing)
        }
    }
}
