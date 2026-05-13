import SwiftUI

/// Inline legend for the per-utterance fusion-contribution strip
/// that sits below the diarizer + emotion timelines. Lives on the
/// left pane's affect page below `SERAggregateCard` so the legend
/// is on the same surface the user is reading the bias data from.
///
/// The strip itself is intentionally lean (a 6pt band with no axis
/// labels) so an explainer card off to the side carries the
/// meaning. Two color swatches + one-line definitions, plus a
/// short footnote covering the all-one-color edge cases and the
/// ASR-confidence coupling.
struct FusionLegendCard: View {
    /// Acoustic share tint — must mirror
    /// `FusionContributionStrip.acousticTint` exactly so the legend
    /// reads against the strip without drift.
    private static let acousticTint = Color(red: 0.30, green: 0.55, blue: 0.85)
    private static let textTint = Color(red: 0.95, green: 0.62, blue: 0.30)
    /// Same fill opacity the strip uses for its segments — keeps
    /// the swatches looking like an extract of the strip, not a
    /// brighter advertisement of one.
    private static let segmentOpacity: Double = 0.78

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "fusion.legend.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            // Miniature strip preview so the user has the visual
            // anchor inline next to the text below.
            HStack(spacing: 0) {
                Rectangle()
                    .fill(Self.acousticTint.opacity(Self.segmentOpacity))
                Rectangle()
                    .fill(Self.textTint.opacity(Self.segmentOpacity))
            }
            .frame(height: 6)
            .clipShape(Capsule())

            legendRow(
                tint: Self.acousticTint,
                label: String(localized: "fusion.legend.acoustic"),
                description: String(localized: "fusion.legend.acoustic.detail")
            )
            legendRow(
                tint: Self.textTint,
                label: String(localized: "fusion.legend.text"),
                description: String(localized: "fusion.legend.text.detail")
            )

            Text(String(localized: "fusion.legend.footnote"))
                .font(.caption2)
                .foregroundStyle(.tertiary)
                .fixedSize(horizontal: false, vertical: true)
                .padding(.top, 2)
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .glassEffect(in: RoundedRectangle(cornerRadius: 12, style: .continuous))
    }

    @ViewBuilder
    private func legendRow(
        tint: Color,
        label: String,
        description: String
    ) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 8) {
            // Square swatch (not a capsule) so each color reads as a
            // sample rather than a button. Sized to align roughly
            // with the surrounding caption metrics.
            RoundedRectangle(cornerRadius: 3, style: .continuous)
                .fill(tint.opacity(Self.segmentOpacity))
                .frame(width: 12, height: 12)
                .alignmentGuide(.firstTextBaseline) { d in d.height - 2 }
            VStack(alignment: .leading, spacing: 1) {
                Text(label)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.primary)
                Text(description)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
