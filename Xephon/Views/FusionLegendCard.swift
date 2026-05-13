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
    /// Palette endpoint tints — pulled from the strip's own RGB
    /// constants so the legend can't drift out of sync with the
    /// rendering it explains.
    private static let acousticTint = Color(
        red: FusionContributionStrip.acousticRGB.r,
        green: FusionContributionStrip.acousticRGB.g,
        blue: FusionContributionStrip.acousticRGB.b
    )
    private static let textTint = Color(
        red: FusionContributionStrip.textRGB.r,
        green: FusionContributionStrip.textRGB.g,
        blue: FusionContributionStrip.textRGB.b
    )
    private static let neutralTint = Color(
        red: FusionContributionStrip.neutralRGB.r,
        green: FusionContributionStrip.neutralRGB.g,
        blue: FusionContributionStrip.neutralRGB.b
    )
    /// Same fill opacity the strip uses, so the gradient preview
    /// looks like an extract of the strip rather than a brighter
    /// advertisement.
    private static let segmentOpacity: Double = 0.85

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text(String(localized: "fusion.legend.header"))
                    .font(.caption.bold())
                    .foregroundStyle(.secondary)
                Spacer()
            }
            // Miniature gradient preview so the user has the
            // visual anchor inline next to the text below. The
            // three-stop linear gradient mirrors the strip's
            // divergent palette: text on the left (−1), neutral
            // in the middle (0), acoustic on the right (+1).
            LinearGradient(
                stops: [
                    .init(color: Self.textTint.opacity(Self.segmentOpacity), location: 0.0),
                    .init(color: Self.neutralTint.opacity(Self.segmentOpacity), location: 0.5),
                    .init(color: Self.acousticTint.opacity(Self.segmentOpacity), location: 1.0),
                ],
                startPoint: .leading,
                endPoint: .trailing
            )
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
