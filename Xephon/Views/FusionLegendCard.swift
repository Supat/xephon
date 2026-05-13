import SwiftUI
import Fusion

/// Inline legend + adjustable controls for the per-utterance
/// fusion-contribution strip. Lives on the left pane's affect page
/// below `SERAggregateCard` so the explanation, the live sliders,
/// and the footnote all sit on the same surface the user is
/// reading the strip from.
///
/// Layout, top-to-bottom:
///   1. Gradient swatch preview spanning text → neutral → acoustic.
///   2. Two labeled rows naming the endpoint modalities.
///   3. Adjustable controls — sliders for `acousticWeight` and
///      `textWeightFloor`, plus a Reset affordance.
///   4. Footnote covering the color-mapping math and the ASR-
///      confidence coupling.
///
/// Slider changes hit the controller immediately and propagate
/// into the strip + per-row inspector on the next render. New
/// utterances fuse under the new weights; existing utterances
/// keep their cached fused V/A/D until manually re-evaluated.
struct FusionLegendCard: View {
    let recorder: RecordingController
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
    private static let segmentOpacity: Double = 1.0

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

            Divider()
            controlsSection

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
    private var controlsSection: some View {
        let acousticBinding = Binding<Float>(
            get: { recorder.fusionAcousticWeight },
            set: { recorder.setFusionAcousticWeight($0) }
        )
        let textFloorBinding = Binding<Float>(
            get: { recorder.fusionTextWeightFloor },
            set: { recorder.setFusionTextWeightFloor($0) }
        )
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .firstTextBaseline) {
                Text(String(localized: "fusion.controls.header"))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.tertiary)
                    .textCase(.uppercase)
                Spacer()
                Button(role: .destructive) {
                    recorder.resetFusionWeights()
                } label: {
                    Text(String(localized: "fusion.controls.reset"))
                        .font(.caption2)
                }
                .buttonStyle(.borderless)
                .disabled(
                    recorder.fusionAcousticWeight == LateFusion.defaultAcousticWeight
                        && recorder.fusionTextWeightFloor == LateFusion.defaultTextWeightFloor
                )
            }
            sliderRow(
                label: String(localized: "fusion.controls.acousticWeight"),
                value: acousticBinding,
                range: 0...2,
                step: 0.05
            )
            sliderRow(
                label: String(localized: "fusion.controls.textWeightFloor"),
                value: textFloorBinding,
                range: 0...1,
                step: 0.05
            )
        }
    }

    @ViewBuilder
    private func sliderRow(
        label: String,
        value: Binding<Float>,
        range: ClosedRange<Float>,
        step: Float
    ) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack {
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Spacer()
                Text(String(format: "%.2f", value.wrappedValue))
                    .font(.caption2.monospacedDigit())
                    .foregroundStyle(.primary)
                    .frame(width: 40, alignment: .trailing)
            }
            Slider(value: value, in: range, step: step)
                .controlSize(.small)
        }
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
