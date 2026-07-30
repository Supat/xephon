import SwiftUI

/// The paper sheet's two per-item scales, rendered as they appear
/// on the form: a relative-strength axis (強い −1 … +1 弱い, minor
/// ticks at the template's step, labeled majors every 0.5) and a
/// preference axis (嫌い 1 … 9 好き). A stated value draws a solid
/// tint marker; an inferred value draws an orange marker (same
/// suggestion-not-entry coding as the score badge); no value leaves
/// the empty track — which is itself information, matching an
/// unfilled row on the sheet.
///
/// Hand-drawn with GeometryReader shapes (the app's strip idiom)
/// rather than Swift Charts — these are 12 pt annotated axes, not
/// data plots.

/// Shared geometry + drawing for one horizontal scale.
struct EvalScaleAxis: View {
    struct Tick {
        let position: CGFloat   // 0…1 along the track
        let label: String?      // nil = minor tick
    }

    let ticks: [Tick]
    let leadingLabel: String
    let trailingLabel: String
    /// (normalized position, isInferred) — nil for an empty track.
    let marker: (position: CGFloat, inferred: Bool)?

    private static let trackHeight: CGFloat = 1
    private static let majorTick: CGFloat = 7
    private static let minorTick: CGFloat = 4
    private static let markerSize: CGFloat = 9
    private static let axisHeight: CGFloat = 12
    private static let labelHeight: CGFloat = 10

    /// Normalized 0…1 position of `value` on [minimum, maximum].
    static func normalizedPosition(
        _ value: Double,
        minimum: Double,
        maximum: Double
    ) -> CGFloat {
        guard maximum > minimum else { return 0 }
        let clamped = min(max(value, minimum), maximum)
        return CGFloat((clamped - minimum) / (maximum - minimum))
    }

    var body: some View {
        HStack(alignment: .center, spacing: 4) {
            Text(verbatim: leadingLabel)
                .font(.system(size: 8))
                .foregroundStyle(.secondary)
            GeometryReader { geo in
                let width = geo.size.width
                ZStack(alignment: .topLeading) {
                    // Track.
                    Rectangle()
                        .fill(Color.secondary.opacity(0.55))
                        .frame(width: width, height: Self.trackHeight)
                        .offset(y: Self.axisHeight / 2)
                    // Ticks, centered on the track.
                    ForEach(Array(ticks.enumerated()), id: \.offset) { _, tick in
                        let height = tick.label != nil ? Self.majorTick : Self.minorTick
                        Rectangle()
                            .fill(Color.secondary.opacity(tick.label != nil ? 0.7 : 0.4))
                            .frame(width: 1, height: height)
                            .offset(
                                x: tick.position * width - 0.5,
                                y: Self.axisHeight / 2 - height / 2
                            )
                        if let label = tick.label {
                            Text(verbatim: label)
                                .font(.system(size: 7).monospacedDigit())
                                .foregroundStyle(.tertiary)
                                .fixedSize()
                                // Center the label under its tick;
                                // 30 is comfortably wider than any
                                // "-0.5"-class label at 7 pt.
                                .frame(width: 30)
                                .offset(
                                    x: tick.position * width - 15,
                                    y: Self.axisHeight + 1
                                )
                        }
                    }
                    // Marker.
                    if let marker {
                        Circle()
                            .fill(marker.inferred ? Color.orange : Color.accentColor)
                            .frame(width: Self.markerSize, height: Self.markerSize)
                            .offset(
                                x: marker.position * width - Self.markerSize / 2,
                                y: (Self.axisHeight - Self.markerSize) / 2
                            )
                            .opacity(marker.inferred ? 0.85 : 1)
                    }
                }
            }
            .frame(height: Self.axisHeight + Self.labelHeight)
            Text(verbatim: trailingLabel)
                .font(.system(size: 8))
                .foregroundStyle(.secondary)
        }
    }
}

/// 強い −1 … +1 弱い, built from the template's strength scale.
struct StrengthScaleView: View {
    let scale: EvalFormTemplate.StrengthScale
    let stated: Double?
    let inferred: Double?

    var body: some View {
        EvalScaleAxis(
            ticks: ticks,
            leadingLabel: "強い",
            trailingLabel: "弱い",
            marker: marker
        )
    }

    private var ticks: [EvalScaleAxis.Tick] {
        // Minor tick at every quantized step, labeled major every
        // 0.5 — the sheet's layout.
        scale.allowedValues.map { value in
            let isMajor = abs(value.truncatingRemainder(dividingBy: 0.5)) < 0.0005
            let label = isMajor
                ? (value > 0 ? "+\(String(format: "%g", value))" : String(format: "%g", value))
                : nil
            return EvalScaleAxis.Tick(
                position: EvalScaleAxis.normalizedPosition(
                    value, minimum: scale.minimum, maximum: scale.maximum
                ),
                label: label
            )
        }
    }

    private var marker: (position: CGFloat, inferred: Bool)? {
        if let stated {
            return (EvalScaleAxis.normalizedPosition(
                stated, minimum: scale.minimum, maximum: scale.maximum
            ), false)
        }
        if let inferred {
            return (EvalScaleAxis.normalizedPosition(
                inferred, minimum: scale.minimum, maximum: scale.maximum
            ), true)
        }
        return nil
    }
}

/// 嫌い 1 … 9 好き, built from the template's preference scale.
struct PreferenceScaleView: View {
    let scale: EvalFormTemplate.PreferenceScale
    let value: Int?

    var body: some View {
        EvalScaleAxis(
            ticks: ticks,
            leadingLabel: "嫌い",
            trailingLabel: "好き",
            marker: value.map { v in
                (EvalScaleAxis.normalizedPosition(
                    Double(v),
                    minimum: Double(scale.minimum),
                    maximum: Double(scale.maximum)
                ), false)
            }
        )
    }

    private var ticks: [EvalScaleAxis.Tick] {
        (scale.minimum...scale.maximum).map { v in
            let midpoint = (scale.minimum + scale.maximum) / 2
            let isMajor = v == scale.minimum || v == scale.maximum || v == midpoint
            return EvalScaleAxis.Tick(
                position: EvalScaleAxis.normalizedPosition(
                    Double(v),
                    minimum: Double(scale.minimum),
                    maximum: Double(scale.maximum)
                ),
                label: isMajor ? "\(v)" : nil
            )
        }
    }
}
