import SwiftUI
import Fusion

/// Per-utterance horizontal bar showing how much each modality
/// contributed to the fused-label argmax for that row. Each row's
/// slot on the strip spans its `[start, end]` audio range and is
/// filled with a single color drawn from a divergent palette: full
/// text → orange, balanced → neutral gray, full acoustic → blue,
/// all values in between interpolated linearly on the
/// `net = (acoustic − text) / (acoustic + text)` axis (range
/// `[-1, +1]`). Mirrors the heatmap's "one color says it all"
/// rendering style; rows with only one modality saturate to that
/// modality's endpoint.
///
/// Sits below the emotion-timeline strip in the transcript pane so
/// the three strips read top-down as: speaker → emotion label →
/// modality balance. The X axis is shared across all three.
struct FusionContributionStrip: View {
    let utterances: [UtteranceEstimate]
    let totalDuration: TimeInterval
    /// Selected utterance's `[start, end]` window. Renders a
    /// primary-color stroke over that range — matches the diarizer
    /// and emotion strips so the three highlights line up at the
    /// same audio time when a row is focused.
    let selectedRange: (start: TimeInterval, end: TimeInterval)?
    /// Fires when the user taps a point on the strip; the
    /// `TimeInterval` is the audio-time the tap location maps to,
    /// clamped to `[0, totalDuration]`. Same hook the other two
    /// strips use to route the selection back through ContentView.
    let onTapAtTime: ((TimeInterval) -> Void)?

    private static let height: CGFloat = 6
    /// Endpoint RGB for the acoustic side (W2V2 + emotion2vec).
    /// Stored as a tuple, not a `Color`, so the per-slot blend can
    /// interpolate component-wise without round-tripping through a
    /// runtime color-extraction API. `FusionLegendCard` mirrors
    /// these values when rendering the gradient swatch — keep the
    /// two in sync.
    static let acousticRGB: (r: Double, g: Double, b: Double) = (0.30, 0.55, 0.85)
    /// Endpoint RGB for the text side (DeBERTa / Apple FM).
    static let textRGB: (r: Double, g: Double, b: Double) = (0.95, 0.62, 0.30)
    /// Midpoint RGB used when the two modalities contribute
    /// roughly equally. Neutral mid-gray so a balanced row reads
    /// as "neither modality dominates" rather than a muddy average
    /// of orange + blue.
    static let neutralRGB: (r: Double, g: Double, b: Double) = (0.55, 0.55, 0.55)
    /// Fill opacity of the per-slot color. Lower than the
    /// diarizer strip's runs so the strip reads as supporting
    /// detail, not a peer-level signal.
    private static let segmentOpacity: Double = 0.85
    /// Background track tint behind the segments — visible only in
    /// the gaps between utterance slots.
    private static let trackTintOpacity: Double = 0.06

    var body: some View {
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .glassEffect(
                        .regular.tint(.secondary.opacity(Self.trackTintOpacity)),
                        in: Capsule()
                    )
                    .frame(height: Self.height)
                ForEach(utterances) { utt in
                    let share = shareFor(utt)
                    if let geometry = slotGeometry(for: utt, width: geo.size.width) {
                        Rectangle()
                            .fill(
                                Self.contributionColor(
                                    acoustic: share.acoustic,
                                    text: share.text
                                )
                                .opacity(Self.segmentOpacity)
                            )
                            .frame(width: geometry.width, height: Self.height)
                            .offset(x: geometry.x)
                    }
                }
            }
            // Same primary-stroke selection overlay the diarizer
            // and emotion strips use, so all three highlighters
            // line up at the focused row's audio range.
            .overlay(alignment: .leading) {
                if let sel = selectedRange, totalDuration > 0 {
                    let clampedStart = max(0, min(sel.start, totalDuration))
                    let clampedEnd = max(clampedStart, min(sel.end, totalDuration))
                    let x = geo.size.width * CGFloat(clampedStart / totalDuration)
                    let w = geo.size.width * CGFloat((clampedEnd - clampedStart) / totalDuration)
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.9), lineWidth: 1.5)
                        .frame(width: max(2, w), height: Self.height)
                        .offset(x: x)
                }
            }
            .clipShape(Capsule())
            .contentShape(Rectangle())
            .onTapGesture(coordinateSpace: .local) { location in
                guard let onTapAtTime, totalDuration > 0, geo.size.width > 0 else { return }
                let t = totalDuration * Double(location.x / geo.size.width)
                onTapAtTime(max(0, min(t, totalDuration)))
            }
        }
        .frame(height: Self.height)
    }

    /// X-offset + width of a row's slot in screen coordinates. nil
    /// when the row's range falls outside the strip's audio axis
    /// (negative width or zero-width slot — those would render as
    /// invisible hairlines anyway).
    private func slotGeometry(
        for utt: UtteranceEstimate,
        width: CGFloat
    ) -> (x: CGFloat, width: CGFloat)? {
        guard totalDuration > 0 else { return nil }
        let start = max(0, utt.start)
        let end = min(totalDuration, utt.end)
        guard end > start else { return nil }
        let x = width * CGFloat(start / totalDuration)
        let w = width * CGFloat((end - start) / totalDuration)
        return (x: x, width: max(1, w))
    }

    /// Map a `(acoustic, text)` share pair to a single color on
    /// the diverging palette. `net = (acoustic − text) /
    /// (acoustic + text)` is in `[-1, +1]`; we interpolate the
    /// neutral midpoint toward `acousticRGB` for positive `net`
    /// and toward `textRGB` for negative. Returns neutral when
    /// neither side contributed (defensive — the caller filters
    /// these out earlier, but keep the function total).
    static func contributionColor(acoustic: Float, text: Float) -> Color {
        let total = Double(acoustic + text)
        guard total > 1e-6 else {
            return Color(
                red: neutralRGB.r,
                green: neutralRGB.g,
                blue: neutralRGB.b
            )
        }
        let net = (Double(acoustic) - Double(text)) / total
        let endpoint = net >= 0 ? acousticRGB : textRGB
        let t = abs(net)
        let r = neutralRGB.r * (1 - t) + endpoint.r * t
        let g = neutralRGB.g * (1 - t) + endpoint.g * t
        let b = neutralRGB.b * (1 - t) + endpoint.b * t
        return Color(red: r, green: g, blue: b)
    }

    /// Compute the (acoustic, text) fusion share for one row.
    /// Falls back to "this side carries 100%" when only one
    /// modality was present, mirroring the row inspector's
    /// `fusionWeightSummary` logic.
    private func shareFor(
        _ utt: UtteranceEstimate
    ) -> (acoustic: Float, text: Float) {
        let hasAcoustic = utt.acousticCategorical != nil
        let hasText = utt.plutchik != nil
        switch (hasAcoustic, hasText) {
        case (false, false):
            return (acoustic: 0, text: 0)
        case (true, false):
            return (acoustic: 1, text: 0)
        case (false, true):
            return (acoustic: 0, text: 1)
        case (true, true):
            if let share = LateFusion.defaultLabelFusionShare(
                acoustic: utt.acousticCategorical,
                plutchik: utt.plutchik,
                asrConfidence: utt.asrConfidence ?? 0.5
            ) {
                return share
            }
            return LateFusion.defaultVAFusionShare(
                asrConfidence: utt.asrConfidence ?? 0.5
            )
        }
    }
}
