import SwiftUI
import Fusion

/// Per-utterance horizontal bar showing how much each modality
/// contributed to the fused-label argmax for that row. Each row's
/// slot on the strip spans its `[start, end]` audio range; the slot
/// is split horizontally into an acoustic-tinted segment and a
/// text-tinted segment, sized by
/// `LateFusion.defaultLabelFusionShare`. Rows with only one
/// modality fall back to a full slot in that modality's color.
///
/// Sits below the emotion-timeline strip in the transcript pane so
/// the three strips read top-down as: speaker → emotion label →
/// modality balance. The X axis is shared across all three.
struct FusionContributionStrip: View {
    let utterances: [UtteranceEstimate]
    let totalDuration: TimeInterval

    private static let height: CGFloat = 6
    /// Color for the acoustic share (W2V2 + emotion2vec). Matches
    /// the row's "Acoustic SER" section header convention — a
    /// neutral blue that doesn't compete with the speaker palette
    /// in the strip above.
    private static let acousticTint = Color(red: 0.30, green: 0.55, blue: 0.85)
    /// Color for the text share (DeBERTa / Apple FM). Warm orange,
    /// distinct from any speaker tint that might appear in the
    /// diarizer strip; the two strips don't share a meaning so
    /// keeping the palettes orthogonal is intentional.
    private static let textTint = Color(red: 0.95, green: 0.62, blue: 0.30)
    /// Fill opacity of the per-slot segments. Lower than the
    /// diarizer strip's runs so the strip reads as supporting
    /// detail, not a peer-level signal.
    private static let segmentOpacity: Double = 0.78
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
                        slot(share: share)
                            .frame(width: geometry.width, height: Self.height)
                            .offset(x: geometry.x)
                    }
                }
            }
            .clipShape(Capsule())
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

    @ViewBuilder
    private func slot(share: (acoustic: Float, text: Float)) -> some View {
        let total = max(share.acoustic + share.text, 1e-6)
        let acousticFraction = CGFloat(share.acoustic / total)
        GeometryReader { slotGeo in
            HStack(spacing: 0) {
                Rectangle()
                    .fill(Self.acousticTint.opacity(Self.segmentOpacity))
                    .frame(width: slotGeo.size.width * acousticFraction)
                Rectangle()
                    .fill(Self.textTint.opacity(Self.segmentOpacity))
            }
        }
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
