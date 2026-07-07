import SwiftUI
import Fusion

/// Timeline strip marking WHERE color-tagged keywords occur in the
/// session — the keyword companion to the emotion and fusion
/// strips, sharing their X axis, selection overlay, and
/// tap-to-scroll behavior so a glance down the stack reads
/// "speaker / emotion / modality / keyword, at this audio time".
///
/// Each utterance that matches at least one TAGGED keyword renders
/// a slot at its time span; when several tags match the same
/// utterance, the slot splits into equal-width bands in keyword-
/// list order (same within-slot subdivision idea as the fusion
/// strip). Untagged keywords never paint — the strip stays empty
/// until the user assigns colors in the Keywords card.
struct KeywordTimelineStrip: View {
    let utterances: [UtteranceEstimate]
    /// utteranceID → ordered, color-deduped tags matched in that
    /// row. Computed + memoized by TranscriptFilterModel
    /// (`keywordTagMatches`) so this view stays render-only.
    let tagsByUtterance: [UUID: [KeywordTagColor]]
    let totalDuration: TimeInterval
    let selectedRange: (start: TimeInterval, end: TimeInterval)?
    let onTapAtTime: ((TimeInterval) -> Void)?

    private static let height: CGFloat = 6
    private static let trackTintOpacity: Double = 0.06

    var body: some View {
        TimelineStripCanvas(
            totalDuration: totalDuration,
            selectedRange: selectedRange,
            onTapAtTime: onTapAtTime,
            height: Self.height,
            trackTintOpacity: Self.trackTintOpacity
        ) { width in
            ForEach(utterances) { utt in
                if let tags = tagsByUtterance[utt.id],
                   !tags.isEmpty,
                   let slot = slotGeometry(for: utt, width: width) {
                    let band = slot.width / CGFloat(tags.count)
                    ForEach(Array(tags.enumerated()), id: \.offset) { i, tag in
                        Rectangle()
                            .fill(tag.color)
                            .frame(width: max(1, band), height: Self.height)
                            .offset(x: slot.x + band * CGFloat(i))
                    }
                }
            }
        }
    }

    /// Same slot math as FusionContributionStrip.
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
}
