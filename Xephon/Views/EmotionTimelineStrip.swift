import SwiftUI
import Fusion

/// Per-session emotion timeline — a thin horizontal strip showing
/// each utterance's fused top label across the whole recording.
/// Sits directly under the diarizer timeline strip and uses the
/// same X-axis scale so a tap or a glance lines up across both
/// strips at the same audio time.
///
/// Unlike the diarizer timeline (~5 overlapping observations per
/// audio moment, needs a per-instant majority sweep), emotion data
/// is per-utterance and non-overlapping. We just walk the sorted
/// utterance list, emit one run per utterance whose `fusedTopLabel`
/// is set, and coalesce consecutive same-label runs so a long
/// stretch of the same emotion renders as one continuous band
/// instead of a series of touching slabs. Utterances with no
/// `fusedTopLabel` produce gaps (the base track shows through).
struct EmotionTimelineStrip: View {
    let utterances: [UtteranceEstimate]
    /// Conversation duration in seconds — same value the diarizer
    /// strip uses, so both strips share an X axis.
    let totalDuration: TimeInterval
    /// Selected utterance's `[start, end]` window if any, used to
    /// outline the strip region corresponding to that row.
    let selectedRange: (start: TimeInterval, end: TimeInterval)?
    /// Fires when the user taps a point on the strip; the
    /// `TimeInterval` is the audio-time the tap location maps to,
    /// clamped to `[0, totalDuration]`. ContentView resolves it to
    /// the nearest utterance and asks `TranscriptList` to scroll
    /// there. Optional so the strip is useful even without a tap-
    /// routing parent.
    let onTapAtTime: ((TimeInterval) -> Void)?

    /// Half the diarizer strip's 12 pt height. Keeps the secondary
    /// strip visually subordinate to the primary speaker strip,
    /// while still being thick enough that small runs stay legible.
    private static let height: CGFloat = 6
    /// Tint strength for the base track's glass effect. Matches
    /// the diarizer strip's track so both strips read as one unit.
    private static let trackTintOpacity: Double = 0.08
    /// Fill opacity for each emotion run. Same as the diarizer
    /// strip's run fill so a coincident speaker-and-emotion change
    /// reads at the same visual weight on both strips.
    private static let runFillOpacity: Double = 0.75

    var body: some View {
        let runs = Self.coalescedRuns(utterances: utterances)
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                Capsule()
                    .glassEffect(
                        .regular.tint(.secondary.opacity(Self.trackTintOpacity)),
                        in: Capsule()
                    )
                    .frame(height: Self.height)
                ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                    let x = geo.size.width * CGFloat(run.start / totalDuration)
                    let w = geo.size.width * CGFloat((run.end - run.start) / totalDuration)
                    Rectangle()
                        .fill(emotionTint(for: run.label).opacity(Self.runFillOpacity))
                        .frame(width: max(1, w), height: Self.height)
                        .offset(x: x)
                }
            }
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

    /// Walk the utterance list in start-time order, emit one run
    /// per utterance whose `fusedTopLabel` is set, and merge
    /// adjacent runs that share the same label (or are within a
    /// utterance-end-to-next-utterance-start gap small enough to
    /// be silence between two parts of the same emotional beat —
    /// we still merge across small gaps so a hesitant pause inside
    /// one mood doesn't fragment the band). Utterances without a
    /// fused top label produce no run; the base track shows
    /// through there. Returns `[]` on empty input so the
    /// rendering ForEach is a no-op.
    static func coalescedRuns(
        utterances: [UtteranceEstimate]
    ) -> [(label: String, start: TimeInterval, end: TimeInterval)] {
        guard !utterances.isEmpty else { return [] }
        let sorted = utterances.sorted { $0.start < $1.start }
        var runs: [(label: String, start: TimeInterval, end: TimeInterval)] = []
        for u in sorted {
            guard let label = u.fusedTopLabel, !label.isEmpty else { continue }
            if var last = runs.last, last.label == label, u.start <= last.end + Self.mergeGapSec {
                last.end = max(last.end, u.end)
                runs[runs.count - 1] = last
            } else {
                runs.append((label: label, start: u.start, end: u.end))
            }
        }
        return runs
    }

    /// Max gap (s) between two same-label utterances that still
    /// merges them into one run. Larger than a typical inter-
    /// utterance silence so a brief pause inside a continuous
    /// emotional stretch doesn't split the band, but small enough
    /// that two distant utterances that happen to share a label
    /// still render as two distinct runs.
    private static let mergeGapSec: TimeInterval = 0.75
}
