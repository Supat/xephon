import SwiftUI
import Diarization
import XephonUtilities

/// Per-row windowed view of the diarizer timeline. Renders the
/// segments overlapping `[utteranceStart - pad, utteranceEnd + pad]`,
/// with the utterance's own range outlined so the user can read at a
/// glance what the diarizer was hearing right before, during, and
/// right after this row's audio. Same per-instant majority sweep as
/// `DiarizationTimelineStrip` so the row strip and the session strip
/// at the top of the pane tell a consistent story.
///
/// Hidden entirely when no segment overlaps the window — common for
/// rows that finalized before the first continuous-diarize fire
/// (~10 s of audio), where surfacing an empty track would just add
/// noise.
struct UtteranceDiarizationStrip: View {
    let segments: [DiarizedSegment]
    let utteranceStart: TimeInterval
    let utteranceEnd: TimeInterval

    /// Context window padding on each side of the utterance. 1.5 s
    /// is just long enough to see the trailing edge of the prior
    /// speaker and the leading edge of the next, without making
    /// short utterances visually dominated by empty padding.
    private static let padSec: TimeInterval = 1.5
    /// Strip height. Thinner than the session-level strip (12 pt) —
    /// this is row chrome, not a primary surface, so it has to read
    /// without crowding the transcript text above it.
    private static let height: CGFloat = 5
    /// Sample interval for the per-instant majority sweep. 50 ms
    /// keeps run boundaries crisp at this strip's narrow pixel
    /// budget; the per-session strip's 200 ms is too coarse here
    /// because the whole window is only a few seconds wide.
    private static let sampleStepSec: TimeInterval = 0.05

    var body: some View {
        let windowStart = max(0, utteranceStart - Self.padSec)
        let windowEnd = utteranceEnd + Self.padSec
        let windowDuration = windowEnd - windowStart
        // Cheap pre-filter: drop everything that can't possibly touch
        // the window before the more expensive sweep runs.
        let windowed: [DiarizedSegment] = segments.compactMap { seg in
            let s = max(seg.start, windowStart)
            let e = min(seg.end, windowEnd)
            guard s < e else { return nil }
            return DiarizedSegment(
                speakerID: seg.speakerID,
                start: s - windowStart,
                end: e - windowStart
            )
        }
        if windowed.isEmpty || windowDuration <= 0 {
            EmptyView()
        } else {
            let runs = DiarizationTimelineStrip.majorityRuns(
                segments: windowed,
                totalDuration: windowDuration,
                sampleStep: Self.sampleStepSec
            )
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.secondary.opacity(Self.trackTintOpacity))
                        .frame(height: Self.height)
                    ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                        let x = geo.size.width * CGFloat(run.start / windowDuration)
                        let w = geo.size.width * CGFloat((run.end - run.start) / windowDuration)
                        Rectangle()
                            .fill(speakerTint(for: run.speakerID).opacity(Self.runFillOpacity))
                            .frame(width: max(1, w), height: Self.height)
                            .offset(x: x)
                    }
                }
                // Outline the utterance's own [start, end] within the
                // window so the user can distinguish "what the
                // diarizer heard during the row" from "what it heard
                // in the padding before/after."
                .overlay(alignment: .leading) {
                    let utterLocalStart = max(0, utteranceStart - windowStart)
                    let utterLocalEnd = min(windowDuration, utteranceEnd - windowStart)
                    let x = geo.size.width * CGFloat(utterLocalStart / windowDuration)
                    let w = geo.size.width * CGFloat((utterLocalEnd - utterLocalStart) / windowDuration)
                    RoundedRectangle(cornerRadius: 1.5, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.7), lineWidth: 0.75)
                        .frame(width: max(2, w), height: Self.height)
                        .offset(x: x)
                }
                .clipShape(Capsule())
            }
            .frame(height: Self.height)
            .accessibilityLabel(
                String(
                    format: String(localized: "row.diarization.a11y"),
                    Int(utteranceStart),
                    Int(utteranceEnd)
                )
            )
        }
    }

    private static let trackTintOpacity: Double = 0.10
    private static let runFillOpacity: Double = 0.80
}
