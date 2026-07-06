import SwiftUI
import Diarization
import XephonUtilities

/// One contiguous speaker run inside an utterance row's strip,
/// expressed in window-local seconds (so the strip can render
/// without re-doing any of the windowing math). Produced by
/// `UtteranceDiarizationStrip.computeRuns` and passed in by the
/// parent — separating data from rendering lets `TranscriptList`
/// memoize the per-row run array across body re-evals so a
/// 500-row session doesn't re-sweep the timeline on every scroll
/// tick.
struct DiarizationRun: Hashable, Sendable {
    let speakerID: String
    let start: TimeInterval
    let end: TimeInterval
}

/// Per-row windowed view of the diarizer timeline. Renders the
/// segments overlapping `[utteranceStart - pad, utteranceEnd + pad]`,
/// with the utterance's own range outlined so the user can read at a
/// glance what the diarizer was hearing right before, during, and
/// right after this row's audio. Same per-instant majority sweep as
/// `DiarizationTimelineStrip` (with a stable tie-break) so the per-
/// row strip, the session strip, and the chip's speaker label all
/// resolve ties to the same speaker.
///
/// Hidden entirely when no segment overlaps the window — common for
/// rows that finalized before the first continuous-diarize fire
/// (~10 s of audio), where surfacing an empty track would just add
/// noise.
///
/// Computation vs rendering: `computeRuns` does the segment scan and
/// majority sweep; the view just paints rectangles. The parent
/// computes the runs once (memoized on the timeline version) and
/// hands the same array back into the view on subsequent renders.
struct UtteranceDiarizationStrip: View {
    let runs: [DiarizationRun]
    let utteranceStart: TimeInterval
    let utteranceEnd: TimeInterval

    /// Context window padding on each side of the utterance. 1.5 s
    /// is just long enough to see the trailing edge of the prior
    /// speaker and the leading edge of the next, without making
    /// short utterances visually dominated by empty padding.
    static let padSec: TimeInterval = 1.5
    /// Strip height. Thinner than the session-level strip (12 pt) —
    /// this is row chrome, not a primary surface, so it has to read
    /// without crowding the transcript text above it.
    private static let height: CGFloat = 5
    /// Sample interval for the per-instant majority sweep. 50 ms
    /// keeps run boundaries crisp at this strip's narrow pixel
    /// budget; the per-session strip's 200 ms is too coarse here
    /// because the whole window is only a few seconds wide.
    static let sampleStepSec: TimeInterval = 0.05

    /// Window bounds for an utterance, with leading clamp at 0 so
    /// short-prefix audio doesn't push the outline off-strip.
    /// Shared between `computeRuns` and the view so both agree on
    /// the coordinate origin.
    static func window(
        utteranceStart: TimeInterval,
        utteranceEnd: TimeInterval
    ) -> (start: TimeInterval, end: TimeInterval, duration: TimeInterval) {
        let s = max(0, utteranceStart - padSec)
        let e = utteranceEnd + padSec
        return (s, e, max(0, e - s))
    }

    /// Compute the windowed per-instant majority runs for one
    /// utterance. Output is in window-local seconds (0…duration),
    /// ready to render. Returns `[]` when no segment overlaps the
    /// window or the window has zero duration — the view then
    /// renders `EmptyView` and the row slot collapses.
    /// Convenience single-shot variant: sorts + measures the
    /// timeline itself. Fine for one-off calls; batch callers
    /// (TranscriptList's per-row memo) MUST pre-sort once and use
    /// the `sortedByStart:` overload below — sorting per row would
    /// reintroduce the O(rows × segments log segments) recompute
    /// this split exists to kill.
    static func computeRuns(
        segments: [DiarizedSegment],
        utteranceStart: TimeInterval,
        utteranceEnd: TimeInterval
    ) -> [DiarizationRun] {
        computeRuns(
            sortedByStart: segments.sorted { $0.start < $1.start },
            maxSegmentDuration: segments.lazy.map { $0.end - $0.start }.max() ?? 0,
            utteranceStart: utteranceStart,
            utteranceEnd: utteranceEnd
        )
    }

    /// Batch-optimized windowing: `segments` pre-sorted by `start`,
    /// `maxSegmentDuration` = the longest segment in the timeline.
    /// Candidates for the window are found with two binary searches
    /// — a segment can only overlap `[win.start, win.end)` if its
    /// start lies in `[win.start − maxSegmentDuration, win.end)` —
    /// so the per-row cost is O(log S + k) instead of O(S). With
    /// the memoized recompute firing on every diarizer tick (~2 s)
    /// during recording, the old full-scan grew quadratically with
    /// session length (rows × segments ≈ 10⁶ checks per tick at
    /// the one-hour mark); this caps k at the handful of
    /// observations overlapping one utterance's window.
    static func computeRuns(
        sortedByStart segments: [DiarizedSegment],
        maxSegmentDuration: TimeInterval,
        utteranceStart: TimeInterval,
        utteranceEnd: TimeInterval
    ) -> [DiarizationRun] {
        let win = window(
            utteranceStart: utteranceStart,
            utteranceEnd: utteranceEnd
        )
        guard win.duration > 0, !segments.isEmpty else { return [] }
        // Lower bound: first segment whose start could still reach
        // into the window given the longest observed duration.
        let minStart = win.start - maxSegmentDuration
        var lo = 0
        var hi = segments.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if segments[mid].start < minStart { lo = mid + 1 } else { hi = mid }
        }
        let lower = lo
        // Upper bound: first segment starting at/after the window's
        // end — nothing from there on can overlap.
        hi = segments.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if segments[mid].start < win.end { lo = mid + 1 } else { hi = mid }
        }
        let upper = lo
        guard lower < upper else { return [] }
        var windowed: [DiarizedSegment] = []
        windowed.reserveCapacity(upper - lower)
        for seg in segments[lower..<upper] {
            let s = max(seg.start, win.start)
            let e = min(seg.end, win.end)
            if s < e {
                windowed.append(DiarizedSegment(
                    speakerID: seg.speakerID,
                    start: s - win.start,
                    end: e - win.start
                ))
            }
        }
        guard !windowed.isEmpty else { return [] }
        let rawRuns = DiarizationTimelineStrip.majorityRuns(
            segments: windowed,
            totalDuration: win.duration,
            sampleStep: sampleStepSec
        )
        return rawRuns.map {
            DiarizationRun(speakerID: $0.speakerID, start: $0.start, end: $0.end)
        }
    }

    var body: some View {
        let win = Self.window(
            utteranceStart: utteranceStart,
            utteranceEnd: utteranceEnd
        )
        if runs.isEmpty || win.duration <= 0 {
            EmptyView()
        } else {
            GeometryReader { geo in
                ZStack(alignment: .leading) {
                    Capsule()
                        .fill(Color.secondary.opacity(Self.trackTintOpacity))
                        .frame(height: Self.height)
                    ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                        let x = geo.size.width * CGFloat(run.start / win.duration)
                        let w = geo.size.width * CGFloat((run.end - run.start) / win.duration)
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
                    let utterLocalStart = max(0, utteranceStart - win.start)
                    let utterLocalEnd = min(win.duration, utteranceEnd - win.start)
                    let x = geo.size.width * CGFloat(utterLocalStart / win.duration)
                    let w = geo.size.width * CGFloat((utterLocalEnd - utterLocalStart) / win.duration)
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
