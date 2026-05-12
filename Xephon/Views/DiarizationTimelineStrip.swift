import SwiftUI
import Diarization
import Fusion

/// Per-session diarizer timeline — a thin horizontal strip showing
/// which speaker the diarizer thinks is talking across the whole
/// recording, with the currently-selected utterance's window
/// highlighted. Sits below the speaker filter chip bar in the
/// transcript pane.
///
/// The cumulative timeline has ~5 overlapping observations per
/// audio moment (the continuous-diarize task fires every 2 s on a
/// 10 s window). To render a clean strip we compute the per-instant
/// majority winner at fixed-rate sample points and compress
/// consecutive same-speaker samples into runs. That voting matches
/// `AnalysisPipeline.dominantSpeakerInSegments`'s rule, so the strip
/// reflects the same speaker assignment the row labels do.
struct DiarizationTimelineStrip: View {
    /// Cumulative timeline snapshot at render time. May be empty
    /// (no diarization run yet, or freshly loaded session — the
    /// timeline isn't persisted).
    let segments: [DiarizedSegment]
    /// Conversation duration in seconds. Derived by the caller from
    /// `fileTotalAudioDuration` when available, else the max segment
    /// end — either way it sets the strip's X axis scale.
    let totalDuration: TimeInterval
    /// Selected utterance's `[start, end]` window if any, used to
    /// outline the strip region corresponding to that row.
    let selectedRange: (start: TimeInterval, end: TimeInterval)?

    /// Strip height. Thin enough to fit between the chip bar and
    /// the transcript list without crowding either; tall enough to
    /// keep narrow runs legible.
    private static let height: CGFloat = 12
    /// Sample interval for the per-instant majority sweep. 200 ms
    /// is well below the diarizer's segment granularity and gives
    /// O(samples × ~5 active) ≈ 5×duration_seconds operations —
    /// trivial for any realistic session.
    private static let sampleStepSec: TimeInterval = 0.2

    var body: some View {
        let runs = Self.majorityRuns(
            segments: segments,
            totalDuration: totalDuration,
            sampleStep: Self.sampleStepSec
        )
        GeometryReader { geo in
            ZStack(alignment: .leading) {
                // Backdrop for the "no diarized speech" regions of
                // the timeline. Subtle so the colored runs read as
                // the foreground.
                RoundedRectangle(cornerRadius: Self.height / 2, style: .continuous)
                    .fill(Color.secondary.opacity(0.08))
                ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                    let x = geo.size.width * CGFloat(run.start / totalDuration)
                    let w = geo.size.width * CGFloat((run.end - run.start) / totalDuration)
                    Rectangle()
                        .fill(speakerTint(for: run.speakerID).opacity(0.7))
                        .frame(width: max(1, w), height: Self.height)
                        .offset(x: x)
                }
                if let sel = selectedRange, totalDuration > 0 {
                    let clampedStart = max(0, min(sel.start, totalDuration))
                    let clampedEnd = max(clampedStart, min(sel.end, totalDuration))
                    let x = geo.size.width * CGFloat(clampedStart / totalDuration)
                    let w = geo.size.width * CGFloat((clampedEnd - clampedStart) / totalDuration)
                    RoundedRectangle(cornerRadius: 2, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.85), lineWidth: 1.5)
                        .frame(width: max(2, w), height: Self.height)
                        .offset(x: x)
                }
            }
            .clipShape(RoundedRectangle(cornerRadius: Self.height / 2, style: .continuous))
        }
        .frame(height: Self.height)
    }

    /// Per-instant majority winner sweep across `[0, totalDuration]`,
    /// compressing consecutive same-speaker samples into runs.
    /// Returns `[]` when the timeline is empty or the duration is
    /// zero. Linear in `samples × active_per_sample`; the active
    /// set is maintained with a single sweep pointer over the
    /// sorted segments.
    static func majorityRuns(
        segments: [DiarizedSegment],
        totalDuration: TimeInterval,
        sampleStep: TimeInterval
    ) -> [(speakerID: String, start: TimeInterval, end: TimeInterval)] {
        guard !segments.isEmpty, totalDuration > 0, sampleStep > 0 else { return [] }
        let sorted = segments.sorted { $0.start < $1.start }
        let sampleCount = max(1, Int((totalDuration / sampleStep).rounded(.up)))
        var runs: [(speakerID: String, start: TimeInterval, end: TimeInterval)] = []
        var currentSpeaker: String?
        var runStart: TimeInterval = 0
        // Sweep pointer past which no segment has yet started.
        var sweepHead = 0
        // Set of segment indices currently overlapping the sample
        // point. Rebuilt incrementally — segments enter when their
        // `start` ≤ t, exit when their `end` < t.
        var active: [Int] = []
        for i in 0..<sampleCount {
            let t = TimeInterval(i) * sampleStep
            while sweepHead < sorted.count && sorted[sweepHead].start <= t {
                active.append(sweepHead)
                sweepHead += 1
            }
            active.removeAll { sorted[$0].end <= t }
            var votes: [String: Int] = [:]
            for idx in active {
                votes[sorted[idx].speakerID, default: 0] += 1
            }
            let winner = votes.max(by: { $0.value < $1.value })?.key
            if winner != currentSpeaker {
                if let prev = currentSpeaker {
                    runs.append((speakerID: prev, start: runStart, end: t))
                }
                currentSpeaker = winner
                runStart = t
            }
        }
        if let last = currentSpeaker {
            runs.append((speakerID: last, start: runStart, end: totalDuration))
        }
        return runs
    }
}
