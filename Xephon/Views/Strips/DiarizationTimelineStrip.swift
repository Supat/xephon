import SwiftUI
import Diarization
import Fusion
import XephonUtilities

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
    /// Fires when the user taps a point on the strip; the
    /// `TimeInterval` is the audio-time the tap location maps to,
    /// clamped to `[0, totalDuration]`. ContentView resolves it
    /// to the nearest utterance and asks `TranscriptList` to
    /// scroll there. Optional so the strip is useful even without
    /// a tap-routing parent.
    let onTapAtTime: ((TimeInterval) -> Void)?

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
                // Glass base track. Provides the Liquid Glass frame
                // for the strip (rounded capsule profile + edge
                // refraction + tinted backdrop) without the heavy
                // backdrop blur the per-run glasses introduced.
                Capsule()
                    .glassEffect(
                        .regular.tint(.secondary.opacity(Self.trackTintOpacity)),
                        in: Capsule()
                    )
                    .frame(height: Self.height)
                // Speaker runs are now flat-tinted fills, not
                // glass — they're contiguous across the timeline
                // so per-run glass blur was dominating the strip
                // and softening the speaker colors. Keeping them
                // as semi-translucent fills retains a hint of
                // glassiness from the track showing through at
                // low opacity, while the colors themselves read
                // crisply.
                ForEach(Array(runs.enumerated()), id: \.offset) { _, run in
                    let x = geo.size.width * CGFloat(run.start / totalDuration)
                    let w = geo.size.width * CGFloat((run.end - run.start) / totalDuration)
                    Rectangle()
                        .fill(speakerTint(for: run.speakerID).opacity(Self.runFillOpacity))
                        .frame(width: max(1, w), height: Self.height)
                        .offset(x: x)
                }
            }
            // Selection overlay — plain primary-color stroke
            // marking the focused range. Kept outside any glass
            // material so the edge stays crisp against the
            // colored speaker runs underneath.
            .overlay(alignment: .leading) {
                if let sel = selectedRange, totalDuration > 0 {
                    let clampedStart = sel.start.clamped(to: 0...totalDuration)
                    let clampedEnd = sel.end.clamped(to: clampedStart...totalDuration)
                    let x = geo.size.width * CGFloat(clampedStart / totalDuration)
                    let w = geo.size.width * CGFloat((clampedEnd - clampedStart) / totalDuration)
                    RoundedRectangle(cornerRadius: 3, style: .continuous)
                        .strokeBorder(Color.primary.opacity(0.9), lineWidth: 1.5)
                        .frame(width: max(2, w), height: Self.height)
                        .offset(x: x)
                }
            }
            // Clip the whole thing to the strip's capsule shape so
            // run rectangles that overlap the rounded ends inherit
            // the capsule profile.
            .clipShape(Capsule())
            // Whole-strip hit area for taps. `contentShape` is
            // needed so the gaps between the (mostly contiguous)
            // run rectangles still register, and so the
            // GeometryReader's coordinate space is what the gesture
            // reports against — `location.x / width` directly maps
            // to the strip's audio-time axis.
            .contentShape(Rectangle())
            .onTapGesture(coordinateSpace: .local) { location in
                guard let onTapAtTime, totalDuration > 0, geo.size.width > 0 else { return }
                let t = totalDuration * Double(location.x / geo.size.width)
                onTapAtTime(t.clamped(to: 0...totalDuration))
            }
        }
        .frame(height: Self.height)
    }

    /// Tint strength for the base track's glass effect. Subtle
    /// enough to let the colored runs dominate visually but strong
    /// enough that the strip is still visible against any
    /// background a user might set.
    private static let trackTintOpacity: Double = 0.08
    /// Fill opacity for each speaker run. Bolder than the track so
    /// the speaker color reads clearly; not 1.0 so a hint of the
    /// underlying glass track shows through and the strip retains
    /// the Liquid Glass feel without the heavy backdrop blur the
    /// per-run glass effect introduced.
    private static let runFillOpacity: Double = 0.75

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
