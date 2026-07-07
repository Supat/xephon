import Foundation
import Testing
@testable import Xephon
import Diarization

/// Pins the binary-search-windowed `dominantSpeakerInSegments`
/// overload to the original full-scan algorithm's semantics. The
/// windowed form exists because the batch mismatch sweeps
/// (`TranscriptList.speakerMismatchedIDs`,
/// `TranscriptFilterModel.mismatchedUtteranceIDs`) call the vote
/// once per utterance against the whole cumulative timeline — the
/// old sort-per-call + scan-every-started-segment-per-sample shape
/// hard-froze the first render after loading a `.xph` that carried
/// a long session's persisted timeline. The optimization must be
/// invisible: same winner, same tie-breaks, same nearest-midpoint
/// fallback for zero-overlap ranges.
@Suite("Dominant-speaker vote windowing")
struct DominantSpeakerVoteTests {

    /// The pre-optimization algorithm, verbatim, as the oracle.
    private func referenceVote(
        _ segments: [DiarizedSegment],
        from start: TimeInterval,
        to end: TimeInterval,
        fallback: String
    ) -> String {
        guard !segments.isEmpty, end > start else { return fallback }
        let sorted = segments.sorted { $0.start < $1.start }
        let dt = AnalysisPipeline.speakerVoteSampleStepSec
        let sampleCount = min(
            AnalysisPipeline.speakerVoteSampleCountMax,
            max(
                AnalysisPipeline.speakerVoteSampleCountMin,
                Int(((end - start) / dt).rounded(.up))
            )
        )
        let step = (end - start) / TimeInterval(sampleCount)
        var votes: [String: Int] = [:]
        var upperBound = 0
        for i in 0..<sampleCount {
            let t = start + (TimeInterval(i) + 0.5) * step
            while upperBound < sorted.count && sorted[upperBound].start <= t {
                upperBound += 1
            }
            var instant: [String: Int] = [:]
            for j in 0..<upperBound where t <= sorted[j].end {
                instant[sorted[j].speakerID, default: 0] += 1
            }
            if let winner = instant.max(by: { ($0.value, $1.key) < ($1.value, $0.key) })?.key {
                votes[winner, default: 0] += 1
            }
        }
        if let mode = votes.max(by: { ($0.value, $1.key) < ($1.value, $0.key) })?.key {
            return mode
        }
        let mid = (start + end) / 2
        return sorted.min(by: {
            abs(($0.start + $0.end) / 2 - mid) < abs(($1.start + $1.end) / 2 - mid)
        })?.speakerID ?? fallback
    }

    /// Deterministic LCG so the generated timeline is stable across
    /// runs (failures must be reproducible).
    private struct SeededGenerator: RandomNumberGenerator {
        var state: UInt64
        mutating func next() -> UInt64 {
            state = state &* 6364136223846793005 &+ 1442695040888963407
            return state
        }
    }

    /// A timeline shaped like the continuous diarizer's cumulative
    /// output: overlapping ~2–10 s observations, several speakers,
    /// segments appended roughly-but-not-exactly in start order.
    private func syntheticTimeline(
        durationSec: TimeInterval,
        rng: inout SeededGenerator
    ) -> [DiarizedSegment] {
        var segments: [DiarizedSegment] = []
        var cursor: TimeInterval = 0
        while cursor < durationSec {
            let speaker = String(format: "S%02d", Int.random(in: 1...6, using: &rng))
            let length = TimeInterval.random(in: 0.4...10.0, using: &rng)
            let jitter = TimeInterval.random(in: -1.5...0.5, using: &rng)
            let start = max(0, cursor + jitter)
            segments.append(DiarizedSegment(
                speakerID: speaker,
                start: start,
                end: start + length
            ))
            cursor += TimeInterval.random(in: 0.3...2.0, using: &rng)
        }
        return segments
    }

    @Test func windowedMatchesReferenceOnSyntheticTimeline() {
        var rng = SeededGenerator(state: 0x5EED)
        let timeline = syntheticTimeline(durationSec: 600, rng: &rng)
        let sorted = timeline.sorted { $0.start < $1.start }
        let maxDur = timeline.lazy.map { $0.end - $0.start }.max() ?? 0

        // Utterance-shaped query ranges across (and past) the
        // timeline, including sub-second rows and long spans.
        var queryStart: TimeInterval = 0
        while queryStart < 640 {
            let length = TimeInterval.random(in: 0.2...12.0, using: &rng)
            let windowed = AnalysisPipeline.dominantSpeakerInSegments(
                sortedByStart: sorted,
                maxSegmentDuration: maxDur,
                from: queryStart,
                to: queryStart + length,
                fallback: "FB"
            )
            let reference = referenceVote(
                timeline,
                from: queryStart,
                to: queryStart + length,
                fallback: "FB"
            )
            #expect(
                windowed == reference,
                "range [\(queryStart), \(queryStart + length)]: windowed=\(windowed) reference=\(reference)"
            )
            queryStart += TimeInterval.random(in: 0.5...4.0, using: &rng)
        }
    }

    @Test func zeroOverlapFallsBackToNearestMidpoint() {
        let timeline = [
            DiarizedSegment(speakerID: "S01", start: 0, end: 4),
            DiarizedSegment(speakerID: "S02", start: 10, end: 14),
        ]
        // Range beyond every segment: no votes; nearest midpoint is
        // S02's (12 vs 2) — both the convenience and windowed forms
        // must agree.
        let viaConvenience = AnalysisPipeline.dominantSpeakerInSegments(
            timeline, from: 100, to: 101, fallback: "FB"
        )
        let viaWindowed = AnalysisPipeline.dominantSpeakerInSegments(
            sortedByStart: timeline,
            maxSegmentDuration: 4,
            from: 100,
            to: 101,
            fallback: "FB"
        )
        #expect(viaConvenience == "S02")
        #expect(viaWindowed == "S02")
    }

    @Test func emptyAndDegenerateInputsReturnFallback() {
        #expect(AnalysisPipeline.dominantSpeakerInSegments(
            [], from: 0, to: 1, fallback: "FB"
        ) == "FB")
        let one = [DiarizedSegment(speakerID: "S01", start: 0, end: 1)]
        #expect(AnalysisPipeline.dominantSpeakerInSegments(
            one, from: 5, to: 5, fallback: "FB"
        ) == "FB")
        #expect(AnalysisPipeline.dominantSpeakerInSegments(
            sortedByStart: one,
            maxSegmentDuration: 1,
            from: 5,
            to: 5,
            fallback: "FB"
        ) == "FB")
    }
}
