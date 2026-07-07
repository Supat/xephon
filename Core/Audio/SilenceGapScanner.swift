import Foundation

/// Frame-RMS silence-gap detection over 16 kHz mono Float32
/// samples. Used by the re-evaluation flow to MEASURE temporal pad
/// boundaries instead of guessing them: the front pad snaps to the
/// silence immediately preceding the utterance's true onset, and
/// the short-utterance path cuts its back-extension at the first
/// sustained gap after the fragment — one ASR pass at a measured
/// boundary instead of a grow-1s-and-rerun-ASR loop.
///
/// Deliberately NOT a neural VAD: gap-finding needs "is anything
/// audible here", not "is this speech", and a 20 ms RMS gate with
/// an adaptive threshold is deterministic, dependency-free, and
/// costs microseconds. The threshold adapts to the chunk's own
/// noise floor (10th-percentile frame RMS) so quiet-speaker
/// recordings — the project's known hard case — aren't judged
/// against an absolute level. Degrades safely at the extremes:
/// all-speech chunks produce no gaps (floor ≈ speech level pushes
/// the threshold above everything → callers fall back to their
/// fixed-pad behavior), all-silence chunks produce one giant gap.
public enum SilenceGapScanner {
    /// One sustained silent region, in buffer-local seconds.
    public struct Gap: Sendable, Equatable {
        public let start: TimeInterval
        public let end: TimeInterval
        public var duration: TimeInterval { end - start }
    }

    /// RMS analysis frame. 20 ms resolves inter-word pauses without
    /// flickering on intra-phoneme energy dips.
    public static let frameSec: TimeInterval = 0.02
    /// Minimum sustained silence to count as a gap. 300 ms sits
    /// above Japanese mora-boundary micro-pauses (< 150 ms) and
    /// below turn-taking / sentence pauses (typically > 400 ms).
    public static let defaultMinGapSec: TimeInterval = 0.3

    /// All sustained gaps in `samples`, ascending. A silent run
    /// touching the end of the buffer counts (the caller may be
    /// mid-file; the read window just ended in silence).
    public static func gaps(
        in samples: [Float],
        sampleRate: Double,
        minGapSec: TimeInterval = defaultMinGapSec
    ) -> [Gap] {
        guard sampleRate > 0, !samples.isEmpty else { return [] }
        let frameLen = max(1, Int(frameSec * sampleRate))
        let frameCount = samples.count / frameLen
        guard frameCount >= 3 else { return [] }

        var rms = [Float](repeating: 0, count: frameCount)
        for f in 0..<frameCount {
            var sum: Float = 0
            let base = f * frameLen
            for i in base..<(base + frameLen) {
                let s = samples[i]
                sum += s * s
            }
            rms[f] = (sum / Float(frameLen)).squareRoot()
        }

        // Adaptive threshold: noise floor × 3, with a small
        // fraction of the loud end as a secondary floor (covers
        // recordings whose "silence" carries room tone well above
        // digital zero), and an absolute epsilon so dithered
        // digital silence doesn't defeat the floor estimate.
        let sorted = rms.sorted()
        let floor = sorted[frameCount / 10]
        let loud = sorted[min(frameCount - 1, frameCount * 95 / 100)]
        let threshold = max(floor * 3, loud * 0.02, 1e-5)

        var result: [Gap] = []
        var runStart: Int? = nil
        let minFrames = max(1, Int(minGapSec / frameSec))
        for f in 0..<frameCount {
            if rms[f] < threshold {
                if runStart == nil { runStart = f }
            } else if let start = runStart {
                if f - start >= minFrames {
                    result.append(Gap(
                        start: TimeInterval(start) * frameSec,
                        end: TimeInterval(f) * frameSec
                    ))
                }
                runStart = nil
            }
        }
        if let start = runStart, frameCount - start >= minFrames {
            result.append(Gap(
                start: TimeInterval(start) * frameSec,
                end: TimeInterval(samples.count) / sampleRate
            ))
        }
        return result
    }
}
