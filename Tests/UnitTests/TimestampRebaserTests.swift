import Foundation
import Testing
@testable import Audio

/// Pins the engine-rebuild rebasing in `TimestampRebaser`. The
/// load-bearing case is the USB-reconnect rebuild that anchored
/// ~64000s ahead in the field — a forward `sampleTime` discontinuity
/// the rebaser must clamp, or a single utterance ends up spanning
/// ~64000s and freezes the per-row diarization strip's O(duration)
/// majority sweep.
struct TimestampRebaserTests {

    @Test func forwardJumpBeyondCapReanchors() {
        let r = TimestampRebaser()
        #expect(r.sessionTime(forRaw: 100.0) == 0)
        #expect(abs(r.sessionTime(forRaw: 122.8) - 22.8) < 1e-6)

        // Fresh engine after a USB reconnect: sampleTime jumps ~64000s
        // ahead. Must clamp to the prior session time, not produce a
        // 64000s leap.
        r.markEngineRebuildBoundary()
        let t = r.sessionTime(forRaw: 64_148.0)   // would-be ~64048s
        #expect(abs(t - 22.8) < 1e-6)
        #expect(r.sessionTime(forRaw: 64_148.5) > t)   // still monotonic
    }

    @Test func backwardResetReanchors() {
        let r = TimestampRebaser()
        _ = r.sessionTime(forRaw: 100.0)
        let last = r.sessionTime(forRaw: 110.0)        // 10s
        r.markEngineRebuildBoundary()
        let after = r.sessionTime(forRaw: 0.5)         // sampleTime reset to ~0
        #expect(after >= last)                          // no backward jump
    }

    @Test func smallRecoveryGapPreserved() {
        let r = TimestampRebaser()
        _ = r.sessionTime(forRaw: 100.0)
        _ = r.sessionTime(forRaw: 110.0)               // 10s
        r.markEngineRebuildBoundary()
        // A genuine ~2s recovery gap is real elapsed audio — kept.
        #expect(abs(r.sessionTime(forRaw: 112.0) - 12.0) < 1e-6)
    }
}
