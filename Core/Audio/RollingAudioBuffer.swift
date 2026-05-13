import Foundation

/// Bounded rolling buffer over a 16 kHz mono Float32 capture stream.
/// Encapsulates the trim-before-append, deep-copy-on-snapshot, and
/// origin-tracking invariants that previously lived as scattered
/// methods + comments on RecordingController.
///
/// Shape of the contract:
///
/// 1. `append(_:)` enforces `count ≤ maxSamples` *before* adding new
///    samples, so the array never overshoots its reserved capacity.
///    Without this, Swift Array's 2× growth strategy produces multi-MB
///    reallocation attempts that fail under memory pressure
///    (`Fatal error: failed to allocate <N> bytes`). Trimming first
///    keeps capacity pinned to the initial `reserveCapacity` slot.
///
/// 2. `snapshotForDiarization()` deep-copies into a fresh buffer
///    rather than aliasing `samples`. CoW would otherwise hand the
///    snapshot a buffer that shares backing storage with the live
///    array; the next `append` would trigger CoW and reallocate
///    `samples` with a new (potentially larger) capacity, defeating
///    invariant #1 across many snapshot/append cycles.
///
/// 3. `slice(start:end:)` materializes a fresh array (Swift's slice
///    initializer copies), so callers can hand the result to a
///    background actor without contending for `samples`.
///
/// 4. `origin` advances every time samples are dropped from the head.
///    Audio-time → buffer-index mapping uses
///    `(audioTime - origin) * sampleRate`, so callers don't need to
///    track trimming themselves.
public struct RollingAudioBuffer: Sendable {
    public let sampleRate: Double
    /// Hard cap on the rolling capture buffer, sized for the worst
    /// case where ASR stalls and the segment-driven trim doesn't
    /// fire — the raw pump's append still respects this and drops
    /// the oldest samples to keep `count` bounded.
    public let maxSamples: Int
    /// How much audio behind the latest finalized segment to keep
    /// for cross-segment speaker diarization context.
    public let contextSeconds: TimeInterval

    public private(set) var samples: [Float] = []
    /// Audio-time (seconds) of `samples[0]`. Advances each time
    /// samples are dropped from the head.
    public private(set) var origin: TimeInterval = 0

    public var count: Int { samples.count }
    public var isEmpty: Bool { samples.isEmpty }

    public init(
        maxSeconds: TimeInterval,
        contextSeconds: TimeInterval,
        sampleRate: Double = PipelineAudio.sampleRate
    ) {
        self.sampleRate = sampleRate
        self.maxSamples = Int(maxSeconds * sampleRate)
        self.contextSeconds = contextSeconds
    }

    /// Reset for a fresh session. Keeps the underlying buffer's
    /// capacity (so the first append doesn't reallocate) and
    /// re-reserves up to `maxSamples`.
    public mutating func reset() {
        samples.removeAll(keepingCapacity: true)
        // Pre-reserve so Array's exponential capacity growth never
        // triggers a multi-MB allocation when the buffer approaches
        // the cap. `reserveCapacity` is a no-op if the existing
        // capacity already covers `maxSamples`.
        samples.reserveCapacity(maxSamples)
        origin = 0
    }

    /// Append new samples, trimming the head first if the result
    /// would exceed `maxSamples`. The pre-trim discipline is the
    /// invariant that pins capacity to the initial reservation —
    /// see the type's doc comment for why post-trim doesn't work.
    public mutating func append(_ incoming: [Float]) {
        let projected = samples.count + incoming.count
        if projected > maxSamples {
            // `min` guards against a hypothetical capture chunk
            // larger than `maxSamples` itself — without it
            // `removeFirst` would trap on n > count.
            let excess = min(projected - maxSamples, samples.count)
            samples.removeFirst(excess)
            origin += Double(excess) / sampleRate
        }
        samples.append(contentsOf: incoming)
    }

    /// Drop processed audio from the head, keeping the last
    /// `contextSeconds` of audio behind `boundary` as cross-segment
    /// speaker context for the diarizer. Caller is responsible for
    /// having sliced any segments below `boundary` first; the SER
    /// task already owns its own copy of the slice, so trimming is
    /// safe.
    public mutating func trimProcessed(below boundary: TimeInterval) {
        let cutoff = boundary - contextSeconds
        let dropSeconds = cutoff - origin
        guard dropSeconds > 0 else { return }
        let frames = min(Int(dropSeconds * sampleRate), samples.count)
        guard frames > 0 else { return }
        samples.removeFirst(frames)
        origin += Double(frames) / sampleRate
    }

    /// Materialize a fresh `[Float]` for the requested audio-time
    /// range. Returns an empty chunk when the range falls outside
    /// the current buffer. The slice is a real copy (`Array(_:)`
    /// from a slice copies), so the caller can hand it off to a
    /// background actor without contending for the live buffer.
    public func slice(start: TimeInterval, end: TimeInterval) -> AudioChunk {
        let localStart = max(0, start - origin)
        let localEnd   = max(localStart, end - origin)
        let startIndex = min(Int(localStart * sampleRate), samples.count)
        let endIndex   = min(Int(localEnd * sampleRate), samples.count)
        guard startIndex < endIndex else {
            return AudioChunk(samples: [], sampleRate: sampleRate, timestamp: start)
        }
        let slice = Array(samples[startIndex..<endIndex])
        return AudioChunk(samples: slice, sampleRate: sampleRate, timestamp: start)
    }

    /// Sendable snapshot of the entire current buffer for FluidAudio
    /// diarization. Forces a deep copy via
    /// `unsafeUninitializedCapacity` rather than letting Swift
    /// Array's CoW share storage with `samples`. See invariant #2
    /// in the type-level doc comment.
    public func snapshotForDiarization() -> AudioChunk {
        let isolated = samples.withUnsafeBufferPointer { src -> [Float] in
            guard let base = src.baseAddress else { return [] }
            return [Float](unsafeUninitializedCapacity: src.count) { dst, initializedCount in
                if let dstBase = dst.baseAddress {
                    dstBase.update(from: base, count: src.count)
                }
                initializedCount = src.count
            }
        }
        return AudioChunk(
            samples: isolated,
            sampleRate: sampleRate,
            timestamp: origin
        )
    }

    /// Deep-copy snapshot of the trailing `seconds` of audio. Used
    /// by the continuous-diarization path which only needs a short
    /// (~10 s) sliding window, not the full rolling buffer. Same
    /// CoW-isolation guarantee as `snapshotForDiarization` so the
    /// snapshot doesn't alias the live buffer.
    public func snapshotTail(seconds: TimeInterval) -> AudioChunk {
        guard !samples.isEmpty else {
            return AudioChunk(samples: [], sampleRate: sampleRate, timestamp: origin)
        }
        let totalSeconds = Double(samples.count) / sampleRate
        if seconds >= totalSeconds {
            return snapshotForDiarization()
        }
        let count = Int(seconds * sampleRate)
        let startIndex = samples.count - count
        let timestamp = origin + Double(startIndex) / sampleRate
        let isolated = [Float](unsafeUninitializedCapacity: count) { dst, initializedCount in
            samples.withUnsafeBufferPointer { src in
                if let dstBase = dst.baseAddress, let srcBase = src.baseAddress {
                    dstBase.update(from: srcBase + startIndex, count: count)
                }
            }
            initializedCount = count
        }
        return AudioChunk(samples: isolated, sampleRate: sampleRate, timestamp: timestamp)
    }
}
