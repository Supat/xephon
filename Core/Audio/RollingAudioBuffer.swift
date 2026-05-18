import Foundation
import XephonUtilities

/// Bounded rolling buffer over a 16 kHz mono Float32 capture stream.
/// Encapsulates the trim-before-append, deep-copy-on-snapshot, and
/// file-time tracking invariants that previously lived as scattered
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
/// 4. **Indexing is file-time, not output-time.** Each `append(_:)`
///    records an anchor `(sampleIndex, fileTime)` from the incoming
///    chunk's `timestamp`. Slice/trim convert file-time → sample-index
///    via piecewise-linear interpolation across those anchors. This
///    closes the drift gap between SpeechAnalyzer's reported result
///    ranges (which are corrected to file-time in
///    `StreamingSpeechAnalyzerTranscriber.handleResult`) and the
///    sample-position math used to slice the rolling buffer. With
///    plain `i / sampleRate` indexing, a 30-min 44.1 kHz file
///    accumulates 1-3 s of mismatch between ASR ranges and the audio
///    fed to acoustic SER — slicing in file-time fixes that
///    end-to-end.
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

    /// `(sampleIndex, fileTime)` anchor: at `samples[sampleIndex]`, the
    /// source-file timeline reads `fileTime`. Maintained sorted by
    /// `sampleIndex` ascending, with `anchors.first?.sampleIndex == 0`
    /// when the buffer is non-empty.
    private struct Anchor: Sendable {
        var sampleIndex: Int
        var fileTime: TimeInterval
    }
    private var anchors: [Anchor] = []

    /// File-time of `samples[0]`. Derived from the head anchor.
    /// Zero when the buffer is empty (no audio yet).
    public var origin: TimeInterval {
        anchors.first?.fileTime ?? 0
    }

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
        samples.reserveCapacity(maxSamples)
        anchors.removeAll(keepingCapacity: false)
    }

    /// Append a captured chunk. Records a `(sampleIndex, fileTime)`
    /// anchor at the chunk's start so subsequent file-time queries
    /// (slice / trim / snapshotTail) can interpolate sample positions
    /// accurately even when the source file's sample-rate-conversion
    /// to 16 kHz accumulates per-chunk rounding drift.
    ///
    /// `chunk.timestamp` must be monotonic; non-monotonic chunks
    /// (rare AVAudioEngine glitches on route changes) are dropped on
    /// the floor — their samples are still appended, but the prior
    /// anchor's slope continues to govern interpolation through the
    /// dropped region.
    public mutating func append(_ chunk: AudioChunk) {
        let incomingCount = chunk.samples.count
        guard incomingCount > 0 else { return }

        // Trim head first to keep capacity pinned — see invariant #1.
        let projected = samples.count + incomingCount
        if projected > maxSamples {
            let excess = min(projected - maxSamples, samples.count)
            dropHead(excess)
        }

        // Record the anchor before appending so `sampleIndex` reflects
        // where this chunk's first sample lands in the buffer.
        let chunkStartIndex = samples.count
        let monotonic = anchors.last.map { chunk.timestamp > $0.fileTime } ?? true
        if monotonic {
            anchors.append(Anchor(sampleIndex: chunkStartIndex, fileTime: chunk.timestamp))
        }

        samples.append(contentsOf: chunk.samples)
    }

    /// Drop processed audio from the head, keeping the last
    /// `contextSeconds` of audio behind `boundary` (file-time) as
    /// cross-segment speaker context for the diarizer.
    public mutating func trimProcessed(below boundary: TimeInterval) {
        let cutoff = boundary - contextSeconds
        let cutoffIndex = indexForFileTime(cutoff)
        let frames = cutoffIndex.clamped(to: 0...samples.count)
        guard frames > 0 else { return }
        dropHead(frames)
    }

    /// Materialize a fresh `[Float]` for the requested file-time
    /// range. Returns an empty chunk when the range falls outside
    /// the current buffer.
    ///
    /// The returned chunk's `timestamp` is the file-time of its
    /// **actual** first sample — which may be later than `start` if
    /// the head has been evicted past `start`. Without that, a
    /// downstream consumer like `FluidAudioDiarizer.atTime` would
    /// anchor its segments to a time the audio doesn't correspond
    /// to.
    public func slice(start: TimeInterval, end: TimeInterval) -> AudioChunk {
        guard !samples.isEmpty else {
            return AudioChunk(samples: [], sampleRate: sampleRate, timestamp: start)
        }
        let startIdx = indexForFileTime(start).clamped(to: 0...samples.count)
        let endIdx = indexForFileTime(end).clamped(to: startIdx...samples.count)
        guard startIdx < endIdx else {
            return AudioChunk(samples: [], sampleRate: sampleRate, timestamp: start)
        }
        let slice = Array(samples[startIdx..<endIdx])
        let actualStart = fileTimeForIndex(startIdx)
        return AudioChunk(samples: slice, sampleRate: sampleRate, timestamp: actualStart)
    }

    /// Sendable snapshot of the entire current buffer for FluidAudio
    /// diarization. Forces a deep copy via
    /// `unsafeUninitializedCapacity` rather than letting Swift
    /// Array's CoW share storage with `samples`. The chunk's
    /// timestamp is the file-time of `samples[0]`.
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
    /// (~10 s) sliding window, not the full rolling buffer.
    /// `seconds` is wall-clock seconds; the slice is sized off the
    /// buffer's nominal sample rate (the diarizer doesn't care about
    /// per-chunk drift internally — it just needs the most recent
    /// audio in chronological order). The returned chunk's
    /// `timestamp` is the file-time of its first sample so the
    /// diarizer's `atTime` anchor lands in the correct timeline.
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
        let timestamp = fileTimeForIndex(startIndex)
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

    // MARK: - Anchor-based file-time math

    /// Map sample index → file-time using anchor interpolation.
    /// Outside the anchor range, extrapolates from the nearest pair's
    /// slope; with a single anchor, falls back to nominal sample-rate
    /// stepping (good enough until the second anchor lands).
    private func fileTimeForIndex(_ index: Int) -> TimeInterval {
        guard let first = anchors.first else {
            return Double(index) / sampleRate
        }
        if anchors.count == 1 {
            return first.fileTime + Double(index - first.sampleIndex) / sampleRate
        }
        // Binary-search for the first anchor with sampleIndex > index.
        var lo = 0
        var hi = anchors.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if anchors[mid].sampleIndex <= index {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        let upperIdx = lo
        if upperIdx == 0 {
            let a = anchors[0]
            let b = anchors[1]
            return interpolateFileTime(at: index, between: a, and: b)
        }
        if upperIdx == anchors.count {
            let a = anchors[anchors.count - 2]
            let b = anchors[anchors.count - 1]
            return interpolateFileTime(at: index, between: a, and: b)
        }
        let a = anchors[upperIdx - 1]
        let b = anchors[upperIdx]
        return interpolateFileTime(at: index, between: a, and: b)
    }

    /// Map file-time → sample index (inverse of `fileTimeForIndex`).
    private func indexForFileTime(_ t: TimeInterval) -> Int {
        guard let first = anchors.first else {
            return Int(t * sampleRate)
        }
        if anchors.count == 1 {
            return first.sampleIndex + Int((t - first.fileTime) * sampleRate)
        }
        var lo = 0
        var hi = anchors.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if anchors[mid].fileTime <= t {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        let upperIdx = lo
        if upperIdx == 0 {
            let a = anchors[0]
            let b = anchors[1]
            return interpolateIndex(at: t, between: a, and: b)
        }
        if upperIdx == anchors.count {
            let a = anchors[anchors.count - 2]
            let b = anchors[anchors.count - 1]
            return interpolateIndex(at: t, between: a, and: b)
        }
        let a = anchors[upperIdx - 1]
        let b = anchors[upperIdx]
        return interpolateIndex(at: t, between: a, and: b)
    }

    private func interpolateFileTime(at index: Int, between a: Anchor, and b: Anchor) -> TimeInterval {
        let span = b.sampleIndex - a.sampleIndex
        if span <= 0 { return a.fileTime }
        let t = Double(index - a.sampleIndex) / Double(span)
        return a.fileTime + t * (b.fileTime - a.fileTime)
    }

    private func interpolateIndex(at fileTime: TimeInterval, between a: Anchor, and b: Anchor) -> Int {
        let span = b.fileTime - a.fileTime
        if span <= 0 { return a.sampleIndex }
        let t = (fileTime - a.fileTime) / span
        return a.sampleIndex + Int(t * Double(b.sampleIndex - a.sampleIndex))
    }

    /// Drop `count` samples from the head, shifting anchors so
    /// `anchors[0].sampleIndex == 0` afterwards.
    private mutating func dropHead(_ count: Int) {
        guard count > 0, count <= samples.count else { return }
        // Compute the file-time at the new head BEFORE mutating anchors.
        let newOriginFileTime = fileTimeForIndex(count)
        samples.removeFirst(count)
        // Shift remaining anchors and drop those that now sit at
        // negative sample indices.
        anchors = anchors.compactMap { anchor in
            let shifted = anchor.sampleIndex - count
            guard shifted >= 0 else { return nil }
            return Anchor(sampleIndex: shifted, fileTime: anchor.fileTime)
        }
        // Ensure the head anchor pins sampleIndex 0 to the right
        // file-time. If the first remaining anchor isn't at index 0,
        // synthesize one from the interpolated origin.
        if anchors.first?.sampleIndex != 0 {
            anchors.insert(
                Anchor(sampleIndex: 0, fileTime: newOriginFileTime),
                at: 0
            )
        }
    }
}
