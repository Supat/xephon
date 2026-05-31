import Foundation
import Audio
import XephonLogging

/// Streaming wrapper around `Qwen3ASRTranscriber` for the live
/// recording path. Qwen3-ASR is fundamentally a one-shot
/// transcribe-this-buffer API — it doesn't produce incremental
/// results or per-token timestamps — so this wrapper buffers
/// incoming audio until it has accumulated `chunkSizeSeconds`
/// of samples, transcribes that chunk, and yields a single
/// `ASRSegment` covering the chunk's timeline range. Chunks
/// transcribe serially via a chained-Task queue, so emitted
/// segments arrive in chronological order even if a slow chunk
/// runs longer than the next chunk's accumulation time.
///
/// Wall-clock latency between speaking and seeing a segment is
/// roughly `chunkSizeSeconds + inference time`. On an M-class
/// iPad Pro, Qwen3-ASR inference is on the order of real-time
/// for short clips, so an 8 s chunk finishes ~8 s of recording
/// plus a few seconds of inference — segments lag the live
/// audio by ~10–15 s. That's tolerable for a secondary backend;
/// `StreamingSpeechAnalyzerTranscriber` remains the right
/// default for tight live preview.
///
/// No volatile-text preview — Qwen3 has no notion of partial
/// transcription, so the `volatileText` property always returns
/// an empty string. The transcript pane's live preview row stays
/// blank between chunk emissions when this backend is active.
public actor StreamingQwen3ASRTranscriber: StreamingTranscriber {
    public let locale: Locale
    private let inner: Qwen3ASRTranscriber

    /// Output continuation for the segment stream. Nil before
    /// `start()` and after `finish()`.
    private var outputCont: AsyncStream<ASRSegment>.Continuation?
    /// Accumulated audio samples that haven't yet been packaged
    /// into a chunk for transcription. Truncated to empty (with
    /// reserved capacity) when a chunk dispatches.
    private var audioBuffer: [Float] = []
    /// Sample rate of the most recent `feed`. The pipeline
    /// guarantees 16 kHz mono Float32 per CLAUDE.md, but the
    /// value is tracked dynamically so a future capture-format
    /// change doesn't silently mis-time chunks.
    private var sampleRate: Double = 16_000
    /// File-time stamp of the first sample currently in
    /// `audioBuffer`. Advances by the chunk's duration each
    /// time a chunk dispatches.
    private var bufferStartTime: TimeInterval = 0
    /// True once `feed(_:)` has been called at least once.
    /// Used to seed `bufferStartTime` from the first incoming
    /// chunk's timestamp rather than assuming zero.
    private var hasReceivedAudio: Bool = false
    /// Chain handle for the serial transcription queue. Each
    /// new chunk's task `await`s this before starting, then
    /// becomes the new tail. `finish()` awaits the tail to
    /// drain any in-flight transcription before closing the
    /// output stream.
    private var chainTail: Task<Void, Never>?

    /// Target audio duration per chunk. 8 s balances latency
    /// (delay before a segment shows up) against utterance
    /// integrity (smaller risks splitting sentences mid-word;
    /// larger leaves the user waiting). Tuned for conversational
    /// Japanese / English speech.
    private static let chunkSizeSeconds: Double = 8.0

    public init(locale: Locale = Locale(identifier: "ja_JP")) {
        self.locale = locale
        self.inner = Qwen3ASRTranscriber(locale: locale)
    }

    public func start() async throws -> AsyncStream<ASRSegment> {
        // Unbounded buffering: chunks emit at ~chunkSizeSeconds
        // cadence and the controller drains immediately, so a
        // bounded policy would drop segments only if the
        // controller's drainer fell behind by many chunks —
        // unlikely in practice and would be a separate bug.
        let (stream, cont) = AsyncStream<ASRSegment>.makeStream(
            bufferingPolicy: .unbounded
        )
        outputCont = cont
        audioBuffer.removeAll(keepingCapacity: true)
        hasReceivedAudio = false
        bufferStartTime = 0
        AppLog.asr.info("Qwen3 streaming session started (locale=\(self.locale.identifier, privacy: .public))")
        return stream
    }

    public func feed(_ buffer: AudioChunk) async {
        guard outputCont != nil else { return }
        if !hasReceivedAudio {
            hasReceivedAudio = true
            bufferStartTime = buffer.timestamp
        }
        sampleRate = buffer.sampleRate
        audioBuffer.append(contentsOf: buffer.samples)

        // Dispatch in a loop so a very long single `feed` (rare
        // — capture pumps at ~100 ms granularity normally — but
        // possible during file-mode catch-up) emits multiple
        // chunks rather than one giant one.
        while Double(audioBuffer.count) / sampleRate >= Self.chunkSizeSeconds {
            dispatchChunk(maxSamples: Int(Self.chunkSizeSeconds * sampleRate))
        }
    }

    public func finish() async {
        // Flush any tail audio as a final partial chunk so the
        // last few seconds of speech aren't lost.
        if !audioBuffer.isEmpty {
            dispatchChunk(maxSamples: audioBuffer.count)
        }
        // Drain the serial chain so all pending transcriptions
        // complete before closing the output stream — otherwise
        // the controller's drainer sees `.finish()` while
        // chunks are still in flight and they fall on the floor.
        await chainTail?.value
        outputCont?.finish()
        outputCont = nil
        chainTail = nil
        AppLog.asr.info("Qwen3 streaming session finished")
    }

    /// Snapshot `maxSamples` samples from the front of
    /// `audioBuffer`, advance `bufferStartTime` past them, and
    /// chain a transcription task onto the serial queue. Each
    /// task awaits the previous one's completion before
    /// starting, so segments emit in chronological order.
    private func dispatchChunk(maxSamples: Int) {
        guard let cont = outputCont else { return }
        let take = min(maxSamples, audioBuffer.count)
        guard take > 0 else { return }

        let chunkSamples = Array(audioBuffer.prefix(take))
        audioBuffer.removeFirst(take)
        let chunkStartTime = bufferStartTime
        let chunkSampleRate = sampleRate
        bufferStartTime = chunkStartTime + Double(take) / sampleRate

        let prior = chainTail
        let transcriber = inner
        chainTail = Task {
            // Wait for the prior chunk to finish so segments
            // arrive in order. Nil on the first dispatch — the
            // optional chain just no-ops.
            await prior?.value
            let chunk = AudioChunk(
                samples: chunkSamples,
                sampleRate: chunkSampleRate,
                timestamp: chunkStartTime
            )
            do {
                let segments = try await transcriber.transcribe(chunk)
                for segment in segments {
                    cont.yield(segment)
                }
            } catch {
                AppLog.asr.warning(
                    "Qwen3 streaming chunk failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
    }
}
