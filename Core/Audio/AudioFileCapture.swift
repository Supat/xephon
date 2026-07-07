import Foundation
@preconcurrency import AVFoundation
import XephonLogging

/// `AudioCapture` implementation that streams an audio file through the
/// same pipeline used for the mic. Reads `fileURL` in fixed-size chunks,
/// resamples to 16 kHz mono Float32, and yields chunks to the raw and
/// processed streams **as fast as the downstream consumers can drain them**.
///
/// Pacing is not real-time. The pump uses `.bufferingOldest(N)` continuations
/// and retries on `.dropped`, so the producer naturally throttles to the
/// slowest downstream stage (typically ASR). This eliminates the silent
/// audio loss that real-time pacing produced on long files when
/// `SpeechAnalyzer`'s drain rate dipped below 1× — at the cost of no
/// side-channel speaker playback during analysis (the file's audible
/// timeline would no longer match the analysis cursor, so playback was
/// removed entirely).
///
/// The speech-boost EQ doesn't apply to pre-recorded material, but the
/// ASR-branch speech leveler does: when enabled, the `processed` stream
/// is run through `SoftwareSpeechLeveler` (the sample-domain twin of the
/// live engine's DynamicsProcessor node) while `raw` stays untouched —
/// same branch discipline as the mic graph. Input selection is disabled
/// while a file is the active source.
public actor AudioFileCapture: AudioCapture {
    private let fileURL: URL
    private let chunkFrames: AVAudioFrameCount
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var pumpTask: Task<Void, Never>?
    private var isAccessingScopedResource = false
    /// Set when the pump ends on a mid-file read error; surfaced
    /// through `captureEndReason()` so the controller's stream-end
    /// watcher can distinguish "file fully processed" from "file
    /// silently truncated" — previously indistinguishable, and the
    /// user saw a partial transcript presented as complete.
    private var endReason: CaptureEndReason?

    /// ASR-branch leveler toggle, propagated from RecordingController
    /// at session start (startFromFile) and live-flippable mid-run —
    /// the pump reads this per chunk. Default off so directly
    /// constructed instances (tests) stay pass-through.
    private var speechLevelerEnabled = false

    public init(
        fileURL: URL,
        chunkFrames: AVAudioFrameCount = 4096
    ) {
        self.fileURL = fileURL
        self.chunkFrames = chunkFrames
    }

    public func start() async throws -> CaptureStreams {
        // FileImporter hands us a security-scoped URL; the wrapper below
        // ensures we can read its bytes for the duration of the pump.
        if fileURL.startAccessingSecurityScopedResource() {
            isAccessingScopedResource = true
        }

        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: fileURL)
        } catch {
            stopAccessingScopedResource()
            throw AudioError.underlying(error)
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ),
        let converter = AVAudioConverter(from: file.processingFormat, to: outputFormat) else {
            stopAccessingScopedResource()
            throw AudioError.unsupportedFormat(
                expected: "16 kHz mono Float32",
                got: String(describing: file.processingFormat)
            )
        }
        converter.primeMethod = .none

        // `.bufferingOldest(N)` + retry-on-drop is the backpressure
        // discipline. Yielding into a full queue returns `.dropped(_)`,
        // which the pump catches with a short sleep + retry. Producer
        // self-paces to whatever the slowest consumer (usually ASR) can
        // sustain. N=64 ≈ 6 s of audio at 4096-frame chunks: ample
        // headroom for the analyzer to ride out transient stalls
        // without the pump pausing.
        let bufferingPolicy: AsyncStream<AudioChunk>.Continuation.BufferingPolicy =
            .bufferingOldest(64)
        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        self.rawCont = rawCont
        self.processedCont = processedCont

        let totalFrames = file.length
        AppLog.audio.info(
            "File capture started: \(self.fileURL.lastPathComponent, privacy: .public) (\(totalFrames, privacy: .public) frames @ \(file.processingFormat.sampleRate, privacy: .public) Hz)"
        )

        let chunkFrames = self.chunkFrames
        pumpTask = Task { [weak self] in
            await self?.pump(
                file: file,
                converter: converter,
                outputFormat: outputFormat,
                chunkFrames: chunkFrames,
                rawCont: rawCont,
                processedCont: processedCont
            )
        }

        return CaptureStreams(raw: rawStream, processed: processedStream)
    }

    public func stop() async {
        pumpTask?.cancel()
        await pumpTask?.value
        pumpTask = nil
        rawCont?.finish()
        processedCont?.finish()
        rawCont = nil
        processedCont = nil
        stopAccessingScopedResource()
        AppLog.audio.info("File capture stopped")
    }

    // MARK: - AudioCapture conformance (no-ops for file mode)

    public func availableInputs() async -> [AudioInputDescription] { [] }
    public func currentInput() async -> AudioInputDescription? { nil }
    public func setPreferredInput(_ uid: String?) async throws {}
    public var isSpeechBoostEnabled: Bool { get async { false } }
    public func setSpeechBoostEnabled(_ enabled: Bool) async {}
    public func captureEndReason() async -> CaptureEndReason? { endReason }
    public var isSpeechLevelerEnabled: Bool { get async { speechLevelerEnabled } }
    public func setSpeechLevelerEnabled(_ enabled: Bool) async {
        speechLevelerEnabled = enabled
        AppLog.audio.info("File-capture speech leveler \(enabled ? "ON" : "OFF", privacy: .public)")
    }

    // MARK: - File pump

    private func pump(
        file: AVAudioFile,
        converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        chunkFrames: AVAudioFrameCount,
        rawCont: AsyncStream<AudioChunk>.Continuation,
        processedCont: AsyncStream<AudioChunk>.Continuation
    ) async {
        let inputFormat = file.processingFormat
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inputFormat, frameCapacity: chunkFrames) else {
            rawCont.finish()
            processedCont.finish()
            return
        }
        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate

        // One streaming leveler for the whole file so the envelope
        // carries across chunk boundaries (a per-chunk reset would
        // re-attack on every chunk edge).
        let leveler = SoftwareSpeechLeveler(sampleRate: PipelineAudio.sampleRate)
        var elapsed: TimeInterval = 0
        while !Task.isCancelled {
            inputBuffer.frameLength = 0
            do {
                try file.read(into: inputBuffer, frameCount: chunkFrames)
            } catch {
                // AVAudioFile.read THROWS at exact end-of-file for
                // some containers (YouTube-sourced AAC observed
                // on-device) instead of returning zero frames — a
                // blanket fileReadFailed here flagged a fully-read
                // file as truncated. Only frames genuinely left
                // unread count as truncation; a sub-chunk remainder
                // is a trailing partial packet (< ~85 ms, inaudible)
                // and completes normally.
                let remaining = file.length - file.framePosition
                if remaining >= AVAudioFramePosition(chunkFrames) {
                    AppLog.audio.warning(
                        "file read error with \(remaining, privacy: .public) frames unread: \(String(describing: error), privacy: .public)"
                    )
                    endReason = .fileReadFailed(String(describing: error))
                } else {
                    AppLog.audio.info(
                        "file read ended at EOF (\(remaining, privacy: .public) frames of trailing packet); treating as normal completion"
                    )
                }
                break
            }
            if inputBuffer.frameLength == 0 { break }
            // Advance the file-time clock HERE, not at the loop
            // bottom: the conversion guards below `continue` past
            // the old accumulation point, and a single skipped
            // chunk (converter error / zero output frames) shifted
            // every later chunk's timestamp ~one chunk early —
            // misaligning ASR/diarization/SER for the rest of the
            // file. The read already consumed these frames, so the
            // clock must advance whether or not they convert; the
            // chunk itself is stamped with the PRE-advance time
            // (its start position in the file).
            let chunkStart = elapsed
            elapsed += Double(inputBuffer.frameLength) / inputFormat.sampleRate

            let outputCapacity = AVAudioFrameCount(
                (Double(inputBuffer.frameLength) * sampleRateRatio).rounded(.up)
            )
            guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
                continue
            }
            var convError: NSError?
            // `@unchecked Sendable` is safe: one-shot latch consumed only
            // by the AVAudioConverter input block, which the converter
            // calls serially on a single thread per `convert(...)` call.
            final class Once: @unchecked Sendable { var fired = false }
            let once = Once()
            let inputBlock: AVAudioConverterInputBlock = { _, status in
                if once.fired { status.pointee = .noDataNow; return nil }
                once.fired = true
                status.pointee = .haveData
                return inputBuffer
            }
            let result = converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
            guard result != .error,
                  let channelData = outBuffer.floatChannelData,
                  outBuffer.frameLength > 0 else {
                continue
            }
            let frameCount = Int(outBuffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
            let chunk = AudioChunk(
                samples: samples,
                sampleRate: PipelineAudio.sampleRate,
                timestamp: chunkStart
            )
            // Yield to both streams with retry-on-drop. Each call returns
            // promptly when the consumer has space; under sustained
            // backpressure (analyzer slow), it loops on `.dropped` with
            // a brief sleep so no audio is lost.
            await Self.yieldWithBackpressure(chunk, to: rawCont)
            // ASR branch: levelled copy when enabled; raw pass-through
            // otherwise. `speechLevelerEnabled` is actor state and the
            // pump is actor-isolated, so a mid-run Settings toggle
            // takes effect on the next chunk.
            let processedChunk = speechLevelerEnabled
                ? AudioChunk(
                    samples: leveler.process(samples),
                    sampleRate: PipelineAudio.sampleRate,
                    timestamp: chunkStart
                )
                : chunk
            await Self.yieldWithBackpressure(processedChunk, to: processedCont)
        }

        rawCont.finish()
        processedCont.finish()
    }

    /// Yield with retry-on-drop. Under `.bufferingOldest(N)`, attempting
    /// to yield into a full buffer returns `.dropped(value:)` instead of
    /// kicking out an existing element. Sleeping briefly and retrying
    /// gives the consumer a chance to drain a slot, turning the queue
    /// into real backpressure on the upstream pump.
    private static func yieldWithBackpressure(
        _ chunk: AudioChunk,
        to cont: AsyncStream<AudioChunk>.Continuation
    ) async {
        while !Task.isCancelled {
            switch cont.yield(chunk) {
            case .enqueued, .terminated:
                return
            case .dropped:
                try? await Task.sleep(nanoseconds: 5_000_000)
            @unknown default:
                return
            }
        }
    }

    private func stopAccessingScopedResource() {
        if isAccessingScopedResource {
            fileURL.stopAccessingSecurityScopedResource()
            isAccessingScopedResource = false
        }
    }
}
