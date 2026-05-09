import Foundation
@preconcurrency import AVFoundation
import XephonLogging

/// `AudioCapture` implementation that streams an audio file through the
/// same pipeline used for the mic. Reads `fileURL` in fixed-size chunks,
/// resamples to 16 kHz mono Float32, and yields chunks to the raw and
/// processed streams.
///
/// Two pacing modes:
///   - `realTimePacing: true`  — sleeps the chunk's source duration after
///     each yield, so volatile previews and stage animations look the same
///     as a live recording. A 5-min file analyzes in 5 min.
///   - `realTimePacing: false` (default) — yields back-to-back so the
///     analyzer ingests audio as fast as it can decode. SpeechAnalyzer's
///     volatile/final logic doesn't depend on wall-clock time, so finals
///     still stabilize correctly; the whole file finishes in roughly the
///     time it takes the analyzer + SER to process it (typically several
///     times faster than real time on M-series silicon).
///
/// Streams use unbounded buffering in fast mode so the pump never has to
/// drop chunks while downstream catches up — bounded by file size, since
/// the pump exits at EOF.
///
/// Both `raw` and `processed` streams carry the same content — the speech
/// boost EQ doesn't apply to pre-recorded material. Input selection is
/// disabled while a file is the active source.
public actor AudioFileCapture: AudioCapture {
    private let fileURL: URL
    private let chunkFrames: AVAudioFrameCount
    private let realTimePacing: Bool
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var pumpTask: Task<Void, Never>?
    private var isAccessingScopedResource = false

    public init(
        fileURL: URL,
        chunkFrames: AVAudioFrameCount = 4096,
        realTimePacing: Bool = false
    ) {
        self.fileURL = fileURL
        self.chunkFrames = chunkFrames
        self.realTimePacing = realTimePacing
    }

    public func start() async throws -> CaptureStreams {
        // FileImporter hands us a security-scoped URL; the wrapper below
        // ensures we can read its bytes for the duration of playback.
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

        // Fast mode pumps faster than the consumer can drain, so unbounded
        // buffering avoids dropping chunks. Memory is bounded by file size
        // (16-bit-equivalent at 16 kHz mono Float32 = ~3.8 MB / minute).
        // Real-time mode keeps the bounded policy used for the live mic
        // since the producer is naturally rate-limited.
        let bufferingPolicy: AsyncStream<AudioChunk>.Continuation.BufferingPolicy =
            realTimePacing ? .bufferingNewest(64) : .unbounded
        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        self.rawCont = rawCont
        self.processedCont = processedCont

        let totalFrames = file.length
        AppLog.audio.info(
            "File capture started: \(self.fileURL.lastPathComponent, privacy: .public) (\(totalFrames, privacy: .public) frames @ \(file.processingFormat.sampleRate, privacy: .public) Hz)"
        )

        let chunkFrames = self.chunkFrames
        let realTimePacing = self.realTimePacing
        pumpTask = Task { [weak self] in
            await self?.pump(
                file: file,
                converter: converter,
                outputFormat: outputFormat,
                chunkFrames: chunkFrames,
                realTimePacing: realTimePacing,
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

    // MARK: - File pump

    private func pump(
        file: AVAudioFile,
        converter: AVAudioConverter,
        outputFormat: AVAudioFormat,
        chunkFrames: AVAudioFrameCount,
        realTimePacing: Bool,
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
        let chunkSecondsAtFile = Double(chunkFrames) / inputFormat.sampleRate

        var elapsed: TimeInterval = 0
        while !Task.isCancelled {
            inputBuffer.frameLength = 0
            do {
                try file.read(into: inputBuffer, frameCount: chunkFrames)
            } catch {
                AppLog.audio.warning("file read error: \(String(describing: error), privacy: .public)")
                break
            }
            if inputBuffer.frameLength == 0 { break }

            let outputCapacity = AVAudioFrameCount(
                (Double(inputBuffer.frameLength) * sampleRateRatio).rounded(.up)
            )
            guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
                continue
            }
            var convError: NSError?
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
                timestamp: elapsed
            )
            // Both streams emit the same content — there's no upstream EQ to
            // diverge here, and downstream slicing/feeding code already
            // assumes parallel timelines.
            rawCont.yield(chunk)
            processedCont.yield(chunk)

            elapsed += chunkSecondsAtFile
            if realTimePacing {
                // Match wall-clock to file timeline so volatile previews and
                // stage animations look identical to a live recording.
                try? await Task.sleep(nanoseconds: UInt64(chunkSecondsAtFile * 1_000_000_000))
            } else {
                // Yield the actor briefly so we don't starve consumers on the
                // same MainActor / scheduling pool. Zero-duration cooperative
                // yield, no real wall-clock delay.
                await Task.yield()
            }
        }

        rawCont.finish()
        processedCont.finish()
    }

    private func stopAccessingScopedResource() {
        if isAccessingScopedResource {
            fileURL.stopAccessingSecurityScopedResource()
            isAccessingScopedResource = false
        }
    }
}
