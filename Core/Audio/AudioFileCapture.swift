import Foundation
@preconcurrency import AVFoundation
import XephonLogging

/// `AudioCapture` implementation that streams an audio file as if it were
/// being captured live. Reads `fileURL` in fixed-size chunks, resamples to
/// the pipeline's 16 kHz mono Float32 format, and yields chunks paced to
/// real-time so the rest of the pipeline (SpeechAnalyzer's volatile previews,
/// per-segment SER, the live UI) behaves exactly as it does for the mic.
///
/// Both `raw` and `processed` streams carry the same content — the speech
/// boost EQ doesn't apply to pre-recorded material. Input selection is
/// disabled while a file is the active source.
public actor AudioFileCapture: AudioCapture {
    private let fileURL: URL
    private let chunkFrames: AVAudioFrameCount
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var pumpTask: Task<Void, Never>?
    private var isAccessingScopedResource = false

    public init(fileURL: URL, chunkFrames: AVAudioFrameCount = 4096) {
        self.fileURL = fileURL
        self.chunkFrames = chunkFrames
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

        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
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
            // Real-time pace so SpeechAnalyzer's volatile→final stabilization
            // and the SER pipeline behave as they do for live capture.
            try? await Task.sleep(nanoseconds: UInt64(chunkSecondsAtFile * 1_000_000_000))
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
