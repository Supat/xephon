import Foundation
@preconcurrency import AVFoundation
import Speech
import Audio
import XephonLogging

// Streaming variant of SpeechAnalyzerTranscriber. Keeps a single SpeechAnalyzer
// alive across the recording session, feeds audio buffers as they arrive, and
// emits ASRSegments only after `volatileRangeChangedHandler` reports that their
// range has been committed (i.e., the model is no longer revising them).
//
// In Apple's iOS 26 SpeechAnalyzer model:
//   - Each Result has a CMTimeRange describing the audio span it transcribes.
//   - The "volatile range" is the trailing region the model may still revise.
//   - A Result is final when its `range.end <= volatileRange.start`.
public actor StreamingSpeechAnalyzerTranscriber: StreamingTranscriber {
    public let locale: Locale

    private nonisolated(unsafe) var analyzer: SpeechAnalyzer?
    private var inputCont: AsyncStream<AnalyzerInput>.Continuation?
    private var outputCont: AsyncStream<ASRSegment>.Continuation?
    private var pending: [Double: SpeechTranscriber.Result] = [:]
    private var emittedKeys: Set<Double> = []
    private var resultDrainer: Task<Void, Never>?
    private var analyzerStartTask: Task<Void, Never>?
    private var targetFormat: AVAudioFormat?

    public init(locale: Locale = Locale(identifier: "ja_JP")) {
        self.locale = locale
    }

    public func start() async throws -> AsyncStream<ASRSegment> {
        guard SpeechTranscriber.isAvailable else {
            throw ASRError.neuralEngineUnavailable
        }
        guard let resolvedLocale = await SpeechTranscriber.supportedLocale(equivalentTo: locale) else {
            throw ASRError.unsupportedLocale(locale.identifier)
        }

        let transcriber = SpeechTranscriber(
            locale: resolvedLocale,
            transcriptionOptions: [],
            reportingOptions: [],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        try await ensureAssetInstalled(for: transcriber, locale: resolvedLocale)

        // Pick a format the transcriber accepts; fall back to 16 kHz mono Float32
        // (which is what the rest of the pipeline produces).
        let formats = await transcriber.availableCompatibleAudioFormats
        self.targetFormat = formats.first ?? AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        )

        let (inputStream, inputCont) = AsyncStream<AnalyzerInput>.makeStream()
        let (outputStream, outputCont) = AsyncStream<ASRSegment>.makeStream()
        self.inputCont = inputCont
        self.outputCont = outputCont
        self.pending.removeAll()
        self.emittedKeys.removeAll()

        let analyzer = SpeechAnalyzer(modules: [transcriber], options: nil)
        self.analyzer = analyzer

        // Volatile range moves forward each time a chunk of audio is committed.
        // When that happens, every cached result whose end is now ≤ the new
        // volatile.start is a finalized sentence — emit it.
        await analyzer.setVolatileRangeChangedHandler { [weak self] newRange, _, _ in
            let boundary = newRange.start.seconds
            Task { await self?.flushFinalized(below: boundary) }
        }

        // Drain results from the transcriber. We replace prior revisions of
        // the same range; the volatile-range handler decides when to emit.
        resultDrainer = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    await self?.appendResult(result)
                }
            } catch {
                AppLog.asr.error("streaming drainer failed: \(String(describing: error), privacy: .public)")
            }
        }

        // analyzer.start blocks until the input sequence finishes. We run it
        // in a background task and let `finish()` close the input.
        analyzerStartTask = Task { [outputCont] in
            do {
                try await analyzer.start(inputSequence: inputStream)
            } catch {
                AppLog.asr.error("streaming analyzer.start failed: \(String(describing: error), privacy: .public)")
                outputCont.finish()
            }
        }

        AppLog.asr.info("Streaming SpeechAnalyzer started (locale=\(resolvedLocale.identifier, privacy: .public))")
        return outputStream
    }

    public func feed(_ buffer: AudioChunk) async {
        guard let inputCont, let targetFormat else { return }
        guard let pcm = makePCMBuffer(samples: buffer.samples, target: targetFormat) else { return }
        // bufferStartTime is left nil; the analyzer infers timeline from sample-rate sequencing.
        inputCont.yield(AnalyzerInput(buffer: pcm))
    }

    public func finish() async {
        inputCont?.finish()
        do {
            try await analyzer?.finalizeAndFinishThroughEndOfInput()
        } catch {
            AppLog.asr.error("streaming finalize failed: \(String(describing: error), privacy: .public)")
        }
        // Anything still pending is now final by definition.
        flushFinalized(below: .infinity)
        outputCont?.finish()
        resultDrainer?.cancel()
        analyzerStartTask?.cancel()
        cleanup()
    }

    // MARK: - State updates (actor-isolated)

    private func appendResult(_ result: SpeechTranscriber.Result) {
        let startKey = result.range.start.seconds
        guard !emittedKeys.contains(startKey) else { return }
        pending[startKey] = result
    }

    private func flushFinalized(below boundary: Double) {
        let finalized = pending.values
            .filter { $0.range.end.seconds <= boundary && !emittedKeys.contains($0.range.start.seconds) }
            .sorted { $0.range.start.seconds < $1.range.start.seconds }
        for result in finalized {
            let segment = ASRSegment(
                text: String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines),
                start: result.range.start.seconds,
                end: result.range.end.seconds,
                confidence: nil
            )
            guard !segment.text.isEmpty else {
                pending.removeValue(forKey: result.range.start.seconds)
                emittedKeys.insert(result.range.start.seconds)
                continue
            }
            outputCont?.yield(segment)
            pending.removeValue(forKey: result.range.start.seconds)
            emittedKeys.insert(result.range.start.seconds)
        }
    }

    private func cleanup() {
        analyzer = nil
        inputCont = nil
        outputCont = nil
        targetFormat = nil
        pending.removeAll()
        emittedKeys.removeAll()
    }

    // MARK: - Asset install

    private func ensureAssetInstalled(
        for transcriber: SpeechTranscriber,
        locale: Locale
    ) async throws {
        let status = await AssetInventory.status(forModules: [transcriber])
        switch status {
        case .installed:
            return
        case .unsupported:
            throw ASRError.unsupportedLocale(locale.identifier)
        case .supported, .downloading:
            do {
                if let request = try await AssetInventory.assetInstallationRequest(supporting: [transcriber]) {
                    AppLog.asr.info("Downloading SpeechAnalyzer asset for \(locale.identifier, privacy: .public)…")
                    try await request.downloadAndInstall()
                }
            } catch {
                throw ASRError.modelUnavailable(reason: "asset install: \(error)")
            }
        @unknown default:
            throw ASRError.modelUnavailable(reason: "unknown AssetInventory status")
        }
    }

    // MARK: - PCM buffer construction

    private func makePCMBuffer(samples: [Float], target: AVAudioFormat) -> AVAudioPCMBuffer? {
        let isAlreadyTarget = target.sampleRate == PipelineAudio.sampleRate
            && target.channelCount == AVAudioChannelCount(PipelineAudio.channelCount)
            && target.commonFormat == .pcmFormatFloat32
            && !target.isInterleaved
        if isAlreadyTarget {
            return Self.fillBuffer(samples: samples, format: target)
        }
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ),
        let source = Self.fillBuffer(samples: samples, format: sourceFormat),
        let converter = AVAudioConverter(from: sourceFormat, to: target)
        else { return nil }

        let ratio = target.sampleRate / sourceFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(source.frameLength) * ratio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: outCapacity) else { return nil }

        var error: NSError?
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let block: AVAudioConverterInputBlock = { _, status in
            if once.fired { status.pointee = .endOfStream; return nil }
            once.fired = true
            status.pointee = .haveData
            return source
        }
        let result = converter.convert(to: outBuffer, error: &error, withInputFrom: block)
        return result == .error ? nil : outBuffer
    }

    private static func fillBuffer(samples: [Float], format: AVAudioFormat) -> AVAudioPCMBuffer? {
        guard let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else { return nil }
        pcm.frameLength = AVAudioFrameCount(samples.count)
        guard let dst = pcm.floatChannelData?[0] else { return nil }
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                dst.update(from: base, count: samples.count)
            }
        }
        return pcm
    }
}
