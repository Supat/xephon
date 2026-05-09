import Foundation
@preconcurrency import AVFoundation
import Speech
import Audio
import XephonLogging

// Streaming variant of SpeechAnalyzerTranscriber. Keeps a single SpeechAnalyzer
// alive across the recording session, feeds audio buffers as they arrive, and
// emits one ASRSegment per finalized result.
//
// Canonical pattern per Apple WWDC25 sample code and FluidInference/swift-scribe:
//   - reportingOptions includes `.volatileResults` so the transcriber emits
//     both volatile and final results.
//   - For each result, check `result.isFinal`:
//       * isFinal == true  → committed sentence chunk; emit downstream.
//                            Final results are non-overlapping by contract.
//       * isFinal == false → rolling preview of trailing audio; replace the
//                            volatile-text snapshot used by the live UI.
//   - No need for volatileRangeChangedHandler, no result-dictionary, no
//     range-overlap dedup. The previous design that tried to reconstruct
//     finalization from `volatileRangeChangedHandler` produced the duplicated
//     transcripts users saw.
public actor StreamingSpeechAnalyzerTranscriber: StreamingTranscriber {
    public let locale: Locale

    private nonisolated(unsafe) var analyzer: SpeechAnalyzer?
    private var inputCont: AsyncStream<AnalyzerInput>.Continuation?
    private var outputCont: AsyncStream<ASRSegment>.Continuation?
    private var volatileTextValue: String = ""
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
            reportingOptions: [.volatileResults],
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

        // Bounded input queue with retry-on-drop in `feed()` below. The
        // default `.unbounded` policy lets PCM buffers accumulate
        // forever when the analyzer drains slower than the producer
        // yields — particularly under fast-pace file analysis, where
        // the file pump runs 8x faster than SpeechAnalyzer can possibly
        // process. Each `AnalyzerInput` wraps a heap-allocated
        // AVAudioPCMBuffer (~20–50 KB), so a few minutes of backlog is
        // hundreds of MB. `.bufferingOldest` returns `.dropped(_)`
        // rather than evicting in place, which `feed()` catches and
        // retries — that turns the unbounded queue into real
        // backpressure on the upstream pump. 64 buffers ≈ 16 s of audio
        // at 4096-frame chunks, ample headroom for normal jitter
        // without sacrificing the bound.
        let (inputStream, inputCont) = AsyncStream<AnalyzerInput>.makeStream(
            bufferingPolicy: .bufferingOldest(64)
        )
        let (outputStream, outputCont) = AsyncStream<ASRSegment>.makeStream()
        self.inputCont = inputCont
        self.outputCont = outputCont
        self.volatileTextValue = ""

        let analyzer = SpeechAnalyzer(modules: [transcriber], options: nil)
        self.analyzer = analyzer

        // Drain results. Finals are emitted as ASRSegments; volatiles update
        // the rolling preview text.
        resultDrainer = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    await self?.handleResult(result)
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
        let input = AnalyzerInput(buffer: pcm)
        // Retry-on-drop loop: with `.bufferingOldest(64)`, a yield into
        // a full queue returns `.dropped(_)` rather than overwriting an
        // existing entry. We wait briefly for the analyzer to drain a
        // slot and retry, never silently losing audio. Mirrors the
        // backpressure pattern in `AudioFileCapture.yieldWithBackpressure`.
        while !Task.isCancelled {
            switch inputCont.yield(input) {
            case .enqueued, .terminated:
                return
            case .dropped:
                try? await Task.sleep(nanoseconds: 5_000_000)
            @unknown default:
                return
            }
        }
    }

    public func finish() async {
        inputCont?.finish()
        do {
            try await analyzer?.finalizeAndFinishThroughEndOfInput()
        } catch {
            AppLog.asr.error("streaming finalize failed: \(String(describing: error), privacy: .public)")
        }
        // Wait for the drainer to consume any tail finals from the analyzer
        // before we close the segment stream — otherwise downstream consumers
        // miss the last sentence of the session.
        await resultDrainer?.value
        outputCont?.finish()
        analyzerStartTask?.cancel()
        cleanup()
    }

    // MARK: - State updates (actor-isolated)

    public var volatileText: String {
        get async { volatileTextValue }
    }

    private func handleResult(_ result: SpeechTranscriber.Result) {
        let text = String(result.text.characters).trimmingCharacters(in: .whitespacesAndNewlines)
        let kind = result.isFinal ? "FINAL" : "vol  "
        AppLog.asr.debug("\(kind, privacy: .public) [\(result.range.start.seconds, privacy: .public)–\(result.range.end.seconds, privacy: .public)s] \"\(text, privacy: .public)\"")

        if result.isFinal {
            volatileTextValue = ""
            guard !text.isEmpty else { return }
            let segment = ASRSegment(
                text: text,
                start: result.range.start.seconds,
                end: result.range.end.seconds,
                confidence: SpeechAttributes.averageConfidence(in: result.text)
            )
            outputCont?.yield(segment)
        } else {
            volatileTextValue = text
        }
    }

    private func cleanup() {
        analyzer = nil
        inputCont = nil
        outputCont = nil
        targetFormat = nil
        volatileTextValue = ""
        resultDrainer = nil
        analyzerStartTask = nil
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
