import Foundation
@preconcurrency import AVFoundation
import Speech
import Audio
import XephonLogging

// Primary ASR: Apple SpeechAnalyzer + SpeechTranscriber (iPadOS 26+, ja_JP).
//
// Caveats per CLAUDE.md:
//  - Requires 16-core Neural Engine (M-series iPad Pro). Always check
//    SpeechTranscriber.isAvailable and degrade gracefully.
//  - Does not run in the iOS Simulator's older OS images.
//  - No Custom Vocabulary; for domain terms, re-score with
//    SFSpeechRecognizer.contextualStrings on short clips.
public actor SpeechAnalyzerTranscriber: Transcriber {
    public let locale: Locale

    public init(locale: Locale = Locale(identifier: "ja_JP")) {
        self.locale = locale
    }

    public func transcribe(_ buffer: AudioChunk) async throws -> [ASRSegment] {
        try await transcribe(buffer, onVolatileText: nil)
    }

    /// Variant that also forwards rolling volatile-result text to the
    /// caller via `onVolatileText` as SpeechAnalyzer's stabilization
    /// timers refine its hypothesis. The handler is invoked on the
    /// MainActor (suitable for direct UI state writes) and is awaited
    /// inside the actor, so by the time `transcribe` returns no
    /// further callbacks are in flight — callers can safely clear any
    /// preview text without racing a stray late firing.
    ///
    /// `.volatileResults` is always enabled. The single-arg overload
    /// gets the same final-only behavior because `collectResults`
    /// filters on `result.isFinal`; the only cost when no callback is
    /// supplied is a few extra `isFinal == false` results we drop.
    public func transcribe(
        _ buffer: AudioChunk,
        onVolatileText: (@Sendable @MainActor (String) -> Void)?
    ) async throws -> [ASRSegment] {
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

        let pcmBuffer = try await makeCompatibleBuffer(samples: buffer.samples, for: transcriber)

        let (inputStream, cont) = AsyncStream<AnalyzerInput>.makeStream()
        cont.yield(AnalyzerInput(buffer: pcmBuffer))
        cont.finish()

        let analyzer = SpeechAnalyzer(modules: [transcriber])

        // Drain results concurrently with feeding the analyzer. Without an
        // explicit `finalizeAndFinishThroughEndOfInput()`, `transcriber.results`
        // never terminates and the for-await loop hangs forever.
        async let collected: [ASRSegment] = Self.collectResults(
            from: transcriber,
            onVolatileText: onVolatileText
        )
        do {
            _ = try await analyzer.analyzeSequence(inputStream)
            try await analyzer.finalizeAndFinishThroughEndOfInput()
        } catch {
            await analyzer.cancelAndFinishNow()
            throw ASRError.underlying(error)
        }
        return try await collected
    }

    // MARK: - Asset installation

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

    // MARK: - Audio buffer construction

    private func makeCompatibleBuffer(
        samples: [Float],
        for transcriber: SpeechTranscriber
    ) async throws -> AVAudioPCMBuffer {
        guard let sourceFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ) else {
            throw ASRError.audioFormatMismatch(expected: "16kHz mono Float32", got: "n/a")
        }

        let sourceBuffer = try Self.makePCMBuffer(samples: samples, format: sourceFormat)

        let formats = await transcriber.availableCompatibleAudioFormats
        guard let targetFormat = formats.first else {
            // Transcriber didn't advertise any compatible format yet — feed it the source as-is.
            return sourceBuffer
        }

        if sourceFormat.sampleRate == targetFormat.sampleRate
            && sourceFormat.channelCount == targetFormat.channelCount
            && sourceFormat.commonFormat == targetFormat.commonFormat
            && sourceFormat.isInterleaved == targetFormat.isInterleaved {
            return sourceBuffer
        }

        guard let converter = AVAudioConverter(from: sourceFormat, to: targetFormat) else {
            throw ASRError.audioFormatMismatch(
                expected: String(describing: targetFormat),
                got: String(describing: sourceFormat)
            )
        }

        let ratio = targetFormat.sampleRate / sourceFormat.sampleRate
        let outCapacity = AVAudioFrameCount(Double(sourceBuffer.frameLength) * ratio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: targetFormat, frameCapacity: outCapacity) else {
            throw ASRError.audioFormatMismatch(expected: "PCM alloc", got: "failed")
        }

        var error: NSError?
        // The AVAudioConverter input block is treated as @Sendable under Swift 6
        // strict concurrency, so a captured `var` won't compile. Wrap the
        // single-shot "have we fed yet?" flag in a reference type.
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let block: AVAudioConverterInputBlock = { _, status in
            if once.fired {
                status.pointee = .endOfStream
                return nil
            }
            once.fired = true
            status.pointee = .haveData
            return sourceBuffer
        }
        let result = converter.convert(to: outBuffer, error: &error, withInputFrom: block)
        if result == .error {
            throw ASRError.audioFormatMismatch(
                expected: String(describing: targetFormat),
                got: error.map { String(describing: $0) } ?? "convert failed"
            )
        }
        return outBuffer
    }

    private static func makePCMBuffer(samples: [Float], format: AVAudioFormat) throws -> AVAudioPCMBuffer {
        guard let pcm = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(samples.count)) else {
            throw ASRError.audioFormatMismatch(expected: "PCM alloc", got: "failed")
        }
        pcm.frameLength = AVAudioFrameCount(samples.count)
        guard let dst = pcm.floatChannelData?[0] else {
            throw ASRError.audioFormatMismatch(expected: "Float channel data", got: "nil")
        }
        samples.withUnsafeBufferPointer { src in
            if let base = src.baseAddress {
                dst.update(from: base, count: samples.count)
            }
        }
        return pcm
    }

    // MARK: - Result draining

    private static func collectResults(
        from transcriber: SpeechTranscriber,
        onVolatileText: (@Sendable @MainActor (String) -> Void)?
    ) async throws -> [ASRSegment] {
        var segments: [ASRSegment] = []
        // Hold the last volatile we saw so we can fall back to it if
        // `finalizeAndFinishThroughEndOfInput()` closes the stream
        // without promoting any volatile to final. SpeechAnalyzer
        // appears to do this for short offline clips that don't
        // contain a stable sentence boundary — its volatile-
        // stabilization timer doesn't get a chance to fire before
        // the input ends, and the analyzer treats the pending
        // hypothesis as "still tentative" rather than committing it
        // on finalize. Without this fallback the segment list would
        // come back empty and `pipeline.reevaluate` would return nil
        // for clips like a single 4-second utterance — exactly the
        // common case for the per-utterance re-evaluate button.
        var lastVolatile: (
            text: String,
            start: TimeInterval,
            end: TimeInterval,
            confidence: Float?,
            tokens: [ASRSegment.Token]
        )?
        for try await result in transcriber.results {
            let plainText = String(result.text.characters)
            if !result.isFinal {
                if let handler = onVolatileText {
                    await handler(plainText)
                }
                let start = result.range.start.seconds
                let end = start + result.range.duration.seconds
                lastVolatile = (
                    text: plainText,
                    start: start.isFinite ? start : 0,
                    end: end.isFinite ? end : 0,
                    confidence: SpeechAttributes.averageConfidence(in: result.text),
                    tokens: SpeechAttributes.tokens(in: result.text)
                )
                continue
            }
            let start = result.range.start.seconds
            let duration = result.range.duration.seconds
            let end = start + duration
            segments.append(ASRSegment(
                text: plainText,
                start: start.isFinite ? start : 0,
                end: end.isFinite ? end : 0,
                confidence: SpeechAttributes.averageConfidence(in: result.text),
                tokens: SpeechAttributes.tokens(in: result.text)
            ))
        }
        if segments.isEmpty, let v = lastVolatile,
           !v.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            AppLog.asr.info(
                "offline ASR: no finals emitted; using last volatile as result (\(v.text.count, privacy: .public) chars)"
            )
            segments.append(ASRSegment(
                text: v.text,
                start: v.start,
                end: v.end,
                confidence: v.confidence,
                tokens: v.tokens
            ))
        }
        AppLog.asr.info(
            "offline ASR collected: \(segments.count, privacy: .public) segment(s), lastVolatile=\(lastVolatile == nil ? "nil" : "set", privacy: .public)"
        )
        return segments
    }
}
