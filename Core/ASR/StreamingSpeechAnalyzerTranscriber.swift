import Foundation
@preconcurrency import AVFoundation
import Speech
import Audio
import XephonLogging

// Streaming variant of SpeechAnalyzerTranscriber. Keeps a SpeechAnalyzer
// alive across the recording session, feeds audio buffers as they arrive,
// and emits one ASRSegment per finalized result.
//
// Canonical pattern per Apple WWDC25 sample code and FluidInference/swift-scribe:
//   - reportingOptions includes `.volatileResults` so the transcriber emits
//     both volatile and final results.
//   - For each result, check `result.isFinal`:
//       * isFinal == true  → committed sentence chunk; emit downstream.
//                            Final results are non-overlapping by contract.
//       * isFinal == false → rolling preview of trailing audio; replace the
//                            volatile-text snapshot used by the live UI.
//
// Long-session handling: Apple's SpeechAnalyzer degrades on long inputs
// (~20 min mark, finals start emitting with long ranges and sparse text).
// To keep accuracy consistent across files of any length, this actor
// periodically tears down the current `SpeechAnalyzer`/`SpeechTranscriber`/
// drainer trio and spins up a fresh one once a configurable amount of
// audio time has been fed. The shared `outputCont` (the
// `AsyncStream<ASRSegment>` the controller is iterating) persists across
// restarts so the controller never sees the boundary. Per-analyzer state
// (`cumulativeOutputFrames`, `timeAnchors`) resets each restart so the new
// analyzer's output-time mapping is fresh; `baseTimestamp` does NOT reset
// because it's the session-level file-time origin.
public actor StreamingSpeechAnalyzerTranscriber: StreamingTranscriber {
    public let locale: Locale

    // Per-analyzer-instance state. Replaced wholesale on each restart.
    private nonisolated(unsafe) var analyzer: SpeechAnalyzer?
    private var inputCont: AsyncStream<AnalyzerInput>.Continuation?
    private var resultDrainer: Task<Void, Never>?
    private var analyzerStartTask: Task<Void, Never>?

    // Session-level state. Persists across analyzer restarts.
    private var outputCont: AsyncStream<ASRSegment>.Continuation?
    private var volatileTextValue: String = ""
    private var targetFormat: AVAudioFormat?
    private var resolvedLocale: Locale?

    // Drift correction: SpeechAnalyzer computes `result.range` from
    // cumulative output-frame counts at the PCM buffer's sample rate.
    // AVAudioConverter resampling from 44.1/48 kHz → 16 kHz consistently
    // undershoots by ~1 frame per chunk, so analyzer-time falls behind
    // source-file-time by 1-3 s over a 30-min file. We can't feed
    // `bufferStartTime` to fix this (the continuity check rejects the
    // chunks and the analyzer stops emitting finals), so we correct on
    // the output side: anchor (analyzer-time, source-time) at every
    // chunk boundary, then interpolate result ranges through the
    // mapping. Mic-mode chunks have `buffer.timestamp` in engine time,
    // not session-zero, so `baseTimestamp` normalizes the first feed.
    private struct TimeAnchor: Sendable {
        let outputSeconds: Double
        let sourceSeconds: Double
    }
    private var timeAnchors: [TimeAnchor] = []
    private var cumulativeOutputFrames: Int = 0
    /// Total audio duration (seconds) fed to the CURRENT analyzer
    /// instance. Drives the periodic restart decision in `feed`.
    private var sessionAudioFedSeconds: Double = 0
    /// Session-level file-time origin. Persists across analyzer restarts
    /// (resetting it would re-anchor the new analyzer's mapping to the
    /// restart point's wall-clock instead of session zero).
    private var baseTimestamp: Double?

    /// After this much audio has been fed to a single analyzer instance,
    /// finalize it and spin up a fresh one. 10 min is well below the
    /// ~20-min degradation point on M-class iPad, with plenty of margin.
    /// Configurable here only — exposing it as init param adds surface
    /// without obvious wins; revisit if a different threshold proves
    /// better empirically.
    private static let analyzerSessionAudioBudgetSeconds: Double = 10 * 60

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
        self.resolvedLocale = resolvedLocale

        // Pre-resolve target format using a throwaway transcriber. The
        // actual transcribers used per analyzer-session are rebuilt by
        // `startAnalyzerSession`, but they all share the same target
        // format so we only need to probe once.
        let probe = SpeechTranscriber(
            locale: resolvedLocale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )
        try await ensureAssetInstalled(for: probe, locale: resolvedLocale)
        let formats = await probe.availableCompatibleAudioFormats
        self.targetFormat = formats.first ?? AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        )

        // Shared output stream — persists across analyzer restarts.
        let (outputStream, outputCont) = AsyncStream<ASRSegment>.makeStream()
        self.outputCont = outputCont
        self.volatileTextValue = ""
        self.baseTimestamp = nil

        try await startAnalyzerSession()
        AppLog.asr.info("Streaming SpeechAnalyzer started (locale=\(resolvedLocale.identifier, privacy: .public))")
        return outputStream
    }

    public func feed(_ buffer: AudioChunk) async {
        guard let targetFormat else { return }
        guard let pcm = makePCMBuffer(samples: buffer.samples, target: targetFormat) else { return }

        // If we've fed this analyzer enough audio that it's approaching
        // the degradation point, roll to a fresh one before this chunk.
        // Doing it BEFORE the yield means the new chunk lands in the
        // fresh analyzer rather than being orphaned by the finalize.
        if sessionAudioFedSeconds >= Self.analyzerSessionAudioBudgetSeconds {
            await rotateAnalyzerSession()
        }

        guard let inputCont else { return }

        // Record (analyzer-time, source-time) anchor at this chunk's
        // start, BEFORE incrementing `cumulativeOutputFrames`. See the
        // TimeAnchor doc-comment for why this is needed.
        let baseSource: Double
        if let base = baseTimestamp {
            baseSource = base
        } else {
            baseTimestamp = buffer.timestamp
            baseSource = buffer.timestamp
        }
        let outputStart = Double(cumulativeOutputFrames) / pcm.format.sampleRate
        let sourceStart = buffer.timestamp - baseSource
        // Drop non-monotonic anchors. AVAudioEngine has been observed
        // to occasionally report a slightly-earlier `sampleTime` on
        // route changes; ignoring those keeps the binary-search
        // interpolation well-defined.
        if let last = timeAnchors.last, sourceStart <= last.sourceSeconds {
            // skip
        } else {
            timeAnchors.append(TimeAnchor(outputSeconds: outputStart, sourceSeconds: sourceStart))
        }
        cumulativeOutputFrames += Int(pcm.frameLength)
        sessionAudioFedSeconds += Double(pcm.frameLength) / pcm.format.sampleRate

        // bufferStartTime is intentionally nil: providing it caused
        // SpeechAnalyzer to stop emitting finals on file input
        // (volatile previews still appeared, but isFinal=true never
        // fired) — the analyzer's continuity check is stricter than
        // the per-buffer seconds resolution we can supply once the
        // file rate is rounded to 16 kHz. The analyzer infers
        // timeline from buffer frame counts instead, and we correct
        // the drift on the output side via `correctedSeconds(_:)`.
        let input = AnalyzerInput(buffer: pcm)
        // Retry-on-drop. With `.bufferingOldest(64)`, yielding into a
        // full queue returns `.dropped(_)` rather than evicting an
        // existing entry. Sleep briefly + retry until the analyzer
        // drains a slot — the queue becomes real backpressure on the
        // producer, never silently losing audio. Crucial for long
        // files where the analyzer can stall transiently.
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
        await finishCurrentAnalyzerSession()
        outputCont?.finish()
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
            // Apply file-time correction to both segment bounds AND
            // per-token bounds. Tokens flow into `splitIntoSentences`
            // for hand-edit / re-evaluation splits, so the two have
            // to live in the same timeline — otherwise a split's
            // sub-segment endpoints (token-derived) would disagree
            // with the parent's endpoints (corrected) by the drift
            // amount.
            let rawTokens = SpeechAttributes.tokens(in: result.text)
            let correctedTokens = rawTokens.map { token in
                ASRSegment.Token(
                    text: token.text,
                    start: correctedSeconds(token.start),
                    end: correctedSeconds(token.end)
                )
            }
            let segment = ASRSegment(
                text: text,
                start: correctedSeconds(result.range.start.seconds),
                end: correctedSeconds(result.range.end.seconds),
                confidence: SpeechAttributes.averageConfidence(in: result.text),
                tokens: correctedTokens
            )
            outputCont?.yield(segment)
        } else {
            volatileTextValue = text
        }
    }

    /// Map an analyzer-reported time (cumulative-output-frames / 16 kHz)
    /// to source-file time using the `timeAnchors` mapping recorded at
    /// each chunk boundary in `feed`. Before the first anchor we apply
    /// a constant offset (first anchor's source−output difference);
    /// after the last anchor we extrapolate via the last segment's
    /// slope — important for the tail finals SpeechAnalyzer flushes
    /// from `finalizeAndFinishThroughEndOfInput()`.
    private func correctedSeconds(_ raw: Double) -> Double {
        guard !timeAnchors.isEmpty else { return raw }
        // Binary-search for first anchor with outputSeconds > raw.
        var lo = 0
        var hi = timeAnchors.count
        while lo < hi {
            let mid = (lo + hi) / 2
            if timeAnchors[mid].outputSeconds <= raw {
                lo = mid + 1
            } else {
                hi = mid
            }
        }
        let upperIdx = lo

        if upperIdx == 0 {
            let first = timeAnchors[0]
            return raw + (first.sourceSeconds - first.outputSeconds)
        }
        if upperIdx == timeAnchors.count {
            let last = timeAnchors[timeAnchors.count - 1]
            if timeAnchors.count >= 2 {
                let prev = timeAnchors[timeAnchors.count - 2]
                let outSpan = last.outputSeconds - prev.outputSeconds
                if outSpan > 0 {
                    let slope = (last.sourceSeconds - prev.sourceSeconds) / outSpan
                    return last.sourceSeconds + slope * (raw - last.outputSeconds)
                }
            }
            return last.sourceSeconds + (raw - last.outputSeconds)
        }
        let lower = timeAnchors[upperIdx - 1]
        let upper = timeAnchors[upperIdx]
        let outSpan = upper.outputSeconds - lower.outputSeconds
        if outSpan <= 0 {
            return lower.sourceSeconds
        }
        let t = (raw - lower.outputSeconds) / outSpan
        return lower.sourceSeconds + t * (upper.sourceSeconds - lower.sourceSeconds)
    }

    private func cleanup() {
        analyzer = nil
        inputCont = nil
        outputCont = nil
        targetFormat = nil
        resolvedLocale = nil
        volatileTextValue = ""
        resultDrainer = nil
        analyzerStartTask = nil
        timeAnchors.removeAll(keepingCapacity: false)
        cumulativeOutputFrames = 0
        sessionAudioFedSeconds = 0
        baseTimestamp = nil
    }

    // MARK: - Analyzer session lifecycle

    /// Build a fresh `SpeechTranscriber` + `SpeechAnalyzer` + result
    /// drainer trio. Resets per-analyzer state (output-frame counter,
    /// time anchors, audio-fed counter). Caller must have already
    /// torn down any prior analyzer via `finishCurrentAnalyzerSession`.
    private func startAnalyzerSession() async throws {
        guard let resolvedLocale, let outputCont else { return }
        let transcriber = SpeechTranscriber(
            locale: resolvedLocale,
            transcriptionOptions: [],
            reportingOptions: [.volatileResults],
            attributeOptions: [.audioTimeRange, .transcriptionConfidence]
        )

        // `.bufferingOldest(64)` + retry-on-drop in `feed` is the
        // backpressure discipline. 64 buffers ≈ 6 s of audio at
        // 4096-frame chunks gives generous headroom for transient
        // analyzer slowdowns before the producer pauses.
        let (inputStream, inputCont) = AsyncStream<AnalyzerInput>.makeStream(
            bufferingPolicy: .bufferingOldest(64)
        )
        self.inputCont = inputCont
        self.timeAnchors.removeAll(keepingCapacity: true)
        self.cumulativeOutputFrames = 0
        self.sessionAudioFedSeconds = 0

        let analyzer = SpeechAnalyzer(modules: [transcriber], options: nil)
        self.analyzer = analyzer

        resultDrainer = Task { [weak self] in
            do {
                for try await result in transcriber.results {
                    await self?.handleResult(result)
                }
            } catch {
                AppLog.asr.error("streaming drainer failed: \(String(describing: error), privacy: .public)")
            }
        }

        analyzerStartTask = Task { [outputCont] in
            do {
                try await analyzer.start(inputSequence: inputStream)
            } catch {
                AppLog.asr.error("streaming analyzer.start failed: \(String(describing: error), privacy: .public)")
                // Don't finish the shared outputCont here; the session
                // owner decides when downstream is done. Just log.
                _ = outputCont
            }
        }
    }

    /// Finish the current analyzer's input stream, await
    /// finalize-through-end-of-input, drain any tail finals, then
    /// clear the per-analyzer state. Safe to call multiple times.
    private func finishCurrentAnalyzerSession() async {
        inputCont?.finish()
        do {
            try await analyzer?.finalizeAndFinishThroughEndOfInput()
        } catch {
            AppLog.asr.error("streaming finalize failed: \(String(describing: error), privacy: .public)")
        }
        await resultDrainer?.value
        analyzerStartTask?.cancel()
        analyzer = nil
        inputCont = nil
        resultDrainer = nil
        analyzerStartTask = nil
    }

    /// Roll to a fresh analyzer mid-session. Drains the current one and
    /// builds a new trio. Called from `feed` when the per-analyzer
    /// audio-fed budget is exhausted. Throws are swallowed (logged) —
    /// failure here just means transcription pauses for this session.
    private func rotateAnalyzerSession() async {
        let prevAudio = sessionAudioFedSeconds
        await finishCurrentAnalyzerSession()
        do {
            try await startAnalyzerSession()
            AppLog.asr.info(
                "SpeechAnalyzer rotated after \(prevAudio, privacy: .public)s of audio (long-session degradation guard)"
            )
        } catch {
            AppLog.asr.error("analyzer rotation failed: \(String(describing: error), privacy: .public)")
        }
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
        // `@unchecked Sendable` is safe: one-shot latch consumed only
        // by the AVAudioConverter input block, which the converter
        // calls serially on a single thread per `convert(...)` call.
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
