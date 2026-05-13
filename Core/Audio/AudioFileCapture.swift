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
///   - `realTimePacing: false` (default) — fixed N× real-time per
///     `fastPaceMultiplier`. Sleeps `chunkDuration / N` after each yield,
///     so a 5-min file finishes in `5 min / N`. Empirically 4–8× still
///     preserves SpeechAnalyzer's volatile-stabilization timing relative
///     to real-time, which (because `capForSER` quantizes audio length
///     into 2/4/8 s bins) is what keeps SER labels stable across modes.
///     Going significantly higher shifts segment finalization
///     timestamps and can flip dominant labels. The earlier "as fast as
///     the analyzer can keep up" approach was unbounded and produced
///     both label drift AND OOMs.
///
/// Streams use unbounded buffering in fast mode so the pump never has to
/// drop chunks while downstream catches up — bounded by file size, since
/// the pump exits at EOF.
///
/// Both `raw` and `processed` streams carry the same content — the speech
/// boost EQ doesn't apply to pre-recorded material. Input selection is
/// disabled while a file is the active source.
public actor AudioFileCapture: AudioCapture {
    /// Multiplier used by fast-pace mode. Empirically 4–8× preserves
    /// SpeechAnalyzer's segment-boundary alignment with real-time mode
    /// on M-class iPads; above this range the volatile-stabilization
    /// timers fire at different timestamps, sometimes flipping a
    /// `capForSER` bin and producing different SER labels for the same
    /// audio. 8× is the throughput sweet spot but has been observed
    /// to nudge a few SER labels off versus real-time on edge cases;
    /// 4× cuts that drift roughly in half at the cost of doubled
    /// wall-clock analysis time. Bumping above 8× requires
    /// re-validating SER label stability against a reference set.

    private let fileURL: URL
    private let chunkFrames: AVAudioFrameCount
    private let realTimePacing: Bool
    /// Multiplier applied to fast-pace chunk-sleep duration. Ignored
    /// when `realTimePacing` is true. 8× is the default; the file
    /// importer dialog also exposes 4× when accuracy matters more
    /// than throughput.
    private let fastPaceMultiplier: Int
    /// When true and `realTimePacing` is also true, the file is played
    /// out the device speaker alongside the analysis pump. Independent
    /// of fast pacing, where playing the file at the analyzer's wall
    /// clock would just produce chipmunked-up noise.
    private let audioOutputEnabled: Bool
    /// Position to seek to (in file-time seconds) before starting
    /// the pump. Nonzero only on a pause/resume recreation — the
    /// controller computes the elapsed audio-time at pause and
    /// passes it here so the new capture instance picks up where
    /// the previous one stopped instead of replaying from frame 0.
    private let startingSecond: TimeInterval
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var pumpTask: Task<Void, Never>?
    private var isAccessingScopedResource = false
    private var audioPlayer: AVAudioPlayer?

    public init(
        fileURL: URL,
        chunkFrames: AVAudioFrameCount = 4096,
        realTimePacing: Bool = false,
        fastPaceMultiplier: Int = 8,
        audioOutputEnabled: Bool = false,
        startingSecond: TimeInterval = 0
    ) {
        self.fileURL = fileURL
        self.chunkFrames = chunkFrames
        self.realTimePacing = realTimePacing
        self.fastPaceMultiplier = fastPaceMultiplier
        self.audioOutputEnabled = audioOutputEnabled
        self.startingSecond = startingSecond
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
        // Seek before pump start. AVAudioFile.framePosition is the
        // next-read offset in file-native frames, so multiply
        // `startingSecond` by the file's native sample rate (NOT
        // the pipeline's 16 kHz — those can differ for sources at
        // 44.1/48 kHz). Clamp to file.length so a stale offset
        // doesn't drive `framePosition` past EOF and produce zero
        // reads forever.
        if startingSecond > 0 {
            let nativeRate = file.processingFormat.sampleRate
            let targetFrame = AVAudioFramePosition(
                (startingSecond * nativeRate).rounded()
            )
            file.framePosition = min(targetFrame, file.length)
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

        // Real-time keeps `.bufferingNewest(64)` — the producer is
        // rate-limited by `Task.sleep` so the buffer rarely fills, and
        // newest-keeps-newest is the right policy for a live mic.
        //
        // Fast-pace uses `.bufferingOldest(256)` PAIRED with retry-on-
        // drop in the pump (`yieldWithBackpressure` below). With this
        // policy, attempting to yield into a full buffer returns
        // `.dropped(value:)` instead of overwriting the oldest entry —
        // the pump catches that and sleeps until the consumer drains a
        // slot. The result is true backpressure: the pump runs as fast
        // as the consumer can keep up, never faster, no chunks lost.
        // (An earlier `.bufferingNewest(256)` here silently dropped the
        // oldest 99% of the file — fast-pace would only produce the
        // final few utterances. `.unbounded` before that let the queue
        // grow to 100s of MB and OOM'd the app.)
        let bufferingPolicy: AsyncStream<AudioChunk>.Continuation.BufferingPolicy =
            realTimePacing ? .bufferingNewest(64) : .bufferingOldest(256)
        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: bufferingPolicy)
        self.rawCont = rawCont
        self.processedCont = processedCont

        let totalFrames = file.length
        AppLog.audio.info(
            "File capture started: \(self.fileURL.lastPathComponent, privacy: .public) (\(totalFrames, privacy: .public) frames @ \(file.processingFormat.sampleRate, privacy: .public) Hz)"
        )

        // Optional speaker playback: a separate AVAudioPlayer reads the
        // same file from disk at native rate. The analysis pump runs
        // alongside at its own wall-clock pace; both stay roughly
        // aligned because real-time pacing matches the file timeline.
        // We don't tap the player's output (no feedback loop into ASR);
        // it's purely for the user's ear.
        if realTimePacing && audioOutputEnabled {
            startAudioPlayback()
        }

        let chunkFrames = self.chunkFrames
        let realTimePacing = self.realTimePacing
        let fastPaceMultiplier = self.fastPaceMultiplier
        pumpTask = Task { [weak self] in
            await self?.pump(
                file: file,
                converter: converter,
                outputFormat: outputFormat,
                chunkFrames: chunkFrames,
                realTimePacing: realTimePacing,
                fastPaceMultiplier: fastPaceMultiplier,
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
        audioPlayer?.stop()
        audioPlayer = nil
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
        fastPaceMultiplier: Int,
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
            //
            // In fast-pace, the bounded `.bufferingOldest(256)` policy can
            // return `.dropped(_)` when the consumer is behind. Retry
            // until the chunk is enqueued so no audio is lost — this is
            // the actual backpressure mechanism, replacing the previous
            // (broken) "let memory grow" / "drop the oldest" approaches.
            // Real-time pace's `.bufferingNewest(64)` doesn't return
            // `.dropped`, so the helper is a no-op there.
            await Self.yieldWithBackpressure(chunk, to: rawCont)
            await Self.yieldWithBackpressure(chunk, to: processedCont)

            elapsed += chunkSecondsAtFile
            if realTimePacing {
                // Match wall-clock to file timeline so volatile previews and
                // stage animations look identical to a live recording.
                try? await Task.sleep(nanoseconds: UInt64(chunkSecondsAtFile * 1_000_000_000))
            } else {
                // Configurable fast-pace multiplier — see the
                // type-level comment for the rationale. Critically,
                // this is a *paced* fast mode rather than "as fast
                // as possible": SpeechAnalyzer gets the same
                // wall-clock breathing room as real-time for its
                // stabilization timers, so segment boundaries line
                // up with what real-time would produce, so SER
                // labels stay stable across modes.
                try? await Task.sleep(
                    nanoseconds: UInt64(chunkSecondsAtFile * 1_000_000_000 / Double(fastPaceMultiplier))
                )
            }
        }

        rawCont.finish()
        processedCont.finish()
    }

    /// Yield with retry-on-drop. Under `.bufferingOldest(N)`, attempting
    /// to yield into a full buffer returns `.dropped(value:)` rather than
    /// kicking out an existing element — we sleep briefly and retry until
    /// the consumer drains a slot, giving the pump real backpressure.
    /// Under `.bufferingNewest(N)` this is a single yield with no retry
    /// since that policy returns `.enqueued` even when "full" (it just
    /// silently evicts the oldest, which is what real-time mode wants).
    private static func yieldWithBackpressure(
        _ chunk: AudioChunk,
        to cont: AsyncStream<AudioChunk>.Continuation
    ) async {
        while !Task.isCancelled {
            switch cont.yield(chunk) {
            case .enqueued, .terminated:
                return
            case .dropped:
                // Short wait — long enough to let the consumer pull a
                // few chunks, short enough that the pump stays close to
                // consumer speed instead of pacing at fixed intervals.
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

    /// Start the side-channel `AVAudioPlayer` for speaker playback.
    /// Failures are non-fatal — the analysis pump runs regardless;
    /// the user just doesn't hear anything.
    private func startAudioPlayback() {
        do {
            #if os(iOS) || targetEnvironment(macCatalyst)
            // The active session may have been left in `.record` mode by
            // a prior live recording. Switch to `.playback` so the device
            // unmutes the output route. Mode `.default` is fine — file
            // analysis isn't a measurement context.
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default)
            try session.setActive(true)
            #endif
            let player = try AVAudioPlayer(contentsOf: fileURL)
            player.prepareToPlay()
            if player.play() {
                audioPlayer = player
                AppLog.audio.info("audio playback started")
            } else {
                AppLog.audio.warning("audio playback play() returned false")
            }
        } catch {
            AppLog.audio.warning("audio playback failed: \(String(describing: error), privacy: .public)")
        }
    }
}
