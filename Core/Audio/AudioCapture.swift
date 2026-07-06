import Foundation
@preconcurrency import AVFoundation
import XephonLogging

/// Two parallel streams from the same physical mic. Both are 16 kHz mono
/// Float32, but `processed` has been routed through a speech-band EQ — feed
/// it to ASR, keep `raw` for SER and other prosody-sensitive analyses.
public struct CaptureStreams: Sendable {
    public let raw: AsyncStream<AudioChunk>
    public let processed: AsyncStream<AudioChunk>

    public init(raw: AsyncStream<AudioChunk>, processed: AsyncStream<AudioChunk>) {
        self.raw = raw
        self.processed = processed
    }
}

/// Container/codec for the optional native-rate session recording.
/// `aac` (`.m4a`) keeps sessions small enough to embed in the session
/// bundle; `wav` is bit-exact lossless at the cost of ~10× the size.
public enum RecordingAudioFormat: String, Sendable, CaseIterable, Codable {
    case aac
    case wav

    /// File extension for the chosen container.
    public var fileExtension: String {
        switch self {
        case .aac: return "m4a"
        case .wav: return "wav"
        }
    }
}

/// Abnormal termination cause for a capture session. The recording
/// controller checks this when the capture streams finish while a
/// recording is still in progress and surfaces it to the user —
/// without it, a mid-session engine death is a silent stall.
public enum CaptureEndReason: Sendable, Equatable {
    /// The session was interrupted (call, Siri, another app seizing the
    /// mic) and the OS did not offer resumption.
    case interruptedNotResumable
    /// Engine recovery after a config change, interruption, or
    /// media-services reset failed.
    case recoveryFailed(String)
}

public protocol AudioCapture: Actor {
    /// Starts the audio engine and returns two parallel streams: a raw mic
    /// capture (for SER) and a speech-boosted copy (for ASR). Throws
    /// `AudioError.permissionDenied` if microphone access is not granted.
    func start() async throws -> CaptureStreams
    func stop() async

    /// Configure native-rate recording-to-disk for the NEXT `start()`.
    /// `url == nil` disables it. Default is a no-op — only the live mic
    /// engine records; file-mode capture ignores this.
    func setRecordingDestination(_ url: URL?, format: RecordingAudioFormat) async

    /// Why the most recent capture ended, when it ended abnormally.
    /// nil = still running, never started, or ended via `stop()`.
    func captureEndReason() async -> CaptureEndReason?

    func availableInputs() async -> [AudioInputDescription]
    func currentInput() async -> AudioInputDescription?
    func setPreferredInput(_ uid: String?) async throws

    /// Enables/disables the speech-band EQ on the ASR-bound branch. When
    /// disabled, the `processed` stream pass-throughs raw audio so SER and
    /// ASR see identical input.
    var isSpeechBoostEnabled: Bool { get async }
    func setSpeechBoostEnabled(_ enabled: Bool) async

    /// Enables/disables the AGC-style speech leveler (compressor +
    /// makeup gain) on the ASR-bound branch. Same branch-scoping as
    /// the speech boost: raw stream is never levelled.
    var isSpeechLevelerEnabled: Bool { get async }
    func setSpeechLevelerEnabled(_ enabled: Bool) async
}

/// Maps the running `AVAudioTime.sampleTime` of each tap callback to a
/// session-relative seconds value, even across engine restarts.
///
/// One instance per capture session, shared between the raw and
/// processed taps so they observe the same time origin. The audio
/// thread calls `sessionTime(forRaw:)`; the capture actor calls
/// `markEngineRebuildBoundary()` whenever a recovery path is about to
/// rebuild the engine (`recoverEngine`, `handleMediaServicesReset`).
///
/// On boundary, the rebaser inspects the next chunk's raw time and:
/// - leaves the anchor alone if the new raw time would still produce a
///   forward session time (engine `sampleTime` continued — common when
///   only taps were reinstalled), or
/// - re-anchors so the session timeline continues from
///   `lastSessionTime` (engine `sampleTime` reset to ~0 — happens on a
///   fresh `AVAudioEngine` and is implementation-defined on
///   `engine.stop()`/`start()`).
///
/// Without this, a fresh engine after `handleMediaServicesReset` (or a
/// post-renegotiation tap reinstall, depending on platform behavior)
/// silently produced session times that jumped backward by tens of
/// seconds — corrupting the rolling buffer's anchors, the diarize
/// cursor, and the analyzer's source-time mapping.
///
/// Thread-safe via NSLock; contention is minimal (audio thread reads
/// per chunk, actor writes only on rebuild).
final class TimestampRebaser: @unchecked Sendable {
    private let lock = NSLock()
    private var firstRawTime: Double?
    private var offset: Double = 0
    private var lastSessionTime: Double = 0
    private var pendingBoundary = false

    /// Largest plausible audio gap across an engine rebuild. A
    /// recovery (config-change rebuild, USB reconnect, media-services
    /// reset) takes sub-second to a few seconds, so anything past this
    /// is a `sampleTime` discontinuity, not real elapsed audio — clamp
    /// it. ponytail: 30s ceiling; raise only if a legitimately longer
    /// recovery gap ever shows up in the field. Observed failure: a USB
    /// reconnect rebuild anchored ~64000s ahead, producing a single
    /// utterance spanning ~64000s that froze the per-row diarization
    /// strip's O(duration) majority sweep.
    private static let maxRecoveryGapSec: Double = 30

    init() {}

    func markEngineRebuildBoundary() {
        lock.lock(); defer { lock.unlock() }
        pendingBoundary = true
    }

    func sessionTime(forRaw raw: Double) -> Double {
        lock.lock(); defer { lock.unlock() }
        if pendingBoundary {
            pendingBoundary = false
            if let prev = firstRawTime {
                let wouldBeSession = (raw - prev) + offset
                // Re-anchor on a discontinuity in EITHER direction so
                // the next chunk lands at lastSessionTime, collapsing
                // the rebuild gap. Backward: a fresh engine's sampleTime
                // reset toward 0. Forward-beyond-cap: a fresh engine's
                // sampleTime jumped far ahead (USB reconnect rebuild was
                // seen anchoring ~64000s ahead). A small forward delta
                // is left alone — that's the genuine audio gap during
                // recovery.
                if wouldBeSession < lastSessionTime
                    || wouldBeSession - lastSessionTime > Self.maxRecoveryGapSec {
                    offset = lastSessionTime
                    firstRawTime = raw
                }
            }
        }
        if firstRawTime == nil {
            firstRawTime = raw
        }
        let session = (raw - (firstRawTime ?? raw)) + offset
        if session > lastSessionTime { lastSessionTime = session }
        return session
    }
}

/// Audio-thread diagnostic: detects capture discontinuities by comparing
/// each tap callback's `sampleTime` against where the previous buffer
/// ended. A gap means frames were lost upstream of the tap (HAL overrun,
/// USB glitch, engine stall) — the signature of periodic dropouts in the
/// session recording. One fresh instance per engine build (like the
/// converters), so a rebuild's sampleTime reset doesn't false-positive.
///
/// Lock-free single-consumer: only the input tap's serial callback
/// thread touches the state.
final class TapGapDetector: @unchecked Sendable {
    private var expectedNext: AVAudioFramePosition?
    private var gapCount = 0
    private var gapFramesTotal: AVAudioFramePosition = 0

    func check(time: AVAudioTime, frameLength: AVAudioFrameCount) {
        guard time.isSampleTimeValid, time.sampleRate > 0 else { return }
        defer { expectedNext = time.sampleTime + AVAudioFramePosition(frameLength) }
        guard let expected = expectedNext, time.sampleTime != expected else { return }
        let gap = time.sampleTime - expected
        gapCount += 1
        gapFramesTotal += max(0, gap)
        let ms = Double(gap) / time.sampleRate * 1000
        let totalMs = Double(gapFramesTotal) / time.sampleRate * 1000
        AppLog.audio.warning("input tap discontinuity #\(self.gapCount, privacy: .public): \(gap, privacy: .public) frames (\(String(format: "%.1f", ms), privacy: .public) ms) at sampleTime=\(time.sampleTime, privacy: .public); lost so far ≈\(String(format: "%.0f", totalMs), privacy: .public) ms")
    }
}

public extension AudioCapture {
    func captureEndReason() async -> CaptureEndReason? { nil }
    func availableInputs() async -> [AudioInputDescription] { [] }
    func currentInput() async -> AudioInputDescription? { nil }
    func setPreferredInput(_ uid: String?) async throws {}
    var isSpeechBoostEnabled: Bool { get async { false } }
    func setSpeechBoostEnabled(_ enabled: Bool) async {}
    var isSpeechLevelerEnabled: Bool { get async { false } }
    func setSpeechLevelerEnabled(_ enabled: Bool) async {}
    func setRecordingDestination(_ url: URL?, format: RecordingAudioFormat) async {}
}

public actor AVAudioEngineCapture: AudioCapture {
    // Rebuilt fresh on every `start()`. Reusing a single AVAudioEngine
    // across stop/start cycles leaves stale hardware-format bindings
    // behind that there's no public API to clear in place — the second
    // session ends up reading 44.1 kHz from `inputNode.outputFormat`
    // when the first session ended at that rate, even though the live
    // hardware has since reverted to 48 kHz. A fresh engine sidesteps
    // this; each `inputNode` gets to query HW from scratch.
    private var engine: AVAudioEngine?
    private var eq: AVAudioUnitEQ?
    private var leveler: AVAudioUnitEffect?
    private var processedSink: AVAudioMixerNode?
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var rawConverter: AVAudioConverter?
    private var processedConverter: AVAudioConverter?
    private var preferredInputUID: String?
    private var configChangeObserver: NSObjectProtocol?
    private var interruptionObserver: NSObjectProtocol?
    private var mediaResetObserver: NSObjectProtocol?
    private var endReason: CaptureEndReason?
    private var rebaser: TimestampRebaser?
    private var speechBoostEnabled: Bool = true
    private var speechLevelerEnabled: Bool = true

    // Optional native-rate recording-to-disk. The file is opened once
    // (first engine build) at a FIXED PCM format; later rebuilds (USB
    // reconnect changing rate/channels) recreate `recordConverter` to
    // map the new input format into that same fixed format, so the one
    // output file stays valid across recovery. Writing happens off the
    // audio thread: the tap yields converted buffers into `recordCont`
    // and `recordWriterTask` drains them with `file.write`.
    private var recordingURL: URL?
    private var recordingFormat: RecordingAudioFormat = .aac
    private var recordFile: AVAudioFile?
    private var recordPCMFormat: AVAudioFormat?
    private var recordConverter: AVAudioConverter?
    private var recordCont: AsyncStream<SendableBuffer>.Continuation?
    private var recordWriterTask: Task<Void, Never>?

    /// Ownership-transfer box for handing a freshly-allocated,
    /// never-reused PCM buffer from the audio-thread tap to the
    /// off-thread writer. `@unchecked` is safe: the tap allocates a
    /// fresh buffer per chunk and never touches it after yielding.
    private struct SendableBuffer: @unchecked Sendable {
        let buffer: AVAudioPCMBuffer
    }

    public init() {}

    public func setRecordingDestination(_ url: URL?, format: RecordingAudioFormat) async {
        recordingURL = url
        recordingFormat = format
    }

    public func start() async throws -> CaptureStreams {
        guard await Self.requestPermission() else {
            throw AudioError.permissionDenied
        }

        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            try Self.bindPreferredInput(
                to: effectiveInputUID(session: session),
                session: session
            )
        } catch {
            throw AudioError.engineUnavailable(reason: "audio session: \(error)")
        }
        #endif

        endReason = nil
        rebaser = TimestampRebaser()
        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        self.rawCont = rawCont
        self.processedCont = processedCont

        // Spin up the off-thread recording writer before the engine so
        // the very first converted chunk has a consumer. The file
        // itself is opened inside buildAndStartEngine, once the input
        // format is known; the writer no-ops until then.
        if recordingURL != nil {
            let (recStream, recCont) = AsyncStream<SendableBuffer>.makeStream(
                bufferingPolicy: .bufferingNewest(256)
            )
            self.recordCont = recCont
            self.recordWriterTask = Task { [weak self] in
                for await box in recStream {
                    await self?.appendRecordBuffer(box.buffer)
                }
            }
        }

        do {
            try buildAndStartEngine()
        } catch {
            rawCont.finish()
            processedCont.finish()
            self.rawCont = nil
            self.processedCont = nil
            throw error
        }

        registerSessionObservers()
        return CaptureStreams(raw: rawStream, processed: processedStream)
    }

    /// Build a fresh AVAudioEngine + graph against the session's current
    /// route and start it, yielding into the already-stored stream
    /// continuations. Used by `start()` and by media-services-reset
    /// recovery, where the previous engine instance is invalid and must
    /// be discarded rather than reused.
    private func buildAndStartEngine() throws {
        guard let rawCont, let processedCont, let rebaser else {
            throw AudioError.engineUnavailable(reason: "no active capture streams")
        }
        let capturedRebaser = rebaser

        // Fresh engine + nodes per session — see the property comment.
        let engine = AVAudioEngine()
        let eq = SpeechBoost.makeEQ()
        eq.bypass = !speechBoostEnabled
        let leveler = SpeechLeveler.makeLeveler()
        leveler.bypass = !speechLevelerEnabled
        let processedSink = AVAudioMixerNode()
        engine.attach(eq)
        engine.attach(leveler)
        engine.attach(processedSink)

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)

        // A 0 Hz / 0 ch input format means the session is inactive or
        // bound to a no-input category at this instant (e.g. another
        // session toucher raced our activation). `engine.connect`
        // with such a format raises an uncatchable
        // IsFormatSampleRateAndChannelCountValid NSException — throw
        // a typed error instead so the controller surfaces it and
        // the user can retry.
        guard inputFormat.sampleRate > 0, inputFormat.channelCount > 0 else {
            throw AudioError.unsupportedFormat(
                expected: "non-zero input sample rate and channel count",
                got: "\(inputFormat.sampleRate) Hz × \(inputFormat.channelCount) ch (session inactive or no input bound)"
            )
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ) else {
            throw AudioError.unsupportedFormat(expected: "16 kHz mono Float32", got: "n/a")
        }

        guard let rawConverter = AVAudioConverter(from: inputFormat, to: outputFormat),
              let processedConverter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioError.unsupportedFormat(
                expected: String(describing: outputFormat),
                got: String(describing: inputFormat)
            )
        }
        // Skip the resampler's primer — its job is to ramp output amplitude,
        // but it shifts the output timeline relative to the input by the
        // primer length, which manifests as a small leading delay/glitch in
        // every chunk we yield. Matches the pattern from swift-scribe's
        // BufferConverter.
        rawConverter.primeMethod = .none
        processedConverter.primeMethod = .none

        // input → eq → leveler → processedSink. Tap input for raw, tap
        // processedSink for the speech-boosted + levelled copy. The mixer
        // sink avoids tapping an effect's output bus directly (see comment
        // on `processedSink`).
        engine.connect(input, to: eq, format: inputFormat)
        engine.connect(eq, to: leveler, format: inputFormat)
        engine.connect(leveler, to: processedSink, format: inputFormat)

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate

        // Open the recording file (first build) + (re)make the record
        // converter for this input format. Locals captured into the tap
        // so the audio-thread closure never touches actor state.
        configureRecording(inputFormat: inputFormat)
        let recConverter = recordConverter
        let recFormat = recordPCMFormat
        let recCont = recordCont
        let gapDetector = TapGapDetector()

        // Tap A — raw input.
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            gapDetector.check(time: time, frameLength: buffer.frameLength)
            Self.yieldResampled(
                buffer,
                time: time,
                rebaser: capturedRebaser,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: rawConverter,
                continuation: rawCont
            )
            if let recConverter, let recFormat, let recCont {
                Self.recordCopy(buffer, converter: recConverter, recordFormat: recFormat, continuation: recCont)
            }
        }

        // Tap B — sink mixer (= EQ-processed audio).
        processedSink.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
                rebaser: capturedRebaser,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: processedConverter,
                continuation: processedCont
            )
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            processedSink.removeTap(onBus: 0)
            throw AudioError.engineUnavailable(reason: String(describing: error))
        }

        // Mid-session config changes (USB clock renegotiation) auto-stop
        // the engine. Pinned to this engine instance; re-registered when
        // media-services-reset recovery swaps in a fresh engine.
        configChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            AppLog.audio.info("AVAudioEngineConfigurationChange received")
            Task { [weak self] in
                await self?.recoverEngine(trigger: "configChange")
            }
        }

        self.engine = engine
        self.eq = eq
        self.leveler = leveler
        self.processedSink = processedSink
        self.rawConverter = rawConverter
        self.processedConverter = processedConverter

        AppLog.audio.info("Capture started: input=\(inputFormat.sampleRate, privacy: .public) Hz × \(inputFormat.channelCount, privacy: .public) ch → 16 kHz mono (raw + speech-boosted)")
    }

    /// Session-level lifecycle observers — interruption and media-services
    /// reset. Engine-level config changes are handled by the per-engine
    /// observer in `buildAndStartEngine`; these two are different beasts:
    /// neither fires `AVAudioEngineConfigurationChange`, and before this
    /// pair existed both stalled a recording silently.
    private func registerSessionObservers() {
        #if os(iOS) || targetEnvironment(macCatalyst)
        let session = AVAudioSession.sharedInstance()
        interruptionObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.interruptionNotification,
            object: session,
            queue: nil
        ) { [weak self] note in
            // Extract primitives before hopping into the actor —
            // Notification isn't Sendable.
            let typeRaw = note.userInfo?[AVAudioSessionInterruptionTypeKey] as? UInt
            let optionsRaw = note.userInfo?[AVAudioSessionInterruptionOptionKey] as? UInt
            Task { [weak self] in
                await self?.handleInterruption(typeRaw: typeRaw, optionsRaw: optionsRaw)
            }
        }
        mediaResetObserver = NotificationCenter.default.addObserver(
            forName: AVAudioSession.mediaServicesWereResetNotification,
            object: session,
            queue: nil
        ) { [weak self] _ in
            AppLog.audio.warning("mediaServicesWereReset received")
            Task { [weak self] in
                await self?.handleMediaServicesReset()
            }
        }
        #endif
    }

    #if os(iOS) || targetEnvironment(macCatalyst)
    private func handleInterruption(typeRaw: UInt?, optionsRaw: UInt?) async {
        guard let typeRaw,
              let type = AVAudioSession.InterruptionType(rawValue: typeRaw) else { return }
        switch type {
        case .began:
            // The OS has already stopped the engine. Recovery (or
            // abandonment) is decided at `.ended`. Note `.ended` is not
            // guaranteed to arrive (e.g. the user answers the call); in
            // that case capture stays stalled until the user stops it.
            AppLog.audio.warning("audio session interruption began — engine stopped by OS; awaiting .ended")
        case .ended:
            let options = AVAudioSession.InterruptionOptions(rawValue: optionsRaw ?? 0)
            if options.contains(.shouldResume) {
                AppLog.audio.info("interruption ended with .shouldResume — recovering engine")
                await recoverEngine(trigger: "interruptionEnded")
            } else {
                AppLog.audio.error("interruption ended without .shouldResume — ending capture")
                endReason = .interruptedNotResumable
                rawCont?.finish()
                processedCont?.finish()
            }
        @unknown default:
            break
        }
    }

    /// mediaserverd died and came back. Every AVAudio object from before
    /// the reset is invalid — including the engine the config-change
    /// observer is pinned to — so the engine-reuse recovery path can't
    /// help. Drop everything and rebuild session + engine from scratch,
    /// yielding into the same stream continuations.
    private func handleMediaServicesReset() async {
        guard engine != nil, rawCont != nil, processedCont != nil else {
            AppLog.audio.info("mediaServicesReset: no active capture; nothing to rebuild")
            return
        }
        if let obs = configChangeObserver {
            NotificationCenter.default.removeObserver(obs)
            configChangeObserver = nil
        }
        engine = nil
        eq = nil
        leveler = nil
        processedSink = nil
        // Fresh engine = fresh `sampleTime` origin near 0. Without
        // this mark, the next chunk's session-relative timestamp
        // would jump backward by the entire session duration so far.
        rebaser?.markEngineRebuildBoundary()
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            try Self.bindPreferredInput(to: effectiveInputUID(session: session), session: session)
            try buildAndStartEngine()
            AppLog.audio.info("mediaServicesReset: session + engine rebuilt")
        } catch {
            AppLog.audio.error("mediaServicesReset: rebuild failed — \(String(describing: error), privacy: .public); ending capture")
            endReason = .recoveryFailed("media services reset: \(error)")
            rawCont?.finish()
            processedCont?.finish()
        }
    }
    #endif

    public func captureEndReason() async -> CaptureEndReason? {
        endReason
    }

    /// Recover the existing engine after the OS auto-stopped it —
    /// `AVAudioEngineConfigurationChange` (USB clock renegotiation;
    /// 48 ↔ 44.1 mid-session is common) or an interruption that ended
    /// with `.shouldResume`. We tear down the taps + connections first
    /// (calling `connect()` while old-format taps are installed trips
    /// -10868), rebuild the graph against the live HW format, then
    /// reinstall taps. Reuses the engine instance — valid for both
    /// triggers; only a media-services reset invalidates it (handled
    /// separately by `handleMediaServicesReset`).
    private func recoverEngine(trigger: String) async {
        guard let engine, let eq, let leveler, let processedSink else {
            AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: nothing to restart (engine/eq/leveler/processedSink nil) — capture already stopped")
            return
        }
        guard configChangeObserver != nil else {
            AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: observer cleared — capture is stopping; bailing")
            return
        }
        guard !engine.isRunning else {
            AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: engine still running, skipping rebuild")
            return
        }

        let input = engine.inputNode
        let oldFormat = input.outputFormat(forBus: 0)
        AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: begin (old inputFormat=\(oldFormat.sampleRate, privacy: .public) Hz × \(oldFormat.channelCount, privacy: .public) ch)")

        // Old taps gone first — they reference the prior format.
        input.removeTap(onBus: 0)
        processedSink.removeTap(onBus: 0)
        engine.disconnectNodeInput(eq)
        engine.disconnectNodeInput(leveler)
        engine.disconnectNodeInput(processedSink)

        // Re-assert the preferred input. A USB clock renegotiation (or a
        // brief drop / re-enumerate) fires this notification AND lets the
        // OS re-arbitrate the route — the same auto-routing
        // `bindPreferredInput` fights at start(). Without re-pinning here,
        // the rebuilt graph faithfully captures from whatever device the OS
        // flipped to, so the mic "flips" mid-session. Re-binding verifies
        // the route via currentRoute.inputs and switches back if it moved.
        // Source the post-renegotiation format from AVAudioSession. The
        // input node's `outputFormat(forBus:)` is stale here: we reuse the
        // same AVAudioEngine across recovery (see the engine property
        // comment), so its inputNode still reports the format it cached at
        // engine creation — typically the pre-renegotiation rate. A USB
        // clock switch (48 ↔ 44.1) leaves the node at the old rate while
        // the session has moved on, and connecting at the stale format
        // throws "Input HW format and tap format not matching" out of
        // `engine.connect`. AVAudioSession is the authoritative source
        // once the config-change notification has fired and we've cycled
        // the session in `bindPreferredInput` above.
        #if os(iOS) || targetEnvironment(macCatalyst)
        let session = AVAudioSession.sharedInstance()
        do {
            try Self.bindPreferredInput(to: effectiveInputUID(session: session), session: session)
        } catch {
            AppLog.audio.error("recoverEngine[\(trigger, privacy: .public)]: re-bind failed — \(String(describing: error), privacy: .public)")
        }
        AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: route after re-bind=\(session.currentRoute.inputs.first?.uid ?? "<none>", privacy: .public) sessionRate=\(session.sampleRate, privacy: .public) sessionChannels=\(session.inputNumberOfChannels, privacy: .public)")
        let liveSampleRate = session.sampleRate
        let liveChannelCount = AVAudioChannelCount(session.inputNumberOfChannels)
        #else
        let liveFormat = input.outputFormat(forBus: 0)
        let liveSampleRate = liveFormat.sampleRate
        let liveChannelCount = liveFormat.channelCount
        #endif

        guard liveSampleRate > 0, liveChannelCount > 0,
              let inputFormat = AVAudioFormat(
                standardFormatWithSampleRate: liveSampleRate,
                channels: liveChannelCount
              ),
              let outputFormat = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: PipelineAudio.sampleRate,
                channels: AVAudioChannelCount(PipelineAudio.channelCount),
                interleaved: false
              ),
              let newRawConverter = AVAudioConverter(from: inputFormat, to: outputFormat),
              let newProcessedConverter = AVAudioConverter(from: inputFormat, to: outputFormat),
              let rawCont = rawCont, let processedCont = processedCont else {
            AppLog.audio.error("recoverEngine[\(trigger, privacy: .public)]: invalid state — liveSampleRate=\(liveSampleRate, privacy: .public) liveChannelCount=\(liveChannelCount, privacy: .public) hasRawCont=\(self.rawCont != nil, privacy: .public) hasProcessedCont=\(self.processedCont != nil, privacy: .public); ending capture")
            endReason = .recoveryFailed("\(trigger): invalid post-change state (rate=\(liveSampleRate), channels=\(liveChannelCount))")
            rawCont?.finish()
            processedCont?.finish()
            return
        }
        AppLog.audio.info("recoverEngine[\(trigger, privacy: .public)]: rebuilding with inputFormat=\(inputFormat.sampleRate, privacy: .public) Hz × \(inputFormat.channelCount, privacy: .public) ch")
        newRawConverter.primeMethod = .none
        newProcessedConverter.primeMethod = .none
        rawConverter = newRawConverter
        processedConverter = newProcessedConverter

        engine.connect(input, to: eq, format: inputFormat)
        engine.connect(eq, to: leveler, format: inputFormat)
        engine.connect(leveler, to: processedSink, format: inputFormat)

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate
        // Capture the rebaser locally so it survives any actor-state
        // reset during tap callbacks. nil is impossible here (we'd
        // have bailed at the top guard) but defend against future
        // refactors.
        guard let capturedRebaser = rebaser else {
            AppLog.audio.error("recoverEngine[\(trigger, privacy: .public)]: rebaser nil; ending capture")
            endReason = .recoveryFailed("\(trigger): rebaser nil")
            rawCont.finish()
            processedCont.finish()
            return
        }
        // Remap the (unchanged, fixed-format) record file from the new
        // input format. The file was opened on the first build; only the
        // converter changes here.
        if let rpf = recordPCMFormat {
            recordConverter = AVAudioConverter(from: inputFormat, to: rpf)
            recordConverter?.primeMethod = .none
        }
        let recConverter = recordConverter
        let recFormat = recordPCMFormat
        let recCont = recordCont
        let gapDetector = TapGapDetector()
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            gapDetector.check(time: time, frameLength: buffer.frameLength)
            Self.yieldResampled(
                buffer,
                time: time,
                rebaser: capturedRebaser,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: newRawConverter,
                continuation: rawCont
            )
            if let recConverter, let recFormat, let recCont {
                Self.recordCopy(buffer, converter: recConverter, recordFormat: recFormat, continuation: recCont)
            }
        }
        processedSink.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
                rebaser: capturedRebaser,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: newProcessedConverter,
                continuation: processedCont
            )
        }

        // Mark a rebuild boundary so the rebaser can correct for an
        // `engine.stop()`/`start()` that resets `AVAudioTime.sampleTime`
        // (implementation-defined for the same engine instance). If
        // sampleTime continues forward, the rebaser passes through
        // unchanged; if it resets to ~0, the rebaser re-anchors to
        // maintain monotonicity.
        capturedRebaser.markEngineRebuildBoundary()
        engine.prepare()
        do {
            try engine.start()
            AppLog.audio.info("Capture engine recovered [\(trigger, privacy: .public)]: input=\(inputFormat.sampleRate, privacy: .public) Hz × \(inputFormat.channelCount, privacy: .public) ch")
        } catch {
            AppLog.audio.error("recoverEngine[\(trigger, privacy: .public)]: engine.start() failed — \(String(describing: error), privacy: .public); ending capture")
            endReason = .recoveryFailed("\(trigger): engine restart: \(error)")
            rawCont.finish()
            processedCont.finish()
        }
    }

    // MARK: - Recording-to-disk

    /// Open the recording file on the first build (fixing its PCM
    /// format to the initial input format) and (re)make the converter
    /// that maps the current input format into it. Disables recording
    /// on file-open failure rather than aborting the whole capture.
    private func configureRecording(inputFormat: AVAudioFormat) {
        guard let url = recordingURL else { return }
        if recordPCMFormat == nil {
            guard let pcm = AVAudioFormat(
                commonFormat: .pcmFormatFloat32,
                sampleRate: inputFormat.sampleRate,
                channels: inputFormat.channelCount,
                interleaved: false
            ) else { return }
            do {
                recordFile = try AVAudioFile(
                    forWriting: url,
                    settings: Self.recordSettings(format: recordingFormat, pcm: pcm),
                    commonFormat: .pcmFormatFloat32,
                    interleaved: false
                )
                recordPCMFormat = pcm
                AppLog.audio.info("recording → \(url.lastPathComponent, privacy: .public) (\(self.recordingFormat.rawValue, privacy: .public), \(pcm.sampleRate, privacy: .public) Hz × \(pcm.channelCount, privacy: .public) ch)")
            } catch {
                AppLog.audio.error("recording file open failed: \(String(describing: error), privacy: .public); recording disabled this session")
                recordingURL = nil
                return
            }
        }
        if let recordPCMFormat {
            recordConverter = AVAudioConverter(from: inputFormat, to: recordPCMFormat)
            recordConverter?.primeMethod = .none
        }
    }

    private static func recordSettings(format: RecordingAudioFormat, pcm: AVAudioFormat) -> [String: Any] {
        switch format {
        case .aac:
            return [
                AVFormatIDKey: kAudioFormatMPEG4AAC,
                AVSampleRateKey: pcm.sampleRate,
                AVNumberOfChannelsKey: pcm.channelCount,
                AVEncoderBitRateKey: 192_000,
            ]
        case .wav:
            return [
                AVFormatIDKey: kAudioFormatLinearPCM,
                AVSampleRateKey: pcm.sampleRate,
                AVNumberOfChannelsKey: pcm.channelCount,
                AVLinearPCMBitDepthKey: 16,
                AVLinearPCMIsFloatKey: false,
                AVLinearPCMIsBigEndianKey: false,
                AVLinearPCMIsNonInterleaved: false,
            ]
        }
    }

    /// Off-thread file write, invoked by `recordWriterTask`.
    private func appendRecordBuffer(_ buffer: AVAudioPCMBuffer) {
        guard let file = recordFile else { return }
        do {
            try file.write(from: buffer)
        } catch {
            AppLog.audio.error("recording write failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Audio-thread helper: convert the native tap buffer into the
    /// fixed record format and hand the fresh (owned) buffer to the
    /// off-thread writer. Mirrors `yieldResampled`'s converter dance.
    private static func recordCopy(
        _ buffer: AVAudioPCMBuffer,
        converter: AVAudioConverter,
        recordFormat: AVAudioFormat,
        continuation: AsyncStream<SendableBuffer>.Continuation
    ) {
        let ratio = recordFormat.sampleRate / buffer.format.sampleRate
        let cap = AVAudioFrameCount((Double(buffer.frameLength) * ratio).rounded(.up))
        guard cap > 0,
              let out = AVAudioPCMBuffer(pcmFormat: recordFormat, frameCapacity: cap) else { return }
        var convError: NSError?
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if once.fired { status.pointee = .noDataNow; return nil }
            once.fired = true
            status.pointee = .haveData
            return buffer
        }
        let result = converter.convert(to: out, error: &convError, withInputFrom: inputBlock)
        guard result != .error, out.frameLength > 0 else {
            AppLog.audio.warning("recordCopy: converter produced no output (result=\(result.rawValue, privacy: .public)) — \(buffer.frameLength, privacy: .public) input frames not recorded")
            return
        }
        // `bufferingNewest` silently evicts the oldest queued buffer
        // when the writer backlogs — surface that as a definite drop
        // in the recorded file.
        if case .dropped = continuation.yield(SendableBuffer(buffer: out)) {
            AppLog.audio.warning("recordCopy: record queue full — writer backlogged; dropped \(out.frameLength, privacy: .public) frames from the recording")
        }
    }

    public func stop() async {
        if let obs = configChangeObserver {
            NotificationCenter.default.removeObserver(obs)
            configChangeObserver = nil
        }
        if let obs = interruptionObserver {
            NotificationCenter.default.removeObserver(obs)
            interruptionObserver = nil
        }
        if let obs = mediaResetObserver {
            NotificationCenter.default.removeObserver(obs)
            mediaResetObserver = nil
        }
        if let engine {
            engine.inputNode.removeTap(onBus: 0)
            processedSink?.removeTap(onBus: 0)
            if engine.isRunning {
                engine.stop()
            }
        }
        rawCont?.finish()
        processedCont?.finish()
        rawCont = nil
        processedCont = nil
        rawConverter = nil
        processedConverter = nil
        rebaser = nil
        // Finalize the recording: stop accepting buffers, drain the
        // writer so every queued chunk lands, then drop the file so
        // AVAudioFile flushes its header. Cleared for the next session;
        // the controller keeps the URL it handed in via
        // setRecordingDestination.
        recordCont?.finish()
        recordCont = nil
        await recordWriterTask?.value
        recordWriterTask = nil
        if let recordFile, let recordPCMFormat {
            let secs = Double(recordFile.length) / recordPCMFormat.sampleRate
            AppLog.audio.info("recording finalized: \(recordFile.length, privacy: .public) frames ≈ \(String(format: "%.1f", secs), privacy: .public) s @ \(recordPCMFormat.sampleRate, privacy: .public) Hz — compare against wall-clock session length to quantify dropped audio")
        }
        recordFile = nil
        recordConverter = nil
        recordPCMFormat = nil
        recordingURL = nil
        // Drop the engine + nodes entirely. The next `start()` builds
        // fresh instances — see the engine property comment.
        engine = nil
        eq = nil
        leveler = nil
        processedSink = nil
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Deactivate the session so the `.record / .measurement`
        // config we set in `start()` doesn't linger as the
        // system-wide active session.
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
        #endif
        AppLog.audio.info("Capture stopped")
    }

    // MARK: - Resample helper (audio thread)

    private static func yieldResampled(
        _ buffer: AVAudioPCMBuffer,
        time: AVAudioTime,
        rebaser: TimestampRebaser,
        sampleRateRatio: Double,
        outputFormat: AVAudioFormat,
        converter: AVAudioConverter,
        continuation: AsyncStream<AudioChunk>.Continuation
    ) {
        // Capacity = exactly what one input buffer's worth of audio resamples
        // to (rounded up). The converter will fill up to this and then ask
        // for more input; we respond with `.noDataNow` so it returns with
        // what it has. Capacity equal to expected output (not padded) avoids
        // inviting an additional input request that would risk doubling.
        let expectedOutputFrames = AVAudioFrameCount(
            (Double(buffer.frameLength) * sampleRateRatio).rounded(.up)
        )
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: expectedOutputFrames) else {
            return
        }

        // The block-based API is required for sample-rate conversion. The
        // canonical pattern (per FluidInference/swift-scribe BufferConverter)
        // returns `.noDataNow` after the single buffer is consumed — NOT
        // `.endOfStream`. `.noDataNow` makes the converter return with what
        // it has produced so far, while keeping it reusable for the next
        // call. `.endOfStream` permanently finalizes the converter and
        // breaks subsequent tap callbacks.
        var convError: NSError?
        // `@unchecked Sendable` is safe: one-shot latch consumed only
        // by the AVAudioConverter input block, which the converter
        // calls serially on a single thread per `convert(...)` call.
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let inputBlock: AVAudioConverterInputBlock = { _, status in
            if once.fired {
                status.pointee = .noDataNow
                return nil
            }
            once.fired = true
            status.pointee = .haveData
            return buffer
        }
        let result = converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
        guard result != .error,
              let channelData = outBuffer.floatChannelData else {
            return
        }

        let frameCount = Int(outBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: frameCount))
        // Session-relative timestamp via the rebaser — survives engine
        // restarts (config-change recovery and media-services reset)
        // without backward jumps. See `TimestampRebaser` for the
        // boundary logic.
        let rawTimestamp = time.sampleRate > 0
            ? Double(time.sampleTime) / time.sampleRate
            : 0
        let timestamp = rebaser.sessionTime(forRaw: rawTimestamp)

        // Per-source-channel perceptual levels for the multi-bar
        // meter. Read from the PRE-downmix input buffer so a stereo
        // mic shows L/R separately — the mono `samples` array we
        // just produced has the channels averaged together.
        let channelLevels = computeChannelLevels(buffer)

        continuation.yield(AudioChunk(
            samples: samples,
            sampleRate: PipelineAudio.sampleRate,
            timestamp: timestamp,
            channelLevels: channelLevels
        ))
    }

    /// Per-channel perceptual level (0…1) for the raw input buffer,
    /// matching the dB-normalization the UI's existing single-bar
    /// meter uses (`(20·log10(rms) + 60) / 60` clamped). Runs on the
    /// audio thread; pulls directly from `floatChannelData` so it
    /// works for arbitrary channel counts without allocating per-
    /// channel sample arrays.
    private static func computeChannelLevels(_ buffer: AVAudioPCMBuffer) -> [Float] {
        guard let channelData = buffer.floatChannelData else { return [] }
        let channelCount = Int(buffer.format.channelCount)
        let frameLength = Int(buffer.frameLength)
        guard frameLength > 0 else {
            return Array(repeating: 0, count: channelCount)
        }
        var levels: [Float] = []
        levels.reserveCapacity(channelCount)
        for ch in 0..<channelCount {
            let chan = channelData[ch]
            var sum: Float = 0
            for i in 0..<frameLength {
                let s = chan[i]
                sum += s * s
            }
            let rms = (sum / Float(frameLength)).squareRoot()
            let db = 20 * log10(max(rms, 1e-6))
            let normalized = (db + 60) / 60
            levels.append(min(1, max(0, normalized)))
        }
        return levels
    }

    // MARK: - Input selection

    public func availableInputs() async -> [AudioInputDescription] {
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Pure query — must not change the session category. Earlier
        // this helper called `configureCategoryIfNeeded` to make
        // `session.availableInputs` non-nil under arbitrary session
        // state, but that turned an innocent UI refresh into a
        // session-trampling side effect: every route-change
        // notification (AirPods plug, screen-off, etc.) would flip
        // the active category to `.record / .measurement` and
        // silently break the next `AVAudioPlayer.play()`. Returning
        // an empty list when the current category doesn't expose
        // inputs is acceptable — the picker shows a default label
        // and the next mic-record start (`AVAudioEngineCapture.start`)
        // configures the category itself.
        let session = AVAudioSession.sharedInstance()
        return (session.availableInputs ?? []).map(Self.describe)
        #else
        return []
        #endif
    }

    public func currentInput() async -> AudioInputDescription? {
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Same purity rule as `availableInputs` above — no category
        // writes here.
        let session = AVAudioSession.sharedInstance()
        if let port = session.currentRoute.inputs.first {
            return Self.describe(port)
        }
        if let uid = preferredInputUID,
           let port = (session.availableInputs ?? []).first(where: { $0.uid == uid }) {
            return Self.describe(port)
        }
        return (session.availableInputs ?? []).first.map(Self.describe)
        #else
        return nil
        #endif
    }

    #if os(iOS) || targetEnvironment(macCatalyst)
    /// The input UID we actually want bound: the user's explicit pick
    /// when set, otherwise the built-in mic. iPadOS otherwise auto-routes
    /// to whatever USB-C audio device happens to be plugged in, so a user
    /// who never touched the picker gets silently switched to USB.
    /// Returns nil only when neither is resolvable, letting the OS pick.
    private func effectiveInputUID(session: AVAudioSession) -> String? {
        if let preferredInputUID {
            return preferredInputUID
        }
        if let builtIn = (session.availableInputs ?? []).first(where: { $0.portType == .builtInMic }) {
            AppLog.audio.info("no explicit preferred input; falling back to built-in mic \(builtIn.uid, privacy: .public)")
            return builtIn.uid
        }
        return nil
    }

    /// Activate the session and bind the user's preferred input,
    /// fighting iPadOS 26's tendency to silently auto-route to USB-C
    /// audio devices regardless of the app's preference.
    ///
    /// The naive sequence (`setPreferredInput` → `setActive(true)`) is
    /// documented as "most effective" but is still treated as a hint;
    /// when a USB-C mic is plugged in the OS overrides it at
    /// activation. Reapplying after activation usually works, but on
    /// some sessions it doesn't either — apparently because the
    /// active session is already bound to the USB route and the OS
    /// won't switch on a simple hint.
    ///
    /// The reliable lever is a deactivate / reactivate cycle with the
    /// preferred input set at both ends. Once the session is torn
    /// down and brought back up with an explicit preferredInput set,
    /// the OS treats it as a fresh activation against our preference
    /// rather than overriding an existing route.
    ///
    /// Logs the route at each step so we can see which step actually
    /// switches the route in the field.
    private static func bindPreferredInput(
        to preferredUID: String?,
        session: AVAudioSession
    ) throws {
        func resolveTarget() -> AVAudioSessionPortDescription? {
            guard let preferredUID else { return nil }
            return (session.availableInputs ?? []).first { $0.uid == preferredUID }
        }
        func routeUID() -> String {
            session.currentRoute.inputs.first?.uid ?? "<none>"
        }
        func portTypeDescription() -> String {
            session.currentRoute.inputs.first?.portType.rawValue ?? "<none>"
        }

        let preTarget = resolveTarget()
        if let preTarget {
            // Best-effort pre-activation hint. Tolerated to fail —
            // some category states refuse it before activation.
            try? session.setPreferredInput(preTarget)
        }
        try session.setActive(true)

        let target = resolveTarget()
        let actualUID = routeUID()
        let wantedUID = target?.uid

        if let wantedUID, actualUID != wantedUID, let target {
            AppLog.audio.warning("OS auto-routed to \(actualUID, privacy: .public) (\(portTypeDescription(), privacy: .public)) despite preferredInputUID=\(wantedUID, privacy: .public); attempting hint override")
            try? session.setPreferredInput(target)

            // If the hint didn't take, do a deactivate/reactivate cycle
            // with the preferred input set at both ends. This forces
            // the OS to bind a fresh route against our preference
            // instead of holding the existing USB binding.
            if routeUID() != wantedUID {
                AppLog.audio.warning("hint override didn't switch route; cycling session deactivate/reactivate")
                try? session.setActive(false, options: .notifyOthersOnDeactivation)
                try? session.setPreferredInput(target)
                try session.setActive(true)
                try? session.setPreferredInput(target)
                let finalUID = routeUID()
                if finalUID != wantedUID {
                    AppLog.audio.error("could not bind preferred input \(wantedUID, privacy: .public); session routed to \(finalUID, privacy: .public) (\(portTypeDescription(), privacy: .public))")
                } else {
                    AppLog.audio.info("preferred input bound after cycle: \(wantedUID, privacy: .public)")
                }
            } else {
                AppLog.audio.info("preferred input bound after hint: \(wantedUID, privacy: .public)")
            }
        } else if let wantedUID {
            AppLog.audio.info("preferred input honored on first activation: \(wantedUID, privacy: .public)")
        } else if let preferredUID {
            // We had a UID but couldn't resolve it in `availableInputs`.
            // Either the route changed between picker render and start,
            // or the port we stored is no longer enumerated by the OS.
            let names = (session.availableInputs ?? [])
                .map { "\($0.portType.rawValue):\($0.uid)" }
                .joined(separator: ", ")
            AppLog.audio.error("preferred input UID=\(preferredUID, privacy: .public) NOT FOUND in availableInputs=[\(names, privacy: .public)]; session bound to \(actualUID, privacy: .public)")
        } else {
            AppLog.audio.info("no preferred input set; session bound to \(actualUID, privacy: .public) (\(portTypeDescription(), privacy: .public))")
        }
    }
    #endif

    public var isSpeechBoostEnabled: Bool {
        get async { speechBoostEnabled }
    }

    public func setSpeechBoostEnabled(_ enabled: Bool) async {
        // AVAudioUnitEffect.bypass can flip safely while the engine is running;
        // the EQ continues to forward audio so the dual-tap structure stays
        // valid — only the spectral shaping turns off. When the engine
        // doesn't exist yet (toggled before the first `start()`), we just
        // record the intent so the next `start()` honors it.
        speechBoostEnabled = enabled
        eq?.bypass = !enabled
        AppLog.audio.info("Speech boost \(enabled ? "ON" : "OFF", privacy: .public)")
    }

    public var isSpeechLevelerEnabled: Bool {
        get async { speechLevelerEnabled }
    }

    public func setSpeechLevelerEnabled(_ enabled: Bool) async {
        // Same live-flip semantics as the speech boost: bypass on an
        // AVAudioUnitEffect toggles safely while the engine runs, so
        // the graph structure never changes — only whether the
        // dynamics stage processes or passes through.
        speechLevelerEnabled = enabled
        leveler?.bypass = !enabled
        AppLog.audio.info("Speech leveler \(enabled ? "ON" : "OFF", privacy: .public)")
    }

    public func setPreferredInput(_ uid: String?) async throws {
        preferredInputUID = uid
        AppLog.audio.info("AudioCapture.setPreferredInput stored preferredInputUID=\(uid ?? "<nil>", privacy: .public)")
        #if os(iOS) || targetEnvironment(macCatalyst)
        let session = AVAudioSession.sharedInstance()
        configureCategoryIfNeeded(session)
        let target = uid.flatMap { id in (session.availableInputs ?? []).first(where: { $0.uid == id }) }
        do {
            try session.setPreferredInput(target)
            AppLog.audio.info("AudioCapture.setPreferredInput: session.setPreferredInput → target=\(target?.uid ?? "<nil>", privacy: .public) route=\(session.currentRoute.inputs.first?.uid ?? "<none>", privacy: .public)")
        } catch {
            throw AudioError.engineUnavailable(reason: "setPreferredInput: \(error)")
        }
        #endif
    }

    #if os(iOS) || targetEnvironment(macCatalyst)
    private func configureCategoryIfNeeded(_ session: AVAudioSession) {
        if session.category != .record {
            do {
                try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            } catch {
                AppLog.audio.warning("setCategory failed: \(String(describing: error), privacy: .public)")
            }
        }
    }

    private static func describe(_ port: AVAudioSessionPortDescription) -> AudioInputDescription {
        let kind: AudioInputDescription.Kind
        switch port.portType {
        case .builtInMic:
            kind = .builtInMic
        case .headsetMic:
            kind = .wiredHeadset
        case .bluetoothHFP, .bluetoothLE, .bluetoothA2DP:
            kind = .bluetooth
        case .usbAudio:
            kind = .usb
        case .airPlay:
            kind = .airPlay
        case .carAudio:
            kind = .carPlay
        default:
            kind = .other
        }
        return AudioInputDescription(uid: port.uid, displayName: port.portName, kind: kind)
    }
    #endif

    private static func requestPermission() async -> Bool {
        #if os(macOS) && !targetEnvironment(macCatalyst)
        return await AVCaptureDevice.requestAccess(for: .audio)
        #else
        return await withCheckedContinuation { (cont: CheckedContinuation<Bool, Never>) in
            AVAudioApplication.requestRecordPermission { granted in
                cont.resume(returning: granted)
            }
        }
        #endif
    }
}
