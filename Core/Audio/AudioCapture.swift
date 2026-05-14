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

public protocol AudioCapture: Actor {
    /// Starts the audio engine and returns two parallel streams: a raw mic
    /// capture (for SER) and a speech-boosted copy (for ASR). Throws
    /// `AudioError.permissionDenied` if microphone access is not granted.
    func start() async throws -> CaptureStreams
    func stop() async

    func availableInputs() async -> [AudioInputDescription]
    func currentInput() async -> AudioInputDescription?
    func setPreferredInput(_ uid: String?) async throws

    /// Enables/disables the speech-band EQ on the ASR-bound branch. When
    /// disabled, the `processed` stream pass-throughs raw audio so SER and
    /// ASR see identical input.
    var isSpeechBoostEnabled: Bool { get async }
    func setSpeechBoostEnabled(_ enabled: Bool) async
}

public extension AudioCapture {
    func availableInputs() async -> [AudioInputDescription] { [] }
    func currentInput() async -> AudioInputDescription? { nil }
    func setPreferredInput(_ uid: String?) async throws {}
    var isSpeechBoostEnabled: Bool { get async { false } }
    func setSpeechBoostEnabled(_ enabled: Bool) async {}
}

public actor AVAudioEngineCapture: AudioCapture {
    private let engine = AVAudioEngine()
    private let eq: AVAudioUnitEQ
    // Sink mixer that receives the EQ output. Tapping AVAudioUnitEQ's output
    // bus directly is unreliable — its internal format isn't always inherited
    // from `engine.connect(...format:)`, leading to "Failed to create tap due
    // to format mismatch" at runtime. AVAudioMixerNode negotiates formats
    // predictably, so we route input → eq → sink and tap the sink.
    private let processedSink = AVAudioMixerNode()
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var rawConverter: AVAudioConverter?
    private var processedConverter: AVAudioConverter?
    private var preferredInputUID: String?

    public init() {
        self.eq = SpeechBoost.makeEQ()
        self.engine.attach(self.eq)
        self.engine.attach(self.processedSink)
    }

    public func start() async throws -> CaptureStreams {
        guard await Self.requestPermission() else {
            throw AudioError.permissionDenied
        }

        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            if let uid = preferredInputUID,
               let target = (session.availableInputs ?? []).first(where: { $0.uid == uid }) {
                try session.setPreferredInput(target)
            }
            try session.setActive(true)
        } catch {
            throw AudioError.engineUnavailable(reason: "audio session: \(error)")
        }
        #endif

        let input = engine.inputNode
        let inputFormat = input.outputFormat(forBus: 0)

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
        self.rawConverter = rawConverter
        self.processedConverter = processedConverter

        // input → eq → processedSink. Tap input for raw, tap processedSink for
        // the speech-boosted copy. The mixer sink avoids tapping the EQ output
        // bus directly (see comment on `processedSink`).
        engine.connect(input, to: eq, format: inputFormat)
        engine.connect(eq, to: processedSink, format: inputFormat)

        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        self.rawCont = rawCont
        self.processedCont = processedCont

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate

        // Tap A — raw input. Captures: rawCont (Sendable), rawConverter,
        // outputFormat (each touched only on the audio thread).
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: rawConverter,
                continuation: rawCont
            )
        }

        // Tap B — sink mixer (= EQ-processed audio). The same per-buffer
        // audio after speech-band shaping.
        processedSink.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
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
            rawCont.finish()
            processedCont.finish()
            self.rawCont = nil
            self.processedCont = nil
            self.rawConverter = nil
            self.processedConverter = nil
            throw AudioError.engineUnavailable(reason: String(describing: error))
        }

        AppLog.audio.info("Capture started: input=\(inputFormat.sampleRate, privacy: .public) Hz → 16 kHz mono (raw + speech-boosted)")
        return CaptureStreams(raw: rawStream, processed: processedStream)
    }

    public func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        processedSink.removeTap(onBus: 0)
        if engine.isRunning {
            engine.stop()
        }
        rawCont?.finish()
        processedCont?.finish()
        rawCont = nil
        processedCont = nil
        rawConverter = nil
        processedConverter = nil
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Deactivate the session so the `.record / .measurement /
        // [.allowBluetoothHFP]` config we set in `start()` doesn't
        // linger as the system-wide active session. If we leave it
        // active, subsequent code that switches to a playback-only
        // category (e.g. per-utterance audio review) inherits the
        // recording route and `play()` reports success while the
        // actual output stays silent.
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
        let timestamp = time.sampleRate > 0
            ? Double(time.sampleTime) / time.sampleRate
            : 0

        continuation.yield(AudioChunk(
            samples: samples,
            sampleRate: PipelineAudio.sampleRate,
            timestamp: timestamp
        ))
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

    public var isSpeechBoostEnabled: Bool {
        get async { !eq.bypass }
    }

    public func setSpeechBoostEnabled(_ enabled: Bool) async {
        // AVAudioUnitEffect.bypass can flip safely while the engine is running;
        // the EQ continues to forward audio so the dual-tap structure stays
        // valid — only the spectral shaping turns off.
        eq.bypass = !enabled
        AppLog.audio.info("Speech boost \(enabled ? "ON" : "OFF", privacy: .public)")
    }

    public func setPreferredInput(_ uid: String?) async throws {
        preferredInputUID = uid
        #if os(iOS) || targetEnvironment(macCatalyst)
        let session = AVAudioSession.sharedInstance()
        configureCategoryIfNeeded(session)
        let target = uid.flatMap { id in (session.availableInputs ?? []).first(where: { $0.uid == id }) }
        do {
            try session.setPreferredInput(target)
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
