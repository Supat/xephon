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
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var rawConverter: AVAudioConverter?
    private var processedConverter: AVAudioConverter?
    private var preferredInputUID: String?

    public init() {
        self.eq = SpeechBoost.makeEQ()
        self.engine.attach(self.eq)
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
        self.rawConverter = rawConverter
        self.processedConverter = processedConverter

        // Connect inputNode → EQ. The EQ output bus (bus 0) is what we tap for
        // the processed stream; inputNode's bus 0 (a tee point) gives us raw.
        engine.connect(input, to: eq, format: inputFormat)

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

        // Tap B — EQ output. The same per-buffer audio after speech-band shaping.
        eq.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
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
            eq.removeTap(onBus: 0)
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
        eq.removeTap(onBus: 0)
        if engine.isRunning {
            engine.stop()
        }
        rawCont?.finish()
        processedCont?.finish()
        rawCont = nil
        processedCont = nil
        rawConverter = nil
        processedConverter = nil
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
        let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * sampleRateRatio) + 1024
        guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity) else {
            return
        }

        var convError: NSError?
        let inputBlock: AVAudioConverterInputBlock = { _, status in
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
        let session = AVAudioSession.sharedInstance()
        configureCategoryIfNeeded(session)
        return (session.availableInputs ?? []).map(Self.describe)
        #else
        return []
        #endif
    }

    public func currentInput() async -> AudioInputDescription? {
        #if os(iOS) || targetEnvironment(macCatalyst)
        let session = AVAudioSession.sharedInstance()
        configureCategoryIfNeeded(session)
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
