import Foundation
@preconcurrency import AVFoundation
import XephonLogging

public protocol AudioCapture: Actor {
    /// Starts the audio engine and returns a stream of resampled buffers
    /// (16 kHz mono Float32 per `PipelineAudio`). Throws
    /// `AudioError.permissionDenied` if microphone access is not granted.
    func start() async throws -> AsyncStream<AudioChunk>
    func stop() async

    /// Inputs currently advertised by the OS (built-in mic, AirPods, USB-C
    /// audio, etc.). May be empty before the audio session is configured.
    func availableInputs() async -> [AudioInputDescription]
    /// The input the OS will actually use on the next `start()`.
    func currentInput() async -> AudioInputDescription?
    /// Persist a preferred input. Pass `nil` to defer to the system default.
    /// The chosen UID corresponds to `AudioInputDescription.uid`.
    func setPreferredInput(_ uid: String?) async throws
}

public extension AudioCapture {
    func availableInputs() async -> [AudioInputDescription] { [] }
    func currentInput() async -> AudioInputDescription? { nil }
    func setPreferredInput(_ uid: String?) async throws {}
}

public actor AVAudioEngineCapture: AudioCapture {
    private let engine = AVAudioEngine()
    private var continuation: AsyncStream<AudioChunk>.Continuation?
    private var converter: AVAudioConverter?
    private var preferredInputUID: String?

    public init() {}

    public func start() async throws -> AsyncStream<AudioChunk> {
        guard await Self.requestPermission() else {
            throw AudioError.permissionDenied
        }

        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            // `.allowBluetooth` enables AirPods / Bluetooth-headset mic input
            // (HFP profile). Without it, AirPods-as-input is hidden from
            // `availableInputs` even when AirPods are paired.
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
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

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioError.unsupportedFormat(
                expected: String(describing: outputFormat),
                got: String(describing: inputFormat)
            )
        }
        self.converter = converter

        let (stream, cont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        self.continuation = cont

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate

        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
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

            cont.yield(AudioChunk(
                samples: samples,
                sampleRate: PipelineAudio.sampleRate,
                timestamp: timestamp
            ))
        }

        engine.prepare()
        do {
            try engine.start()
        } catch {
            input.removeTap(onBus: 0)
            cont.finish()
            self.continuation = nil
            self.converter = nil
            throw AudioError.engineUnavailable(reason: String(describing: error))
        }

        AppLog.audio.info("Capture started: input=\(inputFormat.sampleRate, privacy: .public) Hz → 16 kHz mono")
        return stream
    }

    public func stop() async {
        engine.inputNode.removeTap(onBus: 0)
        if engine.isRunning {
            engine.stop()
        }
        continuation?.finish()
        continuation = nil
        converter = nil
        AppLog.audio.info("Capture stopped")
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
                try session.setCategory(.record, mode: .measurement, options: [.allowBluetooth])
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
