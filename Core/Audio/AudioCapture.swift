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
    // Rebuilt fresh on every `start()`. Reusing a single AVAudioEngine
    // across stop/start cycles leaves stale hardware-format bindings
    // behind that there's no public API to clear in place — the second
    // session ends up reading 44.1 kHz from `inputNode.outputFormat`
    // when the first session ended at that rate, even though the live
    // hardware has since reverted to 48 kHz. A fresh engine sidesteps
    // this; each `inputNode` gets to query HW from scratch.
    private var engine: AVAudioEngine?
    private var eq: AVAudioUnitEQ?
    private var processedSink: AVAudioMixerNode?
    private var rawCont: AsyncStream<AudioChunk>.Continuation?
    private var processedCont: AsyncStream<AudioChunk>.Continuation?
    private var rawConverter: AVAudioConverter?
    private var processedConverter: AVAudioConverter?
    private var preferredInputUID: String?
    private var configChangeObserver: NSObjectProtocol?
    private var speechBoostEnabled: Bool = true

    public init() {}

    public func start() async throws -> CaptureStreams {
        guard await Self.requestPermission() else {
            throw AudioError.permissionDenied
        }

        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.record, mode: .measurement, options: [.allowBluetoothHFP])
            // Fall back to the built-in mic when the user hasn't
            // explicitly picked anything. iPadOS otherwise auto-routes
            // to whatever USB-C audio device happens to be plugged in,
            // which means a user who's never touched the input picker
            // gets silently switched to USB — surprising behavior that
            // makes the picker feel decorative. Mirrors how Voice
            // Memos and Ferrite handle the no-preference case.
            let effectiveUID: String?
            if let preferredInputUID {
                effectiveUID = preferredInputUID
            } else if let builtIn = (session.availableInputs ?? []).first(where: { $0.portType == .builtInMic }) {
                effectiveUID = builtIn.uid
                AppLog.audio.info("no explicit preferred input; falling back to built-in mic \(builtIn.uid, privacy: .public)")
            } else {
                effectiveUID = nil
            }
            try Self.bindPreferredInput(
                to: effectiveUID,
                session: session
            )
        } catch {
            throw AudioError.engineUnavailable(reason: "audio session: \(error)")
        }
        #endif

        // Fresh engine + nodes per session — see the property comment.
        let engine = AVAudioEngine()
        let eq = SpeechBoost.makeEQ()
        eq.bypass = !speechBoostEnabled
        let processedSink = AVAudioMixerNode()
        engine.attach(eq)
        engine.attach(processedSink)

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

        // input → eq → processedSink. Tap input for raw, tap processedSink for
        // the speech-boosted copy. The mixer sink avoids tapping the EQ output
        // bus directly (see comment on `processedSink`).
        engine.connect(input, to: eq, format: inputFormat)
        engine.connect(eq, to: processedSink, format: inputFormat)

        let (rawStream, rawCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))
        let (processedStream, processedCont) = AsyncStream<AudioChunk>.makeStream(bufferingPolicy: .bufferingNewest(64))

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate

        // Tap A — raw input.
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

        // Tap B — sink mixer (= EQ-processed audio).
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
            throw AudioError.engineUnavailable(reason: String(describing: error))
        }

        // Mid-session config changes (USB clock renegotiation, media-services
        // restart) auto-stop the engine. Pinned to this engine instance.
        configChangeObserver = NotificationCenter.default.addObserver(
            forName: .AVAudioEngineConfigurationChange,
            object: engine,
            queue: nil
        ) { [weak self] _ in
            Task { [weak self] in
                await self?.restartEngineAfterConfigurationChange()
            }
        }

        self.engine = engine
        self.eq = eq
        self.processedSink = processedSink
        self.rawCont = rawCont
        self.processedCont = processedCont
        self.rawConverter = rawConverter
        self.processedConverter = processedConverter

        AppLog.audio.info("Capture started: input=\(inputFormat.sampleRate, privacy: .public) Hz × \(inputFormat.channelCount, privacy: .public) ch → 16 kHz mono (raw + speech-boosted)")
        return CaptureStreams(raw: rawStream, processed: processedStream)
    }

    /// Recover from `AVAudioEngineConfigurationChange`. The engine has
    /// been auto-stopped by the OS; the input's HW sample rate may
    /// have renegotiated (USB-C mics commonly switch 48 ↔ 44.1 mid-session).
    /// We tear down the taps + connections first (calling `connect()`
    /// while old-format taps are installed trips -10868), rebuild the
    /// graph against the live HW format, then reinstall taps.
    private func restartEngineAfterConfigurationChange() async {
        guard let engine, let eq, let processedSink else { return }
        guard configChangeObserver != nil else { return }
        guard !engine.isRunning else { return }

        let input = engine.inputNode

        // Old taps gone first — they reference the prior format.
        input.removeTap(onBus: 0)
        processedSink.removeTap(onBus: 0)
        engine.disconnectNodeInput(eq)
        engine.disconnectNodeInput(processedSink)

        // Live HW sample rate. After the config-change notification has
        // fired, the OS has committed the new route, so
        // `AVAudioSession.sampleRate` is authoritative here (unlike the
        // window right after `setActive(true)`).
        let liveChannelCount = input.outputFormat(forBus: 0).channelCount
        #if os(iOS) || targetEnvironment(macCatalyst)
        let liveSampleRate = AVAudioSession.sharedInstance().sampleRate
        #else
        let liveSampleRate = input.outputFormat(forBus: 0).sampleRate
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
            AppLog.audio.error("capture rebuild after config change: invalid state")
            rawCont?.finish()
            processedCont?.finish()
            return
        }
        newRawConverter.primeMethod = .none
        newProcessedConverter.primeMethod = .none
        rawConverter = newRawConverter
        processedConverter = newProcessedConverter

        engine.connect(input, to: eq, format: inputFormat)
        engine.connect(eq, to: processedSink, format: inputFormat)

        let sampleRateRatio = outputFormat.sampleRate / inputFormat.sampleRate
        input.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: newRawConverter,
                continuation: rawCont
            )
        }
        processedSink.installTap(onBus: 0, bufferSize: 4096, format: inputFormat) { buffer, time in
            Self.yieldResampled(
                buffer,
                time: time,
                sampleRateRatio: sampleRateRatio,
                outputFormat: outputFormat,
                converter: newProcessedConverter,
                continuation: processedCont
            )
        }

        engine.prepare()
        do {
            try engine.start()
            AppLog.audio.info("Capture engine restarted after config change: input=\(inputFormat.sampleRate, privacy: .public) Hz × \(inputFormat.channelCount, privacy: .public) ch")
        } catch {
            AppLog.audio.error("capture restart after config change failed: \(String(describing: error), privacy: .public)")
            rawCont.finish()
            processedCont.finish()
        }
    }

    public func stop() async {
        if let obs = configChangeObserver {
            NotificationCenter.default.removeObserver(obs)
            configChangeObserver = nil
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
        // Drop the engine + nodes entirely. The next `start()` builds
        // fresh instances — see the engine property comment.
        engine = nil
        eq = nil
        processedSink = nil
        #if os(iOS) || targetEnvironment(macCatalyst)
        // Deactivate the session so the `.record / .measurement /
        // [.allowBluetoothHFP]` config we set in `start()` doesn't
        // linger as the system-wide active session.
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
