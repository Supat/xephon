import Foundation
import Observation
@preconcurrency import AVFoundation
import Audio
import ASR
import Fusion
import Export
import XephonLogging

@MainActor
@Observable
final class RecordingController {
    enum Phase: Sendable {
        case idle
        case warmingUp
        case recording
        case analyzing
    }

    private(set) var phase: Phase = .idle
    private(set) var samplesCaptured: Int = 0
    private(set) var errorMessage: String?
    private(set) var utterances: [UtteranceEstimate] = []
    private(set) var inputLevel: Float = 0
    private(set) var availableInputs: [AudioInputDescription] = []
    private(set) var currentInputUID: String?
    private(set) var isSpeechBoostEnabled: Bool = true

    var isRecording: Bool { phase == .recording }
    var isAnalyzing: Bool { phase == .analyzing }
    var isWarmingUp: Bool { phase == .warmingUp }
    var elapsedSeconds: Double {
        Double(samplesCaptured) / PipelineAudio.sampleRate
    }

    private let capture: any AudioCapture
    private let streamingTranscriber: any StreamingTranscriber
    private var pipeline: AnalysisPipeline?
    private var pipelineTask: Task<AnalysisPipeline, Never>?
    private let exporter = JSONExporter()
    private var rawTask: Task<Void, Never>?
    private var feedTask: Task<Void, Never>?
    private var analysisTask: Task<Void, Never>?
    private var routeWatcherTask: Task<Void, Never>?
    private var capturedSamples: [Float] = []

    init(
        capture: any AudioCapture = AVAudioEngineCapture(),
        streamingTranscriber: (any StreamingTranscriber)? = nil,
        pipeline: AnalysisPipeline? = nil
    ) {
        self.capture = capture
        self.streamingTranscriber = streamingTranscriber ?? StreamingSpeechAnalyzerTranscriber()
        self.pipeline = pipeline
        // Pre-warm the pipeline in the background at first construction so heavy
        // SER constructors (W2V2 ONNX ~631 MB) and the SpeechAnalyzer asset
        // install can complete before the user finishes their first sentence.
        if pipeline == nil {
            self.pipelineTask = Task.detached(priority: .userInitiated) {
                let configured = await AnalysisPipeline.autoConfigured()
                AppLog.app.info("Pipeline pre-warm complete")
                return configured
            }
        }

        // Initial input list + observe route changes (e.g. AirPods connect / disconnect).
        Task { @MainActor [weak self] in
            await self?.refreshInputs()
        }
        routeWatcherTask = Task { @MainActor [weak self] in
            let notifications = NotificationCenter.default.notifications(
                named: AVAudioSession.routeChangeNotification
            )
            for await _ in notifications {
                guard let self else { break }
                await self.refreshInputs()
            }
        }
    }

    private func ensurePipeline() async -> AnalysisPipeline {
        if let pipeline { return pipeline }
        if let task = pipelineTask {
            let configured = await task.value
            self.pipeline = configured
            self.pipelineTask = nil
            return configured
        }
        let configured = await AnalysisPipeline.autoConfigured()
        self.pipeline = configured
        return configured
    }

    func toggle() async {
        switch phase {
        case .idle:
            await start()
        case .recording:
            await stop()
        case .warmingUp, .analyzing:
            break
        }
    }

    /// Start streaming: open capture + a long-lived ASR analyzer. Each finalized
    /// (post-volatile) ASR segment is processed through SER+fusion and appended
    /// to `utterances` live.
    func start() async {
        do {
            let segmentStream = try await streamingTranscriber.start()
            let streams = try await capture.start()

            phase = .recording
            errorMessage = nil
            samplesCaptured = 0
            utterances = []
            capturedSamples.removeAll(keepingCapacity: true)

            // Pump 1 (raw): drain → SER buffer + level meter. The raw stream
            // preserves prosody for SER and prosody analyses.
            rawTask = Task { @MainActor [weak self] in
                guard let self else { return }
                for await buffer in streams.raw {
                    self.capturedSamples.append(contentsOf: buffer.samples)
                    self.samplesCaptured = self.capturedSamples.count
                    self.inputLevel = Self.smoothLevel(
                        previous: self.inputLevel,
                        current: Self.perceptualLevel(buffer.samples)
                    )
                }
                self.inputLevel = 0
            }

            // Pump 2 (processed): drain → ASR analyzer. Speech-band EQ applied
            // upstream by AVAudioUnitEQ (see SpeechBoost).
            feedTask = Task { @MainActor [weak self] in
                guard let self else { return }
                for await buffer in streams.processed {
                    await self.streamingTranscriber.feed(buffer)
                }
            }

            // Pump 2: drain finalized ASR segments → SER+fuse → append.
            analysisTask = Task { @MainActor [weak self] in
                guard let self else { return }
                let pipeline = await self.ensurePipeline()
                for await segment in segmentStream {
                    do {
                        let segmentBuffer = self.sliceForSegment(segment)
                        let estimate = try await pipeline.processSegment(
                            asr: segment,
                            segmentAudio: segmentBuffer,
                            speakerID: "S01"
                        )
                        self.utterances.append(estimate)
                    } catch {
                        AppLog.app.error("segment process failed: \(String(describing: error), privacy: .public)")
                    }
                }
            }
        } catch {
            errorMessage = String(describing: error)
            phase = .idle
            AppLog.app.error("recording start failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Stop capture, finalize the analyzer, drain remaining segments, then idle.
    func stop() async {
        await capture.stop()
        await rawTask?.value
        await feedTask?.value
        rawTask = nil
        feedTask = nil

        // Flushing remaining utterances may take a few seconds (SpeechAnalyzer
        // finalize + per-segment SER for any tail audio).
        phase = .analyzing
        await streamingTranscriber.finish()
        await analysisTask?.value
        analysisTask = nil
        phase = .idle
    }

    // MARK: - Audio input selection

    func refreshInputs() async {
        let inputs = await capture.availableInputs()
        let current = await capture.currentInput()
        self.availableInputs = inputs
        self.currentInputUID = current?.uid
    }

    func setSpeechBoostEnabled(_ enabled: Bool) async {
        await capture.setSpeechBoostEnabled(enabled)
        self.isSpeechBoostEnabled = enabled
    }

    func selectInput(uid: String?) async {
        do {
            try await capture.setPreferredInput(uid)
            await refreshInputs()
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error("setPreferredInput failed: \(String(describing: error), privacy: .public)")
        }
    }

    private func sliceForSegment(_ asr: ASRSegment) -> AudioChunk {
        let full = AudioChunk(
            samples: capturedSamples,
            sampleRate: PipelineAudio.sampleRate,
            timestamp: 0
        )
        return AnalysisPipeline.slice(full, start: asr.start, end: asr.end)
    }

    // MARK: - Level meter helpers

    /// RMS-of-buffer mapped from dB into a perceptual [0, 1].
    /// −60 dB → 0, 0 dB → 1.
    private static func perceptualLevel(_ samples: [Float]) -> Float {
        guard !samples.isEmpty else { return 0 }
        var sumSquares: Float = 0
        for s in samples { sumSquares += s * s }
        let rms = (sumSquares / Float(samples.count)).squareRoot()
        let db = 20 * log10f(max(rms, 1e-7))
        return max(0, min(1, (db + 60) / 60))
    }

    /// Fast attack, slow release — the classic VU-meter feel.
    private static func smoothLevel(previous: Float, current: Float) -> Float {
        if current >= previous {
            return current
        }
        return previous * 0.85 + current * 0.15
    }

    func exportJSON() async -> URL? {
        guard !utterances.isEmpty else { return nil }
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("xephon-\(Int(Date().timeIntervalSince1970)).json")
        do {
            try await exporter.write(utterances, to: url)
            return url
        } catch {
            errorMessage = String(describing: error)
            return nil
        }
    }
}
