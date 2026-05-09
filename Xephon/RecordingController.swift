import Foundation
import Observation
@preconcurrency import AVFoundation
import Audio
import ASR
import Fusion
import Export
import SERText
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
    private(set) var availableTextSERBackends: [SwitchingTextSER.Backend] = []
    private(set) var currentTextSERBackend: SwitchingTextSER.Backend?
    private(set) var volatileText: String = ""
    private(set) var lastAcousticDuration: TimeInterval?
    private(set) var lastTextDuration: TimeInterval?
    private(set) var lastSegmentTotal: TimeInterval?
    private(set) var lastExportAt: Date?
    private(set) var inflightSegments: Int = 0

    var isRecording: Bool { phase == .recording }
    var isAnalyzing: Bool { phase == .analyzing }
    var isWarmingUp: Bool { phase == .warmingUp }
    var elapsedSeconds: Double {
        Double(samplesCaptured) / PipelineAudio.sampleRate
    }

    private var capture: any AudioCapture
    private let micCapture: any AudioCapture
    private let streamingTranscriber: any StreamingTranscriber
    private(set) var sourceMode: SourceMode = .microphone

    enum SourceMode: Equatable {
        case microphone
        case file(URL)
    }
    private var pipeline: AnalysisPipeline?
    private var pipelineTask: Task<AnalysisPipeline, Never>?
    private let exporter = JSONExporter()
    private var rawTask: Task<Void, Never>?
    private var feedTask: Task<Void, Never>?
    private var analysisTask: Task<Void, Never>?
    private var routeWatcherTask: Task<Void, Never>?
    private var volatilePollTask: Task<Void, Never>?
    private var fileEndWatcherTask: Task<Void, Never>?
    private var capturedSamples: [Float] = []
    /// Audio-time (seconds) of `capturedSamples[0]`. Starts at 0 and advances
    /// each time we trim already-processed audio from the head of the buffer.
    /// Without this, sample-rate × audio-time indexing would land at the
    /// wrong frame after a trim. Long sessions would otherwise grow
    /// `capturedSamples` linearly forever (~64 KB / sec at 16 kHz mono Float32).
    private var capturedSamplesOrigin: TimeInterval = 0

    init(
        capture: any AudioCapture = AVAudioEngineCapture(),
        streamingTranscriber: (any StreamingTranscriber)? = nil,
        pipeline: AnalysisPipeline? = nil
    ) {
        self.capture = capture
        self.micCapture = capture
        self.streamingTranscriber = streamingTranscriber ?? StreamingSpeechAnalyzerTranscriber()
        self.pipeline = pipeline
        // Pre-warm the pipeline in the background at first construction so heavy
        // SER constructors (W2V2 ONNX ~631 MB) and the SpeechAnalyzer asset
        // install can complete before the user finishes their first sentence.
        if pipeline == nil {
            let warmTask = Task.detached(priority: .userInitiated) {
                let configured = await AnalysisPipeline.autoConfigured()
                AppLog.app.info("Pipeline pre-warm complete")
                return configured
            }
            self.pipelineTask = warmTask
            // Surface text-SER backend availability as soon as the pre-warm
            // resolves, so the DeBERTa/Apple FM picker can appear before the
            // user records for the first time.
            Task { @MainActor [weak self] in
                let configured = await warmTask.value
                guard let self, self.pipeline == nil else { return }
                self.pipeline = configured
                self.pipelineTask = nil
                await self.refreshTextSERState(from: configured)
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
            await refreshTextSERState(from: configured)
            return configured
        }
        let configured = await AnalysisPipeline.autoConfigured()
        self.pipeline = configured
        await refreshTextSERState(from: configured)
        return configured
    }

    private func refreshTextSERState(from pipeline: AnalysisPipeline) async {
        availableTextSERBackends = await pipeline.availableTextSERBackends()
        currentTextSERBackend = await pipeline.currentTextSERBackend()
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
            // If `capture.start()` throws here, the transcriber's analyzer +
            // drainer are already alive and would leak (subsequent
            // recordings would stack new ones on top). Tear it down
            // explicitly before propagating, so failure paths leave the
            // controller in a clean idle state.
            let streams: CaptureStreams
            do {
                streams = try await capture.start()
            } catch {
                await streamingTranscriber.finish()
                throw error
            }

            phase = .recording
            errorMessage = nil
            samplesCaptured = 0
            utterances = []
            capturedSamples.removeAll(keepingCapacity: true)
            capturedSamplesOrigin = 0

            // Pump 1 (raw): drain → SER buffer + level meter. The raw stream
            // preserves prosody for SER and prosody analyses.
            //
            // `samplesCaptured` is a monotonic session counter (drives the
            // elapsed-time display); it must NOT track `capturedSamples.count`
            // because the rolling buffer is trimmed each time a segment is
            // sliced for SER, which would make the timer jump backward.
            rawTask = Task { @MainActor [weak self] in
                guard let self else { return }
                for await buffer in streams.raw {
                    self.capturedSamples.append(contentsOf: buffer.samples)
                    self.samplesCaptured += buffer.samples.count
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

            // Volatile-text poll for the live ASR preview in the pipeline card.
            // 5 Hz keeps it fluid without taxing the actor.
            volatilePollTask = Task { @MainActor [weak self] in
                while !Task.isCancelled {
                    guard let self else { return }
                    self.volatileText = await self.streamingTranscriber.volatileText
                    try? await Task.sleep(nanoseconds: 200_000_000)
                }
            }

            // Pump 2: drain finalized ASR segments → SER+fuse → append.
            // Each segment is processed concurrently (TaskGroup) so a slow
            // text-SER LLM call on segment N doesn't block segment N+1.
            // Results are inserted in start-time order regardless of which
            // task finishes first.
            analysisTask = Task { @MainActor [weak self] in
                guard let self else { return }
                let pipeline = await self.ensurePipeline()
                await withTaskGroup(of: Void.self) { group in
                    for await segment in segmentStream {
                        let segmentBuffer = self.sliceForSegment(segment)
                        // The SER task owns `segmentBuffer` (a copy);
                        // it's safe to drop the head of `capturedSamples`
                        // up to this segment's end now. Bounds session
                        // memory: an 8-hour recording stays at a few MB
                        // instead of growing to ~1.8 GB.
                        self.trimProcessedAudio(below: segment.end)
                        self.beginSegmentInflight()
                        group.addTask { [weak self] in
                            do {
                                let (estimate, metrics) = try await pipeline.processSegment(
                                    asr: segment,
                                    segmentAudio: segmentBuffer,
                                    speakerID: "S01"
                                )
                                await self?.applySegmentResult(estimate: estimate, metrics: metrics)
                            } catch {
                                AppLog.app.error("segment process failed: \(String(describing: error), privacy: .public)")
                            }
                            await self?.endSegmentInflight()
                        }
                    }
                }
            }

            // File-backed sources have a natural end. Watch for the feed loop
            // to drain (the AudioFileCapture pump finishes both streams when
            // the file is exhausted) and auto-stop so the user doesn't have
            // to tap Stop. Mic captures never drain on their own, so this is
            // a no-op for live recording. Spawned outside the awaited task
            // graph so calling stop() from here doesn't deadlock against
            // stop()'s own `await feedTask?.value`.
            if case .file = sourceMode {
                let watchedFeed = feedTask
                fileEndWatcherTask = Task { @MainActor [weak self] in
                    await watchedFeed?.value
                    guard let self, self.isRecording else { return }
                    AppLog.app.info("file source exhausted; auto-stopping")
                    await self.stop()
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
        // Re-entrancy guard. Manual Stop and the file-end watcher can both
        // race to call this — the watcher's `await self.stop()` is queued
        // before we cancel its task on the manual path, so two `stop()`
        // calls reach this point sequentially (MainActor serializes them).
        // Without this guard, the second call re-runs
        // `streamingTranscriber.finish()` on an already-finalized analyzer,
        // which is undefined behavior.
        guard phase == .recording else { return }

        // Don't await the auto-stop watcher (it's the caller in the
        // file-end path). Cancel it so manual Stop also tears it down.
        fileEndWatcherTask?.cancel()
        fileEndWatcherTask = nil

        await capture.stop()
        await rawTask?.value
        await feedTask?.value
        rawTask = nil
        feedTask = nil

        volatilePollTask?.cancel()
        volatilePollTask = nil
        volatileText = ""

        // Flushing remaining utterances may take a few seconds (SpeechAnalyzer
        // finalize + per-segment SER for any tail audio).
        phase = .analyzing
        await streamingTranscriber.finish()
        await analysisTask?.value
        analysisTask = nil
        phase = .idle

        // File-backed sessions are one-shot: drop back to the live mic so
        // the next Record tap behaves normally. The file analysis output
        // remains in `utterances` for inspection/export.
        if case .file = sourceMode {
            sourceMode = .microphone
            capture = micCapture
            await refreshInputs()
        }
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

    func setTextSERBackend(_ backend: SwitchingTextSER.Backend) async {
        let pipeline = await ensurePipeline()
        await pipeline.setTextSERBackend(backend)
        await refreshTextSERState(from: pipeline)
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

    // MARK: - File-backed source

    /// Switch to a file-backed audio source and immediately begin streaming
    /// it through the same pipeline used for the microphone.
    ///
    /// - Parameter realTimePacing: when `true`, audio is yielded at the
    ///   file's wall-clock duration so SpeechAnalyzer behaves identically
    ///   to a live recording (best ASR/SER quality). When `false` (default),
    ///   audio is yielded as fast as the analyzer can ingest, which can
    ///   complete several times faster than real time at some risk to
    ///   accuracy on long files.
    func startFromFile(_ url: URL, realTimePacing: Bool = false) async {
        guard phase == .idle else { return }
        sourceMode = .file(url)
        capture = AudioFileCapture(fileURL: url, realTimePacing: realTimePacing)
        availableInputs = []
        currentInputUID = nil
        await start()
    }

    /// Restore the microphone as the active source. Called automatically when
    /// a file-backed session ends.
    func resetToMicrophone() async {
        guard phase == .idle else { return }
        sourceMode = .microphone
        capture = micCapture
        await refreshInputs()
    }

    /// Called from a TaskGroup child once a segment finishes processing.
    /// Inserts the utterance at the correct chronological index and updates
    /// the per-stage latency snapshot used by the pipeline visualization.
    private func applySegmentResult(estimate: UtteranceEstimate, metrics: ProcessingMetrics) {
        // Stamp speech-boost state so the row badge reflects how the audio
        // was captured. Best-effort: reads the live toggle, which is fine
        // for the common case where it's not flipped mid-recording.
        let stamped = estimate.withSpeechBoost(isSpeechBoostEnabled)
        let index = utterances.firstIndex { $0.start > stamped.start } ?? utterances.endIndex
        utterances.insert(stamped, at: index)
        lastAcousticDuration = metrics.acousticDuration
        lastTextDuration = metrics.textDuration
        lastSegmentTotal = metrics.totalDuration
    }

    fileprivate func beginSegmentInflight() {
        inflightSegments += 1
    }

    fileprivate func endSegmentInflight() {
        if inflightSegments > 0 { inflightSegments -= 1 }
    }

    private func sliceForSegment(_ asr: ASRSegment) -> AudioChunk {
        let sampleRate = PipelineAudio.sampleRate
        let localStart = max(0, asr.start - capturedSamplesOrigin)
        let localEnd   = max(localStart, asr.end - capturedSamplesOrigin)
        let startIndex = min(Int(localStart * sampleRate), capturedSamples.count)
        let endIndex   = min(Int(localEnd   * sampleRate), capturedSamples.count)
        guard startIndex < endIndex else {
            return AudioChunk(samples: [], sampleRate: sampleRate, timestamp: asr.start)
        }
        let slice = Array(capturedSamples[startIndex..<endIndex])
        return AudioChunk(samples: slice, sampleRate: sampleRate, timestamp: asr.start)
    }

    /// Drop processed audio up to `boundary` (audio-time seconds) from the
    /// head of `capturedSamples`. Called after each segment is sliced —
    /// the SER task already owns its own copy of the slice, so trimming
    /// here is safe. Bounds session-long memory growth: peak capture
    /// memory is now roughly the longest single utterance plus the
    /// not-yet-finalized rolling buffer, regardless of session length.
    private func trimProcessedAudio(below boundary: TimeInterval) {
        let dropSeconds = boundary - capturedSamplesOrigin
        guard dropSeconds > 0 else { return }
        let frames = min(Int(dropSeconds * PipelineAudio.sampleRate), capturedSamples.count)
        guard frames > 0 else { return }
        capturedSamples.removeFirst(frames)
        capturedSamplesOrigin += Double(frames) / PipelineAudio.sampleRate
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
            lastExportAt = Date()
            return url
        } catch {
            errorMessage = String(describing: error)
            return nil
        }
    }
}
