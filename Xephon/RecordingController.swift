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
    /// Wall-clock delay from "speaker finished this utterance"
    /// (`sessionStartedAt + segment.end`) to "analyzer emitted the final
    /// for it" (Date() at the analysisTask receipt). Indicates how long
    /// SpeechAnalyzer's volatile-stabilization window held the segment
    /// before promoting it to a final — typical 200–800 ms on M-class.
    /// Nil until the first finalize lands, and nil under fast-pace file
    /// analysis (audio time != wall-clock time, so the delta has no
    /// physical meaning).
    private(set) var lastASRFinalizeLatency: TimeInterval?
    private(set) var lastExportAt: Date?
    private(set) var inflightSegments: Int = 0
    private(set) var conversationSummary: ConversationSummary = ConversationSummary()
    /// Set once `modelStore.ensureModels()` succeeds. Until then the
    /// SetupView is shown in place of the main UI.
    private(set) var modelsReady: Bool = false

    var isRecording: Bool { phase == .recording }
    var isAnalyzing: Bool { phase == .analyzing }
    var isWarmingUp: Bool { phase == .warmingUp }
    var elapsedSeconds: Double {
        Double(samplesCaptured) / PipelineAudio.sampleRate
    }
    /// Live size of the rolling capture buffer (the chunk SER + diarization
    /// slice from). Differs from `samplesCaptured`, which is monotonic for
    /// the whole session — this one shrinks every time `trimProcessedAudio`
    /// drops audio older than the diarization context window. Surfaced in
    /// the pipeline visualization so the developer can see the buffer's
    /// peak/steady-state footprint.
    var bufferedSamples: Int { capturedAudio.count }
    /// Distinct speakers detected across the session. Derived from the
    /// `speakerID` field stamped onto each utterance by the diarization
    /// path; surfaces in the pipeline visualization's Diarizer row.
    var distinctSpeakers: Int {
        Set(utterances.map { $0.speakerID }).count
    }

    /// MainActor-confined progress mirror the SetupView observes during
    /// first-launch model hydration.
    let modelDownload: ModelDownloadState
    /// Optional because `ModelStore.init` can throw if Application Support
    /// is inaccessible (sandbox edge cases). Nil here surfaces immediately
    /// as a hydration failure instead of crashing in the controller's init.
    private let modelStore: ModelStore?
    /// Per-modality construction failures captured during pipeline
    /// pre-warm. Surfaced in the main UI as a small banner so the user
    /// knows when, e.g., the DeBERTa text SER silently dropped out
    /// because its FP16 model failed ORT load. Empty when everything
    /// loaded cleanly.
    private(set) var pipelineDiagnostics: [String] = []

    private var capture: any AudioCapture
    private let micCapture: any AudioCapture
    private let streamingTranscriber: any StreamingTranscriber
    private(set) var sourceMode: SourceMode = .microphone

    enum SourceMode: Equatable {
        case microphone
        case file(URL)
    }
    private var pipeline: AnalysisPipeline?
    private var pipelineTask: Task<AutoConfiguredPipeline, Never>?
    private let exporter = JSONExporter()
    private var rawTask: Task<Void, Never>?
    private var feedTask: Task<Void, Never>?
    private var analysisTask: Task<Void, Never>?
    private var routeWatcherTask: Task<Void, Never>?
    private var volatilePollTask: Task<Void, Never>?
    private var fileEndWatcherTask: Task<Void, Never>?
    /// Lock-Screen / Dynamic Island integration. See
    /// `LiveActivityController` for the activity-id, coalescing, and
    /// nonisolated update plumbing.
    private let liveActivity = LiveActivityController()
    /// Wall-clock time the current session began, captured at `start()`.
    /// Used by the Lock Screen / Dynamic Island clock so it ticks at
    /// real time even when fast-pace file analysis is racing through
    /// audio at multi-x speed (where `samplesCaptured / sampleRate`
    /// would jump in fast-forward).
    private var sessionStartedAt: Date?
    /// True when audio time progresses at wall-clock rate, so the ASR
    /// finalize-latency metric is physically meaningful. Mic mode is
    /// always wall-clock; file mode is wall-clock only when
    /// `realTimePacing` was passed to `startFromFile`.
    private var asrLatencyMeaningful: Bool = true
    /// Bounded rolling buffer over the raw capture stream. Owns the
    /// trim-before-append cap, the deep-copy-on-snapshot discipline,
    /// and the audio-time → buffer-index origin tracking. See
    /// `RollingAudioBuffer` for the invariants enforced.
    private var capturedAudio = RollingAudioBuffer(
        maxSeconds: 120,
        contextSeconds: 60
    )

    init(
        capture: any AudioCapture = AVAudioEngineCapture(),
        streamingTranscriber: (any StreamingTranscriber)? = nil,
        pipeline: AnalysisPipeline? = nil,
        modelStore: ModelStore? = nil
    ) {
        self.capture = capture
        self.micCapture = capture
        self.streamingTranscriber = streamingTranscriber ?? StreamingSpeechAnalyzerTranscriber()
        self.pipeline = pipeline
        // ModelDownloadState is @MainActor — RecordingController is too,
        // so it's built here (synchronously) and shared with the
        // non-isolated ModelStore actor by reference.
        let downloadState = ModelDownloadState()
        self.modelDownload = downloadState
        let resolvedStore: ModelStore?
        if let modelStore {
            resolvedStore = modelStore
        } else {
            do {
                resolvedStore = try ModelStore(state: downloadState)
            } catch {
                AppLog.app.error("ModelStore init failed: \(String(describing: error), privacy: .public)")
                downloadState.markFailed("Couldn't open the app's Application Support directory. Restart the app and try again. (\(error.localizedDescription))")
                resolvedStore = nil
            }
        }
        self.modelStore = resolvedStore
        // If a pipeline was injected (tests), assume models are ready.
        // Otherwise the SetupView gates the main UI until the
        // background hydration completes.
        if pipeline != nil {
            self.modelsReady = true
        }
        // Pre-warm the pipeline in the background at first construction so heavy
        // SER constructors (W2V2 ONNX ~631 MB) and the SpeechAnalyzer asset
        // install can complete before the user finishes their first sentence.
        // First-launch hydration of the models (download from GitHub Release
        // when the bundle doesn't ship them) happens inside the Task too —
        // SER constructors all depend on URLs the ModelStore resolves.
        if pipeline == nil && resolvedStore != nil {
            Task { @MainActor [weak self] in
                guard let self else { return }
                await self.hydrateAndWarm()
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
            let result = await task.value
            self.pipeline = result.pipeline
            self.pipelineDiagnostics = result.diagnostics
            self.pipelineTask = nil
            await refreshTextSERState(from: result.pipeline)
            return result.pipeline
        }
        guard let modelStore else {
            // Should be unreachable: modelsReady gates the main UI on
            // modelStore init success. Construct an empty pipeline so the
            // app degrades gracefully rather than hanging on an await.
            let configured = AnalysisPipeline()
            self.pipeline = configured
            return configured
        }
        let result = await AnalysisPipeline.autoConfigured(modelStore: modelStore)
        self.pipelineDiagnostics = result.diagnostics
        self.pipeline = result.pipeline
        await refreshTextSERState(from: result.pipeline)
        return result.pipeline
    }

    /// Run model hydration → pipeline warm → text-SER state refresh.
    /// Splits cleanly so the SetupView can call `retryModelDownload()`
    /// after a failure without re-spawning the wrapping init Task.
    private func hydrateAndWarm() async {
        guard let modelStore else { return }
        do {
            try await modelStore.ensureModels()
            self.modelsReady = true
        } catch {
            await MainActor.run {
                self.modelDownload.markFailed(String(describing: error))
            }
            AppLog.app.error("model hydration failed: \(String(describing: error), privacy: .public)")
            return
        }

        let warmTask = Task.detached(priority: .userInitiated) {
            let result = await AnalysisPipeline.autoConfigured(modelStore: modelStore)
            AppLog.app.info("Pipeline pre-warm complete")
            return result
        }
        self.pipelineTask = warmTask
        let result = await warmTask.value
        guard self.pipeline == nil else { return }
        self.pipeline = result.pipeline
        self.pipelineDiagnostics = result.diagnostics
        self.pipelineTask = nil
        await self.refreshTextSERState(from: result.pipeline)
    }

    /// Called from SetupView when the user taps Retry after a download
    /// failure. Resets the install dir and re-runs hydration.
    func retryModelDownload() async {
        guard let modelStore else { return }
        do {
            try await modelStore.resetForRedownload()
        } catch {
            AppLog.app.warning("resetForRedownload failed: \(String(describing: error), privacy: .public)")
        }
        modelDownload.reset()
        await hydrateAndWarm()
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
            conversationSummary.reset()
            capturedAudio.reset()
            sessionStartedAt = Date()
            lastASRFinalizeLatency = nil
            // Speaker history reset is sequenced inside `analysisTask`
            // below — it must complete BEFORE the first ingest, and a
            // fire-and-forget Task here can't make that guarantee.
            // Surface the session on the Lock Screen / Dynamic Island.
            // Has to happen AFTER `sourceMode` is set (handled by
            // `startFromFile` for file-mode) so the activity captures
            // the right source label.
            liveActivity.start(sourceLabel: liveActivitySourceLabel)

            // Pump 1 (raw): drain → SER buffer + level meter. The raw stream
            // preserves prosody for SER and prosody analyses.
            //
            // `samplesCaptured` is a monotonic session counter (drives the
            // elapsed-time display); it must NOT track `capturedAudio.count`
            // because the rolling buffer is trimmed each time a segment is
            // sliced for SER, which would make the timer jump backward.
            rawTask = Task { @MainActor [weak self] in
                guard let self else { return }
                for await buffer in streams.raw {
                    // The trim-before-append cap and origin advance
                    // both live inside RollingAudioBuffer. See its
                    // doc comment for why the order matters.
                    self.capturedAudio.append(buffer.samples)
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
            //
            // Concurrency is capped at `maxConcurrentSegments` to bound peak
            // memory during fast-pace file analysis. Without the cap, the
            // file capture pump can deliver tens of segments before any have
            // finalized, each holding its own audio slice + ONNX I/O tensors
            // — enough to OOM on long files. The poll-and-yield wait is
            // coarse but lets the MainActor service updates between checks.
            analysisTask = Task { @MainActor [weak self] in
                guard let self else { return }
                let pipeline = await self.ensurePipeline()
                // Clear cumulative speaker history before the first
                // segment lands. Awaiting the actor reset ensures the
                // tracker is empty by the time `processSegment` calls
                // `speakerTracker.ingest` — a previous fire-and-forget
                // Task could be scheduled after the first segment's
                // ingest, leaking last session's speaker numbering.
                await pipeline.resetSpeakerTracking()
                await withTaskGroup(of: Void.self) { group in
                    for await segment in segmentStream {
                        // Record finalize latency at the moment of receipt:
                        // wall-clock now minus the wall-clock when the audio
                        // for this utterance ended. Only meaningful when
                        // audio time tracks wall-clock (mic / real-time
                        // file). Under fast-pace the audio pump runs
                        // multi-x faster than real, so segment.end is in
                        // accelerated audio time and the delta would be
                        // negative — surface nil instead.
                        if self.asrLatencyMeaningful, let started = self.sessionStartedAt {
                            let endWallClock = started.addingTimeInterval(segment.end)
                            let latency = Date().timeIntervalSince(endWallClock)
                            self.lastASRFinalizeLatency = max(0, latency)
                        }
                        while self.inflightSegments >= Self.maxConcurrentSegments {
                            try? await Task.sleep(nanoseconds: 5_000_000)
                        }
                        let segmentBuffer = self.sliceForSegment(segment)
                        // Order matters here: slice → trim → snapshot.
                        //
                        // Slicing first reads the segment's range out of
                        // the full pre-trim buffer, so the SER task gets
                        // the right audio.
                        //
                        // Trimming next drops everything older than the
                        // diarization context window (~60 s before
                        // segment.end), which is exactly the audio the
                        // diarizer doesn't need anyway.
                        //
                        // Snapshotting AFTER the trim is the memory win —
                        // an earlier version snapshotted first and copied
                        // the entire pre-trim buffer (potentially 100s of
                        // MB if rawTask was ahead of analysisTask), making
                        // peak memory `inflightCap × pre-trim buffer size`.
                        // Now the snapshot is at most ~60 s × 16 kHz × 4 B
                        // = ~3.8 MB regardless of how far ASR is behind.
                        self.trimProcessedAudio(below: segment.end)
                        let diarContext = self.snapshotForDiarization()
                        self.beginSegmentInflight()
                        group.addTask { [weak self] in
                            // Diarize once, then split the segment at
                            // speaker-change token boundaries. Each sub
                            // gets its own SER+fusion pass with the
                            // pre-resolved speaker — passing
                            // `diarizationContext: nil` to processSegment
                            // skips re-running the diarizer per sub.
                            // Single-speaker segments come back as a
                            // one-element array, so the loop is uniform.
                            let splits = await pipeline.splitOnSpeakerChange(
                                asr: segment,
                                segmentAudio: segmentBuffer,
                                diarizationContext: diarContext
                            )
                            for split in splits {
                                do {
                                    let (estimate, metrics) = try await pipeline.processSegment(
                                        asr: split.asr,
                                        segmentAudio: split.audio,
                                        diarizationContext: nil,
                                        fallbackSpeakerID: split.speaker
                                    )
                                    await self?.applySegmentResult(estimate: estimate, metrics: metrics)
                                } catch {
                                    AppLog.app.error("segment process failed: \(String(describing: error), privacy: .public)")
                                }
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
        liveActivity.scheduleUpdate(currentLiveActivityState)
        await streamingTranscriber.finish()
        await analysisTask?.value
        analysisTask = nil
        phase = .idle
        await liveActivity.end(finalState: currentLiveActivityState)
        sessionStartedAt = nil

        // File-backed sessions are one-shot: drop back to the live mic so
        // the next Record tap behaves normally. The file analysis output
        // remains in `utterances` for inspection/export.
        if case .file = sourceMode {
            sourceMode = .microphone
            asrLatencyMeaningful = true
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
    /// - Parameter audioOutputEnabled: only meaningful with real-time
    ///   pacing — plays the file out the speaker alongside analysis so
    ///   the user can hear what's being transcribed. Ignored under
    ///   fast pacing (no useful audio at multi-x speed).
    func startFromFile(
        _ url: URL,
        realTimePacing: Bool = false,
        audioOutputEnabled: Bool = false
    ) async {
        guard phase == .idle else { return }
        sourceMode = .file(url)
        // ASR finalize latency only makes sense when audio time tracks
        // wall-clock time. Fast-pace pumps audio multi-x faster than
        // real, so segment.end ≠ wall-clock-end, and the latency
        // computation would yield negative or meaningless deltas.
        asrLatencyMeaningful = realTimePacing
        capture = AudioFileCapture(
            fileURL: url,
            realTimePacing: realTimePacing,
            audioOutputEnabled: audioOutputEnabled
        )
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
        conversationSummary.update(with: stamped)
        lastAcousticDuration = metrics.acousticDuration
        lastTextDuration = metrics.textDuration
        lastSegmentTotal = metrics.totalDuration
        // Coalesce Live Activity updates instead of spawning one task per
        // segment — fast-pace fires segments faster than the system can
        // process Activity updates, and the queued tasks themselves
        // contribute to MainActor congestion.
        liveActivity.scheduleUpdate(currentLiveActivityState)
    }

    // MARK: - Live Activity state

    /// Source label rendered in the Lock Screen / Dynamic Island.
    /// Pulled from `sourceMode` at start time.
    private var liveActivitySourceLabel: String {
        switch sourceMode {
        case .microphone: return "Microphone"
        case .file(let url): return url.lastPathComponent
        }
    }

    /// Current snapshot for the Live Activity. Lock Screen clock uses
    /// wall-clock since session start, NOT `samplesCaptured /
    /// sampleRate` — the latter accelerates wildly during fast-pace
    /// file analysis since the pump yields hours of audio in minutes.
    /// Recenters V/A to [-1, +1] for parity with the in-app summary.
    private var currentLiveActivityState: XephonActivityAttributes.ContentState {
        let liveElapsed = sessionStartedAt.map { Date().timeIntervalSince($0) } ?? elapsedSeconds
        return XephonActivityAttributes.ContentState(
            elapsedSeconds: liveElapsed,
            utteranceCount: utterances.count,
            topLabel: conversationSummary.topLabel,
            valence: conversationSummary.meanValence.map { $0 * 2 - 1 },
            arousal: conversationSummary.meanArousal.map { $0 * 2 - 1 },
            isAnalyzing: phase == .analyzing
        )
    }

    fileprivate func beginSegmentInflight() {
        inflightSegments += 1
    }

    fileprivate func endSegmentInflight() {
        if inflightSegments > 0 { inflightSegments -= 1 }
    }

    private func sliceForSegment(_ asr: ASRSegment) -> AudioChunk {
        capturedAudio.slice(start: asr.start, end: asr.end)
    }

    /// Cap on concurrent SER+fusion tasks. Fast-pace file analysis can
    /// otherwise stack dozens of in-flight segments before any finalize,
    /// each retaining its audio slice + ONNX I/O tensors — sufficient to
    /// OOM on long files. 2 is empirically the highest value that doesn't
    /// trigger `IOSurface creation failed: kIOReturnNoMemory` cascades
    /// from CoreML EP under sustained fast-pace pressure on 16 GB iPad
    /// Pro: each acoustic SER inference allocates IOSurface-backed
    /// tensors per running call, and 2 segments × 2 acoustic models
    /// (× 3 input bins worth of compiled MLModels each) saturates the
    /// system IOSurface pool. ASR typically emits 1-2 segments/sec so
    /// 2-wide concurrency is rarely the throughput bottleneck anyway.
    private static let maxConcurrentSegments: Int = 2

    private func trimProcessedAudio(below boundary: TimeInterval) {
        capturedAudio.trimProcessed(below: boundary)
    }

    private func snapshotForDiarization() -> AudioChunk {
        capturedAudio.snapshotForDiarization()
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
