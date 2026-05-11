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
    /// Total audio duration (in source seconds) of the file currently
    /// being analyzed, or nil when not in file mode / before the file
    /// has been probed. Used by the status line to render a completion
    /// percentage alongside wall-time and sample count.
    private(set) var fileTotalAudioDuration: TimeInterval?
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
    /// File URL whose utterances are currently loaded for playback.
    /// Non-nil iff the most recent (or in-progress) session was a
    /// file analysis. Cleared when a microphone session starts so
    /// the playback button doesn't linger on mic-recorded rows.
    /// Mutated through `setPlaybackSourceURL` so the matching
    /// security-scoped access ref is balanced.
    private(set) var playbackSourceURL: URL?
    /// URL we currently hold a `startAccessingSecurityScopedResource`
    /// ref on. Distinct from `playbackSourceURL` because the start
    /// call can fail (e.g. for a non-scoped URL); we only stash here
    /// after a successful start so the matching stop is balanced.
    /// `AudioFileCapture` takes its own ref during analysis — refs
    /// are independent, so dropping its ref at the end of analysis
    /// doesn't invalidate ours.
    private var scopedPlaybackURL: URL?
    /// ID of the utterance currently playing back, nil when nothing
    /// is playing. The row uses this to flip its play icon to stop
    /// and to disable other rows' buttons while one is mid-playback.
    private(set) var playingUtteranceID: UUID?
    private var playbackPlayer: AVAudioPlayer?
    private var playbackStopTask: Task<Void, Never>?
    /// Set once `modelStore.ensureModels()` succeeds. Until then the
    /// SetupView is shown in place of the main UI.
    private(set) var modelsReady: Bool = false

    var isRecording: Bool { phase == .recording }
    var isAnalyzing: Bool { phase == .analyzing }
    var isWarmingUp: Bool { phase == .warmingUp }
    var elapsedSeconds: Double {
        Double(samplesCaptured) / PipelineAudio.sampleRate
    }
    /// Fraction of the file's audio that has been captured so far, in
    /// `[0, 1]`. Nil when not in file mode or the duration probe at
    /// `startFromFile` couldn't read the file. Compares the pipeline's
    /// captured-sample audio time against the file's total duration —
    /// independent of pacing (fast-pace pumps samples through faster
    /// in wall-clock, but each sample still represents 1/16 ms of
    /// audio either way).
    var fileCompletionFraction: Double? {
        guard let total = fileTotalAudioDuration, total > 0 else { return nil }
        return min(1.0, max(0.0, elapsedSeconds / total))
    }
    /// Live size of the rolling capture buffer (the chunk SER + diarization
    /// slice from). Differs from `samplesCaptured`, which is monotonic for
    /// the whole session — this one shrinks every time `trimProcessedAudio`
    /// drops audio older than the diarization context window. Surfaced in
    /// the pipeline visualization so the developer can see the buffer's
    /// peak/steady-state footprint.
    var bufferedSamples: Int { capturedAudio.count }
    /// Distinct speakers detected across the most recent segment's
    /// sentence splits (i.e. the last `splitForProcessing` call's
    /// output). Surfaces in the pipeline visualization's Diarizer row
    /// as a per-chunk indicator, distinct from a session-wide
    /// cumulative — the row's job is to show what the diarizer just
    /// did, not the running total. 0 before any segment has been
    /// processed and resets each `start()`.
    private(set) var lastChunkSpeakerCount: Int = 0
    /// Sentence count of the most recently-finalized ASR segment —
    /// the number of sub-segments `splitIntoSentences` produced
    /// from it. Surfaces in the pipeline visualization's ASR row.
    /// 0 before the first segment finalizes; reset each `start()`.
    private(set) var lastChunkSentenceCount: Int = 0

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
    /// Sliding-window continuous diarization. Fires every
    /// `continuousDiarizeStrideSec` while recording and feeds the
    /// last `continuousDiarizeWindowSec` of audio to the pipeline's
    /// speaker timeline. The timeline is what `dominantSpeaker`
    /// queries per sentence — running it continuously rather than
    /// per-segment gives sharper speaker boundaries on fast-pace
    /// turn-takes (the per-segment 60 s context blurs them).
    private var continuousDiarizeTask: Task<Void, Never>?
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
            lastChunkSpeakerCount = 0
            lastChunkSentenceCount = 0
            // Stop any prior playback. For mic-mode this also clears
            // any prior file's playback ref via setPlaybackSourceURL.
            // File-mode acquires its scope earlier in `startFromFile`
            // — before any async hop — so the security-scoped URL is
            // pinned while it's still fresh from the picker.
            stopPlayback()
            if case .microphone = sourceMode {
                setPlaybackSourceURL(nil)
                fileTotalAudioDuration = nil
            }
            // Reset per-segment latencies so the pipeline visualization's
            // SER rows return to .idle when a new session starts.
            // Without this, lastAcousticDuration / lastTextDuration
            // carry over from the previous session, leaving the SER
            // glyphs latched to .ready before the first segment of the
            // new session has even processed.
            lastAcousticDuration = nil
            lastTextDuration = nil
            lastSegmentTotal = nil
            // Reset speaker recognition (cumulative timeline AND
            // FluidAudio's embedding-based SpeakerManager database)
            // BEFORE any pump task starts. Doing it here, in the
            // synchronous prologue of `start()`, guarantees no
            // observation from the new session can land before the
            // reset — earlier this lived inside `analysisTask` where
            // `continuousDiarizeTask` could in principle fire its
            // first tick before the reset awaited. The reset has to
            // run after `ensurePipeline` so the diarizer instance is
            // available to clear, but before the recording-time
            // tasks are spawned below.
            await ensurePipeline().resetSpeakerTracking()
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

            // Continuous diarization: every stride, snapshot the last
            // window of audio and feed it to the speaker timeline. The
            // timeline is what dominantSpeaker queries per sentence —
            // keeping it fresh means assignment is based on a recent
            // ~10 s window rather than a 60 s lump that averages
            // embeddings across many turns.
            //
            // First tick waits one stride, so a sentence finalizing in
            // the first ~2 s falls back to the supplied default ID via
            // dominantSpeaker's empty-timeline path. Subsequent ticks
            // get increasingly accurate coverage.
            //
            // Awaits `ensurePipeline` first because the diarizer lives
            // inside the pipeline. Skipped when the pipeline failed to
            // load a diarizer (`ingestDiarizationWindow` is a no-op).
            continuousDiarizeTask = Task { @MainActor [weak self] in
                guard let self else { return }
                let pipeline = await self.ensurePipeline()
                while !Task.isCancelled {
                    try? await Task.sleep(
                        nanoseconds: UInt64(Self.continuousDiarizeStrideSec * 1_000_000_000)
                    )
                    if Task.isCancelled { return }
                    guard self.isRecording else { continue }
                    let window = self.capturedAudio.snapshotTail(
                        seconds: Self.continuousDiarizeWindowSec
                    )
                    guard !window.samples.isEmpty else { continue }
                    await pipeline.ingestDiarizationWindow(window)
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
                // Speaker tracking + FluidAudio's SpeakerManager are
                // already reset in `start()`'s synchronous prologue
                // before this task is spawned, so no further reset
                // is needed here.
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
                        self.beginSegmentInflight()
                        group.addTask { [weak self] in
                            // Sentence-level split (pause + punctuation),
                            // then per-sentence speaker assignment from the
                            // cumulative diarizer timeline. Each split gets
                            // its own SER+fusion pass with the pre-resolved
                            // speaker. Single-sentence segments come back as
                            // a one-element array, so the loop is uniform.
                            // See `AnalysisPipeline.splitForProcessing`.
                            let splits = await pipeline.splitForProcessing(
                                asr: segment,
                                segmentAudio: segmentBuffer
                            )
                            // Distinct speakers across the splits → Diarizer
                            // pipeline row metric.
                            await self?.recordChunkSpeakerCount(
                                Set(splits.map(\.speaker)).count
                            )
                            await self?.recordChunkSentenceCount(splits.count)
                            for split in splits {
                                do {
                                    let (estimate, metrics) = try await pipeline.processSegment(
                                        asr: split.asr,
                                        segmentAudio: split.audio,
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

        continuousDiarizeTask?.cancel()
        continuousDiarizeTask = nil
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

        // File-backed sessions are one-shot: drop back to the live mic
        // so the next Record tap behaves normally. The file analysis
        // output remains in `utterances` for inspection/export.
        //
        // We deliberately DO NOT call `refreshInputs()` here. Doing so
        // routes through `AudioCapture.availableInputs()`, which
        // forces the AVAudioSession into `.record / .measurement /
        // [.allowBluetoothHFP]`. That config persists in the route
        // graph even when we later set `.playback` on top of it, and
        // `AVAudioPlayer.play()` returns true while the actual output
        // is silent. The inputs list will refresh the next time the
        // user presses Record (mic `start()` configures the session
        // explicitly anyway).
        if case .file = sourceMode {
            sourceMode = .microphone
            asrLatencyMeaningful = true
            capture = micCapture
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
        fastPaceMultiplier: Int = 8,
        audioOutputEnabled: Bool = false
    ) async {
        guard phase == .idle else { return }
        // Acquire the playback scope synchronously, before any await
        // hop. The picker's implicit grant for the URL is freshest
        // right after the dialog dismisses; by the time the analysis
        // task and AudioFileCapture's own scope ref have run, opening
        // a new ref later (e.g. when the user taps Playback) can fail
        // with permErr (-54). Holding our own ref here keeps the URL
        // readable for the lifetime of `playbackSourceURL`.
        setPlaybackSourceURL(url)
        // Probe the file's length so the status line can render a
        // completion percentage. AVAudioFile open is cheap and is
        // immediately discarded — the capture pump opens its own.
        // Falls back to nil when the file can't be parsed; the UI
        // hides the percentage in that case.
        fileTotalAudioDuration = {
            guard let f = try? AVAudioFile(forReading: url) else { return nil }
            let rate = f.processingFormat.sampleRate
            guard rate > 0, f.length > 0 else { return nil }
            return TimeInterval(Double(f.length) / rate)
        }()
        sourceMode = .file(url)
        // ASR finalize latency only makes sense when audio time tracks
        // wall-clock time. Fast-pace pumps audio multi-x faster than
        // real, so segment.end ≠ wall-clock-end, and the latency
        // computation would yield negative or meaningless deltas.
        asrLatencyMeaningful = realTimePacing
        capture = AudioFileCapture(
            fileURL: url,
            realTimePacing: realTimePacing,
            fastPaceMultiplier: fastPaceMultiplier,
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

    // MARK: - Per-utterance playback

    /// Assign `playbackSourceURL` while keeping the security-scoped
    /// access ref balanced. File-picker URLs only stay readable while
    /// some part of the app holds a ref via
    /// `startAccessingSecurityScopedResource()`; AudioFileCapture's
    /// ref drops at the end of analysis, so we hold our own ref here
    /// for the duration that the URL is exposed for playback. The
    /// `start` can fail for already-accessible URLs (e.g. in tests),
    /// which is fine — we just don't stash a stop counterpart.
    private func setPlaybackSourceURL(_ newURL: URL?) {
        if let scoped = scopedPlaybackURL {
            scoped.stopAccessingSecurityScopedResource()
            scopedPlaybackURL = nil
        }
        playbackSourceURL = newURL
        if let url = newURL {
            let ok = url.startAccessingSecurityScopedResource()
            AppLog.app.info(
                "playback scope start: \(ok ? "ok" : "skipped", privacy: .public) for \(url.lastPathComponent, privacy: .public)"
            )
            if ok { scopedPlaybackURL = url }
        }
    }

    /// Toggle playback of the audio range `[utterance.start, utterance.end]`
    /// from `playbackSourceURL`. No-op when there's no source URL (mic
    /// session) or when analysis is still running — the row gates the
    /// button so this is defense-in-depth. Tapping the row that's
    /// currently playing stops it; tapping a different row stops the
    /// previous playback and starts the new one.
    func togglePlayback(for utterance: UtteranceEstimate) {
        AppLog.app.info(
            "togglePlayback called: utt=\(utterance.id, privacy: .public) src=\(self.playbackSourceURL?.lastPathComponent ?? "nil", privacy: .public) phase=\(String(describing: self.phase), privacy: .public) scoped=\(self.scopedPlaybackURL?.lastPathComponent ?? "nil", privacy: .public)"
        )
        guard let url = playbackSourceURL else {
            AppLog.app.warning("togglePlayback: no playbackSourceURL")
            return
        }
        guard phase == .idle else {
            AppLog.app.warning("togglePlayback: phase not idle: \(String(describing: self.phase), privacy: .public)")
            return
        }
        if playingUtteranceID == utterance.id {
            stopPlayback()
            return
        }
        stopPlayback()
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            AppLog.app.info(
                "playback session BEFORE: category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
            AppLog.app.info(
                "playback session AFTER:  category=\(session.category.rawValue, privacy: .public) mode=\(session.mode.rawValue, privacy: .public) outputs=\(session.currentRoute.outputs.map(\.portType.rawValue).joined(separator: ","), privacy: .public)"
            )
        } catch {
            AppLog.app.warning("playback session setup failed: \(String(describing: error), privacy: .public)")
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            // Retain the player BEFORE calling play(). If the local
            // reference is the only retain at the moment of play(),
            // the optimizer is free to release it on the next line —
            // not normally an issue, but we've seen the case where
            // state doesn't progress past idle on the second session.
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, utterance.start)
            let didStart = player.play()
            AppLog.app.info(
                "togglePlayback: play() returned \(didStart ? "true" : "false", privacy: .public), duration=\(player.duration, privacy: .public)s, seek=\(player.currentTime, privacy: .public)s"
            )
            guard didStart else {
                AppLog.app.warning("playback failed to start for \(utterance.id, privacy: .public)")
                playbackPlayer = nil
                return
            }
            playingUtteranceID = utterance.id
            AppLog.app.info("togglePlayback: playingUtteranceID set to \(utterance.id, privacy: .public)")
            let duration = max(0, utterance.end - utterance.start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error("playback open failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Stop any in-flight playback. Safe to call when nothing is
    /// playing — it just clears the latch.
    func stopPlayback(caller: String = #function) {
        if playingUtteranceID != nil || playbackPlayer != nil {
            AppLog.app.info("stopPlayback called by \(caller, privacy: .public), playingID=\(self.playingUtteranceID?.uuidString ?? "nil", privacy: .public)")
        }
        playbackStopTask?.cancel()
        playbackStopTask = nil
        playbackPlayer?.stop()
        playbackPlayer = nil
        playingUtteranceID = nil
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

    /// Update the per-chunk speaker count surfaced on the Diarizer
    /// pipeline row. Called from the analysisTask immediately after
    /// `splitForProcessing` returns, so the row reflects the
    /// just-processed chunk rather than a running session total.
    fileprivate func recordChunkSpeakerCount(_ count: Int) {
        lastChunkSpeakerCount = count
    }

    fileprivate func recordChunkSentenceCount(_ count: Int) {
        lastChunkSentenceCount = count
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

    /// Sliding-window length for continuous diarization. 10 s is the
    /// shortest window where Sortformer's speaker embeddings are
    /// reliable — much shorter and the embedding noise dominates the
    /// turn-take signal. Going wider averages embeddings across more
    /// turns and undoes the whole reason we're running continuously.
    private static let continuousDiarizeWindowSec: TimeInterval = 10
    /// Stride between consecutive continuous-diarize calls. 2 s gives
    /// each new boundary < 2 s of latency before the timeline learns
    /// about it, while keeping diarizer load to ~0.5 calls/s. Going
    /// shorter spends CPU on overlapping windows that mostly agree;
    /// going longer makes fast-pace turn-takes show up late.
    private static let continuousDiarizeStrideSec: TimeInterval = 2

    private func trimProcessedAudio(below boundary: TimeInterval) {
        capturedAudio.trimProcessed(below: boundary)
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
