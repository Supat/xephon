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
    /// Bumped on every utterance mutation that doesn't change
    /// `utterances.count` (re-evaluate replaces in place, session
    /// import can in theory load a same-length list). The ContentView
    /// filter memo uses this as part of its dependency key so an
    /// in-place mutation forces the cache to rebuild — without it the
    /// re-evaluated row's old content would stay on screen.
    private(set) var utterancesVersion: Int = 0
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
    /// ID of the utterance currently being re-evaluated, nil when no
    /// re-evaluation is in flight. Used to drive the per-row spinner
    /// and to disable all playback / re-evaluate buttons across the
    /// list while one runs (offline ASR + SER serialize naturally and
    /// we don't want competing audio reads).
    private(set) var reevaluatingUtteranceID: UUID?
    /// Pre-first-reeval snapshots keyed by utterance id. Captured in
    /// `applyReevaluation` only on the FIRST re-evaluation of each
    /// row (so subsequent re-evals don't overwrite the truly-
    /// original streaming result). Cleared per row by
    /// `revertReevaluation`, and wholesale by `start()` / `loadSession`.
    /// Not persisted — revert history is session-scoped only.
    private var preReevaluationSnapshots: [UUID: UtteranceEstimate] = [:]
    /// Extra rows produced when a hand-edit commit contained more
    /// than one sentence. Keyed by the *parent* utterance id (the
    /// one that retains the original `id` after the split) and
    /// holding the new ids of every sibling row. Used by
    /// `revertReevaluation` so reverting the parent also cleans up
    /// the siblings the split spawned. Not persisted.
    private var handEditChildren: [UUID: [UUID]] = [:]
    /// User-supplied display names keyed by stored speaker id
    /// (e.g. `"S01" → "Alice"`). When present, takes precedence
    /// over the default `S01` / `M01` formatting that
    /// `formatSpeakerLabel` produces. Cleared at session start;
    /// restored from the `.xph` bundle on `loadSession`. The stored
    /// id stays canonical — diarizer matching, JSON identity, and
    /// per-speaker tint keying all keep operating on the original
    /// `S01`-style key.
    private(set) var speakerNameOverrides: [String: String] = [:]
    private var playbackPlayer: AVAudioPlayer?
    private var playbackStopTask: Task<Void, Never>?
    /// True while a `playRange(start:end:)` preview is in flight.
    /// Distinct from row-level playback (`playingUtteranceID`) so
    /// the Edit Utterance sheet's play button can toggle to a stop
    /// glyph without lighting up unrelated row controls. Flipped on
    /// inside `playRange` immediately after `player.play()` returns
    /// true, and back off in `stopPlayback()` (covers both the
    /// duration-elapsed auto-stop and the user-initiated tap).
    private(set) var isPreviewPlaying: Bool = false
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
                await self.handleAudioRouteChange()
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
        // Audible cue fires first, before any session-category
        // changes — the chime plays through the speaker while the
        // transcriber and capture spin up. `capture.start()` will
        // flip the session to `.record` a few hundred ms later;
        // by then the begin-record chime has already finished.
        UISounds.playRecordingStart()
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
            resetSessionState()
            await prepareSessionForRecording()
            spawnRawAudioPump(streams: streams)
            spawnTranscriberFeedPump(streams: streams)
            spawnVolatilePollPump()
            spawnContinuousDiarizeTask()
            spawnSegmentAnalysisTask(segmentStream: segmentStream)
            spawnFileEndWatcherIfNeeded()
        } catch {
            errorMessage = String(describing: error)
            phase = .idle
            AppLog.app.error("recording start failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Wipe the per-session state that `start()` should always
    /// start fresh: utterances, snapshots, override names,
    /// summary, audio buffer, capture counters, latencies. Pure
    /// in-memory clear — no async, no capture/pipeline mutation
    /// (those live in `prepareSessionForRecording`). Splitting the
    /// two halves means a failure in the async half leaves the
    /// in-memory state already coherent.
    private func resetSessionState() {
        errorMessage = nil
        samplesCaptured = 0
        utterances = []
        preReevaluationSnapshots.removeAll()
        handEditChildren.removeAll()
        speakerNameOverrides.removeAll()
        conversationSummary.reset()
        capturedAudio.reset()
        sessionStartedAt = Date()
        lastASRFinalizeLatency = nil
        lastChunkSpeakerCount = 0
        lastChunkSentenceCount = 0
        // Reset per-segment latencies so the pipeline visualization's
        // SER rows return to .idle when a new session starts.
        // Without this, lastAcousticDuration / lastTextDuration
        // carry over from the previous session, leaving the SER
        // glyphs latched to .ready before the first segment of the
        // new session has even processed.
        lastAcousticDuration = nil
        lastTextDuration = nil
        lastSegmentTotal = nil
    }

    /// External-state half of session startup: stop any prior
    /// playback, clear mic-mode playback refs, reset the diarizer's
    /// SpeakerManager + cumulative timeline, and start the Live
    /// Activity. Has to run after `resetSessionState` (so the new
    /// session's startedAt is set when LiveActivity captures it)
    /// and before any pump task spawns (so the SpeakerManager
    /// reset can't race a continuous-diarize tick).
    private func prepareSessionForRecording() async {
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
        // Reset speaker recognition (cumulative timeline AND
        // FluidAudio's embedding-based SpeakerManager database)
        // BEFORE any pump task starts.
        await ensurePipeline().resetSpeakerTracking()
        // Surface the session on the Lock Screen / Dynamic Island.
        // Has to happen AFTER `sourceMode` is set (handled by
        // `startFromFile` for file-mode) so the activity captures
        // the right source label.
        liveActivity.start(sourceLabel: liveActivitySourceLabel)
    }

    /// Pump 1 (raw): drain → SER buffer + level meter. The raw
    /// stream preserves prosody for SER and prosody analyses.
    /// `samplesCaptured` is a monotonic session counter (drives
    /// the elapsed-time display); it must NOT track
    /// `capturedAudio.count` because the rolling buffer is trimmed
    /// each time a segment is sliced for SER, which would make
    /// the timer jump backward.
    private func spawnRawAudioPump(streams: CaptureStreams) {
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
    }

    /// Pump 2 (processed): drain → ASR analyzer. Speech-band EQ
    /// is applied upstream by `AVAudioUnitEQ` (see `SpeechBoost`).
    private func spawnTranscriberFeedPump(streams: CaptureStreams) {
        feedTask = Task { @MainActor [weak self] in
            guard let self else { return }
            for await buffer in streams.processed {
                await self.streamingTranscriber.feed(buffer)
            }
        }
    }

    /// Volatile-text poll for the live ASR preview in the pipeline
    /// card. 5 Hz keeps it fluid without taxing the actor.
    private func spawnVolatilePollPump() {
        volatilePollTask = Task { @MainActor [weak self] in
            while !Task.isCancelled {
                guard let self else { return }
                self.volatileText = await self.streamingTranscriber.volatileText
                try? await Task.sleep(nanoseconds: 200_000_000)
            }
        }
    }

    /// Continuous diarization: every stride, snapshot the last
    /// window of audio and feed it to the speaker timeline. The
    /// timeline is what `dominantSpeaker` queries per sentence —
    /// keeping it fresh means assignment is based on a recent
    /// ~10 s window rather than a 60 s lump that averages
    /// embeddings across many turns.
    ///
    /// First tick waits one stride, so a sentence finalizing in
    /// the first ~2 s falls back to the supplied default ID via
    /// `dominantSpeaker`'s empty-timeline path. Subsequent ticks
    /// get increasingly accurate coverage.
    ///
    /// Awaits `ensurePipeline` first because the diarizer lives
    /// inside the pipeline. Skipped when the pipeline failed to
    /// load a diarizer (`ingestDiarizationWindow` is a no-op).
    private func spawnContinuousDiarizeTask() {
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
    }

    /// Drain finalized ASR segments → split → SER+fuse → append.
    /// Each segment processes concurrently (TaskGroup) so a slow
    /// text-SER LLM call on segment N doesn't block segment N+1.
    /// Results are inserted in start-time order regardless of which
    /// task finishes first.
    ///
    /// Concurrency is capped at `maxConcurrentSegments` to bound
    /// peak memory during fast-pace file analysis. Without the cap,
    /// the file capture pump can deliver tens of segments before
    /// any have finalized, each holding its own audio slice + ONNX
    /// I/O tensors — enough to OOM on long files. The poll-and-
    /// yield wait is coarse but lets the MainActor service updates
    /// between checks.
    private func spawnSegmentAnalysisTask(segmentStream: AsyncStream<ASRSegment>) {
        analysisTask = Task { @MainActor [weak self] in
            guard let self else { return }
            let pipeline = await self.ensurePipeline()
            // Speaker tracking + FluidAudio's SpeakerManager are
            // already reset in `prepareSessionForRecording()` before
            // this task is spawned, so no further reset is needed.
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
    }

    /// File-backed sources have a natural end. Watch for the feed
    /// loop to drain (the AudioFileCapture pump finishes both
    /// streams when the file is exhausted) and auto-stop so the
    /// user doesn't have to tap Stop. Mic captures never drain on
    /// their own, so this is a no-op for live recording. Spawned
    /// outside the awaited task graph so calling stop() from here
    /// doesn't deadlock against stop()'s own `await feedTask?.value`.
    private func spawnFileEndWatcherIfNeeded() {
        guard case .file = sourceMode else { return }
        let watchedFeed = feedTask
        fileEndWatcherTask = Task { @MainActor [weak self] in
            await watchedFeed?.value
            guard let self, self.isRecording else { return }
            AppLog.app.info("file source exhausted; auto-stopping")
            await self.stop()
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

        // End-of-record chime fires after capture has stopped (so
        // the session is no longer pinned to `.record`) but before
        // the longer SpeechAnalyzer finalize tail runs. Audible
        // confirmation lands within ~100 ms of the button press.
        UISounds.playRecordingStop()

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
        #if os(iOS) || targetEnvironment(macCatalyst)
        // `session.availableInputs` filters by the current category.
        // Under `.playback` (or default) it returns only the built-in
        // mic; Bluetooth inputs like AirPods only show up under
        // `.record` / `.playAndRecord` with bluetooth options. To
        // make the input picker reflect a freshly-connected AirPods
        // without committing to mic recording, briefly set the
        // category to `.playAndRecord` for the query, then restore
        // the previous declared category.
        //
        // We never call `setActive(true)`, so this is metadata-only
        // — the hardware route doesn't change, which is what kept
        // the playback-silence bug from coming back. Gated on idle
        // + no active playback because changing category on an
        // already-active session leaves the route latched (see
        // docs/playback_silence_postmortem.md).
        let canReconfigure = phase == .idle && playbackPlayer == nil
        let session = AVAudioSession.sharedInstance()
        let priorCategory = session.category
        let priorMode = session.mode
        let priorOptions = session.categoryOptions
        if canReconfigure {
            try? session.setCategory(
                .playAndRecord,
                mode: .default,
                options: [.allowBluetoothHFP, .defaultToSpeaker]
            )
        }
        #endif
        let inputs = await capture.availableInputs()
        let current = await capture.currentInput()
        self.availableInputs = inputs
        self.currentInputUID = current?.uid
        #if os(iOS) || targetEnvironment(macCatalyst)
        if canReconfigure {
            try? session.setCategory(priorCategory, mode: priorMode, options: priorOptions)
        }
        #endif
    }

    /// React to `AVAudioSession.routeChangeNotification`. Refreshes
    /// the visible input list and, when we're idle, fully deactivates
    /// the shared session so the next playback or record activation
    /// rebinds against the current route.
    ///
    /// Without this, AirPods that disconnect mid-app-session and then
    /// reconnect leave the session bound to a stale (dead) route:
    /// both AirPods playback and the built-in speaker silently fail
    /// until the user backgrounds the app, which forces the OS to
    /// reclaim the session. Deactivating here mimics that recovery
    /// path without requiring user intervention.
    ///
    /// Active playback gets a hard stop on any route change — Apple's
    /// human-interface guidance is that an unplugged or disconnected
    /// output should pause / stop playback rather than silently
    /// continue through whatever the OS falls back to.
    private func handleAudioRouteChange() async {
        await refreshInputs()
        if playbackPlayer != nil {
            stopPlayback()
        }
        guard phase == .idle else { return }
        #if os(iOS) || targetEnvironment(macCatalyst)
        try? AVAudioSession.sharedInstance().setActive(
            false,
            options: .notifyOthersOnDeactivation
        )
        #endif
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

    // MARK: - Session save / load

    /// Build a `SessionDocument` snapshot for the current state. For
    /// file-mode sessions, embeds the source audio bytes inline so
    /// playback round-trips after import; mic-mode sessions skip
    /// the audio block per the schema's no-playback-for-mic contract.
    ///
    /// Throws when the audio file can't be read (e.g. the picker's
    /// scope expired). Callers should ensure `playbackSourceURL` is
    /// fresh before calling — `togglePlayback`-tested URLs are good.
    func makeSessionDocument() async throws -> SessionDocument {
        let utts = utterances
        let names = speakerNameOverrides.isEmpty ? nil : speakerNameOverrides
        // Snapshot the diarizer's SpeakerManager so re-diarization
        // after Open Session lands on the same session-stable IDs
        // the original recording assigned. Nil when the diarizer
        // wasn't engaged (mic mode with no speech) or hasn't loaded
        // its models yet — both are normal and don't surface to the
        // user. Awaiting the pipeline here is what forced this
        // function to become async; callers run it from a Task.
        let speakerDB = await pipelineForExport()?.exportSpeakerDatabase()
        // Carry the pre-edit revert state alongside the utterances
        // so a long-press revert on a row that was hand-edited or
        // re-evaluated keeps working after Save → Open. Filter
        // snapshots whose target is no longer in `utterances` —
        // those would be orphans and the revert path would no-op
        // for them anyway. Empty maps round-trip as nil so v1-shaped
        // bundles (no revert state) stay byte-identical on save.
        let liveIDs = Set(utts.map(\.id))
        let snapshotsForExport = preReevaluationSnapshots.filter { liveIDs.contains($0.key) }
        let childrenForExport = handEditChildren.filter { liveIDs.contains($0.key) }
        let snapshots = snapshotsForExport.isEmpty ? nil : snapshotsForExport
        let children = childrenForExport.isEmpty ? nil : childrenForExport
        if let url = playbackSourceURL {
            let stillScoped = url.startAccessingSecurityScopedResource()
            defer {
                if stillScoped { url.stopAccessingSecurityScopedResource() }
            }
            do {
                let audioData = try Data(contentsOf: url)
                return SessionDocument(
                    sourceKind: .file,
                    audioFilename: url.lastPathComponent,
                    audio: audioData,
                    utterances: utts,
                    speakerNames: names,
                    speakerDatabase: speakerDB,
                    originalSnapshots: snapshots,
                    handEditChildren: children
                )
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "audio read failed: \(error.localizedDescription)"
                )
            }
        }
        return SessionDocument(
            sourceKind: .microphone,
            audioFilename: nil,
            audio: nil,
            utterances: utts,
            speakerNames: names,
            speakerDatabase: speakerDB,
            originalSnapshots: snapshots,
            handEditChildren: children
        )
    }

    /// Read-only access to the existing pipeline, without forcing
    /// initialization. `makeSessionDocument` uses it to skip the
    /// diarizer DB snapshot when no pipeline ever spun up (e.g.
    /// saving an imported session that was never re-analyzed).
    private func pipelineForExport() -> AnalysisPipeline? {
        pipeline
    }

    /// Replace the current session state with the contents of a
    /// previously-saved bundle. No-op when not idle so we never
    /// clobber an in-flight recording. Extracts the bundle's audio
    /// (if any) to a sandboxed temp file and wires it as the new
    /// playback source.
    func loadSession(_ document: SessionDocument) async throws {
        guard phase == .idle else { return }
        stopPlayback()
        utterances = document.utterances
        // Restore the pre-edit revert state from the bundle so a
        // long-press on a row's Edited / completed marker after
        // Open Session still rolls the row back to its original
        // streaming-pass record. `removeAll` first to drop any
        // leftover state from a prior session that wasn't cleared
        // (no recording started yet).
        preReevaluationSnapshots.removeAll()
        handEditChildren.removeAll()
        if let saved = document.originalSnapshots {
            preReevaluationSnapshots = saved
        }
        if let savedChildren = document.handEditChildren {
            handEditChildren = savedChildren
        }
        speakerNameOverrides = document.speakerNames ?? [:]
        // Same-length imports would otherwise hit the filter memo;
        // bump defensively so the cache rebuilds for any load.
        commitUtteranceChanges()
        lastChunkSpeakerCount = 0
        lastChunkSentenceCount = 0
        lastAcousticDuration = nil
        lastTextDuration = nil
        lastSegmentTotal = nil
        lastASRFinalizeLatency = nil
        // Imported sessions act like a finished file analysis: in
        // the .microphone source mode (so Record starts fresh) with
        // a playback URL pointing at the extracted audio (so the
        // per-row play button works).
        sourceMode = .microphone
        capture = micCapture
        if let audioData = document.audio, !audioData.isEmpty {
            let filename = document.audioFilename ?? "audio"
            let tempDir = FileManager.default.temporaryDirectory
                .appendingPathComponent(
                    "xephon-session-\(UUID().uuidString)",
                    isDirectory: true
                )
            do {
                try FileManager.default.createDirectory(
                    at: tempDir,
                    withIntermediateDirectories: true
                )
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "temp dir create failed: \(error.localizedDescription)"
                )
            }
            let destination = tempDir.appendingPathComponent(filename)
            do {
                try audioData.write(to: destination, options: .atomic)
            } catch {
                throw SessionBundle.BundleError.ioFailure(
                    "audio extract failed: \(error.localizedDescription)"
                )
            }
            setPlaybackSourceURL(destination)
            fileTotalAudioDuration = {
                guard let f = try? AVAudioFile(forReading: destination) else { return nil }
                let rate = f.processingFormat.sampleRate
                guard rate > 0, f.length > 0 else { return nil }
                return TimeInterval(Double(f.length) / rate)
            }()
        } else {
            setPlaybackSourceURL(nil)
            fileTotalAudioDuration = nil
        }
        // Restore the FluidAudio speaker DB when the saved bundle
        // carries one. Without this, a re-diarize on a hand-edited
        // slice would cluster against an empty SpeakerManager and
        // assign brand-new IDs that don't correspond to the
        // utterance rows' speaker labels. We swallow throws — a
        // stale blob (e.g. FluidAudio's `Speaker` schema changed
        // across versions) shouldn't prevent the user from opening
        // their session; they just lose the diarizer-restore
        // benefit and re-diarize starts from scratch.
        if let blob = document.speakerDatabase, !blob.isEmpty {
            do {
                try await ensurePipeline().importSpeakerDatabase(blob)
            } catch {
                AppLog.app.warning(
                    "loadSession: speaker DB restore failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
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

    // MARK: - Per-utterance re-evaluate

    /// Front-only padding applied before the original utterance's
    /// start when re-feeding audio to offline ASR. The streaming
    /// pass's finalizer cuts segments at the volatile-stabilization
    /// boundary, which often clips the first phoneme of an utterance;
    /// 500 ms of lead-in gives offline ASR a chance to recover it.
    /// No back padding is applied in the long-utterance path — the
    /// segment's tail is already preserved by streaming, and the
    /// sentence-aware trim in `AnalysisPipeline.reevaluate` drops
    /// anything past the last terminator anyway.
    private static let reevaluationPaddingSec: TimeInterval = 0.5

    /// Below this duration, the original streaming utterance is
    /// probably an incomplete fragment (the volatile-stabilization
    /// boundary cut mid-sentence). Re-evaluate then enters a retry
    /// loop, growing the back pad in steps until offline ASR
    /// produces a transcript containing a sentence terminator.
    private static let shortUtteranceThresholdSec: TimeInterval = 1.0
    /// Initial and per-iteration step for back padding when retrying
    /// the short-utterance case.
    private static let reevaluationBackPadStepSec: TimeInterval = 1.0
    /// Hard cap on back padding so a recording with no clean sentence
    /// boundary anywhere ahead doesn't keep growing the read forever.
    /// 10 s is comfortably longer than any realistic Japanese
    /// conversational sentence.
    private static let reevaluationMaxBackPadSec: TimeInterval = 10.0

    /// Surrounding-context pad (each side) for the audio chunk fed
    /// to the diarizer when re-resolving a speaker after a hand-
    /// edit commit or re-evaluation. Sortformer's input
    /// `chunkDuration` is 10 s; a typical edited slice is 2–5 s,
    /// which gets zero-padded — `createSegmentIfValid` then drops
    /// segments whose duration falls under `minSpeechDuration`,
    /// causing `runDiarization` to return empty. Giving the model
    /// real surrounding audio (the same audio it ran on during
    /// streaming) reliably yields segments. We still vote for
    /// speakers in the user-supplied range only; the wider window
    /// just gives Sortformer something to chew on. ~4 s on each
    /// side brings most slices comfortably above the 10 s window.
    private static let rediarizePadSec: TimeInterval = 4.0

    /// Re-feed the utterance's audio (padded by `reevaluationPaddingSec`
    /// on each side) to offline ASR, then run SER + fusion on the new
    /// result and replace the utterance in `utterances` in place. The
    /// utterance's `id`, `start`, `end`, `speakerID`, and `speechBoost`
    /// are preserved so list position, selection, and Save/Load
    /// identity all hold across the re-evaluation.
    ///
    /// No-op when there's no source audio (mic mode), when the
    /// session is mid-recording / mid-analysis, or when another
    /// re-evaluation is already in flight. The button gates these
    /// in the UI; the guards here are defense-in-depth.
    func reevaluate(_ utterance: UtteranceEstimate) async {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let url = playbackSourceURL else { return }

        reevaluatingUtteranceID = utterance.id
        // Reset any leftover preview from a prior recording. The
        // defer below ensures we also clear on every exit path,
        // including the early returns inside the do-block.
        volatileText = ""
        defer {
            reevaluatingUtteranceID = nil
            volatileText = ""
        }

        // Anchor padding to the *truly-original* utterance bounds —
        // the pre-first-reeval snapshot when one exists, otherwise
        // the current row's start/end (which on the first pass *is*
        // the original).
        //
        // Why: each re-evaluation produces a corrected start/end
        // that may differ from the original (the sentence-aware
        // trim landed on per-token anchors). If a subsequent
        // re-evaluation anchored its padding to the corrected
        // values, retrying on the same row would compound the shift
        // — pad of 500 ms before a slightly-earlier start, plus
        // 1 s after a slightly-later end, on each retry — and the
        // utterance's duration would grow without bound. Sourcing
        // anchors from the snapshot keeps re-evaluation idempotent.
        let snapshot = preReevaluationSnapshots[utterance.id]
        let originalStart = snapshot?.start ?? utterance.start
        let originalEnd = snapshot?.end ?? utterance.end
        let extendedStart = max(0, originalStart - Self.reevaluationPaddingSec)
        let speakerID = utterance.speakerID
        let originalDuration = originalEnd - originalStart
        let utteranceID = utterance.id

        let pipeline = await ensurePipeline()
        let volatileHandler: @Sendable @MainActor (String) -> Void = { [weak self] text in
            // Stream the offline ASR's rolling hypothesis into the
            // same `volatileText` slot the live ASR uses, so the
            // pipeline panel's preview animates during a
            // re-evaluation the same way it does during recording.
            // Awaited inside the transcriber, so no callbacks race
            // the defer's final clear.
            self?.volatileText = text
        }

        do {
            // Long-utterance path: single read, single ASR call, done.
            if originalDuration >= Self.shortUtteranceThresholdSec {
                let chunk = try await Task.detached(priority: .userInitiated) {
                    try Self.readAudioChunkForReevaluation(
                        fileURL: url,
                        start: extendedStart,
                        end: originalEnd
                    )
                }.value
                guard !chunk.samples.isEmpty else {
                    AppLog.app.warning("reevaluate: extended audio range was empty")
                    return
                }
                guard let (fresh, _) = try await pipeline.reevaluate(
                    audio: chunk,
                    originalStart: originalStart,
                    originalEnd: originalEnd,
                    speakerID: speakerID,
                    onVolatileText: volatileHandler
                ) else {
                    AppLog.app.warning("reevaluate: offline ASR returned no transcript")
                    return
                }
                let speaker = await rediarizedSpeaker(
                    url: url,
                    pipeline: pipeline,
                    correctedStart: fresh.start,
                    correctedEnd: fresh.end,
                    fallback: speakerID
                )
                applyReevaluation(
                    utteranceID: utteranceID,
                    fresh: fresh.withSpeakerID(speaker)
                )
                return
            }

            // Short-utterance path: streaming likely cut mid-sentence.
            // Grow back-padding in steps and rerun offline ASR until
            // the transcript contains a sentence terminator (or the
            // pad cap is reached, or the file is exhausted).
            //
            // No front pad on this path: when the original is shorter
            // than 1 s the streaming pass usually finalized late
            // (volatile-stabilization caught it mid-sentence), so
            // `utterance.start` already sits well inside the sentence
            // rather than at its true beginning. Adding 500 ms of
            // lead-in then drags in the *previous* sentence's tail,
            // and that tail's own terminator confuses the
            // `segmentsContainFullSentence` check — the loop sees a
            // terminator on iteration 1 from the prior sentence and
            // commits to a result that doesn't actually cover the
            // utterance we wanted to refresh.
            AppLog.app.info(
                "reevaluate: short utterance (\(originalDuration, privacy: .public)s) — entering back-pad retry loop"
            )
            var backPad: TimeInterval = Self.reevaluationBackPadStepSec
            var previousSampleCount = -1
            var matchedSegments: [ASRSegment]?
            var matchedAudio: AudioChunk?

            while backPad <= Self.reevaluationMaxBackPadSec {
                let currentEnd = originalEnd + backPad
                let chunk = try await Task.detached(priority: .userInitiated) {
                    try Self.readAudioChunkForReevaluation(
                        fileURL: url,
                        start: originalStart,
                        end: currentEnd
                    )
                }.value
                if chunk.samples.isEmpty {
                    AppLog.app.warning("reevaluate: empty read at backPad=\(backPad, privacy: .public)s")
                    break
                }
                if chunk.samples.count == previousSampleCount {
                    // File-end clamping returned the same audio as
                    // the previous iteration. No point in growing
                    // the pad further; offline ASR will produce the
                    // same output again.
                    AppLog.app.info(
                        "reevaluate: file exhausted at backPad=\(backPad, privacy: .public)s; stopping retry"
                    )
                    break
                }
                previousSampleCount = chunk.samples.count

                let segments = try await pipeline.transcribeForReevaluation(
                    audio: chunk,
                    onVolatileText: volatileHandler
                )
                if AnalysisPipeline.segmentsContainFullSentence(segments) {
                    AppLog.app.info(
                        "reevaluate: found full sentence at backPad=\(backPad, privacy: .public)s"
                    )
                    matchedSegments = segments
                    matchedAudio = chunk
                    break
                }
                AppLog.app.info(
                    "reevaluate: no terminator at backPad=\(backPad, privacy: .public)s; growing"
                )
                backPad += Self.reevaluationBackPadStepSec
            }

            guard let segments = matchedSegments, let chunk = matchedAudio else {
                AppLog.app.warning(
                    "reevaluate: no full sentence found within \(Self.reevaluationMaxBackPadSec, privacy: .public)s of back pad; preserving original"
                )
                return
            }
            guard let (fresh, _) = try await pipeline.reevaluateFromSegments(
                segments: segments,
                audio: chunk,
                originalStart: originalStart,
                originalEnd: originalEnd,
                speakerID: speakerID
            ) else {
                AppLog.app.warning("reevaluate: pipeline rejected segments after retry")
                return
            }
            let speaker = await rediarizedSpeaker(
                url: url,
                pipeline: pipeline,
                correctedStart: fresh.start,
                correctedEnd: fresh.end,
                fallback: speakerID
            )
            applyReevaluation(
                utteranceID: utteranceID,
                fresh: fresh.withSpeakerID(speaker)
            )
        } catch {
            AppLog.app.error("reevaluate failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Re-resolve the speaker for an utterance's corrected
    /// `[start, end]` window after re-evaluation. Reads a
    /// `rediarizePadSec`-wider chunk so Sortformer has enough
    /// surrounding speech to actually emit segments, then votes
    /// for the speaker covering the corrected range against the
    /// fresh segments only (so the cumulative timeline's ~5
    /// observations from the streaming pass don't outvote a
    /// genuinely-different verdict). Returns `fallback` on any
    /// read failure, on empty audio, or when the diarizer
    /// produces no segments for the slice.
    private func rediarizedSpeaker(
        url: URL,
        pipeline: AnalysisPipeline,
        correctedStart: TimeInterval,
        correctedEnd: TimeInterval,
        fallback: String
    ) async -> String {
        let totalDur: TimeInterval = fileTotalAudioDuration
            ?? max(correctedEnd + Self.rediarizePadSec, correctedEnd)
        let diarStart = max(0, correctedStart - Self.rediarizePadSec)
        let diarEnd = min(totalDur, correctedEnd + Self.rediarizePadSec)
        do {
            let chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: diarStart,
                    end: diarEnd
                )
            }.value
            guard !chunk.samples.isEmpty else { return fallback }
            let speakers = await pipeline.resolveSpeakersForRanges(
                audio: chunk,
                ranges: [(start: correctedStart, end: correctedEnd)],
                fallback: fallback
            )
            return speakers.first ?? fallback
        } catch {
            AppLog.app.warning(
                "reevaluate: speaker re-diarize read failed: \(String(describing: error), privacy: .public)"
            )
            return fallback
        }
    }

    /// Replace the utterance whose `id == utteranceID` with a merge
    /// of the original's identity (`id`, `speakerID`, `speechBoost`)
    /// and the freshly-computed content. `start` and `end` now come
    /// from `fresh` so the row's displayed timestamp updates to the
    /// re-evaluated audio range — typically a tighter span than the
    /// streaming pass produced, since the sentence-aware trim
    /// landed on per-token anchors. Stamps `wasReevaluated = true`
    /// so the marker rides along with the utterance through
    /// Save/Load and JSON export. Rebuilds the running conversation
    /// summary from scratch — ConversationSummary is an incremental
    /// fold with no "replace" path, and N is small enough that
    /// re-folding is cheap.
    ///
    /// After the replacement, re-sorts `utterances` if the corrected
    /// `start` moved the row out of chronological order — without
    /// this, a substantially-shifted re-eval could leave the list
    /// out of order, and List's identity-stable rendering would
    /// keep it at its old position visually.
    /// Bump `utterancesVersion` (so ContentView's filter memo
    /// invalidates after in-place mutations that leave
    /// `utterances.count` unchanged) and rebuild the conversation
    /// summary from the current `utterances`. Called after every
    /// path that mutates a row's content or replaces an entry.
    private func commitUtteranceChanges() {
        utterancesVersion &+= 1
        conversationSummary.reset()
        for u in utterances { conversationSummary.update(with: u) }
    }

    /// Build a merged `UtteranceEstimate` from the fresh SER/fusion
    /// output combined with stable origin fields (id, speakerID,
    /// speechBoost) and the per-flow edit flags. Used by the three
    /// "replace this row with new SER results" paths
    /// (`applyReevaluation`, `applyHandEdit`, `applyHandEditSplit`)
    /// to keep their constructor footprints small and consistent.
    private static func mergedEstimate(
        id: UUID,
        speakerID: String,
        speechBoost: Bool?,
        fresh: UtteranceEstimate,
        wasReevaluated: Bool?,
        wasHandEdited: Bool?
    ) -> UtteranceEstimate {
        UtteranceEstimate(
            id: id,
            speakerID: speakerID,
            start: fresh.start,
            end: fresh.end,
            transcript: fresh.transcript,
            asrConfidence: fresh.asrConfidence,
            dimensional: fresh.dimensional,
            acousticCategorical: fresh.acousticCategorical,
            plutchik: fresh.plutchik,
            textBackend: fresh.textBackend,
            speechBoost: speechBoost,
            wasReevaluated: wasReevaluated,
            wasHandEdited: wasHandEdited,
            fusedValence: fresh.fusedValence,
            fusedArousal: fresh.fusedArousal,
            fusedDominance: fresh.fusedDominance,
            fusedTopLabel: fresh.fusedTopLabel
        )
    }

    private func applyReevaluation(utteranceID: UUID, fresh: UtteranceEstimate) {
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else {
            return
        }
        let original = utterances[index]
        // Capture the truly-original (pre-first-reeval) snapshot so a
        // later long-press can restore it. Only on the FIRST re-eval
        // per row — subsequent re-evals leave the existing snapshot
        // alone, so revert always returns to the streaming result,
        // not the previous re-eval.
        if preReevaluationSnapshots[utteranceID] == nil {
            preReevaluationSnapshots[utteranceID] = original
        }
        // Carry `fresh.speakerID` — the caller in `reevaluate`
        // ran the diarizer on a context-padded window covering
        // the corrected `[fresh.start, fresh.end]` range and
        // stamped the resolved verdict on `fresh` via
        // `withSpeakerID`. The original streaming-pass speaker
        // assignment isn't authoritative for the corrected window,
        // so trusting `fresh` lets a re-evaluation fix speaker-
        // boundary errors at the same time it fixes the
        // transcript.
        utterances[index] = Self.mergedEstimate(
            id: original.id,
            speakerID: fresh.speakerID,
            speechBoost: original.speechBoost,
            fresh: fresh,
            wasReevaluated: true,
            wasHandEdited: nil
        )
        // If the corrected start moved the row out of chronological
        // order with its neighbours, re-sort. Cheap (small N, stable
        // sort) and keeps the list consistent for filtering /
        // selection /scroll. No-op when the shift was small enough
        // that the row's still in the right place.
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    // MARK: - Per-utterance hand-edit

    /// Commit a hand-edited utterance: new transcript text and new
    /// time range, with SER + fusion re-run on the audio slice
    /// `[newStart, newEnd]`. Identity is preserved (`id`,
    /// `speakerID`, `speechBoost`), `wasHandEdited` is stamped
    /// true, `wasReevaluated` is cleared. Captures the pre-edit
    /// snapshot first (sharing the same `preReevaluationSnapshots`
    /// store used by the re-evaluate revert path) so the long-press
    /// revert restores the truly-original streaming row.
    ///
    /// No-op when mic-mode (no source audio), when something else
    /// is already running, when the range is empty, or when the
    /// trimmed transcript is empty.
    func commitHandEdit(
        utteranceID: UUID,
        newText: String,
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) async {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let url = playbackSourceURL else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        guard newEnd > newStart else { return }
        let trimmedText = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }

        // Split on sentence-terminator punctuation (`。！？．.!?`) and
        // newlines. One sentence → existing single-row rewrite path;
        // multiple sentences → split the range proportionally by
        // character count and run SER+fusion per slice.
        let sentences = Self.splitTranscriptIntoSentences(trimmedText)
        guard !sentences.isEmpty else { return }
        let plans = Self.planHandEditSentences(
            sentences,
            newStart: newStart,
            newEnd: newEnd
        )
        guard !plans.isEmpty else { return }

        let original = utterances[index]
        reevaluatingUtteranceID = utteranceID
        defer { reevaluatingUtteranceID = nil }

        if preReevaluationSnapshots[utteranceID] == nil {
            preReevaluationSnapshots[utteranceID] = original
        }

        let fallbackSpeaker = original.speakerID
        let pipeline = await ensurePipeline()

        guard let chunks = await readHandEditAudio(
            url: url,
            newStart: newStart,
            newEnd: newEnd
        ) else { return }
        let (diarChunk, serChunk) = chunks

        if plans.count == 1 {
            await runSingleSentenceHandEdit(
                utteranceID: utteranceID,
                plan: plans[0],
                diarChunk: diarChunk,
                serChunk: serChunk,
                fallbackSpeaker: fallbackSpeaker,
                pipeline: pipeline
            )
            return
        }

        let freshEstimates = await runMultiSentenceHandEdit(
            plans: plans,
            diarChunk: diarChunk,
            url: url,
            fallbackSpeaker: fallbackSpeaker,
            pipeline: pipeline
        )
        guard !freshEstimates.isEmpty else { return }
        applyHandEditSplit(utteranceID: utteranceID, sentences: freshEstimates)
    }

    /// Read the two audio chunks `commitHandEdit` needs: a wide
    /// diarizer window (slice ± `rediarizePadSec`, clamped to
    /// file bounds) so Sortformer has real surrounding audio to
    /// work with, and the unpadded SER window the user committed.
    /// Both reads happen on the same detached task so the file
    /// I/O + AVAudioConverter pass don't hitch the MainActor.
    /// Returns nil on read failure or when the SER chunk came back
    /// empty (e.g. the user dialed past EOF).
    private func readHandEditAudio(
        url: URL,
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) async -> (diar: AudioChunk, ser: AudioChunk)? {
        let totalDur: TimeInterval = fileTotalAudioDuration
            ?? max(newEnd + Self.rediarizePadSec, newEnd)
        let diarStart = max(0, newStart - Self.rediarizePadSec)
        let diarEnd = min(totalDur, newEnd + Self.rediarizePadSec)
        let diarChunk: AudioChunk
        let serChunk: AudioChunk
        do {
            (diarChunk, serChunk) = try await Task.detached(priority: .userInitiated) {
                let dc = try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: diarStart,
                    end: diarEnd
                )
                let sc = try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: newStart,
                    end: newEnd
                )
                return (dc, sc)
            }.value
        } catch {
            AppLog.app.error(
                "commitHandEdit: parent slice read failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        guard !serChunk.samples.isEmpty else {
            AppLog.app.warning("commitHandEdit: empty audio slice")
            return nil
        }
        AppLog.app.info(
            "commitHandEdit: SER window [\(newStart, privacy: .public)..\(newEnd, privacy: .public)] s, diarizer window [\(diarStart, privacy: .public)..\(diarEnd, privacy: .public)] s (pad=\(Self.rediarizePadSec, privacy: .public)s)"
        )
        return (diarChunk, serChunk)
    }

    /// Single-sentence hand-edit: resolve the speaker for the
    /// whole window from the diarizer, then run SER + fusion on
    /// the unpadded SER chunk and stamp the result onto the row
    /// via `applyHandEdit`. No-op on pipeline error.
    private func runSingleSentenceHandEdit(
        utteranceID: UUID,
        plan: HandEditPlan,
        diarChunk: AudioChunk,
        serChunk: AudioChunk,
        fallbackSpeaker: String,
        pipeline: AnalysisPipeline
    ) async {
        let speakers = await pipeline.resolveSpeakersForRanges(
            audio: diarChunk,
            ranges: [(start: plan.start, end: plan.end)],
            fallback: fallbackSpeaker
        )
        let speaker = speakers.first ?? fallbackSpeaker
        do {
            let asr = ASRSegment(
                text: plan.text,
                start: plan.start,
                end: plan.end,
                // User-verified text is presumed correct, so weight
                // the text side at full confidence in fusion.
                confidence: 1.0,
                tokens: []
            )
            let (fresh, _) = try await pipeline.processSegment(
                asr: asr,
                segmentAudio: serChunk,
                fallbackSpeakerID: speaker
            )
            applyHandEdit(utteranceID: utteranceID, fresh: fresh)
        } catch {
            AppLog.app.error("commitHandEdit failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// Multi-sentence hand-edit: diarize ONCE on the full parent
    /// window (so Sortformer has speaker-boundary context the per-
    /// slice reads couldn't provide), then loop the SER pipeline
    /// per sentence-slice with the resolved speaker as fallback.
    /// Returns the freshly-fused per-sentence estimates in order
    /// (skipping any slice whose audio came back empty or whose
    /// pipeline call threw). Caller hands the result to
    /// `applyHandEditSplit`.
    private func runMultiSentenceHandEdit(
        plans: [HandEditPlan],
        diarChunk: AudioChunk,
        url: URL,
        fallbackSpeaker: String,
        pipeline: AnalysisPipeline
    ) async -> [UtteranceEstimate] {
        let sliceSpeakers = await pipeline.resolveSpeakersForRanges(
            audio: diarChunk,
            ranges: plans.map { ($0.start, $0.end) },
            fallback: fallbackSpeaker
        )
        var freshEstimates: [UtteranceEstimate] = []
        freshEstimates.reserveCapacity(plans.count)
        for (i, plan) in plans.enumerated() {
            let speaker = i < sliceSpeakers.count ? sliceSpeakers[i] : fallbackSpeaker
            do {
                let chunk = try await Task.detached(priority: .userInitiated) {
                    try Self.readAudioChunkForReevaluation(
                        fileURL: url,
                        start: plan.start,
                        end: plan.end
                    )
                }.value
                guard !chunk.samples.isEmpty else { continue }
                let asr = ASRSegment(
                    text: plan.text,
                    start: plan.start,
                    end: plan.end,
                    confidence: 1.0,
                    tokens: []
                )
                let (fresh, _) = try await pipeline.processSegment(
                    asr: asr,
                    segmentAudio: chunk,
                    fallbackSpeakerID: speaker
                )
                freshEstimates.append(fresh)
            } catch {
                AppLog.app.error(
                    "commitHandEdit split[\(i, privacy: .public)] failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
        return freshEstimates
    }

    /// One sentence's slot in a multi-sentence hand-edit commit:
    /// the trimmed sentence text plus the time range carved out of
    /// the user-supplied `[newStart, newEnd]` window for it.
    private struct HandEditPlan {
        let text: String
        let start: TimeInterval
        let end: TimeInterval
    }

    /// Allocate per-sentence time slots inside the user's edit
    /// window proportionally to character count. The last slot
    /// pins to `newEnd` exactly so floating-point drift doesn't
    /// leave a sub-millisecond gap. Returns `[]` when the input
    /// can't be sliced sensibly (empty sentences, zero total
    /// chars, or the slot collapsing to zero width).
    private static func planHandEditSentences(
        _ sentences: [String],
        newStart: TimeInterval,
        newEnd: TimeInterval
    ) -> [HandEditPlan] {
        let totalChars = sentences.reduce(0) { $0 + $1.count }
        guard totalChars > 0, newEnd > newStart else { return [] }
        let totalDuration = newEnd - newStart
        var plans: [HandEditPlan] = []
        plans.reserveCapacity(sentences.count)
        var charsConsumed = 0
        for (i, sentence) in sentences.enumerated() {
            let sliceStart = newStart
                + Double(charsConsumed) / Double(totalChars) * totalDuration
            charsConsumed += sentence.count
            let sliceEnd = i == sentences.count - 1
                ? newEnd
                : newStart
                    + Double(charsConsumed) / Double(totalChars) * totalDuration
            guard sliceEnd > sliceStart else { continue }
            plans.append(HandEditPlan(text: sentence, start: sliceStart, end: sliceEnd))
        }
        return plans
    }

    /// Mirror of `applyReevaluation` for the hand-edit path.
    /// Stamps `wasHandEdited = true` and clears `wasReevaluated`
    /// (the row is no longer a model-driven re-eval, it's a user
    /// correction).
    private func applyHandEdit(utteranceID: UUID, fresh: UtteranceEstimate) {
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]
        // Carry `fresh.speakerID` (resolved against the cumulative
        // diarizer timeline in `commitHandEdit`) rather than the
        // pre-edit `original.speakerID`, so a hand-edit that moved
        // the range across a speaker boundary actually re-labels.
        utterances[index] = Self.mergedEstimate(
            id: original.id,
            speakerID: fresh.speakerID,
            speechBoost: original.speechBoost,
            fresh: fresh,
            wasReevaluated: nil,
            wasHandEdited: true
        )
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    /// Multi-sentence hand-edit apply path. Replaces the parent row
    /// in place with the first sentence (keeping the original `id`,
    /// `speakerID`, and `speechBoost`) and inserts the remaining
    /// sentences as fresh-id siblings immediately after it. Every
    /// row gets `wasHandEdited = true` and `wasReevaluated = nil`.
    /// Sibling ids are tracked in `handEditChildren` so a revert on
    /// the parent can also remove the siblings the split spawned.
    private func applyHandEditSplit(
        utteranceID: UUID,
        sentences: [UtteranceEstimate]
    ) {
        guard !sentences.isEmpty else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]

        var newRows: [UtteranceEstimate] = []
        newRows.reserveCapacity(sentences.count)
        var siblingIDs: [UUID] = []
        for (i, fresh) in sentences.enumerated() {
            // First row keeps the original id (so existing
            // references stay valid); siblings get fresh ids and
            // are tracked for revert-time cleanup. Speaker is the
            // per-sentence diarizer verdict — different sentences
            // in a split can legitimately differ in speaker.
            let rowID = i == 0 ? original.id : UUID()
            if i > 0 { siblingIDs.append(rowID) }
            newRows.append(Self.mergedEstimate(
                id: rowID,
                speakerID: fresh.speakerID,
                speechBoost: original.speechBoost,
                fresh: fresh,
                wasReevaluated: nil,
                wasHandEdited: true
            ))
        }

        utterances.remove(at: index)
        utterances.insert(contentsOf: newRows, at: index)
        if !siblingIDs.isEmpty {
            handEditChildren[utteranceID] = siblingIDs
        }
        // Sentences carve up the original `[newStart, newEnd]` window
        // in order, so their starts are already monotonic among
        // themselves. Resort once to be safe in case a neighbouring
        // row overlaps the window (e.g. overlapping speakers).
        utterances.sort { $0.start < $1.start }

        commitUtteranceChanges()
    }

    /// Split a user-typed transcript into sentences. Terminators
    /// (`。！？．.!?` and `\n`) stay with the preceding sentence;
    /// trailing whitespace is trimmed; empty fragments are dropped.
    /// Returns `[trimmed]` when no terminator appears so the
    /// single-sentence path stays a no-op refactor of the previous
    /// behaviour.
    private static func splitTranscriptIntoSentences(_ text: String) -> [String] {
        let terminators: Set<Character> = ["。", "！", "？", "．", ".", "!", "?", "\n"]
        var sentences: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if terminators.contains(ch) {
                let trimmed = current.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty { sentences.append(trimmed) }
                current = ""
            }
        }
        let tail = current.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { sentences.append(tail) }
        return sentences
    }

    /// Play `[start, end]` from the source file. Used by the Edit
    /// Utterance dialog's preview button — the dialog's spinners
    /// hold arbitrary times that don't correspond to an existing
    /// utterance row, so `togglePlayback(for:)` doesn't apply.
    func playRange(start: TimeInterval, end: TimeInterval) {
        guard let url = playbackSourceURL else { return }
        guard phase == .idle else { return }
        guard end > start else { return }
        stopPlayback()
        #if os(iOS) || targetEnvironment(macCatalyst)
        do {
            let session = AVAudioSession.sharedInstance()
            try session.setCategory(.playback, mode: .default, options: [])
            try session.setActive(true)
        } catch {
            AppLog.app.warning(
                "playRange: session setup failed: \(String(describing: error), privacy: .public)"
            )
        }
        #endif
        do {
            let player = try AVAudioPlayer(contentsOf: url)
            playbackPlayer = player
            player.prepareToPlay()
            player.currentTime = max(0, start)
            guard player.play() else { return }
            isPreviewPlaying = true
            let duration = max(0, end - start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: UInt64(duration * 1_000_000_000))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error(
                "playRange failed: \(String(describing: error), privacy: .public)"
            )
        }
    }

    /// Set a custom display name for `stored` (e.g. `S01`). Pass an
    /// empty / whitespace-only `name` to clear the override (revert
    /// to the default `S01` / `M01` formatting). Bumps
    /// `utterancesVersion` so the ContentView filter memo
    /// invalidates and every row re-renders with the new label.
    func renameSpeaker(stored: String, to name: String) {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty {
            guard speakerNameOverrides.removeValue(forKey: stored) != nil else { return }
            AppLog.app.info(
                "speaker rename: cleared override for \(stored, privacy: .public)"
            )
        } else {
            if speakerNameOverrides[stored] == trimmed { return }
            speakerNameOverrides[stored] = trimmed
            AppLog.app.info(
                "speaker rename: \(stored, privacy: .public) → \(trimmed, privacy: .public)"
            )
        }
        utterancesVersion &+= 1
    }

    /// Custom display name for `stored` if the user has renamed it,
    /// nil otherwise. Convenience for the view layer.
    func speakerDisplayName(forStored stored: String) -> String? {
        speakerNameOverrides[stored]
    }

    /// Manually reassign one utterance's `speakerID`. Used by the
    /// speaker chip's action sheet to override the diarizer's
    /// verdict on a row whose voice the user knows belongs to a
    /// different speaker. The mutation is in-place — `id`, times,
    /// transcript, SER/fusion outputs all carry over — so list
    /// position and selection are stable. Bumps
    /// `utterancesVersion` so the filter memo invalidates and the
    /// chip re-renders with the new tint/label. No-op when the row
    /// is already on that speaker, when no row matches `utteranceID`,
    /// or when `newSpeakerID` is empty / whitespace-only.
    func reassignSpeaker(utteranceID: UUID, to newSpeakerID: String) {
        let trimmed = newSpeakerID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let original = utterances[index]
        guard original.speakerID != trimmed else { return }
        utterances[index] = original.withSpeakerID(trimmed)
        AppLog.app.info(
            "speaker reassigned: utt=\(original.id, privacy: .public) \(original.speakerID, privacy: .public) → \(trimmed, privacy: .public)"
        )
        commitUtteranceChanges()
    }

    /// Sorted, deduplicated list of speaker IDs currently appearing
    /// in the session — surface for the reassignment menu. Includes
    /// the currently-assigned speaker so the menu reads as a Picker
    /// (with the active row visually marked, not removed).
    func knownSpeakerIDs() -> [String] {
        Array(Set(utterances.map(\.speakerID))).sorted()
    }

    /// Restore the utterance to its pre-first-reeval state. Invoked
    /// by a 5-second long-press on the re-evaluate button. No-op
    /// when there's no snapshot for this id (the row was never
    /// re-evaluated) or when a re-evaluation is in flight (would
    /// race the upcoming write). Re-sorts if the original start
    /// time moved the row out of order, and bumps
    /// `utterancesVersion` so the filter memo invalidates.
    func revertReevaluation(_ utterance: UtteranceEstimate) {
        guard reevaluatingUtteranceID == nil else { return }
        guard phase == .idle else { return }
        guard let snapshot = preReevaluationSnapshots[utterance.id] else {
            return
        }
        // If this row spawned siblings via a multi-sentence hand-edit
        // commit, drop them so the revert leaves a single row again
        // rather than orphan sentences alongside the restored
        // original.
        if let siblingIDs = handEditChildren.removeValue(forKey: utterance.id), !siblingIDs.isEmpty {
            let drop = Set(siblingIDs)
            utterances.removeAll { drop.contains($0.id) }
        }
        guard let index = utterances.firstIndex(where: { $0.id == utterance.id }) else {
            preReevaluationSnapshots.removeValue(forKey: utterance.id)
            return
        }
        AppLog.app.info(
            "reevaluate: reverting utterance \(utterance.id, privacy: .public) to original snapshot"
        )
        utterances[index] = snapshot
        preReevaluationSnapshots.removeValue(forKey: utterance.id)
        if !Self.isChronologicallyOrdered(utterances, around: index) {
            utterances.sort { $0.start < $1.start }
        }
        commitUtteranceChanges()
    }

    /// True when the element at `index` is in chronological order
    /// relative to its immediate neighbours. Cheap O(1) check used
    /// by `applyReevaluation` to skip the sort when the corrected
    /// timestamp didn't actually move the row past anyone.
    private static func isChronologicallyOrdered(
        _ utterances: [UtteranceEstimate],
        around index: Int
    ) -> Bool {
        if index > 0, utterances[index - 1].start > utterances[index].start {
            return false
        }
        if index + 1 < utterances.count, utterances[index].start > utterances[index + 1].start {
            return false
        }
        return true
    }

    /// Read a sub-range of `fileURL` and resample to the pipeline's
    /// 16 kHz mono Float32 format. Used by `reevaluate` — runs on a
    /// detached task so the file I/O and AVAudioConverter pass don't
    /// hitch the MainActor. `start` and `end` are absolute seconds in
    /// the file's native timeline; both are clamped to `[0, duration]`
    /// here so a padded-past-EOF range comes back as a partial chunk
    /// rather than failing.
    nonisolated private static func readAudioChunkForReevaluation(
        fileURL: URL,
        start: TimeInterval,
        end: TimeInterval
    ) throws -> AudioChunk {
        let file: AVAudioFile
        do {
            file = try AVAudioFile(forReading: fileURL)
        } catch {
            throw AudioError.underlying(error)
        }
        let inputFormat = file.processingFormat
        let inputRate = inputFormat.sampleRate
        guard inputRate > 0 else {
            throw AudioError.unsupportedFormat(
                expected: ">0 Hz",
                got: "\(inputRate) Hz"
            )
        }
        let totalDuration = TimeInterval(Double(file.length) / inputRate)
        let clampedStart = max(0, min(start, totalDuration))
        let clampedEnd = max(clampedStart, min(end, totalDuration))
        let startFrame = AVAudioFramePosition(clampedStart * inputRate)
        let endFrame = AVAudioFramePosition(clampedEnd * inputRate)
        let inputFrames = AVAudioFrameCount(max(0, endFrame - startFrame))
        guard inputFrames > 0 else {
            return AudioChunk(
                samples: [],
                sampleRate: PipelineAudio.sampleRate,
                timestamp: clampedStart
            )
        }

        file.framePosition = startFrame
        guard let inputBuffer = AVAudioPCMBuffer(
            pcmFormat: inputFormat,
            frameCapacity: inputFrames
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "input PCM buffer",
                got: "alloc failed"
            )
        }
        do {
            try file.read(into: inputBuffer, frameCount: inputFrames)
        } catch {
            throw AudioError.underlying(error)
        }

        guard let outputFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: PipelineAudio.sampleRate,
            channels: AVAudioChannelCount(PipelineAudio.channelCount),
            interleaved: false
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "16 kHz mono Float32",
                got: "alloc failed"
            )
        }

        let sameFormat = inputFormat.sampleRate == outputFormat.sampleRate
            && inputFormat.channelCount == outputFormat.channelCount
            && inputFormat.commonFormat == outputFormat.commonFormat
            && inputFormat.isInterleaved == outputFormat.isInterleaved
        if sameFormat {
            guard let channelData = inputBuffer.floatChannelData else {
                throw AudioError.unsupportedFormat(
                    expected: "Float32 channel data",
                    got: "nil"
                )
            }
            let count = Int(inputBuffer.frameLength)
            let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
            return AudioChunk(
                samples: samples,
                sampleRate: PipelineAudio.sampleRate,
                timestamp: clampedStart
            )
        }

        guard let converter = AVAudioConverter(from: inputFormat, to: outputFormat) else {
            throw AudioError.unsupportedFormat(
                expected: String(describing: outputFormat),
                got: String(describing: inputFormat)
            )
        }
        converter.primeMethod = .none

        let ratio = outputFormat.sampleRate / inputFormat.sampleRate
        let outCapacity = AVAudioFrameCount(
            (Double(inputBuffer.frameLength) * ratio).rounded(.up)
        ) + 1024
        guard let outBuffer = AVAudioPCMBuffer(
            pcmFormat: outputFormat,
            frameCapacity: outCapacity
        ) else {
            throw AudioError.unsupportedFormat(
                expected: "output PCM buffer",
                got: "alloc failed"
            )
        }

        var convError: NSError?
        final class Once: @unchecked Sendable { var fired = false }
        let once = Once()
        let block: AVAudioConverterInputBlock = { _, status in
            if once.fired {
                status.pointee = .endOfStream
                return nil
            }
            once.fired = true
            status.pointee = .haveData
            return inputBuffer
        }
        let result = converter.convert(to: outBuffer, error: &convError, withInputFrom: block)
        if result == .error {
            throw AudioError.underlying(
                convError ?? NSError(domain: "Reevaluate", code: -1)
            )
        }

        guard let channelData = outBuffer.floatChannelData else {
            throw AudioError.unsupportedFormat(
                expected: "Float32 channel data",
                got: "nil"
            )
        }
        let count = Int(outBuffer.frameLength)
        let samples = Array(UnsafeBufferPointer(start: channelData[0], count: count))
        return AudioChunk(
            samples: samples,
            sampleRate: PipelineAudio.sampleRate,
            timestamp: clampedStart
        )
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
        isPreviewPlaying = false
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
