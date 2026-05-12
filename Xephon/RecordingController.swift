import Foundation
import Observation
import os
@preconcurrency import AVFoundation
import Audio
import ASR
import Diarization
import Fusion
import Export
import FoundationModels
import SERText
import Summarizer
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
    /// Rotates every time the session-level utterance list changes
    /// identity (new recording, new file analysis, imported `.xph`).
    /// Views observe this to flush transient `@State` keyed by the
    /// prior session's UUIDs — `visibleUtteranceIDs`,
    /// `expandedUtteranceIDs`, `selectedUtteranceID`, etc. The
    /// controller can't reach those bindings directly, so a token
    /// the view watches is the cleanest seam.
    private(set) var sessionToken: UUID = UUID()
    private(set) var utterances: [UtteranceEstimate] = []
    private(set) var inputLevel: Float = 0
    private(set) var availableInputs: [AudioInputDescription] = []
    private(set) var currentInputUID: String?
    private(set) var isSpeechBoostEnabled: Bool = true
    private(set) var availableTextSERBackends: [SwitchingTextSER.Backend] = []
    private(set) var currentTextSERBackend: SwitchingTextSER.Backend?
    /// Active session language. Drives the ASR locale (Apple
    /// SpeechTranscriber + offline transcriber), the FoundationModels
    /// prompt opener, and the DeBERTa-WRIME availability gate
    /// (Japanese-only model — hidden from the Text SER picker when
    /// the session isn't Japanese). User-selectable in the Settings
    /// card while idle; locked while a session is in flight.
    /// Persists across launches via UserDefaults.
    private(set) var sessionLanguage: SessionLanguage

    /// Whether the on-device session summarizer (Qwen2.5-7B 4-bit
    /// MLX) is opted-in. Flipping this on triggers the ~4 GB model
    /// download via `ModelStore.ensureOptional`; flipping it off
    /// unloads the resident weights but leaves the on-disk files
    /// alone (the user can free them via "Remove summarizer model").
    /// Persists in UserDefaults so the choice survives launch.
    private(set) var summarizerEnabled: Bool
    /// User's chosen summarizer backend. Apple FM is the default
    /// — system-managed, no model download, lighter on memory.
    /// Qwen is the higher-quality opt-in (requires ~4.3 GB
    /// install). Persisted via UserDefaults so the choice
    /// survives launch.
    private(set) var summarizerBackend: SummarizerBackend
    /// Whether Apple's `SystemLanguageModel.default` is available
    /// on this device. Refreshed at init and on backend change.
    /// Folded into `summarizerReady` for the toolbar button gate.
    private(set) var summarizerAppleFMAvailable: Bool = false
    /// True iff every file declared by the summarizer's optional
    /// manifest entry is present on disk. Doesn't re-hash on every
    /// read — fast enough for the Settings card to bind to.
    private(set) var summarizerModelInstalled: Bool = false
    /// True while `ModelStore.ensureOptional` is in flight. Drives
    /// the inline progress chrome next to the Settings toggle.
    private(set) var summarizerDownloading: Bool = false
    /// True while a `summarizeSession()` call is generating tokens.
    /// Disables the "Summarize session" toolbar button mid-run.
    private(set) var summarizerInferenceRunning: Bool = false
    /// Last successful summary, cached so the result sheet survives
    /// re-presentation without re-running inference. Cleared on
    /// session start.
    private(set) var lastSessionSummary: SessionSummary?

    /// The actor that owns the loaded MLX weights once they're in
    /// memory. Lazy-created on first `summarizeSession()` call and
    /// dropped when the user disables the summarizer or starts a
    /// new session, so the ~4 GB working set doesn't linger.
    private var summarizerActor: MLXQwenSummarizer?

    /// True while a `reviewSession()` call is in flight. Disables
    /// the toolbar button + footer affordance during the LLM pass.
    private(set) var transcriptionReviewRunning: Bool = false
    /// Last successful suggestion list, kept around so the review
    /// sheet survives re-presentation without re-running inference.
    /// Suggestions are removed from this list as the user accepts /
    /// rejects them. Cleared on session start.
    private(set) var transcriptionSuggestions: [TranscriptionSuggestion] = []
    /// The Qwen reviewer's actor — separate from `summarizerActor`
    /// because each owns its own `ModelContainer`. The controller
    /// is responsible for ensuring only one of the two is loaded
    /// at a time (loading both = ~9 GB resident, well over the
    /// per-app ceiling); the helpers below unload the sibling
    /// before invoking.
    private var reviewerActor: MLXQwenTranscriptionReviewer?

    /// Snapshot of the FluidAudio diarizer's speaker database
    /// taken right before the analysis pipeline is released for
    /// summarization. The blob rides through the pipeline rebuild
    /// and gets imported back into the freshly-warmed diarizer
    /// so embedding-based speaker matching survives a summarize
    /// pass. Nil outside the summarize window.
    private var summarizerSavedSpeakerDB: Data?
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
    /// Speaker IDs that appeared in `utterances` at the previous
    /// `commitUtteranceChanges` boundary. The auto-demote pass
    /// diffs this against the current state to find ids that
    /// fell out of the conversation entirely, then asks the
    /// diarizer to drop them from its DB. Reset alongside other
    /// per-session state.
    private var lastKnownSpeakerIDs: Set<String> = []

    /// Cached sorted list of distinct speaker ids in `utterances`,
    /// surfaced to the view layer via `knownSpeakerIDs()`. Without
    /// this, the speaker-chip reassign menu's input was rebuilt
    /// per visible row per body re-eval — `Array(Set(map)).sorted()`
    /// is O(N) per call, called ~30× per render in a 500-row
    /// session. Maintained in lockstep with `lastKnownSpeakerIDs`
    /// at every utterance-mutation boundary.
    private(set) var cachedKnownSpeakerIDs: [String] = []
    /// Cached count of distinct speaker ids in `utterances`, used
    /// to gate the per-row speaker chip rendering. Same motivation
    /// as `cachedKnownSpeakerIDs` — recomputing `Set` membership
    /// per body re-eval scaled with list length.
    private(set) var distinctSpeakerCountCache: Int = 0

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
    /// Latest snapshot of the cumulative diarizer timeline.
    /// Refreshed by the continuous-diarize task after each ingest;
    /// reset at session start / loadSession. Drives the per-session
    /// timeline strip in the transcript pane. Not persisted with
    /// `makeSessionDocument` — after Open Session this stays empty
    /// until re-recording or hand-edit/reeval flows repopulate it.
    private(set) var diarizationTimeline: [DiarizedSegment] = []

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
    private var streamingTranscriber: any StreamingTranscriber
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

    /// True when the streaming transcriber wasn't externally injected
    /// and we're free to swap it on a language change. Tests inject a
    /// stub via the init param; for them this flag stays false so
    /// `setSessionLanguage` leaves their stub in place rather than
    /// silently replacing it with a real Apple transcriber.
    private let canRecreateStreamingTranscriber: Bool

    private static let summarizerEnabledKey = "xephon.summarizerEnabled"
    private static let summarizerBackendKey = "xephon.summarizerBackend"

    init(
        capture: any AudioCapture = AVAudioEngineCapture(),
        streamingTranscriber: (any StreamingTranscriber)? = nil,
        pipeline: AnalysisPipeline? = nil,
        modelStore: ModelStore? = nil
    ) {
        self.capture = capture
        self.micCapture = capture
        let initialLanguage = SessionLanguage.loadFromDefaults()
        self.sessionLanguage = initialLanguage
        self.summarizerEnabled = UserDefaults.standard.bool(forKey: Self.summarizerEnabledKey)
        let rawBackend = UserDefaults.standard.string(forKey: Self.summarizerBackendKey) ?? ""
        self.summarizerBackend = SummarizerBackend(rawValue: rawBackend) ?? .appleFM
        self.summarizerAppleFMAvailable = SystemLanguageModel.default.isAvailable
        self.canRecreateStreamingTranscriber = (streamingTranscriber == nil)
        self.streamingTranscriber = streamingTranscriber
            ?? StreamingSpeechAnalyzerTranscriber(locale: initialLanguage.locale)
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
        // Apply the persisted session language to the freshly-warmed
        // pipeline before reading its text-SER state. `autoConfigured`
        // builds its offline transcriber against the default Japanese
        // locale and treats DeBERTa as available; if the user picked
        // English on a previous launch we need to retarget here so
        // the offline transcriber matches and `availableBackends`
        // excludes DeBERTa.
        await pipeline.setLocale(
            sessionLanguage.locale,
            languageLabel: sessionLanguage.label
        )
        availableTextSERBackends = await pipeline.availableTextSERBackends()
        currentTextSERBackend = await pipeline.currentTextSERBackend()
        // Also opportunistically refresh the summarizer install
        // state — `modelStore` is guaranteed live by the time this
        // runs, and we don't want the Settings card to render
        // "Not installed" on first launch when the files are
        // actually present from a previous session.
        refreshSummarizerInstallState()
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
        // Bump first so views observing the token can drop their
        // stale UUID-keyed state before they see the empty
        // utterance list — `@Observable` change notifications fire
        // synchronously and ContentView's reset modifier reads
        // utterances during cleanup.
        sessionToken = UUID()
        utterances = []
        preReevaluationSnapshots.removeAll()
        handEditChildren.removeAll()
        speakerNameOverrides.removeAll()
        conversationSummary.reset()
        capturedAudio.reset()
        diarizationTimeline = []
        lastKnownSpeakerIDs = []
        cachedKnownSpeakerIDs = []
        distinctSpeakerCountCache = 0
        // New session invalidates the previous summary — the
        // utterance list it was generated against is gone.
        lastSessionSummary = nil
        // Same goes for any pending transcription suggestions: they
        // pointed at utterance IDs that no longer exist in this
        // fresh session.
        transcriptionSuggestions = []
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
                self.diarizationTimeline = await pipeline.diarizationTimelineSnapshot()
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

    // MARK: - Session summarizer

    /// Convenience for the toolbar / UI: true iff the chosen
    /// backend is ready to summarize. Apple FM is "ready" when
    /// the system model is available on this device; Qwen is
    /// "ready" when its 4.3 GB on-disk install is complete.
    var summarizerReady: Bool {
        switch summarizerBackend {
        case .appleFM: return summarizerAppleFMAvailable
        case .qwen:    return summarizerModelInstalled
        }
    }

    /// Flip the summarizer's enabled flag. When turning on:
    /// persist the choice, refresh backend-specific readiness,
    /// and (only for the Qwen backend) kick off the on-demand
    /// model download if the weights aren't on disk yet. When
    /// turning off: persist the choice and unload the resident
    /// Qwen weights (Apple FM is system-managed; nothing to
    /// unload there). All paths are idle-safe — flipping the
    /// toggle during recording / analysis is permitted because
    /// the download / unload doesn't touch the active capture
    /// pipeline.
    func setSummarizerEnabled(_ enabled: Bool) async {
        guard summarizerEnabled != enabled else { return }
        summarizerEnabled = enabled
        UserDefaults.standard.set(enabled, forKey: Self.summarizerEnabledKey)
        AppLog.app.info(
            "summarizer enabled → \(enabled, privacy: .public)"
        )
        if !enabled {
            await summarizerActor?.unload()
            summarizerActor = nil
            return
        }
        refreshAppleFMAvailability()
        refreshSummarizerInstallState()
        if summarizerBackend == .qwen,
           !summarizerModelInstalled,
           !summarizerDownloading {
            await triggerSummarizerDownload()
        }
    }

    /// Switch the summarizer backend. Apple FM has no
    /// install step (the system model is in-process via
    /// FoundationModelsSER); selecting Qwen kicks off the
    /// download if the weights aren't on disk yet.
    func setSummarizerBackend(_ backend: SummarizerBackend) async {
        guard summarizerBackend != backend else { return }
        summarizerBackend = backend
        UserDefaults.standard.set(backend.rawValue, forKey: Self.summarizerBackendKey)
        AppLog.app.info(
            "summarizer backend → \(backend.rawValue, privacy: .public)"
        )
        if backend == .qwen,
           summarizerEnabled,
           !summarizerModelInstalled,
           !summarizerDownloading {
            await triggerSummarizerDownload()
        }
        if backend == .appleFM {
            // Reclaim Qwen's RAM if it was loaded.
            await summarizerActor?.unload()
            summarizerActor = nil
        }
        refreshAppleFMAvailability()
    }

    private func refreshAppleFMAvailability() {
        summarizerAppleFMAvailable = SystemLanguageModel.default.isAvailable
    }

    /// Drive the on-demand download via `ModelStore.ensureOptional`.
    /// Wraps the call in `summarizerDownloading` so the Settings
    /// card can render an inline progress indicator. Errors surface
    /// on `errorMessage` and the toggle stays on (so the user can
    /// see they tried) — re-enabling retries.
    private func triggerSummarizerDownload() async {
        guard let modelStore else { return }
        summarizerDownloading = true
        defer { summarizerDownloading = false }
        do {
            try await modelStore.ensureOptional(id: ModelManifest.summarizerID)
            refreshSummarizerInstallState()
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error(
                "Summarizer download failed: \(String(describing: error), privacy: .public)"
            )
        }
    }

    /// Refresh `summarizerModelInstalled` from the filesystem.
    /// Cheap — just an existence check per declared file. Called
    /// after enable / download / remove and on session reset, so
    /// the UI reflects whatever's actually on disk.
    private func refreshSummarizerInstallState() {
        guard let modelStore else {
            summarizerModelInstalled = false
            return
        }
        Task {
            let installed = await modelStore.isOptionalInstalled(
                id: ModelManifest.summarizerID
            )
            await MainActor.run {
                self.summarizerModelInstalled = installed
            }
        }
    }

    /// Run the summarizer over the current session. Returns nil
    /// when the model isn't installed, the session is empty, or
    /// inference is already in flight. Side-effect: stamps
    /// `lastSessionSummary` so the result sheet can re-present
    /// without re-running.
    ///
    /// Memory orchestration: the 4.3 GB Qwen weights plus the
    /// AnalysisPipeline's already-loaded inference actors (W2V2 ~
    /// 600 MB, emotion2vec ~330 MB, DeBERTa-WRIME ~250 MB, plus
    /// the FluidAudio diarizer's Core ML models) consistently
    /// trip iOS Jetsam on a 16 GB iPad once Qwen prefill kicks
    /// off. We release the analysis pipeline before engaging the
    /// summarizer, then unload Qwen and re-warm the pipeline in
    /// the background after the user has their result. The
    /// in-memory pipeline state we lose this way (speaker tracker
    /// cumulative history, FluidAudio SpeakerManager DB) is
    /// per-session-recording state — for the typical "record →
    /// summarize" flow there's no live recording to lose state
    /// in, and the speaker DB is persisted into the `.xph`
    /// anyway. Re-loading on next use takes ~2–5 s.
    func summarizeSession() async -> SessionSummary? {
        guard !summarizerInferenceRunning else { return nil }
        guard !utterances.isEmpty else { return nil }
        // Both backends benefit from releasing the analysis
        // pipeline before invoking — even Apple FM, which is
        // light on RAM in our process, can trip Jetsam when the
        // device is already under pressure (2-3 GB of resident
        // ONNX models + a fat speaker DB + a long utterance list
        // before we even allocate anything for the summary). The
        // pipeline lazy-rewarms in the deferred cleanup.
        logAvailableMemory(label: "summarize start (before pipeline release)")
        await releasePipelineForSummarization()
        logAvailableMemory(label: "summarize start (after pipeline release)")
        switch summarizerBackend {
        case .appleFM:
            return await summarizeWithAppleFM()
        case .qwen:
            return await summarizeWithQwen()
        }
    }

    /// Log how much memory the process can still allocate before
    /// iOS Jetsam will start culling. `os_proc_available_memory()`
    /// is the canonical sentinel — when it drops near zero, a
    /// SIGKILL is imminent. Surfaced around the summarize call so
    /// we can see exactly how much headroom we have at each stage
    /// when a crash report goes missing.
    private func logAvailableMemory(label: String) {
        let bytes = os_proc_available_memory()
        let mb = bytes / (1024 * 1024)
        AppLog.app.info(
            "memory available [\(label, privacy: .public)]: \(mb, privacy: .public) MB"
        )
    }

    /// Apple FM path. The system model itself is system-managed,
    /// but the path goes through a `LanguageModelSession` which
    /// allocates per-call buffers in our process; combined with
    /// the still-resident analysis pipeline this can OOM even
    /// though FM is "lightweight on paper." We rely on
    /// `summarizeSession` having already released the pipeline.
    private func summarizeWithAppleFM() async -> SessionSummary? {
        guard SystemLanguageModel.default.isAvailable else {
            errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMSummarizer()
        summarizerInferenceRunning = true
        defer {
            summarizerInferenceRunning = false
            scheduleSummarizerUnloadAndPipelineRewarm()
        }
        logAvailableMemory(label: "summarize Apple FM (before respond)")
        do {
            let summary = try await backend.summarize(
                utterances: utterances,
                speakerNames: speakerNameOverrides
            )
            logAvailableMemory(label: "summarize Apple FM (after respond)")
            lastSessionSummary = summary
            return summary
        } catch is CancellationError {
            // User dismissed the sheet mid-generation. Silent —
            // this isn't an error condition from their POV.
            AppLog.app.info("summarizeWithAppleFM cancelled by user")
            return nil
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithAppleFM failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Qwen path — releases the analysis pipeline before loading
    /// the 4.3 GB weights to fit under iOS's per-app memory
    /// ceiling, runs inference, then unloads Qwen and re-warms
    /// the pipeline (with the speaker DB restored). See the
    /// helpers below for the orchestration detail.
    private func summarizeWithQwen() async -> SessionSummary? {
        guard let modelStore else {
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(
            id: ModelManifest.summarizerID
        ) else {
            errorMessage = String(describing: SummarizerError.modelNotInstalled)
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }

        // Pipeline release happens up in `summarizeSession`
        // before this dispatch, so we don't repeat it here.

        let actor: MLXQwenSummarizer
        if let existing = summarizerActor {
            actor = existing
        } else {
            actor = MLXQwenSummarizer(
                modelIdentifier: ModelManifest.summarizerID,
                modelDirectory: directory
            )
            summarizerActor = actor
        }

        summarizerInferenceRunning = true
        defer {
            summarizerInferenceRunning = false
            scheduleSummarizerUnloadAndPipelineRewarm()
        }
        do {
            let summary = try await actor.summarize(
                utterances: utterances,
                speakerNames: speakerNameOverrides
            )
            lastSessionSummary = summary
            return summary
        } catch is CancellationError {
            AppLog.app.info("summarizeWithQwen cancelled by user")
            return nil
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error(
                "summarizeWithQwen failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Drop strong references to the analysis pipeline so ARC can
    /// reclaim the ~1.5 GB of resident ONNX session memory before
    /// Qwen claims its 4.3 GB. Yields cooperatively after nilling
    /// so the runtime gets a tick to release before MLX starts
    /// allocating prefill memory.
    ///
    /// Before nilling, we snapshot the FluidAudio diarizer's
    /// speaker database so embedding-based matching survives the
    /// rebuild — without this, a re-evaluation after summarize
    /// would no longer recognize the existing rows' voices and
    /// could reassign them to fresh IDs. The blob is restored
    /// in `scheduleSummarizerUnloadAndPipelineRewarm`.
    private func releasePipelineForSummarization() async {
        AppLog.app.info(
            "releasing analysis pipeline before summarization (free ~1.5 GB)"
        )
        if let pipeline {
            summarizerSavedSpeakerDB = await pipeline.exportSpeakerDatabase()
            if let blob = summarizerSavedSpeakerDB {
                AppLog.app.info(
                    "snapshotted speaker DB before summarize (\(blob.count, privacy: .public) bytes)"
                )
            }
        }
        pipelineTask?.cancel()
        pipelineTask = nil
        pipeline = nil
        await Task.yield()
    }

    /// Unload Qwen and re-warm the analysis pipeline in the
    /// background. Runs in `defer` so it fires whether
    /// summarization succeeded or failed. The user is now reading
    /// their summary (or seeing an error), so the few seconds of
    /// background re-load aren't user-visible. Restores the
    /// pre-summarize speaker DB snapshot after the fresh diarizer
    /// is warm, so post-summarize re-evaluations / hand-edits
    /// still match the established speaker IDs.
    private func scheduleSummarizerUnloadAndPipelineRewarm() {
        Task { @MainActor [weak self] in
            guard let self else { return }
            // Unload BOTH Qwen actors. Only one is loaded at a time
            // by construction, but the cleanup path is shared between
            // summarize and review flows, and a stale reference here
            // would keep ~4.6 GB of weights resident across the
            // pipeline rewarm.
            await self.summarizerActor?.unload()
            self.summarizerActor = nil
            await self.reviewerActor?.unload()
            self.reviewerActor = nil
            AppLog.app.info("Qwen unloaded; re-warming analysis pipeline")
            let pipeline = await self.ensurePipeline()
            if let saved = self.summarizerSavedSpeakerDB {
                do {
                    try await pipeline.importSpeakerDatabase(saved)
                    AppLog.app.info(
                        "restored speaker DB after pipeline re-warm"
                    )
                } catch {
                    AppLog.app.warning(
                        "speaker DB restore after summarize failed: \(String(describing: error), privacy: .public)"
                    )
                }
                self.summarizerSavedSpeakerDB = nil
            }
        }
    }

    /// Remove the summarizer model from disk. Called by the
    /// "Remove model" action in Settings — the toggle stays on or
    /// off according to the user's preference, but the ~4 GB
    /// weights blob goes away.
    func removeSummarizerModel() async {
        await summarizerActor?.unload()
        summarizerActor = nil
        // The reviewer shares the same weights — drop its handle too
        // so the freshly-deleted model can't be lazily reloaded by a
        // stale actor reference.
        await reviewerActor?.unload()
        reviewerActor = nil
        do {
            try await modelStore?.removeOptional(id: ModelManifest.summarizerID)
        } catch {
            AppLog.app.warning(
                "removeOptional failed: \(String(describing: error), privacy: .public)"
            )
        }
        refreshSummarizerInstallState()
    }

    // MARK: - Transcription review

    /// Walk the current utterance list through the on-device LLM and
    /// collect transcription suggestions. Same orchestration as
    /// `summarizeSession`: release the analysis pipeline, run the
    /// reviewer, unload + rewarm. Suggestions are cached in
    /// `transcriptionSuggestions` so the review sheet can re-present
    /// without re-running inference; the caller is expected to
    /// guard repeated entries via that field.
    func reviewSession() async -> [TranscriptionSuggestion]? {
        guard !transcriptionReviewRunning else { return nil }
        guard !summarizerInferenceRunning else { return nil }
        guard !utterances.isEmpty else { return nil }

        logAvailableMemory(label: "review start (before pipeline release)")
        await releasePipelineForSummarization()
        logAvailableMemory(label: "review start (after pipeline release)")
        switch summarizerBackend {
        case .appleFM:
            return await reviewWithAppleFM()
        case .qwen:
            return await reviewWithQwen()
        }
    }

    private func reviewWithAppleFM() async -> [TranscriptionSuggestion]? {
        guard SystemLanguageModel.default.isAvailable else {
            errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }
        let backend = AppleFMTranscriptionReviewer()
        transcriptionReviewRunning = true
        defer {
            transcriptionReviewRunning = false
            scheduleSummarizerUnloadAndPipelineRewarm()
        }
        logAvailableMemory(label: "review Apple FM (before respond)")
        do {
            let suggestions = try await backend.review(
                utterances: utterances,
                speakerNames: speakerNameOverrides,
                language: reviewLanguage()
            )
            logAvailableMemory(label: "review Apple FM (after respond)")
            transcriptionSuggestions = suggestions
            return suggestions
        } catch is CancellationError {
            AppLog.app.info("reviewWithAppleFM cancelled by user")
            return nil
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithAppleFM failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    private func reviewWithQwen() async -> [TranscriptionSuggestion]? {
        guard let modelStore else {
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }
        guard let directory = await modelStore.optionalDirectory(
            id: ModelManifest.summarizerID
        ) else {
            errorMessage = String(describing: TranscriptionReviewError.modelNotInstalled)
            scheduleSummarizerUnloadAndPipelineRewarm()
            return nil
        }

        // Belt-and-braces: drop the summarizer actor before bringing
        // up the reviewer. Both share Qwen3-8B's 4.6 GB weights;
        // holding both = ~9 GB resident and a guaranteed Jetsam.
        await summarizerActor?.unload()
        summarizerActor = nil

        let actor: MLXQwenTranscriptionReviewer
        if let existing = reviewerActor {
            actor = existing
        } else {
            actor = MLXQwenTranscriptionReviewer(
                modelIdentifier: ModelManifest.summarizerID,
                modelDirectory: directory
            )
            reviewerActor = actor
        }

        transcriptionReviewRunning = true
        defer {
            transcriptionReviewRunning = false
            scheduleSummarizerUnloadAndPipelineRewarm()
        }
        do {
            let suggestions = try await actor.review(
                utterances: utterances,
                speakerNames: speakerNameOverrides,
                language: reviewLanguage()
            )
            transcriptionSuggestions = suggestions
            return suggestions
        } catch is CancellationError {
            AppLog.app.info("reviewWithQwen cancelled by user")
            return nil
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error(
                "reviewWithQwen failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
    }

    /// Map the controller's `SessionLanguage` onto the Summarizer
    /// module's `ReviewLanguage`. Lives here (not on
    /// `SessionLanguage`) so the latter type doesn't grow a
    /// dependency on Summarizer just for one bridge.
    private func reviewLanguage() -> ReviewLanguage {
        switch sessionLanguage {
        case .japanese: return .japanese
        case .english:  return .english
        }
    }

    /// Drop a suggestion without applying it. Called from the
    /// review sheet's per-row reject button.
    func dismissTranscriptionSuggestion(id: UUID) {
        transcriptionSuggestions.removeAll { $0.id == id }
    }

    /// Drop every cached suggestion. Used when the user explicitly
    /// clears the review state or when they re-run a review (the
    /// fresh pass replaces the list wholesale, but the regenerate
    /// path benefits from clearing first so the empty state is
    /// distinguishable from a no-op result).
    func clearTranscriptionSuggestions() {
        transcriptionSuggestions = []
    }

    /// Accept a suggestion: route through the existing hand-edit
    /// path so SER + fusion are re-run on the corrected transcript,
    /// then remove the suggestion from the list. No-op if the
    /// suggestion's text is empty (no concrete fix was proposed)
    /// or if the row no longer exists or its transcript has
    /// drifted from the snapshot the reviewer saw — better to keep
    /// the row untouched than to blindly overwrite a manual edit
    /// the user already made.
    func acceptTranscriptionSuggestion(id: UUID) async {
        guard let suggestion = transcriptionSuggestions.first(where: { $0.id == id }) else { return }
        guard !suggestion.suggestedText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        guard let utterance = utterances.first(where: { $0.id == suggestion.utteranceID }) else {
            transcriptionSuggestions.removeAll { $0.id == id }
            return
        }
        guard utterance.transcript == suggestion.originalText else {
            // Stale: the row was edited between review and accept.
            // Drop the suggestion so the user re-runs review with
            // the current state.
            transcriptionSuggestions.removeAll { $0.id == id }
            return
        }
        await commitHandEdit(
            utteranceID: utterance.id,
            newText: suggestion.suggestedText,
            newStart: utterance.start,
            newEnd: utterance.end
        )
        transcriptionSuggestions.removeAll { $0.id == id }
    }

    /// Switch the session language. Re-creates the streaming
    /// transcriber against the new locale (unless the streaming
    /// transcriber was externally injected by tests), forwards the
    /// new locale + language label to the pipeline so the offline
    /// transcriber and FoundationModels prompt opener follow along,
    /// and refreshes the text-SER state so the picker re-renders
    /// without DeBERTa for non-Japanese sessions. Persists the
    /// choice to UserDefaults for the next launch. No-op when not
    /// idle — language can't change mid-session because the
    /// already-running transcriber is locked to its initial locale.
    func setSessionLanguage(_ language: SessionLanguage) async {
        guard phase == .idle else { return }
        guard sessionLanguage != language else { return }
        sessionLanguage = language
        language.saveToDefaults()
        if canRecreateStreamingTranscriber {
            streamingTranscriber = StreamingSpeechAnalyzerTranscriber(
                locale: language.locale
            )
        }
        let pipeline = await ensurePipeline()
        await pipeline.setLocale(language.locale, languageLabel: language.label)
        await refreshTextSERState(from: pipeline)
        AppLog.app.info(
            "session language → \(language.rawValue, privacy: .public)"
        )
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
        // Serialize the diarizer timeline so the per-session
        // visualization strip in the transcript pane survives
        // Save → Open. JSON-encoded so the Export layer doesn't
        // need to import the Diarization module's segment type.
        // Nil when no diarization ran (mic-mode pre-roll on a save
        // with no audio, or an analysis path that never engaged
        // the diarizer).
        let timelineBlob: Data? = {
            guard !diarizationTimeline.isEmpty else { return nil }
            return try? JSONEncoder().encode(diarizationTimeline)
        }()
        // Persist the last LLM summary alongside the utterances so
        // reopening a `.xph` shows the cached result without
        // re-running the multi-second LLM pass. JSON-encoded so the
        // Export layer doesn't need to depend on Summarizer (which
        // would drag MLX into a module that has no business with
        // it). Nil when no summary has been produced this session.
        let summaryBlob: Data? = {
            guard let summary = lastSessionSummary else { return nil }
            return try? JSONEncoder().encode(summary)
        }()
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
                    handEditChildren: children,
                    diarizationTimeline: timelineBlob,
                    sessionSummary: summaryBlob
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
            handEditChildren: children,
            diarizationTimeline: timelineBlob,
            sessionSummary: summaryBlob
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
        // Bump the session token so ContentView's `@State` keyed
        // by UUID (`visibleUtteranceIDs`, `expandedUtteranceIDs`,
        // `selectedUtteranceID`, `scrollRequestUtteranceID`,
        // `normalizedTranscriptCache`) is dropped before the new
        // utterance list takes over. Without this, stale view
        // state from the previous file leaks into the loaded one
        // and timeline strips (which mix `recorder.utterances`
        // with the view-side visibility set) can render wrong.
        sessionToken = UUID()
        utterances = document.utterances
        // Restore the pre-edit revert state from the bundle so a
        // long-press on a row's Edited / completed marker after
        // Open Session still rolls the row back to its original
        // streaming-pass record. `removeAll` first to drop any
        // leftover state from a prior session that wasn't cleared
        // (no recording started yet).
        preReevaluationSnapshots.removeAll()
        handEditChildren.removeAll()
        diarizationTimeline = []
        lastKnownSpeakerIDs = []
        // The cached summary was generated against the previous
        // session's utterances. Default to clearing it; restore
        // below if the loaded `.xph` carries a persisted one.
        lastSessionSummary = nil
        if let blob = document.sessionSummary,
           let restored = try? JSONDecoder().decode(SessionSummary.self, from: blob) {
            lastSessionSummary = restored
        }
        // Pending suggestions don't persist — their utterance IDs
        // belonged to the previous in-memory session and would
        // alias onto unrelated rows here. Re-run review if needed.
        transcriptionSuggestions = []
        if let saved = document.originalSnapshots {
            preReevaluationSnapshots = saved
        }
        if let savedChildren = document.handEditChildren {
            handEditChildren = savedChildren
        }
        if let timelineBlob = document.diarizationTimeline,
           let restored = try? JSONDecoder().decode([DiarizedSegment].self, from: timelineBlob) {
            diarizationTimeline = restored
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
    ///
    /// Also runs the auto-demote pass: any speaker id that was
    /// present in `lastKnownSpeakerIDs` but no longer appears in
    /// any utterance has been fully reassigned away and gets
    /// cleaned up — drop the speaker name override, scrub
    /// cumulative-timeline observations with that id, and remove
    /// the entry from the diarizer DB (keeping permanent
    /// user-promoted entries). Gated on `phase == .idle` so
    /// mid-streaming flux doesn't trigger premature deletion of
    /// a speaker the streaming pass is still about to use.
    private func commitUtteranceChanges() {
        utterancesVersion &+= 1
        conversationSummary.reset()
        for u in utterances { conversationSummary.update(with: u) }
        let current = Set(utterances.map(\.speakerID))
        let removed = lastKnownSpeakerIDs.subtracting(current)
        if !removed.isEmpty, phase == .idle {
            sweepUnreferencedSpeakers(removed)
        }
        lastKnownSpeakerIDs = current
        cachedKnownSpeakerIDs = current.sorted()
        distinctSpeakerCountCache = current.count
    }

    /// Drop every trace of speakers that no row references
    /// anymore. Synchronous parts run inline (overrides, timeline
    /// observations); the diarizer-DB removal is fired off in a
    /// detached MainActor task so it doesn't block the commit
    /// pass. Failure of the async removal is non-fatal — the
    /// embedding stays in the DB but nothing else in the app
    /// references the id, so it's only memory waste.
    private func sweepUnreferencedSpeakers(_ ids: Set<String>) {
        for id in ids {
            speakerNameOverrides.removeValue(forKey: id)
            diarizationTimeline.removeAll { $0.speakerID == id }
        }
        AppLog.app.info(
            "auto-demoting unreferenced speakers: \(Array(ids).sorted().joined(separator: ", "), privacy: .public)"
        )
        Task { @MainActor [weak self] in
            guard let self else { return }
            let pipeline = await self.ensurePipeline()
            for id in ids {
                try? await pipeline.removeSpeakerFromDB(id: id, keepIfPermanent: true)
            }
        }
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
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return }
        let trimmedText = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedText.isEmpty else { return }
        let sentences = Self.splitTranscriptIntoSentences(trimmedText)
        guard !sentences.isEmpty else { return }

        let original = utterances[index]
        // Two flows split on whether the session has source audio.
        // File-mode (`playbackSourceURL != nil`) follows the full
        // pipeline: re-read the audio, re-diarize, re-run SER +
        // fusion. Mic-mode (live or imported) has no audio to
        // re-slice, so we inherit the parent's speaker, time
        // range, dimensional + acoustic SER, and only re-run text
        // SER + fusion. The Edit Utterance dialog hides time
        // controls + play button on the mic-mode path; the
        // controller treats `newStart`/`newEnd` as advisory and
        // overrides them with `original.start`/`original.end`.
        if playbackSourceURL == nil {
            reevaluatingUtteranceID = utteranceID
            defer { reevaluatingUtteranceID = nil }
            if preReevaluationSnapshots[utteranceID] == nil {
                preReevaluationSnapshots[utteranceID] = original
            }
            let plans = Self.planHandEditSentences(
                sentences,
                newStart: original.start,
                newEnd: original.end
            )
            guard !plans.isEmpty else { return }
            let pipeline = await ensurePipeline()
            await runTextOnlyHandEdit(
                utteranceID: utteranceID,
                plans: plans,
                original: original,
                pipeline: pipeline
            )
            return
        }

        guard let url = playbackSourceURL else { return }
        guard newEnd > newStart else { return }
        let plans = Self.planHandEditSentences(
            sentences,
            newStart: newStart,
            newEnd: newEnd
        )
        guard !plans.isEmpty else { return }

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

    /// Mic-mode (no source audio) hand-edit dispatcher. Inherits
    /// the parent's dimensional / acoustic-categorical / speech-
    /// boost; the speaker is *re-voted* against the cumulative
    /// diarizer timeline for each plan's sub-range, so a commit
    /// can shift the row's speaker assignment when the timeline
    /// has accumulated more observations since the row was
    /// originally finalized. Falls back to the parent's
    /// `speakerID` when the timeline has no overlap with the
    /// plan's window. Routes to `applyHandEdit` for one sentence
    /// and `applyHandEditSplit` for many — proportional
    /// per-character time allocation keeps the overall duration
    /// `[original.start, original.end]` consistent across the
    /// resulting rows.
    private func runTextOnlyHandEdit(
        utteranceID: UUID,
        plans: [HandEditPlan],
        original: UtteranceEstimate,
        pipeline: AnalysisPipeline
    ) async {
        if plans.count == 1 {
            do {
                let speaker = revoteSpeakerFromTimeline(
                    plan: plans[0],
                    fallback: original.speakerID
                )
                let parent = inheritedTimeStub(
                    from: original,
                    plan: plans[0],
                    speakerID: speaker
                )
                let fresh = try await pipeline.reanalyzeTextOnly(
                    text: plans[0].text,
                    inheriting: parent
                )
                applyHandEdit(utteranceID: utteranceID, fresh: fresh)
            } catch {
                AppLog.app.error(
                    "commitHandEdit (text-only) failed: \(String(describing: error), privacy: .public)"
                )
            }
            return
        }
        var freshEstimates: [UtteranceEstimate] = []
        freshEstimates.reserveCapacity(plans.count)
        for (i, plan) in plans.enumerated() {
            do {
                let speaker = revoteSpeakerFromTimeline(
                    plan: plan,
                    fallback: original.speakerID
                )
                let parent = inheritedTimeStub(
                    from: original,
                    plan: plan,
                    speakerID: speaker
                )
                let fresh = try await pipeline.reanalyzeTextOnly(
                    text: plan.text,
                    inheriting: parent
                )
                freshEstimates.append(fresh)
            } catch {
                AppLog.app.error(
                    "commitHandEdit (text-only) split[\(i, privacy: .public)] failed: \(String(describing: error), privacy: .public)"
                )
            }
        }
        guard !freshEstimates.isEmpty else { return }
        applyHandEditSplit(utteranceID: utteranceID, sentences: freshEstimates)
    }

    /// Per-instant majority over the cumulative diarizer timeline
    /// for `plan.[start, end]`. Used by the mic-mode (no-audio)
    /// hand-edit path to re-attribute a row's speaker without
    /// running the diarizer model — the timeline still holds the
    /// streaming-pass observations (and is persisted across Save),
    /// so re-tallying with the now-fuller record can move the
    /// vote even though no fresh audio is processed.
    private func revoteSpeakerFromTimeline(
        plan: HandEditPlan,
        fallback: String
    ) -> String {
        AnalysisPipeline.dominantSpeakerInSegments(
            diarizationTimeline,
            from: plan.start,
            to: plan.end,
            fallback: fallback
        )
    }

    /// Build a synthetic "parent" utterance that carries the
    /// inherited dimensional / acoustic-categorical / speech-
    /// boost fields, the plan's time range, and the supplied
    /// `speakerID` (either the inherited parent's or a fresh
    /// timeline re-vote). Fed to `reanalyzeTextOnly` so the
    /// fusion step sees the right inputs for this slice.
    private func inheritedTimeStub(
        from original: UtteranceEstimate,
        plan: HandEditPlan,
        speakerID: String
    ) -> UtteranceEstimate {
        UtteranceEstimate(
            id: original.id,
            speakerID: speakerID,
            start: plan.start,
            end: plan.end,
            transcript: plan.text,
            asrConfidence: 1.0,
            dimensional: original.dimensional,
            acousticCategorical: original.acousticCategorical,
            plutchik: nil,
            textBackend: nil,
            speechBoost: original.speechBoost,
            wasReevaluated: nil,
            wasHandEdited: nil,
            fusedValence: nil,
            fusedArousal: nil,
            fusedDominance: nil,
            fusedTopLabel: nil
        )
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
    /// (with the active row visually marked, not removed). Backed
    /// by a cache maintained at every utterance-mutation boundary
    /// so this is O(1) per call — it's invoked from
    /// `TranscriptList.row(for:)` per visible row per render.
    func knownSpeakerIDs() -> [String] {
        cachedKnownSpeakerIDs
    }

    /// Corrective reassignment: extract the row's audio embedding
    /// and fold it into `targetSpeakerID`'s centroid in the
    /// diarizer DB via EMA, reassign the row, and rewrite the
    /// timeline range so the per-instant majority for the window
    /// reflects the corrected speaker. Future re-eval / hand-edit
    /// on similar audio will be more likely to match the target,
    /// not just for this row but anywhere in the session.
    ///
    /// Distinct from the pure-annotation `reassignSpeaker(...)`,
    /// which only swaps the row's label and leaves the diarizer
    /// state untouched.
    ///
    /// No-op when no source audio, the row doesn't exist, the
    /// audio slice is empty, embedding extraction is unavailable,
    /// the target id is the same as the current id, or the
    /// underlying diarizer call throws. Returns true on success.
    @discardableResult
    func correctUtteranceSpeaker(
        utteranceID: UUID,
        to targetSpeakerID: String
    ) async -> Bool {
        let trimmed = targetSpeakerID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard let url = playbackSourceURL else { return false }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return false }
        let utt = utterances[index]
        guard utt.speakerID != trimmed else { return false }
        let chunk: AudioChunk
        do {
            chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: utt.start,
                    end: utt.end
                )
            }.value
        } catch {
            AppLog.app.error(
                "correct: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("correct: empty audio slice")
            return false
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("correct: embedding extractor unavailable")
            return false
        }
        let duration = Float(max(0, utt.end - utt.start))
        do {
            try await pipeline.correctSpeaker(
                id: trimmed,
                embedding: embedding,
                duration: duration
            )
        } catch {
            AppLog.app.error(
                "correct: DB update failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        utterances[index] = utt.withSpeakerID(trimmed)
        rewriteTimelineRange(speakerID: trimmed, start: utt.start, end: utt.end)
        AppLog.app.info(
            "speaker corrected: utt=\(utt.id, privacy: .public) \(utt.speakerID, privacy: .public) → \(trimmed, privacy: .public) (taught diarizer)"
        )
        commitUtteranceChanges()
        return true
    }

    /// Extract an embedding from `utteranceID`'s audio, register it
    /// in the diarizer's SpeakerManager DB under a freshly-minted
    /// `S0N` id, reassign the row to that id, and re-write the
    /// cumulative timeline so the new id wins the majority vote
    /// for the utterance's window. Future re-eval / hand-edit
    /// passes on similar audio will match this entry.
    ///
    /// No-op when: there's no source audio (mic mode without a
    /// playback URL), no row matches `utteranceID`, the audio
    /// slice came back empty, the diarizer's embedding extractor
    /// isn't available, or promotion throws. Returns the new id
    /// on success, nil on any failure path — caller can surface
    /// a toast on nil.
    func promoteUtteranceToNewSpeaker(utteranceID: UUID) async -> String? {
        guard let url = playbackSourceURL else { return nil }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return nil }
        let utt = utterances[index]
        let chunk: AudioChunk
        do {
            chunk = try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url,
                    start: utt.start,
                    end: utt.end
                )
            }.value
        } catch {
            AppLog.app.error(
                "promote: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("promote: empty audio slice")
            return nil
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("promote: embedding extractor unavailable")
            return nil
        }
        let newID = Self.nextNewSpeakerID(in: knownSpeakerIDs())
        do {
            try await pipeline.promoteSpeaker(id: newID, embedding: embedding)
        } catch {
            AppLog.app.error(
                "promote: register failed: \(String(describing: error), privacy: .public)"
            )
            return nil
        }
        utterances[index] = utt.withSpeakerID(newID)
        rewriteTimelineRange(speakerID: newID, start: utt.start, end: utt.end)
        AppLog.app.info(
            "speaker promoted: utt=\(utt.id, privacy: .public) → \(newID, privacy: .public) [\(utt.start, privacy: .public)..\(utt.end, privacy: .public)]"
        )
        commitUtteranceChanges()
        return newID
    }

    /// Smallest unused `S0N` id given the supplied existing ids.
    /// Matches the chip menu's `nextNewSpeakerID` logic so both
    /// the "New Speaker" and "Promote New Speaker" paths agree
    /// on which slot to fill.
    private static func nextNewSpeakerID(in existing: [String]) -> String {
        var used: Set<Int> = []
        for id in existing {
            guard id.hasPrefix("S") else { continue }
            if let n = Int(id.dropFirst()) { used.insert(n) }
        }
        var candidate = 1
        while used.contains(candidate) { candidate += 1 }
        return String(format: "S%02d", candidate)
    }

    /// Replace every cumulative-timeline observation that covers
    /// any part of `[start, end]` so the result has exactly one
    /// segment for `speakerID` over that window. Overlapping
    /// segments get split at the boundary (the portions outside
    /// `[start, end]` survive intact, the portion inside is
    /// dropped). Used after `promoteUtteranceToNewSpeaker` so the
    /// per-instant majority for the promoted range cleanly
    /// reflects the new speaker — without this, the streaming
    /// pass's ~5 votes for the old id would still win the
    /// `dominantSpeakerInSegments` tally and the mismatch
    /// warning would persist.
    private func rewriteTimelineRange(
        speakerID: String,
        start: TimeInterval,
        end: TimeInterval
    ) {
        guard end > start else { return }
        var rewritten: [DiarizedSegment] = []
        rewritten.reserveCapacity(diarizationTimeline.count + 1)
        for seg in diarizationTimeline {
            if seg.end <= start || seg.start >= end {
                rewritten.append(seg)
            } else if seg.start < start && seg.end > end {
                // Segment surrounds the promoted range — split.
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: start))
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: end, end: seg.end))
            } else if seg.start < start {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: start))
            } else if seg.end > end {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: end, end: seg.end))
            }
            // Fully inside → drop.
        }
        rewritten.append(DiarizedSegment(speakerID: speakerID, start: start, end: end))
        rewritten.sort { $0.start < $1.start }
        diarizationTimeline = rewritten
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
        // Keep the speaker-id cache current for the streaming path
        // too (only `commitUtteranceChanges` is reached on edits;
        // streaming inserts bypass it). Cheap: insertion into a
        // small set + sort only when a new speaker appears.
        if !lastKnownSpeakerIDs.contains(stamped.speakerID) {
            lastKnownSpeakerIDs.insert(stamped.speakerID)
            cachedKnownSpeakerIDs = lastKnownSpeakerIDs.sorted()
            distinctSpeakerCountCache = lastKnownSpeakerIDs.count
        }
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
        // Stamp the active rename (`speakerNameOverrides[id]`) onto
        // each row before encoding, so the JSON carries the human
        // name alongside the canonical `speakerID`. We don't keep
        // the name on `utterances` itself because it lives one
        // layer above the fusion stage — the override map is the
        // source of truth, and a row's stored `speakerName` would
        // go stale the moment the user renamed the speaker without
        // re-stamping every row. Renames + ID reassignments are
        // both reflected this way: the id change rides in
        // `withSpeakerID` (already applied in place) and the
        // friendly name is filled in here.
        let stamped: [UtteranceEstimate]
        if speakerNameOverrides.isEmpty {
            stamped = utterances
        } else {
            stamped = utterances.map { u in
                u.withSpeakerName(speakerNameOverrides[u.speakerID])
            }
        }
        do {
            try await exporter.write(stamped, to: url)
            lastExportAt = Date()
            return url
        } catch {
            errorMessage = String(describing: error)
            return nil
        }
    }
}
