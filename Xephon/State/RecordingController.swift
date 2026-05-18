import Foundation
import Observation
import os
@preconcurrency import AVFoundation
import Audio
import ASR
import Diarization
import Fusion
import Export
import SERText
import Summarizer
import XephonLogging
import XephonUtilities

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
    var utterancesVersion: Int = 0
    private(set) var samplesCaptured: Int = 0
    /// Total audio duration (in source seconds) of the file currently
    /// being analyzed, or nil when not in file mode / before the file
    /// has been probed. Used by the status line to render a completion
    /// percentage alongside wall-time and sample count.
    var fileTotalAudioDuration: TimeInterval?
    /// Surface-level error from the controller or any of its
    /// coordinators. The Settings card binds an alert to this; nil
    /// = no pending message. Setter is internal (rather than
    /// `private(set)`) so sibling coordinators in this target can
    /// report their own errors here.
    var errorMessage: String?
    /// Rotates every time the session-level utterance list changes
    /// identity (new recording, new file analysis, imported `.xph`).
    /// Views observe this to flush transient `@State` keyed by the
    /// prior session's UUIDs — `visibleUtteranceIDs`,
    /// `expandedUtteranceIDs`, `selectedUtteranceID`, etc. The
    /// controller can't reach those bindings directly, so a token
    /// the view watches is the cleanest seam.
    var sessionToken: UUID = UUID()
    /// Internal-write so hand-edit / re-evaluation extensions can
    /// replace rows in place. Views still read-only via `recorder.utterances`.
    var utterances: [UtteranceEstimate] = []
    private(set) var inputLevel: Float = 0
    /// Per-source-channel perceptual level (0…1), one entry per
    /// channel the input device delivers. Drives the multi-bar level
    /// meter so a stereo USB mic surfaces L/R balance instead of a
    /// single mixed-down bar. Empty when no audio has been captured
    /// yet. `inputLevel` is kept around as the max-of-channels for
    /// the existing PipelineCard binding.
    private(set) var inputChannelLevels: [Float] = []
    private(set) var availableInputs: [AudioInputDescription] = []
    /// The OS's reported current-route input. Useful for diagnostics
    /// but NOT what the picker should display — iPadOS 26 flips
    /// `session.currentRoute.inputs.first` between built-in and USB-C
    /// mid-session even when the engine's input bound is stable, so
    /// binding the picker label to this would make it flicker.
    private(set) var currentInputUID: String?
    /// What the user explicitly picked in the input menu. Nil means
    /// no explicit choice yet; the recording path then falls back to
    /// the built-in mic, so the picker should also display built-in
    /// in that case. This is what the picker label binds to so the
    /// displayed input reflects the user's intent and stays stable
    /// while iOS shuffles routes underneath.
    private(set) var selectedInputUID: String?
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

    /// Owns the entire on-device summarization + transcription-review
    /// flow. The properties + methods below are thin forwarders so
    /// existing view-layer reads (`recorder.summarizerEnabled`,
    /// `recorder.lastSessionSummary`, `recorder.transcriptionIssues`,
    /// …) keep working without a rename pass. The coordinator's
    /// `@Observable` state propagates through these computed reads,
    /// so view tracking stays correct.
    private(set) var summarizer: SummarizerCoordinator!
    var summarizerEnabled: Bool { summarizer.enabled }
    var summarizerBackend: SummarizerBackend { summarizer.backend }
    var summarizerAppleFMAvailable: Bool { summarizer.appleFMAvailable }
    var summarizerModelInstalled: Bool { summarizer.modelInstalled }
    var summarizerDownloading: Bool { summarizer.downloading }
    var summarizerInferenceRunning: Bool { summarizer.inferenceRunning }
    var summarizerInferenceStart: Date? { summarizer.inferenceStart }
    var summarizerReady: Bool { summarizer.ready }
    var lastSessionSummary: SessionSummary? { summarizer.lastSessionSummary }
    var transcriptionReviewRunning: Bool { summarizer.reviewRunning }
    var transcriptionReviewStart: Date? { summarizer.reviewStart }
    var transcriptionIssues: [TranscriptionIssue] { summarizer.issues }

    func setSummarizerEnabled(_ enabled: Bool) async { await summarizer.setEnabled(enabled) }
    func setSummarizerBackend(_ backend: SummarizerBackend) async { await summarizer.setBackend(backend) }
    func summarizeSession() async -> SessionSummary? { await summarizer.summarize() }
    func reviewSession() async -> [TranscriptionIssue]? { await summarizer.review() }
    func removeSummarizerModel() async { await summarizer.removeModel() }
    func dismissTranscriptionIssue(id: UUID) { summarizer.dismissIssue(id: id) }
    var volatileText: String = ""
    var lastAcousticDuration: TimeInterval?
    var lastTextDuration: TimeInterval?
    var lastSegmentTotal: TimeInterval?
    /// Wall-clock delay from "speaker finished this utterance"
    /// (`sessionStartedAt + segment.end`) to "analyzer emitted the final
    /// for it" (Date() at the analysisTask receipt). Indicates how long
    /// SpeechAnalyzer's volatile-stabilization window held the segment
    /// before promoting it to a final — typical 200–800 ms on M-class.
    /// Nil until the first finalize lands.
    var lastASRFinalizeLatency: TimeInterval?
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
    var reevaluatingUtteranceID: UUID?
    /// Pre-first-reeval snapshots keyed by utterance id. Captured in
    /// `applyReevaluation` only on the FIRST re-evaluation of each
    /// row (so subsequent re-evals don't overwrite the truly-
    /// original streaming result). Cleared per row by
    /// `revertReevaluation`, and wholesale by `start()` / `loadSession`.
    /// Not persisted — revert history is session-scoped only.
    /// Internal-write (not `private`) so the hand-edit extension in
    /// `RecordingController+HandEdit.swift` can capture pre-edit
    /// snapshots alongside the re-evaluate flow.
    var preReevaluationSnapshots: [UUID: UtteranceEstimate] = [:]
    /// Speaker IDs that appeared in `utterances` at the previous
    /// `commitUtteranceChanges` boundary. The auto-demote pass
    /// diffs this against the current state to find ids that
    /// fell out of the conversation entirely, then asks the
    /// diarizer to drop them from its DB. Reset alongside other
    /// per-session state.
    var lastKnownSpeakerIDs: Set<String> = []

    /// Cached sorted list of distinct speaker ids in `utterances`,
    /// surfaced to the view layer via `knownSpeakerIDs()`. Without
    /// this, the speaker-chip reassign menu's input was rebuilt
    /// per visible row per body re-eval — `Array(Set(map)).sorted()`
    /// is O(N) per call, called ~30× per render in a 500-row
    /// session. Maintained in lockstep with `lastKnownSpeakerIDs`
    /// at every utterance-mutation boundary.
    var cachedKnownSpeakerIDs: [String] = []

    /// Extra rows produced when a hand-edit commit contained more
    /// than one sentence. Keyed by the *parent* utterance id (the
    /// one that retains the original `id` after the split) and
    /// holding the new ids of every sibling row. Used by
    /// `revertReevaluation` so reverting the parent also cleans up
    /// the siblings the split spawned. Not persisted.
    /// Internal-write so the hand-edit extension can register
    /// split children alongside its commit path.
    var handEditChildren: [UUID: [UUID]] = [:]
    /// User-supplied display names keyed by stored speaker id
    /// (e.g. `"S01" → "Alice"`). When present, takes precedence
    /// over the default `S01`-style label `formatSpeakerLabel`
    /// produces. Cleared at session start;
    /// restored from the `.xph` bundle on `loadSession`. The stored
    /// id stays canonical — diarizer matching, JSON identity, and
    /// per-speaker tint keying all keep operating on the original
    /// `S01`-style key.
    var speakerNameOverrides: [String: String] = [:]
    private var playbackPlayer: AVAudioPlayer?
    private var playbackStopTask: Task<Void, Never>?
    /// Latest snapshot of the cumulative diarizer timeline.
    /// Refreshed by the continuous-diarize task after each ingest;
    /// reset at session start / loadSession. Drives the per-session
    /// timeline strip in the transcript pane. Not persisted with
    /// `makeSessionDocument` — after Open Session this stays empty
    /// until re-recording or hand-edit/reeval flows repopulate it.
    var diarizationTimeline: [DiarizedSegment] = [] {
        didSet { diarizationTimelineVersion &+= 1 }
    }
    /// Bumps on every `diarizationTimeline` assignment. Lets the
    /// mismatch + filter memos invalidate even when the rewrite
    /// happens to leave the segment count unchanged — e.g. a
    /// single covering segment replaced inline by
    /// `rewriteTimelineRange`. Without this, two consumers that
    /// memo against `timelineCount` could return stale (and
    /// divergent) mismatch sets after a correction / affirmation.
    private(set) var diarizationTimelineVersion: Int = 0

    /// Latest snapshot of the diarizer's speaker cluster — every
    /// known speaker's averaged centroid plus a tail window of raw
    /// observations. Drives the heatmap + PCA cluster panels.
    /// Refreshed on every continuous-diarize tick and on demand via
    /// `refreshClusterSnapshot()` (the panel kicks a 1 Hz loop while
    /// visible so the views stay live even when not recording).
    var speakerCluster: SpeakerClusterSnapshot =
        SpeakerClusterSnapshot(speakers: [])

    /// Per-utterance speaker embedding, captured at the moment the
    /// utterance is processed so the UI can match it back to the
    /// specific node in the cluster scatter. Lives outside
    /// `UtteranceEstimate` because the embedding doesn't belong in
    /// the JSON export (it's a 256-D blob with no analytical value
    /// to downstream consumers), and we don't want it bloating
    /// every `.xph` either — orphan entries cleared by the same
    /// retention logic that drops stale snapshot entries.
    var utteranceEmbeddings: [UUID: [Float]] = [:]

    /// Per-utterance pinned diarizer observation, keyed by the
    /// utterance id. Captured at the same site that fills
    /// `utteranceEmbeddings`: right after computing the embedding
    /// for a finalized row, we ask the diarizer which of its
    /// currently-resident observations is closest to that vector
    /// and stash its stable `RawEmbedding.segmentId`. The cluster
    /// scatter's tap path then resolves observation → utterance by
    /// looking up this map, instead of running a second cosine
    /// argmin at tap time — exact for observations still in the
    /// tail window, robust to later trimming/reorders. Nil for an
    /// utterance means the diarizer hadn't ingested a matching
    /// observation yet (rare; falls back to embedding-distance
    /// matching at tap time).
    var utteranceObservationSegmentIDs: [UUID: UUID] = [:]

    /// Ask the diarizer which currently-resident observation is
    /// closest to `embedding` for `speakerID` and stash its
    /// `segmentId` against `utteranceID`. Called immediately after
    /// every site that writes `utteranceEmbeddings`. No-op when
    /// the diarizer hasn't seen any matching observation yet
    /// (rare; the tap path falls back to embedding-distance
    /// matching). Awaiting from a `@MainActor` context is fine —
    /// the diarizer hop is the only suspension point and the
    /// caller is typically already in an async flow.
    func pinObservationSegmentID(
        utteranceID: UUID,
        embedding: [Float],
        speakerID: String
    ) async {
        guard let pipeline else { return }
        if let sid = await pipeline.bestMatchingObservationID(
            forEmbedding: embedding,
            speakerID: speakerID
        ) {
            utteranceObservationSegmentIDs[utteranceID] = sid
        }
    }

    /// Cap on raw observations carried in each `speakerCluster`
    /// refresh. 50 covers ~50 s of contiguous speech per speaker
    /// at the diarizer's 1 s windowing — enough cloud density for
    /// the PCA scatter to read as a cluster, but bounded so the
    /// projection step stays sub-10 ms even with 10 speakers.
    private static let clusterObservationsPerSpeaker = 50

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
    /// True iff a recorded transcript exists and we're idle enough
    /// to run actions over it (summarize / review / search /
    /// export). Each toolbar button starts from this and adds its
    /// own per-action busy flag.
    var isIdleWithTranscript: Bool {
        !utterances.isEmpty && !isRecording && !isAnalyzing
    }
    var isWarmingUp: Bool { phase == .warmingUp }
    var elapsedSeconds: Double {
        Double(samplesCaptured) / PipelineAudio.sampleRate
    }
    /// Fraction of the file's audio that has been captured so far, in
    /// `[0, 1]`. Nil when not in file mode or the duration probe at
    /// `startFromFile` couldn't read the file. Compares the pipeline's
    /// captured-sample audio time against the file's total duration.
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
    var lastChunkSpeakerCount: Int = 0
    /// Sentence count of the most recently-finalized ASR segment —
    /// the number of sub-segments `splitIntoSentences` produced
    /// from it. Surfaces in the pipeline visualization's ASR row.
    /// 0 before the first segment finalizes; reset each `start()`.
    var lastChunkSentenceCount: Int = 0

    /// MainActor-confined progress mirror the SetupView observes during
    /// first-launch model hydration.
    let modelDownload: ModelDownloadState
    /// Optional because `ModelStore.init` can throw if Application Support
    /// is inaccessible (sandbox edge cases). Nil here surfaces immediately
    /// as a hydration failure instead of crashing in the controller's init.
    /// Visible to sibling coordinators (e.g. `SummarizerCoordinator`)
    /// for their own model-hydration calls.
    let modelStore: ModelStore?
    /// Per-modality construction failures captured during pipeline
    /// pre-warm. Surfaced in the main UI as a small banner so the user
    /// knows when, e.g., the DeBERTa text SER silently dropped out
    /// because its FP16 model failed ORT load. Empty when everything
    /// loaded cleanly.
    private(set) var pipelineDiagnostics: [String] = []

    var capture: any AudioCapture
    let micCapture: any AudioCapture
    private var streamingTranscriber: any StreamingTranscriber
    var sourceMode: SourceMode = .microphone

    enum SourceMode: Equatable {
        case microphone
        case file(URL)
    }
    /// Internal access (not `private`) so sibling coordinators can
    /// release/reload the pipeline for memory orchestration around
    /// large-LLM inference.
    var pipeline: AnalysisPipeline?
    var pipelineTask: Task<AutoConfiguredPipeline, Never>?
    /// Most recent value seen by `setBackgroundMode(_:)`. Stored so a
    /// pipeline built after the app has already moved into background
    /// (rare but possible — e.g. resumed from a backgrounded scene)
    /// can be put into the correct EP state immediately on creation
    /// rather than waiting for the next scene-phase transition.
    private var latestBackgroundMode: Bool = false

    /// Per-component readiness shims for the "Models" card. Each
    /// proxies through to the active pipeline's nonisolated snapshot
    /// flags, or returns `false` while the pipeline is still pre-
    /// warming (the card hides under SetupView during that window
    /// anyway). Kept here rather than threaded through the view so
    /// `ModelsCard` doesn't have to know how `AnalysisPipeline` is
    /// stored.
    var pipelineHasDiarizer: Bool { pipeline?.hasDiarizer ?? false }
    var pipelineHasDimensionalSER: Bool { pipeline?.hasDimensionalSER ?? false }
    var pipelineHasCategoricalSER: Bool { pipeline?.hasCategoricalSER ?? false }
    var pipelineHasAgeGenderSER: Bool { pipeline?.hasAgeGenderSER ?? false }
    var pipelineHasDeBERTaTextSER: Bool { pipeline?.hasDeBERTaTextSER ?? false }

    private let exporter = JSONExporter()
    private var rawTask: Task<Void, Never>?
    private var feedTask: Task<Void, Never>?
    private var analysisTask: Task<Void, Never>?
    private var routeWatcherTask: Task<Void, Never>?
    /// Hardware-level connect/disconnect notification observers.
    /// `AVAudioSession.routeChangeNotification` only fires for active
    /// sessions, so while the app is idle between recordings, USB-C
    /// plug/unplug events don't reach it on iPadOS 26 — the picker
    /// then stays stale until the user explicitly opens it. The
    /// `AVCaptureDevice` connect/disconnect notifications fire
    /// globally for any audio (and video) device the system sees,
    /// regardless of session state, so we use those as the primary
    /// signal to refresh the input list on plug/unplug.
    private var deviceConnectedWatcherTask: Task<Void, Never>?
    private var deviceDisconnectedWatcherTask: Task<Void, Never>?
    /// Polling fallback for USB-C audio plug/unplug detection.
    /// On iPadOS 26, neither `AVAudioSession.routeChangeNotification`
    /// (silent for inactive sessions) nor `AVCaptureDevice.wasConnected
    /// /Disconnected` fires for USB-C mic events while the app is
    /// idle. No other documented notification covers this gap, so we
    /// poll the input list at 2 s intervals while idle. The poll is a
    /// metadata-only category swap + read; no audio session activation
    /// happens, so the cost is negligible.
    private var inputPollTask: Task<Void, Never>?
    private var volatilePollTask: Task<Void, Never>?
    private var fileEndWatcherTask: Task<Void, Never>?
    /// Sliding-window continuous diarization. Fires every
    /// `continuousDiarizeStrideSec` while recording and feeds the
    /// last `continuousDiarizeWindowSec` of audio to the pipeline's
    /// speaker timeline. The timeline is what `dominantSpeaker`
    /// queries per sentence — running it continuously rather than
    /// per-segment gives sharper speaker boundaries on rapid
    /// turn-takes (the per-segment 60 s context blurs them).
    private var continuousDiarizeTask: Task<Void, Never>?
    /// First chunk's timestamp from the raw audio pump, used to
    /// rebase capture timestamps into session-relative time. In file
    /// mode this is always 0; in mic mode it's the engine's
    /// `sampleTime / sampleRate` at first tap, which can be any
    /// non-zero value (the engine's clock continues across
    /// start/stop cycles). The transcriber does the same rebase
    /// internally for its anchors — we need it on the raw branch
    /// too so `capturedAudio`'s anchors and the diarizer's
    /// cumulative timeline land in the same timeline as ASR.
    private var rawPumpBaseTimestamp: TimeInterval?
    /// Latest session-relative file-time seen by the raw audio pump.
    /// Tracked here because `samplesCaptured / sampleRate` is
    /// output-time, which diverges from file-time by the per-chunk
    /// resampling drift (1-3 s over a 30 min file at 44.1 kHz).
    /// The continuous-diarize task reads this so its slice / fire
    /// cursors stay in the same timeline as `ASRSegment` ranges
    /// and the diarizer's `atTime` anchor.
    private var latestCapturedFileTime: TimeInterval = 0
    /// Latest file-time the continuous-diarize task has processed.
    /// Used as a lower bound when trimming the rolling capture
    /// buffer so audio isn't evicted before diarize reaches it.
    /// With the non-realtime file pump, ASR can race ahead of the
    /// diarize task transiently; without this guard, the
    /// per-segment trim would drop audio diarize still needs and
    /// the cumulative timeline would gap.
    private var lastDiarizedAudioTime: TimeInterval = 0
    /// Lock-Screen / Dynamic Island integration. See
    /// `LiveActivityController` for the activity-id, coalescing, and
    /// nonisolated update plumbing.
    private let liveActivity = LiveActivityController()
    /// Wall-clock time the current session began, captured at `start()`.
    /// Used by the Lock Screen / Dynamic Island clock so it ticks at
    /// real time alongside the audio timeline.
    private var sessionStartedAt: Date?
    /// True when audio time progresses at wall-clock rate, so the
    /// ASR finalize-latency metric is physically meaningful. Set
    /// true in mic mode (engine clock = wall clock) and false in
    /// file mode (the buffer-pipeline pump races ahead at multiple
    /// × real time, so "latency" relative to wall clock would
    /// report meaningless numbers).
    private var asrLatencyMeaningful: Bool = true
    /// Bounded rolling buffer over the raw capture stream. Owns the
    /// trim-before-append cap, the deep-copy-on-snapshot discipline,
    /// and the audio-time → buffer-index origin tracking. See
    /// `RollingAudioBuffer` for the invariants enforced.
    private var capturedAudio = RollingAudioBuffer(
        // Sized generously so the non-realtime pump can race ahead of
        // the continuous-diarize task without the head-cap evicting
        // audio diarize still needs. ASR is the typical bottleneck
        // (~5-15× real-time on M-class); diarize is faster but can
        // transiently fall behind on contention. 300 s gives ~5 min
        // of slack at ~38 MB resident (16 kHz × 4 B × 300 s).
        maxSeconds: 300,
        contextSeconds: 60
    )

    /// True when the streaming transcriber wasn't externally injected
    /// and we're free to swap it on a language change. Tests inject a
    /// stub via the init param; for them this flag stays false so
    /// `setSessionLanguage` leaves their stub in place rather than
    /// silently replacing it with a real Apple transcriber.
    private let canRecreateStreamingTranscriber: Bool

    private static let fusionAcousticWeightKey = "xephon.fusionAcousticWeight"
    private static let fusionTextWeightFloorKey = "xephon.fusionTextWeightFloor"
    private static let diarizerClusteringThresholdKey = "xephon.diarizerClusteringThreshold"


    /// Current weight applied to the acoustic modality during late
    /// fusion. Drives the live `LateFusion` instance on the
    /// analysis pipeline AND the per-row inspector / fusion strip's
    /// contribution-share rendering. Persists across launches via
    /// `fusionAcousticWeightKey`. Adjustable from
    /// `FusionLegendCard`'s sliders; the controller does NOT
    /// retroactively re-fuse existing utterances on a change —
    /// users re-evaluate specific rows if they want the new
    /// weights applied historically.
    private(set) var fusionAcousticWeight: Float
    /// Floor for the text modality weight (text weight =
    /// `max(floor, asrConfidence)`). Drives the same call sites
    /// as `fusionAcousticWeight`.
    private(set) var fusionTextWeightFloor: Float

    /// FluidAudio clustering threshold. Lower = stricter (more
    /// distinct speakers); higher = looser (merge similar voices).
    /// Persists via UserDefaults; pushed into the live pipeline on
    /// every bring-up.
    private(set) var diarizerClusteringThreshold: Float

    /// Shared "Teach diarizer" toggle state across every row's
    /// speaker popover. Flipping the switch in one row's popover
    /// propagates to every other row's popover, so a user
    /// correcting a batch of misattributions doesn't have to flip
    /// it repeatedly. Session-only (not persisted) — resets to off
    /// on launch so the heavier centroid-folding behavior never
    /// silently survives a cold start.
    var teachingDiarizer: Bool = false

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
        // Restore fusion weights from UserDefaults if the user has
        // touched the sliders before; otherwise inherit LateFusion's
        // compiled-in defaults. The `0` sentinel covers the
        // first-launch case where UserDefaults returns 0 for an
        // unset Float key (UserDefaults Float lookup returns 0.0
        // when the key is absent — distinguishable from a
        // deliberately-saved 0 only via `object(forKey:)`).
        if UserDefaults.standard.object(forKey: Self.fusionAcousticWeightKey) != nil {
            self.fusionAcousticWeight = UserDefaults.standard.float(forKey: Self.fusionAcousticWeightKey)
        } else {
            self.fusionAcousticWeight = LateFusion.defaultAcousticWeight
        }
        if UserDefaults.standard.object(forKey: Self.fusionTextWeightFloorKey) != nil {
            self.fusionTextWeightFloor = UserDefaults.standard.float(forKey: Self.fusionTextWeightFloorKey)
        } else {
            self.fusionTextWeightFloor = LateFusion.defaultTextWeightFloor
        }
        if UserDefaults.standard.object(forKey: Self.diarizerClusteringThresholdKey) != nil {
            self.diarizerClusteringThreshold = UserDefaults.standard.float(forKey: Self.diarizerClusteringThresholdKey)
        } else {
            self.diarizerClusteringThreshold = FluidAudioDiarizer.defaultClusteringThreshold
        }
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
        // Now that every stored property is set, build the coordinator.
        // Holds an unowned ref back to self, so init must finish here
        // before it's safe to construct.
        self.summarizer = SummarizerCoordinator(parent: self)
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
        // Hardware-level connect/disconnect — fires for USB-C audio
        // devices regardless of audio-session activation state, which
        // is what we need to keep the picker up to date while the app
        // is idle.
        deviceConnectedWatcherTask = Task { @MainActor [weak self] in
            let notifications = NotificationCenter.default.notifications(
                named: AVCaptureDevice.wasConnectedNotification
            )
            for await _ in notifications {
                guard let self else { break }
                AppLog.app.info("AVCaptureDevice.wasConnected fired; refreshing inputs")
                await self.refreshInputs()
            }
        }
        deviceDisconnectedWatcherTask = Task { @MainActor [weak self] in
            let notifications = NotificationCenter.default.notifications(
                named: AVCaptureDevice.wasDisconnectedNotification
            )
            for await _ in notifications {
                guard let self else { break }
                AppLog.app.info("AVCaptureDevice.wasDisconnected fired; refreshing inputs")
                await self.refreshInputs()
            }
        }
        // Polling fallback. iPadOS 26 doesn't fire any notification
        // we've found for USB-C audio plug/unplug while idle, so we
        // diff the input list against its last-known state every
        // ~2 s. Only triggers a full UI refresh when the set of
        // available input UIDs actually changes, so steady state is
        // a cheap array compare with no observable side effects.
        inputPollTask = Task { @MainActor [weak self] in
            var lastKnownUIDs: Set<String> = []
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(Self.inputPollIntervalSec))
                if Task.isCancelled { return }
                guard let self else { return }
                guard self.phase == .idle, self.playbackPlayer == nil else { continue }
                let currentUIDs = Set(self.availableInputs.map(\.uid))
                // Cheap probe: snapshot what session.availableInputs
                // reports right now (no category swap) and compare to
                // our cached UID set. Only do the full enumeration if
                // the snapshot differs from what the picker currently
                // shows.
                #if os(iOS) || targetEnvironment(macCatalyst)
                let session = AVAudioSession.sharedInstance()
                let probeUIDs = Set((session.availableInputs ?? []).map(\.uid))
                #else
                let probeUIDs = currentUIDs
                #endif
                if probeUIDs != currentUIDs || lastKnownUIDs != probeUIDs {
                    AppLog.app.info("input poll: detected change probe=\(probeUIDs.count, privacy: .public) cached=\(currentUIDs.count, privacy: .public); refreshing")
                    lastKnownUIDs = probeUIDs
                    await self.refreshInputs()
                }
            }
        }
    }

    /// Internal access (not `private`) so sibling coordinators can
    /// rewarm the pipeline after a memory-orchestrated release.
    func ensurePipeline() async -> AnalysisPipeline {
        if let pipeline { return pipeline }
        if let task = pipelineTask {
            let result = await task.value
            self.pipeline = result.pipeline
            self.pipelineDiagnostics = result.diagnostics
            self.pipelineTask = nil
            await applyConfiguration(to: result.pipeline)
            await syncTextSERStateFromPipeline(from: result.pipeline)
            return result.pipeline
        }
        guard let modelStore else {
            // Should be unreachable: modelsReady gates the main UI on
            // modelStore init success. Construct an empty pipeline so the
            // app degrades gracefully rather than hanging on an await.
            let configured = AnalysisPipeline()
            applyFusionWeights(to: configured)
            self.pipeline = configured
            return configured
        }
        let result = await AnalysisPipeline.autoConfigured(modelStore: modelStore)
        self.pipelineDiagnostics = result.diagnostics
        self.pipeline = result.pipeline
        await applyConfiguration(to: result.pipeline)
        await syncTextSERStateFromPipeline(from: result.pipeline)
        return result.pipeline
    }

    /// Re-apply the most recent scene-phase signal to a freshly-
    /// built pipeline. Cheap no-op when `latestBackgroundMode` is
    /// the default `false` (typical case — app launched into
    /// foreground and ensurePipeline ran during normal startup).
    private func applyLatestBackgroundMode(to pipeline: AnalysisPipeline) async {
        guard latestBackgroundMode else { return }
        await pipeline.setBackgroundMode(true)
    }

    /// Apply the user's current fusion-weight settings to the
    /// supplied pipeline so fresh utterances fuse under them.
    /// Called at every pipeline bring-up site and whenever the
    /// sliders move.
    private func applyFusionWeights(to pipeline: AnalysisPipeline) {
        pipeline.setFusionWeights(
            acousticWeight: fusionAcousticWeight,
            textWeightFloor: fusionTextWeightFloor
        )
    }

    /// Set the acoustic-modality weight applied during late
    /// fusion. Persists across launches and is pushed into the
    /// live pipeline so subsequent utterances fuse under the new
    /// value. Existing utterances keep their cached fused V/A/D —
    /// re-evaluate a row to apply the new weight to it.
    /// `utterancesVersion &+=` flips the filter-memo invalidation
    /// counter so any view computing contribution shares from
    /// `LateFusion.vaFusionShare`/`labelFusionShare` re-renders
    /// against the new value.
    func setFusionAcousticWeight(_ value: Float) {
        let clamped = value.clamped(to: 0...2)
        guard clamped != fusionAcousticWeight else { return }
        fusionAcousticWeight = clamped
        UserDefaults.standard.set(clamped, forKey: Self.fusionAcousticWeightKey)
        if let pipeline { applyFusionWeights(to: pipeline) }
        utterancesVersion &+= 1
        AppLog.app.info(
            "fusion.acousticWeight → \(clamped, privacy: .public)"
        )
    }

    /// Set the text-modality weight floor. Same persistence +
    /// pipeline-push semantics as `setFusionAcousticWeight`.
    func setFusionTextWeightFloor(_ value: Float) {
        let clamped = value.clamped(to: 0...1)
        guard clamped != fusionTextWeightFloor else { return }
        fusionTextWeightFloor = clamped
        UserDefaults.standard.set(clamped, forKey: Self.fusionTextWeightFloorKey)
        if let pipeline { applyFusionWeights(to: pipeline) }
        utterancesVersion &+= 1
        AppLog.app.info(
            "fusion.textWeightFloor → \(clamped, privacy: .public)"
        )
    }

    /// Restore both fusion weights to LateFusion's compiled-in
    /// defaults. Clears the persisted values so a future launch
    /// also inherits the defaults.
    func resetFusionWeights() {
        fusionAcousticWeight = LateFusion.defaultAcousticWeight
        fusionTextWeightFloor = LateFusion.defaultTextWeightFloor
        UserDefaults.standard.removeObject(forKey: Self.fusionAcousticWeightKey)
        UserDefaults.standard.removeObject(forKey: Self.fusionTextWeightFloorKey)
        if let pipeline { applyFusionWeights(to: pipeline) }
        utterancesVersion &+= 1
        AppLog.app.info("fusion weights reset to defaults")
    }

    /// Takes effect on the next diarize call; already-clustered
    /// observations keep their assignment — re-running the session
    /// re-clusters them.
    func setDiarizerClusteringThreshold(_ value: Float) async {
        let clamped = value.clamped(to: FluidAudioDiarizer.clusteringThresholdRange)
        // Epsilon, not `!=`: a slider re-emitting the same logical
        // value after a Float round-trip can still differ by 1 ULP,
        // which would defeat the early-return and re-write
        // UserDefaults on every drag tick.
        guard abs(clamped - diarizerClusteringThreshold) > .ulpOfOne else { return }
        diarizerClusteringThreshold = clamped
        UserDefaults.standard.set(clamped, forKey: Self.diarizerClusteringThresholdKey)
        if let pipeline {
            await pipeline.setDiarizerClusteringThreshold(clamped)
        }
        AppLog.app.info(
            "diarizer.clusteringThreshold → \(clamped, privacy: .public)"
        )
    }

    /// Restore the default threshold and clear the persisted
    /// override.
    func resetDiarizerClusteringThreshold() async {
        let defaultValue = FluidAudioDiarizer.defaultClusteringThreshold
        diarizerClusteringThreshold = defaultValue
        UserDefaults.standard.removeObject(forKey: Self.diarizerClusteringThresholdKey)
        if let pipeline {
            await pipeline.setDiarizerClusteringThreshold(defaultValue)
        }
        AppLog.app.info("diarizer clustering threshold reset to default")
    }

    /// Push every persisted setting into a freshly-built pipeline:
    /// fusion weights, diarizer sensitivity, and the latest
    /// background-mode snapshot. Centralizes the "what does this
    /// pipeline need to know" list so adding a new setting later
    /// is a one-line edit instead of N call-site edits.
    /// `syncTextSERStateFromPipeline` is intentionally separate — it reads
    /// state *out of* the pipeline rather than pushing into it.
    private func applyConfiguration(to pipeline: AnalysisPipeline) async {
        applyFusionWeights(to: pipeline)
        await pipeline.setDiarizerClusteringThreshold(diarizerClusteringThreshold)
        await applyLatestBackgroundMode(to: pipeline)
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
        await applyConfiguration(to: result.pipeline)
        await self.syncTextSERStateFromPipeline(from: result.pipeline)
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

    private func syncTextSERStateFromPipeline(from pipeline: AnalysisPipeline) async {
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
        summarizer.syncInstallState()
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
        speakerCluster = SpeakerClusterSnapshot(speakers: [])
        lastKnownSpeakerIDs = []
        cachedKnownSpeakerIDs = []
        // New session invalidates the previous summary + pending
        // issues — both pointed at utterances that are gone.
        summarizer.clearLastSummary()
        summarizer.clearIssues()
        sessionStartedAt = Date()
        lastASRFinalizeLatency = nil
        lastChunkSpeakerCount = 0
        lastChunkSentenceCount = 0
        lastDiarizedAudioTime = 0
        latestCapturedFileTime = 0
        rawPumpBaseTimestamp = nil
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
                // Rebase chunk timestamp into session-relative time
                // (subtract the first chunk's stamp). Mic-mode chunks
                // carry the engine's running sampleTime which can be
                // any value at the start of a fresh session — without
                // this, capturedAudio's anchors and the diarize
                // cursor would live at engine-time while ASR uses
                // session-relative time, producing a cumulative
                // timeline that doesn't line up with utterances and
                // a strip whose totalDuration explodes.
                let base = self.rawPumpBaseTimestamp ?? buffer.timestamp
                if self.rawPumpBaseTimestamp == nil {
                    self.rawPumpBaseTimestamp = base
                }
                let rebasedChunk = AudioChunk(
                    samples: buffer.samples,
                    sampleRate: buffer.sampleRate,
                    timestamp: buffer.timestamp - base
                )
                // Backpressure on diarize lag. The rolling buffer
                // has a hard `maxSeconds` cap; once it's hit,
                // `append()` silently evicts the oldest samples and
                // the continuous-diarize task's next slice into the
                // evicted range produces a gap in the cumulative
                // timeline. Pausing here lets diarize catch up
                // before the cap is reached. Audio is still being
                // pulled from the file pump on demand (AudioFileCapture
                // backpressures on us via `.bufferingOldest` +
                // retry-on-drop), so blocking here just slows the
                // upstream pipeline to the diarize-bound rate.
                let chunkEndFileTime =
                    rebasedChunk.timestamp + Double(rebasedChunk.samples.count) / rebasedChunk.sampleRate
                while !Task.isCancelled,
                      chunkEndFileTime - self.lastDiarizedAudioTime > Self.maxDiarizeLagSeconds {
                    try? await Task.sleep(for: .seconds(Self.diarizeBackpressurePollSec))
                }
                // The trim-before-append cap and origin advance
                // both live inside RollingAudioBuffer. See its
                // doc comment for why the order matters.
                self.capturedAudio.append(rebasedChunk)
                self.samplesCaptured += rebasedChunk.samples.count
                // Track latest file-time so the continuous-diarize
                // task can advance its cursor in file-time (matching
                // the timeline the diarizer's `atTime` anchor + the
                // ASR-corrected segment ranges both use). Chunk
                // duration is approximated at the nominal 16 kHz
                // rate; tiny per-chunk error vs. true file-rate is
                // absorbed by the diarizer's own segmentation.
                self.latestCapturedFileTime =
                    rebasedChunk.timestamp + Double(rebasedChunk.samples.count) / rebasedChunk.sampleRate
                // Per-channel smoothing. AudioCapture attaches one
                // level per source channel (before the mono downmix);
                // we smooth each channel against its prior value so a
                // stereo mic shows independent L/R movement instead of
                // a single mixed bar. If the channel count changes
                // mid-session (e.g. USB renegotiates), we reset the
                // smoothing state to match.
                if !buffer.channelLevels.isEmpty {
                    if self.inputChannelLevels.count != buffer.channelLevels.count {
                        self.inputChannelLevels = buffer.channelLevels
                    } else {
                        self.inputChannelLevels = zip(self.inputChannelLevels, buffer.channelLevels).map {
                            Self.smoothLevel(previous: $0.0, current: $0.1)
                        }
                    }
                    self.inputLevel = self.inputChannelLevels.max() ?? 0
                } else {
                    // File capture / no per-channel data: fall back to
                    // the mono samples for backward compatibility.
                    self.inputLevel = Self.smoothLevel(
                        previous: self.inputLevel,
                        current: Self.perceptualLevel(buffer.samples)
                    )
                    self.inputChannelLevels = [self.inputLevel]
                }
            }
            self.inputLevel = 0
            self.inputChannelLevels = []
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
                try? await Task.sleep(for: .seconds(Self.volatilePumpIntervalSec))
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
            // Fire on AUDIO-TIME progress, not wall-clock. With the
            // non-realtime file pump, wall-clock can run several×
            // faster than audio time — a single sleep-then-fire
            // iteration can cover many seconds of audio, exceeding
            // the 10-s window length and leaving gaps in the
            // cumulative timeline. The catch-up loop below fires
            // one diarize call per stride until we've covered all
            // the audio added since the last fire. Per-fire window
            // stays at the canonical `[fireEnd - windowSec, fireEnd]`
            // so observations overlap by `windowSec - strideSec` as
            // designed, regardless of how fast audio is arriving.
            var lastFiredAudioTime: TimeInterval = 0
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(Self.continuousDiarizeTickSec))
                if Task.isCancelled { return }
                guard self.isRecording else { continue }
                // File-time cursor — same timeline `slice`,
                // `FluidAudioDiarizer.atTime`, and ASR-corrected
                // segment ranges all use. Output-time
                // (`samplesCaptured / sampleRate`) would drift by
                // 1-3 s over a 30 min file and put diarize segments
                // in a different frame than ASR's, producing
                // dominantSpeaker lookup misses that look like
                // gaps in the cumulative timeline.
                let audioTime = self.latestCapturedFileTime
                var fired = false
                while true {
                    // First fire of the session waits for
                    // `minFirstDiarizeWindowSec` of audio rather than
                    // the usual stride. With only 2 s of audio the
                    // WeSpeaker embedding is too noisy to be stable,
                    // and the SpeakerManager registers a phantom
                    // "Speaker 1" whose centroid sits far from the
                    // real voice's embedding — so the second fire
                    // (with cleaner audio) crosses the new-speaker
                    // distance threshold and spawns "Speaker 2" for
                    // what is actually the same person. Holding off
                    // the first fire until the window has enough
                    // clean speech gives the initial centroid a
                    // chance to be representative.
                    let isFirstFire = lastFiredAudioTime == 0
                    let fireEnd: TimeInterval = isFirstFire
                        ? Self.minFirstDiarizeWindowSec
                        : lastFiredAudioTime + Self.continuousDiarizeStrideSec
                    guard fireEnd <= audioTime else { break }
                    let fireStart = max(0, fireEnd - Self.continuousDiarizeWindowSec)
                    lastFiredAudioTime = fireEnd
                    let window = self.capturedAudio.slice(start: fireStart, end: fireEnd)
                    guard !window.samples.isEmpty else { continue }
                    await pipeline.ingestDiarizationWindow(window)
                    // Publish progress so `trimProcessedAudio` can
                    // hold the buffer back if ASR races ahead.
                    self.lastDiarizedAudioTime = fireEnd
                    fired = true
                }
                guard fired else { continue }
                self.diarizationTimeline = await pipeline.diarizationTimelineSnapshot()
                self.speakerCluster = await pipeline.clusterSnapshot(
                    maxObservationsPerSpeaker: Self.clusterObservationsPerSpeaker
                )
                // Self-trim to keep the rolling buffer bounded
                // even when ASR is between finalizations (which
                // would otherwise gate the per-segment trim). The
                // `trimProcessedAudio` helper already respects the
                // `min(asr, lastDiarized) - context` floor, so this
                // is safe to call from here too — it just lets
                // diarize-side progress drive the trim cursor when
                // ASR's hasn't moved yet.
                self.trimProcessedAudio(below: self.lastDiarizedAudioTime)
            }
        }
    }

    /// Pull a fresh `speakerCluster` from the diarizer. View-facing
    /// hook for the heatmap + PCA panels — they spin a 1 Hz loop
    /// while visible so the panels stay live even when no recording
    /// is in progress. Cheap (just hands back resident `[Float]`
    /// arrays); safe to call when no pipeline is up (no-op).
    func refreshClusterSnapshot() async {
        guard let pipeline = self.pipeline else { return }
        let next = await pipeline.clusterSnapshot(
            maxObservationsPerSpeaker: Self.clusterObservationsPerSpeaker
        )
        // No-write when the new snapshot equals the resident one.
        // `clusterSnapshot()` builds fresh value-typed structs every
        // call, so a naive assignment fires @Observable didChange
        // even when the underlying speaker DB hasn't moved. With the
        // 1 Hz refresh loop in ContentView's TabView, that's a
        // didChange per second forever, cascading re-renders into
        // every ContentView child that reads `speakerCluster`
        // (cluster card, heatmap, roster) AND every sibling view in
        // the same parent body that SwiftUI then has to re-diff —
        // including the cumulative diarization strip in the
        // transcript pane, whose `.glassEffect` repaints visibly on
        // each rebuild. Equating first kills the loop's churn at
        // the source.
        if next != speakerCluster { speakerCluster = next }
    }

    /// Drain finalized ASR segments → split → SER+fuse → append.
    /// Each segment processes concurrently (TaskGroup) so a slow
    /// text-SER LLM call on segment N doesn't block segment N+1.
    /// Results are inserted in start-time order regardless of which
    /// task finishes first.
    ///
    /// Concurrency is capped at `maxConcurrentSegments` to bound
    /// peak memory during long file analysis. Without the cap, the
    /// capture pump can deliver tens of segments before any have
    /// finalized, each holding its own audio slice + ONNX I/O
    /// tensors — enough to OOM on long files. The poll-and-yield
    /// wait is coarse but lets the MainActor service updates between
    /// checks.
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
                    // for this utterance ended. Audio time tracks
                    // wall-clock for both mic and file capture today.
                    if self.asrLatencyMeaningful, let started = self.sessionStartedAt {
                        let endWallClock = started.addingTimeInterval(segment.end)
                        let latency = Date().timeIntervalSince(endWallClock)
                        self.lastASRFinalizeLatency = max(0, latency)
                    }
                    while self.inflightSegments >= Self.maxConcurrentSegments {
                        try? await Task.sleep(for: .seconds(Self.serSlotWaitSec))
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
                                // Stamp the per-utterance speaker
                                // embedding so the cluster scatter
                                // can later arrow at the exact node
                                // for this row. Concurrent with the
                                // result apply — the controller
                                // serializes both writes.
                                let embedding = await pipeline.extractSpeakerEmbedding(
                                    audio: split.audio.samples
                                )
                                await self?.applySegmentResult(
                                    estimate: estimate,
                                    metrics: metrics,
                                    embedding: embedding
                                )
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

        // Drain the continuous-diarize task so the cumulative
        // timeline covers all captured audio before we finalize.
        // Bounded wait so a misbehaving diarizer can't hang stop().
        // (capture / raw / feed have all drained above, so
        // `latestCapturedFileTime` is final at this point.)
        let diarizeDeadline = Date().addingTimeInterval(30)
        while lastDiarizedAudioTime + Self.continuousDiarizeStrideSec < latestCapturedFileTime,
              Date() < diarizeDeadline {
            try? await Task.sleep(for: .seconds(Self.diarizeDrainPollSec))
        }

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

        // Reconcile every utterance's speaker against the now-final
        // cumulative timeline. The streaming-pass assignment uses
        // the timeline as it stood when the utterance was first
        // emitted — by end-of-file, later observations may have
        // moved the per-instant majority. File mode only: mic
        // sessions have no "final" moment (the user could keep
        // recording), and reconciliation should run against a
        // settled timeline.
        if case .file = sourceMode {
            reconcileSpeakersWithTimeline()
        }
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

        // Re-enumerate the input list now that we're back in .idle.
        // Route-change notifications during `capture.stop()` and the
        // bindPreferredInput deactivate/reactivate cycle all fired
        // with `phase != .idle`, so `refreshInputs` couldn't query
        // authoritatively and skipped the assignment to preserve the
        // prior list. With phase now idle we can briefly switch to
        // .playAndRecord, enumerate all inputs, and restore — which
        // re-enables the picker.
        await refreshInputs()
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
        #if os(iOS) || targetEnvironment(macCatalyst)
        if canReconfigure {
            try? session.setCategory(priorCategory, mode: priorMode, options: priorOptions)
        }
        #endif
        AppLog.app.info("refreshInputs: phase=\(String(describing: self.phase), privacy: .public) canReconfigure=\(canReconfigure, privacy: .public) inputs.count=\(inputs.count, privacy: .public) current=\(current?.uid ?? "<nil>", privacy: .public)")
        // Only commit the refreshed inputs when the query had a chance
        // to enumerate all ports. Without `canReconfigure` we read with
        // whatever category the session happens to be in (mid-record,
        // mid-deactivate, etc.), and `session.availableInputs` can
        // return a transient subset — or [] when the session was just
        // deactivated. Overwriting `availableInputs` with that stale
        // snapshot leaves the picker disabled (count ≤ 1) after the
        // first recording, since route-change notifications from the
        // bindPreferredInput deactivate/reactivate cycle all fire
        // while phase is still .recording. Keep the prior list when
        // we couldn't query authoritatively; the next idle refresh
        // (or the post-stop refresh) updates it cleanly. Also keep
        // the prior list when an idle query came back empty — that's
        // typically a deactivation-transient (the OS hasn't re-
        // enumerated the USB device yet) rather than a legitimate
        // "no inputs connected" state.
        if !inputs.isEmpty {
            self.availableInputs = inputs
            self.currentInputUID = current?.uid
        } else {
            AppLog.app.info("refreshInputs: empty query result; keeping prior list (count=\(self.availableInputs.count, privacy: .public))")
        }
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
        #if os(iOS) || targetEnvironment(macCatalyst)
        let reason = (AVAudioSession.sharedInstance().currentRoute.inputs.first?.portName ?? "<none>")
        AppLog.app.info("handleAudioRouteChange fired; currentInput=\(reason, privacy: .public) phase=\(String(describing: self.phase), privacy: .public)")
        #endif
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
        await syncTextSERStateFromPipeline(from: pipeline)
    }

    /// Forward a foreground/background lifecycle transition to the
    /// SER pipeline so each acoustic actor can swap its ORT session
    /// between CoreML EP (foreground) and CPU (background). Driven
    /// by the scene-phase observer in `ContentView`. Stored locally
    /// in `latestBackgroundMode` so a freshly-built pipeline picks
    /// up the right state if the app happened to be backgrounded
    /// during init.
    func setBackgroundMode(_ inBackground: Bool) async {
        latestBackgroundMode = inBackground
        guard let pipeline else { return }
        await pipeline.setBackgroundMode(inBackground)
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
        await syncTextSERStateFromPipeline(from: pipeline)
        AppLog.app.info(
            "session language → \(language.rawValue, privacy: .public)"
        )
    }

    func selectInput(uid: String?) async {
        AppLog.app.info("selectInput called with uid=\(uid ?? "<nil>", privacy: .public)")
        // Record the user's intent immediately so the picker label
        // reflects it even if the OS route doesn't actually switch
        // (or flickers back). This is the picker's source of truth;
        // `currentInputUID` is only used as diagnostic.
        self.selectedInputUID = uid
        do {
            try await capture.setPreferredInput(uid)
            await refreshInputs()
            AppLog.app.info("selectInput post-refresh: selectedInputUID=\(self.selectedInputUID ?? "<nil>", privacy: .public) currentInputUID=\(self.currentInputUID ?? "<nil>", privacy: .public)")
        } catch {
            errorMessage = String(describing: error)
            AppLog.app.error("setPreferredInput failed: \(String(describing: error), privacy: .public)")
        }
    }

    /// The input that the next recording will actually use. Matches
    /// the bindPreferredInput fallback in `AudioCapture.start()`:
    /// explicit user pick → that UID; no pick → built-in mic UID
    /// from the current available-inputs list. The picker label
    /// binds to this so what the user sees is what they get.
    var effectiveInputUID: String? {
        if let selectedInputUID { return selectedInputUID }
        return availableInputs.first { $0.kind == .builtInMic }?.uid
    }

    // MARK: - File-backed source

    /// Switch to a file-backed audio source and immediately begin
    /// streaming it through the same pipeline used for the microphone.
    /// File analysis runs **non-realtime** under the buffer-pipeline
    /// pump: chunks emit as fast as downstream consumers can swallow,
    /// with retry-on-drop backpressure. Chunk timestamps are still
    /// anchored to the source file's audio-time axis, so the pipeline
    /// sees a continuous monotonic file-time clock — see
    /// `RollingAudioBuffer`'s anchor-based mapping for the detail.
    func startFromFile(_ url: URL) async {
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
        // File analysis runs as fast as the consumers can drain — see
        // AudioFileCapture's doc-comment. `asrLatencyMeaningful` stays
        // false here because chunk timestamps (file-time) decouple
        // from wall-clock when the pump runs faster than 1×.
        asrLatencyMeaningful = false
        capture = AudioFileCapture(fileURL: url)
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
    func setPlaybackSourceURL(_ newURL: URL?) {
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
                try? await Task.sleep(for: .seconds(duration))
                guard !Task.isCancelled else { return }
                self?.stopPlayback()
            }
        } catch {
            AppLog.app.error("playback open failed: \(String(describing: error), privacy: .public)")
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
    func commitUtteranceChanges() {
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
    static func mergedEstimate(
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
            ageGender: fresh.ageGender,
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


    /// Play `[start, end]` from the source file. Used by the Edit
    /// Utterance dialog's preview button — the dialog's spinners
    /// hold arbitrary times that don't correspond to an existing
    /// utterance row, so `togglePlayback(for:)` doesn't apply.
    ///
    /// `owner` ties the playback session to a specific utterance id
    /// when one applies — the review sheet's inline play buttons use
    /// this so each card knows whether it's the active one. Pass
    /// nil for arbitrary-range previews (the edit dialog's preview
    /// button, where the spinners may not correspond to any row).
    func playRange(
        start: TimeInterval,
        end: TimeInterval,
        owner: UUID? = nil
    ) {
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
            playingUtteranceID = owner
            let duration = max(0, end - start)
            playbackStopTask = Task { @MainActor [weak self] in
                try? await Task.sleep(for: .seconds(duration))
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
    /// to the default `S01`-style label). Bumps
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

    /// Walk every utterance once and reassign its `speakerID` to
    /// the cumulative timeline's per-instant majority verdict for
    /// its `[start, end]` window. Streaming-pass assignment uses
    /// the timeline as it stood at finalize-time; by end-of-file,
    /// later observations may have shifted the majority. This pass
    /// brings the labels into sync with the strip so the two
    /// visualizations agree before the session goes idle.
    ///
    /// Mutates in place + a single `commitUtteranceChanges()` at
    /// the end, so the auto-demote sweep and version bump fire
    /// once for the whole batch instead of N times.
    private func reconcileSpeakersWithTimeline() {
        guard !diarizationTimeline.isEmpty, !utterances.isEmpty else { return }
        var changed = 0
        for i in utterances.indices {
            let utt = utterances[i]
            let dominant = AnalysisPipeline.dominantSpeakerInSegments(
                diarizationTimeline,
                from: utt.start,
                to: utt.end,
                fallback: utt.speakerID
            )
            if dominant != utt.speakerID {
                utterances[i] = utt.withSpeakerID(dominant)
                changed += 1
            }
        }
        guard changed > 0 else { return }
        AppLog.app.info(
            "finalize: reconciled \(changed, privacy: .public) utterance speaker assignments against cumulative timeline"
        )
        commitUtteranceChanges()
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

    /// Utterance whose `[start, end)` contains `t`, or the one
    /// whose midpoint is closest if no row contains `t`. Used by
    /// the timeline strips' tap-to-scroll.
    func nearestUtterance(toTime t: TimeInterval) -> UtteranceEstimate? {
        if let containing = utterances.first(where: { $0.start <= t && t < $0.end }) {
            return containing
        }
        return utterances.min {
            abs(($0.start + $0.end) / 2 - t) < abs(($1.start + $1.end) / 2 - t)
        }
    }

    /// Exact-id resolution from a diarizer observation segment id
    /// back to its emitting utterance. Nil for sessions that
    /// pre-date observation pinning, and for centroid taps which
    /// pass `nil` as the segment id by convention.
    func utterance(forSegmentID sid: UUID) -> UtteranceEstimate? {
        guard let uid = utteranceObservationSegmentIDs.first(
            where: { $0.value == sid }
        )?.key else { return nil }
        return utterances.first(where: { $0.id == uid })
    }

    /// Argmin Euclidean distance from `query` over utterances
    /// belonging to `speakerID`. Constraining to one speaker
    /// matters because overlapping clouds in the cluster scatter
    /// otherwise let a global argmin land in a neighbor's cloud.
    /// Nil when the speaker has no utterance with a stored
    /// embedding (older session, mic-mode without diarizer, or
    /// auto-demoted centroid).
    func nearestUtterance(
        toEmbedding query: [Float],
        speakerID: String
    ) -> UtteranceEstimate? {
        guard !utteranceEmbeddings.isEmpty else { return nil }
        let speakerByID = utterances.reduce(into: [UUID: String]()) {
            $0[$1.id] = $1.speakerID
        }
        var bestID: UUID?
        var bestDist: Float = .infinity
        for (id, e) in utteranceEmbeddings {
            guard speakerByID[id] == speakerID else { continue }
            let n = min(query.count, e.count)
            guard n > 0 else { continue }
            var sum: Float = 0
            for j in 0..<n {
                let d = e[j] - query[j]
                sum += d * d
            }
            if sum < bestDist {
                bestDist = sum
                bestID = id
            }
        }
        guard let id = bestID else { return nil }
        return utterances.first(where: { $0.id == id })
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

    /// "Affirm": tell the diarizer that the row's currently-assigned
    /// speaker is correct. Extracts the row's audio embedding and
    /// folds it into the **current** speaker's centroid via EMA, the
    /// same teaching path `correctUtteranceSpeaker` uses for a
    /// different target — so subsequent re-eval / hand-edit on
    /// acoustically similar audio is more likely to match this
    /// speaker. No row reassignment, no cumulative-timeline rewrite,
    /// no utterance mutation: the assignment is already what the
    /// user wants; this just reinforces it in the diarizer DB.
    ///
    /// No-op when source audio is absent (mic mode), the row doesn't
    /// exist, the audio slice is empty, embedding extraction is
    /// unavailable, or the underlying diarizer call throws. Returns
    /// true on success.
    @discardableResult
    func affirmUtteranceSpeaker(utteranceID: UUID) async -> Bool {
        guard let url = playbackSourceURL else { return false }
        guard let index = utterances.firstIndex(where: { $0.id == utteranceID }) else { return false }
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
                "affirm: audio read failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        guard !chunk.samples.isEmpty else {
            AppLog.app.warning("affirm: empty audio slice")
            return false
        }
        let pipeline = await ensurePipeline()
        guard let embedding = await pipeline.extractSpeakerEmbedding(audio: chunk.samples) else {
            AppLog.app.warning("affirm: embedding extractor unavailable")
            return false
        }
        let duration = Float(max(0, utt.end - utt.start))
        do {
            try await pipeline.correctSpeaker(
                id: utt.speakerID,
                embedding: embedding,
                duration: duration
            )
        } catch {
            AppLog.app.error(
                "affirm: DB update failed: \(String(describing: error), privacy: .public)"
            )
            return false
        }
        // Declare the affirmed speaker as authoritative over the
        // row's audio range, the same way `correctUtteranceSpeaker`
        // does for a different target. Without this the cumulative
        // timeline may still hold competing observations for
        // `[utt.start, utt.end]`, so the mismatch detector keeps
        // flagging the row and the orange caution glyph stays put
        // — even though the user just confirmed the assignment.
        rewriteTimelineRange(
            speakerID: utt.speakerID,
            start: utt.start,
            end: utt.end
        )
        AppLog.app.info(
            "speaker affirmed: utt=\(utt.id, privacy: .public) \(utt.speakerID, privacy: .public) (reinforced diarizer, timeline range rewritten)"
        )
        // The timeline rewrite already bumped `diarizationTimelineVersion`
        // via the didSet, which is what the mismatch / filter memos
        // key on — no need for `commitUtteranceChanges()` here since
        // utterance content is unchanged.
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
        let newID = nextAvailableSpeakerID()
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

    /// Smallest `S0N` id not in use anywhere the user might see it:
    /// neither in the current utterance roster nor in the diarizer's
    /// internal speaker DB. The latter matters because FluidAudio's
    /// streaming `assignSpeaker` mints raw numeric ids ("4") that
    /// our snapshot formats to "S04" for display — picking that same
    /// "S04" for a promote would leave two distinct DB entries
    /// rendering as the same chip and trigger SwiftUI's "id occurs
    /// multiple times" warning on the cluster panels. Used by every
    /// "new speaker" allocation path (chip-menu reassign, promote).
    func nextAvailableSpeakerID() -> String {
        var used = cachedKnownSpeakerIDs
        used.append(contentsOf: speakerCluster.speakers.map(\.id))
        return Self.nextNewSpeakerID(in: used)
    }

    /// Smallest unused `S0N` id given the supplied existing ids.
    /// Backs `nextAvailableSpeakerID` — kept static + pure so callers
    /// that already have a specific id set in hand can use it
    /// directly. Both "New Speaker" and "Promote New Speaker" route
    /// through `nextAvailableSpeakerID` so they agree on which slot
    /// to fill.
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
    func rewriteTimelineRange(
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

    /// Replace every cumulative-timeline observation overlapping
    /// `[coveringStart, coveringEnd]` with the per-slice segments
    /// in `slices`. Segments outside the covered range survive
    /// intact; segments straddling either boundary get trimmed at
    /// the boundary; segments fully inside are dropped. Used after
    /// a multi-sentence hand-edit re-diarizes the parent window —
    /// without this, the streaming pass's older per-instant
    /// majority would still drive the timeline strip and disagree
    /// with the post-edit per-row speaker labels.
    func rewriteTimelineWithSlices(
        coveringStart: TimeInterval,
        coveringEnd: TimeInterval,
        slices: [DiarizedSegment]
    ) {
        guard coveringEnd > coveringStart, !slices.isEmpty else { return }
        var rewritten: [DiarizedSegment] = []
        rewritten.reserveCapacity(diarizationTimeline.count + slices.count)
        for seg in diarizationTimeline {
            if seg.end <= coveringStart || seg.start >= coveringEnd {
                rewritten.append(seg)
            } else if seg.start < coveringStart && seg.end > coveringEnd {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: coveringStart))
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: coveringEnd, end: seg.end))
            } else if seg.start < coveringStart {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: seg.start, end: coveringStart))
            } else if seg.end > coveringEnd {
                rewritten.append(DiarizedSegment(speakerID: seg.speakerID, start: coveringEnd, end: seg.end))
            }
            // Fully inside → drop.
        }
        for slice in slices where slice.end > slice.start {
            rewritten.append(slice)
        }
        rewritten.sort { $0.start < $1.start }
        diarizationTimeline = rewritten
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
    private func applySegmentResult(
        estimate: UtteranceEstimate,
        metrics: ProcessingMetrics,
        embedding: [Float]? = nil
    ) async {
        // Stamp speech-boost state so the row badge reflects how the audio
        // was captured. Best-effort: reads the live toggle, which is fine
        // for the common case where it's not flipped mid-recording.
        let stamped = estimate.withSpeechBoost(isSpeechBoostEnabled)
        let index = utterances.firstIndex { $0.start > stamped.start } ?? utterances.endIndex
        utterances.insert(stamped, at: index)
        if let embedding = embedding {
            utteranceEmbeddings[stamped.id] = embedding
            await pinObservationSegmentID(
                utteranceID: stamped.id,
                embedding: embedding,
                speakerID: stamped.speakerID
            )
        }
        conversationSummary.update(with: stamped)
        // Keep the speaker-id cache current for the streaming path
        // too (only `commitUtteranceChanges` is reached on edits;
        // streaming inserts bypass it). Cheap: insertion into a
        // small set + sort only when a new speaker appears.
        if !lastKnownSpeakerIDs.contains(stamped.speakerID) {
            lastKnownSpeakerIDs.insert(stamped.speakerID)
            cachedKnownSpeakerIDs = lastKnownSpeakerIDs.sorted()
        }
        lastAcousticDuration = metrics.acousticDuration
        lastTextDuration = metrics.textDuration
        lastSegmentTotal = metrics.totalDuration
        // Coalesce Live Activity updates instead of spawning one task per
        // segment — dense turn-taking can fire segments faster than the
        // system can process Activity updates, and the queued tasks
        // themselves contribute to MainActor congestion.
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
    /// wall-clock since session start for parity with the audio
    /// timeline. Recenters V/A to [-1, +1] for parity with the in-app
    /// summary.
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
    /// from CoreML EP under sustained load on 16 GB iPad Pro: each
    /// acoustic SER inference allocates IOSurface-backed tensors per
    /// running call, and 2 segments × 2 acoustic models (× 3 input
    /// bins worth of compiled MLModels each) saturates the system
    /// IOSurface pool. ASR typically emits 1-2 segments/sec so 2-wide
    /// concurrency is rarely the throughput bottleneck anyway.
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
    /// going longer makes rapid turn-takes show up late.
    private static let continuousDiarizeStrideSec: TimeInterval = 2
    /// Minimum audio-time before the FIRST continuous-diarize call
    /// fires. Set equal to `continuousDiarizeWindowSec` so the initial
    /// fire processes the canonical full-length window, matching what
    /// every steady-state fire sees. A shorter first window builds a
    /// tight WeSpeaker centroid around only the early seconds of
    /// speech; the next overlapping fire then includes slightly later
    /// audio that lands just beyond `clusteringThreshold × 1.2` from
    /// that tight centroid, which FluidAudio's SpeakerManager treats
    /// as a brand-new speaker and registers a phantom ID covering
    /// the region. Earlier iterations tried 2 s (the stride) and 5 s
    /// (one-stride past stride); both produced consistent phantoms
    /// at the [stride, window] boundary. Utterances finalized before
    /// the first fire still fall back to `dominantSpeaker`'s empty-
    /// timeline default ID.
    private static let minFirstDiarizeWindowSec: TimeInterval = 10
    /// Maximum audio-time gap the raw audio pump allows between
    /// `latestCapturedFileTime` and `lastDiarizedAudioTime` before
    /// blocking to let diarize catch up. Sized below
    /// `capturedAudio.maxSeconds` minus `contextSeconds` minus a
    /// safety margin so the rolling buffer's hard-cap eviction
    /// never trips on diarize-side lag — eviction silently drops
    /// audio and produces gaps in the cumulative timeline.
    private static let maxDiarizeLagSeconds: TimeInterval = 220

    private func trimProcessedAudio(below boundary: TimeInterval) {
        // Don't trim past where the continuous-diarize task has
        // processed. Otherwise on a non-realtime pump where ASR
        // races ahead of diarize, the head-eviction would drop
        // audio the diarize task still wants to slice — producing
        // gaps in the cumulative timeline. While diarize hasn't
        // fired yet (lastDiarizedAudioTime == 0), this gate just
        // means we don't trim — bounded by `maxSeconds` anyway.
        let effective = min(boundary, lastDiarizedAudioTime)
        guard effective > 0 else { return }
        capturedAudio.trimProcessed(below: effective)
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
        return ((db + 60) / 60).clamped(to: 0...1)
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
