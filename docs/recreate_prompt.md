# Build Prompt: Xephon — Japanese Conversational Affect Analyzer (iPadOS)

> A real-talk first: no single prompt will *completely, accurately, correctly*
> recreate this app. Tens of thousands of lines, many design choices that came
> from empirical testing (specific bin sizes, thresholds, race fixes), and UI
> polish from iteration won't all be re-derived from a description. This prompt
> reliably recreates the **architecture, constraints, and intent**, and most of
> the load-bearing design decisions — but the implementing model will make
> different micro-choices, and tuning constants will need re-calibration on
> real audio.

---

You are building an **iPadOS 26+ research app** (macOS 15+ as a secondary
"Designed for iPad" target) that takes Japanese conversational audio and
produces **per-utterance multimodal emotion estimates** — both **categorical**
(Plutchik 8 / Ekman 7) and **dimensional** (valence, arousal, dominance).
Status: research-only, sideload via Xcode / personal TestFlight, **not** App
Store.

## Hard constraints (non-negotiable)

1. **Strictly on-device.** No cloud LLMs, no cloud ASR, no analytics, no
   telemetry. The user's audio never leaves the device. This overrides any
   convenience argument.
2. **Swift 6, strict concurrency.** All inference is `async`. No
   `@unchecked Sendable` without a comment explaining why.
3. **One model = one actor.** Inference actors hold the `MLModel`/ORT
   session and serialize calls. Never share an `MLModel` across actors.
4. **Typed errors** per subsystem (`ASRError`, `SERError`, `AudioError`, …).
   No `throws Error`.
5. **`os.Logger`** with per-subsystem categories (`AppLog.app`, `AppLog.asr`,
   `AppLog.serText`, `AppLog.diarization`, `AppLog.fusion`). No `print()`.
6. **16 kHz mono Float32 throughout.** Resample once at capture; never
   re-resample mid-pipeline.
7. **ANE-first compute.** Use Core ML `MLComputeUnits.cpuAndNeuralEngine`
   unless a model is empirically faster on `.all`. For ONNX Runtime models,
   prefer the CoreML EP in foreground and fall back to CPU when backgrounded.
8. **No `.wav` / `.m4a` / participant audio in git.** Ever.
9. **`SpeechTranscriber` requires a 16-core Neural Engine** (M-series iPad
   Pro). Always check `SpeechTranscriber.isAvailable` and degrade gracefully.
   **It also doesn't run in the iOS Simulator** — gate eval tests accordingly.
10. **No Custom Vocabulary in `SpeechAnalyzer`.** For domain terminology,
    re-score short clips with `SFSpeechRecognizer.contextualStrings`, or fall
    back to WhisperKit / Qwen3-ASR.

## Canonical pipeline

```
AVAudioEngine (16 kHz mono Float32)
  → FluidAudio: Silero VAD + Sortformer diarization (Core ML, ANE)
  → ASR
       primary:   SpeechAnalyzer + SpeechTranscriber (ja_JP, on-device)
       fallback:  WhisperKit + Kotoba-Whisper-v2.0 (Core ML)
       fallback:  FluidAudio Qwen3-ASR (Core ML, iOS 18+)
  → Acoustic SER (parallel with text SER)
       audeering wav2vec2-large-robust-12-ft-emotion-msp-dim  → V/A/D
       emotion2vec_plus_large                                  → 9-class softmax
       audeering W2V2 age-gender                                → demographics
  → Text SER
       fine-tuned Japanese DeBERTa-v3-large on WRIME           → 8-Plutchik
       (optional) Apple Foundation Models 3B                   → structured V/A
  → Late fusion (weighted, ASR-confidence-aware) — NOT a trained head
  → Per-utterance JSON / .xph export
```

**Late fusion is the default.** Do NOT introduce a trained cross-modal head
without an in-domain calibration dataset and explicit user go-ahead.

## Build system

- **XcodeGen** for project generation (`project.yml` → `Xephon.xcodeproj`,
  gitignored). Provide `scripts/generate_project.sh`.
- **Swift Package Manager** for dependencies, pinned in `Package.resolved`:
  - `argmaxinc/WhisperKit`
  - `FluidInference/FluidAudio` (>= 0.5.0)
  - `microsoft/onnxruntime-swift-package-manager` (with CoreML EP)
  - `huggingface/swift-transformers` (1.0.x — keep pinned until
    mlx-swift-examples moves past 1.0)
  - `ml-explore/mlx-swift` + `mlx-swift-examples` (only for the summarizer)
- System frameworks: `Speech`, `AVFoundation`, `CoreML`, `Accelerate`,
  `FoundationModels` (iPadOS 26+), `SwiftUI`.
- ML model weights **are not committed**. `scripts/fetch_models.sh` hydrates
  them from a GitHub Release attached to a `models-v1` tag, runs
  `coremltools` conversions where needed, and installs into
  `Application Support/Models/<tag>/`.
- Per-file checksum verification (`ModelManifest` + `ModelStore`) with
  `isExcludedFromBackup`.

## Repository layout

```
Xephon/                        SwiftUI app target
  App/                         ContentView, XephonApp, SetupView
  Recording/                   RecordingController (+extension files
                               per concern: +Inputs, +Playback, +HandEdit,
                               +Reevaluation, +SpeakerEditing, +SessionPersistence,
                               +FileSource), AnalysisPipeline,
                               SummarizerCoordinator
  Models/                      ModelManifest, ModelStore
  Session/                     SessionFileDocument, SessionFileCoordinator,
                               SessionFileBridge, SessionLanguage
  Glossary/                    GlossaryStore (user-editable lexicon)
  Search/                      JapaneseSearchNormalizer
  Views/
    Main/                      ControlPaneView, TranscriptPaneView,
                               MainToolbar, LLMSheetBridge
    Cards/Settings/            SettingsCard, PipelineCard
    Cards/Affect/              SERAggregateCard, FusionLegendCard, ...
    Cards/Speakers/            SpeakerClusterCard, SpeakerHeatmapCard,
                               TurnTakingCard, ReactivityCard, ...
    Cards/Summarizer/          SummarizerCard, ModelsCard
    Sheets/                    CustomGlossarySheet,
                               TranscriptionReviewSheet/Coordinator,
                               SearchReplaceSheet/Coordinator,
                               SessionSummarySheet, ShareSheet
    Transcript/                UtteranceRow, UtteranceBadges,
                               TranscriptList, TranscriptFilterBar,
                               EditUtteranceSheet, NewUtteranceCapsule
    Status/                    PipelineDiagnosticsBanner, LevelMeterView,
                               CircularDownloadProgress
    Strips/                    DiarizationTimelineStrip,
                               EmotionTimelineStrip, FusionContributionStrip
    Shared/                    DisplayFormatting (badge view-models,
                               ClockTimeFormatStyle, ElapsedTimeLabel)
  Resources/                   Localizable.strings (en + ja)
  Info.plist, entitlements
Core/
  Audio/                       AudioCapture protocol + AVAudio impl,
                               AudioFileCapture, RollingAudioBuffer,
                               AudioChunk
  ASR/                         ASRSegment, SpeechAnalyzerTranscriber,
                               WhisperKitTranscriber, QwenTranscriber
  Diarization/                 FluidAudioDiarizer, FluidAudioVAD,
                               DiarizedSegment, SpeechSegment,
                               StreamingSpeakerTracker, StreamingVADTracker
  SER/
    Acoustic/                  DimensionalSER (W2V2), CategoricalSER
                               (emotion2vec+), AgeGenderSER,
                               VADScore, CategoricalEmotion, AgeGenderEstimate
    Text/                      TextSER protocol, WRIMETextSER,
                               FoundationModelsSER, SwitchingTextSER,
                               LexiconBias, TextSERError
  Fusion/                      LateFusion (default Fuser),
                               UtteranceEstimate, UtteranceFusionReport,
                               AffectiveSynchrony, TurnTakingDynamics,
                               ModalityDisagreement, SpeakerOrdering,
                               SynchronyAxis
  Export/                      SessionBundle (binary plist .xph),
                               JSONExporter
  Summarizer/                  AppleFMSummarizer, MLXQwenSummarizer,
                               TranscriptionIssue, SessionSummary
  XephonLogging/               AppLog enum with per-subsystem Logger
  XephonUtilities/             Clamped, small extensions
Tests/
  UnitTests/                   UtteranceCodableRoundtripTests, ...
  EvalTests/                   ASR WER/CER + SER metrics on held-out clips
docs/                          feasibility.md, output_schema.md,
                               models.md, eval_log.md, privacy.md,
                               ser_speaker_active_slicing.md, etc.
scripts/                       generate_project.sh, fetch_models.sh,
                               upload_models_to_github_release.sh
```

## Audio capture

- `AudioCapture` protocol with two implementations: `AVAudioCapture` (mic)
  and `AudioFileCapture` (file source).
- Single `RollingAudioBuffer` storing 16 kHz mono Float32 with bounded
  `maxSeconds`. Used for diarize windows.
- Speech-boost EQ toggle (mic mode only) — boosts the 1–4 kHz speech band
  for ASR clarity. Applies to ASR feed only, never to acoustic SER input.
- Audio route change handling via `AVAudioSession.routeChangeNotification`.
  Polling fallback for USB-C plug/unplug while idle (no public notification).
- Honor user's input selection over iPadOS auto-route (e.g., USB mic plug
  shouldn't override the user's "Built-in" pick).

## Diarization (FluidAudio)

- `FluidAudioDiarizer` wraps FluidAudio's Sortformer + WeSpeaker stack.
  Exposes `diarize(_:)` for one-shot and
  `resolveSpeakersForRanges(audio:ranges:fallback:preserveSpeakerDatabase:)`
  for hand-edit / re-eval lookups.
- `FluidAudioVAD` wraps the standalone Silero `VadManager`. Returns
  `[SpeechSegment]` at ~30 ms resolution — pre-aggregation, so it catches
  intra-utterance silences the diarizer's speaker-track segments absorb.
- User-tunable clustering threshold via a "speaker sensitivity" slider in
  Settings (inverts threshold so "right = more speakers"). Double-tap label
  to reset.
- `StreamingSpeakerTracker` (actor) maintains cumulative per-speaker
  timeline across diarize fires; append-and-vote semantics with a
  `cumulativeCap`.
- `StreamingVADTracker` (actor) maintains cumulative VAD timeline;
  merge-on-ingest (union of overlapping speech segments).

## ASR

- Primary: `SpeechAnalyzerTranscriber` wrapping `SpeechAnalyzer` +
  `SpeechTranscriber`. Emits both volatile (in-progress) and final segments.
  Volatile callback drives a 5 Hz UI poll for live preview text. ASR feed
  pump is separate from the analysis pump.
- Re-evaluation path (`AnalysisPipeline.reevaluate` /
  `transcribeForReevaluation`): re-feeds an already-saved audio slice
  through offline ASR with front-pad (500 ms) and adaptive back-pad retry
  for short utterances. Sentence-aware trim drops anything past the last
  terminator.
- Transcribe-arbitrary-range helper
  (`RecordingController.transcribeRange(start:end:)`) for the Edit Utterance
  sheet's Transcribe button.

## Acoustic SER

- Three independent models, all run in parallel per segment:
  - **DimensionalSER** (W2V2-MSP-DIM via ONNX Runtime): outputs `VADScore`.
  - **CategoricalSER** (emotion2vec+ via ONNX Runtime): outputs
    `CategoricalEmotion` with 9-class probabilities.
  - **AgeGenderSER** (W2V2 age-gender via ONNX Runtime): outputs
    `AgeGenderEstimate`.
- **`capForSER` binning**: snap audio length to one of `[2, 4, 8]` seconds
  before inference.
  - Reasons (any one is sufficient): models are trained on ≤10 s, ONNX
    CoreML EP compiles per-input-shape MLModels (unbounded shapes → OOM
    around 15 min of varied-length file analysis), latency scales linearly,
    longer-than-needed clips dilute the signal.
  - **Repeat-pad** under-bin clips, **center-crop** over-bin clips.
    Repeat-pad rather than zero-pad because acoustic models are mean-pool
    classifiers and silence has a non-trivial bias (W2V2 → `A≈0.61, V≈0.39`,
    emotion2vec → `sad≈99 %`). Looping the original samples keeps the
    mean-pool window representing the utterance, not a speech/silence
    blend. The micro-clicks at loop seams are spectrally negligible
    compared to the 25 % silence-mean shift zero-pad introduces.
  - On a literally-empty input, skip the model entirely (not "pad zeros
    and call it"). See "Empty-slice guard" below.
- **Speaker-active trim**:
  `trimToSpeakerActive(segmentAudio:asr:speakerID:timeline:vadTimeline:) -> SpeakerActiveTrim?`
  clips `segmentAudio` to portions of `[asr.start, asr.end]` where the
  diarizer placed `speakerID`, then AND-s with the cumulative VAD timeline.
  Concatenates kept intervals. Always opted-in by the streaming pipeline
  (both mic and file mode); hand-edit / re-evaluation skip (they run on
  isolated chunks with no cumulative VAD context).
  - Empty-intersection returns `nil` → caller falls back to whole-window
    scoring. Predictable degradation beats hallucination.
  - Possible future improvement (noted in code, not implemented): score
    per-interval and aggregate by sample-count-weighted mean instead of
    concatenating.

## Text SER

- `TextSER` protocol
  (`func classify(_ text: String) async throws -> PlutchikScore`).
- `SwitchingTextSER` actor holds both the WRIME-tuned text SER and Apple
  FoundationModelsSER, forwards to `currentBackend`.
  - The WRIME text SER is Japanese-only; auto-drops from `availableBackends`
    when session language is not `"ja"`.
  - On FoundationModels failure (often happens when app is backgrounded —
    iOS revokes ANE/GPU access), transparently fall through to the WRIME
    text SER for that row if available. Don't flip `preferredBackend`; the next
    foreground call returns to FM automatically.
  - Apple FM safety-guardrail decline: stamp
    `textBackend = SwitchingTextSER.foundationModelsGuardrailBackend`
    sentinel; UI renders a dedicated "Apple FM ✕" chip.
- **Lexicon bias** (`LexiconBias`): user-editable glossary of
  `(term, Plutchik label, weight ∈ [0,1])` entries. Applied post-classify
  in logit space:
  - Sum `weight × logitScale` (default `logitScale = 2.0`) per matched
    class.
  - Convert each probability to logit (`ln p`); 0-prob labels get
    `zeroLogitFloor = -4.0` (≈ ln 0.018) so they (a) survive into the
    post-bias dictionary (UI consistency — all 8 bars render), and (b)
    can be lifted by a max-weight glossary entry to a meaningful share
    (~10 %).
  - Add bias, re-softmax, return `(biased, matched)` where `matched` is
    the deduplicated list of term strings that fired.
- `classifyBiased(_:) -> (score, rawScore, matchedTerms)` exposes both the
  biased score AND the raw pre-bias score so the **reapply path** can
  replay the lexicon against the stored raw without re-running DeBERTa.

## Glossary

- `GlossaryStore` (`@Observable @MainActor`). Holds `isEnabled: Bool`,
  `entries: [LexiconBiasEntry]`, an `onChange` callback. Write-through JSON
  persistence to `Application Support/Glossary/glossary.json`.
- **Critical**: `didSet`-triggered `didMutate()` must defer persist +
  `onChange?()` via `Task { @MainActor in ... }`. `@Observable`'s
  synthesized `_modify` accessor runs `didSet` *inside* the exclusive-access
  scope; re-reading `entries` / `isEnabled` from within `didSet` (which
  `exportDocument()` does inside `persist`, and `currentLexicon` does
  inside the controller's `onChange`) trips Swift's runtime exclusivity
  check.
- JSON import/export via `.fileImporter` / `.fileExporter`
  (`GlossaryFileDocument: FileDocument`). Glossary is **app-level**, not
  bundled with `.xph` sessions.
- `CustomGlossarySheet`:
  - Top: Apply-Bias toggle (gated explanation in hint footer).
  - Middle: `ScrollView` (NOT `List` — `List`'s row reuse fights the
    focus-and-scroll pattern). `ScrollViewReader` wraps it. Each entry
    card has `.id(entry.id)`.
  - Per-card: emotion chip + delete button row, term TextField, emotion
    picker (`.fixedSize(horizontal: true)` so "Anticipation" doesn't
    wrap), weight stepper with monospaced-digit readout.
  - `@FocusState var focusedEntryID: UUID?` bound on each TextField.
    `onChange(of: focusedEntryID)` calls
    `proxy.scrollTo(newID, anchor: .center)` so the keyboard doesn't
    occlude.
  - Bottom action bar: full-width "Add Entry" button. After
    `store.add(newEntry)`, defer `focusedEntryID = newEntry.id` via
    `Task { @MainActor in ... }` so the ForEach has rendered the row
    before `@FocusState` binds.
  - Delete: capture id, defer mutation via
    `Task { @MainActor in store.entries.removeAll { $0.id == id } }`.
    Synchronous removal from `ForEach($collection)` while the binding
    into the deleted element is still live crashes.
  - Toolbar: Done (calls `onDismiss` + triggers
    `RecordingController.reapplyGlossaryBias()`) + ellipsis menu with
    Import/Export.
- **Reapply on Done**: `RecordingController.reapplyGlossaryBias()`
  iterates utterances. For each, `pipeline.rebiasAndFuse(_:)` reads
  `plutchikRaw` (or falls back to `plutchik` when no prior bias was
  stamped), re-applies the current lexicon, re-runs `fuser.fuse(...)`,
  and stamps `plutchik`, `lexiconBiasMatched`, and the new fused
  V/A/D/top-label via `UtteranceEstimate.withRebiased(...)`. Acoustic
  fields (`dimensional`, `acousticCategorical`) are preserved unchanged.
- Per-row chip on `UtteranceRow`: purple "Glossary N" badge (with the
  matched terms in the accessibility label) when
  `utterance.lexiconBiasMatched` is non-nil.

## Late fusion (`LateFusion`)

- V: weighted average of acoustic `dimensional.valence` and a Plutchik-
  derived valence. Text weight = `max(textWeightFloor, asrConfidence)`.
  Acoustic weight = constant `defaultAcousticWeight = 0.35`.
- A: same shape using arousal.
- D: acoustic only (text-only models don't estimate dominance).
- Top label: argmax over acoustic-categorical + Plutchik on a shared label
  subspace, with sink-bucket fallback (`other` / `unknown`) to text-side
  top Plutchik.
- `plutchikToValence` and `plutchikToArousal` are Russell-style polar
  mappings; coefficients are conservative defaults — tune from a
  calibration set and log to `docs/eval_log.md`.

## `UtteranceEstimate` (the canonical row)

Optional-rich Codable struct. Stored in session bundles. Fields:

- Identity: `id: UUID`, `speakerID: String`, `speakerName: String?`
  (stamped at export time from rename map)
- Range: `start`, `end: TimeInterval`
- Text: `transcript: String`, `asrConfidence: Float?`
- Acoustic: `dimensional: VADScore?`, `acousticCategorical: CategoricalEmotion?`,
  `ageGender: AgeGenderEstimate?`
- Text-SER: `plutchik: PlutchikScore?`, `plutchikRaw: PlutchikScore?`
  (pre-bias snapshot for reapply), `textBackend: String?`
- Flags: `speechBoost: Bool?`, `wasReevaluated: Bool?`,
  `wasHandEdited: Bool?`, `lexiconBiasMatched: [String]?`
- Fused: `fusedValence`, `fusedArousal`, `fusedDominance: Float?`,
  `fusedTopLabel: String?`
- Every new field is `Optional` with default `nil` so existing `.xph`
  bundles decode without a `formatVersion` bump.
- Provide `with*` builders for each field that the pipeline mutates after
  fusion (`withBounds`, `withTextBackend`, `withAgeGender`,
  `withPlutchikRaw`, `withLexiconBiasMatched`, `withRebiased`,
  `withSpeakerID`, `withSpeakerName`, `withSpeechBoost`, `withTranscript`).
  Every builder preserves *all* other fields.

## Session bundle (`SessionBundle` / `SessionDocument`)

Single binary-plist `.xph` file:

- `formatVersion: Int` (current = 1)
- `createdAt: Date`, `sourceKind: .microphone | .file`
- `audioFilename: String?`, `audio: Data?` (inlined for file-mode sessions
  only; mic-mode omits audio)
- `utterances: [UtteranceEstimate]`
- `speakerNames: [String: String]?` (user renames keyed by stored id)
- `speakerDatabase: Data?` (opaque diarizer DB blob — `[Speaker]` JSON
  with embeddings)
- `originalSnapshots: [UUID: UtteranceEstimate]?` (pre-first-reeval /
  pre-hand-edit snapshots for revert)
- `handEditChildren: [UUID: [UUID]]?` (sibling ids spawned by
  multi-sentence hand-edit splits)
- `diarizationTimeline: Data?`, `sessionSummary: Data?`,
  `transcriptionIssues: Data?`,
  `transcriptionIssueTranscriptSnapshots: [UUID: String]?`,
  `utteranceEmbeddings: [UUID: [Float]]?`,
  `utteranceObservationSegmentIDs: [UUID: UUID]?`

Diarizer / summarizer fields are `Data?` blobs — Export must not take an
upward dependency on those modules.

## RecordingController (the main MainActor coordinator)

`@Observable @MainActor`. Owns:

- Capture, transcriber, analysis pipeline, summarizer coordinator
- `utterances: [UtteranceEstimate]`, speaker tracker, cluster snapshot,
  diarization timeline
- Session state (phase, language, source mode, source URL, input pick)
- Glossary store (forwards `onChange` to `pipeline.setLexicon(...)`)
- Per-session caches (snapshots, hand-edit children, etc.)

Pump tasks spawned on `start()`:

1. **Raw-audio pump** — feeds `capturedAudio` from `AudioCapture`,
   advances `latestCapturedFileTime`. Back-pressures when
   `latestCapturedFileTime - lastDiarizedAudioTime > maxDiarizeLagSeconds`.
2. **Transcriber-feed pump** — feeds the streaming ASR. Separate from the
   analysis pump because ASR's view of audio is independent of the rolling
   buffer.
3. **Volatile-poll pump** — 5 Hz read of the analyzer's volatile hypothesis
   for UI preview.
4. **Continuous-diarize task** — fires on AUDIO time (not wall-clock) to
   handle the non-realtime file pump. 10 s sliding window, 2 s stride,
   first fire waits for a full 10 s window. Also runs the VAD on each
   window. Catch-up loop fires one diarize call per stride until all
   audio-since-last-fire is covered.
5. **Segment-analysis task** — drains finalized ASR segments. Each goes
   through `splitForProcessing` (sentence-aware split → per-sentence
   speaker assignment via cumulative timeline) → `processSegment` per
   split → `applySegmentResult`. Capped at `maxConcurrentSegments = 2`
   to bound IOSurface allocations under CoreML EP.
6. **File-end watcher** (file mode only) — awaits the feed pump and calls
   `stop()` when the file is exhausted.

## CRITICAL: file-mode `sliceForSegment` correctness

The diarize-loop self-trim races ahead of ASR's finalize cursor. When
diarize processes past t=180 s, the trim wipes audio below 180 s, and
any ASR segment finalized afterward for an earlier range gets an empty
slice — manifests as ONNX "empty audio" errors on the three acoustic
models.

The fix is **not** to delay ASR finalize (the audio is already gone). The
fix is:

```swift
private func sliceForSegment(_ asr: ASRSegment) async -> AudioChunk {
    if case .file = sourceMode, let url = playbackSourceURL {
        do {
            return try await Task.detached(priority: .userInitiated) {
                try Self.readAudioChunkForReevaluation(
                    fileURL: url, start: asr.start, end: asr.end
                )
            }.value
        } catch {
            AppLog.app.warning("file read failed, falling back to buffer: \(error)")
        }
    }
    return capturedAudio.slice(start: asr.start, end: asr.end)
}
```

The source file is on disk, authoritative, and re-reads are cheap relative
to SER cost. The rolling buffer is still used for diarize windows
(separate concern). Mic mode keeps using the buffer (no source file to
re-read from); the trim race exists there in theory but the realtime
capture rate can't outrun ASR by more than the diarize stride.

Belt-and-suspenders: `processSegment` skips the three acoustic models
entirely (single debug log) when the input slice is `samples.isEmpty`.

## Per-segment `processSegment` (the SER + fusion chokepoint)

```
1. Optional speaker-active trim (when applyDiarizerTrim opt-in).
   serAudio = trim?.audio ?? segmentAudio.
2. Skip acoustic entirely if serAudio.samples.isEmpty (with debug log).
3. async let dimensional / categorical / demographics / plutchik in parallel.
4. fuser.fuse(asr, speakerID, dimensional, acousticCategorical, plutchik).
5. Chain:
   .withTextBackend(textBackend)
   .withAgeGender(ageGender)
   .withPlutchikRaw(textResult.rawScore)
   .withLexiconBiasMatched(textResult.matchedTerms)
6. Propagate trim bounds to utterance.start/end (clamped to original ASR
   range so trim can never widen).
```

## UI structure

- Two-pane layout via `NavigationSplitView` (or equivalent). Left: control
  pane. Right: transcript pane. Sidebar collapses in portrait.
- **Settings card** — language picker, text-SER backend picker,
  speech-boost toggle, diarizer-sensitivity slider, Custom Glossary row.
  Compact picker layout in portrait, side-by-side in landscape (use
  `ViewThatFits`).
- **Pipeline card** — live indicators for each stage with throughput +
  status chips.
- **Transcript list** — `UtteranceRow` per row with: speaker chip (tap →
  reassign popover with "Teach diarizer" toggle), text-backend badge,
  modality-disagreement badge, hand-edited badge (with 2 s long-press
  revert), glossary chip, transcript text (0.5 s long-press → Edit
  Utterance sheet), play + re-evaluate buttons, expandable detail panel
  with V/A/D readouts, fused-label bars, per-modality probability bars,
  V/A scatter mini.
- **Affect panels** — `SERAggregateCard`, `FusionLegendCard`,
  `FusionVAScatterMini`, `StatisticsCard`, `SummaryCard`.
- **Speaker panels** — `SpeakerClusterCard` (PCA scatter),
  `SpeakerHeatmapCard`, `TurnTakingCard`, `ReactivityCard`,
  `AccommodationCohesionCard`, `InfluenceContagionCard`,
  `AffectiveSynchronyCard`, `SynchronyArcCard`.
- **Sheets** — `EditUtteranceSheet`, `CustomGlossarySheet`,
  `TranscriptionReviewSheet`, `SearchReplaceSheet`, `SessionSummarySheet`,
  `ShareSheet`. All use opaque `Color(uiColor: .systemBackground)`
  backdrop so inner `.glassEffect` cards don't refract the underlying
  content.
- **`EditUtteranceSheet`** — TextEditor + time-range stepper controls
  (custom −/+ buttons because system Stepper doesn't compress under
  keyboard) + play button + **Transcribe button** that calls
  `recorder.transcribeRange(start:end:)` to re-run offline ASR on the
  current range and replace the transcript field (TextEditor locks while
  in flight).

## Japanese-specific gotchas

- **Pitch accent vs. emotional prosody.** F0 in Japanese carries lexical
  info; cross-lingual SER can misattribute to arousal. Run emotion2vec+
  as a second opinion alongside audeering W2V2.
- **Public Japanese SER datasets are small and stylistically narrow**
  (JTES = acted/read, OGVC = game chat, STUDIES = acted dialogue,
  JVNV = script-generated). Treat zero-shot outputs as relative, not
  absolute.
- **Politeness register confounds text emotion.** WRIME-trained
  classifiers under-detect strong affect when the speaker uses 敬語.
  Document in user-facing reports.
- **Conversational vs. read-speech mismatch.** ReazonSpeech and Common
  Voice under-represent disfluencies (えーと, あの) and backchannels
  (うん, そう). Expect higher CER than published benchmarks suggest.

## Localization

- `Localizable.strings` for `en` and `ja` with full coverage of UI
  strings.
- All user-facing strings use `String(localized: "key")`.
- App ships with `ja` and `en` as primary locales.

## Always / Never

- **Always** evaluate ASR/SER on the project's held-out calibration set
  after any model swap; log to `docs/eval_log.md`.
- **Always** record `asr_confidence` and propagate into fusion weights.
- **Always** preserve audio + speaker + identity fields when builder
  helpers mutate single fields.
- **Never** commit `.wav`, `.m4a`, participant audio, or model weights
  to git.
- **Never** add a cloud provider, telemetry, or analytics. Period.
- **Never** fine-tune on a user's data silently. Fine-tuning is
  out-of-band scripts, not in-app actions.
- **Never** mock the database / models in eval tests. Use real fixtures
  (`Tests/Fixtures/`, git-lfs, ≤30 s consented clips).

## Implementation order (suggested)

1. SPM packages + XcodeGen project skeleton + `ModelStore` +
   `fetch_models.sh`.
2. Audio capture (mic + file), `RollingAudioBuffer`, `AudioChunk`.
3. ASR (start with `SpeechAnalyzerTranscriber`).
4. Diarization (FluidAudio adapter, then VAD, then trackers).
5. Acoustic SER (one model at a time — start with W2V2 dimensional).
6. Text SER (DeBERTa first, FoundationModels later).
7. Fusion + `UtteranceEstimate` + JSON export.
8. `RecordingController` + analysis pipeline + pump tasks.
9. Transcript UI + per-row detail.
10. Settings, pipeline, affect cards.
11. Session bundle save/load.
12. Hand-edit + re-evaluate paths.
13. Glossary + reapply + chip.
14. Speaker analysis panels (heatmap, cluster, etc.).
15. Summarizer (Apple FM + MLX Qwen).
16. Transcription review sheet.

Calibrate constants (`defaultAcousticWeight`, `logitScale`,
`zeroLogitFloor`, `clusteringThreshold` range, `serBinSeconds`,
`maxConcurrentSegments`, `maxDiarizeLagSeconds`) against real audio after
the pipeline runs end-to-end. The defaults above are reasonable starting
points but came from iteration — expect to retune.

---

**What this prompt cannot reproduce:**

- The exact tuning constants (came from measurement on real recordings).
- Specific bug fixes the implementing model won't re-encounter
  (route-change edge cases, IOSurface OOM under sustained CoreML EP
  load, USB-C polling on idle, etc.).
- Per-card layout polish and the exact set of analysis panels — those
  grew organically.
- The model conversion pipelines (W2V2 / emotion2vec / WRIME RoBERTa →
  Core ML / ONNX) — those need their own attention.

Use the prompt as a load-bearing spec; expect a 2–4 week round-trip of
iteration before the output feels like the original.
