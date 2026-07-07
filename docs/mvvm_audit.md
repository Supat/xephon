# Architecture audit: MVVM conformance

*2026-07-07. Full-codebase survey (view layer + model/coordinator layer +
Core modules). Inventory verified with file:line references at audit time.*

## Verdict

Xephon is **not classic MVVM and doesn't pretend to be** — it's the modern
SwiftUI "MV + coordinators" shape: `@Observable` model objects owned at the
App/ContentView level, views reading them directly via observation tracking,
with a *de facto* view-model layer that exists but isn't named or placed as
one. Measured against MVVM's actual goals (testable presentation logic,
views free of business logic, models free of UI), the codebase scores
**good on model purity, mixed on view purity, weak on layer naming/placement
and on view-input narrowing**.

Grade by MVVM criterion:

| Criterion | Grade | Evidence |
|---|---|---|
| Model layer free of UI | **A** | Zero SwiftUI imports across all Core/ modules; inference is actor-isolated |
| ViewModel-ish layer exists | **B** | 8 `@Observable` companions do the job — but 6 live under `Views/` |
| Views free of business logic | **C+** | ~10 analysis cards compute in `body`; 2 heavyweight coordinators are 100% algorithm |
| Views take narrow inputs | **D** | 22 views receive the whole `RecordingController` |
| Single source of truth | **B+** | One process-wide controller, versioned invalidation, memo discipline |
| Testability of presentation logic | **B−** | Companions are plain classes (testable) but entangled with `RecordingController` |

## What's genuinely good

1. **Core/ is immaculate.** Audio, ASR, Diarization, SER, Fusion, Summarizer,
   Export: no SwiftUI anywhere, typed errors, one-model-one-actor. This is
   the hardest MVVM property to retrofit and it's already true.
2. **A real (if unnamed) view-model layer exists.** `TranscriptFilterModel`
   (482 ln), `SearchReplaceCoordinator` (712 ln), `SERAggregateModel`,
   `SpeakerScatterModel`, `LLMSheetCoordinator`, `TranscriptionReviewCoordinator`,
   `AffectiveSynchronyViewModel` — all `@MainActor @Observable`, all owned via
   `@State`, all doing exactly what view models do (derive presentation state,
   memoize, hold sheet workflow state).
3. **Invalidation discipline.** `utterancesVersion` / `timelineVersion` keys +
   memo classes (`FilterMemo`, `MismatchMemo`, `StripRunsMemo`,
   `KeywordHighlightMemo`, …) give the app explicit, auditable recompute
   boundaries — better than most MVVM codebases manage.
4. **Communication patterns are consistent**: closure hooks
   (`onChange`, `requestAutoSummary`, `requestCancelInference`), UUID-token
   command bus (`MenuCommands`), unowned parent refs. No NotificationCenter
   soup between models.

## Findings (ranked by architectural weight)

### F1 — `RecordingController` is a 5,589-line god object *(the big one)*
Core file 2,350 lines + 7 extensions (HandEdit 718, Reevaluation 612,
SpeakerEditing 463, Playback 412, SessionPersistence 385, Undo 320,
Inputs 253). Eight distinct responsibility clusters. Every view-facing
concern flows through it; `SummarizerCoordinator` (1,337 ln) already
demonstrates the extraction pattern that works here (unowned parent,
thin forwarders). MVVM's "Model" this is not — it's Model + ViewModel +
service locator fused.

### F2 — 22 views take the whole `RecordingController`
No narrowing facade anywhere: cards deep-read
`recorder.utterances` / `recorder.keywords` / `recorder.summarizerInferenceRunning`
etc. Consequences: (a) every card is coupled to the god object's full
surface, (b) card previews/tests need a full controller, (c) observation
granularity is fine (thanks to `@Observable` per-property tracking) but
*compile-time* coupling is total.

### F3 — Analysis cards compute in `body`
TurnTakingCard, ReactivityCard, AccommodationCohesionCard, SynchronyArcCard,
InfluenceContagionCard, SpeakerBehaviorCard, AffectiveSynchronyCard call
`*.compute(utterances:)` directly in `body` — O(N)–O(N²) fusion analytics
re-run on every body re-eval of a visible card. `SERAggregateModel` and
`SpeakerScatterModel` show the house pattern for doing this right
(recompute in `.task` on version change / detached task); the other seven
cards never adopted it. This is both an MVVM violation *and* a live perf
hazard of exactly the kind just fixed twice (keyword highlights, strip runs).

### F4 — The view-model layer is misplaced and inconsistently named
Six `@Observable` companions live under `Views/`; one is named `*ViewModel`,
others `*Model` / `*Coordinator` / `*Digest`. `SearchReplaceCoordinator` is
712 lines of pure matching algorithm (three-pass search, fuzzy ranges,
staged edits) — the least "view" code in the app, filed under `Views/Sheets/`.
Placement doesn't break anything (single app target), but it obscures the
architecture the app actually has.

### F5 — Infrastructure leaks in views *(minor, localized)*
- `FileManager.default.temporaryDirectory` in 3 sheets (Markdown export
  staging) — belongs in an exporter helper.
- `UIDevice` orientation + `UIApplication.connectedScenes` introspection in
  `LMStudioServerSection` — belongs in a size-class/environment adapter.
- `@AppStorage` duplicated across `MainToolbar` / `TranscriptPaneView` /
  `XephonApp` for strip toggles — deliberate (documented bus-via-defaults),
  acceptable.
- `AppLog` calls inside two button actions in `UtteranceRow` — trivial.

### F6 — Domain state scattered in `@State`
`ContentView` holds selection, expansion, editing-snapshot state;
`KeywordsCard` holds add/rename workflow text. Defensible as "UI state,"
but selection (`selectedUtteranceID`) is consumed by four strips + list +
keyboard nav and is arguably session state. Low priority — the bindings
plumb it correctly today.

## Recommendations (cost-ordered; none urgent)

1. **Adopt the SERAggregateModel pattern for the seven compute-in-body
   analysis cards** *(low cost, real perf + purity win)*. One
   `@Observable` model per card (or one shared `SpeakerDynamicsModel`),
   recompute keyed on `utterancesVersion` via `.task(id:)`.
2. **Relocate + rename the view-model layer** *(mechanical)*:
   `Xephon/ViewModels/` (or `Presentation/`) for TranscriptFilterModel,
   SearchReplace/TranscriptionReview/LLMSheet coordinators, SERAggregateModel,
   SpeakerScatterModel. Naming: pick `*Model` or `*ViewModel`, one
   convention. Zero behavior change; makes the real architecture visible.
3. **Narrow view inputs opportunistically, not wholesale** *(medium)*.
   Don't retrofit 22 facades at once; instead, when a card is next touched,
   pass the slices it reads (`utterances`, a display-name closure, a version)
   instead of `recorder`. Cards already receiving narrow inputs
   (`UtteranceRow` post-refactors) show the direction.
4. **Continue carving `RecordingController` along the SummarizerCoordinator
   seam** *(highest cost, highest payoff)*. Natural next extractions, in
   order of independence: `PlaybackCoordinator` (playback + session-audio
   export accessors), `SpeakerEditingCoordinator`,
   `SessionPersistenceCoordinator`. Each keeps the unowned-parent + thin
   forwarders pattern so views don't churn.
5. **Sweep F5's three FileManager sites** into a `MarkdownExportStager`
   helper next time an export bug is touched.

## What NOT to do

- Don't rewrite toward textbook MVVM (protocol-abstracted view models,
  per-view VM classes, Combine pipelines). SwiftUI + `@Observable`
  observation tracking makes that ceremony net-negative, and the research
  goals (CLAUDE.md's three load-bearing capabilities) are better served by
  the current pragmatic shape with the fixes above.
- Don't move `TranscriptFilterModel`'s memo machinery into Core — it is
  presentation logic (what the *list shows*), correctly MainActor-bound.
- Don't split `UtteranceRow` (1,024 ln) for size alone — it's ~60% layout
  with the logic already extracted; splitting would scatter one visual
  component across files.
