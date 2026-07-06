# Feasibility: auto-summarize on session end + re-run on settings change

*2026-07-05. Assessment against the current `SummarizerCoordinator` /
`RecordingController` / `LLMSheetCoordinator` machinery. No code changed.*

## The ask, split into two features

- **A.** When a recording session ends, start summarization automatically in
  the background (no user tap, no sheet).
- **B.** When the user changes a setting that affects the summary, re-run the
  summarization with the new parameters.

Both are **feasible**, but "in the background" collides hard with an existing,
load-bearing constraint. The honest verdict: **feasible as a foreground,
opt-in, debounced auto-run; NOT feasible as true OS-background execution for
the MLX backends.**

## What already exists in our favor

- `RecordingController.summarizeSession()` → `SummarizerCoordinator.summarize()`
  is **already UI-independent**. It reads `parent.utterances`, runs, and writes
  the result to `lastSessionSummary` via `writeback`. The sheet only *presents*
  that cached value — nothing about generation requires the sheet to be open.
  So a headless trigger is essentially "call `summarizeSession()` without
  opening the sheet." (`SummarizerCoordinator.swift:296`)
- **Cancellation + supersede is already built.** `withInferenceGate` +
  `inferenceGenerationToken` + `userCancelledSummary()` mean a second run
  cleanly supersedes a first (`SummarizerCoordinator.swift:355`). Feature B's
  "re-run with new params" reuses this directly — it's what Regenerate does.
- **Backend/mode/enabled setters exist** and already do the right teardown:
  `setBackend` unloads the resident actor, `setMode` just persists
  (`SummarizerCoordinator.swift:120,166,187`).
- `stop()` ends with `phase = .analyzing` → `.idle` and a natural completion
  point to hook A onto (`RecordingController.swift:1721`, tail ~1790).

## The blocking constraint: "background" has two meanings

1. **Background thread, app foregrounded** — a detached `Task` that doesn't
   block the UI. ✅ Fully feasible; this is how summarize already runs.
2. **App actually suspended** (screen locked, app switched away —
   `scenePhase == .background`) — ❌ **not feasible for MLX**, and partly not
   for Apple FM either:
   - MLX's next Metal command buffer after backgrounding returns
     `kIOGPUCommandBufferCallbackErrorBackgroundExecutionNotPermitted`, an
     uncatchable C++ exception that **crashes the process**. This is documented
     on `ContentView` (`ContentView.swift:12-22`) and is *why*
     `LLMSheetBridge` **cancels in-flight summarize/review the instant the app
     backgrounds** (`ContentView.swift:420`).
   - The app declares only `UIBackgroundModes: [audio]` (`Info.plist:43`) — no
     `processing` mode, no `BGTaskScheduler`. Even with them, GPU/ANE work is
     revoked for backgrounded apps regardless of declared modes (noted in
     `project.yml:95`). SER already does a proactive CPU-rebuild dance to
     survive this; MLX has no such fallback.

**The cruel irony for Feature A:** the single most common user action right
after tapping Stop is to *lock the iPad or switch away* — which fires
`scenePhase == .background` and the existing cancel path would kill the
auto-started summary within milliseconds. So a naive "auto-start on stop" would
usually be cancelled before producing anything, or (without the cancel) would
crash.

## Feature A — feasibility per backend

| Backend | Auto-run on stop (app stays foreground) | If user backgrounds mid-run |
|---|---|---|
| **Apple FM** | ✅ Cheap (~no extra RAM), fast, best fit for auto-run | Cancels cleanly (CancellationError); safe, just no result |
| **LM Studio** | ✅ Network round-trip, no local RAM cost; but requires reachable server | URLSession cancels; safe |
| **Qwen / Llama (MLX)** | ⚠️ Works but heavy: releases the ~1.5 GB pipeline, loads ~4.3 GB weights, runs for **tens of seconds to minutes**, then rewarms the pipeline | Would crash if not cancelled; the existing cancel saves it but wastes the whole run |

**Memory/timing cost is the real tax.** `runSummarize` tears down the analysis
pipeline and rewarms it afterward (`SummarizerCoordinator.swift:901,923`).
Auto-firing MLX on *every* session end means every recording is followed by a
multi-minute, 4 GB, pipeline-thrashing operation the user didn't ask for — and
if they immediately start editing utterances or re-evaluating, they hit a
released/rewarming pipeline. Apple FM sidesteps most of this.

## Feature B — re-run on settings change

The settings that actually change summary output:

| Setting | Setter | Re-run trigger cost |
|---|---|---|
| Mode (trailing/heuristic/deep/meeting) | `setMode` | cheap to detect; full re-run |
| Backend | `setBackend` | already unloads actor; full re-run + possibly model load |
| Enabled | `setEnabled` | on→ re-run; off→ nothing |
| Keywords (heuristic/meeting boost) | `KeywordsCard` | affects `boostedUtteranceIDs` only in 2 of 4 modes |
| Speaker name overrides | rename flow | changes prompt; cheap to detect |
| Session/UI language | `setSessionLanguage` (idle-only) | changes response-language directive |
| LM Studio settings | `LMStudioSettings` | only when backend == lmStudio |

Feasible, but **re-running a 4 GB MLX pass on every slider nudge is
user-hostile**. Two hard requirements fall out:

1. **Debounce** — coalesce rapid changes (e.g. adjusting keywords) into one
   trailing re-run after the user settles (~2–3 s idle), not one per keystroke.
2. **Only auto-re-run when a summary already exists** for this session
   (`lastSessionSummary != nil`) — otherwise a settings tweak on a session the
   user never chose to summarize would spontaneously spin up the model.

Everything Feature B needs for *correctness* (supersede a running pass, flip
UI flags) is already present via the generation-token machinery.

## Recommended design (if we build it)

A single **`autoSummarize` user preference (default OFF)**, plus:

1. **Trigger A** — in `stop()`'s tail, once `phase == .idle` and
   `!utterances.isEmpty` and `summarizerEnabled && summarizerReady` and
   `autoSummarize`, fire `summarizeSession()` in a detached Task. **Gate MLX
   behind `scenePhase == .active`** — if the app isn't foreground, skip (don't
   start something the background-cancel will kill). Strongly consider
   **restricting auto-run to Apple FM / LM Studio** and leaving MLX
   manual-only, or at least warning that MLX auto-run is best-effort.
2. **Trigger B** — a debounced observer on the summary-affecting settings that
   calls `startSummarization` **only if `lastSessionSummary != nil`**, reusing
   the existing cancel-and-supersede path.
3. **A visible, cancellable indicator** — auto-run must be as interruptible as
   the manual path (it already is via `userCancelledSummary`), and the user
   must be able to see it's happening (reuse the `inferenceRunning` flag; a
   toolbar spinner or a banner). Silent 4 GB background work is a battery/heat
   surprise otherwise.
4. **Respect a "results are stale" marker instead of eager re-run** as a
   lighter-weight alternative to Trigger B: mark the cached summary stale on a
   relevant settings change and show a "Regenerate (settings changed)" affordance,
   rather than auto-spending tokens. This is the least surprising option and I'd
   recommend it as the default behavior, with true auto-re-run as an opt-in.

## Verdict

- **A (auto-run on end):** feasible foreground; **recommend Apple FM / LM
  Studio only** for auto-run, MLX manual or explicitly best-effort. Must gate on
  `scenePhase == .active` to avoid firing into the background-cancel.
- **B (re-run on settings change):** feasible via the existing supersede
  machinery, but **must be debounced and gated on an existing summary**;
  recommend a *stale-marker + one-tap regenerate* as the default over eager
  auto-re-run.
- **True OS-background summarization (screen locked / app switched):**
  **infeasible for MLX** (GPU revocation → documented crash), and the current
  architecture actively cancels it. Would require a fundamentally different
  execution path (CPU-only inference, or chunked BGProcessingTask work that MLX
  doesn't support) — out of scope for a settings-driven convenience feature.

The single biggest design decision is **which backends may auto-run**. Apple FM
is the clean fit: light, fast, cancels safely. MLX auto-run fights memory,
latency, and the background-crash guard simultaneously — supportable, but only
with eyes open.
