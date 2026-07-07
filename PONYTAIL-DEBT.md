# Ponytail debt ledger

Deliberate shortcuts marked `ponytail:` in the codebase, plus deferred gaps
and tombstones of approaches that died on-device — so a deferral can't
quietly become permanent and a dead end can't be re-walked. Regenerate the
ledger rows with `grep -rnE '(#|//) ?ponytail:' .`; append deferrals and
tombstones as they're earned.

## Deferred — known gaps, revisit when they bite

### MLX backgrounding: two residual crash windows
`MLXCancellablePrefill` shrank the un-cancellable GPU window from the whole
prompt to one 128-token chunk, but (a) a chunk in flight at the exact
transition instant can still submit post-background (ms-scale), and
(b) model WEIGHT LOADING is un-cancellable GPU work with no chunk
boundaries at all. **upgrade if either bites:** cancel at `.inactive`
instead of `.background` and re-fire on return to `.active` — costs
spurious cancels on Control Center pulls, buys a ~1 s foreground runway.

### Session-open progress popup is indeterminate
`SessionFileBridge`'s overlay is a spinner; decode + `loadSession` expose
no progress granularity. **upgrade:** phase plumbing (read / decode /
restore) only if users ask "how long".

### SpeakerScatterModel hand-rolls power-iteration PCA (~80 lines)
Accelerate/LAPACK ships SVD natively, but the swap would change the
deterministic-seed + sign-stabilization behavior the scatter relies on.
**upgrade:** only with a fixture pinning projection stability across the
swap. (Repo audit 2026-07-07, flagged-not-scored.)

## Ledger

### Core/Summarizer/MLXLLMSummarizerCore.swift:131
Meeting-mode prompt cap. **ceiling:** flat 250 selected utterances shared by
both MLX families. **upgrade:** if real meetings routinely exceed ~250, add a
map-reduce meeting pass (mirror `summarizeDeep`) — do NOT just bump the
number; single-pass coverage degrades past ~10k prompt tokens.

### Core/Summarizer/AppleFMSummarizer.swift:63
Apple-FM meeting-mode prompt cap. **ceiling:** 35 utterances — empirical
limit that keeps text rows + reserved response inside the 3B model's window.
**upgrade:** only alongside handling for `exceededContextWindowSize`; raising
the number alone reintroduces the crash it dodges.

### Core/Summarizer/LMStudio/LMStudioSummarizer.swift:245
LM Studio meeting-mode prompt cap. **ceiling:** 400 text-only rows (~3–4×
cheaper per row than `.all`'s full-SER lines). **upgrade:** none needed until
a real meeting exceeds 400 distinct rows; the hard cap exists so a
pathological session can't blow a remote server's context window.

3 markers, 0 with no trigger.

## Tombstones — fuzzy-search stack (2026-07-07, audiorecord)

Lessons paid for with on-device rounds; don't re-walk these.

### Normalizer retranscription: equality guards can't terminate fold cycles
`JapaneseSearchNormalizer.tokens`'s nested pass was guarded on "the NFKC fold
changed something" — and froze the app on session open. For `、` the
tokenizer's latin-transcription attribute returns a variant form that NFKC
folds straight back: a 1-cycle the equality guard re-triggers on every level
(~1,800 frames on-device). **Rule:** recursion over tokenizer/Unicode
transforms must be depth-limited structurally (`allowRetranscription`, depth
1 — fold → kana → latin/ASCII never needs more), never terminated by "did
the output change" checks. Regression pinned in
`FuzzySearchFixtureTests.punctuationTerminates`.

### Memoized @Observable accessors: the memo-hit path must still read the
### observed state
`KeywordReviewModel.suspects()` keyed its memo on an `@ObservationIgnored`
generation counter. On a memo hit it read nothing observable, so any view
whose first render landed on a warm memo (the Keywords card always warms it
before the review sheet opens) never registered a dependency — taps mutated
state and the sheet never re-rendered. **Rule:** in an `@Observable` model, a
memoized accessor must read every observable input on BOTH hit and miss
paths; keying the memo on the observable value itself (here the
`rejectedIDs` set) does this for free and also dedups correctly across undo
cycles where a plain count collides.

### Session UndoManager: no bare registerUndo, ever
The recorder's `UndoManager` runs `groupsByEvent = false`; a bare
`registerUndo` outside a group throws `NSInternalInconsistencyException`
("must begin a group"). Registration is two-mode, mirroring
`RecordingController.registerUndoStep` / `apply`: user-initiated → wrap in
`beginUndoGrouping`/`endUndoGrouping` with `setActionName` inside the group;
during an undo/redo invocation (`isUndoing`/`isRedoing`) → register bare,
the manager is already inside its own group. Any new undoable surface must
follow this split or route through `registerUndoStep`.

## Tombstones — pipeline & lifecycle (2026-07-07, audiorecord)

### Per-arrival memo invalidation is O(n²) during file ingest
Memos keyed on `utterancesVersion`/`utterances.count` recompute on EVERY
arriving utterance; file mode pumps arrivals faster than realtime, so an
O(session) sweep in any always-materialized view (`.page` TabView keeps
sibling pages alive) becomes O(n²·k) on the MainActor and throttles the
pipeline itself. **Rule:** any O(session) UI sweep must gate on ingest and
serve its stale memo until the phase flips to idle. Decisive cheap test:
empty the keyword list (or equivalent input) and compare wall time.

### File read-in runs under `phase == .recording`, not `.analyzing`
`startFromFile` → `start()` — `.analyzing` is only the post-stop drain. A
busy-gate written against `isAnalyzing` silently misses the entire file
ingest; that exact miss shipped once. Gate on
`RecordingController.isFileIngestRunning` (file source + not idle).

### Library GPU loops can't be stopped by Task.cancel from outside
MLXLMCommon prefills the whole prompt inside `TokenIterator.init` with no
cancellation points; cancelling the wrapping Task does nothing until the
loop finishes — fatal when iOS revokes GPU access on backgrounding. The
scenePhase→cancel machinery was correct and still couldn't work. **Rule:**
when a library loops GPU submissions, run the loop yourself in cancellable
chunks (`MLXCancellablePrefill` pattern: prime the KV cache chunk-by-chunk
with `Task.checkCancellation`, hand the iterator the remainder).

### Blind subagent splices corrupt files that share substrings
An automated body-replacement matched `.frame(height: Self.height)` inside
a deeper-indented line (substring, not line match) and left four strip
files half-rewritten. **Rule:** structural refactors over near-identical
files need line-anchored (or brace-counted) edits, then a build before
moving on — the build caught it immediately.
