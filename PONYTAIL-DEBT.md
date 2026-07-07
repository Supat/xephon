# Ponytail debt ledger

Deliberate shortcuts marked `ponytail:` in the codebase, plus tombstones of
approaches that died on-device — so a deferral can't quietly become permanent
and a dead end can't be re-walked. Regenerate the ledger rows with
`grep -rnE '(#|//) ?ponytail:' .`; append tombstones as they're earned.

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
