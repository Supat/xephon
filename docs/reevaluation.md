# Per-utterance re-evaluation

Each utterance row in a file-analysis session has a re-evaluate button
below its playback button. Tapping it re-runs the offline ASR + SER +
fusion pipeline on that single utterance's audio, and replaces the
row's content with the new result in place. The original utterance's
identity (`id`, `start`, `end`, `speakerID`, `speechBoost`) is
preserved so list position, selection state, and Save/Load identity
all hold across the re-evaluation.

This document covers what the feature does, why each piece is shaped
the way it is, and the bugs we ran into building it.

## Why it exists

Streaming ASR has to commit a result before it knows what's coming
next. SpeechAnalyzer's volatile-stabilization timer fires at the
boundary between accumulated and not-yet-stable text — which often
falls *inside* a phoneme or just before a sentence-final particle.
The streaming pass therefore tends to:

- clip the first phoneme of an utterance (it ran the timer before
  enough audio had arrived);
- truncate the sentence-final particle (the timer fired before the
  trailing `ね` / `よ` / `か` stabilized);
- lock in a transcript that, with hindsight + neighbouring context,
  is obviously wrong but irreversible.

The user can fix any one of those with the re-evaluate button: it
re-runs the offline ASR on the same audio with a small front-side
pad so SpeechAnalyzer sees the lead-in it missed, then refreshes
the row's text, SER, and fusion outputs.

## Pipeline

```
Tap row's re-evaluate button
    ↓
RecordingController.reevaluate(_:)
    ↓
read audio from file:
    [utterance.start − 500 ms, utterance.end]
    (front-only padding, clamped to file bounds)
    ↓
AnalysisPipeline.reevaluate(audio, originalStart, originalEnd, speakerID, onVolatileText)
    ↓
SpeechAnalyzerTranscriber.transcribe(audio, onVolatileText:)
    .volatileResults enabled
    volatile hypotheses stream to caller's onVolatileText
    finals collected as ASRSegment[]
    if no finals emit: use last volatile as fallback
    ↓
trim to "every full sentence":
    walk concatenated tokens, find LAST token whose text ends with
    a sentence terminator → cut text + audio at that token's end
    ↓
processSegment(asr, segmentAudio, fallbackSpeakerID)
    runs acoustic dimensional SER, acoustic categorical SER, text SER
    in parallel, fuses via LateFusion
    ↓
applyReevaluation(utteranceID, fresh)
    merge: keep id/start/end/speaker/speechBoost from original,
           take transcript/asrConfidence/SER/fusion from fresh
    stamp wasReevaluated = true
    bump utterancesVersion (invalidates ContentView filter memo)
    re-fold conversationSummary
```

## Front-only padding

500 ms before `utterance.start`, **nothing** after `utterance.end`.

Three iterations got us here:

1. **500 ms / 500 ms** (original). Worked for most cases but left
   some sentence-final particles still clipped.
2. **1 s / 1 s** then **2 s / 2 s** (briefly). Wider context helped
   ASR but the back pad routinely picked up a neighbour's opening
   phoneme, which the acoustic SER then mixed into the original
   utterance's V/A/D.
3. **500 ms / 0 ms** (current). The sentence-aware trim
   (`AnalysisPipeline.allFullSentences`) drops anything past the
   last terminator anyway, so back-padding is wasted bytes; only
   the front side meaningfully helps ASR recover a clipped leading
   phoneme.

The padding constant lives at
`RecordingController.reevaluationPaddingSec`.

## Sentence-aware trim

Offline ASR over `[start − 500 ms, end]` happily transcribes the
leading phoneme of a neighbouring utterance if the pad crossed a
real silence. SpeechAnalyzer doesn't know about utterance
boundaries — it just emits whatever it heard.

We trim the result to *every committed sentence* — everything up to
and including the LAST sentence terminator (`。 ！ ？ ． . ! ?`).
Unterminated trailing fragments are dropped.

**Both text and audio are trimmed in lockstep** using per-token
audio-time anchors:

- `SpeechAttributes.tokens(in:)` extracts per-run `audioTimeRange`
  anchors when `attributeOptions: [.audioTimeRange]` is set on the
  `SpeechTranscriber`.
- `lastSentenceEndTokenIndex` finds the last token whose text ends
  with a terminator.
- That token's `end` (in buffer-local seconds, i.e. relative to
  sample 0 of the audio chunk) is the cut point for both `audio`
  and the concatenated `text`.

When tokens are missing (a transcriber that doesn't expose them),
the code falls back to text-only trim; audio passes through whole.

## Volatile preview during re-evaluation

The pipeline panel's ASR row already has a slot for italic rolling
preview text (used during live recording). During re-evaluation we
populate the same slot:

- `SpeechTranscriber` is constructed with
  `reportingOptions: [.volatileResults]`.
- `collectResults` distinguishes via `result.isFinal`:
  - `isFinal == false` → call the `onVolatileText` handler with
    the rolling hypothesis;
  - `isFinal == true` → append to the segment list.
- The handler is `@Sendable @MainActor (String) -> Void` and is
  awaited inside the transcriber actor, so by the time
  `transcribe(_:onVolatileText:)` returns no callback is in flight
  — `RecordingController.reevaluate`'s `defer { volatileText = "" }`
  doesn't race a late firing.

The `PipelineCard` is unchanged: it binds to `recorder.volatileText`
either way, and the source (streaming poll task vs re-eval callback)
is invisible to the view.

## Volatile fallback for short clips

`SpeechAnalyzer` exhibits a subtle quirk on short offline clips:
when `.volatileResults` is on and the audio doesn't contain a
stable sentence boundary, `finalizeAndFinishThroughEndOfInput()`
closes the stream **without** promoting the pending volatile to a
final. The segment list comes back empty.

Symptom: ASR volatile preview animates correctly, but the utterance
never updates because `pipeline.reevaluate` returned nil for empty
segments.

Fix: `collectResults` stashes the last volatile it saw. If no
finals arrived but a volatile did, the fallback path synthesizes
one segment from that volatile (including its tokens, so the
sentence-aware trim still works).

Two log lines mark the path:

```
offline ASR collected: N segment(s), lastVolatile=(set|nil)
offline ASR: no finals emitted; using last volatile as result (X chars)
```

## Identity preservation

`applyReevaluation` constructs the replacement utterance by:

| field | source |
| --- | --- |
| `id` | original — keeps list position, selection, `.xph` round-trip identity |
| `speakerID` | original — diarization isn't re-run; the streaming pass's verdict stands |
| `start`, `end` | original — the re-eval is about content, not boundary |
| `speechBoost` | original — captures the capture-time state, doesn't change |
| `transcript`, `asrConfidence` | fresh from offline ASR |
| `dimensional`, `acousticCategorical`, `plutchik`, `textBackend` | fresh from SER |
| `fusedValence`, `fusedArousal`, `fusedDominance`, `fusedTopLabel` | fresh from fusion |
| `wasReevaluated` | always `true` after a successful re-eval |

The `wasReevaluated` flag lives on `UtteranceEstimate` itself —
which means it rides along the existing serialization paths:

- `.xph` session bundle: utterances are embedded, so the green
  marker survives Save/Load with no schema version bump;
- JSON export: `JSONExporter` encodes `[UtteranceEstimate]`, so
  external tooling sees `"wasReevaluated": true` next to the
  affect data.

`conversationSummary` is re-folded from scratch after the
replacement — `ConversationSummary` is an incremental fold with no
"replace" path, and N is small enough that re-folding is cheap.

## State machine

Two enums in `UtteranceRow` drive what each row's button column
shows.

`PlaybackAvailability`:

| state | meaning |
| --- | --- |
| `unavailable` | mic session — no source file; button hidden |
| `disabled` | session is recording/analyzing/re-evaluating — grey button |
| `idle` | tap to start playback — blue |
| `playing` | this row is playing — stop icon, blue |

`ReevaluateAvailability`:

| state | meaning |
| --- | --- |
| `unavailable` | mic session — button hidden |
| `running` | this row's re-evaluation is in flight — spinner |
| `completed` | `wasReevaluated == true` — green icon, still tappable |
| `disabled` | session busy elsewhere — grey, untappable |
| `idle` | never re-evaluated this session — blue |

`.completed` is checked **before** the busy `.disabled` branches so
the green marker doesn't flicker away while another row's
re-evaluation runs. The controller's own guards
(`reevaluatingUtteranceID == nil`, `phase == .idle`) keep the
no-op safety net intact for any tap that arrives during the busy
window.

## Cross-row gating

While one re-evaluation runs, every other row's playback and
re-evaluate buttons go disabled:

- Audio reads compete for the same file handle.
- SER and ASR serialize on the pipeline's actor anyway.
- Two re-evaluations in flight would interleave their
  `volatileText` writes into the pipeline panel's preview slot,
  producing visual garbage.

This is enforced in `ContentView.playbackAvailability` and
`reevaluateAvailability` via `recorder.reevaluatingUtteranceID !=
nil`, with a defense-in-depth guard in
`RecordingController.reevaluate` itself.

## Pipeline panel reaction

The ASR / Acoustic SER / Text SER / Fusion rows in `PipelineCard`
flip from `.ready` to `.active(0)` while re-evaluation runs,
mirroring how a live streaming segment lights them up. Capture,
Diarizer, and Export deliberately stay idle:

- Capture isn't running — audio is read from a file directly.
- Diarizer isn't re-run — the original speakerID is preserved.
- Export isn't touched — the `lastExportAt` latch is unchanged.

`PipelineCard.isReevaluating` reads
`recorder.reevaluatingUtteranceID != nil` and that signal is
plumbed into the four relevant state computed properties.

## The filter-memo cache bug

`ContentView` memoizes `filteredIndexedUtterances` and
`displayedSummary` behind a `FilterDepsKey` to avoid re-filtering
on every body re-run. Original key fields: normalized search
query, label filter, speaker filter, utterance count.

Re-evaluation mutates `utterances[index] = merged` in place. Count
stays the same. **Memo cache hits, returns the pre-re-eval
snapshot, List renders stale rows.** Pipeline / fusion run fine
end-to-end and produce correct values that the view layer never
reads.

Fix: `RecordingController` now exposes a `utterancesVersion`
counter that bumps in `applyReevaluation` (and `loadSession` as
defense-in-depth). `FilterDepsKey` includes the version, so any
in-place mutation invalidates the cache.

The general lesson: a memo dependency key has to capture *every*
input that can change the derived value. Count is sufficient when
the only mutation pattern is append-only; the moment any in-place
update is possible, content needs a fingerprint too. A monotonic
counter on the source actor is the cheapest fingerprint.

## Files

| file | role |
| --- | --- |
| `Xephon/Views/UtteranceRow.swift` | renders the re-evaluate button column, owns the two availability enums |
| `Xephon/ContentView.swift` | maps recorder state → row state, wires `onReevaluate`, holds the filter memo |
| `Xephon/RecordingController.swift` | reads audio from disk, calls pipeline, applies the merge, owns `reevaluatingUtteranceID` / `utterancesVersion` |
| `Xephon/AnalysisPipeline.swift` | offline ASR call, sentence-aware trim, processSegment forward |
| `Core/ASR/SpeechAnalyzerTranscriber.swift` | offline transcribe with `.volatileResults` + callback + volatile-fallback |
| `Core/Fusion/UtteranceEstimate.swift` | carries `wasReevaluated: Bool?` through serialization |
