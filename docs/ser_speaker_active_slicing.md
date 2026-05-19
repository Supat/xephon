# Tight-slice SER — implementation plan

**Status:** to-implement.
**Goal:** SER should only see the portion of an ASR window where the
assigned speaker is actually voicing — not the leading/trailing
silence, the inter-word pauses, or another speaker's audio bleeding
into the window. Applies to acoustic SER (W2V2 V/A/D + emotion2vec
9-class) and W2V2 age/gender. Text SER is unaffected — it already
operates on the transcript string, which is what was said.

Related context: [`acoustic_ser_bias.md`](acoustic_ser_bias.md) (why
the encoders are sensitive to non-speech),
[`ser_padding_postmortem.md`](ser_padding_postmortem.md) (the prior
silence-padding incident).

---

## Decisions to lock first

- **Slicing source.** Use the cumulative diarizer timeline (already
  maintained on `RecordingController` at ~0.5 Hz) AND-ed with the
  existing Silero VAD output. Diarizer alone is leaky at segment
  edges; VAD alone can't separate speakers in overlap. Both are
  already computed; this is plumbing, not new ML.
- **Inference shape.** Run acoustic SER per interval, then average
  per-class probabilities / V/A/D weighted by interval sample count.
  Concatenating non-contiguous speech is cheaper but Mel features at
  the join discontinuity confuse both encoders. Per-interval is
  ~1.5–2× slower per utterance; measure on M4 before optimizing.
- **Minimum interval.** Drop intervals shorter than 0.3 s; if the
  *aggregate* across all kept intervals is < 0.5 s, fall back to
  whole window and stamp `serSlicingMode = .wholeWindowShort`. Don't
  run W2V2 on sub-encoder-window clips — the receptive field needs
  ~25 ms × 5–10 hops minimum and the prosodic features need an
  entire syllable.
- **Live + file mode parity.** Apply in both. Initially framed as
  live-only, but the same dilution hits file-mode analysis; gating
  to live would create two pipelines diverging from day one.

## Schema changes (`UtteranceEstimate`)

- `serActiveDuration: TimeInterval?` — total per-speaker speech-time
  actually fed to acoustic SER.
- `serActiveRatio: Float?` — `serActiveDuration / (end - start)`.
  Drives the per-row inspector readout.
- `serSlicingMode: SERSlicingMode?` — one of `.wholeWindow`
  (fallback / legacy), `.wholeWindowShort`, `.diarizerTrimmed`,
  `.diarizerAndVAD`. Default nil for sessions saved before this
  change → treat as `.wholeWindow` on load.
- Session bundle codec round-trip + JSON export schema update +
  `docs/output_schema.md`.

## Code surface

- **`Core/Audio/`** — new `SpeakerActiveIntervals.swift`: pure
  function `(diarizerTimeline, vadTimeline, utterance) ->
  [Range<TimeInterval>]`. Returns the AND-intersection sliced to
  the speaker's segments. Stateless, unit-testable.
- **`Core/SER/Acoustic/`** — extend the acoustic SER entry point to
  accept `[Range<TimeInterval>]` instead of a single `(start, end)`.
  Internally:
  - Slice the audio buffer per interval.
  - Run W2V2 + emotion2vec on each kept interval.
  - Aggregate: V/A/D → sample-count weighted mean; categorical →
    sample-count weighted mean of softmax then re-normalize.
  - Return aggregate + `serActiveDuration`.
- **`Core/SER/Acoustic/AgeGender.swift`** — same treatment for the
  W2V2 age/gender head. Age regression weighted mean, gender vote
  weighted by sample count.
- **`Recording/AnalysisPipeline.swift`** — at utterance finalization,
  compute the intervals once, hand them to the acoustic SER actor
  and the age-gender actor. Text SER path unchanged.
- **`Recording/RecordingController+Reevaluation.swift`** — the
  row-level re-eval path needs to read the *current* diarizer + VAD
  timelines (not the snapshot from when the row was first finalized)
  so a re-eval after the diarizer has caught up uses tighter slices.
- **Live-mode timing guard.** Utterance finalization must wait for
  the diarizer's continuous-tick to cover the utterance's `[start,
  end]` range. Currently the diarizer tail is ~1 s behind capture.
  Either:
  - (a) Add a 1.5 s grace delay before kicking off SER. Simpler.
  - (b) Finalize SER with whatever diarizer state exists and
    re-eval automatically when the timeline catches up. Avoids any
    user-visible "SER not run yet" gap.

  Pick (a) for v1.

## UX surfacing

- **Per-row inspector** (`UtteranceRow` detail section): add a
  `metaLine("SER fed", "2.4s / 4.1s · diarizer+VAD")` row when
  `serActiveRatio < 1.0`. Uses the existing `metaLine` helper.
- **Pipeline diagnostics banner** (`PipelineDiagnosticsBanner`):
  when ≥ 20% of utterances in a session fell back to `.wholeWindow`,
  emit a one-line warning. Routes through the existing
  `recorder.pipelineDiagnostics` stream.
- **No toggle.** Default-on. Off-switch invites the "is my eval
  comparable to last week's?" trap. If the change regresses, revert;
  don't ship a feature flag.

## Validation

- **EvalTests** — extend `Tests/EvalTests/` with a sliced-vs-whole
  comparison harness over the JTES + STUDIES fixtures. Numbers: CCC
  for V/A/D, F1 (macro) for the 9-class. Both per-utterance and
  per-speaker aggregate. Write results into `docs/eval_log.md` keyed
  by build hash.
- **Sanity microbench** — instrument the per-utterance SER call
  site, log `(window_duration, active_duration, latency_ms)` to
  `os.Logger` under a new `.ser-slicing` subsystem. Pull on-device
  traces for the first 50 utterances of a real session and look for:
  high fallback rate (silence-dominated room), tight active ratio
  (single-speaker close-mic) — both should validate the slicer's
  behavior matches expectation.

## Risks / open questions

- **W2V2 minimum input.** Need to confirm the audeering model's
  minimum input length empirically. ONNX Runtime won't error on a
  200 ms clip but the embedding will be dominated by edge effects.
  If we see a CCC regression on short utterances, raise the
  per-interval floor from 0.3 s to 0.5 s.
- **Multi-speaker overlap.** When the diarizer assigns two speakers
  to overlapping segments inside one utterance window, we trim to
  the stored speaker only. If that strips most of the audio, falls
  back to whole window with the new "you may want to reassign"
  mismatch flag the timeline already raises.
- **Acted vs spontaneous gap.** JTES and STUDIES are acted; OGVC is
  game chat. Tight slicing might help acted (clean turns) less than
  spontaneous (long pauses). Worth running a comparison on whichever
  spontaneous Japanese set has labels — note in `docs/eval_log.md`
  if the gap is large.
- **First-pass vs re-eval divergence.** The (a)-grace-delay approach
  above means a row's SER is computed once with the diarizer state
  at `finalizationTime + 1.5s`. If the diarizer later changes its
  mind about who spoke during that window, the SER doesn't auto-
  rerun. The row's mismatch glyph already surfaces this case —
  accept it for v1, document the boundary.

## Sequencing

1. Schema + bundle codec + JSON export (so old sessions still load).
2. `SpeakerActiveIntervals` helper + unit tests.
3. Acoustic SER + AgeGender adapters to take `[Range]`.
4. Pipeline + reeval wiring + the 1.5 s grace.
5. Per-row inspector readout + diagnostics warning.
6. EvalTests harness + `docs/eval_log.md` row.

(1)–(2) are isolated and mergeable independently. (3) is the only
step that touches model code paths — biggest risk surface, do under
careful eval gating.
