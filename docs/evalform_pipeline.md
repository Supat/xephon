# A1 Eval pipeline — how a session becomes a filled evaluation sheet

Reference for `Plugins/VehiclePerformanceMetricsA_1_Straight/` as built (2026-07-15).
Companions: `docs/eval_form_autofill_research.md` (design rationale),
`docs/plugin_architecture.md` (the host it runs on),
`docs/eval_log.md` (trial results + harness usage).

## 0. Design principles (govern every stage)

1. **Stated-only for sheet entries.** A score/preference field is
   filled only with values the evaluator *spoke*. Model suggestions
   live in a separate, visually-marked channel (`inferred`).
2. **Null-first.** Emptiness is cheap at every layer (schema,
   merge, render) so nothing is filled to please.
3. **Evidence on everything.** Every filled field carries 1-based
   utterance row numbers; the UI reveals them in the transcript on tap.
4. **Deterministic beats generative.** Regex-captured values are
   ground truth; the model cannot override them, only fill gaps.
5. **Degrade loudly, never sink the run.** A failed pass yields a
   per-item conflict note or an empty field, plus a log line with
   the raw model output tail.
6. **The reviewer signs off.** Output is a draft; the 未検出 list
   states the remaining manual work explicitly.

## 1. Inputs

- **Session**: the live `SessionSnapshot` — `utterances`
  (chronological `UtteranceEstimate`s; row numbers below are always
  1-based indexes into this array), `speakerNames`,
  `utterancesVersion` (staleness stamp).
- **Template pack** (`EvalFormTemplate`): the active sheet
  definition. Embedded default = A-1 (直線路走行専用); an imported
  JSON pack replaces it (persisted cross-session in plugin
  storage). Fields used by the pipeline:
  - `items[]`: id, printed number, titles, definition,
    `referenceRoads`, `vocabulary` (the surface forms that mark a
    mention — onomatopoeia + variants).
  - `strengthScale`: min/max/step (−1…+1, 0.125) +
    `anchors` (the sheet's ※ rubric: magnitude → meaning).
  - `preferenceScale`: 1…9.
  - `metadataFields` + `metadataCues` (header extraction).
- **Inference backend**: whatever the summarizer settings select
  (Apple FM / Qwen3-MLX / Llama-MLX / LM Studio), reached through
  the plugin host's `InferenceService`.

## 2. Activation-time side effects

On plugin activation (app launch / toggle-on):
- The ACTIVE template's `vocabulary` is seeded into the keyword
  bank under a group named after the template (idempotent; never
  duplicates user entries). This lights up the app's existing
  keyword highlighting, homophone review, and timeline strips for
  the evaluation vocabulary — independent of any fill run.
- The stored draft (if the loaded `.xph` carries one) is restored
  version-aware (§8).

## 3. Run gates and the batch envelope

"Fill Form" (`EvalFormModel.run`) checks, in order:
1. not already running;
2. session non-empty;
3. inference available (summarizer enabled + backend ready).

Then the entire fill executes inside **one inference batch**
(`InferenceService.withBatch`): on MLX backends the host releases
the analysis pipeline (SER/diarizer, ~1.5 GB) once, loads the LLM
once, and schedules a single unload + pipeline re-warm + speaker-DB
restore when the batch ends. Without the envelope each of the 6–8
generate calls paid a full load/unload cycle (first field trial:
most of the ~5-minute wall clock). Each call still passes the
coordinator's inference gate, serializing against built-in
summarize/review runs.

The runner itself (`EvalFormRunner.fill`) is headless — the same
function serves the plugin page and the eval harness. It throws
only `CancellationError`; everything else degrades per-item.

## 4. Per-item extraction (the core loop)

For each template item, in order:

### 4a. Candidate rows

`candidateRowNumbers`: a row is a **mention** iff its transcript
(NFKC-folded, lowercased) contains any of the item's vocabulary
surfaces as a substring. No fuzzy matching here — ASR-garbled
onomatopoeia are the keyword-review feature's job, upstream of the
fill.

`contextExpandedRows` then adds **±2 neighbour rows** around every
mention: ASR splits Japanese judgments across segments, so the
verdict — often the spoken score — lands in the row after the
onomatopoeia. Attribution rules keep the window from bleeding
between items:
- a context row attaches to the item whose mention is NEAREST;
- equidistant ties attach to every tied item (the merge conflict
  machinery covers the rare double-capture);
- another item's own mention row is at distance 0 of that item, so
  it can never be captured as context here.

Empty expanded set (= item never mentioned) → the item is recorded
empty, **no model call**, loop continues.

All expanded rows are also unioned into `claimedRows` for the
supplementary pass (§5).

### 4b. Deterministic tier (`SpokenScoreParser`)

Runs over the expanded rows. Cannot hallucinate by construction.

**Strength scores** — grammar, applied to the NFKC-folded text:
- `プラマイ|プラスマイナス|±` + `ゼロ|0` → 0 (checked first so the
  generic pattern can't eat its suffix).
- REQUIRED sign marker + decimal:
  `(マイナス|ﾏｲﾅｽ|プラス|ﾌﾟﾗｽ|[-−+ー])\s*number`. The long-vowel
  mark `ー` counts as minus only directly before a digit (a common
  ASR rendering of spoken マイナス). **Unsigned numbers are never
  scores** — speeds, road names, counts are everywhere in drive
  talk.
- **Quantization gate**: the value must equal one of
  `strengthScale.allowedValues` (±0.0005). "マイナス0.3" is
  conversation, not a sheet entry.
- Multiple captures per row/item all record (revision handling in
  §4e).

**Preference (好き嫌い)** — deliberately conservative: one
utterance must contain BOTH a preference word (好き/嫌い) AND
`digit+点` within the scale range. First hit wins.

### 4c. LLM tier — prompt

One generate call per item (`extractionPrompt`), containing:
- role line + sheet name; item number/title/definition;
- scale description (relative to 基準仕様);
- **POLARITY rule**: negative = stronger than baseline
  (強い・増えた side), positive = weaker (弱い・減った・なくなった);
  the sign must agree with the described direction;
- **magnitude rubric**: the template's ※ anchors verbatim
  (±0.125 = なんとなく違う（5:5）… ±1.0 = まったく違う) — the
  same calibration a human reads on the sheet;
- STRICT RULES: statedScore/likeDislike only for explicitly spoken
  values, else null; inferredScore only when nothing stated AND
  the wording clearly implies a direction, on a legal step;
  comment = short Japanese distillation or null; evidenceRows =
  only numbers that appear below;
- **pre-verified stated scores** from §4b injected as ground truth
  the model must not contradict;
- the numbered rows (`[n] speaker: text`, session-global numbers,
  cap 60 — a note explains neighbours are included and that the
  verdict often follows the mention);
- "Return ONLY the JSON object; first character `{`".

Schema (`itemSchemaJSON`, all fields required, all nullable):
`statedScore: number|null, inferredScore: number|null,
likeDislike: integer|null, comment: string|null,
evidenceRows: [integer]`.

### 4d. Generation, backend enforcement, parsing

`inference.generate(prompt:schemaJSON:maxOutputTokens: 512)` routes
through `SummarizerCoordinator.pluginGenerate`:
- **LM Studio + structured-output ON**: the schema goes as a native
  `response_format` json_schema (constrained server-side); the
  prompt is sent as-is.
- **Apple FM / MLX / LM Studio without structured output**: the
  schema is appended to the prompt as a contract
  ("Return ONLY a valid JSON object conforming to…").
- **MLX specifics** (inside the actors' `generateRaw`): the family
  turn directives are always appended (`/no_think` on Qwen —
  without it Qwen3 burns the whole token budget inside a think
  block; first-trial root cause), and `stripThinkBlocks` runs on
  the output so a brace inside a think block can't poison slicing.

Parsing (`parseItemResponse` → `parseJSONObject`):
1. slice from the first `{` to the last `}` and strict-decode;
2. on failure, **truncation repair** over the tail from the first
   `{`: walk the text tracking in-string/escape state and an open
   bracket stack → close an unterminated string, trim trailing
   whitespace, drop a dangling comma / null-complete a dangling
   `key:`, close brackets in reverse. Balanced-but-invalid input
   returns nil (repair recovers truncation, it doesn't invent
   structure);
3. decoding is **lenient at the scalar level** (custom Codable):
   numbers-as-strings ("−0.25" incl. full-width signs), "null"
   strings, integer-typed doubles, string row numbers. Semantic
   gates still apply at merge — leniency of syntax, not meaning.

**Failure handling**: parse-nil or a thrown generate → log
(with the raw output's last 240 chars on parse failures) and retry
ONCE. Still nothing → the item proceeds with deterministic
findings only, plus the llmFailed conflict note.

### 4e. Merge policy (`EvalFormExtractor.merge`)

Precedence and gates, in order:
1. **Stated strength**: deterministic captures win. One distinct
   value → filled, its rows become evidence. Multiple distinct
   values → chronologically LAST wins + a conflict note carrying
   the full revision trail (`-0.5@[5] → -0.25@[9]`).
2. Model `statedScore` is accepted only when the deterministic
   pass found nothing AND the value is a legal step; if it
   contradicts a deterministic capture, the deterministic value
   stands and the disagreement is flagged.
3. **Inferred**: accepted only when no stated value exists, and
   only on a legal step. Stored in `strengthScoreInferred` — a
   separate field end-to-end.
4. **Preference**: deterministic wins; model value fills the gap
   when within range.
5. **Comment**: model's, nil when empty.
6. **Evidence**: deterministic rows ∪ model rows filtered to the
   candidate set (invented numbers are silently dropped), sorted.
7. **Polarity sanity check** (inferred only; stated is the
   evaluator's own words): count direction words in the cited
   rows' text (強/増え/大きく/悪化 vs 弱/減っ/なくなっ/小さく/改善).
   A clear contradiction — opposing hits with zero supporting —
   appends 「推定スコアの極性要確認…」. Never a silent sign flip
   (the heuristic can't see negation like 強くない).
8. **Reference-road cross-check** (soft flag, never a filter):
   the callout segmentation assigns each row a road; when ≥2 of
   the item's cited rows have assignments and the majority lie
   OUTSIDE the item's 評価路 column, append 「根拠発話の多くが
   評価路以外の区間…」. Skipped entirely when the session has no
   callouts; unassigned rows count neither way; ties pass.

## 5. Supplementary pass (補足コメント)

- Candidates: rows **no item claimed** (complement of all expanded
  sets), with a substantive-length floor (≥8 chars — drops うん /
  そう backchannels), chronological, cap 40.
- One generate call: distill 2–4 short Japanese sentences of
  observations worth recording, or null; evidenceRows from the
  shown numbers. Schema mirrors the item pass.
- Evidence filtered to the candidate set. Any failure → no
  supplementary comment (additive field; never sinks the run).

## 6. Header metadata pass

- Candidates: session-opening rows (first 20) ∪ any row hitting a
  template `metadataCue` (車両/アブソーバー/仕様/天気/気温/路面/…),
  sorted, cap 30 — so a spec restated at an absorber swap or a
  mid-drive weather change is visible.
- One generate call: JSON object with exactly the template's
  `metadataFields` as keys; value = the stated value verbatim or
  null; NEVER guess; when restated, the LATEST statement wins.
- Values are extracted as-said — ASR garbles included (that is the
  stated-only policy working; hand-correcting the source rows is
  the fix, not model cleanup).

## 7. Timing profile (Qwen3-8B-4bit, M4 iPad, field runs)

One model load ~15 s (batch envelope), prefill 4–9 s per call
(~600–1300 prompt tokens), decode ~20 tok/s. A six-item session
with supplementary + metadata ≈ 8 calls ≈ 1½–2 minutes total. A
parse-failure retry adds one call.

## 8. Draft persistence and migration

The result is an `EvalFormDraft` (templateID, the input
`utterancesVersion` as a staleness stamp, per-item results,
supplementary comment + evidence, metadata map, reviewedItemIDs),
JSON-encoded into the plugin's `.xph` payload (payload v2).
Restore is version-aware: v2 decodes; v1 (pre-review-state)
migrates with empty `reviewedItemIDs`; payloads from a NEWER
plugin build are left untouched in the bundle (no guess-decode).
Optional fields added within v2 (e.g. `supplementaryEvidenceRows`)
are missing-key tolerant — same rule `SessionDocument` uses.

## 9. Review UI (the card) and exports

Card, top to bottom: header metadata → per-item rows → 補足コメント
→ 未検出 list → status lines → the sheet's ※ rubric footnote
(always visible). Each item row: reviewed-confirmed toggle
(payload v2 state), title, numeric badge, then the sheet's two
axes as printed — 強い −1…+1 弱い with 0.125 minor ticks, and
嫌い 1…9 好き. Marker coding: solid tint = stated; orange (+`?`
badge) = inferred suggestion; empty track = undetected. Evidence
chips (tap = jump + highlight the transcript row) wrap in an adaptive grid and collapse
past 6 chips behind a "+N" expander; when the draft carries road
provenance the chips group under small road labels and the item
gains a 走行路 line listing the distinct full road names its
evidence came from (canonical labels, first-appearance order).
Conflicts render orange.

Exports (markdown + CSV, through the root file picker) mirror the
card: detections with evidence numbers, inferred values explicitly
marked 要確認, and the 未検出（要手動記入） section — missing
header fields plus per-item gap phrases (評点（推定のみ・要確認） /
評点 / 好き嫌い / コメント / 言及なし) — computed by the same
`EvalFormCoverage` helper as the card, so report and UI cannot
disagree. CSV uses RFC-4180 quoting and filterable `undetected`
rows.

## 10. Road sections (adjacent deterministic feature)

"Detect road sections" matches the template's road-callout lexicon
and proposes callout-to-callout segments to the Sections page
(repeat visits numbered; existing titles never duplicated). The
lexicon is the pack's `roadCallouts` when present — per road, a
canonical label (the section title) plus the surfaces that count
as its callout, so partial forms are accepted exactly where
they're unambiguous (A-1 ships "E3" for E3路 and "スペイン" for
スペイン歩道; the single-letter roads also accept the bare letter,
which the matcher restricts to STANDALONE occurrences — neither
neighbour a Latin alphanumeric — so 「Dに入ります」 opens D路 while
4WD/HD never can) and course-specific naming
(5ヘルツ…) is a pack edit. Without `roadCallouts` it falls back to
the exact labels derived from `referenceRoads`. Matching folds
width and case on both sides; different surfaces of one road are
one road (no spurious segment splits). It assumes live callouts
("次、F路60キロ") — retrospective references (一番最初の5ヘルツ…)
still mis-segment, a known limitation.

The same segmentation feeds the fill: each cited row's road is
frozen into the draft (`roadByRow`) at fill time, rendered as
grouped evidence in the card and exports ("D路 [16] [36] ・ F路
[7]"; CSV `evidenceRoads` column), summarized per item as the
走行路 line (card + markdown, via `roadsForItem`), and drives the
merge's reference-road cross-check (§4e·8).

## 11. Degradation table

| Failure | Behaviour |
|---|---|
| item never mentioned | empty item, no model call, 言及なし in 未検出 |
| LLM output unparseable twice | deterministic values only + llmFailed note + raw-tail log |
| output truncated at token cap | repair recovers completed fields; rest as above |
| model invents evidence rows | dropped at merge (candidate-set filter) |
| model contradicts regex capture | deterministic wins + conflict note |
| inferred sign contradicts evidence wording | 極性要確認 flag |
| supplementary / metadata call fails | field(s) empty, run completes |
| user cancels | run aborts, prior draft untouched |
| session edited after fill | staleness banner (version stamp mismatch) |

## 12. Verification status

- Deterministic tier + merge policy + parsers: pinned by unit
  fixtures (`EvalFormPluginTests`).
- Whole-pipeline extraction fidelity: synthetic known-answer
  harness (`EvalFormSynthetic` + `EvalFormLiveEvalTests`,
  `XEPHON_LMSTUDIO_URL`-gated) with a pinned offline floor —
  see docs/eval_log.md for the run command and results table.
- Judgment validity (do inferred magnitudes match what a human
  would put on the sheet): NOT yet validated — requires the
  human ground-truth eval (research doc §6).
