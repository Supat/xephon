# EvalForm Tier 3a — inferred 好き嫌い (plan)

Status: implemented on `plugin-arch` alongside this document.
Scope: the A1 Eval plugin (`VehiclePerformanceMetricsA_1_Straight`).

## Problem

The 好き嫌い (1–9 preference) column fills only from explicit
statements: the deterministic parser needs 好き/嫌い + `digit+点`
in one utterance, and the LLM's `likeDislike` field is restricted
to explicitly spoken ratings. Field sessions rarely contain either
— evaluators say 「これは好みじゃないな」「うん、いいね」 without a
number — so the column is almost always a gap.

## Approach

Add an **inferred preference channel** mirroring the existing
stated-vs-inferred split for the strength score
(`strengthScore` / `strengthScoreInferred`): a new
`inferredLikeDislike` field the LLM may fill from qualitative
wording ONLY, landing in a separate draft field
(`likeDislikeInferred`) that is marked 要確認 everywhere it
renders and never counts as a sheet entry.

Explicitly **out of scope** (Tier 3b, gated on the ground-truth
eval): estimating preference from the acoustic/text affect stream
(valence, Plutchik). Also out of scope: any trained
valence→preference mapping (CLAUDE.md forbids trained cross-modal
heads without in-domain calibration data).

## Rules (mirroring `inferredScore`, plus one of its own)

1. Filled only when no stated preference exists (deterministic
   capture or explicit spoken value) — stated always suppresses
   inferred.
2. Must be an integer within the template's preference scale;
   out-of-range suggestions are dropped at merge, not clamped.
3. **Never derived from the strength score.** The prompt says so
   outright: stronger/weaker than baseline is a measurement, not
   a preference — a bigger difference is not automatically
   disliked. Only wording that expresses the evaluator's own
   liking/disliking (好み・気に入った・嫌だ…) qualifies.
   *Revised 2026-07-31 (see Revision below): evaluative wording
   about the behaviour now also qualifies.*
4. Coverage still reports the item's 好き嫌い as unfilled —
   qualified as 「好き嫌い（推定のみ・要確認）」, the same way an
   inferred-only 評点 reads.

## Touch list

- `EvalFormExtractor`: `itemSchemaJSON` + STRICT RULES prompt
  line + `ItemWire.inferredLikeDislike` (lenient Codable) + merge
  branch (stated wins → wire stated fills → wire inferred fills
  `likeDislikeInferred`).
- `EvalFormDraft.ItemResult`: `likeDislikeInferred: Int?` —
  optional-tolerant addition within payload v2 (missing key
  decodes nil; no version bump), same rule as
  `supplementaryEvidenceRows`.
- `EvalFormCoverage`: `ItemGaps.inferredPreferencePresent`,
  mentioned-check includes the new field, gap phrase gains the
  推定のみ qualifier.
- `EvalFormMarkdown` / `EvalFormCSV`: render 「（推定 N —
  要確認）」 / new `likeDislikeInferred` column.
- `EvalFormCard` / `EvalFormScaleViews`: orange `♥N?` badge and
  orange marker on the preference axis — the exact coding the
  strength axis uses for suggestions.
- Tests: merge gating (accepted / suppressed-by-stated /
  out-of-range), wire parsing leniency, prompt rule presence,
  coverage phrase, both renderers.
- Docs: `evalform_pipeline.md` §4c/§4e/§8/§9 sync.

## Revision 2026-07-31 — loosened evidence bar

First field runs produced the value too rarely: the original rule
accepted only explicit liking/disliking words, and its STRICT /
NEVER / otherwise-null framing made the small quantized models
default to null. Loosened, prompt-only (merge gates unchanged):

- Evidence now includes **evaluative wording about the
  behaviour** (良い・悪い・気になる・不快・うるさい・改善した・
  収まりが悪い…), not just preference words — an evaluator
  saying the settling is bad IS expressing an impression.
- Added a coarse **anchor guide** (2-3 clearly negative, 4
  mildly negative, 5 mixed, 6 mildly positive, 7-8 clearly
  positive) so the model has somewhere to land besides null.
- Added a **prefer-mild-over-null nudge** for borderline cases —
  directly counters small-model conservatism.
- Null is still correct for pure measurement talk, and the
  never-from-the-strength-score guard is unchanged.

Tradeoff, accepted deliberately: recall up, precision down. The
channel stays quarantined (separate field, 要確認 marking,
reviewer confirmation), so a wrong mild guess costs one reviewer
glance; an always-empty column costs the feature. The ground-truth
eval should score this channel's false-positive rate.

## Revision 2026-07-31 (2) — split into its own call

Field observation after the loosening: strength-score outputs
shifted too. Cause: all fields of the item call are one
autoregressive generation over one prompt — any prompt edit
conditions every field, and the prefer-mild-over-null nudge in
particular plausibly bleeds into `inferredScore` on a small
model (plus ordinary temperature-0.2 run-to-run variance; no
ground truth existed to rank the two behaviours).

Fix: the inferred-preference instructions moved OUT of the item
extraction into a dedicated per-item call
(`preferencePrompt` / `preferenceSchemaJSON` / `PreferenceWire`,
run by `EvalFormRunner.inferPreference`). The item prompt,
schema, and `ItemWire` reverted byte-for-byte to their
pre-Tier-3a shape, so the strength-score path is exactly what it
was before this feature existed. The new call runs only when the
merge produced no stated preference (the stated-suppresses-
inferred gate, now at the call site), sees the same rows, carries
the preference scale only (no strength scale / rubric / POLARITY
— isolation cuts both ways), and adds a guard that spoken
±scores in the rows are measurements, never convertible.
Single attempt, range-gated, failure degrades to an empty cell.

Cost: one extra generate per mentioned item without a stated
preference — roughly doubles the item-pass wall time
(prefill-dominated over the same rows; decode is ~a dozen
tokens). Accepted in exchange for a strength-score path that no
future preference-prompt tuning can perturb.

## Validation

Unit tests only for now. The synthetic harness scores stated
recovery and is untouched; the inferred channel joins the
existing caveat (docs/evalform_plugin.md): unvalidated until the
ground-truth eval, reviewer confirmation required.
