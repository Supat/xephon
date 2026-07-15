# Eval log

Running CER / WER / CCC / F1 numbers per model swap. Append, never rewrite.

Format:
```
## YYYY-MM-DD — short title
- Held-out set: <name>, N=<count>, total duration <hh:mm:ss>
- Models: <asr> | <ser_acoustic> | <ser_text>
- ASR: CER=<x>, WER=<y>
- SER (acoustic dim): valence CCC=<v>, arousal CCC=<a>, dominance CCC=<d>
- SER (acoustic cat): macro-F1=<f>
- SER (text 8-Plutchik): macro-F1=<f>
- Notes: <free text>
```

---

## (no entries yet)

## 2026-07-08 — Meeting (Experiment) mode: evidence citations + glossary + map-reduce

Shipped as a SEPARATE mode (`.meetingExperimental`); `.meeting`
stays the untouched baseline schema. The experiment: numbered rows,
one validated evidence row citation per stance (invalid citations
stripped at parse, logged), keyword-bank + glossary homophone
hints, and — past the single-pass cap — chronological map-reduce
(window 150, global row numbering) instead of TF-IDF sampling.

Field results so far (171-row Japanese vehicle-evaluation session,
Qwen3-8B-4bit):
- Array-valued evidence was catastrophic: the model ignored the
  "3-5 rows" soft bound and enumerated 20+ rows per claim, blowing
  the whole 4096-token output cap on numbers (two runs, both
  truncated mid-topics). Scalar evidence (one row per stance) is
  the shipped contract — structurally spray-proof.
- Even with scalar evidence, the user judged the baseline schema's
  output MORE COHERENT than the experiment's on this session —
  hence the mode split rather than replacement.

TODO before promoting the experiment: side-by-side on 2-3 held-out
meetings — topic recall (coverage), claim precision (spot-check
evidence rows), and a subjective coherence rating vs the baseline.
Also compare map-reduce vs baseline TF-IDF on a >250-row session.

## EvalForm synthetic harness (2026-07-17)

Model-independent eval for the A-1 auto-fill: seeded synthetic
sessions with by-construction truth (planted stated scores /
qualitative-only mentions / absent items / metadata / distractors),
scored on stated-score exact+MAE, false fills, comment recall,
evidence validity, metadata accuracy. Measures extraction fidelity,
NOT ride-judgment validity — the human ground-truth eval (research
doc §6) remains the final gate.

Run (LM Studio serving the candidate model):

    XEPHON_LMSTUDIO_URL=http://127.0.0.1:1234 \
    XEPHON_LMSTUDIO_MODEL=<served-model-id> \
    xcodebuild -scheme XephonEval \
      -destination 'platform=iOS Simulator,name=iPad Pro 13-inch (M5)' test \
      -only-testing:XephonEvalTests/EvalFormLiveEvalTests

Offline floor (deterministic tier only, pinned in UnitTests):
stated scores 2/2 exact, 0 false fills, evidence 2/2 — any live run
scoring below this floor is a regression, anything above it is what
the model adds. Paste per-model reports below.

| date | model | scores exact | false fills | comments | evidence | metadata |
|------|-------|--------------|-------------|----------|----------|----------|
| —    | —     | —            | —           | —        | —        | —        |

### First real-session trial — Qwen3-8B-4bit on-device (2026-07-15)

Tiguan session (.xph, ASR uncorrected). After the /no_think +
batch-envelope fix: full fill completed, one model load for the
whole run. Qualitative read (no ground truth yet):

- Policy guards held on real data: zero fabricated STATED scores
  (all four scores landed in the inferred channel, ?-marked), all
  inferred values on legal 0.125 steps, two items left honestly
  empty, evidence rows on every filled field + supplementary.
- Metadata extracted verbatim incl. ASR garbles (オリジナジナル,
  バネル上が) — stated-only working as designed, but 路面状況 got
  "5ヘルツ,15ヘルツ": a field filled with topically-adjacent but
  type-invalid content → template needs per-field allowed values.
- Item 12 comment reads self-contradictory; item 10 cites 13
  evidence rows (near the whole candidate set); ゴツゴツ has
  evidence rows with no filled field. Prompt/merge tightening
  candidates. Reviewer adjudication via evidence chips worked.
