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

## 2026-08-06 — turbo branch field confirmation (residency + prefix cache + progress)

Device: Kiyosumi (iPad Pro M4-class, 15.14 GB reported, iOS 26.6).
Model: Qwen3-8B-4bit. Timing observations, not accuracy numbers.

- Meeting summarize, 171 utterances / 5096 prompt tokens: prefill
  30.6 s (~167 tok/s), decode 954 tokens at 17.5 → 16.4 tok/s
  (mild thermal droop), 88.9 s total. A1 plugin call, 1120 prompt
  tokens: prefill 8.48 s (~132 tok/s), 74 output tokens, 12.2 s
  total.
- Residency policy's first field decision: "11583 MB available,
  floor 4096 MB → keep resident"; the riskiest path — pipeline
  re-warm WITH 4.6 GB of weights held — completed cleanly in ~2.5 s
  (mlmodelc compiles cached: wespeaker 5.4 s cold at launch vs
  49 ms on re-warm), speaker DB snapshot/restore intact, no memory
  kill.
- Prefix cache engaged: "reusing 3 of 1120 prompt tokens" on a
  cross-item plugin call — tiny LCP is expected with the reverted
  instructions-first prompts; the big reuse case (verbatim
  parse-failure retry) did not occur in this run.
- Caveat on the residency floor: os_proc_available_memory() on this
  device reads ~15000 MB idle and 16078 MB after pipeline release —
  it measures against the entitled Jetsam limit, not physical RAM,
  and can exceed it. On 16 GB-class hardware the 4096 MB floor is
  therefore effectively always-keep; the operative guards there are
  the memory-pressure eviction and the backgrounding eviction. The
  floor still does its intended job on smaller-RAM devices.
- Not yet exercised: a second run against the resident model (the
  log ends before one) — expect no "MLXQwenSummarizer loading" line
  and the run starting straight at prompt prep.

Follow-up (same day, second field run): A1 extraction quality
confirmed back to pre-turbo behavior after the prompt revert
(fca98ac), and the live progress line confirmed rendering after the
action-bar mount (404c76d). Still unobserved in the field: the
second-consecutive-run residency payoff (no model reload) and a
large prefix-cache reuse on a verbatim retry.
