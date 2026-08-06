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

Since 2026-08-06 the battery also covers the Tier 3a inferred
好き嫌い channel (positive/negative wording must land in band 6–8 /
2–4; measurement-only wording must leave the cell nil) and the
harness has two backends:

Run A — the PRODUCTION MLX path (MLXQwenSummarizer.generateRaw,
prompt-contract schema, KV prefix cache), on an Apple silicon Mac
via the Designed-for-iPad destination (no server, no device). Two
gotchas, both load-bearing: `TEST_RUNNER_` vars must be environment
variables ON the xcodebuild process (as trailing arguments they
become build settings and the suite silently skips), and the test
host is sandboxed — the model directory must live INSIDE the app
container (`cp -cRL` = APFS clonefile, instant and free):

    CONTAINER=~/Library/Containers/<xephon-container-uuid>/Data
    mkdir -p "$CONTAINER/Library/Application Support/eval-models"
    cp -cRL <qwen3-8b-4bit dir> \
      "$CONTAINER/Library/Application Support/eval-models/qwen3-8b-4bit"

    TEST_RUNNER_XEPHON_MLX_MODEL_DIR="$CONTAINER/Library/Application Support/eval-models/qwen3-8b-4bit" \
    xcodebuild -project Xephon.xcodeproj -scheme XephonEval \
      -destination 'platform=macOS,arch=arm64,variant=Designed for iPad' \
      test -only-testing:XephonEvalTests/EvalFormLiveEvalTests

Run B — any served model via LM Studio (cross-model bake-off,
native json_schema enforcement):

    TEST_RUNNER_XEPHON_LMSTUDIO_URL=http://127.0.0.1:1234 \
    TEST_RUNNER_XEPHON_LMSTUDIO_MODEL=<served-model-id> \
    xcodebuild -project Xephon.xcodeproj -scheme XephonEval \
      -destination 'platform=macOS,arch=arm64,variant=Designed for iPad' \
      test -only-testing:XephonEvalTests/EvalFormLiveEvalTests

MLX wins when both are set. Note Run A on a Mac shares the model
family and the exact prompt path with the iPad but not its silicon
— treat cross-run deltas as meaningful, absolute latencies as
Mac-only.

Offline floor (deterministic tier only, pinned in UnitTests):
stated scores 2/2 exact, 0 false fills, evidence 2/2 — any live run
scoring below this floor is a regression, anything above it is what
the model adds. Paste per-model reports below.

| date | model | scores exact | false fills | inferred in-band | false inferred | comments | evidence | metadata |
|------|-------|--------------|-------------|------------------|----------------|----------|----------|----------|
| —    | —     | —            | —           | —                | —              | —        | —        | —        |

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

## 2026-08-06 — back-to-back Summary → A1 fill: residency payoff confirmed

Same device/session as above. Meeting summarize (5096 tokens,
prefill 32.6 s, 945 tokens out, 94.2 s) → A1 eval fill started
immediately after.

- **Residency payoff observed**: the plugin batch began with NO
  "MLXQwenSummarizer loading" line — the model stayed resident from
  the summary run straight into the fill, saving the ~15 s reload.
  Headroom stable at ~11.2-11.6 GB throughout; both post-run
  decisions kept the model.
- A1 fill: 12 calls in ~175 s, zero parse-failure retries (so the
  retry-reuse path remains field-unobserved). Call pattern: six
  item extractions (~960-1800 prompt tokens, 71-176 out) each
  followed by a Tier 3a preference call (13 output tokens behind an
  8-11 s full prefill of the same rows). The six preference
  prefills total ~55-60 s of the 175 s — the concrete, field-
  measured cost the shared-prefix prompt layout would remove once
  the ground-truth harness can gate the prompt reorder.
- Cross-call prefix reuse is 3-5 tokens (chat-template header) as
  expected with the instructions-first prompts.
- Progress "first emission" logged on the summary and on every
  plugin call — delivery chain confirmed everywhere.
- Minor churn observed: the pipeline re-warmed (~2.5 s) after the
  summary and was immediately released again when the fill started.
  A short rewarm debounce (cancel if another LLM run starts within
  a few seconds) would remove it.

## 2026-08-06 — first live harness run: Qwen3-8B-4bit via production MLX path

Mac (Designed for iPad, sandboxed test host; model APFS-cloned into
the app container), battery of 4 specs incl. the new tier3a, 111.6 s
wall. Aggregates:

| date | model | scores exact | false fills | inferred in-band | false inferred | comments | evidence | metadata |
|------|-------|--------------|-------------|------------------|----------------|----------|----------|----------|
| 2026-08-06 | mlx:qwen3-8b-4bit (Mac) | 5/5 | 0 | 4/5 | 0 | 4/11 | 8/8 | 4/4 (+1 false) |

Read:
- Anti-hallucination floor is clean under the live model: zero false
  score fills, zero false comments, zero false inferred preferences
  (measurement-only rows correctly left nil), evidence 8/8 valid.
- Tier 3a direction is good: 4/5 in band incl. the positive case,
  zero missed (the always-null failure mode is gone), one
  out-of-band (mixed-2's negative-wording item).
- Comments 4/11 is depressed by a REAL defect the harness caught:
  on two items the model ECHOED THE APPENDED JSON SCHEMA as its
  output (both attempts unparseable → item degraded to
  deterministic-only, no comment). The production prompt-contract
  appendix does not forbid schema echo.
- The verbatim-retry KV reuse worked in production form: retry
  reused 827/828 tokens, prefill 0.04 s vs 2.35 s cold.

### Candidate: schema-echo fix (same day)

One appendix line added to the shared prompt contract ("Do NOT copy
or repeat the schema itself — output only the data object"), both
copies (coordinator + harness) in lockstep. Same battery, same model:

| run | scores exact | false fills | inferred in-band | false inferred | comments | evidence | metadata | wall |
|-----|--------------|-------------|------------------|----------------|----------|----------|----------|------|
| baseline | 5/5 | 0 | 4/5 | 0 | 4/11 | 8/8 | 4/4 +1 false | 111.6 s |
| schema-echo fix | 5/5 | 0 | 3/5 | 0 | 9/11 | 11/11 | 4/4 +3 false | 75.3 s |

- Targeted defect eliminated: zero schema echoes, zero parse
  failures, zero retries (the wall-clock drop is the two retry
  calls plus the wasted echo decodes). Comment recall 4/11 → 9/11
  and evidence 11/11 follow directly — echoed items had degraded
  to deterministic-only.
- Single-run caveats (small deltas, treat as noise until repeated):
  one tier3a inferred flipped in-band → missed (nil), and tier3a
  grew 2 false metadata fields. Greedy decode is not bit-stable
  across runs (Metal reduction order); per the measurement
  discipline, repeat/interleave before reading anything into
  ±1-count changes. The parse-failure elimination is causal and
  mechanism-understood; the rest needs N>1.
