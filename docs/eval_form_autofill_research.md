# LLM-based auto-fill of the vehicle sensory-evaluation form — research & design options

**Status:** research only — nothing here is implemented.
**Date:** 2026-07-14
**Scope note:** the target form (車両評価性能指標Ａ－１, 直線路走行専用) is marked
関係者外秘. This document describes its *structure* for engineering purposes and
must stay in this private repository; no form specifics were sent to external
services during this research.

---

## 1. The task

Given a Xephon session (diarized, per-utterance Japanese transcript of a test
drive — e.g. the Tiguan sensory-comment sessions), automatically produce a
filled evaluation form. The A-1 form's fields, from the PDF:

**Header metadata** — date/time, vehicle, evaluation item, evaluator name,
absorber specs (基準SA仕様 vs 評価SA仕様), evaluation position (運転席/助手席),
weather (fine/cloud/rain), temperature, road surface (dry/half wet/wet).

**Six ride-feel items**, each with *three* outputs:

| # | Item | Perceptual definition | Reference roads |
|---|------|----------------------|-----------------|
| 10 | フラット感 | sprung-mass motion balance (あおり/ピッチ/ロール) | E3 60, D 40, G 60 |
| 11 | ヒョコヒョコ | under-damped bobbing without stroke feel, 3–8 Hz | E3 60, F 60 |
| 12 | ブルブル | unsprung shake from damping deficit, 8–15 Hz | F 60, G 60, 段差路 |
| 13 | ゴツゴツ | input texture, 15–30 Hz | D 40, G 60 |
| 13 | ビリビリ | input texture, ≥30 Hz | H 60 |
| 14 | ハーシュネス | input-G magnitude, edge feel (shock/noise/damping) | 段差路, スペイン歩道 |

Per item: **(a)** relative strength vs baseline on −1…+1 in 0.125 steps
(0.125 = arguable, 0.25 = 30% of people notice, 0.5 = 70%, 0.75 = 100%,
1.0 = totally different), **(b)** like/dislike 1–9, **(c)** free comment.
Plus a session-level 補足コメント block.

Two properties make this much more tractable than generic form-filling:

1. **The item names are the vocabulary.** Evaluators literally say
   ヒョコヒョコ/ブルブル/ゴツゴツ/ビリビリ while driving — the form rows are
   Japanese onomatopoeia that appear verbatim (or as near-homophone ASR
   errors) in the transcript. Xephon's keyword bank + fuzzy/phonetic matcher
   was built for exactly this vocabulary class.
2. **Scores are frequently spoken.** "ヒョコヒョコはマイナス0.25かな" is a
   direct utterance→field mapping, not an inference. The road protocol
   ("次、F路60キロ") gives segmentation anchors that map to the form's
   reference-road column.

## 2. Architecture options considered

### Option A — prompt-engineered general LLM in the existing summarizer stack (baseline; recommended)

Add an **evaluation-form mode** alongside the existing `SummarizeMode` cases,
reusing the Meeting (Experiment) machinery almost wholesale:

- **Numbered text-only rows with evidence citations.** Meeting (Experiment)
  already makes the model cite the single row number supporting each claim and
  validates citations against the row list in the parser (invented numbers get
  stripped). For a form-filler this is not a nicety — every score and comment
  written into the form must be traceable to an utterance for human review.
- **Schema-constrained output.** All three backends already have a structured
  path: Apple FM via `@Generable` guided generation (schema guaranteed by the
  OS), LM Studio via `response_format` JSON schema (`LMStudioSchemas` exists),
  MLX via prompt contract + tolerant parser today — upgradeable to true
  token-masked constrained decoding (see §4).
- **Glossary folding.** The meeting-experimental prompt already folds the
  keyword bank + custom glossary as "domain terms that may appear
  mis-transcribed" — for this task the glossary is the six onomatopoeia, road
  names (D路/E3路/F路/G路/H路/段差路/スペイン歩道), and absorber spec terms.
- **Map-reduce.** Long sessions exceed single-pass caps; the meeting deep path
  (per-window partial output with globally numbered rows → merge) transfers
  directly. A form even merges more naturally than minutes: per-item fields
  union, conflicts flagged.

Cost: one new mode + schema + prompts + parser + a form renderer. No new
model, no new dependency, works fully on-device.

### Option B — specialized extraction LLM (worth a bake-off, not a replacement)

The "specialized LLM" that actually matches this task is **not** an
"evaluation" model but a **structured-extraction** model:

- **NuExtract** (NuMind) — models fine-tuned solely for
  "document + JSON template → filled JSON". NuExtract 2.0 ships in 2B/4B/8B
  on Qwen-VL bases (multilingual, multimodal; 4B/8B under Qwen research
  license, 2B commercially usable; GGUF available for local runtimes).
  **NuExtract 3** (May 2026) is a 4B Apache-2.0 model that reportedly beats
  Qwen3.5-9B on NuMind's extraction benchmark (0.651 vs 0.479) and fits
  consumer hardware. Fit: the "fill this template" framing is literally its
  training objective, and template-following removes most prompt engineering.
  Concerns: (a) niche Japanese automotive onomatopoeia are far from its
  training distribution — unvalidated; (b) it extracts *stated* information
  and does not reason well about *implied* scores ("これはちょっと気になるね"
  → 0.25?) — though §3 argues implied scores should be flagged for humans
  anyway, which turns this weakness into policy alignment; (c) no native
  evidence-row mechanism (the template can demand a `sourceRow` field, but
  it's not trained for citation discipline); (d) served via LM Studio /
  llama.cpp on the Mac, not on-iPad (MLX port would be extra work).
- **GLiNER2 / GoLLIE-class extractors** — schema-driven IE systems; stronger
  for entity/relation extraction than for judgment-bearing fields with
  numeric scales; less relevant here.

Verdict: run NuExtract 3 (and 2.0-4B-GGUF) as a *candidate backend* in the
same eval harness as Option A's models. If its field-level accuracy on real
sessions matches Qwen3-32B-class prompting at 4B size, it becomes the
recommended LM Studio backend for this mode. Do not build the feature around
it.

### Option C — "LLM-as-judge" evaluator models (investigated, rejected)

Prometheus 2, JudgeLM, Auto-J, etc. are open models specialized in
*evaluating LLM outputs against rubrics* (direct assessment / pairwise
ranking). Despite the word "evaluation", the task is inverted here: we are
not scoring generated text, we are **extracting a human's already-made
evaluation** from speech. A judge model's rubric-scoring skill does not
transfer to onomatopoeia-anchored Japanese extraction, and Prometheus 2's
bases (Mistral 7B / 8x7B) are weak in Japanese. One narrow reuse *is* worth
noting for later: a judge model (or a second general model in judge mode)
could serve as the **verifier** in a generate→verify pipeline, checking each
filled field against its cited row — same adversarial-verification shape the
transcription reviewer already uses.

### Option D — hybrid deterministic + LLM (recommended refinement on top of A)

Xephon already has deterministic machinery that removes the highest-risk work
from the LLM:

1. **Item tagging without the LLM.** Seed the keyword bank with the six
   onomatopoeia (color-tagged). The existing normalized/phonetic matcher then
   marks candidate rows per item, and `KeywordReviewModel`'s homophone tier
   catches ASR mis-hearings of them. The LLM receives per-item *pre-filtered
   row sets* instead of hunting through the whole transcript — smaller
   prompts, fewer misses, and the keyword timeline strip becomes a visual
   audit of coverage.
2. **Numeric score capture by rule, not generation.** Spoken scores follow a
   tiny grammar (マイナス/プラス × 0.125|0.25|0.5|0.75|1 / 1–9 for 好き嫌い).
   A regex/lexicon pass over rows near an item mention captures stated scores
   with effectively zero hallucination risk; the LLM's job reduces to
   *attribution* (which item does this score belong to) and *adjudication*
   (two conflicting scores for one item → keep both, flag).
3. **Road segmentation → the Sections feature.** Road callouts ("F路60キロ
   入ります") segment the session; sections map to the form's reference-road
   column, which both scopes each item's candidate rows and fills the header's
   road context. Detection can be keyword-driven with LLM fallback.

The LLM then does what only it can: distill per-item free comments from the
candidate rows, resolve ambiguous attribution, and draft the 補足コメント —
each output carrying evidence row numbers.

## 3. The policy that matters more than the model: stated vs. inferred

The single biggest failure mode in the closest commercial analog — ambient
clinical scribes (AWS HealthScribe, Abridge, DAX) — is plausible-but-wrong
content in structured fields; studies of GPT-4-based note generation found
~24 errors per case, while structured/objective sections consistently score
highest (~87% accuracy). The transferable lessons:

- **Extract, don't estimate — by default.** A score field is filled only when
  the evaluator *stated* a score. Qualitative-only mentions ("ゴツゴツは
  ちょっと強いね") produce a filled *comment* and an **empty score flagged
  for review** — never a silently invented number. An optional "suggest"
  mode may propose a score from qualitative language, but visually marked as
  inferred (own JSON field, own UI treatment), mirroring how the form's own
  0.125-step semantics encode confidence.
- **Null is a first-class answer.** Items never discussed stay empty. The
  schema must make emptiness cheap (`nullable` everywhere) so the model isn't
  pressured to fill.
- **Every filled field carries evidence rows.** Already proven in the meeting
  pipeline; the review UI can jump from field → utterance → audio playback
  (row-level playback already exists in the keyword-review sheet).
- **Human sign-off is the product.** The deliverable is a *pre-filled draft*
  the evaluator reviews in minutes instead of transcribing for an hour. This
  matches both the research-app posture and the confidentiality context.

## 4. Backend matrix

| Backend | Where | Schema enforcement | Japanese strength | Notes for this task |
|---|---|---|---|---|
| Apple FM 3B | on-iPad | `@Generable` guided generation (guaranteed) | adequate, weak on long sessions | best schema guarantee; smallest capacity — use with hybrid pre-filtering (D) and per-item calls |
| Qwen3 (MLX, on-iPad) | on-iPad | prompt contract + tolerant parser today; [mlx-swift-structured](https://github.com/petrukha-ivan/mlx-swift-structured) (Apache-2.0, v0.2.0, token-masked constrained decoding incl. `@Generable` types, Qwen3-tested) is the upgrade path — early-stage, evaluate before adopting | good | the default on-device path; per-item map-reduce keeps prompts small |
| Llama-3-Swallow (MLX) | on-iPad | same as Qwen3 path | strong Japanese | candidate; weaker instruction-following on strict JSON in current use |
| LM Studio (Mac, LAN) | remote (opt-in) | `response_format` json_schema (already wired) | model-dependent | where the big/special models live: Qwen3-32B-class, **LLM-jp-4-32B-A3B** (NII, Apr 2026 — 3.8B active params, JA MT-Bench 7.82 > GPT-4o 7.29, ~65k context, open license) is the strongest open Japanese option and cheap to run as MoE; **NuExtract 3 / 2.0** as the specialized-extractor candidate |
| Cloud APIs | opt-in only | native structured output | best | effectively ruled out for this use: the form is 関係者外秘; existing "no cloud without toggle + privacy note" rule stands, and LAN LM Studio makes cloud unnecessary |

ASR side (feeds everything): the six onomatopoeia and road names must survive
transcription. Actions that don't require new research: put them in the
keyword bank (drives the existing homophone review), and use them as
`contextualStrings` in the `SFSpeechRecognizer` re-scoring path (constraint
#3 in CLAUDE.md) for critical rows.

## 5. Proposed pipeline (target design, not yet built)

```
Session (.xph): utterances + speakers + timings
  │
  ├─ 0. Form template (JSON Schema per form type; A-1 first, others later)
  │
  ├─ 1. Deterministic pass (no LLM)
  │      keyword/phonetic tagging per item ─ onomatopoeia bank
  │      spoken-score regex capture (±0.125…1.0, 1–9)
  │      road-callout sectioning → Sections
  │      evaluator identification (dominant speaker / user pick)
  │
  ├─ 2. LLM extraction (schema-constrained, evidence-cited)
  │      per-item: candidate rows → {score(stated|null), scoreInferred?,
  │        likeDislike(stated|null), comment, evidenceRows[]}
  │      header metadata from session-start rows
  │      補足コメント from untagged substantive rows
  │      (map-reduce when over cap; merge = per-field union + conflict flags)
  │
  ├─ 3. (later) Verify pass — second model refutes each filled field
  │      against its cited rows (transcription-reviewer pattern)
  │
  └─ 4. Render + review UI
         JSON → markdown/CSV export first (Export module);
         field↔row↔audio navigation for sign-off;
         PDF overlay is a separate, later concern
```

Draft output schema sketch (per item):

```json
{
  "itemId": "11_hyokohyoko",
  "strengthScore": -0.25,          // null unless stated
  "strengthScoreInferred": null,    // optional suggest-mode value, kept separate
  "likeDislike": null,              // 1–9, null unless stated
  "comment": "…",                  // distilled from evidence rows, ja
  "evidenceRows": [42, 47],
  "conflicts": []                   // e.g. two different stated scores
}
```

## 6. Evaluation plan (before any implementation is trusted)

- **Ground truth:** human-filled forms for a handful of real sessions
  (the Tiguan session + upcoming 2026-04-15 data). Store alongside `.xph` as
  labeled fixtures (consented, ≤ existing fixture policy).
- **Metrics:** field-presence F1 (did it fill what the human filled, and
  nothing else) · stated-score exact match + MAE in 0.125 steps ·
  evidence validity (cited rows actually support the field) · comment
  faithfulness (no claims without evidence) · reviewer time-to-sign-off vs
  manual filling.
- **Bake-off:** same harness over {Apple FM, Qwen3-MLX, Swallow-MLX,
  Qwen3-32B\@LM Studio, LLM-jp-4-32B-A3B\@LM Studio, NuExtract 3\@LM Studio}
  × {with/without hybrid pre-filtering}. Numbers go to `docs/eval_log.md`
  per repo convention.

## 7. Recommendation

1. **Build Option A+D** (general LLM in the existing summarizer stack +
   deterministic pre-passes), because it reuses proven machinery (evidence
   rows, glossary, structured output, map-reduce, keyword matching) and keeps
   the confidential data on-device/LAN.
2. **Policy first:** stated-vs-inferred separation, null-first schema,
   evidence on every field, human sign-off UI. This determines trustworthiness
   more than model choice.
3. **Treat specialized models as backends, not architecture:** bake off
   NuExtract 3 and LLM-jp-4-32B-A3B against Qwen3-class prompting on the LM
   Studio path; adopt whichever wins the eval, behind the same schema.
4. **Skip judge-LLMs** for extraction; reconsider the judge *pattern* later as
   a verify pass.
5. **Before building:** collect 2–3 human-filled ground-truth forms and run a
   manual prompt prototype (PromptsCard's Real Prompt export + LM Studio) to
   validate the schema and the stated-score density in real sessions — this
   costs hours, not days, and de-risks the whole design.

## Sources

- [NuExtract-2.0-8B](https://huggingface.co/numind/NuExtract-2.0-8B) · [NuExtract-2.0-4B](https://huggingface.co/numind/NuExtract-2.0-4B) · [NuExtract-2.0-4B-GGUF](https://huggingface.co/numind/NuExtract-2.0-4B-GGUF) · [NuExtract 3 release note](https://7minai.com/news/nuextract-3-4b-vlm-release/) · [NuExtract platform](https://numind.ai/blog/nuextract-platform-the-new-information-extraction)
- [mlx-swift-structured (constrained decoding for MLX Swift)](https://github.com/petrukha-ivan/mlx-swift-structured) · [MLX Swift @Generable exploration](https://rudrank.com/exploring-mlx-swift-structured-generation-with-generable-macro)
- [LLM-jp-4 release (NII)](https://www.nii.ac.jp/news/release/2026/0403.html)
- [Prometheus 2 paper](https://arxiv.org/abs/2405.01535) · [prometheus-eval](https://github.com/prometheus-eval/prometheus)
- Ambient clinical documentation accuracy: [GPT-4 structured notes study](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11074889/) · [ambient scribe validation](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1691499/full) · [AWS HealthScribe](https://aws.amazon.com/healthscribe/)
- Industry context: [AVL sensory-evaluation quantification](https://xtech.nikkei.com/atcl/nxt/column/18/00063/00030/) · [GLiNER2](https://arxiv.org/pdf/2507.18546)
