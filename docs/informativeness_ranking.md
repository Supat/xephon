# Per-utterance informativeness ranking

Algorithm that ranks each utterance in a session by how
**semantically distinctive** it is, with optional speaker-
balanced selection on top. Lives in
`Core/Fusion/Informativeness.swift`. Used by the heuristic
summarizer mode (`SummarizeMode.heuristic`) to pick which
utterances feed the LLM when the session exceeds the
backend's prompt budget. Deterministic, no LLM, no network —
runs in milliseconds even on hundreds of utterances.

## What "informative" means here

Informativeness is **distinctive vocabulary** relative to the
session, not **semantic importance**. A verbose speaker who
casually drops "辞めることにしました" ("I quit my job") once
between paragraphs of small talk will see that line under-
ranked because their own session-level IDF is washed out by
everything else they said. For semantic ranking you'd add an
LLM re-rank pass over the top-N TF-IDF candidates — out of
scope here.

The scorer outputs a value in `[0, 1]` where `1.0` = the
single most distinctive utterance in this session, others are
proportional. Scores are session-relative — adding or
removing one utterance changes every other utterance's
score, which is why scores are computed on demand and never
persisted on `UtteranceEstimate`.

## Pipeline

```
transcript → tokenize → per-token IDF → sum per utterance
            → length-normalize → backchannel penalty
            → normalize to [0, 1] across the session
```

### 1. Tokenization

`String.enumerateSubstrings(in:, options: .byWords)` —
delegates to `CFStringTokenizer`, which has built-in Japanese
segmentation. Not as morphologically accurate as Mecab but
ships with iOS / iPadOS and is good enough at TF-IDF scale
(over- and under-segmentation tend to wash out across the
session's aggregate). ASCII lowercased; CJK preserved
verbatim; empty / whitespace-only tokens dropped.

### 2. IDF

Standard smoothed inverse document frequency over the per-
utterance token lists:

```
idf(t) = log((1 + N) / (1 + df(t))) + 1
```

where `N` = number of utterances and `df(t)` = number of
utterances containing token `t`. The `+1` shift keeps every
observed token at `idf ≥ 1` so high-frequency content words
still carry weight; the `1+` smoothing avoids `log(0)` on
singletons.

### 3. Per-utterance raw score

```
sum_idf  = Σ idf(t) for t in tokens
normed   = sum_idf / sqrt(tokenCount)
penalty  = 1 - 0.9 × (backchannelTokenCount / tokenCount)
raw      = normed × penalty
if utteranceID ∈ boostedIDs:
    raw  = raw × 4.0
```

**Why `sqrt` for length normalization** — linear
normalization over-penalizes natural-length utterances; no
normalization lets `"はい はい はい そう そう"` outscore a
short content utterance by repetition. `sqrt` is the BM25 /
classical-IR middle ground.

**Backchannel penalty** — proportional to what fraction of
the row is composed of tokens in the inline Japanese
backchannel/filler lexicon (`うん`, `そう`, `はい`, `えーと`,
`あの`, `まあ`, …). 100% backchannel → 0.1 (kept above zero
so callers can decide whether to threshold). The lexicon is
drawn from CSJ / CallHome / OGVC backchannel inventories;
kept inline because it's small and rarely changes.

**Keyword boost (`boostedIDs`)** — a caller-supplied
`Set<UUID>` of utterance IDs flagged as containing one or
more user-curated keywords. Each such row is multiplied by
**4.0** (the `keywordBoost` constant) before session
normalization, so keyword-hit utterances reliably dominate
the top of the ranking — the user said "if you care enough
to put this in your keyword list, treat it as evidence the
row matters." The factor is large enough to push a keyword
hit above most non-keyword rows yet small enough that, in a
session full of keyword hits, the underlying TF-IDF mass
still differentiates the most informative among them (vs. a
flat 0/1 boost which would flatten that quality ordering
inside the boosted subset). The matching itself happens
**outside** Informativeness — see `SummarizerCoordinator
.keywordBoostedIDs(in:)` for the canonical implementation
(uses `JapaneseSearchNormalizer` for cross-script Kanji ↔
kana ↔ romaji unification, so the boost behaves consistently
with the transcript pane's keyword-count chips). Keeping
matching out of Informativeness lets the module stay free of
locale-specific normalization dependencies.

### 4. Session normalization

`score(utterances:)` divides every raw score by the session's
max, giving `[0, 1]`. When every score is 0 (e.g. a single
all-backchannel utterance), returns `{id: 0}` for every id.

## The two ranker entry points

### `topN(_ n: Int, utterances:) -> Set<UUID>`

Pure global ranking. Returns the IDs of the N highest-scoring
utterances regardless of speaker. Cheap, unsurprising, but
can leave a quiet speaker entirely absent from the selection
on multi-speaker sessions — a problem for summarizer
prompting.

### `topNBalancedBySpeaker(_ n:, utterances:, maxSpeakers: 10)`

**The heuristic-mode default.** Guarantees each speaker (up
to `maxSpeakers`) gets at least one slot when the budget
allows, then proportions the rest by per-speaker total
informativeness.

#### Allocation algorithm

1. **Group** utterances by speaker. Cap to `maxSpeakers = 10`
   by dropping the lowest-total-weight speakers first.
   Defensive — typical sessions are well under 10.
2. **Fallback** when `n < speakerCount`: the
   include-every-speaker promise is impossible. Degenerate to
   `topN(n:)` — pure global ranking, no balancing.
3. **Floor**: every active speaker gets 1 slot.
4. **Hamilton's largest-remainder method** on the remaining
   `n - speakerCount` slots, weighted by each speaker's
   **total informativeness sum** (Σ of their utterances'
   scores). This single weight folds in two signals:
   - **Volume** — more utterances → larger sum (rewards
     contribution).
   - **Quality** — more distinctive utterances → larger
     individual terms (rewards information density).
   - Mean would only capture quality; raw count would only
     capture volume; sum is the right primitive.
5. **Saturation cap + redistribute**: if a speaker is
   allocated more slots than they actually have utterances,
   cap them at their count, mark saturated, re-Hamilton the
   freed surplus among non-saturated speakers. Iterate to a
   fixed point. Bounded by `maxSpeakers` rounds because each
   round either saturates ≥ 1 speaker or exits.
6. **Pick**: per speaker, take the top-K by per-utterance
   informativeness from their own pool.

Returned `Set<UUID>` is unordered. Chronology is restored at
the call site: `utterances.filter { ids.contains($0.id) }`.

#### When all weights are zero

If every speaker's informativeness sum is 0 (rare: every
utterance is all-backchannel after the penalty), Hamilton
degenerates to as-even-as-possible distribution. The
selection still spreads across speakers rather than
collapsing onto whoever happens to come first in the
dictionary.

#### When a session has > `maxSpeakers` speakers

Speakers beyond the cap are dropped before allocation, by
ascending total informativeness — the speakers with the least
session-relative content get cut. This is defensive: most
sessions are 2–6 speakers; > 10 typically means diarization
over-segmented. If diarization quality improves, the cap can
be raised without algorithmic change.

## Worked example

Session with 4 speakers, `n = 10` slots:

| Speaker | utterance count | informativeness sum |
|---|---|---|
| S01 | 30 | 12.0 |
| S02 | 20 | 8.0 |
| S03 | 2  | 1.5 |
| S04 | 8  | 4.5 |

1. Floor: each gets 1 → 4 slots used, 6 remaining.
2. Hamilton on remaining 6, weighted by sum (total = 26):
   - S01: 6 × 12/26 = 2.77 → 2 + leftover 0.77
   - S02: 6 × 8/26  = 1.85 → 1 + leftover 0.85
   - S03: 6 × 1.5/26 = 0.35 → 0 + leftover 0.35
   - S04: 6 × 4.5/26 = 1.04 → 1 + leftover 0.04
   - Distributed: 4. Leftover 2 → goes to S02 (0.85) and S01
     (0.77) by descending fraction.
   - Extras: S01=3, S02=2, S03=0, S04=1.
3. Sum with floor: S01=4, S02=3, S03=1, S04=2. Total 10. ✓
4. No saturation (every speaker has enough utterances).
5. Pick per-speaker top-K by informativeness.

S03 (the quiet speaker with only 2 utterances) is guaranteed
representation despite low weight. S01 gets the largest share
because they contributed both volume and distinctive content.

## Where it's used

- **MLX summarizer (`MLXQwenSummarizer.summarizeSinglePass`)**
  — `selection: .heuristicTopN` calls
  `topNBalancedBySpeaker(maxPromptUtterances, utterances:
  boostedIDs:)`. The truncation note in the prompt informs
  the LLM the rows are a representative TF-IDF sample (NOT
  the most recent), shown chronologically, with gaps
  expected. Different prompt framing vs. `.trailing` to
  avoid the model treating the last-chronological row as
  "the current state."
- **Apple FM summarizer** — same selection mechanism, same
  per-prompt note adjustment, smaller `maxPromptUtterances`
  (15) because of FM's 4 k context.
- **Keyword boost producer:** `SummarizerCoordinator
  .keywordBoostedIDs(in:)` runs before each `.heuristic`
  invocation, normalizes both user keywords and each
  utterance's transcript via `JapaneseSearchNormalizer`,
  and produces the `Set<UUID>` passed as `boostedIDs`.
  The `.fast` and `.deep` paths bypass this — `.fast`
  always picks the trailing window regardless of content,
  and `.deep` processes every utterance so no selection
  bias applies.

## Limitations and extension points

- **Informativeness ≠ semantic importance.** TF-IDF underranks
  one-off semantically-loaded statements made by verbose
  speakers. An LLM re-rank pass over the top-2N candidates
  (selecting the final N) is the natural next layer; not
  implemented because it'd add an inference call to a path
  whose entire point is being LLM-free.
- **Backchannel lexicon is Japanese-only.** The list at the
  bottom of `Informativeness.swift` covers JP conversational
  patterns. Sessions in other languages still benefit from
  the TF-IDF + length normalization but lose the specific
  filler/backchannel downweight. Extending: the lexicon could
  be made locale-conditional (read `SummarizerLocale` for the
  active session) without algorithmic change.
- **Tokenization quality varies by script.** CFStringTokenizer
  segments Japanese decently but a true morphological
  analyzer (Mecab via `vapor-community/swift-mecab` or
  similar) would tighten token boundaries — only worth the
  dependency if the heuristic mode's quality plateaus and we
  trace the cap to tokenization noise.
- **No cross-utterance context.** Each utterance is scored
  independently. A turn-taking model that boosts utterances
  introducing new topic shifts (vs. continuing the prior
  speaker's frame) would catch a different signal; out of
  scope for the current scorer.
- **Keyword matcher lives in the caller.** Informativeness
  doesn't know how `boostedIDs` was computed — by design.
  This keeps the module free of locale-specific dependencies
  (JP normalizer, Mecab, romaji transliteration). Callers can
  build the set with whatever matcher fits their domain;
  Xephon happens to use cross-script JP normalization because
  that's what the rest of the app's search uses.
