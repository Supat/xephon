# Fuzzy-search stack audit — hit-rate & accuracy opportunities

*2026-07-07. Scope: `JapaneseSearchNormalizer`, `NormalizedSearchQuery`,
`LatinLetterReadings`, `FuzzySubstringMatcher`, the threshold functions on
`SearchReplaceCoordinator`, and their consumers (transcript filter,
find-and-replace, keyword counts/highlighter/review).*

## Architecture (as built)

One normalized space: `CFStringTokenizer`'s latin transcription collapses
kanji/hiragana/katakana/romaji to lowercased Hepburn-ish romaji, per token,
with original-range mapping. Matching tiers, tightest→loosest:

1. Raw case-insensitive substring (original text)
2. Exact normalized substring (cross-script; homophones collapse here)
3. Boundary-form token runs (single Latin letters → spoken readings)
4. Windowed Levenshtein per token, threshold `max(1, L/4)`
5. Longest-common-substring per token, min-run `max(3, ~0.4·L)`

The layering is sound and the app-wide consistency (one notion of "sounds
close", reused by search, keywords, and review) is a real strength. The
opportunities below are ranked by expected hit-rate/accuracy return.

---

## R1 — Romaji canonicalization pass *(recall; cheap; low risk)*

The tokenizer emits **Hepburn** ("shi, chi, tsu, fu, ji"), but Latin input
passes through **raw** (only lowercased). Consequences:

- A user typing Kunrei/Nihon-shiki romaji — `si`, `ti`, `tu`, `hu`, `zi`,
  `sya`, `tya`, `zya` — never exact-matches tier 2 and must survive on
  Levenshtein budget, wasting edits on systematic spelling variance rather
  than genuine differences.
- Long-vowel spellings fracture the space: transcript `shiryou` vs typed
  `shiryo`, `shiryō`, `shiryoh`, `shiryoo` — four ways to write one word,
  only one of which matches exactly.

**Fix:** a deterministic canonicalization table applied to every normalized
token (both sides see it, so it's symmetric): Kunrei→Hepburn digraph map,
macron/`oh`/doubled-vowel → one canonical long-vowel spelling. Strictly
increases recall; precision cost ≈ 0 because the collapsed forms are
pronunciation-identical.

## R2 — NFKC compatibility fold before tokenizing *(recall; one line)*

Full-width ASCII（ＡＢＣ１２３）and half-width katakana（ｶﾀｶﾅ）are not folded
before tokenization — `lowercased()` doesn't touch them, and whether the
tokenizer transcribes half-width kana is unverified. ASR output is unlikely
to contain them, but user-typed queries, pasted text, and imported keyword
lists can. `precomposedStringWithCompatibilityMapping` at the top of
`tokens(_:)` closes the class.

## R3 — Phonetic-key tier for ASR confusion classes *(the big one)*

Unit-cost Levenshtein treats every substitution equally, but Japanese ASR
errors are **structured**: long↔short vowel (`toukyou`/`tokyo`),
geminate presence (`kitte`/`kite`), voiced↔unvoiced onset (`k/g`, `s/z`,
`t/d`, `h/b/p`), `n/m` assimilation. Today these burn generic edit budget —
a 6-char query gets 1 edit total, so *one* systematic difference exhausts it
and a second (extremely common in combination) causes a miss. Loosening the
threshold instead would raise false positives *everywhere*.

**Fix:** a **phonetic key** derived from the canonical romaji — collapse
long vowels, degeminate, devoice onsets, `m→n` — compared for **exact
equality** as tier 3.5 (between exact-normalized and Levenshtein). Two
strings share a key iff they differ only by known confusion classes: high
recall on exactly the errors ASR makes, near-zero precision cost, **no
length floor** (recovers fuzzy coverage for the short words tier 4 excludes
by `minQueryLengthForSimilar = 4`). This tier is what the keyword-review
feature wants most — its whole premise is ASR confusions.

## R4 — Semi-global alignment; fuzzy over the joined text *(recall + perf)*

Two coupled issues:

- **Granularity gap:** tiers 4–5 run **per token**. A near-match whose
  reading spans a token boundary is invisible — and `CFStringTokenizer`
  splits the same reading differently depending on context (documented in
  `NormalizedSearchQuery` for exactly this reason), so where the boundary
  falls is not under our control. The exact-match tier scans the *joined*
  normalized string; the fuzzy tiers can't afford to, because…
- **Algorithm:** `hasSimilarSubstring` slides every window length in
  `[L−t, L+t]` over every start with a full Levenshtein per window —
  O((2t+1)·n·L²)-ish. The standard replacement is **semi-global alignment**
  (Sellers): one DP pass with a zero-initialized first row computes the
  minimum edit distance of the query against *any substring* of the text in
  O(L·n), same semantics, no window loops.

Together: swap the matcher core for semi-global, then run tier 4 over the
**joined** normalized transcript and map hits back through the token
offsets (the reconstruction machinery already exists for tier 2). Boundary-
spanning near-misses become visible, and `hasHighlightableSimilarToken`'s
row-gating workaround (which exists because joined-only matches couldn't be
highlighted) becomes unnecessary.

## R5 — Tighten or scope the LCS tier *(precision)*

`hasLongCommonSubstring` fires on any shared run of `max(3, ~0.4·L)` chars.
In romaji space, 3-char runs (`shi`, `kai`, `you`, `tta`) are ubiquitous —
for 4–7-char queries this tier is the stack's main false-positive source,
and it now leaks into keyword review as junk suspects. Options, compatible:

- Require **containment ratio**, not just absolute run length: run ≥ 50% of
  the query AND ≥ 40% of the candidate token, so `dia` matching both
  `midiamu` and `mediatte` still works but `you` alone doesn't fire.
- **Scope it**: keep LCS in explicit "Include similar" search (user opted
  into loose matching) but drop it from keyword-review detection, where
  precision matters more (every suspect demands user attention) and R3
  covers the legitimate cases better.

## R6 — Damerau (transposition-aware) Levenshtein *(minor recall)*

Adjacent-character swaps (`ie`↔`ei` etc.) cost 2 edits today. One-line
extension to the DP. Mostly benefits typed-query typos rather than ASR
errors; low priority.

## R7 — Measure before/after *(process; makes the rest safe)*

Nothing above should land blind. The repo has the right precedent
(`LetterNameSearchTests`) but the scheme currently attaches **zero test
targets** — fixing that is prerequisite. Then: a small labeled fixture set
of (query, transcript, expected-hit?) pairs covering homophones, long-vowel
variants, voicing confusions, boundary-spanning readings, and known
false-positive traps — run per change, results noted in `docs/eval_log.md`
per the CLAUDE.md eval rule. R1/R3/R5 all shift the recall/precision
balance; the fixture set is what proves each moved it the right way.

## Accepted limitations (document, don't fight)

- **Single-reading transcription**: the tokenizer emits one contextual
  reading per token; kanji homographs read differently in query vs
  transcript (行った as *itta* vs *okonatta*) can miss. Fixing this needs a
  multi-reading dictionary — out of scope for the return.
- **Fused-token boundary forms**: ジーパン never matches letter "G" — the
  documented, intended tightening.

## Suggested order

R2 (one line) → R1 (table) → R7 (fixtures) → R3 (phonetic key) → R4
(semi-global + joined text) → R5 (LCS scoping) → R6 (optional). R1–R3 are
independent of R4–R5 and each is safe to ship alone once fixtures exist.
