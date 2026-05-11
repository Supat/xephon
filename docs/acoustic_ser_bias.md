# Acoustic SER bias toward "happy"

Observed: the fused top label on Japanese conversational audio
disproportionately resolves to `happy`, even on stretches that read
as neutral or mildly negative to a human listener.

This is a real bias, not a regression. It has two converging
structural causes plus a smaller third one. Each is documented
below with the relevant code reference, followed by a list of
mitigations and their trade-offs.

## Cause 1 — Plutchik→label mapping is 3:1 in favour of "happy"

The fused top label is an argmax over a bucket score that combines
acoustic-categorical (emotion2vec) probabilities with Plutchik
(text SER) probabilities, mapped onto the acoustic-categorical
label space. The mapping lives at
`Core/Fusion/LateFusion.swift::plutchikToAcousticLabelMapping`:

```swift
public static let plutchikToAcousticLabelMapping: [PlutchikScore.Label: String] = [
    .joy: "happy", .sadness: "sad", .anger: "angry",
    .fear: "fearful", .disgust: "disgusted", .surprise: "surprised",
    .trust: "happy", .anticipation: "happy",
]
```

Three Plutchik labels (`.joy`, `.trust`, `.anticipation`) collapse
into the same `"happy"` bucket. The other five each get their own.
Inside `topLabel`:

```swift
for (k, v) in p.probabilities {
    guard let mapped = plutchikToAcousticLabelMapping[k] else { continue }
    scores[mapped, default: 0] += v * asrConfidence
}
```

The contributions are **summed**, not maxed. So a text SER output
like:

| Plutchik label | probability |
|---|---|
| joy           | 0.20 |
| trust         | 0.15 |
| anticipation  | 0.10 |
| sadness       | 0.30 |

after the mapping produces:

| acoustic bucket | accumulated score |
|---|---|
| happy | 0.45 (= 0.20 + 0.15 + 0.10) |
| sad   | 0.30 |

`happy` wins despite `sadness` being the largest single Plutchik
label. The user experiences "everything is happy" because the
mapping has effectively three votes for happy and one vote for
each other emotion.

This is a well-known artefact of reducing Plutchik-8 to a smaller
label space. `.trust` and `.anticipation` aren't *wrong* mappings
on Russell's circumplex — both sit in the +V region — but giving
them equal weight to `.joy` over-claims for "happy" specifically.

## Cause 2 — Pitch-accent / emotional-prosody confound

Japanese F0 movement carries lexical (pitch accent) information
that cross-lingual SER models like audeering W2V2 and emotion2vec
weren't trained to distinguish from emotional prosody. The models
were trained predominantly on languages where F0 contour signals
affect, not lexical content, so neutral declarative Japanese
speech with normal pitch movement often reads as mildly aroused +
mildly positive — i.e. `happy`.

This is already noted in:

- `CLAUDE.md` § Japanese-specific gotchas:
  > Pitch accent vs. emotional prosody. F0 movements in Japanese
  > carry lexical information; cross-lingual SER models can
  > mis-attribute them to arousal.

- `docs/feasibility.md` § Japanese SER:
  > The cross-corpus picture: Japanese SER datasets are an order
  > of magnitude smaller than English MSP-Podcast/IEMOCAP and
  > biased toward acted or game-chat speech, not natural
  > research-interview speech.

The confound interacts with cause 1: if the acoustic side already
biases mildly toward "happy" for engaged-prosody Japanese, and the
text side also accumulates disproportionately into "happy", the
combined argmax is even more skewed than either modality alone.

## Cause 3 — `plutchikToValence` is mildly positive-leaning

The Russell mapping used for V/A in
`Core/Fusion/LateFusion.swift::plutchikToValence`:

```swift
let pos = (p.probabilities[.joy] ?? 0)
        + (p.probabilities[.trust] ?? 0) * 0.6
        + (p.probabilities[.anticipation] ?? 0) * 0.3
let neg = (p.probabilities[.sadness] ?? 0)
        + (p.probabilities[.fear] ?? 0)
        + (p.probabilities[.disgust] ?? 0)
        + (p.probabilities[.anger] ?? 0) * 0.7
```

Positive contribution sums to a max of `1 + 0.6 + 0.3 = 1.9`;
negative caps at `1 + 1 + 1 + 0.7 = 3.7`. The negative side is
actually larger in theory — but in practice, text SER on Japanese
research-interview speech rarely emits high probabilities for
`.disgust` or `.anger`, while `.trust` and `.anticipation` are
near-default outputs for polite/declarative content. The
*observed* sum is positive-leaning on conversational Japanese.

This affects `fusedValence` and through it `dominantSpeaker` /
trajectory plots, but not the categorical top label directly.

## Why this is hard to fix cleanly

The "right" fix would be a Japanese-specific calibration set —
record neutral conversational Japanese, label by hand, fit a per-
class prior and subtract it. That's `docs/feasibility.md`'s
recommended path, but the calibration set doesn't exist yet (see
`docs/eval_log.md`).

In the meantime, the heuristics are picking the highest-effort
patches and accepting they're approximations:

- The Plutchik→label mapping was chosen for *interpretability*
  (every Plutchik label corresponds to some emotion2vec class)
  rather than per-label calibration.
- The V/A coefficients are documented as "conservative defaults;
  tune from a calibration set per `docs/eval_log.md`".

A sharper fix waits on data.

## Mitigation options (sketches)

### A. Down-weight `.trust` and `.anticipation` for top label

`plutchikToAcousticLabelMapping` could be replaced with a weighted
mapping:

```swift
let weight: [PlutchikScore.Label: (String, Float)] = [
    .joy:          ("happy",     1.0),
    .trust:        ("happy",     0.4),
    .anticipation: ("happy",     0.3),
    .sadness:      ("sad",       1.0),
    .anger:        ("angry",     1.0),
    .fear:         ("fearful",   1.0),
    .disgust:      ("disgusted", 1.0),
    .surprise:     ("surprised", 1.0),
]
```

So the "happy" bucket's accumulation becomes `joy + trust × 0.4 +
anticipation × 0.3`, matching the proportional contributions
already used in `plutchikToValence`. This is the smallest possible
change.

Trade-off: arbitrary coefficients. The right values depend on the
text SER's empirical Plutchik distribution on real input, which
varies by backend (DeBERTa vs Foundation Models) and by speech
register (formal vs casual). Conservative defaults are guesses.

### B. Add a `.neutral` path for low-confidence Plutchik

If no Plutchik label exceeds a confidence threshold (e.g., max
probability < 0.4), contribute the residual to a "neutral" bucket
in `topLabel`. emotion2vec already has a `.neutral` class, so the
bucket exists; it's just never reachable from the text side today.

This wouldn't change the V/A computation, just the categorical
top-label argmax for ambiguous text.

Trade-off: requires picking a threshold without empirical
grounding. Too high → most utterances become "neutral"; too low →
no effect.

### C. Subtract a per-class prior

Profile the fused-label distribution on a held-out Japanese
neutral conversation set (e.g., 5 min of news read-aloud) and
subtract the resulting per-class frequencies from each utterance's
bucket scores before argmax. This is the principled fix, equivalent
to per-class calibration on a "neutral" reference.

Trade-off: requires building the calibration set, which is the
gating step `docs/feasibility.md` was already going to need for
proper per-class normalization. Worth doing once and reusing
across both this bias and the broader cross-lingual calibration
gap.

### D. Replace mean-pool acoustic head with a learned classifier

The deepest fix: emotion2vec_plus_large's mean-pool head was
trained on cross-lingual data weighted heavily toward
English/Mandarin. A small linear probe fine-tuned on Japanese-only
data would correct the pitch-accent confound at the model level.

Trade-off: requires training infrastructure, labelled Japanese
data, and a deployment story for the additional weights. Out of
scope for a research app.

## Reproducing

To verify the cause-1 mechanism interactively:

1. Pick an utterance where the green fused label is "Happy" but the
   acoustic categorical detail shows ≤ 0.4 probability on `happy`
   alone (tap the row to expand it).
2. Look at the Plutchik detail. If the highest single Plutchik
   probability is *not* `.joy` but the top label is still "happy",
   you're seeing the joy + trust + anticipation accumulation
   dominate.

The accumulation logic is `LateFusion.topLabel`.

## Status

Not currently mitigated. Documented so future work has a hand-off
point. The cleanest place to start is **option A** (proportional
re-weight matching `plutchikToValence` coefficients) — minimal
code change, zero new dependencies, and consistent with the V/A
math already in place. Validation should land in `docs/eval_log.md`
against a held-out Japanese set.
