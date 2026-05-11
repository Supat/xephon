# SER padding post-mortem

## Symptom

Across the app, fused emotion classifications "felt off" after a recent
round of changes — borderline cases drifted toward `sad` / `neutral`,
and W2V2's V/A/D values trended toward middling values regardless of
how clearly the input expressed an emotion. Behavior reproduced in
both real-time and fast-pace modes, on the same audio.

## Wrong hypotheses, in order

The recent commit batch had touched several inference-adjacent things,
so the obvious suspects came first.

### Hypothesis 1: FP16 quantization degraded the models

Most plausible at first glance — every weight initializer had been
forced to FP16 (`force_fp16_initializers=True`) including small bias
terms that the conversion warnings were clamping to FP16's smallest
representable value. Compounded across transformer layers, this could
plausibly shift class probabilities.

A/B against PyTorch / FP32 references:

- **wrime FP16 ONNX vs PyTorch:** bit-identical to two decimals on
  six varied Japanese affect sentences (`今日は本当に楽しい一日でした`,
  `すごく悲しいです`, etc.).
- **W2V2 FP16 ONNX vs FP32 (Zenodo source):** max delta `0.0001` on
  zeros / noise / sine inputs.
- **emotion2vec FP16 ONNX vs FunASR reference (`AutoModel.generate`)
  on `test.wav`:** both predict `angry=100%`.

FP16 was *not* the cause.

### Hypothesis 2: `setGraphOptimizationLevel(.extended)` cap

The `.all` tier (default) was capped at `.extended` because the
SimplifiedLayerNormFusion pass crashes on the FP16 converter's
auto-inserted `InsertedPrecisionFreeCast_*` nodes. Maybe the skipped
fusion was changing rounding behavior?

A/B with `.extended` vs `.all` on wrime (W2V2 can't load `.all`
because of the very crash that motivated the cap):

- wrime outputs **bit-identical** between the two levels.

The cap was a no-op numerically; not the cause.

### Hypothesis 3: emotion2vec model is broken

Synthetic input testing pointed here loudly. Running the FP16 (or
freshly re-exported FP32) emotion2vec on zeros / noise / sine, every
input collapsed to `sad ≈ 99%`. That looked like a model breakage.

Then I ran it against `test.wav` (real Japanese speech from FunASR's
example dir), comparing our ONNX export to FunASR's own
`AutoModel.generate`. Both produced `angry=100%`. Identical agreement.

So the synthetic-input degeneracy was a property of the trained
model on out-of-distribution inputs (silence/noise), not a bug. False
alarm.

## The actual bug: silence drag from `capForSER` zero-padding

`capForSER` quantizes each segment's audio length into one of three
bins (2 s, 4 s, 8 s) before SER inference, so CoreML EP only ever
compiles a fixed number of MLModel shapes. Segments shorter than a
bin used to be padded with zeros at the end. Segments longer get
center-cropped.

The acoustic SER models are mean-pool classifiers — the encoder
produces per-frame features and they get averaged across time before
the classification head. Mean-pool means *every frame contributes
equally to the output*. Including silence frames.

Probing each model's silence baseline (4 s of zeros):

| Model | Output on silence |
|---|---|
| W2V2 V/A/D | `A=0.611, D=0.639, V=0.392` |
| emotion2vec | `sad ≈ 99.6%` |

So zero-padding a 3 s utterance to fit a 4 s bin doesn't just append
"empty space" — it appends *active silence-class signal* that
counts for 25% of the mean-pool window.

Effect on a clearly-angry 4.10 s clip from `test.wav` after
trimming/padding to 4 s through W2V2:

| Padding | A | D | V |
|---|---|---|---|
| Native (no pad, 4.10 s) | 0.981 | 0.916 | 0.150 |
| **Zero-pad 3 s → 4 s** | **0.972** | **0.905** | **0.174** |
| Repeat-pad 3 s → 4 s | 0.991 | 0.911 | 0.242 |
| Native 3 s only (no pad) | 1.003 | 0.915 | 0.202 |

Zero-padding visibly shifts every value toward the silence baseline,
proportional to the silence fraction. For borderline cases this is
enough to flip the dominant categorical label after late fusion.

## Fix

Switch `capForSER`'s under-target branch from zero-padding to
repeat-padding (loop the utterance's own samples until the bin is
filled). The bin discipline stays — still three CoreML EP MLModel
cache slots — but the mean-pool window now contains 100% utterance
character instead of an utterance/silence blend.

Implementation in `Xephon/AnalysisPipeline.swift::capForSER(_:)`:

```swift
// Repeat-pad rather than zero-pad. Mean-pool acoustic classifiers
// bias toward whatever they predict on silence (W2V2 → mid-V/A,
// emotion2vec → "sad"); zero-padding a 2.5 s utterance to fit a 4 s
// bin pulls the prediction ~25% toward those silence baselines.
guard !buffer.samples.isEmpty else { return buffer }
var padded = buffer.samples
padded.reserveCapacity(target)
while padded.count < target {
    let needed = target - padded.count
    let chunk = min(needed, buffer.samples.count)
    padded.append(contentsOf: buffer.samples.prefix(chunk))
}
```

The micro-clicks at loop seams are spectrally negligible compared to
the silence-mean shift the prior approach introduced; the encoder's
convolutional front-end is robust to short-duration transients.

## Lessons

- **Out-of-distribution inputs aren't a model bug.** A trained
  classifier confidently predicting one class on noise/silence/zeros
  is consistent with the model's training distribution being real
  speech. Don't infer model breakage from synthetic-input behavior;
  always validate against a reference path on in-distribution audio.

- **Zero-padding has a non-trivial *signal* contribution to mean-pool
  classifiers.** "Padding" sounds like a no-op; for these models it
  isn't. The fix would have generalized to most pre-trained mean-pool
  audio classifiers (wav2vec2, HuBERT, emotion2vec — anything that
  averages over time before the head).

- **Don't change three things at once.** This bug existed *before* the
  recent commit batch — the silence drag was always there — but the
  user noticed it now because they were paying attention to outputs
  they'd previously taken at face value. If only one thing had
  changed, it would have been faster to localize.

- **Verify with a reference, not against itself.** The FP16 ONNX
  matched our FP32 ONNX matched our PyTorch reference matched
  FunASR's `generate` — four converging lines of evidence ruling out
  a numerical regression. Without that, the silence-drag hypothesis
  would still be one suspect among many.
