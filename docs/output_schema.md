# Per-utterance JSON output schema

Canonical shape produced by `Core/Export/JSONExporter`. The exporter emits a
single JSON **array** of utterance objects (pretty-printed, keys sorted),
not JSON Lines — historical drift from this doc; the array form is what the
shipped encoder produces. Keys are written using the Swift property name
verbatim (`speakerID`, `asrConfidence`, …), not snake_case — `JSONExporter`
sets no `keyEncodingStrategy`, so the JSON output is what synthesized
`Codable` writes on `UtteranceEstimate`.

```json
[
  {
    "id": "1F8E8B1C-…",
    "speakerID": "S01",
    "speakerName": "Alice",
    "start": 12.34,
    "end": 14.71,
    "transcript": "今日は本当に楽しかった",
    "asrConfidence": 0.87,

    "dimensional": {
      "valence": 0.78,
      "arousal": 0.62,
      "dominance": 0.55
    },
    "acousticCategorical": {
      "happy": 0.71, "neutral": 0.18, "surprised": 0.06,
      "sad": 0.02, "angry": 0.01, "fearful": 0.01,
      "disgusted": 0.005, "other": 0.005, "unknown": 0.0
    },
    "ageGender": {
      "age": 0.32,
      "female": 0.95,
      "male": 0.03,
      "child": 0.02
    },

    "plutchik": {
      "joy": 0.81, "trust": 0.34, "anticipation": 0.22,
      "surprise": 0.09, "sadness": 0.04, "fear": 0.02,
      "anger": 0.02, "disgust": 0.01
    },
    "textBackend": "deberta",

    "speechBoost": true,
    "wasReevaluated": false,
    "wasHandEdited": false,

    "fusedValence": 0.79,
    "fusedArousal": 0.61,
    "fusedDominance": 0.55,
    "fusedTopLabel": "joy"
  }
]
```

## Field notes

- **`id`** — stable per-utterance UUID; survives Save → Open via `.xph` and
  is the cross-reference key for external tooling.
- **`speakerID`** — canonical session id like `S01`. Always present.
- **`speakerName`** — user-supplied display name; omitted when the speaker
  was never renamed.
- **`asrConfidence`** — `[0, 1]`; absent if the underlying transcriber
  doesn't expose one. Propagates into fusion weights
  (see `Core/Fusion/LateFusion`).
- **`dimensional`** — audeering W2V2 V/A/D, each in `[0, 1]`; absent if the
  acoustic dimensional model wasn't available or the utterance was too
  short to score.
- **`acousticCategorical`** — emotion2vec+ 9-class softmax; missing labels
  imply 0.
- **`ageGender`** — audeering W2V2 age-gender. `age` is in `[0, 1]`
  (multiply by 100 for years); `female`/`male`/`child` are softmax
  probabilities summing to 1. Absent if the model wasn't loaded or the
  clip was unusable. The Swift adapter at
  `Core/SER/Acoustic/W2V2AgeGenderSER.swift` documents the gotcha that
  the order is `[female, male, child]` per the audeering tutorial repo
  — not the order the Hugging Face model card's prose suggests.
- **`plutchik`** — DeBERTa-WRIME 8-class softmax; missing labels imply 0.
- **`textBackend`** — identifier for the text-SER backend that produced
  `plutchik` (e.g. `"deberta"`, `"foundationModels"`). Nil when text SER
  was skipped.
- **`speechBoost`** — whether the capture-side speech-boost EQ was on
  when the utterance was recorded. Nil for batch-imported audio where
  capture-time state is unknown.
- **`wasReevaluated`** — `true` after the user has manually re-evaluated
  the row (offline ASR re-run with padded boundaries + SER + fusion
  redone). Nil for rows that came straight from the streaming pipeline.
- **`wasHandEdited`** — `true` after the user has hand-edited the row
  (transcript and/or time range). A later re-evaluation clears this back
  to nil. Nil otherwise.
- **`fused*`** — late-fusion outputs. `fusedValence` / `fusedArousal` /
  `fusedDominance` are in `[0, 1]`; `fusedTopLabel` is the argmax across
  the fused Plutchik distribution.

## Save (`.xph`) ↔ JSON parity

`SessionDocument` (binary plist) and the JSON export both serialize the
same `UtteranceEstimate` Codable shape, so every field above survives a
Save → Open round-trip. The `.xph` bundle additionally carries audio,
the diarizer's speaker DB, the cumulative diarization timeline, and the
last LLM session summary — none of which appear in the JSON export
(JSON is the per-utterance affect record, not a full session snapshot).
