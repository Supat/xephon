# Per-utterance JSON output schema

Canonical shape produced by `Core/Export/JSONExporter`. One JSON Lines record
per utterance.

```json
{
  "speaker_id": "S01",
  "start": 12.34,
  "end": 14.71,
  "transcript": "今日は本当に楽しかった",
  "asr_confidence": 0.87,
  "acoustic": {
    "dimensional": { "valence": 0.78, "arousal": 0.62, "dominance": 0.55 },
    "categorical": {
      "happy": 0.71, "neutral": 0.18, "surprised": 0.06,
      "sad": 0.02, "angry": 0.01, "fearful": 0.01,
      "disgusted": 0.005, "other": 0.005, "unknown": 0.0
    }
  },
  "text": {
    "plutchik": {
      "joy": 0.81, "trust": 0.34, "anticipation": 0.22,
      "surprise": 0.09, "sadness": 0.04, "fear": 0.02,
      "anger": 0.02, "disgust": 0.01
    }
  },
  "fused": {
    "valence": 0.79,
    "arousal": 0.61,
    "dominance": 0.55,
    "top_label": "joy"
  },
  "model_versions": {
    "asr": "speechanalyzer/ja_JP",
    "acoustic_dim": "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim@onnx",
    "acoustic_cat": "emotion2vec/emotion2vec_plus_large",
    "text": "deberta-v3-large/wrime-takenaka-2025"
  }
}
```

## Notes

- `asr_confidence` is in [0, 1]; absent if the underlying transcriber doesn't
  expose one. It propagates into fusion weights — see `Core/Fusion/LateFusion`.
- Both categorical blocks are normalized; missing labels imply 0.
- `model_versions` is required for every record so retrospective re-fusion is
  possible if a model is swapped (and `docs/eval_log.md` is updated).
