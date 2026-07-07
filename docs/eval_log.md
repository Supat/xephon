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
