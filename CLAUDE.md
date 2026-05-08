# CLAUDE.md — Japanese Conversational Affect Analyzer (iPadOS)

Persistent project context for Claude Code. Keep this file under ~200 lines.

## What this project is

An **iPadOS 26+ research app** (macOS 15+ as a secondary target) that takes
Japanese conversational audio, transcribes it, and produces **per-utterance
multimodal emotion estimates** — both **categorical** (Plutchik 8 / Ekman 7)
and **dimensional** (valence, arousal, dominance).

- **Status:** research-only. Sideload via Xcode / personal TestFlight. **Not** App Store.
- **Privacy posture:** fully on-device by default. Cloud APIs are an opt-in fallback.
- **Primary device:** iPad Pro M4/M5 (16 GB SKU strongly preferred).
- **License of the source code:** TBD (do not assume MIT).

## Architecture (the canonical pipeline)

```
AVAudioEngine (16 kHz mono Float32)
  → FluidAudio: Silero VAD + Sortformer/LS-EEND diarization (Core ML, ANE)
  → ASR
       primary:   SpeechAnalyzer + SpeechTranscriber (ja_JP, on-device)
       fallback:  WhisperKit + Kotoba-Whisper-v2.0 (Core ML)
       fallback:  FluidAudio Qwen3-ASR (Core ML, iOS 18+)
  → Acoustic SER (run in parallel on the same segments)
       audeering wav2vec2-large-robust-12-ft-emotion-msp-dim  → V/A/D
       emotion2vec_plus_large                                  → 9-class softmax
  → Text SER
       fine-tuned Japanese DeBERTa-v3-large on WRIME           → 8-Plutchik
       (optional) Apple Foundation Models 3B                   → structured V/A
  → Late fusion (weighted, ASR-confidence-aware)
  → Per-utterance JSON export (see `docs/output_schema.md`)
```

Late fusion is the default. **Do not** introduce a trained cross-modal head
without an in-domain calibration dataset and explicit go-ahead.

## Repository layout

```
Xephon/               SwiftUI app target (iPadOS; "Designed for iPad" on macOS)
Core/
  Audio/              AVAudioEngine capture, resampling, VAD wrapper
  ASR/                SpeechAnalyzer wrapper, WhisperKit adapter, Qwen3 adapter
  Diarization/        FluidAudio adapter
  SER/
    Acoustic/         W2V2 + emotion2vec inference (Core ML / ONNX)
    Text/             DeBERTa-WRIME inference, Foundation Models adapter
  Fusion/             Late-fusion logic, calibration, V/A/D mapping
  Export/             JSON / CSV writers
Models/               *.mlpackage and *.onnx (git-lfs, see below)
Tests/
  UnitTests/
  EvalTests/          Reference WER/CER and SER metrics on small held-out set
docs/
  feasibility.md      The architecture & feasibility report
  output_schema.md    Per-utterance JSON schema
  models.md           Each model's source, license, conversion command
```

## Build & run

`Xephon.xcodeproj` is gitignored and generated from `project.yml` by
[XcodeGen](https://github.com/yonaskolb/XcodeGen). Regenerate any time
`project.yml` or the file layout under `Xephon/` changes.

```bash
# One-time tooling
brew install xcodegen git-lfs
git lfs install

# Generate Xephon.xcodeproj from project.yml
./scripts/generate_project.sh

# Open in Xcode (Xcode 16+, Swift 6, iOS/iPadOS 26 SDK)
open Xephon.xcodeproj

# CLI build
xcodebuild -scheme Xephon -destination 'platform=iOS,name=Sui iPad Pro' build

# Run unit tests on macOS (Catalyst); eval tests need physical iPad device
xcodebuild -scheme Xephon -destination 'generic/platform=iOS Simulator' test

# Hydrate Core ML / ONNX models (do this once — large download)
./scripts/fetch_models.sh
```

Models are **not committed** to git — `fetch_models.sh` hydrates them from
Hugging Face and runs `coremltools` conversions where needed. The git-lfs
filter in `.gitattributes` is defense-in-depth in case a `.mlpackage` ever
lands in the tree.

## Dependencies

Swift Package Manager only. Pinned versions live in `Package.resolved`.

- `argmaxinc/WhisperKit`
- `FluidInference/FluidAudio`
- `ml-explore/mlx-swift` (only if/when an LLM path is enabled)
- `microsoft/onnxruntime-swift-package-manager` (CoreML EP)

System frameworks: `Speech`, `AVFoundation`, `CoreML`, `Accelerate`,
`FoundationModels` (iPadOS 26+), `SwiftUI`.

## Critical constraints — read before changing the pipeline

1. **`SpeechTranscriber` requires a 16-core Neural Engine** (M-series iPad Pro).
   Always check `SpeechTranscriber.isAvailable` and degrade gracefully.
2. **`SpeechTranscriber` does not run in the iOS Simulator.** Eval tests that
   exercise it must be gated to physical-device or macOS targets.
3. **No Custom Vocabulary in `SpeechAnalyzer`.** For domain terminology,
   re-score with `SFSpeechRecognizer.contextualStrings` on short clips, or
   fall back to WhisperKit/Qwen3-ASR.
4. **All ML inference targets the ANE first**, GPU second, CPU last. Use Core
   ML `MLComputeUnits.cpuAndNeuralEngine` unless a model is empirically faster
   on `.all`.
5. **Audio never leaves the device** unless the user toggles the explicit
   "Cloud ASR fallback" switch. The toggle's state must be visible in the UI
   while recording.
6. **Sample rate is 16 kHz mono Float32 throughout the pipeline.** Resample at
   capture; never re-resample mid-pipeline.

## Japanese-specific gotchas

- **Pitch accent vs. emotional prosody.** F0 movements in Japanese carry
  lexical information; cross-lingual SER models can mis-attribute them to
  arousal. Use `emotion2vec+` as a second opinion alongside audeering W2V2.
- **All public Japanese SER datasets are small and stylistically narrow**
  (JTES = acted/read; OGVC = game chat; STUDIES = acted dialogue;
  JVNV = script-generated). Treat zero-shot outputs as relative, not absolute.
- **Politeness register confounds text emotion.** WRIME-trained classifiers
  under-detect strong affect when the speaker uses 敬語. Document this in
  any user-facing report.
- **Conversational vs. read-speech mismatch.** ReazonSpeech (TV) and Common
  Voice (read) under-represent disfluencies (えーと, あの) and backchannels
  (うん, そう). Expect higher CER than published benchmarks suggest.

## Conventions

- **Swift 6, strict concurrency.** All inference is `async`. No `@unchecked Sendable`
  without a comment explaining why.
- **One model = one actor.** Inference actors hold the `MLModel` instance and
  serialize calls. Never share an `MLModel` across actors.
- **Errors are typed.** Use `enum`-based errors per subsystem
  (`ASRError`, `SERError`, …). No `throws Error`.
- **No print().** Use `os.Logger` with per-subsystem categories.
- **Tests use real audio fixtures**, never synthesized tones, for ASR/SER paths.
  Fixtures live in `Tests/Fixtures/` (git-lfs, ≤30 s clips, consented).
- **UI strings are localizable** (`String(localized:)`); the app ships with
  `ja` and `en` localizations.

## Always / Never

- **Always** evaluate ASR/SER on the project's held-out calibration set after
  any model swap. Numbers go in `docs/eval_log.md`.
- **Always** record `asr_confidence` and propagate it into fusion weights.
- **Never** commit `.wav`, `.m4a`, or any participant audio to git.
- **Never** add a cloud provider without (a) an explicit UI toggle and
  (b) a privacy note in `docs/privacy.md`.
- **Never** fine-tune on a user's data silently. Fine-tuning workflows are
  out-of-band scripts, not in-app actions.

## Where to look for more context

- `docs/feasibility.md` — full architecture & model-selection rationale,
  realistic accuracy ceilings, effort estimates, citations.
- `docs/models.md` — for each model: source, license, conversion command,
  measured size and latency on M4/M5 iPad Pro.
- `docs/output_schema.md` — canonical per-utterance JSON.
- `docs/eval_log.md` — running CER/WER/CCC/F1 numbers per model swap.
