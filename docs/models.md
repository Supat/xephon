# Models

For each model: source, license, conversion command, measured size and latency.
Fill rows in as `scripts/fetch_models.sh` is implemented and benchmarks run.

## ASR

| Model | Source | License | Conversion | Size | Latency (M4 iPad Pro) | Notes |
|---|---|---|---|---|---|---|
| SpeechTranscriber ja_JP | Apple system asset catalog | Apple SDK | n/a (system download) | n/a | n/a | Primary. Requires 16-core ANE; not on Simulator. |
| Kotoba-Whisper-v2.0 | `kotoba-tech/kotoba-whisper-v2.0` | Apache-2.0 | distil-whisper → ONNX → Core ML (or `whisper.cpp` ggml → CML encoder) | TBD | TBD | WhisperKit fallback. |
| Qwen3-ASR | FluidAudio Core ML port | Apache-2.0 (verify) | bundled by FluidAudio | TBD | TBD | iOS 18+. |

## Diarization

| Model | Source | License | Notes |
|---|---|---|---|
| Sortformer | FluidAudio | Apache-2.0 | ≤4 speakers, very stable. Default. |
| LS-EEND | FluidAudio | Apache-2.0 | ≤10 speakers, lighter. |
| Silero VAD | FluidAudio bundle | MIT | Voice activity detection. |

## SER (acoustic)

| Model | Source | License | Conversion | Notes |
|---|---|---|---|---|
| audeering W2V2-large-robust-12-ft-emotion-msp-dim | Zenodo `10.5281/zenodo.6221127` | CC-BY-NC 4.0 ⚠️ | ONNX → Core ML or ORT CoreML EP | Dimensional V/A/D. **Non-commercial; research-only fits this project.** |
| emotion2vec_plus_large | `emotion2vec/emotion2vec_plus_large` | Apache-2.0 | ONNX → Core ML | 9-class. Cross-lingual. |
| Bagus/wav2vec2-xlsr-japanese-SER | HF | research demo | optional | JTES-trained; third opinion only. |

## Speaker demographics (optional)

| Model | Source | License | Conversion | Notes |
|---|---|---|---|---|
| audeering W2V2-large-robust-6-ft-age-gender | `audeering/wav2vec2-large-robust-6-ft-age-gender` | CC-BY-NC 4.0 ⚠️ | ONNX → ORT CoreML EP (FP16-quantized via `scripts/quantize_onnx_fp16.py`) | Continuous age regression + 3-class gender softmax (female/male/child). Same W2V2 backbone as the V/A/D model so the existing ONNX-via-ORT path applies. Non-commercial; fits the research-only posture. Off by default — wire up when the demographics surface is built. |

## SER (text)

| Model | Source | License | Notes |
|---|---|---|---|
| DeBERTa-v3-large WRIME (Takenaka 2025) | `deberta-emotion-predictor` (pip) | TBD | Convert to Core ML for on-device. 8-Plutchik. |
| `patrickramos/bert-base-japanese-v2-wrime-fine-tune` | HF | TBD | Smaller alternative. |
| Apple Foundation Models 3B | iPadOS 26+ system | Apple SDK | Optional second opinion via guided generation. |

## Session summarizer (LLM)

| Model | Source | License | Notes |
|---|---|---|---|
| Qwen3-8B (4-bit MLX) | `mlx-community/Qwen3-8B-4bit` (HF direct) | Apache 2.0 | On-device, opt-in. Runs through `mlx-swift-examples` (`MLXLLM` + `MLXLMCommon`). ~4.6 GB on disk, ~5 GB resident. Requires the `com.apple.developer.kernel.increased-memory-limit` entitlement on the app target so iOS's per-app Jetsam ceiling on a 16 GB iPad lifts from ~5 GB to ~10–11 GB — without that, the Qwen weights + analysis pipeline trip Jetsam during prefill. Sourced from Hugging Face directly (GitHub Releases' 2 GB Free-tier asset cap rules out the safetensors). Developer-side pull via `scripts/fetch_models.sh --with-summarizer`. |

## License obligations to track

- audEERING dimensional W2V2 weights are **CC-BY-NC**. Compatible with
  research-only sideload; would block any commercial release.
- DeBERTa-v3-large base model is MIT; the WRIME fine-tune license depends on
  the publisher — verify before redistribution.
