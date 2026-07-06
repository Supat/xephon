# Speech enhancement before ASR / diarization — research findings

*2026-07-03. Deep-research pass: 25 sources fetched, 117 claims extracted,
25 top claims adversarially verified (3-vote panels): 24 confirmed, 1 refuted.
Full citations inline.*

## Headline

**Generic single-channel neural denoising before modern noise-robust ASR and
before speaker-embedding extraction more often HURTS than helps.** The
dominant failure mode is enhancement *artifacts*, not residual noise.
Every backend Xephon uses or plans (SpeechAnalyzer, Parakeet-TDT, Qwen3-ASR,
WhisperKit) is in the noise-robust class where the measured effect of
denoising is negative. The current architecture — raw branch for SER,
optional speech-band EQ for ASR only — is already close to what the evidence
supports.

## Verified findings

### 1. Artifacts, not noise, degrade ASR — and "observation adding" fixes it *(high confidence, 3-0)*

Iwamoto et al. (NTT, Interspeech 2022 + TASLP 2024 extension) identified the
artifact component of enhanced speech as the main cause of post-enhancement
ASR degradation, and proved that **observation adding (OA)** — blending a
scaled copy of the raw signal back into the enhanced output — monotonically
increases signal-to-artifact ratio. Experimentally, OA at dry/wet ratios of
0.3–0.8 recovered ~20 % relative WER on CHiME-3. Independently corroborated
by de Oliveira et al. 2026.
→ *If a denoiser is ever fed to ASR, never use it fully wet.*
[arXiv:2201.06685](https://arxiv.org/abs/2201.06685) ·
[arXiv:2404.14860](https://arxiv.org/abs/2404.14860) ·
[arXiv:2605.12107](https://arxiv.org/pdf/2605.12107)

### 2. Denoising actively hurts noise-robust ASR — including Parakeet-TDT *(high, 3-0)*

On EARS-WHAM (SNR −2.5…10 dB), **all five** DNN enhancers tested reduced word
accuracy for **Parakeet TDT v2 (95.0 % noisy → 89.8 % best-enhanced, 73.4 %
worst)** and all four Whisper variants (Large v3 Turbo 94.1 % → 91.4 % best).
Separately, SAM-Audio denoising before zero-shot Whisper *increased* WER/CER
despite better signal-level quality metrics — acoustically cleaner audio is
recognized worse. Notable against-interest evidence: the Hamburg group
reports their own enhancer (SGMSE+) as the worst offender.
[arXiv:2605.12107](https://arxiv.org/pdf/2605.12107) ·
[arXiv:2603.04710](https://arxiv.org/abs/2603.04710) ·
[arXiv:2512.17562](https://arxiv.org/abs/2512.17562)

**Refuted (0-3):** "degradation scales with Whisper model size" — do not
carry that narrative.

### 3. The decision is per-backend, not global *(high, 3-0)*

The same enhancers that hurt Parakeet/Whisper substantially helped weak CTC
models without large-scale noisy training: QuartzNet 58.2 → 72.7 % WAcc,
wav2vec2 LARGE 70.2 → 81.1 %. All of Xephon's backends are in the robust
class → default expectation is harm, but per CLAUDE.md every swap must be
confirmed on the held-out Japanese calibration set.
[arXiv:2605.12107](https://arxiv.org/pdf/2605.12107)

### 4. Blind enhancement degrades speaker embeddings *(high, 3-0 / 2-1 on mechanism)*

DeepFilterNet3 before ECAPA-TDNN (VoxCeleb1+MUSAN): **EER 12.53 % → 19.08 %
at −10 dB**; at 0 dB the raw noisy signal beat enhanced for both embedding
models tested (ECAPA 3.31 vs 7.68 EER). Attributed to enhancement distorting
speaker-intrinsic characteristics (mechanism attribution survived 2-1).
→ *Do not denoise the input to LS-EEND/Sortformer or the FluidSpeakerDB
enrollment/verification path.*
[arXiv:2508.18913](https://arxiv.org/html/2508.18913v1) ·
[arXiv:1904.03601](https://arxiv.org/abs/1904.03601)

### 5. Speaker-CONDITIONED enhancement is the exception *(high, 3-0)*

Target-speaker enhancement **jointly trained** with the embedding network
(TASE-SVNet) cut EER 15.91 % → 6.02 % (+63.8 % relative) on heavily
overlapped speech and +30.3 % relative under vehicle noise. The gains
required joint training — an off-the-shelf TSE bolted onto a frozen verifier
was not what was measured. Relevant future direction, not a drop-in.
[arXiv:2103.08781](https://arxiv.org/abs/2103.08781) ·
[arXiv:1902.02546](https://arxiv.org/abs/1902.02546)

### 6. Diarization: denoising is a double-edged sword *(high, 3-0)*

Noisy classroom audio: denoising cut DER 50.5 → 36.7 % by slashing missed
speech (39.4 → 11.4 %) **but** raised false alarms (7.0 → 13.9 %) and
*deleted low-amplitude speech entirely — quiet speakers vanished*; in the
all-speaker configuration denoising **worsened** DER (71.3 → 82.2 %). The
authors' resolution: use denoising as **training-time augmentation**, keep
raw audio at inference. DIHARD III top systems credited enhancement without
an isolating ablation.
→ *At most, denoise a VAD/segmentation aid; attribute + embed from raw.*
[arXiv:2505.10879](https://arxiv.org/html/2505.10879v1) ·
[arXiv:2012.01477](https://arxiv.org/abs/2012.01477) ·
[arXiv:2103.10661](https://arxiv.org/abs/2103.10661)

### 7. Deployability: GTCRN is the only trivially shippable candidate *(high, 3-0)*

- **GTCRN**: 48.2 K params (23.7 K claimed + unlearnable ERB), 33 MMACs/s,
  MIT, pretrained DNS3/VCTK checkpoints, causal streaming variant RTF 0.07
  (i5 CPU), ONNX + sherpa-onnx (iOS-supported), **16 kHz native** — matches
  the pipeline. Caveats: repo reports only signal-quality metrics (no ASR/DER
  evidence), ANE residency unverified (GRU/dual-path ops may fall back to
  CPU/GPU under Core ML).
  [github.com/Xiaobin-Rong/gtcrn](https://github.com/Xiaobin-Rong/gtcrn)
- **DeepFilterNet v2/v3** *(medium, 2-1 on the rate constraint)*: real-time
  embedded-capable (~2.3 M params, MIT/Apache-2.0) but **48 kHz-only** —
  integrating it forces a 16→48→16 resample round-trip, violating the
  "resample at capture; never mid-pipeline" constraint unless it runs
  capture-side at 48 kHz before the single downsample. Rust/PyTorch/ONNX
  only; ERB/deep-filter ops are nontrivial to port to Core ML/ANE.
  [github.com/rikorose/deepfilternet](https://github.com/rikorose/deepfilternet)

## Per-branch recommendation

| Branch | Recommendation |
|---|---|
| **SER (emotion)** | **Strictly raw** — unchanged. Denoisers smooth exactly the spectral/prosodic detail (F0, formants, harmonics) W2V2/emotion2vec depend on; the speaker-distortion evidence generalizes to paralinguistic embeddings. |
| **ASR** | **Default raw** (plus the existing optional SpeechBoost EQ, which is user-gated and mild). If a denoiser is ever added for very-low-SNR sessions: GTCRN + observation-adding blend (grid 0.3–0.8 dry), per-backend, gated on held-out Japanese CER logged to `docs/eval_log.md`. Expectation going in: SpeechAnalyzer/Parakeet/Qwen3 will be *hurt* by fully-wet denoising. |
| **Diarizer / speaker embeddings** | **Raw at inference** — no denoising into LS-EEND/Sortformer or FluidSpeakerDB. If missed-speech in noisy sessions becomes a problem, explore a *denoised-VAD / raw-embedding split* (denoiser only informs segmentation boundaries). Training-time denoised augmentation only if the diarizer is ever fine-tuned (out-of-band per CLAUDE.md). |
| **Recording-to-disk** | Keep native/raw — it's the archival source for re-evaluation; any processing bakes artifacts in permanently. |

## Caveats (verified as part of the research)

- **All quantitative evidence is English or Mandarin.** No surviving claim
  reports WER/CER/DER deltas on conversational Japanese; pitch-accent and
  backchannel characteristics could shift the picture. The held-out-set eval
  rule is load-bearing.
- **The harm evidence comes from heavier enhancers.** GTCRN/RNNoise/DTLN-class
  ultra-light denoisers were *not* among those shown to hurt robust ASR — the
  direction is presumptive for them, not proven.
- **No surviving evidence at all** on Apple Voice Processing I/O / voice
  isolation, AGC/loudness normalization, dereverberation, speech-band EQ, or
  VAD-gated processing — those remain *unanswered*, not answered-negative.
  (Note: VPIO's AEC/AGC is also incompatible with `.measurement` mode
  semantics currently used at capture.)
- Several key sources are recent arXiv preprints, not yet peer-reviewed;
  GTCRN/DeepFilterNet performance numbers are self-reported and CPU-measured
  — no Core ML/ANE port or on-iPad latency measurement exists for any
  candidate.

## Open questions worth an in-house experiment

1. Does GTCRN (or Apple voice isolation) help or hurt SpeechAnalyzer ja_JP /
   Parakeet-ja / Qwen3 on the calibration set? (No published data; cheap A/B.)
2. Optimal OA dry/wet ratio per backend and SNR band — fixed vs
   estimated-SNR-adaptive.
3. Does a denoised-VAD / raw-embedding split reduce DER on iPad tabletop
   recordings with streaming EEND?
4. Does mild level normalization (as opposed to denoising) help or hurt the
   prosody-dependent SER branch? The "strictly raw" rule is currently
   asserted, not measured.
