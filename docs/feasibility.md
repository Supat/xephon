# Building a Japanese ASR + Multimodal Emotion-Recognition iPad App in 2026: A Feasibility & Architecture Report

## TL;DR

- **It is feasible to build this entirely on-device on an M4/M5 iPad Pro for research use**, but the recommended hybrid stack is: **Apple `SpeechAnalyzer`/`SpeechTranscriber` (iOS/iPadOS 26)** as the primary Japanese ASR (a Japanese model ships in the system asset catalog), with **WhisperKit + Kotoba-Whisper-v2.0/Distil-Whisper** or **FluidAudio Qwen3-ASR** as a fallback for higher-accuracy long-form / domain-mismatched audio; **audeering's wav2vec2-large-robust-12-ft-emotion-msp-dim** (ONNX→Core ML) for dimensional valence/arousal/dominance from acoustics; **emotion2vec+ large** (categorical, language-agnostic, trained with Japanese data) for categorical labels from acoustics; and a **WRIME-fine-tuned Japanese DeBERTa-v3-large** (or BERT-base) for text emotion. Late fusion via weighted averaging of audio-categorical, audio-dimensional, and text-categorical outputs is the pragmatic starting point.
- **Realistic accuracy ceiling fully offline on iPad Pro M4/M5 in 2026:** ~5–10% CER on clean read Japanese (Kotoba-Whisper-v2.0/SpeechTranscriber territory) and ~12–25% CER on noisy/spontaneous conversation; SER categorical accuracy ~55–70% on 7-class (vs. ~75–80% supervised in-domain), valence CCC ~0.55–0.65, arousal CCC ~0.65–0.75. Text WRIME 8-emotion macro-F1 ~0.55–0.65 (reader-perspective). The audio-side numbers on Japanese are *softer* than English because every public Japanese SER corpus (JTES, OGVC, STUDIES, JVNV) is small and stylistically narrow.
- **Effort estimate:** ~**6–10 person-weeks** for a single competent Swift/ML developer to build a usable research prototype; **3–5 person-months** for a polished, well-evaluated tool with diarization, custom evaluation harness, and reasonable UI. The biggest unknown unknowns are (a) Japanese pitch-accent vs. emotional-prosody confusion, and (b) the conversational/read-speech mismatch between every public Japanese emotion dataset and your actual research recordings — budget time for domain fine-tuning or at least a small calibration set.

---

## Key Findings

1. **iPadOS 26's `SpeechAnalyzer` / `SpeechTranscriber` is the new default for Japanese ASR.** `SpeechTranscriber.supportedLocales` explicitly returns `ja_JP`. It is on-device only, optimized for long-form conversational audio, and has no 1-minute limit (unlike legacy `SFSpeechRecognizer`). Apple's model is reported to be ~2× faster than Whisper Large-v3 Turbo on macOS with no perceptible quality drop in English long-form tests (MacStories' "Yap" benchmark). However, **`SpeechTranscriber` requires a 16-core Neural Engine** — confirmed for M4/M5 iPad Pro, but not for older iPads; it also does not run in the simulator. There is no public Japanese accuracy benchmark from Apple, so you should plan to evaluate it against your own data.

2. **Kotoba-Whisper-v2.0 is the strongest open ASR specifically for Japanese.** It is a distilled Whisper-large-v3 (full encoder + 2-layer decoder, 756M params, 6.3× faster than large-v3) trained on 7.2M ReazonSpeech clips, and matches or beats large-v3 on JSUT-basic5000 and Common Voice 8 ja. **ReazonSpeech-k2-v2** (Next-gen Kaldi, ONNX) and **ReazonSpeech-NeMo-v2** (FastConformer-RNNT, 619M, long-form) are the other top-tier Japanese-native models. Kotoba-whisper-v2.2 adds integrated punctuation and diarization in the HF pipeline.

3. **WhisperKit (Argmax) is the most production-ready Whisper stack on Apple Silicon**, with Core ML conversions of Whisper tiny/base/small/medium/large-v3/large-v3-turbo and self-distilled `d750` variants, ANE-optimized. Their published WhisperKit paper benchmarks Japanese CER on Common Voice 17 — Japanese is one of the languages where the self-distilled turbo is reported to retain accuracy. WhisperKit lacks Custom Vocabulary in Apple's new SpeechAnalyzer; Argmax's own SDK does support it.

4. **FluidAudio is the most attractive Apple-native multi-feature audio SDK.** It bundles Core ML versions of (a) Parakeet TDT v3 (0.6B, 25 European languages — *not Japanese*), (b) **Parakeet TDT 0.6B Japanese** (a hybrid CTC/TDT CoreML port with a published JSUT-basic5000 CER benchmark), (c) **Qwen3-ASR** for excellent Japanese/Chinese/Korean/Vietnamese transcription on macOS 15+ / iOS 18+, (d) **Pyannote-derived speaker diarization on the ANE**, (e) Sortformer (≤4 speakers, very stable) and LS-EEND (≤10 speakers, lighter) diarizers, and (f) Silero VAD. It is Apache 2.0 and runs on iOS 17+ Apple Silicon. ~110× RTF on M4 Pro for batch ASR.

5. **For dimensional emotion (valence/arousal/dominance), the field standard is audEERING's `wav2vec2-large-robust-12-ft-emotion-msp-dim`** — pruned from 24 to 12 transformer layers, fine-tuned on MSP-Podcast v1.7, available as an ONNX export on Zenodo. It outputs A/D/V in [0,1]. audEERING's 2024 *Wav2Small* paper distills the same teacher into 72k–MobileNet-scale students (valence CCC = 0.676 on MSP-Podcast — a new SoTA at publication), making sub-1MB on-device deployment realistic.

6. **For categorical SER, `emotion2vec+ large` (~300M, Apache 2.0) is the most defensible language-agnostic choice.** It is explicitly trained as a "Whisper of speech emotion," tested across 13 datasets in 10 languages, with a 9-class output (angry, disgusted, fearful, happy, neutral, other, sad, surprised, unknown). For Japanese-specific fine-tunes, the only well-known public model is `Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition` (XLSR-53, ~300M, fine-tuned on JTES). It is a research demo, not production-grade.

7. **For Japanese text emotion, the 2025 SoTA on WRIME 8-Plutchik binary classification is fine-tuned DeBERTa-v3-large** (mean accuracy 0.860, mean F1 0.662, Takenaka 2025), substantially beating ChatGPT-4o (F1 0.527) and TinySwallow-1.5B (F1 0.292) — i.e., for this task a dedicated fine-tuned encoder is still better than a small LLM. `patrickramos/bert-base-japanese-v2-wrime-fine-tune` is a well-cited intensity-regression baseline. WRIME-ver2 also includes 5-point sentiment polarity, which maps nicely to a valence axis.

8. **Apple's Foundation Models framework (iOS/iPadOS 26)** exposes the on-device ~3B Apple Intelligence LLM with Japanese support, free at inference time, with guided generation and tool calling — usable as a zero-cost text-emotion classifier via prompt + structured output, but expect quality below a fine-tuned WRIME classifier. Requires Apple Intelligence-capable device.

9. **MLX runs on iPad** (Swift API supports macOS, iOS, iPadOS, visionOS). **For your M4-vs-M5 question:** the M5 GPU's per-core Neural Accelerators deliver Apple's claimed 4× LLM prompt-processing improvement, *but only when the MLX/Metal stack actually uses the new TensorOps/MPP path*. As of MacStories' M5 iPad Pro reviews (Oct/Nov 2025 → Mar 2026), early MLX builds on iPad showed only ~1.2× generation speedup (12.2 vs 10.2 tok/s on Qwen3-8B-4bit) until a later Metal-accelerated MLX branch landed; with that, prompt-processing/TTFT gains on iPad now match Apple's claims, but token-generation is still memory-bandwidth-bound and the M5 iPad Pro does not have the M5 Pro/Max's 307–614 GB/s bandwidth jump (those are MacBook Pro chips, not iPad). Net: **for this app, M4 vs M5 iPad Pro is a small (~1.2–2×) difference for ASR/SER/text inference; the much bigger lever is *which* models you pick.**

10. **Realistic on-device hybrid recommendation:** ASR → SpeechTranscriber primary, Kotoba-Whisper-v2.0 (WhisperKit Core ML) as fallback. Diarization → FluidAudio Sortformer/LS-EEND. Acoustic emotion → audeering ONNX (dimensional) + emotion2vec+ (categorical) via ONNX Runtime CoreML EP or Core ML conversion. Text emotion → fine-tuned DeBERTa-v3-large (WRIME) converted to Core ML, optionally backed up by Apple Foundation Models framework for zero-shot/structured-output reasoning. Fusion → late, weighted, with a calibration set on your own data.

---

## Details

### 1. Japanese ASR options for iPadOS / macOS

**Apple `SFSpeechRecognizer` (legacy).** Supports Japanese both server- and on-device. Hard limits: ≤1 minute audio per request, 1,000 requests/device/hour, no streaming beyond ~1 min, `requiresOnDeviceRecognition = true` works for ja_JP on devices with the Japanese dictation pack downloaded. Includes `SFVoiceAnalytics` (pitch, jitter, shimmer per segment) — a free side-channel of acoustic features that you may want to log even if you don't use it for emotion. **Verdict for your app: not adequate as the primary ASR for a research tool processing multi-minute conversational audio, but the `SFVoiceAnalytics` outputs and the `contextualStrings` phrase-bias feature (~100 phrases) are useful.**

**Apple `SpeechAnalyzer` / `SpeechTranscriber` (iOS/iPadOS 26+).** Modular API (`SpeechAnalyzer` coordinator + `SpeechTranscriber`, `DictationTranscriber`, `SpeechDetector` modules). On-device only, no session limit, optimized for long-form conversational and lecture-style audio. **Japanese (`ja_JP`) is in `SpeechTranscriber.supportedLocales`** along with `yue_CN`, `zh_CN/HK/TW`, `ko_KR`, etc. Locale assets are downloaded on demand into a system asset catalog (zero app-bundle bloat; shared across apps). Caveats: model needs initial download; not available on simulator; requires a 16-core Neural Engine (so M-series iPad Pro is fine, A14/A15 iPads are not). No public WER/CER number for Japanese yet; Argmax's WhisperKit team benchmarked English long-form (Earnings22) and reports it sits at "mid-tier Whisper" speed/accuracy. **Verdict: this should be your default ASR.** Argmax also notes that lack of Custom Vocabulary is a significant gap for domain terms — your research domain may need a re-scoring pass.

**WhisperKit (Argmax, Swift Package).** Production-grade Core ML port of Whisper, ANE-optimized, supports tiny/base/small/medium/large-v2/v3/turbo and Argmax's self-distilled `d750` Whisper-Large-v3-Turbo. Repository `argmaxinc/WhisperKit`. They publish a HuggingFace benchmarks dashboard (`argmaxinc/whisperkit-evals-dataset`). Their paper explicitly benchmarks Japanese on a Common Voice 17 subset; the d750 turbo retains usable Japanese CER. WhisperKit Pro (Argmax SDK, paid) adds Custom Vocabulary, real-time speakers, and frontier accuracy with ~5× higher transcription speed than SpeechAnalyzer per Argmax's own benchmarks. **Verdict: best fallback when SpeechAnalyzer underperforms or when you need word timestamps and detailed control.**

**Kotoba-Whisper.** Best Japanese-specific Whisper variant. v1.0 (1,253h ReazonSpeech), v2.0 (full ~7.2M clips ReazonSpeech), and v2.2 (adds diarization + punctuation in the HF pipeline). Architecture: full Whisper-large-v3 encoder + 2-layer decoder, 756M params, 6.3× faster than large-v3, lower CER/WER than large-v3 on in-domain ReazonSpeech test, competitive on JSUT-basic5000 and Common Voice 8 ja out-of-domain. Apache 2.0. Convertible to Core ML / WhisperKit via standard distil-whisper ONNX → coremltools or `whisper.cpp` ggml → Core ML encoder. **Verdict: this is the model to deploy when SpeechAnalyzer's accuracy is insufficient.**

**ReazonSpeech ecosystem.** `reazonspeech-nemo-v2` (619M FastConformer-RNNT with Longformer attention, supports several-hour audio) is the long-form champion but requires NVIDIA NeMo — *not iPad-friendly without significant porting*. `reazonspeech-k2-v2` is the more iPad-relevant: Next-gen-Kaldi / Icefall, distributed in ONNX, int8-quantized variant available, designed to run on-device without GPU. Apache 2.0. **Verdict: k2-v2 is a strong second-or-third-place option via sherpa-onnx.**

**`whisper.cpp`.** Mature C/C++ port with iOS support, Metal decoder, Core ML encoder, Q5/Q6/Q8 quantization. ScribeAI is a public iOS reference. Japanese support is the same as upstream Whisper (so noisy on conversational unless you use a Japanese fine-tune). Some users report Japanese hallucinations in `large-v3` consistent with the well-known Whisper-v3 hallucination issue (Deepgram's analysis, openai/whisper#1762). **Verdict: viable but Kotoba-Whisper via WhisperKit is generally a better choice in 2026.**

**MLX-Whisper / `swift-parakeet-mlx`.** MLX Swift implementations exist; FluidInference's own `swift-parakeet-mlx` was deprecated by them in favor of FluidAudio's Core ML path because MLX was "quite resource-intensive" in production on iOS. **Verdict: use MLX for LLMs, not Whisper, on iPad.**

**FluidAudio Parakeet/Qwen3-ASR.** FluidInference's `parakeet-tdt-0.6b-v3` is multilingual but covers 25 European languages — *not Japanese*. They have a **separate Japanese-specific Parakeet TDT 0.6B** Core ML port with a published JSUT-basic5000 CER benchmark. They also offer **Qwen3-ASR** in Core ML with explicit "excellent Japanese, Chinese, Vietnamese support" — the most attractive non-Whisper option. macOS 15+ / iOS 18+ required for Qwen3-ASR. **Verdict: very strong candidate; benchmark Qwen3-ASR vs Kotoba-Whisper-v2.0 on your own conversational data before committing.**

**Vosk / Kaldi.** Vosk has a small Japanese model. Quality is well below modern Whisper/Conformer systems. Useful only for ultra-low-power streaming. **Verdict: not recommended for research-quality transcription.**

**Cloud APIs as fallback.** For Japanese specifically, recent third-party comparisons (Paul Kuo, "6.4%: Pushing Japanese Speech Recognition Accuracy") show **Google Chirp 3 substantially outperforming Whisper variants on Japanese business conversation** — average CER 13.5% vs. Groq Whisper 47.8% across four scenarios; Chirp 3 also exposes a Speech Adaptation feature accepting up to 5,000 phrases for terminology biasing, which is critically useful in domain-specific Japanese (technical/medical/business). OpenAI Whisper API, Deepgram Nova-3, AssemblyAI, AWS Transcribe, and Azure all support Japanese; Deepgram and AssemblyAI typically focus on English. **Verdict: if you ever need a cloud fallback for high-stakes accuracy, Google Chirp 3 with Speech Adaptation is currently the strongest Japanese cloud option.**

### 2. Speech Emotion Recognition (acoustic side)

**State-of-the-art in 2025/2026.** SER has decisively shifted to two paradigms: (a) self-supervised foundation models (wav2vec 2.0, HuBERT, WavLM) fine-tuned on dimensional A/D/V regression, evaluated by Concordance Correlation Coefficient (CCC), and (b) emotion-specific pretraining (`emotion2vec`). The audEERING *Wav2Small* paper (Kounadis-Bastian et al., 2024) sets the SoTA at **valence CCC = 0.676 on MSP-Podcast** with the wav2vec2-large teacher; their distilled 72k-parameter Wav2Small student is the smallest competitive model and is explicitly designed for low-resource deployment.

**Recommended models for your app:**

- **`audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`** — outputs valence, arousal, dominance in ~[0,1]. Pruned from 24 to 12 transformer layers (~165M params). ONNX export available on Zenodo (`doi:10.5281/zenodo.6221127`). Trained on MSP-Podcast (English), but the model has been used cross-lingually with reasonable CCC, including Japanese in third-party reports — it captures language-agnostic prosodic/paralinguistic features. **Convert ONNX → Core ML or run via ONNX Runtime CoreML EP.**
- **`emotion2vec/emotion2vec_plus_large`** (~300M) — categorical 9-class output (angry/disgusted/fearful/happy/neutral/other/sad/surprised/unknown), trained as a universal SER foundation model with explicit cross-lingual evaluation.
- **`Bagus/wav2vec2-xlsr-japanese-speech-emotion-recognition`** (~300M, XLSR-53 fine-tuned on JTES) — a Japanese-specific demo model. Useful as a third opinion or for in-domain fine-tuning with JTES/STUDIES/OGVC/JVNV; not necessarily better than emotion2vec+.

**Japanese SER datasets — be aware of the limitations.**

- **JTES (Japanese Twitter-based Emotional Speech)** — read speech of 50 phrases × 4 emotions × 100 speakers (acted, balanced).
- **OGVC (Online Gaming Voice Chat Corpus)** — spontaneous voice-chat speech labeled with Plutchik categories. Subsequent research finds OGVC has *low emotion recognizability in human forced-choice tests* (Xin et al. 2024 JVNV paper) — anger/fear/disgust are particularly hard to elicit in casual game chat. Only ~3% of utterances have unanimous annotator agreement. **Caveat: training-only on OGVC will likely yield disappointing real-world performance.**
- **STUDIES** — empathetic dialogue, 8 hours, 3 professional speakers, 4 emotions (anger/happiness/sadness/neutral). Acted, but conversational style.
- **JNV / JVNV (Japanese Verbal–Nonverbal Vocalizations)** — recent (2024) ChatGPT-script-generated emotional corpus *with* nonverbal vocalizations (laughter, sighs). Higher human emotion recognizability than JTES/OGVC/STUDIES.
- **UUDB (Utsunomiya University Dialogue DB)** — task-oriented dialogue with dimensional emotion labels.

The cross-corpus picture: Japanese SER datasets are an order of magnitude smaller than English MSP-Podcast/IEMOCAP and **biased toward acted or game-chat speech**, not natural research-interview speech. Plan to either (a) fine-tune a cross-lingual model (audeering W2V2 or emotion2vec+) on a small target-domain calibration set you collect and label yourself, or (b) accept that the open-domain output is a noisy estimate.

**Acoustic features that matter.** Modern foundation-model SER subsumes hand-engineered features, but you can additionally log a Geneva Minimalistic Acoustic Parameter Set (GeMAPS / eGeMAPS) via openSMILE-compatible code (or simpler proxies via AVAudioEngine + Accelerate vDSP): F0 mean/std/range, jitter, shimmer, HNR, energy/intensity, speaking rate (from word timings of the ASR), spectral tilt, MFCC stats. These help **interpretability** (Russell-style scatter plots, prosody-emotion correlation analysis for research), even if they do not improve a foundation-model classifier on top.

**Important Japanese-specific gotcha — pitch accent vs. emotional prosody.** Japanese is a lexical pitch-accent language: F0 contour distinguishes `hashi` (chopsticks, HL) from `hashi` (bridge, LH). Models that haven't seen Japanese during pre-training can confuse lexical-prosodic F0 movements with emotional-prosodic ones, especially for arousal. Empirically the audeering W2V2 model still works cross-lingually (its reasoning is paralinguistic, not lexical-tonal), but you should expect somewhat noisier F0-driven dimensions in Japanese than in English. This is one motivation for *also* using emotion2vec+ (which has been pre-trained on Mandarin and other tonal/pitch-accented languages) as a second opinion.

### 3. Text-based sentiment/emotion analysis for Japanese

**Best in 2025/2026 for WRIME 8-Plutchik:** Takenaka 2025 (arXiv:2505.00013) systematically compared BERT, RoBERTa, DeBERTa-v3-base, DeBERTa-v3-large, ChatGPT-4o, and TinySwallow-1.5B on WRIME with reader-perspective annotations. **DeBERTa-v3-large** wins with mean accuracy 0.860 and mean F1 0.662; the LLMs lag substantially (4o: 0.527 F1; TinySwallow: 0.292). The fine-tuned model is published as `pip install deberta-emotion-predictor`. Lesson: for this specific task, fine-tuned encoders > prompt-only LLMs, even capable cloud ones.

**Other strong options:**

- **`patrickramos/bert-base-japanese-v2-wrime-fine-tune`** — 0.1B, MSE 0.6 (writers) / 0.2 (readers) on WRIME intensity regression. Smaller, easy to deploy. Backbone: `cl-tohoku/bert-base-japanese-v2`.
- **`tohoku-nlp/bert-base-japanese-v3`** (CC-100 + Wikipedia, 392M+34M sentences, MeCab+UniDic+WordPiece). Best-quality public Japanese BERT base; many WRIME fine-tunes derive from it (e.g., `kynea0b/cl-tohoku-bert-base-japanese-v3-wrime-8-emotions`).
- **WRIME-ver2** also has 5-point sentiment polarity (-2..+2) — directly useful as a valence proxy for fusion.

**Datasets beyond WRIME:** ChABSA (aspect-based sentiment, financial domain), JSentiment, Amazon-review fine-tunes. None has the breadth of WRIME for emotion. ISEAR has been translated; SemEval-2018 Task 1 (intensity) is English-only. There is no high-quality Japanese GoEmotions equivalent at WRIME's scale.

**LLM-based approaches.** Apple's on-device Foundation Models framework (iPadOS 26) exposes ~3B-param Apple Intelligence LLM with Japanese support and structured-output guided generation; you can prompt for `{joy: float, sadness: float, ...}`. Expect quality somewhere between TinySwallow (F1 0.292) and 4o (F1 0.527), worse than a fine-tuned BERT/DeBERTa, but it is **free, on-device, and zero-setup**. **Recommended use: pair fine-tuned Japanese DeBERTa as primary; use Apple FM for second-opinion / explanation generation / open-ended affect description**, not as the classifier of record.

For local LLMs on iPad: Qwen3 (1.7B/4B/8B) runs via MLX-Swift; quantized 4-bit Qwen3-8B was the M4/M5-iPad-Pro reference workload in MacStories' MLXBenchmark (~10–12 tok/s). Llama 3.2 1B/3B and Gemma 3 1B/4B also run via llama.cpp. Japanese-specialized small LLMs include Sakana's TinySwallow-1.5B (worse on WRIME than fine-tuned BERT per Takenaka 2025) and Rinna/ABEJA/Calm-2 variants.

### 4. Multimodal fusion

**Research SoTA on IEMOCAP-style benchmarks** uses cross-attention fusion between wav2vec 2.0 and BERT representations (Sun et al. 2023, Zhao et al. 2022; UA ~79.7% on IEMOCAP with auxiliary tasks). MAVEN (2025, arXiv:2503.12623) extends this to bi-directional cross-modal attention over visual+audio+text and predicts polar (valence, arousal) coordinates following Russell's circumplex; CCC = 0.3061 on Aff-Wild2. WavFusion (2024, arXiv:2412.05558) uses wav2vec2 + gated cross-modal attention with multimodal-homogeneous-feature-discrepancy learning.

**For your app, you have three pragmatic fusion options:**

1. **Late, weighted late fusion (recommended starting point).** Run audeering W2V2 (V/A/D), emotion2vec+ (categorical softmax over 9 classes), and DeBERTa-WRIME (categorical softmax over 8 Plutchik + sentiment polarity for valence) independently. Map all to a common space (Plutchik 8 + V/A/D triplet), then fuse per-utterance with a learned linear/logistic combination calibrated on a held-out set you collect. Pros: trivial to implement, debuggable, models are independently swappable. Cons: ignores within-utterance cross-modal alignment.
2. **Score fusion with ASR confidence weighting.** Same as (1) but down-weight the text branch when ASR confidence is low (which `SpeechTranscriber` returns). Particularly useful for Japanese where ASR errors on emotional/disfluent speech can poison text emotion.
3. **Trained cross-attention fusion head.** Take wav2vec2 pooled embeddings + Japanese DeBERTa CLS embedding, train a small (~5–10M-param) cross-attention head on an in-domain set (STUDIES/JVNV + your own data). Best research-grade quality, but requires a labeled training set. Realistic only if you collect data anyway.

**Categorical → dimensional mapping (Russell's circumplex).** Map each Plutchik/Ekman category to a fixed (valence, arousal) anchor (e.g., happy ≈ +0.8 V, +0.6 A; angry ≈ −0.6 V, +0.8 A; sad ≈ −0.6 V, −0.4 A; neutral ≈ 0,0; fear ≈ −0.6V, +0.7A; surprise ≈ +0.2V, +0.8A; disgust ≈ −0.7V, +0.3A). For a categorical softmax distribution, output the soft-weighted sum of these anchors. This gives you a "secondary" dimensional estimate from the categorical models that you can sanity-check against the audeering W2V2 V/A/D regression — large disagreements are useful research signals.

**Language-specific multimodal SoTA references:** M3ED is Chinese; IEMOCAP is English; there is no Japanese IEMOCAP-equivalent. The closest is JTES + WRIME used in cross-modal experiments by individual labs, but no public benchmark leaderboard.

### 5. iPadOS / macOS implementation stack

**Audio capture.** `AVAudioEngine` with `installTap(onBus:)` is the right choice for buffered streaming (you get `AVAudioPCMBuffer` at any sample rate, then resample to 16 kHz mono Float32 for all the ML models). For file-based research workflows, `AVAudioFile` reads FLAC/WAV/M4A/MP3 directly. Microphone permission requires `NSMicrophoneUsageDescription` plus `NSSpeechRecognitionUsageDescription` (legacy) or just the microphone string (new Speech framework). For research, using a high-quality external USB or Lightning mic into the iPad reduces noise floor dramatically.

**Inference runtimes available on iPadOS 26:**

- **Core ML** (built-in) — best ANE utilization. Tools: `coremltools` 7+ for PyTorch/ONNX → mlmodelc; `ane-transformers` reference patterns from Apple.
- **MLX-Swift** — runs on iPad/iOS, full GPU/Metal path; the M5's per-GPU-core Neural Accelerators are exposed via Metal 4 TensorOps. Best for LLM workloads (Qwen3, Llama 3.2, Gemma 3) — see Apple's "Exploring LLMs with MLX and the Neural Accelerators in the M5 GPU" (Apple Machine Learning Research). MLX **does not** directly target the ANE; it runs on GPU+CPU. So for small classifiers, Core ML on ANE is more power-efficient; for LLMs, MLX-on-GPU is faster.
- **`llama.cpp` (ggml) Swift Package** — official `.binaryTarget` XCFramework on the GitHub release page (e.g., `llama-b5046-xcframework.zip`). Metal backend works on M-series iPads (older A12-class iPads have a Metal SIMD limitation noted in llama.cpp#2550). Stanford's SpeziLLM wraps it ergonomically.
- **`whisper.cpp`** — same ggml backend, mature iOS support (ScribeAI reference app).
- **WhisperKit** — Swift Package, Core ML, ANE-optimized.
- **FluidAudio** — Swift Package, Core ML on ANE, iOS 17+/macOS 14+.
- **ONNX Runtime mobile with CoreML EP** — supported via CocoaPods (`onnxruntime-c`, `onnxruntime-objc`), iOS 13+. Useful for shipping the audeering W2V2 ONNX directly without conversion. Performance: typically slightly worse than a hand-converted Core ML mlmodelc but much less integration work.

**MLX on iPad — practical caveats.** MLX Swift supports iPadOS officially (Apple WWDC25 session 315 explicitly demos iPadOS targets). The MLXBenchmark community app shows the M5 iPad Pro's measured early-software benefit was ~1.2× over M4 for token generation; the new Metal-accelerated MLX branch (post-March 2026) brings prompt-processing TTFT in line with Apple's 4× claim, but token-generation is bandwidth-bound (M5 iPad Pro is **not** M5 Pro/Max — 153 GB/s vs 307–614 GB/s). Practically: budget ~10–25 tok/s for 4-bit quantized 3–8B Japanese LLMs on M4/M5 iPad Pro, which is fast enough for utterance-by-utterance emotion classification but not for streaming dialogue.

**Memory / thermal envelope on M4/M5 iPad Pro.** 8 GB (256/512 GB SKUs) or 16 GB (1/2 TB SKUs) unified memory. For your stack: SpeechTranscriber model is downloaded by the system and shared (free in your budget); audeering W2V2 (~330 MB FP16) + emotion2vec+ (~600 MB FP16) + Japanese DeBERTa-large (~1.5 GB FP16) + Kotoba-Whisper-v2.0 (~1.5 GB FP16) → ~4 GB working set without an LLM. Add Qwen3-8B-4bit (~4.5 GB) and you're at ~8.5 GB peak, which is **borderline on 8 GB iPads** and comfortable on 16 GB. Recommendation: use the 16 GB iPad Pro variant. Thermal: M4/M5 iPad Pro can sustain ~80% of peak under continuous load before throttling; for batch processing of long recordings, expect throttling during multi-minute Whisper runs but not for short utterances.

**Diarization.** The single best fit on iPad in 2026 is **FluidAudio** with either its Sortformer (≤4 speakers, very stable, ANE-resident) or LS-EEND (≤10 speakers, lighter, M4 Max CPU-capable) backend. Both are based on pyannote-derived models converted to Core ML. The legacy `DiarizerManager` (online VBx) is also bundled. `sherpa-onnx` provides an ONNX Runtime path with Pyannote 3.1; this works on iPad but FluidAudio is more idiomatic Swift. `pyannote.audio` Python directly is server-side only.

### 6. Practical feasibility verdict

**Realistic accuracy ceiling, fully offline, M4/M5 iPad Pro, mid-2026:**

- **Japanese ASR (clean read):** SpeechTranscriber or Kotoba-Whisper-v2.0 — CER ~5–8% on JSUT-class clean speech; ReazonSpeech-k2-v2 is in the same ballpark.
- **Japanese ASR (spontaneous conversation):** ~12–25% CER depending on noise, overlap, dialect; Google Chirp 3 cloud would be ~10–14%; on-device Apple/Kotoba-Whisper realistically ~15–22%. Expect domain mismatch with research interviews — ReazonSpeech (TV) and Common Voice (read) don't fully cover counseling/clinical/research-interview style.
- **Acoustic emotion (categorical 7-class), cross-lingual model on Japanese:** ~55–70% accuracy out-of-the-box. With 1–3 hours of in-domain fine-tuning data: ~70–80%.
- **Acoustic emotion (dimensional V/A/D), cross-lingual:** valence CCC ~0.45–0.60, arousal CCC ~0.60–0.75 (arousal is easier and more language-agnostic). Wav2Small/audEERING-12 sets the upper bar at ~0.68 valence on MSP-Podcast (English in-domain).
- **Text emotion (WRIME 8-Plutchik binary):** ~0.66 mean F1 with DeBERTa-v3-large; on transcribed (not written) text expect 5–10 points lower.
- **Multimodal late fusion:** typically 3–6 points absolute improvement over the better single modality on validation; not guaranteed without a calibration set.

**Effort estimate (single competent iOS+ML developer):**

- **Weeks 1–2:** project setup, microphone capture, file ingest, AVAudioEngine pipeline, basic SwiftUI UI, integrate SpeechAnalyzer + Foundation Models + WhisperKit fallback toggle. Deliverable: Japanese transcription with timestamps.
- **Weeks 3–4:** integrate FluidAudio diarization + VAD; add WhisperKit + Kotoba-Whisper-v2.0 Core ML conversion; evaluate ASR accuracy on small in-domain set.
- **Weeks 5–6:** convert audeering W2V2 ONNX → Core ML (or run via ONNX Runtime CoreML EP); convert emotion2vec+ via coremltools; wire utterance-level emotion outputs.
- **Weeks 7–8:** convert / fine-tune Japanese DeBERTa-WRIME to Core ML; implement late fusion with calibration; categorical→dimensional mapping; build research export (CSV/JSON of per-utterance text, transcript, timestamps, speaker, audio-emotion, text-emotion, fused).
- **Weeks 9–10:** UI polish, evaluation harness, packaging for sideload via Xcode/TestFlight.

A **6-week minimum viable research prototype** is realistic; **3–5 months** for a polished tool with a custom diarization-aware fusion head and an evaluation report on collected data.

**Recommended concrete architecture:**

```
[iPad Pro M4/M5 — research app, sideloaded via Xcode/TestFlight]

  Audio in
    │ AVAudioEngine (16 kHz mono Float32 buffer)
    ▼
  ┌──────────────────────────────────────┐
  │ FluidAudio Silero VAD + Sortformer/  │
  │  LS-EEND diarization (Core ML, ANE)  │
  └────────┬─────────────────────────────┘
           │ per-speaker, per-utterance segments
   ┌───────┴─────────┐
   ▼                 ▼
 ASR           Acoustic SER
  │              ├─ audeering W2V2-12 ONNX → Core ML  ⇒ V/A/D
  │              └─ emotion2vec+ large (Core ML)      ⇒ 9-class softmax
  │
  ▼ (Japanese transcript with word timestamps)
 Apple SpeechAnalyzer / SpeechTranscriber (ja_JP)  ← primary
   │  (fallback) → WhisperKit Kotoba-Whisper-v2.0 (Core ML)
   │  (fallback) → FluidAudio Qwen3-ASR (Core ML)
   ▼
 Text SER
  ├─ DeBERTa-v3-large WRIME-8-Plutchik (Core ML)   ⇒ 8-class softmax
  └─ Apple Foundation Models (3B on-device)         ⇒ structured V/A explanation
   │
   ▼
 Fusion (late, calibrated weighted average)
   │
   ▼
 Output: per-utterance JSON
  { speaker, t_start, t_end, transcript_ja,
    audio_categorical: {…},  audio_VAD: {v,a,d},
    text_categorical: {…},
    fused_categorical: {…},  fused_VAD: {v,a,d},
    asr_confidence, prosody_features (F0, energy, rate) }
```

**Hybrid (cloud) recommendation.** Stay fully on-device for: SER (privacy of voice samples is a serious research-ethics concern, and cloud SER quality is not meaningfully better than on-device W2V2/emotion2vec+); diarization; text emotion. Consider cloud only for ASR fallback when SpeechTranscriber + Kotoba-Whisper both underperform on a particular dataset — Google Chirp 3 with Speech Adaptation is currently the strongest Japanese cloud STT and can take 5,000 phrases of domain biasing.

**Pitfalls and limitations specifically for Japanese:**

1. **Pitch accent vs. emotional prosody.** Lexical F0 movements in Japanese can be mis-attributed to arousal by acoustic models pre-trained mostly on English. Mitigation: use emotion2vec+ as a second opinion (it has more cross-lingual training); validate on JTES/STUDIES first.
2. **Conversational vs. read-speech mismatch.** All major Japanese ASR training data (ReazonSpeech: TV; Common Voice: read; JSUT: read) under-represents spontaneous conversation with disfluencies, fillers (えーと, あの), backchannels (うん, そう, はい). Whisper-class models are robust to this in English but less so in Japanese. Mitigation: collect 30–60 minutes of in-domain audio, transcribe carefully, evaluate WER/CER per system, and if needed fine-tune Kotoba-Whisper with LoRA on your data (a few thousand utterances suffice).
3. **Emotion datasets being small and stylistically narrow.** OGVC (game-chat), JTES (acted, read), STUDIES (acted dialogue), JVNV (script-generated). None is research-interview-style. Expect domain shift; budget time for either (a) a small calibration label-set you collect or (b) accepting noisy outputs as relative-not-absolute signals.
4. **Honorific/politeness register confounds.** Japanese politeness levels (敬語, です/ます vs plain) carry social-affective information that doesn't map cleanly onto Plutchik's 8 emotions. The text emotion classifier will under-detect "felt" emotions when the speaker uses formal register over hot affect, a well-known limitation of WRIME-trained models. Document this in your research write-up.
5. **`SpeechTranscriber` requires 16-core ANE.** Confirmed for M4/M5 iPad Pro; not for older iPads or A-series chips. Verify `SpeechTranscriber.isAvailable` at runtime and fall back to WhisperKit.
6. **Apple Foundation Models** requires Apple Intelligence to be enabled and is region-gated; check availability at runtime.
7. **No Custom Vocabulary in `SpeechAnalyzer` (yet).** If your research domain uses specialized terminology (clinical, legal, technical), expect mistranscriptions. Either keep `SFSpeechRecognizer` with `contextualStrings` for terminology-heavy short clips, or use Argmax SDK Pro / Google Chirp 3 (cloud) Speech Adaptation as fallback.
8. **MLX on iPad is GPU-bound, not ANE-bound.** Don't expect ANE-class power efficiency from MLX-resident LLMs. For long sessions plug in / actively cool the iPad. M5 iPad Pro vs M4 iPad Pro difference is ~1.2–2× depending on workload (MacStories' measured numbers; Apple's 3.5–4× claims apply more cleanly to Mac chips with bandwidth headroom).

---

## Caveats

- **Apple's `SpeechTranscriber` Japanese accuracy is not publicly benchmarked.** The Argmax/MacStories comparisons that report mid-tier-Whisper-equivalent quality are English-only. Plan for a custom Japanese evaluation on your own audio before committing to it as the primary.
- **The MacStories M5 iPad Pro benchmarks are early-software** (Oct 2025 review showed only 1.2× MLX speedup; March 2026 follow-up with Metal-optimized MLX branch shows the claimed 4× prompt-processing improvement). The exact numbers you'll see in mid-2026 will depend on which MLX/Metal/iPadOS build is current; recheck before committing to M5 over M4.
- **The Apple M5 Pro / M5 Max numbers (614 GB/s, 128 GB) reported by InsiderLLM and AIProductivity are for MacBook Pros**, not iPads. iPad Pro M5 has the base M5 with ~153 GB/s memory bandwidth. Don't conflate.
- **Some sources cited speak in marketing or future-tense terms.** Apple's "2× faster than Whisper Large-v3 Turbo" claim for SpeechAnalyzer comes from a single MacStories speed test on a 7 GB English video; quality parity was reported as "no noticeable difference" — a subjective, single-file judgment, not a WER measurement. Treat as directional, not authoritative.
- **The 6–10 person-week effort estimate assumes a developer already comfortable with Swift, Core ML, ONNX, AVAudioEngine, and at least one of WhisperKit/FluidAudio.** Add ~50% if any of those is new.
- **The "realistic accuracy ceiling" numbers are based on cross-corpus literature (audEERING Wav2Small CCC = 0.676; emotion2vec multi-language results; WhisperKit Common Voice ja CER reports; Takenaka 2025 WRIME F1 = 0.662; Paul Kuo 2025 Japanese STT comparison)** and represent upper bounds you might achieve. Real-world performance on uncontrolled research-interview audio may be 5–15 points worse and is unpredictable without a pilot evaluation. Always run a calibration on your own data before reporting findings.
- **Ethical and privacy considerations** for emotion analysis on identifiable voice samples are non-trivial and vary by jurisdiction (GDPR Art. 9 treats biometric and emotional inferences as special-category data; the EU AI Act restricts emotion recognition in some workplace/educational contexts). Even for "research use" outside the App Store, secure your IRB/ethics-board sign-off and keep audio on-device; this is one of the strongest arguments for the offline architecture above.