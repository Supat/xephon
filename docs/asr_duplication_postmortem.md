# ASR character duplication post-mortem

## Symptom

`SpeechTranscriber` finals contained duplicated characters that the user
clearly hadn't said. Example from a real session:

```
FINAL [3.96–9.18s]   "明日横浜マナに行く。"        ← clean
FINAL [16.68–22.62s] "明日明日横横浜前に行く。"   ← "明日" and "横" doubled
```

Duplicates were always *internal* to a single final's text (not two finals
yielding overlapping ranges), and the doubled phonemes consistently
matched phonemes the model had heard earlier in the same session.

## Wrong hypotheses, in order

I went through several incorrect theories before finding the bug. Each
shaped a fix that didn't hold up. Recording them here because the same
shape of failure could trip up future debugging.

### 1. "SpeechTranscriber emits overlapping finals; we need to dedup."

The original `StreamingSpeechAnalyzerTranscriber` keyed pending results
by `range.start.seconds`, gated emission on
`volatileRangeChangedHandler`, and tried to drop "split" revisions. I
added a `lastEmittedEnd` high-water mark, then a flush-time overlap
guard, then append-time containment checks. None of these helped because
the duplication wasn't in the *ranges* — it was in the *text content of
a single final*.

### 2. "We're using the wrong reportingOption / missing isFinal."

The `SpeechTranscriber.Result` swiftinterface lists `range`,
`resultsFinalizationTime`, `text`, `alternatives` — no `isFinal`. I
assumed `isFinal` didn't exist and that `volatileRangeChangedHandler`
was the only finalization signal. **Wrong:** `isFinal` is a default
extension on the `SpeechModuleResult` *protocol*, so every
`SpeechTranscriber.Result` has it. Verified with FluidInference's
swift-scribe (`Scribe/Transcription/Transcription.swift:93–105`), whose
canonical loop is:

```swift
for try await case let result in transcriber.results {
    if result.isFinal {
        finalizedTranscript += result.text
    } else {
        volatileTranscript = result.text
    }
}
```

I rewrote the transcriber to that pattern with
`reportingOptions: [.volatileResults]`. Cleaner code, but **the
duplication persisted**, which finally pointed the audit downstream of
the transcriber.

### 3. "Apple is redelivering or emitting cumulative finals."

Wrote three speculative dedup branches in `handleResult` (redelivery,
suffix-only emission, generic overlap drop). All speculation; no actual
evidence Apple does any of those things. With diagnostic logging
enabled, I asked for real data — that's when the screenshot below
landed.

## What the data showed

The volatile previews grew like this for utterance 2 (real captured
log):

```
vol [9.18–23.288s] "明"
vol [9.18–23.288s] "明日"
vol [9.18–23.288s] "明日明日"           ← duplication appears HERE,
vol [9.18–23.288s] "明日明日横"            during volatile growth
vol [9.18–23.288s] "明日明日横横"
vol [9.18–23.288s] "明日明日横横浜前に行く。"
```

The duplication was already present in the volatile previews. The model
was faithfully transcribing audio that contained doubled phonemes. So
the bug had to be upstream of the transcriber — in what we were feeding
it.

## Actual root cause

`Core/Audio/AudioCapture.swift`, the per-tap resampler input block:

```swift
// BUGGY
let outputCapacity = AVAudioFrameCount(Double(buffer.frameLength) * sampleRateRatio) + 1024
guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: outputCapacity)
    else { return }
let inputBlock: AVAudioConverterInputBlock = { _, status in
    status.pointee = .haveData
    return buffer
}
let result = converter.convert(to: outBuffer, error: &convError, withInputFrom: inputBlock)
```

Two compounding mistakes:

1. **`outputCapacity` was way too large.** `inputFrames × ratio + 1024`
   for 48 kHz → 16 kHz with a 4096-frame input asks for 2390 output
   frames; one input only produces ≈ 1365. The converter saw extra
   capacity and immediately called the input block again.
2. **The input block always answered `.haveData` with the same
   `buffer`.** So the converter spliced in the same audio twice, then
   stopped. The 2390-frame output contained the input audio doubled.

`SpeechTranscriber` was correctly transcribing what we sent it: the
sound `"明日"` repeated twice in a row, the sound `"横"` repeated twice
in a row.

## Fix

Adopted the canonical pattern from FluidInference's `BufferConverter`
(`Scribe/Helpers/BufferConversion.swift:42–50`):

```swift
let expectedOutputFrames = AVAudioFrameCount(
    (Double(buffer.frameLength) * sampleRateRatio).rounded(.up)
)
guard let outBuffer = AVAudioPCMBuffer(pcmFormat: outputFormat, frameCapacity: expectedOutputFrames)
    else { return }

final class Once: @unchecked Sendable { var fired = false }
let once = Once()
let inputBlock: AVAudioConverterInputBlock = { _, status in
    if once.fired {
        status.pointee = .noDataNow
        return nil
    }
    once.fired = true
    status.pointee = .haveData
    return buffer
}
```

Three things matter:

- **`outputCapacity = ceil(inputFrames × ratio)`**, no padding. The
  converter has no extra room to invite a refill.
- **`.noDataNow`, not `.endOfStream`.** `.endOfStream` *permanently*
  finalizes the persistent converter; subsequent tap callbacks find a
  dead converter and produce zero frames (this trapped me on attempt
  #1 of the fix — the user saw "frozen at 1600 samples"). `.noDataNow`
  tells the converter "no more input *for this call*" while keeping the
  resampler tail state alive for the next call.
- **`primeMethod = .none`** on the converters. Skips the resampler's
  amplitude-ramp primer that would otherwise put a leading glitch on
  every chunk.

Removed all the speculative dedup logic in
`StreamingSpeechAnalyzerTranscriber` once we knew the transcriber
itself was clean — defensive workarounds for non-existent problems made
the real bug harder to see.

## Lessons

- **A persistent `AVAudioConverter` for streaming SRC needs the
  `.noDataNow` pattern.** `.endOfStream` is one-shot.
  `convert(to:from:)` doesn't replace the block API for SRC: it has a
  hidden `outputCapacity ≥ inputBuffer.frameLength` precondition that
  crashes during downsampling, and the docs explicitly send you to the
  block API for "variable input frame counts (e.g. variable rate,
  sample-rate conversion)."
- **Diagnose with data, not speculation.** Three rounds of speculative
  dedup wasted real time. As soon as I logged what
  `transcriber.results` actually emitted, the bug pointed unambiguously
  upstream of the transcriber within minutes.
- **Look outside the failing component.** The duplication was in the
  audio path; everyone reaches for the ASR code first because that's
  where the duplicate text *appears*. Audit upstream early.
- **Mirror canonical reference implementations.** swift-scribe's
  `BufferConverter` had the exact pattern we needed. Reading reference
  code for an unfamiliar Apple framework is cheaper than rediscovering
  the API contract through trial and error.

## Files touched

- `Core/Audio/AudioCapture.swift` — the actual fix.
- `Core/ASR/StreamingSpeechAnalyzerTranscriber.swift` — rewritten to
  the canonical `isFinal` pattern; speculative dedup removed.
