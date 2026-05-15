# SpeechAnalyzer drift + long-session degradation post-mortem

Why file-mode analysis stops being trustworthy after ~20 min of audio,
and the buffer-pipeline rewrite that fixes it.

## Two distinct problems, stacked

**Drift.** ASR output timestamps slowly diverge from input
(file-clock) timestamps as a session runs. After 20–40 minutes of
audio the slip is large enough that an utterance's `[start, end]`
no longer aligns with the audio it was transcribed from — the
cumulative timeline strip drifts visibly, and SER segments resliced
from those timestamps capture the wrong region.

**Long-session degradation.** Independently, after ~20 minutes of
audio fed into a single `SpeechAnalyzer` session, finals start
arriving truncated. The same minute of audio that produced a
two-sentence final at 5:00 produces a six-character final at 25:00.
This is observable pre-fast-pace-removal too — it's a property of
the SpeechAnalyzer session itself, not of our pump.

These compound: a stalled transcriber emits sparse, short finals
*and* those finals carry increasingly wrong timestamps, so the
trailing portion of a long session is the most damaged.

## What "drift" actually is

`SpeechTranscriber` consumes 16 kHz PCM and reports `range` against
its own running output clock. Internally there's an
`AVAudioConverter` step (input-format → analyzer-format) and the
analyzer's own framing — neither produces an exact integer-ratio
mapping from input samples to reported output time. The error per
buffer is tiny; cumulative over many minutes it slips by hundreds of
milliseconds, then seconds.

For mic mode this barely matters because there's no objective
"file time" to drift relative to. For file mode it's a real bug:
the user knows the audio's true offsets, and the UI is supposed to
reflect them.

## The fix, in two parts

### 1. Anchor-based file-time mapping

`RollingAudioBuffer` (in `Core/Audio/`) carries a list of
`TimeAnchor(sampleIndex, fileTime)` records, one per appended chunk.
`indexForFileTime(_:)` and `fileTimeForIndex(_:)` interpolate
piecewise-linearly between anchors. All slicing / trimming /
snapshot-tail operations go through these helpers, so a slice
asked for `[60.0s, 65.0s]` returns audio that's actually at
`[60.0s, 65.0s]` on the user's file even if the writer's
sample-index clock has drifted.

`slice` returns `timestamp: actualStart = fileTimeForIndex(startIdx)`
rather than the requested start — defense against eviction (an
older anchor having rolled off).

Mic mode is unified with file mode by rebasing each engine
`sampleTime` to a session-relative origin
(`rawPumpBaseTimestamp` in `RecordingController`). The buffer then
sees the same monotonic, session-zeroed clock in both modes.

### 2. Periodic SpeechAnalyzer rotation

`StreamingSpeechAnalyzerTranscriber` rotates its
`SpeechAnalyzer` session every **10 minutes of audio fed** (not 10
minutes of wall time). Each analyzer has its own:

- `analyzer`, `inputCont`, `resultDrainer`, `analyzerStartTask`
- `cumulativeOutputFrames` — running output-time anchor
- `timeAnchors` — per-analyzer time-correction anchors
- `sessionAudioFedSeconds` — budget tracker

Session-level state (`outputCont`, `targetFormat`, `baseTimestamp`)
survives rotation. On rotation we drain finals from the old analyzer,
spin up a fresh one, and let the new analyzer's output clock continue
from the rotated anchor — so the rest of the pipeline sees one
continuous stream of finals with monotonically-increasing times,
even though three or four analyzer sessions handled the audio.

Each analyzer thus never accumulates more than ~10 minutes of
internal state, dodging the degradation entirely.

### Token-level time correction

Token times reported by the analyzer are corrected via
`correctedSeconds(...)` using the same per-analyzer anchors, so
word-level timing in finals lands on the right file offsets too.

## Why non-realtime

The original pipeline paced the file-mode pump with `Task.sleep`
keyed to real-time playback (audio was also being played back via
`AVAudioPlayer` for monitoring). Under that pacing, long files
*were* analyzed at real-time speed — 30 min of audio took 30 min of
wall-clock to finish. That made the drift more visible and the
degradation more punishing.

The rewrite (branch `bufferpipeline`, commit `f0b7d1b`):

- **Drops the pacing.** The pump emits chunks as fast as downstream
  consumers can swallow them, with `.bufferingOldest(64)` plus a
  retry-on-drop `yieldWithBackpressure` helper on both raw and
  processed streams. This applies backpressure naturally — if SER is
  slow, the pump waits.
- **Drops audio playback.** The `AVAudioPlayer` and its
  `muted` / `setFileAudioMuted` toggles are gone. The UI no longer
  needs to gate "audio is leaving the device" mid-recording for
  the file path.
- **Adds a rawTask backpressure ceiling.** Continuous diarization
  must not lag the pump by more than `maxDiarizeLagSeconds = 220`
  (~3.5 min); past that the pump pauses chunk yields until the
  diarize loop catches up. Combined with the catch-up loop firing
  per `continuousDiarizeStrideSec` and trimming bounded by
  `lastDiarizedAudioTime`, this prevents both buffer eviction gaps
  and unbounded RAM growth.

End-to-end the rewrite means: file analysis runs as fast as the
device can chew, ASR output times stay correct against the source
audio at any duration, and SpeechAnalyzer is reset often enough that
its quality stays constant from minute 0 to minute 120.

## Where this lives in code

- `Core/Audio/RollingAudioBuffer.swift` — `TimeAnchor`,
  `indexForFileTime`, `fileTimeForIndex`, anchor-driven slice/trim.
- `Core/Audio/AudioFileCapture.swift` — non-realtime pump,
  `yieldWithBackpressure`, no playback.
- `Core/ASR/StreamingSpeechAnalyzerTranscriber.swift` — periodic
  rotation, per-analyzer anchors, `correctedSeconds`.
- `Xephon/RecordingController.swift` — `rawPumpBaseTimestamp`
  (mic-mode rebasing), `latestCapturedFileTime`,
  `lastDiarizedAudioTime`, catch-up diarize loop,
  `maxDiarizeLagSeconds` backpressure, `trimProcessedAudio` gated
  by the diarized boundary, end-of-file
  `reconcileSpeakersWithTimeline()`.

## Open caveats

- **Mic mode still pays real-time.** Audio arrives at engine clock;
  there's no faster-than-real-time path. The same rotation + drift
  correction applies, just on a slower cadence.
- **Rotation seam.** A rotation lands at a hard 10-min boundary
  in fed audio. If a sentence happens to straddle that seam, one
  final ends slightly before it and another begins slightly after.
  Late fusion and the speaker reconciliation pass both tolerate
  this; word-level alignment across the seam may briefly look
  jittery on display.
- **The 220 s diarize-lag ceiling is empirical.** It's set to
  cover worst-case SER + diarize stalls observed on M4 iPad Pro.
  If diarization gets faster or slower in a future model swap,
  re-tune.
