# Privacy

Xephon is a research tool. All processing is on-device by default.

## Data flow

- Audio capture is performed locally via `AVAudioEngine` and never persisted
  to disk unless the user explicitly exports a recording.
- ASR, diarization, and emotion estimation run locally on the Neural Engine /
  GPU / CPU.
- No cloud transcription or analysis path is wired up today. If one is added
  later, it must follow the rules in "Adding a new cloud provider" below.

## What we do not do

- We never silently fine-tune on user data. Fine-tuning workflows are
  out-of-band scripts, not in-app actions.
- We never upload audio without an explicit user-visible toggle.
- We never commit participant audio (`.wav`, `.m4a`, …) to the repository.

## Adding a new cloud provider

A pull request that introduces a cloud dependency must:

1. Add a UI toggle (and respect it everywhere).
2. Update this document with the data classes sent and the provider's
   retention/usage policy.
3. Surface the data classes in the in-app privacy disclosure.
