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

## Optional remote LLM (LM Studio)

The Summarizer + Reviewer can optionally off-load inference to a user-run
[LM Studio](https://lmstudio.ai/) server on the local network. This is
**off by default** and gated behind two UI conditions:

- The "Remote LLM Server (LM Studio)" toggle in Settings must be enabled.
- A reachable host + port must be configured.

When both conditions hold and the user selects the `LM Studio (remote)`
backend in the Summarizer picker, the app POSTs to
`http://<host>:<port>/v1/chat/completions` with these data classes:

- **Post-ASR transcript text** (per-utterance lines, speaker ids, fused
  emotion labels, V/A/D scores)
- **Display names** for speakers the user has renamed
- **No audio.** The wav stream never leaves the device, and the request
  body contains no acoustic features beyond the V/A/D scalars already in
  the JSON utterance schema.

The first time the app opens a TCP connection to a local IP iOS surfaces a
permission prompt; the `NSLocalNetworkUsageDescription` string explains
what the connection is for. ATS is configured with
`NSAllowsLocalNetworking = true` — plaintext HTTP is permitted only to
local-network destinations (link-local, `.local` Bonjour, RFC 1918
addresses); it does **not** permit plaintext to the public internet.

LM Studio is not a cloud provider — it runs on hardware the user
controls — but the local-network hop is still off-device, so the same
disclosure-first rule applies.

## Adding a new cloud provider

A pull request that introduces a cloud dependency must:

1. Add a UI toggle (and respect it everywhere).
2. Update this document with the data classes sent and the provider's
   retention/usage policy.
3. Surface the data classes in the in-app privacy disclosure.
