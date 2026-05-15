# Xephon

Japanese conversational affect analyzer for iPadOS 26+ (research-only).

See [`CLAUDE.md`](CLAUDE.md) for the architecture, constraints, and conventions.
See [`docs/feasibility.md`](docs/feasibility.md) for the full feasibility report.

## Quickstart

```bash
# 1. One-time tooling
brew install xcodegen git-lfs
git lfs install

# 2. Generate Xephon.xcodeproj (gitignored)
./scripts/generate_project.sh

# 3. Hydrate Core ML / ONNX models (large download)
./scripts/fetch_models.sh

# 4. Open in Xcode
open Xephon.xcodeproj
```

## Runtime requirements

### Apple Intelligence (text SER on non-Japanese sessions)

The Japanese path uses bundled DeBERTa-WRIME for text SER, so no system
service is required. **English (and any other non-Japanese) sessions
route text SER through Apple FoundationModels**, which means
[Apple Intelligence](https://www.apple.com/apple-intelligence/) must be
enabled on the device:

- iPad / iPhone: **Settings → Apple Intelligence & Siri → Apple Intelligence**.
- macOS (including "Designed for iPad" builds running on Apple Silicon
  Macs): **System Settings → Apple Intelligence & Siri → Apple Intelligence**.

When Apple Intelligence is off / not yet downloaded / the device region
is unsupported, `SystemLanguageModel.default.isAvailable` returns
`false`, every English utterance logs
`text SER skipped: foundationModelsUnavailable`, and rows arrive without
a Plutchik score or text-backend chip. Japanese rows are unaffected (they
keep using DeBERTa).

## Repository status

This repo is a freshly scaffolded project. Module stubs throw
`*Error.notImplemented` until wired up — see `Core/` and the per-subsystem
notes in `CLAUDE.md`.
