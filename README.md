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

## Repository status

This repo is a freshly scaffolded project. Module stubs throw
`*Error.notImplemented` until wired up — see `Core/` and the per-subsystem
notes in `CLAUDE.md`.
