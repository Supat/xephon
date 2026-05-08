#!/usr/bin/env bash
# Regenerate Xephon.xcodeproj from project.yml.
# Requires: xcodegen (`brew install xcodegen`).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "error: xcodegen not found. Install with: brew install xcodegen" >&2
  exit 1
fi

xcodegen generate
echo "Generated Xephon.xcodeproj"
