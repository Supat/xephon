#!/usr/bin/env bash
# Regenerate Xephon.xcodeproj from project.yml.
# Requires: xcodegen (`brew install xcodegen`).
set -euo pipefail

cd "$(dirname "$0")/.."

if ! command -v xcodegen >/dev/null 2>&1; then
  echo "error: xcodegen not found. Install with: brew install xcodegen" >&2
  exit 1
fi

# Bootstrap the per-developer signing xcconfig if missing. The
# project's configFiles point at Config/Local.xcconfig; XcodeGen
# writes a reference to that path into the generated project but
# never touches the file's contents, so DEVELOPMENT_TEAM /
# CODE_SIGN_STYLE set in it survive regenerations. The file is
# gitignored — each developer keeps their own.
LOCAL_XCCONFIG="Config/Local.xcconfig"
LOCAL_EXAMPLE="Config/Local.xcconfig.example"
if [ ! -f "$LOCAL_XCCONFIG" ]; then
  if [ ! -f "$LOCAL_EXAMPLE" ]; then
    echo "error: $LOCAL_EXAMPLE missing — can't bootstrap $LOCAL_XCCONFIG" >&2
    exit 1
  fi
  cp "$LOCAL_EXAMPLE" "$LOCAL_XCCONFIG"
  echo "Created $LOCAL_XCCONFIG from example. Edit it to set DEVELOPMENT_TEAM."
fi

xcodegen generate
echo "Generated Xephon.xcodeproj"

# Surface a hint if the developer hasn't filled in their team id
# yet. Empty value = unsigned build; the user will hit signing
# errors on Run / Archive.
if grep -E '^DEVELOPMENT_TEAM[[:space:]]*=[[:space:]]*$' "$LOCAL_XCCONFIG" >/dev/null 2>&1; then
  echo
  echo "note: DEVELOPMENT_TEAM is empty in $LOCAL_XCCONFIG. On-device"
  echo "      builds will fail to sign until you fill it in."
fi
