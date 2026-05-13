#!/usr/bin/env bash
# Upload the FP16-quantized on-device ONNX models + tokenizer to a
# GitHub Release the app downloads from on first launch.
#
# Usage:
#   scripts/upload_models_to_github_release.sh                      # tag = manifest default
#   scripts/upload_models_to_github_release.sh --tag models-v2      # override tag
#   scripts/upload_models_to_github_release.sh --dry-run            # only print SHAs / planned uploads
#
# After upload, the script prints the Swift `sha256: "..."` constants
# to paste into Xephon/ModelManifest.swift. It does NOT modify the
# Swift file in place — that's a manual step so you can review.
#
# Requires:
#   - gh CLI authenticated (`gh auth status`)
#   - Models/ already populated + FP16-quantized (run scripts/fetch_models.sh first)
set -euo pipefail

cd "$(dirname "$0")/.."

# Mapping: local path → GitHub Release asset name.
# Order matters only for human readability of the printed Swift
# constants (matches ModelManifest.swift's entry order).
#
# Note: the Qwen2.5 summarizer files are deliberately *not* listed
# here. The 4-bit safetensors blob is 4.28 GB, well past GitHub
# Releases' 2 GB Free-tier asset limit, so the manifest sources
# every Qwen file from `huggingface.co/mlx-community/Qwen2.5-7B-
# Instruct-4bit` directly via `ModelFile.directRemoteURL`. No
# GitHub upload needed.
declare -a UPLOADS=(
  "Models/w2v2-msp-dim/model.onnx|w2v2-model.onnx|W2V2_MODEL"
  "Models/w2v2-msp-dim/model.onnx.data|w2v2-model.data|W2V2_DATA"
  "Models/w2v2-age-gender/model.onnx|w2v2-age-gender-model.onnx|W2V2_AGE_GENDER_MODEL"
  "Models/w2v2-age-gender/model.onnx.data|w2v2-age-gender-model.data|W2V2_AGE_GENDER_DATA"
  "Models/emotion2vec-plus-large/emotion2vec_onnx/model.onnx|emotion2vec-model.onnx|EMOTION2VEC_MODEL"
  "Models/emotion2vec-plus-large/emotion2vec_onnx/model.data|emotion2vec-model.data|EMOTION2VEC_DATA"
  "Models/wrime-roberta/model.onnx|wrime-model.onnx|WRIME_MODEL"
  "Models/wrime-roberta/tokenizer.json|wrime-tokenizer.json|WRIME_TOKENIZER_JSON"
  "Models/wrime-roberta/tokenizer_config.json|wrime-tokenizer_config.json|WRIME_TOKENIZER_CONFIG"
  "Models/wrime-roberta/config.json|wrime-config.json|WRIME_CONFIG"
  "Models/wrime-roberta/special_tokens_map.json|wrime-special_tokens_map.json|WRIME_SPECIAL_TOKENS_MAP"
)

DEFAULT_TAG="$(grep -E 'static let releaseTag' Xephon/ModelManifest.swift | sed -E 's/.*"([^"]+)".*/\1/')"
TAG="${DEFAULT_TAG:-models-v1}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tag)     TAG="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,15p' "$0" | sed 's/^# *//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

echo "[plan] release tag: $TAG"
echo

# 1. Verify all source files exist + compute SHA-256.
# macOS bash 3.2 has no associative arrays — collect SHAs into a
# parallel array indexed the same as UPLOADS so we can re-emit them in
# step 4 without re-hashing.
echo "[hash] computing SHA-256 of each asset…"
SHAS=()
for entry in "${UPLOADS[@]}"; do
  IFS='|' read -r path asset key <<< "$entry"
  if [ ! -f "$path" ]; then
    echo "[error] missing: $path  (run scripts/fetch_models.sh first)" >&2
    exit 3
  fi
  sha="$(shasum -a 256 "$path" | awk '{print $1}')"
  SHAS+=("$sha")
  size="$(du -h "$path" | awk '{print $1}')"
  printf "  %-12s  %-40s  %s  %s\n" "$size" "$asset" "$sha" "$path"
done
echo

if [ "$DRY_RUN" -eq 1 ]; then
  echo "[dry-run] skipping gh release create / upload"
else
  # 2. Create the release if it doesn't exist.
  if gh release view "$TAG" >/dev/null 2>&1; then
    echo "[release] $TAG already exists; new assets will be appended (existing assets won't change)"
  else
    echo "[release] creating $TAG"
    gh release create "$TAG" \
      --title "On-device models ($TAG)" \
      --notes "FP16-quantized ONNX weights + tokenizer for Xephon. Downloaded by the app on first launch (see Xephon/ModelStore.swift)."
  fi

  # 3. Stage the files under their target asset names. `gh release
  # upload`'s `path#label` syntax only sets a display label — the
  # released asset's filename is always the path's basename. Several of
  # our source files share the same basename (`model.onnx` appears
  # under three different model dirs), so without a rename step every
  # upload would collide on `model.onnx` and `--clobber` would leave
  # only the last one. Symlink-staging gives us the prefixes
  # (`w2v2-model.onnx`, `wrime-model.onnx`, …) without copying ~840 MB
  # of bytes.
  STAGE_DIR="$(mktemp -d)"
  trap 'rm -rf "$STAGE_DIR"' EXIT
  for entry in "${UPLOADS[@]}"; do
    IFS='|' read -r path asset key <<< "$entry"
    ln -s "$(pwd)/$path" "$STAGE_DIR/$asset"
  done

  # 4. Upload each asset. `--clobber` so re-runs replace stale uploads.
  echo "[upload] pushing assets to ${TAG}…"
  for entry in "${UPLOADS[@]}"; do
    IFS='|' read -r path asset key <<< "$entry"
    echo "  → $asset"
    gh release upload "$TAG" "$STAGE_DIR/$asset" --clobber
  done
fi

# 4. Emit Swift constants to paste into ModelManifest.swift.
echo
echo "════════════════════════════════════════════════════════════════"
echo "  Paste the SHA-256 values below into Xephon/ModelManifest.swift"
echo "  (replace each TODO_PASTE_..._SHA256 placeholder)"
echo "════════════════════════════════════════════════════════════════"
echo
for i in "${!UPLOADS[@]}"; do
  entry="${UPLOADS[$i]}"
  IFS='|' read -r path asset key <<< "$entry"
  printf "TODO_PASTE_%s_SHA256  →  \"%s\"\n" "$key" "${SHAS[$i]}"
done
echo
echo "Then bump the build, install on device, confirm SetupView shows 'downloading'"
echo "for each file and 'completed' once finished."
