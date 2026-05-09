#!/usr/bin/env bash
# Hydrate Core ML / ONNX / SafeTensors models into Models/.
#
# Usage:
#   scripts/fetch_models.sh            # fetch the full default set
#   scripts/fetch_models.sh --list     # show what would be fetched
#   scripts/fetch_models.sh --only NAME [NAME...]  # subset by short name
#
# Models are NOT committed to git. The .gitattributes filter is defense-in-depth.
set -euo pipefail

cd "$(dirname "$0")/.."

VENV_DIR=".venv"
PYTHON_BIN="${PYTHON_BIN:-python3}"

# -----------------------------------------------------------------------------
# Bootstrap: venv + huggingface_hub
# -----------------------------------------------------------------------------
if [ ! -d "$VENV_DIR" ]; then
  echo "[bootstrap] creating $VENV_DIR via $PYTHON_BIN"
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

if ! python -c "import huggingface_hub" 2>/dev/null; then
  echo "[bootstrap] installing scripts/requirements.txt"
  pip install --quiet --upgrade pip
  pip install --quiet -r scripts/requirements.txt
fi

# -----------------------------------------------------------------------------
# Model registry
#   short_name | source | location | extra
# `source`:
#   hf:<repo>          → huggingface-cli download <repo>
#   zenodo:<record>    → curl from https://zenodo.org/records/<record>/files/<extra>
# `extra`:
#   for hf:    optional --include glob (semicolon-separated for multiple)
#   for zenodo: filename
# -----------------------------------------------------------------------------
MODELS=(
  "kotoba-whisper-v2.0|hf:kotoba-tech/kotoba-whisper-v2.0|"
  "w2v2-msp-dim|hf:audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim|"
  "emotion2vec-plus-large|hf:emotion2vec/emotion2vec_plus_large|"
)

# DeBERTa-WRIME (Takenaka 2025) ships via `pip install deberta-emotion-predictor`.
# The exact upstream HF repo isn't documented. As an interim, we use
# MuneK/roberta-base-japanese-finetuned-wrime exported to ONNX via
# scripts/export_wrime_onnx.py (called at the end of this script).

# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
ACTION=fetch
FILTER=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --list)  ACTION=list; shift ;;
    --only)  shift; while [[ $# -gt 0 && "$1" != --* ]]; do FILTER+=("$1"); shift; done ;;
    -h|--help)
      sed -n '2,9p' "$0" | sed 's/^# *//'
      exit 0
      ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

want() {
  local name="$1"
  if [ ${#FILTER[@]} -eq 0 ]; then return 0; fi
  for f in "${FILTER[@]}"; do
    [[ "$name" == "$f" ]] && return 0
  done
  return 1
}

# -----------------------------------------------------------------------------
# List mode
# -----------------------------------------------------------------------------
if [ "$ACTION" = "list" ]; then
  printf "%-30s  %s\n" "name" "source"
  printf "%-30s  %s\n" "----" "------"
  for entry in "${MODELS[@]}"; do
    IFS='|' read -r name source _ <<< "$entry"
    printf "%-30s  %s\n" "$name" "$source"
  done
  exit 0
fi

# -----------------------------------------------------------------------------
# Fetch
# -----------------------------------------------------------------------------
mkdir -p Models

fetch_hf() {
  local repo="$1" outdir="$2" include="$3"
  local args=(download "$repo" --local-dir "$outdir")
  if [ -n "$include" ]; then
    IFS=';' read -ra patterns <<< "$include"
    for p in "${patterns[@]}"; do args+=(--include "$p"); done
  fi
  huggingface-cli "${args[@]}"
}

fetch_zenodo() {
  local record="$1" outdir="$2" filename="$3"
  mkdir -p "$outdir"
  local url="https://zenodo.org/records/${record}/files/${filename}"
  curl --fail --location --progress-bar -o "$outdir/$filename" "$url"
}

for entry in "${MODELS[@]}"; do
  IFS='|' read -r name source extra <<< "$entry"
  want "$name" || continue

  outdir="Models/$name"
  if [ -d "$outdir" ] && [ -n "$(ls -A "$outdir" 2>/dev/null)" ]; then
    echo "[skip] $name (already in $outdir)"
    continue
  fi

  echo "[fetch] $name ← $source"
  case "$source" in
    hf:*)     fetch_hf     "${source#hf:}"     "$outdir" "$extra" ;;
    zenodo:*) fetch_zenodo "${source#zenodo:}" "$outdir" "$extra" ;;
    *) echo "unknown source scheme: $source" >&2; exit 3 ;;
  esac
  echo "[done] $name"
done

echo
echo "[step] Exporting WRIME RoBERTa to ONNX (text SER)…"
if [ ! -f Models/wrime-roberta/model.onnx ]; then
    python scripts/export_wrime_onnx.py
else
    echo "[skip] wrime-roberta (already exported)"
fi

echo
echo "[step] Exporting emotion2vec_plus_large to ONNX (acoustic categorical SER)…"
if [ ! -f Models/emotion2vec-plus-large/emotion2vec_model.onnx ]; then
    python scripts/export_emotion2vec_onnx.py
else
    echo "[skip] emotion2vec (already exported)"
fi

echo
echo "Models/ tree:"
ls -la Models/
