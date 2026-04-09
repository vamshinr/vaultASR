#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VaultASR — Download additional Whisper models
# Usage: ./scripts/download_model.sh <model>
#   e.g. ./scripts/download_model.sh base.en
#        ./scripts/download_model.sh small.en
#        ./scripts/download_model.sh medium.en
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$REPO_ROOT/models/whisper"

MODEL="${1:-}"

usage() {
    echo "Usage: $0 <model>"
    echo ""
    echo "Available models:"
    echo "  tiny.en    (~75 MB)   — fastest, English only"
    echo "  base.en    (~148 MB)  — good balance, English only"
    echo "  small.en   (~466 MB)  — high quality, English only"
    echo "  medium.en  (~1.5 GB)  — highest quality, English only"
    echo "  tiny       (~75 MB)   — fastest, multilingual"
    echo "  base       (~148 MB)  — balanced, multilingual"
    echo "  small      (~466 MB)  — high quality, multilingual"
    echo "  medium     (~1.5 GB)  — highest quality, multilingual"
    echo ""
    exit 1
}

[[ -z "$MODEL" ]] && usage

FILENAME="ggml-${MODEL}.bin"
DEST="$REPO_ROOT/models/whisper/$FILENAME"
URL="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$FILENAME"

if [[ -f "$DEST" ]]; then
    echo "✓ $FILENAME already exists at models/whisper/"
    exit 0
fi

echo "Downloading $FILENAME..."
curl -L --progress-bar "$URL" -o "$DEST"
echo "✓ Saved to models/whisper/$FILENAME ($(du -sh "$DEST" | cut -f1))"
echo ""
echo "Use with: vaultasr --model $MODEL your_audio.mp3"
