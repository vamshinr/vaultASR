#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# VaultASR v2 — One-shot setup script for Apple Silicon (macOS)
# Run this once before building.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

step() { echo -e "\n${CYAN}${BOLD}▶ $1${RESET}"; }
ok()   { echo -e "${GREEN}✓ $1${RESET}"; }
warn() { echo -e "${YELLOW}⚠ $1${RESET}"; }
err()  { echo -e "${RED}✗ $1${RESET}" >&2; exit 1; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo -e "${BOLD}${CYAN}"
echo "╔══════════════════════════════════════════╗"
echo "║      VaultASR v2 — Setup Script          ║"
echo "║      Apple Silicon / macOS               ║"
echo "╚══════════════════════════════════════════╝"
echo -e "${RESET}"

# ── Check macOS + Apple Silicon ─────────────────────────────────────────────
step "Checking system"
[[ "$(uname)" == "Darwin" ]] || err "This script is for macOS only."
[[ "$(uname -m)" == "arm64" ]] || warn "Not ARM64 — Metal/CoreML may not be optimal."
ok "macOS $(sw_vers -productVersion) on $(uname -m)"

# ── Homebrew ─────────────────────────────────────────────────────────────────
step "Checking Homebrew"
if ! command -v brew &>/dev/null; then
    err "Homebrew not found. Install from https://brew.sh"
fi
ok "Homebrew $(brew --version | head -1)"

# ── System dependencies ──────────────────────────────────────────────────────
step "Installing system dependencies via Homebrew"

BREW_DEPS=(ffmpeg onnxruntime sqlite cmake pkg-config)
for dep in "${BREW_DEPS[@]}"; do
    if brew list "$dep" &>/dev/null; then
        ok "$dep already installed"
    else
        echo "  Installing $dep..."
        brew install "$dep"
        ok "$dep installed"
    fi
done

# ── Git submodules ───────────────────────────────────────────────────────────
step "Initializing git submodules"

git submodule update --init --recursive external/whisper.cpp
ok "whisper.cpp"

git submodule update --init --recursive external/rnnoise
ok "rnnoise"

git submodule update --init --recursive external/libxlsxwriter
ok "libxlsxwriter"

git submodule update --init --recursive external/miniz
ok "miniz"

# ── Models ───────────────────────────────────────────────────────────────────
step "Downloading models"

mkdir -p models/whisper

# Silero VAD v5
if [[ ! -f models/silero_vad_v5.onnx ]]; then
    echo "  Downloading Silero VAD v5..."
    curl -L --progress-bar \
        "https://github.com/snakers4/silero-vad/raw/master/files/silero_vad.onnx" \
        -o models/silero_vad_v5.onnx
    ok "silero_vad_v5.onnx ($(du -sh models/silero_vad_v5.onnx | cut -f1))"
else
    ok "silero_vad_v5.onnx already exists"
fi

# WeSpeaker ResNet34
if [[ ! -f models/wespeaker_resnet34.onnx ]]; then
    echo "  Downloading WeSpeaker ResNet34..."
    # Try to use the existing file if present (renamed from old location)
    if [[ -f models/wespeaker_pyannote.onnx ]]; then
        cp models/wespeaker_pyannote.onnx models/wespeaker_resnet34.onnx
        ok "wespeaker_resnet34.onnx (copied from existing file)"
    else
        curl -L --progress-bar \
            "https://huggingface.co/pyannote/wespeaker-voxceleb-resnet34-LM/resolve/main/wespeaker_resnet34.onnx" \
            -o models/wespeaker_resnet34.onnx
        ok "wespeaker_resnet34.onnx ($(du -sh models/wespeaker_resnet34.onnx | cut -f1))"
    fi
else
    ok "wespeaker_resnet34.onnx already exists"
fi

# Whisper tiny.en (default)
if [[ ! -f models/whisper/ggml-tiny.en.bin ]]; then
    echo "  Downloading Whisper tiny.en (~75 MB)..."
    curl -L --progress-bar \
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-tiny.en.bin" \
        -o models/whisper/ggml-tiny.en.bin
    ok "ggml-tiny.en.bin ($(du -sh models/whisper/ggml-tiny.en.bin | cut -f1))"
else
    ok "ggml-tiny.en.bin already exists"
fi

# Copy existing base.en if present
if [[ -f models/ggml-base.en.bin ]] && [[ ! -f models/whisper/ggml-base.en.bin ]]; then
    cp models/ggml-base.en.bin models/whisper/ggml-base.en.bin
    ok "ggml-base.en.bin (moved to models/whisper/)"
fi

# ── Build ────────────────────────────────────────────────────────────────────
step "Building VaultASR"

cmake -B build -DCMAKE_BUILD_TYPE=Release \
    -DVAULTASR_USE_COREML=ON \
    -DVAULTASR_USE_METAL=ON

cmake --build build --parallel "$(sysctl -n hw.logicalcpu)"

ok "Build complete: build/vaultasr"

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}╔══════════════════════════════════════════╗${RESET}"
echo -e "${GREEN}${BOLD}║        Setup complete! ✓                 ║${RESET}"
echo -e "${GREEN}${BOLD}╚══════════════════════════════════════════╝${RESET}"
echo ""
echo "  Run: ./build/vaultasr --help"
echo "  Try: ./build/vaultasr your_audio.mp3"
echo ""
echo "  Optional larger models:"
echo "    scripts/download_model.sh base.en    (~148 MB)"
echo "    scripts/download_model.sh small.en   (~466 MB)"
echo "    scripts/download_model.sh medium.en  (~1.5 GB)"
echo ""
