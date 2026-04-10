# VaultASR v2

VaultASR is a high-performance, private, and local speech-to-text pipeline designed for macOS. It combines the power of OpenAI's Whisper with advanced Voice Activity Detection (VAD) and Speaker Diarization, all running locally on your hardware with Metal GPU acceleration.

## Features

- **Local & Private**: No data ever leaves your machine. Perfect for sensitive meetings or personal notes.
- **Advanced Pipeline**:
    - **Whisper (STT)**: Industry-leading transcription accuracy using `whisper.cpp`.
    - **Silero VAD v5**: Precise voice activity detection to split audio into logical segments.
    - **Speaker Diarization**: Multi-speaker identification using WeSpeaker-based embeddings.
    - **RNNoise Denoising**: Intelligent noise suppression for high-quality results in noisy environments.
- **Hardware Accelerated**: Fully optimized for Apple Silicon (M1/M2/M3) using **Metal** and **CoreML**.
- **Versatile Exports**: Support for Text, JSON, CSV, XLSX, SRT, Markdown, Docx, and SQLite.

## Prerequisites

Ensure you have the following installed (via Homebrew):

```bash
brew install cmake ffmpeg onnxruntime sqlite3
```

## Building

1. **Setup**: Run the setup script to initialize submodules and download required models.
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

2. **Build**:
   ```bash
   cmake -B build -DCMAKE_BUILD_TYPE=Release -DVAULTASR_USE_COREML=ON -DVAULTASR_USE_METAL=ON
   cmake --build build --parallel
   ```

## Usage

### Quick Start
Transcribe an audio file with default settings (`tiny.en` model):
```bash
./build/vaultasr meeting.mp3
```

### High Quality with Denoising
Use a larger model, enable noise suppression, and export to JSON:
```bash
./build/vaultasr --model base.en --denoise --format json --output report.json recording.wav
```

### Common Options
- `--model <name>`: `tiny.en`, `base.en`, `small.en`, `medium.en`
- `--no-diarize`: Disable speaker identification (faster)
- `--denoise`: Clean up background noise
- `--format <fmt>`: `text`, `json`, `xlsx`, `srt`, `markdown`, `docx`
- `--no-gpu`: Force CPU-only mode

## Project Structure

- `src/v2/`: Core C++ pipeline logic
- `external/`: Submodules (whisper.cpp, rnnoise, libxlsxwriter, miniz)
- `models/`: ONNX and GGML models
- `scripts/`: Initialization and helper scripts

## Roadmap

While the current version is optimized for macOS (Metal/CoreML), I plan to expand to other hardware acceleration paths:

- [ ] **Windows (DirectML)**: Support for Windows GPU acceleration via ONNX Runtime DirectML EP.
- [ ] **NVIDIA (CUDA)**: Support for high-performance CUDA execution providers.
- [ ] **AMD (ROCm/Vulcan)**: Broader cross-vendor GPU support.
- [ ] **Mobile Port**: Exploring lightweight execution for iOS/Android.

## 🤝 Contributing

Contributions are welcome! If you have experience with any of the following, we'd love your help:

| Area | What's Needed |
| :--- | :--- |
| 🪟 **Windows / DirectML** | Port the CMake build to Windows and wire up the ONNX Runtime DirectML execution provider |
| ⚡ **NVIDIA / CUDA** | Add CUDA EP support for VAD and speaker embedder inference |
| 🔴 **AMD / ROCm** | ROCm-based GPU acceleration on Linux/Windows |
| 📱 **Mobile** | Lightweight iOS/Android inference using Core ML or NNAPI |
| 🧪 **Testing** | Unit and integration tests for the pipeline stages |
| 📖 **Docs** | Improve setup guides, add architecture diagrams, API docs |

### How to contribute
1. Fork the repo and create a feature branch
2. Open an issue first to discuss major changes
3. Submit a pull request with a clear description of what you changed and why

All skill levels are welcome. Even fixing a typo or improving the README helps. 🙏

---
*VaultASR is optimized for speed and privacy. Transcribe hours of audio in minutes, entirely offline.*
