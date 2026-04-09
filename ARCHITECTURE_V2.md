# VaultASR v2 — Architecture Design Document

## Executive Summary

VaultASR v2 is a complete local speech-to-text pipeline built as a single C++17 executable. It performs denoising, voice activity detection, speaker diarization, and speech recognition entirely on-device. The initial target is Apple Silicon (M1–M4) via CoreML and Metal, with the architecture designed for future CUDA/HIP/Vulkan backends.

**Design principles:** zero-copy memory flow, low memory footprint while being accurate, streaming-capable, low-end machine friendly (4GB RAM target), single static binary with no Python/runtime dependencies.

---

## 1. Pipeline Overview

```
Audio File (any format)
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Stage 0: AUDIO DECODE (FFmpeg)                      │
│  Input:  file path                                   │
│  Output: float32 PCM, 16kHz mono                     │
│  Memory: streaming chunks, not full file in RAM       │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 1: DENOISE (RNNoise)                          │
│  Input:  raw PCM float32 @ 48kHz (resampled up)      │
│  Output: denoised PCM float32 → resample back 16kHz  │
│  Memory: ~2MB model + frame buffer                   │
│  Toggle: --denoise flag (off by default)              │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 2: VOICE ACTIVITY DETECTION (Silero VAD v5)   │
│  Input:  denoised 16kHz PCM                          │
│  Output: list of speech segments [{start, end}, ...]  │
│  Memory: ~2MB ONNX model                             │
│  Config: --vad-threshold (default 0.5)                │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 3: SPEAKER DIARIZATION                        │
│  3a. Mel Filterbank extraction (Kaldi Fbank)         │
│  3b. Speaker embedding (WeSpeaker ResNet34, CoreML)  │
│  3c. Spectral clustering (auto speaker count)        │
│  Input:  speech segments + raw audio                 │
│  Output: labeled segments [{start, end, speaker}, ...]│
│  Memory: ~26MB ONNX model                            │
│  Config: --max-speakers (default 10)                  │
│          --cluster-threshold (default 0.55)           │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 4: SPEECH RECOGNITION (whisper.cpp + Metal)   │
│  Input:  labeled audio segments                      │
│  Output: text with word-level timestamps + confidence │
│  Memory: 75MB (tiny) to 1.5GB (medium)               │
│  Config: --model tiny.en|base.en|small.en|medium.en   │
│          --language en (default, or auto-detect)      │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 5: POST-PROCESSING                            │
│  - Merge adjacent same-speaker segments              │
│  - Punctuation restoration (rule-based + Whisper)     │
│  - Capitalization normalization                       │
│  - Timestamp alignment                               │
└──────────────────┬───────────────────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────┐
│  Stage 6: EXPORT                                     │
│  Formats: stdout, JSON, CSV, XLSX, SRT, Markdown,    │
│           DOCX, SQLite3                              │
│  Config: --output <path> --format <fmt>              │
└──────────────────────────────────────────────────────┘
```

---

## 2. Model Selection & Memory Budget

| Stage | Model | Size on Disk | Peak RAM | Acceleration | Rationale |
|-------|-------|-------------|----------|-------------|-----------|
| Denoise | RNNoise (xiph) | ~85KB (embedded C weights) | <1MB | CPU only (fast enough) | Pure C library, ~20ms latency per frame, battle-tested. No GPU needed — processes in real-time on any CPU. |
| VAD | Silero VAD v5 | 2.2MB ONNX | ~5MB | CoreML via ONNX Runtime | Best accuracy-to-size ratio. 512-sample windows at 16kHz. Stateful (carries context). Supports 6000+ languages. |
| Embeddings | WeSpeaker ResNet34 (Pyannote) | 26MB ONNX | ~40MB | CoreML via ONNX Runtime | Strong speaker discrimination. 256-dim embeddings. Already integrated and working. CAM++/ECAPA-TDNN are similar size with marginal gains not worth the migration. |
| Fbank | Kaldi Fbank (custom C++) | 0 (code only) | <1MB | CPU (FFT via libavutil) | Already implemented. 80-mel bins, Hamming window, 25ms frames, 10ms shift. No model needed. |
| ASR | whisper.cpp GGML | 75MB–1.5GB | 150MB–3GB | Metal GPU | User-selectable. Default: tiny.en (75MB). Supports CoreML encoder acceleration for 3–5x speedup on Apple Neural Engine. |

**Total minimum memory footprint (tiny.en):** ~220MB peak RAM
**Total maximum memory footprint (medium.en):** ~3.2GB peak RAM

### Model file layout

```
models/
├── rnnoise_weights.bin          # 85KB  — or compiled into binary
├── silero_vad_v5.onnx           # 2.2MB
├── wespeaker_resnet34.onnx      # 26MB
└── whisper/
    ├── ggml-tiny.en.bin         # 75MB  (default)
    ├── ggml-base.en.bin         # 148MB
    ├── ggml-small.en.bin        # 466MB (optional download)
    └── ggml-medium.en.bin       # 1.5GB (optional download)
```

---

## 3. Module Design

### 3.1 `AudioDecoder` (audio_decoder.h / audio_decoder.cpp)

**Replaces:** current `audio_utils.cpp`

**Key changes from v1:**
- Streaming decode — reads audio in chunks instead of loading entire file into memory
- Supports seeking for resume capability
- Reports duration and format metadata before full decode

```cpp
class AudioDecoder {
public:
    struct AudioMeta {
        double duration_sec;
        int sample_rate;
        int channels;
        std::string codec_name;
        std::string format_name;
    };

    // Probe file without decoding
    static AudioMeta probe(const std::string& path);

    // Streaming decode: callback receives chunks of float32 PCM @ 16kHz mono
    // Returns total samples decoded
    using ChunkCallback = std::function<void(const float* data, size_t num_samples)>;
    static size_t decode_streaming(const std::string& path, ChunkCallback cb,
                                    size_t chunk_size = 16000 * 30); // 30s chunks

    // Full decode (for short files or when random access needed)
    static std::vector<float> decode_full(const std::string& path);
};
```

**Supported formats:** Everything FFmpeg supports — MP3, WAV, FLAC, OGG, AAC, M4A, WMA, OPUS, plus video containers (MP4, MOV, MKV, AVI, WMV, FLV, MTS) where it extracts the audio track.

### 3.2 `Denoiser` (denoiser.h / denoiser.cpp)

**New module.** Wraps RNNoise.

```cpp
class Denoiser {
public:
    Denoiser();   // loads RNNoise model (embedded weights or file)
    ~Denoiser();

    // In-place denoise. Audio must be 16kHz mono float32.
    // Internally resamples to 48kHz (RNNoise native rate), denoises,
    // resamples back to 16kHz.
    void process(std::vector<float>& audio);

    // Streaming: process a single 10ms frame (160 samples @ 16kHz)
    // Returns denoised frame
    std::vector<float> process_frame(const float* data, size_t len);

private:
    DenoiseState* rnn_state_;
    // Internal 16k→48k→16k resampler state (libswresample)
    SwrContext* upsample_ctx_;
    SwrContext* downsample_ctx_;
};
```

**Integration notes:**
- RNNoise operates on 480-sample frames at 48kHz (10ms)
- We resample 16kHz→48kHz, denoise, then 48kHz→16kHz
- The resamplers add ~5ms latency per direction — acceptable for offline processing
- RNNoise weights can be compiled directly into the binary (no external file)

### 3.3 `VoiceActivityDetector` (vad.h / vad.cpp)

**Replaces:** current `silero_vad.cpp` (keeps same model, fixes state management)

**Key fixes from v1:**
- Proper state carry between frames (v1 had a bug where state dimensions were mismatched)
- Configurable threshold with hysteresis
- Minimum speech/silence duration to prevent fragmentation
- Segment merging for short gaps

```cpp
struct SpeechSegment {
    double start_sec;
    double end_sec;
    float avg_confidence;  // mean VAD probability across segment
};

class VoiceActivityDetector {
public:
    struct Config {
        float threshold = 0.5f;          // speech probability threshold
        float neg_threshold = 0.35f;     // threshold to end speech (hysteresis)
        double min_speech_sec = 0.25;    // minimum speech duration
        double min_silence_sec = 0.3;    // minimum silence to split
        double max_segment_sec = 30.0;   // force split at this length (for ASR)
        double speech_pad_sec = 0.1;     // padding around speech segments
    };

    VoiceActivityDetector(const std::string& model_path,
                          Ort::Env& env,
                          const Config& config = {});

    // Process full audio buffer, return speech segments
    std::vector<SpeechSegment> detect(const std::vector<float>& audio_16k);

    // Reset internal state (for new file)
    void reset();

private:
    Ort::Session session_;
    Ort::MemoryInfo memory_info_;
    Config config_;

    // Silero v5 state: shape [2, 1, 128]
    std::vector<float> state_;
    int64_t sample_rate_ = 16000;

    float infer_frame(const float* data);  // single 512-sample frame
};
```

### 3.4 `SpeakerEmbedder` (speaker_embedder.h / speaker_embedder.cpp)

**Replaces:** `kaldi_fbank.cpp` + `wespeaker_embedder.cpp` (merges into one class)

```cpp
class SpeakerEmbedder {
public:
    SpeakerEmbedder(const std::string& model_path,
                     Ort::Env& env,
                     bool use_coreml = true);

    // Extract 256-dim speaker embedding from raw audio segment
    // Handles fbank computation internally
    std::vector<float> embed(const float* audio_16k, size_t num_samples);

    // Batch embed multiple segments
    std::vector<std::vector<float>> embed_batch(
        const std::vector<float>& full_audio,
        const std::vector<SpeechSegment>& segments);

private:
    Ort::Session session_;
    KaldiFbank fbank_;

    // L2-normalize embedding in-place
    void normalize(std::vector<float>& emb);
};
```

### 3.5 `SpeakerClusterer` (speaker_cluster.h / speaker_cluster.cpp)

**Replaces:** `clustering.cpp` — switches from AHC to Spectral Clustering

**Why spectral clustering over AHC:**
- Better at automatically determining number of speakers (eigenvalue gap heuristic)
- More robust with cosine-similarity affinity matrices
- Scales better than O(n³) AHC when segment count is high
- No need for manual distance threshold — uses data-driven cluster count

```cpp
struct DiarizedSegment {
    double start_sec;
    double end_sec;
    int speaker_id;        // 0-indexed
    float confidence;      // clustering confidence
};

class SpeakerClusterer {
public:
    struct Config {
        int max_speakers = 10;
        float min_cluster_similarity = 0.3f;  // minimum affinity to merge
        bool auto_num_speakers = true;         // use eigenvalue gap
        int fixed_num_speakers = 0;            // override if > 0
    };

    SpeakerClusterer(const Config& config = {});

    // Cluster embeddings, return speaker labels (0-indexed)
    // Uses spectral clustering with normalized Laplacian
    std::vector<int> cluster(const std::vector<std::vector<float>>& embeddings);

private:
    Config config_;

    // Build cosine similarity affinity matrix
    std::vector<std::vector<float>> build_affinity(
        const std::vector<std::vector<float>>& embeddings);

    // Estimate number of speakers from eigenvalue gap
    int estimate_num_speakers(const std::vector<float>& eigenvalues);

    // Simple k-means on spectral embedding
    std::vector<int> kmeans(const std::vector<std::vector<float>>& data, int k);
};
```

**Spectral clustering algorithm:**
1. Build cosine similarity matrix S from embeddings
2. Compute normalized graph Laplacian: L = D^(-1/2) * (D - S) * D^(-1/2)
3. Find k smallest eigenvectors (k = estimated speakers via eigenvalue gap)
4. Stack eigenvectors as rows, normalize
5. Run k-means on rows to get cluster assignments

**Eigenvalue gap heuristic:** The number of speakers = argmax(gap between consecutive eigenvalues) for eigenvalues sorted ascending. This is bounded by `max_speakers`.

**Fallback:** If spectral clustering produces degenerate results (single cluster when clearly multiple speakers), fall back to AHC with the existing code. Keep AHC as a private method.

### 3.6 `Transcriber` (transcriber.h / transcriber.cpp)

**Replaces:** inline whisper logic in `main_metal.cpp`

```cpp
struct WordInfo {
    std::string text;
    double start_sec;       // word-level timestamp
    double end_sec;
    float probability;      // token probability from Whisper
};

struct TranscriptSegment {
    int speaker_id;
    double start_sec;
    double end_sec;
    std::string text;                   // full segment text
    std::vector<WordInfo> words;        // word-level detail
    float avg_confidence;
};

class Transcriber {
public:
    struct Config {
        std::string model_path;         // path to ggml-*.bin
        std::string language = "en";    // or "auto" for detection
        bool use_gpu = true;            // Metal on macOS
        int n_threads = 4;
        bool token_timestamps = true;   // word-level timestamps
        bool translate = false;         // translate to English
        float no_speech_threshold = 0.6f;
    };

    Transcriber(const Config& config);
    ~Transcriber();

    // Transcribe a single audio segment
    TranscriptSegment transcribe(const float* audio_16k, size_t num_samples,
                                  int speaker_id, double offset_sec);

    // Batch transcribe all diarized segments
    std::vector<TranscriptSegment> transcribe_all(
        const std::vector<float>& full_audio,
        const std::vector<DiarizedSegment>& segments);

private:
    whisper_context* ctx_;
    Config config_;
};
```

**Word-level timestamps:** whisper.cpp supports `WHISPER_SAMPLING_GREEDY` with `token_timestamps = true`. Each token's timing is available via `whisper_full_get_segment_t0/t1` and probability via `whisper_full_get_segment_p`.

### 3.7 `PostProcessor` (post_processor.h / post_processor.cpp)

**New module.**

```cpp
class PostProcessor {
public:
    // Merge adjacent segments from same speaker
    static std::vector<TranscriptSegment> merge_speakers(
        const std::vector<TranscriptSegment>& segments,
        double max_gap_sec = 1.0);

    // Basic punctuation restoration (Whisper usually handles this,
    // but fix edge cases at segment boundaries)
    static void fix_punctuation(std::vector<TranscriptSegment>& segments);

    // Capitalize sentence starts
    static void fix_capitalization(std::vector<TranscriptSegment>& segments);

    // Remove duplicate text at segment boundaries
    // (overlapping windows can cause whisper to repeat phrases)
    static void deduplicate_boundaries(std::vector<TranscriptSegment>& segments);
};
```

### 3.8 `Exporter` (exporter.h / exporter.cpp)

**New module.** Handles all output formats.

```cpp
enum class ExportFormat {
    STDOUT,     // human-readable console output
    JSON,       // structured JSON
    CSV,        // flat CSV
    XLSX,       // Excel via libxlsxwriter (C library, ~1MB)
    SRT,        // SubRip subtitle format
    MARKDOWN,   // formatted Markdown
    DOCX,       // Word document via custom minimal writer
    SQLITE3,    // full analysis database
};

class Exporter {
public:
    static void export_to(const std::vector<TranscriptSegment>& transcript,
                          ExportFormat format,
                          const std::string& output_path);

private:
    static void to_stdout(const std::vector<TranscriptSegment>& t);
    static void to_json(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_csv(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_xlsx(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_srt(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_markdown(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_docx(const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_sqlite(const std::vector<TranscriptSegment>& t, const std::string& path);
};
```

**SQLite3 schema:**

```sql
CREATE TABLE transcripts (
    id INTEGER PRIMARY KEY,
    audio_file TEXT NOT NULL,
    created_at TEXT DEFAULT (datetime('now')),
    model TEXT,
    duration_sec REAL
);

CREATE TABLE segments (
    id INTEGER PRIMARY KEY,
    transcript_id INTEGER REFERENCES transcripts(id),
    speaker_id INTEGER NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    text TEXT NOT NULL,
    confidence REAL
);

CREATE TABLE words (
    id INTEGER PRIMARY KEY,
    segment_id INTEGER REFERENCES segments(id),
    word TEXT NOT NULL,
    start_sec REAL NOT NULL,
    end_sec REAL NOT NULL,
    confidence REAL
);

CREATE INDEX idx_segments_transcript ON segments(transcript_id);
CREATE INDEX idx_words_segment ON words(segment_id);
CREATE INDEX idx_segments_speaker ON segments(speaker_id);
```

### 3.9 `Pipeline` (pipeline.h / pipeline.cpp)

**New module.** Orchestrates the full flow.

```cpp
class Pipeline {
public:
    struct Config {
        // Audio
        std::string audio_path;

        // Denoise
        bool denoise = false;

        // VAD
        VoiceActivityDetector::Config vad_config;

        // Diarization
        std::string embedder_model = "models/wespeaker_resnet34.onnx";
        SpeakerClusterer::Config cluster_config;
        bool skip_diarization = false;  // single-speaker mode

        // ASR
        Transcriber::Config transcriber_config;

        // Export
        ExportFormat export_format = ExportFormat::STDOUT;
        std::string output_path;

        // Runtime
        bool verbose = false;
        bool use_coreml = true;
    };

    Pipeline(const Config& config);

    // Run full pipeline, return transcript
    std::vector<TranscriptSegment> run();

    // Progress callback for UI integration later
    using ProgressCallback = std::function<void(const std::string& stage, float pct)>;
    void set_progress_callback(ProgressCallback cb);

private:
    Config config_;
    ProgressCallback progress_cb_;
    Ort::Env ort_env_;
};
```

---

## 4. CLI Interface

```
USAGE:
    vaultasr [OPTIONS] <audio-file> [<audio-file> ...]

AUDIO INPUT:
    <audio-file>             One or more audio/video files to transcribe

MODEL SELECTION:
    --model <name>           Whisper model: tiny.en (default), base.en,
                             small.en, medium.en, tiny, base, small, medium
    --model-path <path>      Custom path to GGML model file
    --models-dir <path>      Directory containing model files (default: ./models)

PIPELINE OPTIONS:
    --denoise                Enable RNNoise denoising (off by default)
    --no-diarize             Disable speaker diarization (single speaker mode)
    --language <lang>        Language code (default: en, use "auto" to detect)
    --max-speakers <n>       Maximum number of speakers (default: 10)

VAD TUNING:
    --vad-threshold <f>      VAD speech probability threshold (default: 0.5)
    --min-speech <f>         Minimum speech duration in seconds (default: 0.25)
    --min-silence <f>        Minimum silence to split in seconds (default: 0.3)
    --speech-pad <f>         Padding around speech in seconds (default: 0.1)

OUTPUT:
    --output <path>          Output file path (default: stdout)
    --format <fmt>           Output format: text (default), json, csv, xlsx,
                             srt, markdown, docx, sqlite
    --timestamps             Include timestamps in text output

PERFORMANCE:
    --threads <n>            Number of CPU threads (default: 4)
    --no-gpu                 Disable GPU acceleration (CPU only)

GENERAL:
    --verbose                Show detailed pipeline progress
    --version                Show version information
    --help                   Show this help message

EXAMPLES:
    vaultasr meeting.mp3
    vaultasr --model base.en --denoise --format json -o out.json call.wav
    vaultasr --no-diarize --format srt lecture.m4a
    vaultasr --format sqlite --output analysis.db *.mp3
```

### File Queue Behavior

When multiple files are given, VaultASR processes them sequentially, sharing the loaded models across files (load once, transcribe many). Output behavior:

- **stdout/text:** prints each file's transcript separated by headers
- **json:** produces array of transcript objects
- **sqlite:** appends all transcripts to the same database
- **other formats:** creates `<basename>.<ext>` for each file

---

## 5. Build System

### CMakeLists.txt (redesigned)

```cmake
cmake_minimum_required(VERSION 3.16)
project(VaultASR VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(VAULTASR_USE_COREML "Enable CoreML acceleration (macOS)" ON)
option(VAULTASR_USE_METAL "Enable Metal GPU acceleration (macOS)" ON)
option(VAULTASR_USE_CUDA "Enable CUDA GPU acceleration (future)" OFF)
option(VAULTASR_EMBED_RNNOISE "Compile RNNoise weights into binary" ON)

# === Dependencies ===

# FFmpeg (required)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED
    libavformat libavcodec libavutil libswresample)

# ONNX Runtime (required for VAD + diarization)
find_package(onnxruntime REQUIRED)
# Fallback: find via homebrew
if(NOT onnxruntime_FOUND)
    find_library(ONNXRUNTIME_LIB onnxruntime PATHS /opt/homebrew/lib)
    set(ONNXRUNTIME_INCLUDE /opt/homebrew/include)
endif()

# SQLite3 (for export)
find_package(SQLite3 REQUIRED)

# whisper.cpp (submodule)
set(WHISPER_METAL ${VAULTASR_USE_METAL} CACHE BOOL "" FORCE)
set(WHISPER_COREML OFF CACHE BOOL "" FORCE)  # we use GGML Metal, not CoreML whisper
add_subdirectory(external/whisper.cpp)

# RNNoise (submodule or bundled)
add_subdirectory(external/rnnoise)

# libxlsxwriter (for XLSX export)
add_subdirectory(external/libxlsxwriter)

# === Apple Frameworks ===
if(APPLE)
    find_library(COREML_FW CoreML)
    find_library(FOUNDATION_FW Foundation)
    find_library(METAL_FW Metal)
    find_library(METALKIT_FW MetalKit)
    find_library(ACCELERATE_FW Accelerate)
endif()

# === Main Library ===
add_library(vaultasr_lib STATIC
    src/audio_decoder.cpp
    src/denoiser.cpp
    src/vad.cpp
    src/kaldi_fbank.cpp
    src/speaker_embedder.cpp
    src/speaker_cluster.cpp
    src/transcriber.cpp
    src/post_processor.cpp
    src/exporter.cpp
    src/pipeline.cpp
)

target_include_directories(vaultasr_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${FFMPEG_INCLUDE_DIRS}
    ${ONNXRUNTIME_INCLUDE}
)

target_link_libraries(vaultasr_lib PUBLIC
    ${FFMPEG_LIBRARIES}
    onnxruntime
    whisper
    rnnoise
    xlsxwriter
    SQLite::SQLite3
)

if(APPLE)
    target_link_libraries(vaultasr_lib PUBLIC
        ${COREML_FW}
        ${FOUNDATION_FW}
        ${METAL_FW}
        ${METALKIT_FW}
        ${ACCELERATE_FW}
    )
endif()

# === CLI Executable ===
add_executable(vaultasr src/main.cpp)
target_link_libraries(vaultasr PRIVATE vaultasr_lib)

# === Tests ===
enable_testing()
add_subdirectory(tests)

# === Install ===
install(TARGETS vaultasr RUNTIME DESTINATION bin)
```

### Directory structure (v2)

```
vaultasr/
├── CMakeLists.txt
├── src/
│   ├── main.cpp                  # CLI entry point + arg parsing
│   ├── pipeline.h / .cpp         # orchestrator
│   ├── audio_decoder.h / .cpp    # FFmpeg decode
│   ├── denoiser.h / .cpp         # RNNoise wrapper
│   ├── vad.h / .cpp              # Silero VAD v5
│   ├── kaldi_fbank.h / .cpp      # Mel filterbank (kept from v1)
│   ├── speaker_embedder.h / .cpp # WeSpeaker embedding
│   ├── speaker_cluster.h / .cpp  # Spectral clustering
│   ├── transcriber.h / .cpp      # whisper.cpp wrapper
│   ├── post_processor.h / .cpp   # text cleanup
│   └── exporter.h / .cpp         # all export formats
├── external/
│   ├── whisper.cpp/              # git submodule
│   ├── rnnoise/                  # git submodule (xiph/rnnoise)
│   └── libxlsxwriter/           # git submodule (for XLSX)
├── models/
│   ├── silero_vad_v5.onnx
│   ├── wespeaker_resnet34.onnx
│   └── whisper/
│       └── ggml-tiny.en.bin
├── tests/
│   ├── test_vad.cpp
│   ├── test_clustering.cpp
│   ├── test_exporter.cpp
│   └── fixtures/
│       └── test_audio_3sec.wav
└── scripts/
    └── download_models.sh        # fetch models from HuggingFace
```

---

## 6. Dependency Summary

| Dependency | Type | Size Impact | Purpose | License |
|-----------|------|------------|---------|---------|
| FFmpeg (libavformat, libavcodec, libswresample, libavutil) | System lib | linked dynamically | Audio decode + resample | LGPL-2.1 |
| ONNX Runtime | System lib | linked dynamically | VAD + speaker embedding inference | MIT |
| whisper.cpp | Git submodule | compiled in | ASR | MIT |
| RNNoise | Git submodule | ~85KB compiled | Denoising | BSD-3 |
| libxlsxwriter | Git submodule | ~200KB compiled | XLSX export | BSD-2 |
| SQLite3 | System lib | linked dynamically | SQLite export | Public domain |
| Apple CoreML/Metal/Accelerate | System framework | 0 (OS provided) | GPU acceleration | Apple |

**User-installed requirements (via Homebrew):**
```bash
brew install ffmpeg onnxruntime sqlite3
```

---

## 7. Memory Optimization Strategy

### Streaming architecture
The pipeline does NOT load the entire audio file into RAM. Instead:

1. **Decode** in 30-second chunks
2. **Denoise** each chunk in-place (10ms frames)
3. **VAD** processes the full audio but only stores segment boundaries (tiny)
4. **Embedding** extracts per-segment — only one segment's audio in memory at a time
5. **Clustering** operates on embedding vectors only (256 floats × N segments ≈ KB)
6. **ASR** transcribes one segment at a time, freeing chunk after

For a 1-hour file at 16kHz mono:
- Full audio: 16000 × 3600 × 4 bytes = **230MB**
- Streaming (30s chunks): 16000 × 30 × 4 bytes = **1.9MB**

However, Whisper needs random access to segments, so we keep the full decoded audio for now but can memory-map it for very large files.

**Low-end machine strategy:**
- Default model: tiny.en (75MB) — runs on 4GB RAM Macs comfortably
- VAD filters out silence first → less audio to process
- Segments processed sequentially (not batched)
- No concurrent model loading — pipeline is sequential

### Memory-map for large files (future)
For files >30 minutes, decode to a temp file (raw PCM) and memory-map it. This lets the OS manage paging and keeps RAM usage constant regardless of file length.

---

## 8. Spectral Clustering — Implementation Detail

The switch from AHC to spectral clustering is the most significant algorithmic change. Here's the detailed approach:

```
Input: N embeddings of dimension 256

Step 1: Affinity Matrix
    For each pair (i, j), compute:
        S[i][j] = max(0, cosine_similarity(emb_i, emb_j))
    Set diagonal to 0.

Step 2: Degree Matrix
    D[i][i] = sum(S[i][j]) for all j

Step 3: Normalized Laplacian
    L_norm = I - D^(-1/2) * S * D^(-1/2)

Step 4: Eigendecomposition
    Compute smallest k eigenvectors of L_norm.
    Use power iteration or Lanczos (avoid full eigendecomposition for large N).

Step 5: Determine k (number of speakers)
    eigenvalues = sort(eigenvalues, ascending)
    gaps = [eigenvalues[i+1] - eigenvalues[i] for i in 0..max_speakers]
    k = argmax(gaps) + 1
    Clamp: 1 <= k <= max_speakers

Step 6: K-means
    Take first k eigenvectors as columns → NxK matrix.
    Row-normalize.
    Run k-means (10 iterations, 5 restarts).
    Labels = cluster assignments.
```

For typical meetings (10-300 segments), this runs in <100ms on CPU. No GPU needed.

We implement eigendecomposition using **Accelerate framework** (Apple's LAPACK) via `ssyev_` for real symmetric matrix eigensolve. This is already optimized for Apple Silicon.

---

## 9. Future Extensibility

### Hardware backends (planned)
The `Pipeline` class abstracts acceleration. Adding new backends means:
1. New ONNX Runtime execution provider (CUDA, DirectML, etc.)
2. New whisper.cpp backend flag (CUDA, Vulkan, etc.)
3. RNNoise stays CPU (it's fast enough universally)

```
Platform detection at build time:
    macOS   → CoreML (ONNX) + Metal (whisper) + Accelerate (eigen)
    Linux   → CUDA (ONNX) + CUDA (whisper) + OpenBLAS (eigen)
    Windows → DirectML (ONNX) + CUDA/Vulkan (whisper) + OpenBLAS (eigen)
```

### GUI (planned)
The `pipeline.h` library interface is GUI-ready:
- `ProgressCallback` for progress bars
- `TranscriptSegment` data structure maps directly to UI
- Pipeline is a single function call: `Pipeline::run()`
- Future: build Qt/SwiftUI frontend that links `vaultasr_lib`

---

## 10. Implementation Order

| Phase | Module | Est. Effort | Dependencies |
|-------|--------|------------|-------------|
| 1 | Restructure project directories + CMake | 1 day | None |
| 2 | `AudioDecoder` (refactor from v1) | 0.5 day | FFmpeg |
| 3 | `VoiceActivityDetector` (fix + enhance Silero) | 1 day | ONNX Runtime |
| 4 | `Denoiser` (integrate RNNoise) | 1 day | RNNoise submodule |
| 5 | `SpeakerEmbedder` (merge fbank + embedder) | 0.5 day | ONNX Runtime |
| 6 | `SpeakerClusterer` (spectral clustering) | 2 days | Accelerate |
| 7 | `Transcriber` (whisper wrapper + word timestamps) | 1 day | whisper.cpp |
| 8 | `PostProcessor` | 0.5 day | None |
| 9 | `Exporter` (all 8 formats) | 2 days | SQLite3, libxlsxwriter |
| 10 | `Pipeline` orchestrator + CLI | 1 day | All above |
| 11 | Testing + integration | 2 days | Test fixtures |

**Total estimated: ~12 working days**

---

## 11. Open Questions

1. **RNNoise vs skip:** RNNoise operates at 48kHz natively. The resample overhead (16k→48k→16k) adds latency and slight quality loss. Consider making denoise truly optional and off by default — Whisper is already quite robust to moderate noise.

2. **Silero VAD v5 ONNX state format:** Need to verify the exact state tensor dimensions for v5 (v4 used [2,1,128], v5 may differ). Test with the actual model file.

3. **Spectral clustering eigendecomposition:** For very large segment counts (1000+), full eigendecomposition is O(n³). May need Lanczos iteration for 2+ hour recordings. Profile first.

4. **DOCX export complexity:** Generating valid DOCX from C++ without a library is non-trivial (it's a ZIP of XML files). Options: (a) use a minimal C++ DOCX library, (b) generate via embedded template, (c) defer to a simpler Markdown→DOCX conversion. Recommend (b).

5. **Model download UX:** First run needs models downloaded. Ship a `download_models.sh` script? Or build download into the binary with a `vaultasr --download-models` command?
