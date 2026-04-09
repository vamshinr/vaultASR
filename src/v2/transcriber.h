#pragma once

#include "types.h"
#include <string>
#include <vector>

// Forward declare whisper.cpp types
struct whisper_context;

namespace vaultasr {

// ─── Whisper.cpp ASR wrapper ───────────────────────────────────────────────
//
// Wraps whisper.cpp for speech-to-text with Metal GPU acceleration.
// Supports word-level timestamps and confidence scores.
//
class Transcriber {
public:
    struct Config {
        std::string model_path;             // path to ggml-*.bin
        std::string language  = "en";       // "en", "auto", etc.
        bool        use_gpu   = true;       // Metal on macOS
        int         n_threads = 4;          // CPU threads
        bool        word_timestamps = true; // extract word-level timing
        bool        translate = false;      // translate to English
        float       no_speech_threshold = 0.6f;
    };

    explicit Transcriber(const Config& config);
    ~Transcriber();

    // Non-copyable
    Transcriber(const Transcriber&) = delete;
    Transcriber& operator=(const Transcriber&) = delete;

    // Transcribe a single audio segment
    TranscriptSegment transcribe_segment(
        const float* audio_16k,
        size_t num_samples,
        int speaker_id,
        double offset_sec);

    // Transcribe all diarized segments
    // Calls progress_cb after each segment with streaming text
    std::vector<TranscriptSegment> transcribe_all(
        const std::vector<float>& full_audio,
        const std::vector<DiarizedSegment>& segments,
        ProgressCallback progress_cb = nullptr);

    const Config& config() const { return config_; }

private:
    Config config_;
    whisper_context* ctx_;
};

}  // namespace vaultasr
