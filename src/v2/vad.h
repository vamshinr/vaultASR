#pragma once

#include "types.h"
#include <memory>
#include <string>
#include <vector>

// Forward declare ONNX Runtime types to keep header clean
namespace Ort {
class Env;
class Session;
class SessionOptions;
class MemoryInfo;
}  // namespace Ort

namespace vaultasr {

// ─── Silero VAD v5 wrapper ─────────────────────────────────────────────────
//
// Processes 16kHz mono audio through Silero VAD v5 ONNX model.
// Uses 512-sample windows (32ms) with stateful recurrence.
// Returns merged speech segments with hysteresis-based thresholding.
//
class VoiceActivityDetector {
public:
    struct Config {
        float  threshold       = 0.5f;    // speech probability threshold to trigger
        float  neg_threshold   = 0.35f;   // threshold to end speech (hysteresis)
        double min_speech_sec  = 0.25;    // discard speech shorter than this
        double min_silence_sec = 0.3;     // min silence duration to split
        double max_segment_sec = 30.0;    // force split long segments (for Whisper)
        double speech_pad_sec  = 0.1;     // padding around speech boundaries
    };

    VoiceActivityDetector(const std::string& model_path, Config config);
    explicit VoiceActivityDetector(const std::string& model_path);  // default config overload
    ~VoiceActivityDetector();

    // Non-copyable
    VoiceActivityDetector(const VoiceActivityDetector&) = delete;
    VoiceActivityDetector& operator=(const VoiceActivityDetector&) = delete;

    // Detect speech segments in 16kHz mono audio
    std::vector<SpeechSegment> detect(const std::vector<float>& audio_16k);

    // Reset internal state for new file
    void reset();

    const Config& config() const { return config_; }

private:
    Config config_;

    // ONNX Runtime handles (pimpl to avoid header pollution)
    struct OrtState;
    std::unique_ptr<OrtState> ort_;

    // Run inference on a single 512-sample window, return speech probability
    float infer_frame(const float* data);

    // Merge raw triggers into clean segments
    std::vector<SpeechSegment> post_process(
        const std::vector<std::pair<double, float>>& raw_probs,
        double total_duration);
};

}  // namespace vaultasr
