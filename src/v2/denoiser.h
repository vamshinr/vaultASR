#pragma once

#include <vector>

// Forward declaration — RNNoise is a C library
struct DenoiseState;
struct SwrContext;

namespace vaultasr {

// ─── RNNoise denoiser wrapper ──────────────────────────────────────────────
//
// RNNoise operates at 48kHz with 480-sample frames (10ms).
// This class handles the 16kHz↔48kHz resampling transparently.
//
class Denoiser {
public:
    Denoiser();
    ~Denoiser();

    // Non-copyable
    Denoiser(const Denoiser&) = delete;
    Denoiser& operator=(const Denoiser&) = delete;

    // In-place denoise. Audio must be 16kHz mono float32.
    void process(std::vector<float>& audio_16k);

    // Reset internal state (for new file)
    void reset();

private:
    DenoiseState* rnn_state_;
    SwrContext*   up_ctx_;    // 16k → 48k
    SwrContext*   down_ctx_;  // 48k → 16k
};

}  // namespace vaultasr
