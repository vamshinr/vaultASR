#include "denoiser.h"
#include "logger.h"
#include <cstring>
#include <stdexcept>

// RNNoise C API
extern "C" {
#include <rnnoise.h>
}

extern "C" {
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace vaultasr {

// ─── Helper: create a resampler ────────────────────────────────────────────
static SwrContext* make_resampler(int in_rate, int out_rate) {
    SwrContext* ctx = nullptr;

    AVChannelLayout mono_layout;
    av_channel_layout_default(&mono_layout, 1);

    int ret = swr_alloc_set_opts2(
        &ctx,
        &mono_layout, AV_SAMPLE_FMT_FLT, out_rate,
        &mono_layout, AV_SAMPLE_FMT_FLT, in_rate,
        0, nullptr);

    if (ret < 0 || !ctx) {
        throw std::runtime_error("Failed to create denoiser resampler");
    }

    if (swr_init(ctx) < 0) {
        swr_free(&ctx);
        throw std::runtime_error("Failed to init denoiser resampler");
    }

    return ctx;
}

// ─── Denoiser implementation ───────────────────────────────────────────────

Denoiser::Denoiser() {
    rnn_state_ = rnnoise_create(nullptr);  // nullptr = use built-in model
    if (!rnn_state_) {
        throw std::runtime_error("Failed to create RNNoise state");
    }

    up_ctx_   = make_resampler(16000, 48000);
    down_ctx_ = make_resampler(48000, 16000);

    LOG_INFO("RNNoise denoiser initialized (16kHz <-> 48kHz resampling)");
}

Denoiser::~Denoiser() {
    if (rnn_state_) rnnoise_destroy(rnn_state_);
    if (up_ctx_)    swr_free(&up_ctx_);
    if (down_ctx_)  swr_free(&down_ctx_);
}

void Denoiser::reset() {
    if (rnn_state_) rnnoise_destroy(rnn_state_);
    rnn_state_ = rnnoise_create(nullptr);
}

void Denoiser::process(std::vector<float>& audio_16k) {
    if (audio_16k.empty()) return;

    LOG_DEBUG("Denoising %zu samples (%.2fs)...",
              audio_16k.size(), audio_16k.size() / 16000.0);

    // Step 1: Upsample 16kHz → 48kHz
    int in_samples = static_cast<int>(audio_16k.size());
    int out_48k_count = in_samples * 3 + 256;  // 48k/16k = 3x, plus some padding
    std::vector<float> audio_48k(out_48k_count);

    const uint8_t* in_data[1] = {reinterpret_cast<const uint8_t*>(audio_16k.data())};
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(audio_48k.data())};

    int converted = swr_convert(up_ctx_, out_data, out_48k_count,
                                 in_data, in_samples);
    if (converted <= 0) {
        LOG_WARN("Upsample produced 0 samples, skipping denoise");
        return;
    }
    audio_48k.resize(converted);

    LOG_TRACE("Upsampled to %d samples at 48kHz", converted);

    // Step 2: Process through RNNoise in 480-sample frames (10ms @ 48kHz)
    // RNNoise expects float input in range [-32768, 32767] (int16 scale)
    const int frame_size = 480;  // RNNoise frame size
    std::vector<float> rnn_frame(frame_size);

    for (size_t i = 0; i + frame_size <= audio_48k.size(); i += frame_size) {
        // Scale up to int16 range for RNNoise
        for (int j = 0; j < frame_size; j++) {
            rnn_frame[j] = audio_48k[i + j] * 32768.0f;
        }

        float vad_prob = rnnoise_process_frame(rnn_state_, rnn_frame.data(), rnn_frame.data());
        (void)vad_prob;  // RNNoise also returns a VAD probability, unused here

        // Scale back to float range
        for (int j = 0; j < frame_size; j++) {
            audio_48k[i + j] = rnn_frame[j] / 32768.0f;
        }
    }

    LOG_TRACE("RNNoise processed %zu frames", audio_48k.size() / frame_size);

    // Step 3: Downsample 48kHz → 16kHz
    int out_16k_count = static_cast<int>(audio_48k.size()) / 3 + 256;
    std::vector<float> denoised_16k(out_16k_count);

    const uint8_t* in48_data[1] = {reinterpret_cast<const uint8_t*>(audio_48k.data())};
    uint8_t* out16_data[1] = {reinterpret_cast<uint8_t*>(denoised_16k.data())};

    converted = swr_convert(down_ctx_, out16_data, out_16k_count,
                             in48_data, static_cast<int>(audio_48k.size()));

    if (converted > 0) {
        denoised_16k.resize(converted);
        // Replace original audio with denoised version
        // Trim or pad to match original length
        if (denoised_16k.size() >= audio_16k.size()) {
            std::memcpy(audio_16k.data(), denoised_16k.data(),
                        audio_16k.size() * sizeof(float));
        } else {
            std::memcpy(audio_16k.data(), denoised_16k.data(),
                        denoised_16k.size() * sizeof(float));
            // Zero-fill remainder
            std::memset(audio_16k.data() + denoised_16k.size(), 0,
                        (audio_16k.size() - denoised_16k.size()) * sizeof(float));
        }
    }

    LOG_INFO("Denoising complete: %zu samples processed", audio_16k.size());
}

}  // namespace vaultasr
