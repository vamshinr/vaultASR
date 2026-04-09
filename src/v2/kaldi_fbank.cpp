#include "kaldi_fbank.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

extern "C" {
#include <libavutil/tx.h>
}

namespace vaultasr {

KaldiFbank::KaldiFbank(int sample_rate, int num_mel_bins,
                       int frame_length_samples, int frame_shift_samples,
                       int fft_size)
    : sample_rate_(sample_rate)
    , num_mel_bins_(num_mel_bins)
    , frame_length_samples_(frame_length_samples)
    , frame_shift_samples_(frame_shift_samples)
    , fft_size_(fft_size)
    , preemph_coeff_(0.97f)
    , tx_ctx_(nullptr)
    , tx_fn_(nullptr)
{
    // Build Hamming window
    window_.resize(frame_length_samples_);
    double two_pi_over_n = 2.0 * M_PI / (frame_length_samples_ - 1);
    for (int i = 0; i < frame_length_samples_; i++) {
        window_[i] = static_cast<float>(0.54 - 0.46 * std::cos(i * two_pi_over_n));
    }

    // Build triangular Mel filterbank
    int num_freq_bins = fft_size_ / 2 + 1;
    mel_filters_.resize(num_mel_bins_ * num_freq_bins, 0.0f);

    double mel_low  = hz_to_mel(20.0);
    double mel_high = hz_to_mel(sample_rate_ / 2.0);

    int n_points = num_mel_bins_ + 2;
    std::vector<double> hz_pts(n_points);
    for (int i = 0; i < n_points; i++) {
        hz_pts[i] = mel_to_hz(mel_low + i * (mel_high - mel_low) / (num_mel_bins_ + 1));
    }

    for (int m = 0; m < num_mel_bins_; m++) {
        double fl = hz_pts[m];
        double fc = hz_pts[m + 1];
        double fh = hz_pts[m + 2];

        for (int k = 0; k < num_freq_bins; k++) {
            double fk = k * static_cast<double>(sample_rate_) / fft_size_;
            float w = 0.0f;
            if (fk >= fl && fk < fc && fc > fl) {
                w = static_cast<float>((fk - fl) / (fc - fl));
            } else if (fk >= fc && fk <= fh && fh > fc) {
                w = static_cast<float>((fh - fk) / (fh - fc));
            }
            mel_filters_[m * num_freq_bins + k] = w;
        }
    }

    // Initialize FFmpeg FFT
    float scale = 1.0f;
    int ret = av_tx_init(&tx_ctx_, &tx_fn_, AV_TX_FLOAT_FFT, 0, fft_size_, &scale, 0);
    if (ret < 0) {
        throw std::runtime_error("Failed to initialize FFT context");
    }

    LOG_DEBUG("KaldiFbank: %d mel bins, %d Hz, frame=%d shift=%d fft=%d",
              num_mel_bins_, sample_rate_, frame_length_samples_,
              frame_shift_samples_, fft_size_);
}

KaldiFbank::~KaldiFbank() {
    if (tx_ctx_) av_tx_uninit(&tx_ctx_);
}

double KaldiFbank::hz_to_mel(double hz) {
    return 1127.0 * std::log(1.0 + hz / 700.0);
}

double KaldiFbank::mel_to_hz(double mel) {
    return 700.0 * (std::exp(mel / 1127.0) - 1.0);
}

std::vector<std::vector<float>> KaldiFbank::compute(const std::vector<float>& waveform) {
    int n = static_cast<int>(waveform.size());
    int num_frames = 1 + (n - frame_length_samples_) / frame_shift_samples_;
    if (num_frames <= 0) return {};

    int num_freq_bins = fft_size_ / 2 + 1;
    std::vector<std::vector<float>> result(num_frames, std::vector<float>(num_mel_bins_, 0.0f));

    // Pre-allocate working buffers
    std::vector<float> frame(frame_length_samples_);

    // AVComplexFloat = struct { float re, im; }
    struct ComplexFloat { float re; float im; };
    std::vector<ComplexFloat> in_buf(fft_size_);
    std::vector<ComplexFloat> out_buf(fft_size_);
    std::vector<float> power(num_freq_bins);

    for (int fi = 0; fi < num_frames; fi++) {
        int offset = fi * frame_shift_samples_;

        // Copy frame and scale to int16 range (Kaldi convention)
        float dc_sum = 0.0f;
        for (int i = 0; i < frame_length_samples_; i++) {
            frame[i] = waveform[offset + i] * 32768.0f;
            dc_sum += frame[i];
        }

        // DC removal
        float dc = dc_sum / frame_length_samples_;
        for (int i = 0; i < frame_length_samples_; i++) {
            frame[i] -= dc;
        }

        // Pre-emphasis
        float prev = frame[0];
        frame[0] *= (1.0f - preemph_coeff_);
        for (int i = 1; i < frame_length_samples_; i++) {
            float curr = frame[i];
            frame[i] -= preemph_coeff_ * prev;
            prev = curr;
        }

        // Zero-pad and apply Hamming window into FFT input buffer
        for (int i = 0; i < fft_size_; i++) {
            in_buf[i].re = 0.0f;
            in_buf[i].im = 0.0f;
        }
        for (int i = 0; i < frame_length_samples_; i++) {
            in_buf[i].re = frame[i] * window_[i];
        }

        // FFT
        tx_fn_(tx_ctx_, out_buf.data(), in_buf.data(), sizeof(ComplexFloat));

        // Power spectrum
        for (int k = 0; k < num_freq_bins; k++) {
            power[k] = out_buf[k].re * out_buf[k].re + out_buf[k].im * out_buf[k].im;
        }

        // Apply mel filterbank + log
        for (int m = 0; m < num_mel_bins_; m++) {
            float energy = 0.0f;
            for (int k = 0; k < num_freq_bins; k++) {
                energy += power[k] * mel_filters_[m * num_freq_bins + k];
            }
            result[fi][m] = std::log(std::max(energy, 1e-10f));
        }
    }

    // Cepstral Mean Normalization (CMN)
    std::vector<float> means(num_mel_bins_, 0.0f);
    for (int fi = 0; fi < num_frames; fi++) {
        for (int m = 0; m < num_mel_bins_; m++) {
            means[m] += result[fi][m];
        }
    }
    for (int m = 0; m < num_mel_bins_; m++) {
        means[m] /= num_frames;
    }
    for (int fi = 0; fi < num_frames; fi++) {
        for (int m = 0; m < num_mel_bins_; m++) {
            result[fi][m] -= means[m];
        }
    }

    LOG_TRACE("Fbank: %d frames x %d bins from %d samples",
              num_frames, num_mel_bins_, n);

    return result;
}

}  // namespace vaultasr
