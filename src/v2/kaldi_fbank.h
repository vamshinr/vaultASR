#pragma once

#include <vector>

// Forward declare FFmpeg types
struct AVTXContext;

namespace vaultasr {

// ─── Kaldi-compatible Mel filterbank feature extractor ─────────────────────
//
// Computes 80-bin log-mel filterbank features compatible with WeSpeaker/Pyannote
// speaker embedding models. Uses FFmpeg's libavutil FFT for performance.
//
// Parameters match Kaldi defaults:
//   - 25ms frame length (400 samples @ 16kHz)
//   - 10ms frame shift  (160 samples @ 16kHz)
//   - 80 mel bins, 20–8000 Hz
//   - Hamming window
//   - Pre-emphasis coefficient 0.97
//   - Cepstral Mean Normalization (CMN) applied per-utterance
//
class KaldiFbank {
public:
    KaldiFbank(int sample_rate = 16000,
               int num_mel_bins = 80,
               int frame_length_samples = 400,
               int frame_shift_samples = 160,
               int fft_size = 512);
    ~KaldiFbank();

    // Non-copyable
    KaldiFbank(const KaldiFbank&) = delete;
    KaldiFbank& operator=(const KaldiFbank&) = delete;

    // Compute filterbank features from raw audio
    // Returns: [num_frames x num_mel_bins] matrix
    std::vector<std::vector<float>> compute(const std::vector<float>& waveform);

    int num_mel_bins() const { return num_mel_bins_; }

private:
    int sample_rate_;
    int num_mel_bins_;
    int frame_length_samples_;
    int frame_shift_samples_;
    int fft_size_;
    float preemph_coeff_;

    std::vector<float> window_;       // Hamming window coefficients
    std::vector<float> mel_filters_;  // [num_mel_bins x num_freq_bins] flat

    // FFmpeg FFT context
    AVTXContext* tx_ctx_;
    void (*tx_fn_)(AVTXContext*, void*, void*, ptrdiff_t);

    static double hz_to_mel(double hz);
    static double mel_to_hz(double mel);
};

}  // namespace vaultasr
