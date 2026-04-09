#pragma once
#include <vector>

extern "C" {
#include <libavutil/tx.h>
}

class KaldiFbank {
public:
    KaldiFbank(int sample_rate = 16000, int num_mel_bins = 80, 
               int frame_length_samples = 400, int frame_shift_samples = 160, 
               int fft_size = 512);
    ~KaldiFbank();

    std::vector<std::vector<float>> compute(const std::vector<float>& waveform);

private:
    struct AVTXContext *tx_ctx;
    av_tx_fn tx_fn;

    int sample_rate_;
    int num_mel_bins_;
    int frame_length_samples_;
    int frame_shift_samples_;
    int fft_size_;
    float preemph_coeff_;

    std::vector<float> window_;
    std::vector<float> mel_filters_;
    
    double hz_to_mel(double hz);
    double mel_to_hz(double mel);
};
