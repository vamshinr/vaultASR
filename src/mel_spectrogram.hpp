#pragma once
#include <vector>

extern "C" {
#include <libavutil/tx.h>
}

class MelSpectrogram {
public:
    MelSpectrogram();
    ~MelSpectrogram();
    
    // Whisper standard mapping: 16k mono audio -> log-mel spectrogram tensor data [80 x 3000]
    std::vector<float> compute(const std::vector<float>& audio);

private:
    struct AVTXContext *tx_ctx;
    av_tx_fn tx_fn;
    float* window_hann;
    int n_fft;
    int hop_length;
    int n_mels;
};
