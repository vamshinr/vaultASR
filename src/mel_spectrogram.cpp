#include "mel_spectrogram.hpp"
#include "whisper_mel_filters.hpp"
#include <cmath>
#include <algorithm>

MelSpectrogram::MelSpectrogram() {
    n_fft = 400;
    hop_length = 160;
    n_mels = 80;
    
    window_hann = new float[n_fft];
    for(int i = 0; i < n_fft; i++) {
        window_hann[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / n_fft));
    }
    
    float scale = 1.0f;
    av_tx_init(&tx_ctx, &tx_fn, AV_TX_FLOAT_FFT, 0, n_fft, &scale, 0);
}

MelSpectrogram::~MelSpectrogram() {
    delete[] window_hann;
    av_tx_uninit(&tx_ctx);
}

std::vector<float> MelSpectrogram::compute(const std::vector<float>& audio) {
    int n_samples = 480000;
    std::vector<float> padded(n_samples, 0.0f);
    int cpy_len = std::min((int)audio.size(), n_samples);
    std::copy(audio.begin(), audio.begin() + cpy_len, padded.begin());
    
    int n_frames = 3000;

    std::vector<float> mel(n_mels * n_frames, 0.0f);
    
    std::vector<AVComplexFloat> in_buf(n_fft);
    std::vector<AVComplexFloat> out_buf(n_fft);
    
    for(int i = 0; i < n_frames; i++) {
        int start = i * hop_length;
        for(int j = 0; j < n_fft; j++) {
            in_buf[j].re = padded[start + j] * window_hann[j];
            in_buf[j].im = 0.0f;
        }
        
        tx_fn(tx_ctx, out_buf.data(), in_buf.data(), sizeof(AVComplexFloat));
        
        std::vector<float> spec_power(n_fft / 2 + 1, 0.0f);
        // Add DC
        spec_power[0] = out_buf[0].re * out_buf[0].re + out_buf[0].im * out_buf[0].im;
        for(int j = 1; j <= n_fft / 2; j++) {
            spec_power[j] = out_buf[j].re * out_buf[j].re + out_buf[j].im * out_buf[j].im;
        }
        
        for(int m = 0; m < n_mels; m++) {
            float sum = 0.0f;
            for(int j = 0; j <= n_fft / 2; j++) { 
                sum += spec_power[j] * whisper_mel_filters[m][j];
            }
            sum = std::max(sum, 1e-10f);
            mel[m * n_frames + i] = log10f(sum);
        }
    }
    
    float max_mel = -1e10f;
    for(float v : mel) if(v > max_mel) max_mel = v;
    float cutoff = max_mel - 8.0f;
    for(size_t i = 0; i < mel.size(); i++) {
        mel[i] = (std::max(mel[i], cutoff) + 4.0f) / 4.0f;
    }
    
    return mel;
}
