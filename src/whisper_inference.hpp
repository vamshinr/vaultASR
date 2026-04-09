#pragma once
#include <vector>
#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>

class WhisperInference {
public:
    WhisperInference(const std::string& encoder_path, const std::string& decoder_path, Ort::SessionOptions& session_options, Ort::Env& env);
    
    std::string transcribe(const std::vector<float>& mel_spectrogram);

private:
    Ort::Session encoder_session_{nullptr};
    Ort::Session decoder_session_{nullptr};
    Ort::MemoryInfo memory_info_;
};
