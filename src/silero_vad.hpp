#pragma once
#include <vector>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>

struct VadSegment {
    double start_sec;
    double end_sec;
};

class SileroVad {
public:
    SileroVad(const std::string& model_path, Ort::SessionOptions& session_options, Ort::Env& env);
    std::vector<VadSegment> get_speech_segments(const std::vector<float>& audio_data, float threshold = 0.5f);
private:
    Ort::Session session_{nullptr};
    Ort::MemoryInfo memory_info_;
    std::vector<float> _h;
    std::vector<float> _c;
};
