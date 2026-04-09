#pragma once
#include <vector>
#include <string>
#include <onnxruntime/onnxruntime_cxx_api.h>

class WeSpeakerEmbedder {
public:
    WeSpeakerEmbedder(const std::string& model_path, Ort::SessionOptions& session_options, Ort::Env& env);
    ~WeSpeakerEmbedder();

    std::vector<float> compute_embedding(const std::vector<std::vector<float>>& fbank);

private:
    Ort::Session* session_;
    Ort::AllocatorWithDefaultOptions allocator_;
    
    std::string input_name_;
    std::string output_name_;
};
