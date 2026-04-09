#include "wespeaker_embedder.hpp"
#include <iostream>
#include <cmath>

WeSpeakerEmbedder::WeSpeakerEmbedder(const std::string& model_path, Ort::SessionOptions& session_options, Ort::Env& env) {
    session_ = new Ort::Session(env, model_path.c_str(), session_options);
    
    Ort::TypeInfo type_info = session_->GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    
    // Hardcoded input/output names typical for WeSpeaker Pyannote
    input_name_ = "input_features"; // From huggingface model.onnx
    output_name_ = "last_hidden_state";
}

WeSpeakerEmbedder::~WeSpeakerEmbedder() {
    delete session_;
}

std::vector<float> WeSpeakerEmbedder::compute_embedding(const std::vector<std::vector<float>>& fbank) {
    int num_frames = fbank.size();
    if (num_frames == 0) return std::vector<float>(256, 0.0f);
    
    int num_mels = fbank[0].size(); // Should be 80
    
    std::vector<float> flat_fbank(num_frames * num_mels);
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < num_mels; j++) {
            flat_fbank[i * num_mels + j] = fbank[i][j];
        }
    }
    
    std::vector<int64_t> input_shape = {1, num_frames, num_mels};
    
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, flat_fbank.data(), flat_fbank.size(),
        input_shape.data(), input_shape.size()
    );
    
    const char* input_names[] = { input_name_.c_str() };
    const char* output_names[] = { output_name_.c_str() };
    
    auto output_tensors = session_->Run( Ort::RunOptions{nullptr}, 
        input_names, &input_tensor, 1, 
        output_names, 1
    );
    
    float* out_data = output_tensors[0].GetTensorMutableData<float>();
    auto type_info = output_tensors[0].GetTensorTypeAndShapeInfo();
    auto out_shape = type_info.GetShape();
    
    int embedding_dim = out_shape[1]; // Usually 256
    
    std::vector<float> embedding(embedding_dim);
    float l2_norm = 0.0f;
    for (int i = 0; i < embedding_dim; i++) {
        embedding[i] = out_data[i];
        l2_norm += embedding[i] * embedding[i];
    }
    
    l2_norm = std::sqrt(l2_norm);
    if (l2_norm > 0) {
        for (int i = 0; i < embedding_dim; i++) {
            embedding[i] /= l2_norm;
        }
    }
    
    return embedding;
}
