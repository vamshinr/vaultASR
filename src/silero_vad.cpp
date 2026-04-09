#include "silero_vad.hpp"
#include <iostream>
#include <stdexcept>

SileroVad::SileroVad(const std::string& model_path, Ort::SessionOptions& session_options, Ort::Env& env) 
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    session_ = Ort::Session(env, model_path.c_str(), session_options);
    
    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_input_nodes = session_.GetInputCount();
    std::cout << "[VAD DIAGNOSTICS] Model expects " << num_input_nodes << " inputs:\n";
    for (size_t i = 0; i < num_input_nodes; i++) {
        // Modern approach to get string from unique_ptr
        auto name_ptr = session_.GetInputNameAllocated(i, allocator);
        std::cout << "  - " << name_ptr.get() << "\n";
    }

    // Allocate generic state size for now
    _h.resize(2 * 1 * 128, 0.0f);
    _c.resize(2 * 1 * 128, 0.0f);
}

std::vector<VadSegment> SileroVad::get_speech_segments(const std::vector<float>& audio_data, float threshold) {
    std::vector<VadSegment> segments;
    const int window_size = 512;
    const int sample_rate = 16000;
    
    // reset states
    std::fill(_h.begin(), _h.end(), 0.0f);
    std::fill(_c.begin(), _c.end(), 0.0f);

    int64_t sr = 16000;
    
    std::vector<int64_t> input_node_dims = {1, window_size};
    std::vector<int64_t> hc_node_dims = {2, 1, 128};
    std::vector<int64_t> sr_node_dims = {1};

    bool triggered = false;
    double temp_start = 0.0;

    for (size_t i = 0; i + window_size <= audio_data.size(); i += window_size) {
        std::vector<float> window(audio_data.begin() + i, audio_data.begin() + i + window_size);
        
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info_, window.data(), window.size(), input_node_dims.data(), input_node_dims.size());
        Ort::Value sr_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, &sr, 1, sr_node_dims.data(), sr_node_dims.size());
        
        // Silero V4/V5 typically uses 'state' merged vector instead of h/c, but if it expects h and c, wait.
        // We saw the inputs are: "input", "state", "sr".
        // State is often 2x1x128. I named it _h and _c, let's just use _h representing the single state vector.
        Ort::Value state_tensor = Ort::Value::CreateTensor<float>(memory_info_, _h.data(), _h.size(), hc_node_dims.data(), hc_node_dims.size());

        const char* input_names[] = {"input", "state", "sr"};
        const char* output_names[] = {"output", "stateN"};

        std::vector<Ort::Value> input_tensors;
        input_tensors.push_back(std::move(input_tensor));
        input_tensors.push_back(std::move(state_tensor));
        input_tensors.push_back(std::move(sr_tensor));

        auto output_tensors = session_.Run(Ort::RunOptions{nullptr}, input_names, input_tensors.data(), 3, output_names, 2);
        
        float prob = output_tensors[0].GetTensorMutableData<float>()[0];
        
        // update states
        float* stateN = output_tensors[1].GetTensorMutableData<float>();
        std::copy(stateN, stateN + _h.size(), _h.begin());

        double current_time = static_cast<double>(i) / sample_rate;
        if (prob >= threshold && !triggered) {
            triggered = true;
            temp_start = current_time;
        } else if (prob < (threshold - 0.15) && triggered) {
            triggered = false;
            segments.push_back({temp_start, current_time});
        }
    }
    
    if (triggered) {
        segments.push_back({temp_start, audio_data.size() / static_cast<double>(sample_rate)});
    }
    
    return segments;
}
