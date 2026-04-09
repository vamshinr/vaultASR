#include "whisper_inference.hpp"
#include "whisper_tokens.hpp"
#include <iostream>
#include <algorithm>

WhisperInference::WhisperInference(const std::string& encoder_path, const std::string& decoder_path, Ort::SessionOptions& session_options, Ort::Env& env) 
    : memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    encoder_session_ = Ort::Session(env, encoder_path.c_str(), session_options);
    decoder_session_ = Ort::Session(env, decoder_path.c_str(), session_options);
}

std::string WhisperInference::transcribe(const std::vector<float>& mel_spectrogram) {
    if (mel_spectrogram.size() != 80 * 3000) {
        std::cerr << "Invalid MEL Size\n";
        return "";
    }
    
    std::vector<int64_t> enc_input_dims = {1, 80, 3000};
    Ort::Value enc_input_tensor = Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(mel_spectrogram.data()), mel_spectrogram.size(), enc_input_dims.data(), enc_input_dims.size());
    
    const char* enc_input_names[] = {"input_features"};
    const char* enc_output_names[] = {"last_hidden_state"};
    
    auto enc_outputs = encoder_session_.Run(Ort::RunOptions{nullptr}, enc_input_names, &enc_input_tensor, 1, enc_output_names, 1);
    
    Ort::Value& hidden_state_tensor = enc_outputs[0];
    auto type_info = hidden_state_tensor.GetTensorTypeAndShapeInfo();
    auto hidden_shape = type_info.GetShape(); 
    
    // tiny.en requires exactly [50257, 50362] for <|startoftranscript|> and <|notimestamps|>
    std::vector<int64_t> input_ids = {50257, 50362};
    std::string transcript = "";
    
    const char* dec_input_names[] = {"input_ids", "encoder_hidden_states"};
    const char* dec_output_names[] = {"logits"};

    int max_tokens = 400;
    
    for (int step = 0; step < max_tokens; step++) {
        std::vector<int64_t> dec_input_ids_dims = {1, (int64_t)input_ids.size()};
        Ort::Value dec_input_tensor = Ort::Value::CreateTensor<int64_t>(memory_info_, input_ids.data(), input_ids.size(), dec_input_ids_dims.data(), dec_input_ids_dims.size());
        
        std::vector<Ort::Value> dec_inputs;
        dec_inputs.push_back(std::move(dec_input_tensor));
        
        Ort::Value hs_ref = Ort::Value::CreateTensor<float>(memory_info_, hidden_state_tensor.GetTensorMutableData<float>(), type_info.GetElementCount(), hidden_shape.data(), hidden_shape.size());
        dec_inputs.push_back(std::move(hs_ref));

        auto dec_outputs = decoder_session_.Run(Ort::RunOptions{nullptr}, dec_input_names, dec_inputs.data(), 2, dec_output_names, 1);
        
        auto logits_info = dec_outputs[0].GetTensorTypeAndShapeInfo();
        auto logits_shape = logits_info.GetShape();
        int64_t vocab_size = logits_shape[2];
        
        float* logits_data = dec_outputs[0].GetTensorMutableData<float>();
        float* last_logits = logits_data + ((input_ids.size() - 1) * vocab_size);
        
        int64_t best_token = 0;
        float max_logit = -1e10f;
        for (int64_t v = 0; v < vocab_size; v++) {
            if (last_logits[v] > max_logit) {
                max_logit = last_logits[v];
                best_token = v;
            }
        }
        
        input_ids.push_back(best_token);
        
        if (best_token == 50256) { // <|endoftext|> in tiny.en
            break;
        }
        
        // Exclude special tokens from transcript dynamically
        if (best_token < whisper_vocab.size() && best_token < 50256) {
            std::string token_str = whisper_vocab[best_token];
            
            size_t pos = 0;
            while ((pos = token_str.find("\xc4\xa0", pos)) != std::string::npos) {
                token_str.replace(pos, 2, " ");
                pos += 1;
            }
            if (token_str != "<|startoftranscript|>" && token_str != "<|endoftext|>" && token_str.find("<|") == std::string::npos) {
                transcript += token_str;
            }
        }
    }
    
    return transcript;
}
