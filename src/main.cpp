#include <iostream>
#include <string>
#include <vector>
#include <onnxruntime/coreml_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include "audio_utils.hpp"
#include "silero_vad.hpp"
#include "mel_spectrogram.hpp"
#include "whisper_inference.hpp"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
}

// NOTE: Mac/Apple CoreML Execution Provider Header (dynamically handled by ORT session options)

int main(int argc, char* argv[]) {
    bool trace = false;
    std::string audioPath = "";
    
    // 1. Argument Parsing
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--trace") {
            trace = true;
        } else if (arg == "--audio" && i + 1 < argc) {
            audioPath = argv[++i];
        }
    }
    
    if (audioPath.empty()) {
        std::cerr << "Usage: vernacula_cpp --audio <path> [--trace]\n";
        return 1;
    }
    
    if (trace) {
        std::cout << "[EXECUTION TRACE] Vernacula C++ Engine starting...\n";
        std::cout << "[EXECUTION TRACE] Audio Target: " << audioPath << "\n";
    }
    
    try {
        // 2. Init ONNX Runtime Environment
        if (trace) std::cout << "[EXECUTION TRACE] Initializing ONNX Runtime Environment.\n";
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VernaculaEnv");
        Ort::SessionOptions session_options;
        
        // 3. Attempt to load Apple CoreML EP
        // According to ORT C++ spec, CoreML is configured with flags. 0 = default.
        uint32_t coreml_flags = 0;
        try {
            Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CoreML(session_options, coreml_flags));
            if (trace) {
                std::cout << "[EXECUTION TRACE] ⚡ Apple CoreML (ANE) Execution Provider successfully attached!\n";
            }
        } catch (const Ort::Exception& e) {
            std::cerr << "[EXECUTION TRACE] CoreML EP not available or failed to load: " << e.what() << ". Falling back to CPU.\n";
        }
        
        
        // 4. Ingest Audio via FFmpeg
        if (trace) std::cout << "[EXECUTION TRACE] Calling FFmpeg to decode audio into 16kHz Mono float array...\n";
        std::vector<float> audio_data = load_audio_16k_mono(audioPath, trace);
        
        // 5. Load Silero VAD
        if (trace) std::cout << "[EXECUTION TRACE] Initializing Silero VAD v4/v5 ONNX Session...\n";
        std::string vad_model_path = "../models/silero_vad.onnx";
        SileroVad vad(vad_model_path, session_options, env);
        
        if (trace) std::cout << "[EXECUTION TRACE] Running VAD segmentation over " << audio_data.size() << " samples...\n";
        std::vector<VadSegment> segments = vad.get_speech_segments(audio_data);
        
        if (trace) {
            std::cout << "[EXECUTION TRACE] Found " << segments.size() << " speech segments:\n";
            for (size_t i = 0; i < std::min(segments.size(), (size_t)5); i++) {
                std::cout << "  -> Segment " << i << ": [" << segments[i].start_sec << "s - " << segments[i].end_sec << "s]\n";
            }
            if (segments.size() > 5) std::cout << "  -> ... and " << (segments.size() - 5) << " more.\n";
        }
        
        // 6. Next step: Whisper Inference (Log Mel + Tokenizer mappings)
        if (trace) std::cout << "[EXECUTION TRACE] Initializing C++ Mel Spectrogram & Whisper Tiny ONNX...\n";
        
        MelSpectrogram mel_synthesizer;
        std::string enc_path = "../models/whisper_tiny_en/encoder_model.onnx";
        std::string dec_path = "../models/whisper_tiny_en/decoder_model.onnx";
        WhisperInference whisper(enc_path, dec_path, session_options, env);
        
        std::cout << "\n=============================================\n";
        std::cout << "             TRANSCRIPT OUTPUT               \n";
        std::cout << "=============================================\n\n";

        if (segments.empty()) {
            if (trace) std::cout << "[EXECUTION TRACE] VAD found 0 segments. Falling back to strict 30-second chunks...\n";
            for (size_t i = 0; i < audio_data.size(); i += 16000 * 30) {
                double s_time = i / 16000.0;
                double e_time = std::min(s_time + 30.0, audio_data.size() / 16000.0);
                segments.push_back({s_time, e_time});
            }
        }

        for (size_t i = 0; i < segments.size(); i++) {
            // Cut segment audio
            int start_idx = static_cast<int>(segments[i].start_sec * 16000);
            int end_idx = static_cast<int>(segments[i].end_sec * 16000);
            if (start_idx < 0) start_idx = 0;
            if (end_idx > audio_data.size()) end_idx = audio_data.size();
            
            std::vector<float> chunk(audio_data.begin() + start_idx, audio_data.begin() + end_idx);
            
            if (chunk.size() < 1600) continue; 
            
            // Generate Mel Spec
            std::vector<float> mel_tensor = mel_synthesizer.compute(chunk);
            
            // Execute C++ ONNX Whisper
            std::string text = whisper.transcribe(mel_tensor);
            
            std::cout << "[" << segments[i].start_sec << "s - " << segments[i].end_sec << "s]: " << text << "\n";
            std::cout.flush();
        }
        
        std::cout << "\n=============================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Fatal Error: " << e.what() << "\n";
        return 1;
    }
    
    if (trace) {
        std::cout << "[EXECUTION TRACE] Vernacula C++ Engine shutting down cleanly.\n";
    }
    
    return 0;
}
