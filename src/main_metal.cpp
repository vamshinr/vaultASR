#include "../external/whisper.cpp/include/whisper.h"
#include "audio_utils.hpp"
#include "clustering.hpp"
#include "kaldi_fbank.hpp"
#include "silero_vad.hpp"
#include "wespeaker_embedder.hpp"
#include <chrono>
#include <iostream>
#include <fstream>
#include <onnxruntime/coreml_provider_factory.h>
#include <onnxruntime/onnxruntime_cxx_api.h>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/log.h>
}

int main(int argc, char *argv[]) {
  bool trace = false;
  std::string audio_path = "";
  std::string csv_path = "";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--trace")
      trace = true;
    else if (arg == "--audio" && i + 1 < argc)
      audio_path = argv[++i];
    else if (arg == "--csv" && i + 1 < argc)
      csv_path = argv[++i];
  }

  if (audio_path.empty()) {
    std::cerr << "Usage: ./vaultasr --audio <path_to_audio> [--trace] [--csv <path.csv>]\n";
    return 1;
  }

  if (trace)
    std::cout << "[EXECUTION TRACE] Vernacula Metal Pipeline starting...\n";
  if (trace)
    std::cout << "[EXECUTION TRACE] Target: " << audio_path << "\n";

  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "VernaculaVAD");
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  uint32_t coreml_flags = 0;
  OrtSessionOptionsAppendExecutionProvider_CoreML(session_options,
                                                  coreml_flags);
  if (trace)
    std::cout
        << "[EXECUTION TRACE] VAD utilizing Apple CoreML Execution Provider!\n";

  std::vector<float> audio_data;
  try {
    if (trace)
      std::cout << "[EXECUTION TRACE] Decoding audio into 16kHz Mono float "
                   "array...\n";
    audio_data = load_audio_16k_mono(audio_path, trace);
    if (trace)
      std::cout << "[EXECUTION TRACE] Loaded " << audio_data.size()
                << " samples (" << (audio_data.size() / 16000.0)
                << " seconds).\n";

    // TRUNCATE FOR QUICK TESTING AS REQUESTED BY USER
    if (audio_data.size() > 16000 * 150) {
      if (trace)
        std::cout << "[EXECUTION TRACE] Truncating to 150 seconds for rapid "
                     "testing...\n";
      audio_data.resize(16000 * 150);
    }

    if (trace)
      std::cout
          << "[EXECUTION TRACE] Initializing Silero VAD v5 ONNX Session...\n";
    SileroVad vad("../models/silero_vad.onnx", session_options, env);

    if (trace)
      std::cout
          << "[EXECUTION TRACE] Initializing WeSpeaker Diarization Models...\n";
    KaldiFbank fbank_maker;
    WeSpeakerEmbedder speaker_embedder("../models/wespeaker_pyannote.onnx",
                                       session_options, env);
    Clustering clusterer(
        0.60f); // Lenient 0.60 Cosine distance threshold for unique identities

    if (trace)
      std::cout << "[EXECUTION TRACE] Phase 3: Unrolling continuous "
                   "Sliding-Window Diarization...\n";

    double window_sz = 1.5;
    double hop_sz = 0.5;
    int win_samples = static_cast<int>(window_sz * 16000);
    int hop_samples = static_cast<int>(hop_sz * 16000);

    std::vector<std::vector<float>> all_embs;
    std::vector<double> window_starts;
    std::vector<double> window_ends;

    for (size_t i = 0; i + win_samples <= audio_data.size(); i += hop_samples) {
      std::vector<float> chunk(audio_data.begin() + i,
                               audio_data.begin() + i + win_samples);
      auto fbank = fbank_maker.compute(chunk);
      auto emb = speaker_embedder.compute_embedding(fbank);
      all_embs.push_back(emb);

      // The label applies cleanly to the hop region!
      window_starts.push_back(i / 16000.0);
      window_ends.push_back((i + hop_samples) / 16000.0);
    }

    if (trace)
      std::cout << "[EXECUTION TRACE] Running Agglomerative Hierarchical "
                   "Clustering...\n";
    std::vector<int> raw_labels = clusterer.cluster(all_embs);

    // Dynamically stitch adjacent same-speaker windows into continuous
    // conversation segments!
    std::vector<VadSegment> final_segments;
    std::vector<int> final_speakers;

    if (!raw_labels.empty()) {
      int current_speaker = raw_labels[0];
      double start_t = window_starts[0];
      double end_t = window_ends[0];

      for (size_t i = 1; i < raw_labels.size(); i++) {
        if (raw_labels[i] == current_speaker) {
          end_t = window_ends[i]; // Extend the speaker's segment
        } else {
          final_segments.push_back({start_t, end_t});
          final_speakers.push_back(current_speaker);

          current_speaker = raw_labels[i];
          start_t = window_starts[i];
          end_t = window_ends[i];
        }
      }
      final_segments.push_back({start_t, end_t});
      final_speakers.push_back(current_speaker);
    }

    if (trace)
      std::cout << "[EXECUTION TRACE] Segregated into " << final_segments.size()
                << " discrete continuous speaker blocks!\n";

    if (trace)
      std::cout
          << "[EXECUTION TRACE] Initializing GGML Metal Whisper Backend...\n";

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = true; // explicitly trigger Metal

    struct whisper_context *ctx = whisper_init_from_file_with_params(
        "../models/ggml-base.en.bin", cparams);
    if (!ctx) {
      std::cerr << "Failed to initialize whisper context. Ensure "
                   "ggml-base.en.bin exists in models/\n";
      return 1;
    }

    if (trace)
      std::cout << "[EXECUTION TRACE] GGML Whisper Context successfully loaded "
                   "onto Metal Backend!\n";

    std::cout << "\n=============================================\n";
    std::cout << "             METAL TRANSCRIPT                \n";
    std::cout << "=============================================\n\n";

    struct TranscriptBlock {
        int speaker;
        std::string text;
    };
    std::vector<TranscriptBlock> transcript;

    for (size_t i = 0; i < final_segments.size(); i++) {
      if (final_speakers[i] == -1) continue; // Intentionally hide [SILENCE] from the final Otter-like product
      
      int start_idx = static_cast<int>(final_segments[i].start_sec * 16000);
      int end_idx = static_cast<int>(final_segments[i].end_sec * 16000);
      if (start_idx < 0)
        start_idx = 0;
      if (end_idx > audio_data.size())
        end_idx = audio_data.size();
      std::vector<float> chunk(audio_data.begin() + start_idx,
                               audio_data.begin() + end_idx);
      if (chunk.size() < 1600)
        continue;

      whisper_full_params wparams =
          whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
      wparams.print_progress = false;
      wparams.print_special = false;
      wparams.print_realtime = false;
      wparams.print_timestamps = false;
      wparams.language = "en";
      wparams.n_threads = 4;

      if (whisper_full(ctx, wparams, chunk.data(), chunk.size()) != 0) {
        std::cerr << "failed to process audio chunk\n";
        continue;
      }

      const int n_segments = whisper_full_n_segments(ctx);
      std::string text = "";
      for (int s = 0; s < n_segments; ++s) {
        const char *txt = whisper_full_get_segment_text(ctx, s);
        if (txt)
          text += txt;
      }

      // Strip excess whitespace from whisper output logic to cleanly collapse
      if (!text.empty() && text[0] == ' ') text = text.substr(1);

      if (!transcript.empty() && transcript.back().speaker == final_speakers[i]) {
          transcript.back().text += " " + text;
      } else {
          transcript.push_back({final_speakers[i], text});
      }
    }

    // Print Otter-like Human Readable Output
    for (const auto& block : transcript) {
        std::cout << "Speaker " << block.speaker << ":\n";
        std::cout << block.text << "\n\n";
    }

    // CSV Exporter Loop
    if (!csv_path.empty()) {
        std::ofstream csv_file(csv_path);
        if (csv_file.is_open()) {
            csv_file << "Speaker,Text\n";
            for (const auto& block : transcript) {
                // Escape internal quotes to preserve CSV integrity
                std::string escaped_text = block.text;
                size_t pos = 0;
                while ((pos = escaped_text.find("\"", pos)) != std::string::npos) {
                     escaped_text.replace(pos, 1, "\"\"");
                     pos += 2;
                }
                csv_file << "Speaker " << block.speaker << ",\"" << escaped_text << "\"\n";
            }
            csv_file.close();
            if (trace) std::cout << "[EXECUTION TRACE] Safely exported CSV transcript block to: " << csv_path << "\n";
        } else {
            std::cerr << "Failed to open CSV path for writing: " << csv_path << "\n";
        }
    }

    std::cout << "=============================================\n";

    whisper_free(ctx);
    if (trace)
      std::cout << "[EXECUTION TRACE] Metal Engine shutting down cleanly.\n";

  } catch (const std::exception &e) {
    std::cerr << "Fatal Error: " << e.what() << "\n";
    return 1;
  }

  return 0;
}
