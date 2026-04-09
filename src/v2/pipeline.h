#pragma once

#include "types.h"
#include "audio_decoder.h"
#include "denoiser.h"
#include "vad.h"
#include "speaker_embedder.h"
#include "speaker_cluster.h"
#include "transcriber.h"
#include "post_processor.h"
#include "exporter.h"
#include "progress.h"
#include "logger.h"

#include <memory>
#include <string>
#include <vector>

namespace vaultasr {

// ─── Pipeline configuration ────────────────────────────────────────────────
struct PipelineConfig {
    // Input
    std::string audio_path;

    // Denoising
    bool denoise = false;

    // VAD
    VoiceActivityDetector::Config vad_config;

    // Diarization
    std::string embedder_model_path  = "models/wespeaker_resnet34.onnx";
    SpeakerClusterer::Config cluster_config;
    bool skip_diarization = false;  // treat everything as Speaker 0

    // ASR
    Transcriber::Config transcriber_config;

    // Export
    ExportFormat export_format = ExportFormat::STDOUT;
    std::string  output_path;

    // Runtime
    bool stream_text = true;        // print each segment to stderr as transcribed
    bool use_coreml  = true;
};

// ─── Pipeline ─────────────────────────────────────────────────────────────
class Pipeline {
public:
    explicit Pipeline(const PipelineConfig& config);

    // Run the full pipeline on a single file
    // Returns the final transcript
    std::vector<TranscriptSegment> run();

    // Process multiple files sequentially (models loaded once)
    void run_batch(const std::vector<std::string>& audio_paths);

    // Get the progress bar (for CLI to bind to)
    ProgressBar& progress() { return progress_; }

private:
    PipelineConfig config_;
    ProgressBar    progress_;

    // Lazily initialized so models load once across batch runs
    std::unique_ptr<VoiceActivityDetector> vad_;
    std::unique_ptr<SpeakerEmbedder>       embedder_;
    std::unique_ptr<SpeakerClusterer>      clusterer_;
    std::unique_ptr<Transcriber>           transcriber_;
    std::unique_ptr<Denoiser>              denoiser_;

    void ensure_models_loaded();

    // Internal per-file processing
    std::vector<TranscriptSegment> process_file(const std::string& path);
};

}  // namespace vaultasr
