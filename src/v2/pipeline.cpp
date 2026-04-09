#include "pipeline.h"

#include <chrono>
#include <cstdio>
#include <stdexcept>

namespace vaultasr {

// ─── Constructor ───────────────────────────────────────────────────────────

Pipeline::Pipeline(const PipelineConfig& config) : config_(config) {}

// ─── Lazy model loading ────────────────────────────────────────────────────

void Pipeline::ensure_models_loaded() {
    if (!vad_) {
        LOG_STAGE("Loading Models");

        progress_.update("Loading VAD", 0.0f, "Silero VAD v5");
        LOG_INFO("Loading Silero VAD: %s",
                 config_.vad_config.threshold == 0.5f ? "models/silero_vad_v5.onnx" : "custom");

        // Default model path if not set
        std::string vad_path = "models/silero_vad_v5.onnx";
        vad_ = std::make_unique<VoiceActivityDetector>(vad_path, config_.vad_config);
        progress_.update("Loading VAD", 1.0f, "");
    }

    if (!embedder_ && !config_.skip_diarization) {
        progress_.update("Loading Embedder", 0.0f, "WeSpeaker ResNet34");
        embedder_ = std::make_unique<SpeakerEmbedder>(
            config_.embedder_model_path, config_.use_coreml);
        clusterer_ = std::make_unique<SpeakerClusterer>(config_.cluster_config);
        progress_.update("Loading Embedder", 1.0f, "");
    }

    if (!transcriber_) {
        progress_.update("Loading Whisper", 0.0f, config_.transcriber_config.model_path);
        transcriber_ = std::make_unique<Transcriber>(config_.transcriber_config);
        progress_.update("Loading Whisper", 1.0f, "");
    }

    if (!denoiser_ && config_.denoise) {
        progress_.update("Loading Denoiser", 0.0f, "RNNoise");
        denoiser_ = std::make_unique<Denoiser>();
        progress_.update("Loading Denoiser", 1.0f, "");
    }
}

// ─── Single file processing ────────────────────────────────────────────────

std::vector<TranscriptSegment> Pipeline::process_file(const std::string& path) {
    auto wall_start = std::chrono::steady_clock::now();

    LOG_STAGE("Processing: %s", path.c_str());

    // ── Step 0: Probe ──────────────────────────────────────────────────────
    progress_.update("Probing", 0.5f, path);
    AudioMeta meta = AudioDecoder::probe(path);
    progress_.update("Probing", 1.0f, "");

    LOG_INFO("Audio: %.1fs | %s | %dHz | %d ch",
             meta.duration_sec, meta.codec_name.c_str(),
             meta.sample_rate, meta.channels);

    // ── Step 1: Decode ─────────────────────────────────────────────────────
    progress_.update("Decoding", 0.0f, meta.format_name);
    std::vector<float> audio = AudioDecoder::decode_full(path);
    progress_.update("Decoding", 1.0f, "");

    // ── Step 2: Denoise (optional) ─────────────────────────────────────────
    if (config_.denoise && denoiser_) {
        progress_.update("Denoising", 0.0f, "RNNoise");
        denoiser_->process(audio);
        progress_.update("Denoising", 1.0f, "");
    }

    // ── Step 3: VAD ────────────────────────────────────────────────────────
    auto speech_segments = vad_->detect(audio);

    if (speech_segments.empty()) {
        LOG_WARN("VAD found no speech in %s — returning empty transcript", path.c_str());
        return {};
    }

    // ── Step 4: Speaker diarization ────────────────────────────────────────
    std::vector<DiarizedSegment> diarized;

    if (config_.skip_diarization) {
        // Single-speaker mode: convert speech segments directly
        LOG_INFO("Single-speaker mode: skipping diarization");
        for (const auto& seg : speech_segments) {
            diarized.push_back({seg.start_sec, seg.end_sec, 0, seg.avg_confidence});
        }
    } else {
        // Extract embeddings
        auto embeddings = embedder_->embed_segments(
            audio, speech_segments,
            [this](const std::string& stage, float p, const std::string& detail) {
                progress_.update("Embedding", p, detail);
            });

        // Cluster into speakers
        progress_.update("Clustering", 0.3f, "Spectral clustering");
        auto labels = clusterer_->cluster(embeddings);
        progress_.update("Clustering", 1.0f,
                         std::to_string(clusterer_->num_clusters()) + " speakers");

        // Assign labels back to segments
        diarized = assign_speakers(speech_segments, labels);
    }

    // ── Step 5: Transcribe ─────────────────────────────────────────────────
    progress_.update("Transcribing", 0.0f,
                     std::to_string(diarized.size()) + " segments");

    bool first_segment = true;
    auto raw_transcript = transcriber_->transcribe_all(
        audio, diarized,
        [this, &first_segment](const std::string& /*stage*/, float p, const std::string& detail) {
            progress_.update("Transcribing", p, detail);

            // Stream the text: extract speaker+text from detail for live preview
            if (!detail.empty()) {
                // Parse "Speaker N: text…" format
                auto colon = detail.find(": ");
                if (colon != std::string::npos && colon < 12) {
                    std::string prefix = detail.substr(0, colon);
                    std::string text   = detail.substr(colon + 2);
                    // Strip " (Recommended)" prefix from speaker label if present
                    int spk_id = 0;
                    std::sscanf(prefix.c_str(), "Speaker %d", &spk_id);
                    progress_.stream_text(spk_id, text);
                }
            }
            first_segment = false;
        });

    progress_.update("Transcribing", 1.0f, "");

    // ── Step 6: Post-processing ────────────────────────────────────────────
    progress_.update("Post-processing", 0.5f, "");
    auto final_transcript = PostProcessor::process(raw_transcript);
    progress_.update("Post-processing", 1.0f, "");

    // ── Export ─────────────────────────────────────────────────────────────
    // Count unique speakers
    int num_speakers = 0;
    for (const auto& seg : final_transcript) {
        num_speakers = std::max(num_speakers, seg.speaker_id + 1);
    }

    ExportMeta export_meta;
    export_meta.audio_file   = path;
    export_meta.model_name   = config_.transcriber_config.model_path;
    export_meta.duration_sec = meta.duration_sec;
    export_meta.num_speakers = num_speakers;

    // For stdout: the progress bar takes care of clearing, then we print
    progress_.finish();

    Exporter::write(final_transcript, config_.export_format,
                    config_.output_path, export_meta);

    // ── Timing summary ─────────────────────────────────────────────────────
    auto wall_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(wall_end - wall_start).count();
    double rtf = meta.duration_sec > 0 ? elapsed / meta.duration_sec : 0.0;

    LOG_INFO("Done: %.1fs audio in %.1fs wall-time (RTF=%.2f×)",
             meta.duration_sec, elapsed, rtf);

    return final_transcript;
}

// ─── Public run (single file) ──────────────────────────────────────────────

std::vector<TranscriptSegment> Pipeline::run() {
    ensure_models_loaded();
    return process_file(config_.audio_path);
}

// ─── Batch run (multiple files) ───────────────────────────────────────────

void Pipeline::run_batch(const std::vector<std::string>& audio_paths) {
    ensure_models_loaded();

    for (size_t i = 0; i < audio_paths.size(); i++) {
        LOG_INFO("Batch: file %zu/%zu — %s",
                 i + 1, audio_paths.size(), audio_paths[i].c_str());

        config_.audio_path = audio_paths[i];

        // For multi-file exports other than stdout, derive output path
        if (config_.export_format != ExportFormat::STDOUT &&
            !config_.output_path.empty()) {
            // Use provided output_path for first file; append index for the rest
            if (i > 0) {
                auto dot = config_.output_path.rfind('.');
                if (dot != std::string::npos) {
                    config_.output_path =
                        config_.output_path.substr(0, dot) + "_"
                        + std::to_string(i + 1)
                        + config_.output_path.substr(dot);
                }
            }
        }

        try {
            process_file(audio_paths[i]);
        } catch (const std::exception& e) {
            LOG_ERROR("Failed to process %s: %s", audio_paths[i].c_str(), e.what());
        }

        // Reset VAD state between files
        if (vad_) vad_->reset();
        if (denoiser_) denoiser_->reset();
    }
}

}  // namespace vaultasr
