#include "transcriber.h"
#include "logger.h"

#include "../../external/whisper.cpp/include/whisper.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace vaultasr {

// ─── Constructor / Destructor ──────────────────────────────────────────────

Transcriber::Transcriber(const Config& config) : config_(config), ctx_(nullptr) {
    LOG_STAGE("Loading Whisper Model");
    LOG_INFO("Model: %s", config_.model_path.c_str());
    LOG_INFO("Language: %s, GPU: %s, Threads: %d",
             config_.language.c_str(),
             config_.use_gpu ? "yes" : "no",
             config_.n_threads);

    struct whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = config_.use_gpu;

    ctx_ = whisper_init_from_file_with_params(config_.model_path.c_str(), cparams);
    if (!ctx_) {
        throw std::runtime_error("Failed to load Whisper model: " + config_.model_path);
    }

    LOG_INFO("Whisper model loaded successfully (GPU=%s)",
             config_.use_gpu ? "Metal" : "CPU");
}

Transcriber::~Transcriber() {
    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }
}

// ─── Transcribe single segment ─────────────────────────────────────────────

TranscriptSegment Transcriber::transcribe_segment(
    const float* audio_16k,
    size_t num_samples,
    int speaker_id,
    double offset_sec) {

    TranscriptSegment result;
    result.speaker_id  = speaker_id;
    result.start_sec   = offset_sec;
    result.end_sec     = offset_sec + num_samples / 16000.0;

    if (num_samples < 1600) {  // less than 100ms
        LOG_TRACE("Segment too short (%zu samples), skipping", num_samples);
        return result;
    }

    // Configure whisper parameters
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_progress   = false;
    wparams.print_special    = false;
    wparams.print_realtime   = false;
    wparams.print_timestamps = false;
    wparams.language         = config_.language.c_str();
    wparams.n_threads        = config_.n_threads;
    wparams.translate        = config_.translate;
    wparams.no_speech_thold  = config_.no_speech_threshold;
    wparams.token_timestamps = config_.word_timestamps;

    // Greedy params
    wparams.greedy.best_of = 1;

    LOG_TRACE("Transcribing: speaker=%d, offset=%.2fs, samples=%zu (%.2fs)",
              speaker_id, offset_sec, num_samples, num_samples / 16000.0);

    int ret = whisper_full(ctx_, wparams, audio_16k, static_cast<int>(num_samples));
    if (ret != 0) {
        LOG_ERROR("Whisper inference failed for segment at %.2fs", offset_sec);
        return result;
    }

    // Extract segments
    int n_segments = whisper_full_n_segments(ctx_);
    std::string full_text;
    float total_prob = 0.0f;
    int prob_count = 0;

    for (int s = 0; s < n_segments; s++) {
        const char* seg_text = whisper_full_get_segment_text(ctx_, s);
        if (!seg_text) continue;

        std::string text(seg_text);

        // Get segment-level timestamps (relative to this chunk)
        int64_t t0 = whisper_full_get_segment_t0(ctx_, s);  // in 10ms units
        int64_t t1 = whisper_full_get_segment_t1(ctx_, s);

        double seg_start = offset_sec + t0 * 0.01;
        double seg_end   = offset_sec + t1 * 0.01;

        LOG_TRACE("  Whisper seg %d: [%.2f–%.2f] \"%s\"",
                  s, seg_start, seg_end, text.c_str());

        // Extract word-level tokens if available
        if (config_.word_timestamps) {
            int n_tokens = whisper_full_n_tokens(ctx_, s);
            for (int t = 0; t < n_tokens; t++) {
                whisper_token_data tdata = whisper_full_get_token_data(ctx_, s, t);
                const char* token_text = whisper_full_get_token_text(ctx_, s, t);

                if (!token_text || token_text[0] == '\0') continue;

                // Skip special tokens (IDs < whisper_token_eot)
                if (tdata.id >= whisper_token_eot(ctx_)) continue;

                WordInfo word;
                word.text = token_text;
                word.start_sec = offset_sec + tdata.t0 * 0.01;
                word.end_sec   = offset_sec + tdata.t1 * 0.01;
                word.probability = tdata.p;

                result.words.push_back(word);

                total_prob += tdata.p;
                prob_count++;

                LOG_TRACE("    Token: \"%s\" [%.2f–%.2f] p=%.3f",
                          word.text.c_str(), word.start_sec, word.end_sec, word.probability);
            }
        }

        full_text += text;
    }

    // Clean up leading whitespace
    if (!full_text.empty() && full_text[0] == ' ') {
        full_text = full_text.substr(1);
    }

    result.text = full_text;
    result.avg_confidence = prob_count > 0 ? total_prob / prob_count : 0.0f;

    LOG_DEBUG("Transcribed [%.2f–%.2f] Speaker %d: \"%s\" (conf=%.3f)",
              result.start_sec, result.end_sec, speaker_id,
              result.text.substr(0, 80).c_str(), result.avg_confidence);

    return result;
}

// ─── Transcribe all segments ───────────────────────────────────────────────

std::vector<TranscriptSegment> Transcriber::transcribe_all(
    const std::vector<float>& full_audio,
    const std::vector<DiarizedSegment>& segments,
    ProgressCallback progress_cb) {

    LOG_STAGE("Speech Recognition (ASR)");
    LOG_INFO("Transcribing %zu segments", segments.size());

    std::vector<TranscriptSegment> results;
    results.reserve(segments.size());

    for (size_t i = 0; i < segments.size(); i++) {
        const auto& seg = segments[i];

        // Extract audio slice
        int start_idx = static_cast<int>(seg.start_sec * 16000);
        int end_idx   = static_cast<int>(seg.end_sec * 16000);
        start_idx = std::max(0, start_idx);
        end_idx   = std::min(static_cast<int>(full_audio.size()), end_idx);

        int num_samples = end_idx - start_idx;
        if (num_samples < 1600) {
            LOG_TRACE("Skipping segment %zu: too short (%d samples)", i, num_samples);
            continue;
        }

        auto result = transcribe_segment(
            full_audio.data() + start_idx,
            static_cast<size_t>(num_samples),
            seg.speaker_id,
            seg.start_sec);

        // Skip empty transcriptions
        if (result.text.empty()) {
            LOG_TRACE("Segment %zu produced empty text, skipping", i);
            continue;
        }

        results.push_back(std::move(result));

        // Progress callback with streamed text
        if (progress_cb) {
            const auto& last = results.back();
            std::string detail = "Speaker " + std::to_string(last.speaker_id)
                               + ": " + last.text.substr(0, 60);
            progress_cb("Transcribing",
                        static_cast<float>(i + 1) / segments.size(),
                        detail);
        }
    }

    // Summary stats
    double total_speech = 0;
    size_t total_words = 0;
    for (const auto& r : results) {
        total_speech += r.end_sec - r.start_sec;
        total_words += r.words.size();
    }

    LOG_INFO("Transcription complete: %zu segments, %.1fs speech, %zu words",
             results.size(), total_speech, total_words);

    return results;
}

}  // namespace vaultasr
