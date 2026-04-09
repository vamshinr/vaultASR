#include "vad.h"
#include "logger.h"

#include <algorithm>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <onnxruntime/onnxruntime_cxx_api.h>

#ifdef __APPLE__
#include <onnxruntime/coreml_provider_factory.h>
#endif

namespace vaultasr {

// ─── ONNX Runtime state (PIMPL) ───────────────────────────────────────────

struct VoiceActivityDetector::OrtState {
    Ort::Env             env;
    Ort::Session         session;
    Ort::MemoryInfo      memory_info;

    // Silero v5 state tensor: shape [2, 1, 128]
    std::vector<float>   state;
    int64_t              sr;

    OrtState(const std::string& model_path)
        : env(ORT_LOGGING_LEVEL_WARNING, "VaultASR_VAD")
        , session(nullptr)
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
        , state(2 * 1 * 128, 0.0f)
        , sr(16000)
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(1);
        opts.SetInterOpNumThreads(1);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
        // Use CoreML for VAD acceleration on Apple Silicon
        try {
            uint32_t coreml_flags = 0;
            OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags);
            LOG_DEBUG("VAD: CoreML execution provider enabled");
        } catch (...) {
            LOG_WARN("VAD: CoreML not available, falling back to CPU");
        }
#endif

        session = Ort::Session(env, model_path.c_str(), opts);

        // Log model input/output info
        Ort::AllocatorWithDefaultOptions alloc;
        size_t n_inputs = session.GetInputCount();
        LOG_DEBUG("VAD model loaded: %zu inputs", n_inputs);
        for (size_t i = 0; i < n_inputs; i++) {
            auto name = session.GetInputNameAllocated(i, alloc);
            LOG_TRACE("  input[%zu]: %s", i, name.get());
        }
    }
};

// ─── Constructor / Destructor ──────────────────────────────────────────────

VoiceActivityDetector::VoiceActivityDetector(const std::string& model_path,
                                              Config config)
    : config_(std::move(config))
    , ort_(std::make_unique<OrtState>(model_path))
{
    LOG_INFO("VAD initialized: threshold=%.2f, neg_threshold=%.2f, "
             "min_speech=%.2fs, min_silence=%.2fs",
             config_.threshold, config_.neg_threshold,
             config_.min_speech_sec, config_.min_silence_sec);
}

VoiceActivityDetector::VoiceActivityDetector(const std::string& model_path)
    : VoiceActivityDetector(model_path, Config{})
{}

VoiceActivityDetector::~VoiceActivityDetector() = default;

void VoiceActivityDetector::reset() {
    std::fill(ort_->state.begin(), ort_->state.end(), 0.0f);
}

// ─── Single frame inference ────────────────────────────────────────────────

float VoiceActivityDetector::infer_frame(const float* data) {
    const int window_size = 512;  // Silero v5 window @ 16kHz

    // Input tensors
    std::vector<int64_t> input_dims = {1, window_size};
    std::vector<int64_t> state_dims = {2, 1, 128};
    std::vector<int64_t> sr_dims    = {1};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info,
        const_cast<float*>(data), window_size,
        input_dims.data(), input_dims.size());

    Ort::Value state_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info,
        ort_->state.data(), ort_->state.size(),
        state_dims.data(), state_dims.size());

    Ort::Value sr_tensor = Ort::Value::CreateTensor<int64_t>(
        ort_->memory_info,
        &ort_->sr, 1,
        sr_dims.data(), sr_dims.size());

    // Silero v5 I/O names
    const char* input_names[]  = {"input", "state", "sr"};
    const char* output_names[] = {"output", "stateN"};

    std::vector<Ort::Value> inputs;
    inputs.push_back(std::move(input_tensor));
    inputs.push_back(std::move(state_tensor));
    inputs.push_back(std::move(sr_tensor));

    auto outputs = ort_->session.Run(
        Ort::RunOptions{nullptr},
        input_names, inputs.data(), 3,
        output_names, 2);

    // Extract speech probability
    float prob = outputs[0].GetTensorMutableData<float>()[0];

    // Update state for next frame
    float* new_state = outputs[1].GetTensorMutableData<float>();
    std::copy(new_state, new_state + ort_->state.size(), ort_->state.begin());

    return prob;
}

// ─── Main detection ────────────────────────────────────────────────────────

std::vector<SpeechSegment> VoiceActivityDetector::detect(
    const std::vector<float>& audio_16k) {

    LOG_STAGE("Voice Activity Detection");
    LOG_INFO("Processing %.2f seconds of audio (%zu samples)",
             audio_16k.size() / 16000.0, audio_16k.size());

    reset();

    const int window_size = 512;
    const int sample_rate = 16000;
    double total_duration = audio_16k.size() / static_cast<double>(sample_rate);

    // Collect per-window probabilities
    std::vector<std::pair<double, float>> raw_probs;
    size_t total_windows = (audio_16k.size() - window_size) / window_size + 1;
    size_t window_idx = 0;

    for (size_t i = 0; i + window_size <= audio_16k.size(); i += window_size) {
        float prob = infer_frame(audio_16k.data() + i);
        double time = static_cast<double>(i) / sample_rate;
        raw_probs.push_back({time, prob});

        window_idx++;
        if (window_idx % 500 == 0) {
            LOG_TRACE("VAD frame %zu/%zu (%.1fs): prob=%.3f",
                      window_idx, total_windows, time, prob);
        }
    }

    LOG_DEBUG("VAD: processed %zu windows over %.2fs",
              raw_probs.size(), total_duration);

    // Post-process into clean segments
    auto segments = post_process(raw_probs, total_duration);

    // Log results
    double speech_dur = 0;
    for (const auto& seg : segments) speech_dur += seg.duration();
    LOG_INFO("VAD result: %zu speech segments, %.1fs speech / %.1fs total (%.0f%%)",
             segments.size(), speech_dur, total_duration,
             total_duration > 0 ? (speech_dur / total_duration * 100.0) : 0.0);

    for (size_t i = 0; i < segments.size(); i++) {
        LOG_DEBUG("  Segment %zu: %.2f–%.2fs (%.2fs) conf=%.3f",
                  i, segments[i].start_sec, segments[i].end_sec,
                  segments[i].duration(), segments[i].avg_confidence);
    }

    return segments;
}

// ─── Post-processing: raw probs → clean segments ──────────────────────────

std::vector<SpeechSegment> VoiceActivityDetector::post_process(
    const std::vector<std::pair<double, float>>& raw_probs,
    double total_duration) {

    if (raw_probs.empty()) return {};

    const double window_dur = 512.0 / 16000.0;  // ~32ms per window

    // Step 1: Hysteresis-based trigger/release
    struct RawSegment {
        double start;
        double end;
        std::vector<float> probs;
    };
    std::vector<RawSegment> raw_segments;

    bool triggered = false;
    RawSegment current;

    for (const auto& [time, prob] : raw_probs) {
        if (!triggered && prob >= config_.threshold) {
            triggered = true;
            current = {};
            current.start = time;
            current.probs.push_back(prob);
        } else if (triggered && prob < config_.neg_threshold) {
            triggered = false;
            current.end = time + window_dur;
            raw_segments.push_back(current);
        } else if (triggered) {
            current.probs.push_back(prob);
        }
    }

    // Handle segment that extends to end of audio
    if (triggered) {
        current.end = total_duration;
        raw_segments.push_back(current);
    }

    LOG_TRACE("Raw trigger/release produced %zu segments", raw_segments.size());

    // Step 2: Merge segments separated by less than min_silence
    std::vector<RawSegment> merged;
    for (auto& seg : raw_segments) {
        if (!merged.empty() &&
            seg.start - merged.back().end < config_.min_silence_sec) {
            // Merge: extend previous segment
            merged.back().end = seg.end;
            merged.back().probs.insert(
                merged.back().probs.end(), seg.probs.begin(), seg.probs.end());
        } else {
            merged.push_back(std::move(seg));
        }
    }

    LOG_TRACE("After merging short silences: %zu segments", merged.size());

    // Step 3: Filter out segments shorter than min_speech
    std::vector<SpeechSegment> result;
    for (auto& seg : merged) {
        double dur = seg.end - seg.start;
        if (dur < config_.min_speech_sec) {
            LOG_TRACE("Discarding short segment: %.3f–%.3fs (%.0fms)",
                      seg.start, seg.end, dur * 1000);
            continue;
        }

        // Apply padding
        double padded_start = std::max(0.0, seg.start - config_.speech_pad_sec);
        double padded_end   = std::min(total_duration, seg.end + config_.speech_pad_sec);

        // Calculate average confidence
        float avg_conf = 0.0f;
        if (!seg.probs.empty()) {
            avg_conf = std::accumulate(seg.probs.begin(), seg.probs.end(), 0.0f)
                       / seg.probs.size();
        }

        // Step 4: Split segments longer than max_segment_sec
        if (padded_end - padded_start > config_.max_segment_sec) {
            double t = padded_start;
            while (t < padded_end) {
                double chunk_end = std::min(t + config_.max_segment_sec, padded_end);
                result.push_back({t, chunk_end, avg_conf});
                t = chunk_end;
            }
        } else {
            result.push_back({padded_start, padded_end, avg_conf});
        }
    }

    return result;
}

}  // namespace vaultasr
