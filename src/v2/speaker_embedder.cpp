#include "speaker_embedder.h"
#include "logger.h"

#include <cmath>
#include <stdexcept>

#include <onnxruntime/onnxruntime_cxx_api.h>

#ifdef __APPLE__
#include <onnxruntime/coreml_provider_factory.h>
#endif

namespace vaultasr {

// ─── ONNX Runtime state ───────────────────────────────────────────────────

struct SpeakerEmbedder::OrtState {
    Ort::Env        env;
    Ort::Session    session;
    Ort::MemoryInfo memory_info;
    std::string     input_name;
    std::string     output_name;

    OrtState(const std::string& model_path, bool use_coreml)
        : env(ORT_LOGGING_LEVEL_WARNING, "VaultASR_Speaker")
        , session(nullptr)
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
    {
        Ort::SessionOptions opts;
        opts.SetIntraOpNumThreads(2);
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

#ifdef __APPLE__
        if (use_coreml) {
            try {
                uint32_t coreml_flags = 0;
                OrtSessionOptionsAppendExecutionProvider_CoreML(opts, coreml_flags);
                LOG_DEBUG("Speaker embedder: CoreML execution provider enabled");
            } catch (...) {
                LOG_WARN("Speaker embedder: CoreML not available, using CPU");
            }
        }
#else
        (void)use_coreml;
#endif

        session = Ort::Session(env, model_path.c_str(), opts);

        // Discover input/output names
        Ort::AllocatorWithDefaultOptions alloc;
        auto in_name  = session.GetInputNameAllocated(0, alloc);
        auto out_name = session.GetOutputNameAllocated(0, alloc);
        input_name  = in_name.get();
        output_name = out_name.get();

        LOG_DEBUG("Speaker model: input='%s', output='%s'",
                  input_name.c_str(), output_name.c_str());
    }
};

// ─── SpeakerEmbedder implementation ───────────────────────────────────────

SpeakerEmbedder::SpeakerEmbedder(const std::string& model_path, bool use_coreml)
    : ort_(std::make_unique<OrtState>(model_path, use_coreml))
    , fbank_()
    , embedding_dim_(256)
{
    LOG_INFO("Speaker embedder initialized: %s (dim=%d)",
             model_path.c_str(), embedding_dim_);
}

SpeakerEmbedder::~SpeakerEmbedder() = default;

void SpeakerEmbedder::l2_normalize(std::vector<float>& v) {
    float norm = 0.0f;
    for (float x : v) norm += x * x;
    norm = std::sqrt(norm);
    if (norm > 1e-10f) {
        for (float& x : v) x /= norm;
    }
}

std::vector<float> SpeakerEmbedder::embed(const float* audio_16k, size_t num_samples) {
    // Compute mel filterbank features
    std::vector<float> audio(audio_16k, audio_16k + num_samples);
    auto fbank_features = fbank_.compute(audio);

    if (fbank_features.empty()) {
        LOG_WARN("Empty fbank features for %zu samples, returning zero embedding", num_samples);
        return std::vector<float>(embedding_dim_, 0.0f);
    }

    int num_frames = static_cast<int>(fbank_features.size());
    int num_mels   = static_cast<int>(fbank_features[0].size());

    LOG_TRACE("Embedding: %d frames x %d mels from %zu samples",
              num_frames, num_mels, num_samples);

    // Flatten to [1, num_frames, num_mels]
    std::vector<float> flat(num_frames * num_mels);
    for (int i = 0; i < num_frames; i++) {
        for (int j = 0; j < num_mels; j++) {
            flat[i * num_mels + j] = fbank_features[i][j];
        }
    }

    std::vector<int64_t> input_shape = {1, num_frames, num_mels};

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        ort_->memory_info, flat.data(), flat.size(),
        input_shape.data(), input_shape.size());

    const char* input_names[]  = {ort_->input_name.c_str()};
    const char* output_names[] = {ort_->output_name.c_str()};

    auto outputs = ort_->session.Run(
        Ort::RunOptions{nullptr},
        input_names, &input_tensor, 1,
        output_names, 1);

    // Extract embedding
    float* out_data = outputs[0].GetTensorMutableData<float>();
    auto type_info  = outputs[0].GetTensorTypeAndShapeInfo();
    auto out_shape  = type_info.GetShape();

    int emb_dim = static_cast<int>(out_shape.back());
    embedding_dim_ = emb_dim;

    std::vector<float> embedding(out_data, out_data + emb_dim);
    l2_normalize(embedding);

    return embedding;
}

std::vector<std::vector<float>> SpeakerEmbedder::embed_segments(
    const std::vector<float>& full_audio,
    const std::vector<SpeechSegment>& segments,
    ProgressCallback progress_cb) {

    LOG_STAGE("Speaker Embedding Extraction");
    LOG_INFO("Extracting embeddings for %zu segments", segments.size());

    std::vector<std::vector<float>> embeddings;
    embeddings.reserve(segments.size());

    for (size_t i = 0; i < segments.size(); i++) {
        const auto& seg = segments[i];

        int start_idx = static_cast<int>(seg.start_sec * 16000);
        int end_idx   = static_cast<int>(seg.end_sec * 16000);
        start_idx = std::max(0, start_idx);
        end_idx   = std::min(static_cast<int>(full_audio.size()), end_idx);

        if (end_idx - start_idx < 1600) {  // minimum ~100ms
            LOG_TRACE("Segment %zu too short (%d samples), zero embedding", i, end_idx - start_idx);
            embeddings.push_back(std::vector<float>(embedding_dim_, 0.0f));
            continue;
        }

        auto emb = embed(full_audio.data() + start_idx, end_idx - start_idx);
        embeddings.push_back(std::move(emb));

        LOG_TRACE("Segment %zu/%zu: %.2f–%.2fs → embedding extracted",
                  i + 1, segments.size(), seg.start_sec, seg.end_sec);

        if (progress_cb) {
            progress_cb("Embedding",
                        static_cast<float>(i + 1) / segments.size(),
                        "Segment " + std::to_string(i + 1) + "/" + std::to_string(segments.size()));
        }
    }

    LOG_INFO("Extracted %zu speaker embeddings (dim=%d)", embeddings.size(), embedding_dim_);
    return embeddings;
}

}  // namespace vaultasr
