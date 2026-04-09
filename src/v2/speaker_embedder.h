#pragma once

#include "kaldi_fbank.h"
#include "types.h"
#include <memory>
#include <string>
#include <vector>

namespace vaultasr {

// ─── WeSpeaker ResNet34 speaker embedding extractor ────────────────────────
//
// Wraps the WeSpeaker/Pyannote ONNX model to extract 256-dimensional
// speaker embeddings from raw audio. Handles Fbank computation internally.
//
// Usage:
//   SpeakerEmbedder embedder("models/wespeaker_resnet34.onnx");
//   auto emb = embedder.embed(audio_ptr, num_samples);  // → 256-dim vector
//
class SpeakerEmbedder {
public:
    SpeakerEmbedder(const std::string& model_path, bool use_coreml = true);
    ~SpeakerEmbedder();

    // Non-copyable
    SpeakerEmbedder(const SpeakerEmbedder&) = delete;
    SpeakerEmbedder& operator=(const SpeakerEmbedder&) = delete;

    // Extract speaker embedding from raw 16kHz mono audio
    std::vector<float> embed(const float* audio_16k, size_t num_samples);

    // Extract embeddings for all speech segments
    // Returns one embedding per segment, in order
    std::vector<std::vector<float>> embed_segments(
        const std::vector<float>& full_audio,
        const std::vector<SpeechSegment>& segments,
        ProgressCallback progress_cb = nullptr);

    int embedding_dim() const { return embedding_dim_; }

private:
    struct OrtState;
    std::unique_ptr<OrtState> ort_;
    KaldiFbank fbank_;
    int embedding_dim_;

    // L2-normalize a vector in-place
    static void l2_normalize(std::vector<float>& v);
};

}  // namespace vaultasr
