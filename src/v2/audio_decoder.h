#pragma once

#include "types.h"
#include <functional>
#include <string>
#include <vector>

namespace vaultasr {

class AudioDecoder {
public:
    // Probe file metadata without decoding
    static AudioMeta probe(const std::string& path);

    // Decode full file to 16kHz mono float32 PCM
    static std::vector<float> decode_full(const std::string& path);

    // Streaming decode: callback receives chunks of float32 PCM @ 16kHz mono
    // chunk_samples: how many samples per callback (default 30s worth)
    // Returns total number of samples decoded
    using ChunkCallback = std::function<void(const float* data, size_t num_samples)>;
    static size_t decode_streaming(const std::string& path,
                                    ChunkCallback cb,
                                    size_t chunk_samples = 16000 * 30);
};

}  // namespace vaultasr
