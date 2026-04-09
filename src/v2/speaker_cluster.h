#pragma once

#include "types.h"
#include <vector>

namespace vaultasr {

// ─── Spectral clustering for speaker diarization ───────────────────────────
//
// Clusters speaker embeddings using spectral clustering with:
//   - Cosine similarity affinity matrix
//   - Normalized graph Laplacian
//   - Eigenvalue gap heuristic for automatic speaker count estimation
//   - K-means on spectral embedding
//
// Falls back to Agglomerative Hierarchical Clustering (AHC) if spectral
// clustering produces degenerate results.
//
class SpeakerClusterer {
public:
    struct Config {
        int   max_speakers           = 10;     // upper bound on speaker count
        float min_cluster_similarity = 0.3f;   // minimum affinity to consider
        bool  auto_num_speakers      = true;   // use eigenvalue gap heuristic
        int   fixed_num_speakers     = 0;      // override speaker count if > 0
        float ahc_threshold          = 0.55f;  // AHC fallback threshold
    };

    explicit SpeakerClusterer(Config config);
    SpeakerClusterer();  // default config overload

    // Cluster embeddings, return speaker labels (0-indexed)
    // Labels align 1:1 with input embeddings
    std::vector<int> cluster(const std::vector<std::vector<float>>& embeddings);

    // Get the number of clusters found in the last run
    int num_clusters() const { return num_clusters_; }

    const Config& config() const { return config_; }

private:
    Config config_;
    int    num_clusters_ = 0;

    // ─── Spectral clustering steps ────────────────────────────────────
    // Build cosine similarity matrix (N x N)
    std::vector<std::vector<float>> build_affinity(
        const std::vector<std::vector<float>>& embeddings);

    // Estimate number of speakers from eigenvalue gap
    int estimate_num_speakers(const std::vector<float>& eigenvalues);

    // Eigen decomposition of symmetric matrix
    // Returns eigenvalues (ascending) and eigenvectors (columns)
    void eigen_decompose(const std::vector<std::vector<float>>& matrix,
                         std::vector<float>& eigenvalues,
                         std::vector<std::vector<float>>& eigenvectors);

    // K-means clustering
    std::vector<int> kmeans(const std::vector<std::vector<float>>& data,
                            int k, int max_iters = 50, int n_restarts = 5);

    // ─── AHC fallback ─────────────────────────────────────────────────
    std::vector<int> ahc_cluster(const std::vector<std::vector<float>>& embeddings);

    // ─── Utilities ────────────────────────────────────────────────────
    static float cosine_similarity(const std::vector<float>& a,
                                    const std::vector<float>& b);
    static float cosine_distance(const std::vector<float>& a,
                                  const std::vector<float>& b);
};

// ─── Assign speaker labels to speech segments ──────────────────────────────
//
// Combines speech segments with cluster labels into diarized segments.
// Adjacent segments with the same speaker are merged if gap < max_gap.
//
std::vector<DiarizedSegment> assign_speakers(
    const std::vector<SpeechSegment>& segments,
    const std::vector<int>& labels,
    double max_gap_sec = 0.5);

}  // namespace vaultasr
