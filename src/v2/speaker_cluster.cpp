#include "speaker_cluster.h"
#include "logger.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

// Use Accelerate (LAPACK) on macOS for eigendecomposition
#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#else
// Fallback: simple power iteration for non-Apple platforms
#endif

namespace vaultasr {

// ─── Constructor ───────────────────────────────────────────────────────────

SpeakerClusterer::SpeakerClusterer(Config config) : config_(std::move(config)) {}

SpeakerClusterer::SpeakerClusterer() : SpeakerClusterer(Config{}) {}

// ─── Utility functions ─────────────────────────────────────────────────────

float SpeakerClusterer::cosine_similarity(const std::vector<float>& a,
                                           const std::vector<float>& b) {
    float dot = 0.0f, na = 0.0f, nb = 0.0f;
    for (size_t i = 0; i < a.size(); i++) {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    float denom = std::sqrt(na) * std::sqrt(nb);
    return denom > 1e-10f ? dot / denom : 0.0f;
}

float SpeakerClusterer::cosine_distance(const std::vector<float>& a,
                                          const std::vector<float>& b) {
    return 1.0f - cosine_similarity(a, b);
}

// ─── Affinity matrix ───────────────────────────────────────────────────────

std::vector<std::vector<float>> SpeakerClusterer::build_affinity(
    const std::vector<std::vector<float>>& embeddings) {
    int n = static_cast<int>(embeddings.size());
    std::vector<std::vector<float>> S(n, std::vector<float>(n, 0.0f));

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float sim = cosine_similarity(embeddings[i], embeddings[j]);
            // Threshold: set low similarities to 0
            if (sim < config_.min_cluster_similarity) sim = 0.0f;
            S[i][j] = sim;
            S[j][i] = sim;
        }
        S[i][i] = 0.0f;  // no self-loops
    }

    return S;
}

// ─── Eigendecomposition ────────────────────────────────────────────────────

void SpeakerClusterer::eigen_decompose(
    const std::vector<std::vector<float>>& matrix,
    std::vector<float>& eigenvalues,
    std::vector<std::vector<float>>& eigenvectors) {

    int n = static_cast<int>(matrix.size());
    eigenvalues.resize(n);
    eigenvectors.resize(n, std::vector<float>(n, 0.0f));

    if (n == 0) return;

#ifdef __APPLE__
    // Use Apple Accelerate LAPACK: ssyev for symmetric eigendecomposition
    // LAPACK stores matrices in column-major order

    // Flatten matrix to column-major
    std::vector<float> A(n * n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[j * n + i] = matrix[i][j];  // column-major
        }
    }

    char jobz = 'V';   // compute eigenvectors
    char uplo = 'U';   // upper triangle
    __CLPK_integer N = n;
    __CLPK_integer lda = n;
    __CLPK_integer info = 0;

    // Query optimal workspace
    float work_query;
    __CLPK_integer lwork = -1;
    ssyev_(&jobz, &uplo, &N, A.data(), &lda, eigenvalues.data(),
           &work_query, &lwork, &info);

    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(lwork);

    // Compute
    ssyev_(&jobz, &uplo, &N, A.data(), &lda, eigenvalues.data(),
           work.data(), &lwork, &info);

    if (info != 0) {
        LOG_ERROR("LAPACK ssyev failed with info=%d", (int)info);
        return;
    }

    // Extract eigenvectors (columns of A in column-major → rows of our result)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            eigenvectors[j][i] = A[i * n + j];  // eigenvector i, component j
        }
    }

#else
    // Fallback: Jacobi eigenvalue algorithm for small symmetric matrices
    // (adequate for typical segment counts < 1000)

    // Copy matrix
    std::vector<std::vector<float>> A = matrix;
    std::vector<std::vector<float>> V(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) V[i][i] = 1.0f;  // identity

    const int max_sweeps = 100;
    const float tol = 1e-8f;

    for (int sweep = 0; sweep < max_sweeps; sweep++) {
        // Find largest off-diagonal element
        float max_off = 0.0f;
        int p = 0, q = 1;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (std::fabs(A[i][j]) > max_off) {
                    max_off = std::fabs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max_off < tol) break;

        // Compute rotation
        float theta = 0.5f * std::atan2(2.0f * A[p][q], A[q][q] - A[p][p]);
        float c = std::cos(theta);
        float s = std::sin(theta);

        // Apply Givens rotation
        for (int i = 0; i < n; i++) {
            float api = A[p][i];
            float aqi = A[q][i];
            A[p][i] = c * api - s * aqi;
            A[q][i] = s * api + c * aqi;
        }
        for (int i = 0; i < n; i++) {
            float aip = A[i][p];
            float aiq = A[i][q];
            A[i][p] = c * aip - s * aiq;
            A[i][q] = s * aip + c * aiq;
        }

        // Update eigenvector matrix
        for (int i = 0; i < n; i++) {
            float vip = V[i][p];
            float viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues from diagonal
    std::vector<std::pair<float, int>> eig_pairs;
    for (int i = 0; i < n; i++) {
        eig_pairs.push_back({A[i][i], i});
    }

    // Sort by eigenvalue ascending
    std::sort(eig_pairs.begin(), eig_pairs.end());

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = eig_pairs[i].first;
        int orig_idx = eig_pairs[i].second;
        for (int j = 0; j < n; j++) {
            eigenvectors[i][j] = V[j][orig_idx];
        }
    }
#endif

    LOG_TRACE("Eigendecomposition: %d values, range [%.4f, %.4f]",
              n, eigenvalues.front(), eigenvalues.back());
}

// ─── Estimate number of speakers ───────────────────────────────────────────

int SpeakerClusterer::estimate_num_speakers(const std::vector<float>& eigenvalues) {
    if (eigenvalues.size() <= 1) return 1;

    int max_k = std::min(config_.max_speakers, static_cast<int>(eigenvalues.size()));

    // Find the largest gap between consecutive eigenvalues
    // For the normalized Laplacian, eigenvalues near 0 correspond to clusters.
    // The number of near-zero eigenvalues = number of clusters.
    float max_gap = 0.0f;
    int best_k = 1;

    for (int i = 0; i < max_k - 1; i++) {
        float gap = eigenvalues[i + 1] - eigenvalues[i];
        LOG_TRACE("Eigenvalue gap[%d→%d]: %.6f (ev=%.6f → %.6f)",
                  i, i + 1, gap, eigenvalues[i], eigenvalues[i + 1]);
        if (gap > max_gap) {
            max_gap = gap;
            best_k = i + 1;
        }
    }

    best_k = std::max(1, std::min(best_k, max_k));

    LOG_DEBUG("Estimated %d speakers (max gap=%.4f at position %d)",
              best_k, max_gap, best_k - 1);

    return best_k;
}

// ─── K-means ───────────────────────────────────────────────────────────────

std::vector<int> SpeakerClusterer::kmeans(
    const std::vector<std::vector<float>>& data,
    int k, int max_iters, int n_restarts) {

    int n = static_cast<int>(data.size());
    if (n == 0 || k <= 0) return {};
    if (k >= n) {
        // Each point is its own cluster
        std::vector<int> labels(n);
        std::iota(labels.begin(), labels.end(), 0);
        return labels;
    }

    int dim = static_cast<int>(data[0].size());
    std::vector<int> best_labels(n, 0);
    float best_cost = std::numeric_limits<float>::max();

    std::mt19937 rng(42);  // deterministic seed

    for (int restart = 0; restart < n_restarts; restart++) {
        // K-means++ initialization
        std::vector<std::vector<float>> centers;
        std::vector<int> labels(n, 0);

        // Pick first center randomly
        std::uniform_int_distribution<int> dist(0, n - 1);
        centers.push_back(data[dist(rng)]);

        // Pick remaining centers with probability proportional to distance²
        for (int c = 1; c < k; c++) {
            std::vector<float> min_dists(n, std::numeric_limits<float>::max());
            for (int i = 0; i < n; i++) {
                for (const auto& center : centers) {
                    float d = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        float diff = data[i][j] - center[j];
                        d += diff * diff;
                    }
                    min_dists[i] = std::min(min_dists[i], d);
                }
            }

            // Weighted random selection
            std::discrete_distribution<int> weighted(min_dists.begin(), min_dists.end());
            centers.push_back(data[weighted(rng)]);
        }

        // Iterate
        for (int iter = 0; iter < max_iters; iter++) {
            // Assign
            bool changed = false;
            for (int i = 0; i < n; i++) {
                float best_d = std::numeric_limits<float>::max();
                int best_c = 0;
                for (int c = 0; c < k; c++) {
                    float d = 0.0f;
                    for (int j = 0; j < dim; j++) {
                        float diff = data[i][j] - centers[c][j];
                        d += diff * diff;
                    }
                    if (d < best_d) {
                        best_d = d;
                        best_c = c;
                    }
                }
                if (labels[i] != best_c) {
                    labels[i] = best_c;
                    changed = true;
                }
            }

            if (!changed) break;

            // Update centers
            std::vector<std::vector<float>> new_centers(k, std::vector<float>(dim, 0.0f));
            std::vector<int> counts(k, 0);

            for (int i = 0; i < n; i++) {
                counts[labels[i]]++;
                for (int j = 0; j < dim; j++) {
                    new_centers[labels[i]][j] += data[i][j];
                }
            }

            for (int c = 0; c < k; c++) {
                if (counts[c] > 0) {
                    for (int j = 0; j < dim; j++) {
                        new_centers[c][j] /= counts[c];
                    }
                }
            }
            centers = std::move(new_centers);
        }

        // Compute cost (sum of squared distances)
        float cost = 0.0f;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < dim; j++) {
                float diff = data[i][j] - centers[labels[i]][j];
                cost += diff * diff;
            }
        }

        if (cost < best_cost) {
            best_cost = cost;
            best_labels = labels;
        }
    }

    return best_labels;
}

// ─── AHC fallback ──────────────────────────────────────────────────────────

std::vector<int> SpeakerClusterer::ahc_cluster(
    const std::vector<std::vector<float>>& embeddings) {

    LOG_DEBUG("Running AHC fallback (threshold=%.2f)", config_.ahc_threshold);

    int n = static_cast<int>(embeddings.size());
    if (n == 0) return {};

    std::vector<int> labels(n);
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; i++) {
        clusters[i].push_back(i);
        labels[i] = i;
    }

    // Pairwise distance matrix
    std::vector<std::vector<float>> dist(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float d = cosine_distance(embeddings[i], embeddings[j]);
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }

    std::vector<bool> active(n, true);

    while (true) {
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;

        for (int i = 0; i < n; i++) {
            if (!active[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!active[j]) continue;
                if (dist[i][j] < min_dist) {
                    min_dist = dist[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }

        if (best_i < 0 || min_dist > config_.ahc_threshold) break;

        // Merge j into i
        active[best_j] = false;
        for (int idx : clusters[best_j]) {
            clusters[best_i].push_back(idx);
            labels[idx] = best_i;
        }
        clusters[best_j].clear();

        // Update distances (average linkage)
        for (int k = 0; k < n; k++) {
            if (!active[k] || k == best_i) continue;
            float total = 0.0f;
            int count = 0;
            for (int p : clusters[best_i]) {
                for (int q : clusters[k]) {
                    total += cosine_distance(embeddings[p], embeddings[q]);
                    count++;
                }
            }
            float avg = count > 0 ? total / count : 1.0f;
            dist[best_i][k] = avg;
            dist[k][best_i] = avg;
        }
    }

    // Renumber labels sequentially
    std::vector<int> final_labels(n, -1);
    int new_label = 0;
    for (int i = 0; i < n; i++) {
        if (active[i] && !clusters[i].empty()) {
            for (int idx : clusters[i]) {
                final_labels[idx] = new_label;
            }
            new_label++;
        }
    }

    return final_labels;
}

// ─── Main clustering entry point ───────────────────────────────────────────

std::vector<int> SpeakerClusterer::cluster(
    const std::vector<std::vector<float>>& embeddings) {

    LOG_STAGE("Speaker Clustering");

    int n = static_cast<int>(embeddings.size());
    if (n == 0) return {};
    if (n == 1) {
        num_clusters_ = 1;
        return {0};
    }

    // Handle fixed speaker count
    int target_k = 0;
    if (!config_.auto_num_speakers && config_.fixed_num_speakers > 0) {
        target_k = config_.fixed_num_speakers;
        LOG_INFO("Fixed speaker count: %d", target_k);
    }

    // Step 1: Build affinity matrix
    LOG_DEBUG("Building affinity matrix (%d x %d)", n, n);
    auto S = build_affinity(embeddings);

    // Step 2: Compute degree matrix and normalized Laplacian
    // L_sym = I - D^(-1/2) * S * D^(-1/2)
    LOG_DEBUG("Computing normalized Laplacian");

    std::vector<float> D(n, 0.0f);  // degree
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            D[i] += S[i][j];
        }
    }

    // D^(-1/2)
    std::vector<float> D_inv_sqrt(n, 0.0f);
    for (int i = 0; i < n; i++) {
        D_inv_sqrt[i] = D[i] > 1e-10f ? 1.0f / std::sqrt(D[i]) : 0.0f;
    }

    // L_sym = I - D^(-1/2) * S * D^(-1/2)
    std::vector<std::vector<float>> L(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float val = D_inv_sqrt[i] * S[i][j] * D_inv_sqrt[j];
            L[i][j] = (i == j ? 1.0f : 0.0f) - val;
        }
    }

    // Step 3: Eigendecomposition
    LOG_DEBUG("Computing eigendecomposition");
    std::vector<float> eigenvalues;
    std::vector<std::vector<float>> eigenvectors;
    eigen_decompose(L, eigenvalues, eigenvectors);

    // Step 4: Determine number of speakers
    int k = target_k > 0 ? target_k : estimate_num_speakers(eigenvalues);
    k = std::max(1, std::min(k, n));

    LOG_INFO("Clustering %d segments into %d speakers", n, k);

    if (k == 1) {
        // Single speaker — everything in one cluster
        num_clusters_ = 1;
        return std::vector<int>(n, 0);
    }

    // Step 5: Take first k eigenvectors, form N x k matrix
    std::vector<std::vector<float>> spectral_emb(n, std::vector<float>(k, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            spectral_emb[i][j] = eigenvectors[j][i];
        }
        // Row-normalize
        float norm = 0.0f;
        for (int j = 0; j < k; j++) {
            norm += spectral_emb[i][j] * spectral_emb[i][j];
        }
        norm = std::sqrt(norm);
        if (norm > 1e-10f) {
            for (int j = 0; j < k; j++) {
                spectral_emb[i][j] /= norm;
            }
        }
    }

    // Step 6: K-means on spectral embedding
    LOG_DEBUG("Running k-means (k=%d, %d restarts)", k, 5);
    auto labels = kmeans(spectral_emb, k, 50, 5);

    // Validate: check for degenerate clustering (all same label)
    bool degenerate = true;
    for (int i = 1; i < n; i++) {
        if (labels[i] != labels[0]) {
            degenerate = false;
            break;
        }
    }

    if (degenerate && n > 2) {
        LOG_WARN("Spectral clustering produced degenerate result, falling back to AHC");
        labels = ahc_cluster(embeddings);

        // Count clusters
        int max_label = *std::max_element(labels.begin(), labels.end());
        num_clusters_ = max_label + 1;
    } else {
        num_clusters_ = k;
    }

    // Log cluster distribution
    std::vector<int> counts(num_clusters_, 0);
    for (int l : labels) {
        if (l >= 0 && l < num_clusters_) counts[l]++;
    }
    for (int c = 0; c < num_clusters_; c++) {
        LOG_DEBUG("  Speaker %d: %d segments", c, counts[c]);
    }

    LOG_INFO("Clustering complete: %d speakers identified", num_clusters_);
    return labels;
}

// ─── Assign speakers to segments ───────────────────────────────────────────

std::vector<DiarizedSegment> assign_speakers(
    const std::vector<SpeechSegment>& segments,
    const std::vector<int>& labels,
    double max_gap_sec) {

    if (segments.empty()) return {};

    std::vector<DiarizedSegment> result;

    for (size_t i = 0; i < segments.size(); i++) {
        int speaker = (i < labels.size()) ? labels[i] : 0;

        // Try to merge with previous if same speaker and small gap
        if (!result.empty() &&
            result.back().speaker_id == speaker &&
            segments[i].start_sec - result.back().end_sec <= max_gap_sec) {
            result.back().end_sec = segments[i].end_sec;
            result.back().confidence = (result.back().confidence + segments[i].avg_confidence) / 2.0f;
        } else {
            result.push_back({
                segments[i].start_sec,
                segments[i].end_sec,
                speaker,
                segments[i].avg_confidence
            });
        }
    }

    LOG_DEBUG("Assigned speakers: %zu segments → %zu diarized blocks",
              segments.size(), result.size());

    return result;
}

}  // namespace vaultasr
