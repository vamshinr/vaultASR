#include "clustering.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

Clustering::Clustering(float threshold) : threshold_(threshold) {}

float Clustering::cosine_distance(const std::vector<float>& a, const std::vector<float>& b) {
    float dot = 0.0f;
    float norm_a = 0.0f;
    float norm_b = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    
    // L2 normalized inputs mean norm_a ~ 1 and norm_b ~ 1
    float cosine_similarity = dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-10f);
    // distance = 1 - similarity
    // To match typical AHC thresholds (which can be based on distance rather than similarity)
    return 1.0f - cosine_similarity;
}

std::vector<int> Clustering::cluster(const std::vector<std::vector<float>>& embeddings) {
    int n = embeddings.size();
    std::vector<int> labels(n);
    if (n == 0) return labels;
    
    // Initialize each embedding as its own cluster
    std::vector<std::vector<int>> clusters(n);
    for (int i = 0; i < n; i++) {
        clusters[i].push_back(i);
        labels[i] = i;
    }
    
    // Compute initial pairwise distance matrix
    std::vector<std::vector<float>> dist_matrix(n, std::vector<float>(n, 0.0f));
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            float dist = cosine_distance(embeddings[i], embeddings[j]);
            dist_matrix[i][j] = dist;
            dist_matrix[j][i] = dist;
        }
    }
    
    std::vector<bool> active_clusters(n, true);
    
    while (true) {
        float min_dist = std::numeric_limits<float>::max();
        int best_i = -1, best_j = -1;
        
        for (int i = 0; i < n; i++) {
            if (!active_clusters[i]) continue;
            for (int j = i + 1; j < n; j++) {
                if (!active_clusters[j]) continue;
                
                if (dist_matrix[i][j] < min_dist) {
                    min_dist = dist_matrix[i][j];
                    best_i = i;
                    best_j = j;
                }
            }
        }
        
        if (best_i == -1 || min_dist > threshold_) break;
        
        // Merge cluster j into cluster i
        active_clusters[best_j] = false;
        
        for (int idx : clusters[best_j]) {
            clusters[best_i].push_back(idx);
            labels[idx] = best_i;
        }
        clusters[best_j].clear();
        
        // Update distance matrix (Average Linkage)
        for (int k = 0; k < n; k++) {
            if (!active_clusters[k] || k == best_i) continue;
            
            // Average distance between all points in cluster i and cluster k
            float total_dist = 0.0f;
            int count = 0;
            for (int p1 : clusters[best_i]) {
                for (int p2 : clusters[k]) {
                    total_dist += cosine_distance(embeddings[p1], embeddings[p2]);
                    count++;
                }
            }
            float avg_dist = total_dist / (float)count;
            dist_matrix[best_i][k] = avg_dist;
            dist_matrix[k][best_i] = avg_dist;
        }
    }
    
    // Renumber labels sequentially (0, 1, 2...)
    int new_label = 0;
    std::vector<int> final_labels(n, -1);
    
    for (int i = 0; i < n; i++) {
        if (active_clusters[i]) {
            for (int idx : clusters[i]) {
                final_labels[idx] = new_label;
            }
            new_label++;
        }
    }
    
    return final_labels;
}
