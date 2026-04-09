#pragma once
#include <vector>
#include <string>

class Clustering {
public:
    Clustering(float threshold = 0.5f);
    
    // Pass in all embeddings extracted from the overlapping chunks
    // Returns a vector mapping each chunk to a speaker ID (0, 1, 2...)
    std::vector<int> cluster(const std::vector<std::vector<float>>& embeddings);

private:
    float threshold_;
    float cosine_distance(const std::vector<float>& a, const std::vector<float>& b);
};
