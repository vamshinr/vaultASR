#pragma once

#include "types.h"
#include <vector>

namespace vaultasr {

// ─── Post-processing for transcript cleanup ────────────────────────────────
//
// Cleans up raw Whisper output:
//   - Merges adjacent segments from the same speaker
//   - Fixes capitalization at segment boundaries
//   - Removes duplicate text from overlapping segments
//   - Trims whitespace
//
class PostProcessor {
public:
    // Merge adjacent segments from the same speaker (gap < max_gap_sec)
    static std::vector<TranscriptSegment> merge_speakers(
        const std::vector<TranscriptSegment>& segments,
        double max_gap_sec = 1.5);

    // Fix capitalization: ensure sentences start with uppercase
    static void fix_capitalization(std::vector<TranscriptSegment>& segments);

    // Remove duplicate/overlapping text at segment boundaries
    static void deduplicate_boundaries(std::vector<TranscriptSegment>& segments);

    // Trim whitespace from all text fields
    static void trim_whitespace(std::vector<TranscriptSegment>& segments);

    // Run all post-processing steps
    static std::vector<TranscriptSegment> process(
        const std::vector<TranscriptSegment>& segments);
};

}  // namespace vaultasr
