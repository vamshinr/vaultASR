#include "post_processor.h"
#include "logger.h"

#include <algorithm>
#include <cctype>
#include <sstream>

namespace vaultasr {

// ─── Utility: trim string ──────────────────────────────────────────────────

static std::string trim(const std::string& s) {
    size_t start = s.find_first_not_of(" \t\n\r");
    if (start == std::string::npos) return "";
    size_t end = s.find_last_not_of(" \t\n\r");
    return s.substr(start, end - start + 1);
}

// ─── Merge adjacent same-speaker segments ──────────────────────────────────

std::vector<TranscriptSegment> PostProcessor::merge_speakers(
    const std::vector<TranscriptSegment>& segments,
    double max_gap_sec) {

    if (segments.empty()) return {};

    std::vector<TranscriptSegment> merged;
    merged.push_back(segments[0]);

    for (size_t i = 1; i < segments.size(); i++) {
        auto& prev = merged.back();
        const auto& curr = segments[i];

        // Merge if same speaker and gap is small enough
        if (curr.speaker_id == prev.speaker_id &&
            curr.start_sec - prev.end_sec <= max_gap_sec) {

            // Extend time range
            prev.end_sec = curr.end_sec;

            // Concatenate text with space
            if (!prev.text.empty() && !curr.text.empty()) {
                prev.text += " " + curr.text;
            } else if (!curr.text.empty()) {
                prev.text = curr.text;
            }

            // Merge word lists
            prev.words.insert(prev.words.end(), curr.words.begin(), curr.words.end());

            // Average confidence
            prev.avg_confidence = (prev.avg_confidence + curr.avg_confidence) / 2.0f;

            LOG_TRACE("Merged segment %zu into previous (speaker %d, gap=%.2fs)",
                      i, curr.speaker_id, curr.start_sec - prev.end_sec);
        } else {
            merged.push_back(curr);
        }
    }

    LOG_DEBUG("Merged %zu segments → %zu (max_gap=%.1fs)",
              segments.size(), merged.size(), max_gap_sec);

    return merged;
}

// ─── Fix capitalization ────────────────────────────────────────────────────

void PostProcessor::fix_capitalization(std::vector<TranscriptSegment>& segments) {
    for (auto& seg : segments) {
        if (seg.text.empty()) continue;

        // Capitalize first character
        if (std::islower(static_cast<unsigned char>(seg.text[0]))) {
            seg.text[0] = std::toupper(static_cast<unsigned char>(seg.text[0]));
        }

        // Capitalize after sentence-ending punctuation
        bool capitalize_next = false;
        for (size_t i = 0; i < seg.text.size(); i++) {
            char c = seg.text[i];
            if (c == '.' || c == '!' || c == '?') {
                capitalize_next = true;
            } else if (capitalize_next && std::isalpha(static_cast<unsigned char>(c))) {
                seg.text[i] = std::toupper(static_cast<unsigned char>(c));
                capitalize_next = false;
            } else if (!std::isspace(static_cast<unsigned char>(c))) {
                capitalize_next = false;
            }
        }
    }
}

// ─── Deduplicate boundaries ────────────────────────────────────────────────

void PostProcessor::deduplicate_boundaries(std::vector<TranscriptSegment>& segments) {
    // When segments overlap in time, Whisper can produce duplicate text
    // at boundaries. Detect and remove repeated phrases.

    for (size_t i = 1; i < segments.size(); i++) {
        auto& prev = segments[i - 1];
        auto& curr = segments[i];

        if (prev.text.empty() || curr.text.empty()) continue;

        // Check if the end of prev matches the start of curr
        // Look for overlapping phrases (up to 50 characters)
        size_t max_check = std::min(prev.text.size(), std::min(curr.text.size(), size_t(50)));

        for (size_t len = max_check; len >= 5; len--) {
            std::string prev_tail = prev.text.substr(prev.text.size() - len);
            std::string curr_head = curr.text.substr(0, len);

            if (prev_tail == curr_head) {
                LOG_TRACE("Dedup: removed %zu overlapping chars at boundary %zu-%zu",
                          len, i - 1, i);
                curr.text = curr.text.substr(len);
                break;
            }
        }
    }
}

// ─── Trim whitespace ───────────────────────────────────────────────────────

void PostProcessor::trim_whitespace(std::vector<TranscriptSegment>& segments) {
    for (auto& seg : segments) {
        seg.text = trim(seg.text);

        // Also collapse internal multiple spaces
        std::string cleaned;
        bool prev_space = false;
        for (char c : seg.text) {
            if (std::isspace(static_cast<unsigned char>(c))) {
                if (!prev_space) cleaned += ' ';
                prev_space = true;
            } else {
                cleaned += c;
                prev_space = false;
            }
        }
        seg.text = cleaned;
    }
}

// ─── Run all post-processing ───────────────────────────────────────────────

std::vector<TranscriptSegment> PostProcessor::process(
    const std::vector<TranscriptSegment>& segments) {

    LOG_STAGE("Post-Processing");
    LOG_INFO("Processing %zu raw segments", segments.size());

    // Step 1: Merge same-speaker adjacent segments
    auto result = merge_speakers(segments);

    // Step 2: Deduplicate overlapping text
    deduplicate_boundaries(result);

    // Step 3: Fix capitalization
    fix_capitalization(result);

    // Step 4: Trim whitespace
    trim_whitespace(result);

    // Remove empty segments
    result.erase(
        std::remove_if(result.begin(), result.end(),
                        [](const TranscriptSegment& s) { return s.text.empty(); }),
        result.end());

    LOG_INFO("Post-processing complete: %zu final segments", result.size());
    return result;
}

}  // namespace vaultasr
