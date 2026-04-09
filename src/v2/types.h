#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace vaultasr {

// ─── Speech segment from VAD ───────────────────────────────────────────────
struct SpeechSegment {
    double start_sec;
    double end_sec;
    float  avg_confidence;  // mean VAD probability across segment

    double duration() const { return end_sec - start_sec; }
};

// ─── Diarized segment (speech + speaker label) ────────────────────────────
struct DiarizedSegment {
    double start_sec;
    double end_sec;
    int    speaker_id;      // 0-indexed
    float  confidence;      // clustering confidence

    double duration() const { return end_sec - start_sec; }
};

// ─── Word-level info from Whisper ──────────────────────────────────────────
struct WordInfo {
    std::string text;
    double      start_sec;
    double      end_sec;
    float       probability;    // token log-probability
};

// ─── Full transcript segment ───────────────────────────────────────────────
struct TranscriptSegment {
    int                   speaker_id;
    double                start_sec;
    double                end_sec;
    std::string           text;
    std::vector<WordInfo> words;
    float                 avg_confidence;
};

// ─── Audio file metadata ───────────────────────────────────────────────────
struct AudioMeta {
    double      duration_sec;
    int         sample_rate;
    int         channels;
    std::string codec_name;
    std::string format_name;
};

// ─── Export format enum ────────────────────────────────────────────────────
enum class ExportFormat {
    STDOUT,
    JSON,
    CSV,
    XLSX,
    SRT,
    MARKDOWN,
    DOCX,
    SQLITE,
};

// Parse export format from string (case-insensitive)
inline ExportFormat parse_export_format(const std::string& s) {
    if (s == "json")     return ExportFormat::JSON;
    if (s == "csv")      return ExportFormat::CSV;
    if (s == "xlsx")     return ExportFormat::XLSX;
    if (s == "srt")      return ExportFormat::SRT;
    if (s == "markdown" || s == "md") return ExportFormat::MARKDOWN;
    if (s == "docx")     return ExportFormat::DOCX;
    if (s == "sqlite" || s == "sqlite3" || s == "db") return ExportFormat::SQLITE;
    return ExportFormat::STDOUT;
}

// ─── Progress callback ────────────────────────────────────────────────────
// stage: human-readable stage name
// progress: 0.0–1.0 within that stage
// detail: optional detail string (e.g. current segment text)
using ProgressCallback = std::function<void(
    const std::string& stage, float progress, const std::string& detail)>;

}  // namespace vaultasr
