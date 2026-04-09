#pragma once

#include "types.h"
#include <string>
#include <vector>

namespace vaultasr {

// ─── Transcript exporter ────────────────────────────────────────────────────
//
// Exports a transcript to any of the supported formats.
// All methods are static — no state needed.
//
struct ExportMeta {
    std::string audio_file;
    std::string model_name;
    double      duration_sec = 0.0;
    int         num_speakers = 0;
};

class Exporter {
public:
    static void write(
        const std::vector<TranscriptSegment>& transcript,
        ExportFormat format,
        const std::string& output_path,
        const ExportMeta& meta = {});

private:
    static void to_stdout  (const std::vector<TranscriptSegment>& t, const ExportMeta& m);
    static void to_json    (const std::vector<TranscriptSegment>& t, const ExportMeta& m, const std::string& path);
    static void to_csv     (const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_xlsx    (const std::vector<TranscriptSegment>& t, const ExportMeta& m, const std::string& path);
    static void to_srt     (const std::vector<TranscriptSegment>& t, const std::string& path);
    static void to_markdown(const std::vector<TranscriptSegment>& t, const ExportMeta& m, const std::string& path);
    static void to_docx    (const std::vector<TranscriptSegment>& t, const ExportMeta& m, const std::string& path);
    static void to_sqlite  (const std::vector<TranscriptSegment>& t, const ExportMeta& m, const std::string& path);

    // Utilities
    static std::string srt_timestamp(double sec);
    static std::string json_escape(const std::string& s);
    static std::string csv_escape(const std::string& s);
    static std::string format_time_hms(double sec);
};

}  // namespace vaultasr
