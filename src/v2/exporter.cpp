#include "exporter.h"
#include "logger.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>
#include <stdexcept>

// SQLite3
#include <sqlite3.h>

// libxlsxwriter
#include <xlsxwriter.h>

namespace vaultasr {

// ═══════════════════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════════════════

std::string Exporter::srt_timestamp(double sec) {
    int h   = static_cast<int>(sec) / 3600;
    int m   = (static_cast<int>(sec) % 3600) / 60;
    int s   = static_cast<int>(sec) % 60;
    int ms  = static_cast<int>((sec - static_cast<int>(sec)) * 1000);

    char buf[32];
    std::snprintf(buf, sizeof(buf), "%02d:%02d:%02d,%03d", h, m, s, ms);
    return buf;
}

std::string Exporter::format_time_hms(double sec) {
    int h  = static_cast<int>(sec) / 3600;
    int m  = (static_cast<int>(sec) % 3600) / 60;
    int s  = static_cast<int>(sec) % 60;

    char buf[16];
    if (h > 0)
        std::snprintf(buf, sizeof(buf), "%d:%02d:%02d", h, m, s);
    else
        std::snprintf(buf, sizeof(buf), "%d:%02d", m, s);
    return buf;
}

std::string Exporter::json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char hex[8];
                    std::snprintf(hex, sizeof(hex), "\\u%04x", c);
                    out += hex;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

std::string Exporter::csv_escape(const std::string& s) {
    // Wrap in quotes; double any internal quotes
    std::string out = "\"";
    for (char c : s) {
        if (c == '"') out += "\"\"";
        else          out += c;
    }
    out += '"';
    return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Dispatch
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::write(
    const std::vector<TranscriptSegment>& transcript,
    ExportFormat format,
    const std::string& output_path,
    const ExportMeta& meta) {

    LOG_INFO("Exporting %zu segments → format=%d path=%s",
             transcript.size(), static_cast<int>(format),
             output_path.empty() ? "(stdout)" : output_path.c_str());

    switch (format) {
        case ExportFormat::STDOUT:   to_stdout(transcript, meta);                   break;
        case ExportFormat::JSON:     to_json(transcript, meta, output_path);        break;
        case ExportFormat::CSV:      to_csv(transcript, output_path);               break;
        case ExportFormat::XLSX:     to_xlsx(transcript, meta, output_path);        break;
        case ExportFormat::SRT:      to_srt(transcript, output_path);               break;
        case ExportFormat::MARKDOWN: to_markdown(transcript, meta, output_path);    break;
        case ExportFormat::DOCX:     to_docx(transcript, meta, output_path);        break;
        case ExportFormat::SQLITE:   to_sqlite(transcript, meta, output_path);      break;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// STDOUT — human-readable Otter-style output
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_stdout(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta) {

    // Header
    std::printf("\n\033[1;36m╔══════════════════════════════════════════╗\033[0m\n");
    std::printf("\033[1;36m║         VaultASR Transcript               ║\033[0m\n");
    std::printf("\033[1;36m╚══════════════════════════════════════════╝\033[0m\n");
    if (!meta.audio_file.empty())
        std::printf("  File:    %s\n", meta.audio_file.c_str());
    if (meta.duration_sec > 0)
        std::printf("  Length:  %s\n", format_time_hms(meta.duration_sec).c_str());
    if (meta.num_speakers > 0)
        std::printf("  Speakers:%d\n", meta.num_speakers);
    if (!meta.model_name.empty())
        std::printf("  Model:   %s\n", meta.model_name.c_str());
    std::printf("\n");

    int prev_speaker = -1;
    for (const auto& seg : transcript) {
        // Print speaker header on speaker change
        if (seg.speaker_id != prev_speaker) {
            if (prev_speaker >= 0) std::printf("\n");
            std::printf("\033[1;33m[Speaker %d]\033[0m  \033[90m%s – %s\033[0m\n",
                        seg.speaker_id,
                        format_time_hms(seg.start_sec).c_str(),
                        format_time_hms(seg.end_sec).c_str());
            prev_speaker = seg.speaker_id;
        }

        std::printf("%s\n", seg.text.c_str());
    }

    std::printf("\n\033[90m──────────────────────────────────────────\033[0m\n");
    std::fflush(stdout);
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_json(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta,
    const std::string& path) {

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open for writing: " + path);

    f << "{\n";
    f << "  \"meta\": {\n";
    f << "    \"audio_file\": \"" << json_escape(meta.audio_file) << "\",\n";
    f << "    \"model\": \""      << json_escape(meta.model_name) << "\",\n";
    f << "    \"duration_sec\": " << std::fixed << std::setprecision(3) << meta.duration_sec << ",\n";
    f << "    \"num_speakers\": " << meta.num_speakers << "\n";
    f << "  },\n";
    f << "  \"segments\": [\n";

    for (size_t i = 0; i < transcript.size(); i++) {
        const auto& seg = transcript[i];
        f << "    {\n";
        f << "      \"speaker\": "    << seg.speaker_id << ",\n";
        f << "      \"start\": "      << std::fixed << std::setprecision(3) << seg.start_sec << ",\n";
        f << "      \"end\": "        << std::fixed << std::setprecision(3) << seg.end_sec << ",\n";
        f << "      \"confidence\": " << std::fixed << std::setprecision(4) << seg.avg_confidence << ",\n";
        f << "      \"text\": \""     << json_escape(seg.text) << "\",\n";
        f << "      \"words\": [\n";

        for (size_t wi = 0; wi < seg.words.size(); wi++) {
            const auto& w = seg.words[wi];
            f << "        {\"word\": \""  << json_escape(w.text) << "\","
              << " \"start\": "           << std::fixed << std::setprecision(3) << w.start_sec << ","
              << " \"end\": "             << std::fixed << std::setprecision(3) << w.end_sec << ","
              << " \"prob\": "            << std::fixed << std::setprecision(4) << w.probability << "}";
            if (wi + 1 < seg.words.size()) f << ",";
            f << "\n";
        }

        f << "      ]\n";
        f << "    }";
        if (i + 1 < transcript.size()) f << ",";
        f << "\n";
    }

    f << "  ]\n}\n";
    LOG_INFO("JSON written: %s", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// CSV
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_csv(
    const std::vector<TranscriptSegment>& transcript,
    const std::string& path) {

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open for writing: " + path);

    f << "speaker,start_sec,end_sec,duration_sec,confidence,text\n";

    for (const auto& seg : transcript) {
        f << "Speaker " << seg.speaker_id << ","
          << std::fixed << std::setprecision(3) << seg.start_sec << ","
          << std::fixed << std::setprecision(3) << seg.end_sec << ","
          << std::fixed << std::setprecision(3) << (seg.end_sec - seg.start_sec) << ","
          << std::fixed << std::setprecision(4) << seg.avg_confidence << ","
          << csv_escape(seg.text) << "\n";
    }

    LOG_INFO("CSV written: %s (%zu rows)", path.c_str(), transcript.size());
}

// ═══════════════════════════════════════════════════════════════════════════
// XLSX — using libxlsxwriter
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_xlsx(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta,
    const std::string& path) {

    lxw_workbook*  wb = workbook_new(path.c_str());
    lxw_worksheet* ws = workbook_add_worksheet(wb, "Transcript");

    // ── Formats ──────────────────────────────────────────────────────────
    lxw_format* header_fmt = workbook_add_format(wb);
    format_set_bold(header_fmt);
    format_set_bg_color(header_fmt, 0x1F497D);   // dark blue
    format_set_font_color(header_fmt, 0xFFFFFF);
    format_set_border(header_fmt, LXW_BORDER_THIN);
    format_set_align(header_fmt, LXW_ALIGN_CENTER);

    lxw_format* speaker_fmt = workbook_add_format(wb);
    format_set_bold(speaker_fmt);
    format_set_font_color(speaker_fmt, 0x1F497D);

    lxw_format* time_fmt = workbook_add_format(wb);
    format_set_num_format(time_fmt, "0.000");
    format_set_align(time_fmt, LXW_ALIGN_CENTER);

    lxw_format* conf_fmt = workbook_add_format(wb);
    format_set_num_format(conf_fmt, "0.00%");
    format_set_align(conf_fmt, LXW_ALIGN_CENTER);

    lxw_format* text_fmt = workbook_add_format(wb);
    format_set_text_wrap(text_fmt);

    lxw_format* alt_fmt = workbook_add_format(wb);
    format_set_bg_color(alt_fmt, 0xF2F2F2);
    format_set_text_wrap(alt_fmt);

    // ── Column widths ─────────────────────────────────────────────────────
    worksheet_set_column(ws, 0, 0, 14, nullptr);   // Speaker
    worksheet_set_column(ws, 1, 1, 10, nullptr);   // Start
    worksheet_set_column(ws, 2, 2, 10, nullptr);   // End
    worksheet_set_column(ws, 3, 3, 10, nullptr);   // Duration
    worksheet_set_column(ws, 4, 4, 10, nullptr);   // Confidence
    worksheet_set_column(ws, 5, 5, 70, nullptr);   // Text

    // ── Meta info ─────────────────────────────────────────────────────────
    lxw_worksheet* meta_ws = workbook_add_worksheet(wb, "Info");
    worksheet_write_string(meta_ws, 0, 0, "Audio File",   speaker_fmt);
    worksheet_write_string(meta_ws, 0, 1, meta.audio_file.c_str(), nullptr);
    worksheet_write_string(meta_ws, 1, 0, "Model",        speaker_fmt);
    worksheet_write_string(meta_ws, 1, 1, meta.model_name.c_str(), nullptr);
    worksheet_write_string(meta_ws, 2, 0, "Duration (s)", speaker_fmt);
    worksheet_write_number(meta_ws, 2, 1, meta.duration_sec, time_fmt);
    worksheet_write_string(meta_ws, 3, 0, "Speakers",     speaker_fmt);
    worksheet_write_number(meta_ws, 3, 1, meta.num_speakers, nullptr);
    worksheet_write_string(meta_ws, 4, 0, "Segments",     speaker_fmt);
    worksheet_write_number(meta_ws, 4, 1, static_cast<double>(transcript.size()), nullptr);

    // ── Headers ───────────────────────────────────────────────────────────
    worksheet_write_string(ws, 0, 0, "Speaker",    header_fmt);
    worksheet_write_string(ws, 0, 1, "Start (s)",  header_fmt);
    worksheet_write_string(ws, 0, 2, "End (s)",    header_fmt);
    worksheet_write_string(ws, 0, 3, "Duration",   header_fmt);
    worksheet_write_string(ws, 0, 4, "Confidence", header_fmt);
    worksheet_write_string(ws, 0, 5, "Text",       header_fmt);

    // Freeze top row
    worksheet_freeze_panes(ws, 1, 0);

    // ── Data rows ─────────────────────────────────────────────────────────
    for (size_t i = 0; i < transcript.size(); i++) {
        const auto& seg = transcript[i];
        lxw_row_t row = static_cast<lxw_row_t>(i + 1);
        lxw_format* row_fmt = (i % 2 == 1) ? alt_fmt : text_fmt;

        std::string spk = "Speaker " + std::to_string(seg.speaker_id);
        worksheet_write_string(ws, row, 0, spk.c_str(),     speaker_fmt);
        worksheet_write_number(ws, row, 1, seg.start_sec,   time_fmt);
        worksheet_write_number(ws, row, 2, seg.end_sec,     time_fmt);
        worksheet_write_number(ws, row, 3, seg.end_sec - seg.start_sec, time_fmt);
        worksheet_write_number(ws, row, 4, seg.avg_confidence, conf_fmt);
        worksheet_write_string(ws, row, 5, seg.text.c_str(), row_fmt);

        // Row height for wrapped text (approx 15pt per line)
        size_t lines = seg.text.size() / 80 + 1;
        worksheet_set_row(ws, row, static_cast<double>(lines) * 15.0, nullptr);
    }

    // ── Words sheet ───────────────────────────────────────────────────────
    bool has_words = false;
    for (const auto& seg : transcript) {
        if (!seg.words.empty()) { has_words = true; break; }
    }

    if (has_words) {
        lxw_worksheet* words_ws = workbook_add_worksheet(wb, "Words");
        worksheet_write_string(words_ws, 0, 0, "Speaker",    header_fmt);
        worksheet_write_string(words_ws, 0, 1, "Word",       header_fmt);
        worksheet_write_string(words_ws, 0, 2, "Start (s)",  header_fmt);
        worksheet_write_string(words_ws, 0, 3, "End (s)",    header_fmt);
        worksheet_write_string(words_ws, 0, 4, "Confidence", header_fmt);
        worksheet_set_column(words_ws, 0, 0, 14, nullptr);
        worksheet_set_column(words_ws, 1, 1, 20, nullptr);
        worksheet_set_column(words_ws, 2, 3, 10, nullptr);
        worksheet_set_column(words_ws, 4, 4, 10, nullptr);
        worksheet_freeze_panes(words_ws, 1, 0);

        lxw_row_t wrow = 1;
        for (const auto& seg : transcript) {
            std::string spk = "Speaker " + std::to_string(seg.speaker_id);
            for (const auto& w : seg.words) {
                lxw_format* wfmt = (wrow % 2 == 1) ? alt_fmt : nullptr;
                worksheet_write_string(words_ws, wrow, 0, spk.c_str(),     speaker_fmt);
                worksheet_write_string(words_ws, wrow, 1, w.text.c_str(),  wfmt);
                worksheet_write_number(words_ws, wrow, 2, w.start_sec,     time_fmt);
                worksheet_write_number(words_ws, wrow, 3, w.end_sec,       time_fmt);
                worksheet_write_number(words_ws, wrow, 4, w.probability,   conf_fmt);
                wrow++;
            }
        }
    }

    workbook_close(wb);
    LOG_INFO("XLSX written: %s", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// SRT — SubRip subtitle format
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_srt(
    const std::vector<TranscriptSegment>& transcript,
    const std::string& path) {

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open for writing: " + path);

    int idx = 1;
    for (const auto& seg : transcript) {
        if (seg.text.empty()) continue;

        f << idx++ << "\n";
        f << srt_timestamp(seg.start_sec) << " --> " << srt_timestamp(seg.end_sec) << "\n";
        f << "[Speaker " << seg.speaker_id << "] " << seg.text << "\n\n";
    }

    LOG_INFO("SRT written: %s (%d entries)", path.c_str(), idx - 1);
}

// ═══════════════════════════════════════════════════════════════════════════
// Markdown
// ═══════════════════════════════════════════════════════════════════════════

void Exporter::to_markdown(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta,
    const std::string& path) {

    std::ofstream f(path);
    if (!f.is_open()) throw std::runtime_error("Cannot open for writing: " + path);

    // Header
    f << "# Transcript\n\n";
    if (!meta.audio_file.empty())
        f << "**File:** " << meta.audio_file << "  \n";
    if (meta.duration_sec > 0)
        f << "**Duration:** " << format_time_hms(meta.duration_sec) << "  \n";
    if (meta.num_speakers > 0)
        f << "**Speakers:** " << meta.num_speakers << "  \n";
    if (!meta.model_name.empty())
        f << "**Model:** " << meta.model_name << "  \n";
    f << "\n---\n\n";

    // Build per-speaker color map (Markdown doesn't support colors, use headers)
    int prev_speaker = -1;
    for (const auto& seg : transcript) {
        if (seg.text.empty()) continue;

        if (seg.speaker_id != prev_speaker) {
            if (prev_speaker >= 0) f << "\n";
            f << "### Speaker " << seg.speaker_id
              << " `" << format_time_hms(seg.start_sec) << "`\n\n";
            prev_speaker = seg.speaker_id;
        }

        f << seg.text << "\n\n";
    }

    // Appendix: speaker stats
    f << "---\n\n## Speaker Statistics\n\n";
    f << "| Speaker | Segments | Total Time |\n";
    f << "|---------|----------|------------|\n";

    std::map<int, std::pair<int, double>> stats;
    for (const auto& seg : transcript) {
        stats[seg.speaker_id].first++;
        stats[seg.speaker_id].second += seg.end_sec - seg.start_sec;
    }
    for (const auto& [spk, data] : stats) {
        f << "| Speaker " << spk << " | " << data.first
          << " | " << format_time_hms(data.second) << " |\n";
    }

    LOG_INFO("Markdown written: %s", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// DOCX — minimal Open XML implementation (no external library)
// Generates a valid .docx (ZIP of XML files) without any OOXML library.
// ═══════════════════════════════════════════════════════════════════════════

// We write XML into memory then zip using miniz (header-only).
// miniz is included via external/miniz.

#define MINIZ_HEADER_FILE_ONLY
#include "../../external/miniz/miniz.h"

static void docx_add_file(mz_zip_archive& zip, const char* name, const std::string& content) {
    mz_zip_writer_add_mem(&zip, name, content.data(), content.size(), MZ_BEST_COMPRESSION);
}

static std::string xml_escape(const std::string& s) {
    std::string out;
    for (char c : s) {
        switch (c) {
            case '<':  out += "&lt;";   break;
            case '>':  out += "&gt;";   break;
            case '&':  out += "&amp;";  break;
            case '"':  out += "&quot;"; break;
            case '\'': out += "&apos;"; break;
            default:   out += c;
        }
    }
    return out;
}

void Exporter::to_docx(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta,
    const std::string& path) {

    mz_zip_archive zip;
    memset(&zip, 0, sizeof(zip));

    if (!mz_zip_writer_init_file(&zip, path.c_str(), 0)) {
        throw std::runtime_error("Cannot create DOCX file: " + path);
    }

    // ── [Content_Types].xml ───────────────────────────────────────────────
    docx_add_file(zip, "[Content_Types].xml", R"(<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml"  ContentType="application/xml"/>
  <Override PartName="/word/document.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml"
    ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
</Types>)");

    // ── _rels/.rels ───────────────────────────────────────────────────────
    docx_add_file(zip, "_rels/.rels", R"(<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
    Target="word/document.xml"/>
</Relationships>)");

    // ── word/_rels/document.xml.rels ──────────────────────────────────────
    docx_add_file(zip, "word/_rels/document.xml.rels", R"(<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
    Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
    Target="styles.xml"/>
</Relationships>)");

    // ── word/styles.xml ───────────────────────────────────────────────────
    docx_add_file(zip, "word/styles.xml", R"(<?xml version="1.0" encoding="UTF-8"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:style w:type="paragraph" w:styleId="Normal" w:default="1">
    <w:name w:val="Normal"/>
    <w:rPr><w:sz w:val="24"/><w:szCs w:val="24"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:pPr><w:outlineLvl w:val="0"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="36"/><w:color w:val="1F497D"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:pPr><w:outlineLvl w:val="1"/></w:pPr>
    <w:rPr><w:b/><w:sz w:val="28"/><w:color w:val="4472C4"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="SpeakerLabel">
    <w:name w:val="SpeakerLabel"/>
    <w:rPr><w:b/><w:color w:val="C55A11"/><w:sz w:val="22"/></w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Timestamp">
    <w:name w:val="Timestamp"/>
    <w:rPr><w:color w:val="808080"/><w:sz w:val="18"/></w:rPr>
  </w:style>
</w:styles>)");

    // ── word/document.xml ─────────────────────────────────────────────────
    std::string doc_xml;
    doc_xml.reserve(65536);

    doc_xml += R"(<?xml version="1.0" encoding="UTF-8"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:body>
)";

    // Title
    doc_xml += R"(  <w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr>)";
    doc_xml += R"(<w:r><w:t>Transcript</w:t></w:r></w:p>)";
    doc_xml += "\n";

    // Meta info
    auto meta_line = [&](const std::string& label, const std::string& value) {
        doc_xml += "  <w:p><w:r><w:rPr><w:b/></w:rPr><w:t xml:space=\"preserve\">"
                 + xml_escape(label) + ": </w:t></w:r>"
                 + "<w:r><w:t>" + xml_escape(value) + "</w:t></w:r></w:p>\n";
    };

    if (!meta.audio_file.empty()) meta_line("File",     meta.audio_file);
    if (meta.duration_sec > 0)    meta_line("Duration", format_time_hms(meta.duration_sec));
    if (meta.num_speakers > 0)    meta_line("Speakers", std::to_string(meta.num_speakers));
    if (!meta.model_name.empty()) meta_line("Model",    meta.model_name);

    // Horizontal rule (empty paragraph with bottom border)
    doc_xml += R"(  <w:p><w:pPr><w:pBdr>)";
    doc_xml += R"(<w:bottom w:val="single" w:sz="6" w:space="1" w:color="1F497D"/>)";
    doc_xml += R"(</w:pBdr></w:pPr></w:p>)";
    doc_xml += "\n  <w:p/>\n";

    // Transcript body
    int prev_speaker = -1;
    for (const auto& seg : transcript) {
        if (seg.text.empty()) continue;

        // Speaker heading on change
        if (seg.speaker_id != prev_speaker) {
            doc_xml += "  <w:p><w:pPr><w:pStyle w:val=\"Heading2\"/>"
                       "<w:spacing w:before=\"200\" w:after=\"40\"/></w:pPr>"
                       "<w:r><w:t>Speaker " + std::to_string(seg.speaker_id) + "</w:t></w:r>"
                       "<w:r><w:rPr><w:color w:val=\"808080\"/><w:sz w:val=\"18\"/></w:rPr>"
                       "<w:t xml:space=\"preserve\">  "
                     + format_time_hms(seg.start_sec) + " – "
                     + format_time_hms(seg.end_sec) + "</w:t></w:r></w:p>\n";
            prev_speaker = seg.speaker_id;
        }

        // Text paragraph
        doc_xml += "  <w:p><w:pPr><w:spacing w:after=\"80\"/></w:pPr>"
                   "<w:r><w:t>" + xml_escape(seg.text) + "</w:t></w:r></w:p>\n";
    }

    doc_xml += R"(  <w:sectPr>
    <w:pgSz w:w="12240" w:h="15840"/>
    <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
  </w:sectPr>
</w:body></w:document>)";

    docx_add_file(zip, "word/document.xml", doc_xml);

    if (!mz_zip_writer_finalize_archive(&zip)) {
        mz_zip_writer_end(&zip);
        throw std::runtime_error("Failed to finalize DOCX archive: " + path);
    }
    mz_zip_writer_end(&zip);

    LOG_INFO("DOCX written: %s", path.c_str());
}

// ═══════════════════════════════════════════════════════════════════════════
// SQLite3 — full analysis database
// ═══════════════════════════════════════════════════════════════════════════

static void sql_exec(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        std::string msg = err ? err : "unknown error";
        sqlite3_free(err);
        throw std::runtime_error("SQLite error: " + msg + "\nSQL: " + sql);
    }
}

void Exporter::to_sqlite(
    const std::vector<TranscriptSegment>& transcript,
    const ExportMeta& meta,
    const std::string& path) {

    sqlite3* db = nullptr;
    int rc = sqlite3_open(path.c_str(), &db);
    if (rc != SQLITE_OK) {
        throw std::runtime_error("Cannot open SQLite DB: " + path);
    }

    // Schema
    sql_exec(db, "PRAGMA journal_mode=WAL;");
    sql_exec(db, "PRAGMA synchronous=NORMAL;");

    sql_exec(db, R"(
        CREATE TABLE IF NOT EXISTS transcripts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            audio_file  TEXT,
            model       TEXT,
            duration_sec REAL,
            num_speakers INTEGER,
            created_at  TEXT DEFAULT (datetime('now'))
        );
    )");

    sql_exec(db, R"(
        CREATE TABLE IF NOT EXISTS segments (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            transcript_id INTEGER REFERENCES transcripts(id),
            speaker_id    INTEGER NOT NULL,
            start_sec     REAL    NOT NULL,
            end_sec       REAL    NOT NULL,
            duration_sec  REAL    NOT NULL,
            confidence    REAL,
            text          TEXT    NOT NULL
        );
    )");

    sql_exec(db, R"(
        CREATE TABLE IF NOT EXISTS words (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            segment_id  INTEGER REFERENCES segments(id),
            word        TEXT    NOT NULL,
            start_sec   REAL    NOT NULL,
            end_sec     REAL    NOT NULL,
            confidence  REAL
        );
    )");

    sql_exec(db, R"(
        CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
        CREATE INDEX IF NOT EXISTS idx_segments_speaker    ON segments(speaker_id);
        CREATE INDEX IF NOT EXISTS idx_words_segment       ON words(segment_id);
    )");

    // Insert transcript record
    sql_exec(db, "BEGIN TRANSACTION;");

    {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db,
            "INSERT INTO transcripts (audio_file, model, duration_sec, num_speakers) "
            "VALUES (?, ?, ?, ?);",
            -1, &stmt, nullptr);

        sqlite3_bind_text(stmt, 1, meta.audio_file.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_text(stmt, 2, meta.model_name.c_str(), -1, SQLITE_STATIC);
        sqlite3_bind_double(stmt, 3, meta.duration_sec);
        sqlite3_bind_int(stmt, 4, meta.num_speakers);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    }

    sqlite3_int64 transcript_id = sqlite3_last_insert_rowid(db);

    // Insert segments and words
    sqlite3_stmt* seg_stmt = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO segments (transcript_id, speaker_id, start_sec, end_sec, "
        "duration_sec, confidence, text) VALUES (?, ?, ?, ?, ?, ?, ?);",
        -1, &seg_stmt, nullptr);

    sqlite3_stmt* word_stmt = nullptr;
    sqlite3_prepare_v2(db,
        "INSERT INTO words (segment_id, word, start_sec, end_sec, confidence) "
        "VALUES (?, ?, ?, ?, ?);",
        -1, &word_stmt, nullptr);

    for (const auto& seg : transcript) {
        sqlite3_reset(seg_stmt);
        sqlite3_bind_int64(seg_stmt, 1, transcript_id);
        sqlite3_bind_int(seg_stmt,   2, seg.speaker_id);
        sqlite3_bind_double(seg_stmt, 3, seg.start_sec);
        sqlite3_bind_double(seg_stmt, 4, seg.end_sec);
        sqlite3_bind_double(seg_stmt, 5, seg.end_sec - seg.start_sec);
        sqlite3_bind_double(seg_stmt, 6, seg.avg_confidence);
        sqlite3_bind_text(seg_stmt,   7, seg.text.c_str(), -1, SQLITE_STATIC);
        sqlite3_step(seg_stmt);

        sqlite3_int64 seg_id = sqlite3_last_insert_rowid(db);

        for (const auto& w : seg.words) {
            sqlite3_reset(word_stmt);
            sqlite3_bind_int64(word_stmt,  1, seg_id);
            sqlite3_bind_text(word_stmt,   2, w.text.c_str(), -1, SQLITE_STATIC);
            sqlite3_bind_double(word_stmt, 3, w.start_sec);
            sqlite3_bind_double(word_stmt, 4, w.end_sec);
            sqlite3_bind_double(word_stmt, 5, w.probability);
            sqlite3_step(word_stmt);
        }
    }

    sqlite3_finalize(seg_stmt);
    sqlite3_finalize(word_stmt);

    sql_exec(db, "COMMIT;");
    sqlite3_close(db);

    LOG_INFO("SQLite written: %s (transcript_id=%lld)", path.c_str(), (long long)transcript_id);
}

}  // namespace vaultasr
