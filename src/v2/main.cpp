#include "pipeline.h"
#include "logger.h"
#include "progress.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace vaultasr;

// ─── Version ───────────────────────────────────────────────────────────────
static constexpr const char* VAULTASR_VERSION = "2.0.0";

// ─── Help text ─────────────────────────────────────────────────────────────
static void print_usage(const char* prog) {
    std::fprintf(stderr, R"(
VaultASR v%s — Local speech-to-text pipeline

USAGE:
    %s [OPTIONS] <audio-file> [<audio-file> ...]

AUDIO INPUT:
    <audio-file>              One or more audio or video files to transcribe
                              Supports: MP3, WAV, FLAC, OGG, AAC, M4A, OPUS,
                                        MP4, MOV, MKV, AVI, WMV, FLV, MTS, ...

MODEL SELECTION:
    --model <name>            Whisper model size (default: tiny.en)
                              Options: tiny.en  base.en  small.en  medium.en
                                       tiny     base     small     medium
    --model-path <path>       Explicit path to a GGML model file
    --models-dir <path>       Directory containing model files (default: models/)

PIPELINE OPTIONS:
    --denoise                 Enable RNNoise denoising (default: off)
    --no-diarize              Disable speaker diarization (single-speaker mode)
    --language <lang>         Whisper language code (default: en)
                              Use 'auto' for automatic detection
    --max-speakers <n>        Maximum number of speakers (default: 10)
    --translate               Translate to English

VAD TUNING:
    --vad-threshold <f>       Speech probability threshold (default: 0.50)
    --vad-neg-threshold <f>   Silence threshold / hysteresis (default: 0.35)
    --min-speech <f>          Minimum speech duration, seconds (default: 0.25)
    --min-silence <f>         Minimum silence to split on, seconds (default: 0.30)
    --speech-pad <f>          Padding around speech regions, seconds (default: 0.10)
    --max-segment <f>         Max segment length before forced split (default: 30.0)

CLUSTERING:
    --cluster-threshold <f>   AHC fallback threshold (default: 0.55)
    --fixed-speakers <n>      Skip auto-detection, use exactly N speakers

OUTPUT:
    --output <path>           Output file path (stdout by default)
    --format <fmt>            Export format (default: text)
                              Options: text  json  csv  xlsx  srt  markdown  docx  sqlite
    --no-stream               Suppress live transcript streaming to stderr
    --timestamps              Include timestamps in text output

PERFORMANCE:
    --threads <n>             CPU inference threads (default: 4)
    --no-gpu                  Disable Metal/GPU acceleration (CPU only)

LOGGING:
    --verbose                 INFO + DEBUG log messages
    --trace                   All log messages including per-frame TRACE
    --quiet                   Errors only — suppress INFO and progress

GENERAL:
    --version                 Show version and exit
    --help                    Show this help and exit

EXAMPLES:
    # Transcribe a meeting recording
    vaultasr meeting.mp3

    # High-quality with base model, export to JSON
    vaultasr --model base.en --format json --output transcript.json call.wav

    # Single speaker, SRT subtitles
    vaultasr --no-diarize --format srt lecture.m4a

    # Noisy recording, denoise first, full analysis DB
    vaultasr --denoise --format sqlite --output analysis.db interview.mp3

    # Queue multiple files
    vaultasr *.mp3 --format csv --output transcripts.csv

    # Maximum trace logging for debugging
    vaultasr --trace --model tiny.en audio.wav

)", VAULTASR_VERSION, prog);
}

// ─── Argument helpers ──────────────────────────────────────────────────────
static bool has_flag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; i++)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

static const char* get_arg(int argc, char* argv[], const char* flag, const char* default_val = nullptr) {
    for (int i = 1; i < argc - 1; i++)
        if (std::strcmp(argv[i], flag) == 0) return argv[i + 1];
    return default_val;
}

static float get_float(int argc, char* argv[], const char* flag, float default_val) {
    const char* s = get_arg(argc, argv, flag);
    return s ? std::stof(s) : default_val;
}

static int get_int(int argc, char* argv[], const char* flag, int default_val) {
    const char* s = get_arg(argc, argv, flag);
    return s ? std::stoi(s) : default_val;
}

// Resolve model file path from shorthand or explicit path
static std::string resolve_model(int argc, char* argv[]) {
    const char* explicit_path = get_arg(argc, argv, "--model-path");
    if (explicit_path) return explicit_path;

    const char* models_dir = get_arg(argc, argv, "--models-dir", "models");
    const char* model_name = get_arg(argc, argv, "--model", "tiny.en");

    // Support shorthand: "tiny.en" → "ggml-tiny.en.bin"
    std::string name(model_name);
    if (name.find("ggml-") == std::string::npos) {
        name = "ggml-" + name + ".bin";
    }

    return std::string(models_dir) + "/" + name;
}

// ─── Main ──────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    // Quick exits
    if (argc < 2 || has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_usage(argv[0]);
        return 0;
    }
    if (has_flag(argc, argv, "--version")) {
        std::printf("VaultASR %s\n", VAULTASR_VERSION);
        return 0;
    }

    // ── Logging setup ──────────────────────────────────────────────────────
    if (has_flag(argc, argv, "--trace"))   Logger::instance().set_level(LogLevel::TRACE);
    else if (has_flag(argc, argv, "--verbose")) Logger::instance().set_level(LogLevel::DEBUG);
    else if (has_flag(argc, argv, "--quiet"))   Logger::instance().set_level(LogLevel::ERROR);
    else                                        Logger::instance().set_level(LogLevel::INFO);

    Logger::instance().reset_timer();

    // ── Collect input files ────────────────────────────────────────────────
    std::vector<std::string> audio_files;
    for (int i = 1; i < argc; i++) {
        const char* arg = argv[i];
        if (arg[0] == '-') {
            // Skip flag and its value
            bool has_value = (
                std::strcmp(arg, "--model") == 0 ||
                std::strcmp(arg, "--model-path") == 0 ||
                std::strcmp(arg, "--models-dir") == 0 ||
                std::strcmp(arg, "--language") == 0 ||
                std::strcmp(arg, "--max-speakers") == 0 ||
                std::strcmp(arg, "--vad-threshold") == 0 ||
                std::strcmp(arg, "--vad-neg-threshold") == 0 ||
                std::strcmp(arg, "--min-speech") == 0 ||
                std::strcmp(arg, "--min-silence") == 0 ||
                std::strcmp(arg, "--speech-pad") == 0 ||
                std::strcmp(arg, "--max-segment") == 0 ||
                std::strcmp(arg, "--cluster-threshold") == 0 ||
                std::strcmp(arg, "--fixed-speakers") == 0 ||
                std::strcmp(arg, "--output") == 0 ||
                std::strcmp(arg, "--format") == 0 ||
                std::strcmp(arg, "--threads") == 0
            );
            if (has_value) i++;  // skip next token (it's the flag's value)
        } else {
            audio_files.push_back(arg);
        }
    }

    if (audio_files.empty()) {
        std::fprintf(stderr, "Error: no audio files specified.\n");
        print_usage(argv[0]);
        return 1;
    }

    // ── Build pipeline config ──────────────────────────────────────────────
    PipelineConfig cfg;

    // VAD
    cfg.vad_config.threshold       = get_float(argc, argv, "--vad-threshold",     0.50f);
    cfg.vad_config.neg_threshold   = get_float(argc, argv, "--vad-neg-threshold", 0.35f);
    cfg.vad_config.min_speech_sec  = get_float(argc, argv, "--min-speech",        0.25f);
    cfg.vad_config.min_silence_sec = get_float(argc, argv, "--min-silence",       0.30f);
    cfg.vad_config.speech_pad_sec  = get_float(argc, argv, "--speech-pad",        0.10f);
    cfg.vad_config.max_segment_sec = get_float(argc, argv, "--max-segment",       30.0f);

    // Diarization
    cfg.skip_diarization = has_flag(argc, argv, "--no-diarize");
    cfg.cluster_config.max_speakers = get_int(argc, argv, "--max-speakers", 10);
    cfg.cluster_config.ahc_threshold = get_float(argc, argv, "--cluster-threshold", 0.55f);
    {
        int fixed = get_int(argc, argv, "--fixed-speakers", 0);
        if (fixed > 0) {
            cfg.cluster_config.fixed_num_speakers = fixed;
            cfg.cluster_config.auto_num_speakers  = false;
        }
    }

    // Transcriber
    cfg.transcriber_config.model_path    = resolve_model(argc, argv);
    cfg.transcriber_config.language      = get_arg(argc, argv, "--language", "en");
    cfg.transcriber_config.use_gpu       = !has_flag(argc, argv, "--no-gpu");
    cfg.transcriber_config.n_threads     = get_int(argc, argv, "--threads", 4);
    cfg.transcriber_config.translate     = has_flag(argc, argv, "--translate");
    // --timestamps enables word-level timing in text output; always collect
    // internally (Whisper produces them cheaply), control export separately
    cfg.transcriber_config.word_timestamps = true;
    bool show_timestamps = has_flag(argc, argv, "--timestamps");

    // Denoising
    cfg.denoise = has_flag(argc, argv, "--denoise");

    // CoreML
    cfg.use_coreml = !has_flag(argc, argv, "--no-gpu");

    // Output
    {
        const char* fmt_str = get_arg(argc, argv, "--format", "text");
        cfg.export_format = parse_export_format(fmt_str);
    }
    {
        const char* out = get_arg(argc, argv, "--output");
        if (out) cfg.output_path = out;
    }

    // Streaming / progress
    cfg.stream_text = !has_flag(argc, argv, "--no-stream");
    (void)show_timestamps; // word timestamps always stored; stdout exporter uses them when present

    // ── Print startup banner ───────────────────────────────────────────────
    std::fprintf(stderr,
        "\033[1;36m"
        "╔══════════════════════════════════════════╗\n"
        "║        VaultASR v%s                  ║\n"
        "║  Local, private speech-to-text pipeline  ║\n"
        "╚══════════════════════════════════════════╝"
        "\033[0m\n\n",
        VAULTASR_VERSION);

    std::fprintf(stderr, "  Model   : %s\n",  cfg.transcriber_config.model_path.c_str());
    std::fprintf(stderr, "  Language: %s\n",  cfg.transcriber_config.language.c_str());
    std::fprintf(stderr, "  GPU     : %s\n",  cfg.transcriber_config.use_gpu ? "Metal" : "CPU");
    std::fprintf(stderr, "  Denoise : %s\n",  cfg.denoise ? "RNNoise" : "off");
    std::fprintf(stderr, "  Diarize : %s\n",  cfg.skip_diarization ? "off (single speaker)" : "on");
    std::fprintf(stderr, "  Files   : %zu\n", audio_files.size());
    std::fprintf(stderr, "\n");

    // ── Run ────────────────────────────────────────────────────────────────
    try {
        if (audio_files.size() == 1) {
            cfg.audio_path = audio_files[0];
        }

        Pipeline pipeline(cfg);

        if (audio_files.size() == 1) {
            pipeline.run();
        } else {
            pipeline.run_batch(audio_files);
        }


    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n\033[1;31m[FATAL] %s\033[0m\n", e.what());
        return 1;
    }

    return 0;
}
