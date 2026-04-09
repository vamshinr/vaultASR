#pragma once

#include <chrono>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <string>

namespace vaultasr {

// ─── Log levels ────────────────────────────────────────────────────────────
enum class LogLevel {
    TRACE = 0,   // every frame, every window, every embedding
    DEBUG = 1,   // segment boundaries, cluster assignments, model details
    INFO  = 2,   // stage transitions, file start/end, summaries
    WARN  = 3,   // fallbacks, quality issues
    ERROR = 4,   // failures
    NONE  = 5,   // suppress all
};

// ─── Global logger ─────────────────────────────────────────────────────────
class Logger {
public:
    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(LogLevel level) { level_ = level; }
    LogLevel level() const { return level_; }

    // All logging goes to stderr so stdout is clean for transcript output
    template <typename... Args>
    void log(LogLevel level, const char* file, int line, const char* fmt, Args... args) {
        if (level < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_).count();

        // Extract just the filename from path
        const char* basename = std::strrchr(file, '/');
        basename = basename ? basename + 1 : file;

        const char* tag = level_tag(level);
        const char* color = level_color(level);
        const char* reset = "\033[0m";

        // Format: [  1.234s] [INFO ] [audio_decoder.cpp:42] message
        std::fprintf(stderr, "%s[%7.3fs] [%-5s] [%s:%d] ",
                     color, elapsed, tag, basename, line);
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, "%s\n", reset);
        std::fflush(stderr);
    }

    // Log without file/line (for clean stage headers)
    template <typename... Args>
    void stage(const char* fmt, Args... args) {
        if (LogLevel::INFO < level_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration<double>(now - start_).count();

        std::fprintf(stderr, "\033[1;36m[%7.3fs] ══════ ", elapsed);
        std::fprintf(stderr, fmt, args...);
        std::fprintf(stderr, " ══════\033[0m\n");
        std::fflush(stderr);
    }

    void reset_timer() { start_ = std::chrono::steady_clock::now(); }

private:
    Logger() : level_(LogLevel::INFO), start_(std::chrono::steady_clock::now()) {}

    static const char* level_tag(LogLevel l) {
        switch (l) {
            case LogLevel::TRACE: return "TRACE";
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO ";
            case LogLevel::WARN:  return "WARN ";
            case LogLevel::ERROR: return "ERROR";
            default:              return "?????";
        }
    }

    static const char* level_color(LogLevel l) {
        switch (l) {
            case LogLevel::TRACE: return "\033[90m";        // gray
            case LogLevel::DEBUG: return "\033[37m";        // white
            case LogLevel::INFO:  return "\033[32m";        // green
            case LogLevel::WARN:  return "\033[33m";        // yellow
            case LogLevel::ERROR: return "\033[1;31m";      // bold red
            default:              return "";
        }
    }

    LogLevel level_;
    std::mutex mutex_;
    std::chrono::steady_clock::time_point start_;
};

// ─── Convenience macros ────────────────────────────────────────────────────
#define LOG_TRACE(fmt, ...) \
    vaultasr::Logger::instance().log(vaultasr::LogLevel::TRACE, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) \
    vaultasr::Logger::instance().log(vaultasr::LogLevel::DEBUG, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  \
    vaultasr::Logger::instance().log(vaultasr::LogLevel::INFO,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  \
    vaultasr::Logger::instance().log(vaultasr::LogLevel::WARN,  __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR(fmt, ...) \
    vaultasr::Logger::instance().log(vaultasr::LogLevel::ERROR, __FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_STAGE(fmt, ...) \
    vaultasr::Logger::instance().stage(fmt, ##__VA_ARGS__)

}  // namespace vaultasr
