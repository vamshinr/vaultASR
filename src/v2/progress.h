#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <string>

namespace vaultasr {

// ─── Terminal progress bar with streaming transcript preview ───────────────
//
// Renders to stderr so stdout stays clean for piped transcript output.
// Two display modes:
//   1. Progress bar:  [Transcribing] [████████░░░░] 65% | Seg 13/20 | 2:34 elapsed
//   2. Stream line:   [Speaker 1] "And then we decided to go with the..."
//
class ProgressBar {
public:
    ProgressBar() : enabled_(true), last_render_{} {}

    void set_enabled(bool enabled) { enabled_ = enabled; }

    // Update the progress bar (throttled to ~15fps to avoid flicker)
    void update(const std::string& stage, float progress,
                const std::string& detail = "") {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        auto now = std::chrono::steady_clock::now();
        auto dt = std::chrono::duration<double>(now - last_render_).count();
        if (dt < 0.066 && progress < 1.0f) return;  // throttle to ~15fps
        last_render_ = now;

        // Calculate elapsed time
        if (progress <= 0.01f && stage != stage_) {
            stage_start_ = now;
        }
        stage_ = stage;

        auto elapsed = std::chrono::duration<double>(now - stage_start_).count();

        // Build progress bar
        const int bar_width = 30;
        int filled = static_cast<int>(progress * bar_width);
        filled = std::clamp(filled, 0, bar_width);

        char bar[64];
        for (int i = 0; i < bar_width; i++) {
            if (i < filled)       bar[i] = '#';  // solid block
            else                  bar[i] = '-';
        }
        bar[bar_width] = '\0';

        // Format elapsed time
        int mins = static_cast<int>(elapsed) / 60;
        int secs = static_cast<int>(elapsed) % 60;

        // Truncate detail to fit terminal
        std::string det = detail;
        if (det.size() > 50) det = det.substr(0, 47) + "...";

        // Clear line, write progress, return cursor
        std::fprintf(stderr, "\r\033[K\033[1;34m[%s]\033[0m [%s] %3.0f%% | %d:%02d",
                     stage.c_str(), bar, progress * 100.0f, mins, secs);

        if (!det.empty()) {
            std::fprintf(stderr, " | %s", det.c_str());
        }

        std::fflush(stderr);

        // If complete, move to next line
        if (progress >= 1.0f) {
            std::fprintf(stderr, " \033[32m Done\033[0m\n");
            std::fflush(stderr);
        }
    }

    // Print a streamed transcript line (clears progress bar, prints text, restores)
    void stream_text(int speaker_id, const std::string& text) {
        if (!enabled_) return;

        std::lock_guard<std::mutex> lock(mutex_);

        // Clear current progress line
        std::fprintf(stderr, "\r\033[K");

        // Print transcript line to stderr (user can also see it live)
        std::fprintf(stderr, "\033[33m[Speaker %d]\033[0m %s\n", speaker_id, text.c_str());
        std::fflush(stderr);

        // Progress bar will be redrawn on next update()
    }

    // Finish: ensure cursor is on a new line
    void finish() {
        if (!enabled_) return;
        std::lock_guard<std::mutex> lock(mutex_);
        std::fprintf(stderr, "\r\033[K");
        std::fflush(stderr);
    }

private:
    bool enabled_;
    std::mutex mutex_;
    std::string stage_;
    std::chrono::steady_clock::time_point last_render_;
    std::chrono::steady_clock::time_point stage_start_;
};

}  // namespace vaultasr
