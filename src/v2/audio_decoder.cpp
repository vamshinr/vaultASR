#include "audio_decoder.h"
#include "logger.h"
#include <stdexcept>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libswresample/swresample.h>
}

namespace vaultasr {

// ─── RAII wrappers for FFmpeg resources ────────────────────────────────────

struct FFmpegContext {
    AVFormatContext* format_ctx = nullptr;
    AVCodecContext*  codec_ctx  = nullptr;
    SwrContext*      swr_ctx    = nullptr;
    AVPacket*        packet     = nullptr;
    AVFrame*         frame      = nullptr;
    int              stream_idx = -1;

    ~FFmpegContext() {
        if (frame)      av_frame_free(&frame);
        if (packet)     av_packet_free(&packet);
        if (swr_ctx)    swr_free(&swr_ctx);
        if (codec_ctx)  avcodec_free_context(&codec_ctx);
        if (format_ctx) avformat_close_input(&format_ctx);
    }

    void open(const std::string& path) {
        // Open input
        if (avformat_open_input(&format_ctx, path.c_str(), nullptr, nullptr) != 0) {
            throw std::runtime_error("Could not open audio file: " + path);
        }

        if (avformat_find_stream_info(format_ctx, nullptr) < 0) {
            throw std::runtime_error("Could not retrieve stream info: " + path);
        }

        // Find audio stream
        const AVCodec* decoder = nullptr;
        for (unsigned int i = 0; i < format_ctx->nb_streams; i++) {
            if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_AUDIO) {
                stream_idx = static_cast<int>(i);
                decoder = avcodec_find_decoder(format_ctx->streams[i]->codecpar->codec_id);
                break;
            }
        }

        if (stream_idx < 0 || !decoder) {
            throw std::runtime_error("No audio stream found in: " + path);
        }

        LOG_DEBUG("Audio stream #%d, codec: %s", stream_idx, decoder->name);

        // Open codec
        codec_ctx = avcodec_alloc_context3(decoder);
        avcodec_parameters_to_context(codec_ctx, format_ctx->streams[stream_idx]->codecpar);

        if (avcodec_open2(codec_ctx, decoder, nullptr) < 0) {
            throw std::runtime_error("Could not open audio codec");
        }

        LOG_TRACE("Codec opened: %s, %d Hz, %d channels",
                  decoder->name, codec_ctx->sample_rate,
                  codec_ctx->ch_layout.nb_channels);

        // Setup resampler: source format → 16kHz mono float32
        int ret = swr_alloc_set_opts2(
            &swr_ctx,
            &codec_ctx->ch_layout, AV_SAMPLE_FMT_FLT, 16000,   // output
            &codec_ctx->ch_layout, codec_ctx->sample_fmt, codec_ctx->sample_rate,  // input
            0, nullptr);

        if (ret < 0 || !swr_ctx) {
            throw std::runtime_error("Could not allocate resampler");
        }

        // Force mono output
        AVChannelLayout out_layout;
        av_channel_layout_default(&out_layout, 1);
        av_opt_set_chlayout(swr_ctx, "out_chlayout", &out_layout, 0);

        if (swr_init(swr_ctx) < 0) {
            throw std::runtime_error("Could not initialize resampler");
        }

        packet = av_packet_alloc();
        frame  = av_frame_alloc();
    }
};

// ─── Probe ─────────────────────────────────────────────────────────────────

AudioMeta AudioDecoder::probe(const std::string& path) {
    LOG_DEBUG("Probing file: %s", path.c_str());

    AVFormatContext* fmt_ctx = nullptr;
    if (avformat_open_input(&fmt_ctx, path.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Could not open file: " + path);
    }

    if (avformat_find_stream_info(fmt_ctx, nullptr) < 0) {
        avformat_close_input(&fmt_ctx);
        throw std::runtime_error("Could not read stream info: " + path);
    }

    AudioMeta meta{};
    meta.duration_sec = fmt_ctx->duration > 0
        ? static_cast<double>(fmt_ctx->duration) / AV_TIME_BASE
        : 0.0;
    meta.format_name = fmt_ctx->iformat ? fmt_ctx->iformat->name : "unknown";

    for (unsigned int i = 0; i < fmt_ctx->nb_streams; i++) {
        AVCodecParameters* par = fmt_ctx->streams[i]->codecpar;
        if (par->codec_type == AVMEDIA_TYPE_AUDIO) {
            meta.sample_rate = par->sample_rate;
            meta.channels = par->ch_layout.nb_channels;
            const AVCodec* codec = avcodec_find_decoder(par->codec_id);
            meta.codec_name = codec ? codec->name : "unknown";

            // If container-level duration is zero, try stream-level
            if (meta.duration_sec <= 0.0 && fmt_ctx->streams[i]->duration > 0) {
                AVRational tb = fmt_ctx->streams[i]->time_base;
                meta.duration_sec = fmt_ctx->streams[i]->duration *
                                    av_q2d(tb);
            }
            break;
        }
    }

    avformat_close_input(&fmt_ctx);

    LOG_INFO("File: %s | %.1fs | %s | %d Hz | %d ch",
             path.c_str(), meta.duration_sec, meta.codec_name.c_str(),
             meta.sample_rate, meta.channels);

    return meta;
}

// ─── Full decode ───────────────────────────────────────────────────────────

std::vector<float> AudioDecoder::decode_full(const std::string& path) {
    LOG_DEBUG("Full decode: %s", path.c_str());

    std::vector<float> audio_data;

    decode_streaming(path, [&](const float* data, size_t n) {
        audio_data.insert(audio_data.end(), data, data + n);
    }, 16000 * 10);  // 10s internal chunks

    LOG_INFO("Decoded %zu samples (%.2f seconds)",
             audio_data.size(), audio_data.size() / 16000.0);

    return audio_data;
}

// ─── Streaming decode ──────────────────────────────────────────────────────

size_t AudioDecoder::decode_streaming(const std::string& path,
                                       ChunkCallback cb,
                                       size_t chunk_samples) {
    FFmpegContext ctx;
    ctx.open(path);

    std::vector<float> chunk_buffer;
    chunk_buffer.reserve(chunk_samples);
    size_t total_samples = 0;

    while (av_read_frame(ctx.format_ctx, ctx.packet) >= 0) {
        if (ctx.packet->stream_index != ctx.stream_idx) {
            av_packet_unref(ctx.packet);
            continue;
        }

        avcodec_send_packet(ctx.codec_ctx, ctx.packet);

        while (avcodec_receive_frame(ctx.codec_ctx, ctx.frame) == 0) {
            // Calculate output sample count
            int out_samples = av_rescale_rnd(
                swr_get_delay(ctx.swr_ctx, ctx.codec_ctx->sample_rate) + ctx.frame->nb_samples,
                16000, ctx.codec_ctx->sample_rate, AV_ROUND_UP);

            std::vector<float> tmp(out_samples);
            uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(tmp.data())};

            int converted = swr_convert(
                ctx.swr_ctx, out_data, out_samples,
                const_cast<const uint8_t**>(ctx.frame->data),
                ctx.frame->nb_samples);

            if (converted > 0) {
                for (int i = 0; i < converted; i++) {
                    chunk_buffer.push_back(tmp[i]);
                    if (chunk_buffer.size() >= chunk_samples) {
                        cb(chunk_buffer.data(), chunk_buffer.size());
                        total_samples += chunk_buffer.size();
                        chunk_buffer.clear();

                        LOG_TRACE("Decoded chunk: %zu total samples (%.1fs)",
                                  total_samples, total_samples / 16000.0);
                    }
                }
            }
        }

        av_packet_unref(ctx.packet);
    }

    // Flush resampler
    int flushed = 0;
    do {
        std::vector<float> tmp(1024);
        uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(tmp.data())};
        flushed = swr_convert(ctx.swr_ctx, out_data, 1024, nullptr, 0);
        if (flushed > 0) {
            for (int i = 0; i < flushed; i++) {
                chunk_buffer.push_back(tmp[i]);
            }
        }
    } while (flushed > 0);

    // Emit remaining samples
    if (!chunk_buffer.empty()) {
        cb(chunk_buffer.data(), chunk_buffer.size());
        total_samples += chunk_buffer.size();
    }

    LOG_DEBUG("Streaming decode complete: %zu total samples", total_samples);
    return total_samples;
}

}  // namespace vaultasr
