#include "audio_utils.hpp"
#include <iostream>
#include <stdexcept>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswresample/swresample.h>
#include <libavutil/opt.h>
#include <libavutil/channel_layout.h>
}

std::vector<float> load_audio_16k_mono(const std::string& path, bool trace) {
    if (trace) std::cout << "[EXECUTION TRACE] Opening audio via FFmpeg: " << path << "\n";
    AVFormatContext* format_context = nullptr;
    if (avformat_open_input(&format_context, path.c_str(), nullptr, nullptr) != 0) {
        throw std::runtime_error("Could not open audio file");
    }

    if (avformat_find_stream_info(format_context, nullptr) < 0) {
        throw std::runtime_error("Could not retrieve stream info");
    }

    int audio_stream_idx = -1;
    const AVCodec* decoder = nullptr;
    for (unsigned int i = 0; i < format_context->nb_streams; i++) {
        AVCodecParameters* codec_params = format_context->streams[i]->codecpar;
        if (codec_params->codec_type == AVMEDIA_TYPE_AUDIO) {
            audio_stream_idx = i;
            decoder = avcodec_find_decoder(codec_params->codec_id);
            break;
        }
    }

    if (audio_stream_idx == -1 || !decoder) {
        throw std::runtime_error("No audio stream or decoder found");
    }

    AVCodecContext* codec_context = avcodec_alloc_context3(decoder);
    avcodec_parameters_to_context(codec_context, format_context->streams[audio_stream_idx]->codecpar);

    if (avcodec_open2(codec_context, decoder, nullptr) < 0) {
        throw std::runtime_error("Could not open codec");
    }

    SwrContext* swr_context = nullptr;
    int ret = swr_alloc_set_opts2(&swr_context,
                                  &codec_context->ch_layout, AV_SAMPLE_FMT_FLT, 16000,
                                  &codec_context->ch_layout, codec_context->sample_fmt, codec_context->sample_rate,
                                  0, nullptr);
    if (ret < 0 || !swr_context) {
        throw std::runtime_error("Could not allocate swr context");
    }

    // Force Mono output layout
    AVChannelLayout out_ch_layout;
    av_channel_layout_default(&out_ch_layout, 1);
    av_opt_set_chlayout(swr_context, "out_chlayout", &out_ch_layout, 0);

    if (swr_init(swr_context) < 0) {
        throw std::runtime_error("Could not initialize swr context");
    }

    AVPacket* packet = av_packet_alloc();
    AVFrame* frame = av_frame_alloc();
    std::vector<float> audio_data;

    while (av_read_frame(format_context, packet) >= 0) {
        if (packet->stream_index == audio_stream_idx) {
            avcodec_send_packet(codec_context, packet);
            while (avcodec_receive_frame(codec_context, frame) == 0) {
                int out_samples = av_rescale_rnd(swr_get_delay(swr_context, codec_context->sample_rate) + frame->nb_samples,
                                                 16000, codec_context->sample_rate, AV_ROUND_UP);
                
                std::vector<float> buffer(out_samples);
                uint8_t* out_data[1] = { reinterpret_cast<uint8_t*>(buffer.data()) };
                int out_ret = swr_convert(swr_context, out_data, out_samples,
                                          const_cast<const uint8_t**>(frame->data), frame->nb_samples);
                
                if (out_ret > 0) {
                    audio_data.insert(audio_data.end(), buffer.begin(), buffer.begin() + out_ret);
                }
            }
        }
        av_packet_unref(packet);
    }

    av_frame_free(&frame);
    av_packet_free(&packet);
    swr_free(&swr_context);
    avcodec_free_context(&codec_context);
    avformat_close_input(&format_context);

    if (trace) std::cout << "[EXECUTION TRACE] Loaded " << audio_data.size() << " samples (" 
                         << audio_data.size() / 16000.0 << " seconds).\n";

    return audio_data;
}
