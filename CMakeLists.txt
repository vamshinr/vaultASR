cmake_minimum_required(VERSION 3.16)
project(VaultASR VERSION 2.0.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_C_STANDARD 11)

# ─── Build options ────────────────────────────────────────────────────────
option(VAULTASR_USE_COREML  "Enable CoreML acceleration (macOS)" ON)
option(VAULTASR_USE_METAL   "Enable Metal GPU acceleration (macOS)" ON)

# ─── Default to Release ────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ─── Compiler flags ───────────────────────────────────────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3 -DNDEBUG)
endif()

# Apple Silicon: ensure native ARM64
if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    add_compile_options(-march=native)
endif()

# ─── Dependencies: system libraries ───────────────────────────────────────
find_package(PkgConfig REQUIRED)

pkg_check_modules(FFMPEG REQUIRED
    libavformat
    libavcodec
    libavutil
    libswresample)

find_package(SQLite3 REQUIRED)

# Ensure Homebrew lib dir is in linker search path (needed for pkg-config -l flags on macOS)
if(APPLE)
    link_directories(/opt/homebrew/lib /opt/homebrew/opt/ffmpeg/lib)
endif()


# ONNX Runtime — try pkg-config first, then Homebrew fallback
find_library(ONNXRUNTIME_LIB onnxruntime
    PATHS /opt/homebrew/lib /usr/local/lib
    REQUIRED)
find_path(ONNXRUNTIME_INCLUDE onnxruntime/onnxruntime_cxx_api.h
    PATHS /opt/homebrew/include /usr/local/include
    REQUIRED)

# ─── Apple frameworks ──────────────────────────────────────────────────────
if(APPLE)
    find_library(COREML_FW    CoreML    REQUIRED)
    find_library(FOUNDATION_FW Foundation REQUIRED)
    find_library(METAL_FW     Metal     REQUIRED)
    find_library(METALKIT_FW  MetalKit  REQUIRED)
    find_library(ACCELERATE_FW Accelerate REQUIRED)

    set(APPLE_FRAMEWORKS
        ${COREML_FW}
        ${FOUNDATION_FW}
        ${METAL_FW}
        ${METALKIT_FW}
        ${ACCELERATE_FW})
endif()

# ─── External submodules ───────────────────────────────────────────────────

# whisper.cpp
set(WHISPER_METAL ${VAULTASR_USE_METAL} CACHE BOOL "" FORCE)
set(GGML_METAL   ${VAULTASR_USE_METAL} CACHE BOOL "" FORCE)
set(WHISPER_BUILD_TESTS   OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(external/whisper.cpp)

# RNNoise — build from source
add_library(rnnoise STATIC
    external/rnnoise/src/celt_lpc.c
    external/rnnoise/src/denoise.c
    external/rnnoise/src/kiss_fft.c
    external/rnnoise/src/pitch.c
    external/rnnoise/src/rnn.c
    external/rnnoise/src/rnn_data.c
    external/rnnoise/src/rnn_reader.c)
target_include_directories(rnnoise PUBLIC external/rnnoise/include)
target_compile_options(rnnoise PRIVATE -w)  # suppress RNNoise warnings

# libxlsxwriter — build from source (glob to pick up all sources including rich_value_*)
file(GLOB XLSXWRITER_SRCS
    external/libxlsxwriter/src/*.c
    external/libxlsxwriter/third_party/minizip/ioapi.c
    external/libxlsxwriter/third_party/minizip/zip.c
    external/libxlsxwriter/third_party/tmpfileplus/tmpfileplus.c
    external/libxlsxwriter/third_party/md5/md5.c)
add_library(xlsxwriter STATIC ${XLSXWRITER_SRCS})
target_include_directories(xlsxwriter PUBLIC external/libxlsxwriter/include)
target_link_libraries(xlsxwriter PUBLIC z)  # minizip needs zlib (crc32, deflate)
target_compile_options(xlsxwriter PRIVATE -w)  # suppress warnings

# miniz — build from split .c sources (not header-only on this version)
add_library(miniz STATIC
    external/miniz/miniz.c
    external/miniz/miniz_tdef.c
    external/miniz/miniz_tinfl.c
    external/miniz/miniz_zip.c)
target_include_directories(miniz PUBLIC external/miniz)
target_compile_options(miniz PRIVATE -w)  # suppress miniz warnings

# ─── VaultASR core library ────────────────────────────────────────────────
set(VAULTASR_SOURCES
    src/v2/audio_decoder.cpp
    src/v2/denoiser.cpp
    src/v2/vad.cpp
    src/v2/kaldi_fbank.cpp
    src/v2/speaker_embedder.cpp
    src/v2/speaker_cluster.cpp
    src/v2/transcriber.cpp
    src/v2/post_processor.cpp
    src/v2/exporter.cpp
    src/v2/pipeline.cpp)

add_library(vaultasr_lib STATIC ${VAULTASR_SOURCES})

target_include_directories(vaultasr_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/src/v2
    ${CMAKE_SOURCE_DIR}/external/whisper.cpp/include
    ${CMAKE_SOURCE_DIR}/external/miniz
    ${ONNXRUNTIME_INCLUDE}
    ${FFMPEG_INCLUDE_DIRS})

target_link_libraries(vaultasr_lib PUBLIC
    ${FFMPEG_LIBRARIES}
    ${ONNXRUNTIME_LIB}
    whisper
    rnnoise
    xlsxwriter
    miniz
    SQLite::SQLite3)

if(APPLE)
    target_link_libraries(vaultasr_lib PUBLIC ${APPLE_FRAMEWORKS})
    target_compile_definitions(vaultasr_lib PUBLIC __APPLE__)
endif()

# ─── CLI executable ───────────────────────────────────────────────────────
add_executable(vaultasr src/v2/main.cpp)
target_link_libraries(vaultasr PRIVATE vaultasr_lib)

# ─── Tests ────────────────────────────────────────────────────────────────
enable_testing()

if(EXISTS ${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt)
    add_subdirectory(tests)
endif()

# ─── Install ──────────────────────────────────────────────────────────────
install(TARGETS vaultasr RUNTIME DESTINATION bin)
install(DIRECTORY models/ DESTINATION share/vaultasr/models)

# ─── Print build summary ──────────────────────────────────────────────────
message(STATUS "")
message(STATUS "┌──────────────────────────────────────────┐")
message(STATUS "│  VaultASR v${PROJECT_VERSION} Build Configuration     │")
message(STATUS "├──────────────────────────────────────────┤")
message(STATUS "│  Build type : ${CMAKE_BUILD_TYPE}")
message(STATUS "│  CoreML     : ${VAULTASR_USE_COREML}")
message(STATUS "│  Metal      : ${VAULTASR_USE_METAL}")
message(STATUS "│  ONNX RT    : ${ONNXRUNTIME_LIB}")
message(STATUS "│  FFmpeg     : ${FFMPEG_VERSION}")
message(STATUS "└──────────────────────────────────────────┘")
message(STATUS "")
