# Qwen3-TTS Voice Clone Implementation

find_package(Threads REQUIRED)
find_package(OpenMP)

set(TARGET qwen_tts)
set(SRC_FILES
    main.cpp
    qwen_tts.cpp
    talker.cpp
    tts_transformer.cpp
    speaker_encoder.cpp
    speech_tokenizer_encoder.cpp
    speech_tokenizer_decoder.cpp
    model_defs.cpp
    stft.cpp
)

# Native CP CANN engine: optional, enabled on Ascend where CANN headers exist.
# We dlopen the CANN libs at runtime (see cp_cann_symbols.cpp) so nothing
# needs to be linked — just need the toolkit headers + libdl.
option(QWEN_TTS_CP_CANN "Build native CP CANN engine (Ascend only)" OFF)
if(DEFINED ENV{ASCEND_TOOLKIT_HOME} AND EXISTS "$ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include/acl/acl.h")
    set(QWEN_TTS_CP_CANN ON CACHE BOOL "" FORCE)
endif()
if(QWEN_TTS_CP_CANN)
    list(APPEND SRC_FILES cp_cann_engine.cpp cp_cann_symbols.cpp
                          talker_cann_engine.cpp)
    message(STATUS "qwen_tts: CP CANN engine ENABLED (+ native Talker)")
endif()

# -----------------------------------------------------------------------------
# Path C W4.1: AscendC custom kernels.
#
# The fused_attn_sublayer kernel collapses the CP transformer's attention
# sublayer (12 aclnn ops) into a single AscendC dispatch. Built into a
# STATIC library via the ascendc_library() cmake macro from the CANN
# toolkit. The library is named `qwen_tts_ascendc_kernels`; downstream
# targets (qwen_tts executable, qwen_tts_api shared lib, and the offline
# 4.1.2 diff harness) link it as an internal dep.
#
# Gated on QWEN_TTS_CP_CANN — the kernels are only useful when the CP
# CANN engine is also built. Default off when ASCEND_TOOLKIT_HOME is
# absent (macOS / non-Ascend hosts); default on when we're on an Ascend
# host. Source file lives at tools/qwen_tts/ascendc/.
# -----------------------------------------------------------------------------
option(QWEN_TTS_ASCENDC "Build AscendC custom kernels (W4.1)" OFF)
if(QWEN_TTS_CP_CANN
   AND EXISTS "$ENV{ASCEND_TOOLKIT_HOME}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake"
   AND EXISTS "$ENV{ASCEND_TOOLKIT_HOME}/compiler/ccec_compiler/bin/ccec")
    set(QWEN_TTS_ASCENDC ON CACHE BOOL "" FORCE)
endif()
if(QWEN_TTS_ASCENDC)
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_TOOLKIT_HOME})
    set(SOC_VERSION Ascend910B4)
    set(RUN_MODE npu)
    include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
    ascendc_library(qwen_tts_ascendc_kernels STATIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ascendc/fused_attn_sublayer.cpp)
    message(STATUS "qwen_tts: AscendC kernels ENABLED (W4.1 fused_attn_sublayer)")
endif()

# llama.cpp fallback Talker path: optional, default OFF. When OFF, the
# `--llama_fallback` runtime flag prints a clear error and exits; the
# native TalkerCannEngine is the only backbone path. When ON, llama.cpp
# is linked and `--llama_fallback` runs the original hybrid code path.
option(QWEN_TTS_LLAMA "Link llama.cpp fallback into qwen_tts" OFF)
if(QWEN_TTS_LLAMA)
    message(STATUS "qwen_tts: llama.cpp fallback ENABLED (QWEN_TTS_LLAMA=ON)")
else()
    message(STATUS "qwen_tts: llama.cpp fallback DISABLED "
                   "(native-only; enable with -DQWEN_TTS_LLAMA=ON)")
endif()

# kissfft library for STFT computation
set(KISSFFT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../vendor/kissfft)
set(KISSFFT_SRC
    ${KISSFFT_DIR}/kiss_fft.c
    ${KISSFFT_DIR}/kiss_fftr.c
)
list(APPEND SRC_FILES ${KISSFFT_SRC})

add_executable(${TARGET} ${SRC_FILES})

# Base deps. llama is only linked when QWEN_TTS_LLAMA=ON. The `llama.h`
# header is still included by talker.h for type declarations (llama_batch
# field, etc.) so we keep the include path visible in both configs; the
# actual library is skipped in the OFF case.
target_link_libraries     (${TARGET} PUBLIC ggml qwen_common)
target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/include)
if(QWEN_TTS_LLAMA)
    target_link_libraries     (${TARGET} PUBLIC llama)
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_LLAMA=1)
endif()

# Native CP CANN engine: implemented via dlopen/dlsym at runtime to sidestep
# CANN stub-lib circular deps. We only need libdl + the toolkit headers.
target_link_libraries     (${TARGET} PRIVATE Threads::Threads)
if(QWEN_TTS_CP_CANN)
    target_link_libraries(${TARGET} PRIVATE ${CMAKE_DL_LIBS})
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_include_directories(${TARGET} PRIVATE
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)
endif()
if(QWEN_TTS_ASCENDC)
    # Static link with the AscendC library + the auto-generated host stub
    # header (exposes `aclrtlaunch_fused_attn_sublayer`). The runtime
    # static libs come in via the ascendc.cmake's add_link_deps.
    target_link_libraries(${TARGET} PRIVATE qwen_tts_ascendc_kernels)
    target_include_directories(${TARGET} PRIVATE
        ${CMAKE_BINARY_DIR}/auto_gen/qwen_tts_ascendc_kernels
        ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
    target_compile_definitions(${TARGET} PRIVATE QWEN_TTS_HAS_ASCENDC=1)
endif()
target_include_directories(${TARGET} PUBLIC  .)
target_include_directories(${TARGET} PRIVATE ../.. ../../vendor ${KISSFFT_DIR}
    $ENV{ASCEND_TOOLKIT_HOME}/include)
target_compile_features   (${TARGET} PRIVATE cxx_std_17)

# Enable NEON F16 intrinsics for ARM
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

# OpenMP for parallel CP matvec
if(OpenMP_CXX_FOUND)
    target_link_libraries(${TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

if (BUILD_SHARED_LIBS)
    set_target_properties     (${TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(${TARGET} PRIVATE LLAMA_BUILD)
    target_compile_definitions(${TARGET} PUBLIC  LLAMA_SHARED)
endif()

set_target_properties(${TARGET} PROPERTIES OUTPUT_NAME ${TARGET})
if(LLAMA_TOOLS_INSTALL)
    install(TARGETS ${TARGET} RUNTIME)
endif()
# `common` is llama.cpp's utility lib (PUBLIC-links libllama). Only needed
# when we're actually on the llama.cpp path. qwen_tts sources don't use it.
if(QWEN_TTS_LLAMA)
    target_link_libraries  (${TARGET} PRIVATE common)
endif()
target_link_libraries  (${TARGET} PRIVATE Threads::Threads)
target_compile_features(${TARGET} PRIVATE cxx_std_17)

# =============================================================================
# qwen_tts_api — Shared library exposing the C ABI in qwen_tts_api.h.
#
# Built for OminiX-API FFI (Ascend API Bridge Contract §5 B1). Links the SAME
# internals as the qwen_tts executable (BPETokenizer, TalkerLLM, SpeechTokenizer
# encoder/decoder, SpeakerEncoder, ggml, ggml-cann, kissfft) but exposes ONLY
# the 14 qwen_tts_* C symbols via a linker version script to prevent ggml /
# ggml-cann symbol pollution in downstream binaries (risk register §7).
#
# Does NOT include main.cpp / qwen_tts.cpp (CLI entry points) or the llama.cpp
# fallback path — the library is a pure compute surface. The native CP-CANN
# Talker engine (cp_cann_engine.cpp, cp_cann_symbols.cpp, talker_cann_engine.cpp)
# is included when QWEN_TTS_CP_CANN is ON so Ascend clients get the same
# native-Talker path the executable uses.
# =============================================================================
set(API_TARGET qwen_tts_api)
set(API_SRC_FILES
    qwen_tts_api.cpp
    qwen_tts.cpp                     # B5: QwenTTS::generate* used by qwen_tts_synthesize
    talker.cpp
    tts_transformer.cpp
    speaker_encoder.cpp
    speech_tokenizer_encoder.cpp
    speech_tokenizer_decoder.cpp
    model_defs.cpp
    stft.cpp
    ${KISSFFT_SRC}
)
if(QWEN_TTS_CP_CANN)
    list(APPEND API_SRC_FILES
        cp_cann_engine.cpp
        cp_cann_symbols.cpp
        talker_cann_engine.cpp
    )
endif()

add_library(${API_TARGET} SHARED ${API_SRC_FILES})

# qwen_common is a STATIC dep and is linked into a SHARED lib here, so its
# object files must be PIC. qwen_common's own CMakeLists doesn't set PIC
# (its only other consumer is the qwen_tts executable, which doesn't need
# it). We flip the flag on the existing target from here — this touches
# build flags, not source code in tools/qwen_common/.
set_property(TARGET qwen_common PROPERTY POSITION_INDEPENDENT_CODE ON)

target_link_libraries     (${API_TARGET} PUBLIC ggml qwen_common)
target_link_libraries     (${API_TARGET} PRIVATE Threads::Threads)
target_include_directories(${API_TARGET} PUBLIC  .)
target_include_directories(${API_TARGET} PRIVATE
    ${CMAKE_SOURCE_DIR}/include   # llama.h visible for talker.h (fallback types)
    ../.. ../../vendor ${KISSFFT_DIR}
    $ENV{ASCEND_TOOLKIT_HOME}/include)
target_compile_features   (${API_TARGET} PRIVATE cxx_std_17)
set_target_properties     (${API_TARGET} PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(QWEN_TTS_LLAMA)
    target_link_libraries     (${API_TARGET} PUBLIC llama)
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_LLAMA=1)
endif()

if(QWEN_TTS_CP_CANN)
    target_link_libraries(${API_TARGET} PRIVATE ${CMAKE_DL_LIBS})
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_include_directories(${API_TARGET} PRIVATE
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)
endif()
if(QWEN_TTS_ASCENDC)
    target_link_libraries(${API_TARGET} PRIVATE qwen_tts_ascendc_kernels)
    target_include_directories(${API_TARGET} PRIVATE
        ${CMAKE_BINARY_DIR}/auto_gen/qwen_tts_ascendc_kernels
        ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
    target_compile_definitions(${API_TARGET} PRIVATE QWEN_TTS_HAS_ASCENDC=1)
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${API_TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(${API_TARGET} PRIVATE OpenMP::OpenMP_CXX)
endif()

# SONAME versioning: libqwen_tts_api.so.1.1.0 with .so.1 and .so symlinks.
# Bumped MINOR (1.0.0 -> 1.1.0) for B5's additive qwen_tts_synthesize +
# qwen_tts_pcm_free exports. SOVERSION stays at 1 — existing consumers
# linked against the 14-symbol ABI resolve cleanly against this build.
set_target_properties(${API_TARGET} PROPERTIES
    OUTPUT_NAME qwen_tts_api
    VERSION     1.1.0
    SOVERSION   1
)

# Hide every symbol that isn't in the version script. Prevents ggml/ggml-cann
# internals from leaking into consumer binaries.
set(API_VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/qwen_tts_api.version")
if(UNIX AND NOT APPLE AND EXISTS "${API_VERSION_SCRIPT}")
    target_link_options(${API_TARGET} PRIVATE
        "-Wl,--version-script=${API_VERSION_SCRIPT}"
        "-Wl,--no-undefined")
    set_property(TARGET ${API_TARGET} APPEND PROPERTY
        LINK_DEPENDS "${API_VERSION_SCRIPT}")
    # NOTE: We deliberately do NOT use -fvisibility=hidden at compile time.
    # The version script's `local: *;` already suppresses every non-API symbol
    # at link time, including symbols from static-linked qwen_common / kissfft.
    # Adding -fvisibility=hidden would hide the extern "C" qwen_tts_* symbols
    # as well (they have no __attribute__((visibility("default"))) annotation),
    # leaving nothing for the version script to export.
endif()

# Install library to lib/, public header to include/.
install(TARGETS ${API_TARGET}
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin)
install(FILES qwen_tts_api.h DESTINATION include)

# Test: Code Predictor verification
set(TEST_CP_SRC
    test_code_predictor.cpp
    talker.cpp
    model_defs.cpp
)
add_executable(test_code_predictor ${TEST_CP_SRC})
target_link_libraries     (test_code_predictor PUBLIC ggml llama qwen_common)
target_link_libraries     (test_code_predictor PRIVATE Threads::Threads common)
target_include_directories(test_code_predictor PUBLIC  .)
target_include_directories(test_code_predictor PRIVATE ../.. ../../vendor)
target_compile_definitions(test_code_predictor PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_code_predictor PRIVATE cxx_std_17)

# Test: Talker LLM integration
set(TEST_TALKER_SRC
    test_talker.cpp
    talker.cpp
    model_defs.cpp
)
add_executable(test_talker ${TEST_TALKER_SRC})
target_link_libraries     (test_talker PUBLIC ggml llama qwen_common)
target_link_libraries     (test_talker PRIVATE Threads::Threads common)
target_include_directories(test_talker PUBLIC  .)
target_include_directories(test_talker PRIVATE ../.. ../../vendor)
target_compile_definitions(test_talker PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_talker PRIVATE cxx_std_17)

# Test: Speech Tokenizer Encoder
set(TEST_ENCODER_SRC
    test_encoder.cpp
    speech_tokenizer_encoder.cpp
    model_defs.cpp
)
add_executable(test_encoder ${TEST_ENCODER_SRC})
target_link_libraries     (test_encoder PUBLIC ggml llama qwen_common)
target_link_libraries     (test_encoder PRIVATE Threads::Threads common)
target_include_directories(test_encoder PUBLIC  .)
target_include_directories(test_encoder PRIVATE ../.. ../../vendor)
target_compile_features   (test_encoder PRIVATE cxx_std_17)

# Test: CP autoregressive flow
set(TEST_CP_FLOW_SRC
    test_cp_flow.cpp
    talker.cpp
    model_defs.cpp
)
add_executable(test_cp_flow ${TEST_CP_FLOW_SRC})
target_link_libraries     (test_cp_flow PUBLIC ggml llama qwen_common)
target_link_libraries     (test_cp_flow PRIVATE Threads::Threads common)
target_include_directories(test_cp_flow PUBLIC  .)
target_include_directories(test_cp_flow PRIVATE ../.. ../../vendor)
target_compile_definitions(test_cp_flow PRIVATE QWEN_TTS_LLAMA=1)
target_compile_features   (test_cp_flow PRIVATE cxx_std_17)

# Test: BPE Tokenizer
add_executable(test_tokenizer test_tokenizer.cpp)
target_link_libraries     (test_tokenizer PRIVATE Threads::Threads qwen_common)
target_include_directories(test_tokenizer PRIVATE . ../.. ../../vendor)
target_compile_features   (test_tokenizer PRIVATE cxx_std_17)

# Test: native TalkerCannEngine (Ascend only). Only built when the CANN
# engine source is present — it depends on cp_cann_symbols + the aclnn
# headers. The test pulls in the engine and its symbol loader directly; it
# does NOT link talker.cpp (no need for the llama.cpp Talker path).
if(QWEN_TTS_CP_CANN)
    add_executable(test_talker_native
        test_talker_native.cpp
        talker_cann_engine.cpp
        cp_cann_symbols.cpp
        model_defs.cpp
    )
    target_link_libraries     (test_talker_native PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_talker_native PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_talker_native PUBLIC  .)
    target_include_directories(test_talker_native PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_talker_native PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_talker_native PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_talker_native PRIVATE -march=armv8.2-a+fp16)
    endif()

    # Diagnostic harness: batched vs iterative forward_prefill cos-sim on
    # synthetic scales + real-embedding-dump replay (see Track A §8 entry
    # in NATIVE_TTS_CONTRACT.md for the use case).
    add_executable(test_prefill_diff
        test_prefill_diff.cpp
        talker_cann_engine.cpp
        cp_cann_symbols.cpp
        model_defs.cpp
    )
    target_link_libraries     (test_prefill_diff PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_prefill_diff PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_prefill_diff PUBLIC  .)
    target_include_directories(test_prefill_diff PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_prefill_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_prefill_diff PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_prefill_diff PRIVATE -march=armv8.2-a+fp16)
    endif()

    # W4.1.2 offline-diff harness: dispatch the fused_attn_sublayer AscendC
    # kernel on a synthetic fixture and compare its output against the same
    # math run in host-F32. Gate: max-abs-diff ≤ 1e-2.
    if(QWEN_TTS_ASCENDC)
        add_executable(test_fused_attn_diff
            ascendc/test_fused_attn_diff.cpp
            cp_cann_symbols.cpp
        )
        target_link_libraries     (test_fused_attn_diff PUBLIC ggml llama qwen_common)
        target_link_libraries     (test_fused_attn_diff PRIVATE Threads::Threads common
                                                                 ${CMAKE_DL_LIBS}
                                                                 qwen_tts_ascendc_kernels)
        target_include_directories(test_fused_attn_diff PUBLIC  .)
        target_include_directories(test_fused_attn_diff PRIVATE ../.. ../../vendor
            $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
            $ENV{ASCEND_TOOLKIT_HOME}/include
            ${CMAKE_BINARY_DIR}/include/qwen_tts_ascendc_kernels)
        target_compile_definitions(test_fused_attn_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1
                                                                 QWEN_TTS_HAS_ASCENDC=1)
        target_compile_features   (test_fused_attn_diff PRIVATE cxx_std_17)
        # The ascendc_library() macro produces PIC object files; any target
        # that links it must also be PIC.
        set_target_properties     (test_fused_attn_diff PROPERTIES
                                    POSITION_INDEPENDENT_CODE ON)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(test_fused_attn_diff PRIVATE -march=armv8.2-a+fp16)
        endif()
    endif()

    # Diagnostic: per-row aclnnMm(W, X_col) vs batched aclnnMm(X, W^T strided).
    # Verifies the strided-weight view pattern used in forward_prefill matches
    # the contiguous-weight pattern used in forward_decode.
    add_executable(test_mm_diff
        test_mm_diff.cpp
        cp_cann_symbols.cpp
    )
    target_link_libraries     (test_mm_diff PUBLIC ggml llama qwen_common)
    target_link_libraries     (test_mm_diff PRIVATE Threads::Threads common
                                                           ${CMAKE_DL_LIBS})
    target_include_directories(test_mm_diff PUBLIC  .)
    target_include_directories(test_mm_diff PRIVATE ../.. ../../vendor
        $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
        $ENV{ASCEND_TOOLKIT_HOME}/include)
    target_compile_definitions(test_mm_diff PRIVATE QWEN_TTS_HAS_CP_CANN=1)
    target_compile_features   (test_mm_diff PRIVATE cxx_std_17)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_compile_options(test_mm_diff PRIVATE -march=armv8.2-a+fp16)
    endif()

    # -------------------------------------------------------------------------
    # G1 HARD GATE feasibility probe: test_aclgraph_smoke.
    #
    # Answers the single open question from G0: does
    # `aclmdlRICaptureTaskUpdateBegin/End` accept rebinding of RoPE slice
    # ptrs / KV-cache write offset / FIAv2 `seq_len` on an already-captured
    # `aclOpExecutor`? See ACLGRAPH_CONTRACT.md §5 G1 for the gate logic.
    #
    # Gated on QWEN_TTS_ACLGRAPH_SMOKE=ON (default OFF; only built when the
    # G1 agent is running). No behavioural impact on the main qwen_tts
    # binary — this is a standalone harness.
    # -------------------------------------------------------------------------
    option(QWEN_TTS_ACLGRAPH_SMOKE "Build G1 aclGraph capture/replay smoke test" OFF)
    if(QWEN_TTS_ACLGRAPH_SMOKE)
        add_executable(test_aclgraph_smoke
            test_aclgraph_smoke.cpp
            cp_cann_symbols.cpp
        )
        target_link_libraries     (test_aclgraph_smoke PUBLIC ggml llama qwen_common)
        target_link_libraries     (test_aclgraph_smoke PRIVATE Threads::Threads common
                                                                 ${CMAKE_DL_LIBS})
        target_include_directories(test_aclgraph_smoke PUBLIC  .)
        target_include_directories(test_aclgraph_smoke PRIVATE ../.. ../../vendor
            $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include
            $ENV{ASCEND_TOOLKIT_HOME}/include)
        target_compile_definitions(test_aclgraph_smoke PRIVATE QWEN_TTS_HAS_CP_CANN=1
                                                                TALKER_CP_ACLGRAPH_SMOKE=1)
        target_compile_features   (test_aclgraph_smoke PRIVATE cxx_std_17)
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            target_compile_options(test_aclgraph_smoke PRIVATE -march=armv8.2-a+fp16)
        endif()
        message(STATUS "qwen_tts: G1 aclGraph smoke harness ENABLED")
    endif()
endif()
