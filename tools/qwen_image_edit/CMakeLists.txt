# Qwen-Image-Edit-2511 native DiT engine (contract §Q2 scaffold)
#
# This directory holds the native ImageDiffusionEngine — the Ascend-NPU
# counterpart to `ominix_diffusion`'s ggml-cann path. We gate the whole
# target on the same condition tools/qwen_asr's native binary uses
# (ASCEND_TOOLKIT_HOME defined AND acl.h present) so the scaffold is a
# no-op on the Mac cross-check build.
#
# Phase 1 gate: this file MUST produce `qwen_image_edit_native` as a clean
# build on ac03 with no runtime crashes. Phase 2/3/4 fills the engine body
# without touching this file (except to add Q4 antiquant flag thread-through
# once Q1.1 lands).

find_package(Threads REQUIRED)

if(NOT (DEFINED ENV{ASCEND_TOOLKIT_HOME}
        AND EXISTS "$ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include/acl/acl.h"))
    message(STATUS "qwen_image_edit: native engine disabled "
                   "(ASCEND_TOOLKIT_HOME missing — non-Ascend host)")
    return()
endif()

set(NATIVE_TARGET qwen_image_edit_native)
set(NATIVE_SRC
    native/main_native.cpp
    native/image_diffusion_engine.cpp
    # Reused verbatim from qwen_tts for CANN symbol dlopen/dlsym.
    ../qwen_tts/cp_cann_symbols.cpp
)

add_executable(${NATIVE_TARGET} ${NATIVE_SRC})

target_link_libraries(${NATIVE_TARGET} PUBLIC ggml qwen_common)
target_link_libraries(${NATIVE_TARGET} PRIVATE
    Threads::Threads
    ${CMAKE_DL_LIBS})

target_include_directories(${NATIVE_TARGET} PUBLIC  .)
target_include_directories(${NATIVE_TARGET} PRIVATE
    ../..
    ../../vendor
    $ENV{ASCEND_TOOLKIT_HOME}/include
    $ENV{ASCEND_TOOLKIT_HOME}/aarch64-linux/include)

target_compile_features  (${NATIVE_TARGET} PRIVATE cxx_std_17)
target_compile_definitions(${NATIVE_TARGET} PRIVATE QWEN_TTS_HAS_CP_CANN=1)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    target_compile_options(${NATIVE_TARGET} PRIVATE -march=armv8.2-a+fp16)
endif()

if(BUILD_SHARED_LIBS)
    set_target_properties(${NATIVE_TARGET}
        PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

install(TARGETS ${NATIVE_TARGET} RUNTIME)
message(STATUS "qwen_image_edit: native engine ENABLED "
               "(${NATIVE_TARGET}) — Phase 1 scaffold")
