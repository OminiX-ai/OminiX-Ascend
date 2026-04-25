#!/usr/bin/env bash
# Q2.4.5.4i Step 2 — block-0 CPU reference forward (CPU backend, F32) for
# native-vs-CPU discrimination. Build + run on ac03.
#
# Inputs (must exist before run):
#   $QIE_BLOCK0_INPUTS_DIR/{00_img.f32, 00_txt.f32, 00_t_emb.f32}
# Outputs:
#   $QIE_BLOCK0_OUTPUTS_DIR/{cpu_24_img_resid2.f32, cpu_24_txt_resid2.f32}
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/../../.." && pwd)

cd "$DIR"

# Build dir that produced ggml libs. The CPU reference does NOT need CANN —
# only ggml + ggml-cpu — but we link against the same ggml build dir as the
# native engine probes for consistency.
GGML_BUILD=${GGML_BUILD:-$REPO/build-w1}
if [ ! -d "$GGML_BUILD" ]; then
  GGML_BUILD="$REPO/build-85"
fi
if [ ! -d "$GGML_BUILD" ]; then
  echo "[block0_cpu] missing ggml build at $GGML_BUILD — set GGML_BUILD"
  exit 1
fi

# ominix_diffusion src dir compiled IN-LINE because we need
# the ModelLoader symbols (model.cpp, name_conversion.cpp,
# tokenize_util.cpp, util.cpp, version.cpp, upscaler.cpp,
# stable-diffusion.cpp). For a probe-scale harness we just compile
# what's needed and link the ggml libs.
OMI_SRC_DIR="$REPO/tools/ominix_diffusion/src"

g++ -std=c++17 -O2 -g \
    -I"$REPO" \
    -I"$REPO/tools" \
    -I"$REPO/ggml/include" \
    -I"$REPO/include" \
    -I"$REPO/tools/ominix_diffusion" \
    -I"$REPO/tools/ominix_diffusion/src" \
    -I"$REPO/tools/ominix_diffusion/src/thirdparty" \
    -I"$REPO/tools/ominix_diffusion/common" \
    -DSD_USE_CUBLAS=0 \
    -DGGML_USE_CPU=1 \
    -DGGML_MAX_NAME=128 \
    -o test_qie_block0_cpu_reference \
    test_qie_block0_cpu_reference.cpp \
    "$OMI_SRC_DIR/model.cpp" \
    "$OMI_SRC_DIR/name_conversion.cpp" \
    "$OMI_SRC_DIR/util.cpp" \
    "$OMI_SRC_DIR/tokenize_util.cpp" \
    "$OMI_SRC_DIR/thirdparty/zip.c" \
    -L"$GGML_BUILD/ggml/src" \
    -L"$GGML_BUILD/bin" \
    -lggml-base -lggml -lggml-cpu \
    -ldl -lpthread

echo "--- build OK ---"

export LD_LIBRARY_PATH="$GGML_BUILD/bin:${LD_LIBRARY_PATH:-}"

: "${QIE_BLOCK0_INPUTS_DIR:=/tmp/qie_block0_inputs}"
: "${QIE_BLOCK0_OUTPUTS_DIR:=/tmp/qie_block0_outputs}"
mkdir -p "$QIE_BLOCK0_OUTPUTS_DIR"
export QIE_BLOCK0_INPUTS_DIR QIE_BLOCK0_OUTPUTS_DIR

if [ ! -f "$QIE_BLOCK0_INPUTS_DIR/00_img.f32" ] \
   || [ ! -f "$QIE_BLOCK0_INPUTS_DIR/00_txt.f32" ] \
   || [ ! -f "$QIE_BLOCK0_INPUTS_DIR/00_t_emb.f32" ]; then
  echo "[block0_cpu] missing input dump under $QIE_BLOCK0_INPUTS_DIR"
  echo "             run the native dump probe first:"
  echo "             QIE_DUMP_BLOCK0_DIR=$QIE_BLOCK0_INPUTS_DIR \\"
  echo "               QIE_DEBUG_INTRA_BLOCK0=1 \\"
  echo "               bash tools/probes/qie_q45_real_denoise_smoke/build_and_run.sh"
  exit 1
fi

./test_qie_block0_cpu_reference
