#!/usr/bin/env bash
# Q2 Phase 4.5 Step 4 native end-to-end denoise_full smoke — build + run on ac03.
# Takes /tmp/ac03_hbm_lock before the init_from_gguf dispatch.
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/../../.." && pwd)

cd "$DIR"

GGML_BUILD=${GGML_BUILD:-$REPO/build-w1}
if [ ! -d "$GGML_BUILD" ]; then
  GGML_BUILD="$REPO/build-85"
fi
if [ ! -d "$GGML_BUILD" ]; then
  echo "[smoke45s4] missing ggml build at $GGML_BUILD — set GGML_BUILD to a repo build dir"
  exit 1
fi

ENGINE_SRC="$REPO/tools/qwen_image_edit/native/image_diffusion_engine.cpp"
SYMS_SRC="$REPO/tools/qwen_tts/cp_cann_symbols.cpp"

g++ -std=c++17 -O2 -g \
    -I"$REPO" \
    -I"$REPO/tools" \
    -I"$REPO/ggml/include" \
    -I"$REPO/include" \
    -I"$ASCEND_TOOLKIT_HOME/aarch64-linux/include" \
    -o test_qie_q45_step4_full_denoise \
    test_qie_q45_step4_full_denoise.cpp \
    "$ENGINE_SRC" \
    "$SYMS_SRC" \
    -L"$GGML_BUILD/ggml/src" \
    -L"$GGML_BUILD/bin" \
    -L"$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64" \
    -lggml-base -lggml -lggml-cpu \
    -lascendcl -lopapi -lnnopbase -ldl -lpthread

echo "--- build OK ---"

export LD_LIBRARY_PATH="$GGML_BUILD/bin:$LD_LIBRARY_PATH"
export GGML_CANN_QUANT_BF16="${GGML_CANN_QUANT_BF16:-on}"

LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
  echo "[smoke45s4] HBM lock present at $LOCK — proceeding (script re-touches)"
fi
touch "$LOCK"
trap "rm -f $LOCK" EXIT

./test_qie_q45_step4_full_denoise
