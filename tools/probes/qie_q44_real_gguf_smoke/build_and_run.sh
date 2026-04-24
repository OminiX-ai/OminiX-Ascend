#!/usr/bin/env bash
# Q2 Phase 4.4 real-GGUF forward smoke — build + run on ac03.
# Takes /tmp/ac03_hbm_lock before the init_from_gguf dispatch.
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/../../.." && pwd)

cd "$DIR"

# Same build recipe as Phase 4.2/4.3 smoke. Engine links against ggml static
# libs from an existing repo build dir (engine TU #includes ggml.h/gguf.h for
# the init_from_gguf path).
GGML_BUILD=${GGML_BUILD:-$REPO/build-w1}
if [ ! -d "$GGML_BUILD" ]; then
  # Fallback to Phase 4.2 smoke's historical default.
  GGML_BUILD="$REPO/build-85"
fi
if [ ! -d "$GGML_BUILD" ]; then
  echo "[smoke44] missing ggml build at $GGML_BUILD — set GGML_BUILD to a repo build dir"
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
    -o test_qie_q44_real_gguf_smoke \
    test_qie_q44_real_gguf_smoke.cpp \
    "$ENGINE_SRC" \
    "$SYMS_SRC" \
    -L"$GGML_BUILD/ggml/src" \
    -L"$GGML_BUILD/bin" \
    -L"$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64" \
    -lggml-base -lggml -lggml-cpu \
    -lascendcl -lopapi -lnnopbase -ldl -lpthread

echo "--- build OK ---"

# Runtime ld path: pick up ggml libs from build dir.
export LD_LIBRARY_PATH="$GGML_BUILD/bin:$LD_LIBRARY_PATH"

# Ensure GGML_CANN_QUANT_BF16 default on (per Q2.1 smoke recipe). Caller may
# override via the invocation env.
export GGML_CANN_QUANT_BF16="${GGML_CANN_QUANT_BF16:-on}"

# HBM lock for NPU run. Peak ~17 GiB expected per Q2.1 projection, co-tenants
# MUST honor the lock before starting their own engine.
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
  echo "[smoke44] HBM lock present at $LOCK — proceeding (script re-touches)"
fi
touch "$LOCK"
trap "rm -f $LOCK" EXIT

./test_qie_q44_real_gguf_smoke
