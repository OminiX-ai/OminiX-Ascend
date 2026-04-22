#!/usr/bin/env bash
# Q3 FIAv2 runtime probe — build + run on ac02.
# Reproduction:  bash build_and_run.sh
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

g++ -std=c++17 -O2 -o test_qie_fiav2_seq4352 test_qie_fiav2_seq4352.cpp \
    -I$ASCEND_TOOLKIT_HOME/aarch64-linux/include \
    -L$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64 \
    -lascendcl -lopapi -lnnopbase -ldl

echo "--- build OK ---"
./test_qie_fiav2_seq4352
