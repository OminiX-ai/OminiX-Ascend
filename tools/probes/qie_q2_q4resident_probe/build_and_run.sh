#!/usr/bin/env bash
# QIE-Q2 Q4-resident Gate 0 probe — build + run on ac03.
# Reproduction:  bash build_and_run.sh
# Exit codes:
#   0 = GREEN  (op works, cos_sim > 0.99, reasonable perf)
#   1 = YELLOW (op works but numerics or perf off — escalate to PM)
#   2 = RED    (op rejects W4 — re-open A16W8 / vendor ask)
set -euo pipefail

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:${LD_LIBRARY_PATH:-}
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

DIR=$(cd "$(dirname "$0")" && pwd)
cd "$DIR"

# Cohabit with Agent A4C-PHASE-2-PLUS — contract mandates the ac03 HBM lock.
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
    echo "[probe] HBM lock present at $LOCK — waiting..."
    while [ -e "$LOCK" ]; do sleep 5; done
fi
echo "qie_q2_q4resident_probe $$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

g++ -std=c++17 -O2 -o test_qie_q4resident_probe test_qie_q4resident_probe.cpp \
    -I$ASCEND_TOOLKIT_HOME/aarch64-linux/include \
    -L$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64 \
    -lascendcl -lopapi -lnnopbase -ldl

echo "--- build OK ---"
./test_qie_q4resident_probe
rc=$?
echo "--- probe exit rc=$rc ---"
exit $rc
