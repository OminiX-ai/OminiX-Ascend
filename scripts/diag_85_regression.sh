#!/bin/bash
# scripts/diag_85_regression.sh — Track I: CANN 8.3 vs 8.5 regression investigation
#
# Measures side-by-side timing breakdown (build_emb/prefill/LLM/CP/...) for the
# canonical long utterance on 910B4 ModelArts box, and probes common env-knob
# suspects (TASK_QUEUE_ENABLE, ASCEND_LAUNCH_BLOCKING, ACL_OP_INIT_MODE,
# ASCEND_CACHE_PATH, etc.) to localise the 27% 8.3->8.5 regression.
#
# Usage: bash scripts/diag_85_regression.sh [LABEL]
#   Produces /tmp/diag_85_${LABEL}/{run83,run85,run85_envtune}.log and a
#   side-by-side summary table in /tmp/diag_85_${LABEL}/summary.tsv.
set -euo pipefail

LABEL="${1:-$(date +%s)}"
WORK=/tmp/diag_85_${LABEL}
mkdir -p "$WORK"

cd ~/work/OminiX-Ascend

TEXT="Speech synthesis on neural processing units is a compelling application of modern deep learning."
REF=tools/qwen_tts/data/ref_audios/ellen_ref.wav
REFTXT="$(cat tools/qwen_tts/data/ref_audios/ellen_ref.txt)"

# Keep warmup skip so numbers are stable
export QWEN_TTS_SKIP_WARMUP=1
# ND baseline (per Track I scope: keep TALKER_NZ_WEIGHTS unset)
unset TALKER_NZ_WEIGHTS || true

run_bin() {
  local tag="$1"
  local bin="$2"
  local tkhome="$3"
  local extra_env="$4"
  local out="$WORK/${tag}.log"
  echo "==== [$tag] bin=$bin  toolkit=$tkhome  extra_env=${extra_env:-none} ===="
  (
    set -e
    export ASCEND_TOOLKIT_HOME="$tkhome"
    export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/runtime/lib64:${LD_LIBRARY_PATH:-}
    source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true
    if [ -n "$extra_env" ]; then eval "export $extra_env"; fi
    env | grep -E "^(ASCEND_TOOLKIT_HOME|TASK_QUEUE_ENABLE|ASCEND_LAUNCH_BLOCKING|ACL_OP_INIT_MODE|ASCEND_CACHE_PATH|ASCEND_WORK_PATH)=" || true
    timeout 180 "$bin" -m tools/qwen_tts/gguf/ \
      -t "$TEXT" -r "$REF" --ref_text "$REFTXT" \
      -o "$WORK/${tag}.wav" \
      --native_talker --cp_cann --cp_groups 8 \
      --seed 42 --greedy --max_tokens 200 2>&1
  ) > "$out" 2>&1 || echo "[diag] $tag exit non-zero (see $out)"
  # Extract timing table if present
  grep -E "talker\].*timing|build_emb|prefill|head:|sample:|CP:|EMB:|trailing:|LLM:|loop_sum|Total:|frames/sec|Generate:|generated [0-9]+ codec" "$out" | sed "s|^|[${tag}] |"
}

# ---------------------------------------------------------------
# (1) 8.3 binary — baseline. Binary already built at build/bin/qwen_tts per §8.
# ---------------------------------------------------------------
BIN83="./build/bin/qwen_tts"
TK83=/usr/local/Ascend/ascend-toolkit/latest
if [ ! -x "$BIN83" ]; then
  echo "[diag] building 8.3 binary"
  (cd build && ASCEND_TOOLKIT_HOME=$TK83 cmake --build . --target qwen_tts -j 4)
fi
run_bin run83_a  "$BIN83" "$TK83" ""
run_bin run83_b  "$BIN83" "$TK83" ""
run_bin run83_c  "$BIN83" "$TK83" ""

# ---------------------------------------------------------------
# (2) 8.5 binary — build if absent.
# ---------------------------------------------------------------
TK85="$HOME/Ascend/cann-8.5.0"
BIN85="./build-85/bin/qwen_tts"
if [ ! -x "$BIN85" ]; then
  echo "[diag] building 8.5 binary (build-85/)"
  rm -rf build-85
  mkdir build-85
  (cd build-85 && ASCEND_TOOLKIT_HOME=$TK85 cmake .. -DCMAKE_BUILD_TYPE=Release && ASCEND_TOOLKIT_HOME=$TK85 cmake --build . --target qwen_tts -j 4) 2>&1 | tail -40
fi
run_bin run85_a  "$BIN85" "$TK85" ""
run_bin run85_b  "$BIN85" "$TK85" ""
run_bin run85_c  "$BIN85" "$TK85" ""

# ---------------------------------------------------------------
# (3) 8.5 env-knob probes — try combinations of documented CANN env vars
#     that affect host dispatch, tiling caching, and stream flush.
# ---------------------------------------------------------------
run_bin run85_tqe2       "$BIN85" "$TK85" "TASK_QUEUE_ENABLE=2"
run_bin run85_tqe1       "$BIN85" "$TK85" "TASK_QUEUE_ENABLE=1"
run_bin run85_tqe0       "$BIN85" "$TK85" "TASK_QUEUE_ENABLE=0"
run_bin run85_noblock    "$BIN85" "$TK85" "ASCEND_LAUNCH_BLOCKING=0"
run_bin run85_opinit0    "$BIN85" "$TK85" "ACL_OP_INIT_MODE=0"
# Persistent kernel cache — point both runs at a fresh cache then rerun
CACHE=/tmp/diag_85_${LABEL}/cache85
mkdir -p "$CACHE"
run_bin run85_cache_cold "$BIN85" "$TK85" "ASCEND_CACHE_PATH=$CACHE ASCEND_WORK_PATH=$CACHE"
run_bin run85_cache_warm "$BIN85" "$TK85" "ASCEND_CACHE_PATH=$CACHE ASCEND_WORK_PATH=$CACHE"

# ---------------------------------------------------------------
# (4) Dmesg / ascend logs for driver-toolkit version-mismatch warnings.
# ---------------------------------------------------------------
echo "==== dmesg ascend tail ===="  > "$WORK/dmesg.log"
dmesg 2>/dev/null | grep -i ascend | tail -40 >> "$WORK/dmesg.log" || true
echo "==== ascend_seclog tail ====" >> "$WORK/dmesg.log"
ls /var/log/ascend_seclog/ 2>/dev/null | head >> "$WORK/dmesg.log" || true

# ---------------------------------------------------------------
# (5) Summary
# ---------------------------------------------------------------
SUM="$WORK/summary.tsv"
echo -e "tag\tfps\ttotal_ms\tbuild_emb_ms\tprefill_ms\tLLM_ms\tCP_ms\tEMB_ms\thead_ms\tsample_ms\ttrailing_ms\tloop_sum_ms" > "$SUM"
for L in "$WORK"/run*.log; do
  tag=$(basename "$L" .log)
  fps=$(grep -oE "[0-9.]+ frames/sec" "$L" | head -1 | awk '{print $1}')
  tot=$(grep -oE "TOTAL:\s+[0-9]+ ms" "$L" | head -1 | grep -oE "[0-9]+")
  be=$(grep  -oE "build_emb:\s+[0-9]+ ms" "$L" | head -1 | grep -oE "[0-9]+")
  pf=$(grep  -oE "prefill:\s+[0-9]+ ms"  "$L" | head -1 | grep -oE "[0-9]+")
  llm=$(grep -oE "LLM:\s+[0-9]+ ms"      "$L" | head -1 | grep -oE "[0-9]+")
  cp=$(grep  -oE "CP:\s+[0-9]+ ms"       "$L" | head -1 | grep -oE "[0-9]+")
  emb=$(grep -oE "EMB:\s+[0-9]+ ms"      "$L" | head -1 | grep -oE "[0-9]+")
  hd=$(grep  -oE "head:\s+[0-9]+ ms"     "$L" | head -1 | grep -oE "[0-9]+")
  sm=$(grep  -oE "sample:\s+[0-9]+ ms"   "$L" | head -1 | grep -oE "[0-9]+")
  tr=$(grep  -oE "trailing:\s+[0-9]+ ms" "$L" | head -1 | grep -oE "[0-9]+")
  ls=$(grep  -oE "loop_sum:\s+[0-9]+ ms" "$L" | head -1 | grep -oE "[0-9]+")
  echo -e "$tag\t${fps:-?}\t${tot:-?}\t${be:-?}\t${pf:-?}\t${llm:-?}\t${cp:-?}\t${emb:-?}\t${hd:-?}\t${sm:-?}\t${tr:-?}\t${ls:-?}" >> "$SUM"
done
echo
echo "===== SUMMARY ($SUM) ====="
cat "$SUM"
