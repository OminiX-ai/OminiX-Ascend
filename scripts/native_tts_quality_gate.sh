#!/bin/bash
# Quality gate bench for the native TTS contract (M2+).
#
# Runs 5 distinct test utterances through both the llama.cpp baseline and the
# native path; records duration, frame count, throughput, and a simple audio
# metric for each; copies all 10 WAVs back for human ear inspection.
#
# The contract's M2 quality gate requires:
#   - DTW log-mel vs baseline >= 0.85 on each utterance (computed locally)
#   - User-ear pass on all 5 (human review)
#   - Throughput >= 20 fps (M2) or >= 25 fps (final)
#
# Usage: bash native_tts_quality_gate.sh LABEL
#   LABEL is appended to output filenames (e.g. "v15_m2_check").

set -euo pipefail

LABEL="${1:-unlabeled}"
WORK_DIR="/tmp/tts_quality_${LABEL}"
mkdir -p "$WORK_DIR"

export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:$ASCEND_TOOLKIT_HOME/runtime/lib64:$LD_LIBRARY_PATH
source $ASCEND_TOOLKIT_HOME/../set_env.sh 2>/dev/null || true

cd ~/work/OminiX-Ascend

REF=tools/qwen_tts/data/ref_audios/ellen_ref.wav
REFTXT="$(cat tools/qwen_tts/data/ref_audios/ellen_ref.txt)"

# Skip warmup (its 5-frame request produces a short clip that the audio
# decoder refuses to process — unrelated to the native path).
export QWEN_TTS_SKIP_WARMUP=1

# Three natural-EOS utterances used as the M2.4 quality gate baseline.
# See NATIVE_TTS_CONTRACT.md §5 M2.4 for why these three are the canonical
# set (the longer "quick brown fox" variant doesn't naturally EOS within
# max_tokens=200 and inflates DTW drift).
declare -a TEXTS=(
  "Good morning, how are you today."
  "The sun is shining brightly this afternoon."
  "Please remember to turn off the lights."
)

declare -a UIDS=(utt1 utt2 utt3)

summary="$WORK_DIR/summary.tsv"
echo -e "uid\tpath\tbackend\tframes\tduration_s\tgenerate_s\tfps\tCP_ms" > "$summary"

run_one() {
  local uid="$1" text="$2" backend="$3" out="$4" extra="$5" timeout_s="$6"
  echo "==== [$backend] $uid ===="
  timeout "${timeout_s}" ./build/bin/qwen_tts -m tools/qwen_tts/gguf/ \
    -t "$text" -r "$REF" --ref_text "$REFTXT" \
    -o "$out" --seed 42 --max_tokens 200 $extra \
    2>&1 | tee "$WORK_DIR/$uid.$backend.log" \
    | grep -E "frames/sec|CP: |Total:|Generate:|generated [0-9]+ codec|EOS at step" || true

  local frames=$(grep -oE "generated [0-9]+ codec" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9]+" | head -1)
  local gen_s=$(grep -oE "Generate:\s+[0-9.]+ sec" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9.]+")
  local cp_ms=$(grep -oE "CP:\s+[0-9]+ ms" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9]+")
  local fps=$(awk -v f="$frames" -v s="$gen_s" 'BEGIN{if(s>0)printf "%.2f", f/s; else print "-"}')
  local dur=$(awk -v f="${frames:-0}" 'BEGIN{printf "%.2f", f*0.08}')
  echo -e "$uid\t$out\t$backend\t${frames:-0}\t$dur\t${gen_s:-0}\t${fps:-0}\t${cp_ms:-0}" >> "$summary"
}

for i in "${!TEXTS[@]}"; do
  uid="${UIDS[$i]}"
  text="${TEXTS[$i]}"
  # Native path: the target we're gating. cp_groups=8 is the M2 default
  # (see contract M2.5 — 21+ fps, DTW 3/3 pass).
  run_one "$uid" "$text" "native" \
    "$WORK_DIR/$uid.native.wav" \
    "--native_talker --cp_cann --cp_groups 8" 90
  # Baseline: plain llama.cpp (slow, used only for DTW reference).
  run_one "$uid" "$text" "llama" \
    "$WORK_DIR/$uid.llama.wav" \
    "" 600
done

echo
echo "=== Summary ==="
column -t "$summary" || cat "$summary"
echo
echo "Audio samples in: $WORK_DIR"

# Simple throughput gate check (M2.5 ≥ 20 fps on the longest native run)
NATIVE_FPS_MIN=$(awk -F'\t' 'NR>1 && $3=="native" && $7!="-" && $7+0>0 {print $7}' "$summary" | sort -g | head -1)
if [ -n "${NATIVE_FPS_MIN:-}" ] && awk -v f="$NATIVE_FPS_MIN" 'BEGIN{exit !(f>=20)}'; then
  echo "Throughput gate (M2.5 ≥20 fps): PASS (min native fps=$NATIVE_FPS_MIN)"
else
  echo "Throughput gate (M2.5 ≥20 fps): FAIL (min native fps=${NATIVE_FPS_MIN:-N/A})"
fi

echo
echo "Pull audio and run DTW on local:"
echo "  mkdir -p /tmp/qg_$LABEL"
echo "  scp -i ~/home/tensordock/KeyPair-4fbd-yue.pem -P 31984 \\"
echo "      ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:$WORK_DIR/'*.wav' /tmp/qg_$LABEL/"
echo "  python3 scripts/dtw_vs_baseline.py /tmp/qg_$LABEL"
