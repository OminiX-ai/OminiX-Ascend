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

REF=tools/qwen_tts/data/ref_audios/ellen_ref_24k.wav
REFTXT="Hello there, how are you today."
TALKER=tools/qwen_tts/gguf/qwen_tts_talker_llama_q8_0.gguf

# Five distinct test utterances — each probes a different speech characteristic.
declare -a TEXTS=(
  "This is a short test sentence for benchmarking the TTS pipeline speed."
  "The quick brown fox jumps over the lazy dog near the river bank."
  "Artificial intelligence is transforming how we think about computing and everyday life."
  "She sells seashells by the seashore; the shells she sells are surely seashells."
  "In a world of rapid change, patience, focus, and attention to detail still matter deeply."
)

declare -a UIDS=(short fox ai tongue patience)

summary="$WORK_DIR/summary.tsv"
echo -e "uid\tpath\tbackend\tframes\tduration_s\tgenerate_s\tfps\tCP_ms" > "$summary"

run_one() {
  local uid="$1" text="$2" backend="$3" out="$4" extra="$5"
  echo "==== [$backend] $uid ===="
  local start=$(date +%s.%N)
  ./build/bin/qwen_tts -m tools/qwen_tts/gguf \
    -t "$text" -r "$REF" --ref_text "$REFTXT" \
    --target_lang English --ref_lang English \
    --talker_model "$TALKER" \
    --n_gpu_layers 29 -n 8 -o "$out" -p --seed 42 $extra \
    2>&1 | tee "$WORK_DIR/$uid.$backend.log" \
    | grep -E "frames/sec|CP: |Total:| Generate:|generated [0-9]+ codec|EOS at step"

  # Parse key metrics from the log
  local frames=$(grep -oE "generated [0-9]+ codec" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9]+" | head -1)
  local gen_s=$(grep -oE "Generate:\s+[0-9.]+ sec" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9.]+")
  local cp_ms=$(grep -oE "CP:\s+[0-9]+ ms" "$WORK_DIR/$uid.$backend.log" | tail -1 | grep -oE "[0-9]+")
  local fps=$(awk -v f="$frames" -v s="$gen_s" 'BEGIN{if(s>0)printf "%.2f", f/s; else print "-"}')
  local dur=$(awk -v f="$frames" 'BEGIN{printf "%.2f", f*0.08}')
  echo -e "$uid\t$out\t$backend\t$frames\t$dur\t$gen_s\t$fps\t$cp_ms" >> "$summary"
}

for i in "${!TEXTS[@]}"; do
  uid="${UIDS[$i]}"
  text="${TEXTS[$i]}"
  # Baseline (llama.cpp CP)
  run_one "$uid" "$text" "llama" \
    "$WORK_DIR/$uid.llama.wav" \
    "--cp_model tools/qwen_tts/gguf/qwen_tts_cp_llama.gguf"
  # Native (CANN CP + whatever flags are default in the binary right now)
  run_one "$uid" "$text" "native" \
    "$WORK_DIR/$uid.native.wav" \
    "--cp_cann"
done

echo
echo "=== Summary ==="
column -t "$summary" || cat "$summary"
echo
echo "Audio samples in: $WORK_DIR"
echo "Pull to local with:"
echo "  scp -i ~/home/tensordock/KeyPair-4fbd-yue.pem -P 31984 \\"
echo "      ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com:$WORK_DIR/'*.wav' /tmp/"
