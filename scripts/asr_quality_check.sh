#!/bin/bash
# Automated ASR-based quality check for generated TTS audio.
#
# Runs qwen3-asr on a directory of WAV files and prints transcripts.
# Use this to catch "sounds like voice but doesn't match target text"
# regressions that DTW-log-mel can't detect (both paths can produce
# non-target speech and still match each other structurally).
#
# Requires: qwen3-asr-mlx crate available at ~/home/OminiX-MLX/qwen3-asr-mlx
# and the qwen3-asr-1.7b-8bit model at ~/.OminiX/models/qwen3-asr-1.7b.
#
# Usage:
#   bash scripts/asr_quality_check.sh <audio_dir> [target_text_file]
#
# If target_text_file is a TSV of "<basename>\t<expected text>", the
# script also prints PASS/FAIL per line based on substring match.

set -euo pipefail

AUDIO_DIR="${1:-/tmp/qg_natural}"
TARGETS="${2:-}"
ASR_CRATE="${HOME}/home/OminiX-MLX/qwen3-asr-mlx"

if [ ! -d "$AUDIO_DIR" ]; then
  echo "No such dir: $AUDIO_DIR" >&2; exit 1
fi
if [ ! -d "$ASR_CRATE" ]; then
  echo "qwen3-asr-mlx crate not found at $ASR_CRATE" >&2; exit 1
fi

pushd "$ASR_CRATE" > /dev/null

pass=0
fail=0
for wav in "$AUDIO_DIR"/*.wav; do
  name=$(basename "$wav")
  txt=$(cargo run --release --example transcribe -- "$wav" --language English 2>/dev/null \
        | awk 'flag{print; flag=0} /^=== Transcription/{flag=1}')
  if [ -n "$TARGETS" ] && [ -f "$TARGETS" ]; then
    expect=$(awk -F'\t' -v n="$name" '$1==n {print $2}' "$TARGETS")
    if [ -n "$expect" ]; then
      # Substring match first 3 words
      first3=$(echo "$expect" | awk '{print $1, $2, $3}')
      if echo "$txt" | grep -qiF "$first3"; then
        echo "PASS  $name  $txt"
        pass=$((pass+1))
      else
        echo "FAIL  $name  got=\"$txt\"  want=\"$expect\""
        fail=$((fail+1))
      fi
      continue
    fi
  fi
  echo "      $name  $txt"
done

popd > /dev/null

if [ -n "$TARGETS" ] && [ -f "$TARGETS" ]; then
  echo
  echo "ASR gate: $pass PASS / $fail FAIL"
  [ "$fail" -eq 0 ]
fi
