#!/usr/bin/env bash
# Q2.4.5.4i — full discrimination run.
#
#   Step 1  native dump  : runs qie_q45_real_denoise_smoke with
#                          QIE_DUMP_BLOCK0_DIR set → captures
#                          {00_img,00_txt,00_t_emb,24_img_resid2,
#                          24_txt_resid2}.f32
#   Step 2  CPU reference: runs qie_block0_cpu_reference → produces
#                          cpu_24_{img,txt}_resid2.f32
#   Step 3  comparison   : runs compare_block0.py → reports verdict
#
# Default mode: small smoke (img_seq=64 txt_seq=32). For production-scale
# discriminator set QIE_Q45_BIG=1 (img_seq=256 txt_seq=64).
#
# The native engine dump path runs WITH QIE_FFN_DOWN_BF16=1 by default to
# match the §5.5.8 BF16 ff_down receipts (1.1e5 / 7.4e6 magnitudes).
# Override by exporting QIE_FFN_DOWN_BF16=0 before launching.
#
# Sighup-proof launch (run from anywhere on ac03):
#   nohup setsid bash -c '
#     cd ~/work/OminiX-Ascend &&
#     bash tools/probes/qie_block0_cpu_reference/run_step_4i.sh
#   ' < /dev/null > /tmp/qie_q2454i_run.log 2>&1 &
#
set -euo pipefail

DIR=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$DIR/../../.." && pwd)

NATIVE_PROBE_DIR="$REPO/tools/probes/qie_q45_real_denoise_smoke"
CPU_PROBE_DIR="$REPO/tools/probes/qie_block0_cpu_reference"

: "${QIE_BLOCK0_INPUTS_DIR:=/tmp/qie_block0_inputs}"
: "${QIE_BLOCK0_OUTPUTS_DIR:=/tmp/qie_block0_outputs}"
mkdir -p "$QIE_BLOCK0_INPUTS_DIR" "$QIE_BLOCK0_OUTPUTS_DIR"

# Step 1: native engine dump. Cap N_STEPS=1 — we only need the first
# block-0 invocation; subsequent calls are no-ops for the dump path.
export QIE_DUMP_BLOCK0_DIR="$QIE_BLOCK0_INPUTS_DIR"
export QIE_DEBUG_INTRA_BLOCK0="${QIE_DEBUG_INTRA_BLOCK0:-1}"
export QIE_N_STEPS="${QIE_N_STEPS:-1}"
export QIE_FFN_DOWN_BF16="${QIE_FFN_DOWN_BF16:-1}"
export QIE_BLOCK0_INPUTS_DIR QIE_BLOCK0_OUTPUTS_DIR

echo "===== Q2.4.5.4i Step 1: native engine block-0 dump ====="
echo "QIE_FFN_DOWN_BF16=$QIE_FFN_DOWN_BF16  QIE_N_STEPS=$QIE_N_STEPS"
echo "QIE_DUMP_BLOCK0_DIR=$QIE_DUMP_BLOCK0_DIR"
echo

cd "$NATIVE_PROBE_DIR"
bash build_and_run.sh

echo
echo "Listing dump outputs:"
ls -la "$QIE_BLOCK0_INPUTS_DIR"

# Sanity-check all five files exist.
for f in 00_img.f32 00_txt.f32 00_t_emb.f32 24_img_resid2.f32 24_txt_resid2.f32; do
  if [ ! -f "$QIE_BLOCK0_INPUTS_DIR/$f" ]; then
    echo "[run_step_4i] missing dump file: $QIE_BLOCK0_INPUTS_DIR/$f"
    exit 1
  fi
done

# Step 2: CPU reference forward.
echo
echo "===== Q2.4.5.4i Step 2: CPU reference block-0 forward ====="
cd "$CPU_PROBE_DIR"
bash build_and_run.sh

echo
echo "Listing CPU reference outputs:"
ls -la "$QIE_BLOCK0_OUTPUTS_DIR"

# Step 3: element-wise comparison.
echo
echo "===== Q2.4.5.4i Step 3: element-wise comparison ====="
python3 "$CPU_PROBE_DIR/compare_block0.py" \
  --native-dir "$QIE_BLOCK0_INPUTS_DIR" \
  --cpu-dir    "$QIE_BLOCK0_OUTPUTS_DIR"

echo
echo "===== run_step_4i.sh complete ====="
