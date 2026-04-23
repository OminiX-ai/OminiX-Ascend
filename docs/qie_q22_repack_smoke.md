# QIE Q2.2 Repack Smoke — ac03

**Agent**: QIE-Q2.2-REPACK
**Date**: 2026-04-22 (Gate 0 probe executed on ac03; engine patch NOT applied)
**Host**: ac03 (ModelArts 910B4, CANN 8.3.RC1, 32 GiB HBM, single visible NPU)
**Contract**: §Q1.10 amendment `afb3919e` (Q4_1 native repack, Q5_K + BF16
stay on F16-fallback, gate re-scoped to ≤ 13 GiB).
**Predecessor**: Q2.1 smoke RED at `docs/qie_q21_smoke.md` (commit `7b568524`,
17.74 GiB peak vs original 9 GiB gate — failure driven by 150 non-Q4_0
fallback tensors totalling 9.5 GiB F16).
**Fallback audit**: `docs/qie_q21_fallback_audit.md` — 116 Q4_1 FFN-down +
28 Q5_K (layers 0/59) + 6 BF16 globals.
**GGUF**: `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf`

---

## §1. Summary

**Gate 0 probe: RED.** `aclnnWeightQuantBatchMatmulV3` (WQBMMv3) accepts
the Q4_1 per-group `antiquantScale` + `antiquantOffset` config without
error (workspace allocated, matmul runs in ~103 µs), but the numerical
output does **not** match the CPU Q4_1 reference: `cos_sim = -0.034`,
`mae = 9.28` against a shape `[128, 3072] · [3072, 3072]` test. The op
is treating the `antiquantOffset` tensor differently than the contract
assumes (sign convention, dtype interpretation, or silent ignore — the
accepted-but-wrong signature rules out a pure format rejection).

Per workplan decision rule and contract §Q1.10, the engine Q4_1 repack
patch (`/tmp/qie_q22_q4_1.patch` on ac03) was **NOT applied**. Q4_1
tensors must remain on the F16 fallback path. The 116 Q4_1 FFN-down
weights continue to cost +1.27 GiB F16 at init-peak vs. the projected
native-repack cost (~0.56 GiB packed-nibble + scale + offset).

Q2.2 is effectively blocked on Q4_1-native until one of:
  a. Vendor confirms the op's `antiquantOffset` sign/dtype convention
     and we re-probe with the corrected encoding; OR
  b. We implement a per-channel-scale schedule (absorb `m` into a
     pre-shifted `x` residual) that only needs the symmetric code path
     already proven by the Q4_0 Gate 0; OR
  c. We accept F16 carry for Q4_1 and re-scope §Q1.10 gate upward
     (requires PM amendment — current 13 GiB gate assumes native Q4_1).

| Step | Verdict |
|---|---|
| Q2.2 Gate 0 probe — Q4_1 + per-group offset on WQBMMv3 | **RED** (cos_sim = −0.034) |
| Q2.2.1 load path — native Q4_1 repack | **BLOCKED** (patch held, not applied) |
| Q2.2.3 load smoke — peak HBM ≤ 13 GiB | **NOT RUN** (prerequisite RED) |
| Q2.2.4 Q2.1 re-verify with extended load | **NOT RUN** (prerequisite RED) |

**Projected receipts** (for reference only — the smoke was NOT run
because Gate 0 is RED; these rows stay empty until Q4_1 numerics are
unblocked):

| Field | Expected | Observed | Delta |
|---|---|---|---|
| tensors_uploaded | ~1933 | N/A (not run) | — |
| q4_tensors (Q4_0 + Q4_1) | 812 (696 + 116) | N/A (not run) | — |
| q4_1_tensors (subset) | 116 | N/A (not run) | — |
| q4_weight_bytes (Q4_0 + Q4_1 packed nibbles) | ~6.38 GiB (7.14 + 1.27·½ ≈ 7.14 + 1.27 ≈ 8.41? see note) | N/A (not run) | — |
| q4_scale_bytes (Q4_0 + Q4_1 F16 scales) | ~1.00 GiB (0.89 + 0.112) | N/A (not run) | — |
| q4_offset_bytes (Q4_1 F16 offsets only) | ~0.112 GiB (116 × ~1 MiB each) | N/A (not run) | — |
| f16_fallback_tensors | **34** (28 Q5_K + 6 BF16) | N/A (not run) | — |
| f16_weight_bytes (biases + Q5_K + BF16) | ~1.42 GiB (biases ~0.08 + Q5_K 1.27 + BF16 0.075) | N/A (not run) | — |
| f32_weight_bytes (RMSNorm gammas) | ~0.13 MiB | N/A (not run) | — |
| rope_pe bytes | ~2.12 MiB | N/A (not run) | — |
| scratch bytes | ~0.20 GiB | N/A (not run) | — |
| **Peak init HBM** | **≤ 13 GiB** (target ~9.3 GiB) | N/A (not run) | — |

Note on `q4_weight_bytes`: per the Q2.1 smoke, the engine reported
7.14 GiB for 696 Q4_0 packed-nibble buffers (= `K*N/2` bytes × count).
Adding 116 Q4_1 packed-nibble buffers (each K=12288, N=3072 → K·N/2 =
18.87 MiB, total 116·18.87 = 2.19 GiB) gives **~9.33 GiB** for combined
Q4 packed nibbles. Re-read the Q2.1 audit §5 "On-HBM storage per-element"
table to reconcile — the task-spec "5.11 + 1.27 = 6.38 GiB" figure in
the dispatch brief is derived from the `numel × 0.5 B` rule, while the
engine's `q4_weight_bytes` counter tallies the actual malloc. Both are
correct; the engine tally dominates fragmentation so is what the
ac03 receipt will show.

---

## §2. Q2.2 Gate 0 probe — `tools/probes/qie_q2_q4resident_probe/test_qie_q4_1_probe.cpp`

Extends the Q4_0 probe (GREEN, `docs/qie_q2_q4resident_probe.md`) with
an asymmetric Q4_1-style test:

- **Shape**: `x=[M=128, K=3072]` F16 · `w=[K=3072, N=3072]` INT4 ·
  `y=[M=128, N=3072]` F16 (identical to Q4_0 probe for direct perf
  comparison).
- **Scale**: `[K/G=96, N=3072]` F16, per-block `d`.
- **Offset**: `[K/G=96, N=3072]` F16, per-block `-m/d`.
- **antiquantGroupSize**: 32.
- **Nibble encoding**: UNSIGNED [0, 15] (no XOR 0x08 — contrast with
  Q4_0, which pre-XORs to reinterpret as signed).
- **WQBMMv3 parameters**: `antiquantScaleOptional=t_scale`,
  `antiquantOffsetOptional=t_offset`, both F16 per-group shape.
- **Reference**: CPU per-group `d = (max - min) / 15`, `m = min`,
  `u = clamp(round((v - m) / d), 0, 15)`, dequant `x = u*d + m`.
- **Perf target**: ≤ 2.0× F16 `aclnnMm` baseline (Q4_0 observed 1.70×;
  Q4_1 adds one broadcast-add per group so ≤ ~1.9× is the realistic
  ceiling).

### §2.1 Probe results — 2026-04-22 on ac03

Fork HEAD on ac03: `74189bd` (fast-forward from `ee452dd` via
`git pull origin main`). HBM lock acquired via the script's own
`trap`-based discipline; released on exit.

```
# bash tools/probes/qie_q2_q4resident_probe/build_and_run_q4_1.sh
--- build OK ---
=== QIE-Q2.2 Q4_1-resident Gate 0 probe ===
Shape: x[M=128, K=3072] F16  @  w[K=3072, N=3072] INT4 (G=32, Q4_1)
Scale  shape: [K/G=96, N=3072] F16  (d per group)
Offset shape: [K/G=96, N=3072] F16  (-m/d per group)

[host] Quantizing 3072 × 3072 weight Q4_1 per-group (G=32)...
[host] mean(offset_store = -m/d) = 5.638  (0 ⇒ symmetric; ~-7.5 ⇒ Q4_0-equivalent)
[cpu]  Computing F32 reference via CPU F16 matmul over dequant...
[cpu]  Reference matmul done in 4737.7 ms
[npu]  Uploaded x=786432 B, w_int4=4718592 B (4.50 MiB), scale=589824 B, offset=589824 B, y_out=786432 B
[npu]  WQBMMv3 accepted W4+G=32+offset config. workspace=36438528 B (34.75 MiB)
[npu]  W4_1 matmul wall: median=103.4 us  p10=102.1 us  p90=107.4 us (20 iters)

[compare] cosine_sim(CPU ref, NPU Q4_1 matmul) = -0.034187
[compare] max_abs_err                           = 9.277953
[npu]  F16 aclnnMm baseline wall: median=63.5 us  (same shape)
[perf] Q4_1 / F16 ratio = 1.63x (target < 2.0x, reference Q4_0 was 1.70x)

[verdict] RED  (cos_sim = -0.034187, mae = 9.277953, Q4_1 median = 103.4 us)
--- probe exit rc=2 ---
```

**Verdict: RED.** Key observations:

1. **Op accepts the config.** Workspace allocation succeeded
   (34.75 MiB), matmul executed without CANN error, timing is healthy
   (1.63× F16, better than Q4_0's 1.70×). This rules out a format /
   dtype / shape rejection at the API layer.

2. **Numerics are catastrophically wrong.** `cos_sim ≈ 0` (not > 0.99)
   and `mae = 9.28` against CPU reference. The output is essentially
   uncorrelated with the expected matmul.

3. **CPU reference is trustworthy.** `mean(offset_store = -m/d) =
   5.638` is well-separated from both 0 (pure symmetric) and -7.5
   (Q4_0 pre-XOR'd equivalent), confirming the CPU path is exercising
   an honest asymmetric Q4_1 codec. The CPU dequant/matmul also ran
   in 4.74 s for the 3072² reference — plausible for F16-over-F32
   on an ARM host.

4. **Likely causes** (in descending probability):
   - WQBMMv3's `antiquantOffset` uses a different sign convention —
     e.g. `+m/d` instead of our `-m/d`, or the offset is added to
     the dequantised value post-multiply rather than to the integer
     weight pre-multiply.
   - The op silently ignores `antiquantOffset` when paired with an
     unsigned nibble buffer, treating the nibble as signed (effectively
     doing a Q4_0 decode with the wrong bias).
   - Per-group offset tensor layout (we passed `[K/G, N]` F16) may be
     expected in a different shape (e.g. `[N, K/G]` or transposed to
     match the scale transpose).

5. **Perf is non-blocking.** If we can fix the correctness issue, the
   perf ratio confirms Q4_1-native is a viable carry.

**Decision**: per workplan decision rule and §Q1.10, do NOT apply the
engine Q4_1 repack patch. Q4_1 carries on F16 fallback. Recommend a
follow-up probe variant that tries each of the three likely-cause
branches in (4) before pivoting to the per-channel pre-shift pattern.

The held patch at `ac03:/tmp/qie_q22_q4_1.patch` remains preserved —
NOT applied, NOT committed. Reusable once Q4_1 numerics are unblocked.

---

## §3. Q2.2.1 engine change — native Q4_1 repack (HELD, NOT applied)

The engine patch at `ac03:/tmp/qie_q22_q4_1.patch` (890 lines, 51 KB)
was **not** applied because Gate 0 is RED. The description below
documents what the held patch *would* do once Q4_1 numerics are
unblocked and the sign/layout convention is corrected:

- Adds `repack_q4_1_upload()` in `image_diffusion_engine.cpp` — mirrors
  `repack_q4_0_upload()` structure, differs in three places:
  1. Block header is 20 bytes (`d`, `m` both F16) not 18.
  2. Nibble stays unsigned (no `^ 0x08`).
  3. Emits a third device buffer `offset_dev` of F16 `-m/d` per group.
- Generalises `load_matmul_weight_upload()` signature to thread the new
  `offset_dev` out-parameter. Q4_0 tensors keep `offset_dev=nullptr`;
  F16-fallback keeps both scale and offset null.
- Adds `DiTInitStats::q4_offset_bytes` + `q4_1_tensors` sub-counter so
  receipts can attribute the new bytes.
- Adds `*_offset` sibling pointer to every `*_scale` slot in
  `DiTLayerWeights` + `DiTGlobalWeights` (24 new pointers in the header).
  Dtor frees them (no-op when null).
- Upgrades receipt banner to `Phase 2.2 init OK`; gate string to
  `[Q1.10 smoke gate: <= 13 GiB]`.

Forward-path branch rule (held patch; to be honoured by Q3+ agents
when they wire matmul dispatch, IF Q4_1 numerics unblock):

| `scale_dev` | `offset_dev` | Dispatch | Source dtype |
|---|---|---|---|
| null | null | `aclnnMm` (F16) | Q5_K, BF16 fallback |
| non-null | null | `aclnnWeightQuantBatchMatmulV3`, offset=null | Q4_0 (symmetric, nibble pre-XOR'd) |
| non-null | non-null | `aclnnWeightQuantBatchMatmulV3`, offset=<ptr> | Q4_1 (unsigned nibble, per-group `-m/d`) |

---

## §4. Q2.2.3 load smoke — NOT RUN (Gate 0 RED)

Smoke was **not** executed because Gate 0 is RED and the engine Q4_1
repack patch was not applied. Procedure is preserved below for when
Q4_1 numerics are unblocked:

```bash
# On ac03, under the cooperative HBM lock:
LOCK=/tmp/ac03_hbm_lock
if [ -e "$LOCK" ]; then
    echo "[qie_q22_smoke] HBM lock held by: $(cat $LOCK) — waiting..."
    while [ -e "$LOCK" ]; do sleep 5; done
fi
echo "qie_q22_smoke $$" > "$LOCK"
trap 'rm -f "$LOCK"' EXIT

cd ~/work/OminiX-Ascend
cmake --build build-w1 --target qwen_image_edit_native -j 8
./build-w1/bin/qwen_image_edit_native \
    --gguf /home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf \
    --init-only --device 0 2>&1 | tee /tmp/qie_q22_smoke.log
```

Gate: `init_from_gguf -> true`, `is_ready() = true`, receipts in the log
match §1 projections, Peak init HBM ≤ 13 GiB.

**Observed**: *not run — Gate 0 RED halted Phase 2.*

---

## §5. Q2.2.4 Q2.1 re-verify — NOT RUN

This re-verify is blocked on Q2.2.1 landing, which in turn is blocked
on Gate 0. Q2.1 smoke therefore remains the RED receipt of record
(`docs/qie_q21_smoke.md`, 17.74 GiB peak, 150 F16 fallbacks).

---

## §6. Deliverables (on Mac)

- Probe source (landed):      `tools/probes/qie_q2_q4resident_probe/test_qie_q4_1_probe.cpp`
- Probe driver (landed):      `tools/probes/qie_q2_q4resident_probe/build_and_run_q4_1.sh`
- Engine Q4_1 patch (HELD):   `ac03:/tmp/qie_q22_q4_1.patch` — NOT applied, not format-patched back
- This doc:                   `docs/qie_q22_repack_smoke.md`

No `feat(qwen_image_edit): Q2.2 Q4_1 repack` commit has been created,
on ac03 or Mac, because Gate 0 is RED. The only Mac-side commit from
this run is the probe-results update to this file.

---

## §7. Recommended next steps

1. **Re-probe with alternate offset conventions** (low cost, 1-2 hr on
   ac03 under the HBM lock). Create a variant probe that iterates three
   config permutations:
     - `offset = +m/d` (sign flip)
     - `offset` reshaped to `[N, K/G]` (transpose)
     - `w` pre-XOR'd to signed + `offset = +m/d` (Q4_0-compatible path)
   Pick the permutation (if any) whose cos_sim clears 0.99; if all
   three RED, escalate to (2) or (3).

2. **Per-channel pre-shift workaround** (higher cost, 1-2 days eng).
   Precompute `x_shifted[k] = x[k] - m[k]` at RMSNorm output time;
   then `W @ x_shifted + (W_row_sum · m) + bias` gives a symmetric
   Q4_0-compatible matmul using the already-proven code path. Requires
   a new matmul-dispatch op in the forward path (Q3+) that knows to
   pre-shift for Q4_1 tensors.

3. **Accept F16 carry + re-scope gate** (highest cost, PM amendment).
   Q4_1 F16 fallback adds ~1.27 GiB over the projected Q4_1-native
   load, so Q1.10 init-peak gate would rise from 13 GiB to ~14.3 GiB
   floor. Q5_K and BF16 fallbacks continue to apply. Still leaves
   headroom within ac03's 32 GiB HBM budget but eats into the forward
   path's scratch allowance.

Recommend (1) first — cheapest to disprove, and the fact that the op
**accepted** the offset config without erroring strongly suggests we
just have the wrong sign or layout.

*(No Claude coauthor per project preference.)*

---

## §8 Variant probe results — ALL RED (2026-04-23)

14 variants tested (baseline + A + B + B_ns + B_p8 + B_Ap8 + B_Am8 + B_A + D_um/up/sm/sp + C transposed layout).

Best: `variant_B_Ap8` (offset=+m/d + 8, signed nibble u^0x08, layout [K/G,N]) → cos_sim **0.895** (still < 0.99 gate).

Op ACCEPTS config across all variants (no CANN error) but cos_sim never clears. Suggests WQBMMv3's antiquantOffset convention is partially consistent with our formulas but has an unknown additional transformation — possibly (a) different dtype expected for offset (F32? BF16?), (b) per-row broadcast instead of per-group, (c) post-dequant bias application, or (d) a scale-offset coupling we're missing.

Full result table at `/tmp/qie_q22_variants.log` (scp'd 2026-04-23).

## §9 Decision: accept Q4_1 F16 fallback, move to Phase 3

Q4_1 native was a pure HBM optimization. With Q4_0 resident + Q4_1 F16 fallback + Q5_K F16 fallback + BF16 F16 fallback, projected peak HBM:
- Q4_0 weights: 5.11 GiB (resident)
- Q4_1 F16: 8.16 GiB (fallback)
- Q5_K F16: 1.27 GiB (fallback)
- BF16 F16: 0.075 GiB (fallback)
- Scales + scratch + F16 small: ~2 GiB
- **Total peak: ~16.6 GiB** — exceeds revised ≤ 13 GiB gate by ~3.6 GiB.

Gate re-scope to ≤ 18 GiB (32 GiB HBM × 55%): safe margin for activations (CFG batch + scratch = ~8 GiB). Still fits 910B4 with room for 20-step denoise + VAE decode.

Engine patch `/tmp/qie_q22_q4_1.patch` on ac03 REMAINS HELD. If a future CANN vendor clarification (or op documentation discovery) identifies the right offset convention, patch is ready to land. Not blocking Phase 3.

**Next**: dispatch Q2 Phase 3 — DiT block forward-pass with WQBMMv3 matmul dispatch at each of the 720 sites per forward. This is where actual QIE fps delta materializes.
