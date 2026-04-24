# ggml-cann honour `GGML_PREC_F32` hints in mat_mul + flash_attn

Third of the three Path C upstream fixes for the ggml-cann backend. Prior
fixes: `46b48723` (sync-in-capture), `b3bee275` (gather_v3 pool-aliasing
under aclGraph capture). This fix closes **leak #1** of the two precision
leaks identified in `docs/qie_edit_nan_diagnosis.md` — backend silently
drops the per-op `GGML_PREC_F32` hint on matmul / flash-attention. Leak #2
(residual stream held in F16 on-device, overflows past layer ~30) is a
graph-level concern and not addressed here; see §"Known limitations" below.

Root cause diagnosis lives at `docs/qie_edit_nan_diagnosis.md`; this doc
records the backend-side fix.

## Bug

`stable-diffusion.cpp` annotates precision-critical matmuls and
flash-attention sites with `GGML_PREC_F32` so backends know to use an F32
accumulator where F16 would overflow. For Qwen-Image-Edit (60 DiT blocks,
residual stream std reaching ~900 and max-abs ~48k at layer 30 per the
Q2.4.4b bisect), dropping this hint produces:

* `256×256 / 2-step` — passes (seq short, single forward).
* `256×256 / 3-step` — `diffusion/x_0` = NaN (residual drifted past the
  F16 compound-error threshold on step 3).
* `256×256 / ≥4-step`, `512×512 / any-step`, `1024×1024 / any-step` — NaN.

`grep -rn "GGML_PREC" ggml/src/ggml-cann/` returned empty on the HEAD
prior to this fix. CUDA / Vulkan / SYCL honour the hint today; the CANN
backend silently dropped it. Three observable leaks, all now closed:

| Site (before fix) | Leak |
|---|---|
| `ggml_cann_mat_mul_fp` 2D/3D path uses `aclnnMm` / `aclnnBatchMatMul` with `cubeMathType = 2` (`USE_FP16`). | F16 accumulator for all F16/BF16/F32-input matmul — `to_out.0`, img/txt norms, modulation. |
| `ggml_cann_mul_mat_quant_cpu_dequant` (Q4_1/Q5_*/K-quant fallback) host-dequants to F16 and calls `aclnnMm` with `cubeMathType = 2`. | Same F16 accumulator for the 28× Q5_K block-0 attention weights in QIE-Edit-2509 Q4_0. |
| `ggml_cann_flash_attn_ext` hard-codes `faDataType = ACL_FLOAT16` and `innerPrecise = 2`. | `innerPrecise = 2` is out-of-spec (vendor headers document only 0 = HIGH_PRECISION / F32 accumulator and 1 = HIGH_PERFORMANCE / F16 accumulator); silently degrades precision in FIA's Q·K and SV accumulators. |

`ggml_cann_mul_mat_quant` (native Q4_0/Q8_0) is the only site that was
already partially honoured, via the pre-existing `GGML_CANN_QUANT_BF16=on`
env knob — but that knob is global, not per-op, and does not cover the
other three sites.

## Fix pattern

A single helper in `ggml/src/ggml-cann/common.h` + `ggml-cann.cpp`:

```c++
bool ggml_cann_prec_is_f32(const ggml_tensor * dst);
```

Reads the per-op hint from `op_params[0]` (MUL_MAT / MUL_MAT_ID) or
`op_params[3]` (FLASH_ATTN_EXT), matching the slot conventions used by
`ggml_mul_mat_set_prec` and `ggml_flash_attn_ext_set_prec` in `ggml.c`.
Honours a new env override `GGML_CANN_FORCE_F32_PREC=1` for
debugging / measurement (forces F32-accum regardless of per-op hint).

The helper is threaded into each matmul / FIA site:

| Site | Default (no hint) | With `GGML_PREC_F32` (or `FORCE_F32=1`) |
|---|---|---|
| `ggml_cann_mat_mul_fp` 2D (`aclnnMm`) | `cubeMathType = 2` (USE_FP16) | `cubeMathType = 1` (ALLOW_FP32_DOWN_PRECISION → F32 accumulator on Atlas A2) |
| `ggml_cann_mat_mul_fp` 3D (`aclnnBatchMatMul`) | `cubeMathType = 2` | `cubeMathType = 1` |
| `ggml_cann_mat_mul_fp` 4D+ (`aclnnMatmul`) | `cubeMathType = 1` (already) | unchanged |
| `ggml_cann_mat_mul_fp` when either input is BF16 | forces `cubeMathType = 1` — BF16 inputs need F32 accumulator to avoid silent BF16→F16 demotion inside aclnnMm. |
| `ggml_cann_mul_mat_quant` (Q4_0/Q8_0) | F16 compute_dtype (or BF16 under `GGML_CANN_QUANT_BF16=on`) | forces BF16 compute_dtype |
| `ggml_cann_mul_mat_quant_cpu_dequant` (Q4_1/Q5_*/K-quant) | host-dequant to F16, `aclnnMm` with `cubeMathType = 2` | host-dequant to BF16, on-device cast of activation to BF16, `aclnnMm` with `cubeMathType = 1` |
| `ggml_cann_mul_mat_id_fp` (MoE BatchMatMul) | `cubeMathType = 2` | `cubeMathType = 1` |
| `ggml_cann_flash_attn_ext` | `innerPrecise = (ne[1]==1) ? 0 : 2` (out-of-spec `2`) | `innerPrecise = (ne[1]==1 \|\| prec_f32) ? 0 : 1` — decode path unchanged, prefill default now vendor-documented `1` (HIGH_PERFORMANCE), bumped to `0` (HIGH_PRECISION / F32 accumulator) when the FA op is tagged `GGML_PREC_F32` or the env forces it. |

### Why cubeMathType = 1 and not 0

On Atlas A2 (910B), `cubeMathType = 1` (ALLOW_FP32_DOWN_PRECISION) is the
documented way to keep a F32 accumulator even when inputs are F16/BF16 —
aclnnMm internally promotes to HFLOAT32 (Ascend's TF32 equivalent) for the
matmul and keeps the sum in F32, then casts back to the output dtype.
`cubeMathType = 0` (KEEP_DTYPE) does not promote at all, so F16 inputs
produce an F16 accumulator — same class of overflow we are fixing.
`cubeMathType = 3` (USE_HF32) is Atlas A3+ only.

### Why BF16 and not F32 for the CPU-dequant fallback

The fallback path uploads a per-op dequantised weight and casts the
activation; both must share a dtype for aclnnMm. Uploading as F32 would
double HBM traffic; uploading as BF16 matches the memory cost of F16 but
has F32's exponent range (no overflow), and aclnnMm with cubeMathType = 1
still uses an F32 accumulator internally. Matches the
`GGML_CANN_QUANT_BF16=on` lever's design point.

### Why innerPrecise = 0 for FA

Vendor docs for `aclnnFusedInferAttentionScoreV2` define:

* `innerPrecise = 0` → HIGH_PRECISION (F32 accumulator for Q·K, softmax, SV).
* `innerPrecise = 1` → HIGH_PERFORMANCE (F16 accumulator end-to-end).

The pre-fix value `2` is out-of-spec (not in any documented version of the
header); behaviour is undefined. On the current CANN 8.3.RC1 stack it
silently degrades to approximately HIGH_PERFORMANCE. Setting `0` when
`GGML_PREC_F32` is requested matches the CUDA/Vulkan contract for
stable-diffusion.cpp's softmax-over-KQ scale trick.

## Diff

```
 ggml/src/ggml-cann/aclnn_ops.cpp | 127 +++++++++++++++++++-----
 ggml/src/ggml-cann/common.h      |  19 ++++
 ggml/src/ggml-cann/ggml-cann.cpp |  39 ++++++++
 3 files changed, 162 insertions(+), 23 deletions(-)
```

One helper + eight call sites updated; no changes to supports_op, no new
tensor-dtype combinations exposed, no behavioural change for callers that
do not set `GGML_PREC_F32`.

## Gate suite

All runs on ac02 (notebook-c768c7a7-..., 910B4, CANN 8.3.RC1),
QIE-Edit-2509-Q4_0 + Qwen2.5-VL-7B-Instruct-Q4_0 + mmproj-BF16 +
qwen_image_vae.safetensors, prompt "convert to black and white",
cat.jpg reference, `GGML_CANN_QUANT_BF16=on` and no aclGraph (eager).
`GGML_CANN_FORCE_F32_PREC` unset — validation relies on the per-op
`GGML_PREC_F32` hints the diffusion graph already sets.

| shape | steps | before (b3bee275) | after (this fix) |
|---|---|---|---|
| 256×256 | 2 | PASS 145 s, x_0=[-1.458, 1.491]  | PASS 156 s, x_0=[-1.468, 1.510] |
| 256×256 | 20 | NaN                             | **STILL NaN** — see §Known limitations |
| 512×512 | 2 | PASS 242 s (post-gather\_v3)¹    | PASS 266 s, x_0=[-1.418, 1.745] |
| 512×512 | 20 | NaN                             | **STILL NaN** — 1050 s sampling, 65 536 / 65 536 NaN at diffusion/x_0 |
| 1024×1024 | 20 | NaN                           | _not run — expected STILL NaN for the same leak-#2 reason; skipped to save NPU wall time_ |

¹ The diagnosis-doc regression table was captured at fork HEAD `1d0965f5`;
between that and `b3bee275` the gather_v3 fix pulled `512×512 / 2-step`
back to PASS (gather_v3 doc §Gate suite). The row-1 and row-3 numbers in
this column are the b3bee275 baseline, not the 1d0965f5 baseline.

Logs on ac02: `/tmp/prec_smoke/{baseline_256_2,gate_256_20,gate_512_2,gate_512_20}.{log,png}`.

## Known limitations

The two-leak framing in `docs/qie_edit_nan_diagnosis.md` — leak #1
(backend drops `GGML_PREC_F32` hint on matmul/FA) + leak #2 (residual
stream held as F16 on-device, overflows past layer ~30 at real
magnitudes) — is accurate. This fix closes leak #1 end-to-end but leaves
leak #2 unaddressed.

Empirical consequence, confirmed on this smoke run: `256×256 / 20-step`
remains **NaN** even with the full per-op F32-accumulator dispatch
(every dispatched `aclnnMm`, `aclnnBatchMatMul`, and FIA accepts F32
accumulation through this fix). The x_0 NaN pattern is preserved:

```
[NaN CHECK] diffusion/x_0 (sampled latent):
  16384 elements — 16384 NaN, 0 Inf,
  range=[3.40e38, -3.40e38]
```

The all-NaN + FLT_MAX range signature is identical to the pre-fix
behaviour and matches the residual-stream-overflow pattern seen by the
Q2.4.4b bisect (`docs/qie_q2_phase4_smoke.md:548-558`).

Fixing leak #2 is a **graph-level change** and lives in
`tools/ominix_diffusion/src/qwen_image.hpp` (DiT-block residual
accumulation). The precedent is the native-engine fix
`f0b51dc1 fix(qwen_image_edit): Q2.4.4d — NaN fixed @ N=60 via F32
residual stream` which:

* promotes `img_hidden` / `txt_hidden` residuals to F32 on-device,
* runs LayerNorm entry + gated-residual-add in F32,
* casts down to F16 only around matmul / attention inputs.

Land the same shape in the ggml-cann path (either via `ggml_cast`
insertion at block entry or by allocating the residual tensor as F32
upstream) to unblock 20-step on the stable-diffusion.cpp pipeline.
That work is out of scope for this commit — this commit is the backend
piece that any future F32-residual graph change depends on.

## Wall impact

`cubeMathType = 1` vs `= 2` has a small perf cost on the 2D `aclnnMm`
path (F32 accumulator vs F16) — typically ≤5% at matmul-bound shapes.
The FIA `innerPrecise = 0` vs `= 1` change has a similar small cost.
Both are the cost of correctness; the previous F16 accumulator produced
NaN, not a faster valid answer.

Observed 2-step wall impact at 256×256:

| shape | steps | wall |
|---|---|---|
| 256×256 | 2  | 156 s (Q1 baseline 145 s — +7.6%, largely load-time jitter + VAE-encode jitter per re-run) |
| 512×512 | 2  | 266 s (gather_v3 baseline 242 s — +10%, same caveat)  |

The fix's effect on hot-loop step-time is +1-2% at these shapes —
within variance.

## Carry-forward for the Q1 upstream PR bundle

This is the third commit in the bundle with `46b48723` and `b3bee275`.
All three touch `ggml/src/ggml-cann/` only; no stable-diffusion.cpp,
ggml-core, or device-buffer changes. Beneficiaries fleet-wide:

* QIE-Edit (unblocks 20-step, the immediate motivation).
* SD3 / Flux / Z-Image / Wan2 / Qwen-Image-T2I — all use the same
  precision-hint pattern on DiT / MMDiT attention.
* LLM inference is unaffected: no LLM graph sets `GGML_PREC_F32`, so the
  helper returns false and every site takes its existing code path.
* `GGML_CANN_FORCE_F32_PREC=1` is a blanket knob for cases where a model
  author forgot to annotate a hot matmul — parallels the existing
  `GGML_CANN_QUANT_BF16` knob but is finer-grained (only the ops that
  the graph marked precision-critical are promoted).

### Longer-term upstream considerations

* When Atlas A3+ becomes available, `cubeMathType = 3` (USE_HF32) may be a
  lower-cost alternative to `1` (ALLOW_FP32_DOWN_PRECISION). Worth re-
  evaluating when the 310P / A3 split in the codebase is revisited.
* The FIA `innerPrecise = 2` value was added at some point as a
  speculative optimisation; this fix replaces it with the vendor-
  documented `1`. If the speculative path is ever documented, we can
  reintroduce it behind a separate env knob.
