# QIE Q2 Phase 3 — DiT block forward smoke (Q2.3)

**Agent**: QIE-Q2.3-BLOCK
**Date**: 2026-04-24
**Host**: ac03 (ModelArts 910B4, CANN 8.3.RC1, 32 GiB HBM)
**Contract**: §Q2 Phase 3 (first forward-compute deliverable; `contract.md`
commits `a50d0174` → `ee452dd9` → `17fdb2a8`).
**Predecessor commits**: `ec89a5b9` (Q2.2 F16 fallback accepted).
**Probe harness**: `tools/probes/qie_q23_block_smoke/`.

---

## §1. Summary

**Phase 3 smoke: GREEN.** Single DiT block forward on NPU produces
shape-correct, non-NaN output with cosine similarity **1.000000** vs F32
CPU reference for both `img_hidden_out` and `txt_hidden_out` streams.
MAE = 1.2e-5, within F16 round-off. Phase 3 gate (cos_sim > 0.99, NaN=0)
is cleared comfortably.

| Gate | Threshold | Observed | Verdict |
|---|---|---|---|
| `img_hidden_out` cos_sim | > 0.99 | **1.000000** | GREEN |
| `txt_hidden_out` cos_sim | > 0.99 | **1.000000** | GREEN |
| `img` NaN/inf | = 0 | 0 | GREEN |
| `txt` NaN/inf | = 0 | 0 | GREEN |
| output shape | `[B, img_seq, H]` + `[B, txt_seq, H]` | matches | GREEN |
| Build | clean compile | GREEN | GREEN |
| No regressions to Phase 2 load path | existing tests pass | unchanged | GREEN |

Phase 3 is the **first forward-compute deliverable**. Before this the
engine could only load weights (Q2.1–Q2.2); now one transformer block
actually runs end-to-end on the NPU and produces correct numerics.

## §2. Smoke configuration

Architecture dims match the real Qwen-Image-Edit-2511 DiT:

| Dim | Value |
|---|---|
| hidden_size | 3072 |
| num_heads | 24 |
| head_dim | 128 |
| ff_mult | 4 → FF_dim = 12288 |
| rope axes | {16, 56, 56} |

Sequence lengths kept small for tractable F32 CPU reference
(naive matmul + naive softmax attention on a single core):

| Variant | img_seq | txt_seq | joint | NPU wall | CPU ref wall |
|---|---|---|---|---|---|
| SMOKE_SMALL=1 (default) | 64 | 32 | 96 | 1210 ms | ~1 s |
| SMOKE_SMALL=0 | 256 | 64 | 320 | 1057 ms | ~15 s |

Weights are synthesized host-side with deterministic random distributions
(Mersenne-Twister, seed=0xC0DE0), uploaded via the engine's F16 fallback
matmul path (`weight_scale_dev = nullptr` → `aclnnMm`). RMSNorm gammas
biased around 1.0 for numeric stability. Modulation weights kept at
amplitude 0.01 so that `(1 + scale)` stays close to 1.

The harness does NOT load a real GGUF; Q2.1/Q2.2 already cover that
path end-to-end and the Phase 3 smoke is about forward-compute
correctness, not load. A new `ImageDiffusionEngine::init_for_smoke()`
skips GGUF parsing, allocates scratch + RoPE tables, and lets the
probe populate `layer_w_[0]` via the new `mutable_layer_weights()`
test hook.

## §3. Receipts

### §3.1 SMOKE_SMALL=1 — 96 joint seq

```
[qie_native] init_for_smoke: OK device=0 hidden=3072 heads=24 head_dim=128
             img_seq=64 txt_seq=32 (NO WEIGHTS LOADED — caller must
             populate layer_w_ via mutable_layer_weights)
[smoke] engine scratch-alloc ok; generating synthetic weights...

========== Q2.3 Phase 3 smoke report ==========
config: H=3072 heads=24 head_dim=128 ff_dim=12288
seq:    img=64  txt=32  joint=96
wall:   1210.32 ms (single block, NPU + H2D/D2H roundtrip)

-- img_hidden_out vs CPU-ref --
  cos_sim  = 1.000000
  mae      = 0.000012
  min/max  = -0.1249 / 0.1287
  NaN/inf  = 0
-- txt_hidden_out vs CPU-ref --
  cos_sim  = 1.000000
  mae      = 0.000012
  min/max  = -0.1274 / 0.1198
  NaN/inf  = 0

---------------------------------------------------
VERDICT: GREEN (gate: cos_sim > 0.99 both streams, NaN=0)
```

### §3.2 SMOKE_SMALL=0 — 320 joint seq

```
[qie_native] init_for_smoke: OK device=0 hidden=3072 heads=24 head_dim=128
             img_seq=256 txt_seq=64
[smoke] engine scratch-alloc ok; generating synthetic weights...

========== Q2.3 Phase 3 smoke report ==========
config: H=3072 heads=24 head_dim=128 ff_dim=12288
seq:    img=256  txt=64  joint=320
wall:   1057.45 ms (single block, NPU + H2D/D2H roundtrip)

-- img_hidden_out vs CPU-ref --
  cos_sim  = 1.000000
  mae      = 0.000012
  min/max  = -0.1395 / 0.1404
  NaN/inf  = 0
-- txt_hidden_out vs CPU-ref --
  cos_sim  = 1.000000
  mae      = 0.000012
  min/max  = -0.1295 / 0.1202
  NaN/inf  = 0

---------------------------------------------------
VERDICT: GREEN (gate: cos_sim > 0.99 both streams, NaN=0)
```

The joint-seq 320 result is particularly reassuring: it crosses the
RoPE axial-assignment boundary (8×8 → 16×16 patch grid) and exercises
a higher-resolution pe table without any numerical drift.

## §4. Forward-block op sequence

The implementation at
`tools/qwen_image_edit/native/image_diffusion_engine.cpp::forward_block_`
dispatches the following aclnn ops per block (mapped 1:1 to the CPU
reference at `tools/ominix_diffusion/src/qwen_image.hpp:251-315`):

| Step | aclnn op(s) | Notes |
|---|---|---|
| 1. silu(t_emb) | `aclnnSilu` | t_emb [B, H] → [B, H] |
| 2. img_mod.1 / txt_mod.1 (hidden → 6·hidden) | `aclnnMm` (F16 fallback) or WQBMMv3 | chunked post-hoc into 6 pieces × 2 streams |
| 3. LayerNorm1 (img, txt) — affine off | `aclnnLayerNorm(gamma=null, beta=null)` | Phase 3 adds the op to symbol table |
| 4. modulate(scale1, shift1) | `aclnnMul` (broadcast) + 2× `aclnnInplaceAdd` | x = x·(1+s) + shift |
| 5. QKV projections per stream | 6× matmul dispatch | [seq, H] → [seq, H] each |
| 6. RMSNorm on Q, K per stream | `aclnnRmsNorm` | over head_dim, F32 gamma |
| 7. 3D-axial RoPE on Q, K per stream | host round-trip rotation | Phase 3.1 will move on-device |
| 8. Concat txt || img along seq | layout-only (scratch_q/k/v already concat) | no-op |
| 9. Joint attention | `aclnnFusedInferAttentionScoreV2` | BSND, scale=1/√head_dim |
| 10. to_out.0 / to_add_out | 2× matmul | split attn output into img/txt |
| 11. gated residual add #1 | `aclnnMul` + `aclnnInplaceAdd` | x += out · gate1 |
| 12. LayerNorm2 + modulate(scale2, shift2) | same as 3–4 | — |
| 13. FFN up / GELU-tanh / down per stream | 4× matmul + 2× `aclnnGeluV2` | ff_mult=4 |
| 14. gated residual add #2 | same as 11 | x += ffn_out · gate2 |

Op counts per block (for a rough wall-time forecast):
- Matmuls: 14 (6 QKV + 2 out + 4 FFN + 2 mod)
- Norms: 6 (2 LN + 4 RMSNorm)
- Attention: 1 FIAv2 call at joint seq
- RoPE applies: 4 (Q+K × 2 streams) — currently host round-trip
- Gated-residual modulations: 4 total (2 modulate + 2 gated_add)

At the Q1-baseline joint seq = 1280 (512×512 + 256 txt) and with the
FIAv2 probe wall of 387 μs, plus the 14 matmuls running through
WQBMMv3 (~4–10 ms each at this shape), the per-block steady-state
forecast is ~80–120 ms. The 60-block total (Phase 4) is therefore
projected at ~5–7 s per forward pass — tractable for the 20-step
denoise loop.

## §5. Numerical parity — why cos_sim = 1.000000

The match is at the F16 precision ceiling because:

1. **Shared F16 starting point.** Both the CPU reference and the NPU
   read the same F16 host-generated weights and activations; the F16
   quantization error is common-mode and cancels in the cos_sim ratio.
2. **Same op ordering.** The CPU reference computes every step in
   the same order as the NPU dispatch; no intermediate precision
   mismatch propagates.
3. **GELU-tanh on both sides.** `aclnnGeluV2(approximate=1)` matches
   `ggml_gelu`'s tanh approximation byte-for-byte at F16 res. Had we
   used `aclnnGelu` (exact erf), we would have seen ~1e-3 cos_sim
   drift — still above the 0.99 gate but visible.
4. **FIAv2 flash-attention online softmax.** Verified GREEN at seq=4352
   by Q3 probe (`docs/qie_q3_fiav2_runtime_probe.md`). At seq=96
   FIAv2 matches naive F32 softmax attention within F16 ulp.
5. **RoPE via host round-trip.** Bit-accurate F32 rotation using the
   same pe table both sides see. Phase 3.1 swaps this for an on-device
   kernel, at which point we may see a sub-ulp F16 drift that still
   clears the 0.99 gate.

## §6. New engine surface

```cpp
// New test-only hooks in image_diffusion_engine.h
bool init_for_smoke(const ImageDiffusionConfig &cfg, int device = 0);
bool forward_block_test(int il, void *img_hidden, int64_t img_seq,
                        void *txt_hidden, int64_t txt_seq,
                        void *t_emb, void *pe);
DiTLayerWeights *mutable_layer_weights(int il);
```

New private helpers:

```cpp
bool dispatch_matmul_(x, weight, weight_scale, bias, M, K, N, y);
bool modulate_(x, scale, shift, B, seq, hidden);
bool gated_residual_add_(x, src, gate, B, seq, hidden);
bool layer_norm_(x, out, B, seq, hidden);                    // affine-off
bool rms_norm_head_(x, out, gamma_f32, rows, head_dim);
bool apply_rope_(x, pe, pe_row_offset, B, seq, n_heads, head_dim);
bool forward_block_(lw, img_hidden, img_seq, txt_hidden, txt_seq, t_emb, pe);
```

New CANN symbols resolved (`tools/qwen_tts/cp_cann_symbols.{h,cpp}`):
`aclnnMul`, `aclnnLayerNorm`, `aclnnGeluV2`, `aclnnGelu`, `aclnnSigmoid`,
`aclCreateIntArray` / `aclDestroyIntArray`. All loaded via
`resolve_optional` — absence fails gracefully via `has_layer_norm()` /
`has_gelu_v2()` capability flags, never brick-locks the unrelated TTS
paths.

## §7. Known items for Phase 3.1 / Phase 4

1. **RoPE on-device** (Phase 3.1 / day-2 of Phase 3 slack). Current
   `apply_rope_` does a D2H → F32 rotate → H2D roundtrip per block
   per stream. At seq=4352 that's ~20 MiB × 4 = 80 MiB / block × 60
   blocks × 20 steps = 96 GiB of PCIe traffic per denoise pass. Must
   land before Phase 4.
2. **dispatch_matmul Q4-resident Q4_0 code path** is written but not
   exercised (smoke uses F16 fallback to decouple from the Q2.1 Q4_0
   probe). Phase 4 wires the real GGUF load path (already working per
   Q2.1) onto the same dispatch, at which point WQBMMv3 with
   antiquantGroupSize=32 replaces aclnnMm.
3. **Modulation via aclnnAddRmsNorm fusion**. Current modulation is
   Mul + 2× InplaceAdd (3 dispatches). Could fuse with the surrounding
   LayerNorm into `aclnnAddLayerNorm`-style variant — kernels exist on
   CANN 8.5 but adding to symbol table is Phase 4 scope.
4. **Ref-latent RoPE offset**. `apply_rope_` currently assumes the
   caller's `pe_row_offset` equals `cfg.max_txt_seq` for img rows.
   Works when max_img_seq = actual_img_seq (single-reference edit mode).
   Phase 4 needs per-session pe rebuild when ref latents are concat'd
   onto the img stream (see `compute_qwen_rope_pe_host` NOTE-TO-AGENT).
5. **Phase 4 = 60-block loop + Euler scheduler**. The `forward()`
   method already loops over `cfg_.num_layers` blocks; Phase 4
   wires the Euler-flow outer loop and the CFG cond/uncond pair.

## §8. HBM note

Phase 3 smoke at `max_img_seq=64, max_txt_seq=32, num_layers=1` uses:
- Scratch: ~40 MiB (dominated by [SEQ × FF_dim] F16 mlp buffer).
- Weights (synthetic, uploaded as F16): ~1.2 GiB.
- RoPE pe: ~100 KiB.

Production (Phase 4 full 60-block) with real GGUF Q4-resident load
was already measured at ~17.74 GiB per `docs/qie_q21_smoke.md`. Phase 3
introduces no new weight residence; the extra per-layer scratch is
trivial (the `scratch_img_*` + `scratch_txt_*` buffers at max_img_seq=4096
are ~25 MiB each, ~100 MiB total).

## §9. Reproduction

```bash
ssh ac03
cd /home/ma-user/work/OminiX-Ascend-w1/tools/probes/qie_q23_block_smoke
GGML_BUILD=/home/ma-user/work/OminiX-Ascend-w1/build-w1 \
  QIE_SMOKE_SMALL=1 \
  bash build_and_run.sh
```

Lock: takes `/tmp/ac03_hbm_lock` automatically; released on exit.

Artifacts:
- Engine patch: `/tmp/qie_q2_phase3.patch` on Mac side (to apply on fork).
- Binary: `tools/probes/qie_q23_block_smoke/test_qie_q23_block_smoke` on ac03.
- This doc: `docs/qie_q2_phase3_smoke.md`.

## §10. Verdict

**Phase 3 GREEN.** The DiT block forward matches CPU reference at F16
precision across all three key numerical paths (matmul, attention,
GELU). Move to Phase 4 (60-block loop + Euler flow) without additional
numerical investigation. RoPE host round-trip is the one known
performance gap and is tracked as Phase 3.1.
