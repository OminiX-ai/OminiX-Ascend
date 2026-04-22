# QIE-2511 Optimization: Feasibility & Forward Learnings

**Date**: 2026-04-21. **Author**: Agent QIE-LEARN (synthesis, no
code). **Source base**: `docs/contracts/QWEN_IMAGE_EDIT_2511_CONTRACT.md`,
`docs/qie_q0v2_discovery.md`, `docs/qwen_tts_optimization_learnings.md`,
`docs/fused_op_audit.md`, `docs/qkv_grouped_probe_verdict.md`,
`docs/swiglu_probe_verdict.md`, `docs/fia_v34_probe_verdict.md`,
`docs/ascend_910b4_datasheet.md`, `tools/ominix_diffusion/src/*.hpp`,
`OminiX-MLX/qwen-image-mlx/docs/*.md`.

**Framing**: this doc mirrors `qwen_tts_optimization_learnings.md` but
diverges on content. The TTS doc is a retrospective over delivered work;
this doc is **forward-looking feasibility** — we have zero QIE fps
delivered yet (Q0-v2 YELLOW, three ggml-cann backend gaps block
baseline). The goal is to frame the contract with honest ROI before
spending 8-12 weeks of agent-wall on it.

## Executive summary

- **Zero baseline exists**. Q0-v2 crashed on all three attempted
  runs; three ggml-cann backend bugs (`mul_mat` missing Q4_K/Q5_K/Q6_K,
  `get_rows` missing Q4_0/Q4_1, `ascendc/gather_v3` crash in Qwen2.5-VL
  vision encoder) block any measurement. **Q1 must be backend-unblock,
  not optimization.** 1-2 weeks agent-wall before we have a number.
- **Architecture is not what the contract assumed**. Reading
  `qwen_image.hpp` reveals QwenImage DiT uses **RMSNorm on Q/K**,
  LayerNorm (affine-off) on block norms, **RoPE on every attention**
  via `Rope::attention`, GELU (not SwiGLU) FFN, 60 transformer blocks
  × 24 heads × 128 head_dim. This **partially rescues** the TTS
  playbook: W3b-style RMSNorm fusion is applicable on Q/K-norm sites;
  RoPE-V2 fusion is applicable; FFNV3 is not (GELU, not SwiGLU, and
  no MoE). CFG batching and step distillation remain the biggest
  novel levers.
- **MLX parity framing was wrong**. Apple M4 Max at "30 fps qwen-image
  T2I" is a different workload (tiny model variant / short prompt).
  `OminiX-MLX/qwen-image-mlx/docs/STEP_REDUCTION_RESULTS.md` shows the
  real M4 Max optimized figure at 20 steps is **~80 s diffusion wall
  = ~4 s/step** on the full model. Our Ascend 910B4 target at Q4 after
  full TTS-playbook transfer is **1-2 s/step** — which is **1-4× MLX,
  not 1/30× MLX.** Matching MLX is feasible; beating it is in range.
- **Honest ROI**: Q1 unblock (required, not a lever) + Q2 aclGraph
  step-keyed (step-stable shape is tailor-made for aclGraph: **+15-35%
  realistic** — better than TTS's G2 because no pos-keyed cache
  explosion, only ~20 graphs) + Q4 CFG batching (**+80-100% realistic**
  on per-step wall, the largest single lever) + RoPE pre-compute (**+10-25%**
  per MLX's own measurement of the same optimization) stacks to a
  plausible **2.5-4× over raw ggml-cann baseline**, matching the
  contract's 2-3× target with margin.
- **The real risk is not Q4 quality or HBM; it is Q1's gather_v3 bug.**
  If Q1.3 takes 3 weeks instead of 1, the whole contract slides. That's
  a CANN-backend bug-hunt with no probe coverage (we haven't seen it in
  TTS / ASR). Recommend: budget Q1 at 3 weeks, not 1-2, and schedule Q2
  to start on synthetic baseline (Q4_K_M numerics verified on CPU) in
  parallel so Q1 slippage doesn't stall everything downstream.

## Part 1 — Current state: blocked on 3 ggml-cann backend bugs

Per `qie_q0v2_discovery.md` (verified 2026-04-22, ac02):

### The 3 blockers, ranked by severity

| # | Location | Failure mode | Severity | Est. fix |
|---|---|---|---|---|
| 1 | `aclnn_ops.cpp:2272` `ggml_cann_get_rows` missing Q4_0/Q4_1 | T2I aborts at text-encoder embedding-table lookup | **HARD** — T2I fails immediately | 30-50 LOC mirroring existing Q8_0 branch, ~1-2 days |
| 2 | `aclnn_ops.cpp:2670` `ggml_cann_mul_mat` missing Q4_K/Q5_K/Q6_K | K-quant variants unusable; Q4_0 is the only quant that passes (and it still hits #1) | **HARD** on K-quant, soft on Q4_0 | 50-80 LOC per quant variant, ~3-5 days |
| 3 | `ascendc/gather_v3` crash in Qwen2.5-VL vision encoder on float-bit-pattern indices | Edit-mode aborts (ref-image conditioning path) | **HARD for edit**, soft for T2I | unknown — could be ggml-cann wiring bug (int32 vs float32 confusion in op input), could be genuine AscendC indices-dtype limit. **1-3 weeks, high variance.** |

Bug #3 is the schedule risk. It sits in the Qwen2.5-VL vision encoder
(mmproj → gather indices), which is an edit-specific path that TTS and
ASR have never touched. The crash signature (`gather_v3_base.h:137`
assertion with indices like `-1085232137` on AIV) is the classic
float-bits-being-read-as-int-bits pattern. **Is it a ggml-cann
graph-builder bug (dtype mismatch on input tensor), or a vendor op
limitation?** We don't know. The probe-first principle says: before
dispatching an A-agent for a 1-2 week fix, spend 2-4 hours in ac02
printing the graph tensor-dtypes around the first gather and
comparing to the Qwen2.5-VL MLX reference. If it's a ggml dtype tag,
it's a patch. If it's a genuine op limit, it becomes a CANN vendor
ask.

### What HBM taught us

Prior project framing (pre-Q0-v2) assumed 32 GB was too tight and we
needed to procure 910B3/B2 with 64 GB. **Q0-v2 disproved this**:
Q4 weights = 18-20 GB HBM, leaving 12 GB free. The 910B4 is not the
bottleneck. This is a genuine framing improvement and saved procurement
spend.

### Estimated per-op breakdown (before optimization, inferred)

Unlike TTS (where the TQE=2+NZ+W8 stack already compressed dispatch
overhead to ~2-3 μs eager), we have **no measured breakdown** because
there is no baseline. Inference from the architecture:

| Phase | Est. % of step wall | Notes |
|---|---|---|
| Text encoder (Qwen2.5-VL) — one-shot at prompt time | out of hot loop | prefill-like, amortised across 20 steps |
| DiT forward (cond) | ~40-45% per step | 60 blocks × (QKV-proj, RMSNorm, RoPE, joint attn, GELU-FFN, 2×LayerNorm, modulation) |
| DiT forward (uncond) | ~40-45% per step | second CFG pass, currently un-batched |
| Scheduler (Euler flow) | < 1% | small elementwise |
| VAE decode | 10-25% of total wall (amortised once) | GroupNorm + Conv + AttnBlock — not per-step |

A reasonable Ascend 910B4 ggml-cann baseline guess, extrapolating from
the README's 910B2 Q8 number (1.59 s/step) by the 2× HBM bandwidth gap
(1600 → 800 GB/s) and the Q4 vs Q8 weight-bandwidth gain (~1.5×), is
**~2-3 s/step** at 512×512 on Q4 — if the backend worked. This number
is a placeholder until Q1 green; the contract must not stake any gate
on it.

## Part 2 — TTS playbook transfer (what applies, what doesn't)

| TTS lever | QIE applicability | Expected Δ (realistic after ÷5 discount) | Notes |
|---|---|---|---|
| **W1 NPU lm_head port** | **❌** | — | QIE has no per-frame lm_head; DiT output projection runs once per step, not per token. No cross-layer port analog. |
| **W3b AddRmsNorm fusion** | **⚠️ partial** | +0.5-1% per step | Contract claimed "DiT uses LayerNorm"; source reveals **RMSNorm on norm_q / norm_k / norm_added_q / norm_added_k** = 4 sites × 60 blocks × 20 steps × 2 (CFG) = 9600 RMSNorm calls per image. Add-RmsNorm fusion applies on residual+norm pairs inside attention. Block norms (img_norm1/2, txt_norm1/2) are LayerNorm without affine — Add-LayerNorm fusion (different vendor op) applies there. |
| **G2 aclGraph step-keyed** | **✅ IDEAL FIT** | **+15-35% per step** | This is where QIE beats TTS. Shape is stable per step (same latent size, same token count — unlike TTS where pos varies). **~20 graphs total, one per step**, not 17-128 pos-keyed. Dispatch floor is large (hundreds of per-block dispatches) and no TQE=2 amortisation yet. Biggest single structural lever. |
| **A.1 InplaceAddRmsNorm** | **⚠️ partial** | +0.2-0.5% | Same applicability as W3b — attention's QK-norm sites. Modest because DiT doesn't have TTS's 15-forward-per-frame multiplier; only 20 steps. |
| **M3'new' pos 0+1 batch** | **❌** | — | No autoregressive "pos" dimension in diffusion. Each step is independent. No analog. |
| **WSPOOL async-safe retain** | **✅** | correctness infra | Transfer verbatim. DiT workspace at large batch (CFG=2, 4096-token seq) is non-trivial; retain-list bound at depth-8 prevents pool churn. |
| **W8 quant (aclnnWeightQuantBatchMatmulV3)** | **✅ contingent on Q1.1 unblock** | **+30-50% per step** | DiT is matmul-heavy (60 blocks × 8-10 matmul sites). If Q1.1 lands Q4_K support in ggml-cann and we wire native WQBMMv3 for Q4, this is the headline HBM-bandwidth lever. Risk: `antiquantGroupSize` parameter at per-group INT4 is documented but we haven't probed it for image workloads. |
| **GMM-QKV grouped matmul** | **✅** | +0.3-0.8% per step | `QwenImageAttention` has **to_q, to_k, to_v on img + add_q_proj, add_k_proj, add_v_proj on txt = 6 projections per block × 60 blocks × 20 steps × 2 = 14400 matmul calls.** Grouping into 2 triple-projections per block (img-QKV, txt-QKV) per `qkv_grouped_probe_verdict.md` pattern halves dispatch count on those sites. Smaller lever than TTS because DiT matmuls are larger (hidden=3072) so dispatch overhead is lower relative to compute. |
| **Path C hand-written AscendC** | **❌** | — | Closed in TTS on drift. Don't re-open. |
| **CannFusion A16W8** | **❌** | — | Upstream PR not shipped. Same blocker as TTS. |
| **aclnnFFNV3** | **❌** | — | QwenImage DiT FFN is `Linear → GELU → Linear` (no gating, no SwiGLU). FFNV3 is MoE-only for INT8 and SwiGLU-activation-coded. Wrong shape entirely. |
| **aclnnApplyRotaryPosEmbV2** | **⚠️ layout probe needed** | +0.5-1.5% per step | Every DiT block applies RoPE to Q and K (`Rope::apply_rope` call on both, line 655-656 of `rope.hpp`). V2 fuses Q+K into one dispatch. Current code uses 2 ops × 60 × 20 × 2 = **4800 RoPE calls per image**. BUT — QIE's RoPE is **non-standard**: 3D axial decomposition with `axes_dim = {16, 56, 56}` and shape-wise rope-interleaved mode. V2 probed GREEN on Qwen3-TTS's 1D GQA-mismatched case but FAILED on GQA packed-UB. QIE is MHA (no GQA: `num_heads=24`, no kv heads difference — it's joint-attention not GQA), so the V2 GQA bug doesn't hit; the 3D-axial layout is the new probe question. |
| **aclnnSwiGLU** | **❌** | — | DiT uses GELU. SwiGLU probe was YELLOW on rounding-order anyway. |

### Transfer summary

Seven of the TTS playbook's 13 catalogued levers transfer partially or
fully. The largest single transfer is **W8 quant + G2 aclGraph + GMM-QKV
stacked**, which is the clean path to 2× baseline — but all three
require Q1 unblocked first, and W8 specifically requires Q1.1 to land
Q4_K in `ggml_cann_mul_mat`.

## Part 3 — Diffusion-specific levers (NOT in TTS playbook)

New territory. None of these have probes on Ascend yet; every one needs
a 30-minute probe before scoping — codify as Gate 0 for the contract.

### 3.1 CFG batching — the headline lever

**What**: stable-diffusion.cpp currently runs `out_cond` and `out_uncond`
(and `out_img_cond` for edit, and `out_skip` for SLG) as **sequential
forward passes** per step. Each is a 60-block DiT forward. Batching
cond+uncond into a single batch=2 forward saves one full forward per
step — nominally 50% of DiT wall.

**Expected**: optimistic +80-100% steps/s per step (because
pre-optimisation DiT dominates total wall). Realistic after discount:
**+40-60%**. Real-world batched forward is not exactly 2× over two
sequential forwards because matmul wall at batch=2 is ~1.3-1.5× not 1×,
depending on how close to memory-bound the shape sits.

**Risk**:
- **Asymmetric cross-attention** in some DiT edit variants (cond has
  ref-image tokens concatenated, uncond doesn't). Source check:
  `qwen_image.hpp:454-459` concats `ref_latents` into the img token
  stream — this means **cond and uncond may have different token counts
  in edit mode**, breaking naive batch=2. Mitigation: verify shape
  symmetry at Q0.5 before dispatch.
- **Modulation broadcast**: each block uses 6-way-chunked modulation
  tensor derived from timestep; per-step across cond/uncond this can
  be shared or duplicated. Needs sanity check.
- **Workspace doubles**: batch=2 doubles activation HBM per step.
  Current Q4 weights leave 12 GB; activation budget at 4096 img tokens
  × 3072 hidden × batch=2 × 60 blocks is ~few hundred MB-ish
  (intermediate-retention-dependent) — safe but needs measurement.

**Probe-first**: 30 min to verify cond/uncond token-count symmetry in
QIE-2511 edit mode. If asymmetric, CFG batching is a 1-week refactor,
not a drop-in.

### 3.2 Step distillation — the nuclear option

**What**: LCM / TCD / Hyper / SDXL-Lightning schedulers distill 20-step
denoising to 4-8 steps. External ML work — requires a distilled
checkpoint of the base model. Community has distilled Qwen-Image-T2I
variants (4-step, 8-step); QIE-Edit-2511 distillations are
**not-yet-public** as of release date (Nov 2025, 5 months ago).

**Expected**: 3-5× end-to-end if checkpoint exists. Zero if it
doesn't.

**Risk**: high — depends entirely on checkpoint availability. Budget =
0 engineering days on our side if we wait for community; 2-4 weeks if
we distill ourselves (needs training infra we don't have for image
models on Ascend).

**Recommendation**: **out of scope** for initial contract (as the
contract itself states). Revisit at Q8 if the 2-3× gate isn't met.
Monitor HF for community distillations — if one lands, drop-in at zero
engineering cost.

### 3.3 EasyCache / UCache / DBCache / CacheDIT — already in source

**What**: `stable-diffusion.cpp:1761-1864` shows 4 cache modes already
wired (EasyCache, UCache, DBCache, TaylorSeer CacheDit). These reuse
intermediate-layer outputs across consecutive steps when residuals are
below a threshold — a **step-axis** optimization similar in spirit to
speculative-decode for LLMs. DBCache can skip 30-50% of block computes
at 0.1-threshold with ~1-2% FID cost; TaylorSeer extrapolates across
skipped steps.

**Expected**: **+30-60% per-image** if thresholds tune cleanly. Needs
per-model calibration (threshold too tight → no wins; too loose →
visible artefacts at edit boundaries).

**Risk**: medium — already in code, already gated by `sd_cache_params_t`.
**No engineering work**, only tuning + eye-gate validation. This is
likely the biggest "free" win in the contract — most of the
implementation is done; the remaining work is threshold calibration on
the 20-task canonical edit suite.

**Recommendation**: insert as Q2.5 or Q3.5 — it's a small probe with
large upside. Start with DBCache + TaylorSeer at default thresholds on
the 20-task suite; measure FID delta and wall delta.

### 3.4 Attention @ image-token sequence (FIAv2/v3/v4)

**What**: image-token attention in QwenImage is joint txt+img with
sequence length ~4096 for 64×64 latent (4096 img + ~256 txt ≈ 4352).
`rope.hpp:658` currently calls `ggml_ext_attention_ext` with
`flash_attn_enabled` toggle — this is the ggml-native attention path,
not FIAv2.

**Expected**: wiring FIAv2 (or V3/V4 per `fia_v34_probe_verdict.md`) on
this sequence length vs ggml's naive softmax is **+10-25% per
attention**, which is ~40% of step wall, so **+4-10% per step
realistic**.

**Risk**: low — FIAv2 probed byte-identical at TTS's small seq; V3/V4
forward-compatible. The unknown is whether FIAv2's perf at seq=4352
holds up vs Flash-Attention-style kernels (seq=4k is the target regime
for Flash — exactly where it wins most). **Probe at Q0.5**: measure
FIAv2 wall at seq=4352 vs ggml's attention on ac02. If within 20%, wire
it; if >30% slower, this becomes a CANN ask.

### 3.5 VAE decode fusion

**What**: VAE decode is GroupNorm + Conv2d + AttnBlock (conv-based
QKV attention at mid-layer). Source `vae.hpp` shows GroupNorm32 is
used throughout; Conv2d uses `aclnnConvolution` directly per
README §1. Fusion opportunities: GroupNorm + SiLU + Conv
(`aclnnGroupNormSilu*` family exists per contract Q7.2).

**Expected**: +5-15% VAE decode wall = +0.5-3% total image wall
(since VAE is 10-25% of end-to-end).

**Risk**: medium — CANN GroupNormSilu family exists but we haven't
probed its dtype matrix for the GroupNorm32 (32 groups, bf16)
specific instantiation.

### 3.6 Mixed precision for VAE / encoder paths

**What**: text encoder + VAE can often run in bf16 or f16 without
visible quality loss, while DiT body needs f32 accumulate for stable
flow-matching. `README.md` currently mandates `GGML_CANN_QUANT_BF16=on`
because **FP16 accumulate overflows** on K=12288 matmuls (activation
~131, sum > 65504 → NaN).

**Expected**: +5-10% on mixed-precision paths.

**Risk**: medium — bf16 is already default. Further mixed-precision
requires careful per-layer calibration. Not a day-1 lever.

### 3.7 Step-graph persistent (speculative)

**What**: capture **the entire 20-step denoising loop** as a single
aclGraph, replay end-to-end. Would eliminate all per-step host-side
dispatch cost.

**Expected**: +5-10% dispatch floor if per-step is still dispatch-bound
after G2 lands. After G2, probably +1-3%.

**Risk**: high — speculative on Ascend; nobody has captured a
multi-step-with-scheduler aclGraph. Scheduler's `sigma` is
step-index-dependent — needs parameterisation in-graph, which aclGraph
may or may not support at this granularity.

**Recommendation**: park as Q8+ stretch.

### 3.8 RoPE pre-compute (import from MLX)

**What**: `qwen-image-mlx/docs/OPTIMIZATION_TECHNIQUES.md` §1: RoPE
was computed **inside** the transformer forward, recalculated
`60 blocks × 20 steps × 2 CFG = 2400 times per generation`. Moving
it out of the loop delivered **20-40% wall reduction** on M4 Max
— MLX's largest optimization.

**On Ascend**: source `qwen_image.hpp:544-561` shows `pe_vec =
Rope::gen_qwen_image_pe(...)` called **inside `build_graph`** (per
step). Lifting to session-init is a straightforward refactor.

**Expected**: +10-25% per step realistic. This is directly measured on
MLX; the ggml-cann code path has the same structural issue.

**Risk**: low. Architecture identical to MLX; fix is structural.

**Recommendation**: **must-have**, wire at Q1 or Q2 as a free win.

## Part 4 — Ranked optimization opportunities (honest ROI)

After 3-10× optimism discount per the TTS learnings meta-finding.

### Ranking

| Rank | Lever | Realistic Δ | Effort | Risk | Gate dep |
|---|---|---|---|---|---|
| 1 | **Q1 backend unblock** (get_rows, mul_mat, gather_v3) | — (required, not a lever) | 1-3 weeks | medium (gather_v3 is a schedule risk) | none — blocks everything |
| 2 | **Q2 aclGraph step-keyed** | +15-35% per step | 1-2 weeks | low — shape-stable fits aclGraph ideally | Q1 |
| 3 | **RoPE pre-compute** (MLX import) | +10-25% per step | 2-3 days | low — structural refactor, well-understood | Q1 |
| 4 | **Q4 CFG batching** | +40-60% per step | 1 week (+ asymmetry probe) | medium — edit-mode token asymmetry risk | Q1, Q2 |
| 5 | **W8→Q4 quant via WQBMMv3** | +30-50% per step | 1-2 weeks | medium — Q4 eye-gate quality is new on Ascend | Q1.1 (Q4_K lane) |
| 6 | **CacheDIT/DBCache tuning** | +30-60% per-image (step skip) | 3-5 days | low-medium — already in code, needs threshold calibration | Q1 |
| 7 | **FIAv2/V3 attention** at seq=4352 | +4-10% per step | 3-5 days | low — probe dictates | Q2 |
| 8 | **QKV grouped matmul (6→2 per block)** | +0.5-1% per step | 3-5 days | low — direct TTS transfer | Q2 |
| 9 | **RMSNorm fusion on QK-norm sites** | +0.5-1.5% per step | 3-5 days | low — direct TTS transfer | Q2 |
| 10 | **RoPE-V2 fused Q+K** | +0.5-1.5% per step | 2-3 days + probe | low-medium — 3D axial layout untested | Q2 |
| 11 | **Q7 VAE GroupNormSilu fusion** | +0.5-3% total wall | 1 week | medium — CANN op matrix untested for VAE dtypes | Q1 |
| 12 | **Step distillation** (external) | 3-5× if checkpoint exists | 0 days (community) or 2-4 weeks (ours) | high — checkpoint availability gate | none |
| 13 | **Full-loop aclGraph** | +1-3% | 1-2 weeks | very high — speculative | Q2 |

### Stacking math (realistic, after discount)

Assume Q1 unblocked, ggml-cann baseline = 1× (placeholder, TBD at Q1).

- Q2 aclGraph step-keyed: ×1.20
- RoPE pre-compute: ×1.15
- Q4 CFG batching: ×1.45 (mid of 40-60%)
- W8→Q4 WQBMMv3: ×1.35
- CacheDIT at threshold 0.08 (conservative): ×1.25 on end-to-end
  (not per-step — reduces step count effectively from 20 to ~15)
- QKV+RMSNorm+RoPE-V2 small-dispatch cluster: ×1.03 combined

**Multiplicative stack**: 1.20 × 1.15 × 1.45 × 1.35 × 1.25 × 1.03
= **3.5×** end-to-end over a (notional) Q1-green ggml-cann baseline.

**Hitting the contract's 2-3× gate is realistic with margin.** Stretch
to 3× (the contract's ceiling) is plausible with all levers. 4× would
require CacheDIT to tune aggressively plus a non-trivial FIAv2 win, or
the step-distilled checkpoint appearing mid-contract.

## Part 5 — Unique challenges vs TTS

### 5.1 Subjective quality gate

TTS moved from WER-gated (automatable) to ear-gated (subjective) and
we learned to budget time for human review. ASR restored WER
automation. QIE goes **back to eye-gate**, 20-task canonical edit
suite, human review per milestone. **Budget 2 hours of human
attention per milestone minimum** for CN+EN task set. Codify in
contract M8 gates.

### 5.2 Ref-image conditioning asymmetry

`qwen_image.hpp:454-459` shows `ref_latents` are concatenated into the
**img token stream** inside `QwenImageModel::forward`. Uncond path in
`stable-diffusion.cpp:1884-1906` uses `uncond.c_crossattn` but the
ref-latent path is shared. This means **edit mode breaks the naive
"cond and uncond forward pass have identical shapes" assumption** that
CFG batching depends on. We need to verify whether the ref-image
tokens are included in both forwards (they should be — it's a
conditioning hint, not a condition itself), but this must be **probed
at Q0.5**, not assumed.

### 5.3 VAE is non-trivial slice (not in TTS's hot loop)

TTS had no VAE. QIE has VAE encode (once per ref-image) and VAE decode
(once per generated image). VAE decode at 1024×1024 can be
10-25% of end-to-end wall; at 512×512 it's 5-15%. The TTS playbook has
no VAE-decoder-fusion pattern; Q7 is genuinely new territory.

### 5.4 Two separate "encoder" heavy paths

TTS had mel encoder (audio → embed) and text encoder. QIE has:
- **Qwen2.5-VL vision encoder** (ref image → tokens via mmproj) —
  where the gather_v3 bug lives
- **Qwen2.5-VL text encoder** (prompt → embed, 28-layer transformer)

Both are one-shot prefill but together consume ~10% of end-to-end wall
at 20-step inference. Not-hot-loop, but not negligible. TTS's W1
cross-layer port analog (moving CPU-side computation to NPU) does not
apply here because these encoders are already NPU-resident; the issue
is dispatch coverage, not host↔device boundary.

### 5.5 32 GB mandate is harder than TTS's 32 GB

TTS fit F16 comfortably in 32 GB (model ~14 GB). QIE at Q4 is 18-20 GB;
at Q8 it's 36 GB which **does not fit 910B4**. No F16 fallback for
quality bisection like we had on TTS — every quality debug cycle runs
at Q4 or we're off-device. If Q4 eye-gate fails, the fallback is
procuring 910B3 (64 GB) and running Q8, which is procurement + host
migration + rewire (ac03 is ASR, ac01 is TTS, we'd need ac04/05).
Treat as hard-kill risk in Q8.

## Part 6 — Transferable patterns for future diffusion workloads

QIE is the first diffusion workload in the fleet. Patterns we codify
here serve SD3, Flux, Flux-Kontext, Z-Image, Anima, Wan2 (all
already in `tools/ominix_diffusion/src/*.hpp`) and future
Qwen-Image-Edit-2512 / -2601 variants.

### Diffusion-generic pattern library (to establish)

1. **Step-keyed aclGraph** is the canonical diffusion capture pattern.
   ~N graphs where N = step count (typically 4-30). Far simpler than
   LLM pos-keyed capture. Budget ~5-10 MB HBM per graph × N steps =
   ~100-300 MB. Capture at session init, replay per step. **Promote
   to a reusable `DiffusionAclGraphEngine` template once QIE Q2
   lands.**
2. **CFG batching** is the universal +40-60% lever for guidance-based
   diffusion. Shape-symmetry probe is prerequisite. For edit variants
   with ref-latent concat, extra shape-check at contract gate.
3. **Cache-family step-skip** (EasyCache, UCache, DBCache, TaylorSeer)
   is already modular in `cache_dit.hpp` / `ucache.hpp` / `easycache.hpp`.
   Every diffusion variant inherits this for free — **calibrate
   per-model thresholds, not per-contract re-implementation**.
4. **RoPE pre-compute outside step loop** is a structural refactor
   applicable wherever RoPE is computed per-forward. MLX has measured
   this at +20-40%; ggml-cann will see similar.
5. **VAE is a separate optimization surface.** GroupNorm + Conv + mid-
   attention. Treat as its own Q-phase; don't bundle with DiT-body work.
6. **Quality gate = user-eye on a 20-task canonical suite.** Codify the
   suite once (CN+EN × 10 edit categories) and reuse across Flux /
   Qwen-Image / Z-Image contracts. Save the 2-hour-per-milestone
   subjective review cost from re-defining per contract.
7. **K-quant coverage in ggml-cann backend** is a shared dep across all
   diffusion models. The 3 Q0-v2 bugs are not QIE-specific — they block
   SD3-Q4, Flux-Q4, Z-Image-Q4 too. Fixing them once pays off across
   the whole diffusion fleet.

### Contract-structure template for diffusion workloads

Copy QIE's Q0/Q1/Q2/Q4/Q7/Q8 structure to future diffusion contracts.
Q0 = discovery + baseline; Q1 = backend unblock (will often be needed
if quant is new); Q2 = step-keyed aclGraph (almost always the first
lever); Q4 = CFG batching; Q7 = VAE; Q8 = eye-gate. Skip/compress
depending on pre-existing work.

## Part 7 — Meta-process improvements

Inherit from TTS; add QIE-specific items.

### Inherited from TTS playbook (re-state for clarity)

1. Noise-band measurement at contract start (N≥5 stock runs, std
   reported).
2. Vendor-op catalog grep at Gate 0.
3. Probe-first mandate on every fused-op substitution (30 min standalone
   test before 2-week integration).
4. Projection math: agent estimate ÷ 5 is the planning number.
5. Live-result gate is non-negotiable; offline parity is necessary but
   not sufficient.
6. Patch-file push is the standard workflow (ac0N → Mac PM).
7. No Co-Authored-By on commits.

### New for QIE / diffusion

8. **Eye-gate sample suite is frozen at Q0.** Define the 20-task
   CN+EN canonical edit set once at Q0, lock image+prompt, reuse
   unchanged through Q8. Deviating mid-contract poisons comparability.
9. **Cond/uncond shape-symmetry probe is a prerequisite for CFG
   batching**, not an implementation detail. Dispatch at Q0.5 before
   Q4 scopes the work.
10. **Step-keyed aclGraph capture is at init, not lazy on first-touch.**
    Same lesson as TALKER_CANN_GRAPH scaffold — lazy capture on
    diffusion is an anti-pattern (each step is touched once per image,
    same as each pos in Talker).
11. **MLX optimization doc is required reading before Ascend work.**
    `OminiX-MLX/qwen-image-mlx/docs/OPTIMIZATION_TECHNIQUES.md` already
    documents RoPE pre-compute, timestep embedding caching, step
    reduction, and mixed precision — all directly applicable. Budget
    2 hours per QIE-agent onboarding for MLX-doc read-through.
12. **ggml-cann backend bugs are vendor asks once patched.** The 3
    Q0-v2 bugs become upstream PRs once Q1 lands. Codify PR-to-upstream
    as part of Q1 delivery, not a separate contract.
13. **Quality regression is HARD-KILL, not SOFT-KILL.** TTS's
    ear-gate was sometimes ambiguous; QIE's eye-gate on edit tasks is
    binary (did the edit happen correctly, yes/no). If a gate trip,
    the lever rolls back immediately — no "maybe the next run will be
    better" slow-death.

## Part 8 — Vendor-side asks (ggml-cann backend + CANN fused-op gaps)

Per-priority:

### 8.1 ggml-cann backend (llama.cpp Ascend) — REQUIRED for any Ascend diffusion workload

1. **Q4_K / Q5_K / Q6_K dispatch in `ggml_cann_mul_mat`**
   (`aclnn_ops.cpp:2670`). ~80-150 LOC, mirrors existing Q8_0/Q4_0
   branches.
2. **Q4_0 / Q4_1 dispatch in `ggml_cann_get_rows`**
   (`aclnn_ops.cpp:2272`). ~30-50 LOC.
3. **Fix `ascendc/gather_v3` in Qwen2.5-VL vision encoder wiring.**
   Likely a dtype-tag bug in the graph-builder pre-gather, but needs
   the Q1.3 probe to localise.

All three land as upstream PRs against
`ggerganov/llama.cpp/ggml/src/ggml-cann/`. Co-sign from ymote fork,
standard upstream review cycle. **These are free-standing community
contributions with no CANN-vendor ask — just open-source elbow grease.**

### 8.2 CANN fused-op gaps specific to diffusion

1. **`aclnnGroupNormSiluConv2d` (UNet / VAE epilogue)**. Doesn't exist
   as a single op in 8.3.RC1 per Q7.2 audit; vendor has
   `aclnnGroupNormSilu` but not the Conv pair. UNet / VAE decode path
   would benefit from `GroupNorm → SiLU → Conv2d` single-dispatch.
2. **`aclnnFusedAttentionWithRope2D` for image-token attention.** DiT
   attention has (self-attn on joint txt+img with 2D-RoPE on img
   portion, 1D on txt portion). Current code composes RoPE-apply +
   FIA as two ops. A fused kernel targeting **seq=4k-8k 2D-RoPE +
   MHA** is the diffusion analog of DeepSeek's MLA kernel — a
   workload-specific vendor ask.
3. **`aclnnCFGBatchedDiTStep`.** Ultra-specific: cond+uncond batched
   DiT-block fused op with CFG weighted output. Unlikely to exist;
   asking vendor for it establishes the diffusion workload as a
   first-class target for CANN's op catalog.
4. **`aclnnTimestepEmbed`.** Sinusoidal embedding with cached
   frequencies. Tiny op, but pure elementwise — currently composed as
   several aclnnMul + aclnnExp calls per step. Not high leverage,
   mentioned for completeness.

Priority 1 is the realistic ask in the CANN-team channel; 2 and 3 are
"diffusion-workload-class" signals worth raising at roadmap reviews
but unlikely to land in 8.5 or 8.6.

### 8.3 Inherited asks from TTS learnings (still valid for QIE)

5. **Open-source `libruntime.so`** — every EZ9999 error on QIE will hit
   the same debug-time tax TTS hit.
6. **Local CANN SDK for macOS + Linux dev** — QIE agent onboarding will
   again start with "ssh ac02" just to grep headers. Mac-local SDK
   would collapse audit time.
7. **English-first op documentation + capability-tag catalog** — FIAv2
   at image-token seq lengths, WQBMMv3 at `antiquantGroupSize` per-group
   INT4, GroupNormSilu dtype matrix — all currently require in-header
   grep. Curated index with per-op capability matrix (MHA / GQA / MLA /
   training-only / inference-only / dtype-matrix / MoE-gated / non-MoE)
   remains the top-leverage ecosystem move.

## Part 9 — Realistic ceiling estimate

### Correcting the MLX parity framing

**The "Apple M4 Max = 30 fps" figure in the QIE-LEARN prompt is wrong.**
`OminiX-MLX/qwen-image-mlx/docs/STEP_REDUCTION_RESULTS.md` directly
measures M4 Max at optimized state:

- 20-step inference: 80 s diffusion wall = **~4.0 s/step**
- End-to-end including 25 s overhead (model load + VAE decode): 105 s
  total = **~0.01 fps at image level**, or ~0.25 step/s

Per-step rate 0.25 step/s (M4 Max optimized) — not 30 fps — is the real
Apple benchmark. The 30-fps figure refers to a different workload (text
encoder throughput or a smaller/quantized model variant). **Our Ascend
910B4 target is NOT 1/30th of Apple; it's competitive with Apple.**

### Bandwidth-bound envelope on 910B4

- HBM bandwidth: 800 GB/s
- Q4 DiT weights: ~11-13 GB per full pass
- Memory-bound single-forward floor: 13 GB / 800 GB/s ≈ **16 ms** at
  100% BW utilisation
- Realistic 60-80% BW utilisation post-optimization: ~20-27 ms per
  forward
- CFG batched (effective 1 forward per step): 20-27 ms/step floor
- End-to-end 20 steps: ~0.4-0.5 s diffusion + VAE + encode =
  **~1-2 s/step effective at best-case, ~8-10 s end-to-end**

### M4 Max bandwidth envelope for comparison

- M4 Max bandwidth: 546 GB/s
- Q4 weights similar: ~11-13 GB
- Memory-bound floor: ~24 ms per forward at 100% BW
- Observed: 4 s/step — **dispatch/compute-bound**, not memory-bound
  (170× gap from memory floor indicates significant compute or dispatch
  overhead)

### Honest ceilings

| Scenario | Est. step wall | vs ggml-cann baseline (notional 2-3 s/step) | vs M4 Max (4 s/step) |
|---|---|---|---|
| ggml-cann baseline (post Q1 unblock) | 2-3 s/step | 1× | 1.3-2× faster |
| + Q2 aclGraph | 1.6-2.5 | 1.2× | 1.6-2.5× |
| + RoPE pre-compute + RMSNorm/RoPE-V2 stack | 1.4-2.2 | 1.4× | 1.8-2.9× |
| + CFG batching | 0.9-1.5 | 2.0× | 2.7-4.4× |
| + W8→Q4 WQBMMv3 | 0.7-1.1 | 2.7× | 3.6-5.7× |
| + CacheDIT (step-skip effective) | 0.5-0.9 per effective step | 3.5× end-to-end | 4.4-8× |
| **+ step distillation (if checkpoint lands)** | 0.5-0.9 × (20/4) | 7-15× | 8-30× |

**Realistic landing** (without step distillation): **3-3.5× over raw
ggml-cann baseline, 3-5× over M4 Max optimized**. Meets the contract's
2-3× gate with margin. 4× is plausible with aggressive CacheDIT
tuning.

**Step-distillation is the unbounded upside** — if a Qwen-Image-Edit
distilled variant lands publicly mid-contract (it happens on popular
models; SD3 got Hyper variants within months), we're in the 10-15×
regime.

## Closing

QIE is a genuinely different workload from TTS/ASR and the contract
benefits from this pattern-level analysis before agent dispatch. Key
corrections surfaced:

1. **Three ggml-cann backend bugs block everything** — Q1 is
   mandatory backend-unblock, not optimization.
2. **Architecture is not what the contract brief assumed** — RMSNorm
   DOES exist on Q/K norms, partially rescuing TTS fusion levers.
3. **MLX parity framing was off by 100×** — Apple M4 Max at ~4 s/step
   is matchable, not a far-distant ceiling.
4. **The biggest novel levers are CFG batching and CacheDIT** —
   neither has a TTS analog. Both are realistic and well-scoped.
5. **Step distillation is the free nuclear option** — if a distilled
   checkpoint lands mid-contract, end-to-end quintuples at zero
   engineering cost.

The 8-12 week Q0-Q8 timeline is honest for a 2.5-3.5× lift. Extending
to a 4× target needs CacheDIT-tuned plus all sub-fps levers stacked;
extending to 10× needs the step-distilled checkpoint landing. **The
one schedule risk worth pricing in is Q1.3 (gather_v3) — budget 3
weeks, not 1-2.**

---

**Future diffusion-workload inheritance defaults** (for SD3, Flux,
Z-Image, future QIE variants):

- Step-keyed aclGraph is the canonical capture pattern.
- CFG batching is the headline lever after backend unblock.
- RoPE pre-compute is a structural free win (MLX measurement applies).
- Cache-family step-skip (DBCache / CacheDIT / EasyCache) is already
  modular — tune, don't re-implement.
- Quality gate = frozen 20-task canonical eye-suite, reused across
  diffusion contracts.
- Q4 on 32 GB + f32 compute with bf16 accumulate is the default recipe.
- K-quant coverage in ggml-cann is a shared dep — fix once, benefit
  across the fleet.
