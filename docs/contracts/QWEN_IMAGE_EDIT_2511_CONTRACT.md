# Qwen-Image-Edit-2511 Optimization on Ascend 910B4 (ac02)

## 1. Status & mandate

**Status**: AMENDED 2026-04-22 per `docs/qie_optimization_learnings.md` feasibility deltas (Q0.5 inserted, Q1 widened, Q2.5 inserted). Original draft 2026-04-22, PM signed.
**Target**: optimize Qwen-Image-Edit-2511 image-editing inference on Ascend NPU, starting with 910B4 (32 GB HBM constraint — Q4 quant mandatory) via `tools/ominix_diffusion/` as the starting surface.

**Goal** (subject to Q0-v2 calibration):
- **A) First-landing arc**: current `ominix_diffusion` ggml-cann baseline → native `ImageDiffusionEngine` with aclGraph step-keyed capture + playbook transfer
- **B) Steps-per-second lift**: target 2-3× over ggml-cann baseline at Q4 quant
- **C) Real-number cadence**: README claims 20 steps / 32.04 s on 910B2 Q8 (= 1.6 s/step). 910B4 Q4 expected 2-3× slower by HBM ratio (1.6 TB/s → 800 GB/s); post-optimization target ≈ 0.8-1.2 s/step

**PM role**: supervise, gate, arbitrate. PM does not write kernel code. Every milestone is an agent deliverable with numeric gates.

## 2. Background

Qwen-Image-Edit-2511 is Qwen team's image-editing model (Dec 2025 release). Architecture is DiT (Diffusion Transformer) or UNet-based diffusion with reference-image conditioning.

**Sizing constraint (critical)**:

| Variant | Size | 910B4 fits? | 910B3/B2 fits? |
|---|---|---|---|
| BF16 | 57.7 GB | ❌ | ❌ (barely over) |
| 8-bit | 36.1 GB | ❌ | ✅ |
| **4-bit** | **25.9 GB** | **✅** (margin ~6 GB) | ✅ |

**Mandates 4-bit quant on 910B4** from day one.

**Prior art in OminiX ecosystem**:
- `tools/ominix_diffusion/` — structural port of stable-diffusion.cpp via ggml-cann (SD1.5 / SDXL / SD3 / Flux / Qwen-Image). Single commit `a9521b51` — NOT iteratively optimized. `conditioner.hpp:31` has `std::vector<sd_image_t*> ref_images = {};  // for qwen image edit` — ref-image hook present.
- `qwen-image-mlx` (OminiX-MLX) — Apple Silicon counterpart, text-to-image (not edit), supports BF16 + 8-bit + 4-bit variants
- MLX M3 Max hits 30 fps on qwen-image T2I — hard upper bound on what to expect on Ascend if hardware-bounded by memory bandwidth

**Why this is different from TTS workload**:
- **Shape-stable per denoising step** — big win for aclGraph capture (capture once per step, replay ~20 times per image)
- **Image-token attention** (sequence length 4096 for 64×64 latent) vs TTS's tiny 1-16 seq len
- **No RmsNorm fusion** — DiT uses LayerNorm or GroupNorm, not RmsNorm → W3b doesn't transfer
- **CFG (classifier-free guidance)** — 2 forwards per step (cond + uncond). **Batchable → ~2× savings per step** if wired
- **VAE encode/decode** — non-trivial cost (10-25% of total wall, not in DiT hot loop)
- **Quality gate = user-eye** on edit output, not WER/audio. Subjective again after ASR's automatable WER.

## 3. Scope

**In scope**:
- Q0-v2 discovery (in flight on ac02)
- Q1 native `ImageDiffusionEngine` bring-up — escape ggml-cann hot path
- Q2 aclGraph step-keyed capture (per-step, ~20 graphs per image)
- Q3 DiT attention FIAv2 wiring (image-token long-sequence attention)
- Q4 CFG batching (2 forwards/step → 1 batch=2 forward)
- Q5 Mm and QKV grouped matmul (if applicable to DiT — same op family we just landed for TTS)
- Q6 Weight upload + session-lifetime optimization (DiT weights large; minimize reload)
- Q7 VAE optimization (decode-phase; GroupNorm/Conv fusion)
- Q8 User-eye + steps/s HARD gate

**Out of scope**:
- Step distillation (LCM / TCD / Hyper — would cut 20 → 4 steps = 5× speedup but needs distilled checkpoint; separate contract)
- Full-replace VAE with alternative decoder — external ML work
- Text encoder optimization — one-shot prefill, not in hot loop (≤ 10% of wall)
- Multi-image batch inference — contract scopes single-image edit
- Post-processing / super-res — separate pipeline

## 4. Host plan

- **Primary**: ac02 (port 31210, 910B4, 32 GB HBM, CANN 8.3.RC1). Per PM directive "use different Ascend server" for QIE vs TTS (ac01) and ASR (ac03).
- **Secondary**: could extend to ac02 specifically; ac01 and ac03 have TTS/ASR ownership.
- **Fork**: `ymote/OminiX-Ascend` `main`. Patch-file push mechanism.
- **Reference dataset**: 20 canonical edit tasks (source image + edit prompt) covering: color-change / style-transfer / object-add / object-remove / text-add / background-swap / weather-change / time-of-day / pose-change / composition-change. Mix of CN / EN prompts. Human eye-check at gate.

## 5. Workstreams with gates

### Q0-v2 — Discovery + baseline — **DONE (YELLOW)**

Per `docs/qie_q0v2_discovery.md` (verified-by Agent Q0-v2, 2026-04-22, ac02):
- ✅ ac02 confirmed 910B4 / 32 GB HBM / CANN 8.3.RC1 idle
- ✅ `ominix_diffusion` has full QIE support (`QwenImageEditPlusPipeline` + `-r/--ref-image`)
- ✅ **Weights fit at Q4 — 18-20 GB used, 12 GB margin.** 32 GB HBM is NOT the blocker. Earlier "procure 910B2 for 64 GB" framing was wrong.
- ❌ **All 3 baseline runs crashed** — 3 ggml-cann backend bugs block zero-image output:
  1. `ggml_cann_mul_mat` (`aclnn_ops.cpp:2670`): no Q4_K/Q5_K/Q6_K support
  2. `ggml_cann_get_rows` (`aclnn_ops.cpp:2272`): no Q4_0/Q4_1 support — text-encoder embedding abort
  3. `ascendc/gather_v3`: Qwen2.5-VL vision-encoder crash on float-bit-pattern indices (edit-mode specific)

**Verdict**: YELLOW. Contract **restructured** — must unblock backend before baseline makes sense.

### Q0.5 — Probe-first prerequisites (3-5 days, parallel with Q1, Mac-side OK)

Amended 2026-04-22 per `docs/qie_optimization_learnings.md` §3.1, §3.4, §2 (RoPE-V2). Three probes that MUST complete before Q4, Q3, and Q2-RoPE-V2 scope respectively. Source-reading + short runtime measurements; no full-model dispatch needed.

- [ ] Q0.5.1 **CFG cond/uncond shape-symmetry probe** (edit mode): source-read `qwen_image.hpp:454-459` + `stable-diffusion.cpp:1884-1906`. Confirm whether `ref_latents` concat applies to both cond and uncond forwards. If asymmetric → Q4 is a 1-week refactor, not drop-in. **Output**: `docs/qie_q0_5_cfg_symmetry.md` with verdict.
- [ ] Q0.5.2 **FIAv2 wall at seq=4352**: measure FIAv2 dispatch wall on ac02 at joint-attention sequence length (4096 img + ~256 txt). Compare to ggml's `ggml_ext_attention_ext` baseline. If within 20% → wire at Q3. If >30% slower → becomes CANN vendor ask, deprioritize Q3.
- [ ] Q0.5.3 **RoPE-V2 3D-axial layout probe**: QIE uses `axes_dim = {16, 56, 56}` non-standard 3D decomposition (`rope.hpp:655-658`). Verify `aclnnApplyRotaryPosEmbV2` layout compatibility OR document layout-convert cost. Retire lever if layout-convert eats savings (TTS-V2 precedent).
- [ ] Q0.5.4 **20-task canonical eye-gate suite definition**: CN+EN × 10 categories (color/style/object-add/remove/text/bg-swap/weather/time/pose/composition). Lock image + prompt + baseline output. Reuse across Q8 + future diffusion contracts. **Output**: `docs/qie_eye_gate_suite.md`.

**Gate**: 3 probe verdicts + suite locked. No fps claim. Decides Q3/Q4/Q5-RoPE-V2 scope.

### Q1 — **ggml-cann backend unblock** (1-3 weeks, HARD GATE, replaces original Q1)

Timeline widened from 1-2 weeks per `docs/qie_optimization_learnings.md` §Part 1: Q1.3 gather_v3 is an unknown-territory CANN-backend debug with no TTS/ASR probe coverage. Schedule-risk mitigation: **Q1.3 gets its own 2-4 hr localization probe before A-agent dispatch.**

Fix the 3 blockers above IN `ggml-cann` (llama.cpp Ascend backend). Upstream-worthy contribution once validated.

- [ ] Q1.1 Add Q4_K / Q5_K / Q6_K dispatch in `ggml_cann_mul_mat` (K-quant variants are HF-common for Qwen-Image; without them Q4 is stuck on Q4_0 which has its own get_rows issue)
- [ ] Q1.2 Add Q4_0 / Q4_1 dispatch in `ggml_cann_get_rows` — text-encoder embedding-table path MUST accept common quant formats
- [ ] Q1.3 Debug `ascendc/gather_v3` crash on float-bit-pattern indices in Qwen2.5-VL vision-encoder. **Step 1** (2-4 hr probe before A-agent dispatch): dump graph tensor dtypes around first gather on ac02, compare to Qwen2.5-VL MLX reference. Localize to (a) ggml-cann graph-builder dtype-tag bug → patch, or (b) genuine AscendC indices-dtype limit → CANN vendor ask. **Step 2** (patch or escalate based on Step 1).
- [ ] Q1.4 Runtime smoke: canonical edit task ("convert cat to black and white") completes without crash on ac02 at Q4. First successful baseline run.
- [ ] Q1.5 Land three backend fixes as separate upstream PRs against `ggerganov/llama.cpp/ggml/src/ggml-cann/`. These benefit SD3/Flux/Z-Image/Wan2 — codify upstream PR as part of Q1 delivery, not separate contract.

**Gate**: baseline run produces a valid output image (any quality, any steps/sec — just "doesn't crash"). Fork commit pushed with the 3 bug fixes isolated as separate commits for upstream review.

### Q2 — Native `ImageDiffusionEngine` bring-up (2-3 weeks, gated on Q1 green, previously Q1)

Mirror TTS `CpCannEngine` / `TalkerCannEngine` patterns. Build DiT decoder forward pass dispatching aclnn directly, bypassing ggml-cann.

- [ ] Q1.1 Scaffold `tools/qwen_image_edit/` directory structure (or extend `tools/ominix_diffusion/` with a native subdir)
- [ ] Q1.2 DiT block forward (self-attention + cross-attention + FFN + LayerNorm — transformer blocks at DiT's architecture config)
- [ ] Q1.3 Denoising loop (scheduler step + forward + noise-prediction application)
- [ ] Q1.4 CFG (cond + uncond batched) — start with sequential 2-call, batch in Q4
- [ ] Q1.5 VAE encode (ref image → latent) + VAE decode (latent → output image)
- [ ] Q1.6 Weight loading from GGUF (reuse llama.cpp GGUF parser for Q4)
- [ ] Q1.7 Smoke gate: generate a canonical edit ("convert cat to black and white"), verify output image is recognizable
- [ ] **Q1.8 RoPE pre-compute at session init (MUST-HAVE)** — inserted 2026-04-22 per Q0.5.3 verdict (`docs/qie_q0_5_rope_v2_layout.md`). Move `gen_qwen_image_pe(...)` out of per-step `build_graph` into session init. 3D-axial pe vector (16+56+56) pre-computed once, reused across all 20 steps × 60 blocks × 2 CFG. MLX-measured lift +10-25% per step; structural refactor, 2-3 engineer-days. **Supersedes retired RoPE-V2 lever** — V2 op cannot accept 3D-axial packed `[2,2,64,pos_len]` layout (LLM-decode op, scalar-pos + 1D cos/sin only).

**Gate**: end-to-end native-engine generates valid output image on canonical task. steps/s within ~20% of ggml-cann baseline (not a lift yet — just parity proven on native path). RoPE pre-compute verified at session init (not re-computed per step).

### Q2 — aclGraph step-keyed capture (1-2 weeks)

Diffusion is **shape-stable per denoising step** → ideal fit for aclGraph. Capture each of ~20 step-specific forwards at init, replay per step.

- [ ] Q2.1 Each denoising step has fixed shape (latent size, timestep embedding, cond vs uncond) — capture per step-index OR per (step-index, cond/uncond) combo
- [ ] Q2.2 Memory budget: 20 graphs × ~5-10 MB per graph = ~100-200 MB HBM — fits comfortably in the 6 GB margin after weights
- [ ] Q2.3 Replay path: host selects graph[step], dispatches execute, reads output
- [ ] Q2.4 WSPOOL retain-list infra transferable verbatim

**Gate**: steps/s lift **+10-30%** vs Q1 baseline (big since diffusion is dispatch-heavy per step). Byte-or-ulp parity preferred; eye-check required.

### Q2.5 — CacheDIT threshold calibration (3-5 days, FREE-ENGINEERING LEVER)

Inserted 2026-04-22 per `docs/qie_optimization_learnings.md` §3.3. `stable-diffusion.cpp:1761-1864` already wires 4 cache modes (EasyCache, UCache, DBCache, TaylorSeer CacheDIT) via `sd_cache_params_t`. **Zero kernel work** — threshold calibration + eye-gate validation only. Realistic lift **+30-60% end-to-end** by skipping 30-50% of block computes at 0.1-threshold with ~1-2% FID cost. May carry the 2× gate standalone before any Q3/Q4/Q5 kernel work.

- [ ] Q2.5.1 Measure default-threshold wall + eye-gate quality on 20-task suite for each of 4 cache modes (baseline no-cache + EasyCache + UCache + DBCache + TaylorSeer)
- [ ] Q2.5.2 Sweep DBCache threshold {0.05, 0.08, 0.10, 0.15} — find Pareto point where wall-drop ≥ 25% AND eye-gate ≥ baseline
- [ ] Q2.5.3 TaylorSeer extrapolation: pair with DBCache at best threshold; measure compound lift
- [ ] Q2.5.4 Lock best configuration as default for Q8 final numbers

**Gate**: **+25-50% end-to-end wall** vs Q2 baseline with eye-gate PASS on 20/20 tasks. Regressions on any task → revert threshold.

### Q3 — DiT attention FIAv2 (1 week, gated on Q0.5.2 verdict)

Image-token attention has sequence length 4096 (64×64 latent) — different from TTS's seq=1-16 decode. FIAv2 may or may not be optimal at this seq-len class.

- [ ] Q3.1 Audit current ominix_diffusion's attention: FIAv2? naive softmax? something else?
- [ ] Q3.2 Probe FIAv2 at image seq-lens (4096, 8192 for larger images). Compare to hand-coded or flash-style alternatives.
- [ ] Q3.3 Wire best-of-options

**Gate**: +5-15% step wall at Q3.

### Q4 — CFG batching (1 week)

Classic diffusion trick: batch cond + uncond as batch=2, one forward per step instead of 2.

- [ ] Q4.1 Batch-dimension plumbing through DiT blocks (today: bs=1 or whatever the engine assumes)
- [ ] Q4.2 CFG weighting epilogue (linear combine cond / uncond outputs) on NPU
- [ ] Q4.3 Gate: approximately **2× per-step speedup** (best-case 1.8× after batching overhead)

**Gate**: +80-100% steps/s lift at Q4 (biggest single lever for DiT workloads).

### Q5 — Mm / QKV grouped fusion (1 week)

DiT has QKV matmuls in each attention block. If TTS's GMM-wire (in flight) lands GREEN, directly transfer the `aclnnGroupedMatmulV3` pattern to DiT attention.

**RoPE-V2 retired 2026-04-22** per Q0.5.3 verdict. V2 op is LLM-decode-shaped (scalar-pos + 1D cos/sin), cannot accept QIE's 3D-axial packed `[2,2,64,pos_len]` layout. TTS precedent: 3× contract closures on V2 wiring bugs. **RoPE pre-compute** promoted to Q2.8 as must-have captures the structural value (+10-25%/step MLX-measured) without V2 risk.

- [ ] Q5.1 Wire grouped QKV matmul in DiT attention blocks
- [ ] Q5.2 Gate: +5-10% step wall

### Q6 — Weight upload + session lifetime (3-5 days)

DiT weights at Q4 = 25.9 GB. Load once at session init, keep resident. Session cost > 10 seconds is unacceptable per user experience.

- [ ] Q6.1 Measure weight-upload wall — baseline
- [ ] Q6.2 Parallelize upload across streams if beneficial
- [ ] Q6.3 Weight-format sanity: Q4 tensor layouts compatible with vendor `aclnnWeightQuantBatchMatmulV3` (may need NZ-format conversion — see existing M5.2 FRACTAL_NZ work in CpCannEngine)

### Q7 — VAE optimization (1-2 weeks)

VAE decode is ~15% of total wall for high-res outputs. UNet-style GroupNorm + Conv fusion opportunities.

- [ ] Q7.1 Profile VAE decode — which ops dominate?
- [ ] Q7.2 GroupNorm + Conv fusion via vendor op if available (`aclnnGroupNormSilu*` family exists per CANN-SKILLS)
- [ ] Q7.3 Resolution scaling: VAE decode scales quadratically with output resolution — verify 512×512 vs 1024×1024 tradeoff

**Gate**: +5-15% VAE decode wall.

### Q8 — Final HARD gate (1 week)

- [ ] Q8.1 **User-eye gate**: human visual inspection of 20 canonical edit tasks across CN/EN prompts. Must match ggml-cann baseline quality — no visible artefacts, correct edit semantics, colour/style fidelity.
- [ ] Q8.2 **steps/s gate**: **≥ 2× over Q0-v2 baseline** at 512×512 output. Stretch ≥ 3×.
- [ ] Q8.3 **HBM peak gate**: < 30 GB during inference (margin for future model-swap in same process).
- [ ] Q8.4 **Latency-to-first-pixel** for streaming UX (if streaming scope): first denoising step wall < 3 s.
- [ ] Q8.5 Tag fork at `qwen-image-edit-<N>-landed`.

## 6. Acceptance criteria (summary)

- [ ] Q0-v2 discovery GREEN
- [ ] Q1 native engine generates valid outputs at Q4 quant
- [ ] Q2+Q3+Q4+Q5 stacked: each lands with eye-check pass + steps/s lift
- [ ] Q8 final gates cleared: ≥ 2× baseline, eye-check OK, HBM under budget
- [ ] Contract stamped with commit SHAs + Verified-by per milestone

## 7. Risks

1. **Q4 quant image quality** — diffusion at 4-bit is less-studied than text at 4-bit. Visible artefacts possible. Mitigation: eye-gate is hard kill; fallback to 8-bit if 910B3/B2 procured.
2. **VAE encode/decode dominance** — if VAE is >30% of wall, DiT-focused levers (Q2-Q5) have less headroom. Mitigation: Q7 audit early.
3. **CFG batching breaks some architectures** — some DiT variants handle cond/uncond asymmetrically in cross-attention. Mitigation: Q4 verify cond/uncond shapes symmetric before wiring.
4. **Ref-image conditioning pipeline** — QIE specifically uses ref-image (not just T2I); the `sd_image_t*` hook in conditioner may have ggml-cann-specific assumptions that don't transfer. Mitigation: Q1 scoping surfaces this.
5. **HBM margin 6 GB on 32 GB card** — any cached activations / extra workspace could overflow. Mitigation: WSPOOL retain-list bound at depth 8 (already in TTS fork).
6. **MLX parity impossible** — Apple M4 Max hits ~30 fps on qwen-image T2I; that's 2× our 910B4 HBM bandwidth. We won't match Apple absolute; we target 2× our own baseline as the honest gate.

## 8. Timeline (agent-wall)

Amended 2026-04-22 per feasibility deltas.

- Q0-v2: 1-2 days (DONE)
- Q0.5: 3-5 days (parallel with Q1, Mac-side OK) ← **new**
- Q1: **1-3 weeks** (widened from 1-2; gather_v3 schedule risk)
- Q2 native engine: 2-3 weeks
- Q2 aclGraph step-keyed: 1-2 weeks
- Q2.5 CacheDIT calibration: 3-5 days ← **new, free-engineering lever**
- Q3 FIAv2: 1 week (gated on Q0.5.2)
- Q4 CFG batching: 1 week (gated on Q0.5.1)
- Q5 QKV grouped: 1 week
- Q6 weight upload: 3-5 days
- Q7 VAE: 1-2 weeks
- Q8 final gates: 1 week

**Total**: **~9-13 weeks agent-wall** (was 8-12; +1 week for Q1 widening, Q0.5 + Q2.5 fit in parallel slack).

**First-value milestone** (Q0.5 + Q1 + Q2 + Q2.5 + Q4): ~6-8 weeks — native engine + aclGraph + CacheDIT-tuned + CFG-batching, ≥2× baseline. **Q2.5 alone may hit 2× gate** before Q3/Q4/Q5 kernel work — re-evaluate scope at Q2.5 landing.

## 9. Host rules

- ac02 ONLY for all QIE agent work (ac01 = TTS, ac03 = ASR per PM directive)
- Patch-file push mechanism
- No Claude coauthor
- HARD KILL at eye-gate regression OR steps/s regression
- Eye-gate (subjective) vs WER (ASR, automatable): QIE uses eye-check per milestone; reference 20-task benchmark set defined at Q0-v2
- Q4 quant mandatory on 910B4; revisit only if 910B3/B2 procured
- 4-bit weight integrity: verify at Q0-v2 that 4-bit checkpoints produce acceptable baseline quality BEFORE Q1 dispatch (else no point optimizing broken output)
