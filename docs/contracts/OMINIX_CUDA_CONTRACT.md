# OminiX-CUDA Contract — Pure C++ + ggml-cuda Inference Stack

**Status**: ACTIVE (drafted 2026-04-26, PM signed).
**Repo target**: new repo `ominix-cuda` (forked from OminiX-Ascend, CANN code stripped, CUDA backend wired).
**Mandate**: build a Python-free, vLLM-free, PyTorch-free C++ inference stack on NVIDIA Blackwell (GB10) leveraging llama.cpp + ggml-cuda backend + cuBLAS/cuDNN/CUTLASS direct dispatch.

## Status as of 2026-04-26

Phase 5 dispatch landed; ship summary at `/Users/yuechen/home/ominix-cuda/SHIP_SUMMARY.md`. Phase 4 ASR landed same day, post-Phase-5, via Mac-local build chain (commits `f50488e6` → `0596f1e5` → `abec38ba` → `a8858f86`). Phase 2.7–2.9 TTS (vocoder + E2E + sampling + predictor perf) landed same day, post-Phase-4.6, via Mac-local build chain (commits `3fd4253d` → `3dc149ec` → `1829e741` → `1e280ced` → `e5c92e4d` → `82545bb7` → `7680e727`). Phase 3.6 production CLI FlashAttention enable landed same day, post-Phase-2.9 (commit `cfb31930`).
Repo HEAD: `07fb6b6d` (Phase 3.4d Euler-step fix; first authentic CUDA-native QIE-Edit parity vs Ascend, cossim 1.0000 at n=1, 1024^2).
Phase 4 HEAD (Mac local): `a8858f86` (Phase 4.5 audio encoder cossim 1.000000 vs HF Python across all 11 stages; first authentic CUDA-native ASR transcript on hand).
Phase 2.9 HEAD (Mac local): `7680e727` (predictor device LM-head + CUDA Graphs; total TTS RTF 4.60 → 1.74; steady-state runtime RTF 0.62).
Phase 3.6 HEAD (Mac local): `cfb31930` (production CLI `--diffusion-fa` enable; 1024² 20-step cat PNG 1229 s → **165 s (2:45) = 7.4×**).

Headline production deliverables:
- **First authentic CUDA-native Qwen-Image-Edit-2509 cat PNG** generated on GB10 #2 (`zgx-5b44`) via `ominix-diffusion-cli` (sd.cpp path on ggml-cuda) — 1024² / 20-step Euler / "make the cat smile". Pre-Phase-3.6: 1229 s (20.5 min). **Post-Phase-3.6 with `--diffusion-fa`: 165 s (2:45) = 7.4× wall reduction**, n=20 FA sharp + smile, n=2 FA vs no-FA visually identical. Output: `/tmp/qie_cuda_prod_1024_n20.png` (1.24 MB). No Python in process tree. The native `ImageDiffusionCudaEngine` (Phase 3.x) is correctness-GREEN with byte-parity vs Ascend; native-engine end-to-end ship is gated on a future cuDNN FMHA pass.
- **First authentic CUDA-native Qwen3-ASR transcript** (Phase 4.5 receipt, Mac local). Ellen audiobook 9.36 s WAV → "language English. It might serve you better to be a little less comfortable, but wherever you're listening to this book, please remember to turn off your cell phone and that the taking of flash photographs is strictly forbidden." (43 tok, 226 bytes). Wall 3483 ms (0.37 RTF). Audio encoder cossim 1.000000 vs HF Python at all 11 probed stages. AsrCudaEngine reuses TalkerCudaEngine (Phase 2.x) verbatim for the 28L Qwen3 text decoder — no API changes.
- **First authentic CUDA-native end-to-end Qwen-TTS audio** (Phase 2.7–2.9 receipt, Mac local). Text → 24 kHz WAV via Talker + CodePredictor (device F16 LM-head + CUDA Graphs, 10.3× speedup vs host F32 matvec) + new SpeechTokenizerDecoder vocoder (RVQ + 2× upsample + 4 vocoder blocks + tanh). Total wall: 47019 ms → **17805 ms = RTF 4.60 → 1.74 = 2.65× total speedup**; steady-state runtime RTF **0.62** (excluding 10.2 s init/assets). Sampling (temp + top-k + top-p + rep penalty) defeats greedy mode collapse: prompt-1 unique 7/8 (was 2/8); cross-prompt Pearson 0.0108 (was 0.1277); multi-seed Pearson 0.0142.

**Operational note (production CLI)**: production runs MUST pass `--diffusion-fa`. The flag is wired through `common.hpp:622` → `set_flash_attention_enabled` → `ggml_ext_attention_ext` → `fattn-mma-f16` on SM12.1 GB10. Default-off behavior leaves the naive F32 attention path enabled and is the wall-time bottleneck. Phase 3.5 cat PNG simply hadn't passed the flag; Phase 3.6 is an operational fix, not a code change (commit `cfb31930` is an empty marker).

| Phase | Scope | Status | Receipt |
|---|---|---|---|
| Phase 0 | Repo bootstrap, CANN strip, ggml-cuda backend | DONE | commit `1aaa75d2` |
| Phase 1 | sd.cpp baseline + CFG batching + RoPE pre-compute | DONE (eye-PASS, wall 308.82 s above 140 s target) | commit `ad5ef19c`, `docs/cuda_phase1_baseline.md` |
| Phase 2.1 | TalkerCudaEngine scaffold + GGUF parse | DONE | `d60452a7`, `6f0daf16` |
| Phase 2.2 | 28-layer forward_decode, Q8_0 dequant fix | DONE | `ffa9d313`, `7a2f92c4`, `8851a888` |
| Phase 2.3 | KV cache + autoregressive loop | DONE (53.85 TPS Talker, 18.57 ms steady) | `a027e999` |
| Phase 2.4 | CodePredictor (Qwen3 5L, schema-shared) | DONE (944 TPS w/graphs in hot loop) | `a2c655a0` |
| Phase 2.5 | CUDA Graphs at per-pos capture | PARTIAL (Predictor 1.99x, Talker 1.12x compute-bound) | `f92503b8` |
| Phase 2.6 | FP8 / INT8 quant via cuBLAS | DEFERRED (perf-only; required to clear 80 fps gate) | — |
| Phase 2.7a | SpeechTokenizerDecoder scaffold + RVQ (~665 LoC C++ + test harness; 16 codebooks; host gather + 2× cuBLAS Sgemm; vocoder GGUF 457 MB / 271 tensors / F32) | DONE ✓ (RVQ smoke codes[16,32]→output[512,32] NaN-free std=12.05) | `3fd4253d` |
| Phase 2.7b | Pre_conv + 2× upsample blocks (`decoder_ops.cu` 356 LoC + 7 new kernels: causal_conv1d_im2col, depthwise_conv1d_causal, conv_transpose1d_k2s2, layernorm, gelu_erf, bias_add, residual_add) | DONE ✓ (smoke RVQ→pre_conv→ups0→ups1=[128,1024]@T=32 in 4.11 ms) | `3dc149ec` |
| Phase 2.7c | Vocoder blocks + audio out (4 blocks 1536→768→384→192→96 strides [8,5,4,3] × 3 residual units dilations 1,3,9; new kernels dilated_causal_conv1d_im2col, causal_conv_transpose1d, snake_beta, tanh) | DONE ✓ (smoke codes[16,32]→audio[61440]=32×1920, 192.7 ms wall, WAV `/tmp/qwen_tts_smoke.wav`) | `1829e741` |
| Phase 2.7d | Real-token E2E TTS pipeline (structural; `test_qwen_tts_e2e.cpp` ~383 LoC wires Talker + Predictor + Vocoder; honest scope: text-unconditioned on zgx-3675 codec-only Talker GGUF) | DONE ✓ (10.24 s audio @ RTF 4.15, predictor host-matvec dominated) | `1e280ced` |
| Phase 2.7e | Text conditioning via `qwen3_assets.gguf` (text_embd [151936,2048], codec_embd.{0..15}, proj [1024,2048]; tts_pad/bos/eos + codec_bos/eos special tokens; prefill `[IM_START, ASSISTANT, NEWLINE, TTS_PAD×3, TTS_BOS, text..., TTS_EOS, CODEC_BOS]`) | DONE ✓ (two prompts → different audio Pearson 0.13; greedy mode-collapse handed to 2.8) | `e5c92e4d` |
| Phase 2.8 | Sampling (temp + top-k + top-p + rep penalty); defaults match Ascend `talker.h:25-43` (T=0.9, top_k=50, top_p=1.0, rep=1.05); env-gated `OMINIX_TTS_*` | DONE ✓ (mode collapse defeated: prompt-1 unique 7/8 was 2/8; cross-prompt Pearson 0.0108 was 0.1277; multi-seed Pearson 0.0142; sampling overhead 3-5%) | `82545bb7` |
| Phase 2.9 | Predictor device LM-head + CUDA Graphs (host F32 matvec on 30720-vocab → device cuBLAS GemmEx F16 IO + F32 accum, 60 MB weight upload once; Graphs already wrapped via Phase 2.5 shared `decode_graph_execs_`) | DONE ✓ (**predictor 31983 ms → 3094 ms = 10.3×; total TTS 47019 ms → 17805 ms = RTF 4.60 → 1.74; steady-state runtime RTF 0.62**; greedy Pearson 0.904 vs Phase 2.8 sane F16 drift) | `7680e727` |
| Phase 3.1 | ImageDiffusionCudaEngine scaffold + GGUF parse (Q4_0/Q8_0) | DONE | `fc629955` |
| Phase 3.2 | DiT 1-block forward at 1024^2, NaN-free | DONE (~960 ms/block) | `9d9ded58` |
| Phase 3.3a | Multi-axis NEOX RoPE + mod_vec/t_emb internalization | DONE | `afddaf24` |
| Phase 3.3b | F32 widening through residual chain + 60-block forward + norm_out/proj_out/unpatchify | DONE (latent std 0.238 sane) | `09c5cdae` |
| Phase 3.3c | Host-orchestrated 20-step Euler-flow loop, max_img_seq 4096->8192 | DONE | `09c5cdae` |
| Phase 3.4 | Euler-step semantics fix (proj_out treated as denoised prediction, not velocity) | DONE — bit-parity vs Ascend at n=1, production cat at n=20 | `07fb6b6d` |
| Production cat PNG | 1024^2 / 20-step via `ominix-diffusion-cli` | DONE (NOT via native engine) | `/tmp/qie_cuda_prod_1024_n20.png` |
| Phase 3.6 | Production CLI FlashAttention enable (`--diffusion-fa` flag wired through `common.hpp:622` → `set_flash_attention_enabled` → `ggml_ext_attention_ext` → `fattn-mma-f16` on SM12.1; Phase 3.5 cat PNG just hadn't passed the flag — operational fix, not a code change) | DONE ✓ (per-step 18.55 s → **7.53 s = 2.46×**; total 1024² 20-step 1229 s → **165 s = 2:45 = 7.4×**; cat PNG preserved n=2 FA vs no-FA visually identical, n=20 FA sharp + smile) | `cfb31930` (empty marker) |
| Phase 3.7 | Text encoder + VAE FlashAttention (next perf lever for production CLI; ~50 s of the 165 s wall is text encode + VAE encode/decode) | DEFERRED | — |
| Native-engine cuDNN FMHA | Native `ImageDiffusionCudaEngine` ~960 ms/block at F32 naive attn → cuDNN FMHA target 6-12 s/step | DEFERRED | — |
| Phase 4.1 | AsrCudaEngine + AudioEncoderCudaEngine scaffold + GGUF parse + mel spec port (314 LoC C++ from Ascend) | DONE ✓ | `f50488e6` |
| Phase 4.2 | Audio encoder forward (3× Conv2d via im2col + cuBLAS GemmEx, 24L F32 transformer, output MLP 1024→2048); new kernels `launch_layernorm_affine_f32`, `launch_gelu_erf_f32`, `launch_im2col_f32` | DONE ✓ (encode 178.6 ms / 32 kHz / 9.36 s, NaN-free) | `0596f1e5` |
| Phase 4.3 + 4.4 | Split prefill + E2E transcribe driver (440 LoC `test_qwen_asr_cuda_e2e.cpp`); reuses `TalkerCudaEngine::forward_decode(emb, pos)` for token + embed injection — no API changes | DONE ✓ (3483 ms wall on Ellen 9.36 s WAV = 0.37 RTF) | `abec38ba` |
| Phase 4.5 | Audio encoder cossim parity vs HF Python — `nchw_to_frame_slab` inner-dim ordering fix (`c = ch / H; h = ch % H`) | DONE ✓ (cossim 1.000000 across all 11 probed stages; first authentic CUDA-native ASR transcript) | `a8858f86` |
| Phase 4.6 | CPU mel parity vs HF Python (currently 0.80 cossim — window/log-scale divergence) | ⚠ IN FLIGHT (non-blocking; bypassed via `OMINIX_ASR_USE_MEL_BIN`) | — |
| Phase 4 RTF target | RTF ≤ 0.10, CER=0 Tier-1 13-clip | PARTIAL (first transcript at 0.37 RTF; gated on Phase 2.6 FP8/INT8 + Phase 3.6 cuDNN FMHA — 28L Qwen3 body shared with Talker) | — |
| SpeechTokenizerDecoder vocoder (TTS audio E2E) | C++ port complete (Phase 2.7a–c); wired to Talker + Predictor + text-conditioned (Phase 2.7d–e); sampling on (Phase 2.8); device LM-head + Graphs (Phase 2.9) | DONE ✓ (first authentic CUDA-native E2E TTS audio at RTF 1.74 cold / 0.62 steady-state runtime) | `3fd4253d` → `7680e727` |
| Vocoder → Talker init amortization (TTS warm-start) | ~10.2 s one-shot init dominates cold-vs-steady RTF spread; integration question, not kernel | DEFERRED (OminiX-API embed) | — |
| Phase 5 | Docs + ship | DONE (this dispatch) | `SHIP_SUMMARY.md` + this contract update |

Acceptance roll-up: Phase 0 PASS, Phase 1 partial (eye-PASS; pre-3.6 wall 308.82 s; **post-3.6 prod-CLI cat PNG 165 s = 2:45 with `--diffusion-fa`**), Phase 2 perf gate (≥80 fps) deferred to 2.6 — but **TTS E2E audio shipped Phase 2.7–2.9 at RTF 1.74 cold / 0.62 steady-state runtime**, Phase 3 parity-GREEN with production cat in hand (**Phase 3.6 ✓ 7.4× wall reduction**; native-engine end-to-end perf deferred to cuDNN FMHA), Phase 4 PARTIAL (first authentic CUDA-native transcript landed; 4.1–4.5 ✓; 4.6 mel parity in flight; RTF target gated on 2.6/3.6), Phase 5 PASS.

### Phase 4 ASR perf table (Mac local, post-4.5 fix)

| Stage | ms | Notes |
|---|---:|---|
| mel | 9.7 | CPU port from Ascend; 4.6 parity in flight |
| audio encode | 180.2 | F32 naive attn; 24L transformer + 3× Conv2d (im2col + cuBLAS) |
| prefill | 2591 | 137 positions (9 prefix tok + 122 audio embeds + 6 suffix tok); 28L Qwen3 (TalkerCudaEngine) |
| gen | 682 | 3 tok × 227 ms with cuBLAS warm-up |
| BPE decode | 20.2 | host |
| **total** | **3483** | **RTF 0.37** on Ellen 9.36 s WAV → 43 tok / 226 bytes |


## 1. Why this exists

The codex CUDA Phase 0-2 work delivered honest evidence:
- **Qwen3-TTS via vLLM: 11.5 fps** end-to-end on GB10 — **3× SLOWER than Ascend ship (32.2 fps)**. Bottleneck: code-predictor feedback loop incompatible with vLLM batched generation.
- **Qwen-Image-Edit via diffusers**: 141s baseline + torch.compile (-20%) + CacheDIT (-21%) = ~89s stacked on GB10 1024×1024/20-step.

The Python/PyTorch path doesn't fit Qwen-family TTS architecture. Native direct-dispatch (the same pattern that took Ascend 1→32 fps) is the right play on CUDA too — and CUDA's mature toolchain (cuBLAS, cuDNN, CUTLASS, Flash Attention v3, CUDA Graphs) makes it structurally simpler than the Ascend native engine arc.

## 2. Hardware

| Host | Port | Hostname | GPU | Status |
|---|---|---|---|---|
| GB10 #1 | 6222 | zgx-3675 | NVIDIA GB10 (Blackwell, sm_121) | ssh key installed; codex Phase 0 done |
| GB10 #2 | 6022 | zgx-5b44 | NVIDIA GB10 (Blackwell, sm_121) | ssh key installed; codex Phase 0+1+2 done |

Both: 119 GB unified CPU+GPU memory (Grace ARM aarch64 + Blackwell), CUDA 13.0, Ubuntu 24.04.

## 3. Goals

| Workload | Target | Reference |
|---|---|---|
| Qwen3-TTS end-to-end audio | **80-150 fps on GB10** | Ascend ship 32.2 fps |
| Qwen-Image-Edit 1024×1024/20-step | **30-50s end-to-end** | codex stacked 89s; MLX 80s |
| Qwen3-ASR | **beat A1a RTF 0.142** | shipping reference |

## 4. Scope

**In scope**:
- Pure C++ stack (NO Python anywhere in inference, NO vLLM, NO PyTorch, NO transformers)
- Vendored llama.cpp/ggml + ggml-cuda backend
- Native engines for TTS / QIE / ASR mirroring Ascend pattern
- stable-diffusion.cpp port for QIE/SD/Flux baseline
- CUDA Graphs at step level
- FP8/BF16 matmul via cuBLAS direct (cuBLAS supports both natively on Blackwell)
- Codec C++ via cuDNN

**Out of scope**:
- Python wrappers (anything `import` Python)
- vLLM, sglang, TensorRT-LLM (PyTorch-based)
- HuggingFace transformers
- Step distillation training (separate workstream)
- Multi-GPU tensor parallel (single-GPU ship target)

## 5. Reusability map (from OminiX-Ascend)

### Drop-in (build-flag flip)
- `tools/ominix_diffusion/` — stable-diffusion.cpp port. Set `-DGGML_CUDA=ON` instead of `-DGGML_CANN=ON`.
- GGUF parser, tokenizers, conditioner, scheduler, VAE — backend-agnostic at sd.cpp layer.

### Architecture-port (1:1 design, swap dispatch layer)
- `TalkerCannEngine` → `TalkerCudaEngine` (aclnn → cuBLAS/cuDNN)
- `ImageDiffusionEngine` (post-attention-fix from #84) → `ImageDiffusionCudaEngine`
- `AsrTextDecoderCannEngine` → `AsrTextDecoderCudaEngine`
- W8 quant pattern → cuBLAS INT8/FP8 native
- ACLGraph capture → CUDA Graphs (ggml-cuda has built-in support)
- WSPOOL retain-list → CUDA stream-aware allocator

### Code commits to port directly to ominix-cuda
- `036047de` — CFG batching in stable-diffusion.cpp (sd.cpp level, backend-agnostic)
- `fd7ab97a` — RoPE pre-compute in qwen_image.hpp (sd.cpp level)
- `f0b51dc1` Q2.4.4d — F32 residual + LayerNorm + gated-add pattern (architecture portable)
- `cf16f83e` — BF16 matmul out (NOT needed on CUDA — cuBLAS handles natively, but pattern reusable)
- Native attention fix from #84 (in flight) — port once it lands

### NOT to port
- ggml-cann backend
- aclnn dispatchers
- Path C #1-#5 backend patches (CANN-specific bugs)
- FRACTAL_NZ weight format
- WQBMMv3 BF16-output workarounds

### Lessons-portable (knowledge)
- Substep cossim bisect methodology (commits 9a264391 / 96b67fb4)
- Probe-first discipline
- Defensive env-gate pattern
- 3-10× projection discount rule

## 6. Phase plan with gates

### Phase 0 — Repo bootstrap (1 day, Mac + GB10 #2)

- Fork OminiX-Ascend → new repo `ominix-cuda` (local + GitHub at `ymote/ominix-cuda`)
- Strip CANN-specific code:
  - Delete `ggml/src/ggml-cann/`
  - Delete `tools/qwen_*/native/*cann*` files (will be re-ported in Phase 2/3/4)
  - Delete `docs/qie_q2_phase4_smoke.md` Path-C-* sections (preserve in OminiX-Ascend)
  - Keep all sd.cpp, stable-diffusion logic, tokenizer, GGUF parser
- Add `ggml/src/ggml-cuda/` from llama.cpp upstream (vendored; commit reference)
- Top-level CMake: `option(GGML_CUDA ON)`
- Set up `.github/workflows/` for CI
- README with phase status

**Gate 0**: Repo builds on GB10 #2 via `cmake -B build -DGGML_CUDA=ON && cmake --build build`. `nvidia-smi` shows healthy. No CANN dependencies remain.

### Phase 1 — ominix_diffusion CUDA baseline (2-3 days, GB10 #2)

- Build `ominix-diffusion-cli` with CUDA backend
- Run canonical cat-edit smoke at 1024×1024/20-step
- **Apply commit `036047de` (CFG batching) → measure**
- **Apply commit `fd7ab97a` (RoPE pre-compute) → measure**
- Capture per-step wall + total wall + peak GPU memory + eye-check
- Compare to codex's diffusers baseline (141s)

**Gate 1**: 1024×1024/20-step produces recognizable B&W cat at <140s with CFG batching applied. Measurement committed to docs.

### Phase 2 — Native TalkerCudaEngine (10-14 days, GB10 #1)

This is the headline lever. Direct port of `TalkerCannEngine` architecture.

- Phase 2.1: scaffold + GGUF parse (1-2 days)
- Phase 2.2: per-token forward with cuBLAS dispatch (3-5 days)
- Phase 2.3: KV cache + autoregressive loop (2-3 days)
- Phase 2.4: codec C++ via cuDNN (2-3 days)
- Phase 2.5: CUDA Graphs at per-pos capture (2-3 days)
- Phase 2.6: FP8/INT8 quant via cuBLAS (1-2 days)

Reference: `tools/qwen_tts/talker_cann_engine.cpp` (Ascend native engine that delivered 32 fps).

**Gate 2**: Qwen3-TTS canonical synthesis at **≥ 80 fps end-to-end on GB10 #1**. Audio quality byte-identical or ear-PASS vs Ascend ship state. **No Python in process tree.**

### Phase 3 — Native ImageDiffusionCudaEngine (14-21 days, GB10 #2)

Gated on Ascend native attention fix (#84) landing. Once attention algorithm is correct on Ascend, port the corrected forward to CUDA.

- Phase 3.1: scaffold + GGUF parse (1-2 days)
- Phase 3.2: F32 residual stream pattern (2-3 days)
- Phase 3.3: forward block with cuBLAS dispatch + corrected attention (3-5 days)
- Phase 3.4: 60-block loop (2-3 days)
- Phase 3.5: Euler-flow scheduler + 20-step denoise (2-3 days)
- Phase 3.6: VAE decode C++ (already in stable-diffusion.cpp; reuse)
- Phase 3.7: CUDA Graphs at step capture (2-3 days)

**Gate 3**: 1024×1024/20-step canonical cat-edit at **≤ 50s end-to-end**, eye-check PASS. Beats codex's stacked 89s.

### Phase 4 — Native ASR (5-7 days, GB10 #1 or #2 idle)

Port `AsrTextDecoderCannEngine` to CUDA. Compose with TalkerCudaEngine pattern (Qwen3 decoder is shared architecture).

**Gate 4**: Qwen3-ASR canonical Tier-1 13-clip CER=0, RTF ≤ 0.10 (beat Ascend A1a 0.142).

### Phase 5 — Docs + ship (parallel to Phase 4)

Author `docs/cuda_optimization_learnings.md` mirroring Ascend's `qwen_tts_optimization_learnings.md`. Push repo public.

## 7. Workstream dependencies

```
Phase 0 (bootstrap)
   ├── Phase 1 (sd.cpp baseline + CFG + RoPE)
   │      └── Phase 5 (docs)
   ├── Phase 2 (native TTS) — independent
   └── Phase 3 (native QIE) — gated on Ascend #84 attention fix
          └── Phase 4 (native ASR) — composes with Phase 2
```

## 8. Operating rules

- **No Python in production stack.** Build tools (cmake, ninja) and one-shot scripts (e.g., gguf inspection) may use Python during dev only. Final ship is C++/CUDA only.
- **No vLLM, no PyTorch, no transformers, no diffusers.** If a tool requires them at dev time (e.g., generate a reference latent for parity check), document the dev-only dependency clearly.
- **GGUF is the only model format.** No safetensors loaded by Python.
- **Codec must be C++.** ONNX Runtime via C++ API is acceptable; Python wrappers are not.
- **No remote code push without explicit PM approval.** All commits to local fork; push to `ymote/ominix-cuda` when explicitly green-lit.
- **No Claude/codex coauthor on commits.**
- **SIGHUP-proof launches** for any long remote build: `nohup setsid bash -c '...' < /dev/null > log 2>&1 &`.
- **Defensive env-gate discipline**: every new code path gated by env flag; default-off byte-identical to pre-patch.
- **Probe-first**: every cuBLAS/cuDNN/CUTLASS substitution gets a 30-min standalone correctness probe before integration.

## 9. Honest expectation framing

Per OminiX-Ascend's lesson on optimistic projections:
- Ascend native engine F32-projection extrapolation was 50× too pessimistic at toy shape.
- Ascend native attention had a 2-week-undiscovered algorithm bug (mode collapse) hidden under "numerical green."
- "2-3× stack" rule still holds; raw individual lever projections from one platform don't transfer linearly to another.

**Realistic CUDA targets** (post-discount):
- Qwen3-TTS: **80-150 fps** plausible (vs Ascend 32 fps; Blackwell tensor core 3-5× beats Ascend INT8)
- QIE-Edit: **30-50s/image** plausible (vs codex stacked 89s; native bypass of PyTorch overhead)
- ASR: **RTF ≤ 0.08** plausible (Blackwell + cuBLAS direct vs Ascend ggml-cann)

If reality is 2× worse than these targets, that's still SOTA on CUDA for these workloads.

## 10. Repo location

- **Local mirror**: `/Users/yuechen/home/ominix-cuda/`
- **GitHub fork**: `ymote/ominix-cuda` (private until ship)
- **Build hosts**: GB10 #1 at `~/ominix-cuda/`, GB10 #2 at `~/ominix-cuda/`

## 11. Acceptance

- [x] Phase 0 Gate: builds clean on GB10
- [~] Phase 1 Gate: 1024×1024/20-step cat at <140s with CFG batching, eye-PASS — pre-3.6 308.82 s; **post-3.6 prod-CLI cat PNG 165 s with `--diffusion-fa` (7.4× speedup)**; 25 s above 140 s gate, Phase 3.7 (text encoder + VAE FA) the next lever
- [~] Phase 2 Gate: TTS ≥80 fps end-to-end, no Python in process tree — perf gate deferred to 2.6 FP8/INT8; **TTS E2E audio shipped Phase 2.7–2.9 at RTF 1.74 cold / 0.62 steady-state runtime; first authentic CUDA-native text→24 kHz waveform on hand**
- [~] Phase 3 Gate: QIE-Edit ≤50s/1024×1024/20-step, eye-PASS — **production CLI shipped at 7.53 s/step / 165 s total wall via Phase 3.6 `--diffusion-fa`** (3.3× over 50 s gate); native-engine end-to-end still gated on cuDNN FMHA
- [~] Phase 4 Gate: ASR RTF ≤ 0.10, CER=0 Tier-1 — first authentic CUDA-native transcript landed (43 tok, 0.37 RTF on Ellen 9.36 s WAV); audio encoder cossim 1.000000 vs HF Python at 11 stages; 4.6 mel parity in flight; RTF gated on Phase 2.6 + native-engine cuDNN FMHA perf levers
- [x] Phase 5: docs published (`SHIP_SUMMARY.md` + this contract update); repo `ymote/ominix-cuda` review TBD

Total: ~3-4 weeks agent-wall to all gates.
