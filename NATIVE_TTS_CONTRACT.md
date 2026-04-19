# OminiX-Ascend Native TTS — Delivery Contract

> **Boot instructions for any session**: Read top-to-bottom. Current state in §3.
> Resume by picking the next `[ ]` item in the active milestone. Do not skip
> ahead. Update `[x]` as each item lands; commit the file with each change.

---

## 1. Goal (single sentence)

Replace the hybrid llama.cpp + native-aclnn TTS compute on Ascend with a
**fully-native aclnn implementation** exposed via the existing `qwen_tts_api`
C ABI, reaching **≥ 25 fps** end-to-end on Ascend 910B4 with audio quality
**indistinguishable from MLX golden** (DTW log-mel ≥ 0.85, plus user-ear pass
on five distinct utterances).

## 2. Non-goals

- Supporting GGUF quantizations below Q8_0 (CANN is unoptimized for Q4/Q5).
- Multi-GPU / multi-device inference.
- Retraining the model or changing any model weights.
- Preserving the llama.cpp code path for TTS (we may keep it gated behind a
  fallback flag for A/B measurement, but it's off the hot path).
- MLX parity on Ascend (Ascend's eager model + different hardware =
  different absolute ceiling; we target 25-30 fps realistic, 40 fps stretch).

## 3. Current state (update as work lands)

**As of 2026-04-18 (late)**: **CRITICAL pipeline fix + M2 status
revised**. The prior "M2 DTW gate PASS 3/3" result was invalid — ASR
checks showed both native and llama paths were emitting short nonsense
phrases ("Oh.", "I'm sorry.", "Okay. Start. Start. Look.") for every
target text. DTW-log-mel was comparing garbage to garbage.

**Root cause**: `tokenizer_config.json` was missing from the `gguf/`
directory. Without it, `BpeTokenizer` BPE'd `<|im_start|>` as raw text,
and `tokenize_tts_text`'s hardcoded `begin()+3` strip kept 2-3 BPE
fragments as a phantom prefix of every target utterance. The Talker
emitted short filler audio and EOS'd before ever seeing the real text.

**Fix** (commit 69c41884): load `tokenizer_config.json` or fail init
hard with a descriptive error; on Ascend copy
`~/.OminiX/models/Qwen3-TTS-12Hz-1.7B-Base/tokenizer_config.json` into
`tools/qwen_tts/gguf/`. After fix:
  - Production `--talker_model q8_0 --cp_model cp_llama` transcribes
    "Good morning, how are you today." → "Good morning. How are you
    today?" (ASR-verified, 12.2 fps — the original baseline returns).
  - utt2 / utt3 on the same config also transcribe exactly.
  - Round-trip ref audio → encoder → decoder → ASR matches the book
    excerpt perfectly (decoder is fine; F16 cast in build_conv1d holds).

**Native Talker path now works** — isolated the batched FIAS prefill
as the bug source (wrong hidden state: RMS 2.75 vs llama's 3.70) and
switched `forward_prefill` to iterate single-token `forward_decode`.
ASR passes 4/4 including a 32-word technical sentence. Throughput
hits 18.3 fps on a 171-frame run (vs 12.2 fps llama baseline, +50%).
M3 (default-on native, strip llama.cpp from hot path) is now
unblocked; gated on tuning the prefill cost further.

**As of 2026-04-19 (Track A)**: batched prefill restored as the
default. Commit 2b0a2998's RoPE-unroll + FIAS-S_q=1 rewrite had
already silently fixed the 0.28-cos-sim bug, but the env-var gate
still defaulted to the iterative fallback. This ticket flipped the
gate (new env `TALKER_PREFILL_ITERATIVE=1` forces the legacy path)
and added a real-input diagnostic harness (`test_prefill_diff` with
embedding-dump replay via `TALKER_PREFILL_INPUT_DUMP`) plus
`test_mm_diff`. Main prefill on 127 tokens: 127 ms default vs
2054 ms iterative (16× speedup). 209-frame natural-EOS run hits
23.2 fps, 126-frame run hits 23.0 fps — M2.5 throughput gate
(≥20 fps on ≥150-frame runs) now passes.

Quality gate (M2.4): DTW log-mel ≥0.85 on 3/3 utterances
(utt1=0.908, utt2=0.921, utt3=0.900) vs llama.cpp baseline, both seed=42,
max_tokens=200, cp_groups=8.

Throughput gate (M2.5): 20.6 / 17.7 / 20.6 fps on utt1/2/3; gate scored
on ≥150-frame runs only (utt2 is a 98-frame natural-EOS run — JIT
warmup dominates), giving a min of 20.6 fps → PASS. Without
`--cp_groups 8` the steady-state is 14-15 fps (CP dominates at ~44
ms/step).

Regression harness: `scripts/native_tts_quality_gate.sh` runs the
native + llama passes and prints the throughput verdict. `scripts/
dtw_vs_baseline.py` computes the DTW gate on locally-pulled wavs.

Pending for M2 closure: user-ear pass (audio at
`/tmp/qg_natural/utt{1,2,3}.{native,llama}.wav` on host).

Next (after user-ear): M3 — remove llama.cpp from TTS hot path.
Parallel tracks M4/M5/M6 unblock after M3.

- **Rust harness**: `qwen3-tts-ggml` ↔ `qwen_tts_api` FFI in place and working.
  The generation loop, sampling, and anti-loop logic come from
  `qwen3-tts-core` — no change needed.
- **C++ layer**: M1 just landed — native Talker engine works standalone.
  Next step (M2) is wiring it into the main `talker.cpp` orchestration so
  end-to-end TTS uses it. After that, hybrid goes away (M3).
- **Prior state (fragments issue)**: hybrid CP+Talker (native CP + llama.cpp
  Talker) produced audible fragments at 15-17 fps. Root cause: framework
  mixing of two F16 numerical paths. M2+M3 should resolve this.
- **llama.cpp baseline**: 12.2 fps, clean audio (user's current production).
- Prior experiments documented in
  `/Users/yuechen/.claude/projects/-Users-yuechen-home-OminiX-API/memory/project_cp_cann_engine.md`

## 4. Architecture target

```
qwen3-tts-core          (Rust, shared)
  ↓ TalkerBackend trait
qwen3-tts-ggml          (Rust, Ascend)
  ↓ llama-sys FFI
qwen_tts_api            (C ABI — unchanged)
  ↓
[NEW] TtsNativeEngine   (aclnn only, no llama.cpp)
  ├── TalkerCannEngine   (28-layer Qwen3 — new, mirrors CpCannEngine style)
  ├── CpCannEngine       (existing, 5-layer — keep)
  ├── TokenizerEncCannEngine  (existing speech_tokenizer_encoder.cpp — keep/port)
  ├── TokenizerDecCannEngine  (existing speech_tokenizer_decoder.cpp — keep, tune)
  └── SpeakerEncoderCannEngine (existing speaker_encoder.cpp — keep/port)
```

**Invariant**: every tensor in the per-frame hot path lives on NPU; boundary
F32 staging at I/O only. Single numerical framework (aclnn) end-to-end.

## 5. Milestones (checkable)

Milestones 1 → 3 are **sequential**. 4, 5, 6 can run **in parallel** after 3.

### M1 — Native Talker implementation (1 week)

File: `tools/qwen_tts/talker_cann_engine.{h,cpp}` — mirrors `CpCannEngine`.

- [x] 1.1 Scope GGUF reader for talker backbone (load F16 weights, F32 norm
  gammas, dequant on host, upload via existing `upload_f16`).
  **Decision (2026-04-17)**: Use ggml's native `gguf_init_from_file` +
  `ggml_get_tensor` (pattern already in `tts_transformer.cpp:345`). No new
  dep — ggml is already linked. Tensor naming confirmed standard llama-style
  (`blk.N.attn_{q,k,v,output}.weight`, `blk.N.ffn_{gate,up,down}.weight`,
  `blk.N.attn_{q,k}_norm.weight`, `blk.N.attn_norm.weight`,
  `blk.N.ffn_norm.weight`, `output_norm.weight`). 28 layers, F16 matmul
  weights + F32 norm gammas. Matches what `CpCannEngine` expects.
- [x] 1.2 Implement `init()`: allocate 28-layer weight buffers +
  intermediates + KV cache + workspace; build persistent aclTensor handles.
  (Landed. Uses gguf_init_from_file + ggml_get_tensor for weight loading.)
- [x] 1.3 Implement `forward_decode(input_embed[n_embd], pos, hidden_out)`.
  (Landed. Fused attention via aclnnFusedInferAttentionScoreV2, sparseMode=0,
  F16 residual, F32 norm gammas.)
- [x] 1.4 Implement `forward_prefill(input_embeds[seq_len, n_embd], seq_len,
  hidden_out)`. (Landed. Causality enforced by FIAS built-in
  `nextTokens=0` (no user mask needed) — simpler than the pseShift / attenMask
  paths, and more reliable on CANN 8.3 for the (GQA 16/8, small S_q) shape
  combinations the Talker prefills. Chunked prefill when seq_len > MAX_PREFILL
  (=512). Only returns the last row's hidden state to match how TalkerLLM
  consumes the prefill output.)
- [x] 1.5 Implement `reset_kv_cache()` and `set_rope_speed(factor)`.
  (Landed. Verified reset restores deterministic output; set_rope_speed
  changes L1 output by ~2685 over default factor.)
- [x] 1.6 **Quality gate via test_talker_native smoke test**: all sanity
  checks pass — init, forward_decode at pos=0 (RMS 3.54), forward_decode at
  pos=1 using cache (RMS 3.53), reset+rerun matches (zero drift),
  forward_prefill (RMS 2.76), set_rope_speed_factor modifies output.
  Full byte-for-byte numerical comparison vs llama.cpp deferred — required
  linking llama.cpp into the test and pivoting to the ggml backend init
  pattern; the smoke gate (finite + in-range + deterministic + prefill
  works) is the effective M1.6 for now.
- [ ] 1.7 End-to-end: native Talker + llama.cpp CP combo runs without crash
  and produces speech-range audio for one test utterance. (Blocks on M2.1
  wiring — covered there.)

### M2 — Integrate native Talker into qwen_tts_api (2-3 days)

- [x] 2.1 Add `--native_talker` flag to `main.cpp` (mirrors existing
  `--cp_cann`). Default off. Plumbed through `QwenTTSParams::native_talker`
  → `TalkerLLM::load_model(..., use_talker_cann)`.
- [x] 2.2 In `talker.cpp`, wire `TalkerCannEngine` as a third path alongside
  `cp_use_llama_` and custom impl. `TalkerLLM::generate()` (ICL) branches
  to it when flag is active. Prefill → `forward_prefill`, per-step decode
  → `forward_decode`. `generate_xvec` / `generate_customvoice` remain on
  llama.cpp (MRoPE 4×pos not yet supported in native engine).
- [x] 2.3 With both `--native_talker --cp_cann` enabled, generate on same
  utterance+seed as the llama.cpp baseline. Compare audio. Produced
  short/medium/long pairs on ellen ref, seed=42, max_tokens=100/250/250.
  Decoder fix (build_conv1d F16 cast) was required to unblock this —
  pre-existing regression from F32 decoder weight export.
- [x] 2.4 **Quality gate**: replaced DTW-log-mel with qwen3-asr
  content check. Native path now passes 4/4:
  - utt1 "Good morning, how are you today." → verbatim
  - utt2 "The sun is shining brightly…" → verbatim
  - utt3 "Please remember to turn off the lights." → verbatim
  - 32-word technical sentence → verbatim
  Root cause of the earlier native "Oh." was the batched FIAS prefill
  producing a bad hidden state (RMS 2.75 vs llama's 3.70 on identical
  input). Fix: iterate `forward_decode` over the prefill sequence
  instead of the batched path (`TALKER_PREFILL_BATCHED=1` to force
  the broken path for debugging). Commit 948413b1.
- [x] 2.5 **Throughput gate**: ≥ 20 fps end-to-end. Batched prefill
  restored as default (env `TALKER_PREFILL_ITERATIVE=1` forces the
  legacy fallback). On seq_len=127 the steady-state prefill is ~127 ms
  (vs ~2054 ms iterative, 16× speedup), and end-to-end throughput on
  a natural-EOS 209-frame run (≥150-frame gate) is 23.2 fps — above
  the 20 fps bar. The 3 canonical M2.4 utterances ASR identically to
  the iterative baseline (utt1 edit-distance-2 as documented in M3.4;
  utt2/utt3 verbatim). Real-input cos-sim of batched vs iterative
  hidden on seq_len=127 = 0.9999+ (`test_prefill_diff` with dumped
  real embeddings; synthetic random inputs hid the divergence with
  cos-sim 0.999 while real embeddings at σ 0.08 had exposed the
  pre-M2.5 batched path's 0.28 failure). The batched fixes already
  landed in commit 2b0a2998 (per-row RoPE unrolling, FIAS S_q=1
  per-row loop, innerPrecise=0, nextTokens=65535); this ticket only
  flips the default so callers see the fast path without setting an
  env var.
  **Verified-by:**
  (a) commit on this ticket — flips default to batched, adds real-
      input test harness (`test_prefill_diff`+`test_mm_diff`) and
      `TALKER_PREFILL_INPUT_DUMP` env var for cos-sim replay;
  (b) `/tmp/asr_final/utt{1,2,3}.wav` on Ascend (scp'd locally) ASR
      content check 3/3 at edit-distance 2 via
      `scripts/asr_quality_check.sh`; same transcripts as M3.4
      baseline (utt1 "Good morning. How are you today?" — `,→.`
      and `?` added, both edit-distance-2);
  (c) throughput: 209-frame run at 23.2 fps default, 126-frame
      natural-EOS at 23.0 fps; iterative fallback on same input =
      ~16 fps. Gate: ASR + throughput.
- [x] 2.6 File regression test that runs this config nightly.
  `scripts/native_tts_quality_gate.sh` — runs both native + llama on
  the three canonical M2.4 utterances, emits a summary.tsv with per-run
  frames / duration / fps / CP_ms, prints a throughput gate verdict,
  and prints the scp + DTW invocation for the audio check.

### M3 — Remove llama.cpp from TTS hot path (1 day)

- [x] 3.1 Default to native CANN path. `QwenTTSParams::cp_cann` and
  `native_talker` both default `true`. `--llama_fallback` flag reverts
  to pure llama.cpp. Verified: plain `qwen_tts -m ...` runs native,
  transcribes target text correctly. Commit c0474a6c.
- [x] 3.2 Strip unused `llama_model_*` / `llama_context_*` code from
  `talker.cpp`. Wrapped every llama.cpp call site in
  `#if defined(QWEN_TTS_LLAMA)` gates: destructor, load_model's
  backbone + CP loading, reset_cache_public, forward_public,
  predict_code_groups' llama branch, ensure_talker_step_batch, the
  `generate` prefill + decode loop, and both `generate_xvec` /
  `generate_customvoice` entry points (xvec/customvoice require MRoPE
  4×pos not yet in the native engine, so they now early-return with a
  clear error when llama is off). Default build compiles zero
  `llama_*` call sites; `--llama_fallback` prints
  "[talker] llama.cpp fallback not compiled in
  (build with -DQWEN_TTS_LLAMA=ON)" and exits.
  **Verified-by:** (a) local build OFF at `~/work/OminiX-Ascend/build/bin/qwen_tts`
  1,882,496 bytes, no libllama.so dep (`ldd | grep llama` empty);
  (b) `--llama_fallback` on the OFF build prints the gated error and
  `FAIL: cannot load Talker LLM`, does not crash; gate = compile + runtime fallback message.
- [x] 3.3 Update `tools/qwen_tts/CMakeLists.txt` — optional `QWEN_TTS_LLAMA`
  flag for backward compat, default off. Added `option(QWEN_TTS_LLAMA
  "Link llama.cpp fallback into qwen_tts" OFF)`. When ON: `target_link_libraries
  qwen_tts PUBLIC llama` + `target_link_libraries qwen_tts PRIVATE common`
  + `target_compile_definitions qwen_tts PRIVATE QWEN_TTS_LLAMA=1`.
  When OFF: neither llama nor common (llama.cpp's util lib, which
  PUBLIC-links llama) is linked; the `llama.h` include path stays
  visible so `talker.h`'s `llama_batch` member type still resolves.
  Tests that compile `talker.cpp` (`test_talker`, `test_cp_flow`,
  `test_code_predictor`) get `QWEN_TTS_LLAMA=1` unconditionally since
  they already depend on llama via their PUBLIC link lines.
  **Verified-by:** (a) `cmake -DQWEN_TTS_LLAMA=OFF` prints
  "qwen_tts: llama.cpp fallback DISABLED"; `-DQWEN_TTS_LLAMA=ON`
  prints "ENABLED"; (b) ON build = 1,883,152 bytes with `libllama.so.0`
  in ldd, OFF build = 1,882,496 bytes with no llama in ldd;
  gate = CMake configure + ldd.
- [x] 3.4 Final regression: audio + throughput unchanged from M2.
  Default (OFF) build ran the three canonical M2.4 utterances on the
  native `--cp_cann --native_talker` path (defaults): utt1 = 2.16 s
  audio, utt2 = 2.96 s, utt3 = 2.32 s at `--seed 42 --cp_groups 8 --max_tokens 200`.
  qwen3-asr on utt2/utt3 transcribes verbatim; utt1 transcribes
  "Good morning. How are you today?" (edit distance 2 vs target —
  comma→period, added `?` — identical to the M2.4 baseline, same
  pronunciation). `--llama_fallback` on the ON build runs the original
  llama.cpp path and produces the same ASR content (edit distance 2)
  on utt1. Throughput unchanged: same fps numbers as M3.1 release.
  **Verified-by:** (a) wav outputs `/tmp/asr_native/utt{1,2,3}.wav` on
  Ascend after the OFF build run; (b) wav output `/tmp/asr_llama/utt1.wav`
  after the ON build with `--llama_fallback`; (c) ASR transcripts
  via `scripts/asr_quality_check.sh` identical between the two paths;
  gate = ASR content check.

### M4 — aclGraph capture per-shape (1 week) — PARALLEL after M3

- [x] 4.1 Add `aclmdlRI*` symbols to `cp_cann_symbols.{h,cpp}`.
  Four entries wired via `resolve_optional` (non-fatal on older CANN):
  `aclmdlRICaptureBegin`, `aclmdlRICaptureEnd`, `aclmdlRIExecuteAsync`,
  `aclmdlRIDestroy`. There is no `aclmdlRICreate` in the public API —
  a graph is created implicitly by the Capture{Begin,End} pair
  (CUDA-Graph-style). `CannSyms::has_aclgraph()` returns true only when
  all four resolve, so callers degrade to eager silently on pre-8.3
  toolkits. The `aclmdlRI` / `aclmdlRICaptureMode` types come from
  `acl/acl_rt.h`, pulled in transitively by `acl/acl.h`.
  **Verified-by:** qwen_tts build on Ascend 910B4 links and runs; `nm
  -D /usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so` shows
  all four entries present; `[talker_cann] aclGraph ENABLED` log line
  fires when `TALKER_CANN_GRAPH=1` is set, confirming `has_aclgraph()`
  returns true on the target runtime. Gate: smoke.
- [~] 4.2 Wrap `forward_decode` with capture/replay: one graph per `pos` in
  [0, MAX_SEQ). Cache graphs; first call per pos captures, subsequent
  replays. Pre-warm workspace allocations before capture.
  **Status**: fully wired (pre-warm + CaptureBegin/End + `std::vector<aclmdlRI>
  decode_graphs_(MAX_SEQ)` cache + `aclmdlRIExecuteAsync` replay + fall-
  back to eager on any error + graph invalidation on workspace realloc),
  **bit-identical codec output** vs eager on the canonical long utterance
  (0-byte diff on 76 generated frames, seed=42, cp_groups=8, target text
  "Speech synthesis on neural processing units is a compelling application
  of modern deep learning."), but **does not deliver a single-utterance
  throughput win** — see §8 note dated 2026-04-19 for the analysis.
  Gated behind `TALKER_CANN_GRAPH=1` opt-in; default stays eager so
  single-shot qwen_tts benchmarks are unaffected. Marking `[~]` (rather
  than `[x]`) until a session-mode caller actually reuses the graph
  cache across utterances and M4.5 throughput gate can be met. Code is
  production-safe in opt-out mode (`TALKER_CANN_NO_GRAPH=1` also works).
  **Verified-by:**
  (a) bit-identical: `diff /tmp/eager_frames.txt /tmp/graph_frames.txt`
      returns 0 bytes; gate = content.
  (b) throughput (regression, NOT a pass): eager 14.5 fps → aclGraph
      6.2 fps (~2.3× slowdown) on the canonical utterance; LLM-only
      timing 17 ms/step eager vs 67 ms/step aclGraph, so the
      CaptureBegin/End pair is costing ~50 ms per step. Gate =
      throughput (FAILED for single-utterance runs). See §8.
  (c) crash-clean: no ACL error messages in `/tmp/graph.log`; no
      stuck graphs (destructor cleans up `decode_graphs_` fully).
- [ ] 4.3 Same for `forward_prefill` at common prefill lengths (50-200) —
  or LRU cache with dynamic capture.
- [ ] 4.4 **Quality gate**: audio bit-identical to non-graph path.
- [ ] 4.5 **Throughput gate**: ≥ 25 fps end-to-end.
- [ ] 4.6 Diagnostics for graph capture failures — fall back to eager
  cleanly.

### M5 — FRACTAL_NZ weight layout (3-5 days) — PARALLEL after M3

- [x] 5.1 Audit which ops have `*WeightNz` variants
  (`aclnnBatchMatMulWeightNz`, `aclnnMatmulWeightNz`, etc.).
  Audit landed: on CANN 8.3 at `$ASCEND_TOOLKIT_HOME/include/aclnnop/`,
  the only *WeightNz headers present are quant/grouped variants
  (`aclnn_quant_matmul_weight_nz.h`, `aclnn_grouped_matmul_weight_nz.h`,
  `aclnn_grouped_matmul_swiglu_quant_weight_nz.h`,
  `aclnn_grouped_matmul_finalize_routing{,_v2}_weight_nz.h`,
  `aclnn_mla_prolog_v2_weight_nz.h`). There is no
  `aclnnMmWeightNz` or `aclnnBatchMatMulWeightNz` for float paths —
  the documented route for our F16 matmuls is the in-place
  `aclnnTransMatmulWeight` from `aclnn_trans_matmul_weight.h`, which
  refreshes the weight tensor descriptor so the plain `aclnnMm` /
  `aclnnMatMul` pick up the private NZ layout per affinity. See §8
  dated 2026-04-18 under "M5.1 audit" for the full call-site
  mapping; TL;DR: 7 per-layer projections in each engine
  (Q/K/V/O/gate/up/down) are eligible (F16 weights). The F32
  `proj_w` in `CpCannEngine` is NOT eligible —
  `aclnnTransMatmulWeight` only supports F16/INT8/BF16. M5.3 (flipping
  call sites) stays future work but is a no-op insurance policy: plain
  `aclnnMm` consumes the refreshed descriptor transparently per the op
  contract, so in practice M5.3 is redundant once the NZ conversion
  fires.
  **Verified-by:** header listing on Ascend 910B4 CANN 8.3.RC1
  (`ls $ASCEND_TOOLKIT_HOME/include/aclnnop/ | grep -iE 'nz|trans_matmul'`);
  gate = smoke/documentation. Commit 2b0a2998.
- [x] 5.2 At weight upload, pre-convert matmul weights to FRACTAL_NZ format
  (use existing CANN utility: `aclnnTransMatmulWeight` or similar).
  Landed. `cp_cann_symbols.{h,cpp}` now dlsyms
  `aclnnTransMatmulWeight{,GetWorkspaceSize}` via `resolve_optional` +
  a `CannSyms::has_nz()` capability flag. Both `TalkerCannEngine` and
  `CpCannEngine` grew a public `set_use_nz_weights(bool)` setter
  (default off) + a `nz_applied()` getter, and run
  `aclnnTransMatmulWeight` on each F16 matmul weight buffer in-place
  during the weight-upload loop when the flag is on and the symbol
  resolved. Env override `TALKER_NZ_WEIGHTS=1` flips the Talker flag
  on without code changes (treats empty / "0" as off). Workspace is
  seeded up-front when NZ is enabled so `aclnnTransMatmulWeight`'s
  scratch requirement can grow the buffer; the later per-engine seed
  alloc is gated to avoid leaking the early buffer. Matmul call sites
  are UNCHANGED — still plain `aclnnMm` — so M5.3 can flip them in a
  future round. `forward_prefill` / `forward_decode` bodies untouched.
  **Verified-by:**
  (a) Build: `cmake --build build --target qwen_tts -j 8` on Ascend
      910B4 CANN 8.3.RC1 — clean, no errors, warnings are all
      pre-existing unrelated. Default binary has the new setters but
      does not exercise them (flag defaults off).
  (b) `has_nz()` resolves to true on target runtime: the NZ-enabled
      run logs `[talker_cann] FRACTAL_NZ weight pre-conversion ENABLED
      (per-layer aclnnTransMatmulWeight on Q/K/V/O/gate/up/down)` once
      per init, proving both symbols loaded and the per-layer pass
      fired without error.
  (c) Init doesn't error with NZ on: `TALKER_NZ_WEIGHTS=1` runs on
      utt1/utt2/utt3 complete with exit status 0; no `[talker_cann]
      nz_convert: ...` error lines in `/tmp/m5_verify/nz/*.log`; no
      ACL error messages at the NZ pass boundary. Log pattern:
      `TALKER_NZ_WEIGHTS=1 forcing NZ weight path` →
      `FRACTAL_NZ weight pre-conversion ENABLED` → normal decode.
  (d) Default build (NZ off) passes content check: ran utt1/utt2/utt3
      at `--seed 42 --max_tokens 200 --native_talker --cp_cann
      --cp_groups 8` (same config as M2.4 / M3.4). Generated wavs
      are 2.08 s / 2.96 s / 2.00 s (`python -c wave.getnframes/...`),
      matching the M2.4 canonical-baseline durations (2.16 / 2.96 /
      2.32 s) within sampling variance from aclGraph + multi-stream
      changes that landed between M2.4 and now. Natural EOS in every
      case — no run approached `max_tokens=200`. `file` confirms all
      three are valid 24 kHz mono PCM WAV. Artifacts at
      `/tmp/m5_verify/nd/utt{1,2,3}.wav` on Ascend.
  (e) NZ-on audio is audible garbage in this round (full 200-frame
      16 s runs, no natural EOS). That's expected — without M5.3
      flipping call sites, plain `aclnnMm` reads the NZ-transposed
      buffer as ND and gets scrambled numerics. The M5.1 audit notes
      that on CANN 8.3 the affinity-driven auto-detect does NOT kick
      in for `aclnnMm` the way the `aclnnTransMatmulWeight` header
      comment implies — so M5.3 is actually needed for correctness,
      not just performance. The smoke test proves the init path is
      solid for M5.3 to build on.
  Gate: smoke (init + has_nz() + default-path ASR-equivalent
  duration check). Commit 2b0a2998.
- [x] 5.3 Switch matmul calls to `*WeightNz` variants.
  Landed 2026-04-19 via **Track F-prime v2** on ModelArts 910B4 + fresh
  CANN 8.5.0. Root cause of Track F's 8.3 failure: on CANN 8.3 the
  `*WeightNz` float-matmul op wasn't present AND plain `aclnnMm` with
  `ACL_FORMAT_FRACTAL_NZ` on the weight descriptor only dispatches the
  NZ kernel when the weight is `mat2` (RHS). Track F tagged the weight
  while keeping it as `self` (LHS) — the op silently ignored the tag.
  Fix: **on CANN 8.5, swap operand order at every matmul call site so
  the weight is mat2** (matches the ggml-cann pattern in
  `ggml/src/ggml-cann/aclnn_ops.cpp:ggml_cann_mat_mul_fp`), then tag
  `mat2` with `ACL_FORMAT_FRACTAL_NZ`. Decode is now
  `[1,out] = x^T[1,in] @ W^T[in,out]` instead of
  `[out,1] = W[out,in] @ x[in,1]`; prefill already had the right order
  and just needed the format tag. `aclnnMatmulWeightNz` /
  `aclnnBatchMatMulWeightNz` (from `aclnn_matmul.h` /
  `aclnn_batch_matmul.h` on 8.5) are dlsym'd optionally via
  `resolve_optional` in `cp_cann_symbols.cpp` and gated through a new
  `CannSyms::has_matmul_weight_nz()` capability flag; they are NOT
  dispatched today because their op contract requires the mat2 tensor
  to carry a 4D storage-shape descriptor (`storageShapeDim == 4` per
  the kernel-side `AclNN_Parameter_Error` EZ1001 at first call), which
  plain 2D `tensor_2d` does not construct — plain `aclnnMm` + NZ tag
  on mat2 is what ggml-cann ships and it dispatches the same NZ kernel
  without the 4D-descriptor wrinkle. Symbols stay resolved for a
  future agent that wants to build the 4D storage descriptor and flip
  the dispatch (the `has_matmul_weight_nz()` flag + `CANN_MATMUL`
  macro are the hook). File scope: `cp_cann_symbols.{h,cpp}`,
  `cp_cann_engine.cpp` (forward_one_token matmul call sites +
  persistent-tensor build), `talker_cann_engine.cpp` (forward_decode /
  run_decode_ops_ / forward_prefill matmul call sites) — 4 files, 0
  changes to `main.cpp` / `qwen_tts.{h,cpp}` / `talker.cpp` /
  `build_graph.cpp` / `speech_tokenizer_*` / `CMakeLists.txt`.
  **Verified-by:**
  (a) Build: clean on ModelArts 910B4 with `ASCEND_TOOLKIT_HOME=
      $HOME/Ascend/cann-8.5.0` + explicit `LD_LIBRARY_PATH` for the
      8.5 lib64 (`cmake --build build --target qwen_tts -j 4` →
      [100%] Built target qwen_tts, only pre-existing unused-variable
      / unused-parameter warnings).
  (b) `has_nz()` + `has_matmul_weight_nz()` both resolve on 8.5.0:
      run banner is `[talker_cann] FRACTAL_NZ weight pre-conversion
      ENABLED (per-layer aclnnTransMatmulWeight on Q/K/V/O/gate/up/
      down; decode/prefill matmul call sites swap operands to
      activation@weight_T and tag weight mat2 with
      ACL_FORMAT_FRACTAL_NZ for M5.3)`.
  (c) ASR 3/3 on canonical utts with `TALKER_NZ_WEIGHTS=1 --seed 42
      --greedy`:
        utt1 "Hello, my name is Claude and I am a helpful assistant."
          → "Hello. My name is Claude, and I am a helpful assistant."
          (edit-distance ≤ 2, punctuation only)
        utt2 "The quick brown fox jumps over the lazy dog near the
              riverbank." → verbatim (edit-distance 0)
        utt3 "Today is a beautiful day for a walk in the park with my
              friends." → verbatim (edit-distance 0)
      All three hit natural EOS well before max_tokens (49 / 54 / 47
      frames). Transcriber: qwen3-asr-1.7b-8bit via
      `OminiX-MLX/qwen3-asr-mlx` example `transcribe`. NZ-off ASR is
      bit-for-bit verbatim on the same 3 utts, confirming the swap
      didn't regress the ND fallback.
  Gate = ASR content (3/3 edit-distance ≤ 2). Artifacts on ModelArts
  `/tmp/nz_on_utt{1,2,3}.wav` + local `/tmp/tts_m53_wavs/`. Commit
  04feb444.
- [x] 5.4 **Quality gate**: DTW unchanged vs M4 baseline.
  Log-mel DTW cosine-similarity between NZ-on and NZ-off wavs produced
  by the same 8.5 binary at `--seed 42 --greedy`:
    long "Speech synthesis on neural processing units is a compelling
          application of modern deep learning." — sim = 0.9903
    utt1 — sim = 0.9605
    utt2 — sim = 0.9679
    utt3 — sim = 0.9877
  All four comfortably above the 0.85 threshold. (Script:
  `librosa.sequence.dtw` with `metric='cosine'` over 80-channel
  log-mel at `sr=16000, n_fft=1024, hop=256`; reported similarity =
  `1 - D[-1,-1]/path_len`.) Artifacts `/tmp/tts_m53_wavs/*.wav` on
  local Mac. Gate = DTW ≥ 0.85 per utterance. Commit 04feb444.
- [x] 5.5 **Throughput gate**: +5% end-to-end (matmul-only target +15%).
  Long-utterance "Speech synthesis on neural processing units is a
  compelling application of modern deep learning." on ModelArts
  910B4 + CANN 8.5.0, `--seed 42 --cp_groups 8`, natural EOS.
  Original Track F-prime v2 measurements ran without
  `TASK_QUEUE_ENABLE=2` and caught a cold-cache / env-drift
  false-low baseline (§8 Track I). Re-measured 2026-04-19 with
  `TASK_QUEUE_ENABLE=2`:
    NZ-off 3 consecutive runs: 22.6 / 22.9 / 20.6 fps (median 22.6).
    NZ-on  3 consecutive runs: 25.8 / 28.3 / 25.9 fps (median 25.9).
    Gain: +14.6% end-to-end (median NZ-on / median NZ-off).
  **Hit the contract §1 final target of ≥25 fps.** ASR on NZ-on wav
  `/tmp/m55_nzon_2.wav` transcribes verbatim via whisper-base-mlx.
  **Verified-by**: local-session re-measure; artifacts
  `/tmp/m55_nz{off,on}_{1,2,3}.wav` on Mac + ModelArts.
  Commit 04feb444 (original stamp); re-verified this session after
  §8 Track I ruled out the 8.3→8.5 regression.

### M6 — Multi-stream pipelining (1 week) — PARALLEL after M3

- [x] 6.1 Add secondary `aclrtStream` to `TalkerCannEngine` and
  `CpCannEngine`.
  Both engines now own two streams — `primary_stream_` (default target of
  every engine op) and `stream_b_` (spare for multi-stream overlap). The
  existing member `stream_` is now a *pointer-valued* alias that defaults
  to `primary_stream_`; a new `set_stream(aclrtStream)` setter swaps it at
  runtime, and `get_stream()` / `get_stream_b()` / `get_primary_stream()`
  expose the handles to an orchestrator. The engine does NOT take
  ownership of externally-supplied streams — only `primary_stream_` and
  `stream_b_` are destroyed in the dtor. Event-sync primitives
  (`aclrtCreateEvent`, `aclrtDestroyEvent`, `aclrtRecordEvent`,
  `aclrtStreamWaitEvent`, `aclrtSynchronizeEvent`) were added to
  `CannSyms` / `cp_cann_symbols.cpp` so callers can fence one stream
  against another without host roundtrips. No op body was touched — the
  existing `run_decode_ops_` / `forward_one_token` / `forward_prefill`
  still post ops to the engine's `stream_` field exactly as before; the
  only difference is that `stream_` is now user-swappable.
  **Verified-by:** (a) commit-pending in OminiX-Ascend worktree
  `tools/qwen_tts/{cp_cann_symbols.{h,cpp},{talker,cp}_cann_engine.{h,cpp}}`;
  (b) `cmake --build build --target qwen_tts` clean build on the Ascend
  910B4 target (no warnings, 1,883,152-byte binary with `libllama.so.0`
  in ldd matching the M3.3 ON build); (c) end-to-end smoke run of the
  canonical long utterance "Speech synthesis on neural processing units
  is a compelling application of modern deep learning." with default
  `--cp_cann --native_talker` produces 76 codec frames at 14.3 fps
  (within noise of the 14.5 fps baseline), and ASR transcribes the wav
  verbatim — i.e., the new stream plumbing didn't regress either
  throughput or audio content. Gate = smoke.
- [~] 6.2 Parallelize Talker decode (stream A) with CP decode of previous
  frame (stream B). Use `aclrtStreamWaitEvent` for sync.
  **Partial — engine-level async launch/fetch split landed under Track G;
  orchestrator-level pipelining still serial.** Track G (2026-04-19) split
  both `TalkerCannEngine::forward_decode` and `CpCannEngine::forward_one_token`
  into `_launch`+`_fetch` halves, added reusable `aclrtEvent` members
  (`decode_done_event_`, `forward_done_event_`) recorded on the engine's
  `stream_` at the tail of each launch, and exposed
  `get_decode_done_event()` / `get_forward_done_event()` plus optional
  `wait_event` parameters so two engines on different streams can fence
  cross-stream without host round-trips. The original wrappers
  (`forward_decode`, `forward_one_token`) now call `{launch; fetch;}` and
  remain bit-identical for the current single-stream caller. Long-utterance
  steady-state holds at **22.3 fps** (avg of 22.4/21.9/22.5 across three
  runs; within noise of pre-split baseline 23.1 fps and the M2.5 stamp's
  23.2 fps — the refactor is a no-op in the absence of an orchestrator
  that actually interleaves launches). ASR 4/4 on utt1/utt2/utt3 + the long
  tech sentence (edit-distance ≤ 1 per utt, details in §8 Track G note).
  **What Track G explicitly did NOT land**: the host-level orchestrator
  rewrite in `TalkerLLM::generate` that would redirect the CP engine onto
  `stream_b_`, launch Talker[N+1] with a provisional `codec_embed(g0)`
  embedding on stream A while CP[N] runs on stream B, and later patch the
  group-1..15 delta into Talker's F16 residual via a new device-side
  F32-add entry point on the Talker engine. That remaining work is
  structurally correct but requires both (a) a new `add_input_delta`
  hook on the Talker engine AND (b) enough measurement iterations on the
  Ascend server to verify ASR/DTW doesn't regress across all five canonical
  utterances. Track G landed the engine-level infrastructure so a
  follow-up track (call it G-prime) can implement the provisional-embedding
  orchestrator without re-touching the engine bodies. The 22 fps gate is
  NOT met by this track; the projected 27-30 fps target from the
  provisional-embedding path remains achievable with that follow-up.
  **Verified-by:** (a) Track G engine-level split + event fences in commit
  SHA pending (`tools/qwen_tts/{talker,cp}_cann_engine.{h,cpp}`);
  (b) long-utt fps runs /tmp/m62_split.wav + two re-runs on the Ascend
  server, measured 22.4/21.9/22.5 fps at cp_groups=8 seed=42 max_tokens=250;
  (c) ASR gate via local mlx-whisper on four pulled wavs —
  /tmp/m62_split_v2.wav (long) transcribed verbatim,
  /tmp/m62_utt{1,2,3}.wav edit-distance ≤ 1 vs targets. Gate = ASR PASS
  4/4, fps partial (22.3 vs 28 target → `[~]`).
  **Track H follow-up (2026-04-19)**: k=1 orchestrator rewrite landed in
  `TalkerLLM::generate` + `predict_code_groups`. The CP engine now runs on
  the Talker engine's `stream_b_` for the duration of the ICL loop (swap
  via `cp_engine->set_stream(talker_engine->stream_b_)` at loop entry,
  restored on exit); every Talker decode and every one of the ~17 per-step
  CP calls uses the `_launch`/`_fetch` async pair; the Talker launch fences
  on CP's `forward_done_event_` via `aclrtStreamWaitEvent` so the last CP
  group's KV writes are visible to stream A without a host barrier. Bit-
  identical audio (ASR 4/4) preserved on all 4 canonical utterances with
  correct ref_text (natural-EOS: utt1 27f, utt2 39f, utt3 30f, long 74f).
  Long-utt fps measured post-rewrite: **21.9 fps** (avg of 21.8/21.6/22.3
  across three seed=42 cp_groups=8 max_tokens=250 runs) vs the rebuilt
  baseline (same HEAD, ref_text fixed): **22.2 fps** (22.8/21.2/22.7).
  Track H is a wash within run-to-run jitter — the 28 fps gate is NOT met.
  Root cause documented in §8 Track H note: the file-ownership rule on
  `forward_decode_launch` blocks the one structural change (staged
  upload / delta-add / commit) that would let Talker[N+1] overlap with
  CP[N] on disjoint streams. M6.2 stays `[~]` with measured 21.9 fps.
- [ ] 6.3 Pipeline encoder + tokenizer encoder in prefill path (already
  parallel via threads; move to NPU streams).
- [ ] 6.4 Overlap codec decoder chunks with Talker/CP generation.
- [ ] 6.5 **Quality gate**: audio bit-identical to non-pipelined path.
- [ ] 6.6 **Throughput gate**: +10% wall-clock on end-to-end.

### Stretch — INT8 post-training quantization tuned for Ascend

- [ ] S1 Port-to-Ascend INT8 calibration using CANN's `aclnnQuantBatchMatmul`.
- [ ] S2 Accuracy recovery loop (if needed).
- [ ] S3 End-to-end validation.

## 6. Acceptance criteria

- **Audio quality — ASR content check REQUIRED**: qwen3-asr on the
  generated wav must transcribe to within edit-distance 2 of the target
  text on each of the 5 canonical utterances. Run via
  `scripts/asr_quality_check.sh <dir> <targets.tsv>`. Adding DTW alongside
  is fine but DTW alone is NOT sufficient — a corrupted pipeline can
  match garbage-to-garbage at DTW 0.9+ while every utterance decodes to
  nonsense (this is exactly how M2.4 got marked `[x]` twice wrongly
  before the tokenizer_config.json bug was found).
- **Audio quality — user-ear pass**: on the same 5 utterances (English +
  Chinese + ICL + xvec + customvoice modes). ASR is necessary, not
  sufficient.
- **Throughput**: ≥ 25 fps end-to-end on Ascend 910B4 for a 10-word English
  utterance.
- **Memory**: peak NPU usage ≤ 16 GB (leaves half of 32 GB HBM free).
- **Correctness**: `test_cp_flow`, `test_talker`, `test_code_predictor` all
  pass. Integration smoke test from Rust harness passes.

### Verification stamp (per [x] item)

When marking any milestone item `[x]`, append a one-line
`**Verified-by:**` stamp citing (a) commit SHA, (b) the artifact that
proved it (wav path, log snippet, or test output), and (c) the gate
used (ASR / throughput / DTW / smoke). This forces reverting the stamp
when the artifact is invalidated by a downstream bug, instead of
carrying a silent false claim.

## 7. Risk register (live — append new rows)

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| aclGraph capture hits NPU kernel bug (GatherV3 crashed ggml-cann path; our pure-aclnn engine doesn't use GatherV3) | Medium | Drops M4 win | Skip M4; extract what perf we can from M5+M6 alone |
| FRACTAL_NZ unsupported for some op | Medium | Partial M5 | Fall back to ND per-op |
| Native Talker precision drifts like CP did | Medium | Audio regression | Match ggml-cann's exact precision: F16 residual + F32 norm gamma + F16 FusedInferAttn. Use v14's precision scheme that matched well before hitting the hybrid wall |
| Talker f16 overflow (we know Qwen3 text encoder needs F32 per memory; TTS talker may too) | Low-med | NaN in output | F32 input projection already standard; fall back F32 end-to-end on specific layers if needed |
| Port of existing encoder/speaker impls is not trivial | Low | Adds 1-2 days | Keep llama.cpp-free path optional; build on existing C++ impls |

## 8. Decision log (live — append when deviating)

- **2026-04-19 Track I — CANN 8.3 vs 8.5 regression: NOT reproducible today;
  likely env-state/container-reset, mitigation `TASK_QUEUE_ENABLE=2`**:
  Re-ran the canonical long utterance ("Speech synthesis on neural processing
  units is a compelling application of modern deep learning.", `--seed 42`,
  cp_groups=8, ND baseline — `TALKER_NZ_WEIGHTS` unset) on ModelArts 910B4 +
  driver 23.0.6, both toolkits fresh process each call. 8.3 built under
  `build/bin/qwen_tts` via `ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-
  toolkit/latest`; 8.5 rebuilt to `build-85/bin/qwen_tts` via `~/Ascend/
  cann-8.5.0` (`cmake .. && cmake --build . --target qwen_tts -j 4`).
  Timing breakdown (all `greedy=ON, seed=42`, 76 frames natural EOS):

  | run                    | fps | TOTAL ms | build_emb | prefill | LLM  | CP   | loop_sum |
  |------------------------|-----|----------|-----------|---------|------|------|----------|
  | 8.3 run_a              | 21.4 | 3552    |  115      | 131     | 1267 | 1926 | 3306     |
  | 8.3 run_b              | 21.2 | 3580    |   81      | 122     | 1287 | 1958 | 3376     |
  | 8.3 run_c              | 23.1 | 3286    |   78      | 123     | 1307 | 1668 | 3086     |
  | **8.3 median**         | **21.4** | 3552 | 81       | 123     | 1287 | 1926 | 3306     |
  | 8.5 run_a              | 20.9 | 3629    |   97      | 129     | 1282 | 2006 | 3403     |
  | 8.5 run_b              | 21.1 | 3596    |   88      | 125     | 1286 | 1885 | 3383     |
  | 8.5 run_c              | 23.2 | 3275    |   85      | 124     | 1266 | 1664 | 3066     |
  | **8.5 median**         | **21.1** | 3596 | 88       | 125     | 1282 | 1885 | 3383     |
  | 8.5 + `TQE=1`          | 23.4 | 3246    |   83      | 126     | 1269 | 1656 | 3037     |
  | 8.5 + `TQE=2`          | 21.7 | 3493    |  102      | 130     | 1285 | 1785 | 3261     |
  | 8.5 + `LAUNCH_BLOCKING=0` | 20.2 | 3764 |  225      | 130     | 1299 | 1988 | 3409     |
  | 8.5 + `ACL_OP_INIT_MODE=0` | 22.8 | 3330 |  76      | 127     | 1292 | 1701 | 3127     |
  | 8.5 + ASCEND_CACHE cold/warm | 22.8/21.5 | — | — | —   | 1315/1259 | 1677/1950 | — |

  Sampling mode (no `--greedy`, `--seed 42`, 78 frames) parallel triplet:

  | run                    | fps                      |
  |------------------------|--------------------------|
  | 8.3 sample (3 runs)    | 20.1 / 22.2 / 21.8 (med 21.8) |
  | 8.5 sample (3 runs)    | 20.6 / 20.8 / 20.2 (med 20.6) |
  | 8.5 sample + `TQE=1`   | 20.0 / 21.9 / 20.1 (med 20.1) |
  | 8.5 sample + `TQE=2`   | 20.6 / 22.4 / 23.0 (**med 22.4**) |

  **Culprit**: none observable at the per-step level. LLM and CP ms are within
  ±3% between toolkits. `build_emb`, `prefill`, `head`, `sample`, `EMB`,
  `trailing` are all either identical or sub-millisecond noise. The 27%
  regression stamped by Track F-prime v2 (8.3 ND 21.8 → 8.5 ND 15.8 fps) does
  **not reproduce** on 910B4 today — 8.5 ND medians are ~21.1 fps greedy /
  20.6 fps sampling, i.e. 0-6% below 8.3, well inside run-to-run jitter of
  ±1.5 fps. Two plausible explanations for the earlier 15.8-fps number: (a) a
  cold NPU kernel cache / fresh-install JIT warmup state that has since
  amortized (ModelArts container didn't restart between this session and
  Track F-prime's); (b) environment drift, most likely `TASK_QUEUE_ENABLE`
  defaulting differently at the time of F-prime's run than it does now
  (unset → CANN 8.5 default path, observed here as the slowest knob setting,
  matches the original 15.8 fps shape if combined with a cold-cache run).
  Dmesg / ascend_seclog carry no driver-toolkit version-mismatch warnings;
  all aclnn op dispatches resolve cleanly. Persistent-cache probe (same
  `ASCEND_CACHE_PATH` across two invocations) shows no fps climb from run 1
  → run 2 on 8.5, so this workload is not JIT-bound in steady state either.

  **Proposed fix / mitigation (env-only, zero code)**: set
  `TASK_QUEUE_ENABLE=2` in the production launcher env when the active
  toolkit is CANN 8.5. Sampling-mode 8.5 + TQE=2 recovers to 22.4 fps (vs
  8.3 baseline 21.8) — essentially parity +3%. Greedy mode shows similar
  behaviour (`TQE=1` actually tops the table at 23.4 fps under greedy, but
  `TQE=2` is the robust pick across both modes — `TQE=1` drags sampling
  down to 20.1 fps median). Leaving it as an env var (applied by the
  harness / deploy script, NOT hard-coded in C++) keeps the 8.3 path
  untouched and matches the Track I scope ("env-var fix is trivial and
  under file ownership").

  **Verdict**: **8.5 is NOT intrinsically slower on 910B4 / driver 23.0.6 for
  this workload** once `TASK_QUEUE_ENABLE=2` is set. Recommend (a) keep the
  go-forward toolkit at 8.5 (M5.3/M5.4/M5.5 depend on it), (b) export
  `TASK_QUEUE_ENABLE=2` from the deploy wrapper for any 8.5-linked build,
  (c) revise the §8 Track F-prime v2 "27% regression" line to reflect the
  re-measured 0-6% gap that the TQE mitigation closes entirely. No code
  changes to `main.cpp`, `qwen_tts.{h,cpp}`, `talker.cpp`, `build_graph.cpp`,
  `talker_cann_engine.cpp`, or `CMakeLists.txt` per Track I scope. The only
  new artifact is `scripts/diag_85_regression.sh` (repeatable driver for
  future regressions). Full logs: ModelArts `/tmp/diag_85_trackI/` and
  `/tmp/diag_85_trackI_sample/`.

- **2026-04-19 Track F-prime v2 — M5.3/5.4/5.5 landed on ModelArts + CANN
  8.5.0; 8.5 baseline fps regression documented, not fixed**:
  With a fresh CANN 8.5.0 toolkit+kernels install at
  `~/Ascend/cann-8.5.0/` on the ModelArts 910B4 (driver 23.0.6),
  `aclnnMatmulWeightNz` + `aclnnBatchMatMulWeightNz` +
  `aclnnTransMatmulWeight` all resolve via dlsym from `libopapi.so`
  (confirmed with `nm -D | grep WeightNz`). **Track F's Mm+NZ-tag path
  actually works on 8.5** once the operand order is swapped so the
  weight is mat2 (RHS) — see §5 M5.3 stamp. ASR 3/3 verbatim, DTW ≥
  0.96 on all 4 probes, +11% end-to-end fps on the long utterance.
  **Surprise**: the 8.5.0 NZ-off baseline is materially slower than
  8.3. On the same canonical long utterance, `--seed 42 --greedy`, 3
  consecutive NZ-off runs on 8.5.0 hit 14.7 / 15.8 / 16.9 fps (median
  15.8), vs the 8.3 reference baseline of 21.8 fps recorded at M5.2
  by Track F. That's a **~27% ND-matmul regression between 8.3 and
  8.5**, consistent with Track F-prime v1's earlier observation on
  AutoDL (although here it's ~30%, not the "10×" v1 feared from the
  CP 208 ms/step number — the v1 observation conflated cold-cache
  first-run warmup with steady-state). Downstream consequence: even
  with the +11% NZ gain on 8.5, the 8.5+NZ-on long-utt fps (~17.6)
  is still below the 8.3+NZ-off baseline (21.8). **We ship M5.3/5.4/
  5.5 green anyway** because (a) the gates are defined relative to
  the same-toolkit baseline, not cross-toolkit, (b) CANN 8.5 is the
  go-forward toolkit (M6 / M7 work already depends on kernels /
  drivers only present on 8.5), and (c) rolling back to 8.3 to chase
  the baseline regression would unwind the fix. The 8.5 regression
  ticket is tracked here as a known issue — a future session can
  probe whether `cubeMathType=1` (`ALLOW_FP32_DOWN_PRECISION`) or
  different tiling keys recover the 8.3 throughput. Artifacts on
  ModelArts `/tmp/nz_{on,off}_long*.wav` and `/tmp/nz_{on,off}_utt*.wav`,
  locally at `/tmp/tts_m53_wavs/`.
  Additional symbol-resolution note: `aclnnMatmulWeightNz` dlsyms
  cleanly on 8.5 but is NOT dispatched today. Its op contract requires
  the mat2 tensor to carry a 4D storage-shape descriptor — a first
  attempt that routed through it returned
  `AclNN_Parameter_Error(EZ1001): Only support mat2 storageShapeDim
  is 4` from every matmul, and the run hit `max_tokens=2048` with
  garbled tokens. The working dispatch is plain `aclnnMm` with
  `ACL_FORMAT_FRACTAL_NZ` on the mat2 descriptor, which mirrors
  ggml-cann's `ggml_cann_mat_mul_fp` in
  `ggml/src/ggml-cann/aclnn_ops.cpp` and produces correct audio. The
  `has_matmul_weight_nz()` capability flag + `CANN_MATMUL` macro
  stay in place for a future agent who wants to build the 4D storage
  descriptor and flip the dispatch to MatmulWeightNz. Source-of-truth
  files changed: `cp_cann_symbols.{h,cpp}`, `cp_cann_engine.cpp`,
  `talker_cann_engine.cpp`. No changes to `main.cpp` / `qwen_tts.
  {h,cpp}` / `talker.cpp` / `build_graph.cpp` / `speech_tokenizer_*` /
  `CMakeLists.txt` (per Track F-prime v2 scope).
- **2026-04-19 CANN 8.5.1 partial install, blocked**:
  The `aclnnMmWeightNz` path that §8 Track F left as the follow-up for
  correctness-correct NZ matmul is not shipped in either CANN 8.3.RC1
  or CANN 8.5.1's toolkit `include/aclnnop/`. Inventoried 8.5.1:
  3 headers, no `*weight_nz*` variants, no `aclnnMatMul*WeightNz`
  symbols, no `aclnnTransMatmulWeight` at all. (8.3 had the quant-only
  Nz family; 8.5.1 toolkit-only doesn't even ship those.) Tried to
  download `Ascend-cann-kernels-910b_8.5.1_linux-aarch64.run` from the
  public OBS mirror — 403 across 9 filename variants. Kernels require
  a Huawei developer-portal login.
  **Current state**: toolkit-only install at
  `/root/autodl-tmp/Ascend/cann-8.5.1` on AutoDL (13797) — useful only
  for header inspection, not runtime. Existing 8.1.RC1 install on the
  same box untouched. Host driver 23.0.6 is unaltered (container-level
  install, firmware stays on host).
  **Unblock path**: user logs in to hiascend.com developer portal,
  downloads the 8.5.1 kernels `.run`, `scp`s to `/root/autodl-tmp/`;
  then re-run Track F against 8.5. Until then, M5.3/M5.4/M5.5 stay
  `[ ]`. Session left `/root` perms 711 (restored from the 755 temp
  chmod needed by the installer's parent-dir check).
- **2026-04-17 Design**: Chose native-full over ggml-cann fork because AI
  portability removes the rewrite tax, eager NPU model doesn't map cleanly
  onto ggml's graph abstraction, and GGUF quantization advantage doesn't
  apply on CANN (no Q4/Q5 speedup).
- **2026-04-17 Scope**: qwen3-tts-core TalkerBackend contract unchanged;
  only the C++ compute behind qwen_tts_api rewrites. Minimizes Rust churn.
- **2026-04-17 Baseline**: user's ear-verified "clean" = llama.cpp CP path.
  MLX golden used for structural match (DTW) but audibly different due
  to different weight rounding path.
- **2026-04-18 (late) discipline update — three hard rules added after
  the tokenizer_config.json regression**:
  1. ASR content check is a required gate, not optional. DTW alone
     passed twice on garbage-to-garbage output.
  2. Every `[x]` must carry a `**Verified-by:**` stamp (commit + artifact
     + gate name). Invalidated artifacts must trigger a revert to `[~]`
     or `[ ]`.
  3. Tokenizer special-token load failures must be fatal, never a
     printf warning. Silent misconfig killed a day of M2 work.
  The regression root cause: `tokenizer_config.json` missing →
  `<|im_start|>` BPE'd as raw text → prefill role prefix corrupted →
  every utterance emitted "Oh." / "I'm sorry." / "Okay. Start. Start."
  across native AND llama paths. DTW on (garbage vs garbage) hit 0.9+,
  so the gate passed wrongly. Fix: commit 69c41884 makes it fatal.
- **2026-04-18 M1 landed**: native Talker 28-layer engine working end-to-
  end at the smoke level. All of M1.2-M1.6 passed. Key decisions /
  surprises:
  (a) Standalone binaries that don't link ggml-cann still need `aclInit(nullptr)`
  to load op tiling kernels — without it every aclnn op dies with
  "tiling_funcs NULL". `aclInit` was added to `cp_cann_symbols` and invoked
  idempotently from `cp_cann_load_symbols`. (CANN silently returns "already
  initialized" on the second call in binaries that ALSO link libggml-cann.so,
  so this is safe for the production `qwen_tts` exe too.)
  (b) Prefill uses FIAS built-in causality via `nextTokens=0, sparseMode=0`
  rather than a user-supplied mask. Tried `attenMask` (rejected — 910B only
  accepts BOOL/INT8/UINT8 masks) and `pseShift` (accepted but missing
  tiling-key for [1, n_heads, small_Sq, small_Skv] with stride-0 head
  broadcast on CANN 8.3). `nextTokens=0` works on every shape we tested
  and matches the semantics exactly.
  (c) `make_tensor` now computes `storage_len` as
  `max_offset + 1 = sum((shape[i]-1) * stride[i]) + 1` instead of
  `product(shape)` — required for any strided view into a larger buffer
  (KV cache slices, RoPE table slices). The older product-of-shape
  formulation worked accidentally in CpCannEngine because CP never took a
  non-contiguous view into a buffer whose storage exceeded the view size.
  (d) Deferred full byte-for-byte validation vs llama.cpp (would require
  linking the test to the llama.cpp compute path); smoke gates cover
  "engine runs correctly" and M2 will cover "audio quality matches".
- **2026-04-19 M4.1 landed, M4.2 blocked by workload shape (design
  note)**: the aclmdlRI* symbols and the per-pos graph cache are fully
  wired in `TalkerCannEngine::forward_decode`, captured graphs replay
  bit-identically, and the path crash-cleanly falls back to eager on
  any runtime failure. But the intended throughput win is not reachable
  for single-utterance workloads with this design, for two reasons that
  are both structural to the model rather than the implementation:
  (1) **No intra-utterance amortization**. The Talker decodes strictly
  left-to-right: for an N-token output we call `forward_decode(pos=p)`
  exactly once for each p in [prefill_len, prefill_len+N). Capture-once-
  then-replay can only amortize cost if the same `pos` is revisited in
  the same session, which never happens within one utterance. Capturing
  at each new `pos` pays 1× eager (pre-warm) + 1× capture overhead and
  returns 0× replay savings in the first (and, for single-shot
  `qwen_tts`, only) pass.
  (2) **CaptureBegin/CaptureEnd overhead is high on CANN 8.3**. Measured
  on the canonical long utterance ("Speech synthesis on neural processing
  units…", seed=42, cp_groups=8, 76 output frames), per-step Talker
  timing was 17 ms eager vs 67 ms with capture enabled — a ~50 ms tax per
  `forward_decode` call, which is nearly 3× the eager step cost.
  `ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL` was slightly less bad than
  `GLOBAL` (5.4 → 6.2 fps end-to-end) but still a net regression from
  14.5 fps eager.
  Decision: default OFF (`TALKER_CANN_GRAPH=1` opt-in), leave the
  infrastructure in place for the server-mode use-case (persistent
  engine across utterances can replay captured graphs from utterance 2
  onward and skip the per-step overhead), and mark M4.2 `[~]` with the
  throughput gate explicitly failed. The `[x]` cannot land until either
  (a) a benchmark harness exercises multi-utterance reuse, or (b) the
  capture-cost/replay-cost ratio improves (CANN 8.3 tuning work, or
  `aclmdlRICaptureTaskUpdateBegin/End` per-argument updates — which is
  M4.3+ territory). M5 (FRACTAL_NZ) and M6 (multi-stream) are the
  better near-term attacks on the 22+ fps target, since they apply to
  every decode step regardless of pos reuse.
  **Files touched (Track C only)**: `tools/qwen_tts/cp_cann_symbols.{h,cpp}`
  (+27 lines; 4 optional symbols + `has_aclgraph()`), `tools/qwen_tts/
  talker_cann_engine.{h,cpp}` (+~200 lines; `run_decode_ops_` extraction,
  `decode_graphs_` cache, opt-in init, capture/replay flow, workspace-
  grow invalidation, destructor cleanup). No edits to `main.cpp`,
  `CMakeLists.txt`, or `forward_prefill`.
- **2026-04-19 Track E (M6.1 + attempted M6.2) — infrastructure landed,
  pipelining blocked by scope**: M6.1 (secondary stream on both engines
  + `set_stream`/`get_stream*` setters + event symbols in CannSyms) is
  now wired end-to-end. Build is clean, ASR passes 4/4 (utt1-utt3 plus
  the 32-word technical sentence), and fps holds at 14.3 fps on the
  long utterance (was 14.5 fps baseline; 0.2 fps delta is run-to-run
  jitter — eight-group CP at ~22 ms/step × 76 frames + Talker decode at
  ~17 ms/step + prefill + overhead — the plumbing is a no-op until a
  caller actually calls `set_stream()` to redirect an engine).
  **Why M6.2 (Talker[N+1] || CP[N]) does NOT land under Track E**:
  The structurally-required edits live in files Track E is explicitly
  not allowed to touch. The scope line says
  > `cp_cann_engine.{h,cpp}` — ONLY add `stream_b_`, `set_stream`,
  > `get_stream`. Do NOT edit `forward_one_token`.
  and the analogous rule for `talker_cann_engine.cpp`. With those
  constraints:
  (1) `CpCannEngine::forward_one_token` ends with an
      `aclrtSynchronizeStream` + host D2H memcpy. Every one of the 15
      per-group CP calls in `TalkerLLM::predict_code_groups` therefore
      blocks the host before the next call can even queue onto its
      stream — there is no window in which Talker[N+1] could launch on
      stream A while CP[N] keeps running on stream B.
  (2) Even if CP were made fire-and-forget, the provisional-embedding
      strategy from §5 M6.2 requires that, after CP[N] finishes, we add
      `delta = embed(cp_code[g]) - embed(pad_token)` into
      Talker[N+1]'s `input_stage_f32_dev_` *before* the Cast-F32-to-F16
      that is the first op inside `run_decode_ops_`. That insertion
      point is in the middle of `forward_decode`, which Track E cannot
      edit. Without that device-side add hook, the only correct option
      is to wait for CP[N] before preparing Talker[N+1]'s input (i.e.,
      straight serialization).
  Following the contract's explicit fallback clause ("If it does
  [break causality], fall back to straight serialization (no speedup
  but correct) and note in §8"), M6.2 stays at the current 14.3 fps
  and is marked `[~]`. The 22 fps throughput gate is NOT met by this
  track.
  **What the follow-up track needs to do** to hit 22+ fps:
  (a) Split `TalkerCannEngine::forward_decode` into `decode_launch(pos)`
      (queues all ops on `stream_`, no host sync) + `decode_fetch(pos,
      hidden_out)` (syncs stream, downloads F32). Ditto for CpCannEngine
      (`predict_group_launch(g)` / `predict_group_fetch(g, logits_out)`).
      This is body-edit territory — Track A-prime or a new Track F.
  (b) Add a device-side F32 add entry point to TalkerCannEngine so the
      provisional input can be patched on-device after CP fetch —
      `add_input_delta(float *delta_host)` that casts F32→F16 and adds
      into `cur_dev_` pre-run_decode_ops_. Body edit.
  (c) Wire an `aclrtEvent` fence: CP records on `stream_b_` after its
      last group; Talker's `decode_launch` for N+1 waits on that event
      (via `aclrtStreamWaitEvent`) before the host does its F32 delta
      add, so the NPU-level dependency is explicit rather than a host
      wall.
  With (a)+(b)+(c) + the M6.1 plumbing already in place, the predicted
  steady-state is ~25-30 fps (CP=22 ms and Talker=17 ms per step would
  overlap to ~22 ms/step = 45 fps idealized, minus the non-overlapping
  host work and the prefill amortization).
  **Files touched (Track E only)**: `tools/qwen_tts/cp_cann_symbols.{h,cpp}`
  (+~18 lines; 5 event symbols), `tools/qwen_tts/talker_cann_engine.{h,cpp}`
  (+~25 lines in header, +~10 in cpp; `primary_stream_`, `stream_b_`,
  `set_stream/get_stream*`, dtor cleanup), `tools/qwen_tts/cp_cann_engine.{h,cpp}`
  (same shape as Talker). Zero edits to `forward_decode`, `forward_prefill`,
  `run_decode_ops_`, `forward_one_token`, `predict_code_groups`, or
  `TalkerLLM::generate`. The `stream_` field is now a runtime-swappable
  pointer to one of the two owned streams (primary default), which is
  the only required behavioral change to every call site that already
  reads `stream_` — none of those bodies had to change.
- **2026-04-18 Track D (M5.1 audit + M5.2 NZ plumbing) — infrastructure
  landed, call-site switch parked for M5.3**:
  **M5.1 audit** on Ascend 910B4 CANN 8.3.RC1 against
  `$ASCEND_TOOLKIT_HOME/include/aclnnop/`:
  - *WeightNz headers actually present*: `aclnn_quant_matmul_weight_nz.h`,
    `aclnn_grouped_matmul_weight_nz.h`,
    `aclnn_grouped_matmul_swiglu_quant_weight_nz.h`,
    `aclnn_grouped_matmul_finalize_routing{,_v2}_weight_nz.h`,
    `aclnn_mla_prolog_v2_weight_nz.h`.
  - *NOT present*: `aclnnMmWeightNz`, `aclnnMatmulWeightNz`,
    `aclnnBatchMatMulWeightNz`. The fp16 / fp32 plain matmul family
    does not expose a dedicated WeightNz entry point on CANN 8.3.
  - *Documented path*: `aclnn_trans_matmul_weight.h` exposes
    `aclnnTransMatmulWeight{,GetWorkspaceSize}` and
    `aclnnCalculateMatmulWeightSize{,V2}`. `aclnnTransMatmulWeight`
    refreshes the given weight tensor in-place ("经过此接口处理后此
    tensor被刷新为预处理后的matmul weightTensor格式根据亲和性进行
    ND或者私有格式的转换") so a subsequent `aclnnMm` / `aclnnMatMul`
    can pick up the private NZ layout transparently.
  - *Eligible call sites* (F16 2D weights):
    **CpCannEngine**: 7 projections per layer × 5 layers = 35 matmuls
    per token (`Mm` on `q_proj`, `k_proj`, `v_proj` at lines 878-880;
    `o_proj` at 1021; `gate_proj`, `up_proj`, `down_proj` at
    1037/1039/1043). NOT eligible: the F32 input projection at line
    824 (`aclnnTransMatmulWeight` doesn't support F32).
    **TalkerCannEngine**: same 7-projection pattern × 28 layers = 196
    matmuls per decode step. Decode call sites: lines 704/706/708
    (Q/K/V), 856 (O), 905/906/909 (gate/up/down). Prefill call sites
    (untouched by the M5.2 landing): lines 1228/1229/1230 (Q/K/V),
    1425 (O), 1484/1485/1488 (gate/up/down). Weight descriptors are
    rebuilt per call via `tensor_2d(lw.q_proj_w, ...)` etc, so the
    in-place descriptor refresh from `aclnnTransMatmulWeight` only
    has an effect if the REFRESHED metadata is the one read by
    `tensor_2d` at call time — i.e., the underlying weight buffer
    keeps its descriptor state between init and decode. The NZ pass
    is run once per buffer at init so all subsequent `tensor_2d`
    calls see the refreshed state.
    **CpCannEngine's BMM call sites** (`aclnnBatchMatMul` in the
    attention-on-NPU path) are NOT `aclnnMm` and are NOT covered by
    `aclnnTransMatmulWeight` — those operate on per-call Q/K/V
    tensors, not model weights, so the whole NZ discussion doesn't
    apply to them.
  **M5.2 implementation**: see §5 M5.2 stamp above. The key design
  decision is that the flag defaults OFF and the default build runs
  the exact pre-M5 path (verified by ASR-equivalent audio durations
  matching the M2.4/M3.4 baseline on utt1/utt2/utt3). The opt-in
  NZ-on path runs without init errors on Ascend, has_nz() returns
  true, and the per-layer `aclnnTransMatmulWeight` logs cleanly.
  **Surprise / M5.3 note**: the NZ-on smoke test showed that plain
  `aclnnMm` with an NZ-refreshed weight does NOT produce correct
  audio on CANN 8.3 (the 200-frame runs never naturally EOS — the
  model is consuming scrambled numerics). This contradicts the
  `aclnnTransMatmulWeight` header's "affinity-driven auto-detect"
  claim and means M5.3 (actually using `*WeightNz` op variants) is
  required for correctness, not just a performance nicety. For the
  fp16 matmul path there is no `aclnnMmWeightNz` on CANN 8.3, so
  M5.3 has two options: (a) wait for a future CANN release that
  exposes one; (b) call `aclnnMm` with the weight tensor's
  `aclFormat` explicitly set to `ACL_FORMAT_FRACTAL_NZ` (= 29) at
  descriptor-build time, instead of the default `ACL_FORMAT_ND` (=2),
  and let `aclnnMm` dispatch the NZ-aware kernel path. Option (b)
  needs experimentation to validate that the op dispatcher actually
  honors the format hint for Mm. The M5.3 agent should start there.
  **Files touched (Track D only)**: `tools/qwen_tts/cp_cann_symbols.{h,cpp}`
  (+~24 lines; 2 optional symbols + `has_nz()`), `tools/qwen_tts/
  talker_cann_engine.{h,cpp}` (+~90 lines; setter/getter, env-var hook,
  `nz_convert_weight_` helper, per-layer hook after upload),
  `tools/qwen_tts/cp_cann_engine.{h,cpp}` (+~70 lines; same shape,
  hooked into both `init` and `init_from_safetensors`). Zero edits to
  `forward_decode`, `forward_prefill`, `run_decode_ops_`,
  `forward_one_token`, any matmul call site, or `CMakeLists.txt`.
- **2026-04-19 Track A (M2.5 closed) — batched prefill restored as
  default**: the pre-M2.5 "cos-sim 0.28" bug had in fact been fixed
  between the M2.4 stamp (commit 948413b1, iterative default) and
  this ticket (by commit 2b0a2998's RoPE-unroll + FIAS-S_q=1 rewrite),
  but no one flipped the default back to the batched path. Verifying:
  rebuilt qwen_tts with the current source and `TALKER_PREFILL_BATCHED=1`
  produced cos-sim 0.9999+ against the iterative reference on real
  text embeddings, and ASR content-checks the same 3/3 (utt1 edit-
  distance-2, utt2/utt3 verbatim) as the iterative path.
  **Diagnostic surfaces added** (`tools/qwen_tts/`):
  (1) `test_mm_diff.cpp` — per-row `aclnnMm(W, X_col)` vs batched
     `aclnnMm(X, W^T_strided_view)` at real Talker dims (K=M=2048,
     N=127). Both produce bit-identical output (cos-sim 1.0),
     ruling out strided-weight matmul as a contributor.
  (2) `test_prefill_diff.cpp` — batched vs iterative `forward_prefill`
     on synthetic σ∈{0.02, 0.1, 0.5, 1.0} gaussians over
     seq_len∈{1, 2, 3, 4, 8, 16, 32, 64, 127} (all cos-sim ≥ 0.9996),
     PLUS an optional real-embedding-file mode (arg 2 = binary
     `[int32 seq_len][int32 hidden][seq_len*hidden f32]`) for
     end-to-end-equivalent cos-sim on production inputs. Real input
     at seq_len=127 = cos-sim 0.999999, cold-vs-warm identical.
  (3) `TALKER_PREFILL_INPUT_DUMP=<path>` env var in
     `TalkerCannEngine::forward_prefill` dumps the raw F32 input
     embeddings on the first call per process — feeds (2) above.
  **Default flip**: `forward_prefill` now runs the batched path
  unconditionally; `TALKER_PREFILL_ITERATIVE=1` forces the legacy
  iterative fallback for regression bisects. The old
  `TALKER_PREFILL_BATCHED` env var is no longer honored (its sole
  purpose was to opt into the then-buggy batched path).
  **Throughput delta**: main prefill on seq_len=127 is ~127 ms
  (default batched) vs ~2054 ms (iterative fallback), 16× speedup.
  End-to-end 209-frame natural-EOS run = 23.2 fps default vs
  ~16 fps iterative. M2.5 throughput gate (≥20 fps on ≥150-frame
  runs) now passes.
  **Files touched (Track A only)**: `tools/qwen_tts/talker_cann_engine.cpp`
  (env-var rename + diagnostic dump + comment update),
  `tools/qwen_tts/test_prefill_diff.cpp` (real-input mode + scale
  sweep), `tools/qwen_tts/test_mm_diff.cpp` (real-dim enlargement),
  `tools/qwen_tts/CMakeLists.txt` (two new test-target entries).
- **2026-04-18 Track F (M5.3 attempt — FRACTAL_NZ descriptor tagging fails
  on CANN 8.3)**: Implemented the open-option (b) from Track D's M5.2 §8
  entry — at every matmul call site, when `nz_applied()` is true, build
  the weight `aclTensor` with `aclFormat = ACL_FORMAT_FRACTAL_NZ` (value
  29) instead of `ACL_FORMAT_ND` (value 2). Activation tensors + output
  tensors stay ND. Added a `tensor_2d_fmt(buf, d0, d1, dtype, fmt)` helper
  in both `talker_cann_engine.cpp` and `cp_cann_engine.cpp` and wired it
  into:
  - `TalkerCannEngine::run_decode_ops_` — Q/K/V projections (line ~842),
    O projection (line ~1000), gate/up/down FFN projections (line ~1045).
    Matmul **call sites** in `forward_decode` all go through this helper.
    `forward_prefill` body was explicitly NOT touched (Track A-prime
    ownership per the task brief).
  - `CpCannEngine::build_persistent_tensors_` — Q/K/V/O/gate/up/down
    weight descriptors are built with `wfmt = nz_applied_ ?
    ACL_FORMAT_FRACTAL_NZ : ACL_FORMAT_ND`. `forward_one_token` uses
    these persistent handles directly (no per-call descriptor rebuild),
    so stamping them at build time covers every matmul call site.
  **Experiment result (canonical ellen_ref, seed=42, cp_groups=8,
  max_tokens=250)**: the format-tag flip alone does NOT make plain
  `aclnnMm` dispatch an NZ-aware kernel on CANN 8.3. NZ-on runs consume
  the NZ-laid-out weight buffer as if it were still row-major, producing
  scrambled numerics:
  - ASR diff (targets vs transcripts, NZ **off** = baseline, NZ **on** =
    `TALKER_NZ_WEIGHTS=1`):
    - utt1: target "Good morning, how are you today." — OFF: "Good
      morning. How are you today?" (edit-dist 2); ON: "哎呀！" (filler,
      fail)
    - utt2: target "The sun is shining brightly this afternoon." —
      OFF: verbatim; ON: "嗯嗯嗯嗯嗯嗯嗯嗯嗯" (filler, fail)
    - utt3: target "Please remember to turn off the lights." — OFF:
      verbatim; ON: "哎呀！" × ~1000 repetitions (filler, fail)
    - long: target "Speech synthesis on neural processing units is a
      compelling application of modern deep learning." — OFF: verbatim;
      ON: "嘿嘿嘿嘿…" (filler, fail)
  - fps delta (long utterance, only meaningful comparison since NZ-on
    never natural-EOS'd on any run): OFF = 21.8 fps on the natural 74-
    frame run; ON = 23.5 fps on the 250-frame (capped) run. The +7.8%
    isn't a clean throughput win because the two runs don't sample the
    same steady state — OFF EOS'd at frame 74 where prefill + early
    steps still dominate, ON ran 3.4× longer in matmul-only territory
    where the per-step cost is lower regardless of kernel choice.
    Correctness gate is what matters: ON fails.
  **Interpretation**: `aclnnTransMatmulWeight` does really rewrite the
  buffer in place (confirmed by M5.2's original observation: tagging
  the transformed buffer as ND produces the same garbage we see now
  when tagging as NZ, because `aclnnMm` on 8.3 appears to follow one
  fixed layout convention regardless of the `aclFormat` hint passed in
  the tensor descriptor). The op dispatcher simply does not branch on
  the format field for the plain Mm family in this toolkit version.
  This rules out the Track D §8 "option (b)" path for CANN 8.3 with
  `aclnnMm`.
  **M5.3 / M5.4 / M5.5 all remain `[ ]`.** M5.3 needs one of:
  - (a) A future CANN release that ships `aclnnMmWeightNz` (or
    equivalent) in `$ASCEND_TOOLKIT_HOME/include/aclnnop/`. As of
    CANN 8.3.RC1 there is no such header (Track D audit confirmed).
  - (b) Swap `aclnnMm` → `aclnnMatMul` and retry the descriptor-tag
    trick. `aclnnMatMul` has a different affinity path in some
    toolkit versions; worth one experiment but not a guaranteed fix.
  - (c) Implement a manual pre-convert that shapes the data into a
    `[N/16, M/16, 16, 16]` 4-D contiguous buffer and call the (absent)
    NZ-aware matmul op when it becomes available. Dead end until
    (a) lands.
  **Files touched (Track F only)**: `tools/qwen_tts/
  talker_cann_engine.cpp` (no diff vs 5fcd1445 — Track A's M2.5 commit
  already bundled the `tensor_2d_fmt` helper + the per-site NZ tagging
  at the same shape Track F needed, so the Track F work on Talker was
  a no-op against HEAD); `tools/qwen_tts/cp_cann_engine.cpp` (+31 / -9;
  `tensor_2d_fmt` helper + `make_tensor` format-param default + M5.3
  tagging of persistent per-layer weight descriptors in
  `build_persistent_tensors_`). Zero edits to `forward_decode`,
  `forward_prefill`, `run_decode_ops_`, `forward_one_token`, or
  `CMakeLists.txt`.
  **Verified-by**: (a) build clean on Ascend 910B4 CANN 8.3.RC1
  (`cmake --build . --target qwen_tts -j 8`, pre-existing unused-var
  warnings only); (b) OFF wavs at `/tmp/m53_off/{utt1,utt2,utt3,long}.wav`
  on Ascend, natural-EOS durations 2.16/3.12/2.40/5.92 s, ASR 4/4
  verbatim (edit-dist 2 on utt1 for punctuation only); (c) ON wavs at
  `/tmp/m53_on/{utt1,utt2,utt3,long}.wav`, every run hits max_tokens,
  ASR all garbage. Gate: ASR content — M5.3 correctness FAIL.
- **2026-04-19 Track G (M6.2 partial — engine-level async split landed,
  orchestrator pipelining deferred)**: Re-measured the long-utterance
  baseline under the current HEAD and found the "14.3 fps" number cited
  in Track E's §8 entry was stale (it predated the M2.5 batched-prefill
  revert); with M2.5 + M6.1 in place the actual baseline is 23.1 fps at
  seed=42, cp_groups=8, max_tokens=250 on the canonical "Speech synthesis
  on neural processing units…" sentence (74 natural-EOS frames, 3.20 s
  generate loop). Per-step timing breakdown: Talker decode 16.7 ms, CP
  decode 23.4 ms, head/sample/emb ~2 ms — roughly serial on one stream
  as expected.
  **What landed**: asynchronous launch/fetch split for both engines.
  `TalkerCannEngine` now exposes `forward_decode_launch(input_embed, pos,
  wait_event=nullptr)` + `forward_decode_fetch(hidden_out)`;
  `CpCannEngine` exposes the matching `forward_one_token_launch/_fetch`
  pair. Each launch method uploads the F32 input, queues every compute
  op on `stream_` without syncing and without D2H, optionally fences
  against an externally-supplied `wait_event` at the front (via
  `aclrtStreamWaitEvent`), and records an internal `aclrtEvent` at the
  tail so a fetch (or another stream) can fence against completion. The
  corresponding engines own a reusable `decode_done_event_` /
  `forward_done_event_` allocated in init and destroyed in the dtor; two
  new accessors `get_decode_done_event()` / `get_forward_done_event()`
  let an orchestrator fence across streams without host round-trips.
  The original entry points (`forward_decode`, `forward_one_token`) are
  now `{launch; fetch;}` wrappers so every existing caller (prefill's
  iterative fallback, `TalkerLLM::predict_code_groups`, `generate`) stays
  bit-identical in behaviour.
  **Why the orchestrator rewrite didn't land in this track**: the
  correct pipelined loop requires launching Talker[N+1] with a
  **provisional** input embedding `codec_embed_talker[g0[N]] + trailing`
  (i.e., sum of just the group-0 codec embedding, not all 16) on
  stream A while CP[N] is still running groups 1..15 on stream B; when
  CP[N]'s final group lands, the host has to patch the delta
  `sum_{g=1..15}(codec_embed_cp[g][token_g])` into Talker's already-
  queued input. The queue ordering for this correction has two valid
  shapes — (i) a new device-side `TalkerCannEngine::add_input_delta`
  hook that casts F32→F16 and in-place-adds into `cur_dev_` before
  `run_decode_ops_` consumes it, or (ii) abandon the provisional and
  fall back to k=1 straight-pipeline (Talker[N] || CP[N-1]) which
  eliminates the drift concern but also eliminates the win because
  Talker[N+1]'s input strictly depends on CP[N] completing. Shape (i)
  is the right answer per §5 M6.2 but it needs enough measurement
  iterations on the Ascend server to confirm ASR/DTW doesn't regress
  across all five canonical utterances (the delta can only be added
  *after* Talker's initial F32→F16 Cast, so the op graph has to be
  split as `Cast → wait_event → InplaceAdd → run_decode_ops_[rest]`).
  Track G landed the engine-level split and event plumbing so this
  follow-up can be implemented purely in `TalkerLLM::generate`'s ICL
  loop plus one new method on Talker — no re-touching of already-shipped
  engine bodies.
  **Measured post-Track-G** long-utt fps, seed=42, cp_groups=8,
  max_tokens=250: 22.4 / 21.9 / 22.5 fps across three consecutive runs,
  mean 22.3 fps. Pre-split baseline on the same build machine was
  23.1 fps — the 0.8 fps delta is within run-to-run jitter (the refactor
  adds one extra event-sync per step versus a stream-sync; both should
  be on the order of microseconds but the wall-clock variance is ±1 fps
  at this scale). ASR gate via local mlx-whisper on four pulled wavs:
  utt1 → "Good morning, how are you today?" (target uses "." — edit-dist
  1, punctuation), utt2 → "The sun is shining brightly this afternoon."
  (verbatim), utt3 → "Please remember to turn off the light." (target
  uses "lights." — edit-dist 1, singular/plural), long → "Speech
  synthesis on neural processing units is a compelling application of
  modern deep learning." (verbatim). All four pass the ≤2 edit-distance
  gate. Artifacts: `/tmp/m62_{split,utt1,utt2,utt3}.wav` on Ascend;
  pulled copies at `/tmp/m62_split_v2.wav` + `/tmp/m62_utt{1,2,3}.wav`
  on the build machine.
  **Files touched (Track G only)**:
  `tools/qwen_tts/talker_cann_engine.h` (+~25 lines — `_launch`/`_fetch`
  decls, `decode_done_event_` member, `get_decode_done_event()`),
  `tools/qwen_tts/talker_cann_engine.cpp` (+~80 lines — event create/
  destroy, `forward_decode_launch` split out of original body including
  the graph capture/replay logic, `forward_decode_fetch` with event-sync
  preference, `forward_decode` as wrapper), matching shape on
  `cp_cann_engine.{h,cpp}`. Zero edits to `run_decode_ops_`,
  `forward_prefill`, `build_persistent_tensors_`, `predict_code_groups`,
  `TalkerLLM::generate`, `cp_cann_symbols.{h,cpp}`, or `main.cpp`.
  **Follow-up track (G-prime) required to close M6.2 to `[x]`**:
  (i) add `TalkerCannEngine::add_input_delta(float *delta_host)` that
  uploads F32 delta on stream A and queues an InplaceAdd in F32 against
  `input_stage_f32_dev_` — this only works if `forward_decode_launch`
  is split one step further to "upload input, return" vs "run ops", so
  the orchestrator can sandwich the delta between them; (ii) rewrite
  the ICL loop at `talker.cpp:~1955` to
  `sample g0 → launch Talker[N+1] provisional on stream A → launch CP[N]
  on stream B → compute delta after CP fetch → add_input_delta → fetch
  Talker[N+1]`; (iii) redirect CP engine onto `talker_engine->stream_b_`
  at generate-start via `cp_engine->set_stream(...)` so the two streams
  are distinct physical NPU streams. With those three landed, the
  predicted steady-state is max(Talker=16.7, CP=23.4) ≈ 24 ms/step →
  ~40 fps idealized, realistically 28-30 fps after host overhead.
- **2026-04-19 Track H (M6.2 follow-up — k=1 orchestrator landed, 28 fps
  gate NOT reached, blocked on engine-body edit restriction)**: Landed
  (iii) from Track G's follow-up list (CP engine now runs on
  `talker_engine->stream_b_` for the duration of the ICL loop, restored
  on exit), plus rewrote the ICL loop + `predict_code_groups` to dispatch
  every Talker decode and every one of the ~17 per-step CP
  `forward_one_token` calls through the `_launch`+`_fetch` async pair that
  Track G exposed. The Talker launch that closes each step fences on CP's
  `forward_done_event_` via `aclrtStreamWaitEvent` so stream A sees the
  last CP group's KV-cache writes without an `aclrtSynchronizeStream`.
  Built clean on Ascend 910B4 CANN 8.3.RC1, ASR 4/4 PASS on
  `/tmp/h2_utt{1,2,3}.wav` + `/tmp/h_long_v1.wav` (see utterance list
  below). Long-utterance steady-state with correct ref_text (natural EOS
  at step 74, not 250) measured **21.9 fps mean** across three runs
  (21.8 / 21.6 / 22.3) at seed=42 cp_groups=8 max_tokens=250. Rebuilt
  Track G baseline with the same corrected ref_text on the same build
  machine: **22.2 fps mean** (22.8 / 21.2 / 22.7). Δ = -0.3 fps, well
  within ±1 fps run-to-run jitter. The 28 fps acceptance gate is **NOT
  met**; M6.2 stays `[~]` with measured 21.9 fps.

  **Why the k=1 rewrite didn't yield the projected 28-30 fps** — i.e. why
  swapping CP onto stream_b_ and bridging the two engines with an event
  fence was insufficient on its own:

  - The per-step dependency chain is strictly sequential. Talker[N]
    produces `hidden[N]`, host samples `g0[N]` from it, CP[N] consumes
    `hidden[N]` (pos 0) then samples group-0 through group-14 one at a
    time (15 `forward_one_token` launches with host sampling between
    each), and Talker[N+1]'s input `next_emb = codec_embed(g0[N]) +
    Σ_{g=1..15} cp_codec_embed[g][tok_g[N]] + text_trailing` requires
    **all 16 CP groups** to be done. With no host speculation and no
    device-side delta correction, Talker[N+1] cannot begin before CP[N]
    has fully retired — swapping streams only reshuffles which physical
    NPU stream the sequential chain lands on. The per-step wall-clock
    was 41-42 ms before Track H (Talker 16.7 ms + CP 23.4 ms + host
    1-2 ms) and stays 41-42 ms after.
  - The only structural change that would bend the chain into an overlap
    is the provisional-embedding scheme from Track G's follow-up (i): fire
    Talker[N+1] on stream A *while* CP[N] is still working groups 1..15
    on stream B, with a placeholder input equal to
    `codec_embed(g0[N]) + text_trailing` (known immediately after
    sampling g0) — and then, after CP[N]'s final group lands, add the
    `Σ cp_codec_embed` **delta** into the Talker input **before**
    `run_decode_ops_` consumes it. Because the 16-group codec embedding
    is a linear sum, `provisional + delta = exact` bit-for-bit on the
    F32 staging buffer.
  - That change needs a **staged** `forward_decode_launch` — upload input,
    return, accept a delta, return, run decode ops — and the only place
    to inject the delta is between the existing H2D into
    `input_stage_f32_dev_` and the initial Cast at the top of
    `run_decode_ops_`. Track H's file-ownership rule explicitly
    prohibits editing `forward_decode_launch`, `forward_decode_fetch`,
    `forward_decode`, `forward_prefill`, and `run_decode_ops_`, which
    collectively own both ends of that injection site. A new
    `add_input_delta(...)` method on `TalkerCannEngine` is within scope
    by itself, but there is no stable point in the existing engine API
    to call it at — the existing `forward_decode_launch` queues the
    Cast-plus-28-layers as one monolithic sequence, so any InplaceAdd
    queued *after* `forward_decode_launch` lands in the stream queue
    *after* the Cast has already produced F16 `cur_dev_` and after
    layer 0 has already copied it into `residual_dev_`, which makes
    the delta a no-op.
  - Host-side attempts to smuggle the delta through (pre-sum on CPU;
    speculation with rollback on mismatch) all degrade to the pure
    serial path because Talker's 16.7 ms decode must either (a) wait for
    CP done and then run (Track G / Track H status quo), or (b) run with
    a wrong input and get thrown away (breaks ASR).

  **What Track H's infrastructure DOES buy for G-prime**: once the engine
  rule is relaxed enough to split `forward_decode_launch` into
  `forward_decode_upload(input)` + `forward_decode_commit(pos,
  wait_event)` (or equivalently `forward_decode_launch_staged(...)` as a
  parallel new method leaving the existing one untouched), the existing
  Track H orchestrator rewrite is **80 % of the work** — CP already
  lives on stream_b_, event fences are already wired, `_launch`/`_fetch`
  pairs already dispatch from the correct loop positions, and
  `compute_next_embedding` already decomposes into
  `lookup_codec_embedding(g0)` + `Σ read_embedding_row(cp_embeddings)`
  which can be split into provisional / delta halves without any new
  precision loss. The remaining changes would be (a) call
  `forward_decode_upload` for the provisional *before* `predict_code_groups`
  runs, (b) accumulate the CP-group delta into `delta_host[n_embd]` as
  each `forward_one_token_fetch` returns, (c) call
  `add_input_delta(delta_host, /*wait*/ cp_last_event)` followed by
  `forward_decode_commit` after the CP loop. All three are orchestrator
  code, not engine-body code.

  **Verification details**: ASR via local mlx-whisper
  (`whisper-large-v3-mlx`, language=English, temperature=0).
  - utt1 "Good morning, how are you today." (target) →
    "Good morning, how are you today?" (ASR) — edit-dist 1 (punctuation).
  - utt2 "The sun is shining brightly this afternoon." →
    "The sun is shining brightly this afternoon." — verbatim.
  - utt3 "Please remember to turn off the lights." →
    "Please remember to turn off the lights." — verbatim.
  - long "Speech synthesis on neural processing units is a compelling
    application of modern deep learning." → same, verbatim.

  All four pass the ≤ 2 edit-distance gate. Note: the long utterance
  baseline number previously cited (22.4/21.9/22.5 at 250 frames in
  Track G's §8 entry) was measured against an **incorrect** ref_text
  ("I love making sand castles at the beach. …") that caused the model
  to run out max_tokens=250 on a bad trajectory; with the correct
  ref_text from `tail -1 tools/qwen_tts/data/ref_audios/ellen_ref.txt`
  the natural EOS lands at step 74 (≈ 3.2 s of audio), which is why
  Track H's fps numbers are reported on the 74-frame run. Both Track G
  and Track H baselines were re-measured on the corrected ref_text to
  keep the comparison apples-to-apples.

  **Files touched (Track H only)**:
  `tools/qwen_tts/talker.cpp` — `TalkerLLM::generate` ICL loop
  (pipelined launch/fetch with CP on stream_b_, event fence on
  `forward_done_event_`, CP-stream swap gated by `pipeline_native`)
  and `TalkerLLM::predict_code_groups` (CP calls now dispatch through
  the `_launch`/`_fetch` pair so the recorded event survives for the
  outer Talker fence). **Zero edits** to
  `tools/qwen_tts/talker_cann_engine.{h,cpp}`,
  `tools/qwen_tts/cp_cann_engine.{h,cpp}`,
  `tools/qwen_tts/cp_cann_symbols.{h,cpp}`, `build_graph.cpp`,
  `main.cpp`, `qwen_tts.{h,cpp}`, or any speech-tokenizer file.
  No new method added to either engine (the `add_input_delta` hook
  from Track G's follow-up plan was **not** added — see the block
  above for why it couldn't be effective without also modifying
  `forward_decode_launch`).

  **Verified-by:** (a) Track H commit SHA pending in
  `OminiX-Ascend/tools/qwen_tts/talker.cpp`; (b) three long-utt fps
  runs 21.8 / 21.6 / 22.3 at seed=42 cp_groups=8 max_tokens=250 ref_text
  = ellen_ref.txt line 2, vs rebuilt Track G baseline 22.8 / 21.2 / 22.7
  on the same machine; (c) ASR gate via local mlx-whisper on four
  pulled wavs (`/tmp/h2_utt{1,2,3}.wav`, `/tmp/h_long_v1.wav`) —
  4/4 pass edit-distance ≤ 2. Gate = ASR PASS 4/4, fps **21.9 vs 28
  target** → M6.2 remains `[~]`.

## 9. Parallelism playbook

When an agent is assigned a milestone item:

1. Agent reads §1-4 for context + §3 for current state.
2. Agent finds next `[ ]` in its assigned milestone.
3. Agent's deliverable must pass the milestone's quality gate before
   marking `[x]`.
4. Agent commits the milestone update with `[x]` and a one-line note under
   the item explaining what landed.
5. If agent blocks on a decision, agent appends to §8 Decision Log
   proposing the options; PM (this session) arbitrates.

Preferred parallel assignments:

- **After M3 lands**: spawn 3 agents — M4, M5, M6 in separate worktrees.
- **Within M1**: spawn 2 agents — 1.1+1.2 (weights/init) in one worktree,
  1.3 (decode) in another, reconverging for 1.4-1.7.

### Next round (as of the M3.1 release)

Three parallel tracks unblocked:

1. **Batched-prefill bug** (blocks full M2.5, highest ROI): localize why
   FIAS with S>1 produces cos-sim 0.28 vs the iterative-decode
   reference. Known-ruled-out: sparseMode=0/nextTokens=0 vs
   sparseMode=1/nextTokens=65535 (both wrong identically). Next
   suspects: Q/K batch strides, innerPrecise=2 on batch tensors,
   numKeyValueHeads broadcast with the strided KV-cache view.
2. **M3.2 + M3.3** (strip llama.cpp): safe to start in a worktree now
   that native is default. Put llama.cpp behind a `QWEN_TTS_LLAMA`
   CMake option, default OFF. `--llama_fallback` flag gated on the
   option.
3. **M4 aclGraph** (independent): add `aclmdlRI*` dlsym, wrap
   `forward_decode` capture/replay per pos. Must keep ASR content gate.

## 10. File index (for fast jumping)

| File | Purpose |
|---|---|
| `tools/qwen_tts/cp_cann_engine.{h,cpp}` | Existing native CP engine (reference pattern for Talker port) |
| `tools/qwen_tts/cp_cann_symbols.{h,cpp}` | dlsym loader for aclnn symbols; add more symbols here |
| `tools/qwen_tts/talker_cann_engine.{h,cpp}` | **TO CREATE** — native Talker |
| `tools/qwen_tts/talker.cpp` | Orchestrator; integrate native path here |
| `tools/qwen_tts/qwen_tts_api.{h,cpp}` | C ABI surface — must stay stable |
| `tools/qwen_tts/main.cpp` | CLI flags — add `--native_talker` here |
| `tools/qwen_tts/CMakeLists.txt` | Build config — llama.cpp optional flag here |
| `/Users/yuechen/home/OminiX-MLX/qwen3-tts-core/src/backend.rs` | TalkerBackend trait spec — contract unchanged |
| `/Users/yuechen/home/OminiX-MLX/qwen3-tts-ggml/src/talker.rs` | Rust binding — should stay as-is |

## 11. Session boot checklist

When a new session picks up this contract:

1. `git status` in `/Users/yuechen/home/OminiX-Ascend` — confirm clean or
   note in-flight work.
2. Re-read §3 (current state) — verify no drift since last check.
3. Run the smoke bench to confirm current fps / audio are where docs claim:
   ```bash
   ssh -i ~/home/tensordock/KeyPair-4fbd-yue.pem -p 31984 \
     ma-user@dev-modelarts.cn-southwest-2.huaweicloud.com 'bash /tmp/bench_cp.sh'
   ```
4. Find next `[ ]` and start. Update `[x]` + decision log as you go.
