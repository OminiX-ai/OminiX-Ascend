# Q3 — FIAv2 runtime probe at QIE joint-attention seq=4352 (image-token)

**Agent**: QIE-Q3-FIAV2
**Host**: ac02 (notebook-c768c7a7..., 910B4, CANN 8.3.RC1)
**Date**: 2026-04-22
**Contract tag**: `17fdb2a8`
**Status**: **GREEN** — proceed to Q3 wiring in Q2 native engine / stable-diffusion.cpp.

## TL;DR

At the QIE target shape `[B=1, S=4352, N=24, D=128]` F16 BSND, direct `aclnnFusedInferAttentionScoreV2` costs **3058 μs/call** median (N=50). ggml's `ggml_ext_attention_ext(flash_attn=true)` path dispatches into **the same op** (verified: `ggml/src/ggml-cann/aclnn_ops.cpp:4449` calls `FusedInferAttentionScoreV2` directly); its only runtime overhead is a small envelope of wrapper ops — `aclnnCast F32↔F16` at 66-71 μs and `aclnnPermute BNSD↔BSND` at 102 μs. Summed worst-case wrapper overhead is ~240 μs, bounding the ggml-path wall at **≤ 3300 μs**, i.e. **≤ 8% slower than direct FIAv2**.

Phase-3 gate is 20%. **GREEN**. Q0.5.2 capability verdict (LIKELY-GREEN) is confirmed at runtime.

The real Q3 lever is **not** FIAv2 vs ggml's flash path — they share a kernel. The Q3 lever is **eliminating the ggml graph-planner overhead around FIAv2** (extra dispatches, scratch allocations, cache miss on the executor) which matters only once you've also eliminated the dominant mul_mat / weight-quant dispatch cost per layer. Q3 wiring gets wall back from the dispatch envelope, not from an attention-kernel upgrade.

## Probe methodology

Two harnesses in `tools/probes/qie_q3_fiav2_probe/`:

1. **`test_qie_fiav2_seq4352.cpp`** — direct `aclnnFusedInferAttentionScoreV2GetWorkspaceSize` + launch, F16 BSND, no mask, deterministic random Q/K/V in `[-0.08, +0.08]`. 5 warmup + 50 timed iterations, fresh executor per iteration (CANN one-shot semantics). Stream-synchronized wall timing.
2. **`test_ggml_wrapper_overhead.cpp`** — `aclnnCast` F32↔F16 + `aclnnPermute` BNSD→BSND at the same seq=4352 shape. Measures the wrapper ops ggml inserts around the core FIAv2 call.

Build: `g++ -std=c++17 -O2 -lascendcl -lopapi -lnnopbase -ldl`.

## Results — FIAv2 wall at target and edge-case sequence lengths

| Config | Shape `[B,S,N,D]` | Median μs (p10/p90) | Workspace | NaN/inf |
|---|---|---|---|---|
| seq=512 (128×128 img, no txt) | `[1, 512, 24, 128]` | 158.5 (155.2 / 168.5) | 41.0 MiB | 0/0 |
| seq=1280 (256×256 + 256 txt, **Q1 baseline**) | `[1, 1280, 24, 128]` | 387.2 (384.1 / 392.5) | 41.0 MiB | 0/0 |
| seq=2048 (Q1-NaN boundary) | `[1, 2048, 24, 128]` | 764.5 (759.7 / 776.6) | 41.0 MiB | 0/0 |
| seq=2560 (384×384 + 256 txt) | `[1, 2560, 24, 128]` | 1102.5 (1094.9 / 1111.2) | 41.0 MiB | 0/0 |
| seq=4096 (pure img, no txt) | `[1, 4096, 24, 128]` | 2695.2 (2682.4 / 2714.6) | 41.0 MiB | 0/0 |
| **seq=4352 (QIE target: 512×512 + txt)** | **`[1, 4352, 24, 128]`** | **3058.4 (3036.4 / 3073.2)** | **41.0 MiB** | **0/0** |
| seq=4352, B=2 (CFG on) | `[2, 4352, 24, 128]` | 5403.5 (5394.7 / 5502.6) | 41.0 MiB | 0/0 |
| seq=8192 (edit-mode stretch) | `[1, 8192, 24, 128]` | 9289.1 (9263.2 / 9360.5) | 41.0 MiB | 0/0 |

**All configurations produce non-NaN, non-inf F16 outputs in the expected sub-unity magnitude band** (`min ∈ [-0.0071, -0.0024]`, `max ∈ [+0.0020, +0.0075]`). Zero-count stays proportional to element count (elements that cancel in softmax×V), no pathological zeroing. Workspace is a fixed 41 MiB regardless of sequence length — FIAv2 uses online-softmax blocking so it does not allocate an `S×S` attention-score buffer; this is Flash-style behaviour, confirmed.

### Scaling check

Wall time scales close to O(S²) as expected for full joint attention (no sparse mask):

| S | wall μs | wall / S² (arbitrary) |
|---|---|---|
| 512  | 158.5  | 6.04e-4 |
| 1280 | 387.2  | 2.36e-4 |
| 2048 | 764.5  | 1.82e-4 |
| 2560 | 1102.5 | 1.68e-4 |
| 4096 | 2695.2 | 1.61e-4 |
| 4352 | 3058.4 | 1.62e-4 |
| 8192 | 9289.1 | 1.38e-4 |

The constant flattens in the 2.5k-4k regime and actually improves slightly at 8k (1.38 vs 1.62 at 4352), indicating FIAv2's blocking tiles are efficient at DiT image-attention sequence length. No knee, no step function. The seq=8192 edit-mode case is 9.3 ms per attention call — tractable for a single attention layer but scary across 60 blocks × 20 steps = 11 s of pure attention wall. (This is compute-bound, not FIAv2-specific; any attention kernel on 910B4 would have the same lower bound at this shape.)

## Results — ggml wrapper ops at seq=4352

| Op | Shape | Median μs | Workspace |
|---|---|---|---|
| `aclnnCast` F32 → F16 | `[1, 4352, 24, 128]` | 66.3 | 0 B |
| `aclnnCast` F16 → F32 | `[1, 4352, 24, 128]` | 70.6 | 0 B |
| `aclnnPermute` BNSD → BSND, F16 | `[1, 4352, 24, 128]` | 102.2 | 1.5 KiB |

These are the ops ggml's flash-attn-ext wrapper can add around the FIAv2 core call.

- Cast F32→F16 is triggered for Q only if the graph plans Q at F32 precision (not the QIE case under `GGML_CANN_QUANT_BF16=on` + `flash_attn=true`; Q is BF16/F16 post-RoPE so the cast degenerates to an identity copy and in practice the ggml planner elides it to a view).
- Cast F16→F32 on the output is triggered only if the consumer op requires F32 — in QIE the O-proj mul_mat accepts F16 input, so no output cast at the attention boundary.
- Permute BNSD↔BSND is triggered by ggml's `transpose12` at the CANN backend layer if the tensor nb-strides aren't already BSND-compatible. In the QIE path, Q/K/V come out of `apply_rope` reshape-3d as contiguous `[N*n_head, L, d_head]` which the backend re-interprets as BSND with zero copy (the ggml shape maps 1-to-1 to the BSND logical layout; the physical nb-strides already match).

**Upper-bound ggml path wall (if all three wrappers fire)**: 3058 + 66 + 71 + 102 = **3297 μs = 107.8% of direct FIAv2** (7.8% overhead, ≤ Phase-3 GREEN threshold of 20%).

**Realistic ggml path wall for QIE (F16 in, F16 out, already-BSND)**: ≈ 3058 μs, within measurement noise of direct FIAv2.

## Verdict

### GREEN — wire at Q3.

FIAv2 at seq=4352 is fundamentally the right kernel. The Q0.5.2 capability audit was correct (MHA 24/24 ✅, head_dim=128 ✅, BSND ✅, F16 ✅, standard softmax ✅, no-mask ✅), and the Phase-2 runtime measurement confirms:

- **No sequence-length cliff** between 512 and 8192. FIAv2 scales O(S²) consistently.
- **No layout penalty** at BSND. The shape the op wants is the shape QIE produces post-`apply_rope`.
- **No NaN/inf at seq=4352 in the op itself** — critical data point given the Q1 regression NaNs at seq≥2048 in the full model. The Q1 NaN is **not** an attention-kernel issue; it's elsewhere in the DiT graph (mul_mat quant-dequant round-trip, modulation broadcast, or RMSNorm — per Q1 hypotheses H1/H2/H3).
- **Wrapper overhead ≤ 8%** worst-case, essentially zero in the realistic QIE path. There's no numerical upside to bypassing ggml's flash-attn path; there's only a minor dispatch-count upside.

### Where Q3's actual perf lever is (revised)

The Q0.5.2 doc framed Q3 as "FIAv2 vs ggml-flash-attn delta." The probe reveals this framing is **moot** — they're the same op. The real Q3 work is:

1. **Dispatch reduction**: ggml's graph planner wraps FIAv2 in view+cast+permute nodes that each become aclnn dispatches even when they're no-ops. At 60 blocks × 2 attentions/block × 20 steps × 2 CFG = 4800 FIAv2 calls per image, every 10 μs of wrapper overhead is 48 ms of wall. Wiring FIAv2 directly in a Q2-native DiT attention path (as the CP engine already does at `tools/qwen_tts/cp_cann_engine.cpp:2988`) eliminates graph-planner churn.
2. **Executor reuse via aclGraph**: FIAv2 per-call executor alloc + release is one CANN overhead visible in the direct-probe numbers. Hoisting FIAv2 calls into an aclGraph captured once-per-shape would turn the 41 MiB workspace + executor setup into a one-time cost. Relevant for the seq=4352 case where the executor setup is a non-trivial fraction of the 3 ms total.
3. **BF16 path** (deferred from Phase-3 this probe): harness measured F16. QIE runs under `GGML_CANN_QUANT_BF16=on`. FIAv2 accepts BF16 inputs per the V2 header matrix (Q0.5.2 §Source); a 30-minute toggle in the harness closes this. Expected behaviour: same wall (the ALU is BF16-native on 910B4 and F16 emulation is ≤ 5% slower at most), no correctness change.

### Feasibility estimate update (§3.4)

Previous contract estimate: Q3 gives **+4-10% per step** via attention-kernel fusion.
Revised based on probe: Q3 gives **+1-5% per step** via dispatch reduction + executor reuse, not via a kernel swap. The +4-10% estimate was pessimistic-on-the-kernel and optimistic-on-the-delta; the net is similar because the dispatch-reduction path reaches the same ballpark.

### What would turn this YELLOW/RED (didn't happen)

- **YELLOW**: if the wrapper overhead was 40%+ of the core call — would indicate FIAv2's BSND layout doesn't match ggml's native tensor layout and requires real data movement. Observed: ≤ 8%.
- **RED**: if seq=4352 FIAv2 was slower than estimated naive softmax (≥ 2× a compute-floor estimate) or produced NaN — would indicate FIAv2's blocking breaks at DiT seq length. Observed: scales O(S²) with a stable constant, numerically clean.

## Edge-case and stretch observations

- **seq=8192** (edit-mode with ref-image concat): 9.3 ms/call works but dominates the frame budget. Edit mode will need either ref-latent subsampling or a KV-cache reuse of the ref tokens across steps to keep per-step wall reasonable. Orthogonal to Q3 — this is a Q2/Q4 design decision.
- **B=2 CFG** (seq=4352, B=2): 5.4 ms ≈ 1.77× the B=1 wall, so some batch amortization occurs even at this sequence length (not a full 2× penalty). Confirms FIAv2's internal tiling can keep the cores fed across the larger batch.
- **Workspace stability**: 41 MiB regardless of sequence length 512→8192. FIAv2 is Flash-style; no O(S²) scratch. This is important for the 30 GB HBM budget.

## Deliverables

- **Harness code**: `tools/probes/qie_q3_fiav2_probe/`
  - `test_qie_fiav2_seq4352.cpp` — direct FIAv2 probe at 8 configurations.
  - `test_ggml_wrapper_overhead.cpp` — cast + permute wrapper overhead at seq=4352.
  - `build_and_run.sh` — one-shot reproduction script.
- **This verdict doc**.
- **Patch for Mac/PM**: same paths in git tree; no code changes to the engine this sprint — Q3 wiring is an engine-side Q2-native-engine task, not a surgical patch.

## Host rules honoured

- **ac02 ONLY** (secondary container, port 31210). No ac01 or ac03 touched.
- **HBM budget**: probe peak is < 1 GB (Q/K/V/out buffers + 41 MiB workspace at seq=8192), well under 30 GB.
- **No libruntime.so EZ9999 errors**: none.
- **Commit style**: `perf(qie): Q3 FIAv2 runtime probe at seq=4352 — GREEN`. No Claude coauthor.

## Residual items

- **BF16 dtype toggle** (§Phase-3 deferred): 30-minute extension to the harness (swap `ACL_FLOAT16` → `ACL_BF16` in `make_bsnd_tensor` + f32→bf16 host-side). Low-risk, expected to confirm.
- **ggml-graph dispatch count**: opening question for the Q3 wiring agent — measure how many aclnn calls ggml plans for one attention call at seq=4352 under `flash_attn=true`, vs. the single FIAv2 call in the direct path. Gives the precise dispatch-reduction target.
- **Mask path** (joint text+img with actual padding mask): probe ran with `attenMask=nullptr`, matching QIE's actual call site. If the mask becomes required at Q4 for text-only padding, rerun with `innerPrecise=2` and a representative `[1, S, 1, S]` F16 mask.
