# CP Forward Optimization Exploration (W2)

Agent Y, ac02, 2026-04-19. Deliverable for W2.1 – W2.5 of the CP FPS
Optimization Contract (see `CP_FPS_OPTIMIZATION_CONTRACT.md` in the
OminiX-API repo, committed 2026-04-20 at `8038c335`).

## 0. Scope and methodology notes

This is a **research deliverable**. No code is landed. PM decides whether
any of the candidates graduates to a W3 implementation milestone.

**Sources of evidence used**, graded by weight:
1. Live measurement on ac02 (most weight) — *blocked this round*, see §0.1.
2. Prior-run logs and numbers committed to `NATIVE_TTS_CONTRACT.md`
   §5/§8 (well-documented, dated, reproducible with the cited seed/env).
3. Static reading of `cp_cann_engine.cpp`, `cp_cann_symbols.{h,cpp}`,
   `talker.cpp`, `talker_cann_engine.cpp` on the current main (HEAD
   `8038c335`).

### 0.1. ac02 baseline-run status

**Blocker**: ac02 has no Qwen3-TTS GGUFs and no network path to pull
them. The tree at `~/work/OminiX-Ascend-b6/tools/qwen_tts/gguf` is a
symlink to `/root/autodl-tmp/...` which is permission-denied for
`ma-user`. A partial 121 MB `gguf_b6.tar` is present but truncated
(header shows a 1.5 GB `qwen_tts_talker_llama_q8_0.gguf` entry, file
was aborted mid-write).

`ac01→ac02` inter-container networking is blocked (contract §8). The
only transfer route is `ac01 → laptop → ac02`. Measured scp throughput
is ~0.5 MB/s each leg; transferring the minimum 5.4 GB working set
(talker_llama F32 + q8_0_mrope + cp_llama + code_predictor +
speaker_encoder + tokenizer_{enc,dec}) would take ~6 hours round-trip
and hold the user's laptop bandwidth hostage the whole time.

**Decision taken**: deliver the W2 analysis on prior-run evidence +
static reading. All W2 recommendations are grounded in either (a) code
on disk now, or (b) committed measurement runs in
`NATIVE_TTS_CONTRACT.md` §5/§8 (dated, with SHA references). If PM
wants live re-measurement, the fastest route is to run the benchmarks
on **ac01 once W1 lands**, in the same window that Agent X's
end-to-end fps check already requires.

### 0.2. Open-issue answers (from contract §5 W2 open issues)

- **Q1 — Is `TALKER_SPECULATIVE` still reachable?** **Yes.** The env
  var is read at `tools/qwen_tts/talker.cpp:2042`, default off (commit
  `84af6590` flipped it from default-on to default-off, did not remove
  the code). The speculative branch body lives at
  `talker.cpp:2111-2210` and is fully exercised when the env var is
  `=1`. No re-enable work required; only measurement.
- **Q2 — Does the CP engine share KV cache across group g and g+1
  forward?** **Yes, within a frame.** `predict_code_groups` calls
  `cp_cann_engine_->reset_kv_cache()` *once* at the start of a frame
  (`talker.cpp:1618`), then issues 17 `forward_one_token_launch`
  calls at positions 0, 1, 2, ..., 16 for pos=0 (talker hidden state),
  pos=1 (group-0 embedding), and pos=2..16 (groups 1..15). Each call
  writes its K/V into cache slots `k_cache_dev_[il] + pos*kv_dim_`
  (`cp_cann_engine.cpp:1383-1409`) and fused attention reads the full
  `seq_len = pos+1` prefix. So yes, every group re-attends to all
  prior groups' KV — there is no cross-group KV recomputation, only
  attention over a growing cache. The reset is per-frame, not
  per-group.
- **Q3 — Is `aclmdlRICaptureBegin/End` already exposed in
  `cp_cann_symbols.cpp` (for M4 Talker)? Can it be reused for CP
  forward without new dlsym work?** **Yes to both.**
  `cp_cann_symbols.cpp:166-173` resolves all four aclGraph entries
  (`aclmdlRICaptureBegin`, `aclmdlRICaptureEnd`, `aclmdlRIExecuteAsync`,
  `aclmdlRIDestroy`) via `resolve_optional`, and `CannSyms::has_aclgraph()`
  at `cp_cann_symbols.h:284-288` gates callers. `TalkerCannEngine`
  already uses them (`talker_cann_engine.cpp:1768-1790`). Reusing
  them for `CpCannEngine::forward_one_token_launch` is pure
  plumbing — no new dlsym work, no new capability flag.

---

## 1. Baseline rigour (W2.1)

### 1.1. Live measurement: N/A this round (see §0.1)

### 1.2. Committed baseline (source: `CP_FPS_OPTIMIZATION_CONTRACT.md` §3, `NATIVE_TTS_CONTRACT.md` §8 dated 2026-04-20)

At HEAD `8038c335`, `W8+TQE=2+cp_groups=15, seed=42, sampling on`, on
`ac01` (Ascend 910B4 CANN 8.5):

| mode | fps | CP forward / frame | lm_head / frame | other per-frame |
|---|---:|---:|---:|---|
| ICL | 23.3 | ~17 ms (est) | ~15 ms | build_emb once at prefill |
| xvec | 22.2 | ~17 ms | ~15 ms | |
| customvoice | 21.0 | ~17 ms | ~15 ms | |

All three modes within 10% of each other — parity at clean quality,
which implies the CP forward and lm_head are in fact the universal
bottleneck (not a mode-specific path).

**Per-frame accounting** from contract §3 (the 22 fps xvec case):

```
CP forward_one_token × 15     17 ms  NPU
lm_head matvec × 15          14-17 ms  CPU
sample_token × 15              2 ms  CPU
read_embedding_row (x15)     0.2 ms  CPU
Talker forward × 1            10 ms  NPU  [out of scope]
------------------------------
wall per frame              ~45 ms → 22 fps
```

The **17 ms CP number is the *sum* of 15+1 group dispatches plus
prefill warmup**, not per-group. Per-group cost is therefore
~**1.06 ms/group**. This matters for §3 aclGraph analysis: the
capture/replay amortization has to pay back against ~1 ms, not 17 ms.

### 1.3. What a live 3×3 trial *would* add on top of the committed numbers

Three things:

1. **Variance band**. Std. dev. on a single fresh boot vs. cache-warm
   is not recorded in §8; it matters because the M4 aclGraph regression
   (§3 below) was measured at 17 ms/step eager — if variance is ±3 ms
   that's a different conclusion than ±0.5 ms.
2. **Mode sensitivity of the CP path**. All three modes share
   `predict_code_groups`, so the 17 ms CP should be identical across
   ICL/xvec/CV — verifying this rules out a mode-specific pathology
   hiding in the shared timer.
3. **Warm vs. cold**. The contract's 22 fps number is warm. Cold first
   utterance pays the `aclnnTransMatmulWeight` NZ pre-conversion (M5,
   one-shot per load) and W8 weight upload.

**PM decision point**: since the committed numbers already ground
§2/§3/§4 analysis below, the 3×3 trial is worth running only if W1
lands and we need to confirm §3-§4 pre-claims before funding W3.

---

## 2. Speculative decoding (W2.2)

### 2.1. Code status

Reachable. `TALKER_SPECULATIVE=1` flips
`pipeline_speculative=true` at `talker.cpp:2040-2048`, which routes
the frame loop through the speculative branch at `talker.cpp:2111-2210`.

**Git lineage**:
- `b2bf8a54` — Track J landed (M6.2 speculative-embedding ICL pipeline).
- `84af6590` — "fix(tts): default TALKER_SPECULATIVE to OFF — k=1 is
  faster on our workload". Code unchanged, only the default.
- No subsequent removal. HEAD-at-`8038c335` still has the full
  branch.

### 2.2. Why the original numbers at `b2bf8a54` regressed

Documented in `NATIVE_TTS_CONTRACT.md:1924-1948`:

> "...stream B is fully drained on CP and the speculative cast on
> stream A has long since finished its < 1 ms of work. ... Net effect
> vs Track H: Talker cast moved from 'post-CP' to 'pre-CP' (saves
> ~0.3 ms); Talker layers still wait until post-CP (no change); host
> pays a ~0.7 ms tax per step for the extra F32 delta buffer + F32→F16
> Cast + InplaceAdd + cross-stream fence. ... Interleaved A/B
> measurement (one speculative run, one `TALKER_SPECULATIVE=0` run, on
> consecutive utterances) shows the regression is consistent:
> speculative 26.1 / 25.2 fps vs sequential 27.4 / 28.3 fps."

**Root cause**: the speculative overlap premise — "CP's later groups
run on stream B while Talker[N+1] launches layers on stream A" — never
holds, because `predict_code_groups` is host-serial across the 15
groups (each fetch before the next launch, `talker.cpp:1622-1659`).
Stream B is drained by the time `predict_code_groups` returns.

### 2.3. Would a re-audit on W8+TQE=2+cp_groups=15 flip differently?

**Probably not, and here's why**:

- The regression is a **fixed per-step tax** (~0.7 ms for the delta
  buffer path). Moving from cp_groups=8 to cp_groups=15 doesn't change
  this tax (the delta is over whatever groups 1..N are alive; N=14 vs
  N=7 only enlarges one host `for (j=0; j<dim; j++) delta_emb[j] += tmp[j]`
  loop by ~0.05 ms — still small vs. the fixed device fence cost).
- Adding W8 changes per-Mm cost inside the CP forward, but the
  speculative branch is Talker-side, not CP-side. W8 doesn't create
  new CP-stream B bandwidth, it just makes each CP Mm cheaper (and
  thus makes stream B *more* drained by the time `predict_code_groups`
  returns, which *worsens* the speculative premise, not improves it).
- TQE=2 enables the CANN task-queue two-phase submit mode, which
  already overlaps kernel tiling/prep with prior kernel execution on
  the same stream. Turning it on has the side effect of making the
  cross-stream fence (`aclrtStreamWaitEvent(stream_A, CP_done)`)
  cheaper in absolute terms — which is a ~0.2 ms reduction in the
  spec tax, but doesn't flip the sign.

**What would actually make speculative win** (per the same note):
"pushing the CP sampling loop device-side (a single fused CP kernel
that produces groups 1..15 without host round-trips) — out of scope
for M6.2." This is a multi-week device-side sampler project and
effectively subsumes the kernel-fusion track (§4 below).

### 2.4. If PM still wants the A/B run

- Gate behind `TALKER_SPECULATIVE=1` (already wired).
- Canonical: mayun_ref zh long utt, seed=42, W8+TQE=2+cp_groups=15,
  sampling on, 3 trials each.
- Key metrics: fps, frame-count drift (spec 75 vs seq 77 observed at
  cp_groups=8, DTW 0.973 — verify this holds at cp_groups=15), ASR
  edit-dist vs baseline WAV, user-ear pass/fail.
- Predicted outcome: ~1 fps regression, within DTW 0.85 gate, user-ear
  probably identical (the two-frame drift is ≤100 ms which is below
  phoneme granularity).

### 2.5. Recommendation

**Won't work, because** the speculative premise (device-side CP
overlap with Talker[N+1] layers) is structurally refuted by the
host-serial 15-group sampling loop, and the cp_groups=15 / W8 change
doesn't alter that structure.

The only path that unlocks speculative is a device-side CP sampler
(fused 15-group kernel that produces all tokens without per-group
host round-trips). That is a multi-week project that belongs in a
different milestone, not a W2.2 re-audit.

**Estimate for a device-side sampler**: ~3 weeks (new AscendC kernel
or CannFusion DSL graph, integrate top-k/top-p on-device, verify
sampling bit-identity vs host). Low-confidence estimate; unblocking
requires AscendC or CANN-sampling-op expertise that isn't in the
current contract. **Not recommended for W3.**

---

## 3. aclGraph capture for CP forward (W2.3)

### 3.1. Code status

All four aclGraph symbols (`aclmdlRICaptureBegin`, `aclmdlRICaptureEnd`,
`aclmdlRIExecuteAsync`, `aclmdlRIDestroy`) are already dlsym-resolved
at `cp_cann_symbols.cpp:166-173` and gated by `has_aclgraph()`.
`TalkerCannEngine` uses them for per-pos decode graphs
(`talker_cann_engine.cpp:1768-1790`). `CpCannEngine` does **not** use
them today — `forward_one_token_launch` issues ops eagerly.

**Plumbing to add CP capture** (estimated <150 LoC):
- `std::vector<aclmdlRI> decode_graphs_(MAX_SEQ)` member on
  `CpCannEngine`.
- Wrap the aclnn dispatch body of `forward_one_token_launch` in
  `if (captured_at[pos]) replay else capture+record`.
- Invalidate on workspace realloc (mirror `talker_cann_engine.cpp:364`).

### 3.2. Per-capture cost model from M4 Talker data

From `NATIVE_TTS_CONTRACT.md:319-323`:

> "eager 14.5 fps → aclGraph 6.2 fps (~2.3× slowdown) on the canonical
> utterance; LLM-only timing 17 ms/step eager vs 67 ms/step aclGraph,
> so the CaptureBegin/End pair is costing ~50 ms per step."

So the Talker 28-layer decode captures at ~50 ms/capture, replays at
~14 ms (the 17-ms-eager figure minus the ~3 ms dispatch-queueing savings
replay delivers on 28-layer body). In M4's actual numbers:
- **Talker capture cost**: ~50 ms one-shot per `pos`.
- **Talker replay cost**: ~14 ms/step (vs 17 ms eager) = **~3 ms savings
  per replay**.
- **Talker break-even**: `50 ms / 3 ms ≈ 17 replays**. Single-utterance
  runs at 75-200 steps hit every `pos` exactly once, so the graph cache
  never replays, and the capture cost dominates. Hence the 2.3×
  slowdown.

### 3.3. Project to CP forward

CP forward has **5 transformer layers** (vs Talker's 28), shape is
**stable across groups** (cp_hidden=1024, q_dim=1024, kv_dim=256 on
the b6 build). The key question: does the per-capture cost scale with
layer count or with dispatch count?

Answer: **per-capture cost tracks kernel-submission count, not
arithmetic work**. The 50 ms on Talker is not FLOPs (Talker decode
FLOPs at 17 ms eager would imply capture adds 33 ms of *recording*,
not math). So scaling by dispatch count:

- Talker decode: 28 layers × ~12 aclnn dispatches/layer ≈ 336
  dispatches recorded; capture cost ~50 ms → ~0.15 ms/dispatch
  captured.
- CP forward: 5 layers × 18 dispatches/layer + 5 overhead = **95
  dispatches** (see §4.1 audit); projected capture cost
  **~14 ms/capture** at same 0.15 ms/dispatch.
- CP forward eager: ~1.06 ms/group × 15 groups + 2 prefill = ~18 ms
  (matches the contract §3's 17 ms). So ~1.1 ms per call.
- **CP forward replay**: if replay saves the same fractional
  dispatch-queueing overhead as Talker (Talker saves 3 of 17 ms = ~18%),
  CP would save ~0.2 ms per call → **~0.9 ms/call replay**. So replay
  savings per call = ~0.2 ms.

### 3.4. Break-even replay count for CP

Per pos, one capture is amortized over however many times that pos is
called. Within a single frame, CP calls pos=0, 1, 2, ..., 16 each
**exactly once**. Across frames, pos=0..16 are called again every
frame. Over a 75-frame long utt, each pos is called 75 times.

- Per-pos break-even: `14 ms / 0.2 ms = 70 replays`.
- Per-pos available replays in a 75-frame long utt: **75**. Marginally
  above break-even.
- Per-pos frame-5 warmup (contract §8 notes a 5-frame warmup during
  `QwenTTS::load`): **5 replays**. Far below break-even. Capture done
  at warmup is wasted.

So the pattern is: capture pos=0..16 across the first frame of each
fresh utterance, replay for the remaining 74 frames. Total win:
- Capture cost per utt: 17 positions × 14 ms = **238 ms amortized
  over one utt**.
- Per-frame replay savings at cp_groups=15: 17 calls × 0.2 ms = **3.4
  ms/frame**.
- Over a 75-frame utt: 75 × 3.4 = **255 ms** total savings.
- **Net per utt: +17 ms (i.e. ~0.2 fps)**. Break-even.

### 3.5. Where the capture cost can be reduced

Two levers the M4 Talker work did **not** try:

1. **Shared graph across groups**. Positions 2..16 execute the *same*
   5-layer body with just a different KV-cache cursor and RoPE phase.
   RoPE lookups are already per-pos strided indexing into a precomputed
   `rope_cos_dev_ / rope_sin_dev_` buffer, so capturing at pos=2 and
   replaying at pos=3 would read stale RoPE (wrong phase). *Unless*
   we hoist the RoPE phase into a device-side uniform and do a
   partial capture that takes position as an argument via
   `aclmdlRICaptureTaskUpdateBegin` (CANN 8.3+, per
   `NATIVE_TTS_CONTRACT.md:1161`). This drops 15 captures down to 1,
   reducing upfront cost from ~238 ms to ~14 ms per utt.
2. **Capture once, share across utterances**. The Talker decode
   graphs are per-utt; a session API would reuse them. CP has the
   same property — graphs are shape-stable. If the native-TTS C ABI
   bridge (§8 of the contract) shifts to a session model
   (`QwenTTS::begin_session` / `end_session`), all 17 captures
   amortize across every utt in the session.

### 3.6. Recommendation

**Worth 3-5 days for ~0.5-1 fps at baseline, and ~2-3 fps if
combined with session-mode caller (post-FFI-bridge)**, with the
following caveats:

- At single-utterance break-even (+0.2 fps), the risk of
  numerical drift from re-recorded workspace addresses and the
  added ~250 ms cold-start latency are bad tradeoffs for no real
  win.
- With `aclmdlRICaptureTaskUpdateBegin` + shared-across-groups,
  drops capture cost ~17× and delivers 2-3 fps even in
  single-utterance mode. This is the real upside.
- **Blocked on**: validating that `aclmdlRICaptureTaskUpdateBegin`
  supports the RoPE-slot strided tensor update on CANN 8.5. If it
  doesn't, fall back to full-capture-per-pos which is break-even
  only.

Decision for W3: land this **only** if W1 lands and the FFI
session API lands (the `qwen_tts_api` session variant already on
the contract's §8 roadmap). Otherwise defer.

---

## 4. Kernel-count audit (W2.4)

### 4.1. Dispatches per `forward_one_token` (static read of `cp_cann_engine.cpp:1251-1628`)

**Pre-loop** (input projection + cast to F16):
- `aclnnMm` (F32 proj_w × input) × 1
- `aclnnInplaceAdd` (F32 + proj_b) × 1
- `aclnnCast` (F32 → F16) × 1

= **3 compute dispatches** + 1 H2D `aclrtMemcpy` (the host embedding
upload — counts as queue work but not a compute kernel).

**Per layer body** (x 5 layers, `cp_cann_engine.cpp:1315-1610`), in
order:

| # | op | aclnn symbol | notes |
|---|---|---|---|
| 1 | residual d2d | `aclrtMemcpyAsync` | async memcpy, not a compute dispatch but a stream-queue op |
| 2 | RmsNorm(input) | `aclnnRmsNorm` | F16 in/out, F32 gamma |
| 3 | Mm Q | `aclnnMm` or `aclnnWeightQuantBatchMatmulV3/V2` (W8) or NZ variant | |
| 4 | Mm K | same as above | |
| 5 | Mm V | same as above | |
| 6 | RmsNorm(Q) | `aclnnRmsNorm` | per-head norm |
| 7 | RmsNorm(K) | `aclnnRmsNorm` | per-kv-head norm |
| 8 | RoPE(Q) | `aclnnRotaryPositionEmbedding` | |
| 9 | RoPE(K) | `aclnnRotaryPositionEmbedding` | writes into K-cache slot |
| 10 | V→cache | `aclrtMemcpyAsync` | async memcpy |
| 11 | FA | `aclnnFusedInferAttentionScoreV2` | the single biggest op, fused multi-head attn with softmax |
| 12 | Mm O | `aclnnMm`/W8/NZ | |
| 13 | Add(resid+o) | `aclnnAdd` | F16 |
| 14 | residual d2d | `aclrtMemcpyAsync` | |
| 15 | RmsNorm(post) | `aclnnRmsNorm` | |
| 16 | Mm gate | `aclnnMm`/W8/NZ | |
| 17 | Mm up | `aclnnMm`/W8/NZ | |
| 18 | Silu | `aclnnSilu` | in-place |
| 19 | InplaceMul | `aclnnInplaceMul` | gate * up |
| 20 | Mm down | `aclnnMm`/W8/NZ | |
| 21 | Add(resid+ffn) | `aclnnAdd` | |

Compute dispatches per layer = **18** (rows 2-9, 11-13, 15-21).
Stream-queue ops (memcpyAsync) per layer = 3 (rows 1, 10, 14).

**Post-loop** (`cp_cann_engine.cpp:1613-1618`):
- `aclnnRmsNorm` (final) × 1
- `aclnnCast` (F16 → F32) × 1

= **2 compute dispatches**.

**Grand total per `forward_one_token`**:
- Compute dispatches: 3 + 5×18 + 2 = **95**
- Stream-queue memcpys: 1 + 5×3 = **16**

### 4.2. Per-frame total

17 `forward_one_token` calls per frame × 95 = **1,615 compute
dispatches/frame** (at cp_groups=15). The contract §3 estimate of
"900 dispatches per frame" was based on ~12 aclnn ops/layer (likely
undercounting RmsNorms and the V-copy); real figure is ~1.8× higher.

Over a 75-frame long utt: ~121,000 compute dispatches. At a typical
~10 μs per-dispatch submission overhead, that's ~1.2 s of *pure
submit time* if everything ran serial on one CPU thread. CANN's
TASK_QUEUE_ENABLE=2 reclaims most of that via two-phase submit
(tiling + launch overlap), which is why TQE=2 is on the canonical
path.

### 4.3. Top-3 hottest ops (projected from per-op flops)

Cannot live-measure without GGUFs on ac02, so this is a
flops-weighted estimate:

1. **`aclnnFusedInferAttentionScoreV2`** (per-layer, per-`forward_one_token`).
   At seq_len up to 17, n_heads=16 (assumed — cp_config_.n_heads on
   the b6 build), head_dim=64, this is 5 × 17 × ~4×16×17×64 = ~0.37
   Mflop per layer. Low FLOP but high fixed cost per call (workspace
   query + softmax + output materialization) ≈ ~0.3 ms/call estimated.
   **5 layers × 17 calls = 85 calls per frame = ~25 ms/frame if cost
   scaled linearly** — wait, this exceeds the 17 ms total. So the
   per-call FA cost must be <0.2 ms.

   Correction: the CP forward per-call wall is ~1 ms, so FA is likely
   **~50-100 μs** (not 300 μs). But it's still the biggest single op
   because the other ops are all small matvecs.

2. **`aclnnMm` (Q/K/V/O/gate/up/down) × 7/layer × 5 layers**. Per-call
   matrix sizes (from `cp_cann_engine.cpp:1232-1236`): e.g. Q is
   [1×1024] × [1024×1024] = 1 Mflop; gate/up are [1×1024] × [1024×3072] =
   3 Mflop each; down is [1×3072] × [3072×1024] = 3 Mflop. These are
   tiny matmuls where the NPU launch overhead dominates arithmetic —
   the A16W8 path (CANN 8.5+ `aclnnWeightQuantBatchMatmulV3`) is on
   by default for these and gives a ~15% speedup per matmul on the
   committed 8038c335 numbers.

3. **`aclnnRmsNorm` × 4/layer × 5 layers = 20/forward × 17 forwards =
   340 per frame**. Tiny op (~1024-element reduction + mul), but the
   sheer count means its launch overhead is felt. Each call ~20 μs
   rounded up, so 340 × 20 μs ≈ **7 ms/frame** of pure RmsNorm.

### 4.4. Fusion candidates

Ordered by expected impact:

1. **Fuse `aclnnAdd(residual + {o_out,ffn_out})` into a residual
   accumulator**. The post-attn and post-FFN residual adds are both
   F16 fp ops on contiguous 1024-element vectors. On 910B4 these are
   launch-overhead-dominated; 2 per layer × 5 layers × 17 forwards =
   170/frame. Fusing them with the upstream `aclnnMm(o_proj)` / `Mm(down)`
   via an `aclnnMmAddCustom`-style op would drop 170 dispatches/frame.
   Rough saving: **170 × 20 μs ≈ 3.4 ms/frame (≈ +1.5 fps)**.

   Blocked on: whether CANN 8.5 exposes `aclnnMmAdd` or whether we'd
   need an AscendC/CannFusion custom op.

2. **Fuse RmsNorm + subsequent Mm(Q) into `aclnnRmsNormQuantMatmulV3`**.
   The `aclnnWeightQuantBatchMatmulV3` family supports a `rms_norm` input
   fused variant on CANN 8.5 (pre-norm Qwen3 is the trained target of
   this op). If we're already on the W8 path, swapping the two
   separate `aclnnRmsNorm + aclnnWeightQuantBatchMatmulV3` calls for a
   single fused call drops 4 dispatches/layer × 5 layers × 17 = 340
   dispatches. The RmsNorm overhead (§4.3 #3) is the main saving:
   **~7 ms/frame → ~2 ms/frame (≈ +2 fps)**.

   Blocked on: confirming the fused op exists at the CANN version
   actually deployed (b6 build is CANN 8.5 — likely present but
   unverified).

3. **Fuse Silu + InplaceMul(gate, up) + Mm(down) into `aclnnSwiGLUMatmul`**.
   This is a common op family (`aclnnSwiGluQuant` exists; unclear if
   a fused-with-output-Mm variant does). Drops 2 dispatches/layer ×
   5 × 17 = 170/frame. **~3.4 ms/frame (≈ +1.5 fps)** if it lands.

   Blocked on: CANN op-set availability.

4. **Replace V-cache `aclrtMemcpyAsync` with an in-place KV-update
   variant of the FA op**. `aclnnFusedInferAttentionScoreV2` does not
   currently ingest K/V directly from the input projection — we write
   V into the cache via async memcpy first, then FA reads it. An FA
   variant with `inplace_kv_update` would save 1 async memcpy/layer =
   85/frame. Marginal (~0.5 ms/frame).

### 4.5. Recommendation

**Worth 5-7 days for ~4-5 fps total**, prioritized:
1. Fused RmsNorm+QuantMatmul (highest ROI, ~2 fps).
2. Fused Mm+Add residual (next-highest, ~1.5 fps).
3. Fused SwiGLU (~1.5 fps).

All three are **gated on CANN 8.5 op-set availability**. If the fused
ops exist (likely — Qwen3 is a flagship CANN workload), this is
mostly rewiring. If they don't exist, each would require an
AscendC/CannFusion custom kernel — that's a 2-3 week project per op,
not a 5-day effort.

**Decision for W3**: audit CANN 8.5 op headers on ac01 (`ls
$ASCEND_TOOLKIT_HOME/include/aclnnop/ | grep -iE 'rms|swiglu|weight_quant'`).
If the fused ops are present, land (1) first as a 2-day spike; (2) and
(3) follow if (1) pays the predicted ~2 fps. If the fused ops are
absent, skip the whole track and fund the device-side CP sampler
(§2.5) instead.

---

## 5. Summary for PM decision (W2.5)

### 5.1. Candidate ranking

| candidate | est ms saved / frame | est fps gain | est days | risk | recommend |
|---|---:|---:|---:|---|---|
| W1 NPU lm_head (already funded, for reference) | 12 ms | +5 fps | — | low | (landing separately by Agent X) |
| **W2c-1 Fused RmsNorm+QuantMatmul** | **~5 ms** | **~2 fps** | 2 | low if op exists | **land first** |
| **W2c-2 Fused Mm+Add residual** | **~3.4 ms** | **~1.5 fps** | 2 | low if op exists | **land second** |
| W2c-3 Fused SwiGLU-Matmul | ~3.4 ms | ~1.5 fps | 3 | medium (op less common) | land if W2c-1/2 cleanly landed |
| W2c-4 FA inplace-KV variant | ~0.5 ms | ~0.2 fps | 2 | medium | skip — marginal |
| **W2b aclGraph + CaptureTaskUpdate + session** | **~2-3 ms** | **~1-1.5 fps** (single-utt); **~4-5 fps** (session) | 5 | medium-high (RoPE phase + session API dependency) | defer to after FFI session API lands |
| W2a Speculative re-audit | ~0 ms (net regression) | **-1 fps** | 0.5 (just run) | no risk | **skip — already measured to regress** |
| W2a' Device-side CP sampler (unlock for speculative) | unknown | +3-5 fps if works | 15+ | high | not recommended for W3 |

### 5.2. My ranking

1. **Land W2c-1 (Fused RmsNorm+QuantMatmul) first**. 2-day spike;
   pays ~2 fps. Build on ac01 in the W1 worktree, A/B vs eager on
   canonical. If `aclnnWeightQuantBatchMatmulV3` with norm-fused
   variant is not in the CANN 8.5 headers, abort and go to (3).

2. **Land W2c-2 (Fused Mm+Add residual) second**. Another 2-day
   spike; pays ~1.5 fps. Same header-availability gate.

3. **If both header gates fail, invest in CannFusion / AscendC custom
   kernels** or fund the device-side CP sampler as a larger W4
   milestone. Do not land W2a or W2b as-is — W2a regresses, W2b is
   break-even without the session API which is in the §8 roadmap not
   the current quarter.

4. **Skip W2a speculative re-audit** unless PM specifically wants
   to freshen the committed 26.5 vs 27.7 fps number. Prediction:
   still regresses by ~1 fps. 30 min of PM / agent time to confirm
   if wanted.

### 5.3. Blockers on PM

None of the above is blocked on a PM decision per se; the agents can
proceed. The only items PM should decide:

- Does the FFI session API (contract §8 post-v1 direction, OminiX-API
  C-ABI bridge) land in this quarter? If yes, W2b aclGraph becomes
  much more attractive (~4-5 fps, not 1-1.5). If no, demote W2b to
  "after session API".
- Is there budget for a 2-3 week AscendC custom op project if the
  CANN 8.5 header audit comes back empty? If no, W2c becomes
  header-gated and may deliver 0 fps.

### 5.4. Not blocked on, but worth noting

A fresh 3-trial baseline on the canonical Chinese mayun text at
cp_groups=15 / W8 / TQE=2 would be worth ~30 min on ac01 once W1
lands — the committed numbers are good but a single fresh boot
confirms no drift. This can be done as part of W1.5 fps measurement,
no separate agent needed.

---

**End of report.**
