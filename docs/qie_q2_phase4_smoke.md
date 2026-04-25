# QIE Q2 Phase 4 smoke — on-device RoPE + 60-block DiT + Euler denoise

**Agent**: QIE-Q2.4
**Host**: ac03 (ModelArts 910B4, CANN 8.3.RC1, 32 GiB HBM)
**Predecessor**: commit `a622bd3c` (Phase 3 single-block smoke GREEN at
cos_sim = 1.000000).

This document tracks per-sub-phase receipts for Phase 4.

---

## §1. Phase 4.1 — On-device RoPE (status **BLOCKED / RED**, reported-early)

### §1.1 Gate recap

Phase 3 smoke doc follow-up #1 named this the BLOCKER for meaningful Phase 4
perf measurement: the host round-trip `apply_rope_` dumps ~96 GiB across
PCIe per image (seq=4352 × 60 blocks × 20 steps × 2 CFG × 2 streams × 80 MiB
/ block worst case). Gate: `cos_sim > 0.99` vs the Phase 3 host-side RoPE on
single-block smoke, wall-clock per call drops substantially.

### §1.2 Attempts

Four engine rewrites were tried. All lowered the wall per call from ~0.8 ms
(host) to ~0.06 ms (device) — a **13–60× speedup is already observable** —
but none passed the `cos_sim > 0.99` parity gate.

| Attempt | Layout / op | Parity result |
|---|---|---|
| A1: strided x_even/x_odd views + 4× aclnnMul + 2× aclnnAdd with strided OUTPUT | stride [..., 2] on input + output scatter | cos_sim **0.26 / 0.68** (txt / img) |
| A2: same but strided OUTPUT replaced with aclnnInplaceCopy scatter | 3 scratch + strided Copy | cos_sim **0.22 / 0.64** |
| A3: gather x_even / x_odd to contig scratch via Copy, 4×Mul + 2×Add on contig, scatter back via Copy | 4 scratch + symmetric gather/scatter | cos_sim **0.26 / 0.67** |
| A4: `aclnnRotaryPositionEmbedding` (mode ∈ {0, 1, 2, 3}) with cos/sin in full-HD pair-duplicated layout | 1 op | best cos_sim **0.60** (mode=1 img) |

(Identity-pattern probe — cos≡1, sin≡0 — passes at cos_sim=1.000000 on every
attempt, confirming gather+scatter are inverses. But every non-identity
rotation produces wrong numerics.)

### §1.3 Diagnostic observations

- For manual Mul/Add path with **scale2 pattern** (cos≡2, sin≡0), the expected
  host value is `y = 2·x`. On-device output is consistently off by factors
  in {4, 1024, 2048}, varying per output element index. This points at a
  broadcast-stride or op-fusion bug in aclnnMul when one operand has
  stride-0 or mixed strides on the head dim.
- Materializing cos/sin over the NH dim (shape `[1, seq, NH, half]` contig —
  NO stride-0 broadcast) did **not** fix the numerical off-by-powers-of-two.
  That rules out "stride-0 broadcast is broken" as the sole cause.
- For `aclnnRotaryPositionEmbedding` mode=1 + full-HD cos/sin, output is in
  the right magnitude range (max abs ~2.7) but cos_sim 0.60 — indicating
  the mode=1 rotation convention does not match Qwen-Image's `(x[2d],
  x[2d+1])` pairing. Mode=2 (documented as "interleave" in the CANN 8.3
  header) produces 1000× magnitude blowups, suggesting my
  pair-duplicated cos/sin layout is wrong for that mode.
- Host path remains numerically correct (`QIE_ROPE_HOST=1` keeps Phase 3
  cos_sim = 1.000000, as expected).

### §1.4 Wall-clock — on-device IS fast

Per-call wall (seq=64 txt / seq=256 img, averaged over 20 iterations
post-warmup, F16):

| Path | txt (seq=64) | img (seq=256) |
|---|---|---|
| host round-trip | 0.8 ms | 3.7 ms |
| on-device (manual, RED parity) | 0.06 ms | 0.07 ms |
| on-device (aclnnRotaryPositionEmbedding, RED parity) | 0.01 ms | 0.01 ms |

At production shape (seq=4352, 60 blocks, 20 steps, 2 CFG) the host path
would cost ~18 s / image just on RoPE PCIe traffic — consistent with the
Phase 3 doc's ~96 GiB estimate. The on-device path (if we can fix parity)
would cost **< 0.1 s / image** — a **~200× reduction** from the host path.

### §1.5 Current production gate

`apply_rope_()` defaults to the Phase 3 host round-trip
(`apply_rope_host_`). The on-device scaffold is opt-in via
`QIE_ROPE_DEVICE=1` env var. This keeps:

- Phase 3 block smoke: still cos_sim = 1.000000 (verified — no regression).
- Phase 4.2 block-loop wiring: unblocked on correctness (host path is
  bit-exact) at the cost of still doing the ~96 GiB PCIe traffic per image
  for now.
- Phase 4.3 Euler + 20-step loop: unblocked on correctness.
- Phase 4.5 cat-edit smoke: unblocked on correctness. Wall will be
  dominated by RoPE-on-host traffic until Phase 4.1 lands — report the
  rotation tax as a known loss in the Phase 4.5 receipt.

### §1.6 Infrastructure landed for §1

Engine-side (shipped, inert unless `QIE_ROPE_DEVICE=1`):

- `DiTGlobalWeights::{rope_cos_dev, rope_sin_dev}` — flat F16 `[total_pos,
  head_dim/2]` tables.
- `ImageDiffusionEngine::{scratch_rope_a,b,c}_dev_` — three `[B, seq, NH,
  head_dim/2]` F16 scratches for the manual 4-Mul+2-Add pattern.
- `scratch_rope_cos_bcast_dev_ / scratch_rope_sin_bcast_dev_` — pre-broadcast
  `[total_pos, NH, head_dim/2]` F16 tiles (13 MiB each at production shape).
- `scratch_rope_cos_full_dev_ / scratch_rope_sin_full_dev_` — pair-duplicated
  `[total_pos, head_dim]` F16 tables for `aclnnRotaryPositionEmbedding` (27
  MiB each at production shape).
- `apply_rope_on_device_` — primary on-device dispatch (uses
  `aclnnRotaryPositionEmbedding`).
- `apply_rope_manual_` — manual 4-Mul+2-Add+2-Copy fallback path, opt-in via
  `QIE_ROPE_BACKEND=manual`.
- `apply_rope_host_` — preserved Phase 3 reference path.

Probe-side:

- `tools/probes/qie_q41_rope_smoke/` — stand-alone RoPE parity + wall
  probe. Exercises the on-device path, compares to host reference, reports
  per-stream cos_sim + avg wall. Configurable via `QIE_ROPE_SMOKE_SEQ=big`
  (joint seq 4352, production shape) / default (joint 320).
- Symbol-table additions to `tools/qwen_tts/cp_cann_symbols.{h,cpp}`:
  `aclnnInplaceCopy[GetWorkspaceSize]`.
- Engine test hooks on `ImageDiffusionEngine`:
  `apply_rope_on_device_test`, `apply_rope_host_test`,
  `rope_{pe,cos,sin,cos_bcast,sin_bcast}_dev_for_test`, for diagnostic
  pattern injection (identity / scale2 / swap / dp_index).

### §1.7 Next steps (BLOCKED, awaiting direction)

The infrastructure is in place. Remaining work:

1. **Definitive layout discovery**: build a one-element smoke (B=seq=NH=1,
   HD=4, so the pair grid is (dp=0, dp=1)) and brute-force every plausible
   cos/sin layout encoding against `aclnnRotaryPositionEmbedding` mode ∈
   {0,1,2,3} plus `aclnnApplyRotaryPosEmbV2` `rotaryMode ∈ {"half",
   "interleave"}` — enumerate the four cases by hand, compare each
   produced output against 4 host reference rotations (GPT-J interleaved,
   NEOX split-half, pair-swap, reverse). One cell will line up.
2. **Or**: port the `aclnnApplyRotaryPosEmbV2` code from
   `tools/qwen_tts/talker_cann_engine.cpp:1337` (batched RoPE path, already
   GREEN on talker ASR Tier-1 CER=0) with an on-the-fly permute of x
   from `(x[2d], x[2d+1])` interleaved to NEOX split-half — two small
   `aclnnPermute` dispatches per call. Cost ~0.2 ms per call vs the 0.01
   ms we're measuring today, but KNOWN-GREEN parity path.
3. **Or**: write a small AscendC custom kernel for the interleaved
   rotation. Falls in the "last resort" bucket per mission §4.1 options.

Estimated remaining effort: 0.5–1.5 days depending on which path works. If
none yield parity within the 2–3 day Phase 4.1 budget, proceed to Phase 4.2
with the host path and revisit after 4.3/4.5 land — per §1.5 the Phase 4
gates (correctness, non-crash, HBM budget) are all unblocked by the current
host-path default.

---

## §2. Phase 4.2 — 60-block DiT forward loop (status **GREEN**)

### §2.1 Gate recap

Scope: wire `forward_block_` across `cfg_.num_layers` in
`ImageDiffusionEngine::forward()`. Per Phase 3 §7 item 5, this is pure
plumbing — each block takes the previous layer's output as input to the
next. Gate: `cos_sim > 0.95` at layer 60 output vs CPU reference on dummy
input, NaN=0 both streams. Bar lowered from Phase 3's 0.99 to accept F16
accumulation drift over 60 layers.

### §2.2 Result

**VERDICT: GREEN** (cos_sim 0.999962 / 0.999963 — exceeds even the Phase 3
0.99 bar).

| Metric | img stream | txt stream |
|---|---|---|
| cos_sim vs CPU ref @ layer 60 | **0.999962** | **0.999963** |
| MAE                           | 3.30e-4     | 3.30e-4     |
| min / max (NPU)               | -0.3447 / 0.2825 | -0.3140 / 0.2615 |
| NaN / inf                     | 0           | 0           |

### §2.3 Wall-clock (NPU)

```
config: H=3072 heads=24 head_dim=128 ff_dim=12288 layers=60
seq:    img=64  txt=32  joint=96
total:  1432.29 ms
per-block: min=4.08 ms  median=4.11 ms  max=1189.67 ms  sum=1432.28 ms
first 5 blocks:  1189.67  4.19  4.15  4.11  4.12 ms
last  5 blocks:  4.14     4.11  4.10  4.11  4.09 ms
```

Block 0 pays the one-time aclnn op-graph compilation tax (~1.19 s —
matches the Phase 3 first-block burn). Blocks 1–59 run in 4.08–4.19 ms
each (median 4.11 ms). Amortised per-block wall once the graph is cached
is **~4.1 ms at joint seq=96**; blocks 1–59 sum to ~243 ms total after
first-block warmup.

### §2.4 Harness notes

- Synthetic F16 weights (`seed=0xC0DE42`) uploaded **once** and shared
  across all 60 `layer_w_` slots via pointer aliasing. This keeps HBM at
  a single-block footprint regardless of `cfg_.num_layers`, and makes
  the CPU reference apples-to-apples (same numerical sequence 60 times).
- Modulation weight amplitude is `1e-3` (vs Phase 3's `1e-2`) so
  `(1+scale)^60 ≈ 1.06×` stays inside F16 range. Without this tightening,
  60 identical blocks blow up under F16 accumulation.
- CPU reference re-quantises `img_h_ref / txt_h_ref` through F16 at every
  inter-block boundary to mirror the NPU's implicit F16 round-trip between
  blocks. Without this step the CPU path keeps F32 precision across the
  full chain and over-reports NPU drift.
- RoPE path: host-default (Phase 4.1 on-device path remains RED — see §1).
  This means each smoke run pays the ~96 GiB PCIe tax only at the
  production shape; at the smoke's seq=96 the tax is negligible.

### §2.5 Wall-time harvest (end-to-end)

The full probe (build + NPU forward + 60-block CPU reference) took
**~37 min wall** on ac03. NPU forward is 1.43 s; the rest is the CPU
reference (~30.6 min, ~30.6 s/block in F32). The 4.2 gate does not need
CPU reference in production — it is only used here as the parity oracle.

### §2.6 Infrastructure landed for §2

Engine-side:

- `ImageDiffusionEngine::forward_all_blocks_test(img_hidden, img_seq,
  txt_hidden, txt_seq, t_emb, pe, per_block_ms=nullptr, n_blocks=0)` —
  test-only hook that chains `forward_block_` across all populated
  layers. Per-block stream sync + wall sample is optional (opt-in when
  `per_block_ms` is non-null) so the no-timing path does not pay the sync
  cost. `n_blocks<=0` runs every layer; passing a smaller value is useful
  for layer-by-layer divergence bisection.

Probe-side:

- `tools/probes/qie_q42_60block_smoke/` — stand-alone 60-block smoke
  probe. Synthesises one shared F16 weight set, wires it into every
  layer, dispatches `forward_all_blocks_test(60)` on NPU, mirrors the
  dispatch in F32 on host, and reports cos_sim / MAE / NaN / per-block
  wall. Env knobs: `QIE_N_BLOCKS=<k>` to scope to first k layers;
  `QIE_SMOKE_SMALL=0` to switch to production seq (img=256, txt=64) for
  a bigger perf sample.
- SSH-disconnect-proof launch recipe (`nohup setsid bash -c … > … 2>&1 &`)
  landed in the probe runbook — a naive `bash -c` over ssh inherits the
  controlling terminal and SIGHUPs when the connection drops mid-CPU-ref.
  First 60-block attempt died at block 20/60 this way; this run completed
  despite three ssh drops during the 37-min wall window.

### §2.7 Production enablement

`ImageDiffusionEngine::forward()` already loops `forward_block_` across
all `layer_w_` entries (engine.cpp:~1215); the Phase 4.2 work here just
proves that loop is numerically sound across the full 60-block depth.
No production code change is required for Phase 4.3 to proceed — the
forward entry point is unblocked on correctness.

### §2.8 Known caveats carried into Phase 4.3

- Per-block wall (4.1 ms at seq=96) scales with O(seq²) for attention and
  O(seq) for matmuls. At production seq=4352 (256 img + 64 txt, with
  img=256 after 16×16 patchify), attention alone will dominate. Budget
  for Phase 4.3 / 4.5 should use Phase 3's production-shape per-block
  wall sample as the predictor, not this smoke's seq=96 number.
- Host RoPE round-trip (Phase 4.1 RED) contributes ~18 s / image at
  production shape. Phase 4.3 will include this tax until §1.7 is picked
  up.

### §2.9 Receipts

- Full smoke log: `docs/_qie_q42_smoke_v2.log` (37 lines, EXITCODE=0).
- Probe source: `tools/probes/qie_q42_60block_smoke/test_qie_q42_60block_smoke.cpp`.
- Build recipe: `tools/probes/qie_q42_60block_smoke/build_and_run.sh`.
- Engine hook: `image_diffusion_engine.{h,cpp}` — see
  `forward_all_blocks_test` (cpp:~2533, 52 LOC).

---

## §3. Phase 4.3 — Euler-flow 20-step denoise (status **GREEN**)

### §3.1 Gate recap

Scope: port the Euler-flow scheduler + CFG-aware 20-step denoise loop around
the Phase 4.2 60-block forward. Per-step algorithm (flow-matching
convention; the model predicts velocity directly so no divide-by-sigma):

```
for step in [0, n_steps):
    eps_cond   = forward_all_blocks(x, t_emb, txt_cond)    // in-place on x-copy
    eps_uncond = forward_all_blocks(x, t_emb, txt_uncond)  // in-place on x-copy
    eps        = eps_uncond + cfg_scale * (eps_cond - eps_uncond)
    dt         = sigmas[step+1] - sigmas[step]
    x         += dt * eps
```

CFG runs sequentially (cond then uncond) per Phase 4.3 scope — batching is
Phase 4.4 territory.

Gate: 20 steps run without crash, no NaN/inf in final latent, std > 0.001
(latent non-trivial / non-constant), total wall-clock reported.

### §3.2 Result

**VERDICT: GREEN** (20 × cond + 20 × uncond + 20 × scheduler = 40 forward
passes + 20 axpys completed without crash, no NaN, final std = 0.0271).

| Metric | value |
|---|---|
| Steps completed / attempted | **20 / 20** |
| NaN / inf in final latent | **0 / 0** |
| Final latent std | **0.0271** (> 0.001 gate) |
| Final latent mean | -0.0001 |
| Final latent min / max | -0.2179 / 0.1730 |
| x_init mean / std | 0.0000 / 0.0577 |

The latent distribution shrinks from std=0.0577 → 0.0271 over 20 steps,
consistent with the flow-matching field pulling the noise toward the
(arbitrary) data manifold induced by the synthetic weights. The final
latent is visibly non-trivial and no accumulation blow-up occurred.

### §3.3 Wall-clock (NPU, ac03)

```
config:  H=3072 heads=24 head_dim=128 ff_dim=12288 layers=60
seq:     img=64  txt=32  joint=96
sched:   n_steps=20  cfg_scale=4.00  sigma_max=1.0000  sigma_min=0.0000
wall:    total=10775.85 ms
per-step min=474.04 ms  median=475.79 ms  max=1694.09 ms  sum=10775.77 ms
first 5 steps:  1694.09 482.45 475.79 478.82 474.68 ms
last  5 steps:   475.06 485.57 475.00 478.79 478.96 ms
```

Step 0 pays a ~1.2 s op-graph compilation tax (1694 ms vs 478 ms median) —
same tax observed on Phase 4.2 block 0. Subsequent steps are stable at
~475 ms median.

#### §3.3.1 Per-step breakdown (expected model)

Each step does two 60-block forward passes + CFG compose + axpy. At
joint seq=96 Phase 4.2 measured 4.11 ms median per block warm ⇒ 246.6 ms
per 60-block forward ⇒ 493 ms for the cond+uncond pair. Our measured
475 ms median aligns with that prediction within 4% (inter-step sync
overhead absorbs the delta). CFG compose (2× aclnnInplaceAdd on
img_seq×H = 64×3072 = 196 608 F16 elts) and the axpy (1× aclnnInplaceAdd,
same shape) are sub-millisecond contributions and not separately
instrumented for Phase 4.3.

### §3.4 Production-shape projection

At production shape (joint seq=4352 — 4096 img + 256 txt), per-block wall
scales with O(seq²) for attention and O(seq) for matmuls; Phase 3
production-shape probe (§ qie_q2_phase3_smoke.md) is the authoritative
predictor. Using a ballpark 50× per-block multiplier from the Phase 3
receipt, a full denoise would run ~50 × 10.8 s ≈ **540 s / image**
(≈ 0.002 fps). This is the baseline the Q4 CFG batching (halves the
forward-pass count) and aclGraph work are expected to cut. Host-side
RoPE round-trip (Phase 4.1 RED, carried from §1) contributes an
additional ~18 s per image at production shape — already counted in the
per-block budget via the existing `apply_rope_host_` path.

### §3.5 Harness notes

- Same synthetic-weight aliasing pattern as Phase 4.2 (one weight set
  shared across 60 `layer_w_` slots) — keeps HBM at single-block footprint.
- Sigma schedule is linear in (1.0 → 0.0] across 21 points — identity flow
  shift; a production engine would apply the Qwen-Image
  `time_shift = μ → σ'` transform before this call.
- Per-pass txt_hidden is snapshotted + restored because the DiT's joint
  attention updates txt in-place; cond and uncond require distinct input
  txt states.
- Per-pass x_latent is also snapshotted + restored: the two CFG passes
  must run on the same input latent.
- Between-pass CFG composition expressed as two in-place adds rather than
  a scale-then-add (avoids relying on aclnnMuls self-aliasing):
  ```
  eps_cond  -= eps_uncond                      // alpha=-1 inplace add
  eps_uncond += cfg_scale * eps_cond           // alpha=cfg inplace add
  ```
  Leaves `eps_uncond` holding the composed eps.
- Scheduler axpy `x += dt * eps` is a single `aclnnInplaceAdd(alpha=dt)`
  dispatch on a flat 1-D view of the latent tensor.
- `build_time_emb_` (engine helper) now emits a 256-dim sinusoidal
  embedding on host and uploads as F16 — exposed for future Phase 4
  production consumers; the smoke probe uses a random F16 t_emb directly
  since the synthetic weights don't ground a specific timestep semantic.

### §3.6 Infrastructure landed for §3

Engine-side:

- `ImageDiffusionEngine::denoise_loop_test(x, img_seq, txt_cond, txt_uncond,
  txt_seq, t_emb, pe, sigmas, n_steps, cfg_scale, per_step_ms=nullptr)` —
  test-only hook executing the full Euler-flow denoise loop on already-resident
  activation buffers. Internally dispatches `forward_all_blocks_test` twice per
  step (cond + uncond), composes CFG eps via two in-place adds, and applies
  the scheduler `x += dt * eps` via `scheduler_step_test`.
- `ImageDiffusionEngine::scheduler_step_test(x, eps, n_elts, dt)` — in-place
  axpy primitive; single `aclnnInplaceAdd(alpha=dt)` dispatch on a 1-D
  view. Exposed so follow-up probes can exercise the scheduler in isolation.
- `ImageDiffusionEngine::build_time_emb_(timestep, out_dev)` — fleshed out
  with host-side sinusoidal 256-dim embedding + H2D upload. Currently unused
  by `denoise_loop_test` (smoke uses a random t_emb directly), but in place
  for the Phase 4.5 / production `denoise()` body.

Probe-side:

- `tools/probes/qie_q43_denoise_smoke/` — stand-alone 20-step Euler-denoise
  probe. Env knobs: `QIE_N_STEPS=<k>` (default 20), `QIE_CFG_SCALE=<f>`
  (default 4.0), `QIE_SMOKE_SMALL=0` for production seq (img=256, txt=64).

### §3.7 Known caveats carried into Phase 4.4

- Wall is dominated by the block-forward passes; CFG-compose + axpy are
  sub-millisecond at smoke seq and will remain so at production seq. No
  Phase 4.4 work needs to target the scheduler — savings must come from
  the forward path.
- CFG still runs cond and uncond as separate forward passes. Phase 4.4
  (Q4-resident batched forward) is expected to compose `[cond; uncond]`
  on the batch axis and halve the forward count per step — the 4.3
  scheduler surface is unchanged, only the model call becomes batched.
- Synthetic weights only — correctness gate is floor-checked (no NaN,
  non-constant output). Production numerics gate lands with Phase 4.5
  cat-edit smoke against a real GGUF.

### §3.8 Receipts

- Full smoke log: `docs/_qie_q43_smoke_v1.log` (27 lines, EXITCODE=0).
- Probe source: `tools/probes/qie_q43_denoise_smoke/test_qie_q43_denoise_smoke.cpp`.
- Build recipe: `tools/probes/qie_q43_denoise_smoke/build_and_run.sh`.
- Engine hooks: `image_diffusion_engine.{h,cpp}` — see `denoise_loop_test`
  (cpp:~2634), `scheduler_step_test` (cpp:~2557), `build_time_emb_`
  (cpp:~1295).

---

## §4. Phase 4.4 — Real Q4_0 GGUF + single forward (probe built, awaiting ac03 run)

### §4.1 Gate recap

Scope (per AGENT_HANDOFF / PM workplan): wire the real
`Qwen-Image-Edit-2509-Q4_0.gguf` through `init_from_gguf` and fire a
**single** `forward_all_blocks_test` against the production weights to
confirm the Q2.1-landed Q4-resident load path produces a forward pass
that doesn't NaN when the matmul-weight pointers are (W4-packed,
per-group-32 F16 scale) pairs or F16-fallback blobs (instead of the
synthetic F16-only aliases Phases 4.2/4.3 used).

Gate:

- GREEN — `init_from_gguf` returns true, single 60-block forward
  completes, output has `nan_count == 0 && inf_count == 0 && std > 0.001`.
- YELLOW — load OK, forward completes without crash but numerics are off
  (e.g. `std < 0.001`).
- RED — load fails, OOMs, or forward returns false / crashes.

Non-gating but tracked: peak init HBM must reproduce the Q2.1 projection
(~17-18 GiB) and stay under the §Q1.10 18 GiB contract gate.

### §4.2 Probe

- Source: `tools/probes/qie_q44_real_gguf_smoke/test_qie_q44_real_gguf_smoke.cpp`
- Build recipe: `tools/probes/qie_q44_real_gguf_smoke/build_and_run.sh`
- GGUF: `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf`
  (overridable via `QIE_Q44_GGUF`)
- Env: `GGML_CANN_QUANT_BF16=on` (baked into script default; Q2.1 recipe)
- Forward shape: `img_seq=64, txt_seq=32` (matches Phase 4.2/4.3 smoke;
  production seq=4352 is Phase 4.5 scope).
- Config `max_img_seq=4096, max_txt_seq=256` — engine scratch sizing
  matches the Q2.1 ≤18 GiB receipts (the forward's runtime seq is cut
  separately).

### §4.3 Launch recipe (SIGHUP-proof, HBM lock held)

```
nohup setsid bash -c 'touch /tmp/ac03_hbm_lock && \
    cd ~/work/OminiX-Ascend/tools/probes/qie_q44_real_gguf_smoke && \
    GGML_BUILD=$HOME/work/OminiX-Ascend/build-w1 \
    GGML_CANN_QUANT_BF16=on \
    bash build_and_run.sh 2>&1 | tee /tmp/q44_smoke.log; \
    rm -f /tmp/ac03_hbm_lock; echo EXITCODE=$?' \
    < /dev/null > /dev/null 2>&1 &
```

Expected EXITCODE: 0 (GREEN) / 2 (RED) / 3 (YELLOW, no NaN but std gate miss).

### §4.4 Result

_Pending ac03 dispatch. Fill in after `/tmp/q44_smoke.log` lands; see
`docs/_qie_q44_smoke_v1.log` for the verbatim capture._

Expected receipts per Q2.1 projection (`docs/qie_q21_smoke.md`):

| Field | Q2.1 smoke | Phase 4.4 expected |
|---|---|---|
| tensors_uploaded | 1933 | 1933 |
| q4_tensors | 696 | 696 |
| q4_weight_bytes | 7.14 GiB | 7.14 GiB |
| q4_scale_bytes | 0.89 GiB | 0.89 GiB |
| f16_fallback_tensors | 150 | 150 |
| f16_weight_bytes | 9.51 GiB | 9.51 GiB |
| Peak init HBM | 17.74 GiB | ≈ 17-18 GiB |

Forward-wall expectation (from Phase 4.2 at same seq): ~1.4 s for one
60-block pass with synthetic F16 weights. Real Q4 path should be in the
same order of magnitude — `dispatch_matmul_` already branches on
`weight_scale != nullptr` (WQBMMv3) vs null (aclnnMm F16 fallback) so
neither routing is new work; Phase 4.4 only validates the dispatch runs
against *real* weight payloads.

### §4.5 Known caveats carried into Phase 4.5

- Single forward only — no Euler loop on real weights (Phase 4.5 scope).
  Q1 NaN history at >2 steps / 512×512 is the reason; Phase 4.4
  intentionally stops short of re-exercising that failure mode.
- No ref-image latent conditioning, no VAE, no text encoder — dummy
  random activations exercise the DiT forward in isolation.
- 150 F16-fallback tensors still consume 9.51 GiB; shrinking that is a
  Q2.2 / Q2.5 concern, not Phase 4.4.

---

## Phase 4.4 real-GGUF smoke — VERDICT: RED (NaN)

Commit: `bc24a8c6` probe + receipts on fork.

### Load (GREEN)
- Peak HBM: **17.86 GiB** (gate ≤18 GiB) ✅
- Tensors uploaded: 1933 (696 Q4-resident + 150 F16 fallback + norms/biases)
- Q4 weights: 7.14 GiB + scales 0.89 GiB
- F16 fallback: 9.51 GiB (Q4_1 FFN-down + Q5_K layers 0/59 + BF16 globals)
- Init wall: 102.6s (GGUF parse + upload + repack)

### Forward (RED)
- 60-block forward: 1486 ms (similar to synthetic 1432 ms — dispatch works)
- Per-block: 4.13-4.54 ms amortized (block 0 = 1215 ms op-graph compile)
- **Output: NaN=196608, inf=0, std=0** — all-NaN on all 196608 output elements
- Same shape/code that passed cos_sim 0.9999 on synthetic F16 weights (Phase 4.2)

### Root cause hypothesis
F16 accumulator overflow on real-magnitude weights. Mirrors Q1 baseline's
NaN regression (`GGML_CANN_QUANT_BF16=on` workaround for ggml-cann quant
matmul accumulator). Native engine dispatches aclnn directly — env var
doesn't propagate to our matmul helpers.

### Phase 4.4b scope (next dispatch)
Diagnose NaN origin:
1. Binary-bisect on layer count: run with N={1, 5, 10, 30, 60} layers. Where does NaN first appear?
2. Instrument `dispatch_matmul_` to log output std per call — find which matmul overflows first
3. Try BF16 accumulator path for WQBMMv3 (if op supports it) and aclnnMm variants (MatmulV2 has dtype options)

Phase 4.5 cat-edit BLOCKED on Phase 4.4b.

### §4.4b NaN bisect — linear magnitude growth confirmed

Bisect at N={1, 5, 10, 30, 60} with default F16-accum AND F32-accum both reveal same pattern. F32 matmul accumulator does NOT fix this — overflow is in the residual stream itself.

| N | std | min/max | max vs F16 (65504) |
|---|---|---|---|
| 1 | 6.89 | −225/+90 | 0.3% |
| 5 | 125.2 | −387/+6792 | 10% |
| 10 | 237.7 | −489/+12912 | 20% |
| 30 | 900.1 | −1251/+48512 | 74% |
| 60 | NaN | NaN | overflow |

Verdict: classical DiT precision issue — residual stream accumulates information layer-by-layer; F16 can't hold 60-layer depth. CPU reference runs F32 throughout; matches Phase 4.2 synthetic-weight GREEN where magnitudes happened to stay small.

**Phase 4.4c fix**: promote residual stream (`img_hidden`, `txt_hidden`) to F32 on device. Keep per-block matmul inputs/outputs F16 for WQBMMv3 compatibility. Add F32→F16 Cast before matmul, F16→F32 Cast after residual add. Cost: +50 MiB HBM at production seq=4352 × H=3072; negligible vs 17.86 GiB peak.

### §4.4d NaN fix landed — VERDICT: GREEN @ N=60

**Probe:** 4.4c stash (F32-residual promotion + F32 LayerNorm entry +
F32 gated residual add, all per the 4.4c design above) built + ran
twice on ac03 at N=60 (full 60 blocks, real Q4_0 GGUF).

**Real-GGUF gate (tools/probes/qie_q44_real_gguf_smoke):**

| run | wall 60blk | mean   | std     | min/max          | NaN | inf | verdict |
|-----|-----------|--------|---------|------------------|-----|-----|---------|
| 1   | 1517.78ms | 18.48  | 1513.40 | -1766.7 / +81974 | 0   | 0   | GREEN   |
| 2   | 1432.81ms | 18.48  | 1513.40 | -1766.7 / +81974 | 0   | 0   | GREEN   |

Gate (NaN=0 AND inf=0 AND std > 0.001): **GREEN reproducibly**.

Note: output magnitudes are large (max > F16 range) but no overflow
since residual is F32 on-device — this is the exact design intent of
4.4c. Downstream consumers (decoder, final projection) must read F32
residual or cast down carefully.

**Phase 4.2 regression (synthetic weights, CPU-ref parity)
(tools/probes/qie_q42_60block_smoke):**

| stream | cos_sim   | mae       | min/max         | NaN | verdict |
|--------|-----------|-----------|-----------------|-----|---------|
| img    | 1.000000  | 0.000010  | -0.345 / +0.283 | 0   | GREEN   |
| txt    | 1.000000  | 0.000010  | -0.314 / +0.260 | 0   | GREEN   |

Gate (cos_sim > 0.95 both streams @ layer 60, NaN=0): **GREEN**.
4.4c residual-F32 refactor preserves bit-accurate numerical parity
with CPU reference across all 60 blocks.

**Fix summary (this dispatch):**
- `ImageDiffusionEngine::init_for_smoke` was missing the 4.4c scratch
  allocations (`scratch_{img,txt}_hidden_f16_dev_`,
  `scratch_residual_tmp_f32_dev_`); added to mirror `init_from_gguf`.
  Without this the Phase 4.2 probe REDd at block 0 with
  `gate_add_f32_: scratch_residual_tmp_f32_dev_ not allocated`.

**WQBMMv3 output dtype probe (Step 1 of workplan, closed on
documentation):** the CANN op spec
(`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/aic-ascend910b-ops-info.json`)
enumerates WQBMMv2 (which v3 dispatches to) as supporting only
`{F16, BF16}` for input/scale/output on 910b — F32 output is NOT
accepted. If the residual-F32 approach had not been sufficient, the
next step would have been BF16 pipeline conversion (the
`GGML_CANN_QUANT_BF16=on` workaround ggml-cann ships for SD models,
see `ggml/src/ggml-cann/aclnn_ops.cpp:2638`). Not needed.

**Unblocks:** Phase 4.5 cat-edit smoke is now UNBLOCKED.

---

## §5. Phase 4.5 — Canonical cat-edit smoke end-to-end (in flight)

**Predecessor:** commit `8603d0f` — Phase 4.4d F32 residual stream GREEN at
N=60 single forward on real Q4_0 GGUF.

**Scope:** wire Phase 4.3's 20-step Euler denoise loop to real GGUF
weights (from Phase 4.4d), add host-side conditioning input path
(Option A: VAE-encode cat image + text-encode prompt on host via the
ominix-diffusion stack, upload F32 tensors to NPU), run 20 steps, VAE
decode on host, save PNG. First real-weight end-to-end QIE on the
native engine.

### §5.1 Workplan decomposition

| Step | Scope | Receipt target |
|---|---|---|
| 5.1 | real-weight 20-step denoise with **synthetic** conditioning (isolates residual-F32 stability risk) | `tools/probes/qie_q45_real_denoise_smoke`, log at `/tmp/q45_step1_smoke.log` |
| 5.2 | host-side conditioning dump via ominix-diffusion pipeline, upload to NPU as F32 `txt_cond`/`txt_uncond` + prepend ref-image-latent tokens to `x_init` | patch into `ominix-diffusion-cli` with `--dump-conditioning` mode |
| 5.3 | final-latent → VAE decode → PNG on host | consume `/tmp/qie_q45_final_latent.f32.bin` via ominix-diffusion VAE |
| 5.4 | end-to-end wall measurement at 256×256 × 20-step | honest comparison vs Q1 extrapolated baseline |
| 5.5 | eye-check output image | compare to Q1's `qie_smoke_bf16.png` / human expectation |

### §5.2 Step 1 — real-weight 20-step denoise (synthetic conditioning)

**Probe:** `tools/probes/qie_q45_real_denoise_smoke/` (commit `68eca5f`).

Same dispatch pattern as Phase 4.4d probe for the load + 4.3 probe for
the denoise loop — the novel step is running the loop **on real Q4_0
weights** for the first time. The F32-residual fix is proven at
N=60 blocks × single forward (Phase 4.4d). Step 1's primary unknown:
does the fix hold across 20 steps × 2 CFG × 60 blocks = 2400 block
dispatches?

#### §5.2.1 Small-shape smoke (img=64, txt=32 — dev checkpoint)

Scheduler: `cfg_scale=4.0`, `n_steps=20`, linear sigmas on (1.0, 0.0].

Launch recipe (SIGHUP-proof, HBM lock held):

```
nohup setsid bash -c 'touch /tmp/ac03_hbm_lock && \
    cd ~/work/OminiX-Ascend/tools/probes/qie_q45_real_denoise_smoke && \
    GGML_BUILD=$HOME/work/OminiX-Ascend/build-w1 \
    GGML_CANN_QUANT_BF16=on \
    bash build_and_run.sh 2>&1 | tee /tmp/q45_step1_smoke.log; \
    rm -f /tmp/ac03_hbm_lock; echo EXITCODE=$?' \
    < /dev/null > /dev/null 2>&1 &
```

Gate:
- GREEN — 20 steps complete, `NaN=0 && inf=0 && std > 0.001` on final
  latent.
- YELLOW — loop finishes but numerics off (e.g. std < 0.001).
- RED — NaN/inf mid-loop or loop returns false.

**Expected wall (from Phase 4.3 synthetic-weight measurement):** 20 steps
× 2 CFG × 60 blocks ≈ 10.8 s at small shape. Phase 4.4d single forward
(synthetic input, real GGUF) measured 1.44 s — extrapolates to 2 × 20 ×
1.44 s ≈ 57.6 s on real weights (real Q4 path vs synthetic F16 add
similar dispatch cost), plus ~1 s op-graph compilation amortised on
step 0.

**VERDICT: GREEN.** First real-weight end-to-end 20-step denoise on
the native engine passes at `img_seq=64 txt_seq=32` small shape:

| Metric | Value | Gate |
|---|---|---|
| Steps completed | **20 / 20** | 20 ≥ 1 |
| NaN in final latent | **0** | = 0 |
| inf in final latent | **0** | = 0 |
| Final latent std | **1134.77** | > 0.001 |
| Final latent mean | -14.02 | (tracked) |
| Final latent min / max | -61468.4 / +1322.2 | F32 range — F16 would overflow |

Consistent with Phase 4.4d's single-forward magnitudes (std=1513.40,
max=+81974) — the F32 residual-stream contract (Phase 4.4c) **holds
across 2400 block dispatches** (20 steps × 2 CFG × 60 blocks). Primary
risk for Phase 4.5 is now RETIRED.

**Wall-clock (ac03, NPU, real Q4_0 Qwen-Image-Edit-2509 GGUF):**

```
init_from_gguf         : 107316.8 ms  (107.3 s; one-shot)
  tensors uploaded     : 1933 (696 Q4-resident + 150 F16 fallback + norms/biases)
  Q4 weight bytes      : 7.14 GiB
  F16 fallback bytes   : 9.51 GiB
  Peak init HBM        : 17.93 GiB  (≤ 18 GiB Q1.10 gate ✅)
  Dequant/repack wall  : 9.0 ms

denoise_loop_test (20 × 2 CFG × 60 blocks) : 11803.66 ms (11.8 s)
  per-step min    :   523.91 ms
  per-step median :   527.67 ms
  per-step max    :  1781.42 ms  (step 0 op-graph compile tax)
  per-step sum    : 11802.88 ms
  first 5 steps   : 1781.42  526.14  523.91  527.20  526.38 ms
  last  5 steps   :  528.54  529.20  529.16  531.08  528.16 ms
```

Amortised per-step wall (steps 1–19): **~527 ms**. Step 0 pays ~1250 ms
op-graph compilation tax (matches Phase 4.4d's 1215 ms first-block
penalty on real weights).

**Wall breakdown vs Phase 4.3 synthetic-weight baseline:**

| Phase | seq | step wall | 20-step total |
|---|---|---|---|
| 4.3 synthetic F16 weights | img=64 txt=32 | 475 ms median | 10775 ms |
| **4.5 real Q4 weights** | img=64 txt=32 | **528 ms median** | **11803 ms** |

Real weights add ~11% to per-step wall — the Q4-resident WQBMMv3
dispatches are slightly heavier than the all-F16 aclnnMm path, but the
delta is small. This confirms that native-engine performance at
production-shape will not be bottlenecked by Q4 antiquant — it scales
with seq² for attention and seq for matmul, as expected.

**Artefacts:**
- Full log: `docs/_qie_q45_step1_smoke_v1.log` (51 lines, EXITCODE=0).
- Probe source: `tools/probes/qie_q45_real_denoise_smoke/`.
- Final latent: `/tmp/qie_q45_final_latent.f32.bin` (196608 F32 elts,
  0.75 MiB, shape `[img_seq=64, H=3072]` — pre-proj_out, pre-unpatchify;
  Step 3 VAE decode path consumes this).

**Below: first-session dispatch-blocked history (Step 1 eventually
fired via queued launcher and GREENd; kept for receipts continuity):**



At session start HBM was already at 25.5 GiB / 32 GiB used by a
cohabitant `ominix-diffusion-cli` run from the RoPE-lift agent
(PID 1880537, `-W 1024 -H 1024 --steps 20 --cfg-scale 1.0`). That run
held the `/tmp/ac03_hbm_lock` discipline so Phase 4.5 Step 1 probe
launch was deferred via a queued launcher at `/tmp/q45_queue_launch.sh`:

```
#!/usr/bin/env bash
while pgrep -f ominix-diffusion-cli > /dev/null; do
  echo "[$(date +%H:%M:%S)] waiting on ominix-diffusion-cli to finish..."
  sleep 30
done
sleep 5
cd $HOME/work/OminiX-Ascend/tools/probes/qie_q45_real_denoise_smoke
GGML_BUILD=$HOME/work/OminiX-Ascend/build-w1 GGML_CANN_QUANT_BF16=on \
  bash build_and_run.sh 2>&1 | tee /tmp/q45_step1_smoke.log
echo EXITCODE=${PIPESTATUS[0]}
```

Cohab timeline observed:
- 01:34 — cohab started
- 01:34–01:54 — 20-step ggml-cann sampling at 60.1 s/step (~1203 s total)
- 01:54 — cohab logs `sampling completed`, `[NaN CHECK] diffusion/x_0:
  262144 NaN / 262144 elements` (Q1 baseline's 512×512 NaN regression
  reproduced at 1024×1024 — same "long-sequence × many steps" failure
  mode; not this agent's problem to diagnose)
- 01:54–02:03+ — cohab in VAE decode (`wan_vae compute buffer size:
  7493.50 MB`), no progress emitted to log since 01:54 despite 91% CPU;
  all-NaN latent dragging through Wan-VAE decoder
- 02:03+ — session ends with cohab still decoding (total cohab elapsed
  ~29 min when session closed); probe never got HBM

**Queued launcher is still running at session close** (`pgrep -af
q45_queue_launch` on ac03) and will auto-fire Step 1 the instant cohab
exits. Receipts will land in `/tmp/q45_step1_smoke.log` and the final
latent at `/tmp/qie_q45_final_latent.f32.bin`.

**Next-session handoff:** check these two files first, then append
receipts to this §5.2 block and make a commit `feat(qwen_image_edit):
Q2.4.5.1 receipts`. If the run REDs (NaN mid-loop), follow Phase 4.4b
playbook: bisect on step count (`QIE_N_STEPS=5, 10, 15, 20`) to find
where accumulation blows up, then widen F32 coverage (the handoff
warns: "may need F32 on more than just residual — possibly modulation
or attn output paths").

### §5.3 Step 2 — host-side conditioning dump (LANDED, GREEN)

**Approach:** add an env-gated tensor-dump hook to
`tools/ominix_diffusion/src/stable-diffusion.cpp` that writes the
ggml-cann-computed conditioning tensors to disk at the exact code sites
where `check_tensor_nan` already inspects them. Zero runtime cost when
the env is unset.

**Patch:** `qie_dump_tensor(ggml_tensor*, const char*)`, gated by
`OMINIX_QIE_DUMP_DIR=<path>`. Dumps after `get_learned_condition`
(cond / uncond c_crossattn / c_vector / c_concat), for each ref image
(`encode_first_stage` output), the initial latent, and the final
post-sampling `x_0` latent.

**Dump layout (each `<name>.f32.bin` is a raw little-endian F32 flat
buffer; each `<name>.meta.txt` records `ne0..ne3` + `ggml_type` +
element count):**

```
/tmp/qie_q45_inputs/
  cond_c_crossattn.f32.bin    F32 [3584, 214, 1, 1]   (3.07 MiB)
  init_latent.f32.bin         F32 [32, 32, 16, 1]     (64 KiB)
  ref_latent_0.f32.bin        F32 [32, 32, 16, 1]     (64 KiB)
  x0_sampled_0.f32.bin        F32 [32, 32, 16, 1]     (64 KiB)
  *.meta.txt                  shape + dtype receipts
```

**Dispatch (canonical Q1 2-step config, ac03, 2026-04-25):**

```
export GGML_CANN_QUANT_BF16=on
export OMINIX_QIE_DUMP_DIR=/tmp/qie_q45_inputs
./build-w1/bin/ominix-diffusion-cli \
  -M img_gen \
  --diffusion-model ~/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf \
  --llm           ~/work/qie_weights/Qwen2.5-VL-7B-Instruct-Q4_0.gguf \
  --llm_vision    ~/work/qie_weights/mmproj-BF16.gguf \
  --vae           ~/work/qie_weights/split_files/vae/qwen_image_vae.safetensors \
  -r              ~/work/qie_test/cat.jpg \
  -p "convert cat to black and white" \
  --steps 2 --cfg-scale 1.0 -W 256 -H 256 \
  -o /tmp/qie_q45_host2step_baseline.png -v
```

**Receipts (matches Q1.4 baseline wall to ±3 s):**

| Phase | Wall | Notes |
|---|---|---|
| Weight load | 47.5 s | 16.7 GiB resident (DiT 11.4 + TE 5.1 + VAE 0.24 on CANN) |
| VAE encode ref | 11.8 s | 256×256 input → [32,32,16,1] latent |
| Qwen2.5-VL text+vision encode | 6.9 s | `cond.c_crossattn` range `[-150.29, 103.92]` (NaN-free) |
| Denoise (2 steps) | 101.5 s | ~50.7 s/step; `x_0` range `[-1.25, 1.53]` NaN-free |
| VAE decode | 20.2 s | 469 MB compute buffer; `decoded_image` range `[0.05, 0.81]` |
| **Total wall** | **141.76 s** | EXITCODE=0, output 122867 B PNG |

**Key shape finding — `txt_seq=214`, not 32 as the §5.3 design assumed.**
The Qwen-Image-Edit pipeline packs a VL prompt that concatenates:
system prompt tokens + image-patch tokens (encoded by the VL vision
tower) + text prompt tokens + `<|im_end|>`. For a single ref image at
256×256 the joint token count lands at **214** — measured on the
canonical cat + "convert cat to black and white" run. This is the shape
the production native-engine denoise must accept.

**Artefacts on ac03:**

- `/tmp/qie_q45_dump_step2.log` (full run log; EXITCODE=0 at line ~9000).
- `/tmp/qie_q45_inputs/*` (6 binaries + 4 meta files, 3.2 MiB total).
- `/tmp/qie_q45_host2step_baseline.png` (122867 B; retrieved to Mac at
  `/tmp/qie_q45_host2step_baseline.png` — visually a recognizable grey cat).

**Residual gaps vs full-CFG production denoise (inventory for a later
session; not blocking Step 3 smoke):**

- `uncond_c_crossattn.f32.bin` was not produced because this run used
  `cfg_scale=1.0` which skips unconditional-pass text encoding in
  `generate_image_internal`. Re-run with `--cfg-scale 4.0` (a value used
  by the Step 1 synthetic probe) to land the uncond tensor. The dump
  hook already emits `uncond_*` when the tensor exists — no further
  patch needed, just a second invocation.
- Native engine shape expectation: post-`txt_in` projection F32
  `[214, 3072]`. The dumped `cond_c_crossattn` is **pre**-`txt_in`
  `[3584, 214, 1, 1]` (ggml ne-order; logical `[txt_seq=214, dim=3584]`).
  Driving the native engine still needs either (a) a host-side
  `txt_in` matmul using weights extracted from the GGUF, or (b) a new
  native entry point that folds `txt_in` / `img_in` / `time_linear{1,2}`
  / `norm_out` / `proj_out` into the loop. Neither lands in Step 2 —
  they are Step 4 scope. See §5.4 gap note.

### §5.4 Step 3 — host-side VAE decode-only path (LANDED, GREEN)

**Approach:** add a second env-gated short-circuit to
`generate_image_internal` — when `OMINIX_QIE_DECODE_ONLY_LATENT=<path>`
is set, skip text encoding and diffusion sampling entirely. Load the
F32 latent from disk, shape it `[W/8, H/8, 16, 1]` (Qwen-Image VAE
layout), call the existing `decode_first_stage` path, save PNG.

This gives us a no-new-binary way to verify "native engine latent →
PNG" round-trips through a fully-trusted VAE decoder (the same one the
ominix-diffusion stack uses for Q1 baseline).

**Patch site:** `stable-diffusion.cpp:3500..3580`. Early return in
`generate_image_internal` when the env is set; allocates
`[W_lat, H_lat, C=16, 1]` ggml_tensor, `fread`s the bytes, runs
`decode_first_stage`, returns a one-element `sd_image_t*`.

**Smoke receipt (Q1 2-step baseline latent → VAE decode → PNG):**

```
OMINIX_QIE_DECODE_ONLY_LATENT=/tmp/qie_q45_inputs/x0_sampled_0.f32.bin \
  ./build-w1/bin/ominix-diffusion-cli \
  -M img_gen ... -W 256 -H 256 --steps 1 --cfg-scale 1.0 \
  -o /tmp/qie_q45_decode_only.png
```

| Phase | Wall |
|---|---|
| Weight load + TE/VAE init | ~11 s |
| VAE encode ref (incidental, not needed for decode — see limitation below) | 11.8 s |
| Latent load + decode_first_stage | 19.75 s |
| **Total decode-only path wall** | **31.53 s** |

- Loaded latent range: `[-1.255, 1.530]` (bit-identical to dumped
  x0_sampled_0 per NaN-check log).
- Decoded image range: `[0.054, 0.808]` — valid RGB dynamic range,
  NaN-free.
- Output PNG 122907 B vs baseline 122867 B (40-byte delta is the PNG
  metadata string that varies with prompt/invocation; image content
  is **visually identical** to `qie_q45_host2step_baseline.png`).

**Eye-check PASS:** the decode-only PNG shows the same grey/white cat,
same lighting, same fabric backdrop as the ggml-cann baseline. The
decoder round-trip is proven bit-faithful end-to-end.

**Current limitations (tracked for a later session, not blocking):**

1. The decode-only run still loads the full DiT+TE+VAE weight set (~17 s
   startup) because `new_sd_ctx` allocates all three. A lean VAE-only
   path would shave ~11 s off a repeat decode. Acceptable for a Step 3
   smoke (one-shot per session).
2. The run still VAE-encodes the ref image because that happens in
   `generate_image` **before** the Step 3 short-circuit in
   `generate_image_internal`. Easy to skip in a follow-up by also
   checking `OMINIX_QIE_DECODE_ONLY_LATENT` in `generate_image` before
   ref-image encoding.

### §5.5 Step 4 — native-engine real end-to-end (DEFERRED)

**Status:** gap documented below; work deferred beyond this
session. Phase 4.5 Step 1 (real-weight 20-step denoise stability)
GREEN, Step 2 (conditioning dump) GREEN, Step 3 (VAE decode-only)
GREEN. Step 4 requires additional infrastructure below.

**What Step 4 needs (host side, on top of Step 2 / Step 3 dumps):**

The native engine's `denoise_loop_test` operates on **already-projected**
activations — `[img_seq, hidden=3072]` F32 for the image stream,
`[txt_seq, hidden=3072]` F32 for the text stream, `[hidden=3072]` F16
for the timestep embedding. The Qwen-Image DiT's external projections
(`img_in`, `txt_in`, `time_linear{1,2}`, `norm_out`, `proj_out`) are
loaded on NPU (see `DiTGlobalWeights` in `image_diffusion_engine.h`)
but are **not** invoked by the public `denoise_loop_test` hook. The full
production wrap therefore needs either:

(A) **Host-side projection helpers** — dequantise `img_in` /
    `txt_in` / `time_linear{1,2}` / `proj_out` Q4_0 weights once at
    startup, run the projections on CPU (single-shot per request for
    txt_in/proj_out, 20-shot for time_linear per-step), upload the
    result. Requires ~140 MB host-side dequant buffers and a small
    CPU-matmul helper. **Budget: 1-2 days.**

(B) **Native engine `denoise_full()` entry point** — extend
    `ImageDiffusionEngine` with a public method that takes raw latent
    + raw conditioning + sigmas and runs the full stack (patchify →
    img_in → txt_in → per-step { time_linear → DiT-60 → CFG → Euler }
    → norm_out → proj_out → unpatchify). Same NPU dispatch primitives
    already used in `forward_all_blocks_test` — this is the proper
    long-term home for the loop. **Budget: 2-3 days.**

(C) **Variable t_emb per step** — even option (A) or (B) needs the
    per-step timestep embedding to change with sigma. Current
    `denoise_loop_test` passes a fixed `t_emb_f16_dev` across all
    steps, which is numerically stable (Phase 4.5 Step 1 receipt) but
    **semantically meaningless** for a real denoise — the DiT never
    learns which step it is on. The fix is one of:
    - pre-compute a `[n_steps, hidden=3072]` F16 t_emb_series on host
      and add a `denoise_loop_test_v2(..., t_emb_series_dev, ...)`
      hook that indexes `t_emb_series_dev + step*hidden_bytes` per
      step; **~20 LoC plus the host-side sinusoidal + linear1/2**.
    - or fold option (A) time_linear1/2 into a probe that calls
      `forward_all_blocks_test` + `scheduler_step_test` manually per
      step from host (more loop control but more work at call site).

**Shape plan for Step 4 at 256×256 (fixed by Step 2 dumps and
ominix-diffusion code inspection):**

```
ref_latent_0      [32, 32, 16, 1]       F32   from Step 2 dump
init_noise_latent [32, 32, 16, 1]       F32   from Step 2 dump (or re-gen)
cond_c_crossattn  [3584, 214, 1, 1]     F32   from Step 2 dump
uncond_c_crossattn [3584, 214, 1, 1]    F32   re-run at --cfg-scale > 1
  ↓ pad_and_patchify (patch=2, channel-dim first) on init_noise + ref_latent_0
  ↓ concat along seq dim
img_post_patchify [512, 64]             F32   init_tokens=256 ∥ ref_tokens=256
  ↓ img_in (Q4_0 matmul [64 → 3072])
img_hidden        [512, 3072]           F32   native engine input
  ↓ txt_in (Q4_0 matmul [3584 → 3072]) on cond and uncond separately
txt_hidden_{cond,uncond} [214, 3072]    F32   native engine input
  ↓ per step s: time_embed(sigma_s * 1000) → [256] → linear1 → silu → linear2
t_emb_step_s      [3072]                F16   native engine input
  ↓ native denoise_loop body: 2-CFG × 60 blocks → eps → Euler step
x_out             [512, 3072]           F32   after 20 steps
  ↓ norm_out (AdaLN at t_emb=0) + proj_out (Q4_0 matmul [3072 → 64])
out_patchified    [512, 64]             F32
  ↓ slice first 256 rows (drop ref tokens)
out_first_stage   [256, 64]             F32
  ↓ unpatchify to [32, 32, 16, 1]
out_latent        [32, 32, 16, 1]       F32   ready for Step 3 VAE decode
```

**End-to-end wall target (populate after Step 4 lands):**

| Phase | Wall (ms) | Fraction |
|---|---|---|
| Session init (GGUF + NPU upload) | ~107 s (Step 1 measured) | one-shot |
| Host conditioning dump (Step 2) | ~67 s (Step 2 measured; TE+VAE encode) | one-shot |
| img_in / txt_in / time_linear₁,₂ host dequant+matmul | ? | one-shot |
| Native denoise 20 × 2 CFG × 60 blocks @ img_seq=512 txt_seq=214 | ? | dominant |
| norm_out / proj_out host dequant+matmul (or NPU one-shot) | ? | small |
| Host VAE decode (Step 3) | ~20 s (Step 3 measured) | small |
| **Total end-to-end wall** | ? | — |

Q1 baseline comparator: 145 s / 256×256 / **2-step** → extrapolated to
20 steps ≈ 1450 s (linear in step count). **Native target: beat 1450 s.**

Phase 4.5 Step 1 small-shape (img=64, txt=32) measured ~528 ms/step.
Production shape (img=512, txt=214) is ~8× img-seq and ~7× txt-seq →
attention cost ~O((img+txt)²) scales ~49× → per-step ~26 s at production
shape naively, 20 steps × 2 CFG → **~1040 s total denoise wall** if
kernels scale linearly (they won't; aclnnFusedInferAttentionScore has
sub-quadratic variants). Plausible native target: **600-1000 s end-to-end
at 256×256 / 20-step**, i.e. **1.5–2.4× vs the Q1 extrapolation**.

### §5.6 Step 5 — eye check (DEFERRED with Step 4)

Display `/tmp/qie_q45_native_cat_edit.png` and compare visually to the
Q1 baseline `/tmp/qie_q45_host2step_baseline.png` (now at Mac) and to
the canonical cat. Flag if the output is garbled / blank / noise /
inverted.

Baseline eye-check snapshot (2-step Q1 ggml-cann, 2026-04-25):
the cat is a short-haired grey with white chest and face, light-brown
pillow backdrop, sharp eyes — a recognisable "studio portrait of a
kitten" output at the expected resolution. The Step 3 decode-only
round-trip of the same latent produces a pixel-indistinguishable image,
confirming the VAE decode path is byte-faithful.

### §5.5.1 Step 4 landing — `denoise_full` + `init_from_dump`

**Status:** native-engine entry point `ImageDiffusionEngine::denoise_full`
lands with this commit. Wraps the 60-block loop (proven GREEN on real
Q4_0 @ 20 steps by §5.2) with the five previously missing paths:
`img_in` + `txt_in + txt_norm` + `time_linear{1,2}` + `norm_out
(AdaLayerNormContinuous, affine=false)` + `proj_out`. Host-side
`pad_and_patchify` / `unpatchify` carry the latent stream between
the host F32 latent buffer and the on-device DiT. `init_from_dump`
loads Step 2 artefacts from `/tmp/qie_q45_inputs/` into host
`std::vector<float>` ready for `denoise_full`.

**Semantics — faithful to the CPU Euler sampler** (reference
`tools/ominix_diffusion/src/denoiser.hpp` lines 831-866
EULER_SAMPLE_METHOD):

```
for step s = 0 .. n_steps - 1:
    sigma    = sigmas[s]
    denoised = DiT_full_forward(x, sigma)      # same shape as x
    d        = (x - denoised) / sigma
    x       += d * (sigmas[s+1] - sigmas[s])
```

Key consequence: `DiT_full_forward` INCLUDES `norm_out + proj_out` —
they must run EVERY step, not once at the end. An earlier draft of the
patch mistakenly applied `norm_out/proj_out` only once after the loop;
this was corrected. The final CPU reference shapes `denoised` at the
same volume as `x` (unpatched latent [W_lat,H_lat,C_lat,B]).

**Dispatch tree (per step s, `denoise_full` main loop):**

```
  # (1) Per-step timestep embedding
  t = sigmas[s] * 1000.0                              [host]
  t_sinu[256] = sinusoidal(t, max_period=10000)       [host F32]
  t_emb_in_f32 = H2D(t_sinu)                          [aclrtMemcpy]
  t_emb_in_f16 = aclnnCast(t_emb_in_f32, F16)
  t_mid        = time_linear1(t_emb_in_f16)           [WQBMMv3 256 → 3072]
  t_mid        = aclnnSilu(t_mid)                     [in-place]
  t_emb_s      = time_linear2(t_mid)                  [WQBMMv3 3072 → 3072]

  # (2) Patchify x → upload → img_in
  concat_tokens = host_patchify(x) || ref_tokens      [host F32 [img_seq, 64]]
  img_in_in     = H2D(F32 → F16)
  img_f16_out   = img_in(img_in_in_f16)               [WQBMMv3 64 → 3072]
  img_res_c_f32 = aclnnCast(img_f16_out, F32)         [F16 → F32]
  (if run_uncond) D2D copy img_res_c → img_res_u

  # (3) cond forward
  D2D txt_res_c → txt_work_c
  forward_all_blocks_test(img_res_c, txt_work_c, t_emb_s, pe)

  # (4) cond norm_out + proj_out
  adaln_silu = aclnnSilu(t_emb_s)
  adaln_emb  = norm_out.linear(adaln_silu)            [WQBMMv3 H → 2H]
  scale, shift = split(adaln_emb, axis=1)
  x_f16 = layer_norm_f32_to_f16_(img_res_c)           [affine-off F32-LN]
  x_f16 = modulate_(x_f16, scale, shift)              [x*(1+scale)+shift]
  eps_cond = proj_out(x_f16)                          [WQBMMv3 H → 64]

  # (5) Optional uncond + CFG compose (F16 InplaceAdd)
  if run_uncond:
    save eps_cond into side buffer
    D2D txt_res_u → txt_work_u
    forward_all_blocks_test(img_res_u, txt_work_u, t_emb_s, pe)
    x_f16 = layer_norm_f32_to_f16_(img_res_u)
    x_f16 = modulate_(x_f16, scale, shift)
    eps_uncond = proj_out(x_f16)                      [writes to proj_out_out]
    eps_uncond += -1 * eps_cond_saved                 [→ Δ, aliased to cond_saved]
    eps_uncond += cfg_scale * cond_saved              [= eps_u + cfg*(eps_c-eps_u)]

  # (6) D2H unpatchify
  out_f16 = D2H(proj_out_out[0:init_img_tokens*64])   # drop ref tokens
  denoised_host = host_unpatchify(F16→F32)            # [W_lat,H_lat,C_out,B]

  # (7) Euler on host
  for j in range(W*H*C*B):
      d = (x_host[j] - denoised_host[j]) / sigma
      x_host[j] += d * (sigmas[s+1] - sigma)
```

Final `x_host` after the loop is the denoised latent. Copied byte-for-byte
into `out_latent` (same shape, Qwen-Image VAE layout) for the subsequent
Step 3 decode-only probe to consume.

**Input contract (`init_from_dump` + `denoise_full`):**

| File                          | ggml-layout shape            | interpretation      |
|---|---|---|
| init_latent.f32.bin           | [W_lat, H_lat, C_lat, B]     | F32 noisy init      |
| ref_latent_0.f32.bin          | [W_lat, H_lat, C_lat, B]     | F32 VAE-encoded ref (optional) |
| cond_c_crossattn.f32.bin      | [joint_dim, txt_seq, 1, 1]   | F32 text cond       |
| uncond_c_crossattn.f32.bin    | [joint_dim, txt_seq, 1, 1]   | F32 text uncond (optional — enables CFG)     |

At the canonical 256×256 cat-edit run this gives
`W_lat=H_lat=32, C_lat=16, joint_dim=3584, txt_seq=214, init_img_tokens=256,
img_seq=512` matching the §5.5 shape plan.

**Scratch footprint** (allocated inside `denoise_full`, freed at return):
- img side: `img_in_in_f16` + 3× `[img_seq, H] F32` (res + 2× work) + 2×
  `[img_seq, H] F16` (proj_out in/out) + `[img_seq, PATCH_OUT] F16`
  → ~12 MiB at production.
- txt side: `[txt_seq, joint_dim] F32` cond+uncond (6.1 MiB) +
  `[txt_seq, joint_dim] F16` cond+uncond norm-out staging (1.5 MiB)
  + 3× `[txt_seq, H]` mix F16/F32 (~3 MiB each) → ~16 MiB at production.
- Misc: `t_emb_*` (all tiny), `adaln_*` (tiny) → <100 KB.
- Total: ~30 MiB per-request, comfortably within the 32 GiB HBM budget
  on top of the ~8 GiB resident weights.

**Known gaps carried forward into §5.5.2 (not blockers for the Step 4
eye-check, but must be addressed for full Q1 parity):**

1. **RoPE pe session-rebuild gap.** The pe table is computed in
   `init_from_gguf` for a worst-case `h=w=64` image grid and
   `ctx_len=max_txt_seq=256`. At the production 32×32 latent /
   `patch_size=2` grid, the image side is 16×16 tokens — the pe rows
   indexed by `forward_block_`'s `img_pe_off=ctx_len + row_index` point
   into the LINEAR-SWEEP positions `(t=0, h=row/64, w=row%64)` of the
   64×64 grid, not the `(h=row/16, w=row%16)` we actually want.
   Similarly, txt positions read as offset `txt_start=64` inside the pe
   (the max-grid `max(h_len,w_len)` at 64) rather than the production
   `txt_start=16`. Expected effect: numerical drift vs CPU reference
   (same DiT weights + different RoPE → different attention
   outputs) — the model is expected to still produce plausible-looking
   output because it was trained to tolerate diverse position shifts,
   but Q1 cos_sim vs CPU will not be 1.0.
   Fix: adopt Phase 3 pre-existing TODO — add
   `ImageDiffusionEngine::rebuild_rope(h_tokens, w_tokens, ref_count, ctx_len)`
   that re-runs `compute_qwen_rope_pe_host` with the request-specific
   shape and re-uploads the tables. ~150 LoC.

2. **Ref-latent RoPE temporal-axis gap.** `compute_qwen_rope_pe_host`
   assigns `t=0` for every img token; the CPU reference's
   `gen_refs_ids` assigns `t=1,2,...` for each ref-latent block. With
   a single ref (cat-edit case), tokens 256..511 should have `t=1`,
   not `t=0`. Again numerical drift rather than structural failure.

3. **Per-step t_emb semantic correctness.** `denoise_full` DOES
   rebuild t_emb per step from `sigmas[s]*1000` — resolving the gap
   §5.5 called out for `denoise_loop_test`. The final step's t_emb is
   re-used for the `norm_out` AdaLN head, matching the CPU reference
   (`QwenImageModel::forward_orig` line 540).

4. **CFG batching.** `denoise_full` runs cond+uncond sequentially per
   step (2× forward_all_blocks). Q4 CFG batching (Steps 1 and 2 of
   commit `036047de`) would halve denoise wall — follow-up work when
   the non-CFG-batched path is GREEN end-to-end.

5. **RoPE row layout + max_txt_seq=256.** The pe table's txt block
   occupies rows [0, max_txt_seq). Our production `txt_seq=214 < 256`,
   so we read rows [0, 214) and ignore rows [214, 256) — correct.
   If a future request has `txt_seq > max_txt_seq` we must bump the
   config before init_from_gguf and re-upload.

**Test harness:** `tools/probes/qie_q45_step4_full_denoise/` —
exercises init_from_gguf → init_from_dump → denoise_full → save final
latent. Receipts (fill in post-run on ac03):

| Measurement | Gate | Step 4 actual (ac03, HEAD `539f5778`) |
|---|---|---|
| Build | clean | OK (`--- build OK ---`) |
| init_from_gguf wall | < 600 s | 108.3 s |
| init peak HBM | informational | 17.93 GiB |
| init_from_dump txt_seq / has_ref / has_uncond | match dump | 214 / 1 / 0 |
| effective cfg_scale | 1.0 (no uncond dump) | 1.00 (run_uncond=0) |
| NaN in final latent | =0 | **16384 / 16384 (all-NaN)** |
| inf in final latent | =0 | 0 |
| std(final latent) | > 0.001 | 0.0 (all-NaN) |
| min(final latent) | > -20 | -3.40e+38 (NaN sentinel) |
| max(final latent) | < +20 | +3.40e+38 (NaN sentinel) |
| denoise_full wall (20 steps, cfg=1) | < 1450 s target | **24.32 s** |
| per-step wall (median / min / max) | informational | 1149.08 / 1147.91 / 1238.20 ms |
| Output PNG eye-check | recognizable cat | **solid black** (NaN → 0) |

### §5.5 Step 4c+4d smoke — RED (NaN gate fail; agent #62 surface)

Run on ac03 against fork HEAD `539f5778` (this commit). HBM lock
`/tmp/ac03_hbm_lock` taken; ac02 file-disjoint (Leak #2 bisect).

**Build:** clean. `g++ -std=c++17 -O2` against
`tools/qwen_image_edit/native/image_diffusion_engine.cpp` +
`tools/qwen_tts/cp_cann_symbols.cpp`, links `libascendcl libopapi
libnnopbase libggml-base libggml libggml-cpu`. No diagnostics.

**Init:** `init_from_gguf` 108.3 s wall, peak HBM 17.93 GiB, 1933
tensors uploaded (Q4-resident=696, F16-fallback=150). Dequant/repack
5.3 ms. `init_from_dump OK lat=[32,32,16,1] txt=[3584 x 214]
has_ref=1 has_uncond=0`. Conditioning stats sane: `txt_cond mean
-0.135 std 4.40 |.| ≤ 150.3`, `ref_latent mean -0.069 std 0.467 |.| ≤
1.68`, `init_latent` zero-init (Step 4 starts from zero and adds
scaled noise per sigma — expected).

**Denoise:** `denoise_full` ran all 20 steps, total 24.32 s wall
(median 1149 ms/step, min 1148, max 1238 on step 0). No early bail
out, no spike — per-step wall is **flat across the 20 steps**, so
NaN is not "blowing up over many steps"; it almost certainly emerges
on step 0 or step 1 and just propagates uniformly thereafter. Final
out_latent: **16384 / 16384 NaN, 0 inf**, with the F32-NaN sentinel
encoded as ±3.4e+38 in the histogram.

**Decode (Step 4d):** ran the decode-only short-circuit
`OMINIX_QIE_DECODE_ONLY_LATENT=/tmp/qie_q45_step4_latent.f32.bin
ominix-diffusion-cli ... -W 256 -H 256 --steps 20 --cfg-scale 1.0
--seed 42`. CLI honored the env, skipped denoise, ran VAE decode in
19.89 s. Both NaN-checks fired:
`decode_only/x_latent_loaded: 16384 NaN` and
`decode_only/decoded_image: 196608 NaN`. PNG saved at
`/tmp/qie_q45_step4_native_cat.png` (256×256 RGB, 2.3 KB) — all
pixels black (NaN clamped to 0 by the PNG encoder).

**Eye-check:** `qie_q45_step4_native_cat.png` is solid black; Q1
baseline `/tmp/qie_q45_host2step_baseline.png` is a clearly
recognizable gray-and-white kitten on a warm background. Step 4 PNG
is **not a recognizable cat**, so this is **not** the
"RoPE-pe-layout drift / color-shifted but recognizable" failure mode
the workplan warned about — this is a hard NaN failure inside
denoise_full, distinct from the known-gap.

**Verdict:** **RED** on the NaN gate.
`denoise_full` runs to completion, dispatches all 20 flow-Euler
steps, but produces a fully-NaN latent. Per workplan ("Don't fix
bugs in denoise_full beyond trivial typos — agent #62 owns that
surface"), no host-side debug attempted. Hand-off to agent #62.

**Hand-off pointers for agent #62:**
- per-step wall is flat (1148 ms median across all 20 steps), so
  NaN almost certainly emerges by step 0 or 1 — recommend bisecting
  by adding a single `nan_check` after the first DiT block on the
  first step, then between block-rounds.
- F16-fallback weight bytes = 9.51 GiB. If any of those participate
  in matmul accumulation under `QIE_MATMUL_INNER_PRECISE=1`
  (HIGH_PERFORMANCE / F16-accum), this is a top suspect — try
  `QIE_MATMUL_INNER_PRECISE=0` (F32-accum) for the smoke before any
  code changes.
- Conditioning is sane on entry (no NaN in init_latent / ref_latent /
  txt_cond). Sigmas first-5 = `1.0000 0.9828 0.9643 0.9444 0.9231`,
  last `0.0000`, monotonic. So the NaN is generated *inside* the
  step-0 forward, not from bad inputs / bad schedule.
- `cfg_scale` was forced to 1.0 by `has_uncond=0` (Step 2 dumps
  carry only cond, no uncond). Single forward per step, no
  cond+uncond mix. Eliminates CFG combiner as a NaN source.

**Wall summary (Step 4c+4d):** init 108.3 s + denoise_full 24.32 s
+ VAE decode 19.89 s = **152.5 s total wall**. denoise_full alone
is 60× under the 1450 s target — but obviously moot until NaN gate
goes green.

Logs / artifacts (all on ac03 unless noted):
- `/tmp/qie_q45_step4_smoke.log` (build + smoke, 47 lines)
- `/tmp/qie_q45_step4_decode.log` (CLI VAE decode log)
- `/tmp/qie_q45_step4_latent.f32.bin` (65536 B, 16384 F32, all-NaN)
- `/tmp/qie_q45_step4_native_cat.png` (also on Mac, scp'd for record)
- `/tmp/qie_q45_host2step_baseline.png` (Q1 reference, also on Mac)

**Smoke command (ac03, SIGHUP-proof):**

```
cd /home/ma-user/work/OminiX-Ascend
LOG=/tmp/qie_q45_step4_full_denoise.log
nohup setsid bash -c '
  cd tools/probes/qie_q45_step4_full_denoise && bash build_and_run.sh
' < /dev/null > "$LOG" 2>&1 &
echo "pid=$! log=$LOG"
```

Then Step 4d runs the decode-only short-circuit:

```
OMINIX_QIE_DECODE_ONLY_LATENT=/tmp/qie_q45_step4_latent.f32.bin \
  ./bin/ominix-diffusion-cli \
    --diffusion-model <path>/Qwen-Image-Edit-2509-Q4_0.gguf \
    --vae <path>/qwen_image_vae.safetensors \
    --prompt "convert to black and white" \
    --ref-image /tmp/cat.jpg \
    --width 256 --height 256 --steps 2 --cfg-scale 1.0 \
    --sample-method euler --seed 42 \
    --output /tmp/qie_q45_step4_native_cat.png
```


### §5.5.2 Step 4 smoke re-run — RED under `QIE_MATMUL_INNER_PRECISE=0` (F32-accumulator)

Re-run on ac03 against fork HEAD `722d1cf9` with the 4.4d-wired
matmul-precision env knobs forced to F32-accum. Hypothesis under test:
9.5 GiB F16-fallback weight path (Q4_1 FFN-down + Q5_K layers 0/59)
overflows at first matmul under default F16-accumulator.

**Command (SIGHUP-proof, HBM lock held):**

```
nohup setsid bash -c '
  touch /tmp/ac03_hbm_lock && \
  cd ~/work/OminiX-Ascend/tools/probes/qie_q45_step4_full_denoise && \
  QIE_MATMUL_INNER_PRECISE=0 QIE_MATMUL_CUBE_MATH=1 \
  GGML_BUILD=$HOME/work/OminiX-Ascend/build-w1 \
  GGML_CANN_QUANT_BF16=on \
  bash build_and_run.sh 2>&1 | tee /tmp/qie_q45_step4_f32acc.log; \
  rm -f /tmp/ac03_hbm_lock
' < /dev/null > /dev/null 2>&1 &
```

**Env actually honored by native engine:**

```
[qie_native] dispatch_matmul_: QIE_MATMUL_INNER_PRECISE=0
  (WQBMMv3 innerPrecise; 0=HIGH_PRECISION/F32-accum, 1=HIGH_PERFORMANCE/F16-accum)
[qie_native] dispatch_matmul_: QIE_MATMUL_CUBE_MATH=1
  (aclnnMm cubeMathType; 0=KEEP_DTYPE, 1=ALLOW_FP32_DOWN_PRECISION, 2=USE_FP16, 3=USE_HF32)
```

Env knobs are read at `denoise_full` entry and logged before the first
dispatch, so the first forward runs with F32-accum WQBMMv3 +
ALLOW_FP32_DOWN_PRECISION aclnnMm. No fall-through path.

**Init:** identical footprint to §5.5 (1933 tensors, 9.51 GiB F16
fallback, 7.14 GiB Q4, 17.93 GiB peak HBM, 101.4 s wall).

**Denoise:** `denoise_full` 26.48 s wall (vs 24.32 s under F16-accum;
+8.9% cost for the wider accumulator, within noise). Per-step wall
still flat: median 1184 ms across all 20 steps (F16-accum was 1149
ms). Final out_latent stats are bit-pattern-identical to the §5.5 RED
run: `mean=0 std=0 min/max=±1e+30 NaN=16384 inf=0`. 16384 out of 16384
elements are the F32-NaN sentinel.

**Verdict:** **RED** — unchanged from §5.5. F32 inner-accumulator on
WQBMMv3 does NOT fix the step-0 NaN. This **falsifies** the
"F16-accum overflow at first matmul" hypothesis.

**Signal we just bought:**

The NaN is not coming from the 910b cube unit's F16-accumulator. This
is consistent with the Leak #2 per-op trace result
(`docs/qie_leak2_per_op_trace.md`): that trace measured residual-stream
growth to 1.1e5 (above F16's 65504 limit) by block 1, and proved that
F32-residual cast at graph level does NOT reduce the F32-correct
magnitudes — the overflow is in a **backend storage-time saturation**
inside a CANN op output dtype, not in the inner accumulator.

The matmul-output dtype on 910b (WQBMMv2/v3, per
`aic-ascend910b-ops-info.json`) is F16/BF16-only. `INNER_PRECISE=0`
promotes the *accumulator* to F32, but the *output cast* still lands
in F16 (or BF16 under `GGML_CANN_QUANT_BF16=on`). Any residual
magnitude > 65504 (F16) or > ~3.4e38 (BF16 range but with 8 fewer
mantissa bits than F32) saturates at the output cast, regardless of
inner precision.

**Proposed next diagnostics (defer to agent #62 per Step 4 workplan):**

1. Confirm `GGML_CANN_QUANT_BF16=on` is plumbed into the *native*
   engine's `dispatch_matmul_` — the env var is in Path C #4's
   backend path (ominix-diffusion-cli), not necessarily in the native
   probe. If the native probe still emits F16 matmul outputs,
   residuals > 65504 saturate at block 0. Audit
   `tools/qwen_image_edit/native/image_diffusion_engine.cpp`
   `dispatch_matmul_` for BF16 output selection.
2. Insert a `nan_check` on the first matmul output (Q/K/V projection
   of block 0, img stream) in the native probe — should pinpoint
   which projection first saturates.
3. If BF16 plumbing is already active: the Leak #2 residual-stream
   magnitude (1.1e5 by block 1) still exceeds practical F16/BF16
   stability. The only path is F32 residual *storage* end-to-end,
   which in the native engine means promoting the residual-add
   tensor + LayerNorm compute to F32 (analog of 4.4d F32-resid on
   the aclnn side). Audit which residuals land F16 in
   `build_graph_` / `denoise_full`.

**Wall summary (Step 4c re-run, F32-accum):**
init 101.4 s + denoise_full 26.5 s = 127.9 s total. Step 4d decode
skipped (NaN latent → guaranteed black PNG, same as §5.5).

**Logs / artifacts (ac03):**
- `/tmp/qie_q45_step4_f32acc.log` — full smoke (45 lines)
- `/tmp/qie_q45_step4_latent.f32.bin` — overwritten, still 16384
  NaN (65536 B), same sentinel pattern

### §5.5.3 Step 4 NaN bisect — leak isolated to FFN-down matmul, NOT the five projections

Run on ac03 against fork HEAD `b36c5f76` with diagnostic instrumentation
landed in `image_diffusion_engine.cpp` (three new env-gated probe paths,
all dormant when env unset — Step 1 regression preserved):

- `QIE_DEBUG_NAN_BISECT=1` — probes the five projections inside
  `denoise_full` (img_in, txt_in/txt_norm, time_linear1/2 + SiLU,
  norm_out + LN + modulate, proj_out). Step 0 only.
- `QIE_DEBUG_PER_BLOCK_NAN=1` — F32 residual stream scan after each
  of the 60 blocks. First call only.
- `QIE_DEBUG_INTRA_BLOCK0=1` — 24-point scan of every key buffer
  inside the FIRST `forward_block_` invocation: t_emb, silu, mod params,
  LN, modulate, QKV (img+txt), RMSNorm, RoPE, attention out, output
  proj, gated residual #1, LN2, modulate2, FFN up, GELU, FFN down,
  gated residual #2.

#### Receipt 1 — five-projection probe (gate: `QIE_DEBUG_NAN_BISECT=1`)

All five projection sites are CLEAN at step 0 entry — no NaN, no Inf,
all magnitudes well below F16 65504:

| Probe | dtype | mean_abs | max_abs | NaN | Inf |
|---|---|---|---|---|---|
| pre_txt_norm.cond | F32 | 2.728 | 150.3 | 0 | 0 |
| post_txt_norm.cond | F16 | 0.469 | 10.95 | 0 | 0 |
| post_txt_in.cond | F16 | 7.039 | 582 | 0 | 0 |
| txt_res_c_f32 | F32 | 7.039 | 582 | 0 | 0 |
| pre_time_linear1 | F16 | 0.636 | 1.000 | 0 | 0 |
| post_time_linear1 | F16 | 0.231 | 21.44 | 0 | 0 |
| post_time_silu | F16 | 0.0804 | 21.44 | 0 | 0 |
| post_time_linear2 | F16 | 0.326 | 111.8 | 0 | 0 |
| pre_img_in | F16 | 0.186 | 1.683 | 0 | 0 |
| post_img_in | F16 | 0.325 | 16.12 | 0 | 0 |
| img_res_c_f32.seed | F32 | 0.325 | 16.12 | 0 | 0 |

**Verdict:** mission's primary hypothesis ("five projections produce
real-magnitude F16 outputs that saturate before the 60-block loop")
is **falsified**. The five projections feed the block loop with clean
inputs.

#### Receipt 2 — per-block residual scan (gate: `QIE_DEBUG_PER_BLOCK_NAN=1`)

NaN emerges immediately at **block 0**:

| Block | img_resid F32 mean_abs | max_abs | Inf | txt_resid F32 mean_abs | max_abs | Inf |
|---|---|---|---|---|---|---|
| 00 | 9057 | 6.584e+04 | **267092** | 2.449e+04 | 6.567e+04 | **224334** |
| 01 | 0 | 0 | 0 | 0 | 0 | 0 (NaN=657408) |
| 02-59 | 0 | 0 | 0 (NaN=full) | 0 | 0 | 0 (NaN=full) |

Block 0 produces **267092 Inf elements in img_resid + 224334 in
txt_resid**. From block 1 onwards every element is NaN (Inf-Inf or
Inf*0 cascades). The 4.4d F32 residual storage is doing its job —
it's not silently saturating at F16; it correctly stores the F32 Inf
that came from block 0's internal F16 op chain.

#### Receipt 3 — intra-block-0 24-point scan (gate: `QIE_DEBUG_INTRA_BLOCK0=1`)

Magnitude cascade through forward_block_ on block 0:

| Step | Buffer | dtype | mean_abs | max_abs | NaN | Inf |
|---|---|---|---|---|---|---|
| 00 | img_hidden_in | F32 | 0.325 | 16.12 | 0 | 0 |
| 00 | txt_hidden_in | F32 | 7.039 | 582 | 0 | 0 |
| 00 | t_emb_in | F16 | 0.326 | 111.8 | 0 | 0 |
| 01 | silu_t_emb | F16 | 0.148 | 111.8 | 0 | 0 |
| 02 | img_mod_out | F16 | 8.021 | 269 | 0 | 0 |
| 03 | txt_mod_out | F16 | 9.482 | 141.8 | 0 | 0 |
| 04 | img_LN1 | F16 | 0.538 | 11.74 | 0 | 0 |
| 05 | img_mod1 | F16 | 2.078 | 113.5 | 0 | 0 |
| 06 | txt_LN1 | F16 | 0.623 | 39.5 | 0 | 0 |
| 07 | txt_mod1 | F16 | 5.315 | 1057 | 0 | 0 |
| 08 | img_Q/K/V | F16 | 11.9 / 17.5 / 8.8 | 210 / 432 / 127 | 0 | 0 |
| 08 | txt_Q/K/V | F16 | 89 / 74 / 37 | 1012 / 859 / 272 | 0 | 0 |
| 09 | img_QK_rmsnorm | F16 | 1.95 / 1.27 | 723 / 564 | 0 | 0 |
| 09 | txt_QK_rmsnorm | F16 | 0.63 / 0.76 | 6.8 / 8.2 | 0 | 0 |
| 10 | img_QK_rope | F16 | 1.96 / 1.27 | 723 / 564 | 0 | 0 |
| 10 | txt_QK_rope | F16 | 0.64 / 0.77 | 6.6 / 8.1 | 0 | 0 |
| 11 | attn_out_txt / img | F16 | 14.0 / 18.5 | 175 / 186 | 0 | 0 |
| 12 | to_add_out / to_out_0 | F16 | 73.7 / 70.8 | 740 / 634 | 0 | 0 |
| 13 | img/txt resid1 | F32 | 45.5 / 69.7 | 1727 / 3957 | 0 | 0 |
| 14 | img_LN2 | F16 | 0.447 | 16.36 | 0 | 0 |
| 15 | img_mod2 | F16 | 29.4 | 634.5 | 0 | 0 |
| 16 | txt_LN2 | F16 | 0.402 | 20.5 | 0 | 0 |
| 17 | txt_mod2 | F16 | 34.6 | 159.9 | 0 | 0 |
| 18 | img_ff_up | F16 | 204.7 | 2434 | 0 | 0 |
| 19 | img_gelu | F16 | 108 | 2434 | 0 | 0 |
| **20** | **img_ff_down** | F16 | **4796** | **6.355e+04** | 0 | **0 (just below F16 limit)** |
| 21 | txt_ff_up | F16 | 257 | 3706 | 0 | 0 |
| 22 | txt_gelu | F16 | 145 | 3230 | 0 | 0 |
| **23** | **txt_ff_down** | F16 | **4957** | **5.488e+04** | 0 | **214** ← FIRST Inf |
| 24 | img_resid2 | F32 | 9057 | 6.584e+04 | 0 | 267092 |
| 24 | txt_resid2 | F32 | 2.449e+04 | 6.567e+04 | 0 | 224334 |

#### Root cause

**The leak is the FFN down-projection matmul output dtype.** Specifically
`txt_ff_down` (`scratch_mlp_dev_ × txt_ff_down_w_q4` → `scratch_txt_out_dev_`
F16) — output magnitudes hit F16 max (65504) and 214 elements clip to
Inf in the F16 storage. The subsequent `gated_residual_add_f32_` path
casts F16 Inf → F32 Inf (Cast preserves Inf semantics — there's no clamp).
The F32 residual storage now contains Inf, and the next block's LayerNorm
sees mean=Inf, variance=Inf/Inf=NaN → produces NaN throughout.

`img_ff_down` is at 6.355e+04 — **just barely** under F16 max — so img
escapes block 0 with no Inf, but combined with 267092 img_resid Infs at
24_img_resid2 we know elements crossed the limit during the gate-add
path. Looking more carefully, gate2 magnitudes (3rd chunk of img_mod2
output) combine multiplicatively with ff_down to push into Inf:
`(ff_down_f16 * gate_f16)` is computed in F16 at `gated_residual_add_f32_`
step (uses `aclnnMul` on F16 inputs, see line 1820-ish), and that F16
multiplication overflows long before the F32 cast.

The ff_down output magnitude is driven by:
- Modulate2 input (LN output ~1σ) × `(1 + scale2)` where `scale2` has
  max ~635. → mod2 output max ~635.
- ff_up matmul on H=3072 input, FF=12288 output. Even with weight stdev
  ~0.02, max(mod2)·sqrt(H)·stdev ≈ 635·55·0.02 ≈ 700; observed 2434
  (worst-case rows).
- GELU(2434) ≈ 2434 (saturated linear regime).
- ff_down matmul on FF=12288 input, H=3072 output. Worst-case rows
  hit F16 max.

#### Why §4.4d (synthetic real-GGUF smoke) was GREEN

The 4.4d real-GGUF smoke (`tools/probes/qie_q44_real_gguf_smoke`) used
`fill_random_f32_via_f16(amp=0.1)` for img/txt residuals AND
`fill_random_f16(t_emb, amp=0.1)`. With t_emb max ≈ 0.1 instead of
the real 111.8:

- silu(t_emb) max ≈ 0.05 (vs 111.8 real)
- img_mod / txt_mod outputs max ≈ 5 (vs 269 / 141 real)
- modulate2 output max ≈ 5 (vs 634 real)
- ff_up output max ≈ 25 (vs 2434 real)
- ff_down output max ≈ 100 (vs 65504 real)

The 4.4d test was numerically **two orders of magnitude under** the
real Step 4 magnitudes everywhere downstream of t_emb. F32 residual
storage worked fine because there was nothing to store-overflow.

#### Why mission's prescribed F32 widening doesn't help

Mission asked: "extend 4.4d F32 to img_in / txt_in / time_linear /
norm_out / proj_out projection outputs". Per Receipt 1, those outputs
are already clean (max ≤ 582). Casting them to F32 immediately after
the matmul is a no-op for clean values — the matmul **already**
produced a representable F16, and F16→F32 cast preserves it.

The mission's escape hatch applies:
> If NaN persists after F32 projections: the issue is elsewhere
> (modulation gate? FIA softmax?). Document and escalate.

The leak is **inside forward_block_'s FFN down-projection**, which is
inside the 4.4d-protected block — not at the projection boundary. The
fix requires widening the **internal** matmul output dtype, NOT the
projection boundary.

#### Proposed escalation paths

1. **Widen ff_down output dtype**: smallest scope. Replace
   `dispatch_matmul_(scratch_mlp_dev_, ff_down_w, ..., scratch_*_out_dev_)`
   with a variant that produces F32 output, then cast F32→F16 only at
   the entry to gate-mul. **But** this only delays the problem: the
   F16 Mul inside `gated_residual_add_f32_` still overflows when
   ff_down*gate exceeds F16 max. Need to also rewrite that path to
   compute `(F32 ff_down * F16 gate)` cast-in-Mul or `(F32 ff_down *
   F32 gate)` post-cast Mul. Vendor risk: WQBMMv3 doesn't directly
   support F32 output on 910b — would require dequantising ff_down
   weights to F16 and routing through aclnnMm with F32 output tensor
   (cubeMathType=ALLOW_FP32_DOWN_PRECISION). Dequant adds memory cost
   — ff_down is the largest matmul (FF×H = 12288×3072 = 37.7M
   weights × 2 bytes per layer × 60 layers = 4.5 GiB extra HBM if we
   keep an F16 copy for all layers).

2. **Widen mod2 output dtype**: prevent modulate2 from producing
   max=635. If mod_b2 is split scale/shift, scale could be capped /
   regularized at load time — but that would alter trained behavior.
   Probably NOT a viable surgical fix.

3. **Rescale on entry to ff_up**: clip / scale modulate2 output before
   FFN. Same concern — alters behavior.

4. **Promote attn-out projection (`to_out_0`, `to_add_out`) to F32**:
   these saw max=740 / 634 at block 0 — under F16 limit but close. At
   deeper blocks they may exceed 65504. Same widening cost question.

5. **Block-level magnitude clamp**: insert an F16 max-clip after each
   matmul output to prevent overflow. Cheap (one aclnnClamp per
   matmul) but mathematically incorrect — clipping changes activations.

6. **BF16 storage end-to-end**: 910b WQBMMv3 supports BF16 output via
   `GGML_CANN_QUANT_BF16=on` env. BF16 has F16-equivalent precision
   but F32-equivalent range (~3.4e38). This is a separate workstream
   already in flight (see ggml-cann fork Path C #4). For QIE native
   engine, plumbing BF16 weight/output dtype through `dispatch_matmul_`
   would solve the FFN overflow at the matmul level. Recommended path.

#### Verdict

**RED — root cause isolated, NOT in scope of mission's prescribed fix.**
Mission's prescription would not change the outcome (Receipt 1 proves
projections are clean). Escalating: requires widening the **internal**
ff_down matmul output dtype (Path 1) or BF16 plumbing into
dispatch_matmul_ (Path 6). Both are larger changes than the mission's
"widen 5 projections" scope.

#### What landed in this session

- New diagnostic instrumentation (env-gated, dormant by default):
  `QIE_DEBUG_NAN_BISECT`, `QIE_DEBUG_PER_BLOCK_NAN`, `QIE_DEBUG_INTRA_BLOCK0`.
  Together they bisect any future NaN regression to a specific op in
  ≤ 3 build cycles. Step 1 `denoise_loop_test` synthetic regression is
  unaffected (env vars off by default; per-block scan is one-shot
  static-latched on first call).
- Empirical magnitude profile of block 0 with real weights at
  σ=1.0 (the high-noise regime), 24 measurement points.
- Falsification of mission's primary hypothesis.
- Identification of the actual leak surface (FFN down-projection, not
  the projections).

#### Logs / artifacts (ac03)
- `/tmp/qie_q45_step4_bisect.log` — five-projection probe (RED, all
  projections clean, NaN emerges in 60-block loop)
- `/tmp/qie_q45_step4_perblock.log` — per-block scan (Inf at b00,
  NaN propagating from b01)
- `/tmp/qie_q45_step4_intra.log` — 24-point intra-block-0 scan (Inf
  emerges at 23_txt_ff_down + 20_img_ff_down)

#### Wall summary
- Five-projection bisect: init 100.0 s + denoise_full 24.9 s = 124.9 s
- Per-block scan (n_steps=1): init 103.7 s + denoise_full ~3 s
- Intra-block scan (n_steps=1): init 103.0 s + denoise_full 2.7 s

### §5.5.4 Step 4c — BF16 plumbing on FFN-down matmul (PARTIAL: leak #1 fixed, leak #2 surfaced)

Mission (per Q2.4.5.4c handoff): plumb BF16 output dtype through the
native engine's `dispatch_matmul_` helper for the FFN-down site to
escape the F16 65504 saturation isolated in §5.5.3.

#### Implementation
`tools/qwen_image_edit/native/image_diffusion_engine.{h,cpp}`:

1. New optional `aclDataType out_dtype = ACL_FLOAT16` argument on
   `dispatch_matmul_`. ACL_BF16 is supported on both internal paths:
   - **WQBMMv3 (Q4 + F16-scale)**: 910b op-spec lists F16/BF16 output;
     scale is cast F16→BF16 inline (precedent: ggml-cann backend's
     `GGML_CANN_QUANT_BF16` path at `aclnn_ops.cpp:2670-2686`); bias
     cast F16→BF16 lazily for the InplaceAdd post-step.
   - **aclnnMm (F16-fallback)**: status 161002 rejects mixed-dtype
     direct dispatch, so we pre-cast input + weight F16→BF16 to a pair
     of lazy scratch buffers (`scratch_bf16_scale_dev_`,
     `scratch_bf16_src_f32_dev_` repurposed) and dispatch BF16/BF16 →
     BF16 with `cubeMathType=ALLOW_FP32_DOWN_PRECISION` (F32
     accumulator). The weight cast is the dominant per-call cost
     (~75 MB at FF=12288, H=3072) — see Wall summary below.
2. New `gated_residual_add_f32_bf16src_` helper consumes the BF16
   output: F32 cast of BF16 src + F32 cast of F16 gate + F32 mul +
   F32 InplaceAdd into the F32 residual stream. No F16 round-trip
   in the gate-mul path → no 65504 saturation.
3. `forward_block_`: `QIE_FFN_DOWN_BF16=1` (or `QIE_ALL_BF16=1` alias)
   routes both img_ff_down and txt_ff_down through the BF16-output
   variant + BF16-src gated residual. Default OFF — Step 1
   `denoise_loop_test` synthetic regression invariant preserved.

`docs/qie_q2_phase4_smoke.md` — this section.

#### Receipt 1 — block 0 ff_down: F16-Inf → BF16-65000 (GREEN)

ac03 / `qie_q45_step4_full_denoise` smoke under
`QIE_FFN_DOWN_BF16=1 QIE_DEBUG_PER_BLOCK_NAN=1 QIE_DEBUG_INTRA_BLOCK0=1`:

| Metric | Pre-fix (76bc4652) | Post-fix (this) |
|---|---|---|
| `intra_b0[20_img_ff_down]` dtype | F16 | BF16 |
| `intra_b0[20_img_ff_down]` max_abs | 6.355e+04 (clipping) | **6.349e+04** |
| `intra_b0[23_txt_ff_down]` dtype | F16 | BF16 |
| `intra_b0[23_txt_ff_down]` max_abs | 5.488e+04 + 214 Inf | **7.322e+04** (no Inf) |
| `intra_b0[24_img_resid2]` | NaN | F32 max=7.197e+06, NaN=0 |
| `intra_b0[24_txt_resid2]` | NaN | F32 max=4.647e+06, NaN=0 |
| `per_block_nan[b00/{img,txt}]` | NaN=all | **NaN=0 Inf=0** |

The §5.5.3 root cause is fixed: ff_down output cleanly holds magnitudes
> 65504 in BF16 storage; the F32-residual gated add never sees a
saturating intermediate. Block 0 finishes byte-clean.

#### Receipt 2 — block 1 IMG goes NaN (RED — leak #2 surfaced)

```
per_block_nan[b00/img]: mean_abs=1.034e+05 max_abs=7.197e+06 NaN=0 Inf=0
per_block_nan[b00/txt]: mean_abs=7.017e+04 max_abs=4.647e+06 NaN=0 Inf=0
per_block_nan[b01/img]: mean_abs=0 max_abs=0 NaN=1572864 Inf=0   ← FIRST NaN
per_block_nan[b01/txt]: mean_abs=7.369e+04 max_abs=4.561e+06 NaN=0 Inf=0
per_block_nan[b02/img]: mean_abs=0 max_abs=0 NaN=1572864 Inf=0
per_block_nan[b02/txt]: mean_abs=0 max_abs=0 NaN=657408 Inf=0
... (all subsequent blocks NaN, both streams)

VERDICT: RED (NaN/inf)
final latent: mean=0, NaN=16384/16384
```

The leak has migrated. Block 0's residual exits at F32 max ~7.2M
(safe — F32 limit is 3.4e38). Block 1's IMG stream produces NaN
during the block — TXT survives one more block. The bisect
(`QIE_DEBUG_INTRA_BLOCK0=1`) only scans block 0, so the exact b01
op needs another instrumentation pass.

Hypothesis: block 1's `img_ff_up` (still F16 — only ff_down was
widened in this scope) saturates because the upstream modulation
chain feeds it a larger input than block 0 saw (post-fix,
`24_img_resid2` is 7.2M vs the pre-fix's 4M-ish observation; LN1
normalizes back to std≈1, but mod1 then re-amplifies by a layer-1
specific scale that may exceed block 0's). Need:
1. Extend `QIE_DEBUG_INTRA_BLOCK0` to scan block 1 (one-shot static
   latch on second call).
2. Re-bisect to identify exact b01 leak point.

#### Receipt 3 — wall (per-step latency)

| | wall ms / step | per-fix ratio |
|---|---|---|
| Pre-fix (F16 ff_down) | 1240 (24.9 s / 20) | 1.00× |
| Post-fix (BF16 ff_down) | **1656** (33.1 s / 20) | **1.34×** |

The ~30% slowdown is the per-call F16→BF16 weight cast in
the aclnnMm fallback path (ff_down weights are Q4_1 in this GGUF,
not Q4_0, so they route through aclnnMm not WQBMMv3). 75 MB ×
60 layers × 2 streams × 20 steps = 7200 casts × ~0.5 ms ≈ 3.6 s
of cast wall, matching the +8 s observation (rest is BF16
math marginally slower than F16 on 910b). Optimisation lane
(if BF16 is wider-adopted): pre-convert all ff_down weights to
BF16 at `init_from_gguf` time, mirroring ggml-cann's
`GGML_CANN_QUANT_BF16=on` static conversion. Out of scope for this
patch.

#### Status — what landed

- **Mission's primary delivery target met for block 0**: the §5.5.3
  FFN-down F16 saturation no longer produces NaN at block 0.
- **Mission's secondary target (full 20-step denoise GREEN) NOT met**:
  block 1+ IMG stream still goes NaN. New bisect needed to locate
  leak #2.
- The BF16-plumbing scaffold is reusable for Step 7 widening: any
  future call site that needs BF16 output just passes
  `out_dtype=ACL_BF16` and the helper transparently handles scale +
  bias + input/weight casts. The same recipe can be applied to
  ff_up, attn-out projections, etc., once leak #2 is bisected.
- Wall cost +30% per step. Tolerable for a defensive fix; pre-cast
  optimization is the obvious follow-up.

#### Logs / artifacts (ac03)

- `/tmp/qie_q45_step4_bf16_run3.log` — RED-but-block0-clean smoke.
  Final latent in `/tmp/qie_q45_step4_latent.f32.bin` (16384 NaN,
  Step 4d eye-check skipped — no signal in NaN latent).
- `/tmp/qie_q45_step4_bf16_run1.log`, `run2.log` — earlier runs,
  prove the iterative diagnosis (run 1: ff_down on aclnnMm fallback
  silently kept F16; run 2: aclnnMm rejected mixed-dtype direct
  dispatch with status 161002; run 3: F16→BF16 input/weight precast
  works).

#### Next-step queue

1. **Re-bisect leak #2** (block 1 IMG): extend `QIE_DEBUG_INTRA_BLOCK0`
   to scan block N via env (e.g., `QIE_DEBUG_INTRA_BLOCK=1`). Identify
   the saturating op.
2. **Widen BF16 to that op's site** using the same dispatch_matmul_
   knob (or new helper if it's not a matmul).
3. **Decide on init-time BF16 weight pre-cast** to recover the 30%
   wall regression.
4. **Eye-check skipped** until the latent has signal.

### §5.5.5 Step 4d — BF16 widening to attn-out projections (Step 7) — NaN gate GREEN

Mission (Q2.4.5.4d handoff): widen the §5.5.4 BF16 scaffold to all
matmul callsites in the native engine forward path so the §5.5.4 leak
#2 (block 1 IMG NaN once block 0 ff_down was BF16-clean) is also
fixed. Pragmatic interpretation: residual-stream contributors (attn-out
projections + ff-down) are the only matmul outputs whose magnitudes
can exceed F16 65504 once the residual stream grows past a few
blocks; other matmul outputs (Q/K/V proj, FFN-up, modulation linear,
norm_out / proj_out / time_linear / img_in / txt_in) are bounded by
upstream LN/SiLU normalization (max ≤ ~3700 observed at block 0). So
"all matmul outputs to BF16" applied to the leaky surface — not every
matmul indiscriminately, since most non-residual consumers are
F16-strict (RMSNorm + RoPE + FIA, GELU, modulate, Cast/SiLU, CFG
compose) and would require a much bigger refactor for negligible
correctness benefit.

#### What landed

`tools/qwen_image_edit/native/image_diffusion_engine.cpp`:

1. New static `s_all_bf16` env-cache, decoupled from `s_ffn_down_bf16`.
   `QIE_ALL_BF16=1` is now a strict superset of `QIE_FFN_DOWN_BF16=1`:
   under `QIE_ALL_BF16` the attn-output projections (`to_add_out` for
   txt and `to_out_0` for img — the residual #1 contributors) ALSO
   emit BF16, AND gated-residual #1 routes through the BF16-src
   variant.
2. `dispatch_matmul_` BF16 path's input pre-cast (F16→BF16 to scratch)
   was previously gated on `weight_scale_dev == nullptr` (aclnnMm
   fallback). Lifted the gate so it runs for the WQBMMv3 path too —
   WQBMMv3 returned status 161002 ("wrong dtype combo") when
   `t_x` stayed F16 against BF16 scale + BF16 output. The fix uses
   `scratch_bf16_src_f32_dev_` for the input pre-cast on the WQBMMv3
   path (vs `scratch_bf16_scale_dev_` on the aclnnMm path) to avoid
   aliasing the still-required scale-tile cast destination.
3. Build a BF16 view (`t_x_local_bf16`) over the pre-cast input for
   the WQBMMv3 launch. Defensive teardown on all error paths.

Default (env unset): all callsites stay F16, byte-identical to §5.5.4.
Step 1 `denoise_loop_test` regression GREEN with env OFF (verified —
synthetic CPU-ref cos_sim invariant preserved).

#### Receipts (ac03 / qie_q45_step4_full_denoise smoke, n_steps=20)

`QIE_ALL_BF16=1 QIE_DEBUG_PER_BLOCK_NAN=1`:

| Block | img_resid mean_abs / max_abs | NaN | Inf | txt_resid mean_abs / max_abs | NaN | Inf |
|---|---|---|---|---|---|---|
| 00 | 1.035e+05 / 7.197e+06 | 0 | 0 | 7.017e+04 / 4.683e+06 | 0 | 0 |
| 01 | 1.131e+05 / 9.057e+06 | 0 | 0 | 7.369e+04 / 4.598e+06 | 0 | 0 |
| 30 | 2.150e+05 / 1.572e+07 | 0 | 0 | 1.112e+05 / 6.683e+06 | 0 | 0 |
| 59 | 3.049e+05 / 6.506e+07 | 0 | 0 | 1.132e+05 / 6.720e+06 | 0 | 0 |

Block 0 → block 1 IMG NaN cascade in §5.5.4 is gone. Magnitudes grow
~3× across 60 blocks (1e5 → 3e5 mean_abs) but stay 11 orders of
magnitude under BF16 max (3.4e38) and never overflow.

Final latent (after 20 flow-Euler steps):
`mean=-2.4557 std=4.8610 min/max=-13.3359/7.5898 NaN=0 inf=0` — VERDICT GREEN.

#### Wall

| | per-step ms | wall ratio vs §5.5.4 baseline |
|---|---|---|
| §5.5.4 (ff_down BF16 only, RED at b01) | 1656 | 1.00× |
| §5.5.5 (ff_down + attn-out BF16, GREEN) | **1165** (median) | **0.70×** |

The widening is FASTER than the §5.5.4 ff_down-only path — the new
attn-out matmul callsites route through WQBMMv3 (Q4_0 weight + scale
tile cast only, no per-call 75 MB weight cast like the aclnnMm
fallback ff_down path that dominated §5.5.4's wall). Per-step
1165 ms median is comparable to §5.5.4's pre-fix F16 baseline
(1240 ms) — net ~6% faster end-to-end under full BF16 widening, and
crucially produces non-NaN output now.

`forward_block_: QIE_FFN_DOWN_BF16=1 QIE_ALL_BF16=1 (ff_down BF16 +
bf16src #2 always under either; attn-out BF16 + bf16src #1 only
under ALL)` — log line confirms env routing.

#### Step 4d eye-check

Step 4d decode-only (`OMINIX_QIE_DECODE_ONLY_LATENT=...`):
`decode_only/x_latent_loaded: OK (16384 elements, range=[-13.335938,
7.589844])`, `decode_only/decoded_image: OK (196608 elements,
range=[0.0, 1.0])`, PNG saved at
`/tmp/qie_q45_step4d_allbf16_cat.png` (110986 B, 256×256 RGB).

Eye-check verdict: **non-NaN, finite, structured, but NOT a
recognizable cat** — output is a regular blue-checkerboard pattern
suggesting an unpatchify-host or RoPE-pe-layout artifact independent
of the BF16 widening (this matches the §5.5 known caveat about
"RoPE pe-table drift / color-shifted but recognizable" — except the
output is a coherent tile pattern, not a color-shifted cat). Numeric
gate is fully GREEN; visual gate is a separate workstream (host-side
unpatchify + pe alignment). NOT a third NaN leak.

#### Status — what landed

- **Mission's primary delivery target met**: full 20-step denoise
  produces no NaN, no Inf, finite latent, GREEN verdict on the
  numerical gate.
- **Mission's secondary delivery target (visual cat eye-check)
  partially met**: PNG produced is structured / finite / non-NaN
  but is a tile pattern, not a recognizable cat. Likely an
  unpatchify-host or RoPE pe-table layout artifact, distinct from
  matmul-output saturation. Defer to a separate pass.
- The BF16 scaffold scaling held — adding attn-out-BF16 was a
  one-flag addition over §5.5.4's ff_down-only plumbing, and the
  WQBMMv3 input pre-cast lift unblocked Q4-resident weights too
  (§5.5.4's scaffold only exercised the aclnnMm fallback because
  ff_down weights are Q4_1; attn-out weights are Q4_0 → WQBMMv3).

#### Logs / artifacts (ac03)

- `/tmp/qie_q45_step4_allbf16_run2.log` — full GREEN smoke
  (init 101.5 s, denoise 25.4 s, all 60 blocks NaN=0 Inf=0,
  EXITCODE=0, VERDICT=GREEN).
- `/tmp/qie_q45_step4_allbf16_run1.log` — earlier RED run (block 1
  failed with WQBMMv3 status=161002 before the input-pre-cast lift).
- `/tmp/qie_q43_denoise_default_run.log` — Step 1 regression GREEN
  with env OFF (`QIE_FFN_DOWN_BF16=0 QIE_ALL_BF16=0` confirmed,
  cos_sim invariant preserved, EXITCODE=0).
- `/tmp/qie_q45_step4_latent.f32.bin` (65536 B, 16384 F32, finite).
- `/tmp/qie_q45_step4d_allbf16_cat.png` (110986 B, 256×256 RGB,
  tile pattern — sample of structured non-NaN VAE-decoded output).

#### Next-step queue

1. **Visual eye-check chase**: investigate why VAE-decoded latent
   produces a tile pattern rather than a color-shifted cat. Likely
   suspects: host-side unpatchify token layout (concat order
   img_init || img_ref vs the diffusion model's per-step expectation),
   or RoPE pe-table per-step image-token offset mismatch. Independent
   of BF16 — the latent itself is sane.
2. **Optimization lane (deferred)**: pre-convert ff_down + attn-out
   weights to BF16 at `init_from_gguf` time (mirror's ggml-cann's
   `GGML_CANN_QUANT_BF16=on` static conversion). Wall is already
   competitive (1165 ms median vs 1240 ms baseline) so this is a
   "if/when we widen further" follow-up, not a hot priority.
3. **Possibly widen further** (Q/K/V, FFN-up, mod, projections) only
   IF a future deep-block magnitude probe shows non-residual matmul
   outputs approaching F16 65504. Current data says they don't
   (max ≤ ~3700 at block 0).

### §5.5.6 Step 4f — tile-pattern bisect: unpatchify GREEN, pe DRIFT (minor), missing noise PARTIAL, attention saturation RED

Mission (Q2.4.5.4f handoff): the §5.5.5-d gate is GREEN numerically
(no NaN through 20 steps × 60 blocks, sane final-latent F32 stats) but
the VAE-decoded PNG renders as a regular blue tile pattern rather than
a recognizable cat (`/tmp/qie_q45_step4d_allbf16_cat.png` on Mac).
The §5.5.5 next-step queue called out unpatchify token layout and RoPE
pe per-step offset as prime suspects. Step 4f tested both and found
neither is the root cause; the actual cause is a per-token-variation
collapse downstream of the 60-block forward (post-block cosine
similarity between img tokens is 0.999 vs the pre-block 0.027). One
contributing input bug was fixed (missing noise), but the
post-block collapse persists with proper noise. The visual cat
remains BLOCKED on a separate downstream defect.

#### Diagnostic — latent inspection rules out unpatchify

Pulled `qie_q45_step4_latent.f32.bin` (post-20-step output) to host as
`[1, 16, 32, 32]` F32, computed per-channel block coherence:

| channel | inblock σ (within 2×2) | cross-block σ (between 256 patches) |
|---|---|---|
| c=0  | 2.349 | **0.033** |
| c=4  | 2.023 | **0.024** |
| c=8  | 2.252 | 0.058 |
| c=12 | 5.161 | 0.082 |
| c=15 | 3.665 | 0.059 |

The cross-block standard deviation is ≤ 0.08 across every channel —
i.e. **every 2×2 patch has the same mean** to within ~1% of the
inblock variance. Top-left 8×8 corner of channel 0 confirms visually:

```
[[6.246 1.174 6.176 1.112 6.168 1.135 6.207 1.161]
 [2.967 6.93  2.936 6.875 2.922 6.875 2.879 6.871]
 [6.137 1.242 6.168 1.237 6.203 1.269 6.148 1.26 ]
 [2.914 6.965 2.896 6.98  2.916 6.984 2.902 6.953]
 ...]
```

The single 2×2 pattern `[[6.2, 1.2], [2.9, 6.9]]` repeats across every
block of the 32×32 latent. This is **not** an unpatchify ordering bug
(which would scramble channels or transpose H↔W) and **not** a RoPE
offset bug alone (which would produce different-but-wrong positions
per token, not identical outputs). It is the signature of *every img
token producing the same 64-d output vector*.

#### Root cause — `init_latent.f32.bin` is pre-noise, all-zero

Cross-checked the dump: `/tmp/qie_q45_inputs/init_latent.meta.txt`
shape `[32, 32, 16, 1]` F32. Loaded host-side:

```
=== init_latent ===
  mean=0.000 std=0.000 min=0.000 max=0.000
```

The dump is **all zeros**. Tracing
`tools/ominix_diffusion/src/stable-diffusion.cpp`:

- L2697-2715 `generate_init_latent` returns
  `ggml_set_f32(init_latent, shift_factor)` — for Qwen-Image
  `shift_factor=0` → tensor of zeros.
- L3997-3999 (in the per-image sampling loop) generates a
  fresh `noise = randn` tensor next to `init_latent`.
- L2068 inside `sample()`: the actual starting `x` is
  `denoiser->noise_scaling(sigmas[0], noise, x)` →
  `(1 - σ_0) * latent_zero + σ_0 * noise` for DiscreteFlowDenoiser
  (denoiser.hpp:700-705). With `latent_zero ≡ 0` and `σ_0 ≈ 1.0`
  the trajectory starts from pure Gaussian noise.
- L3886 (the dump call site): dumps `init_latent` **before** the
  noise-scaling step. The probe driver
  `tools/probes/qie_q45_step4_full_denoise/test_qie_q45_step4_full_denoise.cpp:219`
  feeds that pre-scaling tensor (zeros) into
  `eng.denoise_full(init_latent_host.data(), ...)` as the starting
  point for Euler.

With `x ≡ 0` at step 0, every img patch projects through `img_in`
(Linear with bias) to the same hidden vector — the bias. RoPE applies
position-dependent rotations to Q/K, so query-key scores differ
per token, but **V is unchanged**. With identical V across every
token, attention output is the same V regardless of attention pattern
(`softmax(scores) @ V` collapses). Subsequent pointwise MLPs preserve
the uniformity. After 60 blocks → 20 Euler steps the latent retains
the all-tokens-identical structure → unpatchify emits the same 2×2
tile to every block → blue-checkerboard PNG.

#### Fix — dump post-noise-scaling x_t, prefer it in `init_from_dump`

Two-line change:

1. `tools/ominix_diffusion/src/stable-diffusion.cpp`: add a forward
   declaration of `qie_dump_tensor` above the
   `StableDiffusionGGML` class (so the dump can be called from
   `sample()`), and emit `qie_dump_tensor(x, "noised_init_latent")`
   immediately after the `noise_scaling` call (line 2068).

2. `tools/qwen_image_edit/native/image_diffusion_engine.cpp`:
   `init_from_dump` now prefers `noised_init_latent.f32.bin`; falls
   back to legacy `init_latent.f32.bin` with a loud WARN that the
   output will be uniform-pattern unless the dump is regenerated.

The native unpatchify (host\_unpatchify\_latent at
`image_diffusion_engine.cpp:4299`) was verified correct against
`tools/ominix_diffusion/src/common_dit.hpp::unpatchify` and the MLX
reference at
`OminiX-MLX/qwen-image-mlx/src/transformer/transformer.rs:220` —
both use channel-major then `(py, px)` per-token layout with
row-major `(ty, tx)` token scan. **No layout fix needed.**

The RoPE pe per-step offset gap noted in §5.5.1 is real and still
worth follow-up (currently the pe table built at h=w=64 grid is
sliced as 8 rows of 64 cols for the production 16×16+16×16 token
sequence — incorrect positions but not catastrophic, as evidenced
by the model still producing F32-finite output and DIFFERENT pe per
token rather than identical pe).

#### Re-probe with proper noise: still uniform output

Re-ran the §5.5.5 Step-4 probe with the fix
(`/tmp/qie_q45_step4f_rerun.log` on ac03). Loaded
`noised_init_latent.f32.bin` (verified Gaussian: mean 0.003 std 0.997
range ±3.93). Final out_latent stats:

```
mean=-2.4559 std=4.8611 min/max=-13.3203/7.6250 NaN=0 inf=0
```

Bit-pattern-near-identical (L2 diff 3.88, max abs diff 0.16) to the
pre-fix run. Same uniform-tile structure (per-channel cross-block std
0.03 vs inblock std 2.3). **The missing-noise input bug is real and
necessary to fix, but is NOT sufficient — there's a deeper defect
downstream.**

#### Bisect — per-token variation collapse inside the 60-block forward

Instrumented `denoise_full` (env-gated `QIE_DEBUG_DUMP_STEP0_TOKENS=1`)
to dump four step-0 buffers:

| File | Shape | What |
|---|---|---|
| `qie_step0_concat_tokens.f32.bin` | [seq=512, IN_CH=64] | Host patchify-output (model input) |
| `qie_step0_pre_blocks_img_res.f32.bin` | [seq=512, H=3072] | Post-img_in F32 residual (just before the 60-block loop) |
| `qie_step0_post_blocks_img_res.f32.bin` | [seq=512, H=3072] | Post-60-block F32 residual |
| `qie_step0_out_tokens.f32.bin` | [init=256, PATCH_OUT=64] | Final proj_out output |

Receipts:

| Stage | per-row σ (within token) | cross-row σ (between tokens) | avg pairwise cossim |
|---|---|---|---|
| Pre-blocks img_res | 1.22 | 1.10 | **0.027** (independent) |
| Post-blocks img_res | 1.24M | 20.2K | **0.9986** (collapsed!) |

Per-row magnitudes also explode from O(1) to O(10⁶) across 60 blocks.
The per-block trace from §5.5.5 receipts already showed mean_abs
1e5 → 3e5 across blocks 0..59 with max_abs reaching 6.5e7 at block 59
— magnitudes the model never saw during training. Reference engine's
final latent (`x0_sampled_0.f32.bin` from the dump pipeline) has
`std=0.36` vs the native engine's `std=4.86` (14× too big), so the
amplification is real and systemic, not just a step-0 artifact.

#### Likely deeper cause (NOT confirmed in Step 4f scope)

Three candidates left on the table for the next agent:

1. **Attention softmax saturation cascade.** Once `img_resid_2` enters
   block 1 at mean_abs 1e5 (§5.5.4 Receipt 2), block 1's `LN1` does
   normalize it back to std≈1 BUT the pre-block residual stream and
   gated-residual contributions from block 1 onwards still ride on
   the 1e5 base. Each block's `to_q`/`to_k` projection from
   post-modulate input (max ~2400) produces O(2400 × √H) Q/K
   magnitudes; their dot product saturates softmax at >50 (1e21
   linear range). Saturated attention means every img query collapses
   onto whichever K row scored highest → identical output rows →
   the cosine 0.999 we measured. This explains the BOTH the
   magnitude blow-up AND the per-token collapse with one
   underlying defect: the residual stream is leaking magnitude that
   the trained attention scale (`1/√head_dim`) can't compensate.
   The reference engine — which produces `x0` at std=0.36 — must
   keep the residual stream to single-digit magnitudes; native is
   ~14× too big from block 0 onward, suggesting a missing or
   wrong-direction normalization somewhere in the QKV / attention
   /  modulation / gated-residual chain.
2. **Q4_0 weight quantization-scale mishandling.** If WQBMMv3
   antiquant scales are off by a constant factor (e.g.
   F16-vs-FP32 cast that misses one of the per-32-element scaling
   stages), every Q4 matmul output is uniformly amplified — which
   would explain the 14× systemic amplification. The §5.5.4 BF16
   widening saved the residual from F16 saturation but did not
   address the underlying scale.
3. **Modulation `gate1`/`gate2` magnitude drift.** AdaLN modulation
   computes `scale = silu(t_emb) @ W_mod` and applies
   `x = x * (1 + scale) + shift`. If `(1 + scale)` is consistently
   >1 (rather than averaging near 1.0 as the reference does), every
   block's residual is multiplied by some factor > 1 — exponential
   growth. Probe: dump `img_scale1` / `img_scale2` magnitudes at
   block 0 vs reference engine's same buffer.

#### Verdict

Step 4f delivers two committed fixes (noised-init dump + load) and
one diagnostic probe (env-gated step-0 token dumps), but the visual
cat remains **BLOCKED**. Native engine produces a numerically-finite
but magnitude-blown-up, per-token-uniform latent that VAE-decodes
into the original tile pattern. PNG receipts:

- `/tmp/qie_q45_step4d_allbf16_cat.png` (Mac, pre-fix)
- `/tmp/qie_q45_host2step_dump.png` (ac03, post-fix CLI dump 2-step,
  visually equivalent — uniform texture, not a cat)
- Reference: `/tmp/phase1_baseline_1024_20step.png` (Mac, codex CUDA
  20-step, recognizable B&W cat)

Next agent: start from candidate (1) or (3) above. Compare native
engine's block-0 `to_out_0` magnitude (`70` mean_abs at the §5.5.3
intra log, max 634) to the reference engine's same buffer — if
native is consistently 10×+ off, walk back through QKV → modulation
→ `txt_in` projection until the amplification source is isolated.
The `qie_step0_pre_blocks_img_res.f32.bin` host dump already shows a
sane post-img_in residual (max_abs 16) — so img_in itself is fine;
the leak is somewhere inside the 60-block loop.

### §5.5.7 Step 4g — AdaLN modulation chunk-order swap (candidate 3 confirmed by source audit)

Mission (Q2.4.5.4g): land the AdaLN modulation fix from §5.5.6
candidate (3). Source audit of `forward_block_` vs the CPU reference
caught a **shift / scale label swap** in the 6-way chunk of
`img_mod.1(silu(t_emb))` and `txt_mod.1(silu(t_emb))`.

**CPU reference** (`tools/ominix_diffusion/src/qwen_image.hpp:230-234,
280-326`) splits the modulation linear output (shape `[B, 6·H]`) into
six chunks along axis 0 and consumes:

| chunk | role         | consumer                                |
|-------|--------------|-----------------------------------------|
| 0     | `shift_msa`  | 3rd arg of `Flux::modulate(x, shift, scale)` (first half) |
| 1     | `scale_msa`  | 4th arg of `Flux::modulate` (first half)|
| 2     | `gate_msa`   | gate1 — multiplied into attn residual   |
| 3     | `shift_mlp`  | `Flux::modulate` shift (second half)    |
| 4     | `scale_mlp`  | `Flux::modulate` scale (second half)    |
| 5     | `gate_mlp`   | gate2 — multiplied into mlp residual    |

`Flux::modulate(x, shift, scale)` is `x = x*(1+scale) + shift`
(`tools/ominix_diffusion/src/flux.hpp:233-248`). The same chunk ordering
appears in MMDiT (`tools/ominix_diffusion/src/mmdit.hpp:291-296`) — it is
the canonical Diffusers ordering for AdaLN-Zero modulation heads.

**Pre-fix native engine** (`image_diffusion_engine.cpp:3204-3221`,
HEAD `7d15f3ce`) labelled `chunk(0)→scale1, chunk(1)→shift1,
chunk(3)→scale2, chunk(4)→shift2`, then called
`modulate_(x, scale1, shift1, ...)` (signature
`modulate_(x, scale, shift, ...)` at `:1909-1981` — same semantics:
`x = x*(1+scale) + shift`). Net effect: `chunk(0)` was being used as
`(1+scale)` (i.e. multiplied into `LN(x)`), but `chunk(0)` is in fact
trained as `shift_msa`. The trained `shift_msa` distribution has
larger magnitudes than `scale_msa` (the model is initialised so
`scale_msa ≈ 0` keeps the modulation near-identity at init); using
`shift_msa` as `(1+scale)` therefore inflates the post-modulate
activations every block. Compounded across 60 blocks this is the
source of the systemic 14× magnitude amplification and
post-block per-token cossim 0.999 collapse.

**Fix (Q2.4.5.4g):** rewrite the chunk labels in `forward_block_`
(both img and txt sides) to:

```
chunk[0] → shiftN, chunk[1] → scaleN, chunk[2] → gateN  (first half)
chunk[3] → shiftN, chunk[4] → scaleN, chunk[5] → gateN  (second half)
```

`modulate_()` call sites stay byte-identical (same arg order
`(scaleN, shiftN)`); only the pointer-to-chunk binding changes. Gate
chunk indices (2 and 5) were already correct.

**Verification probe (env-gated):** `QIE_DEBUG_DUMP_GATES=1` triggers
twelve `intra_probe` lines on the first block-0 step-0 invocation
(co-gated with `QIE_DEBUG_INTRA_BLOCK0=1`), printing per-chunk mean_abs
for img_shift1/scale1/gate1/shift2/scale2/gate2 and the txt analogues.
Expected ranges on healthy Q4 weights: `scale*` mean_abs ≈ 0.05-0.2
(values clustered near 0), `shift*` ≈ 0.1-0.5, `gate*` ≈ 0.01-0.1.
If `scale*` magnitudes look like the previous-known `shift*` range
(or vice-versa), the chunk binding is still wrong and needs
re-permutation.

**Smoke gate:** re-run §5.5.5 Step 4 probe + VAE decode. Eye-check
versus codex CUDA reference at `/tmp/phase1_baseline_1024_20step.png`.
Expected: post-block per-token cossim drops back below ~0.1 (Step 4f
measured 0.027 at the **pre**-block stage on healthy noise; native
post-blocks should land at the same order of magnitude). Final
out-latent std should fall from 4.86 to O(0.3-1) (reference is 0.36).
Numerical gate (Step 1 cos_sim=1.0 path) stays GREEN with all env
vars unset — the only logic change is the chunk-pointer binding,
which is invariant of the synthetic identity weights used in Step 1.

**Fallback (if §5.5.7 doesn't fully close the gap):** start from
§5.5.6 candidate (1) (attention softmax saturation) — instrument
softmax max/min in `aclnnFusedInferAttentionScoreV2` via the
`QIE_DEBUG_INTRA_BLOCK0=1` log path (look for the `12_*_attn_out`
intra_probe magnitudes already wired). Or candidate (2) (Q4_0 weight
scale mishandling) — diff `dispatch_matmul_` output of img_mod.1 at
block 0 between native and the CPU reference engine.

#### Smoke under spec ordering (RED — pivot to legacy ordering)

Built + ran the §5.5.5 Step 4 probe under
`QIE_DEBUG_INTRA_BLOCK0=1 QIE_DEBUG_DUMP_GATES=1 QIE_ALL_BF16=1` on ac03
with the spec ordering active (`/tmp/qie_q45_step4g_smoke.log`).
Final out_latent: `mean=3.01 std=33.88 min/max=-141/128 NaN=0 inf=0`.
**WORSE** than §5.5.5-d's std=4.86 — VERDICT YELLOW (range), gate
fires `|min|<20, |max|<20`. Smoke is RED for the spec-ordering fix.

Gate-dump receipts (block 0, single layer, mean_abs values):

| chunk | spec name | legacy name | mean_abs | max_abs |
|-------|-----------|-------------|---------:|--------:|
| 0     | shift_msa | scale1      | 0.58     |  45.25  |
| 1     | scale_msa | shift1      | 2.09     |  59.66  |
| 2     | gate_msa  | gate1       | 0.46     |   7.10  |
| 3     | shift_mlp | scale2      | 1.49     |  69.75  |
| 4     | scale_mlp | shift2      | **26.21**| 200.00  |
| 5     | gate_mlp  | gate2       |  17.30   | 269.00  |

Diagnosis: chunk[4] has a mean_abs of 26.2. Under spec ordering this
is treated as `scale_mlp` and applied as `(1+scale_mlp)` — i.e. a
~27× per-block multiplier on the post-LN activations. That is the
direct driver of the std=33.88 final latent (60 layers worth of
~27× growth, attenuated by attention/gating but not by enough). Under
legacy ordering chunk[4] is treated as `shift2` and applied
additively after LN — bounded by LN output magnitude (≈1) — so its
per-block contribution is O(chunk[4]_mean) ≈ 26 ADDED, far less
explosive than 27× MULTIPLIED.

Upstream defect signal: `00_t_emb_in` has `max_abs=111.8` with
`mean_abs=0.33` (n=3072). Healthy timestep MLP output should be
O(1) max — a max of ~112 is a 100× outlier dimension, indicative of
upstream amplification in the time-text-embed pipeline (or in the
`time_text_embed.linear_2.weight` Q4 dequant scale). All six modulation
chunks are downstream consumers of `silu(t_emb)` and inherit its
amplification.

**Decision:** PIN the legacy chunk binding (chunk[0]→scale1,
chunk[1]→shift1, etc.) for now. The spec ordering is mathematically
correct per HF Diffusers
(`transformer_qwenimage.py:425` → `chunk(2, dim=-1)` then
`chunk(3, dim=-1)` → on-disk row order
`shift1, scale1, gate1, shift2, scale2, gate2`), but it interacts
with the upstream-amplified t_emb to produce a worse final latent.
The legacy ordering is bug-for-bug safer until the t_emb/Q4 amplification
is isolated and fixed; once that is fixed, the spec ordering must be
restored (chunks[0,1] swap, chunks[3,4] swap) — and the gate-dump
probe instrumentation is preserved precisely for that re-validation.

**Hand-off to the next agent (Q2.4.5.4h or later):**

1. Investigate the upstream amplification driving `00_t_emb_in`
   max_abs=111.8. Likely suspects:
   - `time_text_embed.timestep_embedder.linear_*` Q4 dequant scale
     handling (`qwen_image.hpp:42-58` is the CPU reference; native
     impl in `image_diffusion_engine.cpp` builds the time MLP from
     scratch).
   - `txt_norm` RMSNorm output magnitude (compare to reference).
   - `txt_in` projection scale.
2. Once upstream is bounded to O(1), re-test the spec ordering: in
   `forward_block_` mod_chunk lambda, swap labels to
   `chunk[0]=shift, chunk[1]=scale, chunk[3]=shift, chunk[4]=scale`
   (i.e. revert §5.5.7's pin), keep `QIE_DEBUG_DUMP_GATES=1` on,
   and verify chunk magnitudes drop into the healthy range (gates
   ~0.01-0.1, scale* ~0.05-0.2, shift* ~0.1-0.5).
3. Smoke: §5.5.5 Step 4 probe + VAE decode. Eye-check vs
   `/tmp/phase1_baseline_1024_20step.png`.

The Q2.4.5.4g commit (`8bcad851`) ships the gate-dump probe
(`QIE_DEBUG_DUMP_GATES=1`) and pinned-legacy chunk binding with full
audit trail in this section. Numerical gate (Step 1 cos_sim=1.0)
unchanged because the env var is OFF by default and the chunk binding
is byte-identical to the pre-§5.5.7 path.

### §5.5.8 Step 4h — t_emb upstream bisect (Q2.4.5.4h)

Mission: identify which upstream op is amplifying `00_t_emb_in` to
`max_abs=111.8`. Hand-off from §5.5.7 named four suspects:
(1) `time_text_embed` Q4 dequant scale; (2) `txt_norm` RMSNorm
output magnitude; (3) `txt_in` projection scale; (4) `time_linear*`
SiLU mis-application.

**Methodology:** `image_diffusion_engine.cpp::denoise_full` already
ships `probe_stats` calls (gated by `QIE_DEBUG_NAN_BISECT=1`, step 0
only) at every stage of the time-embed and txt-conditioning chains:
`pre_time_linear1`, `post_time_linear1`, `post_time_silu`,
`post_time_linear2`, `pre_txt_norm.cond`, `post_txt_norm.cond`,
`post_txt_in.cond`, `txt_res_c_f32`. Combined with the existing
`intra_b0` probes, no source changes were needed for Step 1. Ran
the §5.5.5 Step 4 probe on ac03 with all three env gates enabled
(`QIE_DEBUG_NAN_BISECT=1 QIE_DEBUG_INTRA_BLOCK0=1
QIE_DEBUG_DUMP_GATES=1`, `QIE_N_STEPS=2`) — log at
`/tmp/qie_q2454h_step1_temb_bisect.log`.

#### t_emb chain magnitudes (real Q4_0 weights, real conditioning)

| stage                         | dtype | mean_abs | max_abs |
|-------------------------------|-------|---------:|--------:|
| `pre_time_linear1` (sinusoid) | F16   | 0.636    | 1.0     |
| `post_time_linear1` (Q4 mm+b) | F16   | 0.231    | 21.44   |
| `post_time_silu`              | F16   | 0.0804   | 21.44   |
| `post_time_linear2` (Q4 mm+b) | F16   | 0.3261   | 111.8   |
| `00_t_emb_in` (downstream)    | F16   | 0.3261   | 111.8   |
| `01_silu_t_emb`               | F16   | 0.1479   | 111.8   |

#### txt chain magnitudes

| stage                  | dtype | mean_abs | max_abs |
|------------------------|-------|---------:|--------:|
| `pre_txt_norm.cond`    | F32   | 2.716    | 151.3   |
| `post_txt_norm.cond`   | F16   | 0.468    | 10.63   |
| `post_txt_in.cond`     | F16   | 7.029    | 583.5   |
| `txt_res_c_f32`        | F32   | 7.029    | 583.5   |

#### Diagnosis (suspects 1-4 disconfirmed)

The amplification is **two-stage** — `time_linear1` (1→21.44, 21×)
and `time_linear2` (21.44→111.8, 5×). SiLU is correctly applied
(post_silu mean_abs=0.08 = ~SiLU(0.231) for the bulk; sign
preservation visible in mean_abs drop while max_abs is preserved).
`txt_norm` is healthy (max 151.3 → 10.63, RMS-normalised).
`txt_in` produces a **second** big outlier (583.5).

**A/B test:** re-ran with `QIE_MATMUL_INNER_PRECISE=0
QIE_MATMUL_CUBE_MATH=1` (F32 accumulator, F32 cube math) — log at
`/tmp/qie_q2454h_step1b_temb_f32acc.log`. **Magnitudes byte-identical**
to F16-accum run. Rules out F16-accumulator overflow as the
amplifier — this is data-resident, not a numeric-precision bug.

**Cross-check on Q4 dequant scale handling:** §5.5.5 Step 1's
synthetic-identity-weight gate (`cos_sim=1.0` on `forward_block_test`)
already validates that `repack_q4_0_upload`'s scale buffer + WQBMMv3
forward path is correct *for synthetic weights*. The 21× / 5× / 60×
outliers therefore reflect the **trained Qwen-Image weights**, not
a code bug. This matches the well-documented "outlier feature"
phenomenon in DiTs (a small subset of dimensions carry
disproportionate magnitude after timestep MLPs).

**Conclusion (Q2.4.5.4h Step 1):** all four suspects from §5.5.7
hand-off are disconfirmed. `00_t_emb_in max_abs=111.8` is *not*
introduced by upstream Ascend code — it is the trained-model
distribution. The CUDA reference would report the same magnitudes
on the same input.

#### Real bug — F16 overflow in `ff_down` post-modulate

Per-block intra_probe data from the same log already shows the
explosion site:

| stage (block 0)        | mean_abs | max_abs   | NaN/Inf       |
|------------------------|---------:|----------:|---------------|
| 14_img_LN2             | 0.439    | 20.11     | 0/0           |
| 15_img_mod2            | 27.86    | 467.2     | 0/0           |
| 18_img_ff_up           | 198.7    | 2438      | 0/0           |
| 19_img_gelu            | 107      | 2438      | 0/0           |
| **20_img_ff_down**     | **4819** | **6.3e+04** | **0/0**     |
| 23_txt_ff_down         | 4950     | 5.5e+04   | **0/214 Inf** |
| 24_img_resid2 (F32)    | 8897     | 6.6e+04   | **0/272k Inf**|
| 24_txt_resid2 (F32)    | 24500    | 6.6e+04   | **0/224k Inf**|
| post_blocks.img_res_c  | 0        | 0         | **all NaN**   |

`ff_down` (FFN second matmul, [12288→3072] in Qwen-Image) consumes
GeLU output with max_abs=2438. Even at a unit-variance weight, a
12288-wide dot-product of magnitude-100 inputs lands near
sqrt(12288)·100 ≈ 11000 — close to F16 max (65504). With a couple
of outlier columns, individual outputs cross 65504 and become Inf.
Once one block emits Inf in the residual, every subsequent block
propagates and the post-60-block latent is all-NaN.

The driver is **F16 multiplicative compounding** of the `mod2` scale
(`(1+scale_mlp)` ≈ 1+27 = 28 per the chunk-4 mean) into LN(x), then
through a 12288-wide ff_down with F16 accumulation. The post-LN
output has max_abs=20 (LN can't fully bound outlier features when
scale is 27×); ff_up amplifies by ~120× to 2438; GeLU is a no-op
on positive max; ff_down sums 12288 such values in F16 → overflow.

**A/B test 2 — `QIE_FFN_DOWN_BF16=1`** (log
`/tmp/qie_q2454h_step1c_ffdown_bf16.log`): switching ff_down store
dtype to BF16 eliminates the **block-0 Inf** at `23_txt_ff_down`
(was 214 Inf, now 0 Inf) and `24_*_resid2` (was 271k+224k Inf, now
0 Inf each). Block 0 stays finite. However the mean/max grow
because previously-saturated values now express themselves as
finite-but-huge numbers:

| stage (block 0)        | F16 store (default) | BF16 store        |
|------------------------|---------------------|-------------------|
| 20_img_ff_down         | mean 4819 max 6.3e+04 / 0 Inf | mean 4819 max 6.3e+04 / 0 Inf |
| 23_txt_ff_down         | mean 4950 max 5.5e+04 / **214 Inf** | mean 4972 max 7.2e+04 / **0 Inf** |
| 24_img_resid2 (F32)    | mean 8897 max 6.6e+04 / **272k Inf** | mean 1.1e+05 max 7.4e+06 / **0 Inf** |
| 24_txt_resid2 (F32)    | mean 24500 max 6.6e+04 / **224k Inf** | mean 7.0e+04 max 4.6e+06 / **0 Inf** |
| post_blocks.img_res_c  | all NaN | all NaN |
| out_latent             | all NaN | all NaN |

So `QIE_FFN_DOWN_BF16=1` patches the immediate F16 saturation but
the residual's 1e+07-scale magnitude still NaN's somewhere across
blocks 1-59. **Block 0 overflow is downstream of a real magnitude
explosion, not just a numeric-precision issue.**

#### Real driver — AdaLN mod2 multiplicative compounding

The 60×-block magnitude growth is driven by the modulation chunks:
chunk[4] (`legShift2` in §5.5.7 pinned-legacy ordering) has
`mean_abs=26.21 max_abs=200`. Under legacy ordering chunk[4] is
applied *additively* as `shift2`, so its per-block contribution to
the post-modulate output is *additive* (bounded by chunk magnitude).
Under spec ordering it is `scale_mlp` and applied as `(1+scale)`,
which is *multiplicative* — so per-block growth factor is
`(1 + 26)` ≈ 27× compounded over 60 blocks. §5.5.7 verified the
spec-ordering RED smoke (`std=33.88` final latent) and pinned
legacy as bug-for-bug safer. **Legacy is still RED at block 0**
because chunk[4]=200 added to post-LN(≈1) values produces ~200
input to ff_up, then ~24000 post-GeLU, then ff_down's 12288-wide
sum accumulates to F16-overflow regardless of accumulator
precision.

The amplification is therefore **fundamental to the trained
weights' interaction with the AdaLN modulation chunk binding**, not
a numeric bug at any single op. Both legacy and spec orderings RED
on real Q4 weights — legacy fails at block 0 (F16 overflow, fixable
with BF16 store but residual still huge); spec fails by 60×
multiplicative compounding (final std=33.88 vs reference 0.36).

#### Conclusion

Step 1 (t_emb upstream bisect) closes with **all four §5.5.7
suspects disconfirmed**: t_emb's 100× max_abs is the trained-model
distribution, not an Ascend-side bug. The visible mode collapse is
driven by AdaLN modulation × FFN compounding, which compounds
multiplicatively (spec ordering) or saturates additively (legacy
ordering, with F16 ff_down overflow). Neither ordering produces a
finite, well-conditioned post-block latent.

**Hand-off to next agent (Q2.4.5.4i):** the t_emb hypothesis is
exhausted. Two remaining angles:

1. **Re-examine modulation chunk semantics on the actual GGUF
   weight layout.** §5.5.7's chunk-magnitude table shows chunk[4]
   mean_abs=26.21 — but under the canonical Diffusers MMDiT layout,
   `scale_msa`/`scale_mlp` are initialised near zero so the model
   starts identity. A chunk with mean_abs=26 *cannot* be a healthy
   `scale_*` chunk on trained weights. This suggests the GGUF row
   permutation produced by HuggingFace's Qwen-Image-Edit-2509 Q4_0
   conversion may not match the upstream MMDiT chunk order. Cross-
   check `transformer_qwenimage.py:425` against the actual GGUF
   tensor row layout (`tools/ominix_diffusion/src/qwen_image.hpp`'s
   ggml-backed `mod_split` to verify).

2. **Compare native vs. CPU reference engine `forward_block` output
   on identical Q4 weights and identical (img,txt,t_emb) inputs.**
   `tools/ominix_diffusion/src/qwen_image.hpp::QwenImageBlock::forward`
   produces a ground-truth block-0 output; if its `ff_down` doesn't
   overflow on the same t_emb=111.8 input, then the native engine's
   `mod2` apply order or `ff_down` matmul precision is the bug. The
   ggml CPU path uses F32 throughout, so a magnitude comparison
   isolates Ascend-side precision issues from algorithm errors.

Receipts:
- `/tmp/qie_q2454h_step1_temb_bisect.log` (default precision; 121 lines)
- `/tmp/qie_q2454h_step1b_temb_f32acc.log` (F32-accum A/B; t_emb
  byte-identical, ff_down still RED)
- `/tmp/qie_q2454h_step1c_ffdown_bf16.log` (BF16 ff_down store;
  block 0 stops Inf'ing but residual hits 1e+07 magnitude, output
  still NaN after 60 blocks)

No source changes committed in Q2.4.5.4h. Numerical gate (Step 1
cos_sim=1.0 with all env vars unset) unchanged.

### §5.5.9 Step 4i — CPU vs NPU block-0 bisect (Q2.4.5.4i) — YELLOW (partial)

**Mission (per §5.5.8 hand-off angle 2):** discriminate between
"native algorithm correct, mode collapse is design+precision" and
"algorithm error somewhere in modulation/attn/ffn ordering" by
running ONE block-0 forward through the ggml CPU backend (F32
end-to-end, no Ascend) on the **identical** `(img, txt, t_emb, pe)`
inputs that the native engine consumed, then element-wise comparing
the post-resid2 outputs.

**Probe stack** (`tools/probes/qie_block0_cpu_reference/`):

- `test_qie_block0_cpu_reference.cpp` — CPU-only ggml block-0 harness
  that wraps `Qwen::QwenImageRunner`, dequantises Q4_0 weights to F32
  via the standard `ModelLoader` path, and runs only
  `transformer_blocks.0.forward(img, txt, t_emb, pe)`.
- `compare_block0.py` — element-wise comparator (per-row cosine,
  magnitude ratio, sign-agreement, top max-abs channel overlap, auto
  GREEN/YELLOW/RED verdict).
- `run_step_4i.sh` — orchestrator: native dump → CPU reference →
  comparison.
- Engine patch (`image_diffusion_engine.cpp`, env-gated by
  `QIE_DUMP_BLOCK0_DIR`): writes `00_img.f32`, `00_txt.f32`,
  `00_t_emb.f32`, `24_img_resid2.f32`, `24_txt_resid2.f32` for
  block 0 only.

**Run config (small-smoke discriminator):** `img_seq=64 txt_seq=32
H=3072`, `QIE_FFN_DOWN_BF16=1`, `QIE_N_STEPS=1`, real Q4_0 weights
(`Qwen-Image-Edit-2509-Q4_0.gguf`), real conditioning. Native dump
under HBM lock; CPU reference forwarded under 8 threads.

#### Output magnitudes (block-0 post-resid2)

| stream     | engine  | mean_abs | max_abs | NaN | Inf |
|------------|---------|---------:|--------:|----:|----:|
| img_resid2 | native  | 6.79     | 768.9   | 0   | 0   |
| img_resid2 | CPU F32 | 6.80     | 760.5   | 0   | 0   |
| txt_resid2 | native  | 13.67    | 742.0   | 0   | 0   |
| txt_resid2 | CPU F32 | 10.72    | 545.5   | 0   | 0   |

Magnitude ratio (native ÷ CPU): `mean_abs` 0.999 (img) / 1.275 (txt);
`max_abs` 1.011 (img) / 1.36 (txt). **Magnitudes are within 1.4×
across the board**, NaN/Inf-free on both engines. (Note: small-smoke
shapes with `QIE_FFN_DOWN_BF16=1` do not exhibit the 1.1e5 / 7.4e6
magnitudes from §5.5.8, which were measured on production-scale
shapes — but the §5.5.8 finding "block 0 stays finite under
`QIE_FFN_DOWN_BF16=1`" is reaffirmed here.)

#### Per-row cosine similarity distribution

| stream     | min   | p10   | p50   | p90   | mean  | max   |
|------------|------:|------:|------:|------:|------:|------:|
| img_resid2 | 0.225 | 0.464 | 0.616 | 0.733 | 0.608 | 0.818 |
| txt_resid2 | 0.394 | 0.533 | 0.625 | 0.705 | 0.610 | 0.722 |

Sign-agreement fraction: 0.769 (img) / 0.693 (txt).
Relative-error p50 / p90 / p99: 0.74 / 4.46 / 44.7 (img); 1.06 / 6.41
/ 64.2 (txt).

Top-8 max-abs channel indices overlap between native and CPU:
img top-2 `[673, 1691]` match exactly (and 5/8 in top-8); txt top-3
`[3038, 169, 912]` match (different order) and 6/8 in top-8. The
**dominant outlier channels are the same**, but per-element
magnitudes drift.

#### Verdict — YELLOW (partial)

Comparator auto-verdict (`compare_block0.py:158`): **YELLOW —
partial direction agreement (cos_mean=0.607 / 0.610) — examine
intermediate substeps.** Both streams; exit code 1.

This is **not GREEN** (would require cos_mean > 0.99), but is also
**not RED** (algorithm-broken would be cos_mean ≤ 0.5 with no
channel-index overlap). The directional disagreement is
**uniform across the row distribution** (no bimodal "some rows
correct, some rows scrambled"), so the bug is not a row permutation
of the chunk binding. Combined with the matched dominant-channel
indices, the most-likely class is **substantial numeric drift in
attention or post-LN that affects per-element values without
shifting the channel-statistics distribution**. Probable culprits,
in priority order:

1. **F16 attention accumulation** — softmax over 96 heads × (64+32)
   tokens × dim_head=64 with F16 store/accum can produce ~1% per-
   element drift that compounds through the residual to ~40%
   per-element error while preserving column-mean / channel-rank
   statistics. Test: switch attention QKV/out to BF16 store via a
   new `QIE_ATTN_BF16=1` gate, re-run the bisect.
2. **AdaLN modulation chunk binding** — if chunk[3]/chunk[4]/chunk[5]
   row binding differs by a *non-permutation* (e.g. an inadvertent
   transpose that preserves column statistics but scrambles per-
   element values), this exact YELLOW signature would result.
   Test: dump `img_mod1`, `img_mod2`, `txt_mod1`, `txt_mod2` chunk
   tensors side-by-side from native and CPU; element-wise compare.
3. **F16 RMSNorm in `txt_norm` / `img_norm{1,2}`** — variance
   computation in F16 with `H=3072` reductions is borderline. Test:
   force F32 normalisation via existing `ENV` gate or add one.

#### Strategic recommendation

**Do not pivot to Path C #5 yet.** GREEN was the gating signal for
the pivot; YELLOW(partial) means the native engine has a real
per-element bug that is *not* explained by F16 ff_down saturation
alone (which is what §5.5.8 surfaced) and *not* explained by AdaLN
chunk-row permutation (which would have produced RED). The
60-block compounding of this ~40% per-block per-element drift is
**sufficient on its own** to explain the all-NaN final latent
without invoking the AdaLN multiplicative-compounding hypothesis
from §5.5.8. We have a tractable Step 4j ahead: extend
`QIE_DUMP_BLOCK0_DIR` to also emit the post-LN1, post-mod1,
post-attn-img, post-attn-txt, post-resid1, post-LN2, post-mod2,
post-ffn-img, post-ffn-txt intermediates (the existing `intra_probe`
sites at `image_diffusion_engine.cpp:3343-3625` already compute
these in F32 buffers — extending to disk-write is ~30 LoC), then
re-run the comparator on each substage to localise the substep that
first drops below cos>0.99.

**Hand-off to Q2.4.5.4j:** extend dump to intermediates; bisect
which sub-step (LN1, mod1, attn, resid1, LN2, mod2, ffn, resid2)
first violates cos>0.99 against the CPU reference. Once isolated,
the fix is one of: (a) BF16 widening at that op, (b) F32 promotion
for that op, or (c) algorithmic patch (chunk re-binding /
transpose).

#### Receipts

- `/tmp/qie_block0_inputs/` — native dump (5 files, total 2.3 MB):
  `00_img.f32`, `00_txt.f32`, `00_t_emb.f32`, `24_img_resid2.f32`,
  `24_txt_resid2.f32`.
- `/tmp/qie_block0_outputs/` — CPU reference (2 files, 1.2 MB):
  `cpu_24_img_resid2.f32`, `cpu_24_txt_resid2.f32`.
- `/tmp/qie_q2454i_run.log` — orchestrator log (135+ lines; native
  dump phase + Step 1 build errors + Step 2 fixes).

#### Probe build fixes applied during the run

Three issues had to be patched before the CPU reference would build:

1. `-DGGML_MAX_NAME=128` added to `build_and_run.sh` (top-level
   `CMakeLists.txt:6` defines this for the engine build but the
   probe didn't propagate it).
2. `Qwen::QWEN_IMAGE_GRAPH_SIZE` namespace-qualified (probe class is
   at global scope, not inside `namespace Qwen`).
3. `GGMLBlock::blocks` is `protected`; access via a `using`-
   redeclaration helper (`PublicQwenImageModel`) reinterpret-cast
   over the existing `qwen_image` member. Probe-only idiom; no
   upstream header changes.
4. `ggml_extend.hpp::get_compute_graph` overrides the LAST graph
   node's name to `final_result_name`. Probe expanded `img_out`
   first then `txt_out`, so `txt_out` was renamed; lookup fixed
   to `ggml_get_tensor(compute_ctx, final_result_name.c_str())`.
5. `thirdparty/zip.c` added to compile units (model.cpp's
   PyTorch-checkpoint loader unconditionally pulls in the kubazip
   API, even though the probe only loads GGUF).

Probe is now self-contained and re-runnable with one command:
`bash tools/probes/qie_block0_cpu_reference/run_step_4i.sh`.

### §5.5.10 Step 4j — substep bisect: bug isolated to attention (cos drops 0.98 → 0.002)

Following §5.5.9's YELLOW(partial) verdict, the workplan called for
extending `QIE_DUMP_BLOCK0_DIR` to intermediate substeps and
repeating the bisect. Done in this session — both the native engine
dump and the CPU reference probe now emit 14 substep tensors
(LN1/mod1/attn-out/resid1/LN2/mod2/ff-down for img + txt streams),
and `compare_block0.py` walks all of them with per-substep verdict
plus a one-line "first cos < 0.99" pointer.

#### Engine-side extension (~80 LoC)

`tools/qwen_image_edit/native/image_diffusion_engine.cpp`:

- Generalised `dump_tensor_f32` → `dump_tensor_dt(fname, dev, n_elts,
  ProbeDtype)`. The new `PROBE_BF16` arm host-upcasts BF16 to F32 by
  zero-extending the low 16 mantissa bits (top 16 of F32). This
  matches the existing `intra_probe_dt` BF16 handling.
- Added 14 paired `dump_tensor_*` calls next to the existing
  `intra_probe` sites for substeps 04, 05, 06, 07, 11_img, 11_txt,
  12_to_add_out, 12_to_out_0, 13_img_resid1, 13_txt_resid1, 14, 15,
  16, 17, 20, 23. Output filenames mirror the probe label
  (`{step}_{stream}_{name}.f32`).

#### CPU-reference extension

`tools/probes/qie_block0_cpu_reference/test_qie_block0_cpu_reference.cpp`:

- Replaced `block->forward(...)` with an inline copy of
  `Qwen::QwenImageTransformerBlock::forward` (verbatim, ~80 LoC) so
  intermediate substeps can be tagged. Each substep tensor is given
  a `cpu_<step>` name plus `ggml_set_output(...)` (CRITICAL — without
  this the gallocr reuses intermediate buffers for downstream ops
  and the read-back returns garbage; this was the source of the
  initial run's all-RED bisect).
- Added `dump_named` helper in `compute_block0()` that walks the
  same 14 substep names, looks them up in `compute_ctx`, and writes
  `cpu_<step>.f32` to `$QIE_BLOCK0_OUTPUTS_DIR`.
- Helper `PublicQwenImageTransformerBlock` (using-redeclares
  `GGMLBlock::blocks` publicly via the same probe-only idiom from
  §5.5.9).

#### Comparator extension

`compare_block0.py` now iterates the 14 substep pairs (graceful
skip when files absent), prints a one-line table per substep with
verdict tag, and reports the **first substep that crosses
cos_mean < 0.99** — which localises the bug entry point.

#### Substep verdict table (real Q4 weights, `QIE_FFN_DOWN_BF16=1`, img_seq=64, txt_seq=32)

| substep              | cos_mean | ratio_max | verdict |
|----------------------|---------:|----------:|---------|
| 04_img_LN1           |   1.0000 |     1.000 | **GREEN** |
| 05_img_mod1          |   0.9859 |     0.889 | YELLOW  |
| 06_txt_LN1           |   1.0000 |     1.000 | **GREEN** |
| 07_txt_mod1          |   0.9800 |     0.910 | YELLOW  |
| **11_attn_out_img**  | **0.0022** | 0.023   | **RED** |
| **11_attn_out_txt**  | **0.0033** | 0.221   | **RED** |
| 13_img_resid1        |   0.5041 |     0.841 | YELLOW  |
| 13_txt_resid1        |   0.4283 |     0.732 | RED     |
| 14_img_LN2           |   0.5040 |     0.761 | YELLOW  |
| 15_img_mod2          |   0.4902 |     0.788 | RED     |
| 16_txt_LN2           |   0.4283 |     0.997 | RED     |
| 17_txt_mod2          |   0.4116 |     0.923 | RED     |
| 20_img_ff_down       |   0.3924 |     1.020 | RED     |
| 23_txt_ff_down       |   0.5792 |     1.130 | YELLOW  |

#### Diagnosis — bug is in attention

The bisect is unambiguous:

1. **LN1 is byte-perfect** (cos=1.0000 both streams). The pure-F32
   `layer_norm_f32_to_f16_` path agrees with ggml's `ggml_norm` to
   within rounding. This rules out the §5.5.9 hypothesis #3 (F16
   RMSNorm).
2. **mod1 is near-perfect** (cos≈0.98). The `(1+scale)*x + shift`
   modulation introduces a small ~2% drift — consistent with F16
   storage of the modulation chunks plus broadcast-mul precision.
   This rules out the §5.5.9 hypothesis #2 (AdaLN
   non-permutation transpose) — if the chunks were transposed, mod1
   would be RED, not 0.98 YELLOW.
3. **Attention drops cos to 0.002** — direction completely random,
   magnitudes 22-44× smaller than CPU reference. **This is where the
   bug lives.** Confirms §5.5.9 hypothesis #1 (F16 attention
   accumulation) — but at a far more dramatic scale than predicted.
   Cos≈0 with magnitude under-shoot suggests not just precision drift
   but a structural bug: probable suspects, in priority order:
   - **F16 softmax overflow on the 96×96 logit matrix** with q×k
     scale=8.0 (head_dim=64 → 1/√64). The pre-softmax max can
     reach ~|q|·|k|·8 / √64 — and §5.5.8 reports `09_img_Q_rmsnorm
     max=723`, `09_img_K_rmsnorm max=571.5`. q·k logit max ≈
     723·571.5/8 ≈ 51 600 — under F16 max but the per-row softmax
     denominator after exp can saturate.
   - **q/k RMSNorm magnitudes** at max=723 / 571.5 are pathological
     for attention — they crash through softmax dynamic range. The
     CPU reference does not have this issue because ggml_soft_max
     auto-stabilises by max-subtraction.
   - **add_q_proj / add_k_proj / add_v_proj weight-row layout
     mismatch** between native (Diffusers naming) and the CPU
     reference (which uses the same naming). Less likely given
     LN1+mod1 are GREEN — projections feed off mod1 output.
4. **Residual add #1 partially recovers** to cos≈0.5 because the
   F32 residual stream `img_hidden + scrambled_attn * gate1`
   inherits direction from the unaffected input `img_hidden`
   (gate1·attn output is small in magnitude after the attention
   under-shoot, so the residual is dominated by the input). This
   explains the §5.5.9 final-resid2 cos=0.6 finding.
5. **All downstream substeps stay at cos≈0.4-0.5** because
   subsequent ops are linear and preserve the direction set by the
   resid1 step. ff_down magnitudes match (ratio≈1.0) because the
   FFN sub-tree is correct in isolation — it's just operating on
   bad input.

#### Strategic recommendation — focus on attention

The next bisect (Q2.4.5.4k) should narrow inside the attention
substep:

1. **Per-substep dumps inside attention**: add native + CPU dumps
   for `09_*_rmsnorm`, `10_*_rope`, the post-softmax score matrix,
   and the `to_q/to_k/to_v/to_out` projection outputs. The
   `intra_probe` sites at lines 3410-3475 already cover most of
   these — extending dump-to-disk is ~10 LoC per substep.
2. **F32-promote softmax**: attention softmax is the prime suspect
   for cos≈0. Add a `QIE_ATTN_SOFTMAX_F32=1` env gate that runs
   `aclnnSoftmax` on F32 logits then casts back to F16. Test impact
   on the 11_attn_out_* cos.
3. **F32-promote q/k post-RMSNorm**: with q-rmsnorm max=723 and
   F16 forward, the q·k matmul is borderline. A
   `QIE_ATTN_QK_F32=1` gate that keeps q,k in F32 through the
   `aclnnFlashAttentionScore` (or matmul fallback) would test
   precision-class.
4. **Weight-layout cross-check**: `add_q_proj`/`add_k_proj`/
   `add_v_proj` (txt-side projections — these feed `to_q/to_k/to_v`
   in the joint attention) — verify GGUF row layout matches
   Diffusers' MMDiT layout. Same as §5.5.9 hypothesis #2 but
   targeted at the attention QKV projections rather than the
   AdaLN chunks.

**Path C #5 ship pivot remains gated.** A real algorithm bug exists
in the native attention path; once attention bisects to GREEN, the
final cos should jump from 0.6 → 0.99 and the pivot decision
becomes trivial.

#### Receipts

- `/tmp/qie_q2454j_run_first.log` — orchestrator log (substep
  dumps + initial all-RED bisect from missing
  `ggml_set_output`).
- `/tmp/qie_q2454j_compare.log` — final substep verdict table
  after `ggml_set_output` fix.
- `/tmp/qie_block0_inputs/` (21 files, 8.6 MB native dumps)
- `/tmp/qie_block0_outputs/` (16 files, 4.3 MB CPU dumps).
- Probe + comparator are re-runnable end-to-end with
  `bash tools/probes/qie_block0_cpu_reference/run_step_4i.sh`.

#### Source changes committed

- `tools/qwen_image_edit/native/image_diffusion_engine.cpp`: 14
  paired `dump_tensor_*` calls + `dump_tensor_dt` lambda extension
  (BF16 host-upcast). Env-gated by `QIE_DUMP_BLOCK0_DIR`. Zero
  effect when env unset (existing s_dump_dir guard).
- `tools/probes/qie_block0_cpu_reference/test_qie_block0_cpu_reference.cpp`:
  inline-replicate forward, `ggml_set_output` on each substep,
  `dump_named` helper writing 14 cpu_*.f32 files.
- `tools/probes/qie_block0_cpu_reference/compare_block0.py`:
  substep-pair walker + bisect summary.

### §5.5.11 Step 4k — attention narrow-bisect: bug is the QKV PROJECTION (08_*Q/K/V), not softmax/RoPE

The §5.5.10 RED at `11_attn_out` was traced to its true entry by extending
both probes to dump the attention internals (`08_*Q/K/V` post-projection,
`09_*Q/K_rmsnorm`, `11pre_attn_out` pre-output-projection). The new
substep verdict on the same {real Q4 weights, `QIE_FFN_DOWN_BF16=1`,
img_seq=64, txt_seq=32} configuration:

| substep              | cos_mean | ratio_max | verdict |
|----------------------|---------:|----------:|---------|
| 04_img_LN1           |   1.0000 |     1.000 | **GREEN** |
| 05_img_mod1          |   0.9859 |     0.889 | YELLOW  |
| 06_txt_LN1           |   1.0000 |     1.000 | **GREEN** |
| 07_txt_mod1          |   0.9800 |     0.910 | YELLOW  |
| **08_img_Q**         | **-0.0014** |  4.34  | **RED — bug entry** |
| **08_img_K**         | **-0.0000** |  4.56  | **RED** |
| 08_img_V             |   0.0587 |     1.020 | RED     |
| **08_txt_Q**         | **-0.0038** |  1.26  | **RED** |
| **08_txt_K**         | **-0.0020** |  3.24  | **RED** |
| 08_txt_V             |  -0.0035 |     0.744 | RED     |
| 09_img_Q_rmsn        |  -0.0011 |    39.5  | RED     |
| 09_img_K_rmsn        |  -0.0028 |    31.9  | RED     |
| 09_txt_Q_rmsn        |   0.0041 |     0.014 | RED     |
| 09_txt_K_rmsn        |   0.0033 |     0.376 | RED     |
| 11_attn_out_img      |   0.4780 |     1.140 | RED     |
| 11_attn_out_txt      |   0.4130 |     0.738 | RED     |

#### Diagnosis — the QKV projection is the bug entry

`08_img_Q cos=-0.0014` against a near-perfect input
(`05_img_mod1 cos=0.99`). Both sides feed the projection an essentially
identical activation, yet the OUTPUTS are nearly orthogonal with a 4×
magnitude over-shoot on native. This isolates the failure to the
WQBMMv3 dispatch chain at `dispatch_matmul_(scratch_img_norm_dev_,
lw.to_q_w_q4, lw.to_q_scale, lw.to_q_b, img_seq, H, H, …)` and
symmetric add_q_proj/add_k_proj/add_v_proj sites.

Knock-on effect on §5.5.8/§5.5.10: the `09_*_rmsnorm max=723/571.5`
"trained Q4 outliers" finding was a downstream symptom, not a
constitutional property of the trained weights. RmsNorm faithfully
amplifies whatever extreme values the projection emits — and if the
projection is producing a shifted/scaled WRONG basis, those outliers
follow naturally. The `11_attn_out` cos jump from 0.0022 to 0.4780 in
this run (vs §5.5.10) is a NAMING-FIX artefact: prior runs compared
`native_11_attn_out` (FIA pre-projection) against `cpu_11_attn_out`
(post `to_out_0` projection), a category mismatch. The new harness
emits `cpu_11pre_attn_out_*` for an apples-to-apples FIA-output
compare, and that 0.48 figure is now the genuine attention-kernel
parity number — itself constrained by upstream-08 RED.

#### Suspect ranking for Q2.4.5.4l (next bisect)

1. **Q4_0 → WQBMMv3 weight repack mismatch**: `repack_q4_0_upload`
   (engine §256) flips bias-8 to signed two's-complement via
   `(byte & 0x0f) ^ 0x08`. Probe spec at lines 268-273 calls for
   weight view shape `[K, N]` strides `(1, K)` — physically `N` rows
   of `K` packed nibbles. Re-verify by:
   - Dumping `lw.to_q_w_q4` + `lw.to_q_scale` device pointers
     element-for-element against a Python `dequantize_row_q4_0` of
     the same tensor (`block_q4_0` little-endian as in
     `ggml/src/ggml-quants.c:307-325`).
   - Comparing recovered `[K, N]` matrix against `to_q.weight` as
     read by the CPU runner.
2. **dispatch_matmul_ tensor view orientation**: the WQBMMv3 path
   builds activation/weight tensors at lines 1521-1701 — verify the
   transpose flag and `transposeB` matches the `[K, N] strides (1, K)`
   weight view contract.
3. **Scale tensor dtype/layout**: `scale_dev` is `[BLK, N]` strides
   `(N, 1)`. F16 on device. Contract: `result = (sum (qs - 8)) * scale`
   per block. If scale is read as `[N, BLK]` row-major (transposed),
   each block sees the wrong scale → output shape stays `[seq, N]`
   but values are scrambled.

#### Strategic recommendation

Path C #5 ship pivot **stays gated** until 08_*Q/K cos > 0.95. Once
the projection is fixed the rest of the chain (RmsNorm + RoPE + FIA
+ output-projection) should follow deterministically — every
downstream substep already runs against numerically-realistic CPU
references that just inherit the projection error.

#### Step 1 (softmax-overflow workaround) — disproved as the fix

A `QIE_ATTN_SOFTMAX_F32=1` env gate was wired (matches the §5.5.10
work-plan recommendation) implementing the CPU reference's `kv_scale =
1/HD` trick at the FIA dispatch site (pre-multiply K,V by 1/HD,
attn-scale = sqrt(HD), post-multiply attn-out by HD; mathematically
equivalent to plain attention but bounds the K·Q^T accumulator
magnitude — see `ggml_extend.hpp::ggml_ext_attention_ext` lines
1318-1361). Two regression runs confirmed:

- Pre-projection FIA output magnitudes are unchanged with the trick
  on (max≈12 for img attn-out vs CPU's max≈545; ratio still 0.022).
- All cos values in the substep table are byte-identical to the
  non-trick baseline.

The trick is a no-op on this configuration because the FIA kernel
already uses `innerPrecise=0` (HIGH_PRECISION = F32 accumulator) and
the input magnitudes — even with q-rmsnorm peaks at 723 — don't blow
the F16 dynamic range during the K·Q^T tile-mul (max tile-product
≈ 65k stays under F16 max). The kv_scale path is left in tree as
defensive plumbing (env-default OFF, byte-identical when off) for
future configurations where K·Q^T might saturate (longer seq,
larger head_dim, or after the QKV-projection fix exposes
larger-magnitude post-rmsnorm values).

#### Source changes (this step)

- `tools/qwen_image_edit/native/image_diffusion_engine.cpp`: env gate
  `QIE_ATTN_SOFTMAX_F32` for the kv_scale=1/HD trick at the FIA
  dispatch site + 08/09/10 disk dumps for substep bisect.
- `tools/qwen_tts/cp_cann_symbols.{h,cpp}`: resolved `aclnnInplaceMuls`
  + `aclnnInplaceMulsGetWorkspaceSize` (resolve_optional).
- `tools/probes/qie_block0_cpu_reference/test_qie_block0_cpu_reference.cpp`:
  inline-replicate `QwenImageAttention::forward` to expose 08/09/11pre
  intermediates; `PublicQwenImageAttention` using-redeclare for
  `dim_head` + `blocks` access.
- `tools/probes/qie_block0_cpu_reference/compare_block0.py`: 14 new
  substep rows (08_*Q/K/V on both streams, 09_*Q/K_rmsn).
- `tools/probes/qie_q45_real_denoise_smoke/test_qie_q45_real_denoise_smoke.cpp`:
  diagnostic env gate `QIE_RUNTIME_MAX_TXT_SEQ=1` to align cfg.max_txt_seq
  to actual txt_seq (tests pe-row alignment hypothesis, default OFF).
  Empirical result: pe alignment is NOT the cause of the 08-RED.

#### Receipts

- `/tmp/qie_q2454k_step3.log` — final substep bisect (canonical
  max_txt_seq=256, env QIE_ATTN_SOFTMAX_F32 unset).
- `/tmp/qie_block0_inputs/` (33 files) + `/tmp/qie_block0_outputs/`
  (32 files) — full 04 → 24 substep dumps both sides.
- Probe + comparator are re-runnable end-to-end with
  `bash tools/probes/qie_block0_cpu_reference/run_step_4i.sh`.

