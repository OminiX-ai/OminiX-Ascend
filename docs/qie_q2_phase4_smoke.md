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

## §4. Phase 4.4 — Q4-resident forward dispatch (pending)

Not yet started. Replace F16 fallback matmul path with real Q4_0 WQBMMv3
dispatch. Gate: no NaN, reasonable numerics, same output shape.

---

## §5. Phase 4.5 — Canonical cat-edit smoke (pending)

Not yet started. Gate: any sensible output, no crash, HBM peak ≤ 18 GiB.
Report end-to-end 20-step wall-clock for one 256×256 edit — the first QIE
native fps measurement that isn't Q1 baseline.
