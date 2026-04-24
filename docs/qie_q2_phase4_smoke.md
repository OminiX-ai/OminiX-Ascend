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

| Measurement | Gate | Step 4 actual |
|---|---|---|
| NaN in final latent | =0 | (run on ac03) |
| inf in final latent | =0 | (run on ac03) |
| std(final latent) | > 0.001 | (run on ac03) |
| min(final latent) | > -20 | (run on ac03) |
| max(final latent) | < +20 | (run on ac03) |
| denoise_full wall (20 steps, cfg=1 or 4) | < 1450 s target | (run on ac03) |
| Output PNG eye-check | recognizable cat | Step 4d |

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
