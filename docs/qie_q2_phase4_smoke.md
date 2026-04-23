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

## §2. Phase 4.2 — 60-block DiT forward loop (pending)

Not yet started. Scope: wire `forward_block_` across `cfg_.num_layers` in
`ImageDiffusionEngine::forward()`. Per Phase 3 §7 item 5, this is pure
plumbing — each block takes the previous layer's output as input to the
next. Gate: cos_sim > 0.95 at layer 60 output vs CPU reference on dummy
input.

---

## §3. Phase 4.3 — Euler-flow 20-step denoise (pending)

Not yet started. Port from `tools/ominix_diffusion/src/denoiser.hpp:831-865`.
Gate: 20-step loop runs without crash, output is non-trivial (not
zero/NaN/constant), wall-clock measured.

---

## §4. Phase 4.4 — Q4-resident forward dispatch (pending)

Not yet started. Replace F16 fallback matmul path with real Q4_0 WQBMMv3
dispatch. Gate: no NaN, reasonable numerics, same output shape.

---

## §5. Phase 4.5 — Canonical cat-edit smoke (pending)

Not yet started. Gate: any sensible output, no crash, HBM peak ≤ 18 GiB.
Report end-to-end 20-step wall-clock for one 256×256 edit — the first QIE
native fps measurement that isn't Q1 baseline.
