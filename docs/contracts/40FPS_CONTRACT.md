# 40 fps on Ascend 910B4 — Single-Card Parity with M4 Pro MLX

## 1. Status & mandate

**Status**: NEW (2026-04-21, PM signed). Umbrella contract for the
stretch target. Subsumes `FUSED_OP_LANDING_CONTRACT.md` as its
milestone M1.

**Target**: **≥ 40 fps** on canonical xvec mayun zh LONG text, aclGraph
on, sampling on (ship config), byte-identical user-ear verdict.

**Reference points**:
- Today: **31.6 fps** (aclGraph on, LONG), validated by G4-ext
- M4 Pro Mini MLX: **37 fps** (Agent benchmark)
- M3 Max MLX: **46 fps** (same benchmark)
- **40 fps = parity with M4 Pro**, stretch toward M3 Max

**Why 40 and not 32**: the old ≥32 fps gate was aspirational with no
anchor. **40 fps = "don't lose to a laptop"** — the gate that matters
for customer credibility. Below 40 fps, Ascend 910B4's 2× HBM
bandwidth advantage over Apple Silicon (800 vs 400 GB/s) looks
wasted.

**PM role**: supervise contract milestones; gate each on drift +
perf + user-ear; arbitrate trade-offs between stacked levers.

## 2. Background — what closes the 8-fps gap

Single lever analysis (on 31.6 fps baseline, 33 ms/frame):

| Lever | Projected Δ | Cumulative | Cost | Risk |
|---|---|---|---|---|
| M1: vendor fused ops (Phase A+B) | +1.5 fps | 33 fps | 2-3 days | low |
| M1-stretch: lower-ranked audit candidates | +0.5 fps | 33.5 | 1-2 days | low |
| M2: Talker (LLM) aclGraph | +1-2 fps | 34.5-35.5 | ~1 week | med |
| **M3: CP group-collapse (algorithmic)** | **+5-15 fps** | **40-50** | 1-2 weeks | high |
| M4: W8 → W4 quant | +2-5 fps | 42-55 | 1 week | med (quality risk) |
| M5: Decoder (VAE) fusion / AscendC | RTF only | — | separate | — |

M3 is the structural lever that closes the gap. The CP generates 15
codebook-group tokens per frame autoregressively today; if the
dependency graph permits batching (parallel codebooks, shared hidden
state, or RVQ with inter-frame-only deps), 15 sequential GEMVs
collapse into 1+ GEMMs. That's the 4-10× CP speedup that makes 40 fps
reachable.

**M3 feasibility probe is currently in flight** (Agent GD-audit).

## 3. Scope

**In scope**:
- M1 vendor fused ops (Phase A: RoPE V2 + InplaceAddRmsNorm; Phase B: FFNV3)
- M2 Talker aclGraph (NEW track; uses aclGraph playbook from G0-G4)
- M3 CP group-collapse (conditional on GD-audit verdict)
- M4 W8 → W4 quant (conditional; only if M1+M2+M3 don't hit 40)
- Stacked correctness gates: token drift ≤ 1 per lever + composed
  drift ≤ 1 overall + ear gate each time

**Out of scope**:
- Decoder/VAE fusion (separate RTF-improvement track)
- Cluster (tensor/pipeline parallel) — different contract
- Multi-utterance session amortisation — deferred

## 4. Host plan

- ac01: primary M1 (Phase A in flight), M2 (Talker aclGraph)
- ac02: M3 prototyping (if GD-audit returns GREEN/YELLOW)
- ac03: M4 W4 quant calibration harness
- Patch-file push mechanism per convention

## 5. Milestones with gates

### M1 — Vendor fused ops (→ ~33 fps)

See `FUSED_OP_LANDING_CONTRACT.md` §4 Phase A + Phase B. Already
dispatched.

- [ ] M1.A.1 `aclnnInplaceAddRmsNorm` wired, env-gated
- [ ] M1.A.2 `aclnnApplyRotaryPosEmbV2` wired, env-gated
- [ ] M1.A.3 Combined + aclGraph, ≥ 32 fps LONG, ear clean
- [ ] M1.B.1 Offline W8 gate∥up re-pack script
- [ ] M1.B.2 `aclnnFFNV3` wired, env-gated
- [ ] M1.B.3 Combined, ≥ 33 fps LONG, ear clean

**M1 gate**: fps ≥ 33 on LONG canonical + ear identical to baseline.

### M2 — Talker aclGraph (→ ~34.5-35.5 fps)

Talker is the 28-layer LLM that predicts the next codec group-0 token.
Current cost ~7.8 ms/frame at LONG. aclGraph pattern from G-track
should apply analogously.

- [ ] M2.1 Audit Talker forward op chain (analog of G0 feasibility)
- [ ] M2.2 One-layer Talker smoke (analog of G1)
- [ ] M2.3 Full-forward capture with pos-keyed cache OR TaskUpdate
      (whichever G1-style probe recommends)
- [ ] M2.4 Drift gate on token sequences + ear gate
- [ ] M2.5 Combined with M1 + CP aclGraph, measure fps delta

**M2 gate**: fps ≥ 34.5 LONG + drift ≤ 1 + ear clean.

### M3 — CP group-collapse (→ potentially 40+ fps)

**HARD GATE on GD-audit verdict**:
- FULL-COLLAPSE (parallel codebooks): full batch=15 CP forward →
  ~5× CP speedup. Major engine rewrite but algorithmically clean.
- PARTIAL-COLLAPSE: batch subset of groups. Less speedup, moderate
  rewrite.
- NO-COLLAPSE (strict RVQ): M3 is dead; fps ceiling = M1 + M2.

If GD-audit returns GREEN/YELLOW:

- [ ] M3.1 Design doc: new CP forward signature, batch dimension,
      KV-cache layout changes
- [ ] M3.2 Offline numerical validation: batched CP forward output ==
      sequential (bit-identical at cp_greedy)
- [ ] M3.3 Wire into CpCannEngine behind `TALKER_CP_GROUP_BATCH=N`
      (where N = batch size, e.g. 5 or 15)
- [ ] M3.4 Drift gate + ear gate
- [ ] M3.5 Perf gate: fps ≥ 40 LONG

**M3 gate**: fps ≥ 40 LONG + drift ≤ 1 + ear clean.

### M4 — W8 → W4 quant (conditional, → potentially 42-55)

Only if M1+M2+M3 don't reach 40 fps.

- [ ] M4.1 Per-channel W4 quant calibration (AWQ or similar)
- [ ] M4.2 Validate via `aclnnWeightQuantBatchMatmulV3` W4 lane (if
      supported on 8.3.RC1) or fallback to `aclnnGroupedMatmul`
- [ ] M4.3 Full quality pass (ear gate mandatory; W4 risk is real)

Skip to M4 only if M3 closes the gap.

## 6. Acceptance criteria (summary)

- [ ] M1 lands: ≥ 33 fps LONG + ear clean
- [ ] M2 lands: ≥ 34.5 fps LONG + ear clean
- [ ] M3 GD-audit verdict known; if GREEN/YELLOW, M3 lands ≥ 40 fps LONG + ear clean
- [ ] Total ≥ 40 fps LONG cumulative with all env gates on
- [ ] Token drift ≤ 1 per milestone and composed
- [ ] Coverage wavs delivered; user ear verdict CLEAN
- [ ] Contract stamped with commit SHAs + Verified-by per milestone

## 7. Risks

1. **M3 NO-COLLAPSE**: the model may be strict RVQ. Then M3 is dead
   and ceiling is ~35-36 fps. Mitigation: GD-audit is in flight;
   verdict gates M3 dispatch.
2. **M2 Talker aclGraph may duplicate G1 3e-3 multi-op drift**:
   Talker op count and shape class differs from CP; re-verify
   TaskUpdate semantic on Talker's specific aclOpExecutors. Mitigation:
   apply G2's pos-keyed fallback pattern.
3. **M4 W4 quality regression**: W4 in TTS can introduce audible
   artefacts. Mitigation: ear gate is hard-kill; W4 only dispatches if
   M1+M2+M3 fall short.
4. **Composed drift explosion**: 4 stacked fused ops + group-collapse
   + W4 = many places drift can accumulate. Mitigation: drift gate is
   composed, not just per-milestone.
5. **Env-gate combinatorial space**: 5-6 env vars would create 32-64
   possible configs. Mitigation: ship with `TALKER_CP_40FPS=1`
   umbrella env that enables the default production stack in one flag.

## 8. Host rules

- Each milestone commits locally on host, patches to PM, PM pushes fork
- No Claude coauthor
- HARD KILL at each perf/ear gate
- User-ear outranks fps numerics
- M3 is the load-bearing milestone; M3 failure cuts the contract
  target from 40 to 34-35 realistic

## 9. Timeline (agent-wall)

- M1 (in flight): 2-3 days
- M2: ~1 week
- M3 (conditional): 1-2 weeks
- M4 (conditional on M3 miss): 1 week
- Total if M3 passes: **2-3 weeks agent-wall**
- Total if M3 fails: **1-1.5 weeks**, ceiling ~35 fps
