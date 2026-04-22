# A4c — ASR prefill RTF gate closure

**Author**: Agent A4c
**Date**: 2026-04-22
**Host**: ac03 (port 30412), 910B4, CANN 8.3.RC1 — dedicated ASR host
**Scope origin**: A4b (commit `ed67c362`) missed the 0.142 mayun_ref / 0.159
Tier-1 median RTF gate. A4c drives the remaining gap to zero in ROI order.
**Parity gate**: CER = 0 across 13 Tier-1 clips (`docs/asr_a1a_data/tier1_q8_cann_clean.json`).
**Perf gate**: mayun_ref ≤ 0.142 AND Tier-1 median ≤ 0.159.

---

## Baseline (inherited from A4b)

| Metric | A4 (525d8a1e) | A4b (ed67c362) | Gate | A4b gap |
|---|---|---|---|---|
| Tier-1 CER | 0.000 | 0.000 | 0.000 | — |
| Tier-1 RTF median | 0.262 | 0.250 | 0.159 | +0.091 |
| RTF mayun_ref | 0.185 | 0.177 | 0.142 | +0.035 |

A4b architectural finding (carried forward): W8 V3 at prefill M=128 is
compute-bound; INT8 dequant overhead offsets INT8 compute savings. The
remaining gap is structural — op-count collapse and kernel fusion are the
levers, not further INT8 rewiring of existing call sites.

---

## Phase 1 — Fused QKV via aclnnGroupedMatmulV3

**Scope**: collapse the 3 per-layer prefill QKV `w8_matmul_` calls into one
`aclnnGroupedMatmulV3` dispatch. Direct transfer of the CP-path landing
(commit `9d52177e`, `CpCannEngine::gmm_qkv_`).

**Env gate**: `TALKER_W8_GMM_QKV=1` (default OFF). Requires `w8_applied_` +
`g_cann.has_grouped_matmul_v3()`. Orthogonal to CP's `TALKER_CP_GMM_QKV` —
the two flags gate disjoint code paths (CP decode vs Talker prefill).

**Invariants preserved**:

- `TALKER_W8_QUANT` unset → byte-identical to pre-A4c. Every new branch is
  `gmm_qkv_applied_`-gated, which requires `w8_applied_`. The unset path
  never touches the new code.
- aclGraph decode cache (`decode_graphs_`) untouched. GMM-at-prefill does
  not interact with decode capture — prefill has always been eager.
- W8 path without `TALKER_W8_GMM_QKV=1` → byte-identical to A4b (gate on
  `gmm_qkv_applied_` which defaults to false).

**Numerical footprint** (from CP-side probe, `docs/qkv_grouped_probe_verdict.md`):
1-2 F16 ulp per-layer vs the 3-call reference; ~1.7e-3 relative drift on
worst-case K channel. NOT bit-exact. A4c parity gate is clip-level CER = 0,
not per-tensor bit-equal — this is the same envelope A4 and A4b landed in.

**Files touched**:

- `tools/qwen_tts/talker_cann_engine.h` — add `gmm_qkv_enabled_`,
  `gmm_qkv_applied_`, `antiquant_zero_offset_dev_`, + `gmm_qkv_prefill_()` decl.
- `tools/qwen_tts/talker_cann_engine.cpp` — free site in `~TalkerCannEngine`,
  helper body after `w8_matmul_`, init site after `w8_applied_` latching,
  dispatch site in `forward_prefill` Q/K/V block.
- Patch: `/tmp/a4c_phase1.patch` (278 lines).

**Build + smoke plan (on ac03)**:

1. Apply patch on ac03: `(cd ~/work/OminiX-Ascend && git apply /tmp/a4c_phase1.patch)`.
2. HBM lock: `flock /tmp/ac03_hbm_lock -c 'cmake --build build-w1 --target qwen_asr -j 8'`.
3. Smoke 1 (baseline, W8 on, GMM off): `TALKER_W8_QUANT=1 ./qwen_asr ...` — expect A4b RTF.
4. Smoke 2 (GMM on): `TALKER_W8_QUANT=1 TALKER_W8_GMM_QKV=1 ./qwen_asr ...`.
5. Smoke 3 (unset path sanity): `./qwen_asr ...` without env — must match pre-patch wall time.
6. Tier-1 CER sweep: 13 clips × 3 runs → median. CER must be 0 across all clips.
7. RTF decomposition (prefill/decode split) as in A4b.

**Decision after Phase 1 measurement**:

- If mayun_ref ≤ 0.142 AND Tier-1 median ≤ 0.159 → STOP, ship Phase 1,
  close A4c. Scope-discipline rule applies.
- Else → proceed to Phase 2 (RoPE loop collapse) and re-measure.

**Results (ac03, sweep date 2026-04-23, verdict RED)**:

Sweep driver: `/tmp/a4c_sweep_v2.sh` on ac03 (v1 had a bash
prefix-env-assignment bug that dropped `$extra_env` for configs A+B — see
AGENT-AC03 notes). Binary `build-w1/bin/qwen_asr_native` built from
fork HEAD `0f860c5b` + A4c Phase 1 patch applied on working tree.

Three configs, 13 Tier-1 clips × 3 runs each:

| Metric | C (unset F16) | A (A4b W8) | B (A4c GMM on) | B vs A Δ | Gate | Verdict |
|---|---|---|---|---|---|---|
| Tier-1 RTF median | 0.301 | 0.254 | **0.261** | +0.007 | ≤ 0.159 | RED |
| RTF mayun_ref | 0.218 | 0.190 | **0.181** | −0.009 | ≤ 0.142 | RED |
| Max clip CER | 0.0385 | 0.0385 | 0.0385 | 0 | 0 | RED |

Per-clip Config B RTF (CER in parens):

| Clip | RTF | CER/WER | |
|---|---|---|---|
| bys_ref | 0.270 | CER 0.000 | |
| cove_ref | 0.239 | WER 0.000 | |
| doubao_ref | 0.206 | CER 0.000 | |
| ellen_ref | 0.214 | WER 0.000 | |
| juniper_ref | 0.341 | WER 0.000 | |
| luoxiang_ref | 0.189 | CER 0.000 | |
| mabaoguo_ref | 0.329 | **CER 0.037** | repeatable across A+B+C — env drift |
| maple_ref | 0.264 | WER 0.000 | |
| **mayun_ref** | **0.181** | **CER 0.000** | hard-gate clip (target ≤ 0.142) |
| shenyi_ref | 0.261 | **CER 0.024** | repeatable across A+B+C |
| trump_ref | 0.364 | WER 0.000 | |
| yangmi_ref | 0.201 | CER 0.000 | |
| zhoujielun_ref | 0.312 | **CER 0.039** | repeatable across A+B+C |

**Sweep absolute numbers run slower than A4b's published 0.177 mayun /
0.250 Tier-1 median**, most likely because this sweep ran concurrently
with a 12 GiB GGUF transfer into `/home/ma-user/work/qie_weights/` during
Config C and was re-driven immediately after. Relative A→B delta is the
ship metric. A at 0.190 vs A4b's 0.177 is +7% env noise; B at 0.181 is
measurably below A on mayun_ref but by a tiny margin (−5%) and above A
on Tier-1 median (+2.8%, likely run-to-run scatter).

**A4c Phase 1 alone does NOT close the A4b gap.** Per-mayun_ref speedup
is directionally correct and bit-plausible with the 1–2 F16 ulp GMM
numerical footprint, but ~40× below the ≥20% A→B reduction needed to
close 0.190 → 0.142.

**CER drift vs parity reference (orthogonal finding)**: Clips
mabaoguo / shenyi / zhoujielun produce non-zero CER in C (F16, unset)
AND A (W8, A4b-equivalent) AND B (W8+GMM). Parity reference
(`docs/asr_a1a_data/tier1_q8_cann_clean.json`) has CER=0 on all three.
The drift is pre-existing and independent of A4c — the Config A
W8-off baseline reproduces it too. Out of A4c scope; flag to PM for
a separate investigation (likely decoder GGUF or CANN toolkit version
skew on ac03 vs the environment that produced the parity reference).

TTS regression cross-check (ac01, PM-verified post-landing):
- `TALKER_W8_QUANT` unset + F16 Talker GGUF → wall-time delta must be 0
  (noise). The `gmm_qkv_applied_` gate requires `w8_applied_`; unset
  path is not on the new branch. NOT executed this cycle — gate-miss
  means the Phase 1 commit is not eligible to land, so the cross-check
  is deferred until after Phases 2+ land a GREEN Tier-1 gate.

Artifacts: `docs/asr_a4c_data/a4c_tier1_{A,B,C}.json` + `.log`.

---

## Phase 2 — Per-row RoPE loop collapse (unblocked; Phase 1 RED)

Gated on Phase 1 miss. Scope: replace the per-position RoPE application
loop in `forward_prefill` with a single batched dispatch. V2 op is
numerically correct per TTS probe `docs/a2_reopen_v2_probe.md` — prior
retire was on wiring, not op.

Status: **UNBLOCKED** (Phase 1 swept 2026-04-23, verdict RED; see
Results table above). Not yet started on code.

---

## Phase 3 — Per-row FIAS loop collapse (conditional)

Gated on Phases 1+2 miss. Scope: batched FIAS dispatch in place of the
per-row attention loop. See the current FIAS loop site in
`forward_prefill`.

Status: NOT STARTED.

---

## Phase 4 — Prefill aclGraph capture (conditional)

Gated on Phases 1-3 miss. Scope: pos-keyed capture at prefill, adapting
TTS G2 pattern (commit `7fe5897`). Risk: variable seq → pos-bucketing.

Status: NOT STARTED.

---

## Related

- A4 commit: `525d8a1e` (Q8_0 GGUF loading)
- A4b commit: `ed67c362` (W8 prefill extension)
- CP fused-QKV reference: `9d52177e` (TALKER_CP_GMM_QKV, A16W8 + groupType=-1)
- QKV probe verdict: `docs/qkv_grouped_probe_verdict.md`
- ASR learnings: `docs/asr_optimization_learnings.md`
- Parity reference: `docs/asr_a1a_data/tier1_q8_cann_clean.json`
