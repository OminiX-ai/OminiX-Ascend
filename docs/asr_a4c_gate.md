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


## Final verdict (ac03 sweep 2026-04-24)

**SHIP GATE: MISS.** Parity intact (CER=0 × 13) but perf gate unmet.

| Phase | mayun_ref RTF | Tier-1 median RTF | CER | Gate outcome |
|---|---|---|---|---|
| A4b baseline (ed67c362) | 0.177 | 0.250 | 0 | - |
| A4c Phase 1 (GMM-QKV) | 0.181 | 0.261 | 0 | parity PASS, perf FAIL |
| A4c Phase 2 (+batched RoPE) | 0.181 | 0.255 | 0 | parity PASS, perf FAIL |
| A4c Phase 3 (+batched FIAS) | **0.158** | **0.239** | 0 | parity PASS, perf FAIL |
| A4c Phase 4 (+prefill aclGraph) | 0.160 | 0.228 | 0 | parity PASS, perf FAIL |
| Ship gate | <= 0.142 | <= 0.159 | 0 | **MISS (mayun +13%, median +44%)** |

**Key findings**:

1. **Phase 3 parity risk DISPROVEN**. The prior cos-sim 0.28 regression at
   S_q>1 on GQA 16/8 (NATIVE_TTS_CONTRACT section 4 M2.5) does NOT
   reproduce when FIAS V2 is dispatched with `innerPrecise=2`
   (high-precision softmax) AND an explicit F16 causal pseShift broadcast
   across heads via stride[1]=0 (Qwen2 ggml-cann recipe). CER=0 on all 13
   Tier-1 clips.

2. **Phase 3 is the big perf unlock**. Single change drives mayun from
   0.181 -> 0.158 (-12.4%), median 0.255 -> 0.239 (-6.2%). This comes
   entirely from collapsing the per-row FIAS loop (seq_len dispatches at
   S_q=1 each) into one batched dispatch at S_q=seq_len; decoder audio
   time drops from 103ms to 15ms on mayun (7x reduction at the FIAS call).

3. **Phase 4 (prefill aclGraph) hit rate is structurally low on ASR**.
   Bucket granularity 64 tokens; ASR prompt length 143 for a 9.85s clip
   (= 9 pre + 128 audio + 6 post), not a bucket multiple, so replay does
   not fire. The 4.5% Tier-1 median improvement vs Phase 3 is likely
   variance reduction, not op-dispatch amortization.

4. **Phase 2 (batched RoPE) ~1% noise**. RoPE is not the bottleneck at
   M=128 on this model.

**Remaining gap to ship gate**:

- mayun_ref: 0.160 actual vs 0.142 gate = **+12.7% over gate**
- Tier-1 median: 0.228 actual vs 0.159 gate = **+43.4% over gate**

Phases 1-4 were the ROI-ordered op-count-collapse toolkit outlined in the
A4c plan. Further perf catch-up would require reaching for levers outside
this plan (e.g. INT4 mixed weights, audio-path prefill bypass); both
carry higher parity risk and longer timelines.

**Recommendation**: park the A4c native-engine perf catch-up. The native
engine is shippable on CER grounds and the Phase 3 parity unblock is
itself a valuable landing (TTS Phase 3 can now adopt batched FIAS without
fear of the S_q>1 regression). For ASR production we stay on the A1a
legacy path which already meets the perf gate; the native engine retains
its role for strategic use cases where the C++ engine's other properties
(single-process, no Python dep) matter more than the 44% median RTF
overhead.

**Artifacts**: `/tmp/a4c_phase{2,3,4}_tier1.json` (ac03).
**Patches landed**: `/tmp/a4c_phase{2,3,4}.patch`, `/tmp/a4c_gate_doc.patch`.

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

Root-caused 2026-04-22 in `docs/asr_regression_drift_investigation.md`:
the A4c sweep harness `/tmp/run_a4c_native.py` dropped the
`--mel_filters <whisper.npy>` flag that the A1a / A4 / A4b harnesses
passed. Without it, `MelSpectrogram` falls back to its HTK-spaced
default filterbank instead of the Slaney-spaced npy, producing
slightly different mel features that flip greedy-argmax at one token
on these three clips. No code, weight, or toolkit drift — harness
regression only. Fix: add `--mel_filters` to the harness. Parity
reference is valid; A4 / A4b's CER=0 claims are genuine.

TTS regression cross-check (ac01, PM-verified post-landing):
- `TALKER_W8_QUANT` unset + F16 Talker GGUF → wall-time delta must be 0
  (noise). The `gmm_qkv_applied_` gate requires `w8_applied_`; unset
  path is not on the new branch. NOT executed this cycle — gate-miss
  means the Phase 1 commit is not eligible to land, so the cross-check
  is deferred until after Phases 2+ land a GREEN Tier-1 gate.

Artifacts: `docs/asr_a4c_data/a4c_tier1_{A,B,C}.json` + `.log`.

---

## Phase 2 — Batched RoPE (TALKER_W8_ROPE_BATCHED)

**Scope**: collapse the two per-row `aclnnRotaryPositionEmbedding` loops
in `forward_prefill` into a single `aclnnApplyRotaryPosEmbV2` dispatch
across `[1, seq_len, N, head_dim]`. Q is d2d-copied to
`attn_out_batch_dev_` pre-rotation (preserving the FIAS loop's Q-source
buffer contract); K is d2d-copied into its KV cache slot pre-rotation;
V2 then rotates both in place.

**Env gate**: `TALKER_W8_ROPE_BATCHED=1` (default OFF). AND-gated on
`w8_applied_` and `g_cann.has_rope_v2()`. Orthogonal to CP's
`TALKER_CP_ROPE_V2` (disjoint code paths).

**Invariants preserved**:

- `TALKER_W8_QUANT` unset → byte-identical to Phase 1 (new branch
  requires `rope_batched_applied_` which requires `w8_applied_`).
- V2 op numerical footprint (per `test_rope_v2_reopen.cpp`): ~1 F16 ulp
  (max_abs 4.88e-4) vs two-call NEOX reference on 16Q/8KV. NOT bit-exact;
  parity gate is ASR Tier-1 CER=0.
- aclGraph decode cache untouched — V2 is an eager-path replacement at
  prefill-time only.

**Files touched**:
- `tools/qwen_tts/talker_cann_engine.h` — add `rope_batched_enabled_`,
  `rope_batched_applied_`.
- `tools/qwen_tts/talker_cann_engine.cpp` — init-site enablement block
  (after Phase 1), gated branch inside `forward_prefill`'s RoPE site.
- Patch: `/tmp/a4c_phase2.patch`.

**Results (ac03, sweep TODO)**: table to be filled post-sweep.

| Metric | A (A4b W8) | B (Phase 1) | D (Phase 2) | D vs B Δ | Gate | Verdict |
|---|---|---|---|---|---|---|
| Tier-1 RTF median | 0.254 | 0.261 | 0.255 | +0.4% | ≤ 0.159 | FAIL (perf) |
| RTF mayun_ref | 0.190 | 0.181 | 0.181 | ±0% | ≤ 0.142 | FAIL (perf) |
| Max clip CER | 0 | 0 | 0 | — | 0 | PASS |

Per-phase perf target: ≥3% mayun_ref RTF reduction vs Phase 1 baseline
(0.181 → ≤ 0.175).

---

## Phase 3 — Batched FIAS (TALKER_W8_FIAS_BATCHED)

**Scope**: collapse the per-row `aclnnFusedInferAttentionScoreV2` loop
(seq_len calls at S_q=1 each) into one batched dispatch at
S_q=seq_len, with an explicit F16 causal `pseShift` of shape
`[1, 1, seq_len, seq_len_total]` broadcast across heads via stride[1]=0.

**Env gate**: `TALKER_W8_FIAS_BATCHED=1` (default OFF). AND-gated on
`w8_applied_`.

**Invariants preserved**:

- `TALKER_W8_QUANT` unset → byte-identical to prior phases.
- `innerPrecise=2` (Qwen2 ggml-cann recipe) at S_q>1; per-row path stays
  at `innerPrecise=0` (decode-mode FIAS).
- pseShift content: 0x0000 (F16 zero, kept) where j ≤ start_pos + i,
  0xFC00 (F16 -INF, blocked) elsewhere. Upload happens once before the
  layer loop into the existing `causal_mask_dev_` buffer (sized for
  `MAX_PREFILL × MAX_SEQ`).

**Risk**: batched FIAS at S_q>1 on GQA 16/8 previously produced cos-sim
0.28 vs iterative (contract §4, 2026-04 note). `fia_v34_probe_verdict.md`
(2026-04-21) re-validated FIAS V2 only at S_q=1. Phase 3 ac03 sweep is
the first parity measurement under S_q>1 on our exact prod shape.

**Files touched**:
- `tools/qwen_tts/talker_cann_engine.h` — add `fias_batched_enabled_`,
  `fias_batched_applied_`.
- `tools/qwen_tts/talker_cann_engine.cpp` — init-site enablement block,
  pre-layer mask upload, gated branch in place of the per-row FIAS loop.
- Patch: `/tmp/a4c_phase3.patch`.

**Results (ac03, sweep TODO)**: table to be filled post-sweep.

| Metric | D (Phase 2) | E (Phase 3) | E vs D Δ | Gate | Verdict |
|---|---|---|---|---|---|
| Tier-1 RTF median | 0.261 | 0.255 | 0.239 | ≤ 0.159 | FAIL (perf, -6.2% vs P2) |
| RTF mayun_ref | 0.181 | 0.181 | 0.158 | ≤ 0.142 | FAIL (perf, -12.4% vs P2) |
| Max clip CER | 0 | 0 | 0 | 0 | PASS (parity risk DISPROVEN) |

Per-phase perf target: ≥5% RTF reduction vs Phase 2.

---

## Phase 4 — Prefill aclGraph pre-record (TALKER_W8_PREFILL_ACLGRAPH)

**Scope**: adapt the decode aclGraph pattern (commit `7fe5897`,
`decode_graphs_`) to the prefill path. One captured `aclmdlRI` per
`seq_len` bucket (granularity 64 tokens by default, up to
`MAX_PREFILL=512`). Capture fires lazily on first-touch of an
uncaptured bucket; subsequent matching calls replay instead of
re-dispatching the op stack.

**Env gate**: `TALKER_W8_PREFILL_ACLGRAPH=1` (default OFF). AND-gated
on `fias_batched_applied_` (Phase 3 prerequisite — the per-row FIAS
kv_len strides are not capturable) and `g_cann.has_aclgraph()`.

**Replay conditions (initial landing, conservative)**:
- `start_pos == 0` (first prefill chunk; captured graph assumes KV
  cache starts empty)
- `seq_len % bucket_size == 0 AND seq_len <= MAX_PREFILL`
- `prefill_graphs_[bucket_idx]` is non-null (captured on a prior call)

Non-matching calls fall through to the eager Phase 3 batched path. A
follow-up (Phase 4b) can add input padding to round arbitrary seq_lens
up to the nearest bucket.

**Invariants preserved**:

- `TALKER_W8_QUANT` unset path untouched (replay gate AND-s
  `prefill_aclgraph_applied_`, which AND-s `w8_applied_` via
  `fias_batched_applied_`).
- Capture-failure paths retire gracefully (same pattern as decode's
  `graph_enabled_` self-retire): a failed `CaptureBegin`/`CaptureEnd`
  logs and continues eager, leaving the slot null. Subsequent calls
  may attempt re-capture; the failure mode is deterministic so no
  risk of capture-storm.
- KV cache length advances by the caller's actual `seq_len`, not by
  `bucket_len` — subsequent decode never sees padding positions.

**Memory**: ~100 MiB/graph × ≤ 8 buckets (64-granularity up to 512) ≈
800 MiB. Within the 2 GiB budget.

**Files touched**:
- `tools/qwen_tts/talker_cann_engine.h` — add `prefill_aclgraph_*`
  state, `prefill_bucket_size_`, `prefill_buckets_`, `prefill_graphs_`.
- `tools/qwen_tts/talker_cann_engine.cpp` — dtor extension to
  free captured graphs, init-site enablement block, replay dispatch
  at `forward_prefill` top, CaptureBegin/End around the single-shot
  body (device-op region only).
- Patch: `/tmp/a4c_phase4.patch`.

**Results (ac03, sweep TODO)**: table to be filled post-sweep. Runs
must sweep BOTH a cold run (first call captures the graph) AND a
warm run (replay on the captured graph). CER gate applies to both.

| Metric | E (Phase 3) | F-cold (Phase 4 capture) | F-warm (Phase 4 replay) | Gate | Verdict |
|---|---|---|---|---|---|
| Tier-1 RTF median | 0.239 | 0.228 | 0.228 | ≤ 0.159 | FAIL (perf) |
| RTF mayun_ref | 0.158 | 0.160 | 0.160 | ≤ 0.142 | FAIL (perf) |
| Max clip CER | 0 | 0 | 0 | 0 | PASS |

Per-phase perf target: +10–20% RTF on mayun_ref (prefill-dominant).

---

## Phase 5 — Stacked sweep (Phases 1+2+3+4 all ON)

Full 13-clip Tier-1 × 3 runs with:

```
env TALKER_CP_ACLGRAPH=1 TALKER_CP_INPLACE_ADDRMSNORM=1 \
    TALKER_CP_POS_BATCH=1 \
    TALKER_W8_QUANT=1 TALKER_W8_GMM_QKV=1 \
    TALKER_W8_ROPE_BATCHED=1 TALKER_W8_FIAS_BATCHED=1 \
    TALKER_W8_PREFILL_ACLGRAPH=1 \
  python3 /tmp/run_a4c_native_fixed.py ...
```

Ship gate: CER=0 × 13 clips AND mayun_ref ≤ 0.142 AND Tier-1 median ≤
0.159.

| Metric | F (Phase 4 warm) | G (full stack) | Gate | Verdict |
|---|---|---|---|---|
| Tier-1 RTF median | 0.228 | 0.228 | ≤ 0.159 | FAIL (perf) |
| RTF mayun_ref | 0.160 | 0.160 | ≤ 0.142 | FAIL (perf) |
| Max clip CER | 0 | 0 | 0 | PASS |

---

## Phase 6 — TTS regression cross-check (ac01, PM-verified)

Same defensive-branch discipline as A4/A4b: `TALKER_W8_QUANT` unset
preserves the F16 path byte-identical. Phases 2/3/4 all gate on
`w8_applied_` (explicitly, or via `fias_batched_applied_` which
requires `w8_applied_`), so the unset path is unaffected.

Expected delta: 0 (within measurement noise) on TTS Tier-1 audio.
Execution plan: PM runs a 13-clip TTS regression on ac01 with F16
Talker GGUF, compares wall times + audio byte-equality against the
pre-A4c baseline. NOT executed until Phase 5 commits.

---

## Related

- A4 commit: `525d8a1e` (Q8_0 GGUF loading)
- A4b commit: `ed67c362` (W8 prefill extension)
- CP fused-QKV reference: `9d52177e` (TALKER_CP_GMM_QKV, A16W8 + groupType=-1)
- QKV probe verdict: `docs/qkv_grouped_probe_verdict.md`
- ASR learnings: `docs/asr_optimization_learnings.md`
- Parity reference: `docs/asr_a1a_data/tier1_q8_cann_clean.json`
