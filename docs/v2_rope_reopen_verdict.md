# A.2 V2 RoPE Reopen Verdict

**Agent**: A2-reopen
**Date**: 2026-04-21
**Host**: ac02 (910B4, confirmed via Q0 probe)
**Scope**: standalone numeric probe comparing `aclnnApplyRotaryPosEmbV2` against
`aclnnRotaryPositionEmbedding(mode=0, NEOX)` on the exact talker GQA decode
shape, with both cos/sin table preparations surfaced by the MoYoYoTech/llm_mutil_npu
brief.
**Upstream trigger**: `docs/llm_mutil_npu_brief.md` §Q1 — their TP=16 Qwen3-235B
runs V2 successfully on per-rank Hq=4/Hkv=1 (4:1 GQA), falsifying our original
closed-contract H5a (packed-UB + shared-stride breaks heterogeneous head count).

---

## TL;DR

**YELLOW.** V2 is **numerically correct** on our exact talker shape in isolation
and under prod-slot wiring — matches v1 NEOX within F16 rounding (max_abs=4.88e-4
= 1 ulp). The "cos/sin prep is the delta" hypothesis is falsified: our half-duplicated
prep and MoYoYoTech's half-half prep produce **byte-identical tables** because
the `pair = d < half ? d : d - half` index mapping makes them mathematically
the same. The original A.2 457 vs 434 frame divergence was therefore **not an
op-level semantic bug**, and re-landing A.2 requires prod-wiring investigation,
not a cos/sin table fix.

---

## Probe results (standalone harness)

Source: `tools/qwen_tts/test_rope_v2_reopen.cpp` (committed 4bb1a54 on top of
d835c97). Build is standalone — no CMake entry needed:

```
source /usr/local/Ascend/ascend-toolkit/set_env.sh
g++ -std=c++17 -O2 -o test_rope_v2_reopen \
    tools/qwen_tts/test_rope_v2_reopen.cpp -ldl -lm
./test_rope_v2_reopen
```

Shape: Q=[1,1,16,128], K=[1,1,8,128], F16, head_dim=128, pos=5, theta=1e6.
Payload: deterministic sin-wave `q[i] = sin(0.017*i) * 0.5`, same for k.

### Core diff table

| Path | max_abs vs v1 | rel_l2 vs v1 | first_diff_idx | Verdict |
|---|---|---|---|---|
| v1 ref (RotaryPositionEmbedding mode=0, Prep A, two separate Q/K calls) | 0 (baseline) | 0 | — | baseline |
| V2 + Prep A (half-duplicated)   Q | 4.88e-4 | 2.38e-4 | -1 (no diff > 1e-3) | **PASS** |
| V2 + Prep A (half-duplicated)   K | 4.88e-4 | 2.23e-4 | -1 | **PASS** |
| V2 + Prep B (HF half-half)      Q | 4.88e-4 | 2.38e-4 | -1 | **PASS** |
| V2 + Prep B (HF half-half)      K | 4.88e-4 | 2.23e-4 | -1 | **PASS** |
| V2 prod-slot sim                Q | 4.88e-4 | 2.38e-4 | -1 | **PASS** |
| V2 prod-slot sim                K (written into `[MAX_SEQ, kv_dim]` cache slot at POS=5 with kv_dim stride) | 4.88e-4 | 2.23e-4 | -1 | **PASS** |

`max_abs = 4.88e-4 = 2^-11` is exactly 1 ulp at F16 magnitude ~1. This is
rounding, not semantic mismatch. MoYoYoTech's own `test_rope_fused.cpp` uses
the same `1e-2` tolerance.

Workspace sizes returned by V2 `GetWorkspaceSize`: 16,777,216 bytes (16 MiB)
for all three V2 paths — reasonable.

### Cos/sin table comparison (Prep A vs Prep B)

| Property | Prep A (half-duplicated) | Prep B (HF half-half, MoYoYoTech) |
|---|---|---|
| First 8 values (F16) | `+0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044` | identical |
| Second-half first 8 (indices 64..71) | `+0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044` | identical |
| FNV-1a 64 of full 128-element table (cos) | `7625022a0d3f176d` | `7625022a0d3f176d` |
| Byte-identical on host? | **YES** | — |

**Why.** Talker's `build_rope_tables_` computes cos/sin for pairs `j` in
`[0, half)` and duplicates them into both halves: `cos[j] = cos[j+half] =
cos(pos * freq(j))`. MoYoYoTech's `fill_cos_sin_hf` computes
`pair = d < half ? d : d - half` and writes `cos[d] = cos(pos * freq(pair))`.
For `d < half`, `pair = d`, so `cos[d] = cos(pos * freq(d))`. For
`d >= half`, `pair = d - half`, so `cos[d] = cos(pos * freq(d-half))` —
identical value to `cos[d-half]`. The two formulations are algebraically
the same; the "half-duplicated vs half-half" nomenclature is cosmetic.

This falsifies the leading Q1 candidate in the MN brief at the host layer
without any kernel execution needed.

### Adjacent-cache-slot integrity check (prod-slot sim)

Slots POS-1 and POS+1 initialized to sentinel `-99.0`. After V2 in-place
write to slot POS with K descriptor `[1,1,NK,DH]` at offset `POS*kv_dim`
with strides `[kv_dim, kv_dim, DH, 1]`:

```
cache[POS-1] first 4 elements: -99.00000 -99.00000 -99.00000 -99.00000  (intact)
cache[POS+1] first 4 elements: -99.00000 -99.00000 -99.00000 -99.00000  (intact)
```

**Rules out H5b (in-place cache aliasing out-of-bounds write).** V2 respects
the provided stride descriptor and does not bleed into adjacent slots.

---

## Hypothesis ranking — post-probe update

| Hypothesis | Pre-probe likelihood | Post-probe verdict |
|---|---|---|
| H1 — cos/sin prep expectation differs | LOW (per source reading) | **CONFIRMED LOW**: preps byte-identical on host; V2 accepts both. |
| H2 — `rotaryMode` semantics | LOW | unchanged — V2 accepts `"half"` cleanly. |
| H3 — layout semantics (BSND vs SBND) | LOW | unchanged — `layout=1` accepted. |
| H4 — sin sign convention | LOW | unchanged — numerically matches. |
| H5a — packed-UB shared-stride breaks GQA (Nq≠Nk) | HIGH | **FALSIFIED**: V2 on 16Q/8KV matches v1. |
| H5b — in-place cache-slot aliasing | HIGH | **FALSIFIED**: adjacent cache slots untouched; K rotation correct at prod-slot descriptor. |
| H6 — internal F16 precision drift | MEDIUM | unchanged — 1-ulp consistent with both paths using F16 multiply. |

**Original closed-contract verdict (H5a is the root cause) is overturned by
the probe.** The op is fine on our exact shape.

---

## What could the original A.2 divergence actually have been?

The probe rules out every on-op hypothesis the original debug doc
(`docs/v2_rope_numerics_debug.md`) listed. Yet the original patch produced
457 vs 434 frames, which is a 5% divergence — far larger than numerical
drift can explain. Plausible remaining candidates for a rewiring attempt:

1. **Workspace lifetime bug.** The original patch's V2 dispatch may have
   reused a workspace buffer that was still being read by a prior op. Our
   probe allocates a fresh `DevBuf` per V2 call and syncs after each, which
   would hide this. Prod emits aclnn ops back-to-back without sync.
2. **aclGraph interaction.** The debug doc line 142-144 notes V2 was gated
   to disable aclGraph under `cp_rope_v2_applied_`. If the gate had a race
   or was applied after a graph was already captured, V2 calls could land
   inside a captured graph with stale tensor descriptors. The probe
   bypasses aclGraph entirely.
3. **Tensor descriptor stride drift.** Our prod code builds Q descriptor
   with `strides={q_dim, q_dim, head_dim, 1}` (see talker_cann_engine.cpp:1398).
   q_dim = NQ * head_dim = 16*128 = 2048; the probe uses the equivalent
   `NQ*DH = 2048`. These are identical. But if the original A.2 patch
   changed the Q descriptor (e.g. collapsed the leading two dims to
   `[1, NQ, head_dim]` for V2 while v1 used `[1, 1, NQ, head_dim]`), the
   kernel's layout-1 axis interpretation would differ.
4. **V2 is applied after the QK-norm in a different order from v1.** Our
   prod order is: RmsNorm → Q/K-proj → QK-norm → v1 RoPE → FIA. If the
   A.2 patch moved V2 to a different insertion point, intermediate buffers
   may have stale content.
5. **Off-by-one in pos argument.** Our probe fixes `pos=5`. Prod uses
   `pos` from the decode loop; the original patch may have passed a
   different pos to cos/sin slice arithmetic for V2 than for v1.

Of these, **(1) workspace lifetime** and **(3) descriptor stride** are the
most likely and cheapest to test in a rewire attempt.

---

## Production wiring results

**Not run.** The V2 prod wiring was removed from the tree when A.2 closed
(current HEAD d835c97 has no V2 call sites; `grep -rn 'aclnnApplyRotaryPosEmbV2'
tools/qwen_tts/` returns nothing). Re-wiring V2 into the engine and running
the canonical LONG xvec mayun zh with `TALKER_CP_ROPE_V2=1` is the next
step; it is deferred because the probe's GREEN numeric result changes the
rewiring strategy (the original patch assumed cos/sin prep was the delta,
which is now falsified — a fresh rewiring should focus on workspace
lifetime and descriptor parity instead).

---

## Final verdict

- [ ] GREEN: V2 works with Prep B, A.2 re-lands as +0.15-0.25 fps lever
- [x] **YELLOW**: V2 + prep match in isolation AND under prod-slot wiring,
      but the original A.2 production divergence must have come from
      something other than the op. A.2 reopens as a **rewiring probe**,
      not a cos/sin fix.
- [ ] RED: neither prep matches v1

---

## Patch

Probe binary + source: committed as 4bb1a54 on ac02:~/work/OminiX-Ascend-w1.
Patch file: `ac02:/tmp/a2_reopen.patch` (655 additions, ~31 KB, one file).
Scp'd to Mac: `/tmp/a2_reopen.patch`.

The patch adds **only** `tools/qwen_tts/test_rope_v2_reopen.cpp` — a
self-contained standalone probe that dlopen's libascendcl/libopapi, allocates
Q/K/cos/sin, runs three V2 paths plus a v1 baseline, and reports diff
statistics. No engine code changes. No CMakeLists changes (the probe's
g++ recipe is in the file header comment + commit message).

Apply on Mac (for PM review):

```
cd /Users/yuechen/home/OminiX-Ascend
git apply /tmp/a2_reopen.patch
```

Apply on ac02 (to rebuild / extend):

```
cd ~/work/OminiX-Ascend-w1   # already applied
source /usr/local/Ascend/ascend-toolkit/set_env.sh
g++ -std=c++17 -O2 -o test_rope_v2_reopen \
    tools/qwen_tts/test_rope_v2_reopen.cpp -ldl -lm
./test_rope_v2_reopen
```

---

## Recommendation for PM

1. **Update the A.2 closed contract and debug doc.** The H5a (packed-UB
   shared-stride) root-cause hypothesis is falsified by this probe on our
   own 910B4. The closing narrative should read: "V2 is numerically
   correct on our GQA shape; the original patch's 457 vs 434 frame
   divergence was a wiring bug, not a kernel-level GQA incompatibility.
   Reopening as a rewire probe focused on workspace lifetime, descriptor
   parity, and aclGraph-gate ordering." Landing spot:
   `docs/v2_rope_numerics_debug.md` — add a post-script pointing at this
   verdict doc.

2. **Before rewiring V2, diff the A.2 patch against current HEAD.** The
   patch at commit e64705b0 on the fork (or PM's local copy) has the
   exact descriptor strides, workspace allocation pattern, and aclGraph
   gate ordering that diverged. Walking that patch with the five candidates
   above (workspace, aclGraph, stride, insertion order, pos arg) is a
   <1-day desk review. If a candidate is obviously wrong, rewire with
   just that fix and measure.

3. **Deck update.** The MN brief's §Q1 "V2 RoPE reopened" narrative
   remains correct in spirit but needs a line-edit: "MoYoYoTech's cos/sin
   prep is byte-identical to ours; the delta is elsewhere. Their project
   is still the existence-proof that V2 works on stronger GQA than ours.
   We've now independently reproduced that numerical correctness on
   910B4." This is an even cleaner data point — two implementations,
   different hardware generations, same kernel, same numerical behavior.

4. **Probe-as-regression-test.** The file is 555 lines, standalone,
   dlopen-based, no engine deps. It belongs in the CI smoke set: if a
   future CANN toolkit bump ever regresses V2's GQA handling, this
   catches it in 0.5 sec of NPU time. Add to the per-PR smoke gate.

5. **Upside estimate (if rewire succeeds).** MoYoYoTech measured +17% TG
   from V2 on their 94-layer LLM. Talker is 28 layers; Qwen3-TTS wall is
   ~32 fps and RoPE is a small fraction. Realistic Talker upside:
   **+0.5 to +1.5 fps** (1.5-5%). Worth 1-2 days of rewire work, not a
   week. If the rewire bounces off a subtle CANN version issue, park
   cleanly and annotate closed-contract with this probe as evidence
   that the op itself is sound.

---

## Appendix A — raw probe output

```
[reopen] shape: Q=[1,1,16,128] K=[1,1,8,128] pos=5 theta=1e+06
[reopen] hQ fnv=054954427263bbcd  hK fnv=be13385cae322253
[reopen] cosA fnv=7625022a0d3f176d  cosB fnv=7625022a0d3f176d  cosC fnv=7625022a0d3f176d
  cosA[0..8]: +0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044
  cosB[0..8]: +0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044
  cosC[0..8]: +0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044
  cosA[64..][0..8]: +0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044
  cosB[64..][0..8]: +0.28345 -0.63086 -0.99414 -0.86523 -0.51172 -0.12793 +0.20020 +0.45044
[reopen] Prep A == Prep B (byte-identical)? YES (hypothesis falsified at host)

[reopen] === Path 1: v1 RotaryPositionEmbedding mode=0 NEOX, Prep A ===
  v1 Q_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24951 -0.41040 -0.46265 -0.44434 -0.39014
  v1 K_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24951 -0.41040 -0.46265 -0.44434 -0.39014
  v1 Q fnv=9d9f96762325fada  v1 K fnv=3538a0abe7ea4d54

[reopen] === Path 2: V2 + Prep A, layout=1 BSND ===
[v2] GetWorkspaceSize layout=1 -> status=0, ws=16777216
[v2] exec status=0
  V2A Q_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2A K_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2A Q fnv=b995e57f7f32e54a  V2A K fnv=b40259536686e722

[reopen] === Path 3: V2 + Prep B, layout=1 BSND ===
[v2] GetWorkspaceSize layout=1 -> status=0, ws=16777216
[v2] exec status=0
  V2B Q_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2B K_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2B Q fnv=b995e57f7f32e54a  V2B K fnv=b40259536686e722

[reopen] === DIFF vs v1 reference ===
V2A vs v1 Q          max_abs=0.000488  rel_l2=2.378675e-04  first_diff_idx=-1
V2A vs v1 K          max_abs=0.000488  rel_l2=2.228301e-04  first_diff_idx=-1
V2B vs v1 Q          max_abs=0.000488  rel_l2=2.378675e-04  first_diff_idx=-1
V2B vs v1 K          max_abs=0.000488  rel_l2=2.228301e-04  first_diff_idx=-1

[reopen] === Path 4: V2 prod-wiring sim — K slot in cache ===
[v2-slot] GetWorkspaceSize status=0 ws=16777216
[v2-slot] exec status=0
  V2-slot Q_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2-slot Kslot_out[0..8]: +0.42432 +0.34082 +0.03046 -0.24963 -0.41040 -0.46240 -0.44434 -0.39014
  V2-slot cache[POS-1] first 4: -99.00000 -99.00000 -99.00000 -99.00000 (want -99)
  V2-slot cache[POS+1] first 4: -99.00000 -99.00000 -99.00000 -99.00000 (want -99)

[reopen] === VERDICT ===
  V2 + Prep A: Q PASS, K PASS
  V2 + Prep B: Q PASS, K PASS
  V2 prod-slot: Q PASS, K PASS (rc=0)
    slot Q max_abs=0.000488 rel_l2=2.378675e-04  K max_abs=0.000488 rel_l2=2.228301e-04
  OVERALL: GREEN — V2 works on our GQA shape
```

(The line `OVERALL: GREEN` is narrow: green on the op-numerics axis. Overall
verdict is YELLOW because the production-wiring arm was not rerun — we only
proved the op is sound, not that a fresh rewire will land byte-identical wav.)

---

## Appendix B — budget accounting

- Wall: ~1.5 hr (under budget).
- Hosts: ac02 only. ac01 untouched (WSPOOL). Deck files untouched (HCCL-CHEAT).
- Frame-count gate: not applicable — probe is op-level, not end-to-end.
  Re-landing A.2 through a future rewire MUST gate on frame-count-identity
  per the universal Slide 9.5 rule.
- Sub-agents used: 0.
