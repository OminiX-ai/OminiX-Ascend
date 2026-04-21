# Vendor Fused-Op Landing Contract — Close the ≥32 fps Gate

## 1. Status & mandate

**Status**: NEW (2026-04-21, PM signed). Follow-on to `ACLGRAPH_CONTRACT.md`
G4 miss (31.6 fps on LONG canonical mayun xvec zh, 0.4 fps short of
≥32 fps gate).

**Origin**: Agent FO-audit (2026-04-21) grepped CANN 8.3.RC1's 717
aclnnop headers, found **three previously-unused vendor fused ops**
applicable to our CP decode path with combined fps upside of +0.75 to
+1.55 fps:
- `aclnnFFNV3` (W8 SwiGLU — 5→1 op)
- `aclnnApplyRotaryPosEmbV2` (Q+K RoPE fused)
- `aclnnInplaceAddRmsNorm` (in-place tail)

See `docs/fused_op_audit.md` for the full catalog walk + ranking.

**Ceiling claim to beat**: 31.6 fps (aclGraph ON, LONG canonical xvec
mayun zh, stock sampling). Gate: **≥ 32 fps** with byte-identical
user-ear verdict.

**PM role**: supervise; agents land each op behind an env gate with
drift + ear gates stacked. PM does not write kernel code.

## 2. Scope

**In scope**:
- Phase A: drop-in ops (`aclnnApplyRotaryPosEmbV2` + `aclnnInplaceAddRmsNorm`)
- Phase B (conditional on Phase A result): `aclnnFFNV3` with offline W8 re-pack
- Each op env-gated (`TALKER_CP_ROPE_V2=1`, `TALKER_CP_INPLACE_ADDRMSNORM=1`,
  `TALKER_CP_FFN_V3=1`)
- All env gates layer on top of existing `TALKER_CP_ACLGRAPH=1`
- Re-run G4 benchmark on LONG with each combination, report fps

**Out of scope**:
- New contract/tracks beyond these 3 ops (lower-ranked candidates from
  audit are M7 follow-on)
- Talker LLM fusion work (deferred)
- Integration with CannFusion upstream patch (external timeline)

## 3. Host plan

Primary: ac01 (port 31984), CANN 8.3.RC1. Fork main at `116876d3`
(Path C just CLOSED).

Secondary benchmarking: ac02 / ac03 on demand.

Push via patch-file mechanism per Path C convention.

## 4. Workstreams

### Phase A — Drop-in fusions (half-day, target: clear ≥32 gate)

Order by lowest-risk first so each PR lands with clean gates:

- [ ] A.1 `aclnnInplaceAddRmsNorm` wiring
  - Extend `cp_cann_symbols.cpp` to optionally resolve the symbol
  - Replace the current `aclnnAddRmsNorm → residual-copy` pair in
    W3b fusion path with in-place variant
  - Env gate: `TALKER_CP_INPLACE_ADDRMSNORM=1` (default off)
  - G3-style parity: ≤ 1 token drift on canonical xvec mayun zh
  - G4-style micro-bench: fps delta on LONG text
- [ ] A.2 `aclnnApplyRotaryPosEmbV2` wiring
  - Same symbol-resolve pattern
  - Replace current Q-RoPE + K-RoPE pair with single fused call
  - Env gate: `TALKER_CP_ROPE_V2=1`
  - Same parity + fps gates
- [ ] A.3 Combined run with both + aclGraph enabled
  - All three env vars set (A.1 + A.2 + ACLGRAPH=1)
  - G3 parity + G4 fps HARD GATE

**Phase A Gate**: fps ≥ 32 on LONG canonical xvec mayun zh OR Phase B
required. If fps ≥ 32 AND user-ear clean, ship + close contract.

### Phase B — `aclnnFFNV3` (1-1.5 days, conditional)

Only dispatched if Phase A doesn't clear the gate, OR if PM wants
stretch-goal headroom.

- [ ] B.1 Offline W8 gate∥up re-pack
  - Python script: concatenate gate_proj.weight + up_proj.weight along
    dim 0 into weight1 tensor
  - Re-compute per-channel int8 scale for concatenated weight
  - Save new `.bin` next to existing quantised weights
  - ~2 hr
- [ ] B.2 Engine wiring
  - Symbol-resolve `aclnnFFNV3`
  - Replace 5-op FFN chain (gate-W8Mm + up-W8Mm + SiLU + mul + down-W8Mm)
    with one FFNV3 call + `activation="swiglu"`
  - Env gate: `TALKER_CP_FFN_V3=1`
  - Requires aclGraph capture re-inventory (FFN ops changed)
  - ~2-4 hr
- [ ] B.3 Parity + perf gate

**Phase B Gate**: fps ≥ 32.5 AND drift ≤ 1 AND user-ear clean.

## 5. Acceptance criteria

- [ ] Phase A ships behind env gates, LONG fps ≥ 32 with combined gates
- [ ] (if Phase B) FFNV3 ships with W8 re-packed weights, adds ≥ 0.5 fps
- [ ] All env gates default off, unset run byte-identical to pre-patch
- [ ] Contract stamped with commit SHAs + Verified-by Agent G-name
- [ ] Coverage wavs delivered to PM for ear check

## 6. Risks

1. **aclnnFFNV3 dtype matrix may not include A16W8**: same risk as
   CannFusion. Mitigation: grep header for supported dtypes; if A16W8
   not in the list, Phase B is dead on arrival (same pattern).
2. **Weight re-pack breaks existing GGUF loader**: B.1 changes the
   weight file format — must keep original on side, gate the new one
   on `TALKER_CP_FFN_V3=1` runtime check.
3. **aclGraph capture invalidated by op-count change**: when FFN 5→1,
   the existing 17 pos-keyed graphs need re-capture. Mitigation: capture
   is engine-init, not hot path — negligible cost.
4. **F16 numerical drift on FFNV3 internal reduction**: vendor may
   internally use F32 accum (good) or F16 (risk). Validate by G3 drift
   gate. If drift > 1, fall back.
5. **Inplace variants may clobber buffers aclGraph assumes are
   read-only**: Phase A inplace ops need verification the captured
   graph's tensor descriptors match inplace semantics. Mitigation:
   disable aclGraph during A.1/A.2 initial wiring; re-enable after
   parity.

## 7. Host rules

- All wiring on ac01 under `~/work/OminiX-Ascend-w1/`
- Patch-file mechanism (ac01 has no fork push creds)
- No Claude coauthor
- HARD KILL at each fps/parity gate — no silent continuation
- Ear gate outranks fps numerics: a clean 31.5 fps ships; a drifty
  32.5 fps does not
