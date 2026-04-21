# CANN 8.3.RC1 Fused-Op Audit for Qwen3-TTS CP Forward

## Methodology

Grepped `/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/` (717 headers) on ac01 for fusion-style patterns: `Fus`, `AddX`, `NormX`, `Swi`, `GeGlu`, `Silu`, `Mm[A-Z]`, `QKV`, `Rope[A-Z]`, `Quant.*Matmul`, `Grouped`, `Rms.*[A-Z]`. Filtered out `*Grad*`/`*Backward*` (training-only) and image/conv ops. Cross-referenced the remaining candidates against the 20 call sites in `tools/qwen_tts/cp_cann_engine.cpp::forward_one_token_launch` (attn QKV-Mm, q/k norm, RoPE, FIAv2, O-Mm, gate/up-Mm, Silu+InplaceMul, down-Mm, pre/post-attn Add+RmsNorm W3b) and the 15 lm_head Mms in `CpCannEngine::run_lm_head_`.

Per-frame op count used for dispatch-savings arithmetic: 15 forwards/frame × 5 layers = **75 per-layer passes/frame**; 15 lm_head calls occur once per forward (15×15 = 225/frame).

## Inventory summary

| Category                                            | Count |
|-----------------------------------------------------|-------|
| Total aclnn headers (non-training)                  | 613   |
| Already used in CP engine                           | ~20   |
| Training-only (filtered by `Grad`/`Backward`)       | 104   |
| Fusion-style candidates matched                     | 46    |
| **Applicable to our chain (infer, non-MoE, non-distributed)** | **9** |
| Applicable but lower priority (enum-only or speculative) | 5 |

## Ranked kill-list

| Rank | Op | Chain replaced | Applicable at | ops saved/frame | Est. fps Δ | Confidence |
|------|-----|----------------|---------------|-----------|-----------|-----------|
| 1 | **aclnnFFNV3** | W8-Mm(gate) + W8-Mm(up) + Silu + InplaceMul + W8-Mm(down) | FFN sublayer (every layer, every forward) | 5 × 75 = **375 dispatches** + sizeable HBM save | **+0.5 – 1.2** | **med-high** |
| 2 | **aclnnApplyRotaryPosEmbV2** (in-place on Q+K) | Two separate `aclnnRotaryPositionEmbedding` (Q, K) | attn, every layer (W3b-exit already sets norms; RoPE stays 2 ops today) | 1 × 75 = **75 dispatches** | **+0.2** | high |
| 3 | **aclnnInplaceAddRmsNorm** | Existing `aclnnAddRmsNorm` (we allocate a separate xOut buffer today) | post-attn + post-FFN tail W3b | 0 (same op count, but saves `aclnn_copy_cp_` residual-Add copy afterwards, 1×75=75 calls) | **+0.1 – 0.2** | high |
| 4 | aclnnSwiGlu (gate/up interleaved layout) | Silu + InplaceMul — only if we switch to interleaved gate/up tensor | FFN activation | 2 × 75 = 150 dispatches | +0.1 | low (requires changing FFN layout) |
| 5 | aclnnFusedMatmul | Mm + Add / Mm + Mul via `fusedOpType` enum | Q/K/V/O/Down-Mm (no bias in Qwen3 → epilogue limited) | speculative | 0 | low (no-bias chain) |
| 6 | aclnnAddRmsNormCast | AddRmsNorm + F16→F32 cast | Final-layer tail (1× per forward) | 1 × 15 = 15 | +0.02 | med |
| 7 | aclnnScatterPaKvCache | memcpy V→slot, K→slot (already eliminated via slot-view desc) | not applicable (slot-view desc already folds this in) | 0 | 0 | — |
| 8 | aclnnFusedInferAttentionScoreV4 | newer FIAv2 (we're on v2) | attn | — | unknown | low (API changed; untested) |
| 9 | aclnnDequantRopeQuantKvcache | Dequant + RoPE + Quant + KVcache write — MLA path | MLA models only (Qwen3-TTS is GQA, not MLA) | N/A | 0 | — |

### Rationale for the top three

- **#1 aclnnFFNV3** — Signature accepts `x, weight1, weight2, ..., activation="swiglu"`, int8 weight with `deqScale1/2` per-channel. Qwen3-TTS W8 FFN is exactly this pattern (gate_proj_w_i8 ∥ up_proj_w_i8 → silu*up → down_proj_w_i8). This collapses **5 separate aclnn calls → 1** and keeps gate/up/down intermediates in L1/L2 instead of round-tripping HBM, which is the real win (HBM BW, not dispatch latency). Caveat: the header documents **gate+up packed into `weight1` as a concatenated `[K1, 2*N1]` tensor** — requires offline re-packing of the two W8 matrices + concatenating the two `deqScale` vectors. That's a ~1-2 hr offline script, not runtime work.
- **#2 aclnnApplyRotaryPosEmbV2** — Signature takes `queryRef, keyRef, cos, sin, layout, rotaryMode` and updates Q/K in-place with a **single** kernel. Current code issues two separate `aclnnRotaryPositionEmbedding` calls. Drop-in if tensor layouts line up (they do: Q is `[1,1,n_heads,head_dim]`, K is `[1,1,n_kv,head_dim]`, same dtype).
- **#3 aclnnInplaceAddRmsNorm** — Signature: `x1Ref, x2Ref, gamma` where x1 is updated in-place (= residual + ffn_out → residual). We currently use `aclnnAddRmsNorm` which writes to a separate `xOut` and then we do `aclnn_copy_cp_(residual, cur)` on the next iteration. In-place version skips that copy entirely; saves 75 aclnnAdd zero-copies per frame.

## Notable gaps (ops we wish existed but don't, at 8.3.RC1)

- **`aclnnRmsNormRope`** — not found. Post-attn norm → RoPE chain stays as 2 ops.
- **`aclnnRmsNormQKVMatmul`** — not found. 8.3.RC1 has `aclnnSwinTransformerLnQkvQuant` (Swin-specific, wrong QKV-split layout) but no general RmsNorm+Q+K+V triple-Mm fusion. This was Path C's wishlist op; confirmed absent.
- **`aclnnFusedRopeFIA`** — not found. RoPE-K + KVcache-write + FIA remains 2 ops (already merged via slot-view descriptor).
- **`aclnnMmSilu` / `aclnnMmMul`** — not found as standalone. `aclnnFusedMatmul` exists but its `fusedOpType` enum (from header) supports only Mm+bias+x3 epilogue patterns, not elementwise-activation+mul. Qwen3 has no bias → no epilogue hit.
- **`aclnnMmList`** / batched same-x multi-Mm — not found. Q/K/V remain 3 dispatches; packing into a single `[K, Q+KV+KV]` matrix would require offline pack.

## Top 3 candidates — recommended next action

### #1: aclnnFFNV3 (FFN collapse)
- Header: `aclnn_ffn_v3.h`
- Kills: 5 ops × 75 per-layer passes = **375 dispatches/frame**, plus HBM savings on inter/gate/up intermediates (~16 MB × 75 round-trips/frame eliminated)
- Expected fps: **+0.5 to +1.2** — this alone is likely to clear the 32 fps gate
- Risk: (a) int8 layout: need to pack `[gate_w, up_w]` into `weight1` column-concatenated; (b) per-channel deqScale for two matrices must be concatenated the same way; (c) `activation="swiglu"` behavior must parity-match our manual `silu(gate) * up`; (d) aclGraph capturability of aclnnFFNV3 is untested (FFN-v1 known to capture per CANN release notes)
- Integration cost: **4–6 hours** (2 hr offline re-pack script for gate+up → fused W8 tensor, 2 hr dlsym wiring + one new call site, 2 hr parity test)

### #2: aclnnApplyRotaryPosEmbV2 (Q+K fused RoPE)
- Header: `aclnn_apply_rotary_pos_emb_v2.h`
- Kills: **75 dispatches/frame** (two RoPE calls → one)
- Expected fps: **+0.15 to +0.25**
- Risk: low — same cos/sin tensor reused, both Q and K are updated in place (we already use strided-view K-to-slot, just pass the slot-view as keyRef)
- Integration cost: **1.5 hours** (dlsym + replace the two `CANN_OP(RotaryPositionEmbedding, ...)` calls with one)

### #3: aclnnInplaceAddRmsNorm (skip residual copy)
- Header: `aclnn_inplace_add_rms_norm.h`
- Kills: **75 aclnnAdd copies/frame** (the `aclnn_copy_cp_(residual, cur)` after each AddRmsNorm)
- Expected fps: **+0.10 to +0.20**
- Risk: need to verify x1Ref-updated-in-place semantics don't break the subsequent `aclnnAdd(residual, zero)` pattern — likely trivial
- Integration cost: **1–2 hours**

**Combined ceiling**: +0.75 to +1.55 fps. Just **#2 + #3** (easy wins, low risk, ~3 hours) is +0.25 to +0.45 — exactly the closing margin for the ≥32 gate without touching FFN packing.

## Recommendation for PM

Green-light a focused follow-on in this order:

1. **Phase A (≤ half day)**: Land #2 and #3 together — both are drop-in replacements, no weight re-packing, no aclGraph-capture risk beyond what we already run. Expected: **+0.25 to +0.45 fps**, which on its own is likely to carry us from 31.6 → ≥32.0. Ship at that point.
2. **Phase B (1–1.5 days, only if Phase A falls short or we want headroom)**: Land #1 (aclnnFFNV3). Requires offline W8 re-pack and capture-parity test. Upside is another +0.5 to +1.2 fps for a total of 32.5 – 33.0 fps — comfortable margin plus future-proofs for longer sequences.

If Phase A lands at ≥32 fps, I recommend **shipping at that bar and scheduling Phase B as a M7 follow-on** rather than blocking this milestone. Phase A is lower-risk, shorter, and hits the gate; Phase B is a separate optimization epic with its own parity-test surface area.
