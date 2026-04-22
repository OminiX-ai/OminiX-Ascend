# A16W8-specific fused-op supplement audit

## Method
Read-only header greps on ac01 `/usr/local/Ascend/ascend-toolkit/latest/include/aclnnop/` (717 headers, confirms FO-audit baseline). Filtered with `grep -iE "moe|quant.*matmul|dequant|incre|mmad|groupquant"` (63 hits) plus secondary scans for `ffn`, `swiglu`, `rope`, `gelu_quant`. For each candidate, read the declared dtype line for `x`/`weight`, checked whether `expertTokens` / `groupList` appears as a required parameter, and evaluated epilogue chain and shape fit against our CP forward (M=1, K=1024, N=3072 for down-proj; K=1024, N=4864 for gate/up).

## Candidates found

### aclnnWeightQuantBatchMatmulV3 (header: `aclnn_weight_quant_batch_matmul_v3.h`)
- A16W8 support: **Y** — by construction: `x` fp16/bf16 activation, `weight` INT8, `antiquantScale` fp16/bf16 (per-channel or per-group), `antiquantOffset` optional. This is the canonical A16W8 primitive on Ascend.
- expertTokens required: **N** — parameter does not exist; this is a non-MoE op entirely separate from the `aclnnFFN*` family that tripped Finding 3.
- Shape constraints: standard matmul M/K/N; `antiquantGroupSize` for per-group quant. No hardcoded MoE batching. M=1 single-token is supported.
- Applicable to our CP chain: **Y** — replaces the three A16W8 GEMMs in the CP block (gate_up, down_proj, attn_out). Epilogue is bias-only, so SwiGLU must stay as a separate kernel launch (or composed with `aclnnDequantSwigluQuant` downstream of a quant-out matmul — separate path).
- Integration cost estimate: **4-6 hours** — straight drop-in for `nn.Linear(int8, fp16_act)`; dlsym path identical to existing CP fused-op stubs; plumb `antiquantScale`/`antiquantOffset` from the packed A16W8 weight buffer we already load.

### aclnnQuantMatmulDequant (header: `aclnn_quant_matmul_dequant.h`)
- A16W8 support: **partial** — `x` is fp16 activation, `weight` INT8, `weightScale` fp32/int64. But it internally quantizes x on-the-fly (W8A8 semantics with dynamic activation quant), not the static A16W8 keep-activation-in-fp16 lane we need. Output is fp16.
- expertTokens required: **N**.
- Shape constraints: standard M/K/N; supports NZ weight layout.
- Applicable: **N** — introduces activation requantization error we specifically avoided by choosing A16W8. Only usable if we accept W8A8 numerics.
- Cost: moot.

### aclnnIncreFlashAttentionV4 (header: `aclnn_incre_flash_attention_v4.h`)
- A16W8 support: **N for Q/K/V projections** — this op is the attention score kernel itself (Q·Kᵀ, softmax, ·V), not a projection matmul. It has `antiquantScale`/`antiquantOffset` inputs for the KV-cache antiquant path (fp16 Q vs int8 KV-cache), distinct from our weight quant.
- expertTokens required: **N** — has `numKeyValueHeads` field, explicitly supports **GQA** (solves the packed-UB MHA-only problem from `aclnnApplyRotaryPosEmbV2`).
- Shape constraints: M=1 by design (incremental decode); `numHeads=16`, `numKeyValueHeads=8` fits Qwen3-TTS directly.
- Applicable: **Y** — different layer (attention, not projection) but solves the GQA blocker separately; composes with WQBMMv3 for Q/K/V/O projections.
- Integration cost: **8-12 hours** — V4 adds blocktable+kvPaddingSize paged-KV support; wiring paged cache is the bulk of the cost.

## Conclusion
- [x] **FOUND**: `aclnnWeightQuantBatchMatmulV3` is a documented, non-MoE, non-deprecated A16W8 GEMM that accepts INT8 weight + fp16 activation with no `expertTokens` path. `aclnnIncreFlashAttentionV4` separately unblocks GQA attention with M=1 incremental decode. Together they cover the CP forward's matmul and attention sublayers.
- [ ] NONE

## Recommendation for PM
Wire WQBMMv3 for all three A16W8 projections in the CP block before continuing to chase a monolithic fused path. Keep SwiGLU as a separate `aclnnSwiGluV2` launch for now. If this works, add IFAv4 as a second step for the GQA attention sublayer. This is a two-op composition, not a single fused op — but it exits the MoE dispatch trap entirely.
