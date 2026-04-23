#pragma once
// ============================================================================
// Runtime-resolved CANN symbol table.
//
// We do NOT link against libascendcl/libnnopbase/libopapi at build time to
// avoid the circular-dependency issues between CANN's stub libs and the ops
// that ggml-cann loads dynamically. Instead, we dlopen() the three libs on
// first use and resolve each function pointer via dlsym. Callers use the
// pointers in `g_cann` exactly as they would use the direct functions.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>

// Function-pointer table. All members are null-initialized by the global
// `g_cann`; cp_cann_load_symbols() fills them in. Check is_ready() before use.
struct CannSyms {
    // ---- runtime (libascendcl.so) ----
    aclError (*aclInit)(const char *);
    aclError (*aclrtSetDevice)(int32_t);
    aclError (*aclrtCreateStream)(aclrtStream *);
    aclError (*aclrtDestroyStream)(aclrtStream);
    aclError (*aclrtSynchronizeStream)(aclrtStream);
    aclError (*aclrtMalloc)(void **, size_t, aclrtMemMallocPolicy);
    aclError (*aclrtFree)(void *);
    aclError (*aclrtMemcpy)(void *, size_t, const void *, size_t,
                            aclrtMemcpyKind);
    aclError (*aclrtMemcpyAsync)(void *, size_t, const void *, size_t,
                                  aclrtMemcpyKind, aclrtStream);
    const char *(*aclGetRecentErrMsg)();

    // ---- Event APIs (multi-stream sync, M6) -------------------------------
    // Used to fence one stream against another so Talker[N+1] on stream A can
    // wait for CP[N] on stream B without host roundtrips. Required by the
    // multi-stream pipeline; callers that don't create secondary streams can
    // ignore these.
    aclError (*aclrtCreateEvent)(aclrtEvent *);
    aclError (*aclrtDestroyEvent)(aclrtEvent);
    aclError (*aclrtRecordEvent)(aclrtEvent, aclrtStream);
    aclError (*aclrtStreamWaitEvent)(aclrtStream, aclrtEvent);
    aclError (*aclrtSynchronizeEvent)(aclrtEvent);

    // ---- aclnn base (libnnopbase.so) ----
    aclTensor *(*aclCreateTensor)(const int64_t *viewDims, uint64_t viewDimsNum,
                                   aclDataType dataType, const int64_t *stride,
                                   int64_t offset, aclFormat format,
                                   const int64_t *storageDims,
                                   uint64_t storageDimsNum, void *tensorData);
    aclnnStatus (*aclDestroyTensor)(const aclTensor *);
    aclScalar *(*aclCreateScalar)(void *, aclDataType);
    aclnnStatus (*aclDestroyScalar)(const aclScalar *);
    // IntArray helpers (needed for aclnnLayerNorm normalizedShape, and
    // future ops that take int-array metadata).
    aclIntArray *(*aclCreateIntArray)(const int64_t *value, uint64_t size);
    aclnnStatus (*aclDestroyIntArray)(const aclIntArray *array);

    // ---- aclnn ops (libopapi.so) ----
    aclnnStatus (*aclnnMmGetWorkspaceSize)(const aclTensor *, const aclTensor *,
                                            aclTensor *, int8_t, uint64_t *,
                                            aclOpExecutor **);
    aclnnStatus (*aclnnMm)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnAddGetWorkspaceSize)(const aclTensor *, const aclTensor *,
                                             const aclScalar *, aclTensor *,
                                             uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnAdd)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnInplaceAddGetWorkspaceSize)(const aclTensor *,
                                                    const aclTensor *,
                                                    const aclScalar *,
                                                    uint64_t *,
                                                    aclOpExecutor **);
    aclnnStatus (*aclnnInplaceAdd)(void *, uint64_t, aclOpExecutor *,
                                    aclrtStream);

    aclnnStatus (*aclnnInplaceMulGetWorkspaceSize)(aclTensor *,
                                                    const aclTensor *,
                                                    uint64_t *,
                                                    aclOpExecutor **);
    aclnnStatus (*aclnnInplaceMul)(void *, uint64_t, aclOpExecutor *,
                                    aclrtStream);

    aclnnStatus (*aclnnSiluGetWorkspaceSize)(const aclTensor *, aclTensor *,
                                              uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnSilu)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    aclnnStatus (*aclnnRmsNormGetWorkspaceSize)(const aclTensor *,
                                                 const aclTensor *, double,
                                                 const aclTensor *,
                                                 const aclTensor *,
                                                 uint64_t *,
                                                 aclOpExecutor **);
    aclnnStatus (*aclnnRmsNorm)(void *, uint64_t, aclOpExecutor *,
                                 aclrtStream);

    // ---- Fused Add + RmsNorm (W3b, CANN 8.5+) ------------------------------
    // Collapses the `cur = residual + output; residual = cur; normed =
    // rmsnorm(cur, gamma)` tail into one kernel. Signature matches
    // aclnn_add_rms_norm.h:
    //     xOut = x1 + x2
    //     yOut = RmsNorm(xOut, gamma, eps)
    //     rstdOut = 1 / sqrt(mean(xOut^2) + eps)
    // Inputs/outputs are all F16 (gamma is F32 per the existing RmsNorm
    // convention). x1 and xOut must NOT alias — the non-inplace variant does
    // not support in-place updates. Callers wanting inplace semantics should
    // dispatch `aclnnInplaceAddRmsNorm` instead. Capability flag:
    // `has_add_rms_norm()`; absence means toolkit predates CANN 8.5 and
    // callers keep the unfused Add + RmsNorm path.
    aclnnStatus (*aclnnAddRmsNormGetWorkspaceSize)(
        const aclTensor *x1, const aclTensor *x2, const aclTensor *gamma,
        double epsilon, const aclTensor *yOut, const aclTensor *rstdOut,
        const aclTensor *xOut, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnAddRmsNorm)(void *workspace, uint64_t workspaceSize,
                                    aclOpExecutor *executor,
                                    aclrtStream stream);

    // ---- Inplace Add + RmsNorm (Phase A.1, CANN 8.3+) ----------------------
    // In-place sibling of aclnnAddRmsNorm. Per aclnn_inplace_add_rms_norm.h:
    //     x2Ref  <- x1 + x2            (becomes the updated residual)
    //     x1Ref  <- RmsNorm(x1+x2, gamma, eps)  (becomes the normed tensor)
    //     rstdOut <- 1 / sqrt(mean((x1+x2)^2) + eps)
    // i.e. the x1 slot is overwritten with yOut and the x2 slot is overwritten
    // with xOut — no separate output buffers, no residual-copy needed on the
    // caller side. Dtype / gamma conventions match the non-inplace variant
    // (F16 x1Ref/x2Ref/gamma, F32 rstdOut). Gated by has_inplace_add_rms_norm().
    aclnnStatus (*aclnnInplaceAddRmsNormGetWorkspaceSize)(
        aclTensor *x1Ref, aclTensor *x2Ref, const aclTensor *gamma,
        double epsilon, const aclTensor *rstdOut,
        uint64_t *workspaceSize, aclOpExecutor **executor);
    aclnnStatus (*aclnnInplaceAddRmsNorm)(void *workspace,
                                           uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           aclrtStream stream);

    // Attention-on-NPU ops (added for the v2 engine — no host roundtrips).
    aclnnStatus (*aclnnBatchMatMulGetWorkspaceSize)(const aclTensor *,
                                                     const aclTensor *,
                                                     aclTensor *, int8_t,
                                                     uint64_t *,
                                                     aclOpExecutor **);
    aclnnStatus (*aclnnBatchMatMul)(void *, uint64_t, aclOpExecutor *,
                                     aclrtStream);

    aclnnStatus (*aclnnSoftmaxGetWorkspaceSize)(const aclTensor *, int64_t,
                                                 aclTensor *, uint64_t *,
                                                 aclOpExecutor **);
    aclnnStatus (*aclnnSoftmax)(void *, uint64_t, aclOpExecutor *,
                                 aclrtStream);

    aclnnStatus (*aclnnMulsGetWorkspaceSize)(const aclTensor *,
                                              const aclScalar *, aclTensor *,
                                              uint64_t *, aclOpExecutor **);
    aclnnStatus (*aclnnMuls)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // ---- Mul (elementwise, broadcast-capable) ------------------------------
    // Required by QIE Q2.3 block forward for the modulation `x * (1 + scale)`
    // multiply where `scale` is [B, hidden] and broadcasts over seq. aclnnMul
    // follows numpy-style broadcast per aclnn_mul.h.
    aclnnStatus (*aclnnMulGetWorkspaceSize)(const aclTensor *self,
                                             const aclTensor *other,
                                             aclTensor *out,
                                             uint64_t *workspaceSize,
                                             aclOpExecutor **executor);
    aclnnStatus (*aclnnMul)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // ---- LayerNorm (affine-off supported via nullptr gamma/beta) -----------
    // QIE DiT uses LayerNorm with `affine=false` on img_norm1/2 + txt_norm1/2
    // — per `tools/ominix_diffusion/src/qwen_image.hpp:205-213`. Signature
    // follows aclnn_layer_norm.h: gamma/beta passed as optional tensors so
    // callers can pass nullptr for the affine-off case. `normalizedShape`
    // is an aclIntArray giving the trailing dims to reduce over.
    aclnnStatus (*aclnnLayerNormGetWorkspaceSize)(
        const aclTensor *input, const aclIntArray *normalizedShape,
        const aclTensor *weightOptional, const aclTensor *biasOptional,
        double eps, aclTensor *out, aclTensor *meanOutOptional,
        aclTensor *rstdOutOptional, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnLayerNorm)(void *workspace, uint64_t workspaceSize,
                                   aclOpExecutor *executor, aclrtStream stream);

    // ---- GELU (approximate="tanh" to match ggml CPU reference) -------------
    // QIE DiT FFN activation is GELU. The CPU reference (tools/ominix_diffusion
    // via ggml_gelu) uses the tanh approximation
    //   0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    // aclnnGeluV2 takes an `approximate` string parameter accepting "none"
    // (exact erf form) or "tanh" (matches ggml). We dispatch V2 with "tanh"
    // for numerical parity with CPU reference in Q2.3 smoke.
    //
    // Older CANN without V2: dispatch aclnnGelu (defaults to exact erf on
    // current 8.3; will drift from CPU reference by ~1e-3 cos_sim). Callers
    // gate on has_gelu_v2(); absence triggers the "warn + exact" fallback.
    aclnnStatus (*aclnnGeluV2GetWorkspaceSize)(const aclTensor *self,
                                                int64_t approximate,
                                                aclTensor *out,
                                                uint64_t *workspaceSize,
                                                aclOpExecutor **executor);
    aclnnStatus (*aclnnGeluV2)(void *, uint64_t, aclOpExecutor *, aclrtStream);
    aclnnStatus (*aclnnGeluGetWorkspaceSize)(const aclTensor *self,
                                              aclTensor *out,
                                              uint64_t *workspaceSize,
                                              aclOpExecutor **executor);
    aclnnStatus (*aclnnGelu)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // ---- Sigmoid (used in some paths; QIE uses gate direct-multiply so this
    //     is parked but resolved for convenience). ---------------------------
    aclnnStatus (*aclnnSigmoidGetWorkspaceSize)(const aclTensor *self,
                                                 aclTensor *out,
                                                 uint64_t *workspaceSize,
                                                 aclOpExecutor **executor);
    aclnnStatus (*aclnnSigmoid)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // Non-destructive rotary position embedding (single tensor at a time).
    // `mode`: 0 = NEOX half-rotated, 1 = GPT-J interleaved. We use mode 0.
    // This is what ggml-cann invokes for NEOX-style models.
    aclnnStatus (*aclnnRotaryPositionEmbeddingGetWorkspaceSize)(
        const aclTensor *x, const aclTensor *cos, const aclTensor *sin,
        int64_t mode, aclTensor *out, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnRotaryPositionEmbedding)(void *, uint64_t,
                                                 aclOpExecutor *,
                                                 aclrtStream);

    // ---- Fused in-place Q+K rotary position embedding (Phase A.2, CANN 8.3+) --
    // aclnnApplyRotaryPosEmbV2 fuses two aclnnRotaryPositionEmbedding calls
    // (one for Q, one for K) into a single op with in-place writes back into
    // the queryRef / keyRef buffers. Matches NEOX rotation when invoked with
    // rotaryMode="half" and half-duplicated cos/sin tables (see probe at
    // tools/qwen_tts/test_rope_v2_reopen.cpp for byte-level equivalence).
    //
    // Layout semantics:
    //   layout=0: BNSD (batch, n_heads, seq, head_dim)
    //   layout=1: BSND (batch, seq, n_heads, head_dim)  <- we use this
    //
    // GQA is supported natively: Q and K may have different n_heads. Probe
    // result on 16Q/8KV (talker shape): max_abs vs v1 NEOX = 4.88e-4 = 1 F16
    // ulp on both Q and K outputs.
    //
    // Q and K are both in-place: queryRef and keyRef are the input AND output
    // buffers. Callers must ensure there are no concurrent reads of the Q/K
    // slots on the same stream until the op completes.
    //
    // Capability flag: has_rope_v2(); absence means the toolkit predates the
    // V2 op and callers stay on the two-call aclnnRotaryPositionEmbedding
    // path. Gated by TALKER_CP_ROPE_V2=1 at runtime (Phase A.2).
    aclnnStatus (*aclnnApplyRotaryPosEmbV2GetWorkspaceSize)(
        aclTensor *queryRef, aclTensor *keyRef,
        const aclTensor *cos, const aclTensor *sin,
        int64_t layout, char *rotaryMode,
        uint64_t *workspaceSize, aclOpExecutor **executor);
    aclnnStatus (*aclnnApplyRotaryPosEmbV2)(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             aclrtStream stream);

    // Dtype conversion (used to cast the F32 input embedding down to F16 at
    // engine entry, and the F16 final hidden back to F32 at exit).
    aclnnStatus (*aclnnCastGetWorkspaceSize)(const aclTensor *self,
                                              aclDataType dtype,
                                              aclTensor *out,
                                              uint64_t *workspaceSize,
                                              aclOpExecutor **executor);
    aclnnStatus (*aclnnCast)(void *, uint64_t, aclOpExecutor *, aclrtStream);

    // Strided tensor copy (supports non-contiguous tensors). Used by the
    // Phase 4.1 on-device RoPE scatter-write to avoid relying on aclnnAdd
    // accepting strided output views.
    aclnnStatus (*aclnnInplaceCopyGetWorkspaceSize)(aclTensor *selfRef,
                                                     const aclTensor *src,
                                                     uint64_t *workspaceSize,
                                                     aclOpExecutor **executor);
    aclnnStatus (*aclnnInplaceCopy)(void *, uint64_t, aclOpExecutor *,
                                     aclrtStream);

    // TensorList creation / destruction — used to wrap a single K/V tensor for
    // the FusedInferAttention op (which takes lists to support chunked KV).
    aclTensorList *(*aclCreateTensorList)(const aclTensor *const *value,
                                           uint64_t size);
    aclnnStatus (*aclDestroyTensorList)(const aclTensorList *);

    // Mega-fused attention — the exact op llama.cpp/ggml-cann uses. Replaces
    // my manual BMM+Muls+Softmax+BMM path. Natural BSND layout, GQA handled
    // internally via numKeyValueHeads, softmax in F32 regardless of input
    // precision. See aclnn_fused_infer_attention_score_v2.h for the full
    // parameter list.
    aclnnStatus (*aclnnFusedInferAttentionScoreV2GetWorkspaceSize)(
        const aclTensor *query, const aclTensorList *key,
        const aclTensorList *value, const aclTensor *pseShiftOptional,
        const aclTensor *attenMaskOptional,
        const aclIntArray *actualSeqLengthsOptional,
        const aclIntArray *actualSeqLengthsKvOptional,
        const aclTensor *deqScale1Optional, const aclTensor *quantScale1Optional,
        const aclTensor *deqScale2Optional, const aclTensor *quantScale2Optional,
        const aclTensor *quantOffset2Optional,
        const aclTensor *antiquantScaleOptional,
        const aclTensor *antiquantOffsetOptional,
        const aclTensor *blockTableOptional,
        const aclTensor *queryPaddingSizeOptional,
        const aclTensor *kvPaddingSizeOptional,
        const aclTensor *keyAntiquantScaleOptional,
        const aclTensor *keyAntiquantOffsetOptional,
        const aclTensor *valueAntiquantScaleOptional,
        const aclTensor *valueAntiquantOffsetOptional,
        const aclTensor *keySharedPrefixOptional,
        const aclTensor *valueSharedPrefixOptional,
        const aclIntArray *actualSharedPrefixLenOptional, int64_t numHeads,
        double scaleValue, int64_t preTokens, int64_t nextTokens,
        char *inputLayout, int64_t numKeyValueHeads, int64_t sparseMode,
        int64_t innerPrecise, int64_t blockSize, int64_t antiquantMode,
        bool softmaxLseFlag, int64_t keyAntiquantMode,
        int64_t valueAntiquantMode, const aclTensor *attentionOut,
        const aclTensor *softmaxLse, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnFusedInferAttentionScoreV2)(void *, uint64_t,
                                                    aclOpExecutor *,
                                                    aclrtStream);

    // ---- FRACTAL_NZ weight pre-conversion (M5) ------------------------------
    // aclnnTransMatmulWeight refreshes a matmul weight tensor in-place to the
    // private FRACTAL_NZ layout that the hardware prefers for the subsequent
    // aclnnMm / aclnnMatMul call. The tensor buffer is reused (no new alloc);
    // only the format descriptor + internal tiling changes. Callers that want
    // to pre-bake weights gate on has_nz(); absence of the symbol (older CANN
    // toolkits) means "leave everything ND" and silently fall back.
    // Signatures per aclnnop/aclnn_trans_matmul_weight.h (CANN 8.3+).
    aclnnStatus (*aclnnTransMatmulWeightGetWorkspaceSize)(
        aclTensor *mmWeightRef, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnTransMatmulWeight)(void *workspace, uint64_t workspaceSize,
                                           aclOpExecutor *executor,
                                           aclrtStream stream);

    // ---- aclnnMatmulWeightNz (M5.3, CANN 8.5+) -----------------------------
    // Explicit NZ matmul: self = activation (ND), mat2 = weight (NZ layout,
    // pre-converted via aclnnTransMatmulWeight). Computes self @ mat2 just
    // like aclnnMm, but actually consumes the NZ-laid-out weight buffer
    // instead of silently treating it as ND (which is what plain aclnnMm does
    // on CANN 8.3 and produces garbled output). Signature identical to
    // aclnnMm / aclnnMmGetWorkspaceSize. Per aclnn_matmul.h in CANN 8.5.
    // Capability gated via has_matmul_weight_nz().
    aclnnStatus (*aclnnMatmulWeightNzGetWorkspaceSize)(
        const aclTensor *self, const aclTensor *mat2, aclTensor *out,
        int8_t cubeMathType, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnMatmulWeightNz)(void *workspace, uint64_t workspaceSize,
                                        aclOpExecutor *executor,
                                        aclrtStream stream);

    // ---- aclnnBatchMatMulWeightNz (M5.3, CANN 8.5+) -------------------------
    // Batched NZ matmul — reserved for future multi-batch paths. Not used by
    // the single-token decode / single-batch prefill matmul call sites but
    // surfaced here for symmetry so callers that grow into batched matmuls
    // can check has_matmul_weight_nz() and dispatch the batched variant
    // without another symbol-table bump. Per aclnn_batch_matmul.h in CANN 8.5.
    aclnnStatus (*aclnnBatchMatMulWeightNzGetWorkspaceSize)(
        const aclTensor *self, const aclTensor *mat2, aclTensor *out,
        int8_t cubeMathType, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnBatchMatMulWeightNz)(void *workspace,
                                             uint64_t workspaceSize,
                                             aclOpExecutor *executor,
                                             aclrtStream stream);

    // ---- aclnnWeightQuantBatchMatmulV3 (Stretch S1, CANN 8.5+) ---------------
    // A16W8 pseudo-quantized matmul. The activation `x` stays F16/BF16 and the
    // weight `weight` is INT8 (per-output-channel scales in `antiquantScale`,
    // optional per-channel offsets in `antiquantOffsetOptional`). Computes
    // y = x @ dequant(weight, antiquantScale, antiquantOffsetOptional) in
    // a fused kernel — no host-side dequant pass. Scale/offset tensors must
    // match y's dtype (F16 here; we use symmetric quantization so offset is
    // null). See aclnn_weight_quant_batch_matmul_v3.h in CANN 8.5.
    //
    // V3 differs from V2 by taking an extra `innerPrecise` parameter between
    // `antiquantGroupSize` and `y`. We resolve both; callers prefer V3 when
    // available and fall back to V2 if V3 misbehaves.
    //
    // Capability gated via has_w8_quant().
    aclnnStatus (*aclnnWeightQuantBatchMatmulV3GetWorkspaceSize)(
        const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale,
        const aclTensor *antiquantOffsetOptional,
        const aclTensor *quantScaleOptional,
        const aclTensor *quantOffsetOptional,
        const aclTensor *biasOptional,
        int antiquantGroupSize, int innerPrecise,
        const aclTensor *y, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnWeightQuantBatchMatmulV3)(void *workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor *executor,
                                                  aclrtStream stream);

    aclnnStatus (*aclnnWeightQuantBatchMatmulV2GetWorkspaceSize)(
        const aclTensor *x, const aclTensor *weight,
        const aclTensor *antiquantScale,
        const aclTensor *antiquantOffsetOptional,
        const aclTensor *quantScaleOptional,
        const aclTensor *quantOffsetOptional,
        const aclTensor *biasOptional,
        int antiquantGroupSize,
        const aclTensor *y, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnWeightQuantBatchMatmulV2)(void *workspace,
                                                  uint64_t workspaceSize,
                                                  aclOpExecutor *executor,
                                                  aclrtStream stream);

    // ---- aclnnGroupedMatmulV3 (Phase GMM-wire, CANN 8.3+) ------------------
    // Fused multi-group matmul. Canonical contract: y[i] = x[i] @ weight[i].
    // For our QKV projection use we pass three "groups" sharing the same
    // activation (three distinct aclTensor views of the same device buffer)
    // and three distinct weights w_q / w_k / w_v — then the op emits one
    // kernel per group with the group scheduling decided on-device.
    //
    // Probe finding (docs/qkv_grouped_probe_verdict.md, GREEN 2026-04):
    //   - A16W8 accepted when groupType=-1 + groupListOptional=nullptr.
    //   - antiquantOffset is Optional in the header but
    //     CheckGroupedMatmulAntiQuant runtime check requires non-null. Pass a
    //     shared zero-filled F16 offset tensor list (one per group).
    //   - ~94-100 μs median vs ~98-102 μs for 3 × aclnnWeightQuantBatchMatmulV3.
    //   - Numerical drift 1.53e-5 .. 6.13e-5 abs (~1-2 F16 ulp, ~1.7e-3 rel)
    //     vs 3× WQBMMv3 reference — NOT bit-exact; non-canonical fusion.
    //
    // Capability gated via has_grouped_matmul_v3().
    aclnnStatus (*aclnnGroupedMatmulV3GetWorkspaceSize)(
        const aclTensorList *x, const aclTensorList *weight,
        const aclTensorList *biasOptional,
        const aclTensorList *scaleOptional,
        const aclTensorList *offsetOptional,
        const aclTensorList *antiquantScaleOptional,
        const aclTensorList *antiquantOffsetOptional,
        const aclTensor *groupListOptional,
        int64_t splitItem, int64_t groupType,
        const aclTensorList *y, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnGroupedMatmulV3)(void *workspace, uint64_t workspaceSize,
                                         aclOpExecutor *executor,
                                         aclrtStream stream);

    // ---- aclnnFFNV3 (Phase B, CANN 8.5+) -----------------------------------
    // Fused FFN: y = activation(x @ W1 + b1) @ W2 + b2.
    // For our TTS CP path we dispatch it as the no-expert SwiGLU variant:
    //   x:              F16 [1, cp_hidden]
    //   weight1:        INT8 [cp_hidden, 2*inter]  (gate||up concatenated on N)
    //   weight2:        INT8 [inter, cp_hidden]    (down_proj)
    //   antiquantScale1: F16 [2*inter]             (per-channel, gate||up concat)
    //   antiquantScale2: F16 [cp_hidden]           (per-channel, down_proj)
    //   activation:     "swiglu"
    //   y:              F16 [1, cp_hidden]
    // This replaces the 5-op chain (gate-Mm + up-Mm + SiLU + mul + down-Mm)
    // behind the TALKER_CP_FFN_V3=1 env gate. Capability flag: has_ffn_v3().
    aclnnStatus (*aclnnFFNV3GetWorkspaceSize)(
        const aclTensor *x, const aclTensor *weight1, const aclTensor *weight2,
        const aclTensor *expertTokensOptional,
        const aclTensor *bias1Optional, const aclTensor *bias2Optional,
        const aclTensor *scaleOptional, const aclTensor *offsetOptional,
        const aclTensor *deqScale1Optional, const aclTensor *deqScale2Optional,
        const aclTensor *antiquantScale1Optional,
        const aclTensor *antiquantScale2Optional,
        const aclTensor *antiquantOffset1Optional,
        const aclTensor *antiquantOffset2Optional,
        const char *activation, int64_t innerPrecise, bool tokensIndexFlag,
        const aclTensor *y, uint64_t *workspaceSize,
        aclOpExecutor **executor);
    aclnnStatus (*aclnnFFNV3)(void *workspace, uint64_t workspaceSize,
                               aclOpExecutor *executor, aclrtStream stream);

    // ---- aclGraph (aclmdlRI*) — runtime graph capture/replay ---------------
    // Present on CANN 8.3+. The aclmdlRI / aclmdlRICaptureMode types come from
    // `acl/acl_rt.h` (pulled in transitively by `acl/acl.h` above). If the
    // toolkit is older and the symbols don't resolve, has_aclgraph() returns
    // false and callers fall back to eager execution silently. Note: there is
    // no `aclmdlRICreate` in the public API — a graph is created implicitly
    // by the CaptureBegin/CaptureEnd pair, similar to CUDA's
    // cudaStreamBeginCapture/cudaStreamEndCapture flow.
    aclError (*aclmdlRICaptureBegin)(aclrtStream, aclmdlRICaptureMode);
    aclError (*aclmdlRICaptureEnd)(aclrtStream, aclmdlRI *);
    aclError (*aclmdlRIExecuteAsync)(aclmdlRI, aclrtStream);
    aclError (*aclmdlRIDestroy)(aclmdlRI);

    // ---- aclGraph task-group + task-update (G1 HARD GATE) ------------------
    // Present on CANN 8.3+ (sibling of the base capture API above). These
    // enable the vllm-ascend-style "capture once, rebind params per replay"
    // flow: `CaptureTaskGrpBegin/End` wraps a sub-range of the capture as a
    // named task group, and `CaptureTaskUpdateBegin/End` rewrites the tensor
    // bindings of that group on a side stream between replays. Absence on
    // the toolkit means we cannot parameter-rebind and must either fall back
    // to a pos-keyed graph cache or abandon aclGraph entirely. Types:
    //   aclrtTaskGrp is defined as `void *` in acl/acl_base.h:63.
    aclError (*aclmdlRICaptureTaskGrpBegin)(aclrtStream);
    aclError (*aclmdlRICaptureTaskGrpEnd)(aclrtStream, aclrtTaskGrp *);
    aclError (*aclmdlRICaptureTaskUpdateBegin)(aclrtStream, aclrtTaskGrp);
    aclError (*aclmdlRICaptureTaskUpdateEnd)(aclrtStream);

    bool is_ready() const { return aclrtMalloc != nullptr; }

    // Separate capability flag: aclGraph is optional. Callers that want to
    // use capture/replay check this first and silently fall back to eager
    // mode if it returns false (e.g., running on CANN < 8.3).
    bool has_aclgraph() const {
        return aclmdlRICaptureBegin != nullptr &&
               aclmdlRICaptureEnd   != nullptr &&
               aclmdlRIExecuteAsync != nullptr &&
               aclmdlRIDestroy      != nullptr;
    }

    // Capability flag for the task-group + task-update sub-family (G1 HARD
    // GATE). True iff all 4 new symbols resolve in addition to the base
    // aclGraph set. vllm-ascend's paged-attention `update_graph_params` path
    // requires these; our CP forward per-pos rebinding (RoPE slice, KV-slot,
    // FIAv2 seq_len) does the same. If false on 8.3.RC1, the aclGraph
    // feasibility track aborts — the base capture-only path has no way to
    // per-frame rebind.
    bool has_aclgraph_task_update() const {
        return has_aclgraph() &&
               aclmdlRICaptureTaskGrpBegin    != nullptr &&
               aclmdlRICaptureTaskGrpEnd      != nullptr &&
               aclmdlRICaptureTaskUpdateBegin != nullptr &&
               aclmdlRICaptureTaskUpdateEnd   != nullptr;
    }

    // Capability flag for the FRACTAL_NZ weight pre-conversion path (M5).
    // On CANN 8.3+ both entry points resolve. On older toolkits one or both
    // will be null and callers must leave weights in ND layout.
    bool has_nz() const {
        return aclnnTransMatmulWeightGetWorkspaceSize != nullptr &&
               aclnnTransMatmulWeight                 != nullptr;
    }

    // Capability flag for aclnnMatmulWeightNz (M5.3, CANN 8.5+). When present,
    // the engines dispatch this op at matmul call sites instead of plain
    // aclnnMm so the pre-converted NZ weight buffer is actually consumed. On
    // older CANN toolkits (8.3) this resolves nullptr and callers stay on
    // plain aclnnMm. Independent from has_nz(): even if TransMatmulWeight is
    // available, without MatmulWeightNz the NZ layout is dead weight.
    bool has_matmul_weight_nz() const {
        return aclnnMatmulWeightNzGetWorkspaceSize != nullptr &&
               aclnnMatmulWeightNz                 != nullptr;
    }

    // Capability flag for aclnnWeightQuantBatchMatmulV3/V2 (Stretch S1,
    // CANN 8.5+). True when either V3 OR V2 resolved — callers dispatch
    // whichever is non-null (preferring V3). Absence means the toolkit is
    // pre-8.5 or the W8Q op families aren't available, and the engines must
    // fall back to plain F16 matmul (NZ or ND).
    bool has_w8_quant() const {
        return (aclnnWeightQuantBatchMatmulV3GetWorkspaceSize != nullptr &&
                aclnnWeightQuantBatchMatmulV3                 != nullptr) ||
               (aclnnWeightQuantBatchMatmulV2GetWorkspaceSize != nullptr &&
                aclnnWeightQuantBatchMatmulV2                 != nullptr);
    }

    // Capability flag for aclnnGroupedMatmulV3 (Phase GMM-wire, CANN 8.3+).
    // True iff both the workspace-size entry point and the launch entry
    // point resolved. Callers (CpCannEngine) gate the fused QKV path on
    // (TALKER_CP_GMM_QKV=1 + w8_applied_ + has_grouped_matmul_v3()); absence
    // falls back silently to the 3 × aclnnWeightQuantBatchMatmulV3 path.
    bool has_grouped_matmul_v3() const {
        return aclnnGroupedMatmulV3GetWorkspaceSize != nullptr &&
               aclnnGroupedMatmulV3                 != nullptr;
    }

    // Capability flag for aclnnAddRmsNorm (W3b, CANN 8.5+). When present,
    // CpCannEngine can fuse the `Add(residual, output) + RmsNorm(cur, gamma)`
    // tail at the end of each attention/FFN sublayer into one dispatch. Gated
    // at runtime by TALKER_CP_FUSION=1; absence silently falls back to the
    // unfused path.
    bool has_add_rms_norm() const {
        return aclnnAddRmsNormGetWorkspaceSize != nullptr &&
               aclnnAddRmsNorm                 != nullptr;
    }

    // Capability flag for aclnnInplaceAddRmsNorm (Phase A.1, CANN 8.3+).
    // When present + TALKER_CP_INPLACE_ADDRMSNORM=1, CpCannEngine writes
    // rmsnorm(x1+x2) back into x1Ref and the sum into x2Ref, eliminating the
    // residual-copy that follows aclnnAddRmsNorm. Absence means caller keeps
    // the non-inplace path.
    bool has_inplace_add_rms_norm() const {
        return aclnnInplaceAddRmsNormGetWorkspaceSize != nullptr &&
               aclnnInplaceAddRmsNorm                 != nullptr;
    }

    // Capability flag for aclnnFFNV3 (Phase B, CANN 8.5+). When present +
    // TALKER_CP_FFN_V3=1 + w8_applied_, the 5-op FFN chain (gate-Mm +
    // up-Mm + SiLU + mul + down-Mm) collapses into a single aclnnFFNV3
    // call with activation="swiglu". Absence means caller stays on the
    // per-op W8 matmul path.
    bool has_ffn_v3() const {
        return aclnnFFNV3GetWorkspaceSize != nullptr &&
               aclnnFFNV3                 != nullptr;
    }

    // Capability flag for aclnnApplyRotaryPosEmbV2 (Phase A.2, CANN 8.3+).
    // When present + TALKER_CP_ROPE_V2=1, the two per-step
    // aclnnRotaryPositionEmbedding calls (Q + K) collapse into a single
    // fused in-place V2 call. Absence means the toolkit predates the V2
    // op and callers stay on the two-call path.
    bool has_rope_v2() const {
        return aclnnApplyRotaryPosEmbV2GetWorkspaceSize != nullptr &&
               aclnnApplyRotaryPosEmbV2                 != nullptr;
    }

    // Capability flag for aclnnGeluV2 (Q2.3 / QIE DiT FFN). When present,
    // callers dispatch with approximate="tanh" to match the ggml CPU
    // reference. Absence means only the exact-erf aclnnGelu is available
    // and callers accept a ~1e-3 cos_sim drift vs CPU reference (still
    // above 0.99 in practice).
    bool has_gelu_v2() const {
        return aclnnGeluV2GetWorkspaceSize != nullptr &&
               aclnnGeluV2                 != nullptr;
    }

    // Capability flag for aclnnLayerNorm (Q2.3 / QIE block norms). Required
    // for the DiT block's img_norm1/2 + txt_norm1/2. Absence is fatal for
    // QIE Phase 3; we ship a loud error rather than a silent wrong-norm.
    bool has_layer_norm() const {
        return aclnnLayerNormGetWorkspaceSize != nullptr &&
               aclnnLayerNorm                 != nullptr;
    }
};

extern CannSyms g_cann;

// Idempotent. Dlopens libascendcl.so, libnnopbase.so, libopapi.so and resolves
// every member of g_cann. Returns true on success. On failure, prints which
// symbol / library could not be loaded and leaves g_cann in a partial state
// (caller must check is_ready()).
bool cp_cann_load_symbols();
