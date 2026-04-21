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

    // Dtype conversion (used to cast the F32 input embedding down to F16 at
    // engine entry, and the F16 final hidden back to F32 at exit).
    aclnnStatus (*aclnnCastGetWorkspaceSize)(const aclTensor *self,
                                              aclDataType dtype,
                                              aclTensor *out,
                                              uint64_t *workspaceSize,
                                              aclOpExecutor **executor);
    aclnnStatus (*aclnnCast)(void *, uint64_t, aclOpExecutor *, aclrtStream);

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

    // Capability flag for aclnnAddRmsNorm (W3b, CANN 8.5+). When present,
    // CpCannEngine can fuse the `Add(residual, output) + RmsNorm(cur, gamma)`
    // tail at the end of each attention/FFN sublayer into one dispatch. Gated
    // at runtime by TALKER_CP_FUSION=1; absence silently falls back to the
    // unfused path.
    bool has_add_rms_norm() const {
        return aclnnAddRmsNormGetWorkspaceSize != nullptr &&
               aclnnAddRmsNorm                 != nullptr;
    }
};

extern CannSyms g_cann;

// Idempotent. Dlopens libascendcl.so, libnnopbase.so, libopapi.so and resolves
// every member of g_cann. Returns true on success. On failure, prints which
// symbol / library could not be loaded and leaves g_cann in a partial state
// (caller must check is_ready()).
bool cp_cann_load_symbols();
