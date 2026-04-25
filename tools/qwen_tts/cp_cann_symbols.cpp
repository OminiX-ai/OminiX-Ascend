// ============================================================================
// Runtime CANN symbol loader. See cp_cann_symbols.h for rationale.
// ============================================================================

#include "cp_cann_symbols.h"

#include <dlfcn.h>
#include <cstdio>
#include <cstring>
#include <mutex>

CannSyms g_cann = {};

namespace {

void *dlopen_first(const char *const *candidates) {
    for (const char *const *p = candidates; *p; ++p) {
        void *h = dlopen(*p, RTLD_LAZY | RTLD_GLOBAL);
        if (h) return h;
    }
    return nullptr;
}

template <typename FnPtr>
bool resolve(void *handle, const char *name, FnPtr &out) {
    dlerror();
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    if (!sym || err) {
        fprintf(stderr, "[cp_cann] dlsym failed for '%s': %s\n", name,
                err ? err : "null");
        return false;
    }
    out = reinterpret_cast<FnPtr>(sym);
    return true;
}

// Same as resolve() but doesn't log/fail when the symbol is missing. Used for
// optional symbols (aclGraph entry points on older CANN toolkits).
template <typename FnPtr>
void resolve_optional(void *handle, const char *name, FnPtr &out) {
    dlerror();
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    if (sym && !err) out = reinterpret_cast<FnPtr>(sym);
    // else: leave nullptr, checked via has_aclgraph()
}

bool load_once() {
    // Each of these lives in a different CANN shared object. We search a few
    // common paths (flat + aarch64-linux) to be robust against toolkit layout
    // changes. RTLD_GLOBAL so the aclnn ops can find their base symbols even
    // though we loaded them via dlopen rather than the linker.
    const char *runtime_candidates[] = {
        "libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libascendcl.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libascendcl.so",
        nullptr,
    };
    const char *nnopbase_candidates[] = {
        "libnnopbase.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libnnopbase.so",
        "/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64/libnnopbase.so",
        nullptr,
    };
    const char *opapi_candidates[] = {
        "libopapi.so",
        "/usr/local/Ascend/ascend-toolkit/latest/lib64/libopapi.so",
        "/usr/local/Ascend/ascend-toolkit/latest/opp/lib64/libopapi.so",
        nullptr,
    };

    void *h_rt = dlopen_first(runtime_candidates);
    if (!h_rt) {
        fprintf(stderr, "[cp_cann] dlopen libascendcl.so failed: %s\n",
                dlerror());
        return false;
    }
    void *h_base = dlopen_first(nnopbase_candidates);
    if (!h_base) {
        fprintf(stderr, "[cp_cann] dlopen libnnopbase.so failed: %s\n",
                dlerror());
        return false;
    }
    void *h_op = dlopen_first(opapi_candidates);
    if (!h_op) {
        fprintf(stderr, "[cp_cann] dlopen libopapi.so failed: %s\n",
                dlerror());
        return false;
    }

    bool ok = true;
    // runtime (libascendcl)
    ok &= resolve(h_rt,   "aclInit",                  g_cann.aclInit);
    ok &= resolve(h_rt,   "aclrtSetDevice",           g_cann.aclrtSetDevice);
    ok &= resolve(h_rt,   "aclrtCreateStream",        g_cann.aclrtCreateStream);
    ok &= resolve(h_rt,   "aclrtDestroyStream",       g_cann.aclrtDestroyStream);
    ok &= resolve(h_rt,   "aclrtSynchronizeStream",   g_cann.aclrtSynchronizeStream);
    ok &= resolve(h_rt,   "aclrtMalloc",              g_cann.aclrtMalloc);
    ok &= resolve(h_rt,   "aclrtFree",                g_cann.aclrtFree);
    ok &= resolve(h_rt,   "aclrtMemcpy",              g_cann.aclrtMemcpy);
    ok &= resolve(h_rt,   "aclrtMemcpyAsync",         g_cann.aclrtMemcpyAsync);
    ok &= resolve(h_rt,   "aclGetRecentErrMsg",       g_cann.aclGetRecentErrMsg);

    // Event APIs (multi-stream sync — M6). Required on every supported CANN.
    ok &= resolve(h_rt,   "aclrtCreateEvent",         g_cann.aclrtCreateEvent);
    ok &= resolve(h_rt,   "aclrtDestroyEvent",        g_cann.aclrtDestroyEvent);
    ok &= resolve(h_rt,   "aclrtRecordEvent",         g_cann.aclrtRecordEvent);
    ok &= resolve(h_rt,   "aclrtStreamWaitEvent",     g_cann.aclrtStreamWaitEvent);
    ok &= resolve(h_rt,   "aclrtSynchronizeEvent",    g_cann.aclrtSynchronizeEvent);

    // aclnn base (libnnopbase)
    ok &= resolve(h_base, "aclCreateTensor",          g_cann.aclCreateTensor);
    ok &= resolve(h_base, "aclDestroyTensor",         g_cann.aclDestroyTensor);
    ok &= resolve(h_base, "aclCreateScalar",          g_cann.aclCreateScalar);
    ok &= resolve(h_base, "aclDestroyScalar",         g_cann.aclDestroyScalar);
    // IntArray helpers (QIE Q2.3 LayerNorm); optional — absence is only
    // fatal at a LayerNorm call site, and the capability flag has_layer_norm()
    // keeps that isolated.
    resolve_optional(h_base, "aclCreateIntArray",     g_cann.aclCreateIntArray);
    resolve_optional(h_base, "aclDestroyIntArray",    g_cann.aclDestroyIntArray);

    // ops (libopapi)
    ok &= resolve(h_op,   "aclnnMmGetWorkspaceSize",
                  g_cann.aclnnMmGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnMm",                  g_cann.aclnnMm);
    ok &= resolve(h_op,   "aclnnAddGetWorkspaceSize",
                  g_cann.aclnnAddGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnAdd",                 g_cann.aclnnAdd);
    ok &= resolve(h_op,   "aclnnInplaceAddGetWorkspaceSize",
                  g_cann.aclnnInplaceAddGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnInplaceAdd",          g_cann.aclnnInplaceAdd);
    ok &= resolve(h_op,   "aclnnInplaceMulGetWorkspaceSize",
                  g_cann.aclnnInplaceMulGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnInplaceMul",          g_cann.aclnnInplaceMul);
    ok &= resolve(h_op,   "aclnnSiluGetWorkspaceSize",
                  g_cann.aclnnSiluGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnSilu",                g_cann.aclnnSilu);
    ok &= resolve(h_op,   "aclnnRmsNormGetWorkspaceSize",
                  g_cann.aclnnRmsNormGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnRmsNorm",             g_cann.aclnnRmsNorm);

    ok &= resolve(h_op,   "aclnnBatchMatMulGetWorkspaceSize",
                  g_cann.aclnnBatchMatMulGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnBatchMatMul",         g_cann.aclnnBatchMatMul);
    ok &= resolve(h_op,   "aclnnSoftmaxGetWorkspaceSize",
                  g_cann.aclnnSoftmaxGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnSoftmax",             g_cann.aclnnSoftmax);
    ok &= resolve(h_op,   "aclnnMulsGetWorkspaceSize",
                  g_cann.aclnnMulsGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnMuls",                g_cann.aclnnMuls);
    // Optional: aclnnInplaceMuls. Used by QIE Q2.4.5.4k softmax-overflow
    // workaround. Older CANN toolkits may lack the symbol; engine call sites
    // null-check before dispatch and surface a precise error.
    resolve_optional(h_op, "aclnnInplaceMulsGetWorkspaceSize",
                     g_cann.aclnnInplaceMulsGetWorkspaceSize);
    resolve_optional(h_op, "aclnnInplaceMuls",  g_cann.aclnnInplaceMuls);
    ok &= resolve(h_op,   "aclnnRotaryPositionEmbeddingGetWorkspaceSize",
                  g_cann.aclnnRotaryPositionEmbeddingGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnRotaryPositionEmbedding",
                  g_cann.aclnnRotaryPositionEmbedding);
    ok &= resolve(h_op,   "aclnnCastGetWorkspaceSize",
                  g_cann.aclnnCastGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnCast",                g_cann.aclnnCast);

    // Phase 4.1 on-device RoPE scatter. aclnnInplaceCopy supports non-contig
    // src/dst — used to scatter rotated halves back into interleaved x.
    ok &= resolve(h_op,   "aclnnInplaceCopyGetWorkspaceSize",
                  g_cann.aclnnInplaceCopyGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnInplaceCopy",         g_cann.aclnnInplaceCopy);

    // TensorList lives in libnnopbase like the other aclCreateTensor helpers.
    ok &= resolve(h_base, "aclCreateTensorList",      g_cann.aclCreateTensorList);
    ok &= resolve(h_base, "aclDestroyTensorList",     g_cann.aclDestroyTensorList);

    ok &= resolve(h_op,   "aclnnFusedInferAttentionScoreV2GetWorkspaceSize",
                  g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnFusedInferAttentionScoreV2",
                  g_cann.aclnnFusedInferAttentionScoreV2);

    // Optional: aclGraph (aclmdlRI*). Present on CANN 8.3+ only. Absence is
    // not fatal — callers that want capture/replay gate on has_aclgraph().
    resolve_optional(h_rt, "aclmdlRICaptureBegin",
                     g_cann.aclmdlRICaptureBegin);
    resolve_optional(h_rt, "aclmdlRICaptureEnd",
                     g_cann.aclmdlRICaptureEnd);
    resolve_optional(h_rt, "aclmdlRIExecuteAsync",
                     g_cann.aclmdlRIExecuteAsync);
    resolve_optional(h_rt, "aclmdlRIDestroy",
                     g_cann.aclmdlRIDestroy);

    // Optional: aclGraph task-group + task-update (G1 HARD GATE). Sibling of
    // the base aclGraph API above. Callers that want parameter rebinding on
    // an already-captured graph gate on has_aclgraph_task_update(). Absence
    // on a toolkit that does expose the base aclGraph API means the vendor
    // has not shipped the rebind primitive — callers must pivot to a
    // pos-keyed graph cache or abandon aclGraph.
    resolve_optional(h_rt, "aclmdlRICaptureTaskGrpBegin",
                     g_cann.aclmdlRICaptureTaskGrpBegin);
    resolve_optional(h_rt, "aclmdlRICaptureTaskGrpEnd",
                     g_cann.aclmdlRICaptureTaskGrpEnd);
    resolve_optional(h_rt, "aclmdlRICaptureTaskUpdateBegin",
                     g_cann.aclmdlRICaptureTaskUpdateBegin);
    resolve_optional(h_rt, "aclmdlRICaptureTaskUpdateEnd",
                     g_cann.aclmdlRICaptureTaskUpdateEnd);

    // Optional: FRACTAL_NZ weight pre-conversion (M5). Lives in libopapi.so
    // alongside the other aclnn ops; absence means older CANN and callers
    // gate via has_nz() before touching weights.
    resolve_optional(h_op, "aclnnTransMatmulWeightGetWorkspaceSize",
                     g_cann.aclnnTransMatmulWeightGetWorkspaceSize);
    resolve_optional(h_op, "aclnnTransMatmulWeight",
                     g_cann.aclnnTransMatmulWeight);

    // Optional: aclnnMatmulWeightNz (M5.3, CANN 8.5+). Matmul op that actually
    // consumes an NZ-laid-out weight buffer (as opposed to plain aclnnMm on
    // CANN 8.3 which silently falls back to the ND kernel). Gated via
    // has_matmul_weight_nz(); callers use plain aclnnMm when the symbol is
    // absent. The Nz variant of BatchMatMul is pulled in for parity with a
    // possible future batched path — still optional.
    resolve_optional(h_op, "aclnnMatmulWeightNzGetWorkspaceSize",
                     g_cann.aclnnMatmulWeightNzGetWorkspaceSize);
    resolve_optional(h_op, "aclnnMatmulWeightNz",
                     g_cann.aclnnMatmulWeightNz);
    resolve_optional(h_op, "aclnnBatchMatMulWeightNzGetWorkspaceSize",
                     g_cann.aclnnBatchMatMulWeightNzGetWorkspaceSize);
    resolve_optional(h_op, "aclnnBatchMatMulWeightNz",
                     g_cann.aclnnBatchMatMulWeightNz);

    // Optional: aclnnWeightQuantBatchMatmulV3/V2 (Stretch S1, CANN 8.5+).
    // Both variants live in libopapi.so. Either can land null on older
    // toolkits — has_w8_quant() gates both; callers dispatch V3 when present
    // and fall back to V2 otherwise. Absence of both means "W8 quant
    // unavailable" and callers keep the F16 path.
    resolve_optional(h_op, "aclnnWeightQuantBatchMatmulV3GetWorkspaceSize",
                     g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize);
    resolve_optional(h_op, "aclnnWeightQuantBatchMatmulV3",
                     g_cann.aclnnWeightQuantBatchMatmulV3);
    resolve_optional(h_op, "aclnnWeightQuantBatchMatmulV2GetWorkspaceSize",
                     g_cann.aclnnWeightQuantBatchMatmulV2GetWorkspaceSize);
    resolve_optional(h_op, "aclnnWeightQuantBatchMatmulV2",
                     g_cann.aclnnWeightQuantBatchMatmulV2);

    // Optional: aclnnGroupedMatmulV3 (Phase GMM-wire, CANN 8.3+). Fused
    // multi-group matmul; our use wires three Q/K/V groups sharing the same
    // activation. Absence means the toolkit lacks the op and callers stay on
    // 3 × aclnnWeightQuantBatchMatmulV3. Gated via has_grouped_matmul_v3().
    resolve_optional(h_op, "aclnnGroupedMatmulV3GetWorkspaceSize",
                     g_cann.aclnnGroupedMatmulV3GetWorkspaceSize);
    resolve_optional(h_op, "aclnnGroupedMatmulV3",
                     g_cann.aclnnGroupedMatmulV3);

    // Optional: aclnnAddRmsNorm (W3b, CANN 8.5+). Lives in libopapi.so.
    // Absence means the toolkit predates CANN 8.5 and callers keep the
    // unfused Add + RmsNorm path. Gated via has_add_rms_norm().
    resolve_optional(h_op, "aclnnAddRmsNormGetWorkspaceSize",
                     g_cann.aclnnAddRmsNormGetWorkspaceSize);
    resolve_optional(h_op, "aclnnAddRmsNorm",
                     g_cann.aclnnAddRmsNorm);

    // Optional: aclnnInplaceAddRmsNorm (Phase A.1, CANN 8.3+). Lives in
    // libopapi.so alongside the non-inplace variant. Absence means the
    // toolkit lacks the in-place sibling and callers must stay on
    // aclnnAddRmsNorm + residual-copy. Gated via has_inplace_add_rms_norm().
    resolve_optional(h_op, "aclnnInplaceAddRmsNormGetWorkspaceSize",
                     g_cann.aclnnInplaceAddRmsNormGetWorkspaceSize);
    resolve_optional(h_op, "aclnnInplaceAddRmsNorm",
                     g_cann.aclnnInplaceAddRmsNorm);

    // Optional: aclnnFFNV3 (Phase B, CANN 8.5+). Lives in libopapi.so.
    // Fused SwiGLU FFN with per-channel antiquant dequant. Absence means
    // callers stay on the 5-op W8 matmul chain. Gated via has_ffn_v3().
    resolve_optional(h_op, "aclnnFFNV3GetWorkspaceSize",
                     g_cann.aclnnFFNV3GetWorkspaceSize);
    resolve_optional(h_op, "aclnnFFNV3",
                     g_cann.aclnnFFNV3);

    // Optional: aclnnApplyRotaryPosEmbV2 (Phase A.2, CANN 8.3+). Fused Q+K
    // in-place RoPE. Absence means callers stay on the two-call
    // aclnnRotaryPositionEmbedding path. Gated via has_rope_v2().
    resolve_optional(h_op, "aclnnApplyRotaryPosEmbV2GetWorkspaceSize",
                     g_cann.aclnnApplyRotaryPosEmbV2GetWorkspaceSize);
    resolve_optional(h_op, "aclnnApplyRotaryPosEmbV2",
                     g_cann.aclnnApplyRotaryPosEmbV2);

    // Required for QIE Q2.3 block forward — aclnnMul (broadcast elementwise),
    // aclnnLayerNorm (affine-off block norm), aclnnGelu/GeluV2 (FFN
    // activation), aclnnSigmoid (convenience). CANN 8.3+. Marked
    // `resolve_optional` so a missing symbol doesn't brick the unrelated
    // TTS paths; QIE-specific callers check the capability helpers.
    resolve_optional(h_op, "aclnnMulGetWorkspaceSize",
                     g_cann.aclnnMulGetWorkspaceSize);
    resolve_optional(h_op, "aclnnMul",
                     g_cann.aclnnMul);
    resolve_optional(h_op, "aclnnLayerNormGetWorkspaceSize",
                     g_cann.aclnnLayerNormGetWorkspaceSize);
    resolve_optional(h_op, "aclnnLayerNorm",
                     g_cann.aclnnLayerNorm);
    resolve_optional(h_op, "aclnnGeluV2GetWorkspaceSize",
                     g_cann.aclnnGeluV2GetWorkspaceSize);
    resolve_optional(h_op, "aclnnGeluV2",
                     g_cann.aclnnGeluV2);
    resolve_optional(h_op, "aclnnGeluGetWorkspaceSize",
                     g_cann.aclnnGeluGetWorkspaceSize);
    resolve_optional(h_op, "aclnnGelu",
                     g_cann.aclnnGelu);
    resolve_optional(h_op, "aclnnSigmoidGetWorkspaceSize",
                     g_cann.aclnnSigmoidGetWorkspaceSize);
    resolve_optional(h_op, "aclnnSigmoid",
                     g_cann.aclnnSigmoid);

    if (!ok) {
        // Wipe partial state so is_ready() reports false.
        g_cann = {};
        return false;
    }
    // Idempotent: aclInit can only be called once per process. If ggml-cann
    // is loaded in the same binary (e.g., the main qwen_tts exe links
    // libggml-cann.so), it will have already called aclInit — in that case
    // the second call returns an "already initialized" error code which we
    // ignore. Standalone binaries (e.g., test_talker_native) that don't use
    // ggml-cann need this init call to register the op-tiling kernels, or
    // every aclnn op fails with "tiling_funcs NULL".
    g_cann.aclInit(nullptr);
    return true;
}

} // namespace

bool cp_cann_load_symbols() {
    static std::once_flag once;
    static bool cached = false;
    std::call_once(once, [] { cached = load_once(); });
    return cached;
}
