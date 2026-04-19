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
    ok &= resolve(h_op,   "aclnnRotaryPositionEmbeddingGetWorkspaceSize",
                  g_cann.aclnnRotaryPositionEmbeddingGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnRotaryPositionEmbedding",
                  g_cann.aclnnRotaryPositionEmbedding);
    ok &= resolve(h_op,   "aclnnCastGetWorkspaceSize",
                  g_cann.aclnnCastGetWorkspaceSize);
    ok &= resolve(h_op,   "aclnnCast",                g_cann.aclnnCast);

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
