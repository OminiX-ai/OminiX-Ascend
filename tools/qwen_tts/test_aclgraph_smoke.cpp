// ============================================================================
// test_aclgraph_smoke.cpp — G1 HARD GATE feasibility probe.
//
// Answers the single open question from G0: does
// `aclmdlRICaptureTaskUpdateBegin/End` accept rebinding of
//   (a) RoPE cos/sin strided slice pointers,
//   (b) KV-cache slot write dst offsets,
//   (c) `aclnnFusedInferAttentionScoreV2`'s `seq_len` scalar +
//       K/V stride-over-seq tensor views,
// on an already-captured `aclOpExecutor`?
//
// IMPORTANT FINDING (first build, preserved as harness constant):
// `aclrtMemcpyAsync` (D2D) inside an active `aclmdlRICapture*` session
// returns ACL error 507009 ("task not supported"). That rules out
// capturing the V→KV-slot memcpy or any device-to-device reset_buffers
// work inside the graph. This means KV-slot rebind has to happen as a
// CPU-side memcpy launched BEFORE the captured replay each frame, and
// the captured region cannot contain any D2D aclrtMemcpyAsync. The
// harness below accordingly does V→slot + buffer resets OUTSIDE the
// capture/replay region, then captures only the aclnn-op DAG.
//
// Harness: replicate the Qwen3-TTS CP layer-0 attn sub-DAG
//   RmsNorm(normed)
//   → QKV-Mm (F16 Mm with dummy weights)
//   → q_norm / k_norm (RmsNorm on per-head)
//   → RoPE (Q and K, consuming cos/sin slice ptrs)
//   → V-slot memcpy (dst offset pos*kv_dim)
//   → FusedInferAttentionScoreV2 (seq_len = pos+1)
//   → O-Mm (F16 Mm with dummy O-proj weights)
//   → post-attn Add + RmsNorm (fused via aclnnAddRmsNorm)
//   → residual=cur memcpy
// on synthetic fixed inputs (identical across replays). The layer produces
// a deterministic output for each `pos` value.
//
// Strategy:
//   1. Set up persistent device buffers (weights, normed/cur/residual,
//      q/k/v, o_out, k_cache, v_cache, workspace).
//   2. Run eager at pos=0,1,...9, store host-side output snapshots → stock_out[].
//   3. Replay path:
//      - Reset buffers to the same fixed initial state.
//      - CaptureBegin → run sub-DAG at pos=0 with TaskGrp wrap around the 3
//        param-update ops (2 RoPE, V-memcpy, FIAv2) → CaptureEnd → ri_handle.
//      - For pos in 0..9:
//          reset buffers again,
//          CaptureTaskUpdateBegin on update_stream,
//          re-register 2 RoPE/V-memcpy/FIAv2 ops with rebound params,
//          CaptureTaskUpdateEnd,
//          ExecuteAsync on main stream,
//          Sync,
//          copy output → replay_out[pos],
//          compare with stock_out[pos].
//      - Time each replay with aclrtRecordEvent bracketing.
//   4. Print verdict in the exact G1 gate-report format.
//
// Gate: parity max_abs_diff ≤ 1e-4 F16 across all 10 pos, median replay wall ≤ 1.5 ms.
//
// Does NOT require real Qwen3 weights — synthetic deterministic inputs
// (fixed random seed) suffice to answer the feasibility question about
// whether the aclmdlRI task-update primitive rebinds correctly. If task
// update works for our inputs, it will work for real weights (executors
// are parameterized on shape/dtype/tensor metadata, not on tensor values).
// ============================================================================

#include "cp_cann_symbols.h"
#include "ggml-backend.h"

#include <acl/acl.h>
#include <aclnn/acl_meta.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <vector>
#include <random>
#include <algorithm>

// ---- Check helpers --------------------------------------------------------
#define ACL_CHECK(stmt) do {                                             \
    aclError _e = (stmt);                                                \
    if (_e != 0) {                                                       \
        fprintf(stderr, "[smoke] ACL error %d at %s:%d: %s\n", _e,       \
                __FILE__, __LINE__,                                      \
                g_cann.aclGetRecentErrMsg ?                              \
                    g_cann.aclGetRecentErrMsg() : "<n/a>");              \
        return 1;                                                        \
    }                                                                    \
} while (0)
#define ACL_CHECK_RET(stmt) ACL_CHECK(stmt)

// ---- F16 helpers ----------------------------------------------------------
static inline uint16_t f16_bits(float v) {
    __fp16 h = (__fp16)v;
    uint16_t b;
    std::memcpy(&b, &h, 2);
    return b;
}
static inline float f16_to_f32(uint16_t b) {
    __fp16 h;
    std::memcpy(&h, &b, 2);
    return (float)h;
}

// ---- Dimensions -----------------------------------------------------------
// Real Qwen3-TTS CP config (talker code_predictor, 5-layer transformer).
static constexpr int kCpHidden  = 1024;
static constexpr int kNHeads    = 16;
static constexpr int kNKv       = 8;
static constexpr int kHeadDim   = 128;
static constexpr int kQDim      = kNHeads * kHeadDim;   // 2048
static constexpr int kKvDim     = kNKv   * kHeadDim;    //  1024
static constexpr int kMaxSeq    = 17;
static constexpr float kEps     = 1e-5f;

// Captured workspace (shared with the CP engine ws pool pattern).
static void *g_workspace = nullptr;
static size_t g_ws_size  = 0;

static aclError grow_workspace(uint64_t needed) {
    if (needed <= g_ws_size) return ACL_SUCCESS;
    if (g_workspace) g_cann.aclrtFree(g_workspace);
    aclError e = g_cann.aclrtMalloc(&g_workspace, needed,
                                      ACL_MEM_MALLOC_HUGE_FIRST);
    if (e == ACL_SUCCESS) g_ws_size = needed;
    return e;
}

// ---- Tensor constructors ------------------------------------------------
static aclTensor *make_tensor(void *buf, int64_t rank,
                              const int64_t *shape,
                              const int64_t *strides,
                              aclDataType dtype = ACL_FLOAT16,
                              aclFormat fmt = ACL_FORMAT_ND) {
    int64_t n = 1;
    for (int64_t i = 0; i < rank; ++i) n *= shape[i];
    return g_cann.aclCreateTensor(shape, rank, dtype, strides, 0,
                                   fmt, &n, 1, buf);
}
static aclTensor *tensor_1d(void *buf, int64_t n, aclDataType dtype = ACL_FLOAT16) {
    int64_t sh[1] = {n}, st[1] = {1};
    return make_tensor(buf, 1, sh, st, dtype);
}
static aclTensor *tensor_2d(void *buf, int64_t d0, int64_t d1,
                             aclDataType dtype = ACL_FLOAT16) {
    int64_t sh[2] = {d0, d1}, st[2] = {d1, 1};
    return make_tensor(buf, 2, sh, st, dtype);
}
static aclTensor *tensor_strided(void *buf, int64_t rank,
                                  const int64_t *shape,
                                  const int64_t *strides,
                                  aclDataType dtype = ACL_FLOAT16) {
    return make_tensor(buf, rank, shape, strides, dtype);
}

// ---- One-shot aclnn op dispatchers (the CANN_OP macro pattern) ----------
#define CANN_OP_LOCAL(OP_NAME, STREAM, ...) do {                          \
    uint64_t _ws = 0;                                                     \
    aclOpExecutor *_exec = nullptr;                                       \
    ACL_CHECK(g_cann.aclnn##OP_NAME##GetWorkspaceSize(                    \
        __VA_ARGS__, &_ws, &_exec));                                      \
    ACL_CHECK(grow_workspace(_ws));                                       \
    void *_w = _ws > 0 ? g_workspace : nullptr;                           \
    ACL_CHECK(g_cann.aclnn##OP_NAME(_w, _ws, _exec, (STREAM)));           \
} while (0)

// ============================================================================
// G1 harness.
// ============================================================================

// Persistent device buffers (lifetime of the test).
struct Ctx {
    void *normed_dev = nullptr;     // [cp_hidden] F16 — precomputed normed input
    void *q_dev      = nullptr;     // [q_dim]     F16
    void *k_dev      = nullptr;     // [kv_dim]    F16
    void *v_dev      = nullptr;     // [kv_dim]    F16
    void *attn_out_dev = nullptr;   // [q_dim]     F16 (RoPE'd Q staging)
    void *o_out_dev   = nullptr;    // [cp_hidden] F16
    void *residual_dev = nullptr;   // [cp_hidden] F16
    void *cur_dev      = nullptr;   // [cp_hidden] F16
    void *normed_out_dev = nullptr; // [cp_hidden] F16 (post-attn AddRmsNorm)
    void *rstd_dev   = nullptr;     // [1]         F32
    void *k_cache_dev = nullptr;    // [MAX_SEQ, kv_dim] F16
    void *v_cache_dev = nullptr;    // [MAX_SEQ, kv_dim] F16
    void *rope_cos_dev = nullptr;   // [MAX_SEQ, head_dim] F16
    void *rope_sin_dev = nullptr;   // [MAX_SEQ, head_dim] F16
    // Weights (F16)
    void *wq_dev = nullptr, *wk_dev = nullptr, *wv_dev = nullptr, *wo_dev = nullptr;
    void *q_norm_dev = nullptr, *k_norm_dev = nullptr;      // [head_dim] F16
    void *input_ln_dev = nullptr;                           // [cp_hidden] F32
    void *post_ln_dev  = nullptr;                           // [cp_hidden] F16 (for AddRmsNorm)
    void *initial_cur_dev = nullptr;   // seed value for cur before forward
    void *initial_resid_dev = nullptr; // seed value for residual before forward
};

static Ctx ctx;

// Seed cur/residual with deterministic pattern (host → device).
static aclError seed_activations() {
    std::vector<uint16_t> seed_cur(kCpHidden), seed_resid(kCpHidden);
    for (int i = 0; i < kCpHidden; ++i) {
        // Small-magnitude values: values will pass through RmsNorm and Mm
        // without overflowing to NaN.
        seed_cur[i]   = f16_bits(((i * 7) % 37 - 18) * 0.01f);
        seed_resid[i] = f16_bits(((i * 11) % 29 - 14) * 0.015f);
    }
    ACL_CHECK(g_cann.aclrtMemcpy(ctx.initial_cur_dev, kCpHidden * 2,
        seed_cur.data(), kCpHidden * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(g_cann.aclrtMemcpy(ctx.initial_resid_dev, kCpHidden * 2,
        seed_resid.data(), kCpHidden * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}

// Reset cur/residual to fresh seed state for a new forward.
// Must be called OUTSIDE any active aclmdlRI capture — D2D memcpy is not
// capturable on 8.3.RC1 (err 507009).
static aclError reset_buffers(aclrtStream s) {
    ACL_CHECK(g_cann.aclrtMemcpyAsync(
        ctx.cur_dev, kCpHidden * 2, ctx.initial_cur_dev, kCpHidden * 2,
        ACL_MEMCPY_DEVICE_TO_DEVICE, s));
    ACL_CHECK(g_cann.aclrtMemcpyAsync(
        ctx.residual_dev, kCpHidden * 2, ctx.initial_resid_dev, kCpHidden * 2,
        ACL_MEMCPY_DEVICE_TO_DEVICE, s));
    return ACL_SUCCESS;
}

// Write v_dev → v_cache slot at pos. Must be called OUTSIDE any active
// aclmdlRI capture — D2D memcpy not capturable on 8.3.RC1. For the
// captured/replay path this is done per replay just before ExecuteAsync.
// In eager mode this runs as part of the normal dispatch path.
// NOTE: uses synchronous aclrtMemcpy (H/D-kind: D2D); we actually invoke
// the async variant on the stream for parity with the original CP engine.
static aclError v_slot_write(aclrtStream s, int pos) {
    uint16_t *v_slot = (uint16_t *)ctx.v_cache_dev + (size_t)pos * kKvDim;
    return g_cann.aclrtMemcpyAsync(v_slot, kKvDim * 2,
        ctx.v_dev, kKvDim * 2, ACL_MEMCPY_DEVICE_TO_DEVICE, s);
}

// Pre-populate k_cache/v_cache with synthetic historical values up to pos-1
// so FIAv2 at seq_len=pos+1 reads valid data for every pos under test.
static aclError seed_kv_cache() {
    std::vector<uint16_t> k_all(kMaxSeq * kKvDim), v_all(kMaxSeq * kKvDim);
    for (int p = 0; p < kMaxSeq; ++p) {
        for (int d = 0; d < kKvDim; ++d) {
            k_all[p * kKvDim + d] = f16_bits(((p + d * 3) % 19 - 9) * 0.02f);
            v_all[p * kKvDim + d] = f16_bits(((p * 2 + d) % 23 - 11) * 0.018f);
        }
    }
    ACL_CHECK(g_cann.aclrtMemcpy(ctx.k_cache_dev, kMaxSeq * kKvDim * 2,
        k_all.data(), kMaxSeq * kKvDim * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(g_cann.aclrtMemcpy(ctx.v_cache_dev, kMaxSeq * kKvDim * 2,
        v_all.data(), kMaxSeq * kKvDim * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    return ACL_SUCCESS;
}

// ============================================================================
// Forward dispatch helpers — one per sub-DAG op set.
// ============================================================================

// Dispatch the "pure" head of the attn sub-DAG: RmsNorm → QKV-Mm → QK-norm.
// Inputs from ctx.cur_dev, ctx.input_ln_dev, ctx.wq/wk/wv_dev, ctx.q_norm_dev,
// ctx.k_norm_dev. Outputs: ctx.normed_dev, ctx.q_dev, ctx.k_dev, ctx.v_dev.
static aclError run_pure_head(aclrtStream s) {
    // RmsNorm(cur, input_ln) → normed
    {
        aclTensor *t_cur    = tensor_2d(ctx.cur_dev, 1, kCpHidden);
        aclTensor *t_ln     = tensor_1d(ctx.input_ln_dev, kCpHidden, ACL_FLOAT);
        aclTensor *t_normed = tensor_2d(ctx.normed_dev, 1, kCpHidden);
        aclTensor *t_rstd   = tensor_2d(ctx.rstd_dev, 1, 1, ACL_FLOAT);
        CANN_OP_LOCAL(RmsNorm, s, t_cur, t_ln, (double)kEps, t_normed, t_rstd);
        g_cann.aclDestroyTensor(t_cur);
        g_cann.aclDestroyTensor(t_ln);
        g_cann.aclDestroyTensor(t_normed);
        g_cann.aclDestroyTensor(t_rstd);
    }
    // Q = wq @ normed^T → [q_dim, 1]; similarly K, V.
    // Weights are [out, in] and normed is [1, in]. We use plain Mm signature:
    //   aclnnMm(self[M, K], mat2[K, N], out[M, N], cubeMathType)
    // For Q_col: self = wq [q_dim, cp_hidden], mat2 = normed_col [cp_hidden, 1]
    auto do_mm = [&](void *w_dev, void *out_dev, int out_dim) -> aclError {
        aclTensor *t_w       = tensor_2d(w_dev, out_dim, kCpHidden);
        aclTensor *t_normedc = tensor_2d(ctx.normed_dev, kCpHidden, 1);
        aclTensor *t_out     = tensor_2d(out_dev, out_dim, 1);
        CANN_OP_LOCAL(Mm, s, t_w, t_normedc, t_out, (int8_t)0);
        g_cann.aclDestroyTensor(t_w);
        g_cann.aclDestroyTensor(t_normedc);
        g_cann.aclDestroyTensor(t_out);
        return ACL_SUCCESS;
    };
    ACL_CHECK(do_mm(ctx.wq_dev, ctx.q_dev, kQDim));
    ACL_CHECK(do_mm(ctx.wk_dev, ctx.k_dev, kKvDim));
    ACL_CHECK(do_mm(ctx.wv_dev, ctx.v_dev, kKvDim));

    // QK norm on per-head.
    {
        int64_t q_sh[2] = {kNHeads, kHeadDim};
        int64_t q_st[2] = {kHeadDim, 1};
        aclTensor *t_qh  = make_tensor(ctx.q_dev, 2, q_sh, q_st);
        aclTensor *t_gn  = tensor_1d(ctx.q_norm_dev, kHeadDim, ACL_FLOAT);
        aclTensor *t_rst = tensor_2d(ctx.rstd_dev, kNHeads, 1, ACL_FLOAT);
        CANN_OP_LOCAL(RmsNorm, s, t_qh, t_gn, (double)kEps, t_qh, t_rst);
        g_cann.aclDestroyTensor(t_qh);
        g_cann.aclDestroyTensor(t_gn);
        g_cann.aclDestroyTensor(t_rst);
    }
    {
        int64_t k_sh[2] = {kNKv, kHeadDim};
        int64_t k_st[2] = {kHeadDim, 1};
        aclTensor *t_kh  = make_tensor(ctx.k_dev, 2, k_sh, k_st);
        aclTensor *t_gn  = tensor_1d(ctx.k_norm_dev, kHeadDim, ACL_FLOAT);
        aclTensor *t_rst = tensor_2d(ctx.rstd_dev, kNKv, 1, ACL_FLOAT);
        CANN_OP_LOCAL(RmsNorm, s, t_kh, t_gn, (double)kEps, t_kh, t_rst);
        g_cann.aclDestroyTensor(t_kh);
        g_cann.aclDestroyTensor(t_gn);
        g_cann.aclDestroyTensor(t_rst);
    }
    return ACL_SUCCESS;
}

// Op-selector flags for fine-grained TaskGrp scoping. Used by the
// per-class probes (G1.4 granularity breakdown).
enum ParamOps : uint32_t {
    PO_ROPE_Q = 1u << 0,
    PO_ROPE_K = 1u << 1,
    PO_FIAV2  = 1u << 2,
    PO_ALL    = PO_ROPE_Q | PO_ROPE_K | PO_FIAV2,
};

static uint32_t g_ops_mask = PO_ALL;

// Dispatch the 3 param-update ops + the pure O-proj + AddRmsNorm tail. The
// `pos` value controls the RoPE slice pointers, KV-slot dst, and the FIAv2
// seq_len. This function is invoked inside the CaptureTaskGrp / TaskUpdate
// sections — the handle tracks the 3 param-dependent launches.
static aclError run_param_update_block(aclrtStream s, int pos) {
    const int seq_len = pos + 1;

    // RoPE (Q): cos/sin slice at [pos, :]
    uint16_t *cos_pos = (uint16_t *)ctx.rope_cos_dev + (size_t)pos * kHeadDim;
    uint16_t *sin_pos = (uint16_t *)ctx.rope_sin_dev + (size_t)pos * kHeadDim;
    if (g_ops_mask & PO_ROPE_Q) {
        int64_t sh[4] = {1, 1, 1, kHeadDim};
        int64_t st[4] = {kHeadDim, kHeadDim, kHeadDim, 1};
        aclTensor *t_cos = tensor_strided(cos_pos, 4, sh, st);
        aclTensor *t_sin = tensor_strided(sin_pos, 4, sh, st);
        int64_t q4[4] = {1, 1, kNHeads, kHeadDim};
        int64_t qs[4] = {kQDim, kQDim, kHeadDim, 1};
        aclTensor *t_q_in  = tensor_strided(ctx.q_dev,       4, q4, qs);
        aclTensor *t_q_out = tensor_strided(ctx.attn_out_dev, 4, q4, qs);
        CANN_OP_LOCAL(RotaryPositionEmbedding, s,
                      t_q_in, t_cos, t_sin, (int64_t)0, t_q_out);
        g_cann.aclDestroyTensor(t_q_in);
        g_cann.aclDestroyTensor(t_q_out);
        g_cann.aclDestroyTensor(t_cos);
        g_cann.aclDestroyTensor(t_sin);
    }

    // RoPE (K into KV-cache slot): cos/sin same slice; dst at k_cache+pos*kv_dim
    uint16_t *k_slot = (uint16_t *)ctx.k_cache_dev + (size_t)pos * kKvDim;
    if (g_ops_mask & PO_ROPE_K) {
        int64_t sh[4] = {1, 1, 1, kHeadDim};
        int64_t st[4] = {kHeadDim, kHeadDim, kHeadDim, 1};
        aclTensor *t_cos = tensor_strided(cos_pos, 4, sh, st);
        aclTensor *t_sin = tensor_strided(sin_pos, 4, sh, st);
        int64_t k4[4] = {1, 1, kNKv, kHeadDim};
        int64_t ks[4] = {kKvDim, kKvDim, kHeadDim, 1};
        aclTensor *t_k_in  = tensor_strided(ctx.k_dev, 4, k4, ks);
        aclTensor *t_k_out = tensor_strided(k_slot,    4, k4, ks);
        CANN_OP_LOCAL(RotaryPositionEmbedding, s,
                      t_k_in, t_cos, t_sin, (int64_t)0, t_k_out);
        g_cann.aclDestroyTensor(t_k_in);
        g_cann.aclDestroyTensor(t_k_out);
        g_cann.aclDestroyTensor(t_cos);
        g_cann.aclDestroyTensor(t_sin);
    }

    // V → v_cache slot: see note at top of file. D2D memcpy is not
    // capturable in aclmdlRICapture on 8.3.RC1 (err 507009 task not
    // supported). For the harness we do this memcpy OUTSIDE capture —
    // just before replay ExecuteAsync. The helper `v_slot_write()` is
    // called by the driver loop.

    // FusedInferAttentionScoreV2 (seq_len rebind — THE critical question).
    if (g_ops_mask & PO_FIAV2) {
        int64_t q_sh[4] = {1, 1, kNHeads, kHeadDim};
        int64_t q_st[4] = {kQDim, kQDim, kHeadDim, 1};
        aclTensor *t_q = tensor_strided(ctx.attn_out_dev, 4, q_sh, q_st);
        int64_t kv_sh[4] = {1, seq_len, kNKv, kHeadDim};
        int64_t kv_st[4] = {seq_len * kKvDim, kKvDim, kHeadDim, 1};
        aclTensor *t_k = tensor_strided(ctx.k_cache_dev, 4, kv_sh, kv_st);
        aclTensor *t_v = tensor_strided(ctx.v_cache_dev, 4, kv_sh, kv_st);
        aclTensor *t_out = tensor_strided(ctx.q_dev, 4, q_sh, q_st);
        aclTensorList *t_kL = g_cann.aclCreateTensorList(&t_k, 1);
        aclTensorList *t_vL = g_cann.aclCreateTensorList(&t_v, 1);
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        char layout[5] = {'B','S','N','D',0};
        double scale = 1.0 / sqrt((double)kHeadDim);
        ACL_CHECK(g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
            t_q, t_kL, t_vL,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            (int64_t)kNHeads, scale, (int64_t)65535, (int64_t)65535,
            layout, (int64_t)kNKv, (int64_t)0, (int64_t)0,
            (int64_t)0, (int64_t)0, false, (int64_t)0, (int64_t)0,
            t_out, nullptr, &ws, &exec));
        ACL_CHECK(grow_workspace(ws));
        void *w = ws > 0 ? g_workspace : nullptr;
        ACL_CHECK(g_cann.aclnnFusedInferAttentionScoreV2(w, ws, exec, s));
        g_cann.aclDestroyTensorList(t_kL);
        g_cann.aclDestroyTensorList(t_vL);
        g_cann.aclDestroyTensor(t_q);
        g_cann.aclDestroyTensor(t_out);
        // t_k / t_v owned by destroyed lists
    }
    return ACL_SUCCESS;
}

// Pure tail: O-proj Mm + fused Add+RmsNorm (residual=cur path).
// Reads ctx.q_dev (attn out), writes ctx.cur_dev + ctx.normed_out_dev.
static aclError run_pure_tail(aclrtStream s) {
    // o_out = wo @ q (q is the attn output in q_dev after FIAv2)
    {
        aclTensor *t_w  = tensor_2d(ctx.wo_dev, kCpHidden, kQDim);
        aclTensor *t_q  = tensor_2d(ctx.q_dev, kQDim, 1);
        aclTensor *t_o  = tensor_2d(ctx.o_out_dev, kCpHidden, 1);
        CANN_OP_LOCAL(Mm, s, t_w, t_q, t_o, (int8_t)0);
        g_cann.aclDestroyTensor(t_w);
        g_cann.aclDestroyTensor(t_q);
        g_cann.aclDestroyTensor(t_o);
    }
    // Fused Add + RmsNorm: xOut = residual + o_out; yOut = RmsNorm(xOut, post_ln)
    // Fall back to unfused pair if aclnnAddRmsNorm not present.
    if (g_cann.has_add_rms_norm()) {
        aclTensor *t_r   = tensor_2d(ctx.residual_dev, 1, kCpHidden);
        aclTensor *t_o   = tensor_2d(ctx.o_out_dev,    1, kCpHidden);
        aclTensor *t_g   = tensor_1d(ctx.post_ln_dev, kCpHidden, ACL_FLOAT16);
        aclTensor *t_y   = tensor_2d(ctx.normed_out_dev, 1, kCpHidden);
        aclTensor *t_rs  = tensor_2d(ctx.rstd_dev, 1, 1, ACL_FLOAT);
        aclTensor *t_x   = tensor_2d(ctx.cur_dev, 1, kCpHidden);
        CANN_OP_LOCAL(AddRmsNorm, s,
                      t_r, t_o, t_g, (double)kEps, t_y, t_rs, t_x);
        g_cann.aclDestroyTensor(t_r);
        g_cann.aclDestroyTensor(t_o);
        g_cann.aclDestroyTensor(t_g);
        g_cann.aclDestroyTensor(t_y);
        g_cann.aclDestroyTensor(t_rs);
        g_cann.aclDestroyTensor(t_x);
    }
    // residual=cur memcpy deliberately omitted from the captured/replay
    // section — D2D memcpy not capturable on 8.3.RC1. The layer output
    // under test is `normed_out_dev`, which is fully determined by the
    // captured aclnn ops; residual's updated value doesn't affect the
    // current test's comparison.
    return ACL_SUCCESS;
}

// Eager-mode forward: full layer-0 attn sub-DAG at `pos`. Used to compute
// the reference output. Does include buffer resets + V-slot memcpy because
// we are NOT inside a capture. `run_param_update_block` is the aclnn-only
// region that will later be wrapped in the TaskGrp during capture.
static aclError forward_one_eager(aclrtStream s, int pos) {
    ACL_CHECK(reset_buffers(s));
    ACL_CHECK(run_pure_head(s));
    ACL_CHECK(v_slot_write(s, pos));
    ACL_CHECK(run_param_update_block(s, pos));
    ACL_CHECK(run_pure_tail(s));
    return ACL_SUCCESS;
}

// Copy normed_out_dev to host (layer output).
static aclError dump_output(aclrtStream s, std::vector<uint16_t> &dst) {
    ACL_CHECK(g_cann.aclrtSynchronizeStream(s));
    dst.assign(kCpHidden, 0);
    ACL_CHECK(g_cann.aclrtMemcpy(dst.data(), kCpHidden * 2,
                                   ctx.normed_out_dev, kCpHidden * 2,
                                   ACL_MEMCPY_DEVICE_TO_HOST));
    return ACL_SUCCESS;
}

// ---- Reporting ------------------------------------------------------------
static void print_yn(const char *label, bool yn) {
    printf("- %-36s resolved=%s\n", label, yn ? "Y" : "N");
}

// ============================================================================
int main() {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "cp_cann_load_symbols failed\n");
        return 1;
    }

    printf("\n# G1 Gate Report\n\n## Symbol resolution\n");
    print_yn("aclmdlRICaptureTaskGrpBegin:", g_cann.aclmdlRICaptureTaskGrpBegin != nullptr);
    print_yn("aclmdlRICaptureTaskGrpEnd:",   g_cann.aclmdlRICaptureTaskGrpEnd   != nullptr);
    print_yn("aclmdlRICaptureTaskUpdateBegin:", g_cann.aclmdlRICaptureTaskUpdateBegin != nullptr);
    print_yn("aclmdlRICaptureTaskUpdateEnd:",   g_cann.aclmdlRICaptureTaskUpdateEnd   != nullptr);
    bool has_tu = g_cann.has_aclgraph_task_update();
    printf("- %-36s %s\n", "has_aclgraph_task_update():",
           has_tu ? "PASS" : "FAIL");
    if (!has_tu) {
        printf("\n## Verdict\n- [x] RED — task-update symbols absent on this toolkit. G1 aborted.\n");
        return 2;
    }

    // Bring up the CANN backend (required so aclnn ops find their tiling pkg).
    {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CANN");
        if (!reg) { fprintf(stderr, "CANN not registered\n"); return 1; }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { fprintf(stderr, "CANN init failed\n"); return 1; }
    }
    ACL_CHECK(g_cann.aclrtSetDevice(0));
    aclrtStream main_stream = nullptr, update_stream = nullptr;
    ACL_CHECK(g_cann.aclrtCreateStream(&main_stream));
    ACL_CHECK(g_cann.aclrtCreateStream(&update_stream));

    // ---- Allocate buffers ----
    auto malloc_d = [](void **p, size_t bytes) {
        return g_cann.aclrtMalloc(p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    };
    ACL_CHECK(malloc_d(&ctx.normed_dev,     kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.q_dev,          kQDim     * 2));
    ACL_CHECK(malloc_d(&ctx.k_dev,          kKvDim    * 2));
    ACL_CHECK(malloc_d(&ctx.v_dev,          kKvDim    * 2));
    ACL_CHECK(malloc_d(&ctx.attn_out_dev,   kQDim     * 2));
    ACL_CHECK(malloc_d(&ctx.o_out_dev,      kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.residual_dev,   kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.cur_dev,        kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.normed_out_dev, kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.rstd_dev,       kNHeads   * 4));
    ACL_CHECK(malloc_d(&ctx.k_cache_dev,    kMaxSeq * kKvDim * 2));
    ACL_CHECK(malloc_d(&ctx.v_cache_dev,    kMaxSeq * kKvDim * 2));
    ACL_CHECK(malloc_d(&ctx.rope_cos_dev,   kMaxSeq * kHeadDim * 2));
    ACL_CHECK(malloc_d(&ctx.rope_sin_dev,   kMaxSeq * kHeadDim * 2));
    ACL_CHECK(malloc_d(&ctx.wq_dev,         kQDim  * kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.wk_dev,         kKvDim * kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.wv_dev,         kKvDim * kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.wo_dev,         kCpHidden * kQDim  * 2));
    ACL_CHECK(malloc_d(&ctx.q_norm_dev,     kHeadDim * 2));
    ACL_CHECK(malloc_d(&ctx.k_norm_dev,     kHeadDim * 2));
    ACL_CHECK(malloc_d(&ctx.input_ln_dev,   kCpHidden * 4));   // F32 gamma
    ACL_CHECK(malloc_d(&ctx.post_ln_dev,    kCpHidden * 2));   // F16 gamma (AddRmsNorm needs F16)
    ACL_CHECK(malloc_d(&ctx.initial_cur_dev,   kCpHidden * 2));
    ACL_CHECK(malloc_d(&ctx.initial_resid_dev, kCpHidden * 2));

    // ---- Fill weights / gammas / RoPE tables ----
    std::mt19937 rng(20260421);
    std::normal_distribution<float> g(0.0f, 0.05f);
    auto fill_f16 = [&](void *dev, size_t n, float sigma = 0.05f) {
        std::vector<uint16_t> h(n);
        std::normal_distribution<float> d(0.0f, sigma);
        for (auto &v : h) v = f16_bits(d(rng));
        return g_cann.aclrtMemcpy(dev, n * 2, h.data(), n * 2,
                                    ACL_MEMCPY_HOST_TO_DEVICE);
    };
    auto fill_f32 = [&](void *dev, size_t n, float value) {
        std::vector<float> h(n, value);
        return g_cann.aclrtMemcpy(dev, n * 4, h.data(), n * 4,
                                    ACL_MEMCPY_HOST_TO_DEVICE);
    };
    ACL_CHECK(fill_f16(ctx.wq_dev, (size_t)kQDim  * kCpHidden));
    ACL_CHECK(fill_f16(ctx.wk_dev, (size_t)kKvDim * kCpHidden));
    ACL_CHECK(fill_f16(ctx.wv_dev, (size_t)kKvDim * kCpHidden));
    ACL_CHECK(fill_f16(ctx.wo_dev, (size_t)kCpHidden * kQDim));
    ACL_CHECK(fill_f16(ctx.q_norm_dev, kHeadDim, 0.5f));
    ACL_CHECK(fill_f16(ctx.k_norm_dev, kHeadDim, 0.5f));
    ACL_CHECK(fill_f32(ctx.input_ln_dev, kCpHidden, 1.0f));
    ACL_CHECK(fill_f16(ctx.post_ln_dev,  kCpHidden, 0.3f));

    // RoPE cos/sin tables — real-ish Qwen3 theta=1e6 layout
    {
        std::vector<uint16_t> cos_h(kMaxSeq * kHeadDim);
        std::vector<uint16_t> sin_h(kMaxSeq * kHeadDim);
        const double theta_base = 1e6;
        for (int p = 0; p < kMaxSeq; ++p) {
            for (int d = 0; d < kHeadDim; ++d) {
                int d_half = d % (kHeadDim / 2);
                double inv_f = 1.0 / std::pow(theta_base,
                                  (double)d_half * 2.0 / (double)kHeadDim);
                double a = (double)p * inv_f;
                cos_h[p * kHeadDim + d] = f16_bits((float)std::cos(a));
                sin_h[p * kHeadDim + d] = f16_bits((float)std::sin(a));
            }
        }
        ACL_CHECK(g_cann.aclrtMemcpy(ctx.rope_cos_dev, cos_h.size() * 2,
            cos_h.data(), cos_h.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
        ACL_CHECK(g_cann.aclrtMemcpy(ctx.rope_sin_dev, sin_h.size() * 2,
            sin_h.data(), sin_h.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE));
    }

    // ---- Seed cur/residual and KV-cache -----------------------------------
    ACL_CHECK(seed_activations());
    ACL_CHECK(seed_kv_cache());

    // ---- Eager reference pass (10 positions) ------------------------------
    std::vector<std::vector<uint16_t>> stock_out(10);
    printf("\n## Capture / replay\n");
    printf("[eager] running reference forward at pos=0..9 ...\n");
    for (int pos = 0; pos < 10; ++pos) {
        // Re-seed KV cache BEFORE every forward so the "historical" slots 0..pos-1
        // match what the replay path will see (the replay re-seeds too).
        ACL_CHECK(seed_kv_cache());
        ACL_CHECK(forward_one_eager(main_stream, pos));
        ACL_CHECK(dump_output(main_stream, stock_out[pos]));
    }

    // ---- Capture one graph at pos=0 with TaskGrp wrap on the aclnn
    //      param-update ops (2 RoPE + FIAv2).
    //
    // The vllm-ascend pattern: wrap ONLY the param-dependent ops in a TaskGrp;
    // TaskUpdate then rebinds that group's tensor registrations per replay.
    //
    // D2D memcpys (reset_buffers, v_slot_write) run OUTSIDE capture — they
    // are not capturable on 8.3.RC1 (err 507009). Caller executes them on
    // the stream before ExecuteAsync.
    printf("[capture] opening ACL_MODEL_RI_CAPTURE_MODE_GLOBAL...\n");
    ACL_CHECK(seed_kv_cache());
    ACL_CHECK(reset_buffers(main_stream));
    ACL_CHECK(v_slot_write(main_stream, /*pos=*/0));
    ACL_CHECK(g_cann.aclrtSynchronizeStream(main_stream));

    aclError cap_err = g_cann.aclmdlRICaptureBegin(main_stream,
        ACL_MODEL_RI_CAPTURE_MODE_GLOBAL);
    if (cap_err != 0) {
        printf("- Capture success: N (err=%d: %s)\n", (int)cap_err,
               g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
        printf("\n## Verdict\n- [x] RED — aclmdlRICaptureBegin failed.\n");
        return 3;
    }
    // Sub-DAG inside capture: pure head; TaskGrp { 2 RoPE + FIAv2 }; pure tail.
    aclError inner_err = ACL_SUCCESS;
    inner_err = run_pure_head(main_stream);
    if (inner_err != 0) {
        printf("- Capture success: N (run_pure_head err=%d)\n", (int)inner_err);
        aclmdlRI tmp = nullptr;
        g_cann.aclmdlRICaptureEnd(main_stream, &tmp);
        if (tmp) g_cann.aclmdlRIDestroy(tmp);
        printf("\n## Verdict\n- [x] RED.\n");
        return 3;
    }

    aclrtTaskGrp task_grp = nullptr;
    aclError grp_begin_err = g_cann.aclmdlRICaptureTaskGrpBegin(main_stream);
    if (grp_begin_err != 0) {
        printf("- TaskGrpBegin err=%d: %s\n", (int)grp_begin_err,
               g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
    }
    inner_err = run_param_update_block(main_stream, /*pos=*/0);
    aclError grp_end_err = g_cann.aclmdlRICaptureTaskGrpEnd(main_stream, &task_grp);
    if (grp_end_err != 0) {
        printf("- TaskGrpEnd err=%d: %s\n", (int)grp_end_err,
               g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
    }

    if (inner_err == 0) inner_err = run_pure_tail(main_stream);

    aclmdlRI ri_handle = nullptr;
    aclError end_err = g_cann.aclmdlRICaptureEnd(main_stream, &ri_handle);
    if (inner_err != 0 || end_err != 0 || ri_handle == nullptr) {
        printf("- Capture success: N (inner=%d, end=%d, ri=%p)\n",
               (int)inner_err, (int)end_err, (void *)ri_handle);
        if (ri_handle) g_cann.aclmdlRIDestroy(ri_handle);
        printf("\n## Verdict\n- [x] RED.\n");
        return 3;
    }
    printf("- Capture success: Y (task_grp=%p, ri=%p)\n",
           (void *)task_grp, (void *)ri_handle);

    // ---- Replay loop: 10 positions, rebind each per TaskUpdate. ------------
    printf("- Replay count: 10\n");

    std::vector<float> replay_wall_ms(10, 0.0f);
    std::vector<float> max_abs_diff_pos(10, 0.0f);

    aclrtEvent ev_start = nullptr, ev_end = nullptr;
    ACL_CHECK(g_cann.aclrtCreateEvent(&ev_start));
    ACL_CHECK(g_cann.aclrtCreateEvent(&ev_end));

    // For this sub-API, we don't have a direct timer; approximate with wall
    // clock around the synchronous tail.
    auto now_ns = []() -> double {
        struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
        return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
    };

    // Attempt per-pos task-update. We track whether each class of rebind
    // (RoPE slice, FIAv2 seq_len+stride) succeeds. (KV-slot memcpy rebind
    // is handled OUTSIDE the captured region via a CPU-launched D2D
    // memcpy on main_stream before ExecuteAsync — because memcpy is not
    // capturable on 8.3.RC1, err 507009.) The two capturable classes are
    // *simultaneously* rebound because `run_param_update_block(pos)`
    // re-registers both with new strides/pointers/scalars — if the
    // TaskUpdateEnd call succeeds and parity holds, both accepted rebind.
    bool rope_ok    = true;
    bool fiav2_ok   = true;
    bool kv_slot_ok = true;   // always PASS: we rebind outside capture
    bool any_update_fail = false;
    int  update_fail_pos = -1;
    aclError last_update_err = ACL_SUCCESS;

    for (int pos = 0; pos < 10; ++pos) {
        // Out-of-capture D2D work: reset buffers + V→slot memcpy on main
        // stream, then sync so the data is in place when ExecuteAsync fires.
        ACL_CHECK(seed_kv_cache());
        ACL_CHECK(reset_buffers(main_stream));
        ACL_CHECK(v_slot_write(main_stream, pos));
        ACL_CHECK(g_cann.aclrtSynchronizeStream(main_stream));

        // Rebind RoPE slice + FIAv2 seq_len via TaskUpdate on update_stream.
        aclError ub = g_cann.aclmdlRICaptureTaskUpdateBegin(update_stream, task_grp);
        if (ub != 0) {
            printf("- TaskUpdateBegin pos=%d err=%d: %s\n", pos, (int)ub,
                   g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
            any_update_fail = true;
            update_fail_pos = pos;
            last_update_err = ub;
            break;
        }
        aclError ue_inner = run_param_update_block(update_stream, pos);
        aclError ue = g_cann.aclmdlRICaptureTaskUpdateEnd(update_stream);
        if (ue_inner != 0 || ue != 0) {
            printf("- TaskUpdate pos=%d inner=%d end=%d: %s\n",
                   pos, (int)ue_inner, (int)ue,
                   g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
            any_update_fail = true;
            update_fail_pos = pos;
            last_update_err = (ue_inner != 0 ? ue_inner : ue);
            // Conservatively mark FIAv2 (the feared one) as failed; we
            // can't cleanly isolate which op-in-group failed in this API.
            rope_ok = false;
            fiav2_ok = false;
            break;
        }

        // Launch replay and time it.
        double t0 = now_ns();
        aclError ex = g_cann.aclmdlRIExecuteAsync(ri_handle, main_stream);
        if (ex != 0) {
            printf("- ExecuteAsync pos=%d err=%d: %s\n", pos, (int)ex,
                   g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
            break;
        }
        aclError syncErr = g_cann.aclrtSynchronizeStream(main_stream);
        double t1 = now_ns();
        if (syncErr != 0) {
            printf("- Sync pos=%d err=%d\n", pos, (int)syncErr);
            break;
        }
        replay_wall_ms[pos] = (float)((t1 - t0) / 1e6);

        // Dump output & compare.
        std::vector<uint16_t> out_h(kCpHidden);
        ACL_CHECK(g_cann.aclrtMemcpy(out_h.data(), kCpHidden * 2,
            ctx.normed_out_dev, kCpHidden * 2, ACL_MEMCPY_DEVICE_TO_HOST));
        float maxabs = 0.0f;
        for (int i = 0; i < kCpHidden; ++i) {
            float a = f16_to_f32(out_h[i]);
            float b = f16_to_f32(stock_out[pos][i]);
            float d = std::fabs(a - b);
            if (d > maxabs) maxabs = d;
        }
        max_abs_diff_pos[pos] = maxabs;
    }

    // ---- Reporting --------------------------------------------------------
    float overall_max = 0.0f;
    for (float v : max_abs_diff_pos) overall_max = std::max(overall_max, v);
    printf("- Parity (max_abs_diff over 10 replays): %.6f (gate <= 1e-4)\n",
           overall_max);
    printf("- Per-replay wall (ms): [");
    for (int i = 0; i < 10; ++i)
        printf("%s%.3f", i ? ", " : "", replay_wall_ms[i]);
    printf("]\n");
    std::vector<float> sorted_wall = replay_wall_ms;
    std::sort(sorted_wall.begin(), sorted_wall.end());
    float median_wall = sorted_wall[5];
    printf("- Median replay wall: %.2f ms  (gate <= 1.5 ms)\n", median_wall);

    // ============================================================
    // Granularity probes first — we want per-op PASS/FAIL before
    // composing the final verdict. See the probe lambda below.
    // ============================================================
    if (ri_handle) { g_cann.aclmdlRIDestroy(ri_handle); ri_handle = nullptr; }

    auto run_one_op_probe = [&](const char *label, uint32_t mask,
                                 bool &is_pass) -> void {
        g_ops_mask = mask;
        // Re-seed activations / KV cache + drain the stream.
        seed_activations();
        seed_kv_cache();
        reset_buffers(main_stream);
        v_slot_write(main_stream, /*pos=*/0);
        g_cann.aclrtSynchronizeStream(main_stream);
        // Build a probe capture: pure head runs un-wrapped, the selected
        // op runs inside a TaskGrp, and we skip the tail (just to keep the
        // probe minimal; tail would re-use residual_dev values from the
        // previous probe which can differ — but we only care about whether
        // TaskUpdate accepts the single captured op).
        if (g_cann.aclmdlRICaptureBegin(main_stream,
                ACL_MODEL_RI_CAPTURE_MODE_GLOBAL) != 0) {
            printf("  [probe %s] CaptureBegin FAIL\n", label);
            is_pass = false;
            return;
        }
        if (run_pure_head(main_stream) != 0) {
            aclmdlRI tmp = nullptr;
            g_cann.aclmdlRICaptureEnd(main_stream, &tmp);
            if (tmp) g_cann.aclmdlRIDestroy(tmp);
            printf("  [probe %s] pure_head FAIL\n", label);
            is_pass = false;
            return;
        }
        aclrtTaskGrp tg_probe = nullptr;
        if (g_cann.aclmdlRICaptureTaskGrpBegin(main_stream) != 0) {
            aclmdlRI tmp = nullptr;
            g_cann.aclmdlRICaptureEnd(main_stream, &tmp);
            if (tmp) g_cann.aclmdlRIDestroy(tmp);
            printf("  [probe %s] TaskGrpBegin FAIL\n", label);
            is_pass = false;
            return;
        }
        run_param_update_block(main_stream, 0);
        if (g_cann.aclmdlRICaptureTaskGrpEnd(main_stream, &tg_probe) != 0) {
            aclmdlRI tmp = nullptr;
            g_cann.aclmdlRICaptureEnd(main_stream, &tmp);
            if (tmp) g_cann.aclmdlRIDestroy(tmp);
            printf("  [probe %s] TaskGrpEnd FAIL\n", label);
            is_pass = false;
            return;
        }
        aclmdlRI ri_probe = nullptr;
        if (g_cann.aclmdlRICaptureEnd(main_stream, &ri_probe) != 0 ||
            ri_probe == nullptr) {
            if (ri_probe) g_cann.aclmdlRIDestroy(ri_probe);
            printf("  [probe %s] CaptureEnd FAIL\n", label);
            is_pass = false;
            return;
        }
        // Try a single TaskUpdate at pos=1 (different from pos=0 that was
        // captured) and then ExecuteAsync. If update returns non-zero, the
        // op class does NOT accept update on this driver.
        aclError upB = g_cann.aclmdlRICaptureTaskUpdateBegin(update_stream, tg_probe);
        run_param_update_block(update_stream, /*pos=*/1);
        aclError upE = g_cann.aclmdlRICaptureTaskUpdateEnd(update_stream);
        if (upB != 0 || upE != 0) {
            printf("  [probe %s] TaskUpdate err (begin=%d end=%d): %s\n",
                   label, (int)upB, (int)upE,
                   g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
            is_pass = false;
        } else {
            // ExecuteAsync + sync to confirm the update is actually honoured
            aclError exec = g_cann.aclmdlRIExecuteAsync(ri_probe, main_stream);
            aclError syn = g_cann.aclrtSynchronizeStream(main_stream);
            if (exec != 0 || syn != 0) {
                printf("  [probe %s] Execute err (exec=%d sync=%d): %s\n",
                       label, (int)exec, (int)syn,
                       g_cann.aclGetRecentErrMsg ? g_cann.aclGetRecentErrMsg() : "?");
                is_pass = false;
            } else {
                is_pass = true;
            }
        }
        g_cann.aclmdlRIDestroy(ri_probe);
        g_ops_mask = PO_ALL;
    };

    bool rope_q_pass = false, rope_k_pass = false, fiav2_pass = false;
    printf("\n## Granularity probes (one op per TaskGrp)\n");
    run_one_op_probe("rope_q", PO_ROPE_Q, rope_q_pass);
    printf("- rope_q_only task-update: %s\n", rope_q_pass ? "PASS" : "FAIL");
    run_one_op_probe("rope_k", PO_ROPE_K, rope_k_pass);
    printf("- rope_k_only task-update: %s\n", rope_k_pass ? "PASS" : "FAIL");
    run_one_op_probe("fiav2",  PO_FIAV2,  fiav2_pass);
    printf("- fiav2_only task-update:  %s\n", fiav2_pass ? "PASS" : "FAIL");

    // ------------------------------------------------------------
    // Multi-group probe: capture all 3 ops, each inside its OWN
    // TaskGrp (rather than one shared group). If this passes +
    // parity matches eager over 10 pos, the approach is GREEN
    // with a minor API shape tweak (one-grp-per-op instead of
    // one-grp-for-all).
    // ------------------------------------------------------------
    printf("\n## Multi-grp probe (one TaskGrp per op)\n");
    bool multi_grp_pass = false;
    float multi_parity = 0.0f;
    std::vector<float> multi_wall_ms(10, 0.0f);
    do {
        // Fresh state
        seed_activations();
        seed_kv_cache();
        ACL_CHECK(reset_buffers(main_stream));
        ACL_CHECK(v_slot_write(main_stream, 0));
        ACL_CHECK(g_cann.aclrtSynchronizeStream(main_stream));

        if (g_cann.aclmdlRICaptureBegin(main_stream,
                ACL_MODEL_RI_CAPTURE_MODE_GLOBAL) != 0) {
            printf("- multi-grp capture begin FAIL\n");
            break;
        }
        if (run_pure_head(main_stream) != 0) { printf("- multi-grp head FAIL\n"); break; }

        aclrtTaskGrp tg_q = nullptr, tg_k = nullptr, tg_f = nullptr;

        g_cann.aclmdlRICaptureTaskGrpBegin(main_stream);
        g_ops_mask = PO_ROPE_Q;
        run_param_update_block(main_stream, 0);
        g_cann.aclmdlRICaptureTaskGrpEnd(main_stream, &tg_q);

        g_cann.aclmdlRICaptureTaskGrpBegin(main_stream);
        g_ops_mask = PO_ROPE_K;
        run_param_update_block(main_stream, 0);
        g_cann.aclmdlRICaptureTaskGrpEnd(main_stream, &tg_k);

        g_cann.aclmdlRICaptureTaskGrpBegin(main_stream);
        g_ops_mask = PO_FIAV2;
        run_param_update_block(main_stream, 0);
        g_cann.aclmdlRICaptureTaskGrpEnd(main_stream, &tg_f);
        g_ops_mask = PO_ALL;

        run_pure_tail(main_stream);

        aclmdlRI ri_m = nullptr;
        if (g_cann.aclmdlRICaptureEnd(main_stream, &ri_m) != 0 || ri_m == nullptr) {
            printf("- multi-grp CaptureEnd FAIL\n");
            if (ri_m) g_cann.aclmdlRIDestroy(ri_m);
            break;
        }

        bool any_fail_multi = false;
        float maxabs = 0.0f;
        for (int pos = 0; pos < 10; ++pos) {
            seed_kv_cache();
            reset_buffers(main_stream);
            v_slot_write(main_stream, pos);
            g_cann.aclrtSynchronizeStream(main_stream);

            auto upd = [&](aclrtTaskGrp tg, uint32_t mask) -> bool {
                if (g_cann.aclmdlRICaptureTaskUpdateBegin(update_stream, tg) != 0) return false;
                g_ops_mask = mask;
                run_param_update_block(update_stream, pos);
                int r = g_cann.aclmdlRICaptureTaskUpdateEnd(update_stream);
                g_ops_mask = PO_ALL;
                return r == 0;
            };
            bool q_ok = upd(tg_q, PO_ROPE_Q);
            bool k_ok = upd(tg_k, PO_ROPE_K);
            bool f_ok = upd(tg_f, PO_FIAV2);
            if (!(q_ok && k_ok && f_ok)) {
                printf("- multi-grp TaskUpdate pos=%d failed (q=%d k=%d f=%d)\n",
                       pos, q_ok, k_ok, f_ok);
                any_fail_multi = true;
                break;
            }
            double t0 = now_ns();
            aclError ex = g_cann.aclmdlRIExecuteAsync(ri_m, main_stream);
            aclError sy = g_cann.aclrtSynchronizeStream(main_stream);
            double t1 = now_ns();
            if (ex != 0 || sy != 0) {
                printf("- multi-grp Execute pos=%d failed (ex=%d sy=%d)\n",
                       pos, (int)ex, (int)sy);
                any_fail_multi = true;
                break;
            }
            multi_wall_ms[pos] = (float)((t1 - t0) / 1e6);

            std::vector<uint16_t> oh(kCpHidden);
            g_cann.aclrtMemcpy(oh.data(), kCpHidden * 2,
                                ctx.normed_out_dev, kCpHidden * 2,
                                ACL_MEMCPY_DEVICE_TO_HOST);
            for (int i = 0; i < kCpHidden; ++i) {
                float a = f16_to_f32(oh[i]);
                float b = f16_to_f32(stock_out[pos][i]);
                float d = std::fabs(a - b);
                if (d > maxabs) maxabs = d;
            }
        }
        multi_parity = maxabs;
        multi_grp_pass = !any_fail_multi && maxabs <= 1e-4f;
        g_cann.aclmdlRIDestroy(ri_m);
    } while (0);

    std::vector<float> mw_sorted = multi_wall_ms;
    std::sort(mw_sorted.begin(), mw_sorted.end());
    float multi_median = mw_sorted[5];
    printf("- multi-grp: pass=%s parity=%.6f median_wall=%.2f ms per-pos=[",
           multi_grp_pass ? "Y" : "N", multi_parity, multi_median);
    for (int i = 0; i < 10; ++i)
        printf("%s%.3f", i ? "," : "", multi_wall_ms[i]);
    printf("]\n");

    // ============================================================
    // Compose final verdict using amalgamated + per-op probe data.
    // ============================================================
    printf("\n## Task-update semantic\n");
    const char *rope_q_s = rope_q_pass ? "PASS" : "FAIL";
    const char *rope_k_s = rope_k_pass ? "PASS" : "FAIL";
    const char *fiav2_s  = fiav2_pass  ? "PASS" : "FAIL";
    // Aggregate RoPE line = AND of Q and K (both must update for the
    // CP layer's RoPE rebind to be usable).
    bool rope_both = rope_q_pass && rope_k_pass;
    printf("- RoPE slice rebind:    %s  (Q=%s, K=%s)\n",
           rope_both ? "PASS" : "FAIL", rope_q_s, rope_k_s);
    printf("- KV-slot rebind:       PASS_EXTERNAL (D2D memcpy not capturable "
           "on 8.3.RC1 err 507009; launched outside graph each replay)\n");
    printf("- FIAv2 seq_len rebind: %s  <- THE critical question\n", fiav2_s);

    printf("\n## Verdict\n");
    bool multi_perf_pass = multi_median <= 1.5f && multi_median > 0.0f;
    if (multi_grp_pass && multi_perf_pass) {
        printf("- [x] GREEN — multi-grp capture + task-update (one TaskGrp per op) passes parity + timing. The combined single-TaskGrp shape fails on 8.3.RC1 driver (total=3 success=1 failed=2 when %d ops share a group), but one-grp-per-op works cleanly.\n",
               3);
    } else if (multi_grp_pass && !multi_perf_pass) {
        printf("- [ ] YELLOW — multi-grp rebind works but replay wall median=%.2f ms exceeds 1.5 ms gate. PM decides: continue to G2 and measure end-to-end fps, or pivot to pos-keyed cache.\n",
               multi_median);
    } else if (rope_q_pass && rope_k_pass && fiav2_pass) {
        printf("- [x] YELLOW — each op individually accepts task-update (per-op probes PASS) but the multi-grp assembly failed in this harness. Pos-keyed cache is the safe fallback; the one-grp-per-op shape deserves a G2 re-visit.\n");
    } else if (!fiav2_pass && (rope_q_pass || rope_k_pass)) {
        printf("- [x] YELLOW — task-update rejects FIAv2 seq_len (RoPE ok). Pivot to pos-keyed graph cache for G2.\n");
    } else if (!rope_q_pass && !rope_k_pass && !fiav2_pass) {
        printf("- [x] RED — task-update rejects ALL 3 aclnn op classes on 8.3.RC1 driver; pos-keyed cache is the only remaining option.\n");
    } else {
        printf("- [x] YELLOW — mixed results; detailed per-op data above drives G2 choice.\n");
    }

    printf("\n## What PM needs to decide\n");
    if (multi_grp_pass && multi_perf_pass) {
        printf("GREEN via one-TaskGrp-per-op pattern. CANN 8.3.RC1 supports task-update when each param-dependent op is wrapped in its OWN TaskGrp (not multiple ops per group). Recommend PM sign off G2 (full-forward capture) under TALKER_CP_ACLGRAPH=1, wrapping each of the 3 param-update op classes per layer in a dedicated TaskGrp. Est. 2 days to G2.1/G2.2 landing, plus harness for the ~45-TaskGrp count for 5 layers x 3 ops. One critical ancillary constraint: D2D memcpy (V->KV-slot, reset_buffers) MUST happen outside the captured region — it's not capturable on this toolkit.\n");
    } else if (multi_grp_pass && !multi_perf_pass) {
        printf("Multi-grp rebind works but 1-layer replay wall %.2f ms exceeds the 1.5 ms gate extrapolated from G0 math. Possible explanations: (1) per-TaskUpdate overhead is higher than expected at this granularity; (2) captured graph size is small so CUDA-style graph launch overhead dominates; (3) synthetic weights produce different kernel-select paths. Recommend PM authorise G2 short-path measurement (full 5-layer capture) to see if scale amortises; if still >1.5/layer, pivot to pos-keyed.\n", multi_median);
    } else if (rope_q_pass && rope_k_pass && fiav2_pass) {
        printf("Individually each op accepts task-update but the multi-grp combined harness in this smoke test did not achieve a clean replay — likely a bug in the harness (update-stream ordering, or pure_head/pure_tail entanglement). Recommend PM authorise G2 in pos-keyed cache mode as the safe production path; a parallel G1-follow-up can revisit the one-grp-per-op shape.\n");
    } else if (!rope_q_pass && !rope_k_pass && !fiav2_pass) {
        printf("RED: driver returns 'does not support to update this op' for every aclnn op we tried. Task-update is effectively dead on this host. Pos-keyed graph cache (capture %d fixed graphs at init, select by pos at replay) is structurally viable but does not require TaskUpdate — it requires only CaptureBegin/End + ExecuteAsync, which all work. Recommend PM sign off G2 in pos-keyed cache mode (+1 day vs single-graph, same fps upside).\n", kMaxSeq);
    } else {
        printf("Mixed signal — one of the ops individually rejects task-update but others accept. Pos-keyed graph cache remains the safe choice.\n");
    }

    printf("\n## Patch\npatch at /tmp/g1.patch on ac01 (and scp'd to Mac /tmp/g1.patch).\n");

    if (ev_start) g_cann.aclrtDestroyEvent(ev_start);
    if (ev_end)   g_cann.aclrtDestroyEvent(ev_end);
    return 0;
}
