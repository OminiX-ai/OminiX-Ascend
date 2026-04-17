// ============================================================================
// Talker CANN Engine — native 28-layer Qwen3 backbone on Ascend NPU.
//
// Mirrors the CpCannEngine pattern (F16 transformer compute, F32 norm gammas,
// F32 I/O staging) but targets the full Talker model:
//   - 28 layers instead of 5
//   - n_embd = 2048, inter = 6144
//   - MAX_SEQ = 4096 (vs CP's 17) to handle long prefill + many codec frames
//   - Accepts float embeddings from caller (no token embedding lookup here —
//     TalkerLLM does text_projection + codec_embedding on CPU)
//
// Precision pedigree: follows ggml-cann's Qwen3 recipe exactly (the reference
// path that sounds right at 12 fps today).
// ============================================================================

#include "talker_cann_engine.h"
#include "cp_cann_symbols.h"
#include "talker.h"

// ggml for GGUF loading
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <limits>

#define ACL_CHECK_RET(stmt) do {                                      \
    aclError _ret = (stmt);                                           \
    if (_ret != 0) {                                                  \
        fprintf(stderr, "[talker_cann] ACL error %d at %s:%d: %s\n",  \
                _ret, __FILE__, __LINE__,                             \
                g_cann.aclGetRecentErrMsg                             \
                    ? g_cann.aclGetRecentErrMsg() : "<n/a>");         \
    }                                                                 \
} while (0)

// Shared workspace macro: auto-grow the engine's workspace buffer so the
// current op has enough scratch memory, then issue the op. Matches the
// pattern used in CpCannEngine.
#define CANN_OP(OP_NAME, ...) do {                                    \
    uint64_t _ws_needed = 0;                                          \
    aclOpExecutor *_exec = nullptr;                                   \
    ACL_CHECK_RET(g_cann.aclnn##OP_NAME##GetWorkspaceSize(            \
        __VA_ARGS__, &_ws_needed, &_exec));                           \
    ensure_workspace_(_ws_needed);                                    \
    void *_ws = _ws_needed > 0 ? workspace_dev_ : nullptr;            \
    ACL_CHECK_RET(g_cann.aclnn##OP_NAME(_ws, _ws_needed, _exec,       \
                                         stream_));                   \
} while (0)

// ---------------------------------------------------------------------------
// fp32 <-> fp16 helpers (aarch64 native __fp16).
// ---------------------------------------------------------------------------

namespace {

inline uint16_t fp32_to_fp16(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

inline float fp16_to_fp32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

aclScalar *make_f16_scalar(float value) {
    uint16_t bits = fp32_to_fp16(value);
    return g_cann.aclCreateScalar(&bits, ACL_FLOAT16);
}

// Upload a contiguous host F16 buffer (already bit-exact F16) to device.
void upload_f16_bits(void *dev, const uint16_t *host, size_t n) {
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        dev, n * sizeof(uint16_t),
        host, n * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
}

// Convert F32 host -> F16 host buffer and upload.
void upload_f32_as_f16(void *dev, const float *host, size_t n) {
    std::vector<uint16_t> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = fp32_to_fp16(host[i]);
    upload_f16_bits(dev, buf.data(), n);
}

// ---------------------------------------------------------------------------
// Tensor factory helpers — brief wrappers around g_cann.aclCreateTensor.
// ---------------------------------------------------------------------------

aclTensor *make_tensor(void *buf, int64_t rank,
                        const int64_t *shape, const int64_t *strides,
                        aclDataType dtype) {
    // Compute storage_len as the maximum offset the strided view can reach
    // (plus one). For contiguous views this equals shape-product, but for
    // strided views over larger buffers (KV cache slices, mask subarrays)
    // it correctly reflects the extent that ACL must be able to reach.
    // Getting this wrong produces "ViewShape overlap" errors from ACL.
    int64_t storage_len = 0;
    if (rank > 0) {
        int64_t max_off = 0;
        for (int64_t i = 0; i < rank; ++i) {
            if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
        }
        storage_len = max_off + 1;
    }
    return g_cann.aclCreateTensor(shape, rank, dtype, strides, 0,
                                   ACL_FORMAT_ND, &storage_len, 1, buf);
}

aclTensor *tensor_1d(void *buf, int64_t n, aclDataType dtype) {
    int64_t shape[1]   = {n};
    int64_t strides[1] = {1};
    return make_tensor(buf, 1, shape, strides, dtype);
}

aclTensor *tensor_2d(void *buf, int64_t d0, int64_t d1, aclDataType dtype) {
    int64_t shape[2]   = {d0, d1};
    int64_t strides[2] = {d1, 1};
    return make_tensor(buf, 2, shape, strides, dtype);
}

aclTensor *tensor_strided(void *buf, int64_t rank,
                           const int64_t *shape, const int64_t *strides,
                           aclDataType dtype) {
    return make_tensor(buf, rank, shape, strides, dtype);
}

// Pull tensor data out of a GGUF as float (handles F32 and F16 sources).
// Returns empty vector on failure.
std::vector<float> load_gguf_tensor_f32(ggml_context *ggml_ctx,
                                         const char *name,
                                         size_t expected_elems) {
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (!t) {
        fprintf(stderr, "[talker_cann] missing tensor: %s\n", name);
        return {};
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        fprintf(stderr,
                "[talker_cann] %s: expected %zu elems, got %zu\n",
                name, expected_elems, n);
        return {};
    }
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; ++i) out[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        fprintf(stderr, "[talker_cann] %s: unsupported dtype %d\n",
                name, (int)t->type);
        return {};
    }
    return out;
}

// Load tensor and upload as F16 (bit-convert F32 -> F16 on host). Used for
// matmul weights. Returns true on success.
bool upload_tensor_f16(ggml_context *ggml_ctx, const char *name,
                       size_t expected_elems, void *&dev) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    // Bit-convert to F16 on host.
    std::vector<uint16_t> f16(expected_elems);
    for (size_t i = 0; i < expected_elems; ++i) f16[i] = fp32_to_fp16(host[i]);
    aclError err = g_cann.aclrtMalloc(&dev, expected_elems * sizeof(uint16_t),
                                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[talker_cann] aclrtMalloc failed for %s\n", name);
        return false;
    }
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        dev, expected_elems * sizeof(uint16_t),
        f16.data(), expected_elems * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    return true;
}

// Load tensor and upload as F32 (norm gammas).
bool upload_tensor_f32(ggml_context *ggml_ctx, const char *name,
                       size_t expected_elems, void *&dev) {
    std::vector<float> host = load_gguf_tensor_f32(ggml_ctx, name,
                                                    expected_elems);
    if (host.empty()) return false;
    aclError err = g_cann.aclrtMalloc(&dev, expected_elems * sizeof(float),
                                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[talker_cann] aclrtMalloc failed for %s\n", name);
        return false;
    }
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        dev, expected_elems * sizeof(float),
        host.data(), expected_elems * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));
    return true;
}

}  // namespace

// ============================================================================
// Lifecycle
// ============================================================================

TalkerCannEngine::~TalkerCannEngine() {
    if (!ready_) return;

    auto free_dev = [](void *&p) {
        if (p) { g_cann.aclrtFree(p); p = nullptr; }
    };

    for (auto &lw : layer_w_) {
        free_dev(lw.q_proj_w);
        free_dev(lw.k_proj_w);
        free_dev(lw.v_proj_w);
        free_dev(lw.o_proj_w);
        free_dev(lw.q_norm_w);
        free_dev(lw.k_norm_w);
        free_dev(lw.gate_proj_w);
        free_dev(lw.up_proj_w);
        free_dev(lw.down_proj_w);
        free_dev(lw.input_ln_w);
        free_dev(lw.post_ln_w);
    }
    free_dev(final_norm_w_dev_);

    free_dev(cur_dev_);
    free_dev(residual_dev_);
    free_dev(normed_dev_);
    free_dev(q_dev_);
    free_dev(k_dev_);
    free_dev(v_dev_);
    free_dev(attn_out_dev_);
    free_dev(o_out_dev_);
    free_dev(gate_dev_);
    free_dev(up_dev_);
    free_dev(ffn_out_dev_);

    free_dev(cur_batch_dev_);
    free_dev(residual_batch_dev_);
    free_dev(normed_batch_dev_);
    free_dev(q_batch_dev_);
    free_dev(k_batch_dev_);
    free_dev(v_batch_dev_);
    free_dev(attn_out_batch_dev_);
    free_dev(o_out_batch_dev_);
    free_dev(gate_batch_dev_);
    free_dev(up_batch_dev_);
    free_dev(ffn_out_batch_dev_);

    free_dev(causal_mask_dev_);
    free_dev(rope_cos_dev_);
    free_dev(rope_sin_dev_);
    free_dev(rstd_dev_);

    for (auto &p : k_cache_dev_) free_dev(p);
    for (auto &p : v_cache_dev_) free_dev(p);

    free_dev(workspace_dev_);
    free_dev(input_stage_f32_dev_);
    free_dev(output_stage_f32_dev_);

    if (one_scalar_f16_) {
        g_cann.aclDestroyScalar(one_scalar_f16_);
        one_scalar_f16_ = nullptr;
    }

    if (stream_) {
        g_cann.aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

// ============================================================================
// Device-side helpers
// ============================================================================

void TalkerCannEngine::alloc_dev_(void **ptr, size_t bytes) {
    ACL_CHECK_RET(g_cann.aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void TalkerCannEngine::ensure_workspace_(size_t needed) {
    if (needed <= workspace_size_) return;
    if (workspace_dev_) g_cann.aclrtFree(workspace_dev_);
    ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, needed,
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    workspace_size_ = needed;
}

void TalkerCannEngine::upload_(void *dev, const void *host, size_t bytes) {
    ACL_CHECK_RET(g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                                      ACL_MEMCPY_HOST_TO_DEVICE));
}

// ============================================================================
// RoPE cos/sin table construction. Uses rope_speed_factor_ to scale angles —
// callers change it via set_rope_speed_factor, which triggers a rebuild.
// ============================================================================

void TalkerCannEngine::build_rope_tables_() {
    const int half = head_dim_ / 2;
    cos_host_.assign((size_t)MAX_SEQ * head_dim_, 0.0f);
    sin_host_.assign((size_t)MAX_SEQ * head_dim_, 0.0f);
    for (int p = 0; p < MAX_SEQ; ++p) {
        // The speed factor scales the position offset — matches MLX's
        // rope_offset = (pos * rope_speed_factor) pattern.
        float pos = (float)p * rope_speed_factor_;
        for (int j = 0; j < half; ++j) {
            float freq  = 1.0f / powf(rope_theta_, (float)(2 * j) / head_dim_);
            float angle = pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            cos_host_[(size_t)p * head_dim_ + j]        = c;
            cos_host_[(size_t)p * head_dim_ + j + half] = c;
            sin_host_[(size_t)p * head_dim_ + j]        = s;
            sin_host_[(size_t)p * head_dim_ + j + half] = s;
        }
    }
    upload_f32_as_f16(rope_cos_dev_, cos_host_.data(),
                       (size_t)MAX_SEQ * head_dim_);
    upload_f32_as_f16(rope_sin_dev_, sin_host_.data(),
                       (size_t)MAX_SEQ * head_dim_);
}

// ============================================================================
// Causal mask construction. FIAS's pseShift path requires a contiguous
// [1, n_heads, S_q, S_kv] tensor. Broadcast across heads via stride[1]=0
// (same convention as ggml-cann for non-ALiBi models). We build the mask per
// prefill call into `causal_mask_dev_` — cheap (a few kB typical).
//
// The build_causal_mask_ helper is a no-op now; the actual upload happens
// inside forward_prefill. Kept as a method for symmetry with CpCannEngine.
// ============================================================================

void TalkerCannEngine::build_causal_mask_() {
    // Device buffer is allocated in init_from_gguf; content is built per
    // prefill call in forward_prefill.
}

// ============================================================================
// init_from_gguf — load all weights + allocate per-layer buffers + KV cache +
// scratch. Same ordering as CpCannEngine::init.
// ============================================================================

bool TalkerCannEngine::init_from_gguf(const std::string &gguf_path,
                                       const TalkerConfig &cfg, int device) {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[talker_cann] symbol load failed; engine disabled\n");
        return false;
    }

    device_ = device;
    ACL_CHECK_RET(g_cann.aclrtSetDevice(device_));
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&stream_));

    // Cache dims from config.
    n_embd_     = cfg.hidden_size;
    n_heads_    = cfg.num_attention_heads;
    n_kv_       = cfg.num_key_value_heads;
    head_dim_   = cfg.head_dim;
    q_dim_      = n_heads_ * head_dim_;
    kv_dim_     = n_kv_    * head_dim_;
    inter_      = cfg.intermediate_size;
    n_layers_   = cfg.num_hidden_layers;
    eps_        = cfg.rms_norm_eps;
    rope_theta_ = cfg.rope_theta;
    rope_speed_factor_ = 1.0f;

    // Open GGUF via ggml's standard path.
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        fprintf(stderr, "[talker_cann] failed to load GGUF: %s\n",
                gguf_path.c_str());
        return false;
    }

    // ---- Per-layer weights ----
    layer_w_.resize(n_layers_);
    char name[128];
    for (int il = 0; il < n_layers_; ++il) {
        auto &lw = layer_w_[il];
#define TFMT(fmt) (snprintf(name, sizeof(name), fmt, il), name)
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.attn_q.weight"),
                               (size_t)q_dim_ * n_embd_, lw.q_proj_w))
            goto fail;
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.attn_k.weight"),
                               (size_t)kv_dim_ * n_embd_, lw.k_proj_w))
            goto fail;
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.attn_v.weight"),
                               (size_t)kv_dim_ * n_embd_, lw.v_proj_w))
            goto fail;
        // Output projection. GGUF stores this as [hidden, q_dim] which is the
        // [out, in] layout aclnnMm wants.
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.attn_output.weight"),
                               (size_t)n_embd_ * q_dim_, lw.o_proj_w))
            goto fail;
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.ffn_gate.weight"),
                               (size_t)inter_ * n_embd_, lw.gate_proj_w))
            goto fail;
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.ffn_up.weight"),
                               (size_t)inter_ * n_embd_, lw.up_proj_w))
            goto fail;
        if (!upload_tensor_f16(ggml_ctx,
                               TFMT("blk.%d.ffn_down.weight"),
                               (size_t)n_embd_ * inter_, lw.down_proj_w))
            goto fail;
        // Norms: F32 gammas (matches aclnnRmsNorm SupportInfo for F16 x +
        // F32 gamma — same path CpCannEngine uses).
        if (!upload_tensor_f32(ggml_ctx,
                               TFMT("blk.%d.attn_q_norm.weight"),
                               head_dim_, lw.q_norm_w)) goto fail;
        if (!upload_tensor_f32(ggml_ctx,
                               TFMT("blk.%d.attn_k_norm.weight"),
                               head_dim_, lw.k_norm_w)) goto fail;
        if (!upload_tensor_f32(ggml_ctx,
                               TFMT("blk.%d.attn_norm.weight"),
                               n_embd_, lw.input_ln_w)) goto fail;
        if (!upload_tensor_f32(ggml_ctx,
                               TFMT("blk.%d.ffn_norm.weight"),
                               n_embd_, lw.post_ln_w)) goto fail;
#undef TFMT
    }
    if (!upload_tensor_f32(ggml_ctx, "output_norm.weight",
                            n_embd_, final_norm_w_dev_)) goto fail;

    // GGUF no longer needed after we've pulled everything.
    gguf_free(gguf_ctx);
    ggml_free(ggml_ctx);
    gguf_ctx = nullptr;
    ggml_ctx = nullptr;

    // ---- Per-forward working buffers (single-token path: F16) ----
    {
        const size_t E = sizeof(uint16_t);
        alloc_dev_(&cur_dev_,       (size_t)n_embd_ * E);
        alloc_dev_(&residual_dev_,  (size_t)n_embd_ * E);
        alloc_dev_(&normed_dev_,    (size_t)n_embd_ * E);
        alloc_dev_(&q_dev_,         (size_t)q_dim_  * E);
        alloc_dev_(&k_dev_,         (size_t)kv_dim_ * E);
        alloc_dev_(&v_dev_,         (size_t)kv_dim_ * E);
        alloc_dev_(&attn_out_dev_,  (size_t)q_dim_  * E);
        alloc_dev_(&o_out_dev_,     (size_t)n_embd_ * E);
        alloc_dev_(&gate_dev_,      (size_t)inter_  * E);
        alloc_dev_(&up_dev_,        (size_t)inter_  * E);
        alloc_dev_(&ffn_out_dev_,   (size_t)n_embd_ * E);

        // Batched staging (prefill path)
        alloc_dev_(&cur_batch_dev_,      (size_t)MAX_PREFILL * n_embd_ * E);
        alloc_dev_(&residual_batch_dev_, (size_t)MAX_PREFILL * n_embd_ * E);
        alloc_dev_(&normed_batch_dev_,   (size_t)MAX_PREFILL * n_embd_ * E);
        alloc_dev_(&q_batch_dev_,        (size_t)MAX_PREFILL * q_dim_  * E);
        alloc_dev_(&k_batch_dev_,        (size_t)MAX_PREFILL * kv_dim_ * E);
        alloc_dev_(&v_batch_dev_,        (size_t)MAX_PREFILL * kv_dim_ * E);
        alloc_dev_(&attn_out_batch_dev_, (size_t)MAX_PREFILL * q_dim_  * E);
        alloc_dev_(&o_out_batch_dev_,    (size_t)MAX_PREFILL * n_embd_ * E);
        alloc_dev_(&gate_batch_dev_,     (size_t)MAX_PREFILL * inter_  * E);
        alloc_dev_(&up_batch_dev_,       (size_t)MAX_PREFILL * inter_  * E);
        alloc_dev_(&ffn_out_batch_dev_,  (size_t)MAX_PREFILL * n_embd_ * E);

        // Mask buffer sized for the worst case: seq_len = MAX_PREFILL,
        // seq_len_total = MAX_SEQ. That's 512 * 4096 * 2 = 4 MB. Content
        // is built + uploaded per prefill call.
        alloc_dev_(&causal_mask_dev_,
                    (size_t)MAX_PREFILL * MAX_SEQ * E);

        // rstd is F32 regardless of input dtype (aclnnRmsNorm requirement).
        // Sized for the largest case: prefill QK-norm over MAX_PREFILL rows
        // × n_heads heads (each row has its own per-head rstd).
        alloc_dev_(&rstd_dev_,
                    (size_t)MAX_PREFILL * n_heads_ * sizeof(float));

        alloc_dev_(&input_stage_f32_dev_,
                    (size_t)MAX_PREFILL * n_embd_ * sizeof(float));
        alloc_dev_(&output_stage_f32_dev_,
                    (size_t)n_embd_ * sizeof(float));

        // RoPE tables (F16, [MAX_SEQ, head_dim] with halves duplicated)
        alloc_dev_(&rope_cos_dev_, (size_t)MAX_SEQ * head_dim_ * E);
        alloc_dev_(&rope_sin_dev_, (size_t)MAX_SEQ * head_dim_ * E);
    }

    // Populate RoPE + causal mask.
    build_rope_tables_();
    build_causal_mask_();

    // ---- KV cache (F16) ----
    k_cache_dev_.assign(n_layers_, nullptr);
    v_cache_dev_.assign(n_layers_, nullptr);
    for (int il = 0; il < n_layers_; ++il) {
        alloc_dev_(&k_cache_dev_[il],
                    (size_t)MAX_SEQ * kv_dim_ * sizeof(uint16_t));
        alloc_dev_(&v_cache_dev_[il],
                    (size_t)MAX_SEQ * kv_dim_ * sizeof(uint16_t));
    }
    kv_cache_len_ = 0;

    // Scalar used by aclnnAdd (alpha=1.0 F16).
    one_scalar_f16_ = make_f16_scalar(1.0f);

    // Workspace seed (will grow on first big op).
    workspace_size_ = 4 * 1024 * 1024;
    ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                      ACL_MEM_MALLOC_HUGE_FIRST));

    ready_ = true;
    printf("[talker_cann] initialized from %s\n", gguf_path.c_str());
    printf("[talker_cann] dims: n_embd=%d q_dim=%d kv_dim=%d inter=%d "
           "heads=%d kv=%d layers=%d\n",
           n_embd_, q_dim_, kv_dim_, inter_, n_heads_, n_kv_, n_layers_);
    return true;

fail:
    if (gguf_ctx) gguf_free(gguf_ctx);
    if (ggml_ctx) ggml_free(ggml_ctx);
    fprintf(stderr, "[talker_cann] init_from_gguf failed\n");
    return false;
}

void TalkerCannEngine::reset_kv_cache() {
    kv_cache_len_ = 0;
}

void TalkerCannEngine::set_rope_speed_factor(float factor) {
    if (std::fabs(factor - rope_speed_factor_) < 1e-6f) return;
    rope_speed_factor_ = factor;
    if (ready_) build_rope_tables_();
}

// ============================================================================
// forward_decode — single-token path. Mirrors CpCannEngine::forward_one_token
// but with 28 layers, larger dims, and (crucially) F16 residual adds instead
// of CP's F32 residual accumulator — Talker follows ggml-cann's convention.
// ============================================================================

void TalkerCannEngine::forward_decode(const float *input_embed, int pos,
                                       float *hidden_out) {
    assert(ready_);
    assert(pos >= 0 && pos < MAX_SEQ);

    // Optional per-op sync probe — stays behind an env var so it's free in
    // production but helps triangulate which op is the first to misbehave
    // during bringup on new CANN toolkit versions.
    const bool dbg = getenv("TALKER_CANN_DEBUG") != nullptr;
    auto dbg_sync = [&](int il, const char *tag) {
        if (!dbg) return;
        aclError s = g_cann.aclrtSynchronizeStream(stream_);
        fprintf(stderr, "[talker_cann][layer %d] %s sync=%d\n", il, tag, s);
    };

    // 1. Upload F32 input embedding, cast to F16 into cur_dev_.
    //    Cast op needs a 2D+ tensor shape on Ascend 910B (1D Cast hits
    //    "tiling_funcs NULL" — the kernel isn't registered for pure 1D). Use
    //    a [1, n_embd] view, matching how CP does its analogous Cast.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        input_embed,          (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));
    {
        aclTensor *t_in_f32 = tensor_2d(input_stage_f32_dev_,
                                         1, n_embd_, ACL_FLOAT);
        aclTensor *t_cur_f16 = tensor_2d(cur_dev_, 1, n_embd_, ACL_FLOAT16);
        CANN_OP(Cast, t_in_f32, ACL_FLOAT16, t_cur_f16);
        g_cann.aclDestroyTensor(t_in_f32);
        g_cann.aclDestroyTensor(t_cur_f16);
    }
    dbg_sync(-1, "initial Cast");

    // RoPE cos/sin row views for this position.
    uint16_t *cos_pos = (uint16_t *)rope_cos_dev_ + (size_t)pos * head_dim_;
    uint16_t *sin_pos = (uint16_t *)rope_sin_dev_ + (size_t)pos * head_dim_;
    auto make_rope_4d = [&](void *buf) {
        int64_t shape[4]   = {1, 1, 1, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)head_dim_, (int64_t)head_dim_,
                               (int64_t)head_dim_, 1};
        return tensor_strided(buf, 4, shape, strides, ACL_FLOAT16);
    };
    aclTensor *t_cos = make_rope_4d(cos_pos);
    aclTensor *t_sin = make_rope_4d(sin_pos);

    const int seq_len_total = pos + 1;  // KV length *after* writing this token.

    for (int il = 0; il < n_layers_; ++il) {
        const auto &lw = layer_w_[il];

        // residual = cur (d2d F16)
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_dev_, (size_t)n_embd_ * sizeof(uint16_t),
            cur_dev_,      (size_t)n_embd_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- Input RmsNorm (F16 x, F32 gamma) ---
        {
            aclTensor *t_cur_row  = tensor_2d(cur_dev_,    1, n_embd_, ACL_FLOAT16);
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_gamma    = tensor_1d(lw.input_ln_w, n_embd_, ACL_FLOAT);
            aclTensor *t_rstd     = tensor_2d(rstd_dev_, 1, 1, ACL_FLOAT);
            CANN_OP(RmsNorm, t_cur_row, t_gamma, (double)eps_,
                    t_norm_row, t_rstd);
            g_cann.aclDestroyTensor(t_cur_row);
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_gamma);
            g_cann.aclDestroyTensor(t_rstd);
        }
        dbg_sync(il, "input RmsNorm");

        // --- Q/K/V projection: [dim, 1] = [dim, n_embd] @ [n_embd, 1] ---
        {
            aclTensor *t_w_q = tensor_2d(lw.q_proj_w, q_dim_, n_embd_, ACL_FLOAT16);
            aclTensor *t_w_k = tensor_2d(lw.k_proj_w, kv_dim_, n_embd_, ACL_FLOAT16);
            aclTensor *t_w_v = tensor_2d(lw.v_proj_w, kv_dim_, n_embd_, ACL_FLOAT16);
            aclTensor *t_norm_col = tensor_2d(normed_dev_, n_embd_, 1, ACL_FLOAT16);
            aclTensor *t_q_col = tensor_2d(q_dev_, q_dim_,  1, ACL_FLOAT16);
            aclTensor *t_k_col = tensor_2d(k_dev_, kv_dim_, 1, ACL_FLOAT16);
            aclTensor *t_v_col = tensor_2d(v_dev_, kv_dim_, 1, ACL_FLOAT16);
            CANN_OP(Mm, t_w_q, t_norm_col, t_q_col, (int8_t)0);
            dbg_sync(il, "q_proj");
            CANN_OP(Mm, t_w_k, t_norm_col, t_k_col, (int8_t)0);
            dbg_sync(il, "k_proj");
            CANN_OP(Mm, t_w_v, t_norm_col, t_v_col, (int8_t)0);
            dbg_sync(il, "v_proj");
            g_cann.aclDestroyTensor(t_w_q);
            g_cann.aclDestroyTensor(t_w_k);
            g_cann.aclDestroyTensor(t_w_v);
            g_cann.aclDestroyTensor(t_norm_col);
            g_cann.aclDestroyTensor(t_q_col);
            g_cann.aclDestroyTensor(t_k_col);
            g_cann.aclDestroyTensor(t_v_col);
        }

        // --- QK-norm (per-head RmsNorm, shared gamma [head_dim]) ---
        {
            aclTensor *t_q_heads = tensor_2d(q_dev_, n_heads_, head_dim_, ACL_FLOAT16);
            aclTensor *t_k_kv    = tensor_2d(k_dev_, n_kv_,    head_dim_, ACL_FLOAT16);
            aclTensor *t_qnorm   = tensor_1d(lw.q_norm_w, head_dim_, ACL_FLOAT);
            aclTensor *t_knorm   = tensor_1d(lw.k_norm_w, head_dim_, ACL_FLOAT);
            aclTensor *t_rstd_q  = tensor_2d(rstd_dev_, n_heads_, 1, ACL_FLOAT);
            aclTensor *t_rstd_k  = tensor_2d(rstd_dev_, n_kv_,    1, ACL_FLOAT);
            CANN_OP(RmsNorm, t_q_heads, t_qnorm, (double)eps_,
                    t_q_heads, t_rstd_q);
            dbg_sync(il, "q_norm");
            CANN_OP(RmsNorm, t_k_kv, t_knorm, (double)eps_,
                    t_k_kv,    t_rstd_k);
            dbg_sync(il, "k_norm");
            g_cann.aclDestroyTensor(t_q_heads);
            g_cann.aclDestroyTensor(t_k_kv);
            g_cann.aclDestroyTensor(t_qnorm);
            g_cann.aclDestroyTensor(t_knorm);
            g_cann.aclDestroyTensor(t_rstd_q);
            g_cann.aclDestroyTensor(t_rstd_k);
        }

        // --- RoPE on Q (writes into attn_out_dev_ — RoPE is non-in-place) ---
        {
            int64_t q_shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t q_strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                     (int64_t)head_dim_, 1};
            aclTensor *t_q_in  = tensor_strided(q_dev_, 4, q_shape, q_strides,
                                                 ACL_FLOAT16);
            aclTensor *t_q_out = tensor_strided(attn_out_dev_, 4, q_shape,
                                                 q_strides, ACL_FLOAT16);
            CANN_OP(RotaryPositionEmbedding,
                    t_q_in, t_cos, t_sin, (int64_t)0, t_q_out);
            g_cann.aclDestroyTensor(t_q_in);
            g_cann.aclDestroyTensor(t_q_out);
        }
        dbg_sync(il, "rope_q");

        // --- RoPE on K (writes directly into KV cache slot at `pos`) ---
        uint16_t *k_slot = (uint16_t *)k_cache_dev_[il] + (size_t)pos * kv_dim_;
        {
            int64_t k_shape[4]   = {1, 1, (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t k_strides[4] = {(int64_t)kv_dim_, (int64_t)kv_dim_,
                                     (int64_t)head_dim_, 1};
            aclTensor *t_k_in  = tensor_strided(k_dev_, 4, k_shape, k_strides,
                                                 ACL_FLOAT16);
            aclTensor *t_k_out = tensor_strided(k_slot, 4, k_shape, k_strides,
                                                 ACL_FLOAT16);
            CANN_OP(RotaryPositionEmbedding,
                    t_k_in, t_cos, t_sin, (int64_t)0, t_k_out);
            g_cann.aclDestroyTensor(t_k_in);
            g_cann.aclDestroyTensor(t_k_out);
        }
        dbg_sync(il, "rope_k");

        // --- V -> cache slot ---
        uint16_t *v_slot = (uint16_t *)v_cache_dev_[il] + (size_t)pos * kv_dim_;
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            v_slot, (size_t)kv_dim_ * sizeof(uint16_t),
            v_dev_, (size_t)kv_dim_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- FusedInferAttentionScoreV2 (BSND, decode S=1) ---
        aclTensor *t_q_bsnd, *t_k_bsnd, *t_v_bsnd, *t_attn_bsnd;
        {
            int64_t q_shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t q_strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                     (int64_t)head_dim_, 1};
            t_q_bsnd = tensor_strided(attn_out_dev_, 4, q_shape, q_strides,
                                       ACL_FLOAT16);
            int64_t kv_shape[4]   = {1, (int64_t)seq_len_total,
                                      (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t kv_strides[4] = {(int64_t)seq_len_total * kv_dim_,
                                      (int64_t)kv_dim_,
                                      (int64_t)head_dim_, 1};
            t_k_bsnd = tensor_strided(k_cache_dev_[il], 4, kv_shape, kv_strides,
                                       ACL_FLOAT16);
            t_v_bsnd = tensor_strided(v_cache_dev_[il], 4, kv_shape, kv_strides,
                                       ACL_FLOAT16);
            t_attn_bsnd = tensor_strided(q_dev_, 4, q_shape, q_strides,
                                          ACL_FLOAT16);
        }
        aclTensorList *t_k_list = g_cann.aclCreateTensorList(&t_k_bsnd, 1);
        aclTensorList *t_v_list = g_cann.aclCreateTensorList(&t_v_bsnd, 1);
        {
            uint64_t fa_ws = 0;
            aclOpExecutor *fa_exec = nullptr;
            char layout[5] = {'B','S','N','D',0};
            double scale = 1.0 / sqrt((double)head_dim_);
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
                t_q_bsnd, t_k_list, t_v_list,
                /*pseShift*/ nullptr, /*attenMask*/ nullptr,
                /*actSeqLen*/ nullptr, /*actSeqLenKv*/ nullptr,
                /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,
                /*deqScale2*/ nullptr, /*quantScale2*/ nullptr,
                /*quantOffset2*/ nullptr,
                /*antiquantScale*/ nullptr, /*antiquantOffset*/ nullptr,
                /*blockTable*/ nullptr,
                /*queryPaddingSize*/ nullptr, /*kvPaddingSize*/ nullptr,
                /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,
                /*valueAntiquantScale*/ nullptr,
                /*valueAntiquantOffset*/ nullptr,
                /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
                /*actualSharedPrefixLen*/ nullptr,
                /*numHeads*/ (int64_t)n_heads_,
                /*scaleValue*/ scale,
                /*preTokens*/ (int64_t)65535,
                /*nextTokens*/ (int64_t)65535,
                /*inputLayout*/ layout,
                /*numKeyValueHeads*/ (int64_t)n_kv_,
                /*sparseMode*/ (int64_t)0,
                /*innerPrecise*/ (int64_t)0,  // decode (S=1) — high precision
                /*blockSize*/ (int64_t)0,
                /*antiquantMode*/ (int64_t)0,
                /*softmaxLseFlag*/ false,
                /*keyAntiquantMode*/ (int64_t)0,
                /*valueAntiquantMode*/ (int64_t)0,
                /*attentionOut*/ t_attn_bsnd,
                /*softmaxLse*/ nullptr,
                &fa_ws, &fa_exec));
            ensure_workspace_(fa_ws);
            void *ws = fa_ws > 0 ? workspace_dev_ : nullptr;
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2(
                ws, fa_ws, fa_exec, stream_));
        }
        g_cann.aclDestroyTensorList(t_k_list);
        g_cann.aclDestroyTensorList(t_v_list);
        g_cann.aclDestroyTensor(t_q_bsnd);
        g_cann.aclDestroyTensor(t_attn_bsnd);
        // t_k_bsnd / t_v_bsnd owned by the destroyed lists.
        dbg_sync(il, "fias");

        // --- O projection [n_embd, 1] = [n_embd, q_dim] @ [q_dim, 1] ---
        {
            aclTensor *t_w_o   = tensor_2d(lw.o_proj_w, n_embd_, q_dim_, ACL_FLOAT16);
            aclTensor *t_q_col = tensor_2d(q_dev_, q_dim_, 1, ACL_FLOAT16);
            aclTensor *t_o_col = tensor_2d(o_out_dev_, n_embd_, 1, ACL_FLOAT16);
            CANN_OP(Mm, t_w_o, t_q_col, t_o_col, (int8_t)0);
            g_cann.aclDestroyTensor(t_w_o);
            g_cann.aclDestroyTensor(t_q_col);
            g_cann.aclDestroyTensor(t_o_col);
        }

        // cur = residual + o_out (F16)
        {
            aclTensor *t_res = tensor_1d(residual_dev_, n_embd_, ACL_FLOAT16);
            aclTensor *t_o   = tensor_1d(o_out_dev_,    n_embd_, ACL_FLOAT16);
            aclTensor *t_cur = tensor_1d(cur_dev_,      n_embd_, ACL_FLOAT16);
            CANN_OP(Add, t_res, t_o, one_scalar_f16_, t_cur);
            g_cann.aclDestroyTensor(t_res);
            g_cann.aclDestroyTensor(t_o);
            g_cann.aclDestroyTensor(t_cur);
        }

        // residual = cur (for FFN)
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_dev_, (size_t)n_embd_ * sizeof(uint16_t),
            cur_dev_,      (size_t)n_embd_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- Post-attention RmsNorm ---
        {
            aclTensor *t_cur_row  = tensor_2d(cur_dev_,    1, n_embd_, ACL_FLOAT16);
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_gamma    = tensor_1d(lw.post_ln_w, n_embd_, ACL_FLOAT);
            aclTensor *t_rstd     = tensor_2d(rstd_dev_, 1, 1, ACL_FLOAT);
            CANN_OP(RmsNorm, t_cur_row, t_gamma, (double)eps_,
                    t_norm_row, t_rstd);
            g_cann.aclDestroyTensor(t_cur_row);
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_gamma);
            g_cann.aclDestroyTensor(t_rstd);
        }

        // --- FFN: gate = silu(gate_proj @ normed) * (up_proj @ normed),
        //     ffn_out = down_proj @ gate ---
        {
            aclTensor *t_w_gate = tensor_2d(lw.gate_proj_w, inter_, n_embd_, ACL_FLOAT16);
            aclTensor *t_w_up   = tensor_2d(lw.up_proj_w,   inter_, n_embd_, ACL_FLOAT16);
            aclTensor *t_w_down = tensor_2d(lw.down_proj_w, n_embd_, inter_, ACL_FLOAT16);
            aclTensor *t_norm_col = tensor_2d(normed_dev_, n_embd_, 1, ACL_FLOAT16);
            aclTensor *t_gate_col = tensor_2d(gate_dev_, inter_, 1, ACL_FLOAT16);
            aclTensor *t_up_col   = tensor_2d(up_dev_,   inter_, 1, ACL_FLOAT16);
            aclTensor *t_gate_flat = tensor_1d(gate_dev_, inter_, ACL_FLOAT16);
            aclTensor *t_up_flat   = tensor_1d(up_dev_,   inter_, ACL_FLOAT16);
            aclTensor *t_ffn_col   = tensor_2d(ffn_out_dev_, n_embd_, 1, ACL_FLOAT16);
            CANN_OP(Mm, t_w_gate, t_norm_col, t_gate_col, (int8_t)0);
            CANN_OP(Mm, t_w_up,   t_norm_col, t_up_col,   (int8_t)0);
            CANN_OP(Silu, t_gate_flat, t_gate_flat);
            CANN_OP(InplaceMul, t_gate_flat, t_up_flat);
            CANN_OP(Mm, t_w_down, t_gate_col, t_ffn_col, (int8_t)0);
            g_cann.aclDestroyTensor(t_w_gate);
            g_cann.aclDestroyTensor(t_w_up);
            g_cann.aclDestroyTensor(t_w_down);
            g_cann.aclDestroyTensor(t_norm_col);
            g_cann.aclDestroyTensor(t_gate_col);
            g_cann.aclDestroyTensor(t_up_col);
            g_cann.aclDestroyTensor(t_gate_flat);
            g_cann.aclDestroyTensor(t_up_flat);
            g_cann.aclDestroyTensor(t_ffn_col);
        }

        // cur = residual + ffn_out (F16)
        {
            aclTensor *t_res = tensor_1d(residual_dev_, n_embd_, ACL_FLOAT16);
            aclTensor *t_ffn = tensor_1d(ffn_out_dev_,  n_embd_, ACL_FLOAT16);
            aclTensor *t_cur = tensor_1d(cur_dev_,      n_embd_, ACL_FLOAT16);
            CANN_OP(Add, t_res, t_ffn, one_scalar_f16_, t_cur);
            g_cann.aclDestroyTensor(t_res);
            g_cann.aclDestroyTensor(t_ffn);
            g_cann.aclDestroyTensor(t_cur);
        }
    }

    // Final RmsNorm
    {
        aclTensor *t_cur_row  = tensor_2d(cur_dev_,    1, n_embd_, ACL_FLOAT16);
        aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
        aclTensor *t_gamma    = tensor_1d(final_norm_w_dev_, n_embd_, ACL_FLOAT);
        aclTensor *t_rstd     = tensor_2d(rstd_dev_, 1, 1, ACL_FLOAT);
        CANN_OP(RmsNorm, t_cur_row, t_gamma, (double)eps_,
                t_norm_row, t_rstd);
        g_cann.aclDestroyTensor(t_cur_row);
        g_cann.aclDestroyTensor(t_norm_row);
        g_cann.aclDestroyTensor(t_gamma);
        g_cann.aclDestroyTensor(t_rstd);
    }

    // Cast F16 normed -> F32 staging, then D2H to caller.
    {
        aclTensor *t_out_f16 = tensor_1d(normed_dev_, n_embd_, ACL_FLOAT16);
        aclTensor *t_out_f32 = tensor_1d(output_stage_f32_dev_, n_embd_,
                                          ACL_FLOAT);
        CANN_OP(Cast, t_out_f16, ACL_FLOAT, t_out_f32);
        g_cann.aclDestroyTensor(t_out_f16);
        g_cann.aclDestroyTensor(t_out_f32);
    }
    ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        hidden_out, (size_t)n_embd_ * sizeof(float),
        output_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST));

    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);

    // Advance cache length (caller doesn't track per-token position, but we
    // expose `reset_kv_cache` so stateful callers can reset between utterances).
    if (pos + 1 > kv_cache_len_) kv_cache_len_ = pos + 1;
}

// ============================================================================
// forward_prefill — batched path, supports seq_len > 1.
//
// Processes `seq_len` tokens in a single batch. Writes KV entries for cache
// positions [start_pos, start_pos + seq_len). Returns only the hidden state
// of the last position (index seq_len - 1 in the batch).
//
// FIAS with innerPrecise=2 and an attention mask is the standard ggml-cann
// prefill recipe. The mask is [seq_len, seq_len_total]; we build a padded
// [MAX_PREFILL, MAX_PREFILL] once at init and stride-view a sub-rectangle.
//
// If seq_len > MAX_PREFILL we fall back to a chunked prefill loop (each
// chunk ≤ MAX_PREFILL) so we don't have to pre-allocate an absurd staging
// buffer up front.
// ============================================================================

void TalkerCannEngine::forward_prefill(const float *input_embeds, int seq_len,
                                        int start_pos, float *last_hidden_out) {
    assert(ready_);
    assert(seq_len > 0);
    assert(start_pos >= 0 && start_pos + seq_len <= MAX_SEQ);

    // Default: iterate forward_decode. The batched-FIAS prefill below gives
    // wrong hidden state (TTS transcribes as "Oh." via qwen3-asr) while
    // iterative single-token decode matches llama.cpp exactly. Keep the
    // batched path compiled in but off-by-default; set TALKER_PREFILL_BATCHED=1
    // to force it (for bug-hunting the batched path).
    if (getenv("TALKER_PREFILL_BATCHED") == nullptr) {
        std::vector<float> scratch(n_embd_);
        for (int i = 0; i < seq_len; i++) {
            const float *tok = input_embeds + (size_t)i * n_embd_;
            bool is_last = (i == seq_len - 1);
            forward_decode(tok, start_pos + i,
                           is_last && last_hidden_out ? last_hidden_out
                                                       : scratch.data());
        }
        return;
    }

    // Chunked prefill: carve the request into pieces of at most MAX_PREFILL.
    // Each chunk writes its own KV cache slots at the correct absolute
    // position; only the very last chunk's last-row hidden is returned.
    if (seq_len > MAX_PREFILL) {
        int processed = 0;
        while (processed < seq_len) {
            int chunk = std::min(MAX_PREFILL, seq_len - processed);
            const float *chunk_in = input_embeds + (size_t)processed * n_embd_;
            bool is_last_chunk = (processed + chunk == seq_len);
            forward_prefill(chunk_in, chunk, start_pos + processed,
                             is_last_chunk ? last_hidden_out : nullptr);
            processed += chunk;
        }
        return;
    }

    // Single-shot prefill path.
    // 1. Upload the batch of F32 embeddings, cast to F16 into cur_batch_dev_.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, (size_t)seq_len * n_embd_ * sizeof(float),
        input_embeds,         (size_t)seq_len * n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));
    {
        aclTensor *t_in_f32 = tensor_2d(input_stage_f32_dev_,
                                          seq_len, n_embd_, ACL_FLOAT);
        aclTensor *t_cur_f16 = tensor_2d(cur_batch_dev_, seq_len, n_embd_,
                                           ACL_FLOAT16);
        CANN_OP(Cast, t_in_f32, ACL_FLOAT16, t_cur_f16);
        g_cann.aclDestroyTensor(t_in_f32);
        g_cann.aclDestroyTensor(t_cur_f16);
    }

    const int seq_len_total = start_pos + seq_len;

    // RoPE cos/sin window for positions [start_pos, start_pos + seq_len).
    // Shape for FIAS/RoPE is [1, seq_len, 1, head_dim].
    uint16_t *cos_win = (uint16_t *)rope_cos_dev_ +
                         (size_t)start_pos * head_dim_;
    uint16_t *sin_win = (uint16_t *)rope_sin_dev_ +
                         (size_t)start_pos * head_dim_;
    auto make_rope_batch = [&](void *buf) {
        int64_t shape[4]   = {1, (int64_t)seq_len, 1, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)head_dim_ * seq_len,
                               (int64_t)head_dim_,
                               (int64_t)head_dim_, 1};
        return tensor_strided(buf, 4, shape, strides, ACL_FLOAT16);
    };
    aclTensor *t_cos = make_rope_batch(cos_win);
    aclTensor *t_sin = make_rope_batch(sin_win);

    // Causality is enforced inside FIAS via `nextTokens=0` (see the
    // GetWorkspaceSize call below). This is simpler than a user-supplied
    // mask and, based on CANN 8.3's kernel tiling tables, more reliable
    // than the pseShift / attenMask paths for the (GQA 16/8, small S_q)
    // shape combinations the Talker hits during prefill.
    (void)causal_mask_dev_;  // retained for future pseShift paths; unused now.

    for (int il = 0; il < n_layers_; ++il) {
        const auto &lw = layer_w_[il];

        // residual = cur (F16 d2d, seq_len * n_embd)
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_batch_dev_,
            (size_t)seq_len * n_embd_ * sizeof(uint16_t),
            cur_batch_dev_,
            (size_t)seq_len * n_embd_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- Input RmsNorm over all `seq_len` rows ---
        {
            aclTensor *t_cur   = tensor_2d(cur_batch_dev_,   seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_norm  = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_gamma = tensor_1d(lw.input_ln_w, n_embd_, ACL_FLOAT);
            aclTensor *t_rstd  = tensor_2d(rstd_dev_, seq_len, 1, ACL_FLOAT);
            CANN_OP(RmsNorm, t_cur, t_gamma, (double)eps_, t_norm, t_rstd);
            g_cann.aclDestroyTensor(t_cur);
            g_cann.aclDestroyTensor(t_norm);
            g_cann.aclDestroyTensor(t_gamma);
            g_cann.aclDestroyTensor(t_rstd);
        }

        // --- Q/K/V projections (batched) ---
        // Q [seq_len, q_dim]  = normed [seq_len, n_embd] @ q_proj^T [n_embd, q_dim]
        //                     = (q_proj [q_dim, n_embd] @ normed^T [n_embd, seq_len])^T
        // aclnnMm computes A @ B with A:[M,K], B:[K,N] -> [M,N]. We have
        // normed:[seq_len, n_embd] and weight:[q_dim, n_embd]. Use
        // B = weight^T viewed as [n_embd, q_dim] via strides = (1, n_embd).
        auto weight_T_tensor = [&](void *buf, int64_t rows, int64_t cols) {
            // original shape [rows, cols] row-major. Transposed view [cols, rows]
            // with strides (1, cols).
            int64_t shape[2]   = {cols, rows};
            int64_t strides[2] = {1, cols};
            return tensor_strided(buf, 2, shape, strides, ACL_FLOAT16);
        };
        {
            aclTensor *t_normed = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_wq_T = weight_T_tensor(lw.q_proj_w, q_dim_,  n_embd_);
            aclTensor *t_wk_T = weight_T_tensor(lw.k_proj_w, kv_dim_, n_embd_);
            aclTensor *t_wv_T = weight_T_tensor(lw.v_proj_w, kv_dim_, n_embd_);
            aclTensor *t_q = tensor_2d(q_batch_dev_, seq_len, q_dim_,  ACL_FLOAT16);
            aclTensor *t_k = tensor_2d(k_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            aclTensor *t_v = tensor_2d(v_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            CANN_OP(Mm, t_normed, t_wq_T, t_q, (int8_t)0);
            CANN_OP(Mm, t_normed, t_wk_T, t_k, (int8_t)0);
            CANN_OP(Mm, t_normed, t_wv_T, t_v, (int8_t)0);
            g_cann.aclDestroyTensor(t_normed);
            g_cann.aclDestroyTensor(t_wq_T);
            g_cann.aclDestroyTensor(t_wk_T);
            g_cann.aclDestroyTensor(t_wv_T);
            g_cann.aclDestroyTensor(t_q);
            g_cann.aclDestroyTensor(t_k);
            g_cann.aclDestroyTensor(t_v);
        }

        // --- QK-norm per head for all rows. View Q as [seq_len*n_heads, head_dim]. ---
        {
            aclTensor *t_q_heads = tensor_2d(q_batch_dev_,
                                               seq_len * n_heads_, head_dim_,
                                               ACL_FLOAT16);
            aclTensor *t_k_kv    = tensor_2d(k_batch_dev_,
                                               seq_len * n_kv_,    head_dim_,
                                               ACL_FLOAT16);
            aclTensor *t_qnorm   = tensor_1d(lw.q_norm_w, head_dim_, ACL_FLOAT);
            aclTensor *t_knorm   = tensor_1d(lw.k_norm_w, head_dim_, ACL_FLOAT);
            aclTensor *t_rstd_q  = tensor_2d(rstd_dev_, seq_len * n_heads_, 1,
                                               ACL_FLOAT);
            aclTensor *t_rstd_k  = tensor_2d(rstd_dev_, seq_len * n_kv_,    1,
                                               ACL_FLOAT);
            CANN_OP(RmsNorm, t_q_heads, t_qnorm, (double)eps_,
                    t_q_heads, t_rstd_q);
            CANN_OP(RmsNorm, t_k_kv,    t_knorm, (double)eps_,
                    t_k_kv,    t_rstd_k);
            g_cann.aclDestroyTensor(t_q_heads);
            g_cann.aclDestroyTensor(t_k_kv);
            g_cann.aclDestroyTensor(t_qnorm);
            g_cann.aclDestroyTensor(t_knorm);
            g_cann.aclDestroyTensor(t_rstd_q);
            g_cann.aclDestroyTensor(t_rstd_k);
        }

        // --- RoPE on Q (batched) -> attn_out_batch_dev_ ---
        {
            int64_t q_shape[4]   = {1, (int64_t)seq_len, (int64_t)n_heads_,
                                     (int64_t)head_dim_};
            int64_t q_strides[4] = {(int64_t)seq_len * q_dim_,
                                     (int64_t)q_dim_,
                                     (int64_t)head_dim_, 1};
            aclTensor *t_q_in  = tensor_strided(q_batch_dev_, 4, q_shape,
                                                  q_strides, ACL_FLOAT16);
            aclTensor *t_q_out = tensor_strided(attn_out_batch_dev_, 4, q_shape,
                                                  q_strides, ACL_FLOAT16);
            CANN_OP(RotaryPositionEmbedding,
                    t_q_in, t_cos, t_sin, (int64_t)0, t_q_out);
            g_cann.aclDestroyTensor(t_q_in);
            g_cann.aclDestroyTensor(t_q_out);
        }

        // --- RoPE on K (batched), write into KV cache slice starting at start_pos ---
        uint16_t *k_slot =
            (uint16_t *)k_cache_dev_[il] + (size_t)start_pos * kv_dim_;
        {
            int64_t k_shape[4]   = {1, (int64_t)seq_len, (int64_t)n_kv_,
                                     (int64_t)head_dim_};
            int64_t k_strides[4] = {(int64_t)seq_len * kv_dim_,
                                     (int64_t)kv_dim_,
                                     (int64_t)head_dim_, 1};
            aclTensor *t_k_in  = tensor_strided(k_batch_dev_, 4, k_shape,
                                                  k_strides, ACL_FLOAT16);
            aclTensor *t_k_out = tensor_strided(k_slot, 4, k_shape, k_strides,
                                                  ACL_FLOAT16);
            CANN_OP(RotaryPositionEmbedding,
                    t_k_in, t_cos, t_sin, (int64_t)0, t_k_out);
            g_cann.aclDestroyTensor(t_k_in);
            g_cann.aclDestroyTensor(t_k_out);
        }

        // --- V -> cache slice ---
        uint16_t *v_slot =
            (uint16_t *)v_cache_dev_[il] + (size_t)start_pos * kv_dim_;
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            v_slot, (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
            v_batch_dev_,
            (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- FusedInferAttentionScoreV2 (BSND, prefill S>1, innerPrecise=2,
        //     causal mask provided) ---
        aclTensor *t_q_bsnd, *t_k_bsnd, *t_v_bsnd, *t_attn_bsnd;
        {
            int64_t q_shape[4]   = {1, (int64_t)seq_len, (int64_t)n_heads_,
                                     (int64_t)head_dim_};
            int64_t q_strides[4] = {(int64_t)seq_len * q_dim_,
                                     (int64_t)q_dim_,
                                     (int64_t)head_dim_, 1};
            t_q_bsnd = tensor_strided(attn_out_batch_dev_, 4, q_shape, q_strides,
                                       ACL_FLOAT16);
            int64_t kv_shape[4]   = {1, (int64_t)seq_len_total,
                                      (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t kv_strides[4] = {(int64_t)seq_len_total * kv_dim_,
                                      (int64_t)kv_dim_,
                                      (int64_t)head_dim_, 1};
            t_k_bsnd = tensor_strided(k_cache_dev_[il], 4, kv_shape, kv_strides,
                                       ACL_FLOAT16);
            t_v_bsnd = tensor_strided(v_cache_dev_[il], 4, kv_shape, kv_strides,
                                       ACL_FLOAT16);
            t_attn_bsnd = tensor_strided(q_batch_dev_, 4, q_shape, q_strides,
                                          ACL_FLOAT16);
        }
        aclTensorList *t_k_list = g_cann.aclCreateTensorList(&t_k_bsnd, 1);
        aclTensorList *t_v_list = g_cann.aclCreateTensorList(&t_v_bsnd, 1);
        {
            uint64_t fa_ws = 0;
            aclOpExecutor *fa_exec = nullptr;
            char layout[5] = {'B','S','N','D',0};
            double scale = 1.0 / sqrt((double)head_dim_);
            // FIAS built-in causal masking: sparseMode=1 (top-left causal)
            // means the op applies a lower-triangular mask internally. No
            // user mask needed. pseShift / attenMask both nullptr. preTokens
            // = 65535 keeps full K history visible (limited only by the
            // causal rule). Matches what ggml-cann uses for standard
            // autoregressive LLM prefill.
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
                t_q_bsnd, t_k_list, t_v_list,
                /*pseShift*/ nullptr,
                /*attenMask*/ nullptr,
                /*actSeqLen*/ nullptr, /*actSeqLenKv*/ nullptr,
                /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,
                /*deqScale2*/ nullptr, /*quantScale2*/ nullptr,
                /*quantOffset2*/ nullptr,
                /*antiquantScale*/ nullptr, /*antiquantOffset*/ nullptr,
                /*blockTable*/ nullptr,
                /*queryPaddingSize*/ nullptr, /*kvPaddingSize*/ nullptr,
                /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,
                /*valueAntiquantScale*/ nullptr,
                /*valueAntiquantOffset*/ nullptr,
                /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
                /*actualSharedPrefixLen*/ nullptr,
                /*numHeads*/ (int64_t)n_heads_,
                /*scaleValue*/ scale,
                /*preTokens*/ (int64_t)65535,
                /*nextTokens*/ (int64_t)0,  // causal: no future tokens visible
                /*inputLayout*/ layout,
                /*numKeyValueHeads*/ (int64_t)n_kv_,
                /*sparseMode*/ (int64_t)0,  // causality enforced via nextTokens=0
                /*innerPrecise*/ (int64_t)2,  // S>1 prefill precision mode
                /*blockSize*/ (int64_t)0,
                /*antiquantMode*/ (int64_t)0,
                /*softmaxLseFlag*/ false,
                /*keyAntiquantMode*/ (int64_t)0,
                /*valueAntiquantMode*/ (int64_t)0,
                /*attentionOut*/ t_attn_bsnd,
                /*softmaxLse*/ nullptr,
                &fa_ws, &fa_exec));
            ensure_workspace_(fa_ws);
            void *ws = fa_ws > 0 ? workspace_dev_ : nullptr;
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2(
                ws, fa_ws, fa_exec, stream_));
        }
        g_cann.aclDestroyTensorList(t_k_list);
        g_cann.aclDestroyTensorList(t_v_list);
        g_cann.aclDestroyTensor(t_q_bsnd);
        g_cann.aclDestroyTensor(t_attn_bsnd);

        // --- O projection (batched): o_out [seq_len, n_embd] = attn [seq_len, q_dim]
        //     @ o_proj^T [q_dim, n_embd] ---
        {
            aclTensor *t_attn = tensor_2d(q_batch_dev_, seq_len, q_dim_,
                                            ACL_FLOAT16);
            aclTensor *t_wo_T = weight_T_tensor(lw.o_proj_w, n_embd_, q_dim_);
            aclTensor *t_o    = tensor_2d(o_out_batch_dev_, seq_len, n_embd_,
                                            ACL_FLOAT16);
            CANN_OP(Mm, t_attn, t_wo_T, t_o, (int8_t)0);
            g_cann.aclDestroyTensor(t_attn);
            g_cann.aclDestroyTensor(t_wo_T);
            g_cann.aclDestroyTensor(t_o);
        }

        // cur = residual + o_out
        {
            aclTensor *t_res = tensor_1d(residual_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            aclTensor *t_o   = tensor_1d(o_out_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            aclTensor *t_cur = tensor_1d(cur_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            CANN_OP(Add, t_res, t_o, one_scalar_f16_, t_cur);
            g_cann.aclDestroyTensor(t_res);
            g_cann.aclDestroyTensor(t_o);
            g_cann.aclDestroyTensor(t_cur);
        }

        // residual = cur
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_batch_dev_,
            (size_t)seq_len * n_embd_ * sizeof(uint16_t),
            cur_batch_dev_,
            (size_t)seq_len * n_embd_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- Post-attn RmsNorm ---
        {
            aclTensor *t_cur   = tensor_2d(cur_batch_dev_,   seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_norm  = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_gamma = tensor_1d(lw.post_ln_w, n_embd_, ACL_FLOAT);
            aclTensor *t_rstd  = tensor_2d(rstd_dev_, seq_len, 1, ACL_FLOAT);
            CANN_OP(RmsNorm, t_cur, t_gamma, (double)eps_, t_norm, t_rstd);
            g_cann.aclDestroyTensor(t_cur);
            g_cann.aclDestroyTensor(t_norm);
            g_cann.aclDestroyTensor(t_gamma);
            g_cann.aclDestroyTensor(t_rstd);
        }

        // --- FFN (batched) ---
        {
            aclTensor *t_normed  = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_wg_T    = weight_T_tensor(lw.gate_proj_w, inter_, n_embd_);
            aclTensor *t_wu_T    = weight_T_tensor(lw.up_proj_w,   inter_, n_embd_);
            aclTensor *t_wd_T    = weight_T_tensor(lw.down_proj_w, n_embd_, inter_);
            aclTensor *t_gate    = tensor_2d(gate_batch_dev_, seq_len, inter_,
                                               ACL_FLOAT16);
            aclTensor *t_up      = tensor_2d(up_batch_dev_,   seq_len, inter_,
                                               ACL_FLOAT16);
            aclTensor *t_gate_fl = tensor_1d(gate_batch_dev_,
                                               (size_t)seq_len * inter_,
                                               ACL_FLOAT16);
            aclTensor *t_up_fl   = tensor_1d(up_batch_dev_,
                                               (size_t)seq_len * inter_,
                                               ACL_FLOAT16);
            aclTensor *t_ffn     = tensor_2d(ffn_out_batch_dev_, seq_len, n_embd_,
                                               ACL_FLOAT16);
            CANN_OP(Mm, t_normed, t_wg_T, t_gate, (int8_t)0);
            CANN_OP(Mm, t_normed, t_wu_T, t_up,   (int8_t)0);
            CANN_OP(Silu, t_gate_fl, t_gate_fl);
            CANN_OP(InplaceMul, t_gate_fl, t_up_fl);
            CANN_OP(Mm, t_gate, t_wd_T, t_ffn, (int8_t)0);
            g_cann.aclDestroyTensor(t_normed);
            g_cann.aclDestroyTensor(t_wg_T);
            g_cann.aclDestroyTensor(t_wu_T);
            g_cann.aclDestroyTensor(t_wd_T);
            g_cann.aclDestroyTensor(t_gate);
            g_cann.aclDestroyTensor(t_up);
            g_cann.aclDestroyTensor(t_gate_fl);
            g_cann.aclDestroyTensor(t_up_fl);
            g_cann.aclDestroyTensor(t_ffn);
        }

        // cur = residual + ffn_out
        {
            aclTensor *t_res = tensor_1d(residual_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            aclTensor *t_ffn = tensor_1d(ffn_out_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            aclTensor *t_cur = tensor_1d(cur_batch_dev_,
                                          (size_t)seq_len * n_embd_, ACL_FLOAT16);
            CANN_OP(Add, t_res, t_ffn, one_scalar_f16_, t_cur);
            g_cann.aclDestroyTensor(t_res);
            g_cann.aclDestroyTensor(t_ffn);
            g_cann.aclDestroyTensor(t_cur);
        }
    }

    // --- Final RmsNorm (batched) ---
    {
        aclTensor *t_cur   = tensor_2d(cur_batch_dev_,   seq_len, n_embd_, ACL_FLOAT16);
        aclTensor *t_norm  = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
        aclTensor *t_gamma = tensor_1d(final_norm_w_dev_, n_embd_, ACL_FLOAT);
        aclTensor *t_rstd  = tensor_2d(rstd_dev_, seq_len, 1, ACL_FLOAT);
        CANN_OP(RmsNorm, t_cur, t_gamma, (double)eps_, t_norm, t_rstd);
        g_cann.aclDestroyTensor(t_cur);
        g_cann.aclDestroyTensor(t_norm);
        g_cann.aclDestroyTensor(t_gamma);
        g_cann.aclDestroyTensor(t_rstd);
    }

    // Cast ONLY the last row's normed output to F32 and return to host (if
    // the caller requested it). The internal batch result is discarded —
    // TalkerLLM only needs the last position for next-token sampling.
    if (last_hidden_out) {
        uint16_t *last_row = (uint16_t *)normed_batch_dev_ +
                               (size_t)(seq_len - 1) * n_embd_;
        aclTensor *t_last_f16 = tensor_1d(last_row, n_embd_, ACL_FLOAT16);
        aclTensor *t_last_f32 = tensor_1d(output_stage_f32_dev_, n_embd_,
                                            ACL_FLOAT);
        CANN_OP(Cast, t_last_f16, ACL_FLOAT, t_last_f32);
        g_cann.aclDestroyTensor(t_last_f16);
        g_cann.aclDestroyTensor(t_last_f32);
        ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            last_hidden_out, (size_t)n_embd_ * sizeof(float),
            output_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_HOST));
    } else {
        // No download — but we still need to make sure the KV writes are
        // visible before the next chunk.
        ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    }

    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);

    int end_pos = start_pos + seq_len;
    if (end_pos > kv_cache_len_) kv_cache_len_ = end_pos;
}
