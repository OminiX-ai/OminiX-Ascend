// ============================================================================
// CP CANN Engine: Direct ACL-based Code Predictor on Ascend NPU
//
// Design principles:
// 1. ALL weights and intermediates pre-allocated on NPU at init().
// 2. Forward pass: only aclnn kernel launches, zero malloc.
// 3. Host<->Device copies: only input (1x 2048 floats) and output (1x 1024).
// 4. RoPE + QK-norm done on host (tiny vectors, not worth kernel launch).
// 5. KV cache lives on NPU, updated via small aclrtMemcpy per step.
//
// Target: <1ms per CP decode step (vs 3ms with llama.cpp).
// ============================================================================

#include "cp_cann_engine.h"
#include "talker.h"   // CodePredictorConfig

#include <acl/acl.h>
#include <aclnn/aclnn_base.h>
#include <aclnnop/aclnn_mm.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_silu.h>
#include <aclnnop/aclnn_rms_norm.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <cassert>

// ============================================================================
// ACL error check macro (simple for prototype)
// ============================================================================
#define ACL_CHECK_RET(stmt) do {                                      \
    aclError _ret = (stmt);                                           \
    if (_ret != 0) {                                                  \
        fprintf(stderr, "[cp_cann] ACL error %d at %s:%d: %s\n",     \
                _ret, __FILE__, __LINE__, aclGetRecentErrMsg());      \
    }                                                                 \
} while(0)

// ============================================================================
// aclnn two-phase op execution macro (standalone, no ggml context needed)
// ============================================================================
#define CANN_OP(stream, ws_dev, ws_size, OP_NAME, ...) do {           \
    uint64_t _ws_needed = 0;                                          \
    aclOpExecutor *_exec = nullptr;                                   \
    ACL_CHECK_RET(aclnn##OP_NAME##GetWorkspaceSize(                   \
        __VA_ARGS__, &_ws_needed, &_exec));                           \
    void *_ws = nullptr;                                              \
    if (_ws_needed > 0) {                                             \
        if (_ws_needed > (ws_size)) {                                 \
            if ((ws_dev)) aclrtFree((ws_dev));                        \
            ACL_CHECK_RET(aclrtMalloc(&(ws_dev), _ws_needed,         \
                          ACL_MEM_MALLOC_HUGE_FIRST));                \
            (ws_size) = _ws_needed;                                   \
        }                                                             \
        _ws = (ws_dev);                                               \
    }                                                                 \
    ACL_CHECK_RET(aclnn##OP_NAME(_ws, _ws_needed, _exec, (stream))); \
} while(0)

// ============================================================================
// Helper: create a 1D or 2D ACL tensor descriptor over a device buffer
// ============================================================================
static aclTensor *make_tensor_1d(void *dev_buf, int64_t n) {
    int64_t shape[1] = {n};
    int64_t strides[1] = {1};
    int64_t storage_len = n;
    return aclCreateTensor(shape, 1, ACL_FLOAT, strides, 0,
                           ACL_FORMAT_ND, &storage_len, 1, dev_buf);
}

// 2D tensor: shape [rows, cols], row-major
static aclTensor *make_tensor_2d(void *dev_buf, int64_t rows, int64_t cols) {
    // ACL uses NCHW-like ordering: shape reversed from row-major
    // For a [rows, cols] row-major matrix:
    //   shape = {rows, cols}, strides = {cols, 1}
    int64_t shape[2] = {rows, cols};
    int64_t strides[2] = {cols, 1};
    int64_t storage_len = rows * cols;
    return aclCreateTensor(shape, 2, ACL_FLOAT, strides, 0,
                           ACL_FORMAT_ND, &storage_len, 1, dev_buf);
}

// ============================================================================
// Lifecycle
// ============================================================================

CpCannEngine::~CpCannEngine() {
    if (!ready_) return;

    auto free_dev = [](void *&p) { if (p) { aclrtFree(p); p = nullptr; } };

    free_dev(proj_w_dev_);
    free_dev(proj_b_dev_);
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
    free_dev(scores_dev_);
    free_dev(attn_weights_dev_);
    free_dev(rope_cos_dev_);
    free_dev(rope_sin_dev_);
    free_dev(rstd_dev_);

    for (auto &p : k_cache_dev_) free_dev(p);
    for (auto &p : v_cache_dev_) free_dev(p);

    free_dev(workspace_dev_);

    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
}

// ============================================================================
// Device memory helpers
// ============================================================================

void CpCannEngine::alloc_dev(void **ptr, size_t bytes) {
    ACL_CHECK_RET(aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void CpCannEngine::upload(void *dev, const float *host, size_t n_floats) {
    ACL_CHECK_RET(aclrtMemcpy(dev, n_floats * sizeof(float),
                               host, n_floats * sizeof(float),
                               ACL_MEMCPY_HOST_TO_DEVICE));
}

void CpCannEngine::download(float *host, const void *dev, size_t n_floats) {
    ACL_CHECK_RET(aclrtMemcpy(host, n_floats * sizeof(float),
                               dev, n_floats * sizeof(float),
                               ACL_MEMCPY_DEVICE_TO_HOST));
}

void CpCannEngine::ensure_workspace(size_t needed) {
    if (needed <= workspace_size_) return;
    if (workspace_dev_) aclrtFree(workspace_dev_);
    ACL_CHECK_RET(aclrtMalloc(&workspace_dev_, needed,
                               ACL_MEM_MALLOC_HUGE_FIRST));
    workspace_size_ = needed;
}

// ============================================================================
// Init: upload weights + allocate buffers
// ============================================================================

bool CpCannEngine::init(const CpWeightsF32 &w, const CodePredictorConfig &cfg,
                        int device) {
    device_ = device;
    ACL_CHECK_RET(aclrtSetDevice(device_));
    ACL_CHECK_RET(aclrtCreateStream(&stream_));

    // Cache dimensions
    talker_hidden_ = cfg.talker_hidden_size;  // 2048
    cp_hidden_ = cfg.hidden_size;             // 1024
    n_heads_ = cfg.num_attention_heads;       // 16
    n_kv_ = cfg.num_key_value_heads;          // 8
    head_dim_ = cfg.head_dim;                 // 128
    q_dim_ = n_heads_ * head_dim_;            // 2048
    kv_dim_ = n_kv_ * head_dim_;             // 1024
    inter_ = cfg.intermediate_size;           // 3072
    n_layers_ = cfg.num_hidden_layers;        // 5
    eps_ = cfg.rms_norm_eps;
    rope_theta_ = cfg.rope_theta;

    // ---- Upload projection weights ----
    alloc_dev(&proj_w_dev_, cp_hidden_ * talker_hidden_ * sizeof(float));
    upload(proj_w_dev_, w.proj_w.data(), cp_hidden_ * talker_hidden_);
    alloc_dev(&proj_b_dev_, cp_hidden_ * sizeof(float));
    upload(proj_b_dev_, w.proj_b.data(), cp_hidden_);

    // ---- Upload per-layer weights ----
    layer_w_.resize(n_layers_);
    for (int il = 0; il < n_layers_; il++) {
        auto &src = w.layers[il];
        auto &dst = layer_w_[il];

        auto upload_mat = [&](void *&dev, const std::vector<float> &host,
                              int rows, int cols) {
            alloc_dev(&dev, (size_t)rows * cols * sizeof(float));
            upload(dev, host.data(), (size_t)rows * cols);
        };

        upload_mat(dst.q_proj_w, src.q_proj_w, q_dim_, cp_hidden_);
        upload_mat(dst.k_proj_w, src.k_proj_w, kv_dim_, cp_hidden_);
        upload_mat(dst.v_proj_w, src.v_proj_w, kv_dim_, cp_hidden_);
        upload_mat(dst.o_proj_w, src.o_proj_w, cp_hidden_, q_dim_);
        upload_mat(dst.gate_proj_w, src.gate_proj_w, inter_, cp_hidden_);
        upload_mat(dst.up_proj_w, src.up_proj_w, inter_, cp_hidden_);
        upload_mat(dst.down_proj_w, src.down_proj_w, cp_hidden_, inter_);

        alloc_dev(&dst.q_norm_w, head_dim_ * sizeof(float));
        upload(dst.q_norm_w, src.q_norm_w.data(), head_dim_);
        alloc_dev(&dst.k_norm_w, head_dim_ * sizeof(float));
        upload(dst.k_norm_w, src.k_norm_w.data(), head_dim_);
        alloc_dev(&dst.input_ln_w, cp_hidden_ * sizeof(float));
        upload(dst.input_ln_w, src.input_ln_w.data(), cp_hidden_);
        alloc_dev(&dst.post_ln_w, cp_hidden_ * sizeof(float));
        upload(dst.post_ln_w, src.post_ln_w.data(), cp_hidden_);
    }

    alloc_dev(&final_norm_w_dev_, cp_hidden_ * sizeof(float));
    upload(final_norm_w_dev_, w.norm_w.data(), cp_hidden_);

    // ---- Cache QK-norm weights on host ----
    qk_norm_host_.resize(n_layers_);
    for (int il = 0; il < n_layers_; il++) {
        qk_norm_host_[il].q_norm_w = w.layers[il].q_norm_w;
        qk_norm_host_[il].k_norm_w = w.layers[il].k_norm_w;
    }

    // ---- Allocate intermediate buffers ----
    alloc_dev(&cur_dev_, cp_hidden_ * sizeof(float));
    alloc_dev(&residual_dev_, cp_hidden_ * sizeof(float));
    alloc_dev(&normed_dev_, cp_hidden_ * sizeof(float));
    alloc_dev(&q_dev_, q_dim_ * sizeof(float));
    alloc_dev(&k_dev_, kv_dim_ * sizeof(float));
    alloc_dev(&v_dev_, kv_dim_ * sizeof(float));
    alloc_dev(&attn_out_dev_, q_dim_ * sizeof(float));
    alloc_dev(&o_out_dev_, cp_hidden_ * sizeof(float));
    alloc_dev(&gate_dev_, inter_ * sizeof(float));
    alloc_dev(&up_dev_, inter_ * sizeof(float));
    alloc_dev(&ffn_out_dev_, cp_hidden_ * sizeof(float));

    // Attention score buffers
    alloc_dev(&scores_dev_, (size_t)n_heads_ * MAX_SEQ * sizeof(float));
    alloc_dev(&attn_weights_dev_, (size_t)n_heads_ * MAX_SEQ * sizeof(float));

    // RmsNorm rstd scratch
    alloc_dev(&rstd_dev_, sizeof(float));

    // ---- Precompute RoPE cos/sin table on host, upload ----
    {
        int half = head_dim_ / 2;
        std::vector<float> cos_table(MAX_SEQ * half);
        std::vector<float> sin_table(MAX_SEQ * half);
        for (int pos = 0; pos < MAX_SEQ; pos++) {
            for (int i = 0; i < half; i++) {
                float freq = 1.0f / powf(rope_theta_, (float)(2 * i) / head_dim_);
                float angle = pos * freq;
                cos_table[pos * half + i] = cosf(angle);
                sin_table[pos * half + i] = sinf(angle);
            }
        }
        alloc_dev(&rope_cos_dev_, MAX_SEQ * half * sizeof(float));
        alloc_dev(&rope_sin_dev_, MAX_SEQ * half * sizeof(float));
        upload(rope_cos_dev_, cos_table.data(), MAX_SEQ * half);
        upload(rope_sin_dev_, sin_table.data(), MAX_SEQ * half);
    }

    // ---- KV cache ----
    k_cache_dev_.resize(n_layers_, nullptr);
    v_cache_dev_.resize(n_layers_, nullptr);
    for (int il = 0; il < n_layers_; il++) {
        alloc_dev(&k_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * sizeof(float));
        alloc_dev(&v_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * sizeof(float));
    }
    kv_cache_len_ = 0;

    // ---- Pre-allocate host scratch buffers ----
    q_host_.resize(q_dim_);
    k_host_.resize(kv_dim_);
    attn_out_host_.resize(q_dim_);
    k_cache_host_.resize(MAX_SEQ * kv_dim_);
    v_cache_host_.resize(MAX_SEQ * kv_dim_);
    score_buf_.resize(MAX_SEQ);

    // Pre-allocate workspace (start with 4MB, grows if needed)
    workspace_size_ = 4 * 1024 * 1024;
    ACL_CHECK_RET(aclrtMalloc(&workspace_dev_, workspace_size_,
                               ACL_MEM_MALLOC_HUGE_FIRST));

    ready_ = true;
    printf("[cp_cann] Engine initialized: %d layers, device %d\n",
           n_layers_, device_);
    printf("[cp_cann] Buffers: cp_hidden=%d, q_dim=%d, kv_dim=%d, inter=%d\n",
           cp_hidden_, q_dim_, kv_dim_, inter_);
    return true;
}

// ============================================================================
// Host-side helpers for RoPE and QK-norm
// (These operate on tiny vectors -- kernel launch overhead > compute time)
// ============================================================================

void CpCannEngine::apply_rope_on_host(float *vec, int dim, int n_vecs,
                                       int pos) {
    int half = dim / 2;
    for (int v = 0; v < n_vecs; v++) {
        float *h = vec + v * dim;
        for (int i = 0; i < half; i++) {
            float freq = 1.0f / powf(rope_theta_, (float)(2 * i) / dim);
            float angle = pos * freq;
            float cos_a = cosf(angle);
            float sin_a = sinf(angle);
            float x0 = h[i];
            float x1 = h[i + half];
            h[i]        = x0 * cos_a - x1 * sin_a;
            h[i + half] = x1 * cos_a + x0 * sin_a;
        }
    }
}

void CpCannEngine::qk_norm_on_host(float *buf, int dim, int n_vecs,
                                     const float *norm_w_host) {
    for (int v = 0; v < n_vecs; v++) {
        float *h = buf + v * dim;
        float sum_sq = 0.0f;
        for (int i = 0; i < dim; i++) sum_sq += h[i] * h[i];
        float scale = 1.0f / sqrtf(sum_sq / dim + eps_);
        for (int i = 0; i < dim; i++) h[i] = h[i] * scale * norm_w_host[i];
    }
}

// ============================================================================
// Reset KV cache
// ============================================================================

void CpCannEngine::reset_kv_cache() {
    kv_cache_len_ = 0;
    // No need to zero memory -- we track length and only read [0..len)
}

// ============================================================================
// Forward one token through CP transformer on NPU
//
// Strategy: Large matmuls (linear projections, attention V*scores, FFN) run
// on NPU via aclnnMm. RmsNorm runs on NPU via aclnnRmsNorm.
// Small per-head ops (QK-norm, RoPE) run on host to avoid launch overhead.
// Attention score computation uses batched matmul on NPU.
// ============================================================================

void CpCannEngine::forward_one_token(const float *input_talker_space,
                                      int pos, float *hidden_out) {
    assert(ready_);
    ACL_CHECK_RET(aclrtSetDevice(device_));

    // ================================================================
    // 1. Upload input and run input projection: cur = proj_w @ input + proj_b
    //    input: [talker_hidden], cur: [cp_hidden]
    //    matmul: [cp_hidden, talker_hidden] x [talker_hidden, 1] -> [cp_hidden, 1]
    // ================================================================

    // Upload input to a temporary device buffer (reuse q_dev_ as temp since
    // it's large enough: q_dim_ = 2048 >= talker_hidden_ = 2048)
    upload(q_dev_, input_talker_space, talker_hidden_);

    {
        // Mm: cur = proj_w @ input  (as 2D matmul: [ch, th] x [th, 1] -> [ch, 1])
        aclTensor *t_w = make_tensor_2d(proj_w_dev_, cp_hidden_, talker_hidden_);
        aclTensor *t_x = make_tensor_2d(q_dev_, talker_hidden_, 1);
        aclTensor *t_y = make_tensor_2d(cur_dev_, cp_hidden_, 1);

        CANN_OP(stream_, workspace_dev_, workspace_size_,
                Mm, t_w, t_x, t_y, 0);

        aclDestroyTensor(t_w);
        aclDestroyTensor(t_x);
        aclDestroyTensor(t_y);
    }

    // Add bias: cur += proj_b
    {
        aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
        aclTensor *t_b = make_tensor_1d(proj_b_dev_, cp_hidden_);
        float alpha_val = 1.0f;
        aclScalar *alpha = aclCreateScalar(&alpha_val, ACL_FLOAT);

        CANN_OP(stream_, workspace_dev_, workspace_size_,
                InplaceAdd, t_cur, t_b, alpha);

        aclDestroyScalar(alpha);
        aclDestroyTensor(t_cur);
        aclDestroyTensor(t_b);
    }

    // ================================================================
    // 2. Transformer layers
    // ================================================================

    for (int il = 0; il < n_layers_; il++) {
        auto &lw = layer_w_[il];

        // -- Save residual: residual = cur --
        ACL_CHECK_RET(aclrtMemcpyAsync(
            residual_dev_, cp_hidden_ * sizeof(float),
            cur_dev_, cp_hidden_ * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // -- Input LayerNorm: normed = rms_norm(cur, input_ln_w) --
        {
            aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
            aclTensor *t_dst = make_tensor_1d(normed_dev_, cp_hidden_);
            aclTensor *t_gamma = make_tensor_1d(lw.input_ln_w, cp_hidden_);
            aclTensor *t_rstd = make_tensor_1d(rstd_dev_, 1);

            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    RmsNorm, t_cur, t_gamma, eps_, t_dst, t_rstd);

            aclDestroyTensor(t_cur);
            aclDestroyTensor(t_dst);
            aclDestroyTensor(t_gamma);
            aclDestroyTensor(t_rstd);
        }

        // -- Q/K/V projections: q = q_proj @ normed, etc --
        // Q: [q_dim, cp_hidden] x [cp_hidden, 1] -> [q_dim, 1]
        {
            aclTensor *t_w = make_tensor_2d(lw.q_proj_w, q_dim_, cp_hidden_);
            aclTensor *t_x = make_tensor_2d(normed_dev_, cp_hidden_, 1);
            aclTensor *t_y = make_tensor_2d(q_dev_, q_dim_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }
        // K: [kv_dim, cp_hidden] x [cp_hidden, 1] -> [kv_dim, 1]
        {
            aclTensor *t_w = make_tensor_2d(lw.k_proj_w, kv_dim_, cp_hidden_);
            aclTensor *t_x = make_tensor_2d(normed_dev_, cp_hidden_, 1);
            aclTensor *t_y = make_tensor_2d(k_dev_, kv_dim_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }
        // V: [kv_dim, cp_hidden] x [cp_hidden, 1] -> [kv_dim, 1]
        {
            aclTensor *t_w = make_tensor_2d(lw.v_proj_w, kv_dim_, cp_hidden_);
            aclTensor *t_x = make_tensor_2d(normed_dev_, cp_hidden_, 1);
            aclTensor *t_y = make_tensor_2d(v_dev_, kv_dim_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }

        // Sync stream before downloading Q/K for host-side QK-norm + RoPE
        ACL_CHECK_RET(aclrtSynchronizeStream(stream_));

        // Download Q and K to host for QK-norm + RoPE
        download(q_host_.data(), q_dev_, q_dim_);
        download(k_host_.data(), k_dev_, kv_dim_);

        // -- QK Norm (per-head RmsNorm with shared weights, cached on host) --
        qk_norm_on_host(q_host_.data(), head_dim_, n_heads_,
                        qk_norm_host_[il].q_norm_w.data());
        qk_norm_on_host(k_host_.data(), head_dim_, n_kv_,
                        qk_norm_host_[il].k_norm_w.data());

        // -- RoPE (NEOX style) --
        apply_rope_on_host(q_host_.data(), head_dim_, n_heads_, pos);
        apply_rope_on_host(k_host_.data(), head_dim_, n_kv_, pos);

        // Upload Q back to device (needed for attention matmul)
        upload(q_dev_, q_host_.data(), q_dim_);

        // -- Store K/V in KV cache --
        // K -> k_cache[il][pos * kv_dim .. (pos+1) * kv_dim)
        ACL_CHECK_RET(aclrtMemcpy(
            (char *)k_cache_dev_[il] + (size_t)pos * kv_dim_ * sizeof(float),
            kv_dim_ * sizeof(float),
            k_host_.data(), kv_dim_ * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE));

        // V stays on device -- download is wasteful, copy device->device
        // But we already have V on device (v_dev_), so copy directly
        ACL_CHECK_RET(aclrtMemcpy(
            (char *)v_cache_dev_[il] + (size_t)pos * kv_dim_ * sizeof(float),
            kv_dim_ * sizeof(float),
            v_dev_, kv_dim_ * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_DEVICE));

        int seq_len = pos + 1;

        // -- Attention: Q @ K^T -> scores, softmax -> weights, weights @ V -> out --
        // GQA: n_heads=16 query heads, n_kv=8 KV heads, group_size=2
        //
        // For this tiny model (seq_len <= 17, head_dim=128), the attention
        // computation is small enough that batched matmul is efficient.
        //
        // Strategy: do attention per-head on host for simplicity and to avoid
        // complex GQA reshaping on NPU. The vectors are tiny.
        {
            // Read all cached K and V for this layer [0..seq_len)
            download(k_cache_host_.data(), k_cache_dev_[il], seq_len * kv_dim_);
            download(v_cache_host_.data(), v_cache_dev_[il], seq_len * kv_dim_);

            int group_size = n_heads_ / n_kv_;
            float kq_scale = 1.0f / sqrtf((float)head_dim_);

            for (int h = 0; h < n_heads_; h++) {
                int kv_h = h / group_size;
                const float *q_h = q_host_.data() + h * head_dim_;
                float *out_h = attn_out_host_.data() + h * head_dim_;

                // Compute attention scores
                float max_s = -INFINITY;
                for (int p = 0; p < seq_len; p++) {
                    const float *k_p = k_cache_host_.data() + (size_t)p * kv_dim_ + kv_h * head_dim_;
                    float dot = 0.0f;
                    for (int d = 0; d < head_dim_; d++) dot += q_h[d] * k_p[d];
                    score_buf_[p] = dot * kq_scale;
                    if (score_buf_[p] > max_s) max_s = score_buf_[p];
                }

                // Softmax
                float sum = 0.0f;
                for (int p = 0; p < seq_len; p++) {
                    score_buf_[p] = expf(score_buf_[p] - max_s);
                    sum += score_buf_[p];
                }
                float inv_sum = 1.0f / sum;
                for (int p = 0; p < seq_len; p++) score_buf_[p] *= inv_sum;

                // Weighted sum of V
                for (int d = 0; d < head_dim_; d++) {
                    float val = 0.0f;
                    for (int p = 0; p < seq_len; p++) {
                        val += score_buf_[p] *
                               v_cache_host_[(size_t)p * kv_dim_ + kv_h * head_dim_ + d];
                    }
                    out_h[d] = val;
                }
            }

            // Upload attention output to device
            upload(attn_out_dev_, attn_out_host_.data(), q_dim_);
        }

        // -- O projection: o_out = o_proj @ attn_out --
        // [cp_hidden, q_dim] x [q_dim, 1] -> [cp_hidden, 1]
        {
            aclTensor *t_w = make_tensor_2d(lw.o_proj_w, cp_hidden_, q_dim_);
            aclTensor *t_x = make_tensor_2d(attn_out_dev_, q_dim_, 1);
            aclTensor *t_y = make_tensor_2d(o_out_dev_, cp_hidden_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }

        // -- Residual add: cur = residual + o_out --
        {
            aclTensor *t_res = make_tensor_1d(residual_dev_, cp_hidden_);
            aclTensor *t_o = make_tensor_1d(o_out_dev_, cp_hidden_);
            aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
            float alpha_val = 1.0f;
            aclScalar *alpha = aclCreateScalar(&alpha_val, ACL_FLOAT);

            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    Add, t_res, t_o, alpha, t_cur);

            aclDestroyScalar(alpha);
            aclDestroyTensor(t_res);
            aclDestroyTensor(t_o);
            aclDestroyTensor(t_cur);
        }

        // -- Post-attention LayerNorm + FFN --
        // Save residual
        ACL_CHECK_RET(aclrtMemcpyAsync(
            residual_dev_, cp_hidden_ * sizeof(float),
            cur_dev_, cp_hidden_ * sizeof(float),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // Post-attention RmsNorm
        {
            aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
            aclTensor *t_dst = make_tensor_1d(normed_dev_, cp_hidden_);
            aclTensor *t_gamma = make_tensor_1d(lw.post_ln_w, cp_hidden_);
            aclTensor *t_rstd = make_tensor_1d(rstd_dev_, 1);

            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    RmsNorm, t_cur, t_gamma, eps_, t_dst, t_rstd);

            aclDestroyTensor(t_cur);
            aclDestroyTensor(t_dst);
            aclDestroyTensor(t_gamma);
            aclDestroyTensor(t_rstd);
        }

        // -- SwiGLU FFN --
        // gate = gate_proj @ normed
        {
            aclTensor *t_w = make_tensor_2d(lw.gate_proj_w, inter_, cp_hidden_);
            aclTensor *t_x = make_tensor_2d(normed_dev_, cp_hidden_, 1);
            aclTensor *t_y = make_tensor_2d(gate_dev_, inter_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }
        // up = up_proj @ normed
        {
            aclTensor *t_w = make_tensor_2d(lw.up_proj_w, inter_, cp_hidden_);
            aclTensor *t_x = make_tensor_2d(normed_dev_, cp_hidden_, 1);
            aclTensor *t_y = make_tensor_2d(up_dev_, inter_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }

        // SwiGLU: output = silu(gate) * up, computed via:
        //   1. silu_out = SiLU(gate)   [use attn_out_dev_ as temp, big enough]
        //   2. silu_out *= up           [in-place multiply]
        //   3. ffn_out = down_proj @ silu_out
        // Note: attn_out_dev_ is [q_dim=2048] but we need [inter=3072].
        // So we use ffn_out_dev_ isn't big enough either ([cp_hidden=1024]).
        // We already have scores_dev_ and attn_weights_dev_ that are small.
        // Simplest: just do SiLU in-place on gate (same buffer, elementwise).
        {
            aclTensor *t_src = make_tensor_1d(gate_dev_, inter_);
            aclTensor *t_dst = make_tensor_1d(gate_dev_, inter_);
            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    Silu, t_src, t_dst);
            aclDestroyTensor(t_src);
            aclDestroyTensor(t_dst);
        }

        // gate *= up (element-wise multiply, in-place on gate)
        {
            aclTensor *t_gate = make_tensor_1d(gate_dev_, inter_);
            aclTensor *t_up = make_tensor_1d(up_dev_, inter_);
            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    InplaceMul, t_gate, t_up);
            aclDestroyTensor(t_gate);
            aclDestroyTensor(t_up);
        }

        // down = down_proj @ gate -> ffn_out
        {
            aclTensor *t_w = make_tensor_2d(lw.down_proj_w, cp_hidden_, inter_);
            aclTensor *t_x = make_tensor_2d(gate_dev_, inter_, 1);
            aclTensor *t_y = make_tensor_2d(ffn_out_dev_, cp_hidden_, 1);
            CANN_OP(stream_, workspace_dev_, workspace_size_, Mm, t_w, t_x, t_y, 0);
            aclDestroyTensor(t_w); aclDestroyTensor(t_x); aclDestroyTensor(t_y);
        }

        // -- Residual add: cur = residual + ffn_out --
        {
            aclTensor *t_res = make_tensor_1d(residual_dev_, cp_hidden_);
            aclTensor *t_ffn = make_tensor_1d(ffn_out_dev_, cp_hidden_);
            aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
            float alpha_val = 1.0f;
            aclScalar *alpha = aclCreateScalar(&alpha_val, ACL_FLOAT);

            CANN_OP(stream_, workspace_dev_, workspace_size_,
                    Add, t_res, t_ffn, alpha, t_cur);

            aclDestroyScalar(alpha);
            aclDestroyTensor(t_res);
            aclDestroyTensor(t_ffn);
            aclDestroyTensor(t_cur);
        }
    }  // end layer loop

    // ================================================================
    // 3. Final RmsNorm: hidden_out = rms_norm(cur, final_norm_w)
    // ================================================================
    {
        aclTensor *t_cur = make_tensor_1d(cur_dev_, cp_hidden_);
        // Write final norm output into normed_dev_ (reuse)
        aclTensor *t_dst = make_tensor_1d(normed_dev_, cp_hidden_);
        aclTensor *t_gamma = make_tensor_1d(final_norm_w_dev_, cp_hidden_);
        aclTensor *t_rstd = make_tensor_1d(rstd_dev_, 1);

        CANN_OP(stream_, workspace_dev_, workspace_size_,
                RmsNorm, t_cur, t_gamma, eps_, t_dst, t_rstd);

        aclDestroyTensor(t_cur);
        aclDestroyTensor(t_dst);
        aclDestroyTensor(t_gamma);
        aclDestroyTensor(t_rstd);
    }

    // Sync and download result
    ACL_CHECK_RET(aclrtSynchronizeStream(stream_));
    download(hidden_out, normed_dev_, cp_hidden_);
}
