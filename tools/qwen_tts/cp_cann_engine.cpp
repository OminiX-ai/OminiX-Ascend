// ============================================================================
// CP CANN Engine v4 — F16 weights + F16 compute end-to-end.
//
// Why F16: llama.cpp's CANN backend runs this model in F16 and the audio
// sounds right. My earlier F32 path (v2/v3) was numerically correct (bit-
// identical to the f32 CPU reference) but produced "garbled" speech because
// the model was trained/tuned at F16 and F32 inference exposes numerical
// artifacts the vocoder can't parse.
//
// Interface stays F32: the caller still passes a `const float *` input and
// receives a `float *` hidden state. We stage the boundary in F32 buffers
// and aclnnCast into/out-of the F16 working tensors.
//
// Mechanics:
//   - All device buffers (weights, intermediates, KV cache, cos/sin, scores,
//     rstd) are F16 — half the memory of v3.
//   - Weights converted F32 -> F16 on host at upload via __fp16 cast.
//   - Scalars (attn_scale, one_scalar) created with ACL_FLOAT16.
//   - aclnnCast at forward entry/exit only.
// ============================================================================

#include "cp_cann_engine.h"
#include "cp_cann_symbols.h"
#include "talker.h"

#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cassert>
#include <vector>
#include <fstream>
#include <sstream>
#include <map>

#define ACL_CHECK_RET(stmt) do {                                      \
    aclError _ret = (stmt);                                           \
    if (_ret != 0) {                                                  \
        fprintf(stderr, "[cp_cann] ACL error %d at %s:%d: %s\n",      \
                _ret, __FILE__, __LINE__,                             \
                g_cann.aclGetRecentErrMsg                             \
                    ? g_cann.aclGetRecentErrMsg() : "<n/a>");         \
    }                                                                 \
} while (0)

#define CANN_OP(OP_NAME, ...) do {                                    \
    uint64_t _ws_needed = 0;                                          \
    aclOpExecutor *_exec = nullptr;                                   \
    ACL_CHECK_RET(g_cann.aclnn##OP_NAME##GetWorkspaceSize(             \
        __VA_ARGS__, &_ws_needed, &_exec));                           \
    if (_ws_needed > workspace_size_) {                               \
        if (workspace_dev_) g_cann.aclrtFree(workspace_dev_);         \
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, _ws_needed, \
                       ACL_MEM_MALLOC_HUGE_FIRST));                   \
        workspace_size_ = _ws_needed;                                 \
    }                                                                 \
    void *_ws = _ws_needed > 0 ? workspace_dev_ : nullptr;            \
    ACL_CHECK_RET(g_cann.aclnn##OP_NAME(_ws, _ws_needed, _exec,       \
                                         stream_));                   \
} while (0)

// ============================================================================
// fp32 <-> fp16 helpers. On aarch64 we have native __fp16.
// ============================================================================

static inline uint16_t fp32_to_fp16(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

static aclScalar *make_f16_scalar(float value) {
    uint16_t bits = fp32_to_fp16(value);
    return g_cann.aclCreateScalar(&bits, ACL_FLOAT16);
}

// ============================================================================
// Tensor-descriptor constructors. Default dtype is F16; override with F32 only
// for the I/O staging tensors.
// ============================================================================

static aclTensor *make_tensor(void *buf, int64_t rank,
                              const int64_t *shape, const int64_t *strides,
                              aclDataType dtype,
                              aclFormat fmt = ACL_FORMAT_ND) {
    int64_t storage_len = 0;
    if (rank > 0) {
        int64_t n = 1;
        for (int64_t i = 0; i < rank; ++i) n *= shape[i];
        storage_len = n;
    }
    return g_cann.aclCreateTensor(shape, rank, dtype, strides, 0,
                                   fmt, &storage_len, 1, buf);
}

static aclTensor *tensor_1d(void *buf, int64_t n,
                             aclDataType dtype = ACL_FLOAT16) {
    int64_t shape[1]   = {n};
    int64_t strides[1] = {1};
    return make_tensor(buf, 1, shape, strides, dtype);
}

static aclTensor *tensor_2d(void *buf, int64_t d0, int64_t d1,
                             aclDataType dtype = ACL_FLOAT16) {
    int64_t shape[2]   = {d0, d1};
    int64_t strides[2] = {d1, 1};
    return make_tensor(buf, 2, shape, strides, dtype);
}

// M5.3 — build a [d0, d1] weight tensor with the caller-selected aclFormat.
// Use ACL_FORMAT_FRACTAL_NZ for buffers that have already been converted in
// place via aclnnTransMatmulWeight so plain aclnnMm dispatches the NZ kernel
// path; pass ACL_FORMAT_ND (or call plain tensor_2d) for activation tensors.
static aclTensor *tensor_2d_fmt(void *buf, int64_t d0, int64_t d1,
                                 aclDataType dtype, aclFormat fmt) {
    int64_t shape[2]   = {d0, d1};
    int64_t strides[2] = {d1, 1};
    return make_tensor(buf, 2, shape, strides, dtype, fmt);
}

static aclTensor *tensor_strided(void *buf, int64_t rank,
                                  const int64_t *shape,
                                  const int64_t *strides,
                                  aclDataType dtype = ACL_FLOAT16) {
    return make_tensor(buf, rank, shape, strides, dtype);
}

// ============================================================================
// Lifecycle
// ============================================================================

CpCannEngine::~CpCannEngine() {
    if (!ready_) return;

    destroy_persistent_tensors_();

    auto free_dev = [](void *&p) {
        if (p) { g_cann.aclrtFree(p); p = nullptr; }
    };

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
    free_dev(scores_f16_dev_);
    free_dev(rope_cos_dev_);
    free_dev(rope_sin_dev_);
    free_dev(rstd_dev_);

    for (auto &p : k_cache_dev_) free_dev(p);
    for (auto &p : v_cache_dev_) free_dev(p);

    free_dev(workspace_dev_);
    free_dev(input_stage_f32_dev_);
    free_dev(output_stage_f32_dev_);
    free_dev(proj_out_f32_dev_);
    free_dev(accum_scratch_f32_dev_);
    free_dev(normed_f32_dev_);

    if (attn_scale_)     { g_cann.aclDestroyScalar(attn_scale_);     attn_scale_     = nullptr; }
    if (attn_scale_f16_) { g_cann.aclDestroyScalar(attn_scale_f16_); attn_scale_f16_ = nullptr; }
    if (one_scalar_)     { g_cann.aclDestroyScalar(one_scalar_);     one_scalar_     = nullptr; }

    // M6.2: destroy the forward-done event before the streams.
    if (forward_done_event_ && g_cann.aclrtDestroyEvent) {
        g_cann.aclrtDestroyEvent(forward_done_event_);
        forward_done_event_ = nullptr;
    }

    // stream_ may be pointing at primary_stream_ (default) or at some
    // externally-owned stream handed in via set_stream() — only destroy the
    // two streams this engine owns.
    if (primary_stream_) {
        g_cann.aclrtDestroyStream(primary_stream_);
        primary_stream_ = nullptr;
    }
    if (stream_b_) {
        g_cann.aclrtDestroyStream(stream_b_);
        stream_b_ = nullptr;
    }
    stream_ = nullptr;
}

// ============================================================================
// Memory helpers
// ============================================================================

void CpCannEngine::alloc_dev(void **ptr, size_t bytes) {
    ACL_CHECK_RET(g_cann.aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void CpCannEngine::upload(void *dev, const float *host, size_t n_floats) {
    ACL_CHECK_RET(g_cann.aclrtMemcpy(dev, n_floats * sizeof(float),
                                      host, n_floats * sizeof(float),
                                      ACL_MEMCPY_HOST_TO_DEVICE));
}

void CpCannEngine::download(float *host, const void *dev, size_t n_floats) {
    ACL_CHECK_RET(g_cann.aclrtMemcpy(host, n_floats * sizeof(float),
                                      dev, n_floats * sizeof(float),
                                      ACL_MEMCPY_DEVICE_TO_HOST));
}

void CpCannEngine::ensure_workspace(size_t needed) {
    if (needed <= workspace_size_) return;
    if (workspace_dev_) g_cann.aclrtFree(workspace_dev_);
    ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, needed,
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    workspace_size_ = needed;
}

// F16 upload: convert F32 host -> uint16 F16 bit pattern -> device.
static void upload_f16(void *dev, const float *host, size_t n) {
    std::vector<uint16_t> buf(n);
    for (size_t i = 0; i < n; ++i) buf[i] = fp32_to_fp16(host[i]);
    ACL_CHECK_RET(g_cann.aclrtMemcpy(dev, n * sizeof(uint16_t),
                                      buf.data(), n * sizeof(uint16_t),
                                      ACL_MEMCPY_HOST_TO_DEVICE));
}

// ============================================================================
// FRACTAL_NZ weight pre-conversion (M5.2). Mutates the weight tensor buffer
// in place to the private NZ layout via aclnnTransMatmulWeight. Safe no-op
// when g_cann.has_nz() is false. The F32 input projection weight is NOT
// eligible — TransMatmulWeight only supports F16/INT8/BF16 — so callers
// only invoke this on F16 Q/K/V/O/gate/up/down weights.
// ============================================================================

void CpCannEngine::nz_convert_weight_(void *weight_dev,
                                       int64_t rows, int64_t cols) {
    if (!g_cann.has_nz() || weight_dev == nullptr) return;

    int64_t shape[2]   = {rows, cols};
    int64_t strides[2] = {cols, 1};
    int64_t storage_len = rows * cols;
    aclTensor *t = g_cann.aclCreateTensor(
        shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND,
        &storage_len, 1, weight_dev);
    if (!t) {
        fprintf(stderr, "[cp_cann] nz_convert: aclCreateTensor failed\n");
        return;
    }

    uint64_t ws_needed = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnTransMatmulWeightGetWorkspaceSize(
        t, &ws_needed, &exec);
    if (s != 0) {
        fprintf(stderr, "[cp_cann] nz_convert: GetWorkspaceSize status=%d "
                        "(shape=%lldx%lld)\n",
                (int)s, (long long)rows, (long long)cols);
        g_cann.aclDestroyTensor(t);
        return;
    }
    ensure_workspace(ws_needed);
    void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
    s = g_cann.aclnnTransMatmulWeight(ws, ws_needed, exec, stream_);
    if (s != 0) {
        fprintf(stderr, "[cp_cann] nz_convert: TransMatmulWeight status=%d "
                        "(shape=%lldx%lld)\n",
                (int)s, (long long)rows, (long long)cols);
    }
    ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    g_cann.aclDestroyTensor(t);
}

// ============================================================================
// Persistent aclTensor construction / destruction
// ============================================================================

void CpCannEngine::build_persistent_tensors_() {
    const int group = n_heads_ / n_kv_;

    // Input-projection tensors — these stay F32. Transformer weights below
    // remain F16. The F16 view of proj_w/proj_b is unused (we upload F32 now),
    // but the member still exists in the struct — just point it at the F32
    // buffer with F32 dtype for symmetry.
    t_.proj_w = tensor_2d(proj_w_dev_, cp_hidden_, talker_hidden_, ACL_FLOAT);
    t_.proj_b = tensor_1d(proj_b_dev_, cp_hidden_,                 ACL_FLOAT);
    t_.proj_w_f32 = t_.proj_w;
    t_.proj_b_f32 = t_.proj_b;
    t_.proj_out_f32_col  = tensor_2d(proj_out_f32_dev_, cp_hidden_, 1,
                                      ACL_FLOAT);
    t_.proj_out_f32_flat = tensor_1d(proj_out_f32_dev_, cp_hidden_,
                                      ACL_FLOAT);
    t_.cur_f16_flat_as_target = tensor_1d(cur_dev_, cp_hidden_,
                                           ACL_FLOAT16);
    t_.accum_scratch_f32_flat = tensor_1d(accum_scratch_f32_dev_,
                                           cp_hidden_, ACL_FLOAT);
    t_.accum_f32_row_2d   = tensor_2d(proj_out_f32_dev_, 1, cp_hidden_,
                                       ACL_FLOAT);
    t_.normed_f32_row_2d  = tensor_2d(normed_f32_dev_, 1, cp_hidden_,
                                       ACL_FLOAT);
    t_.normed_f32_flat    = tensor_1d(normed_f32_dev_, cp_hidden_, ACL_FLOAT);

    // M5.3 — once aclnnTransMatmulWeight has refreshed each F16 matmul weight
    // buffer to the NZ layout, tag the persistent weight descriptors with
    // ACL_FORMAT_FRACTAL_NZ so plain aclnnMm picks the NZ kernel. Activation
    // descriptors (normed_col / q_col / gate_col / ...) stay ND.
    const aclFormat wfmt = nz_applied_ ? ACL_FORMAT_FRACTAL_NZ
                                       : ACL_FORMAT_ND;
    if (nz_applied_) {
        printf("[cp_cann] matmul call sites tag weights with "
               "ACL_FORMAT_FRACTAL_NZ (M5.3)\n");
    }
    t_.layer.resize(n_layers_);
    for (int il = 0; il < n_layers_; ++il) {
        auto &lw = layer_w_[il];
        auto &lt = t_.layer[il];
        lt.q_proj    = tensor_2d_fmt(lw.q_proj_w,    q_dim_,     cp_hidden_, ACL_FLOAT16, wfmt);
        lt.k_proj    = tensor_2d_fmt(lw.k_proj_w,    kv_dim_,    cp_hidden_, ACL_FLOAT16, wfmt);
        lt.v_proj    = tensor_2d_fmt(lw.v_proj_w,    kv_dim_,    cp_hidden_, ACL_FLOAT16, wfmt);
        lt.o_proj    = tensor_2d_fmt(lw.o_proj_w,    cp_hidden_, q_dim_,     ACL_FLOAT16, wfmt);
        // Norms kept F32 to match GGUF storage + aclnnRmsNorm SupportInfo[0].
        lt.q_norm    = tensor_1d(lw.q_norm_w,    head_dim_,  ACL_FLOAT);
        lt.k_norm    = tensor_1d(lw.k_norm_w,    head_dim_,  ACL_FLOAT);
        lt.gate_proj = tensor_2d_fmt(lw.gate_proj_w, inter_,     cp_hidden_, ACL_FLOAT16, wfmt);
        lt.up_proj   = tensor_2d_fmt(lw.up_proj_w,   inter_,     cp_hidden_, ACL_FLOAT16, wfmt);
        lt.down_proj = tensor_2d_fmt(lw.down_proj_w, cp_hidden_, inter_,     ACL_FLOAT16, wfmt);
        lt.input_ln  = tensor_1d(lw.input_ln_w,  cp_hidden_, ACL_FLOAT);
        lt.post_ln   = tensor_1d(lw.post_ln_w,   cp_hidden_, ACL_FLOAT);
    }
    t_.final_norm = tensor_1d(final_norm_w_dev_, cp_hidden_, ACL_FLOAT);

    t_.cur_row       = tensor_2d(cur_dev_, 1, cp_hidden_);
    t_.cur_flat      = tensor_1d(cur_dev_, cp_hidden_);
    t_.residual_flat = tensor_1d(residual_dev_, cp_hidden_);
    t_.normed_row    = tensor_2d(normed_dev_, 1,         cp_hidden_);
    t_.normed_col    = tensor_2d(normed_dev_, cp_hidden_, 1);

    t_.q_col    = tensor_2d(q_dev_, q_dim_, 1);
    t_.q_heads  = tensor_2d(q_dev_, n_heads_, head_dim_);
    {
        int64_t shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                               (int64_t)head_dim_, 1};
        t_.q_rope_4d = tensor_strided(q_dev_, 4, shape, strides);
    }
    {
        int64_t shape[3]   = {(int64_t)n_kv_, (int64_t)group,
                               (int64_t)head_dim_};
        int64_t strides[3] = {(int64_t)group * head_dim_,
                               (int64_t)head_dim_, 1};
        t_.q_gqa = tensor_strided(q_dev_, 3, shape, strides);
    }

    t_.k_col = tensor_2d(k_dev_, kv_dim_, 1);
    t_.k_kv  = tensor_2d(k_dev_, n_kv_, head_dim_);

    t_.v_col = tensor_2d(v_dev_, kv_dim_, 1);

    t_.attn_out_col = tensor_2d(attn_out_dev_, q_dim_, 1);
    {
        int64_t shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                               (int64_t)head_dim_, 1};
        t_.attn_out_4d = tensor_strided(attn_out_dev_, 4, shape, strides);
    }
    {
        int64_t shape[3]   = {(int64_t)n_kv_, (int64_t)group,
                               (int64_t)head_dim_};
        int64_t strides[3] = {(int64_t)group * head_dim_,
                               (int64_t)head_dim_, 1};
        t_.attn_out_gqa = tensor_strided(attn_out_dev_, 3, shape, strides);
    }

    t_.o_out_col  = tensor_2d(o_out_dev_, cp_hidden_, 1);
    t_.o_out_flat = tensor_1d(o_out_dev_, cp_hidden_);

    t_.gate_col  = tensor_2d(gate_dev_, inter_, 1);
    t_.gate_flat = tensor_1d(gate_dev_, inter_);
    t_.up_col    = tensor_2d(up_dev_,   inter_, 1);
    t_.up_flat   = tensor_1d(up_dev_,   inter_);

    t_.ffn_out_col  = tensor_2d(ffn_out_dev_, cp_hidden_, 1);
    t_.ffn_out_flat = tensor_1d(ffn_out_dev_, cp_hidden_);

    // rstd MUST be F32 even with F16 input — aclnnRmsNorm only supports
    // rstd dtype=F32 on Ascend (validated against the op's SupportInfo list).
    t_.rstd_11    = tensor_2d(rstd_dev_, 1,        1,    ACL_FLOAT);
    t_.rstd_heads = tensor_2d(rstd_dev_, n_heads_, 1,    ACL_FLOAT);
    t_.rstd_kv    = tensor_2d(rstd_dev_, n_kv_,    1,    ACL_FLOAT);

    // --- Boundary staging tensors (F32) for aclnnCast entry/exit --
    t_.input_f32  = tensor_1d(input_stage_f32_dev_,  talker_hidden_, ACL_FLOAT);
    t_.input_f16  = tensor_1d(q_dev_,                talker_hidden_, ACL_FLOAT16);
    t_.output_f16 = tensor_1d(normed_dev_,           cp_hidden_,     ACL_FLOAT16);
    t_.output_f32 = tensor_1d(output_stage_f32_dev_, cp_hidden_,     ACL_FLOAT);
}

void CpCannEngine::destroy_persistent_tensors_() {
    auto drop = [](aclTensor *&t) {
        if (t) { g_cann.aclDestroyTensor(t); t = nullptr; }
    };
    drop(t_.proj_w);
    drop(t_.proj_b);
    for (auto &lt : t_.layer) {
        drop(lt.q_proj); drop(lt.k_proj); drop(lt.v_proj); drop(lt.o_proj);
        drop(lt.q_norm); drop(lt.k_norm);
        drop(lt.gate_proj); drop(lt.up_proj); drop(lt.down_proj);
        drop(lt.input_ln); drop(lt.post_ln);
    }
    t_.layer.clear();
    drop(t_.final_norm);
    drop(t_.cur_row);   drop(t_.cur_flat);
    drop(t_.residual_flat);
    drop(t_.normed_row); drop(t_.normed_col);
    drop(t_.q_col); drop(t_.q_heads); drop(t_.q_rope_4d); drop(t_.q_gqa);
    drop(t_.k_col); drop(t_.k_kv);
    drop(t_.v_col);
    drop(t_.attn_out_col); drop(t_.attn_out_4d); drop(t_.attn_out_gqa);
    drop(t_.o_out_col); drop(t_.o_out_flat);
    drop(t_.gate_col); drop(t_.gate_flat);
    drop(t_.up_col);   drop(t_.up_flat);
    drop(t_.ffn_out_col); drop(t_.ffn_out_flat);
    drop(t_.rstd_11); drop(t_.rstd_heads); drop(t_.rstd_kv);
    drop(t_.input_f32); drop(t_.input_f16);
    drop(t_.output_f16); drop(t_.output_f32);
    // proj_w_f32 / proj_b_f32 are aliases of proj_w / proj_b (already dropped).
    t_.proj_w_f32 = nullptr;
    t_.proj_b_f32 = nullptr;
    drop(t_.proj_out_f32_col); drop(t_.proj_out_f32_flat);
    drop(t_.cur_f16_flat_as_target);
    drop(t_.accum_scratch_f32_flat);
    drop(t_.accum_f32_row_2d);
    drop(t_.normed_f32_row_2d);
    drop(t_.normed_f32_flat);
}

// ============================================================================
// Minimal safetensors reader for the MLX-extracted CP weight file.
//
// Format: uint64 header_size little-endian | JSON header | raw tensor bytes.
// Header JSON maps `name` → {dtype, shape, data_offsets: [start, end]}.
// We don't pull in a real JSON library — the entries we need are looked up
// by string search, and data_offsets is a flat numeric array.
// ============================================================================

namespace {

struct StEntry {
    uint64_t data_start = 0;  // offset relative to tensor data region
    uint64_t data_end   = 0;
};

// Find `"<name>":{...}` and extract its `"data_offsets":[start,end]`.
bool st_lookup(const std::string &header, const std::string &name,
               StEntry &out) {
    std::string key = "\"" + name + "\"";
    size_t p = header.find(key);
    if (p == std::string::npos) return false;
    size_t do_pos = header.find("\"data_offsets\"", p);
    if (do_pos == std::string::npos) return false;
    size_t lb = header.find('[', do_pos);
    size_t rb = header.find(']', lb);
    if (lb == std::string::npos || rb == std::string::npos) return false;
    std::string inner = header.substr(lb + 1, rb - lb - 1);
    size_t comma = inner.find(',');
    if (comma == std::string::npos) return false;
    out.data_start = std::stoull(inner.substr(0, comma));
    out.data_end   = std::stoull(inner.substr(comma + 1));
    return true;
}

// Convert a buffer of BF16 values (stored as uint16_t big-endian-ish halfs of
// the original F32) into F16 values ready for NPU upload. For values within
// F16's exponent range (true for trained transformer weights), this conversion
// is effectively lossless — F16 has more mantissa bits than BF16, so the
// BF16 → F32 → F16 round-trip fills the extra bits with zeros.
void bf16_buf_to_fp16_buf(const uint16_t *bf16_src,
                           uint16_t *fp16_dst, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        union { uint32_t u; float f; } v;
        v.u = ((uint32_t)bf16_src[i]) << 16;
        __fp16 h = (__fp16)v.f;
        std::memcpy(&fp16_dst[i], &h, sizeof(uint16_t));
    }
}

}  // namespace

bool CpCannEngine::init_from_safetensors(const std::string &path,
                                          const CodePredictorConfig &cfg,
                                          int device) {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[cp_cann] symbol load failed; engine disabled\n");
        return false;
    }

    // Cache dims first.
    device_ = device;
    talker_hidden_ = cfg.talker_hidden_size;
    cp_hidden_     = cfg.hidden_size;
    n_heads_       = cfg.num_attention_heads;
    n_kv_          = cfg.num_key_value_heads;
    head_dim_      = cfg.head_dim;
    q_dim_         = n_heads_ * head_dim_;
    kv_dim_        = n_kv_ * head_dim_;
    inter_         = cfg.intermediate_size;
    n_layers_      = cfg.num_hidden_layers;
    eps_           = cfg.rms_norm_eps;
    rope_theta_    = cfg.rope_theta;

    // Open + read header.
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        fprintf(stderr, "[cp_cann] safetensors open failed: %s\n", path.c_str());
        return false;
    }
    uint64_t header_size = 0;
    f.read(reinterpret_cast<char *>(&header_size), 8);
    std::string header(header_size, '\0');
    f.read(&header[0], header_size);
    const std::streamoff data_base = 8 + (std::streamoff)header_size;

    ACL_CHECK_RET(g_cann.aclrtSetDevice(device_));
    // Primary + secondary streams (M6.1). Primary is the default target;
    // stream_b_ is spare for multi-stream pipelining.
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&primary_stream_));
    stream_ = primary_stream_;
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&stream_b_));
    // M6.2: reusable event recorded at the end of forward_one_token_launch.
    if (g_cann.aclrtCreateEvent && !forward_done_event_) {
        ACL_CHECK_RET(g_cann.aclrtCreateEvent(&forward_done_event_));
    }

    // Helper: load one tensor by name, convert BF16 → F16 on host, upload.
    auto load_one_f16 = [&](const std::string &name, void *&dev,
                            size_t expected_elems) {
        StEntry e;
        if (!st_lookup(header, name, e)) {
            fprintf(stderr, "[cp_cann] missing tensor: %s\n", name.c_str());
            return false;
        }
        size_t nbytes = e.data_end - e.data_start;
        if (nbytes != expected_elems * 2) {
            fprintf(stderr,
                    "[cp_cann] size mismatch for %s: got %zu bytes, "
                    "expected %zu (%zu bf16 elems)\n",
                    name.c_str(), nbytes, expected_elems * 2, expected_elems);
            return false;
        }
        std::vector<uint16_t> bf16_buf(expected_elems);
        f.seekg(data_base + (std::streamoff)e.data_start);
        f.read(reinterpret_cast<char *>(bf16_buf.data()), nbytes);
        std::vector<uint16_t> fp16_buf(expected_elems);
        bf16_buf_to_fp16_buf(bf16_buf.data(), fp16_buf.data(), expected_elems);
        alloc_dev(&dev, expected_elems * 2);
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            dev, nbytes, fp16_buf.data(), nbytes,
            ACL_MEMCPY_HOST_TO_DEVICE));
        return true;
    };
    // For tensors we keep as F32 on device (norm gammas, proj_w/b for the
    // F32 input projection). BF16 → F32 is exact (just high-bit shift).
    auto load_one_f32 = [&](const std::string &name, void *&dev,
                            size_t expected_elems) {
        StEntry e;
        if (!st_lookup(header, name, e)) {
            fprintf(stderr, "[cp_cann] missing tensor: %s\n", name.c_str());
            return false;
        }
        size_t nbytes = e.data_end - e.data_start;
        if (nbytes != expected_elems * 2) {
            fprintf(stderr, "[cp_cann] size mismatch for %s\n", name.c_str());
            return false;
        }
        std::vector<uint16_t> bf16_buf(expected_elems);
        f.seekg(data_base + (std::streamoff)e.data_start);
        f.read(reinterpret_cast<char *>(bf16_buf.data()), nbytes);
        std::vector<float> f32_buf(expected_elems);
        for (size_t i = 0; i < expected_elems; ++i) {
            union { uint32_t u; float f; } v;
            v.u = ((uint32_t)bf16_buf[i]) << 16;
            f32_buf[i] = v.f;
        }
        alloc_dev(&dev, expected_elems * sizeof(float));
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            dev, expected_elems * sizeof(float), f32_buf.data(),
            expected_elems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
        return true;
    };

    const std::string pfx = "talker.code_predictor.";
    const std::string mpfx = pfx + "model.";

    // Input projection: F32 on device to match MLX's F32 input embedding
    // precision + our F32 projection compute path.
    if (!load_one_f32(pfx + "small_to_mtp_projection.weight", proj_w_dev_,
                      (size_t)cp_hidden_ * talker_hidden_)) return false;
    if (!load_one_f32(pfx + "small_to_mtp_projection.bias",   proj_b_dev_,
                      cp_hidden_)) return false;

    // ---- FRACTAL_NZ gating (M5.2) ------------------------------------------
    const bool nz_enabled = use_nz_weights_ && g_cann.has_nz();
    if (use_nz_weights_ && !g_cann.has_nz()) {
        printf("[cp_cann] NZ weights requested but "
               "aclnnTransMatmulWeight unresolved — falling back to ND\n");
    }
    if (nz_enabled && workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Per-layer weights.
    layer_w_.resize(n_layers_);
    for (int il = 0; il < n_layers_; ++il) {
        auto &dst = layer_w_[il];
        const std::string lp = mpfx + "layers." + std::to_string(il) + ".";
        if (!load_one_f16(lp + "self_attn.q_proj.weight", dst.q_proj_w,
                          (size_t)q_dim_ * cp_hidden_)) return false;
        if (!load_one_f16(lp + "self_attn.k_proj.weight", dst.k_proj_w,
                          (size_t)kv_dim_ * cp_hidden_)) return false;
        if (!load_one_f16(lp + "self_attn.v_proj.weight", dst.v_proj_w,
                          (size_t)kv_dim_ * cp_hidden_)) return false;
        if (!load_one_f16(lp + "self_attn.o_proj.weight", dst.o_proj_w,
                          (size_t)cp_hidden_ * q_dim_)) return false;
        if (!load_one_f16(lp + "mlp.gate_proj.weight", dst.gate_proj_w,
                          (size_t)inter_ * cp_hidden_)) return false;
        if (!load_one_f16(lp + "mlp.up_proj.weight", dst.up_proj_w,
                          (size_t)inter_ * cp_hidden_)) return false;
        if (!load_one_f16(lp + "mlp.down_proj.weight", dst.down_proj_w,
                          (size_t)cp_hidden_ * inter_)) return false;
        if (nz_enabled) {
            nz_convert_weight_(dst.q_proj_w,    q_dim_,     cp_hidden_);
            nz_convert_weight_(dst.k_proj_w,    kv_dim_,    cp_hidden_);
            nz_convert_weight_(dst.v_proj_w,    kv_dim_,    cp_hidden_);
            nz_convert_weight_(dst.o_proj_w,    cp_hidden_, q_dim_);
            nz_convert_weight_(dst.gate_proj_w, inter_,     cp_hidden_);
            nz_convert_weight_(dst.up_proj_w,   inter_,     cp_hidden_);
            nz_convert_weight_(dst.down_proj_w, cp_hidden_, inter_);
        }
        // Norms kept F32 (matches the F32 norm path in v9/v11).
        if (!load_one_f32(lp + "self_attn.q_norm.weight", dst.q_norm_w,
                          head_dim_)) return false;
        if (!load_one_f32(lp + "self_attn.k_norm.weight", dst.k_norm_w,
                          head_dim_)) return false;
        if (!load_one_f32(lp + "input_layernorm.weight", dst.input_ln_w,
                          cp_hidden_)) return false;
        if (!load_one_f32(lp + "post_attention_layernorm.weight",
                          dst.post_ln_w, cp_hidden_)) return false;
    }
    if (!load_one_f32(mpfx + "norm.weight", final_norm_w_dev_, cp_hidden_))
        return false;

    // --- The rest of init (buffers, scalars, cos/sin, persistent tensors,
    //     workspace) is identical to init(); reuse by setting up the shared
    //     tail. Factor it out to keep this maintainable. ---
    const size_t E = sizeof(uint16_t);

    alloc_dev(&cur_dev_,       cp_hidden_ * E);
    alloc_dev(&residual_dev_,  cp_hidden_ * E);
    alloc_dev(&normed_dev_,    cp_hidden_ * E);
    alloc_dev(&q_dev_,         q_dim_     * E);
    alloc_dev(&k_dev_,         kv_dim_    * E);
    alloc_dev(&v_dev_,         kv_dim_    * E);
    alloc_dev(&attn_out_dev_,  q_dim_     * E);
    alloc_dev(&o_out_dev_,     cp_hidden_ * E);
    alloc_dev(&gate_dev_,      inter_     * E);
    alloc_dev(&up_dev_,        inter_     * E);
    alloc_dev(&ffn_out_dev_,   cp_hidden_ * E);
    alloc_dev(&scores_dev_,    (size_t)n_heads_ * MAX_SEQ * sizeof(float));
    alloc_dev(&scores_f16_dev_,(size_t)n_heads_ * MAX_SEQ * E);
    alloc_dev(&rstd_dev_,      (size_t)n_heads_ * sizeof(float));
    alloc_dev(&input_stage_f32_dev_,  talker_hidden_ * sizeof(float));
    alloc_dev(&output_stage_f32_dev_, cp_hidden_     * sizeof(float));
    alloc_dev(&proj_out_f32_dev_,     cp_hidden_     * sizeof(float));
    alloc_dev(&accum_scratch_f32_dev_, cp_hidden_    * sizeof(float));
    alloc_dev(&normed_f32_dev_,        cp_hidden_    * sizeof(float));

    {
        const int half = head_dim_ / 2;
        std::vector<float> cos_table((size_t)MAX_SEQ * head_dim_);
        std::vector<float> sin_table((size_t)MAX_SEQ * head_dim_);
        for (int p = 0; p < MAX_SEQ; ++p) {
            for (int j = 0; j < half; ++j) {
                float freq  = 1.0f / powf(rope_theta_, (float)(2 * j) / head_dim_);
                float angle = (float)p * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                cos_table[(size_t)p * head_dim_ + j]        = c;
                cos_table[(size_t)p * head_dim_ + j + half] = c;
                sin_table[(size_t)p * head_dim_ + j]        = s;
                sin_table[(size_t)p * head_dim_ + j + half] = s;
            }
        }
        alloc_dev(&rope_cos_dev_, (size_t)MAX_SEQ * head_dim_ * E);
        alloc_dev(&rope_sin_dev_, (size_t)MAX_SEQ * head_dim_ * E);
        upload_f16(rope_cos_dev_, cos_table.data(),
                   (size_t)MAX_SEQ * head_dim_);
        upload_f16(rope_sin_dev_, sin_table.data(),
                   (size_t)MAX_SEQ * head_dim_);
    }

    k_cache_dev_.resize(n_layers_, nullptr);
    v_cache_dev_.resize(n_layers_, nullptr);
    for (int il = 0; il < n_layers_; ++il) {
        alloc_dev(&k_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * E);
        alloc_dev(&v_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * E);
    }
    kv_cache_len_ = 0;

    {
        float scale_val = 1.0f / sqrtf((float)head_dim_);
        attn_scale_     = g_cann.aclCreateScalar(&scale_val, ACL_FLOAT);
    }
    attn_scale_f16_ = make_f16_scalar(1.0f / sqrtf((float)head_dim_));
    one_scalar_     = make_f16_scalar(1.0f);

    // Skip the seed alloc if NZ pre-conversion already set it up.
    if (workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }
    nz_applied_ = nz_enabled;

    build_persistent_tensors_();

    ready_ = true;
    printf("[cp_cann] BF16-weights engine initialized from %s\n", path.c_str());
    printf("[cp_cann] dims: cp_hidden=%d q_dim=%d kv_dim=%d inter=%d head=%d\n",
           cp_hidden_, q_dim_, kv_dim_, inter_, head_dim_);
    return true;
}

// ============================================================================
// init — allocate F16 buffers, convert weights F32->F16 at upload.
// ============================================================================

bool CpCannEngine::init(const CpWeightsF32 &w, const CodePredictorConfig &cfg,
                        int device) {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[cp_cann] symbol load failed; engine disabled\n");
        return false;
    }

    device_ = device;
    ACL_CHECK_RET(g_cann.aclrtSetDevice(device_));
    // Primary + secondary streams (M6.1). Primary is the default target;
    // stream_b_ is spare for multi-stream pipelining.
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&primary_stream_));
    stream_ = primary_stream_;
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&stream_b_));
    // M6.2: reusable event recorded at the end of forward_one_token_launch.
    if (g_cann.aclrtCreateEvent && !forward_done_event_) {
        ACL_CHECK_RET(g_cann.aclrtCreateEvent(&forward_done_event_));
    }

    talker_hidden_ = cfg.talker_hidden_size;
    cp_hidden_     = cfg.hidden_size;
    n_heads_       = cfg.num_attention_heads;
    n_kv_          = cfg.num_key_value_heads;
    head_dim_      = cfg.head_dim;
    q_dim_         = n_heads_ * head_dim_;
    kv_dim_        = n_kv_ * head_dim_;
    inter_         = cfg.intermediate_size;
    n_layers_      = cfg.num_hidden_layers;
    eps_           = cfg.rms_norm_eps;
    rope_theta_    = cfg.rope_theta;

    // All device buffers are F16 (2 bytes/elem).
    const size_t E = sizeof(uint16_t);

    // --- Upload projection weights as F32 (matches llama.cpp's CP path,
    //    which does the input projection on CPU with F32 precision). This
    //    is the single most precision-sensitive matmul in the CP because it
    //    converts between talker-space (high variance) and cp-space. ---
    alloc_dev(&proj_w_dev_,
              (size_t)cp_hidden_ * talker_hidden_ * sizeof(float));
    upload(proj_w_dev_, w.proj_w.data(),
           (size_t)cp_hidden_ * talker_hidden_);
    alloc_dev(&proj_b_dev_, cp_hidden_ * sizeof(float));
    upload(proj_b_dev_, w.proj_b.data(), cp_hidden_);

    // ---- FRACTAL_NZ gating (M5.2) ------------------------------------------
    const bool nz_enabled = use_nz_weights_ && g_cann.has_nz();
    if (use_nz_weights_ && !g_cann.has_nz()) {
        printf("[cp_cann] NZ weights requested but "
               "aclnnTransMatmulWeight unresolved — falling back to ND\n");
    }
    // Seed workspace early so nz_convert_weight_ has scratch to grow from.
    if (nz_enabled && workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }

    layer_w_.resize(n_layers_);
    auto upload_mat_f16 = [&](void *&dev, const std::vector<float> &host,
                               int rows, int cols) {
        alloc_dev(&dev, (size_t)rows * cols * E);
        upload_f16(dev, host.data(), (size_t)rows * cols);
    };
    for (int il = 0; il < n_layers_; ++il) {
        const auto &src = w.layers[il];
        auto &dst = layer_w_[il];
        upload_mat_f16(dst.q_proj_w,    src.q_proj_w,    q_dim_,     cp_hidden_);
        upload_mat_f16(dst.k_proj_w,    src.k_proj_w,    kv_dim_,    cp_hidden_);
        upload_mat_f16(dst.v_proj_w,    src.v_proj_w,    kv_dim_,    cp_hidden_);
        upload_mat_f16(dst.o_proj_w,    src.o_proj_w,    cp_hidden_, q_dim_);
        upload_mat_f16(dst.gate_proj_w, src.gate_proj_w, inter_,     cp_hidden_);
        upload_mat_f16(dst.up_proj_w,   src.up_proj_w,   inter_,     cp_hidden_);
        upload_mat_f16(dst.down_proj_w, src.down_proj_w, cp_hidden_, inter_);
        if (nz_enabled) {
            // proj_w / proj_b are F32 (F32 input projection path); keep those
            // in ND since TransMatmulWeight doesn't support F32 weights.
            nz_convert_weight_(dst.q_proj_w,    q_dim_,     cp_hidden_);
            nz_convert_weight_(dst.k_proj_w,    kv_dim_,    cp_hidden_);
            nz_convert_weight_(dst.v_proj_w,    kv_dim_,    cp_hidden_);
            nz_convert_weight_(dst.o_proj_w,    cp_hidden_, q_dim_);
            nz_convert_weight_(dst.gate_proj_w, inter_,     cp_hidden_);
            nz_convert_weight_(dst.up_proj_w,   inter_,     cp_hidden_);
            nz_convert_weight_(dst.down_proj_w, cp_hidden_, inter_);
        }
        // RMSNorm gammas are stored as F32 in the GGUF and aclnnRmsNorm
        // natively accepts F32 gamma with F16 input (SupportInfo[0]).
        // Casting them to F16 loses precision for a 1-in-1024-sized vector
        // that gets applied 5+ times per decode — noticeable in output.
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

    // --- Fixed-size working buffers (F16) ---
    alloc_dev(&cur_dev_,       cp_hidden_ * E);
    alloc_dev(&residual_dev_,  cp_hidden_ * E);
    alloc_dev(&normed_dev_,    cp_hidden_ * E);
    alloc_dev(&q_dev_,         q_dim_     * E);  // also F16 input stage post-cast (q_dim ≥ talker_hidden)
    alloc_dev(&k_dev_,         kv_dim_    * E);
    alloc_dev(&v_dev_,         kv_dim_    * E);
    alloc_dev(&attn_out_dev_,  q_dim_     * E);
    alloc_dev(&o_out_dev_,     cp_hidden_ * E);
    alloc_dev(&gate_dev_,      inter_     * E);
    alloc_dev(&up_dev_,        inter_     * E);
    alloc_dev(&ffn_out_dev_,   cp_hidden_ * E);

    // Attention scores buffer — kept F32 so softmax can run in F32 (matches
    // the precision of llama.cpp's aclnnFusedInferAttentionScoreV2 internal
    // softmax, which the user-audible F16-throughout softmax can't replicate).
    alloc_dev(&scores_dev_, (size_t)n_heads_ * MAX_SEQ * sizeof(float));
    // Separate F16 scratch for the attn_weights BMM input (F16 required).
    alloc_dev(&scores_f16_dev_, (size_t)n_heads_ * MAX_SEQ * E);
    // rstd is F32 (aclnnRmsNorm spec); sized for the largest case (n_heads).
    alloc_dev(&rstd_dev_,   (size_t)n_heads_ * sizeof(float));

    // Boundary F32 staging buffers
    alloc_dev(&input_stage_f32_dev_,  talker_hidden_ * sizeof(float));
    alloc_dev(&output_stage_f32_dev_, cp_hidden_     * sizeof(float));
    // F32 scratch for the input-projection output AND the running F32
    // residual accumulator used through all transformer layers.
    alloc_dev(&proj_out_f32_dev_,     cp_hidden_     * sizeof(float));
    alloc_dev(&accum_scratch_f32_dev_, cp_hidden_    * sizeof(float));
    alloc_dev(&normed_f32_dev_,        cp_hidden_    * sizeof(float));

    // --- Precompute cos/sin tables as F16 ---
    {
        const int half = head_dim_ / 2;
        std::vector<float> cos_table((size_t)MAX_SEQ * head_dim_);
        std::vector<float> sin_table((size_t)MAX_SEQ * head_dim_);
        for (int p = 0; p < MAX_SEQ; ++p) {
            for (int j = 0; j < half; ++j) {
                float freq  = 1.0f / powf(rope_theta_, (float)(2 * j) / head_dim_);
                float angle = (float)p * freq;
                float c = cosf(angle);
                float s = sinf(angle);
                cos_table[(size_t)p * head_dim_ + j]        = c;
                cos_table[(size_t)p * head_dim_ + j + half] = c;
                sin_table[(size_t)p * head_dim_ + j]        = s;
                sin_table[(size_t)p * head_dim_ + j + half] = s;
            }
        }
        alloc_dev(&rope_cos_dev_, (size_t)MAX_SEQ * head_dim_ * E);
        alloc_dev(&rope_sin_dev_, (size_t)MAX_SEQ * head_dim_ * E);
        upload_f16(rope_cos_dev_, cos_table.data(), (size_t)MAX_SEQ * head_dim_);
        upload_f16(rope_sin_dev_, sin_table.data(), (size_t)MAX_SEQ * head_dim_);
    }

    // --- KV cache (F16) ---
    k_cache_dev_.resize(n_layers_, nullptr);
    v_cache_dev_.resize(n_layers_, nullptr);
    for (int il = 0; il < n_layers_; ++il) {
        alloc_dev(&k_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * E);
        alloc_dev(&v_cache_dev_[il], (size_t)MAX_SEQ * kv_dim_ * E);
    }
    kv_cache_len_ = 0;

    // --- Scalars: F32 attn_scale for the F32 softmax path; F16 one_scalar
    //    for Add/InplaceAdd on F16 tensors. ---
    {
        float scale_val = 1.0f / sqrtf((float)head_dim_);
        attn_scale_     = g_cann.aclCreateScalar(&scale_val, ACL_FLOAT);
    }
    attn_scale_f16_ = make_f16_scalar(1.0f / sqrtf((float)head_dim_));
    one_scalar_     = make_f16_scalar(1.0f);

    // --- Workspace seed ---
    // Skip if the NZ pre-conversion path already allocated and possibly
    // grew the workspace earlier; we don't want to leak that buffer.
    if (workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }
    nz_applied_ = nz_enabled;

    build_persistent_tensors_();

    ready_ = true;
    printf("[cp_cann] v4 F16 engine initialized: %d layers, device %d\n",
           n_layers_, device_);
    printf("[cp_cann] dims: cp_hidden=%d q_dim=%d kv_dim=%d inter=%d head=%d\n",
           cp_hidden_, q_dim_, kv_dim_, inter_, head_dim_);
    return true;
}

void CpCannEngine::reset_kv_cache() {
    kv_cache_len_ = 0;
}

// ============================================================================
// forward_one_token — F16 end-to-end with F32 staging at the boundaries.
// ============================================================================

// M6.2: launch path queues every op for position `pos` on `stream_` and
// returns without syncing or D2H. Records `forward_done_event_` so callers
// can fence from another stream (Talker[N+1] waiting on CP[N] final group).
void CpCannEngine::forward_one_token_launch(const float *input_talker_space,
                                             int pos,
                                             aclrtEvent wait_event) {
    assert(ready_);

    if (wait_event && g_cann.aclrtStreamWaitEvent) {
        ACL_CHECK_RET(g_cann.aclrtStreamWaitEvent(stream_, wait_event));
    }

    // 1. Upload F32 input embedding to f32 stage buffer.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, talker_hidden_ * sizeof(float),
        input_talker_space,   talker_hidden_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // 2. Input projection in F32 (proj_w/proj_b are F32; input is F32).
    //    Output goes to proj_out_f32_dev_ (F32). This matches the precision
    //    llama.cpp uses — cp_matvec_f32 on CPU — which is the only reference
    //    that produces audibly-correct output.
    {
        aclTensor *t_x = tensor_2d(input_stage_f32_dev_,
                                    talker_hidden_, 1, ACL_FLOAT);
        CANN_OP(Mm, t_.proj_w_f32, t_x, t_.proj_out_f32_col,
                /*cubeMathType=*/0);
        g_cann.aclDestroyTensor(t_x);
    }
    // Add proj_b in F32: F32 Add with alpha=1.
    {
        float one = 1.0f;
        aclScalar *alpha_f32 = g_cann.aclCreateScalar(&one, ACL_FLOAT);
        CANN_OP(InplaceAdd, t_.proj_out_f32_flat, t_.proj_b_f32, alpha_f32);
        g_cann.aclDestroyScalar(alpha_f32);
    }
    // Cast F32 projected -> F16 cur_dev_ so the first RmsNorm can run F16
    // with F32 gamma (matches ggml-cann's Qwen3 precision convention).
    // From this point on the transformer runs F16 end-to-end, residual adds
    // included — this is the llama.cpp/ggml-cann convention.
    CANN_OP(Cast, t_.proj_out_f32_flat, ACL_FLOAT16,
            t_.cur_f16_flat_as_target);

    const int seq_len = pos + 1;
    const int group   = n_heads_ / n_kv_;

    uint16_t *cos_pos = (uint16_t *)rope_cos_dev_ + (size_t)pos * head_dim_;
    uint16_t *sin_pos = (uint16_t *)rope_sin_dev_ + (size_t)pos * head_dim_;
    aclTensor *t_cos = [&]() {
        int64_t shape[4]   = {1, 1, 1, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)head_dim_, (int64_t)head_dim_,
                               (int64_t)head_dim_, 1};
        return tensor_strided(cos_pos, 4, shape, strides);
    }();
    aclTensor *t_sin = [&]() {
        int64_t shape[4]   = {1, 1, 1, (int64_t)head_dim_};
        int64_t strides[4] = {(int64_t)head_dim_, (int64_t)head_dim_,
                               (int64_t)head_dim_, 1};
        return tensor_strided(sin_pos, 4, shape, strides);
    }();

    // Hidden state lives in cur_dev_ (F16) throughout the transformer.
    // Residual saved via F16 d2d memcpy at the start of each sublayer, then
    // added back with aclnnAdd (F16) at the end. This is the ggml-cann/
    // llama.cpp convention — the model was trained with F16 residuals.
    const int layers_to_run = (active_layers_ > 0 && active_layers_ < n_layers_)
                               ? active_layers_ : n_layers_;
    for (int il = 0; il < layers_to_run; ++il) {
        const auto &lt = t_.layer[il];

        // residual = cur (F16 d2d async memcpy)
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_dev_, cp_hidden_ * sizeof(uint16_t),
            cur_dev_,      cp_hidden_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        CANN_OP(RmsNorm, t_.cur_row, lt.input_ln, (double)eps_,
                t_.normed_row, t_.rstd_11);

        CANN_OP(Mm, lt.q_proj, t_.normed_col, t_.q_col, /*cubeMathType=*/0);
        CANN_OP(Mm, lt.k_proj, t_.normed_col, t_.k_col, /*cubeMathType=*/0);
        CANN_OP(Mm, lt.v_proj, t_.normed_col, t_.v_col, /*cubeMathType=*/0);

        CANN_OP(RmsNorm, t_.q_heads, lt.q_norm, (double)eps_,
                t_.q_heads, t_.rstd_heads);
        CANN_OP(RmsNorm, t_.k_kv,    lt.k_norm, (double)eps_,
                t_.k_kv,    t_.rstd_kv);

        // RoPE (F16)
        CANN_OP(RotaryPositionEmbedding,
                t_.q_rope_4d, t_cos, t_sin,
                /*mode=*/(int64_t)0, t_.attn_out_4d);

        uint16_t *k_cache_slot =
            (uint16_t *)k_cache_dev_[il] + (size_t)pos * kv_dim_;
        aclTensor *t_k_rope_src = [&]() {
            int64_t shape[4]   = {1, 1, (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)kv_dim_, (int64_t)kv_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(k_dev_, 4, shape, strides);
        }();
        aclTensor *t_k_rope_dst = [&]() {
            int64_t shape[4]   = {1, 1, (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)kv_dim_, (int64_t)kv_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(k_cache_slot, 4, shape, strides);
        }();
        CANN_OP(RotaryPositionEmbedding,
                t_k_rope_src, t_cos, t_sin,
                /*mode=*/(int64_t)0, t_k_rope_dst);
        g_cann.aclDestroyTensor(t_k_rope_src);
        g_cann.aclDestroyTensor(t_k_rope_dst);

        // V -> cache slot
        uint16_t *v_cache_slot =
            (uint16_t *)v_cache_dev_[il] + (size_t)pos * kv_dim_;
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            v_cache_slot, kv_dim_ * sizeof(uint16_t),
            v_dev_,       kv_dim_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- Fused attention: aclnnFusedInferAttentionScoreV2. This is the
        // exact op llama.cpp's ggml-cann backend uses for Qwen3 attention.
        // By calling it here with the same params (layout="BSND",
        // innerPrecise=0 for S=1 decode, no mask), we inherit llama.cpp's
        // numerical behavior — same weights (F16 GGUF) + same op + same
        // params → same output. Prior manual BMM path diverged precisely
        // because it was a custom attention with subtly-different numerics.
        //
        // Input layouts (BSND with B=1):
        //   Q:   [1, 1, n_heads, head_dim]  over attn_out_dev_ (RoPE'd Q)
        //   K:   [1, seq_len, n_kv, head_dim] view of KV cache (native)
        //   V:   same as K, over v_cache_dev_
        //   Out: [1, 1, n_heads, head_dim]  into q_dev_
        aclTensor *t_q_bsnd = [&]() {
            int64_t shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(attn_out_dev_, 4, shape, strides,
                                   ACL_FLOAT16);
        }();
        aclTensor *t_k_bsnd = [&]() {
            int64_t shape[4]   = {1, (int64_t)seq_len, (int64_t)n_kv_,
                                   (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)seq_len * kv_dim_,
                                   (int64_t)kv_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(k_cache_dev_[il], 4, shape, strides,
                                   ACL_FLOAT16);
        }();
        aclTensor *t_v_bsnd = [&]() {
            int64_t shape[4]   = {1, (int64_t)seq_len, (int64_t)n_kv_,
                                   (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)seq_len * kv_dim_,
                                   (int64_t)kv_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(v_cache_dev_[il], 4, shape, strides,
                                   ACL_FLOAT16);
        }();
        aclTensor *t_attn_out_bsnd = [&]() {
            int64_t shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                   (int64_t)head_dim_, 1};
            return tensor_strided(q_dev_, 4, shape, strides, ACL_FLOAT16);
        }();

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
                /*innerPrecise*/ (int64_t)0,  // S=1 → high precision mode
                /*blockSize*/ (int64_t)0,
                /*antiquantMode*/ (int64_t)0,
                /*softmaxLseFlag*/ false,
                /*keyAntiquantMode*/ (int64_t)0,
                /*valueAntiquantMode*/ (int64_t)0,
                /*attentionOut*/ t_attn_out_bsnd,
                /*softmaxLse*/ nullptr,
                &fa_ws, &fa_exec));
            if (fa_ws > workspace_size_) {
                if (workspace_dev_) g_cann.aclrtFree(workspace_dev_);
                ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, fa_ws,
                               ACL_MEM_MALLOC_HUGE_FIRST));
                workspace_size_ = fa_ws;
            }
            void *ws = fa_ws > 0 ? workspace_dev_ : nullptr;
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2(
                ws, fa_ws, fa_exec, stream_));
        }

        g_cann.aclDestroyTensorList(t_k_list);
        g_cann.aclDestroyTensorList(t_v_list);
        g_cann.aclDestroyTensor(t_q_bsnd);
        g_cann.aclDestroyTensor(t_attn_out_bsnd);
        // t_k_bsnd / t_v_bsnd are owned by the destroyed lists — no free here.

        CANN_OP(Mm, lt.o_proj, t_.q_col, t_.o_out_col,
                /*cubeMathType=*/0);

        // cur = residual + o_out  (F16 add)
        CANN_OP(Add, t_.residual_flat, t_.o_out_flat, one_scalar_,
                t_.cur_flat);

        // residual = cur  (for FFN)
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            residual_dev_, cp_hidden_ * sizeof(uint16_t),
            cur_dev_,      cp_hidden_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        CANN_OP(RmsNorm, t_.cur_row, lt.post_ln, (double)eps_,
                t_.normed_row, t_.rstd_11);

        CANN_OP(Mm, lt.gate_proj, t_.normed_col, t_.gate_col,
                /*cubeMathType=*/0);
        CANN_OP(Mm, lt.up_proj,   t_.normed_col, t_.up_col,
                /*cubeMathType=*/0);
        CANN_OP(Silu, t_.gate_flat, t_.gate_flat);
        CANN_OP(InplaceMul, t_.gate_flat, t_.up_flat);
        CANN_OP(Mm, lt.down_proj, t_.gate_col, t_.ffn_out_col,
                /*cubeMathType=*/0);

        // cur = residual + ffn_out  (F16 add)
        CANN_OP(Add, t_.residual_flat, t_.ffn_out_flat, one_scalar_,
                t_.cur_flat);
    }

    // Final RmsNorm (F16 input/output, F32 gamma).
    CANN_OP(RmsNorm, t_.cur_row, t_.final_norm, (double)eps_,
            t_.normed_row, t_.rstd_11);

    // Cast F16 hidden -> F32 staging (queued async; D2H + sync moved to _fetch).
    CANN_OP(Cast, t_.output_f16, ACL_FLOAT, t_.output_f32);

    // Record event so `forward_one_token_fetch` (or another stream via
    // aclrtStreamWaitEvent) can fence against forward completion.
    if (forward_done_event_ && g_cann.aclrtRecordEvent) {
        ACL_CHECK_RET(g_cann.aclrtRecordEvent(forward_done_event_, stream_));
    }

    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);
}

void CpCannEngine::forward_one_token_fetch(float *hidden_out) {
    assert(ready_);
    if (forward_done_event_ && g_cann.aclrtSynchronizeEvent) {
        ACL_CHECK_RET(g_cann.aclrtSynchronizeEvent(forward_done_event_));
    } else {
        ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    }
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        hidden_out, cp_hidden_ * sizeof(float),
        output_stage_f32_dev_, cp_hidden_ * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

void CpCannEngine::forward_one_token(const float *input_talker_space,
                                      int pos, float *hidden_out) {
    forward_one_token_launch(input_talker_space, pos, nullptr);
    forward_one_token_fetch(hidden_out);
}
