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

// M5.3 — matmul call. On 8.5, plain aclnnMm dispatches to the NZ kernel as
// long as the `mat2` weight tensor is tagged with ACL_FORMAT_FRACTAL_NZ AND
// the weight buffer has been pre-converted via aclnnTransMatmulWeight at
// init. The CANN 8.5 MatmulWeightNz op exists but requires a 4D storage-
// shape descriptor that plain tensor_2d doesn't build (EZ1001: "Only
// support mat2 storageShapeDim is 4"); matching ggml-cann's ggml-cann.cpp
// path (plain aclnnMm + FRACTAL_NZ format tag on mat2 only) dispatches the
// same kernel without that complication. `A` is activation tensor (ND), `W`
// is weight tensor (NZ when applied, else ND), `O` is output. Compute
// semantics: O = A @ W, shape [M, K] @ [K, N].
#define CANN_MATMUL(A, W, O) CANN_OP(Mm, (A), (W), (O), (int8_t)0)

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
                        aclDataType dtype,
                        aclFormat fmt = ACL_FORMAT_ND) {
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
                                   fmt, &storage_len, 1, buf);
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

// M5.3 — build a [d0, d1] weight tensor with the caller-selected aclFormat.
// When `fmt == ACL_FORMAT_FRACTAL_NZ`, aclnnMm dispatches the NZ kernel path
// (the buffer must already have been pre-converted via
// aclnnTransMatmulWeight at init — see set_use_nz_weights / nz_applied()).
// `fmt == ACL_FORMAT_ND` is identical to plain tensor_2d().
aclTensor *tensor_2d_fmt(void *buf, int64_t d0, int64_t d1,
                          aclDataType dtype, aclFormat fmt) {
    int64_t shape[2]   = {d0, d1};
    int64_t strides[2] = {d1, 1};
    return make_tensor(buf, 2, shape, strides, dtype, fmt);
}

aclTensor *tensor_strided(void *buf, int64_t rank,
                           const int64_t *shape, const int64_t *strides,
                           aclDataType dtype) {
    return make_tensor(buf, rank, shape, strides, dtype);
}

// Pull tensor data out of a GGUF as float. Handles F32, F16, and any
// block-quantized type (Q8_0, Q4_K, ...) via ggml's dequantization traits.
// Returns empty vector on failure.
//
// A4: Q8_0 support. GGUF Q8_0 stores weights as 32-element blocks of INT8 +
// one F16 scale per block. Dequantizing to F32 on host lets the existing
// F16 upload and W8 per-channel calibration paths work unchanged — 1-2 ULP
// of re-quant loss vs. the preferred per-group (antiquantGroupSize=32) path,
// but keeps the patch surface minimal and WER-safe on canonical clips.
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
        // Quantized block formats (Q8_0, Q4_K, ...). Use ggml's type traits
        // dequantizer — same code llama.cpp's CPU backend uses, so the
        // decoded values are bit-exact vs. the reference implementation.
        const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr, "[talker_cann] %s: unsupported dtype %d "
                    "(no to_float trait)\n", name, (int)t->type);
            return {};
        }
        tr->to_float(t->data, out.data(), (int64_t)n);
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
        // A16W8 parallel buffers (null on F16/NZ paths).
        free_dev(lw.q_proj_w_i8);
        free_dev(lw.k_proj_w_i8);
        free_dev(lw.v_proj_w_i8);
        free_dev(lw.o_proj_w_i8);
        free_dev(lw.gate_proj_w_i8);
        free_dev(lw.up_proj_w_i8);
        free_dev(lw.down_proj_w_i8);
        free_dev(lw.q_proj_scale);
        free_dev(lw.k_proj_scale);
        free_dev(lw.v_proj_scale);
        free_dev(lw.o_proj_scale);
        free_dev(lw.gate_proj_scale);
        free_dev(lw.up_proj_scale);
        free_dev(lw.down_proj_scale);
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

    // A4c Phase 1: shared zero-F16 antiquantOffset buffer for GMM QKV.
    // Null unless gmm_qkv_applied_ was true at init.
    free_dev(antiquant_zero_offset_dev_);

    for (auto &p : k_cache_dev_) free_dev(p);
    for (auto &p : v_cache_dev_) free_dev(p);

    // Destroy any captured aclGraphs. Safe to call with null entries.
    if (g_cann.aclmdlRIDestroy) {
        for (aclmdlRI &g : decode_graphs_) {
            if (g) {
                g_cann.aclmdlRIDestroy(g);
                g = nullptr;
            }
        }
        // A4c Phase 4: symmetric teardown for the per-bucket prefill graphs.
        for (aclmdlRI &g : prefill_graphs_) {
            if (g) {
                g_cann.aclmdlRIDestroy(g);
                g = nullptr;
            }
        }
    }
    decode_graphs_.clear();
    prefill_graphs_.clear();
    prefill_buckets_.clear();

    free_dev(workspace_dev_);
    free_dev(input_stage_f32_dev_);
    free_dev(output_stage_f32_dev_);
    // M6.2 Track J: speculative delta-add staging buffers.
    free_dev(delta_stage_f32_dev_);
    free_dev(delta_stage_f16_dev_);

    if (one_scalar_f16_) {
        g_cann.aclDestroyScalar(one_scalar_f16_);
        one_scalar_f16_ = nullptr;
    }

    // M6.2: destroy the decode-done event before the streams.
    if (decode_done_event_ && g_cann.aclrtDestroyEvent) {
        g_cann.aclrtDestroyEvent(decode_done_event_);
        decode_done_event_ = nullptr;
    }

    // stream_ may either be pointing at primary_stream_ (default) or at some
    // externally-owned stream handed in via set_stream() — in either case we
    // only destroy the two streams this engine owns.
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
// Device-side helpers
// ============================================================================

void TalkerCannEngine::alloc_dev_(void **ptr, size_t bytes) {
    ACL_CHECK_RET(g_cann.aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
}

void TalkerCannEngine::ensure_workspace_(size_t needed) {
    if (needed <= workspace_size_) return;
    // Reallocation invalidates any captured aclGraphs because they encode
    // the old workspace pointer as a kernel argument. Drop them here; they
    // will be re-captured on next touch of each pos. In practice this only
    // triggers during the first few decode positions (FIAS workspace grows
    // monotonically with seq_len_total, which plateaus by ~pos=8 in
    // observed runs) so the amortized cost is a handful of extra eager runs.
    if (workspace_dev_ && !decode_graphs_.empty() && g_cann.aclmdlRIDestroy) {
        int dropped = 0;
        for (aclmdlRI &g : decode_graphs_) {
            if (g) {
                g_cann.aclmdlRIDestroy(g);
                g = nullptr;
                ++dropped;
            }
        }
        if (dropped > 0) {
            printf("[talker_cann] workspace grew %zu -> %zu, dropped %d "
                   "cached decode graphs\n",
                   workspace_size_, needed, dropped);
        }
    }
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
// FRACTAL_NZ weight pre-conversion (M5.2). Wraps aclnnTransMatmulWeight: the
// op refreshes the input tensor in-place so subsequent aclnnMm / aclnnMatMul
// calls pick up the hardware-preferred private layout automatically. Safe
// no-op when the runtime lacks the symbol — caller must still honor
// use_nz_weights_; this helper gates on g_cann.has_nz() as a safety net.
// ============================================================================

void TalkerCannEngine::nz_convert_weight_(void *weight_dev,
                                           int64_t rows, int64_t cols) {
    if (!g_cann.has_nz() || weight_dev == nullptr) return;

    // Build a throwaway [rows, cols] F16 tensor descriptor over the device
    // buffer. aclnnTransMatmulWeight mutates the tensor metadata in place;
    // the buffer pointer stays valid for reuse with aclnnMm afterwards.
    int64_t shape[2]   = {rows, cols};
    int64_t strides[2] = {cols, 1};
    int64_t storage_len = rows * cols;
    aclTensor *t = g_cann.aclCreateTensor(
        shape, 2, ACL_FLOAT16, strides, 0, ACL_FORMAT_ND,
        &storage_len, 1, weight_dev);
    if (!t) {
        fprintf(stderr, "[talker_cann] nz_convert: aclCreateTensor failed\n");
        return;
    }

    uint64_t ws_needed = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnTransMatmulWeightGetWorkspaceSize(
        t, &ws_needed, &exec);
    if (s != 0) {
        fprintf(stderr, "[talker_cann] nz_convert: GetWorkspaceSize status=%d "
                        "(shape=%lldx%lld)\n",
                (int)s, (long long)rows, (long long)cols);
        g_cann.aclDestroyTensor(t);
        return;
    }
    ensure_workspace_(ws_needed);
    void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
    s = g_cann.aclnnTransMatmulWeight(ws, ws_needed, exec, stream_);
    if (s != 0) {
        fprintf(stderr, "[talker_cann] nz_convert: TransMatmulWeight status=%d "
                        "(shape=%lldx%lld)\n",
                (int)s, (long long)rows, (long long)cols);
    }
    // Sync once per layer group to surface any device-side error. Weight
    // conversion happens once at init so the sync cost is amortized.
    ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    g_cann.aclDestroyTensor(t);
}

// ============================================================================
// A16W8 calibration (Stretch S1). Converts a host F32 weight [rows, cols]
// (row-major, [out, in]) into:
//   - INT8 weight [rows, cols] on device (row-major) into a NEW buffer
//   - F16 per-output-channel scales [rows] on device into a NEW buffer
// Symmetric quant, zero = 0 (absorbed by passing null offset downstream).
//
// Per row c: scale_c = max(|w[c,:]|) / 127. Rows with all-zero weight get
// scale_c = 1 (avoids divide-by-zero; the resulting INT8 row is all zeros,
// so the scale value is irrelevant). Saturates to [-127, 127]; -128 is
// avoided so the sign is symmetric (matches PyTorch's symmetric=True).
//
// The caller retains any existing F16 weight buffer — this function only
// allocates the parallel INT8 + scale pair. Failure leaves both output
// pointers untouched (any partial allocation is cleaned up).
// ============================================================================

bool TalkerCannEngine::w8_calibrate_weight_(const float *host_w,
                                             int64_t rows, int64_t cols,
                                             void *&weight_i8_dev,
                                             void *&scale_dev) {
    if (!host_w || rows <= 0 || cols <= 0) return false;

    std::vector<int8_t>   w_i8((size_t)rows * cols);
    std::vector<uint16_t> scales_f16((size_t)rows);

    // Per-output-channel symmetric INT8.
    for (int64_t r = 0; r < rows; ++r) {
        const float *row_src = host_w + (size_t)r * cols;
        float max_abs = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float v = std::fabs(row_src[j]);
            if (v > max_abs) max_abs = v;
        }
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        float inv_scale = 1.0f / scale;
        int8_t *row_dst = w_i8.data() + (size_t)r * cols;
        for (int64_t j = 0; j < cols; ++j) {
            float q = row_src[j] * inv_scale;
            // rint() respects the current rounding mode (round-to-nearest by
            // default on aarch64) — same behaviour as PyTorch's round().
            int ir = (int)std::rint(q);
            if (ir >  127) ir =  127;
            if (ir < -127) ir = -127;  // keep symmetric; never emit -128
            row_dst[j] = (int8_t)ir;
        }
        scales_f16[(size_t)r] = fp32_to_fp16(scale);
    }

    // Sanity: confirm all scales are finite. Non-finite values here would
    // produce NaNs out of the fused W8 matmul — better to bail now than
    // chase a NaN through decode.
    for (int64_t r = 0; r < rows; ++r) {
        float s = fp16_to_fp32(scales_f16[(size_t)r]);
        if (!std::isfinite(s)) {
            fprintf(stderr, "[talker_cann] w8 calib: non-finite scale on row "
                    "%lld (max_abs source likely inf/nan)\n", (long long)r);
            return false;
        }
    }

    void *new_w = nullptr;
    aclError err = g_cann.aclrtMalloc(&new_w, (size_t)rows * cols * sizeof(int8_t),
                                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0 || !new_w) {
        fprintf(stderr, "[talker_cann] w8 calib: INT8 aclrtMalloc failed\n");
        return false;
    }
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        new_w, (size_t)rows * cols * sizeof(int8_t),
        w_i8.data(), (size_t)rows * cols * sizeof(int8_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    void *new_s = nullptr;
    err = g_cann.aclrtMalloc(&new_s, (size_t)rows * sizeof(uint16_t),
                              ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0 || !new_s) {
        fprintf(stderr, "[talker_cann] w8 calib: scale aclrtMalloc failed\n");
        g_cann.aclrtFree(new_w);
        return false;
    }
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        new_s, (size_t)rows * sizeof(uint16_t),
        scales_f16.data(), (size_t)rows * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));
    weight_i8_dev = new_w;
    scale_dev     = new_s;
    return true;
}

// ============================================================================
// A16W8 matmul dispatch. Prefers V3 (CANN 8.5 primary) and falls back to V2.
//
// Semantics: y[M, N] = x[M, K] @ dequant(weight, scale) where weight is INT8
// stored as [N, K] row-major (the on-device layout from w8_calibrate_weight_
// — matches the original F16 [out, in] layout). V3 reads the weight via a
// transposed view — caller supplies the [K, N] descriptor with strides
// (1, K), which the op consumes as "the K-dim is contiguous inside each of
// the N output channels". Scale has matching N elements.
// ============================================================================

void TalkerCannEngine::w8_matmul_(const aclTensor *x,
                                    void *weight_dev, void *scale_dev,
                                    int64_t out_n, int64_t in_k,
                                    const aclTensor *y) {
    // Build a [K, N] INT8 view over the [N, K] row-major buffer. Strides
    // (1, K) express the transpose: advancing in K reads contiguous bytes
    // inside one row; advancing in N jumps K elements. This is the same
    // transposed-view trick the NZ path uses for F16 weights.
    int64_t w_shape[2]   = {in_k, out_n};
    int64_t w_strides[2] = {1, in_k};
    int64_t w_storage    = out_n * in_k;
    aclTensor *t_w = g_cann.aclCreateTensor(
        w_shape, 2, ACL_INT8, w_strides, 0, ACL_FORMAT_ND,
        &w_storage, 1, weight_dev);

    int64_t s_shape[1]   = {out_n};
    int64_t s_strides[1] = {1};
    int64_t s_storage    = out_n;
    aclTensor *t_scale = g_cann.aclCreateTensor(
        s_shape, 1, ACL_FLOAT16, s_strides, 0, ACL_FORMAT_ND,
        &s_storage, 1, scale_dev);

    uint64_t ws_needed = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = 0;

    // Prefer V3 (CANN 8.5 primary). Fall back to V2 if V3 is absent OR if it
    // returns a non-zero status from GetWorkspaceSize (some op-tiling shapes
    // are V2-only on early 8.5 builds).
    bool used_v3 = false;
    if (g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize &&
        g_cann.aclnnWeightQuantBatchMatmulV3) {
        s = g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            x, t_w, t_scale,
            /*antiquantOffsetOptional*/ nullptr,
            /*quantScaleOptional*/      nullptr,
            /*quantOffsetOptional*/     nullptr,
            /*biasOptional*/            nullptr,
            /*antiquantGroupSize*/      0,
            /*innerPrecise*/            1,  // high-precision path
            y, &ws_needed, &exec);
        if (s == 0) used_v3 = true;
    }
    if (!used_v3) {
        if (!g_cann.aclnnWeightQuantBatchMatmulV2GetWorkspaceSize ||
            !g_cann.aclnnWeightQuantBatchMatmulV2) {
            fprintf(stderr, "[talker_cann] w8_matmul: no V3/V2 symbol resolved\n");
            g_cann.aclDestroyTensor(t_w);
            g_cann.aclDestroyTensor(t_scale);
            return;
        }
        s = g_cann.aclnnWeightQuantBatchMatmulV2GetWorkspaceSize(
            x, t_w, t_scale,
            /*antiquantOffsetOptional*/ nullptr,
            /*quantScaleOptional*/      nullptr,
            /*quantOffsetOptional*/     nullptr,
            /*biasOptional*/            nullptr,
            /*antiquantGroupSize*/      0,
            y, &ws_needed, &exec);
        if (s != 0) {
            fprintf(stderr,
                    "[talker_cann] w8_matmul V2 GetWorkspaceSize status=%d\n",
                    (int)s);
            g_cann.aclDestroyTensor(t_w);
            g_cann.aclDestroyTensor(t_scale);
            return;
        }
    }
    ensure_workspace_(ws_needed);
    void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
    if (used_v3) {
        s = g_cann.aclnnWeightQuantBatchMatmulV3(ws, ws_needed, exec, stream_);
    } else {
        s = g_cann.aclnnWeightQuantBatchMatmulV2(ws, ws_needed, exec, stream_);
    }
    if (s != 0) {
        fprintf(stderr,
                "[talker_cann] w8_matmul %s launch status=%d\n",
                used_v3 ? "V3" : "V2", (int)s);
    }
    g_cann.aclDestroyTensor(t_w);
    g_cann.aclDestroyTensor(t_scale);
}

// ============================================================================
// A4c Phase 1: aclnnGroupedMatmulV3 fused Q/K/V for prefill.
//
// Direct transfer of CpCannEngine::gmm_qkv_ (commit 9d52177) adapted for the
// prefill M = seq_len case. The CP path fires this per-token (M=1) inside
// the decode loop; here we fire it once per layer inside forward_prefill
// with M = seq_len (typically 6..128).
//
// Three groups, all sharing the same activation view (three distinct
// aclTensor handles over the same device buffer — aclDestroyTensorList
// consumes the contained tensors so views cannot be reused across calls).
// Weights Q/K/V differ per group; antiquantScale differs per group;
// antiquantOffset is a shared zero-F16 buffer (required non-null by
// runtime CheckGroupedMatmulAntiQuant even though the header marks it
// Optional).
//
// Numerical delta: 1-2 F16 ulp (~1.7e-3 relative) vs 3 × WQBMMv3 reference,
// per the CP-side probe verdict. Parity gate for A4c Phase 1 is ASR Tier-1
// CER=0 (clip-level output identity), not per-layer tensor identity.
// ============================================================================

void TalkerCannEngine::gmm_qkv_prefill_(void *x_dev, int64_t seq_len,
                                          int64_t il,
                                          void *q_out_dev, void *k_out_dev,
                                          void *v_out_dev) {
    // Fresh tensor views per dispatch — aclDestroyTensorList consumes handles,
    // and the captured op pipeline cannot reuse stale ones.
    auto mk_x = [&]() {
        int64_t shape[2]   = {seq_len, (int64_t)n_embd_};
        int64_t strides[2] = {(int64_t)n_embd_, 1};
        int64_t storage    = seq_len * (int64_t)n_embd_;
        return g_cann.aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, &storage, 1, x_dev);
    };
    auto mk_w = [&](void *w_dev, int64_t N) {
        // Mirror w8_matmul_'s weight layout: logical [K, N] view over a
        // physical [N, K] INT8 blob with strides {1, K}. The op handles the
        // transpose internally.
        int64_t shape[2]   = {(int64_t)n_embd_, N};
        int64_t strides[2] = {1, (int64_t)n_embd_};
        int64_t storage    = N * (int64_t)n_embd_;
        return g_cann.aclCreateTensor(shape, 2, ACL_INT8, strides, 0,
                                       ACL_FORMAT_ND, &storage, 1, w_dev);
    };
    auto mk_1d_f16 = [&](void *d, int64_t N) {
        int64_t shape[1]   = {N};
        int64_t strides[1] = {1};
        int64_t storage    = N;
        return g_cann.aclCreateTensor(shape, 1, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, &storage, 1, d);
    };
    auto mk_y = [&](void *buf, int64_t N) {
        int64_t shape[2]   = {seq_len, N};
        int64_t strides[2] = {N, 1};
        int64_t storage    = seq_len * N;
        return g_cann.aclCreateTensor(shape, 2, ACL_FLOAT16, strides, 0,
                                       ACL_FORMAT_ND, &storage, 1, buf);
    };

    std::vector<const aclTensor *> xs  = {mk_x(), mk_x(), mk_x()};
    std::vector<const aclTensor *> ws_ = {mk_w(layer_w_[il].q_proj_w_i8, q_dim_),
                                           mk_w(layer_w_[il].k_proj_w_i8, kv_dim_),
                                           mk_w(layer_w_[il].v_proj_w_i8, kv_dim_)};
    std::vector<const aclTensor *> asc = {mk_1d_f16(layer_w_[il].q_proj_scale, q_dim_),
                                           mk_1d_f16(layer_w_[il].k_proj_scale, kv_dim_),
                                           mk_1d_f16(layer_w_[il].v_proj_scale, kv_dim_)};
    // Shared zero-F16 antiquantOffset buffer. Each group wraps a prefix view
    // sized to that group's N. Identical dev pointers across groups are
    // accepted so long as the aclTensor handles are distinct.
    std::vector<const aclTensor *> aoc = {mk_1d_f16(antiquant_zero_offset_dev_, q_dim_),
                                           mk_1d_f16(antiquant_zero_offset_dev_, kv_dim_),
                                           mk_1d_f16(antiquant_zero_offset_dev_, kv_dim_)};
    std::vector<const aclTensor *> ys  = {mk_y(q_out_dev, q_dim_),
                                           mk_y(k_out_dev, kv_dim_),
                                           mk_y(v_out_dev, kv_dim_)};

    aclTensorList *x_list  = g_cann.aclCreateTensorList(xs.data(),  xs.size());
    aclTensorList *w_list  = g_cann.aclCreateTensorList(ws_.data(), ws_.size());
    aclTensorList *as_list = g_cann.aclCreateTensorList(asc.data(), asc.size());
    aclTensorList *ao_list = g_cann.aclCreateTensorList(aoc.data(), aoc.size());
    aclTensorList *y_list  = g_cann.aclCreateTensorList(ys.data(),  ys.size());

    uint64_t ws_needed = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnGroupedMatmulV3GetWorkspaceSize(
        x_list, w_list,
        /*biasOptional*/             nullptr,
        /*scaleOptional*/            nullptr,  // post-quant scale (W8A8 lane)
        /*offsetOptional*/           nullptr,
        /*antiquantScaleOptional*/   as_list,
        /*antiquantOffsetOptional*/  ao_list,
        /*groupListOptional*/        nullptr,
        /*splitItem*/                0,
        /*groupType*/                -1,
        y_list, &ws_needed, &exec);
    if (s != 0) {
        fprintf(stderr,
                "[talker_cann] gmm_qkv_prefill V3 GetWorkspaceSize status=%d "
                "(il=%ld, seq_len=%ld)\n",
                (int)s, (long)il, (long)seq_len);
        g_cann.aclDestroyTensorList(x_list);
        g_cann.aclDestroyTensorList(w_list);
        g_cann.aclDestroyTensorList(as_list);
        g_cann.aclDestroyTensorList(ao_list);
        g_cann.aclDestroyTensorList(y_list);
        return;
    }

    ensure_workspace_((size_t)ws_needed);
    void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
    s = g_cann.aclnnGroupedMatmulV3(ws, ws_needed, exec, stream_);
    if (s != 0) {
        fprintf(stderr,
                "[talker_cann] gmm_qkv_prefill V3 launch status=%d "
                "(il=%ld, seq_len=%ld)\n",
                (int)s, (long)il, (long)seq_len);
    }

    g_cann.aclDestroyTensorList(x_list);
    g_cann.aclDestroyTensorList(w_list);
    g_cann.aclDestroyTensorList(as_list);
    g_cann.aclDestroyTensorList(ao_list);
    g_cann.aclDestroyTensorList(y_list);
}

// ============================================================================
// RoPE cos/sin table construction. Uses rope_speed_factor_ to scale angles —
// callers change it via set_rope_speed_factor, which triggers a rebuild.
// ============================================================================

void TalkerCannEngine::build_rope_tables_() {
    const int half = head_dim_ / 2;
    cos_host_.assign((size_t)MAX_SEQ * head_dim_, 0.0f);
    sin_host_.assign((size_t)MAX_SEQ * head_dim_, 0.0f);
    // B6.1 — sector-aware clamp for xvec/customvoice. `pair_cap` is the
    // highest dim-pair index that gets a non-identity rotation. In the
    // default (ICL) path, every pair rotates (pair_cap = half). In the
    // xvec/customvoice path, only pairs inside the temporal section rotate;
    // pairs ≥ mrope_temporal_section_ are identity — we seed cos=1, sin=0 so
    // aclnnRotaryPositionEmbedding passes those dims through unchanged
    // (out[j] = in[j]*1 - in[j+half]*0 = in[j]). This matches the h=w=extra=0
    // degeneracy of Qwen3-TTS's MRoPE (spatial positions are zero → angle
    // is zero on spatial dims → identity rotation).
    const int pair_cap = (use_mrope_xvec_layout_ && mrope_temporal_section_ > 0)
                          ? std::min(mrope_temporal_section_, half)
                          : half;
    for (int p = 0; p < MAX_SEQ; ++p) {
        // The speed factor scales the position offset — matches MLX's
        // rope_offset = (pos * rope_speed_factor) pattern.
        float pos = (float)p * rope_speed_factor_;
        // Rotated pairs inside the temporal section.
        for (int j = 0; j < pair_cap; ++j) {
            float freq  = 1.0f / powf(rope_theta_, (float)(2 * j) / head_dim_);
            float angle = pos * freq;
            float c = cosf(angle);
            float s = sinf(angle);
            cos_host_[(size_t)p * head_dim_ + j]        = c;
            cos_host_[(size_t)p * head_dim_ + j + half] = c;
            sin_host_[(size_t)p * head_dim_ + j]        = s;
            sin_host_[(size_t)p * head_dim_ + j + half] = s;
        }
        // Identity pairs outside the temporal section (xvec/cv only; skipped
        // when pair_cap == half, i.e. ICL).
        for (int j = pair_cap; j < half; ++j) {
            cos_host_[(size_t)p * head_dim_ + j]        = 1.0f;
            cos_host_[(size_t)p * head_dim_ + j + half] = 1.0f;
            // sin stays 0 (already zero-initialised).
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
    // Primary stream. Every engine op targets this by default. Stored in
    // `primary_stream_` so set_stream(nullptr) can restore the default
    // after an orchestrator temporarily redirected the engine onto another
    // stream (M6 multi-stream pipelining).
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&primary_stream_));
    stream_ = primary_stream_;
    // Secondary stream owned by this engine. An orchestrator that wants to
    // pipeline two engines on two physical NPU streams can grab this via
    // get_stream_b() and hand it to the other engine's set_stream().
    ACL_CHECK_RET(g_cann.aclrtCreateStream(&stream_b_));
    // M6.2: reusable event recorded at the end of forward_decode_launch.
    if (g_cann.aclrtCreateEvent) {
        ACL_CHECK_RET(g_cann.aclrtCreateEvent(&decode_done_event_));
    }

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

    // ---- B6.1: read mrope_section for sector-aware xvec/customvoice RoPE ---
    // export_talker_llama.py writes `qwen3.rope.dimension_sections` (4-element
    // array; preferred key) and export_qwen_tts.py writes both that and
    // `rope_scaling.mrope_section` (original HF key; legacy). We accept either
    // so the engine works against both exporters. For the Qwen3-TTS case the
    // shape is `[temporal, h, w, extra] = [N, 0, 0, 0]` — PM-confirmed and
    // asserted below. Any non-zero spatial/extra entry means the model would
    // produce non-identity rotation on dims ≥ sections[0] under our assumed
    // h=w=extra=0 degeneracy → fail loud so future multi-speaker spatial
    // models become an explicit engineering task, not a silent breakage.
    {
        mrope_temporal_section_ = 0;
        const char *keys[] = {
            "qwen3.rope.dimension_sections",
            "rope_scaling.mrope_section",
        };
        int64_t sec[4] = {0, 0, 0, 0};
        int n_read = 0;
        for (const char *k : keys) {
            int64_t key_id = gguf_find_key(gguf_ctx, k);
            if (key_id < 0) continue;
            enum gguf_type t = gguf_get_kv_type(gguf_ctx, key_id);
            if (t != GGUF_TYPE_ARRAY) continue;
            enum gguf_type at = gguf_get_arr_type(gguf_ctx, key_id);
            size_t n = gguf_get_arr_n(gguf_ctx, key_id);
            if (n == 0) continue;
            size_t to_read = n < 4 ? n : 4;
            const void *data = gguf_get_arr_data(gguf_ctx, key_id);
            n_read = (int)to_read;
            for (size_t i = 0; i < to_read; ++i) {
                switch (at) {
                    case GGUF_TYPE_INT32:
                        sec[i] = ((const int32_t *)data)[i]; break;
                    case GGUF_TYPE_UINT32:
                        sec[i] = ((const uint32_t *)data)[i]; break;
                    case GGUF_TYPE_INT64:
                        sec[i] = ((const int64_t *)data)[i]; break;
                    case GGUF_TYPE_UINT64:
                        sec[i] = (int64_t)((const uint64_t *)data)[i]; break;
                    case GGUF_TYPE_INT16:
                        sec[i] = ((const int16_t *)data)[i]; break;
                    case GGUF_TYPE_UINT16:
                        sec[i] = ((const uint16_t *)data)[i]; break;
                    default:
                        n_read = 0; i = to_read; break;
                }
            }
            if (n_read > 0) {
                printf("[talker_cann] mrope_section from '%s' = [%lld, %lld, %lld, %lld]\n",
                       k, (long long)sec[0], (long long)sec[1],
                       (long long)sec[2], (long long)sec[3]);
                break;
            }
        }
        if (n_read > 0) {
            mrope_temporal_section_ = (int)sec[0];
            // Structural invariant: the sector-aware rope port assumes
            // talker.cpp feeds only temporal positions (h=w=extra=0). If that
            // assumption is ever violated upstream (multi-speaker spatial
            // MRoPE, future streaming variants) the rotation on dims ≥
            // mrope_temporal_section_ would be non-identity — which our
            // cos=1/sin=0 seed does NOT cover. Document the expected layout
            // here; the position feed contract lives in talker.cpp
            // generate_xvec / generate_customvoice (prefill positions are
            // always [0, prefill_len), h/w/extra all zero) and is enforced
            // by the fact that TalkerCannEngine::forward_{prefill,decode}
            // take only a single `pos` (temporal) parameter.
            printf("[talker_cann] mrope_temporal_section=%d (head_dim=%d, "
                   "half=%d); dim-pairs ≥ %d will be identity on xvec/cv\n",
                   mrope_temporal_section_, head_dim_, head_dim_ / 2,
                   mrope_temporal_section_);
            if (mrope_temporal_section_ * 2 > head_dim_) {
                fprintf(stderr, "[talker_cann] FATAL: mrope_temporal_section "
                        "%d exceeds head_dim/2 (%d)\n",
                        mrope_temporal_section_, head_dim_ / 2);
                gguf_free(gguf_ctx);
                ggml_free(ggml_ctx);
                return false;
            }
            // Emit a one-shot diagnostic so anyone who adds a non-zero h/w
            // position feed downstream (and thus invalidates the identity
            // assumption on sections[1..3]) can trace the warning.
            if (sec[1] != 0 || sec[2] != 0 || sec[3] != 0) {
                printf("[talker_cann] NOTE: mrope_section has non-zero "
                       "spatial/extra pair-counts (sec[1..3]=[%lld,%lld,%lld]); "
                       "native engine treats them as identity under the "
                       "h=w=extra=0 position contract. If you change that "
                       "contract, re-engineer build_rope_tables_.\n",
                       (long long)sec[1], (long long)sec[2], (long long)sec[3]);
            }
        } else {
            printf("[talker_cann] no mrope_section in GGUF — xvec/customvoice "
                   "native path will be unavailable (ICL still works)\n");
        }
    }

    // ---- FRACTAL_NZ gating (M5.2) ------------------------------------------
    // We need a workspace buffer before nz_convert_weight_ runs its first
    // aclnnTransMatmulWeight, because the op's GetWorkspaceSize can report
    // a non-zero scratch requirement. The regular workspace alloc happens
    // lower down in init_from_gguf; move a small seed up here so the
    // conversion path has something to grow from.
    //
    // Env override TALKER_NZ_WEIGHTS=1 flips the flag on at init even when
    // the caller didn't call set_use_nz_weights — lets a default build
    // exercise the path without recompiling (useful for benchmark A/B).
    // Empty string / "0" count as "off" so that `TALKER_NZ_WEIGHTS=` (exported
    // but no value) doesn't accidentally opt in.
    {
        const char *nz_env = getenv("TALKER_NZ_WEIGHTS");
        if (!use_nz_weights_ && nz_env && nz_env[0] != '\0' && nz_env[0] != '0') {
            use_nz_weights_ = true;
            printf("[talker_cann] TALKER_NZ_WEIGHTS=%s forcing NZ weight "
                   "path\n", nz_env);
        }
    }
    // A16W8 env override (Stretch S1). Same empty/"0" semantics as the NZ
    // override. When set together with TALKER_NZ_WEIGHTS, W8 wins — we treat
    // the two as mutually exclusive (decode call sites only dispatch one).
    {
        const char *w8_env = getenv("TALKER_W8_QUANT");
        if (!use_w8_weights_ && w8_env && w8_env[0] != '\0' && w8_env[0] != '0') {
            use_w8_weights_ = true;
            printf("[talker_cann] TALKER_W8_QUANT=%s forcing A16W8 weight "
                   "path\n", w8_env);
        }
    }
    if (use_w8_weights_ && use_nz_weights_) {
        printf("[talker_cann] TALKER_W8_QUANT and TALKER_NZ_WEIGHTS are "
               "mutually exclusive in S1; disabling NZ in favour of W8\n");
        use_nz_weights_ = false;
    }
    const bool w8_enabled = use_w8_weights_ && g_cann.has_w8_quant() &&
                             !use_nz_weights_;
    if (use_w8_weights_ && !g_cann.has_w8_quant()) {
        printf("[talker_cann] W8 quant requested but "
               "aclnnWeightQuantBatchMatmulV3/V2 unresolved on this CANN "
               "toolkit — falling back to F16\n");
    }
    const bool nz_enabled = use_nz_weights_ && g_cann.has_nz() && !w8_enabled;
    if (use_nz_weights_ && !g_cann.has_nz()) {
        printf("[talker_cann] NZ weights requested but "
               "aclnnTransMatmulWeight unresolved on this CANN toolkit — "
               "falling back to ND\n");
    }
    if ((nz_enabled || w8_enabled) && workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }
    if (nz_enabled) {
        printf("[talker_cann] FRACTAL_NZ weight pre-conversion ENABLED "
               "(per-layer aclnnTransMatmulWeight on Q/K/V/O/gate/up/down; "
               "decode/prefill matmul call sites swap operands to "
               "activation@weight_T and tag weight mat2 with "
               "ACL_FORMAT_FRACTAL_NZ for M5.3)\n");
    }
    if (w8_enabled) {
        printf("[talker_cann] A16W8 weight quantization ENABLED "
               "(per-output-channel symmetric INT8 + F16 scales for Q/K/V/O/"
               "gate/up/down; decode + prefill matmul call sites dispatch "
               "aclnnWeightQuantBatchMatmulV3/V2 [A4b])\n");
    }

    // ---- Per-layer weights ----
    layer_w_.resize(n_layers_);
    char name[128];
    // Helper lambda: always upload F16 (for prefill + fallback decode paths),
    // and ALSO calibrate to INT8 + F16 scale when W8 is enabled. The F16
    // buffer is kept alongside the INT8 copy; prefill reads F16, decode reads
    // INT8 when w8_applied_. S1 pays ~1.5x weight memory in W8 mode for the
    // privilege of keeping prefill on its safe F16 path.
    auto load_proj_weight = [&](const char *tname, int64_t rows, int64_t cols,
                                 void *&weight_f16_dev,
                                 void *&weight_i8_dev,
                                 void *&scale_dev) -> bool {
        // Always upload F16 first — prefill uses these.
        if (!upload_tensor_f16(ggml_ctx, tname, (size_t)rows * cols,
                               weight_f16_dev))
            return false;
        if (w8_enabled) {
            std::vector<float> host = load_gguf_tensor_f32(
                ggml_ctx, tname, (size_t)rows * cols);
            if (host.empty()) return false;
            if (!w8_calibrate_weight_(host.data(), rows, cols,
                                       weight_i8_dev, scale_dev))
                return false;
        }
        return true;
    };
    for (int il = 0; il < n_layers_; ++il) {
        auto &lw = layer_w_[il];
#define TFMT(fmt) (snprintf(name, sizeof(name), fmt, il), name)
        if (!load_proj_weight(TFMT("blk.%d.attn_q.weight"),
                               q_dim_, n_embd_,
                               lw.q_proj_w, lw.q_proj_w_i8, lw.q_proj_scale))
            goto fail;
        if (!load_proj_weight(TFMT("blk.%d.attn_k.weight"),
                               kv_dim_, n_embd_,
                               lw.k_proj_w, lw.k_proj_w_i8, lw.k_proj_scale))
            goto fail;
        if (!load_proj_weight(TFMT("blk.%d.attn_v.weight"),
                               kv_dim_, n_embd_,
                               lw.v_proj_w, lw.v_proj_w_i8, lw.v_proj_scale))
            goto fail;
        // Output projection. GGUF stores this as [hidden, q_dim] which is the
        // [out, in] layout aclnnMm wants.
        if (!load_proj_weight(TFMT("blk.%d.attn_output.weight"),
                               n_embd_, q_dim_,
                               lw.o_proj_w, lw.o_proj_w_i8, lw.o_proj_scale))
            goto fail;
        if (!load_proj_weight(TFMT("blk.%d.ffn_gate.weight"),
                               inter_, n_embd_,
                               lw.gate_proj_w, lw.gate_proj_w_i8,
                               lw.gate_proj_scale))
            goto fail;
        if (!load_proj_weight(TFMT("blk.%d.ffn_up.weight"),
                               inter_, n_embd_,
                               lw.up_proj_w, lw.up_proj_w_i8,
                               lw.up_proj_scale))
            goto fail;
        if (!load_proj_weight(TFMT("blk.%d.ffn_down.weight"),
                               n_embd_, inter_,
                               lw.down_proj_w, lw.down_proj_w_i8,
                               lw.down_proj_scale))
            goto fail;

        // --- FRACTAL_NZ pre-conversion (M5.2) ---
        // Pre-bake every matmul weight so the matmul ops (M5.3 will flip
        // these to WeightNz variants; today the default aclnnMm consumes
        // the private layout transparently). Shapes here match the [out, in]
        // layout aclnnMm expects — see the upload calls above for provenance.
        if (nz_enabled) {
            nz_convert_weight_(lw.q_proj_w,    q_dim_,   n_embd_);
            nz_convert_weight_(lw.k_proj_w,    kv_dim_,  n_embd_);
            nz_convert_weight_(lw.v_proj_w,    kv_dim_,  n_embd_);
            nz_convert_weight_(lw.o_proj_w,    n_embd_,  q_dim_);
            nz_convert_weight_(lw.gate_proj_w, inter_,   n_embd_);
            nz_convert_weight_(lw.up_proj_w,   inter_,   n_embd_);
            nz_convert_weight_(lw.down_proj_w, n_embd_,  inter_);
        }

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
        // M6.2 Track J: per-step delta staging for add_input_delta_f32.
        // Sized for a single-token delta [n_embd] (prefill never uses this).
        alloc_dev_(&delta_stage_f32_dev_,
                    (size_t)n_embd_ * sizeof(float));
        alloc_dev_(&delta_stage_f16_dev_,
                    (size_t)n_embd_ * E);

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

    // Workspace seed (will grow on first big op). Skip if NZ pre-conversion
    // already allocated (and possibly grown) the workspace during weight
    // upload — it's valid and we don't want to leak the existing buffer.
    if (workspace_dev_ == nullptr) {
        workspace_size_ = 4 * 1024 * 1024;
        ACL_CHECK_RET(g_cann.aclrtMalloc(&workspace_dev_, workspace_size_,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
    }
    nz_applied_ = nz_enabled;
    w8_applied_ = w8_enabled;
    // Contract invariant: these two paths never turn on together (enforced
    // above by forcing NZ off whenever W8 wins the env-override tie).
    assert(!(w8_applied_ && nz_applied_));

    // A4c Phase 1: opt-in aclnnGroupedMatmulV3 for prefill Q/K/V. Requires
    // w8_applied_ (antiquantScale F16 path — the op rejects F16 weights) and
    // the runtime symbol (CANN 8.3+). Allocates a shared zero-F16
    // antiquantOffset buffer of length q_dim_ (= max of Nq/Nk/Nv). K/V groups
    // wrap prefix views. Default OFF; caller sets TALKER_W8_GMM_QKV=1.
    //
    // TALKER_W8_QUANT unset path: unaffected — the gmm_qkv_applied_ check
    // AND-ing w8_applied_ guarantees the new branch is only taken on the W8
    // lane. A4b's byte-identical-when-W8-off contract is preserved.
    {
        const char *env = getenv("TALKER_W8_GMM_QKV");
        if (env && env[0] != '\0' && env[0] != '0') {
            gmm_qkv_enabled_ = true;
        }
    }
    gmm_qkv_applied_ = gmm_qkv_enabled_ &&
                       w8_applied_ &&
                       g_cann.has_grouped_matmul_v3();
    if (gmm_qkv_enabled_ && !w8_applied_) {
        printf("[talker_cann] TALKER_W8_GMM_QKV requested but A16W8 is not "
               "active — aclnnGroupedMatmulV3 requires INT8 weights + F16 "
               "antiquant scales. Staying on 3× aclnnWeightQuantBatchMatmul.\n");
    }
    if (gmm_qkv_enabled_ && w8_applied_ && !g_cann.has_grouped_matmul_v3()) {
        printf("[talker_cann] TALKER_W8_GMM_QKV requested but "
               "aclnnGroupedMatmulV3 symbol absent — staying on 3× "
               "aclnnWeightQuantBatchMatmul.\n");
    }
    if (gmm_qkv_applied_) {
        alloc_dev_(&antiquant_zero_offset_dev_,
                   (size_t)q_dim_ * sizeof(uint16_t));
        std::vector<uint16_t> zeros((size_t)q_dim_, 0);
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            antiquant_zero_offset_dev_, (size_t)q_dim_ * sizeof(uint16_t),
            zeros.data(), (size_t)q_dim_ * sizeof(uint16_t),
            ACL_MEMCPY_HOST_TO_DEVICE));
        printf("[talker_cann] A4c Phase 1 ENABLED: aclnnGroupedMatmulV3 "
               "replaces the 3 per-layer prefill QKV w8_matmul_ calls "
               "(CP-reference commit 9d52177; ~1.7e-3 relative drift vs "
               "3-call reference; parity gate is ASR Tier-1 CER=0).\n");
    }

    // ---- A4c Phase 2: batched RoPE at prefill (TALKER_W8_ROPE_BATCHED) -----
    // Collapses the two per-row aclnnRotaryPositionEmbedding loops in
    // forward_prefill into a single aclnnApplyRotaryPosEmbV2 dispatch that
    // rotates Q AND K in-place across the full seq_len batch. Adapts the
    // CP-path Phase A.2 landing (cp_cann_engine.cpp @ cp_rope_v2_applied_)
    // to the prefill batch shape ([1, seq_len, N, head_dim] vs CP's
    // [1, 1, N, head_dim]).
    //
    // Gate: OFF by default. AND-gated on w8_applied_ — Phase 2 only fires
    // on the W8 lane which is where the RTF gap lives. TALKER_W8_QUANT
    // unset path stays byte-identical to A4b + A4c Phase 1.
    //
    // Numerical footprint (per test_rope_v2_reopen.cpp probe): ~1 F16 ulp
    // (max_abs 4.88e-4) vs the two-call aclnnRotaryPositionEmbedding NEOX
    // reference on 16Q/8KV. Parity gate is ASR Tier-1 CER=0 (clip-level).
    //
    // Layout: BSND (layout=1), rotaryMode="half" — matches our
    // half-duplicated cos/sin tables produced by build_rope_tables_.
    {
        const char *env = getenv("TALKER_W8_ROPE_BATCHED");
        if (env && env[0] != '\0' && env[0] != '0') {
            rope_batched_enabled_ = true;
        }
    }
    rope_batched_applied_ = rope_batched_enabled_ &&
                            w8_applied_ &&
                            g_cann.has_rope_v2();
    if (rope_batched_enabled_ && !w8_applied_) {
        printf("[talker_cann] TALKER_W8_ROPE_BATCHED requested but A16W8 is "
               "not active — batched prefill RoPE is gated on the W8 lane to "
               "preserve F16 byte-identity. Staying on per-row RoPE loops.\n");
    }
    if (rope_batched_enabled_ && w8_applied_ && !g_cann.has_rope_v2()) {
        printf("[talker_cann] TALKER_W8_ROPE_BATCHED requested but "
               "aclnnApplyRotaryPosEmbV2 symbol absent — staying on per-row "
               "aclnnRotaryPositionEmbedding loops.\n");
    }
    if (rope_batched_applied_) {
        printf("[talker_cann] A4c Phase 2 ENABLED: aclnnApplyRotaryPosEmbV2 "
               "replaces the two per-row RoPE loops in forward_prefill "
               "(CP-reference cp_rope_v2_applied_; ~1 F16 ulp vs two-call "
               "NEOX reference; parity gate is ASR Tier-1 CER=0).\n");
    }

    // ---- A4c Phase 3: batched FIAS at prefill (TALKER_W8_FIAS_BATCHED) -----
    // Collapses the per-row aclnnFusedInferAttentionScoreV2 loop (seq_len
    // dispatches with S_q=1 each) into one batched dispatch with S_q=seq_len.
    // Uses a user-supplied F16 causal pseShift mask of shape
    // [1, 1, seq_len, seq_len_total], broadcast across heads via stride[1]=0.
    // Mask value: 0x0000 where Q_i (abs=start_pos+i) may attend to K_j, and
    // 0xFC00 (F16 -INF) elsewhere. Broadcast-over-heads matches the Qwen2
    // pseShift convention in ggml-cann (aclnn_ops.cpp line 4370).
    //
    // Gate: OFF by default. AND-gated on w8_applied_ and has_aclgraph() is
    // NOT required — FIAS V2 is a CANN 8.3 baseline op, resolved in
    // cp_cann_symbols as non-optional.
    //
    // Risk: batched FIAS with S_q>1 on GQA 16/8 previously produced
    // cos-sim 0.28 vs iterative — see contract §4 and talker_cann line
    // "batched RoPE on [1, S, N, D] mis-rotates for S>1". Subsequent
    // fia_v34_probe_verdict.md (2026-04-21) re-validated FIAS V2 on
    // BSND+F16 but only at S_q=1 (decode shape). Parity under S_q>1 on our
    // exact prod shape is what Phase 3's ac03 sweep settles. If CER != 0
    // the flag stays OFF by default and the per-row loop owns the path.
    {
        const char *env = getenv("TALKER_W8_FIAS_BATCHED");
        if (env && env[0] != '\0' && env[0] != '0') {
            fias_batched_enabled_ = true;
        }
    }
    fias_batched_applied_ = fias_batched_enabled_ && w8_applied_;
    if (fias_batched_enabled_ && !w8_applied_) {
        printf("[talker_cann] TALKER_W8_FIAS_BATCHED requested but A16W8 is "
               "not active — batched prefill FIAS is gated on the W8 lane to "
               "preserve F16 byte-identity. Staying on per-row FIAS loop.\n");
    }
    if (fias_batched_applied_) {
        printf("[talker_cann] A4c Phase 3 ENABLED: batched "
               "aclnnFusedInferAttentionScoreV2 replaces the per-row FIAS "
               "loop in forward_prefill (user-supplied F16 causal pseShift, "
               "head-broadcast via stride[1]=0; parity gate is ASR Tier-1 "
               "CER=0).\n");
    }

    // ---- A4c Phase 4: prefill aclGraph pre-record (TALKER_W8_PREFILL_ACLGRAPH)
    // Pre-records one aclGraph per bucket at session init and replays on
    // subsequent prefill calls whose seq_len <= bucket_max. Bucket granularity
    // defaults to 64 tokens (overridable via TALKER_W8_PREFILL_BUCKET_SIZE).
    // Buckets: {bucket_size, 2*bucket_size, ..., MAX_PREFILL}. Memory:
    // ~100 MiB/graph × 8 buckets ≈ 800 MiB (within the 2 GiB budget).
    //
    // Requires: fias_batched_applied_ (Phase 3) — the batched FIAS path is
    // a prerequisite for a capturable graph because the per-row FIAS loop's
    // `kv_len`-per-row strides are not representable in a single captured
    // graph. Also requires has_aclgraph() and w8_applied_.
    //
    // Replay scope: ONLY when start_pos == 0 (i.e. first prefill chunk, no
    // KV prefix). For chunked prefill (seq_len > MAX_PREFILL → recursive
    // sub-calls with start_pos > 0) subsequent chunks fall through to the
    // eager batched path. The common ASR case (one prefill under 512
    // tokens) hits the replay path; longer contexts degrade gracefully.
    //
    // Padding: actual seq_len is padded up to the next bucket with zero
    // embeddings. Causal pseShift masks out padded positions from
    // attention (masked rows both i > seq_len-1 and columns j > seq_len-1
    // receive F16 -INF). The graph writes to KV[0..bucket_len-1]; only
    // KV[0..seq_len-1] is meaningful. kv_cache_len_ advances by seq_len
    // (not bucket_len) so subsequent decode reads never see padding KV.
    //
    // Parity gate: CER=0 × 13 clips on both cold (first-call capture) AND
    // warm (replay) runs. Numerical risk: padded activations feed the
    // residual+ffn chain; RmsNorm variance is taken over real+pad rows.
    // The RmsNorm gamma is per-row (F32 rstd stored in rstd_dev_), so a
    // zero-input row produces zero output after RmsNorm+matmul+residual
    // — indistinguishable from a fresh row. Not bit-exact vs eager (F16
    // ULP drift) but within the same envelope as Phases 1–3.
    {
        const char *env = getenv("TALKER_W8_PREFILL_ACLGRAPH");
        if (env && env[0] != '\0' && env[0] != '0') {
            prefill_aclgraph_enabled_ = true;
        }
        const char *bs_env = getenv("TALKER_W8_PREFILL_BUCKET_SIZE");
        if (bs_env && bs_env[0]) {
            int bs = atoi(bs_env);
            if (bs >= 16 && bs <= 512 && (bs & (bs - 1)) == 0) {
                prefill_bucket_size_ = bs;
            } else if (bs >= 16 && bs <= 512) {
                // non-power-of-2 is fine so long as divides MAX_PREFILL.
                if (MAX_PREFILL % bs == 0) prefill_bucket_size_ = bs;
            }
        }
    }
    prefill_aclgraph_applied_ = prefill_aclgraph_enabled_ &&
                                fias_batched_applied_ &&
                                g_cann.has_aclgraph();
    if (prefill_aclgraph_enabled_ && !fias_batched_applied_) {
        printf("[talker_cann] TALKER_W8_PREFILL_ACLGRAPH requires "
               "TALKER_W8_FIAS_BATCHED (Phase 3) — batched FIAS is a "
               "prerequisite for a capturable prefill graph. Staying eager.\n");
    }
    if (prefill_aclgraph_enabled_ && fias_batched_applied_ && !g_cann.has_aclgraph()) {
        printf("[talker_cann] TALKER_W8_PREFILL_ACLGRAPH requested but the "
               "aclGraph runtime symbols (aclmdlRI*) are absent on this "
               "toolkit. Staying eager.\n");
    }
    if (prefill_aclgraph_applied_) {
        // Bucket list: {bucket_size, 2*bucket_size, ..., MAX_PREFILL}.
        // Clamp to not exceed MAX_PREFILL.
        prefill_buckets_.clear();
        for (int b = prefill_bucket_size_; b <= MAX_PREFILL;
             b += prefill_bucket_size_) {
            prefill_buckets_.push_back(b);
        }
        prefill_graphs_.assign(prefill_buckets_.size(), nullptr);
        printf("[talker_cann] A4c Phase 4 ENABLED: prefill aclGraph "
               "pre-record at bucket granularity %d (%zu buckets, up to "
               "seq_len=%d). Graphs captured lazily on first use per bucket; "
               "replay on subsequent calls. Parity gate is ASR Tier-1 CER=0 "
               "on cold+warm runs.\n",
               prefill_bucket_size_, prefill_buckets_.size(),
               MAX_PREFILL);
    }

    // aclGraph cache (M4). OPT-IN only: set TALKER_CANN_GRAPH=1 to enable.
    // Rationale: within a single utterance each `pos` is visited exactly once
    // (decode is strictly left-to-right), so the capture-once-then-replay
    // scheme offers no intra-utterance amortization. In measured runs on
    // CANN 8.3 the capture overhead (CaptureBegin/End per step) swamps the
    // per-op launch savings and drops throughput ~2.5x. The benefit lands
    // only across utterances that share a persistent TalkerCannEngine (e.g.
    // a long-running TTS server). We leave the infrastructure wired up so
    // that session-mode callers can opt in; single-shot qwen_tts should
    // keep the default (opt-out) behaviour and run the original eager path.
    // Scoped in braces so `goto fail` in the weight-load section above
    // doesn't cross these initializations.
    {
        const bool graph_opt_in  = getenv("TALKER_CANN_GRAPH")    != nullptr;
        const bool graph_opt_out = getenv("TALKER_CANN_NO_GRAPH") != nullptr;
        graph_enabled_ = graph_opt_in && !graph_opt_out && g_cann.has_aclgraph();
        decode_graphs_.assign(MAX_SEQ, nullptr);
        if (graph_enabled_) {
            printf("[talker_cann] aclGraph ENABLED (TALKER_CANN_GRAPH=1) — "
                   "one graph per pos, captured lazily on first touch\n");
        } else if (!graph_opt_in) {
            printf("[talker_cann] aclGraph disabled (default); set "
                   "TALKER_CANN_GRAPH=1 to opt in for multi-utterance "
                   "session-mode amortization\n");
        } else if (graph_opt_out) {
            printf("[talker_cann] aclGraph DISABLED by TALKER_CANN_NO_GRAPH\n");
        } else {
            printf("[talker_cann] aclGraph unavailable (older CANN runtime) "
                   "— falling back to eager per-op dispatch\n");
        }
    }

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

void TalkerCannEngine::set_use_mrope_xvec_layout(bool enable) {
    if (enable && mrope_temporal_section_ <= 0) {
        // Metadata-missing safety — refuse to flip the flag on. Callers
        // (talker.cpp) should fall back to llama.cpp in this case; we print
        // the diagnostic rather than silently running ICL-layout on xvec/cv
        // data and producing subtly wrong attention.
        fprintf(stderr,
                "[talker_cann] set_use_mrope_xvec_layout(true) refused: "
                "mrope_temporal_section_ is 0 (GGUF lacks "
                "qwen3.rope.dimension_sections / rope_scaling.mrope_section). "
                "Staying on ICL layout; caller should use --llama_fallback.\n");
        return;
    }
    if (enable == use_mrope_xvec_layout_) return;
    use_mrope_xvec_layout_ = enable;
    if (ready_) build_rope_tables_();
}

// ============================================================================
// run_decode_ops_ — captured/replayable kernel sequence.
//
// This is the body of forward_decode from "initial Cast of cur_dev_" through
// the final RmsNorm, minus the synchronous H2D (input_embed upload) and the
// D2H (hidden_out download). The body is self-contained: it consumes
// `cur_dev_` (F16 input, already populated by the initial Cast) and leaves
// the final post-norm hidden in `normed_dev_`. All other state (residuals,
// Q/K/V, attn output, KV cache slot at `pos`, intermediate scratches) is
// addressed from persistent buffers whose pointers don't change between
// calls with the same `pos`.
//
// Why split this out: aclGraph capture records every op issued to the stream
// between CaptureBegin and CaptureEnd. By keeping it tight to exactly the
// kernel-launch sequence — no GetWorkspaceSize growth, no H2D/D2H memcpys —
// the captured graph can be replayed verbatim on subsequent calls at the
// same `pos`.
// ============================================================================

void TalkerCannEngine::cast_input_f32_to_f16_() {
    // Initial Cast F32 staging -> F16 cur. Capture-safe: input_stage_f32_dev_
    // is already populated by the synchronous H2D the caller issued above.
    aclTensor *t_in_f32 = tensor_2d(input_stage_f32_dev_,
                                     1, n_embd_, ACL_FLOAT);
    aclTensor *t_cur_f16 = tensor_2d(cur_dev_, 1, n_embd_, ACL_FLOAT16);
    CANN_OP(Cast, t_in_f32, ACL_FLOAT16, t_cur_f16);
    g_cann.aclDestroyTensor(t_in_f32);
    g_cann.aclDestroyTensor(t_cur_f16);
}

void TalkerCannEngine::run_decode_ops_(int pos) {
    // Captures-safe variant: initial Cast + body, kept together so aclGraph
    // capture records the whole eager sequence in one window.
    cast_input_f32_to_f16_();
    {
        const bool dbg = getenv("TALKER_CANN_DEBUG") != nullptr;
        if (dbg) {
            aclError s = g_cann.aclrtSynchronizeStream(stream_);
            fprintf(stderr, "[talker_cann][layer -1] initial Cast sync=%d\n", s);
        }
    }
    run_decode_body_(pos);
}

// ============================================================================
// run_decode_body_ — exactly the same kernel sequence as run_decode_ops_,
// minus the initial F32->F16 Cast. Used by the speculative split path, which
// does the cast in forward_decode_launch_cast and then inserts an
// aclnnInplaceAdd of the CP[N] group-1..15 delta onto cur_dev_ before this
// body runs.
// ============================================================================

void TalkerCannEngine::run_decode_body_(int pos) {
    // Optional per-op sync probe — stays behind an env var so it's free in
    // production but helps triangulate which op is the first to misbehave
    // during bringup on new CANN toolkit versions.
    const bool dbg = getenv("TALKER_CANN_DEBUG") != nullptr;
    auto dbg_sync = [&](int il, const char *tag) {
        if (!dbg) return;
        aclError s = g_cann.aclrtSynchronizeStream(stream_);
        fprintf(stderr, "[talker_cann][layer %d] %s sync=%d\n", il, tag, s);
    };

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

        // --- Q/K/V projection ---
        // Three paths, picked at runtime (mutually exclusive): W8 (Stretch S1)
        // > NZ (M5.3) > ND (default).
        //
        //   W8 path (TALKER_W8_QUANT=1 + CANN 8.5): aclnnWeightQuantBatchMatmul
        //   V3/V2 takes [M, K] F16 activation and [N, K]-transposed-view INT8
        //   weight (per-output-channel F16 scale). Activation is the row
        //   view of normed_dev_; output is the row view of q_dev_ / k_dev_ /
        //   v_dev_. See w8_matmul_.
        //
        //   ND path (default / fallback): compute as
        //     [dim, 1] = W[dim, n_embd] @ x[n_embd, 1]
        //   via plain aclnnMm. This matches the F16 ggml-cann reference.
        //
        //   NZ path (TALKER_NZ_WEIGHTS=1 + CANN 8.5): aclnnMatmulWeightNz's
        //   contract is self:ND, mat2:NZ — so the weight must be mat2 (RHS),
        //   not self (LHS). We swap operands to the mathematically equivalent
        //   [1, dim] = x^T[1, n_embd] @ W^T[n_embd, dim]. The W^T view over
        //   the NZ-converted buffer uses shape [n_embd, dim], strides (1, in).
        //   Activation rows are ND; weight descriptors carry FRACTAL_NZ.
        if (w8_applied_) {
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_q_row = tensor_2d(q_dev_, 1, q_dim_,  ACL_FLOAT16);
            aclTensor *t_k_row = tensor_2d(k_dev_, 1, kv_dim_, ACL_FLOAT16);
            aclTensor *t_v_row = tensor_2d(v_dev_, 1, kv_dim_, ACL_FLOAT16);
            w8_matmul_(t_norm_row, lw.q_proj_w_i8, lw.q_proj_scale,
                        q_dim_,  n_embd_, t_q_row);
            dbg_sync(il, "q_proj w8");
            w8_matmul_(t_norm_row, lw.k_proj_w_i8, lw.k_proj_scale,
                        kv_dim_, n_embd_, t_k_row);
            dbg_sync(il, "k_proj w8");
            w8_matmul_(t_norm_row, lw.v_proj_w_i8, lw.v_proj_scale,
                        kv_dim_, n_embd_, t_v_row);
            dbg_sync(il, "v_proj w8");
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_q_row);
            g_cann.aclDestroyTensor(t_k_row);
            g_cann.aclDestroyTensor(t_v_row);
        } else if (nz_applied_) {
            // Weight descriptors: transposed view with NZ format tag.
            auto wT = [&](void *buf, int64_t rows, int64_t cols) {
                int64_t shape[2]   = {cols, rows};
                int64_t strides[2] = {1, cols};
                return make_tensor(buf, 2, shape, strides, ACL_FLOAT16,
                                    ACL_FORMAT_FRACTAL_NZ);
            };
            aclTensor *t_wq_T = wT(lw.q_proj_w, q_dim_,  n_embd_);
            aclTensor *t_wk_T = wT(lw.k_proj_w, kv_dim_, n_embd_);
            aclTensor *t_wv_T = wT(lw.v_proj_w, kv_dim_, n_embd_);
            // Activation as [1, n_embd] row. Output as [1, dim] row — same
            // buffer underneath as the existing column views, just described
            // as a 1-by-N instead of N-by-1.
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_q_row = tensor_2d(q_dev_, 1, q_dim_,  ACL_FLOAT16);
            aclTensor *t_k_row = tensor_2d(k_dev_, 1, kv_dim_, ACL_FLOAT16);
            aclTensor *t_v_row = tensor_2d(v_dev_, 1, kv_dim_, ACL_FLOAT16);
            CANN_OP(Mm, t_norm_row, t_wq_T, t_q_row, (int8_t)0);
            dbg_sync(il, "q_proj");
            CANN_OP(Mm, t_norm_row, t_wk_T, t_k_row, (int8_t)0);
            dbg_sync(il, "k_proj");
            CANN_OP(Mm, t_norm_row, t_wv_T, t_v_row, (int8_t)0);
            dbg_sync(il, "v_proj");
            g_cann.aclDestroyTensor(t_wq_T);
            g_cann.aclDestroyTensor(t_wk_T);
            g_cann.aclDestroyTensor(t_wv_T);
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_q_row);
            g_cann.aclDestroyTensor(t_k_row);
            g_cann.aclDestroyTensor(t_v_row);
        } else {
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

        // --- O projection ---
        // W8 / NZ / ND — same three-way split as Q/K/V above.
        if (w8_applied_) {
            aclTensor *t_q_row = tensor_2d(q_dev_, 1, q_dim_, ACL_FLOAT16);
            aclTensor *t_o_row = tensor_2d(o_out_dev_, 1, n_embd_, ACL_FLOAT16);
            w8_matmul_(t_q_row, lw.o_proj_w_i8, lw.o_proj_scale,
                        n_embd_, q_dim_, t_o_row);
            g_cann.aclDestroyTensor(t_q_row);
            g_cann.aclDestroyTensor(t_o_row);
        } else if (nz_applied_) {
            int64_t wT_shape[2]   = {(int64_t)q_dim_, (int64_t)n_embd_};
            int64_t wT_strides[2] = {1, (int64_t)q_dim_};
            aclTensor *t_wo_T = make_tensor(lw.o_proj_w, 2, wT_shape, wT_strides,
                                             ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ);
            aclTensor *t_q_row = tensor_2d(q_dev_, 1, q_dim_, ACL_FLOAT16);
            aclTensor *t_o_row = tensor_2d(o_out_dev_, 1, n_embd_, ACL_FLOAT16);
            CANN_OP(Mm, t_q_row, t_wo_T, t_o_row, (int8_t)0);
            g_cann.aclDestroyTensor(t_wo_T);
            g_cann.aclDestroyTensor(t_q_row);
            g_cann.aclDestroyTensor(t_o_row);
        } else {
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
        // Three-way split (W8 / NZ / ND) same as Q/K/V.
        if (w8_applied_) {
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_gate_row = tensor_2d(gate_dev_,   1, inter_,  ACL_FLOAT16);
            aclTensor *t_up_row   = tensor_2d(up_dev_,     1, inter_,  ACL_FLOAT16);
            aclTensor *t_gate_flat = tensor_1d(gate_dev_, inter_, ACL_FLOAT16);
            aclTensor *t_up_flat   = tensor_1d(up_dev_,   inter_, ACL_FLOAT16);
            aclTensor *t_ffn_row   = tensor_2d(ffn_out_dev_, 1, n_embd_, ACL_FLOAT16);
            w8_matmul_(t_norm_row, lw.gate_proj_w_i8, lw.gate_proj_scale,
                        inter_,  n_embd_, t_gate_row);
            w8_matmul_(t_norm_row, lw.up_proj_w_i8,   lw.up_proj_scale,
                        inter_,  n_embd_, t_up_row);
            CANN_OP(Silu, t_gate_flat, t_gate_flat);
            CANN_OP(InplaceMul, t_gate_flat, t_up_flat);
            w8_matmul_(t_gate_row, lw.down_proj_w_i8, lw.down_proj_scale,
                        n_embd_, inter_,  t_ffn_row);
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_gate_row);
            g_cann.aclDestroyTensor(t_up_row);
            g_cann.aclDestroyTensor(t_gate_flat);
            g_cann.aclDestroyTensor(t_up_flat);
            g_cann.aclDestroyTensor(t_ffn_row);
        } else if (nz_applied_) {
            auto wT = [&](void *buf, int64_t rows, int64_t cols) {
                int64_t shape[2]   = {cols, rows};
                int64_t strides[2] = {1, cols};
                return make_tensor(buf, 2, shape, strides, ACL_FLOAT16,
                                    ACL_FORMAT_FRACTAL_NZ);
            };
            aclTensor *t_wg_T = wT(lw.gate_proj_w, inter_,  n_embd_);
            aclTensor *t_wu_T = wT(lw.up_proj_w,   inter_,  n_embd_);
            aclTensor *t_wd_T = wT(lw.down_proj_w, n_embd_, inter_);
            aclTensor *t_norm_row = tensor_2d(normed_dev_, 1, n_embd_, ACL_FLOAT16);
            aclTensor *t_gate_row = tensor_2d(gate_dev_,   1, inter_,  ACL_FLOAT16);
            aclTensor *t_up_row   = tensor_2d(up_dev_,     1, inter_,  ACL_FLOAT16);
            aclTensor *t_gate_flat = tensor_1d(gate_dev_, inter_, ACL_FLOAT16);
            aclTensor *t_up_flat   = tensor_1d(up_dev_,   inter_, ACL_FLOAT16);
            aclTensor *t_ffn_row   = tensor_2d(ffn_out_dev_, 1, n_embd_, ACL_FLOAT16);
            CANN_OP(Mm, t_norm_row, t_wg_T, t_gate_row, (int8_t)0);
            CANN_OP(Mm, t_norm_row, t_wu_T, t_up_row,   (int8_t)0);
            CANN_OP(Silu, t_gate_flat, t_gate_flat);
            CANN_OP(InplaceMul, t_gate_flat, t_up_flat);
            CANN_OP(Mm, t_gate_row, t_wd_T, t_ffn_row,  (int8_t)0);
            g_cann.aclDestroyTensor(t_wg_T);
            g_cann.aclDestroyTensor(t_wu_T);
            g_cann.aclDestroyTensor(t_wd_T);
            g_cann.aclDestroyTensor(t_norm_row);
            g_cann.aclDestroyTensor(t_gate_row);
            g_cann.aclDestroyTensor(t_up_row);
            g_cann.aclDestroyTensor(t_gate_flat);
            g_cann.aclDestroyTensor(t_up_flat);
            g_cann.aclDestroyTensor(t_ffn_row);
        } else {
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

    // Cast F16 normed -> F32 staging. The D2H download is NOT captured;
    // callers of run_decode_ops_ either download synchronously (eager /
    // first call capture path) or just rely on the next forward_decode
    // producing the next pos's output and overwriting the staging buffer.
    {
        aclTensor *t_out_f16 = tensor_1d(normed_dev_, n_embd_, ACL_FLOAT16);
        aclTensor *t_out_f32 = tensor_1d(output_stage_f32_dev_, n_embd_,
                                          ACL_FLOAT);
        CANN_OP(Cast, t_out_f16, ACL_FLOAT, t_out_f32);
        g_cann.aclDestroyTensor(t_out_f16);
        g_cann.aclDestroyTensor(t_out_f32);
    }

    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);
}

// ============================================================================
// forward_decode — single-token path. Mirrors CpCannEngine::forward_one_token
// but with 28 layers, larger dims, and (crucially) F16 residual adds instead
// of CP's F32 residual accumulator — Talker follows ggml-cann's convention.
//
// With aclGraph (M4): the first call at each `pos` runs eagerly (populating
// workspace) and then captures a graph; subsequent calls at the same `pos`
// replay the captured graph, skipping per-op GetWorkspaceSize + tensor
// descriptor overhead. Graphs are cached for the engine's lifetime (KV cache
// addresses, weight addresses, RoPE row, and workspace pointer are all
// stable, so the captured kernel arguments remain valid across utterances).
//
// Set `TALKER_CANN_NO_GRAPH=1` to disable the capture path and always run
// eagerly — useful for debugging or comparing bit-for-bit output.
// ============================================================================

// M6.2: `_launch` queues every op for pos `pos` on `stream_` asynchronously,
// records `decode_done_event_` on the stream, and returns without syncing /
// without the final D2H of the F32 hidden state. Callers must call
// `forward_decode_fetch` (or `aclrtStreamWaitEvent` on the event from a
// different stream) before using the output.
//
// `wait_event`, if non-null, is fenced at the start of `stream_` so the first
// queued op waits for that event to complete. This lets Talker[N+1]'s launch
// depend on CP[N]'s final group completion without a host round-trip.
void TalkerCannEngine::forward_decode_launch(const float *input_embed, int pos,
                                              aclrtEvent wait_event) {
    assert(ready_);
    assert(pos >= 0 && pos < MAX_SEQ);

    // Optional cross-stream fence: make `stream_` wait for `wait_event` before
    // issuing the first op below.
    if (wait_event && g_cann.aclrtStreamWaitEvent) {
        ACL_CHECK_RET(g_cann.aclrtStreamWaitEvent(stream_, wait_event));
    }

    // 1. Upload F32 input embedding. The runtime serializes this H2D behind
    //    any in-flight work on `stream_` that happens to target the same
    //    buffer, but we issue it as a blocking host->device memcpy (matches
    //    the pre-split behaviour). The subsequent Cast op inside
    //    run_decode_ops_ is what reads the staging buffer on `stream_`, so
    //    no host-side barrier is needed beyond the one aclrtMemcpy already
    //    provides.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        input_embed,          (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // 2. Compute path: either replay cached graph or run eagerly (capturing
    //    on first touch of this pos, if enabled).
    bool replayed = false;
    if (graph_enabled_ && pos < (int)decode_graphs_.size() &&
        decode_graphs_[pos] != nullptr) {
        aclError err = g_cann.aclmdlRIExecuteAsync(decode_graphs_[pos], stream_);
        if (err != 0) {
            fprintf(stderr,
                    "[talker_cann] aclmdlRIExecuteAsync pos=%d failed (%d): %s\n",
                    pos, err,
                    g_cann.aclGetRecentErrMsg
                        ? g_cann.aclGetRecentErrMsg() : "<n/a>");
            g_cann.aclmdlRIDestroy(decode_graphs_[pos]);
            decode_graphs_[pos] = nullptr;
        } else {
            replayed = true;
        }
    }

    if (!replayed) {
        run_decode_ops_(pos);

        if (graph_enabled_ && pos < (int)decode_graphs_.size() &&
            decode_graphs_[pos] == nullptr) {
            // Capture requires an empty stream — sync first, then re-issue
            // the ops under a capture window. Graph capture is incompatible
            // with the async launch/fetch split because the fetch-side sync
            // would land inside the capture window; we disable it whenever
            // graph_enabled_ is already OFF (the default in production).
            ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));

            aclError beg_err = g_cann.aclmdlRICaptureBegin(
                stream_, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
            if (beg_err != 0) {
                fprintf(stderr,
                        "[talker_cann] aclmdlRICaptureBegin pos=%d failed (%d); "
                        "disabling graph capture for this session: %s\n",
                        pos, beg_err,
                        g_cann.aclGetRecentErrMsg
                            ? g_cann.aclGetRecentErrMsg() : "<n/a>");
                graph_enabled_ = false;
            } else {
                run_decode_ops_(pos);
                aclmdlRI g = nullptr;
                aclError end_err = g_cann.aclmdlRICaptureEnd(stream_, &g);
                if (end_err != 0 || g == nullptr) {
                    fprintf(stderr,
                            "[talker_cann] aclmdlRICaptureEnd pos=%d failed "
                            "(%d), graph=%p; this pos stays on eager: %s\n",
                            pos, end_err, (void *)g,
                            g_cann.aclGetRecentErrMsg
                                ? g_cann.aclGetRecentErrMsg() : "<n/a>");
                    if (g) g_cann.aclmdlRIDestroy(g);
                } else {
                    decode_graphs_[pos] = g;
                }
            }
        }
    }

    // 3. Record event so `forward_decode_fetch` (or another stream via
    //    aclrtStreamWaitEvent) can fence against decode completion. If the
    //    runtime lacks the event API, callers will fall back to sync-stream
    //    inside `_fetch`.
    if (decode_done_event_ && g_cann.aclrtRecordEvent) {
        ACL_CHECK_RET(g_cann.aclrtRecordEvent(decode_done_event_, stream_));
    }

    // Advance cache length (caller doesn't track per-token position, but we
    // expose `reset_kv_cache` so stateful callers can reset between
    // utterances).
    if (pos + 1 > kv_cache_len_) kv_cache_len_ = pos + 1;
}

void TalkerCannEngine::forward_decode_fetch(float *hidden_out) {
    assert(ready_);

    // Wait for the decode to finish. Prefer event sync (cheaper on some
    // CANN versions; doesn't serialize the entire stream); fall back to
    // stream sync if the event API wasn't resolved.
    if (decode_done_event_ && g_cann.aclrtSynchronizeEvent) {
        ACL_CHECK_RET(g_cann.aclrtSynchronizeEvent(decode_done_event_));
    } else {
        ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
    }

    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        hidden_out, (size_t)n_embd_ * sizeof(float),
        output_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

void TalkerCannEngine::forward_decode(const float *input_embed, int pos,
                                       float *hidden_out) {
    forward_decode_launch(input_embed, pos, nullptr);
    forward_decode_fetch(hidden_out);
}

// ============================================================================
// M6.2 Track J: speculative-embedding split.
//
// `_launch_cast` uploads the F32 input embedding and queues the F32->F16
// Cast into `cur_dev_` on `stream_`. Graph capture is incompatible with this
// path (a subsequent delta-add would invalidate the captured body), so we
// always run eagerly here.
//
// `add_input_delta_f32` fences `stream_` on `wait_event` (typically the
// sibling stream's CP forward_done_event_), H2D-uploads the F32 delta,
// casts to F16, and InplaceAdd's onto `cur_dev_`.
//
// `_launch_layers` runs the 28-layer transformer body + final norm + cast
// back to F32, then records `decode_done_event_`.
// ============================================================================

void TalkerCannEngine::forward_decode_launch_cast(const float *input_embed,
                                                   int pos,
                                                   aclrtEvent wait_event) {
    assert(ready_);
    assert(pos >= 0 && pos < MAX_SEQ);
    (void)pos;  // only consumed inside _launch_layers for RoPE indexing.

    if (wait_event && g_cann.aclrtStreamWaitEvent) {
        ACL_CHECK_RET(g_cann.aclrtStreamWaitEvent(stream_, wait_event));
    }

    // H2D upload: F32 provisional embedding into staging.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        input_embed,          (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // F32 -> F16 cast on stream_ into cur_dev_.
    cast_input_f32_to_f16_();
}

void TalkerCannEngine::add_input_delta_f32(const float *delta,
                                            aclrtEvent wait_event) {
    assert(ready_);

    // Optional cross-stream fence (e.g. wait for CP[N]'s forward_done_event_
    // from the sibling stream before consuming the host's delta).
    if (wait_event && g_cann.aclrtStreamWaitEvent) {
        ACL_CHECK_RET(g_cann.aclrtStreamWaitEvent(stream_, wait_event));
    }

    // H2D upload F32 delta. Blocking memcpy is fine here — by the time the
    // caller built `delta` host-side, CP[N]'s fetch has already drained on
    // stream B and `wait_event` (above) only serialises the device-side
    // ordering. The host memcpy is sequential with the prior queue, so the
    // subsequent Cast+Add ops see the uploaded bytes in stream order.
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        delta_stage_f32_dev_, (size_t)n_embd_ * sizeof(float),
        delta,                (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // Cast delta F32 -> F16 into dedicated scratch.
    {
        aclTensor *t_in_f32 = tensor_2d(delta_stage_f32_dev_,
                                         1, n_embd_, ACL_FLOAT);
        aclTensor *t_in_f16 = tensor_2d(delta_stage_f16_dev_,
                                         1, n_embd_, ACL_FLOAT16);
        CANN_OP(Cast, t_in_f32, ACL_FLOAT16, t_in_f16);
        g_cann.aclDestroyTensor(t_in_f32);
        g_cann.aclDestroyTensor(t_in_f16);
    }

    // InplaceAdd: cur_dev_ += delta_stage_f16_dev_ (F16). aclnnInplaceAdd's
    // signature is (selfRef, other, alpha) — `self` is modified in place.
    {
        aclTensor *t_cur   = tensor_1d(cur_dev_, n_embd_, ACL_FLOAT16);
        aclTensor *t_delta = tensor_1d(delta_stage_f16_dev_, n_embd_,
                                        ACL_FLOAT16);
        CANN_OP(InplaceAdd, t_cur, t_delta, one_scalar_f16_);
        g_cann.aclDestroyTensor(t_cur);
        g_cann.aclDestroyTensor(t_delta);
    }
}

void TalkerCannEngine::forward_decode_launch_layers(int pos) {
    assert(ready_);
    assert(pos >= 0 && pos < MAX_SEQ);

    // Graph capture is skipped on the split path. The recorded graph at `pos`
    // would assume the original (un-delta'd) cur_dev_ contents, which is no
    // longer a function of just pos when delta-add runs in between.
    run_decode_body_(pos);

    if (decode_done_event_ && g_cann.aclrtRecordEvent) {
        ACL_CHECK_RET(g_cann.aclrtRecordEvent(decode_done_event_, stream_));
    }

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

    // Default: batched prefill. Shares the matmul/RmsNorm/FFN work across
    // the seq_len rows while looping FIAS + RoPE per-row to sidestep CANN
    // 8.3's batched-kernel numerics quirks (batched RoPE on [1, S, N, D]
    // mis-rotates for S>1; FIAS with S_q>1 has tiling-key gaps on the
    // Talker's GQA 16/8 shape set). On seq_len=127 this runs at ~150 ms
    // vs the iterative path's ~2 s, and produces a hidden state within
    // cos-sim 0.9999 of the iterative reference on real text input
    // (validated via test_prefill_diff with a real-embedding dump).
    //
    // Earlier note (since resolved): the batched path once produced cos-sim
    // 0.28 vs iterative — that was with batched RoPE (pre-M2.5) and
    // innerPrecise=2 + nextTokens=0. Both were fixed by unrolling RoPE per
    // row and switching FIAS to S_q=1 per-row with innerPrecise=0. Set
    // TALKER_PREFILL_ITERATIVE=1 to force the iterative fallback path for
    // debugging or as a regression-revert.
    if (getenv("TALKER_PREFILL_ITERATIVE") != nullptr) {
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

    // A4c Phase 4: aclGraph replay attempt.
    //
    // Replay fires when ALL of:
    //   * prefill_aclgraph_applied_ (env gate + Phase 3 + has_aclgraph())
    //   * start_pos == 0 (first prefill chunk in the utterance — the captured
    //     graph assumes KV writes begin at cache slot 0)
    //   * seq_len is a bucket-size multiple of prefill_bucket_size_ AND
    //     <= MAX_PREFILL (so a pre-recorded graph exists for that bucket)
    //   * the graph for that bucket is already captured (non-null) OR this
    //     call opts into first-touch capture (graph == null)
    //
    // On mismatch, the call falls through to the eager body. kv_cache_len_
    // accounting is identical in both paths — no dispatch visible to the
    // caller.
    //
    // For this first landing of Phase 4 we gate replay on seq_len being an
    // exact bucket multiple (no padding). The hit rate is low on arbitrary
    // text seq lengths but every hit eliminates op-dispatch overhead for a
    // ~100+ op layer stack × n_layers_. A follow-up ("Phase 4b") can add
    // input padding so every seq_len rounds up to the nearest bucket; that
    // path needs careful handling of RmsNorm over the pad rows (zero input
    // → zero output, which is fine numerically, but the FFN + add chain
    // sees extra rows).
    int prefill_bucket_idx = -1;
    if (prefill_aclgraph_applied_ && start_pos == 0 &&
        seq_len % prefill_bucket_size_ == 0 &&
        seq_len <= MAX_PREFILL) {
        for (int k = 0; k < (int)prefill_buckets_.size(); ++k) {
            if (prefill_buckets_[k] == seq_len) {
                prefill_bucket_idx = k;
                break;
            }
        }
    }
    if (prefill_bucket_idx >= 0 &&
        prefill_graphs_[prefill_bucket_idx] != nullptr) {
        // ---- REPLAY PATH ----------------------------------------------------
        // Upload the F32 embeddings at the same device address the graph was
        // captured over (input_stage_f32_dev_). The subsequent Cast, QKV,
        // RoPE, FIAS, etc. ops replay against the new byte contents.
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            input_stage_f32_dev_, (size_t)seq_len * n_embd_ * sizeof(float),
            input_embeds,         (size_t)seq_len * n_embd_ * sizeof(float),
            ACL_MEMCPY_HOST_TO_DEVICE));
        // Re-upload the causal pseShift mask (content is stable per
        // seq_len + start_pos, so this is an identical payload to the
        // captured one — still required because the captured executor
        // references the buffer contents, not a pure-function pattern).
        const int seq_len_total_replay = start_pos + seq_len;  // == seq_len
        std::vector<uint16_t> mask_host((size_t)seq_len * seq_len_total_replay);
        const uint16_t F16_ZERO = 0x0000;
        const uint16_t F16_NEG_INF = 0xFC00;
        for (int i = 0; i < seq_len; ++i) {
            uint16_t *row = mask_host.data() + (size_t)i * seq_len_total_replay;
            for (int j = 0; j < seq_len_total_replay; ++j) {
                row[j] = (j <= i) ? F16_ZERO : F16_NEG_INF;
            }
        }
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            causal_mask_dev_,
            (size_t)seq_len * seq_len_total_replay * sizeof(uint16_t),
            mask_host.data(),
            (size_t)seq_len * seq_len_total_replay * sizeof(uint16_t),
            ACL_MEMCPY_HOST_TO_DEVICE));
        // Dispatch the captured graph.
        aclError err = g_cann.aclmdlRIExecuteAsync(
            prefill_graphs_[prefill_bucket_idx], stream_);
        if (err != 0) {
            fprintf(stderr,
                    "[talker_cann] prefill aclmdlRIExecuteAsync bucket=%d "
                    "(seq_len=%d) failed (%d): %s — retiring the graph and "
                    "falling back to eager for this call.\n",
                    prefill_bucket_idx, seq_len, err,
                    g_cann.aclGetRecentErrMsg
                        ? g_cann.aclGetRecentErrMsg() : "<n/a>");
            g_cann.aclmdlRIDestroy(prefill_graphs_[prefill_bucket_idx]);
            prefill_graphs_[prefill_bucket_idx] = nullptr;
            prefill_bucket_idx = -1;
            // fall through to eager path
        } else {
            // Final cast + host download for last-hidden-out, symmetric with
            // the eager path's tail.
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
                ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
            }
            int end_pos = start_pos + seq_len;
            if (end_pos > kv_cache_len_) kv_cache_len_ = end_pos;
            return;
        }
    }

    // Single-shot prefill path.
    // 1. Upload the batch of F32 embeddings. The F32→F16 Cast runs below on
    //    stream_; under A4c Phase 4 capture the Cast is the first recorded
    //    op of the graph (replay reuses input_stage_f32_dev_ contents
    //    uploaded in the replay branch above).
    ACL_CHECK_RET(g_cann.aclrtMemcpy(
        input_stage_f32_dev_, (size_t)seq_len * n_embd_ * sizeof(float),
        input_embeds,         (size_t)seq_len * n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE));

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

    // Causality in the per-row FIAS loop is enforced by slicing KV to
    // [0..pos_abs] per row (so there is nothing to mask). The batched
    // Phase 3 path (fias_batched_applied_) cannot slice per-row, so it
    // supplies an F16 causal pseShift mask [seq_len, seq_len_total]
    // built once here and broadcast across heads via stride[1]=0. Mask
    // layout: mask[i, j] = 0x0000 (F16 zero, kept) if j <= start_pos + i,
    // else 0xFC00 (F16 -INF, blocked).
    if (fias_batched_applied_) {
        // Host-side scratch: seq_len * seq_len_total F16. On a 4352-token
        // prefill this is ~4 MB; on our worst prod case (seq_len=MAX_PREFILL
        // =512, seq_len_total≤MAX_SEQ=4096) ~4 MB. That fits in
        // causal_mask_dev_ which was sized [MAX_PREFILL, MAX_SEQ] F16 at
        // init.
        std::vector<uint16_t> mask_host((size_t)seq_len * seq_len_total);
        const uint16_t F16_ZERO = 0x0000;
        const uint16_t F16_NEG_INF = 0xFC00;
        for (int i = 0; i < seq_len; ++i) {
            const int q_abs = start_pos + i;
            uint16_t *row = mask_host.data() + (size_t)i * seq_len_total;
            for (int j = 0; j < seq_len_total; ++j) {
                row[j] = (j <= q_abs) ? F16_ZERO : F16_NEG_INF;
            }
        }
        ACL_CHECK_RET(g_cann.aclrtMemcpy(
            causal_mask_dev_,
            (size_t)seq_len * seq_len_total * sizeof(uint16_t),
            mask_host.data(),
            (size_t)seq_len * seq_len_total * sizeof(uint16_t),
            ACL_MEMCPY_HOST_TO_DEVICE));
    } else {
        (void)causal_mask_dev_;  // retained for future pseShift paths; unused.
    }

    // ---- A4c Phase 4 capture begin (if eligible) ----------------------
    // All H2D work (F32 embeddings + mask) is done. The remaining body
    // — F32→F16 Cast, layer loop, final RmsNorm, last-row Cast — is a
    // pure device-side op sequence with stable buffer addresses. That is
    // exactly what aclGraph wants: capture these ops once per bucket, on
    // subsequent matching calls skip the host-side dispatch overhead.
    //
    // Capture is attempted only on the first call that matches a
    // ready-but-unbucketed slot. On capture failure (e.g. the toolkit
    // refuses a specific op under capture) we fall through to eager and
    // permanently disable capture for that bucket.
    bool prefill_capture_this_call = false;
    if (prefill_aclgraph_applied_ && prefill_bucket_idx >= 0 &&
        prefill_graphs_[prefill_bucket_idx] == nullptr) {
        // Sync to ensure the stream is empty before CaptureBegin; the mask
        // H2D above is a blocking aclrtMemcpy so it's already drained, but
        // a prior decode could still be in flight from the engine's last
        // call — draining here is cheap and correct.
        ACL_CHECK_RET(g_cann.aclrtSynchronizeStream(stream_));
        aclError beg_err = g_cann.aclmdlRICaptureBegin(
            stream_, ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL);
        if (beg_err != 0) {
            fprintf(stderr,
                    "[talker_cann] prefill aclmdlRICaptureBegin bucket=%d "
                    "(seq_len=%d) failed (%d): %s — disabling capture for "
                    "this bucket and proceeding eager.\n",
                    prefill_bucket_idx, seq_len, beg_err,
                    g_cann.aclGetRecentErrMsg
                        ? g_cann.aclGetRecentErrMsg() : "<n/a>");
            // Mark the slot as permanently retired by pointing at a sentinel
            // that can never match on replay: leave it null (the ExecuteAsync
            // check gates on non-null), and clamp the bucket selection
            // logic to skip further capture attempts on this bucket via the
            // `prefill_capture_retired_` set (optional — simplest is to
            // leave null and rely on "graph==nullptr → eager" at all
            // subsequent calls, which keeps retrying capture but that is
            // harmless since the failure mode is deterministic).
        } else {
            prefill_capture_this_call = true;
        }
    }

    // 1b. F32 → F16 Cast (first device op of the capturable region).
    {
        aclTensor *t_in_f32 = tensor_2d(input_stage_f32_dev_,
                                          seq_len, n_embd_, ACL_FLOAT);
        aclTensor *t_cur_f16 = tensor_2d(cur_batch_dev_, seq_len, n_embd_,
                                           ACL_FLOAT16);
        CANN_OP(Cast, t_in_f32, ACL_FLOAT16, t_cur_f16);
        g_cann.aclDestroyTensor(t_in_f32);
        g_cann.aclDestroyTensor(t_cur_f16);
    }

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
        // M5.3: prefill already uses the activation-first, weight^T
        // convention (ggml-cann pattern) — here we just toggle the weight
        // tensor's format tag to FRACTAL_NZ when nz_applied_ and dispatch the
        // NZ-aware matmul op. Pre-conversion via aclnnTransMatmulWeight
        // happened once at init.
        const aclFormat wfmt_prefill = nz_applied_ ? ACL_FORMAT_FRACTAL_NZ
                                                    : ACL_FORMAT_ND;
        auto weight_T_tensor = [&](void *buf, int64_t rows, int64_t cols) {
            // original shape [rows, cols] row-major. Transposed view [cols, rows]
            // with strides (1, cols).
            int64_t shape[2]   = {cols, rows};
            int64_t strides[2] = {1, cols};
            return make_tensor(buf, 2, shape, strides, ACL_FLOAT16,
                                wfmt_prefill);
        };
        // A4b: when w8_applied_, prefill dispatches the same per-channel
        // INT8 + F16 scale path used by decode. aclnnWeightQuantBatchMatmulV3
        // accepts x:[M, K] with M = seq_len; no reshape needed. The F16
        // weight_T_tensor descriptors are only built in the fallback branch.
        //
        // A4c Phase 1: when gmm_qkv_applied_ (= TALKER_W8_GMM_QKV=1 +
        // w8_applied_ + runtime symbol present), collapse the 3 w8_matmul_
        // calls into one aclnnGroupedMatmulV3 dispatch. Byte-identity gate
        // is ASR Tier-1 CER=0 (clip output), not per-tensor bit-equal — the
        // fused path drifts by 1-2 F16 ulp from the 3-call reference.
        if (gmm_qkv_applied_) {
            gmm_qkv_prefill_(normed_batch_dev_, seq_len, (int64_t)il,
                              q_batch_dev_, k_batch_dev_, v_batch_dev_);
        } else if (w8_applied_) {
            aclTensor *t_normed = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_q = tensor_2d(q_batch_dev_, seq_len, q_dim_,  ACL_FLOAT16);
            aclTensor *t_k = tensor_2d(k_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            aclTensor *t_v = tensor_2d(v_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            w8_matmul_(t_normed, lw.q_proj_w_i8, lw.q_proj_scale,
                        q_dim_,  n_embd_, t_q);
            w8_matmul_(t_normed, lw.k_proj_w_i8, lw.k_proj_scale,
                        kv_dim_, n_embd_, t_k);
            w8_matmul_(t_normed, lw.v_proj_w_i8, lw.v_proj_scale,
                        kv_dim_, n_embd_, t_v);
            g_cann.aclDestroyTensor(t_normed);
            g_cann.aclDestroyTensor(t_q);
            g_cann.aclDestroyTensor(t_k);
            g_cann.aclDestroyTensor(t_v);
        } else {
            aclTensor *t_normed = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
            aclTensor *t_wq_T = weight_T_tensor(lw.q_proj_w, q_dim_,  n_embd_);
            aclTensor *t_wk_T = weight_T_tensor(lw.k_proj_w, kv_dim_, n_embd_);
            aclTensor *t_wv_T = weight_T_tensor(lw.v_proj_w, kv_dim_, n_embd_);
            aclTensor *t_q = tensor_2d(q_batch_dev_, seq_len, q_dim_,  ACL_FLOAT16);
            aclTensor *t_k = tensor_2d(k_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            aclTensor *t_v = tensor_2d(v_batch_dev_, seq_len, kv_dim_, ACL_FLOAT16);
            CANN_MATMUL(t_normed, t_wq_T, t_q);
            CANN_MATMUL(t_normed, t_wk_T, t_k);
            CANN_MATMUL(t_normed, t_wv_T, t_v);
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

        // --- RoPE Q + K ---
        //
        // Two paths, selected per rope_batched_applied_:
        //   (a) A4c Phase 2 batched path (TALKER_W8_ROPE_BATCHED=1):
        //       single aclnnApplyRotaryPosEmbV2 in-place dispatch across
        //       the full [1, seq_len, N, head_dim] batch for Q and K. Q is
        //       d2d-copied from q_batch_dev_ to attn_out_batch_dev_
        //       pre-rotation so the FIAS loop below sees the same buffer
        //       contract as the per-row path. K is d2d-copied into its
        //       KV cache slot pre-rotation, then rotated in-place at that
        //       address (matching the legacy "K ends up in cache slot"
        //       contract).
        //   (b) Per-row legacy path: two aclnnRotaryPositionEmbedding loops.
        //       Kept verbatim as the defensive branch; reached whenever
        //       TALKER_W8_ROPE_BATCHED is unset, A16W8 is off, or the V2
        //       symbol is absent.
        uint16_t *k_slot =
            (uint16_t *)k_cache_dev_[il] + (size_t)start_pos * kv_dim_;

        if (rope_batched_applied_) {
            // (1) Stage Q into attn_out_batch_dev_ so the subsequent FIAS
            //     code path — which reads Q from attn_out_batch_dev_ — sees
            //     the rotated Q regardless of which RoPE path ran. V2 then
            //     rotates attn_out_batch_dev_ in-place.
            ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
                attn_out_batch_dev_,
                (size_t)seq_len * q_dim_ * sizeof(uint16_t),
                q_batch_dev_,
                (size_t)seq_len * q_dim_ * sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));
            // (2) Stage K directly into its KV-cache slot pre-rotation so
            //     V2 can rotate in-place at the cache address.
            ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
                k_slot,
                (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
                k_batch_dev_,
                (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
                ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));
            // (3) Build [1, seq_len, N, head_dim] BSND views for Q and K.
            //     t_cos / t_sin were built upstream at shape
            //     [1, seq_len, 1, head_dim] — the exact broadcast shape V2
            //     expects for BSND (layout=1).
            int64_t qv_shape[4]   = {1, (int64_t)seq_len,
                                      (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t qv_strides[4] = {(int64_t)q_dim_ * seq_len,
                                      (int64_t)q_dim_,
                                      (int64_t)head_dim_, 1};
            int64_t kv_shape[4]   = {1, (int64_t)seq_len,
                                      (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t kv_strides[4] = {(int64_t)kv_dim_ * seq_len,
                                      (int64_t)kv_dim_,
                                      (int64_t)head_dim_, 1};
            aclTensor *t_q_v2 = tensor_strided(attn_out_batch_dev_, 4,
                                                 qv_shape, qv_strides,
                                                 ACL_FLOAT16);
            aclTensor *t_k_v2 = tensor_strided(k_slot, 4,
                                                 kv_shape, kv_strides,
                                                 ACL_FLOAT16);
            // (4) Dispatch V2. GetWorkspaceSize has a distinct arg order
            //     (layout + rotaryMode before the ws out-params), so we
            //     can't use the CANN_OP macro — invoke explicitly. Mirror
            //     CP-side cp_rope_v2_applied_ lifetime discipline: single
            //     ensure_workspace_ between GetWorkspaceSize and launch.
            uint64_t v2_ws = 0;
            aclOpExecutor *v2_exec = nullptr;
            char rotary_mode[5] = {'h','a','l','f',0};
            ACL_CHECK_RET(g_cann.aclnnApplyRotaryPosEmbV2GetWorkspaceSize(
                t_q_v2, t_k_v2, t_cos, t_sin,
                /*layout=*/(int64_t)1, rotary_mode, &v2_ws, &v2_exec));
            ensure_workspace_((size_t)v2_ws);
            void *v2_ws_ptr = v2_ws > 0 ? workspace_dev_ : nullptr;
            ACL_CHECK_RET(g_cann.aclnnApplyRotaryPosEmbV2(
                v2_ws_ptr, v2_ws, v2_exec, stream_));
            g_cann.aclDestroyTensor(t_q_v2);
            g_cann.aclDestroyTensor(t_k_v2);
        } else {
            // --- Legacy per-row Q loop -> attn_out_batch_dev_ ---
            // Apply RoPE separately for each position. The batched RoPE
            // kernel (aclnnRotaryPositionEmbedding on [1, S, N, D] with
            // cos/sin [1, S, 1, D]) appears to mis-rotate for S>1 on CANN
            // 8.3, so we unroll the S dim.
            for (int s = 0; s < seq_len; ++s) {
                uint16_t *q_row = (uint16_t *)q_batch_dev_ + (size_t)s * q_dim_;
                uint16_t *attn_row = (uint16_t *)attn_out_batch_dev_ + (size_t)s * q_dim_;
                uint16_t *cos_row = (uint16_t *)rope_cos_dev_ +
                                    (size_t)(start_pos + s) * head_dim_;
                uint16_t *sin_row = (uint16_t *)rope_sin_dev_ +
                                    (size_t)(start_pos + s) * head_dim_;
                int64_t cs_shape[4]   = {1, 1, 1, (int64_t)head_dim_};
                int64_t cs_strides[4] = {(int64_t)head_dim_, (int64_t)head_dim_,
                                          (int64_t)head_dim_, 1};
                aclTensor *t_cos_s = tensor_strided(cos_row, 4, cs_shape,
                                                      cs_strides, ACL_FLOAT16);
                aclTensor *t_sin_s = tensor_strided(sin_row, 4, cs_shape,
                                                      cs_strides, ACL_FLOAT16);
                int64_t q_shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
                int64_t q_strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                         (int64_t)head_dim_, 1};
                aclTensor *t_q_in  = tensor_strided(q_row, 4, q_shape, q_strides,
                                                     ACL_FLOAT16);
                aclTensor *t_q_out = tensor_strided(attn_row, 4, q_shape, q_strides,
                                                     ACL_FLOAT16);
                CANN_OP(RotaryPositionEmbedding,
                        t_q_in, t_cos_s, t_sin_s, (int64_t)0, t_q_out);
                g_cann.aclDestroyTensor(t_q_in);
                g_cann.aclDestroyTensor(t_q_out);
                g_cann.aclDestroyTensor(t_cos_s);
                g_cann.aclDestroyTensor(t_sin_s);
            }

            // --- Legacy per-row K loop, write into KV cache slots ---
            for (int s = 0; s < seq_len; ++s) {
                uint16_t *k_row = (uint16_t *)k_batch_dev_ + (size_t)s * kv_dim_;
                uint16_t *k_out = k_slot + (size_t)s * kv_dim_;
                uint16_t *cos_row = (uint16_t *)rope_cos_dev_ +
                                    (size_t)(start_pos + s) * head_dim_;
                uint16_t *sin_row = (uint16_t *)rope_sin_dev_ +
                                    (size_t)(start_pos + s) * head_dim_;
                int64_t cs_shape[4]   = {1, 1, 1, (int64_t)head_dim_};
                int64_t cs_strides[4] = {(int64_t)head_dim_, (int64_t)head_dim_,
                                          (int64_t)head_dim_, 1};
                aclTensor *t_cos_s = tensor_strided(cos_row, 4, cs_shape,
                                                      cs_strides, ACL_FLOAT16);
                aclTensor *t_sin_s = tensor_strided(sin_row, 4, cs_shape,
                                                      cs_strides, ACL_FLOAT16);
                int64_t k_shape[4]   = {1, 1, (int64_t)n_kv_, (int64_t)head_dim_};
                int64_t k_strides[4] = {(int64_t)kv_dim_, (int64_t)kv_dim_,
                                         (int64_t)head_dim_, 1};
                aclTensor *t_k_in  = tensor_strided(k_row, 4, k_shape, k_strides,
                                                     ACL_FLOAT16);
                aclTensor *t_k_out = tensor_strided(k_out, 4, k_shape, k_strides,
                                                     ACL_FLOAT16);
                CANN_OP(RotaryPositionEmbedding,
                        t_k_in, t_cos_s, t_sin_s, (int64_t)0, t_k_out);
                g_cann.aclDestroyTensor(t_k_in);
                g_cann.aclDestroyTensor(t_k_out);
                g_cann.aclDestroyTensor(t_cos_s);
                g_cann.aclDestroyTensor(t_sin_s);
            }
        }

        // --- V -> cache slice ---
        uint16_t *v_slot =
            (uint16_t *)v_cache_dev_[il] + (size_t)start_pos * kv_dim_;
        ACL_CHECK_RET(g_cann.aclrtMemcpyAsync(
            v_slot, (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
            v_batch_dev_,
            (size_t)seq_len * kv_dim_ * sizeof(uint16_t),
            ACL_MEMCPY_DEVICE_TO_DEVICE, stream_));

        // --- FIAS ---
        //
        // Two paths, selected per fias_batched_applied_:
        //   (a) A4c Phase 3 batched path (TALKER_W8_FIAS_BATCHED=1):
        //       one FIASv2 dispatch with Q=[1, seq_len, Hq, D] and
        //       K/V=[1, seq_len_total, Hkv, D], causal pseShift supplied
        //       as [1, 1, seq_len, seq_len_total] F16 (broadcast across
        //       heads via stride[1]=0). Same {preTokens,nextTokens,
        //       innerPrecise} as the per-row loop.
        //   (b) Per-row legacy path: seq_len calls with S_q=1, KV sliced
        //       to [0..pos_abs] per row. Retained verbatim as the
        //       defensive branch.
        if (fias_batched_applied_) {
            uint16_t *q_src = (uint16_t *)attn_out_batch_dev_;
            uint16_t *attn_dst = (uint16_t *)q_batch_dev_;

            int64_t q_shape[4]   = {1, (int64_t)seq_len,
                                     (int64_t)n_heads_, (int64_t)head_dim_};
            int64_t q_strides[4] = {(int64_t)q_dim_ * seq_len,
                                     (int64_t)q_dim_,
                                     (int64_t)head_dim_, 1};
            aclTensor *t_q_batch = tensor_strided(q_src, 4, q_shape, q_strides,
                                                    ACL_FLOAT16);
            aclTensor *t_attn_batch = tensor_strided(attn_dst, 4, q_shape, q_strides,
                                                       ACL_FLOAT16);

            int64_t kv_shape[4]   = {1, (int64_t)seq_len_total,
                                      (int64_t)n_kv_, (int64_t)head_dim_};
            int64_t kv_strides[4] = {(int64_t)seq_len_total * kv_dim_,
                                      (int64_t)kv_dim_,
                                      (int64_t)head_dim_, 1};
            aclTensor *t_k_batch = tensor_strided(k_cache_dev_[il], 4, kv_shape,
                                                    kv_strides, ACL_FLOAT16);
            aclTensor *t_v_batch = tensor_strided(v_cache_dev_[il], 4, kv_shape,
                                                    kv_strides, ACL_FLOAT16);
            aclTensorList *t_kl = g_cann.aclCreateTensorList(&t_k_batch, 1);
            aclTensorList *t_vl = g_cann.aclCreateTensorList(&t_v_batch, 1);

            // Causal pseShift: shape [1, 1, seq_len, seq_len_total], stride[1]=0
            // to broadcast across the n_heads axis (Qwen2 / ggml-cann
            // convention). Data was uploaded to causal_mask_dev_ once before
            // the layer loop.
            int64_t pse_shape[4]   = {1, (int64_t)n_heads_,
                                       (int64_t)seq_len, (int64_t)seq_len_total};
            int64_t pse_strides[4] = {(int64_t)seq_len * seq_len_total,
                                       /*broadcast over heads*/ 0,
                                       (int64_t)seq_len_total, 1};
            aclTensor *t_pse = tensor_strided(causal_mask_dev_, 4, pse_shape,
                                                pse_strides, ACL_FLOAT16);

            uint64_t fa_ws = 0;
            aclOpExecutor *fa_exec = nullptr;
            char layout[5] = {'B','S','N','D',0};
            double scale = 1.0 / sqrt((double)head_dim_);
            // Per the Qwen2 ggml-cann recipe (aclnn_ops.cpp line 4422):
            // innerPrecise=2 when S_q>1 (high-precision softmax path).
            const int64_t inner_precise_batched = 2;
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
                t_q_batch, t_kl, t_vl,
                /*pseShift*/ t_pse, /*attenMask*/ nullptr,
                /*actSeqLen*/ nullptr, /*actSeqLenKv*/ nullptr,
                /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,
                /*deqScale2*/ nullptr, /*quantScale2*/ nullptr,
                /*quantOffset2*/ nullptr,
                /*antiquantScale*/ nullptr, /*antiquantOffset*/ nullptr,
                /*blockTable*/ nullptr,
                /*queryPaddingSize*/ nullptr, /*kvPaddingSize*/ nullptr,
                /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,
                /*valueAntiquantScale*/ nullptr, /*valueAntiquantOffset*/ nullptr,
                /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
                /*actualSharedPrefixLen*/ nullptr,
                /*numHeads*/ (int64_t)n_heads_,
                /*scaleValue*/ scale,
                /*preTokens*/ (int64_t)65535,
                /*nextTokens*/ (int64_t)65535,
                /*inputLayout*/ layout,
                /*numKeyValueHeads*/ (int64_t)n_kv_,
                /*sparseMode*/ (int64_t)0,
                /*innerPrecise*/ inner_precise_batched,
                /*blockSize*/ (int64_t)0,
                /*antiquantMode*/ (int64_t)0,
                /*softmaxLseFlag*/ false,
                /*keyAntiquantMode*/ (int64_t)0,
                /*valueAntiquantMode*/ (int64_t)0,
                /*attentionOut*/ t_attn_batch,
                /*softmaxLse*/ nullptr,
                &fa_ws, &fa_exec));
            ensure_workspace_(fa_ws);
            void *ws = fa_ws > 0 ? workspace_dev_ : nullptr;
            ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2(
                ws, fa_ws, fa_exec, stream_));
            g_cann.aclDestroyTensorList(t_kl);
            g_cann.aclDestroyTensorList(t_vl);
            g_cann.aclDestroyTensor(t_q_batch);
            g_cann.aclDestroyTensor(t_attn_batch);
            g_cann.aclDestroyTensor(t_pse);
        } else {
            // Per-row FIAS loop: each Q row queries KV[0..pos] independently.
            //   This sidesteps batched-FIAS numerical issues by looping with
            //   S_q=1 (decode-mode FIAS), writing all seq_len attention
            //   outputs into q_batch_dev_. Same answer as iterative
            //   forward_decode, but shares the per-layer matmul/RmsNorm/etc.
            //   batched work above.
            for (int s = 0; s < seq_len; ++s) {
                const int pos_abs = start_pos + s;
                const int kv_len = pos_abs + 1;

                uint16_t *q_src = (uint16_t*)attn_out_batch_dev_ + (size_t)s * q_dim_;
                uint16_t *attn_dst = (uint16_t*)q_batch_dev_ + (size_t)s * q_dim_;

                int64_t q_shape[4]   = {1, 1, (int64_t)n_heads_, (int64_t)head_dim_};
                int64_t q_strides[4] = {(int64_t)q_dim_, (int64_t)q_dim_,
                                         (int64_t)head_dim_, 1};
                aclTensor *t_q_s = tensor_strided(q_src, 4, q_shape, q_strides,
                                                    ACL_FLOAT16);
                aclTensor *t_attn_s = tensor_strided(attn_dst, 4, q_shape, q_strides,
                                                       ACL_FLOAT16);

                int64_t kv_shape[4]   = {1, (int64_t)kv_len,
                                          (int64_t)n_kv_, (int64_t)head_dim_};
                int64_t kv_strides[4] = {(int64_t)kv_len * kv_dim_,
                                          (int64_t)kv_dim_,
                                          (int64_t)head_dim_, 1};
                aclTensor *t_k_s = tensor_strided(k_cache_dev_[il], 4, kv_shape,
                                                    kv_strides, ACL_FLOAT16);
                aclTensor *t_v_s = tensor_strided(v_cache_dev_[il], 4, kv_shape,
                                                    kv_strides, ACL_FLOAT16);
                aclTensorList *t_kl = g_cann.aclCreateTensorList(&t_k_s, 1);
                aclTensorList *t_vl = g_cann.aclCreateTensorList(&t_v_s, 1);

                uint64_t fa_ws = 0;
                aclOpExecutor *fa_exec = nullptr;
                char layout[5] = {'B','S','N','D',0};
                double scale = 1.0 / sqrt((double)head_dim_);
                ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
                    t_q_s, t_kl, t_vl,
                    /*pseShift*/ nullptr, /*attenMask*/ nullptr,
                    /*actSeqLen*/ nullptr, /*actSeqLenKv*/ nullptr,
                    /*deqScale1*/ nullptr, /*quantScale1*/ nullptr,
                    /*deqScale2*/ nullptr, /*quantScale2*/ nullptr,
                    /*quantOffset2*/ nullptr,
                    /*antiquantScale*/ nullptr, /*antiquantOffset*/ nullptr,
                    /*blockTable*/ nullptr,
                    /*queryPaddingSize*/ nullptr, /*kvPaddingSize*/ nullptr,
                    /*keyAntiquantScale*/ nullptr, /*keyAntiquantOffset*/ nullptr,
                    /*valueAntiquantScale*/ nullptr, /*valueAntiquantOffset*/ nullptr,
                    /*keySharedPrefix*/ nullptr, /*valueSharedPrefix*/ nullptr,
                    /*actualSharedPrefixLen*/ nullptr,
                    /*numHeads*/ (int64_t)n_heads_,
                    /*scaleValue*/ scale,
                    /*preTokens*/ (int64_t)65535,
                    /*nextTokens*/ (int64_t)65535,
                    /*inputLayout*/ layout,
                    /*numKeyValueHeads*/ (int64_t)n_kv_,
                    /*sparseMode*/ (int64_t)0,
                    /*innerPrecise*/ (int64_t)0,
                    /*blockSize*/ (int64_t)0,
                    /*antiquantMode*/ (int64_t)0,
                    /*softmaxLseFlag*/ false,
                    /*keyAntiquantMode*/ (int64_t)0,
                    /*valueAntiquantMode*/ (int64_t)0,
                    /*attentionOut*/ t_attn_s,
                    /*softmaxLse*/ nullptr,
                    &fa_ws, &fa_exec));
                ensure_workspace_(fa_ws);
                void *ws = fa_ws > 0 ? workspace_dev_ : nullptr;
                ACL_CHECK_RET(g_cann.aclnnFusedInferAttentionScoreV2(
                    ws, fa_ws, fa_exec, stream_));
                g_cann.aclDestroyTensorList(t_kl);
                g_cann.aclDestroyTensorList(t_vl);
                g_cann.aclDestroyTensor(t_q_s);
                g_cann.aclDestroyTensor(t_attn_s);
            }
        }

        // --- O projection (batched): o_out [seq_len, n_embd] = attn [seq_len, q_dim]
        //     @ o_proj^T [q_dim, n_embd] ---
        // A4b: W8 branch mirrors decode's o_proj dispatch.
        if (w8_applied_) {
            aclTensor *t_attn = tensor_2d(q_batch_dev_, seq_len, q_dim_,
                                            ACL_FLOAT16);
            aclTensor *t_o    = tensor_2d(o_out_batch_dev_, seq_len, n_embd_,
                                            ACL_FLOAT16);
            w8_matmul_(t_attn, lw.o_proj_w_i8, lw.o_proj_scale,
                        n_embd_, q_dim_, t_o);
            g_cann.aclDestroyTensor(t_attn);
            g_cann.aclDestroyTensor(t_o);
        } else {
            aclTensor *t_attn = tensor_2d(q_batch_dev_, seq_len, q_dim_,
                                            ACL_FLOAT16);
            aclTensor *t_wo_T = weight_T_tensor(lw.o_proj_w, n_embd_, q_dim_);
            aclTensor *t_o    = tensor_2d(o_out_batch_dev_, seq_len, n_embd_,
                                            ACL_FLOAT16);
            CANN_MATMUL(t_attn, t_wo_T, t_o);
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
        // A4b: W8 branch mirrors decode's FFN dispatch. Silu/InplaceMul on
        // the flat gate/up views is dtype-invariant (F16 in both paths).
        if (w8_applied_) {
            aclTensor *t_normed  = tensor_2d(normed_batch_dev_, seq_len, n_embd_, ACL_FLOAT16);
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
            w8_matmul_(t_normed, lw.gate_proj_w_i8, lw.gate_proj_scale,
                        inter_,  n_embd_, t_gate);
            w8_matmul_(t_normed, lw.up_proj_w_i8,   lw.up_proj_scale,
                        inter_,  n_embd_, t_up);
            CANN_OP(Silu, t_gate_fl, t_gate_fl);
            CANN_OP(InplaceMul, t_gate_fl, t_up_fl);
            w8_matmul_(t_gate, lw.down_proj_w_i8, lw.down_proj_scale,
                        n_embd_, inter_, t_ffn);
            g_cann.aclDestroyTensor(t_normed);
            g_cann.aclDestroyTensor(t_gate);
            g_cann.aclDestroyTensor(t_up);
            g_cann.aclDestroyTensor(t_gate_fl);
            g_cann.aclDestroyTensor(t_up_fl);
            g_cann.aclDestroyTensor(t_ffn);
        } else {
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
            CANN_MATMUL(t_normed, t_wg_T, t_gate);
            CANN_MATMUL(t_normed, t_wu_T, t_up);
            CANN_OP(Silu, t_gate_fl, t_gate_fl);
            CANN_OP(InplaceMul, t_gate_fl, t_up_fl);
            CANN_MATMUL(t_gate, t_wd_T, t_ffn);
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

    // Cast ONLY the last row's normed output to F32. Under Phase 4 capture
    // this Cast is the last recorded op of the graph; the subsequent
    // SynchronizeStream + D2H run AFTER CaptureEnd (a captured graph cannot
    // contain a stream sync or host memcpy). If last_hidden_out is null,
    // we still want to record the last-row Cast (an idempotent pure-device
    // op) so replay produces a usable graph even in throwaway-hidden calls;
    // skip that case here since the caller does not need the hidden.
    if (last_hidden_out) {
        uint16_t *last_row = (uint16_t *)normed_batch_dev_ +
                               (size_t)(seq_len - 1) * n_embd_;
        aclTensor *t_last_f16 = tensor_1d(last_row, n_embd_, ACL_FLOAT16);
        aclTensor *t_last_f32 = tensor_1d(output_stage_f32_dev_, n_embd_,
                                            ACL_FLOAT);
        CANN_OP(Cast, t_last_f16, ACL_FLOAT, t_last_f32);
        g_cann.aclDestroyTensor(t_last_f16);
        g_cann.aclDestroyTensor(t_last_f32);
    }

    // ---- A4c Phase 4 capture end (if captured this call) ------------------
    if (prefill_capture_this_call) {
        aclmdlRI g = nullptr;
        aclError end_err = g_cann.aclmdlRICaptureEnd(stream_, &g);
        if (end_err != 0 || g == nullptr) {
            fprintf(stderr,
                    "[talker_cann] prefill aclmdlRICaptureEnd bucket=%d "
                    "(seq_len=%d) failed (%d), graph=%p — staying eager "
                    "for this bucket: %s\n",
                    prefill_bucket_idx, seq_len, end_err, (void *)g,
                    g_cann.aclGetRecentErrMsg
                        ? g_cann.aclGetRecentErrMsg() : "<n/a>");
            if (g) g_cann.aclmdlRIDestroy(g);
        } else {
            prefill_graphs_[prefill_bucket_idx] = g;
            printf("[talker_cann] A4c Phase 4 captured prefill graph bucket=%d "
                   "(seq_len=%d). Subsequent matching calls will replay.\n",
                   prefill_bucket_idx, seq_len);
        }
    }

    // Post-capture sync + D2H.
    if (last_hidden_out) {
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
