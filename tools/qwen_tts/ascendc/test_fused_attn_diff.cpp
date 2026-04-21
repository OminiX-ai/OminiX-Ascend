// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// test_fused_attn_diff.cpp — W4.1.2 offline numerical-validation harness.
//
// Extracts "one layer's" worth of input hidden + Q/K/V/O weights (random
// but deterministic via seeded RNG) into a synthetic fixture, dispatches
// BOTH the fused AscendC kernel AND the stock aclnn op chain on the same
// fixture, and reports max-abs-diff on the residual-update output.
//
// Gate (§5 W4.1.2): max-abs-diff ≤ 1e-2 (F16 noise). If > 1e-2, report
// what differs + where.
//
// The harness uses random fixtures rather than extracting real weights
// because the kernel's math is weight-agnostic and the real-weight
// validation happens at W4.1.4 (live correctness gate under canonical
// mayun xvec zh). W4.1.2's job is to pin the kernel math independently
// of the engine wiring.
//
// Shapes come from cp_cann_engine.h CP config: cp_hidden=1024, q_dim=2048,
// kv_dim=1024, head_dim=128, n_heads=16, n_kv=8, seq_len=8 (a mid-pos
// we can exercise the full KV cache path at).
//
// Build: cmake --build build-w1 --target test_fused_attn_diff
// Run:   build-w1/bin/test_fused_attn_diff
// =============================================================================

#include "cp_cann_symbols.h"
#include "ggml-backend.h"

#include "aclrtlaunch_fused_attn_sublayer.h"

#include <acl/acl.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

// -------------------- constants (match engine + kernel) --------------------
static constexpr int kCpHidden = 1024;
static constexpr int kQDim     = 2048;
static constexpr int kKvDim    = 1024;
static constexpr int kHeadDim  = 128;
static constexpr int kNHeads   = 16;
static constexpr int kNKv      = 8;
static constexpr int kGroup    = kNHeads / kNKv;
static constexpr int kMaxSeq   = 17;
static constexpr int kSeqLen   = 8;    // current pos = 7, so seq_len = pos + 1 = 8
static constexpr float kEps    = 1e-6f;

// -------------------- helpers ---------------------
#define ACL_CHK(s) do { auto _e = (s); if (_e != 0) { \
    fprintf(stderr, "ACL error %d at %s:%d\n", (int)_e, __FILE__, __LINE__); \
    std::exit(1); } } while (0)

static uint16_t f16_from_f32(float v) {
    __fp16 h = (__fp16)v; uint16_t b; std::memcpy(&b, &h, 2); return b;
}
static float f32_from_f16(uint16_t b) {
    __fp16 h; std::memcpy(&h, &b, 2); return (float)h;
}

// Upload a host F16 vector to a freshly-allocated NPU buffer.
static void* upload_f16(const std::vector<uint16_t> &host) {
    void *dev = nullptr;
    size_t bytes = host.size() * 2;
    ACL_CHK(g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHK(g_cann.aclrtMemcpy(dev, bytes, host.data(), bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE));
    return dev;
}
static void* upload_i8(const std::vector<int8_t> &host) {
    void *dev = nullptr;
    size_t bytes = host.size();
    ACL_CHK(g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHK(g_cann.aclrtMemcpy(dev, bytes, host.data(), bytes,
                                 ACL_MEMCPY_HOST_TO_DEVICE));
    return dev;
}
static std::vector<uint16_t> download_f16(void *dev, size_t n_elem) {
    std::vector<uint16_t> host(n_elem);
    ACL_CHK(g_cann.aclrtMemcpy(host.data(), n_elem * 2, dev, n_elem * 2,
                                 ACL_MEMCPY_DEVICE_TO_HOST));
    return host;
}

// W8-quantize a F32 weight row-wise: scale[r] = max|row|/127, then
// clamp-round(row / scale) → int8. Matches cp_cann_engine.cpp
// w8_calibrate_weight_.
static void w8_calibrate(const std::vector<float> &w_f32,
                         int64_t rows, int64_t cols,
                         std::vector<int8_t> &w_i8,
                         std::vector<uint16_t> &scale_f16) {
    w_i8.resize((size_t)rows * cols);
    scale_f16.resize((size_t)rows);
    for (int64_t r = 0; r < rows; ++r) {
        const float *row = w_f32.data() + (size_t)r * cols;
        float max_abs = 0.0f;
        for (int64_t j = 0; j < cols; ++j) {
            float v = std::fabs(row[j]);
            if (v > max_abs) max_abs = v;
        }
        float scale = (max_abs > 0.0f) ? (max_abs / 127.0f) : 1.0f;
        float inv = 1.0f / scale;
        for (int64_t j = 0; j < cols; ++j) {
            int ir = (int)std::rint(row[j] * inv);
            if (ir >  127) ir =  127;
            if (ir < -127) ir = -127;
            w_i8[(size_t)r * cols + j] = (int8_t)ir;
        }
        scale_f16[(size_t)r] = f16_from_f32(scale);
    }
}

// -------------------- reference implementation (host-side, F32) --------
// Runs the stock op chain end-to-end on the host in F32 for the gold
// reference. Using host-F32 isolates kernel vs aclnn-on-device
// discrepancies from kernel vs aclnn-on-device-precision (both sides
// lose precision to F16 + W8 the same way, so we compare both against
// a common F32 ground truth).
//
// Returns the updated residual (cur') in F16 bits.
static std::vector<uint16_t> run_reference(
    const std::vector<uint16_t> &residual_f16,
    const std::vector<uint16_t> &in_ln_f16,
    const std::vector<int8_t>   &wq_i8,
    const std::vector<uint16_t> &wq_scale_f16,
    const std::vector<int8_t>   &wk_i8,
    const std::vector<uint16_t> &wk_scale_f16,
    const std::vector<int8_t>   &wv_i8,
    const std::vector<uint16_t> &wv_scale_f16,
    const std::vector<int8_t>   &wo_i8,
    const std::vector<uint16_t> &wo_scale_f16,
    const std::vector<uint16_t> &q_norm_f16,
    const std::vector<uint16_t> &k_norm_f16,
    const std::vector<uint16_t> &rope_cos_f16,
    const std::vector<uint16_t> &rope_sin_f16,
    std::vector<uint16_t>       &k_cache_f16,  // [kSeqLen, kKvDim]; updated
    std::vector<uint16_t>       &v_cache_f16)
{
    // 1. RmsNorm(residual, in_ln) → normed.
    std::vector<float> x(kCpHidden), gamma(kCpHidden);
    for (int i = 0; i < kCpHidden; ++i) x[i]     = f32_from_f16(residual_f16[i]);
    for (int i = 0; i < kCpHidden; ++i) gamma[i] = f32_from_f16(in_ln_f16[i]);
    float sq = 0.0f;
    for (int i = 0; i < kCpHidden; ++i) sq += x[i] * x[i];
    float rstd = 1.0f / std::sqrt(sq / (float)kCpHidden + kEps);
    std::vector<float> normed(kCpHidden);
    for (int i = 0; i < kCpHidden; ++i)
        normed[i] = x[i] * rstd * gamma[i];

    // 2. QKV: q = normed @ (wq_i8 * wq_scale)^T   shape [q_dim]
    auto w8_mm = [&](const std::vector<int8_t>   &w_i8,
                     const std::vector<uint16_t> &w_scale_f16,
                     int out_n) -> std::vector<float> {
        std::vector<float> y(out_n, 0.0f);
        for (int r = 0; r < out_n; ++r) {
            float scale = f32_from_f16(w_scale_f16[r]);
            float acc   = 0.0f;
            for (int c = 0; c < kCpHidden; ++c) {
                float w = (float)w_i8[(size_t)r * kCpHidden + c] * scale;
                acc += normed[c] * w;
            }
            y[r] = acc;
        }
        return y;
    };
    std::vector<float> q = w8_mm(wq_i8, wq_scale_f16, kQDim);
    std::vector<float> k = w8_mm(wk_i8, wk_scale_f16, kKvDim);
    std::vector<float> v = w8_mm(wv_i8, wv_scale_f16, kKvDim);

    // 3. Q/K norm (per-head).
    auto head_rmsnorm = [&](std::vector<float> &t, int n_heads,
                             const std::vector<uint16_t> &gamma_f16) {
        for (int h = 0; h < n_heads; ++h) {
            float *row = t.data() + (size_t)h * kHeadDim;
            float sq = 0.0f;
            for (int d = 0; d < kHeadDim; ++d) sq += row[d] * row[d];
            float r = 1.0f / std::sqrt(sq / (float)kHeadDim + kEps);
            for (int d = 0; d < kHeadDim; ++d)
                row[d] = row[d] * r * f32_from_f16(gamma_f16[d]);
        }
    };
    head_rmsnorm(q, kNHeads, q_norm_f16);
    head_rmsnorm(k, kNKv,    k_norm_f16);

    // 4. RoPE(Q) / RoPE(K): NEOX rotate_half.
    const int half_d = kHeadDim / 2;
    auto apply_rope = [&](std::vector<float> &t, int n_heads) {
        for (int h = 0; h < n_heads; ++h) {
            float *row = t.data() + (size_t)h * kHeadDim;
            for (int j = 0; j < half_d; ++j) {
                float c  = f32_from_f16(rope_cos_f16[j]);
                float s  = f32_from_f16(rope_sin_f16[j]);
                float x0 = row[j];
                float x1 = row[j + half_d];
                row[j]          = x0 * c - x1 * s;
                row[j + half_d] = x0 * s + x1 * c;
            }
        }
    };
    apply_rope(q, kNHeads);
    apply_rope(k, kNKv);

    // 5. Write K/V into cache at slot kSeqLen - 1.
    int slot = kSeqLen - 1;
    for (int i = 0; i < kKvDim; ++i) {
        k_cache_f16[(size_t)slot * kKvDim + i] = f16_from_f32(k[i]);
        v_cache_f16[(size_t)slot * kKvDim + i] = f16_from_f32(v[i]);
    }

    // 6. Attention: out_h = softmax(Q_h @ K[:seq,kv_h]^T / sqrt(d)) @ V[:seq,kv_h]
    std::vector<float> attn(kQDim);
    const float scale = 1.0f / std::sqrt((float)kHeadDim);
    for (int h = 0; h < kNHeads; ++h) {
        int kv_h  = h / kGroup;
        float *q_h  = q.data() + (size_t)h    * kHeadDim;
        // scores[t] = dot(Q_h, K[t, kv_h])
        std::vector<float> scores(kSeqLen);
        float max_s = -1e30f;
        for (int t = 0; t < kSeqLen; ++t) {
            float acc = 0.0f;
            size_t off = (size_t)t * kKvDim + (size_t)kv_h * kHeadDim;
            for (int d = 0; d < kHeadDim; ++d)
                acc += q_h[d] * f32_from_f16(k_cache_f16[off + d]);
            scores[t] = acc * scale;
            if (scores[t] > max_s) max_s = scores[t];
        }
        // softmax
        float sum = 0.0f;
        for (int t = 0; t < kSeqLen; ++t) {
            scores[t] = std::exp(scores[t] - max_s);
            sum += scores[t];
        }
        float inv_sum = 1.0f / sum;
        for (int t = 0; t < kSeqLen; ++t) scores[t] *= inv_sum;
        // out_h = Σ scores * V
        for (int d = 0; d < kHeadDim; ++d) {
            float acc = 0.0f;
            for (int t = 0; t < kSeqLen; ++t) {
                size_t off = (size_t)t * kKvDim + (size_t)kv_h * kHeadDim;
                acc += scores[t] * f32_from_f16(v_cache_f16[off + d]);
            }
            attn[(size_t)h * kHeadDim + d] = acc;
        }
    }

    // 7. O = W8Mm(attn, Wo).  Wo is [cp_hidden, q_dim] row-major.
    std::vector<float> o(kCpHidden, 0.0f);
    for (int r = 0; r < kCpHidden; ++r) {
        float scale_o = f32_from_f16(wo_scale_f16[r]);
        float acc = 0.0f;
        for (int c = 0; c < kQDim; ++c) {
            float w = (float)wo_i8[(size_t)r * kQDim + c] * scale_o;
            acc += attn[c] * w;
        }
        o[r] = acc;
    }

    // 8. residual' = residual + O (F16 round-trip to match kernel).
    std::vector<uint16_t> out(kCpHidden);
    for (int i = 0; i < kCpHidden; ++i) {
        float r = f32_from_f16(residual_f16[i]) + o[i];
        out[i] = f16_from_f32(r);
    }
    return out;
}

// ---------------------- main ----------------------
int main() {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "cp_cann symbol load failed\n");
        return 1;
    }
    // Bring up ggml-cann (for ACL init order).
    {
        ggml_backend_reg_t reg = ggml_backend_reg_by_name("CANN");
        if (!reg) { fprintf(stderr, "CANN reg missing\n"); return 1; }
        ggml_backend_dev_t dev = ggml_backend_reg_dev_get(reg, 0);
        ggml_backend_t be = ggml_backend_dev_init(dev, nullptr);
        if (!be) { fprintf(stderr, "CANN init failed\n"); return 1; }
    }
    ACL_CHK(g_cann.aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHK(g_cann.aclrtCreateStream(&stream));

    // --- Synthesise fixture ---
    std::mt19937 rng(0xC0FFEE);
    std::normal_distribution<float> gauss(0.0f, 0.1f);

    auto rand_f16 = [&](size_t n) {
        std::vector<uint16_t> v(n);
        for (auto &x : v) x = f16_from_f32(gauss(rng));
        return v;
    };
    auto rand_f32 = [&](size_t n) {
        std::vector<float> v(n);
        for (auto &x : v) x = gauss(rng);
        return v;
    };

    auto residual       = rand_f16(kCpHidden);
    auto in_ln_gamma    = rand_f16(kCpHidden);    // LN gamma (F16)
    auto q_norm_gamma   = rand_f16(kHeadDim);
    auto k_norm_gamma   = rand_f16(kHeadDim);
    // RoPE tables: duplicated-half (cos[j] == cos[j+half]), matching engine.
    std::vector<uint16_t> rope_cos(kHeadDim), rope_sin(kHeadDim);
    {
        const int half = kHeadDim / 2;
        const float pos = 7.0f;  // same as seq_len-1
        const float theta = 1000000.0f;
        for (int j = 0; j < half; ++j) {
            float freq = 1.0f / std::pow(theta, (float)(2 * j) / (float)kHeadDim);
            float angle = pos * freq;
            rope_cos[j]         = f16_from_f32(std::cos(angle));
            rope_cos[j + half]  = rope_cos[j];
            rope_sin[j]         = f16_from_f32(std::sin(angle));
            rope_sin[j + half]  = rope_sin[j];
        }
    }
    // Random fake K/V cache for positions [0, kSeqLen-2]; kSeqLen-1 will be
    // written by kernel / reference.
    auto k_cache = rand_f16(kSeqLen * kKvDim);
    auto v_cache = rand_f16(kSeqLen * kKvDim);

    // Weights: synthesise as F32, then W8-calibrate.
    auto wq_f32 = rand_f32((size_t)kQDim  * kCpHidden);
    auto wk_f32 = rand_f32((size_t)kKvDim * kCpHidden);
    auto wv_f32 = rand_f32((size_t)kKvDim * kCpHidden);
    auto wo_f32 = rand_f32((size_t)kCpHidden * kQDim);
    std::vector<int8_t>   wq_i8, wk_i8, wv_i8, wo_i8;
    std::vector<uint16_t> wq_sc, wk_sc, wv_sc, wo_sc;
    w8_calibrate(wq_f32, kQDim,     kCpHidden, wq_i8, wq_sc);
    w8_calibrate(wk_f32, kKvDim,    kCpHidden, wk_i8, wk_sc);
    w8_calibrate(wv_f32, kKvDim,    kCpHidden, wv_i8, wv_sc);
    w8_calibrate(wo_f32, kCpHidden, kQDim,     wo_i8, wo_sc);

    printf("[diff] fixture: cp_hidden=%d q_dim=%d kv_dim=%d head_dim=%d seq_len=%d\n",
           kCpHidden, kQDim, kKvDim, kHeadDim, kSeqLen);

    // --- Reference (host F32 gold) ---
    // We save a copy of k/v cache before reference because reference
    // writes slot kSeqLen-1.
    auto k_cache_ref = k_cache;
    auto v_cache_ref = v_cache;
    auto residual_ref = run_reference(residual, in_ln_gamma,
                                       wq_i8, wq_sc, wk_i8, wk_sc,
                                       wv_i8, wv_sc, wo_i8, wo_sc,
                                       q_norm_gamma, k_norm_gamma,
                                       rope_cos, rope_sin,
                                       k_cache_ref, v_cache_ref);

    // --- Kernel path (on-device) ---
    // Clone k/v so kernel gets a clean slate (it will write slot kSeqLen-1).
    auto k_cache_ker = k_cache;
    auto v_cache_ker = v_cache;
    auto residual_ker_host = residual;

    // Upload fixture.
    void *d_residual  = upload_f16(residual_ker_host);
    void *d_normed    = upload_f16(std::vector<uint16_t>(kCpHidden, 0));
    void *d_in_ln     = upload_f16(in_ln_gamma);
    void *d_wq_i8     = upload_i8 (wq_i8);
    void *d_wq_scale  = upload_f16(wq_sc);
    void *d_wk_i8     = upload_i8 (wk_i8);
    void *d_wk_scale  = upload_f16(wk_sc);
    void *d_wv_i8     = upload_i8 (wv_i8);
    void *d_wv_scale  = upload_f16(wv_sc);
    void *d_wo_i8     = upload_i8 (wo_i8);
    void *d_wo_scale  = upload_f16(wo_sc);
    void *d_q_norm    = upload_f16(q_norm_gamma);
    void *d_k_norm    = upload_f16(k_norm_gamma);
    void *d_rope_cos  = upload_f16(rope_cos);
    void *d_rope_sin  = upload_f16(rope_sin);
    void *d_k_cache   = upload_f16(k_cache_ker);
    void *d_v_cache   = upload_f16(v_cache_ker);
    void *d_o_out     = upload_f16(std::vector<uint16_t>(kCpHidden, 0));
    void *d_scratch_q = upload_f16(std::vector<uint16_t>(kQDim,     0));
    void *d_scratch_s = upload_f16(std::vector<uint16_t>(kNHeads * kMaxSeq, 0));

    uint32_t eps_bits;
    { float e = kEps; std::memcpy(&eps_bits, &e, 4); }

    uint32_t rc = aclrtlaunch_fused_attn_sublayer(
        /*blockDim=*/1, stream,
        d_residual, d_normed, d_in_ln,
        d_wq_i8, d_wq_scale, d_wk_i8, d_wk_scale, d_wv_i8, d_wv_scale,
        d_wo_i8, d_wo_scale, d_q_norm, d_k_norm,
        d_rope_cos, d_rope_sin, d_k_cache, d_v_cache,
        d_o_out, d_scratch_q, d_scratch_s,
        /*seq_len=*/kSeqLen, /*eps_bits=*/eps_bits, /*opts=*/0);
    if (rc != 0) {
        fprintf(stderr, "[diff] aclrtlaunch_fused_attn_sublayer rc=%u\n", rc);
        return 2;
    }
    ACL_CHK(g_cann.aclrtSynchronizeStream(stream));

    auto residual_ker = download_f16(d_residual, kCpHidden);
    auto k_cache_out  = download_f16(d_k_cache,  kSeqLen * kKvDim);
    auto v_cache_out  = download_f16(d_v_cache,  kSeqLen * kKvDim);

    // --- Diff ---
    auto max_abs_diff = [](const std::vector<uint16_t> &a,
                            const std::vector<uint16_t> &b,
                            const char *tag, size_t print_first = 0) {
        float max_diff = 0.0f;
        int  max_idx  = -1;
        for (size_t i = 0; i < a.size(); ++i) {
            float va = f32_from_f16(a[i]);
            float vb = f32_from_f16(b[i]);
            float d  = std::fabs(va - vb);
            if (d > max_diff) { max_diff = d; max_idx = (int)i; }
        }
        printf("[diff] %s: max_abs_diff=%.6f at idx=%d\n", tag, max_diff, max_idx);
        if (print_first) {
            printf("[diff] %s first %zu: ref vs ker\n", tag, print_first);
            for (size_t i = 0; i < print_first && i < a.size(); ++i) {
                printf("       [%zu] %+.6f vs %+.6f  (delta %+.6f)\n",
                       i, f32_from_f16(a[i]), f32_from_f16(b[i]),
                       f32_from_f16(a[i]) - f32_from_f16(b[i]));
            }
        }
        return max_diff;
    };

    float d_resid = max_abs_diff(residual_ref, residual_ker, "residual", 8);
    float d_kcache = max_abs_diff(k_cache_ref, k_cache_out, "k_cache", 0);
    float d_vcache = max_abs_diff(v_cache_ref, v_cache_out, "v_cache", 0);

    const float gate = 1e-2f;
    bool pass = (d_resid <= gate) && (d_kcache <= gate) && (d_vcache <= gate);
    printf("[diff] gate <= %g; result: %s\n", gate, pass ? "PASS" : "FAIL");
    return pass ? 0 : 3;
}
