// ============================================================================
// Q2.3 Phase 3 smoke — single DiT block forward, cos_sim vs CPU reference.
//
// Harness:
//   - Boots ImageDiffusionEngine via init_for_smoke() (no GGUF load).
//   - Synthesizes deterministic random F16 weights for block 0; uploads them
//     through the F16-fallback matmul path (scale_dev=null per engine
//     dispatch branch).
//   - Generates random F16 img_hidden, txt_hidden, t_emb on host.
//   - Dispatches forward_block_test(il=0) on NPU.
//   - Computes the same block's forward in F32 on host, using the same
//     numerical sequence the NPU runs.
//   - Reports cos_sim(img_out_npu, img_out_cpu_ref), cos_sim(txt_out, ...)
//     plus min/max, NaN count, wall time.
//
// Phase 3 gate: cos_sim > 0.99 for both streams, no NaN, output shape
// matches input.
//
// Build on ac03:
//   cd tools/probes/qie_q23_block_smoke && bash build_and_run.sh
// ============================================================================

#include "../../qwen_image_edit/native/image_diffusion_engine.h"
#include "../../qwen_tts/cp_cann_symbols.h"

#include <acl/acl.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace ominix_qie;

// ---------------------------------------------------------------------------
// F16 <-> F32 helpers (arm64 native __fp16).
// ---------------------------------------------------------------------------
static inline uint16_t f32_to_f16(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}
static inline float f16_to_f32(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

// ---------------------------------------------------------------------------
// Deterministic random fill helpers.
// ---------------------------------------------------------------------------
static void fill_random_f16(std::vector<uint16_t> &out, size_t n,
                             float amp, uint64_t seed) {
    out.assign(n, 0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16(dist(rng));
}

static void fill_random_f32(std::vector<float> &out, size_t n,
                             float amp, uint64_t seed) {
    out.assign(n, 0.0f);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n; ++i) out[i] = dist(rng);
}

// ---------------------------------------------------------------------------
// Upload a host F16 buffer to a fresh device allocation. Returns dev pointer
// or nullptr on failure. For F16-fallback matmul path: GGUF stores weights
// as physical [N=out, K=in] row-major (K contiguous). Our host generator
// emits the same layout.
// ---------------------------------------------------------------------------
static void *upload_f16(const uint16_t *host, size_t n) {
    void *dev = nullptr;
    size_t bytes = n * sizeof(uint16_t);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[smoke] aclrtMalloc(%zu) err=%d\n", bytes, (int)err);
        return nullptr;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        fprintf(stderr, "[smoke] H2D memcpy err=%d\n", (int)err);
        g_cann.aclrtFree(dev);
        return nullptr;
    }
    return dev;
}

static void *upload_f32(const float *host, size_t n) {
    void *dev = nullptr;
    size_t bytes = n * sizeof(float);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) return nullptr;
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) { g_cann.aclrtFree(dev); return nullptr; }
    return dev;
}

// ---------------------------------------------------------------------------
// Weight layouts for the probe:
//   matmul weight  — GGUF-style physical [N, K] row-major (K contig)
//                    stored as F16. Element count = K * N. scale_dev stays
//                    null; engine dispatches aclnnMm with the transposed
//                    view (`[K, N]` strides (1, K)).
//   bias           — F16 [N]
//   rmsnorm gamma  — F32 [head_dim]
//   pe             — F16 [seq_pe_max, head_dim/2, 2, 2]
// ---------------------------------------------------------------------------
struct HostWeights {
    // img side
    std::vector<uint16_t> to_q_w, to_k_w, to_v_w, to_out_w;
    std::vector<uint16_t> to_q_b, to_k_b, to_v_b, to_out_b;
    // txt side
    std::vector<uint16_t> add_q_w, add_k_w, add_v_w, to_add_out_w;
    std::vector<uint16_t> add_q_b, add_k_b, add_v_b, to_add_out_b;
    // rmsnorm gammas (F32, per-head_dim)
    std::vector<float> norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    // modulation (hidden → 6·hidden)
    std::vector<uint16_t> img_mod_w, txt_mod_w;
    std::vector<uint16_t> img_mod_b, txt_mod_b;
    // FFN
    std::vector<uint16_t> img_ff_up_w, img_ff_down_w;
    std::vector<uint16_t> img_ff_up_b, img_ff_down_b;
    std::vector<uint16_t> txt_ff_up_w, txt_ff_down_w;
    std::vector<uint16_t> txt_ff_up_b, txt_ff_down_b;
};

static void gen_host_weights(const ImageDiffusionConfig &cfg, HostWeights &w,
                              uint64_t seed) {
    const int64_t H  = cfg.hidden_size;
    const int64_t HD = cfg.head_dim;
    const int64_t FF = (int64_t)H * cfg.ff_mult;
    // Weight amplitude: small (~1/sqrt(H) order) to keep pre-softmax stable.
    const float W_AMP = 1.0f / std::sqrt((float)H);
    const float B_AMP = 0.02f;
    const float G_AMP = 0.1f;  // norm gammas around zero (GGUF tends to
                                // have pre-trained gammas ~1, but keep small
                                // + add bias 1 implicit in aclnnRmsNorm
                                // formula y = gamma * x_normed). Actually
                                // aclnnRmsNorm uses weight=gamma directly
                                // (no +1). For smoke, keep small.
    auto rw = [&](std::vector<uint16_t> &v, int64_t N, int64_t K, uint64_t s) {
        fill_random_f16(v, (size_t)N * K, W_AMP, s);
    };
    auto rb = [&](std::vector<uint16_t> &v, int64_t N, uint64_t s) {
        fill_random_f16(v, (size_t)N, B_AMP, s);
    };
    auto rg = [&](std::vector<float> &v, int64_t N, uint64_t s) {
        fill_random_f32(v, (size_t)N, G_AMP, s);
        for (auto &x : v) x += 1.0f;   // bias gamma around 1 for stable RMS
    };
    rw(w.to_q_w,   H, H, seed + 1);   rb(w.to_q_b, H, seed + 2);
    rw(w.to_k_w,   H, H, seed + 3);   rb(w.to_k_b, H, seed + 4);
    rw(w.to_v_w,   H, H, seed + 5);   rb(w.to_v_b, H, seed + 6);
    rw(w.to_out_w, H, H, seed + 7);   rb(w.to_out_b, H, seed + 8);
    rw(w.add_q_w,   H, H, seed + 11); rb(w.add_q_b, H, seed + 12);
    rw(w.add_k_w,   H, H, seed + 13); rb(w.add_k_b, H, seed + 14);
    rw(w.add_v_w,   H, H, seed + 15); rb(w.add_v_b, H, seed + 16);
    rw(w.to_add_out_w, H, H, seed + 17); rb(w.to_add_out_b, H, seed + 18);
    rg(w.norm_q_w,       HD, seed + 21);
    rg(w.norm_k_w,       HD, seed + 22);
    rg(w.norm_added_q_w, HD, seed + 23);
    rg(w.norm_added_k_w, HD, seed + 24);
    // Modulation weight has a small amplitude to keep (1 + scale) bounded.
    // We also bias img_mod_b / txt_mod_b so that scale+shift stay small.
    {
        const float mod_w_amp = 0.01f;
        fill_random_f16(w.img_mod_w, (size_t)6 * H * H, mod_w_amp, seed + 31);
        fill_random_f16(w.txt_mod_w, (size_t)6 * H * H, mod_w_amp, seed + 32);
        fill_random_f16(w.img_mod_b, (size_t)6 * H, 0.01f,         seed + 33);
        fill_random_f16(w.txt_mod_b, (size_t)6 * H, 0.01f,         seed + 34);
    }
    rw(w.img_ff_up_w,   FF, H, seed + 41); rb(w.img_ff_up_b,   FF, seed + 42);
    rw(w.img_ff_down_w, H, FF, seed + 43); rb(w.img_ff_down_b, H,  seed + 44);
    rw(w.txt_ff_up_w,   FF, H, seed + 51); rb(w.txt_ff_up_b,   FF, seed + 52);
    rw(w.txt_ff_down_w, H, FF, seed + 53); rb(w.txt_ff_down_b, H,  seed + 54);
}

static bool populate_layer(DiTLayerWeights &lw, const HostWeights &w) {
    auto u16 = [&](const std::vector<uint16_t> &v) {
        return upload_f16(v.data(), v.size());
    };
    auto uf32 = [&](const std::vector<float> &v) {
        return upload_f32(v.data(), v.size());
    };
    // Matmul weights: scale=null (F16 fallback path).
    lw.to_q_w_q4   = u16(w.to_q_w);   lw.to_q_scale   = nullptr;
    lw.to_q_b      = u16(w.to_q_b);
    lw.to_k_w_q4   = u16(w.to_k_w);   lw.to_k_scale   = nullptr;
    lw.to_k_b      = u16(w.to_k_b);
    lw.to_v_w_q4   = u16(w.to_v_w);   lw.to_v_scale   = nullptr;
    lw.to_v_b      = u16(w.to_v_b);
    lw.to_out_0_w_q4 = u16(w.to_out_w); lw.to_out_0_scale = nullptr;
    lw.to_out_0_b  = u16(w.to_out_b);
    lw.add_q_w_q4  = u16(w.add_q_w);  lw.add_q_scale  = nullptr;
    lw.add_q_b     = u16(w.add_q_b);
    lw.add_k_w_q4  = u16(w.add_k_w);  lw.add_k_scale  = nullptr;
    lw.add_k_b     = u16(w.add_k_b);
    lw.add_v_w_q4  = u16(w.add_v_w);  lw.add_v_scale  = nullptr;
    lw.add_v_b     = u16(w.add_v_b);
    lw.to_add_out_w_q4 = u16(w.to_add_out_w);
    lw.to_add_out_scale = nullptr;
    lw.to_add_out_b = u16(w.to_add_out_b);

    lw.norm_q_w       = uf32(w.norm_q_w);
    lw.norm_k_w       = uf32(w.norm_k_w);
    lw.norm_added_q_w = uf32(w.norm_added_q_w);
    lw.norm_added_k_w = uf32(w.norm_added_k_w);

    lw.img_mod_w_q4  = u16(w.img_mod_w);  lw.img_mod_scale = nullptr;
    lw.img_mod_b     = u16(w.img_mod_b);
    lw.txt_mod_w_q4  = u16(w.txt_mod_w);  lw.txt_mod_scale = nullptr;
    lw.txt_mod_b     = u16(w.txt_mod_b);

    lw.img_ff_up_w_q4   = u16(w.img_ff_up_w);   lw.img_ff_up_scale = nullptr;
    lw.img_ff_up_b      = u16(w.img_ff_up_b);
    lw.img_ff_down_w_q4 = u16(w.img_ff_down_w); lw.img_ff_down_scale = nullptr;
    lw.img_ff_down_b    = u16(w.img_ff_down_b);
    lw.txt_ff_up_w_q4   = u16(w.txt_ff_up_w);   lw.txt_ff_up_scale = nullptr;
    lw.txt_ff_up_b      = u16(w.txt_ff_up_b);
    lw.txt_ff_down_w_q4 = u16(w.txt_ff_down_w); lw.txt_ff_down_scale = nullptr;
    lw.txt_ff_down_b    = u16(w.txt_ff_down_b);

    // Block LayerNorm gammas/betas stay null (affine=false).
    return true;
}

// ---------------------------------------------------------------------------
// CPU reference — F32 math mirroring the NPU dispatch sequence.
// All inputs are read from the F16-quantised host buffers so the reference
// sees the same starting point as the NPU path (shared F16 quant error).
// ---------------------------------------------------------------------------

// y[i] = sigmoid(x[i]) * x[i]  = x / (1 + exp(-x))
static void cpu_silu_(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

// C[m, n] = sum_k A[m, k] * B_gguf[n, k]  (since weights are stored [N, K]
// row-major with K contig). bias broadcasts over m.
static void cpu_matmul(const float *A, int64_t M, int64_t K,
                        const uint16_t *B_gguf, int64_t N,
                        const uint16_t *bias, float *C) {
    for (int64_t m = 0; m < M; ++m) {
        for (int64_t n = 0; n < N; ++n) {
            const uint16_t *brow = B_gguf + (size_t)n * K;
            float acc = bias ? f16_to_f32(bias[n]) : 0.0f;
            for (int64_t k = 0; k < K; ++k) {
                acc += A[(size_t)m * K + k] * f16_to_f32(brow[k]);
            }
            C[(size_t)m * N + n] = acc;
        }
    }
}

// LayerNorm over last dim (affine-off; eps applied under sqrt).
static void cpu_layernorm_(float *x, int64_t rows, int64_t cols, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        float *row = x + (size_t)r * cols;
        float mean = 0.0f;
        for (int64_t c = 0; c < cols; ++c) mean += row[c];
        mean /= (float)cols;
        float var = 0.0f;
        for (int64_t c = 0; c < cols; ++c) {
            float d = row[c] - mean;
            var += d * d;
        }
        var /= (float)cols;
        float rstd = 1.0f / std::sqrt(var + eps);
        for (int64_t c = 0; c < cols; ++c) row[c] = (row[c] - mean) * rstd;
    }
}

// RMSNorm over head_dim: y = x * rsqrt(mean(x^2) + eps) * gamma
static void cpu_rmsnorm_head_(float *x, int64_t rows, int64_t head_dim,
                               const float *gamma, float eps) {
    for (int64_t r = 0; r < rows; ++r) {
        float *row = x + (size_t)r * head_dim;
        float ssq = 0.0f;
        for (int64_t c = 0; c < head_dim; ++c) ssq += row[c] * row[c];
        float rstd = 1.0f / std::sqrt(ssq / (float)head_dim + eps);
        for (int64_t c = 0; c < head_dim; ++c)
            row[c] = row[c] * rstd * gamma[c];
    }
}

// modulate — x = x * (1 + scale) + shift, scale/shift broadcast over seq.
static void cpu_modulate_(float *x, int64_t B, int64_t seq, int64_t hidden,
                           const float *scale, const float *shift) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            float *row = x + ((size_t)b * seq + s) * hidden;
            const float *sc = scale + (size_t)b * hidden;
            const float *sh = shift + (size_t)b * hidden;
            for (int64_t h = 0; h < hidden; ++h)
                row[h] = row[h] * (1.0f + sc[h]) + sh[h];
        }
    }
}

// gated residual add — x += src * gate, gate broadcast over seq.
static void cpu_gated_add_(float *x, const float *src, const float *gate,
                            int64_t B, int64_t seq, int64_t hidden) {
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            float *xr = x + ((size_t)b * seq + s) * hidden;
            const float *sr = src + ((size_t)b * seq + s) * hidden;
            const float *gr = gate + (size_t)b * hidden;
            for (int64_t h = 0; h < hidden; ++h)
                xr[h] += sr[h] * gr[h];
        }
    }
}

// GELU-tanh approximation to match ggml CPU reference (and NPU GeluV2 with
// approximate=1).
static void cpu_gelu_tanh_(float *x, size_t n) {
    const float kA = 0.044715f;
    const float kS = 0.7978845608028654f;  // sqrt(2/pi)
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        float arg = kS * (v + kA * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(arg));
    }
}

// RoPE application matching rope.hpp:603 / apply_rope_() engine path.
// x shape [B, seq, heads, head_dim]; pe is [pe_max_rows, head_dim/2, 2, 2]
// where pe[p, dp, 0, :] = [cos, -sin], pe[p, dp, 1, :] = [sin, cos].
static void cpu_apply_rope_(float *x, int64_t B, int64_t seq, int64_t heads,
                              int64_t head_dim,
                              const std::vector<uint16_t> &pe_host,
                              int64_t pe_row_offset) {
    int64_t half = head_dim / 2;
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            size_t pe_base = (size_t)(pe_row_offset + s) * half * 4;
            for (int64_t h = 0; h < heads; ++h) {
                size_t rbase = (((size_t)b * seq + s) * heads + h) * head_dim;
                for (int64_t dp = 0; dp < half; ++dp) {
                    size_t peh = pe_base + (size_t)dp * 4;
                    float pe00 = f16_to_f32(pe_host[peh + 0]);
                    float pe01 = f16_to_f32(pe_host[peh + 1]);
                    float pe10 = f16_to_f32(pe_host[peh + 2]);
                    float pe11 = f16_to_f32(pe_host[peh + 3]);
                    float x0 = x[rbase + (size_t)(2 * dp)];
                    float x1 = x[rbase + (size_t)(2 * dp + 1)];
                    x[rbase + (size_t)(2 * dp)]     = x0 * pe00 + x1 * pe10;
                    x[rbase + (size_t)(2 * dp + 1)] = x0 * pe01 + x1 * pe11;
                }
            }
        }
    }
}

// Joint attention: FIAv2 is online-softmax / flash-style, but for F32
// reference at seq ~ 1280 we can run naive softmax attention with no
// precision loss. Layout: q/k/v are [seq_total, heads, head_dim].
static void cpu_attention_(const float *q, const float *k, const float *v,
                            int64_t S, int64_t heads, int64_t head_dim,
                            float *out) {
    float scale = 1.0f / std::sqrt((float)head_dim);
    std::vector<float> scores((size_t)S);
    for (int64_t h = 0; h < heads; ++h) {
        for (int64_t i = 0; i < S; ++i) {
            const float *qr = q + ((size_t)i * heads + h) * head_dim;
            float m = -std::numeric_limits<float>::infinity();
            for (int64_t j = 0; j < S; ++j) {
                const float *kr = k + ((size_t)j * heads + h) * head_dim;
                float s = 0.0f;
                for (int64_t d = 0; d < head_dim; ++d) s += qr[d] * kr[d];
                s *= scale;
                scores[(size_t)j] = s;
                if (s > m) m = s;
            }
            float sum = 0.0f;
            for (int64_t j = 0; j < S; ++j) {
                scores[(size_t)j] = std::exp(scores[(size_t)j] - m);
                sum += scores[(size_t)j];
            }
            for (int64_t j = 0; j < S; ++j) scores[(size_t)j] /= sum;
            float *or_ = out + ((size_t)i * heads + h) * head_dim;
            for (int64_t d = 0; d < head_dim; ++d) or_[d] = 0.0f;
            for (int64_t j = 0; j < S; ++j) {
                const float *vr = v + ((size_t)j * heads + h) * head_dim;
                float w_ = scores[(size_t)j];
                for (int64_t d = 0; d < head_dim; ++d)
                    or_[d] += w_ * vr[d];
            }
        }
    }
}

// Copy host f16 vector into fresh f32 vector (sharing the F16 quantization
// error with the NPU input path).
static void f16_to_f32_vec(const std::vector<uint16_t> &src,
                            std::vector<float> &dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = f16_to_f32(src[i]);
}

static void vec_cast(const float *src, std::vector<uint16_t> &dst, size_t n) {
    dst.resize(n);
    for (size_t i = 0; i < n; ++i) dst[i] = f32_to_f16(src[i]);
}

// ---------------------------------------------------------------------------
// Full CPU-reference block forward.
// ---------------------------------------------------------------------------
struct CpuBlockRef {
    std::vector<float> img_out;  // [img_seq, H]
    std::vector<float> txt_out;  // [txt_seq, H]
};

static void cpu_block_forward(const ImageDiffusionConfig &cfg,
                               const HostWeights &w,
                               const std::vector<uint16_t> &img_hidden_f16,
                               const std::vector<uint16_t> &txt_hidden_f16,
                               const std::vector<uint16_t> &t_emb_f16,
                               const std::vector<uint16_t> &pe_host,
                               int64_t img_seq, int64_t txt_seq,
                               CpuBlockRef &out) {
    const int64_t B  = 1;
    const int64_t H  = cfg.hidden_size;
    const int64_t HD = cfg.head_dim;
    const int64_t NH = cfg.num_heads;
    const int64_t FF = (int64_t)H * cfg.ff_mult;

    std::vector<float> img_h, txt_h, t_emb;
    f16_to_f32_vec(img_hidden_f16, img_h);
    f16_to_f32_vec(txt_hidden_f16, txt_h);
    f16_to_f32_vec(t_emb_f16, t_emb);

    // 1. silu + mod_params for both streams.
    std::vector<float> silu_t = t_emb;
    cpu_silu_(silu_t.data(), silu_t.size());

    std::vector<float> img_mod((size_t)B * 6 * H);
    std::vector<float> txt_mod((size_t)B * 6 * H);
    cpu_matmul(silu_t.data(), B, H,
                w.img_mod_w.data(), 6 * H,
                w.img_mod_b.data(), img_mod.data());
    cpu_matmul(silu_t.data(), B, H,
                w.txt_mod_w.data(), 6 * H,
                w.txt_mod_b.data(), txt_mod.data());

    // Chunk layout: [B, 6, H] — flatten to [6 · H] per batch since B=1.
    auto img_chunk = [&](int i) { return img_mod.data() + (size_t)i * H; };
    auto txt_chunk = [&](int i) { return txt_mod.data() + (size_t)i * H; };

    // 2. LayerNorm1 + modulate1 per stream.
    std::vector<float> img_normed = img_h;
    cpu_layernorm_(img_normed.data(), img_seq, H, cfg.layernorm_eps);
    cpu_modulate_(img_normed.data(), B, img_seq, H,
                   img_chunk(0), img_chunk(1));
    std::vector<float> txt_normed = txt_h;
    cpu_layernorm_(txt_normed.data(), txt_seq, H, cfg.layernorm_eps);
    cpu_modulate_(txt_normed.data(), B, txt_seq, H,
                   txt_chunk(0), txt_chunk(1));

    // 3. QKV projections per stream.
    std::vector<float> img_q((size_t)img_seq * H);
    std::vector<float> img_k((size_t)img_seq * H);
    std::vector<float> img_v((size_t)img_seq * H);
    cpu_matmul(img_normed.data(), img_seq, H,
                w.to_q_w.data(), H, w.to_q_b.data(), img_q.data());
    cpu_matmul(img_normed.data(), img_seq, H,
                w.to_k_w.data(), H, w.to_k_b.data(), img_k.data());
    cpu_matmul(img_normed.data(), img_seq, H,
                w.to_v_w.data(), H, w.to_v_b.data(), img_v.data());
    std::vector<float> txt_q((size_t)txt_seq * H);
    std::vector<float> txt_k((size_t)txt_seq * H);
    std::vector<float> txt_v((size_t)txt_seq * H);
    cpu_matmul(txt_normed.data(), txt_seq, H,
                w.add_q_w.data(), H, w.add_q_b.data(), txt_q.data());
    cpu_matmul(txt_normed.data(), txt_seq, H,
                w.add_k_w.data(), H, w.add_k_b.data(), txt_k.data());
    cpu_matmul(txt_normed.data(), txt_seq, H,
                w.add_v_w.data(), H, w.add_v_b.data(), txt_v.data());

    // 4. Q/K RMSNorm per stream (rows = seq × NH, last dim = head_dim).
    cpu_rmsnorm_head_(img_q.data(), img_seq * NH, HD,
                       w.norm_q_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(img_k.data(), img_seq * NH, HD,
                       w.norm_k_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(txt_q.data(), txt_seq * NH, HD,
                       w.norm_added_q_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(txt_k.data(), txt_seq * NH, HD,
                       w.norm_added_k_w.data(), cfg.rms_norm_eps);

    // 5. RoPE on Q, K per stream.
    cpu_apply_rope_(txt_q.data(), B, txt_seq, NH, HD, pe_host, 0);
    cpu_apply_rope_(txt_k.data(), B, txt_seq, NH, HD, pe_host, 0);
    cpu_apply_rope_(img_q.data(), B, img_seq, NH, HD, pe_host, cfg.max_txt_seq);
    cpu_apply_rope_(img_k.data(), B, img_seq, NH, HD, pe_host, cfg.max_txt_seq);

    // 6. Concat txt || img along seq dim → joint.
    int64_t S = img_seq + txt_seq;
    std::vector<float> jq((size_t)S * H), jk((size_t)S * H), jv((size_t)S * H);
    std::memcpy(jq.data(), txt_q.data(),
                 txt_q.size() * sizeof(float));
    std::memcpy(jk.data(), txt_k.data(), txt_k.size() * sizeof(float));
    std::memcpy(jv.data(), txt_v.data(), txt_v.size() * sizeof(float));
    std::memcpy(jq.data() + (size_t)txt_seq * H, img_q.data(),
                 img_q.size() * sizeof(float));
    std::memcpy(jk.data() + (size_t)txt_seq * H, img_k.data(),
                 img_k.size() * sizeof(float));
    std::memcpy(jv.data() + (size_t)txt_seq * H, img_v.data(),
                 img_v.size() * sizeof(float));

    // 7. Joint attention.
    std::vector<float> attn((size_t)S * H);
    cpu_attention_(jq.data(), jk.data(), jv.data(), S, NH, HD, attn.data());

    // 8. Split and output project.
    std::vector<float> img_attn_out((size_t)img_seq * H);
    cpu_matmul(attn.data() + (size_t)txt_seq * H, img_seq, H,
                w.to_out_w.data(), H, w.to_out_b.data(), img_attn_out.data());
    std::vector<float> txt_attn_out((size_t)txt_seq * H);
    cpu_matmul(attn.data(), txt_seq, H,
                w.to_add_out_w.data(), H, w.to_add_out_b.data(),
                txt_attn_out.data());

    // 9. Gated residual 1.
    cpu_gated_add_(img_h.data(), img_attn_out.data(), img_chunk(2),
                    B, img_seq, H);
    cpu_gated_add_(txt_h.data(), txt_attn_out.data(), txt_chunk(2),
                    B, txt_seq, H);

    // 10. LayerNorm2 + modulate2.
    std::vector<float> img_normed2 = img_h;
    cpu_layernorm_(img_normed2.data(), img_seq, H, cfg.layernorm_eps);
    cpu_modulate_(img_normed2.data(), B, img_seq, H,
                   img_chunk(3), img_chunk(4));
    std::vector<float> txt_normed2 = txt_h;
    cpu_layernorm_(txt_normed2.data(), txt_seq, H, cfg.layernorm_eps);
    cpu_modulate_(txt_normed2.data(), B, txt_seq, H,
                   txt_chunk(3), txt_chunk(4));

    // 11. FFN per stream.
    std::vector<float> img_ff_mid((size_t)img_seq * FF);
    cpu_matmul(img_normed2.data(), img_seq, H,
                w.img_ff_up_w.data(), FF, w.img_ff_up_b.data(),
                img_ff_mid.data());
    cpu_gelu_tanh_(img_ff_mid.data(), img_ff_mid.size());
    std::vector<float> img_ff_out((size_t)img_seq * H);
    cpu_matmul(img_ff_mid.data(), img_seq, FF,
                w.img_ff_down_w.data(), H, w.img_ff_down_b.data(),
                img_ff_out.data());

    std::vector<float> txt_ff_mid((size_t)txt_seq * FF);
    cpu_matmul(txt_normed2.data(), txt_seq, H,
                w.txt_ff_up_w.data(), FF, w.txt_ff_up_b.data(),
                txt_ff_mid.data());
    cpu_gelu_tanh_(txt_ff_mid.data(), txt_ff_mid.size());
    std::vector<float> txt_ff_out((size_t)txt_seq * H);
    cpu_matmul(txt_ff_mid.data(), txt_seq, FF,
                w.txt_ff_down_w.data(), H, w.txt_ff_down_b.data(),
                txt_ff_out.data());

    // 12. Gated residual 2.
    cpu_gated_add_(img_h.data(), img_ff_out.data(), img_chunk(5),
                    B, img_seq, H);
    cpu_gated_add_(txt_h.data(), txt_ff_out.data(), txt_chunk(5),
                    B, txt_seq, H);

    out.img_out = std::move(img_h);
    out.txt_out = std::move(txt_h);
}

// ---------------------------------------------------------------------------
// Cosine similarity between an F16 array and a F32 reference (cast down).
// ---------------------------------------------------------------------------
struct StatsLine {
    double cos_sim;
    float  min_npu, max_npu;
    int64_t nan_count;
    double mae;
};

static StatsLine compare_f16_f32(const uint16_t *a, const float *b, size_t n) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    float  mn = +1e30f, mx = -1e30f;
    int64_t nanc = 0;
    double mae = 0.0;
    for (size_t i = 0; i < n; ++i) {
        float av = f16_to_f32(a[i]);
        float bv = b[i];
        if (std::isnan(av) || std::isinf(av)) nanc++;
        mn = std::min(mn, av); mx = std::max(mx, av);
        dot += (double)av * (double)bv;
        na  += (double)av * (double)av;
        nb  += (double)bv * (double)bv;
        mae += std::fabs((double)av - (double)bv);
    }
    StatsLine s;
    s.cos_sim = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    s.min_npu = mn; s.max_npu = mx;
    s.nan_count = nanc;
    s.mae = mae / (double)n;
    return s;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
int main(int /*argc*/, char ** /*argv*/) {
    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke] CANN symbol load failed\n");
        return 1;
    }

    // Config: tight smoke. Small seq to keep naive CPU reference tractable
    // (< 1 min on a single core). Architecture dims match the real QwenImage
    // DiT (hidden=3072, heads=24, head_dim=128, ff_mult=4) so the op
    // sequence exercises the same shapes as Phase 4+ full forward.
    //
    // For seq we pick 64 img + 32 txt = 96 joint — keeps the joint attention
    // `seq×seq` matrix tiny in F32 and the (seq, FF) matmuls at ~3.5M ops.
    // RoPE still uses the full 32×32 patch grid via max_img_seq=1024, but
    // the smoke runs seq=max_img_seq if we keep them equal. Here we set
    // max_img_seq=64 so the RoPE pre-compute shrinks to that, keeping
    // everything consistent.
    ImageDiffusionConfig cfg;
    cfg.num_layers    = 1;     // only block 0
    cfg.num_heads     = 24;
    cfg.head_dim      = 128;
    cfg.hidden_size   = 3072;
    cfg.ff_mult       = 4;
    // SEQ_SMALL=1: 8×8 img + 32 txt = 96 joint (first-pass GREEN at this
    // shape, ~1 s CPU ref, ~120 ms NPU).
    // SEQ_SMALL=0: 16×16 img + 64 txt = 320 joint (heavier smoke; CPU ref
    // is still sub-minute). Set by env QIE_SMOKE_SMALL=0|1 at run time.
    bool small = true;
    if (const char *e = std::getenv("QIE_SMOKE_SMALL"))
        small = (e[0] != '0');
    if (small) {
        cfg.max_img_seq = 64;  cfg.max_txt_seq = 32;
    } else {
        cfg.max_img_seq = 256; cfg.max_txt_seq = 64;
    }
    cfg.precompute_rope = true;

    const int64_t img_seq = cfg.max_img_seq;
    const int64_t txt_seq = cfg.max_txt_seq;
    const int64_t B  = 1;
    const int64_t H  = cfg.hidden_size;
    (void)B;

    ImageDiffusionEngine eng;
    if (!eng.init_for_smoke(cfg, /*device*/ 0)) {
        fprintf(stderr, "[smoke] init_for_smoke failed\n");
        return 1;
    }
    printf("[smoke] engine scratch-alloc ok; generating synthetic weights...\n");

    // Populate layer 0 weights.
    HostWeights hw;
    gen_host_weights(cfg, hw, /*seed*/ 0xC0DE0ULL);
    DiTLayerWeights *lw0 = eng.mutable_layer_weights(0);
    if (!lw0) { fprintf(stderr, "[smoke] no layer 0\n"); return 1; }
    if (!populate_layer(*lw0, hw)) {
        fprintf(stderr, "[smoke] populate_layer failed\n");
        return 1;
    }

    // Activations: img_hidden, txt_hidden, t_emb (all F16).
    std::vector<uint16_t> img_h_f16, txt_h_f16, t_emb_f16;
    fill_random_f16(img_h_f16, (size_t)img_seq * H, 0.1f, 0x1111ULL);
    fill_random_f16(txt_h_f16, (size_t)txt_seq * H, 0.1f, 0x2222ULL);
    fill_random_f16(t_emb_f16, (size_t)H,             0.1f, 0x3333ULL);

    void *img_h_dev = upload_f16(img_h_f16.data(), img_h_f16.size());
    void *txt_h_dev = upload_f16(txt_h_f16.data(), txt_h_f16.size());
    void *t_emb_dev = upload_f16(t_emb_f16.data(), t_emb_f16.size());

    // Fetch pe_host (for CPU ref). We computed it inside init_for_smoke
    // but the host copy lives only in that call's vector; re-compute here
    // for the ref. (Same algorithm.)
    // NOTE: we rely on the engine to have built global_w_.rope_pe_dev. For
    // the CPU reference we regenerate pe by calling the same helper
    // (compute_qwen_rope_pe_host is in an anonymous namespace inside the
    // engine TU, so we recompute inline here).
    std::vector<uint16_t> pe_host;
    {
        const int axes_t = cfg.rope_axes_temporal;
        const int axes_h = cfg.rope_axes_h;
        const int axes_w = cfg.rope_axes_w;
        const int head_dim = cfg.head_dim;
        int h_len = (int)std::lround(std::sqrt((double)cfg.max_img_seq));
        int w_len = h_len;
        while (h_len * w_len < cfg.max_img_seq) ++h_len;
        const int img_tokens = h_len * w_len;
        const int ctx_len   = cfg.max_txt_seq;
        const int txt_start = std::max(h_len, w_len);
        int64_t total_pos = (int64_t)ctx_len + img_tokens;
        pe_host.assign((size_t)total_pos * head_dim / 2 * 2 * 2, 0);

        auto pe_set = [&](int64_t pos, int64_t dpair, float cos_v, float sin_v) {
            size_t base = ((size_t)pos * head_dim / 2 + (size_t)dpair) * 4;
            pe_host[base + 0] = f32_to_f16(cos_v);
            pe_host[base + 1] = f32_to_f16(-sin_v);
            pe_host[base + 2] = f32_to_f16(sin_v);
            pe_host[base + 3] = f32_to_f16(cos_v);
        };
        auto axis_omega = [&](int axis_dim, std::vector<float> &omega) {
            int half_axis = axis_dim / 2;
            omega.assign(half_axis, 0.0f);
            if (half_axis == 0) return;
            if (half_axis == 1) { omega[0] = 1.0f; return; }
            const float end_scale = (axis_dim - 2.0f) / (float)axis_dim;
            for (int i = 0; i < half_axis; ++i) {
                float scale = end_scale * (float)i / (float)(half_axis - 1);
                omega[i] = 1.0f / std::pow((float)cfg.rope_theta, scale);
            }
        };
        std::vector<float> omega_t, omega_h, omega_w;
        axis_omega(axes_t, omega_t);
        axis_omega(axes_h, omega_h);
        axis_omega(axes_w, omega_w);
        for (int i = 0; i < ctx_len; ++i) {
            float p = (float)(txt_start + i);
            int64_t pos = i, dp = 0;
            for (int j = 0; j < (int)omega_t.size(); ++j, ++dp) {
                float a = p * omega_t[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
            for (int j = 0; j < (int)omega_h.size(); ++j, ++dp) {
                float a = p * omega_h[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
            for (int j = 0; j < (int)omega_w.size(); ++j, ++dp) {
                float a = p * omega_w[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
        }
        int h_start = -h_len / 2, w_start = -w_len / 2;
        for (int r = 0; r < h_len; ++r) {
            float h_id = (float)(h_start + r);
            for (int c = 0; c < w_len; ++c) {
                float w_id_ = (float)(w_start + c);
                int64_t pos = (int64_t)ctx_len + r * w_len + c;
                if (pos >= total_pos) break;
                int64_t dp = 0;
                float t_id = 0.0f;
                for (int j = 0; j < (int)omega_t.size(); ++j, ++dp) {
                    float a = t_id * omega_t[j];
                    pe_set(pos, dp, std::cos(a), std::sin(a));
                }
                for (int j = 0; j < (int)omega_h.size(); ++j, ++dp) {
                    float a = h_id * omega_h[j];
                    pe_set(pos, dp, std::cos(a), std::sin(a));
                }
                for (int j = 0; j < (int)omega_w.size(); ++j, ++dp) {
                    float a = w_id_ * omega_w[j];
                    pe_set(pos, dp, std::cos(a), std::sin(a));
                }
            }
        }
    }

    // Upload pe table and dispatch.
    void *pe_dev = upload_f16(pe_host.data(), pe_host.size());
    auto t0 = std::chrono::steady_clock::now();
    bool ok = eng.forward_block_test(0,
                                      img_h_dev, img_seq,
                                      txt_h_dev, txt_seq,
                                      t_emb_dev, pe_dev);
    // The engine uses its own primary_stream_ which the probe can't reach
    // directly. aclrtSynchronizeDevice waits for all streams on the current
    // device — simplest valid sync before D2H. Links directly against
    // libascendcl (already on LD_LIBRARY_PATH).
    aclrtSynchronizeDevice();
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    if (!ok) {
        fprintf(stderr, "[smoke] forward_block_test returned false\n");
        return 1;
    }

    // D2H download of outputs.
    std::vector<uint16_t> img_h_out_f16(img_h_f16.size());
    std::vector<uint16_t> txt_h_out_f16(txt_h_f16.size());
    g_cann.aclrtMemcpy(img_h_out_f16.data(),
                        img_h_out_f16.size() * sizeof(uint16_t),
                        img_h_dev,
                        img_h_out_f16.size() * sizeof(uint16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);
    g_cann.aclrtMemcpy(txt_h_out_f16.data(),
                        txt_h_out_f16.size() * sizeof(uint16_t),
                        txt_h_dev,
                        txt_h_out_f16.size() * sizeof(uint16_t),
                        ACL_MEMCPY_DEVICE_TO_HOST);

    // -------- CPU reference --------
    CpuBlockRef ref;
    cpu_block_forward(cfg, hw,
                      img_h_f16, txt_h_f16, t_emb_f16,
                      pe_host, img_seq, txt_seq, ref);

    // -------- Compare --------
    StatsLine img_s = compare_f16_f32(img_h_out_f16.data(),
                                        ref.img_out.data(),
                                        img_h_out_f16.size());
    StatsLine txt_s = compare_f16_f32(txt_h_out_f16.data(),
                                        ref.txt_out.data(),
                                        txt_h_out_f16.size());

    printf("\n========== Q2.3 Phase 3 smoke report ==========\n");
    printf("config: H=%lld heads=%lld head_dim=%lld ff_dim=%lld\n",
           (long long)cfg.hidden_size, (long long)cfg.num_heads,
           (long long)cfg.head_dim,
           (long long)cfg.hidden_size * cfg.ff_mult);
    printf("seq:    img=%lld  txt=%lld  joint=%lld\n",
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    printf("wall:   %.2f ms (single block, NPU + H2D/D2H roundtrip)\n", ms);
    printf("\n-- img_hidden_out vs CPU-ref --\n");
    printf("  cos_sim  = %.6f\n", img_s.cos_sim);
    printf("  mae      = %.6f\n", img_s.mae);
    printf("  min/max  = %.4f / %.4f\n", img_s.min_npu, img_s.max_npu);
    printf("  NaN/inf  = %lld\n", (long long)img_s.nan_count);
    printf("-- txt_hidden_out vs CPU-ref --\n");
    printf("  cos_sim  = %.6f\n", txt_s.cos_sim);
    printf("  mae      = %.6f\n", txt_s.mae);
    printf("  min/max  = %.4f / %.4f\n", txt_s.min_npu, txt_s.max_npu);
    printf("  NaN/inf  = %lld\n", (long long)txt_s.nan_count);

    bool pass = (img_s.cos_sim > 0.99) && (txt_s.cos_sim > 0.99) &&
                (img_s.nan_count == 0) && (txt_s.nan_count == 0);
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s (gate: cos_sim > 0.99 both streams, NaN=0)\n",
           pass ? "GREEN" : "RED");
    return pass ? 0 : 2;
}
