// ============================================================================
// Q2 Phase 4.2 smoke — full 60-block DiT forward, cos_sim vs CPU reference.
//
// Scope: Phase 4.2 per docs/qie_q2_phase4_smoke.md §2. Exercises the block
// loop end-to-end. This is pure plumbing — each block takes the previous
// block's output as input to the next.
//
// Harness:
//   - Boots ImageDiffusionEngine via init_for_smoke() (no GGUF load).
//   - Synthesizes ONE deterministic random F16 weight set and makes every
//     layer_w_[il] point at the SAME device buffers. This keeps HBM at
//     ~single-block budget regardless of cfg_.num_layers, and the CPU
//     reference runs 60 iterations of the same block so the parity check
//     is apples-to-apples.
//   - Generates random F16 img_hidden, txt_hidden, t_emb on host.
//   - Dispatches forward_all_blocks_test(n_blocks=60, per_block_ms).
//   - Computes the same 60-block forward in F32 on host using the same
//     numerical sequence each block runs on NPU.
//   - Reports cos_sim(img_out_npu, img_out_cpu_ref), cos_sim(txt_out, ...),
//     NaN count, total wall, min/median/max per-block wall sample.
//
// Phase 4.2 gate: cos_sim > 0.95 @ layer 60 for both streams; NaN=0. The
// gate is lowered from Phase 3's 0.99 to accept F16 accumulation drift over
// 60 layers — per Phase 4.2 scope doc.
//
// Build on ac03:
//   cd tools/probes/qie_q42_60block_smoke && bash build_and_run.sh
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

static void *upload_f16(const uint16_t *host, size_t n) {
    void *dev = nullptr;
    size_t bytes = n * sizeof(uint16_t);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[smoke42] aclrtMalloc(%zu) err=%d\n", bytes, (int)err);
        return nullptr;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        fprintf(stderr, "[smoke42] H2D memcpy err=%d\n", (int)err);
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
// HostWeights — one shared set used by every layer index on NPU and in the
// CPU reference. Layouts identical to Phase 3 (GGUF [N, K] row-major F16 for
// matmul weights, F16 [N] biases, F32 [head_dim] RMSNorm gammas).
// ---------------------------------------------------------------------------
struct HostWeights {
    std::vector<uint16_t> to_q_w, to_k_w, to_v_w, to_out_w;
    std::vector<uint16_t> to_q_b, to_k_b, to_v_b, to_out_b;
    std::vector<uint16_t> add_q_w, add_k_w, add_v_w, to_add_out_w;
    std::vector<uint16_t> add_q_b, add_k_b, add_v_b, to_add_out_b;
    std::vector<float> norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    std::vector<uint16_t> img_mod_w, txt_mod_w;
    std::vector<uint16_t> img_mod_b, txt_mod_b;
    std::vector<uint16_t> img_ff_up_w, img_ff_down_w;
    std::vector<uint16_t> img_ff_up_b, img_ff_down_b;
    std::vector<uint16_t> txt_ff_up_w, txt_ff_down_w;
    std::vector<uint16_t> txt_ff_up_b, txt_ff_down_b;
};

// Amplitudes chosen to keep activations bounded across 60 iterations with
// identity-ish residual updates. Modulation weight amplitude is ~0 so that
// `(1 + scale)` stays near 1 and `gate` stays near 0 — i.e. each block's
// residual update is small, preventing magnitude blow-up. Same shape of
// amplitudes as Phase 3 (proven stable) just with mod_w_amp an order
// smaller so 60 compositions remain bounded in F16.
static void gen_host_weights(const ImageDiffusionConfig &cfg, HostWeights &w,
                              uint64_t seed) {
    const int64_t H  = cfg.hidden_size;
    const int64_t HD = cfg.head_dim;
    const int64_t FF = (int64_t)H * cfg.ff_mult;
    const float W_AMP = 1.0f / std::sqrt((float)H);
    const float B_AMP = 0.02f;
    const float G_AMP = 0.1f;
    auto rw = [&](std::vector<uint16_t> &v, int64_t N, int64_t K, uint64_t s) {
        fill_random_f16(v, (size_t)N * K, W_AMP, s);
    };
    auto rb = [&](std::vector<uint16_t> &v, int64_t N, uint64_t s) {
        fill_random_f16(v, (size_t)N, B_AMP, s);
    };
    auto rg = [&](std::vector<float> &v, int64_t N, uint64_t s) {
        fill_random_f32(v, (size_t)N, G_AMP, s);
        for (auto &x : v) x += 1.0f;
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
    {
        // Tighten modulation amplitudes relative to Phase 3. With 60 shared
        // blocks iterated back-to-back we want `(1 + scale)` very close to
        // 1 (scale ~ 0.001) and `gate` very close to 0 so the accumulated
        // residual over 60 blocks stays inside F16 range. Phase 3 used
        // 0.01 / 0.01; at 60 compositions that can drift ~(1.01)^60 ≈ 1.8×
        // per element which pushes into numerical instability. 1e-3 keeps
        // drift ≤ (1.001)^60 ≈ 1.06×.
        const float mod_w_amp = 0.001f;
        const float mod_b_amp = 0.001f;
        fill_random_f16(w.img_mod_w, (size_t)6 * H * H, mod_w_amp, seed + 31);
        fill_random_f16(w.txt_mod_w, (size_t)6 * H * H, mod_w_amp, seed + 32);
        fill_random_f16(w.img_mod_b, (size_t)6 * H, mod_b_amp,     seed + 33);
        fill_random_f16(w.txt_mod_b, (size_t)6 * H, mod_b_amp,     seed + 34);
    }
    rw(w.img_ff_up_w,   FF, H, seed + 41); rb(w.img_ff_up_b,   FF, seed + 42);
    rw(w.img_ff_down_w, H, FF, seed + 43); rb(w.img_ff_down_b, H,  seed + 44);
    rw(w.txt_ff_up_w,   FF, H, seed + 51); rb(w.txt_ff_up_b,   FF, seed + 52);
    rw(w.txt_ff_down_w, H, FF, seed + 53); rb(w.txt_ff_down_b, H,  seed + 54);
}

// ---------------------------------------------------------------------------
// UploadedWeights — device pointers for the shared weight set. One instance
// per smoke run, pointed at by every layer's DiTLayerWeights slot.
// ---------------------------------------------------------------------------
struct UploadedWeights {
    // matmul weights (F16 fallback path; scale_dev stays null)
    void *to_q_w = nullptr, *to_k_w = nullptr, *to_v_w = nullptr, *to_out_w = nullptr;
    void *to_q_b = nullptr, *to_k_b = nullptr, *to_v_b = nullptr, *to_out_b = nullptr;
    void *add_q_w = nullptr, *add_k_w = nullptr, *add_v_w = nullptr, *to_add_out_w = nullptr;
    void *add_q_b = nullptr, *add_k_b = nullptr, *add_v_b = nullptr, *to_add_out_b = nullptr;
    void *norm_q_w = nullptr, *norm_k_w = nullptr;
    void *norm_added_q_w = nullptr, *norm_added_k_w = nullptr;
    void *img_mod_w = nullptr, *txt_mod_w = nullptr;
    void *img_mod_b = nullptr, *txt_mod_b = nullptr;
    void *img_ff_up_w = nullptr, *img_ff_down_w = nullptr;
    void *img_ff_up_b = nullptr, *img_ff_down_b = nullptr;
    void *txt_ff_up_w = nullptr, *txt_ff_down_w = nullptr;
    void *txt_ff_up_b = nullptr, *txt_ff_down_b = nullptr;
};

static bool upload_shared(const HostWeights &w, UploadedWeights &u) {
    auto u16 = [&](const std::vector<uint16_t> &v) {
        return upload_f16(v.data(), v.size());
    };
    auto uf32 = [&](const std::vector<float> &v) {
        return upload_f32(v.data(), v.size());
    };
    u.to_q_w = u16(w.to_q_w); u.to_q_b = u16(w.to_q_b);
    u.to_k_w = u16(w.to_k_w); u.to_k_b = u16(w.to_k_b);
    u.to_v_w = u16(w.to_v_w); u.to_v_b = u16(w.to_v_b);
    u.to_out_w = u16(w.to_out_w); u.to_out_b = u16(w.to_out_b);
    u.add_q_w = u16(w.add_q_w); u.add_q_b = u16(w.add_q_b);
    u.add_k_w = u16(w.add_k_w); u.add_k_b = u16(w.add_k_b);
    u.add_v_w = u16(w.add_v_w); u.add_v_b = u16(w.add_v_b);
    u.to_add_out_w = u16(w.to_add_out_w); u.to_add_out_b = u16(w.to_add_out_b);
    u.norm_q_w = uf32(w.norm_q_w);
    u.norm_k_w = uf32(w.norm_k_w);
    u.norm_added_q_w = uf32(w.norm_added_q_w);
    u.norm_added_k_w = uf32(w.norm_added_k_w);
    u.img_mod_w = u16(w.img_mod_w); u.img_mod_b = u16(w.img_mod_b);
    u.txt_mod_w = u16(w.txt_mod_w); u.txt_mod_b = u16(w.txt_mod_b);
    u.img_ff_up_w = u16(w.img_ff_up_w); u.img_ff_up_b = u16(w.img_ff_up_b);
    u.img_ff_down_w = u16(w.img_ff_down_w); u.img_ff_down_b = u16(w.img_ff_down_b);
    u.txt_ff_up_w = u16(w.txt_ff_up_w); u.txt_ff_up_b = u16(w.txt_ff_up_b);
    u.txt_ff_down_w = u16(w.txt_ff_down_w); u.txt_ff_down_b = u16(w.txt_ff_down_b);
    return true;  // upload_f16/f32 abort the process' stderr on failure;
                   // caller still proceeds to let the engine reject nulls.
}

// Point one DiTLayerWeights slot at the shared uploaded buffers.
static void point_layer_at_shared(DiTLayerWeights &lw, const UploadedWeights &u) {
    lw.to_q_w_q4   = u.to_q_w;   lw.to_q_scale   = nullptr; lw.to_q_b = u.to_q_b;
    lw.to_k_w_q4   = u.to_k_w;   lw.to_k_scale   = nullptr; lw.to_k_b = u.to_k_b;
    lw.to_v_w_q4   = u.to_v_w;   lw.to_v_scale   = nullptr; lw.to_v_b = u.to_v_b;
    lw.to_out_0_w_q4 = u.to_out_w; lw.to_out_0_scale = nullptr; lw.to_out_0_b = u.to_out_b;
    lw.add_q_w_q4  = u.add_q_w;  lw.add_q_scale  = nullptr; lw.add_q_b = u.add_q_b;
    lw.add_k_w_q4  = u.add_k_w;  lw.add_k_scale  = nullptr; lw.add_k_b = u.add_k_b;
    lw.add_v_w_q4  = u.add_v_w;  lw.add_v_scale  = nullptr; lw.add_v_b = u.add_v_b;
    lw.to_add_out_w_q4 = u.to_add_out_w; lw.to_add_out_scale = nullptr;
    lw.to_add_out_b = u.to_add_out_b;
    lw.norm_q_w       = u.norm_q_w;
    lw.norm_k_w       = u.norm_k_w;
    lw.norm_added_q_w = u.norm_added_q_w;
    lw.norm_added_k_w = u.norm_added_k_w;
    lw.img_mod_w_q4  = u.img_mod_w;  lw.img_mod_scale = nullptr;
    lw.img_mod_b     = u.img_mod_b;
    lw.txt_mod_w_q4  = u.txt_mod_w;  lw.txt_mod_scale = nullptr;
    lw.txt_mod_b     = u.txt_mod_b;
    lw.img_ff_up_w_q4   = u.img_ff_up_w;   lw.img_ff_up_scale = nullptr;
    lw.img_ff_up_b      = u.img_ff_up_b;
    lw.img_ff_down_w_q4 = u.img_ff_down_w; lw.img_ff_down_scale = nullptr;
    lw.img_ff_down_b    = u.img_ff_down_b;
    lw.txt_ff_up_w_q4   = u.txt_ff_up_w;   lw.txt_ff_up_scale = nullptr;
    lw.txt_ff_up_b      = u.txt_ff_up_b;
    lw.txt_ff_down_w_q4 = u.txt_ff_down_w; lw.txt_ff_down_scale = nullptr;
    lw.txt_ff_down_b    = u.txt_ff_down_b;
}

// ---------------------------------------------------------------------------
// CPU reference — F32 math mirroring the NPU dispatch sequence exactly.
// Copy-paste of the Phase 3 CPU block forward; the 4.2 smoke simply invokes
// it 60× (with output → input) so the reference and NPU paths share the
// identical per-block numerical sequence.
// ---------------------------------------------------------------------------

static void cpu_silu_(float *x, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        x[i] = v / (1.0f + std::exp(-v));
    }
}

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

static void cpu_gelu_tanh_(float *x, size_t n) {
    const float kA = 0.044715f;
    const float kS = 0.7978845608028654f;
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        float arg = kS * (v + kA * v * v * v);
        x[i] = 0.5f * v * (1.0f + std::tanh(arg));
    }
}

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

// Quantize-cast a float buffer to F16 and back to float, matching the
// F16 precision the NPU sees at inter-block boundaries. This is the Phase
// 4.2 wrinkle: after every block the NPU writes F16 [img_seq, H] back out
// and the next block reads F16 [img_seq, H] in. The CPU reference must
// see the same quantization noise at every inter-block boundary or we
// over-report drift.
static void f32_requantize_via_f16(std::vector<float> &v) {
    for (auto &x : v) x = f16_to_f32(f32_to_f16(x));
}

// One CPU reference block — same dispatch sequence as the NPU's
// forward_block_. Input/output are F32 for ease of math; caller is
// responsible for requantizing at block boundaries so the precision
// stays comparable to the NPU F16 in/out contract.
static void cpu_block_forward(const ImageDiffusionConfig &cfg,
                               const HostWeights &w,
                               std::vector<float> &img_h,
                               std::vector<float> &txt_h,
                               const std::vector<float> &t_emb,
                               const std::vector<uint16_t> &pe_host,
                               int64_t img_seq, int64_t txt_seq) {
    const int64_t B  = 1;
    const int64_t H  = cfg.hidden_size;
    const int64_t HD = cfg.head_dim;
    const int64_t NH = cfg.num_heads;
    const int64_t FF = (int64_t)H * cfg.ff_mult;

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

    auto img_chunk = [&](int i) { return img_mod.data() + (size_t)i * H; };
    auto txt_chunk = [&](int i) { return txt_mod.data() + (size_t)i * H; };

    std::vector<float> img_normed = img_h;
    cpu_layernorm_(img_normed.data(), img_seq, H, cfg.layernorm_eps);
    cpu_modulate_(img_normed.data(), B, img_seq, H,
                   img_chunk(0), img_chunk(1));
    std::vector<float> txt_normed = txt_h;
    cpu_layernorm_(txt_normed.data(), txt_seq, H, cfg.layernorm_eps);
    cpu_modulate_(txt_normed.data(), B, txt_seq, H,
                   txt_chunk(0), txt_chunk(1));

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

    cpu_rmsnorm_head_(img_q.data(), img_seq * NH, HD,
                       w.norm_q_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(img_k.data(), img_seq * NH, HD,
                       w.norm_k_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(txt_q.data(), txt_seq * NH, HD,
                       w.norm_added_q_w.data(), cfg.rms_norm_eps);
    cpu_rmsnorm_head_(txt_k.data(), txt_seq * NH, HD,
                       w.norm_added_k_w.data(), cfg.rms_norm_eps);

    cpu_apply_rope_(txt_q.data(), B, txt_seq, NH, HD, pe_host, 0);
    cpu_apply_rope_(txt_k.data(), B, txt_seq, NH, HD, pe_host, 0);
    cpu_apply_rope_(img_q.data(), B, img_seq, NH, HD, pe_host, cfg.max_txt_seq);
    cpu_apply_rope_(img_k.data(), B, img_seq, NH, HD, pe_host, cfg.max_txt_seq);

    int64_t S = img_seq + txt_seq;
    std::vector<float> jq((size_t)S * H), jk((size_t)S * H), jv((size_t)S * H);
    std::memcpy(jq.data(), txt_q.data(), txt_q.size() * sizeof(float));
    std::memcpy(jk.data(), txt_k.data(), txt_k.size() * sizeof(float));
    std::memcpy(jv.data(), txt_v.data(), txt_v.size() * sizeof(float));
    std::memcpy(jq.data() + (size_t)txt_seq * H, img_q.data(),
                 img_q.size() * sizeof(float));
    std::memcpy(jk.data() + (size_t)txt_seq * H, img_k.data(),
                 img_k.size() * sizeof(float));
    std::memcpy(jv.data() + (size_t)txt_seq * H, img_v.data(),
                 img_v.size() * sizeof(float));

    std::vector<float> attn((size_t)S * H);
    cpu_attention_(jq.data(), jk.data(), jv.data(), S, NH, HD, attn.data());

    std::vector<float> img_attn_out((size_t)img_seq * H);
    cpu_matmul(attn.data() + (size_t)txt_seq * H, img_seq, H,
                w.to_out_w.data(), H, w.to_out_b.data(), img_attn_out.data());
    std::vector<float> txt_attn_out((size_t)txt_seq * H);
    cpu_matmul(attn.data(), txt_seq, H,
                w.to_add_out_w.data(), H, w.to_add_out_b.data(),
                txt_attn_out.data());

    cpu_gated_add_(img_h.data(), img_attn_out.data(), img_chunk(2),
                    B, img_seq, H);
    cpu_gated_add_(txt_h.data(), txt_attn_out.data(), txt_chunk(2),
                    B, txt_seq, H);

    std::vector<float> img_normed2 = img_h;
    cpu_layernorm_(img_normed2.data(), img_seq, H, cfg.layernorm_eps);
    cpu_modulate_(img_normed2.data(), B, img_seq, H,
                   img_chunk(3), img_chunk(4));
    std::vector<float> txt_normed2 = txt_h;
    cpu_layernorm_(txt_normed2.data(), txt_seq, H, cfg.layernorm_eps);
    cpu_modulate_(txt_normed2.data(), B, txt_seq, H,
                   txt_chunk(3), txt_chunk(4));

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

    cpu_gated_add_(img_h.data(), img_ff_out.data(), img_chunk(5),
                    B, img_seq, H);
    cpu_gated_add_(txt_h.data(), txt_ff_out.data(), txt_chunk(5),
                    B, txt_seq, H);
}

static void f16_to_f32_vec(const std::vector<uint16_t> &src,
                            std::vector<float> &dst) {
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); ++i) dst[i] = f16_to_f32(src[i]);
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
    // Keep stdout line-buffered so each progress line hits the log even if
    // we OOM in the middle of weight upload or CPU-reference chain.
    setvbuf(stdout, nullptr, _IOLBF, 0);

    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke42] CANN symbol load failed\n");
        return 1;
    }

    // Config: 60 blocks, small seq (keeps CPU reference tractable). Same
    // dimensions as real Qwen-Image-Edit-2511 DiT.
    ImageDiffusionConfig cfg;
    cfg.num_layers    = 60;
    cfg.num_heads     = 24;
    cfg.head_dim      = 128;
    cfg.hidden_size   = 3072;
    cfg.ff_mult       = 4;
    // Small seq: 64 img + 32 txt = 96 joint. CPU ref per block ~1 s × 60
    // blocks ≈ 60 s (parity run) + a one-time burn on NPU (first dispatch
    // compiles the op graph).
    bool small = true;
    if (const char *e = std::getenv("QIE_SMOKE_SMALL"))
        small = (e[0] != '0');
    if (small) {
        cfg.max_img_seq = 64;  cfg.max_txt_seq = 32;
    } else {
        cfg.max_img_seq = 256; cfg.max_txt_seq = 64;
    }
    cfg.precompute_rope = true;

    // Caller can override the number of actually-dispatched blocks via
    // QIE_N_BLOCKS=<k>; defaults to cfg.num_layers (60). Useful for
    // diagnosing divergence layer-by-layer.
    int n_blocks = cfg.num_layers;
    if (const char *e = std::getenv("QIE_N_BLOCKS")) {
        int k = std::atoi(e);
        if (k > 0 && k <= cfg.num_layers) n_blocks = k;
    }

    const int64_t img_seq = cfg.max_img_seq;
    const int64_t txt_seq = cfg.max_txt_seq;
    const int64_t H  = cfg.hidden_size;

    ImageDiffusionEngine eng;
    if (!eng.init_for_smoke(cfg, /*device*/ 0)) {
        fprintf(stderr, "[smoke42] init_for_smoke failed\n");
        return 1;
    }
    printf("[smoke42] engine scratch-alloc ok; generating synthetic weights...\n");
    fflush(stdout);

    // One shared weight set, uploaded once.
    HostWeights hw;
    gen_host_weights(cfg, hw, /*seed*/ 0xC0DE42ULL);
    printf("[smoke42] host weights generated; uploading to NPU...\n");
    fflush(stdout);
    UploadedWeights uw;
    if (!upload_shared(hw, uw)) {
        fprintf(stderr, "[smoke42] upload_shared failed\n");
        return 1;
    }
    printf("[smoke42] weights uploaded; wiring %d layer_w_ slots to shared buffers...\n",
           cfg.num_layers);
    fflush(stdout);
    for (int il = 0; il < cfg.num_layers; ++il) {
        DiTLayerWeights *lw = eng.mutable_layer_weights(il);
        if (!lw) { fprintf(stderr, "[smoke42] no layer %d\n", il); return 1; }
        point_layer_at_shared(*lw, uw);
    }
    printf("[smoke42] all %d layers wired; preparing activations...\n",
           cfg.num_layers);
    fflush(stdout);

    // Activations: img_hidden, txt_hidden, t_emb (all F16). Keep amplitudes
    // as in Phase 3 so the initial block sees the same input distribution.
    std::vector<uint16_t> img_h_f16, txt_h_f16, t_emb_f16;
    fill_random_f16(img_h_f16, (size_t)img_seq * H, 0.1f, 0x1111ULL);
    fill_random_f16(txt_h_f16, (size_t)txt_seq * H, 0.1f, 0x2222ULL);
    fill_random_f16(t_emb_f16, (size_t)H,             0.1f, 0x3333ULL);

    void *img_h_dev = upload_f16(img_h_f16.data(), img_h_f16.size());
    void *txt_h_dev = upload_f16(txt_h_f16.data(), txt_h_f16.size());
    void *t_emb_dev = upload_f16(t_emb_f16.data(), t_emb_f16.size());

    // Re-compute pe tables for CPU reference — same formula as init_for_smoke.
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
        (void)img_tokens;
    }

    void *pe_dev = upload_f16(pe_host.data(), pe_host.size());
    printf("[smoke42] pe uploaded; dispatching %d-block forward on NPU...\n",
           n_blocks);
    fflush(stdout);

    // -------- NPU dispatch: all n_blocks in one sync'd call --------
    std::vector<double> per_block_ms((size_t)n_blocks, 0.0);

    auto t0 = std::chrono::steady_clock::now();
    bool ok = eng.forward_all_blocks_test(img_h_dev, img_seq,
                                            txt_h_dev, txt_seq,
                                            t_emb_dev, pe_dev,
                                            per_block_ms.data(), n_blocks);
    // Final sync before D2H (forward_all_blocks_test only syncs per-block
    // when timing is requested, but for the D2H read we still want a
    // guaranteed full sync).
    g_cann.aclrtSynchronizeStream(nullptr);   // best-effort; per-block syncs
                                                // already ordered the stream
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "[smoke42] forward_all_blocks_test returned false\n");
        return 1;
    }
    printf("[smoke42] NPU forward complete (%.2f ms total); downloading outputs...\n",
           total_ms);
    fflush(stdout);

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

    // -------- CPU reference: n_blocks iterations of the same block --------
    // We apply the F16 requantize step between blocks to model the NPU's
    // F16 [img_seq, H] round-trip. Without it the CPU path keeps F32 precision
    // across the full 60-block chain and over-reports NPU drift.
    std::vector<float> img_h_ref, txt_h_ref, t_emb_ref;
    f16_to_f32_vec(img_h_f16, img_h_ref);
    f16_to_f32_vec(txt_h_f16, txt_h_ref);
    f16_to_f32_vec(t_emb_f16, t_emb_ref);

    printf("[smoke42] running CPU reference for %d blocks...\n", n_blocks);
    fflush(stdout);
    auto ref_t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < n_blocks; ++il) {
        cpu_block_forward(cfg, hw,
                          img_h_ref, txt_h_ref, t_emb_ref,
                          pe_host, img_seq, txt_seq);
        // Inter-block F16 quantization boundary — mirrors NPU's implicit
        // F16 round-trip at the block output.
        f32_requantize_via_f16(img_h_ref);
        f32_requantize_via_f16(txt_h_ref);
        if ((il + 1) % 10 == 0) {
            auto ref_tn = std::chrono::steady_clock::now();
            double ref_ms = std::chrono::duration<double, std::milli>(ref_tn - ref_t0).count();
            printf("[smoke42]   CPU block %d/%d done (cumulative %.1f s)\n",
                   il + 1, n_blocks, ref_ms / 1000.0);
            fflush(stdout);
        }
    }

    // -------- Compare --------
    StatsLine img_s = compare_f16_f32(img_h_out_f16.data(),
                                        img_h_ref.data(),
                                        img_h_out_f16.size());
    StatsLine txt_s = compare_f16_f32(txt_h_out_f16.data(),
                                        txt_h_ref.data(),
                                        txt_h_out_f16.size());

    // Per-block wall stats.
    double min_block = 1e30, max_block = -1e30, sum_block = 0.0;
    for (double ms : per_block_ms) {
        sum_block += ms;
        if (ms < min_block) min_block = ms;
        if (ms > max_block) max_block = ms;
    }
    std::vector<double> sorted_ms = per_block_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    double median_block = sorted_ms.empty()
        ? 0.0 : sorted_ms[sorted_ms.size() / 2];

    printf("\n========== Q2.4.2 Phase 4.2 60-block smoke report ==========\n");
    printf("config: H=%lld heads=%lld head_dim=%lld ff_dim=%lld layers=%d\n",
           (long long)cfg.hidden_size, (long long)cfg.num_heads,
           (long long)cfg.head_dim,
           (long long)cfg.hidden_size * cfg.ff_mult,
           n_blocks);
    printf("seq:    img=%lld  txt=%lld  joint=%lld\n",
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    printf("wall:   total=%.2f ms   per-block min=%.2f ms  median=%.2f ms  "
           "max=%.2f ms  sum=%.2f ms\n",
           total_ms, min_block, median_block, max_block, sum_block);
    printf("per-block ms (first 5 / last 5):  ");
    for (int i = 0; i < std::min(5, n_blocks); ++i)
        printf("%.2f ", per_block_ms[i]);
    printf("... ");
    for (int i = std::max(0, n_blocks - 5); i < n_blocks; ++i)
        printf("%.2f ", per_block_ms[i]);
    printf("\n");

    printf("\n-- img_hidden_out vs CPU-ref @ layer %d --\n", n_blocks);
    printf("  cos_sim  = %.6f\n", img_s.cos_sim);
    printf("  mae      = %.6f\n", img_s.mae);
    printf("  min/max  = %.4f / %.4f\n", img_s.min_npu, img_s.max_npu);
    printf("  NaN/inf  = %lld\n", (long long)img_s.nan_count);
    printf("-- txt_hidden_out vs CPU-ref @ layer %d --\n", n_blocks);
    printf("  cos_sim  = %.6f\n", txt_s.cos_sim);
    printf("  mae      = %.6f\n", txt_s.mae);
    printf("  min/max  = %.4f / %.4f\n", txt_s.min_npu, txt_s.max_npu);
    printf("  NaN/inf  = %lld\n", (long long)txt_s.nan_count);

    // Gate: cos_sim > 0.95 for both streams, NaN=0. 60-layer F16 accumulation
    // drift is expected — bar lowered from Phase 3's 0.99.
    const double gate = 0.95;
    bool pass = (img_s.cos_sim > gate) && (txt_s.cos_sim > gate) &&
                (img_s.nan_count == 0) && (txt_s.nan_count == 0);
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s (gate: cos_sim > %.2f both streams @ layer %d, NaN=0)\n",
           pass ? "GREEN" : "RED", gate, n_blocks);
    return pass ? 0 : 2;
}
