// ============================================================================
// Q2 Phase 4.3 smoke — Euler-flow 20-step denoise loop.
//
// Scope: Phase 4.3 per docs/qie_q2_phase4_smoke.md §3. Exercises the full
// Euler-flow CFG-aware denoise loop end-to-end:
//   for step in [0, 20):
//       eps_cond   = forward_all_blocks(x, t_emb, txt_cond)
//       eps_uncond = forward_all_blocks(x, t_emb, txt_uncond)
//       eps        = eps_uncond + cfg * (eps_cond - eps_uncond)
//       dt         = sigmas[step+1] - sigmas[step]
//       x         += dt * eps
//
// Total NPU work: 20 × (cond_forward + uncond_forward + CFG_compose + axpy)
// = 40 × 60-block forward + 20 × scheduler.
//
// Harness:
//   - Boots ImageDiffusionEngine via init_for_smoke() (no GGUF load).
//   - Synthesizes ONE deterministic random F16 weight set, aliases it into
//     every layer_w_[il] slot (same pattern as Phase 4.2).
//   - Generates random F16 x (latent-shape), txt_cond, txt_uncond, t_emb.
//   - Calls ImageDiffusionEngine::denoise_loop_test with flow-matching
//     sigma schedule linearly spaced on (1.0, 0.0].
//   - Checks: no NaN/inf, std > 0.001 (latent non-trivial), prints per-step
//     wall and min/median/max.
//
// Phase 4.3 gate: 20 steps run without crash; NaN=0; std > 0.001; total
// wall-clock reported for 40 forward passes + 20 scheduler steps.
//
// Build on ac03:
//   cd tools/probes/qie_q43_denoise_smoke && bash build_and_run.sh
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
// F16 <-> F32 helpers.
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
        fprintf(stderr, "[smoke43] aclrtMalloc(%zu) err=%d\n", bytes, (int)err);
        return nullptr;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        fprintf(stderr, "[smoke43] H2D memcpy err=%d\n", (int)err);
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
// HostWeights / UploadedWeights — same layout as Phase 4.2 smoke.
// ---------------------------------------------------------------------------
struct HostWeights {
    std::vector<uint16_t> to_q_w, to_k_w, to_v_w, to_out_w;
    std::vector<uint16_t> to_q_b, to_k_b, to_v_b, to_out_b;
    std::vector<uint16_t> add_q_w, add_k_w, add_v_w, to_add_out_w;
    std::vector<uint16_t> add_q_b, add_k_b, add_v_b, to_add_out_b;
    std::vector<float>    norm_q_w, norm_k_w, norm_added_q_w, norm_added_k_w;
    std::vector<uint16_t> img_mod_w, txt_mod_w;
    std::vector<uint16_t> img_mod_b, txt_mod_b;
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
        // Same tightening as Phase 4.2: modulation amplitude small enough
        // that 60-block composition stays inside F16.
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

struct UploadedWeights {
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
    return true;
}

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
// Same RoPE table construction as Phase 4.2 (keeps smoke self-contained).
// ---------------------------------------------------------------------------
static void build_pe_host(const ImageDiffusionConfig &cfg,
                           std::vector<uint16_t> &pe_host) {
    const int axes_t   = cfg.rope_axes_temporal;
    const int axes_h   = cfg.rope_axes_h;
    const int axes_w   = cfg.rope_axes_w;
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

// ---------------------------------------------------------------------------
// Sigma schedule for flow-matching: linearly space from 1.0 down to 0.0
// across `n_steps + 1` points. Matches Qwen-Image flow-match convention
// (shift=1.0 identity — a real engine will apply the flow-shift transform).
// ---------------------------------------------------------------------------
static std::vector<float> make_flow_sigmas(int n_steps) {
    std::vector<float> sigmas(n_steps + 1, 0.0f);
    for (int i = 0; i <= n_steps; ++i) {
        sigmas[i] = 1.0f - (float)i / (float)n_steps;
    }
    sigmas[n_steps] = 0.0f;  // ensure terminal is exactly zero
    return sigmas;
}

// ---------------------------------------------------------------------------
// Stats helpers.
// ---------------------------------------------------------------------------
struct LatentStats {
    double mean;
    double std;
    float  min_v;
    float  max_v;
    int64_t nan_count;
    int64_t inf_count;
};

static LatentStats compute_f16_stats(const uint16_t *data, size_t n) {
    double sum = 0.0, sumsq = 0.0;
    float  mn = +1e30f, mx = -1e30f;
    int64_t nanc = 0, infc = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = f16_to_f32(data[i]);
        if (std::isnan(v)) { nanc++; continue; }
        if (std::isinf(v)) { infc++; continue; }
        sum   += v;
        sumsq += (double)v * (double)v;
        mn = std::min(mn, v);
        mx = std::max(mx, v);
    }
    LatentStats s;
    int64_t valid = (int64_t)n - nanc - infc;
    if (valid <= 0) {
        s.mean = 0.0; s.std = 0.0;
        s.min_v = 0.0f; s.max_v = 0.0f;
    } else {
        s.mean = sum / (double)valid;
        double var = sumsq / (double)valid - s.mean * s.mean;
        s.std = var > 0.0 ? std::sqrt(var) : 0.0;
        s.min_v = mn; s.max_v = mx;
    }
    s.nan_count = nanc;
    s.inf_count = infc;
    return s;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke43] CANN symbol load failed\n");
        return 1;
    }

    ImageDiffusionConfig cfg;
    cfg.num_layers    = 60;
    cfg.num_heads     = 24;
    cfg.head_dim      = 128;
    cfg.hidden_size   = 3072;
    cfg.ff_mult       = 4;
    // Small seq keeps the 40 × 60-block forward-pass wall tractable
    // (Phase 4.2 measured ~1.4 s per 60-block run at this shape; 40 runs
    // ≈ 56 s denoise wall).
    bool small = true;
    if (const char *e = std::getenv("QIE_SMOKE_SMALL"))
        small = (e[0] != '0');
    if (small) {
        cfg.max_img_seq = 64;  cfg.max_txt_seq = 32;
    } else {
        cfg.max_img_seq = 256; cfg.max_txt_seq = 64;
    }
    cfg.precompute_rope = true;

    // Knobs.
    int n_steps = 20;
    if (const char *e = std::getenv("QIE_N_STEPS")) {
        int k = std::atoi(e);
        if (k > 0 && k <= 200) n_steps = k;
    }
    float cfg_scale = 4.0f;
    if (const char *e = std::getenv("QIE_CFG_SCALE")) {
        float v = (float)std::atof(e);
        if (v > 0.0f && v < 100.0f) cfg_scale = v;
    }

    const int64_t img_seq = cfg.max_img_seq;
    const int64_t txt_seq = cfg.max_txt_seq;
    const int64_t H       = cfg.hidden_size;

    printf("[smoke43] config: layers=%d heads=%d head_dim=%d hidden=%d ff=%d\n",
           cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.hidden_size,
           cfg.hidden_size * cfg.ff_mult);
    printf("[smoke43] seq:    img=%lld txt=%lld joint=%lld\n",
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    printf("[smoke43] sched:  n_steps=%d cfg_scale=%.2f\n",
           n_steps, cfg_scale);
    fflush(stdout);

    ImageDiffusionEngine eng;
    if (!eng.init_for_smoke(cfg, /*device*/ 0)) {
        fprintf(stderr, "[smoke43] init_for_smoke failed\n");
        return 1;
    }
    printf("[smoke43] engine scratch-alloc ok; generating synthetic weights...\n");
    fflush(stdout);

    HostWeights hw;
    gen_host_weights(cfg, hw, /*seed*/ 0xC0DE43ULL);
    printf("[smoke43] host weights generated; uploading to NPU...\n");
    fflush(stdout);
    UploadedWeights uw;
    if (!upload_shared(hw, uw)) {
        fprintf(stderr, "[smoke43] upload_shared failed\n");
        return 1;
    }
    printf("[smoke43] weights uploaded; wiring %d layer_w_ slots...\n",
           cfg.num_layers);
    fflush(stdout);
    for (int il = 0; il < cfg.num_layers; ++il) {
        DiTLayerWeights *lw = eng.mutable_layer_weights(il);
        if (!lw) { fprintf(stderr, "[smoke43] no layer %d\n", il); return 1; }
        point_layer_at_shared(*lw, uw);
    }

    // Activations.
    std::vector<uint16_t> x_f16, txt_cond_f16, txt_uncond_f16, t_emb_f16;
    fill_random_f16(x_f16,           (size_t)img_seq * H, 0.1f, 0x1111ULL);
    fill_random_f16(txt_cond_f16,    (size_t)txt_seq * H, 0.1f, 0x2222ULL);
    fill_random_f16(txt_uncond_f16,  (size_t)txt_seq * H, 0.1f, 0xAAAAULL);
    fill_random_f16(t_emb_f16,       (size_t)H,             0.1f, 0x3333ULL);

    void *x_dev          = upload_f16(x_f16.data(),          x_f16.size());
    void *txt_cond_dev   = upload_f16(txt_cond_f16.data(),   txt_cond_f16.size());
    void *txt_uncond_dev = upload_f16(txt_uncond_f16.data(), txt_uncond_f16.size());
    void *t_emb_dev      = upload_f16(t_emb_f16.data(),      t_emb_f16.size());

    // pe table.
    std::vector<uint16_t> pe_host;
    build_pe_host(cfg, pe_host);
    void *pe_dev = upload_f16(pe_host.data(), pe_host.size());

    // Sigma schedule.
    std::vector<float> sigmas = make_flow_sigmas(n_steps);
    printf("[smoke43] sigma schedule (first 5): ");
    for (int i = 0; i < std::min(5, (int)sigmas.size()); ++i)
        printf("%.4f ", sigmas[i]);
    printf("... (last): %.4f\n", sigmas.back());
    fflush(stdout);

    // Baseline stats of initial x.
    LatentStats s0 = compute_f16_stats(x_f16.data(), x_f16.size());
    printf("[smoke43] x_init: mean=%.4f std=%.4f min=%.4f max=%.4f nan=%lld inf=%lld\n",
           s0.mean, s0.std, s0.min_v, s0.max_v,
           (long long)s0.nan_count, (long long)s0.inf_count);
    fflush(stdout);

    // Run the denoise loop.
    std::vector<double> per_step_ms((size_t)n_steps, 0.0);
    printf("[smoke43] dispatching denoise_loop_test (n_steps=%d, cfg=%.2f)...\n",
           n_steps, cfg_scale);
    fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    bool ok = eng.denoise_loop_test(x_dev, img_seq,
                                      txt_cond_dev, txt_uncond_dev, txt_seq,
                                      t_emb_dev, pe_dev,
                                      sigmas.data(), n_steps, cfg_scale,
                                      per_step_ms.data());
    g_cann.aclrtSynchronizeStream(nullptr);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "[smoke43] denoise_loop_test returned false "
                         "(after %.2f ms); see engine log\n", total_ms);
        return 2;
    }
    printf("[smoke43] denoise_loop_test OK (%.2f ms total)\n", total_ms);
    fflush(stdout);

    // D2H download of final x.
    std::vector<uint16_t> x_out_f16(x_f16.size());
    aclError me = g_cann.aclrtMemcpy(x_out_f16.data(),
                                       x_out_f16.size() * sizeof(uint16_t),
                                       x_dev,
                                       x_out_f16.size() * sizeof(uint16_t),
                                       ACL_MEMCPY_DEVICE_TO_HOST);
    if (me != 0) {
        fprintf(stderr, "[smoke43] D2H memcpy err=%d\n", (int)me);
        return 1;
    }

    LatentStats s1 = compute_f16_stats(x_out_f16.data(), x_out_f16.size());

    // Per-step stats.
    double min_step = 1e30, max_step = -1e30, sum_step = 0.0;
    for (double ms : per_step_ms) {
        sum_step += ms;
        if (ms < min_step) min_step = ms;
        if (ms > max_step) max_step = ms;
    }
    std::vector<double> sorted_ms = per_step_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    double median_step = sorted_ms.empty()
        ? 0.0 : sorted_ms[sorted_ms.size() / 2];

    printf("\n========== Q2.4.3 Phase 4.3 Euler denoise smoke report ==========\n");
    printf("config: H=%lld heads=%lld head_dim=%lld ff_dim=%lld layers=%d\n",
           (long long)cfg.hidden_size, (long long)cfg.num_heads,
           (long long)cfg.head_dim,
           (long long)cfg.hidden_size * cfg.ff_mult,
           cfg.num_layers);
    printf("seq:    img=%lld  txt=%lld  joint=%lld\n",
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    printf("sched:  n_steps=%d cfg_scale=%.2f sigma_max=%.4f sigma_min=%.4f\n",
           n_steps, cfg_scale, sigmas.front(), sigmas.back());
    printf("wall:   total=%.2f ms   per-step min=%.2f ms  median=%.2f ms  "
           "max=%.2f ms  sum=%.2f ms\n",
           total_ms, min_step, median_step, max_step, sum_step);
    printf("per-step ms (first 5 / last 5):  ");
    for (int i = 0; i < std::min(5, n_steps); ++i)
        printf("%.2f ", per_step_ms[i]);
    printf("... ");
    for (int i = std::max(0, n_steps - 5); i < n_steps; ++i)
        printf("%.2f ", per_step_ms[i]);
    printf("\n");

    printf("\n-- final latent x vs input --\n");
    printf("  x_init : mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s0.mean, s0.std, s0.min_v, s0.max_v);
    printf("  x_final: mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s1.mean, s1.std, s1.min_v, s1.max_v);
    printf("  NaN=%lld  inf=%lld\n",
           (long long)s1.nan_count, (long long)s1.inf_count);

    // Gate:
    //   - No NaN/inf in final latent
    //   - std > 0.001 (non-trivial / non-constant)
    //   - all steps completed successfully (already enforced by ok=true)
    const double STD_GATE = 0.001;
    bool pass = (s1.nan_count == 0) && (s1.inf_count == 0)
             && (s1.std > STD_GATE);
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s (gate: NaN=0, inf=0, std > %.4f over final latent)\n",
           pass ? "GREEN" : "RED", STD_GATE);
    return pass ? 0 : 2;
}
