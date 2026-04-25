// ============================================================================
// Q2 Phase 4.5 Step 1 smoke — real Qwen-Image-Edit-2509 Q4_0 GGUF + 20-step
// Euler-flow denoise.
//
// Scope: Phase 4.5 per docs/qie_q2_phase4_smoke.md §5. First real-weight
// end-to-end denoise on the native engine — **the primary unknown** is
// whether the Phase 4.4d F32-residual fix (proven at N=60 blocks, single
// forward) holds across 20 steps × 2 CFG × 60 blocks = 2400 block dispatches.
//
// Step 1 of the Phase 4.5 workplan intentionally stops short of real
// conditioning (text-encoder + VAE-encode) — those arrive in Step 2. Here
// the txt_cond / txt_uncond streams are deterministic random F32 buffers,
// same as Phase 4.3 but with real DiT weights. This isolates the
// "real-weight denoise stability" risk from the "host-side conditioning
// pipeline" risk.
//
// Harness:
//   - Boots ImageDiffusionEngine via init_from_gguf(real Q4_0 gguf) — same
//     load path as Phase 4.4d.
//   - Generates deterministic random F32 x / txt_cond / txt_uncond and
//     F16 t_emb at the Phase 4.3/4.4 small activation shape (img=64 txt=32)
//     OR at production 1024x1024 equivalent (img=256 txt=64 when
//     QIE_Q45_BIG=1) — production-size run is the Phase 4.5 gate, small
//     run is the dev checkpoint.
//   - Calls ImageDiffusionEngine::denoise_loop_test — 20 steps, cfg_scale=4.0.
//   - Downloads final x latent; checks NaN=0 AND std > 0.001 over the
//     whole run.
//
// Phase 4.5 Step 1 gate:
//   GREEN   20 steps complete, NaN=0, inf=0, std > 0.001.
//   YELLOW  loop finishes but numerics off (eg std<gate).
//   RED     NaN/inf during the loop, or loop returns false mid-way.
//
// Build on ac03:
//   cd tools/probes/qie_q45_real_denoise_smoke && bash build_and_run.sh
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
#include <string>
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

static void fill_random_f16(std::vector<uint16_t> &out, size_t n,
                             float amp, uint64_t seed) {
    out.assign(n, 0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16(dist(rng));
}

static void fill_random_f32_via_f16(std::vector<float> &out, size_t n,
                                      float amp, uint64_t seed) {
    out.assign(n, 0.0f);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n; ++i) {
        uint16_t h = f32_to_f16(dist(rng));
        out[i] = f16_to_f32(h);
    }
}

static void *upload_f16(const uint16_t *host, size_t n) {
    void *dev = nullptr;
    size_t bytes = n * sizeof(uint16_t);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[smoke45] aclrtMalloc(%zu) err=%d\n", bytes, (int)err);
        return nullptr;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        fprintf(stderr, "[smoke45] H2D memcpy err=%d\n", (int)err);
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

static LatentStats compute_f32_stats(const float *data, size_t n) {
    double sum = 0.0, sumsq = 0.0;
    float  mn = +1e30f, mx = -1e30f;
    int64_t nanc = 0, infc = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = data[i];
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
// Sigma schedule for flow-matching: linearly space from 1.0 down to 0.0
// across `n_steps + 1` points. Matches Qwen-Image flow-match convention
// (shift=1.0 identity — a real engine will apply the flow-shift transform).
// ---------------------------------------------------------------------------
static std::vector<float> make_flow_sigmas(int n_steps) {
    std::vector<float> sigmas(n_steps + 1, 0.0f);
    for (int i = 0; i <= n_steps; ++i) {
        sigmas[i] = 1.0f - (float)i / (float)n_steps;
    }
    sigmas[n_steps] = 0.0f;
    return sigmas;
}

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    const char *gguf_env = std::getenv("QIE_Q45_GGUF");
    std::string gguf_path = gguf_env
        ? std::string(gguf_env)
        : std::string("/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf");

    printf("[smoke45] GGUF: %s\n", gguf_path.c_str());
    fflush(stdout);

    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke45] CANN symbol load failed\n");
        return 1;
    }

    ImageDiffusionConfig cfg;
    cfg.num_layers    = 60;
    cfg.num_heads     = 24;
    cfg.head_dim      = 128;
    cfg.hidden_size   = 3072;
    cfg.ff_mult       = 4;
    cfg.max_img_seq   = 4096;
    cfg.max_txt_seq   = 256;
    cfg.precompute_rope = true;

    // Activation shapes — default matches Phase 4.3/4.4 small smoke.
    // QIE_Q45_BIG=1 runs at img=256 txt=64 (production-ish, matches Q43
    // non-small mode); production 256x256 edit is seq ~ 512 img + 32 txt.
    int64_t img_seq = 64;
    int64_t txt_seq = 32;
    if (const char *e = std::getenv("QIE_Q45_BIG"); e && e[0] != '0') {
        img_seq = 256;
        txt_seq = 64;
    }
    if (const char *e = std::getenv("QIE_Q45_IMG_SEQ"))  img_seq = std::atoi(e);
    if (const char *e = std::getenv("QIE_Q45_TXT_SEQ"))  txt_seq = std::atoi(e);

    // Q2.4.5.4k diagnostic knob: align cfg.max_txt_seq to actual txt_seq so
    // the pre-built RoPE pe table places img positions at row=txt_seq
    // (matching the CPU reference's gen_qwen_image_pe(context_len=txt_seq)
    // layout). The native engine builds pe with img rows offset by
    // max_txt_seq=256 vs CPU's offset by txt_seq=32 — a layout-mismatch
    // candidate cause for the §5.5.10 attn_out cos=0.002 RED.
    //
    // Empirical result (this run): the override does NOT move 11_attn_out
    // cos — the divergence enters elsewhere (suspected: a deeper layout or
    // weight-loading discrepancy rather than pe-row alignment). Kept as an
    // env-gated diagnostic. Opt in via QIE_RUNTIME_MAX_TXT_SEQ=1 to test
    // pe alignment hypotheses; default OFF preserves Phase 4.4d behaviour.
    bool runtime_max_txt = false;
    if (const char *e = std::getenv("QIE_RUNTIME_MAX_TXT_SEQ"))
        runtime_max_txt = (e[0] != '0');
    if (runtime_max_txt) cfg.max_txt_seq = (int)txt_seq;

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

    const int64_t H = cfg.hidden_size;

    printf("[smoke45] config: layers=%d heads=%d head_dim=%d hidden=%d ff=%d\n",
           cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.hidden_size,
           cfg.hidden_size * cfg.ff_mult);
    printf("[smoke45] max_seq (for scratch): img=%d txt=%d  "
           "forward shape: img=%lld txt=%lld joint=%lld\n",
           cfg.max_img_seq, cfg.max_txt_seq,
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    printf("[smoke45] sched: n_steps=%d cfg_scale=%.2f\n", n_steps, cfg_scale);
    if (const char *q = std::getenv("GGML_CANN_QUANT_BF16")) {
        printf("[smoke45] GGML_CANN_QUANT_BF16=%s\n", q);
    }
    fflush(stdout);

    // ---- init_from_gguf on the real production weights ----
    ImageDiffusionEngine eng;
    auto t_init0 = std::chrono::steady_clock::now();
    bool init_ok = eng.init_from_gguf(gguf_path, cfg, /*device*/ 0);
    auto t_init1 = std::chrono::steady_clock::now();
    double init_wall_ms =
        std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();

    if (!init_ok) {
        fprintf(stderr, "[smoke45] init_from_gguf FAILED after %.1f ms — RED\n",
                init_wall_ms);
        return 2;
    }
    if (!eng.is_ready()) {
        fprintf(stderr, "[smoke45] init_from_gguf returned true but "
                        "is_ready()=false — RED\n");
        return 2;
    }
    printf("[smoke45] init_from_gguf OK (wall %.1f ms)\n", init_wall_ms);
    fflush(stdout);

    const DiTInitStats &st = eng.stats();
    const size_t total_bytes =
        st.q4_weight_bytes + st.q4_scale_bytes +
        st.f16_weight_bytes + st.f32_weight_bytes +
        st.rope_bytes + st.scratch_bytes;
    const double peak_gib = total_bytes / (1024.0 * 1024.0 * 1024.0);

    printf("\n========== Phase 4.5 Step 1 init receipts ==========\n");
    printf("  tensors uploaded:    %lld (Q4-resident=%lld, F16-fallback=%lld)\n",
           (long long)st.tensors_uploaded,
           (long long)st.q4_tensors,
           (long long)st.f16_fallback_tensors);
    printf("  Q4 weight bytes:     %.2f GiB\n",
           st.q4_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  F16 fallback bytes:  %.2f GiB\n",
           st.f16_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Peak init HBM:       %.2f GiB\n", peak_gib);
    printf("  Total init wall:     %.1f ms\n", st.load_wall_ms);
    fflush(stdout);

    // ---- Dummy activations at Phase 4.4c F32 contract ----
    std::vector<float>    x_img_f32, txt_cond_f32, txt_uncond_f32;
    std::vector<uint16_t> t_emb_f16;
    fill_random_f32_via_f16(x_img_f32,      (size_t)img_seq * H, 0.1f, 0x4511ULL);
    fill_random_f32_via_f16(txt_cond_f32,   (size_t)txt_seq * H, 0.1f, 0x4522ULL);
    fill_random_f32_via_f16(txt_uncond_f32, (size_t)txt_seq * H, 0.1f, 0x45AAULL);
    fill_random_f16        (t_emb_f16,      (size_t)H,             0.1f, 0x4533ULL);

    void *x_dev          = upload_f32(x_img_f32.data(),      x_img_f32.size());
    void *txt_cond_dev   = upload_f32(txt_cond_f32.data(),   txt_cond_f32.size());
    void *txt_uncond_dev = upload_f32(txt_uncond_f32.data(), txt_uncond_f32.size());
    void *t_emb_dev      = upload_f16(t_emb_f16.data(),      t_emb_f16.size());
    if (!x_dev || !txt_cond_dev || !txt_uncond_dev || !t_emb_dev) {
        fprintf(stderr, "[smoke45] activation upload failed\n");
        return 1;
    }

    // Reuse the engine's pre-built RoPE pe table (populated by init_from_gguf).
    void *pe_dev = eng.rope_pe_dev_for_test();
    if (!pe_dev) {
        fprintf(stderr, "[smoke45] engine rope_pe_dev not populated\n");
        return 1;
    }

    // Sigma schedule.
    std::vector<float> sigmas = make_flow_sigmas(n_steps);
    printf("[smoke45] sigma schedule (first 5): ");
    for (int i = 0; i < std::min(5, (int)sigmas.size()); ++i)
        printf("%.4f ", sigmas[i]);
    printf("... (last): %.4f\n", sigmas.back());
    fflush(stdout);

    // Baseline stats on initial x (F32).
    LatentStats s0 = compute_f32_stats(x_img_f32.data(), x_img_f32.size());
    printf("\n[smoke45] x_init: mean=%.4f std=%.4f min=%.4f max=%.4f "
           "nan=%lld inf=%lld\n",
           s0.mean, s0.std, s0.min_v, s0.max_v,
           (long long)s0.nan_count, (long long)s0.inf_count);
    fflush(stdout);

    // ---- Run the real-weight 20-step denoise loop ----
    std::vector<double> per_step_ms((size_t)n_steps, 0.0);
    printf("[smoke45] dispatching denoise_loop_test (n_steps=%d, cfg=%.2f, "
           "real Q4_0 GGUF weights)...\n", n_steps, cfg_scale);
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
        fprintf(stderr, "[smoke45] denoise_loop_test returned false "
                         "(after %.2f ms); see engine log — RED\n", total_ms);
        return 2;
    }
    printf("[smoke45] denoise_loop_test OK (%.2f ms)\n", total_ms);
    fflush(stdout);

    // ---- D2H download of final x (F32) ----
    std::vector<float> x_out_f32(x_img_f32.size());
    aclError me = g_cann.aclrtMemcpy(x_out_f32.data(),
                                       x_out_f32.size() * sizeof(float),
                                       x_dev,
                                       x_out_f32.size() * sizeof(float),
                                       ACL_MEMCPY_DEVICE_TO_HOST);
    if (me != 0) {
        fprintf(stderr, "[smoke45] D2H memcpy err=%d\n", (int)me);
        return 1;
    }
    LatentStats s1 = compute_f32_stats(x_out_f32.data(), x_out_f32.size());

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

    printf("\n========== Q2.4.5 Phase 4.5 Step 1 denoise smoke report ==========\n");
    printf("gguf:   %s\n", gguf_path.c_str());
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
    printf("hbm:    peak_init=%.2f GiB\n", peak_gib);
    printf("wall:   init=%.1f ms  denoise_total=%.2f ms   "
           "per-step min=%.2f ms  median=%.2f ms  max=%.2f ms  sum=%.2f ms\n",
           st.load_wall_ms, total_ms, min_step, median_step, max_step, sum_step);
    printf("per-step ms (first 5 / last 5):  ");
    for (int i = 0; i < std::min(5, n_steps); ++i)
        printf("%.2f ", per_step_ms[i]);
    printf("... ");
    for (int i = std::max(0, n_steps - 5); i < n_steps; ++i)
        printf("%.2f ", per_step_ms[i]);
    printf("\n");

    printf("\n-- output latent vs input --\n");
    printf("  x_init : mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s0.mean, s0.std, s0.min_v, s0.max_v);
    printf("  x_out  : mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s1.mean, s1.std, s1.min_v, s1.max_v);
    printf("  NaN=%lld  inf=%lld\n",
           (long long)s1.nan_count, (long long)s1.inf_count);

    const double STD_GATE = 0.001;
    bool no_nan_inf = (s1.nan_count == 0) && (s1.inf_count == 0);
    bool non_trivial = (s1.std > STD_GATE);
    bool pass = no_nan_inf && non_trivial;

    const char *verdict = pass ? "GREEN"
                                : (!no_nan_inf ? "RED (NaN/inf)"
                                               : "YELLOW (std < gate)");
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s  (gate: NaN=0, inf=0, std > %.4f over final latent)\n",
           verdict, STD_GATE);

    // ---- Save final latent for Step 2+ VAE decode consumption ----
    const char *dump_path_env = std::getenv("QIE_Q45_LATENT_OUT");
    std::string dump_path = dump_path_env
        ? std::string(dump_path_env)
        : std::string("/tmp/qie_q45_final_latent.f32.bin");
    FILE *f = std::fopen(dump_path.c_str(), "wb");
    if (f) {
        size_t wrote = std::fwrite(x_out_f32.data(), sizeof(float),
                                    x_out_f32.size(), f);
        std::fclose(f);
        if (wrote == x_out_f32.size()) {
            printf("final latent dumped to %s (%zu F32 elts, %.2f MiB, "
                   "shape [img_seq=%lld, H=%lld])\n",
                   dump_path.c_str(), x_out_f32.size(),
                   x_out_f32.size() * sizeof(float) / (1024.0 * 1024.0),
                   (long long)img_seq, (long long)H);
        } else {
            printf("final latent dump FAILED (fwrite returned %zu)\n", wrote);
        }
    }
    fflush(stdout);

    if (pass) return 0;
    if (no_nan_inf) return 3;
    return 2;
}
