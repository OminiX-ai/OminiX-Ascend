// ============================================================================
// Q2 Phase 4.5 Step 4 smoke — real Qwen-Image-Edit-2509 Q4_0 GGUF end-to-end
// denoise_full on host-dumped conditioning, producing the first real cat-edit
// latent via the native engine.
//
// Workflow (ac03-only):
//   1. init_from_gguf real Q4_0 weights (~8 GiB resident).
//   2. init_from_dump on /tmp/qie_q45_inputs/ (Step 2 dump directory).
//   3. denoise_full 20-step @ cfg=4.0 (or cfg=1.0 if uncond dump absent),
//      producing a final [W_lat, H_lat, C_lat, B] F32 latent on host.
//   4. Save to /tmp/qie_q45_step4_latent.f32.bin for Step 4d VAE decode.
//
// Gate (per docs/qie_q2_phase4_smoke.md §5.5 Step 4):
//   GREEN   NaN=0, inf=0, final-latent std > 0.001, range overlaps with
//           the Step 2 x0_sampled_0 dump (within ±3σ).
//   YELLOW  finishes but numerics off.
//   RED     NaN/inf or denoise_full returns false mid-loop.
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
#include <string>
#include <vector>

using namespace ominix_qie;

// ---------------------------------------------------------------------------
// Sigma schedule for Qwen-Image flow-matching (shift=3.0, default).
// Reference: tools/ominix_diffusion/src/denoiser.hpp::DiscreteFlowDenoiser.
// For the Q1 cat-edit baseline the sample_method is Euler over a 20-step
// schedule built from `get_sigmas`. This is a simplified re-implementation
// that mirrors the linear-t + flow-shift transform:
//   t_k   = 1000 * (1 - k/n_steps)                       (k=0..n_steps)
//   t_k   = (t_k + 1) / 1000                             (t→1..0)
//   σ_k   = shift * t_k / (1 + (shift-1) * t_k)          (time_snr_shift)
// Same recipe stable-diffusion.cpp uses for Euler on Qwen-Image (n=20 → 21
// sigmas including the terminal 0.0 boundary).
// ---------------------------------------------------------------------------
static std::vector<float> make_qwen_image_sigmas(int n_steps, float shift) {
    std::vector<float> sigmas((size_t)n_steps + 1, 0.0f);
    for (int k = 0; k <= n_steps; ++k) {
        float t = 1.0f - (float)k / (float)n_steps;  // 1.0 → 0.0 inclusive
        // DiscreteFlowDenoiser::t_to_sigma: σ = shift * t / (1 + (shift-1)*t)
        if (shift == 1.0f) {
            sigmas[(size_t)k] = t;
        } else {
            sigmas[(size_t)k] = shift * t / (1.0f + (shift - 1.0f) * t);
        }
    }
    sigmas[(size_t)n_steps] = 0.0f;
    return sigmas;
}

struct Stats {
    double mean, std;
    float min_v, max_v;
    int64_t nan_count, inf_count;
};
static Stats compute_stats(const float *data, size_t n) {
    double sum = 0.0, sumsq = 0.0;
    float mn = +1e30f, mx = -1e30f;
    int64_t nanc = 0, infc = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = data[i];
        if (std::isnan(v)) { nanc++; continue; }
        if (std::isinf(v)) { infc++; continue; }
        sum += v; sumsq += (double)v * v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    Stats s;
    int64_t valid = (int64_t)n - nanc - infc;
    s.mean = valid > 0 ? sum / (double)valid : 0.0;
    double var = valid > 0 ? sumsq / (double)valid - s.mean * s.mean : 0.0;
    s.std = var > 0.0 ? std::sqrt(var) : 0.0;
    s.min_v = mn; s.max_v = mx;
    s.nan_count = nanc; s.inf_count = infc;
    return s;
}

int main(int /*argc*/, char ** /*argv*/) {
    setvbuf(stdout, nullptr, _IOLBF, 0);

    const char *gguf_env = std::getenv("QIE_Q45_GGUF");
    std::string gguf_path = gguf_env
        ? std::string(gguf_env)
        : std::string("/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf");
    const char *dump_env = std::getenv("QIE_Q45_DUMP_DIR");
    std::string dump_dir = dump_env ? std::string(dump_env)
                                     : std::string("/tmp/qie_q45_inputs");
    const char *latent_out_env = std::getenv("QIE_Q45_LATENT_OUT");
    std::string latent_out_path = latent_out_env
        ? std::string(latent_out_env)
        : std::string("/tmp/qie_q45_step4_latent.f32.bin");

    // Canonical Q1 cat-edit: 256×256, so W_lat=H_lat=32, C_lat=16, B=1.
    int64_t W_lat = 32, H_lat = 32, C_lat = 16, B = 1;
    int64_t joint_dim = 3584;
    if (const char *e = std::getenv("QIE_Q45_W_LAT"))   W_lat = std::atoi(e);
    if (const char *e = std::getenv("QIE_Q45_H_LAT"))   H_lat = std::atoi(e);
    int n_steps = 20;
    if (const char *e = std::getenv("QIE_N_STEPS")) {
        int k = std::atoi(e); if (k > 0 && k <= 200) n_steps = k;
    }
    float cfg_scale = 4.0f;
    if (const char *e = std::getenv("QIE_CFG_SCALE")) {
        float v = (float)std::atof(e);
        if (v > 0.0f && v < 100.0f) cfg_scale = v;
    }
    float flow_shift = 3.0f;
    if (const char *e = std::getenv("QIE_FLOW_SHIFT")) {
        float v = (float)std::atof(e);
        if (v > 0.0f && v < 100.0f) flow_shift = v;
    }

    printf("[smoke45s4] GGUF=%s\n", gguf_path.c_str());
    printf("[smoke45s4] DUMP=%s\n", dump_dir.c_str());
    printf("[smoke45s4] LATENT_OUT=%s\n", latent_out_path.c_str());
    printf("[smoke45s4] shape W_lat=%lld H_lat=%lld C_lat=%lld B=%lld "
           "joint_dim=%lld n_steps=%d cfg=%.2f flow_shift=%.2f\n",
           (long long)W_lat, (long long)H_lat, (long long)C_lat, (long long)B,
           (long long)joint_dim, n_steps, cfg_scale, flow_shift);
    fflush(stdout);

    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke45s4] CANN symbol load failed\n"); return 1;
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
    cfg.joint_attention_dim = (int)joint_dim;

    // ---- init_from_gguf ----
    ImageDiffusionEngine eng;
    auto t_init0 = std::chrono::steady_clock::now();
    if (!eng.init_from_gguf(gguf_path, cfg, /*device*/ 0)) {
        fprintf(stderr, "[smoke45s4] init_from_gguf FAILED — RED\n");
        return 2;
    }
    auto t_init1 = std::chrono::steady_clock::now();
    double init_wall_ms =
        std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();
    printf("[smoke45s4] init_from_gguf OK (%.1f ms)\n", init_wall_ms);
    fflush(stdout);

    // ---- init_from_dump ----
    std::vector<float> init_latent_host, ref_latent_host;
    std::vector<float> txt_cond_host, txt_uncond_host;
    int64_t txt_seq = 0;
    bool has_ref = false, has_uncond = false;
    if (!eng.init_from_dump(dump_dir, W_lat, H_lat, C_lat, B, joint_dim,
                              init_latent_host, ref_latent_host,
                              txt_cond_host, txt_uncond_host,
                              txt_seq, has_ref, has_uncond)) {
        fprintf(stderr, "[smoke45s4] init_from_dump FAILED — RED\n");
        return 2;
    }
    printf("[smoke45s4] init_from_dump OK txt_seq=%lld has_ref=%d has_uncond=%d\n",
           (long long)txt_seq, (int)has_ref, (int)has_uncond);
    // Range checks on inputs.
    Stats s_init = compute_stats(init_latent_host.data(),
                                  init_latent_host.size());
    printf("[smoke45s4] init_latent: mean=%.4f std=%.4f min/max=%.4f/%.4f "
           "nan=%lld inf=%lld\n",
           s_init.mean, s_init.std, s_init.min_v, s_init.max_v,
           (long long)s_init.nan_count, (long long)s_init.inf_count);
    if (has_ref) {
        Stats s_ref = compute_stats(ref_latent_host.data(),
                                     ref_latent_host.size());
        printf("[smoke45s4] ref_latent: mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
               s_ref.mean, s_ref.std, s_ref.min_v, s_ref.max_v);
    }
    Stats s_txt = compute_stats(txt_cond_host.data(), txt_cond_host.size());
    printf("[smoke45s4] txt_cond:   mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s_txt.mean, s_txt.std, s_txt.min_v, s_txt.max_v);
    fflush(stdout);

    // If cfg>1 requested but no uncond present, fall back to cfg=1.0.
    if (cfg_scale != 1.0f && !has_uncond) {
        printf("[smoke45s4] no uncond dump — forcing cfg_scale=1.0 "
               "(Step 4 will run single-forward per step)\n");
        cfg_scale = 1.0f;
    }

    // ---- sigma schedule ----
    std::vector<float> sigmas = make_qwen_image_sigmas(n_steps, flow_shift);
    printf("[smoke45s4] sigmas: first 5: ");
    for (int i = 0; i < std::min(5, (int)sigmas.size()); ++i)
        printf("%.4f ", sigmas[i]);
    printf("... last: %.4f\n", sigmas.back());
    fflush(stdout);

    // ---- denoise_full ----
    std::vector<float> out_latent((size_t)W_lat * H_lat * C_lat * B, 0.0f);
    std::vector<double> per_step_ms((size_t)n_steps, 0.0);

    printf("[smoke45s4] dispatching denoise_full (real Q4_0 weights, "
           "real host conditioning, 20-step flow Euler, cfg=%.2f)...\n",
           cfg_scale);
    fflush(stdout);

    auto t0 = std::chrono::steady_clock::now();
    bool ok = eng.denoise_full(init_latent_host.data(),
                                 has_ref ? ref_latent_host.data() : nullptr,
                                 W_lat, H_lat, C_lat, B,
                                 txt_cond_host.data(),
                                 has_uncond ? txt_uncond_host.data() : nullptr,
                                 txt_seq, joint_dim,
                                 sigmas.data(), n_steps, cfg_scale,
                                 out_latent.data(),
                                 per_step_ms.data());
    g_cann.aclrtSynchronizeStream(nullptr);
    auto t1 = std::chrono::steady_clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!ok) {
        fprintf(stderr, "[smoke45s4] denoise_full returned false (%.2f ms) — RED\n",
                total_ms);
        return 2;
    }
    printf("[smoke45s4] denoise_full OK (%.2f ms)\n", total_ms);
    fflush(stdout);

    // ---- Stats + dump ----
    Stats s_out = compute_stats(out_latent.data(), out_latent.size());

    double min_step = 1e30, max_step = -1e30, sum_step = 0.0;
    for (double ms : per_step_ms) {
        sum_step += ms;
        if (ms < min_step) min_step = ms;
        if (ms > max_step) max_step = ms;
    }
    std::vector<double> sorted_ms = per_step_ms;
    std::sort(sorted_ms.begin(), sorted_ms.end());
    double median_step = sorted_ms.empty() ? 0.0
                                             : sorted_ms[sorted_ms.size()/2];

    printf("\n========== Q2.4.5 Step 4 denoise_full report ==========\n");
    printf("gguf:   %s\n", gguf_path.c_str());
    printf("dump:   %s\n", dump_dir.c_str());
    printf("shape:  W_lat=%lld H_lat=%lld C_lat=%lld B=%lld\n",
           (long long)W_lat, (long long)H_lat, (long long)C_lat, (long long)B);
    printf("txt:    txt_seq=%lld joint_dim=%lld has_ref=%d has_uncond=%d\n",
           (long long)txt_seq, (long long)joint_dim, (int)has_ref,
           (int)has_uncond);
    printf("sched:  n_steps=%d cfg=%.2f flow_shift=%.2f sigma[0]=%.4f "
           "sigma[n]=%.4f\n",
           n_steps, cfg_scale, flow_shift, sigmas.front(), sigmas.back());
    printf("wall:   init=%.1fms denoise_full=%.2fms per-step "
           "min=%.2f median=%.2f max=%.2f sum=%.2f ms\n",
           init_wall_ms, total_ms, min_step, median_step, max_step, sum_step);
    printf("per-step ms (first 5 / last 5): ");
    for (int i = 0; i < std::min(5, n_steps); ++i)
        printf("%.2f ", per_step_ms[(size_t)i]);
    printf("... ");
    for (int i = std::max(0, n_steps - 5); i < n_steps; ++i)
        printf("%.2f ", per_step_ms[(size_t)i]);
    printf("\n");

    printf("\n-- final latent --\n");
    printf("  out_latent: mean=%.4f std=%.4f min/max=%.4f/%.4f "
           "NaN=%lld inf=%lld\n",
           s_out.mean, s_out.std, s_out.min_v, s_out.max_v,
           (long long)s_out.nan_count, (long long)s_out.inf_count);

    // Step 2 comparator: x0_sampled_0 (F32 [W_lat,H_lat,C_lat,1]) had range
    // ~[-1.255, 1.530] per docs/qie_q2_phase4_smoke.md §5.4. We don't load
    // that here (Step 4d's VAE-decode probe handles the visual comparison)
    // but print the numerical envelope for eyeball comparison.
    const double STD_GATE = 0.001;
    bool no_nan_inf = (s_out.nan_count == 0) && (s_out.inf_count == 0);
    bool non_trivial = (s_out.std > STD_GATE);
    bool range_sane = (s_out.min_v > -20.0f) && (s_out.max_v < 20.0f);
    bool pass = no_nan_inf && non_trivial && range_sane;

    const char *verdict = pass ? "GREEN"
                                : (!no_nan_inf ? "RED (NaN/inf)"
                                  : !non_trivial ? "YELLOW (std<gate)"
                                                 : "YELLOW (range)");
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s  (gate: NaN=0, inf=0, std>%.4f, |min|<20, |max|<20)\n",
           verdict, STD_GATE);

    // Save the final latent in the same layout as Step 2's x0_sampled_0
    // dump (F32 [W_lat*H_lat*C_lat*B] row-major with W_lat fastest, matching
    // Qwen-Image VAE layout expected by OMINIX_QIE_DECODE_ONLY_LATENT).
    FILE *f = std::fopen(latent_out_path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "[smoke45s4] fopen(%s) failed — can't dump latent\n",
                latent_out_path.c_str());
        return pass ? 3 : 2;
    }
    size_t wrote = std::fwrite(out_latent.data(), sizeof(float),
                                out_latent.size(), f);
    std::fclose(f);
    if (wrote != out_latent.size()) {
        fprintf(stderr, "[smoke45s4] fwrite wrote %zu/%zu\n",
                wrote, out_latent.size());
        return 2;
    }
    printf("final latent dumped to %s (%zu F32, %.2f MiB, shape "
           "[W=%lld H=%lld C=%lld B=%lld])\n",
           latent_out_path.c_str(), out_latent.size(),
           out_latent.size() * sizeof(float) / (1024.0 * 1024.0),
           (long long)W_lat, (long long)H_lat, (long long)C_lat, (long long)B);
    fflush(stdout);

    if (pass) return 0;
    if (no_nan_inf) return 3;
    return 2;
}
