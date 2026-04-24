// ============================================================================
// Q2 Phase 4.4 smoke — real Qwen-Image-Edit-2509 Q4_0 GGUF + single forward.
//
// Scope: Phase 4.4 per docs/qie_q2_phase4_smoke.md §4. Exercises the *real*
// `init_from_gguf` load path end-to-end (Q2.1 only smoked the load itself —
// forward pass was never run against production weights) and then fires a
// *single* 60-block forward at the Phase 4.2/4.3 small activation shape
// (img_seq=64, txt_seq=32) to confirm the engine doesn't NaN/crash when the
// matmul-weight pointers are the Q4-resident + F16-fallback pairs that the
// real GGUF produces (as opposed to the synthetic-F16-only aliases used in
// Phases 4.2/4.3).
//
// Harness:
//   - Boots ImageDiffusionEngine via init_from_gguf(gguf_path, cfg_prod).
//     * cfg keeps prod max_img_seq=4096, max_txt_seq=256 so scratch matches
//       Q2.1's 17.74 GiB peak projection; only the forward's *runtime* seq is
//       cut to 64/32.
//   - Prints the Q2.1-style HBM receipts harvested from engine.stats()
//     (Q4 tensors, Q4 bytes, F16-fallback tensors, peak HBM).
//   - Uploads random F16 img/txt activations + t_emb at the smoke shape.
//   - Calls forward_all_blocks_test ONCE (single 60-block pass).
//   - Downloads the img output; checks no NaN, no inf, std > 0.001.
//
// Phase 4.4 gate:
//   GREEN   real GGUF init_from_gguf returns true AND single forward completes
//           with NaN=0, inf=0, std > 0.001.
//   YELLOW  load OK, forward completes without crash but numerics are off
//           (eg std < 0.001 or non-trivial nan count but not all-nan).
//   RED     load crashes/fails OR OOM OR forward returns false/crashes.
//
// Peak HBM gate carried over from Q2.1:  ≤ 18 GiB (contract §Q1.10). Purely
// informational — the probe verdict is numerics-based; HBM is reported for
// the receipts blob only.
//
// Build on ac03:
//   GGML_BUILD=/home/ma-user/work/OminiX-Ascend/build-w1 \
//   GGML_CANN_QUANT_BF16=on \
//   bash build_and_run.sh
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

// ---------------------------------------------------------------------------
// Deterministic random fill helper.
// ---------------------------------------------------------------------------
static void fill_random_f16(std::vector<uint16_t> &out, size_t n,
                             float amp, uint64_t seed) {
    out.assign(n, 0);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-amp, amp);
    for (size_t i = 0; i < n; ++i) out[i] = f32_to_f16(dist(rng));
}

static void *upload_f16(const uint16_t *host, size_t n) {
    void *dev = nullptr;
    size_t bytes = n * sizeof(uint16_t);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        fprintf(stderr, "[smoke44] aclrtMalloc(%zu) err=%d\n", bytes, (int)err);
        return nullptr;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                              ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        fprintf(stderr, "[smoke44] H2D memcpy err=%d\n", (int)err);
        g_cann.aclrtFree(dev);
        return nullptr;
    }
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

    // ---- GGUF path (override via env if needed) ----
    const char *gguf_env = std::getenv("QIE_Q44_GGUF");
    std::string gguf_path = gguf_env
        ? std::string(gguf_env)
        : std::string("/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf");

    printf("[smoke44] GGUF: %s\n", gguf_path.c_str());
    fflush(stdout);

    if (!cp_cann_load_symbols()) {
        fprintf(stderr, "[smoke44] CANN symbol load failed\n");
        return 1;
    }

    // ---- Config: prod-shape max_img_seq/max_txt_seq so scratch sizing
    // matches the Q2.1 ≤18 GiB receipts. Forward uses the smoke small shape. ----
    ImageDiffusionConfig cfg;
    cfg.num_layers    = 60;
    cfg.num_heads     = 24;
    cfg.head_dim      = 128;
    cfg.hidden_size   = 3072;
    cfg.ff_mult       = 4;
    // Prod max_seq (Q2.1 projected peak at 17.74 GiB used these).
    cfg.max_img_seq   = 4096;
    cfg.max_txt_seq   = 256;
    cfg.precompute_rope = true;

    // Forward-pass activation shapes — match Phase 4.2/4.3 smoke (do NOT
    // exercise prod seq=4352 here; that's Phase 4.5 scope).
    const int64_t img_seq = 64;
    const int64_t txt_seq = 32;
    const int64_t H       = cfg.hidden_size;

    printf("[smoke44] config: layers=%d heads=%d head_dim=%d hidden=%d ff=%d\n",
           cfg.num_layers, cfg.num_heads, cfg.head_dim, cfg.hidden_size,
           cfg.hidden_size * cfg.ff_mult);
    printf("[smoke44] max_seq (for scratch): img=%d txt=%d  "
           "forward shape: img=%lld txt=%lld joint=%lld\n",
           cfg.max_img_seq, cfg.max_txt_seq,
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq));
    if (const char *q = std::getenv("GGML_CANN_QUANT_BF16")) {
        printf("[smoke44] GGML_CANN_QUANT_BF16=%s\n", q);
    } else {
        printf("[smoke44] GGML_CANN_QUANT_BF16 not set (engine uses default)\n");
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
        fprintf(stderr, "[smoke44] init_from_gguf FAILED after %.1f ms — RED\n",
                init_wall_ms);
        return 2;
    }
    if (!eng.is_ready()) {
        fprintf(stderr, "[smoke44] init_from_gguf returned true but "
                        "is_ready()=false — RED\n");
        return 2;
    }
    printf("[smoke44] init_from_gguf OK (wall %.1f ms, engine ready)\n",
           init_wall_ms);
    fflush(stdout);

    // ---- Receipts — same layout as Q2.1 smoke ----
    const DiTInitStats &st = eng.stats();
    const size_t total_bytes =
        st.q4_weight_bytes + st.q4_scale_bytes +
        st.f16_weight_bytes + st.f32_weight_bytes +
        st.rope_bytes + st.scratch_bytes;

    printf("\n========== Phase 4.4 real-GGUF init receipts ==========\n");
    printf("  tensors uploaded:    %lld (Q4-resident=%lld, F16-fallback=%lld)\n",
           (long long)st.tensors_uploaded,
           (long long)st.q4_tensors,
           (long long)st.f16_fallback_tensors);
    printf("  Q4 weight bytes:     %zu (%.2f GiB)\n",
           st.q4_weight_bytes,
           st.q4_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Q4 scale  bytes:     %zu (%.2f GiB)\n",
           st.q4_scale_bytes,
           st.q4_scale_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  F16 weight bytes:    %zu (%.2f GiB)  "
           "[biases + F16-fallback weights]\n",
           st.f16_weight_bytes,
           st.f16_weight_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  F32 weight bytes:    %zu (%.2f MiB)  [RMSNorm gammas]\n",
           st.f32_weight_bytes,
           st.f32_weight_bytes / (1024.0 * 1024.0));
    printf("  RoPE pe bytes:       %zu (%.2f MiB)\n",
           st.rope_bytes,
           st.rope_bytes / (1024.0 * 1024.0));
    printf("  Scratch bytes:       %zu (%.2f GiB)\n",
           st.scratch_bytes,
           st.scratch_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Peak init HBM:       %zu (%.2f GiB)  [Phase 4.4 gate: ≤ 18 GiB]\n",
           total_bytes,
           total_bytes / (1024.0 * 1024.0 * 1024.0));
    printf("  Dequant/repack wall: %.1f ms\n", st.dequant_wall_ms);
    printf("  Total init wall:     %.1f ms\n", st.load_wall_ms);
    fflush(stdout);

    const double peak_gib = total_bytes / (1024.0 * 1024.0 * 1024.0);
    const bool hbm_within_budget = peak_gib <= 18.0;
    if (!hbm_within_budget) {
        printf("[smoke44] WARN peak HBM %.2f GiB > 18 GiB Q1.10 gate — "
               "receipts diverge from Q2.1 projection\n", peak_gib);
    }

    // ---- Dummy activations at smoke shape ----
    std::vector<uint16_t> x_img_f16, x_txt_f16, t_emb_f16;
    fill_random_f16(x_img_f16, (size_t)img_seq * H, 0.1f, 0x4411ULL);
    fill_random_f16(x_txt_f16, (size_t)txt_seq * H, 0.1f, 0x4422ULL);
    fill_random_f16(t_emb_f16, (size_t)H,            0.1f, 0x4433ULL);

    void *x_img_dev = upload_f16(x_img_f16.data(), x_img_f16.size());
    void *x_txt_dev = upload_f16(x_txt_f16.data(), x_txt_f16.size());
    void *t_emb_dev = upload_f16(t_emb_f16.data(), t_emb_f16.size());
    if (!x_img_dev || !x_txt_dev || !t_emb_dev) {
        fprintf(stderr, "[smoke44] activation upload failed\n");
        return 1;
    }

    // Reuse the engine's pre-built RoPE pe table — init_from_gguf allocated
    // it at cfg_.precompute_rope=true. forward_all_blocks_test accepts the
    // pointer verbatim; the engine also owns `rope_cos_dev_for_test`/
    // `rope_sin_dev_for_test` flat tables which apply_rope_on_device_
    // consumes directly. Here we pass `rope_pe_dev_for_test()` so the path
    // mirrors Phase 4.2/4.3.
    void *pe_dev = eng.rope_pe_dev_for_test();
    if (!pe_dev) {
        fprintf(stderr, "[smoke44] engine rope_pe_dev not populated "
                        "— init_from_gguf did not build pe table\n");
        return 1;
    }

    // ---- Baseline stats on input x_img ----
    LatentStats s0 = compute_f16_stats(x_img_f16.data(), x_img_f16.size());
    printf("\n[smoke44] x_img_init: mean=%.4f std=%.4f min=%.4f max=%.4f "
           "nan=%lld inf=%lld\n",
           s0.mean, s0.std, s0.min_v, s0.max_v,
           (long long)s0.nan_count, (long long)s0.inf_count);
    fflush(stdout);

    // ---- Single forward: all 60 blocks ----
    std::vector<double> per_block_ms((size_t)cfg.num_layers, 0.0);
    printf("[smoke44] dispatching forward_all_blocks_test "
           "(n_blocks=%d, per_block_ms requested)...\n", cfg.num_layers);
    fflush(stdout);

    auto t_fwd0 = std::chrono::steady_clock::now();
    bool fwd_ok = eng.forward_all_blocks_test(x_img_dev, img_seq,
                                                x_txt_dev, txt_seq,
                                                t_emb_dev, pe_dev,
                                                per_block_ms.data(),
                                                cfg.num_layers);
    g_cann.aclrtSynchronizeStream(nullptr);
    auto t_fwd1 = std::chrono::steady_clock::now();
    double fwd_wall_ms =
        std::chrono::duration<double, std::milli>(t_fwd1 - t_fwd0).count();

    if (!fwd_ok) {
        fprintf(stderr, "[smoke44] forward_all_blocks_test returned false "
                         "(after %.2f ms) — RED\n", fwd_wall_ms);
        return 2;
    }
    printf("[smoke44] forward_all_blocks_test OK (%.2f ms total for 60 blocks)\n",
           fwd_wall_ms);
    fflush(stdout);

    // ---- D2H download + stats ----
    std::vector<uint16_t> x_img_out_f16(x_img_f16.size());
    aclError me = g_cann.aclrtMemcpy(x_img_out_f16.data(),
                                       x_img_out_f16.size() * sizeof(uint16_t),
                                       x_img_dev,
                                       x_img_out_f16.size() * sizeof(uint16_t),
                                       ACL_MEMCPY_DEVICE_TO_HOST);
    if (me != 0) {
        fprintf(stderr, "[smoke44] D2H memcpy err=%d\n", (int)me);
        return 1;
    }
    LatentStats s1 = compute_f16_stats(x_img_out_f16.data(),
                                         x_img_out_f16.size());

    // Per-block stats.
    double min_b = 1e30, max_b = -1e30, sum_b = 0.0;
    for (double ms : per_block_ms) {
        sum_b += ms;
        if (ms < min_b) min_b = ms;
        if (ms > max_b) max_b = ms;
    }
    std::vector<double> sorted_b = per_block_ms;
    std::sort(sorted_b.begin(), sorted_b.end());
    double median_b = sorted_b.empty()
        ? 0.0 : sorted_b[sorted_b.size() / 2];

    printf("\n========== Q2.4.4 Phase 4.4 real-GGUF forward smoke ==========\n");
    printf("gguf:   %s\n", gguf_path.c_str());
    printf("config: H=%lld heads=%lld head_dim=%lld ff_dim=%lld layers=%d\n",
           (long long)cfg.hidden_size, (long long)cfg.num_heads,
           (long long)cfg.head_dim,
           (long long)cfg.hidden_size * cfg.ff_mult,
           cfg.num_layers);
    printf("seq:    img=%lld  txt=%lld  joint=%lld  "
           "(max: img=%d txt=%d for scratch)\n",
           (long long)img_seq, (long long)txt_seq,
           (long long)(img_seq + txt_seq),
           cfg.max_img_seq, cfg.max_txt_seq);
    printf("hbm:    peak=%.2f GiB  (Q4 tensors=%lld, F16-fallback=%lld)\n",
           peak_gib,
           (long long)st.q4_tensors,
           (long long)st.f16_fallback_tensors);
    printf("wall:   init=%.1f ms  forward_60blk=%.2f ms   "
           "per-block min=%.2f ms  median=%.2f ms  max=%.2f ms  sum=%.2f ms\n",
           st.load_wall_ms, fwd_wall_ms, min_b, median_b, max_b, sum_b);
    printf("per-block ms (first 5 / last 5):  ");
    for (int i = 0; i < std::min(5, cfg.num_layers); ++i)
        printf("%.2f ", per_block_ms[i]);
    printf("... ");
    for (int i = std::max(0, cfg.num_layers - 5); i < cfg.num_layers; ++i)
        printf("%.2f ", per_block_ms[i]);
    printf("\n");

    printf("\n-- output img_hidden vs input --\n");
    printf("  x_img_init : mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s0.mean, s0.std, s0.min_v, s0.max_v);
    printf("  x_img_out  : mean=%.4f std=%.4f min/max=%.4f/%.4f\n",
           s1.mean, s1.std, s1.min_v, s1.max_v);
    printf("  NaN=%lld  inf=%lld\n",
           (long long)s1.nan_count, (long long)s1.inf_count);

    // Gate: no NaN/inf AND output has non-trivial variance.
    const double STD_GATE = 0.001;
    bool no_nan_inf = (s1.nan_count == 0) && (s1.inf_count == 0);
    bool non_trivial = (s1.std > STD_GATE);
    bool pass = no_nan_inf && non_trivial;

    const char *verdict = pass ? "GREEN"
                                : (!no_nan_inf ? "RED (NaN/inf)"
                                               : "YELLOW (std < gate)");
    printf("\n---------------------------------------------------\n");
    printf("VERDICT: %s  (gate: NaN=0, inf=0, std > %.4f over output latent; "
           "HBM peak %.2f GiB %s 18 GiB)\n",
           verdict, STD_GATE, peak_gib,
           hbm_within_budget ? "≤" : ">");
    // GREEN → 0, YELLOW (no nan/inf but low variance) → 3, RED → 2.
    if (pass) return 0;
    if (no_nan_inf) return 3;
    return 2;
}
