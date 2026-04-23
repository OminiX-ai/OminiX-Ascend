// ============================================================================
// Q2.4.1 (Phase 4.1) RoPE smoke — on-device vs host round-trip parity + wall.
//
// Harness:
//   - Boots ImageDiffusionEngine via init_for_smoke() (no GGUF load). That
//     allocates cos/sin + pe tables plus the scratch_rope_{a,b,c} buffers.
//   - Synthesizes a deterministic random [B=1, seq, NH, HD] F16 Q-like tensor.
//   - Copies it to two device buffers (`x_ref`, `x_dev`) so we can compare
//     the two code paths on bit-identical input.
//   - Dispatches apply_rope_host_test on `x_ref` (Phase 3 ground truth).
//   - Dispatches apply_rope_on_device_test on `x_dev` N times (perf measure).
//   - Reports cos_sim(x_dev, x_ref), MAE, NaN count.
//
// Phase 4.1 gate:
//   cos_sim > 0.99 vs Phase 3 host-side path on single-block RoPE
//   wall-clock per call drops substantially (expected 10-50×).
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
// F16 <-> F32 helpers (arm64 __fp16).
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

static void *upload_f16(const uint16_t *host, size_t n) {
    void *dev = nullptr;
    if (g_cann.aclrtMalloc(&dev, n * sizeof(uint16_t),
                            ACL_MEM_MALLOC_HUGE_FIRST) != 0) return nullptr;
    if (g_cann.aclrtMemcpy(dev, n * sizeof(uint16_t), host,
                            n * sizeof(uint16_t),
                            ACL_MEMCPY_HOST_TO_DEVICE) != 0) {
        g_cann.aclrtFree(dev);
        return nullptr;
    }
    return dev;
}

static void download_f16(void *dev, std::vector<uint16_t> &out, size_t n) {
    out.resize(n);
    g_cann.aclrtMemcpy(out.data(), n * sizeof(uint16_t), dev,
                        n * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST);
}

struct DiffStats {
    double cos_sim;
    double mae;
    float  min_diff;
    float  max_diff;
    int64_t nan_count;
};

static DiffStats compare(const std::vector<uint16_t> &a,
                          const std::vector<uint16_t> &b) {
    DiffStats d{};
    double dot = 0, norma = 0, normb = 0, abs_sum = 0;
    d.min_diff =  1e30f;
    d.max_diff = -1e30f;
    for (size_t i = 0; i < a.size(); ++i) {
        float x = f16_to_f32(a[i]);
        float y = f16_to_f32(b[i]);
        if (std::isnan(x) || std::isnan(y) || std::isinf(x) || std::isinf(y)) {
            d.nan_count++;
            continue;
        }
        dot   += (double)x * (double)y;
        norma += (double)x * (double)x;
        normb += (double)y * (double)y;
        float diff = x - y;
        abs_sum += std::fabs((double)diff);
        if (diff < d.min_diff) d.min_diff = diff;
        if (diff > d.max_diff) d.max_diff = diff;
    }
    d.cos_sim = (norma > 0 && normb > 0)
                 ? dot / (std::sqrt(norma) * std::sqrt(normb)) : 0.0;
    d.mae = a.empty() ? 0.0 : abs_sum / (double)a.size();
    return d;
}

int main() {
    // Config — match Qwen-Image-Edit dims but shrink the seq for a tractable
    // smoke. We pick two seq points:
    //   (a) seq=320 (txt 64 + img 256) — matches Phase 3 §3.2
    //   (b) seq=4352 (txt 256 + img 4096) — production worst-case per
    //       docs/qie_q0_5_fiav2_seq4352.md
    const char *seq_env = std::getenv("QIE_ROPE_SMOKE_SEQ");
    const bool big = seq_env && std::strcmp(seq_env, "big") == 0;

    ImageDiffusionConfig cfg;
    cfg.num_layers      = 1;
    cfg.num_heads       = 24;
    cfg.head_dim        = 128;
    cfg.hidden_size     = 3072;
    cfg.ff_mult         = 4;
    cfg.max_img_seq     = big ? 4096 : 256;
    cfg.max_txt_seq     = big ? 256  : 64;
    cfg.precompute_rope = true;

    ImageDiffusionEngine engine;
    if (!engine.init_for_smoke(cfg, 0)) {
        std::fprintf(stderr, "init_for_smoke failed\n");
        return 1;
    }

    // Optional micro-test: override cos/sin device buffers with test patterns
    // to verify gather/scatter correctness independent of angle computation.
    if (const char *pat = std::getenv("QIE_ROPE_TEST_PATTERN")) {
        const int64_t total_pos = cfg.max_img_seq + cfg.max_txt_seq;
        const int64_t half = cfg.head_dim / 2;
        std::vector<uint16_t> cos_host((size_t)total_pos * half);
        std::vector<uint16_t> sin_host((size_t)total_pos * half);
        std::printf("[diag] override cos/sin with pattern=%s\n", pat);
        if (std::strcmp(pat, "identity") == 0) {
            for (auto &v : cos_host) v = f32_to_f16(1.0f);
            for (auto &v : sin_host) v = f32_to_f16(0.0f);
        } else if (std::strcmp(pat, "scale2") == 0) {
            for (auto &v : cos_host) v = f32_to_f16(2.0f);
            for (auto &v : sin_host) v = f32_to_f16(0.0f);
        } else if (std::strcmp(pat, "swap") == 0) {
            // cos=0, sin=1: y_even = x_odd, y_odd = -x_even
            for (auto &v : cos_host) v = f32_to_f16(0.0f);
            for (auto &v : sin_host) v = f32_to_f16(1.0f);
        } else if (std::strcmp(pat, "dp_index") == 0) {
            // cos[pos, dp] = dp; sin[pos, dp] = 0. dev output at d=2*dp is
            // x_even[dp] * dp. Lets us see whether dp indexing is right.
            for (int64_t p = 0; p < total_pos; ++p)
                for (int64_t dp = 0; dp < half; ++dp) {
                    cos_host[p*half + dp] = f32_to_f16((float)dp);
                    sin_host[p*half + dp] = f32_to_f16(0.0f);
                }
        }
        g_cann.aclrtMemcpy(engine.rope_cos_dev_for_test(),
                            cos_host.size() * 2, cos_host.data(),
                            cos_host.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
        g_cann.aclrtMemcpy(engine.rope_sin_dev_for_test(),
                            sin_host.size() * 2, sin_host.data(),
                            sin_host.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
        // Also overwrite the pre-broadcast NH-tile used by the on-device path.
        const int64_t NH = cfg.num_heads;
        std::vector<uint16_t> cos_bc((size_t)total_pos * NH * half);
        std::vector<uint16_t> sin_bc((size_t)total_pos * NH * half);
        for (int64_t p = 0; p < total_pos; ++p)
            for (int64_t h = 0; h < NH; ++h) {
                std::memcpy(&cos_bc[((size_t)p*NH + h) * half],
                             &cos_host[(size_t)p * half], half * 2);
                std::memcpy(&sin_bc[((size_t)p*NH + h) * half],
                             &sin_host[(size_t)p * half], half * 2);
            }
        g_cann.aclrtMemcpy(engine.rope_cos_bcast_dev_for_test(),
                            cos_bc.size() * 2, cos_bc.data(),
                            cos_bc.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
        g_cann.aclrtMemcpy(engine.rope_sin_bcast_dev_for_test(),
                            sin_bc.size() * 2, sin_bc.data(),
                            sin_bc.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
        // Also rebuild pe packed layout for the host path to match.
        std::vector<uint16_t> pe_host((size_t)total_pos * half * 2 * 2);
        for (int64_t p = 0; p < total_pos; ++p)
            for (int64_t dp = 0; dp < half; ++dp) {
                float cv = f16_to_f32(cos_host[p*half + dp]);
                float sv = f16_to_f32(sin_host[p*half + dp]);
                size_t base = ((size_t)p * half + (size_t)dp) * 4;
                pe_host[base + 0] = f32_to_f16(cv);
                pe_host[base + 1] = f32_to_f16(-sv);
                pe_host[base + 2] = f32_to_f16(sv);
                pe_host[base + 3] = f32_to_f16(cv);
            }
        g_cann.aclrtMemcpy(engine.rope_pe_dev_for_test(),
                            pe_host.size() * 2, pe_host.data(),
                            pe_host.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE);
    }

    const int64_t B = 1;
    const int64_t NH = cfg.num_heads;
    const int64_t HD = cfg.head_dim;
    const int64_t txt_seq = cfg.max_txt_seq;
    const int64_t img_seq = cfg.max_img_seq;

    // Test both streams like the engine does: txt first (pe_row_offset=0),
    // then img (pe_row_offset=max_txt_seq).
    const int64_t max_n = (size_t)B * std::max(txt_seq, img_seq) * NH * HD;
    std::vector<uint16_t> x_host_txt(max_n);
    std::vector<uint16_t> x_host_img(max_n);
    fill_random_f16(x_host_txt, (size_t)B * txt_seq * NH * HD, 1.0f, 0xC0DE1);
    fill_random_f16(x_host_img, (size_t)B * img_seq * NH * HD, 1.0f, 0xC0DE2);

    auto run_stream = [&](const char *tag, int64_t seq, int64_t pe_off,
                            const std::vector<uint16_t> &x_init) -> int {
        const size_t n_elt = (size_t)B * seq * NH * HD;
        void *x_ref = upload_f16(x_init.data(), n_elt);
        void *x_dev = upload_f16(x_init.data(), n_elt);
        if (!x_ref || !x_dev) { std::fprintf(stderr, "upload failed\n"); return 1; }

        // --- HOST REFERENCE ---------------------------------------------------
        auto t_h0 = std::chrono::steady_clock::now();
        if (!engine.apply_rope_host_test(x_ref, engine.rope_pe_dev_for_test(),
                                            pe_off, B, seq, NH, HD)) {
            std::fprintf(stderr, "apply_rope_host_test failed\n"); return 2;
        }
        g_cann.aclrtSynchronizeStream(nullptr);  // host path is sync anyway
        auto t_h1 = std::chrono::steady_clock::now();
        double host_ms =
            std::chrono::duration<double, std::milli>(t_h1 - t_h0).count();

        // --- ON-DEVICE (warmup + N iters) -------------------------------------
        // One warmup to pay the aclnn JIT cost outside the timer.
        if (!engine.apply_rope_on_device_test(x_dev, pe_off, B, seq, NH, HD)) {
            std::fprintf(stderr, "apply_rope_on_device_test (warmup) failed\n"); return 3;
        }
        g_cann.aclrtSynchronizeStream(nullptr);

        // Reset x_dev to the original input for the timed run.
        g_cann.aclrtMemcpy(x_dev, n_elt * sizeof(uint16_t),
                            x_init.data(), n_elt * sizeof(uint16_t),
                            ACL_MEMCPY_HOST_TO_DEVICE);

        const int N = 20;
        auto t_d0 = std::chrono::steady_clock::now();
        for (int i = 0; i < N; ++i) {
            // Re-populate x_dev each iter so repeated rotations don't diverge.
            // Actually — that H2D would pollute the timing. Instead: accept
            // that the input gets rotated N times; we only care about
            // dispatch wall, not numerical result here. We compare
            // numerical result from a separate single-call path below.
            if (!engine.apply_rope_on_device_test(x_dev, pe_off, B, seq, NH, HD)) {
                std::fprintf(stderr, "apply_rope_on_device_test iter=%d failed\n", i);
                return 3;
            }
        }
        g_cann.aclrtSynchronizeStream(nullptr);
        auto t_d1 = std::chrono::steady_clock::now();
        double dev_ms =
            std::chrono::duration<double, std::milli>(t_d1 - t_d0).count() /
            (double)N;

        // --- PARITY RUN -------------------------------------------------------
        // Re-upload x_init and run apply_rope_on_device_ exactly once, then
        // compare to x_ref (which got apply_rope_host_ once).
        g_cann.aclrtMemcpy(x_dev, n_elt * sizeof(uint16_t),
                            x_init.data(), n_elt * sizeof(uint16_t),
                            ACL_MEMCPY_HOST_TO_DEVICE);
        if (!engine.apply_rope_on_device_test(x_dev, pe_off, B, seq, NH, HD)) {
            std::fprintf(stderr, "apply_rope_on_device_test (parity) failed\n");
            return 3;
        }
        g_cann.aclrtSynchronizeStream(nullptr);

        std::vector<uint16_t> host_out, dev_out;
        download_f16(x_ref, host_out, n_elt);
        download_f16(x_dev, dev_out,  n_elt);
        DiffStats d = compare(dev_out, host_out);

        std::printf("[%-12s] seq=%lld  pe_off=%lld  n_elt=%zu\n",
                     tag, (long long)seq, (long long)pe_off, n_elt);
        std::printf("    wall host=%7.2f ms  dev=%7.3f ms (avg/call over %d)  "
                     "speedup=%.1f×\n",
                     host_ms, dev_ms, N, host_ms / std::max(dev_ms, 1e-6));
        std::printf("    cos_sim=%.6f  mae=%.6f  min/max=%+.4f/%+.4f  NaN=%lld\n",
                     d.cos_sim, d.mae, d.min_diff, d.max_diff,
                     (long long)d.nan_count);

        // Element-wise diagnostic for the first row (b=0, s=0, h=0).
        if (std::getenv("QIE_ROPE_DEBUG")) {
            const int64_t HD = engine.config().head_dim;
            std::printf("    dbg first %d (head 0 of s=0):  in      host    dev   diff\n",
                         (int)HD);
            for (int i = 0; i < (int)HD; ++i) {
                float vi = f16_to_f32(x_init[i]);
                float vh = f16_to_f32(host_out[i]);
                float vd = f16_to_f32(dev_out[i]);
                const bool ok = std::fabs(vh - vd) < 1e-3f;
                std::printf("      d=%3d  %+7.4f  %+7.4f  %+7.4f  %+7.4f %s\n",
                             i, vi, vh, vd, vh - vd, ok ? "" : " <<DIFF");
            }
        }

        g_cann.aclrtFree(x_ref);
        g_cann.aclrtFree(x_dev);

        return (d.cos_sim > 0.99 && d.nan_count == 0) ? 0 : 10;
    };

    std::printf("\n========== Q2.4.1 Phase 4.1 RoPE smoke ==========\n");
    std::printf("config: H=%d heads=%d head_dim=%d max_img_seq=%d max_txt_seq=%d\n",
                 cfg.hidden_size, cfg.num_heads, cfg.head_dim,
                 cfg.max_img_seq, cfg.max_txt_seq);

    int rc = 0;
    rc |= run_stream("txt stream", txt_seq, 0, x_host_txt);
    rc |= run_stream("img stream", img_seq, txt_seq, x_host_img);

    std::printf("\n---------------------------------------------------\n");
    std::printf("VERDICT: %s (gate: cos_sim > 0.99 both streams, NaN=0)\n",
                 rc == 0 ? "GREEN" : "RED");
    return rc;
}
