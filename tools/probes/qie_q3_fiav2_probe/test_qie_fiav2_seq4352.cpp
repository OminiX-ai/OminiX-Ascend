// Q3 FIAv2 runtime probe at QIE joint-attention sequence lengths.
//
// Agent: QIE-Q3-FIAV2 (2026-04-22)
// Contract: runtime measurement of aclnnFusedInferAttentionScoreV2 at the
// QIE image-attention regime — MHA 24/24, head_dim=128, F16 BSND, seq ∈
// {2048, 4096, 4352, 8192} — vs the estimated ggml_ext_attention_ext
// wrapper overhead (ggml builds the same FIAv2 call but adds cast +
// permute-cont ops around it).
//
// Key finding from pre-build audit: ggml_cann_flash_attn_ext IS FIAv2.
// (ggml/src/ggml-cann/aclnn_ops.cpp:4449 calls FusedInferAttentionScoreV2
// directly.) So the meaningful probe is:
//   Path B: direct FIAv2 dispatch (this harness) — measures kernel floor.
//   Path A: ggml's FIAv2 + wrapper ops — measured separately via the
//           wrapper op costs (cast_f32_to_f16, permute+cont, etc.) and
//           added to Path B.
//
// Build on ac02:
//   source /usr/local/Ascend/ascend-toolkit/latest/../set_env.sh
//   g++ -std=c++17 -O2 -o test_qie_fiav2_seq4352 test_qie_fiav2_seq4352.cpp \
//       -I$ASCEND_TOOLKIT_HOME/aarch64-linux/include \
//       -L$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64 \
//       -lascendcl -lopapi -lnnopbase -ldl
//
// Run:
//   export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
//   export LD_LIBRARY_PATH=$ASCEND_TOOLKIT_HOME/aarch64-linux/lib64:$LD_LIBRARY_PATH
//   ./test_qie_fiav2_seq4352

#include <acl/acl.h>
#include <aclnnop/aclnn_fused_infer_attention_score_v2.h>
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

#define ACL_CHECK(expr) do {                                                     \
    aclError __err = (expr);                                                     \
    if (__err != ACL_SUCCESS) {                                                  \
        fprintf(stderr, "ACL error %d at %s:%d: %s\n",                           \
                (int)__err, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

#define ACL_CHECK_NN(expr) do {                                                  \
    aclnnStatus __st = (expr);                                                   \
    if (__st != 0) {                                                             \
        fprintf(stderr, "aclnn error %d at %s:%d: %s\n",                         \
                (int)__st, __FILE__, __LINE__, #expr);                           \
        std::abort();                                                            \
    }                                                                            \
} while (0)

// Convert a float to F16 as a uint16_t (IEEE 754 half, no subnormal flush).
static inline uint16_t f32_to_f16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint16_t res = (uint16_t)(mant >> (14 - exp));
        return (uint16_t)(sign | res);
    } else if (exp >= 31) {
        return (uint16_t)(sign | 0x7c00u | (mant ? 0x200u : 0u));
    }
    return (uint16_t)(sign | (exp << 10) | (mant >> 13));
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp  = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x3ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) { out = sign; }
        else {
            exp = 1;
            while ((mant & 0x400u) == 0) { mant <<= 1; exp--; }
            mant &= 0x3ffu;
            out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &out, sizeof(f));
    return f;
}

struct ProbeConfig {
    int64_t batch;
    int64_t seq;
    int64_t n_heads;
    int64_t n_kv_heads;
    int64_t head_dim;
    const char* tag;
};

struct WallStats {
    double median_us;
    double p10_us;
    double p90_us;
    double mean_us;
};

static WallStats summarize(std::vector<double>& times_us) {
    std::sort(times_us.begin(), times_us.end());
    size_t n = times_us.size();
    WallStats s{};
    s.median_us = times_us[n / 2];
    s.p10_us    = times_us[n / 10];
    s.p90_us    = times_us[(n * 9) / 10];
    double sum  = 0.0;
    for (double v : times_us) sum += v;
    s.mean_us = sum / (double)n;
    return s;
}

// Allocate a device F16 tensor of given logical BSND shape, fill with deterministic
// small-magnitude bf16-range values cast to f16 (so results won't NaN at seq=4352).
static void* alloc_bsnd_f16_device(int64_t B, int64_t S, int64_t N, int64_t D,
                                    uint64_t seed) {
    size_t count = (size_t)B * (size_t)S * (size_t)N * (size_t)D;
    size_t bytes = count * sizeof(uint16_t);
    std::vector<uint16_t> host(count);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    for (size_t i = 0; i < count; ++i) {
        host[i] = f32_to_f16(dist(rng));
    }
    void* dev = nullptr;
    ACL_CHECK(aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(dev, bytes, host.data(), bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    return dev;
}

static aclTensor* make_bsnd_tensor(void* data, int64_t B, int64_t S,
                                    int64_t N, int64_t D) {
    int64_t shape[4]   = {B, S, N, D};
    int64_t strides[4] = {S * N * D, N * D, D, 1};
    return aclCreateTensor(shape, 4, ACL_FLOAT16,
                           strides, /*offset*/ 0, ACL_FORMAT_ND,
                           shape, 4, data);
}

static void run_one_config(const ProbeConfig& cfg, int warmup, int iters,
                            aclrtStream stream) {
    printf("\n=== cfg %s  B=%ld S=%ld N=%ld(Nkv=%ld) D=%ld ===\n",
           cfg.tag, (long)cfg.batch, (long)cfg.seq, (long)cfg.n_heads,
           (long)cfg.n_kv_heads, (long)cfg.head_dim);

    // Allocate Q, K, V, Out (all F16 BSND).
    void* d_q   = alloc_bsnd_f16_device(cfg.batch, cfg.seq, cfg.n_heads,   cfg.head_dim, 0xA1A1);
    void* d_k   = alloc_bsnd_f16_device(cfg.batch, cfg.seq, cfg.n_kv_heads, cfg.head_dim, 0xB2B2);
    void* d_v   = alloc_bsnd_f16_device(cfg.batch, cfg.seq, cfg.n_kv_heads, cfg.head_dim, 0xC3C3);

    size_t out_count = (size_t)cfg.batch * (size_t)cfg.seq *
                       (size_t)cfg.n_heads * (size_t)cfg.head_dim;
    size_t out_bytes = out_count * sizeof(uint16_t);
    void* d_out = nullptr;
    ACL_CHECK(aclrtMalloc(&d_out, out_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_out, out_bytes, 0, out_bytes));

    // Persistent workspace, grown on demand.
    void*    d_ws    = nullptr;
    uint64_t ws_size = 0;

    char layout[5] = {'B','S','N','D',0};
    double scale = 1.0 / std::sqrt((double)cfg.head_dim);

    auto dispatch_once = [&]() {
        aclTensor*     t_q   = make_bsnd_tensor(d_q,   cfg.batch, cfg.seq, cfg.n_heads,    cfg.head_dim);
        aclTensor*     t_k   = make_bsnd_tensor(d_k,   cfg.batch, cfg.seq, cfg.n_kv_heads, cfg.head_dim);
        aclTensor*     t_v   = make_bsnd_tensor(d_v,   cfg.batch, cfg.seq, cfg.n_kv_heads, cfg.head_dim);
        aclTensor*     t_out = make_bsnd_tensor(d_out, cfg.batch, cfg.seq, cfg.n_heads,    cfg.head_dim);
        aclTensorList* tl_k  = aclCreateTensorList(&t_k, 1);
        aclTensorList* tl_v  = aclCreateTensorList(&t_v, 1);

        uint64_t       ws_need = 0;
        aclOpExecutor* exec    = nullptr;
        ACL_CHECK_NN(aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
            t_q, tl_k, tl_v,
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
            /*numHeads*/        cfg.n_heads,
            /*scaleValue*/      scale,
            /*preTokens*/       (int64_t)65535,
            /*nextTokens*/      (int64_t)65535,
            /*inputLayout*/     layout,
            /*numKeyValueHeads*/cfg.n_kv_heads,
            /*sparseMode*/      (int64_t)0,
            /*innerPrecise*/    (int64_t)2,  // S>1 → high-throughput mode
            /*blockSize*/       (int64_t)0,
            /*antiquantMode*/   (int64_t)0,
            /*softmaxLseFlag*/  false,
            /*keyAntiquantMode*/(int64_t)0,
            /*valueAntiquantMode*/(int64_t)0,
            /*attentionOut*/    t_out,
            /*softmaxLse*/      nullptr,
            &ws_need, &exec));

        if (ws_need > ws_size) {
            if (d_ws) aclrtFree(d_ws);
            ACL_CHECK(aclrtMalloc(&d_ws, ws_need, ACL_MEM_MALLOC_HUGE_FIRST));
            ws_size = ws_need;
        }
        void* ws_ptr = (ws_need > 0) ? d_ws : nullptr;
        ACL_CHECK_NN(aclnnFusedInferAttentionScoreV2(ws_ptr, ws_need, exec, stream));

        aclDestroyTensorList(tl_k);
        aclDestroyTensorList(tl_v);
        aclDestroyTensor(t_q);
        aclDestroyTensor(t_out);
    };

    // Warmup
    for (int i = 0; i < warmup; ++i) {
        dispatch_once();
    }
    ACL_CHECK(aclrtSynchronizeStream(stream));

    // Timed iters.
    std::vector<double> wall_us;
    wall_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        dispatch_once();
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
        wall_us.push_back(us);
    }

    WallStats s = summarize(wall_us);
    printf("  wall p10/median/p90/mean μs: %8.1f / %8.1f / %8.1f / %8.1f  (n=%d)\n",
           s.p10_us, s.median_us, s.p90_us, s.mean_us, iters);
    printf("  workspace bytes: %llu  (%.2f MiB)\n",
           (unsigned long long)ws_size, (double)ws_size / (1024.0 * 1024.0));

    // Sanity: check output is not all-zero and not NaN/inf.
    std::vector<uint16_t> out_host(out_count);
    ACL_CHECK(aclrtMemcpy(out_host.data(), out_bytes, d_out, out_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    int nan_count = 0, inf_count = 0, zero_count = 0;
    float omin = INFINITY, omax = -INFINITY, osum = 0.0f;
    for (size_t i = 0; i < out_count; ++i) {
        float v = f16_to_f32(out_host[i]);
        if (std::isnan(v)) nan_count++;
        else if (std::isinf(v)) inf_count++;
        else if (v == 0.0f) zero_count++;
        if (v < omin) omin = v;
        if (v > omax) omax = v;
        osum += v;
    }
    printf("  out stats: min=%.4f  max=%.4f  mean=%.4e  nan=%d inf=%d zeros=%d/%zu\n",
           omin, omax, osum / (float)out_count, nan_count, inf_count,
           zero_count, out_count);

    if (d_ws)  aclrtFree(d_ws);
    aclrtFree(d_q);
    aclrtFree(d_k);
    aclrtFree(d_v);
    aclrtFree(d_out);
}

int main() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));

    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    const int WARMUP = 5;
    const int ITERS  = 50;

    // QIE attention: MHA 24/24, head_dim=128.
    // Scan the joint-attention seq lengths from QIE learnings:
    //   256×256 latent = (32×32) = 1024 img tokens  → seq ≈ 1280  (Q1 baseline shape; currently GREEN)
    //   384×384 latent = (48×48) = 2304            → seq ≈ 2560
    //   512×512 latent = (64×64) = 4096            → seq ≈ 4352  (Q0.5.2 target)
    //   edit-mode 512×512 (2× img concat)          → seq ≈ 8448  (stretch)
    //   pure-img with no txt                       → seq = 4096
    std::vector<ProbeConfig> configs = {
        { 1,  512, 24, 24, 128, "seq=512  (128×128 img, txt none)"   },
        { 1, 1280, 24, 24, 128, "seq=1280 (256×256 img, 256 txt)"    },
        { 1, 2048, 24, 24, 128, "seq=2048 (Q1-NaN boundary, 384 lat)" },
        { 1, 2560, 24, 24, 128, "seq=2560 (384×384 img, 256 txt)"    },
        { 1, 4096, 24, 24, 128, "seq=4096 (pure img, no txt)"         },
        { 1, 4352, 24, 24, 128, "seq=4352 (QIE TARGET: 512×512+txt)"  },
        { 2, 4352, 24, 24, 128, "seq=4352, B=2 (CFG on)"              },
        { 1, 8192, 24, 24, 128, "seq=8192 (edit-mode stretch)"        },
    };

    for (const auto& cfg : configs) {
        run_one_config(cfg, WARMUP, ITERS, stream);
    }

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    printf("\nDONE\n");
    return 0;
}
