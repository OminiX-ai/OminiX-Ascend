// QIE-Q2 Q4-resident Gate 0 probe — aclnnWeightQuantBatchMatmulV3 W4 capability.
//
// Agent: QIE-Q2-Q4RESIDENT (2026-04-22)
// Contract: Q1.9 amendment `32fa76f3` (Q4-RESIDENT supersedes preload-dequant).
//
// Goal: answer one question before rewriting Phase 2 —
//   Does aclnnWeightQuantBatchMatmulV3 accept an INT4 weight tensor with a
//   per-group (groupSize=32) F16 scale, and does it produce numerically
//   correct output at the QIE DiT matmul regime?
//
// Shape (per contract §Q1.9 probe spec):
//   x            = [M=128, K=3072] F16
//   weight (W4)  = [K=3072, N=3072] INT4, symmetric, per-group G=32 along K
//                  → scale shape [K/G=96, N=3072] F16
//   y            = [M=128, N=3072] F16
//
// Note: this is NOT exact Q4_0 block layout. Q4_0 on CPU uses [N rows × K/32
// blocks], each block = 32 × INT4 + one F16 scale per block (no offset). The
// WQBMMv3 API, however, takes:
//     weight:         [K, N] INT4 (or INT8) — contiguous across K
//     antiquantScale: [K/G, N] or [N] F16   — per-group or per-channel
// i.e. the repack direction differs. What we are probing here is:
//
//   1) does the op accept INT4 data-type at our shapes?
//   2) does antiquantGroupSize=32 yield correct per-group dequant?
//   3) is the wall-time reasonable vs F16 aclnnMm baseline?
//
// If GREEN, Phase 1 Q4-resident load path just has to re-tile Q4_0 blocks
// from the GGUF [N, K/32] layout into the [K, N]-packed form WQBMMv3 expects
// (still roughly ~5 GiB resident, no dequant), plus emit the scale tensor.
//
// Build on ac03:
//   bash build_and_run.sh
//
// Exit code 0 = GREEN (op works, cos_sim > 0.99, perf < 1.5× F16 baseline).
// Exit code 1 = YELLOW (op works but numerics or perf off).
// Exit code 2 = RED    (op rejects W4 config or output grossly wrong).
//
// Any non-GREEN verdict MUST be reported to PM before rewriting Phase 2.

#include <acl/acl.h>
#include <aclnnop/aclnn_weight_quant_batch_matmul_v3.h>
#include <aclnnop/aclnn_matmul.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define ACL_CHECK(expr) do {                                                     \
    aclError __err = (expr);                                                     \
    if (__err != ACL_SUCCESS) {                                                  \
        fprintf(stderr, "ACL error %d at %s:%d: %s\n",                           \
                (int)__err, __FILE__, __LINE__, #expr);                          \
        std::abort();                                                            \
    }                                                                            \
} while (0)

// WQBMMv3 probe success is tested on status code. A non-zero status from
// GetWorkspaceSize is the PRIMARY capability signal — op may refuse INT4 at
// this CANN build and return e.g. ACLNN_ERR_PARAM_INVALID. We do NOT abort;
// we record the verdict.
#define ACL_CHECK_NN(expr) do {                                                  \
    aclnnStatus __st = (expr);                                                   \
    if (__st != 0) {                                                             \
        fprintf(stderr, "aclnn error %d at %s:%d: %s\n",                         \
                (int)__st, __FILE__, __LINE__, #expr);                           \
        std::abort();                                                            \
    }                                                                            \
} while (0)

// ---------- F16 <-> F32 (IEEE 754 half, no subnormal flush) ----------
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

// ---------- Shape constants (per contract §Q1.9 probe spec) ----------
static constexpr int64_t M = 128;
static constexpr int64_t K = 3072;
static constexpr int64_t N = 3072;
static constexpr int64_t G = 32;       // antiquantGroupSize (Q4_0 block)
static constexpr int64_t K_G = K / G;  // 96 scale rows

// ---------- CPU Q4_0-style quantize ----------
// For each column n in [0,N), for each group g in [0,K/G):
//   find max|w[g*G..g*G+G, n]|, scale = max/7.0f (so range is [-7..+7]×scale,
//     which is what Q4_0 uses — symmetric 4-bit signed with no offset).
//   quantized int4: q[i] = clamp(round(w[i]/scale), -8, 7)
//   store as unsigned 4-bit (q + 8) → [0, 15]; pack two nibbles per byte.
//
// We emit TWO buffers:
//   w_q4_packed:  K * N / 2 bytes, nibble layout = [K, N] column-major,
//                 i.e. for weight[k, n] the byte index is (n*K + k)/2 and
//                 the nibble within that byte is (n*K + k) & 1. (That is
//                 "transposeB=true" friendly — K is the contiguous inner.)
//   w_scales_f16: K_G * N uint16_t, row-major [K_G, N].
//
// The CPU reference dequants this back to F16 so we can do a CPU F32 matmul
// and compare bit-for-bit against what WQBMMv3 should produce.

struct Q4Pack {
    std::vector<uint8_t>  packed;   // K*N/2 bytes, column-major nibbles
    std::vector<uint16_t> scales;   // K_G * N F16
    std::vector<uint16_t> dequant_f16;  // K * N F16 (CPU reference)
};

static Q4Pack cpu_quantize_q4_symmetric_per_group(const std::vector<float>& w_dense,
                                                   int64_t k, int64_t n, int64_t g) {
    Q4Pack out;
    out.packed.assign((size_t)k * n / 2, 0);
    out.scales.assign((size_t)(k / g) * n, 0);
    out.dequant_f16.assign((size_t)k * n, 0);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t grp = 0; grp < k / g; ++grp) {
            // 1) find max |w| in this group
            float absmax = 0.0f;
            for (int64_t i = 0; i < g; ++i) {
                float v = std::fabs(w_dense[(grp * g + i) * n + col]);
                if (v > absmax) absmax = v;
            }
            float scale = absmax / 7.0f;
            if (scale == 0.0f) scale = 1e-7f;
            out.scales[(size_t)grp * n + col] = f32_to_f16(scale);

            // 2) quantize + pack
            for (int64_t i = 0; i < g; ++i) {
                int64_t k_idx = grp * g + i;
                float v = w_dense[k_idx * n + col];
                int q = (int)std::lrintf(v / scale);
                if (q < -8) q = -8;
                if (q > 7)  q = 7;
                // WQBMMv3 with symmetric quant (antiquantOffset=nullptr)
                // interprets nibble as signed two's-complement 4-bit.
                // -8 → 0b1000 (0x8), -1 → 0b1111 (0xf), 0 → 0x0, 7 → 0x7.
                uint8_t nibble = (uint8_t)(q & 0x0f);

                // Dequant reference: use the same scale + the clamped q.
                out.dequant_f16[k_idx * n + col] = f32_to_f16(q * scale);

                // Pack column-major: linear_index = col*K + k_idx
                size_t lin = (size_t)col * k + k_idx;
                size_t byte_i = lin / 2;
                if ((lin & 1) == 0) {
                    out.packed[byte_i] = (out.packed[byte_i] & 0xf0) | nibble;
                } else {
                    out.packed[byte_i] = (out.packed[byte_i] & 0x0f) |
                                         (uint8_t)(nibble << 4);
                }
            }
        }
    }
    return out;
}

// ---------- CPU F32 matmul: y[M,N] = x[M,K] @ w[K,N] ----------
static void cpu_matmul_f16(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                           std::vector<float>& y) {
    y.assign((size_t)M * N, 0.0f);
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t k = 0; k < K; ++k) {
            float xv = f16_to_f32(x[(size_t)i * K + k]);
            for (int64_t j = 0; j < N; ++j) {
                float wv = f16_to_f32(w[(size_t)k * N + j]);
                y[(size_t)i * N + j] += xv * wv;
            }
        }
    }
}

// ---------- Cosine similarity on flattened vectors ----------
static double cosine_sim(const std::vector<float>& a, const std::vector<float>& b) {
    double dot = 0.0, na = 0.0, nb = 0.0;
    size_t n = a.size();
    for (size_t i = 0; i < n; ++i) {
        dot += (double)a[i] * (double)b[i];
        na  += (double)a[i] * (double)a[i];
        nb  += (double)b[i] * (double)b[i];
    }
    if (na == 0.0 || nb == 0.0) return 0.0;
    return dot / std::sqrt(na * nb);
}

static double max_abs_err(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        double e = std::fabs((double)a[i] - (double)b[i]);
        if (e > m) m = e;
    }
    return m;
}

int main() {
    printf("=== QIE-Q2 Q4-resident Gate 0 probe ===\n");
    printf("Shape: x[M=%lld, K=%lld] F16  @  w[K=%lld, N=%lld] INT4 (G=%lld)\n",
           (long long)M, (long long)K, (long long)K, (long long)N, (long long)G);
    printf("Scale shape: [K/G=%lld, N=%lld] F16\n\n", (long long)K_G, (long long)N);

    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // ---------- 1) Host tensors ----------
    std::mt19937_64 rng(0xC0FFEE);
    std::uniform_real_distribution<float> dist(-0.08f, 0.08f);
    std::uniform_real_distribution<float> wdist(-0.5f, 0.5f);

    std::vector<uint16_t> x_host((size_t)M * K);
    for (auto& v : x_host) v = f32_to_f16(dist(rng));

    std::vector<float> w_dense((size_t)K * N);
    for (auto& v : w_dense) v = wdist(rng);

    printf("[host] Quantizing %lld × %lld weight Q4 per-group (G=%lld)...\n",
           (long long)K, (long long)N, (long long)G);
    Q4Pack q = cpu_quantize_q4_symmetric_per_group(w_dense, K, N, G);

    // ---------- 2) CPU reference ----------
    printf("[cpu]  Computing F32 reference via CPU F16 matmul over dequant...\n");
    std::vector<float> y_cpu;
    auto cpu_t0 = std::chrono::high_resolution_clock::now();
    cpu_matmul_f16(x_host, q.dequant_f16, y_cpu);
    auto cpu_t1 = std::chrono::high_resolution_clock::now();
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_t1 - cpu_t0).count();
    printf("[cpu]  Reference matmul done in %.1f ms\n", cpu_ms);

    // ---------- 3) Device upload ----------
    void *x_dev = nullptr, *w_dev = nullptr, *s_dev = nullptr, *y_dev = nullptr;
    size_t x_bytes = (size_t)M * K * sizeof(uint16_t);
    size_t w_bytes = q.packed.size();                  // K*N/2
    size_t s_bytes = q.scales.size() * sizeof(uint16_t);
    size_t y_bytes = (size_t)M * N * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc(&x_dev, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&w_dev, w_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&s_dev, s_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&y_dev, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(x_dev, x_bytes, x_host.data(), x_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(w_dev, w_bytes, q.packed.data(), w_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(s_dev, s_bytes, q.scales.data(), s_bytes,
                          ACL_MEMCPY_HOST_TO_DEVICE));
    printf("[npu]  Uploaded x=%zu B, w_int4=%zu B (%.2f MiB), scale=%zu B, y_out=%zu B\n",
           x_bytes, w_bytes, (double)w_bytes / (1024.0 * 1024.0), s_bytes, y_bytes);

    // ---------- 4) Build tensors ----------
    //
    // Per WQBMMv3 semantics (and the existing TTS w8_matmul_ template at
    // talker_cann_engine.cpp:562), weight expected shape is [K, N] with K
    // contiguous — we supply strides (1, K) to describe a [K, N] view whose
    // K-dim is contiguous inside each of the N columns. That matches how we
    // packed q.packed above (column-major K inside each N-column).
    //
    // Data type: ACL_INT4. On CANN 8.3 the dtype enum exists; the question
    // the probe answers is whether WQBMMv3 will bind INT4 → F16 at these
    // shapes with G=32.
    int64_t x_shape[2]   = {M, K};
    int64_t x_strides[2] = {K, 1};
    int64_t x_storage[2] = {M, K};
    aclTensor* t_x = aclCreateTensor(
        x_shape, 2, ACL_FLOAT16, x_strides, 0, ACL_FORMAT_ND,
        x_storage, 2, x_dev);

    // W4 tensor: shape [K, N], strides (1, K) so that N is the outer dim
    // (stride K) and K is contiguous inner (stride 1). Storage is the raw
    // packed nibble buffer: K*N/2 bytes, described as K*N elements of INT4.
    int64_t w_shape[2]   = {K, N};
    int64_t w_strides[2] = {1, K};
    int64_t w_storage    = K * N;  // INT4 element count
    aclTensor* t_w = aclCreateTensor(
        w_shape, 2, ACL_INT4, w_strides, 0, ACL_FORMAT_ND,
        &w_storage, 1, w_dev);

    // Scale tensor: shape [K/G, N], strides (N, 1) row-major.
    int64_t s_shape[2]   = {K_G, N};
    int64_t s_strides[2] = {N, 1};
    int64_t s_storage[2] = {K_G, N};
    aclTensor* t_s = aclCreateTensor(
        s_shape, 2, ACL_FLOAT16, s_strides, 0, ACL_FORMAT_ND,
        s_storage, 2, s_dev);

    int64_t y_shape[2]   = {M, N};
    int64_t y_strides[2] = {N, 1};
    int64_t y_storage[2] = {M, N};
    aclTensor* t_y = aclCreateTensor(
        y_shape, 2, ACL_FLOAT16, y_strides, 0, ACL_FORMAT_ND,
        y_storage, 2, y_dev);

    // ---------- 5) Dispatch WQBMMv3 ----------
    uint64_t ws_bytes = 0;
    aclOpExecutor* exec = nullptr;
    aclnnStatus st = aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
        t_x, t_w, t_s,
        /*antiquantOffsetOptional*/ nullptr,
        /*quantScaleOptional*/      nullptr,
        /*quantOffsetOptional*/     nullptr,
        /*biasOptional*/            nullptr,
        /*antiquantGroupSize*/      (int)G,
        /*innerPrecise*/            1,
        t_y, &ws_bytes, &exec);

    if (st != 0) {
        // Op refused the INT4 + per-group config. This is the most likely
        // RED outcome on CANN 8.3 — documented so PM can re-open the
        // 4-option decision surface.
        fprintf(stderr,
                "[npu]  *** WQBMMv3 GetWorkspaceSize REJECTED W4+G=32 config "
                "with status=%d ***\n",
                (int)st);
        fprintf(stderr,
                "[verdict] RED — op does not accept INT4 weight at these shapes.\n"
                "          Next step: vendor ask, or fall back to A16W8 shrink.\n");
        aclDestroyTensor(t_x); aclDestroyTensor(t_w);
        aclDestroyTensor(t_s); aclDestroyTensor(t_y);
        aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev); aclrtFree(y_dev);
        aclrtDestroyStream(stream);
        aclrtResetDevice(0);
        aclFinalize();
        return 2;
    }

    printf("[npu]  WQBMMv3 accepted W4+G=32 config. workspace=%llu B (%.2f MiB)\n",
           (unsigned long long)ws_bytes, (double)ws_bytes / (1024.0 * 1024.0));

    void* ws_dev = nullptr;
    if (ws_bytes > 0) {
        ACL_CHECK(aclrtMalloc(&ws_dev, ws_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    }

    // Warm-up then timed loop.
    const int warmup = 3;
    const int iters  = 20;
    for (int i = 0; i < warmup; ++i) {
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        // Need a fresh executor each call (aclnn single-shot semantics).
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, nullptr, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }

    std::vector<double> q4_times_us;
    q4_times_us.reserve(iters);
    for (int i = 0; i < iters; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3(ws_dev, ws_bytes, exec, stream));
        ACL_CHECK(aclrtSynchronizeStream(stream));
        auto t1 = std::chrono::high_resolution_clock::now();
        q4_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());

        ACL_CHECK_NN(aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
            t_x, t_w, t_s, nullptr, nullptr, nullptr, nullptr,
            (int)G, 1, t_y, &ws_bytes, &exec));
    }
    std::sort(q4_times_us.begin(), q4_times_us.end());
    double q4_median_us = q4_times_us[iters / 2];
    double q4_p10_us    = q4_times_us[iters / 10];
    double q4_p90_us    = q4_times_us[(iters * 9) / 10];
    printf("[npu]  W4 matmul wall: median=%.1f us  p10=%.1f us  p90=%.1f us (%d iters)\n",
           q4_median_us, q4_p10_us, q4_p90_us, iters);

    // ---------- 6) Copy result back, compare ----------
    std::vector<uint16_t> y_npu_h((size_t)M * N);
    ACL_CHECK(aclrtMemcpy(y_npu_h.data(), y_bytes, y_dev, y_bytes,
                          ACL_MEMCPY_DEVICE_TO_HOST));
    std::vector<float> y_npu_f32((size_t)M * N);
    for (size_t i = 0; i < y_npu_h.size(); ++i) y_npu_f32[i] = f16_to_f32(y_npu_h[i]);

    double cos = cosine_sim(y_cpu, y_npu_f32);
    double mae = max_abs_err(y_cpu, y_npu_f32);
    printf("\n[compare] cosine_sim(CPU ref, NPU W4 matmul) = %.6f\n", cos);
    printf("[compare] max_abs_err                          = %.6f\n", mae);

    // ---------- 7) F16 aclnnMm baseline for perf comparison ----------
    // Upload the full-F16 weights so we can time an apples-to-apples matmul.
    void* w_f16_dev = nullptr;
    size_t w_f16_bytes = (size_t)K * N * sizeof(uint16_t);
    ACL_CHECK(aclrtMalloc(&w_f16_dev, w_f16_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(w_f16_dev, w_f16_bytes, q.dequant_f16.data(),
                          w_f16_bytes, ACL_MEMCPY_HOST_TO_DEVICE));

    int64_t w16_shape[2]   = {K, N};
    int64_t w16_strides[2] = {N, 1};
    int64_t w16_storage[2] = {K, N};
    aclTensor* t_w16 = aclCreateTensor(
        w16_shape, 2, ACL_FLOAT16, w16_strides, 0, ACL_FORMAT_ND,
        w16_storage, 2, w_f16_dev);

    uint64_t ws_mm = 0;
    aclOpExecutor* exec_mm = nullptr;
    aclnnStatus st_mm = aclnnMatmulGetWorkspaceSize(
        t_x, t_w16, t_y, /*cubeMathType*/ 1, &ws_mm, &exec_mm);
    if (st_mm != 0) {
        fprintf(stderr, "[npu]  aclnnMatmul GetWorkspaceSize failed status=%d "
                        "(non-fatal — skipping baseline)\n", (int)st_mm);
    } else {
        void* ws_mm_dev = nullptr;
        if (ws_mm > 0) ACL_CHECK(aclrtMalloc(&ws_mm_dev, ws_mm, ACL_MEM_MALLOC_HUGE_FIRST));

        for (int i = 0; i < warmup; ++i) {
            ACL_CHECK_NN(aclnnMatmul(ws_mm_dev, ws_mm, exec_mm, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            ACL_CHECK_NN(aclnnMatmulGetWorkspaceSize(
                t_x, t_w16, t_y, 1, &ws_mm, &exec_mm));
        }
        std::vector<double> mm_times_us;
        mm_times_us.reserve(iters);
        for (int i = 0; i < iters; ++i) {
            auto t0 = std::chrono::high_resolution_clock::now();
            ACL_CHECK_NN(aclnnMatmul(ws_mm_dev, ws_mm, exec_mm, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            mm_times_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            ACL_CHECK_NN(aclnnMatmulGetWorkspaceSize(
                t_x, t_w16, t_y, 1, &ws_mm, &exec_mm));
        }
        std::sort(mm_times_us.begin(), mm_times_us.end());
        double mm_median_us = mm_times_us[iters / 2];
        printf("[npu]  F16 aclnnMm baseline wall: median=%.1f us  (same shape)\n",
               mm_median_us);
        printf("[perf] W4 / F16 ratio = %.2fx (target < 1.5x, win if < 1.0x)\n",
               q4_median_us / mm_median_us);

        if (ws_mm_dev) aclrtFree(ws_mm_dev);
    }

    // ---------- 8) Verdict ----------
    int rc = 0;
    const char* verdict = nullptr;
    if (cos > 0.99) {
        verdict = "GREEN";
        rc = 0;
    } else if (cos > 0.90) {
        verdict = "YELLOW";
        rc = 1;
    } else {
        verdict = "RED";
        rc = 2;
    }
    printf("\n[verdict] %s  (cos_sim = %.6f, mae = %.6f, W4 median = %.1f us)\n",
           verdict, cos, mae, q4_median_us);

    // ---------- Cleanup ----------
    aclDestroyTensor(t_x); aclDestroyTensor(t_w);
    aclDestroyTensor(t_s); aclDestroyTensor(t_y); aclDestroyTensor(t_w16);
    if (ws_dev) aclrtFree(ws_dev);
    aclrtFree(x_dev); aclrtFree(w_dev); aclrtFree(s_dev); aclrtFree(y_dev);
    aclrtFree(w_f16_dev);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return rc;
}
