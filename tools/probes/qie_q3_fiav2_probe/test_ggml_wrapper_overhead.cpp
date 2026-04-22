// Q3 companion probe — measure the ggml "wrapper ops" around FIAv2.
//
// Rationale: ggml_cann_flash_attn_ext IS aclnnFusedInferAttentionScoreV2
// (see ggml/src/ggml-cann/aclnn_ops.cpp:4449). The only delta between
// "ggml flash path" and "direct FIAv2 dispatch" is the wrapper ops ggml
// inserts around the same core call:
//   1. aclnn_cast Q: F32 → F16  (when src0 is F32 — QIE uses F16/BF16 so
//      this is a no-op cast in the best case, still costs a dispatch)
//   2. aclnnPermute + ggml_ext_cont on K/V for BSND layout conformance
//   3. aclnn_cast output: F16 → F32  (when dst is F32)
//
// This harness times those three ops at seq=4352 shape to bound the
// wrapper overhead that would be added on top of the FIAv2 core wall
// measured in test_qie_fiav2_seq4352.

#include <acl/acl.h>
#include <aclnnop/aclnn_cast.h>
#include <aclnnop/aclnn_permute.h>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

static double median_of(std::vector<double>& v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

static aclTensor* make_4d(void* data, int64_t a, int64_t b, int64_t c, int64_t d,
                           aclDataType dt) {
    int64_t shape[4]   = {a, b, c, d};
    int64_t strides[4] = {b * c * d, c * d, d, 1};
    return aclCreateTensor(shape, 4, dt, strides, 0, ACL_FORMAT_ND,
                           shape, 4, data);
}

int main() {
    ACL_CHECK(aclInit(nullptr));
    ACL_CHECK(aclrtSetDevice(0));
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream));

    // QIE target shape: [1, 4352, 24, 128].
    const int64_t B = 1, S = 4352, N = 24, D = 128;
    const size_t  cnt      = (size_t)B * S * N * D;
    const size_t  bytes_16 = cnt * 2;
    const size_t  bytes_32 = cnt * 4;

    void* d_f32_src = nullptr;
    void* d_f16_dst = nullptr;
    void* d_f16_alt = nullptr;
    ACL_CHECK(aclrtMalloc(&d_f32_src, bytes_32, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_f16_dst, bytes_16, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&d_f16_alt, bytes_16, ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemset(d_f32_src, bytes_32, 0, bytes_32));
    ACL_CHECK(aclrtMemset(d_f16_dst, bytes_16, 0, bytes_16));
    ACL_CHECK(aclrtMemset(d_f16_alt, bytes_16, 0, bytes_16));

    // Workspace
    void*    ws  = nullptr;
    uint64_t wsz = 0;
    auto ensure_ws = [&](uint64_t need) {
        if (need > wsz) {
            if (ws) aclrtFree(ws);
            ACL_CHECK(aclrtMalloc(&ws, need, ACL_MEM_MALLOC_HUGE_FIRST));
            wsz = need;
        }
    };

    const int WARMUP = 5;
    const int ITERS  = 50;

    // ---- Cast F32 → F16 at seq=4352 shape ----
    {
        std::vector<double> times;
        for (int i = 0; i < WARMUP + ITERS; ++i) {
            aclTensor*     src  = make_4d(d_f32_src, B, S, N, D, ACL_FLOAT);
            aclTensor*     dst  = make_4d(d_f16_dst, B, S, N, D, ACL_FLOAT16);
            uint64_t       need = 0;
            aclOpExecutor* exec = nullptr;
            ACL_CHECK_NN(aclnnCastGetWorkspaceSize(src, ACL_FLOAT16, dst, &need, &exec));
            ensure_ws(need);
            auto t0 = std::chrono::high_resolution_clock::now();
            ACL_CHECK_NN(aclnnCast(need > 0 ? ws : nullptr, need, exec, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= WARMUP) {
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            aclDestroyTensor(src);
            aclDestroyTensor(dst);
        }
        printf("aclnnCast F32->F16  @ [%ld,%ld,%ld,%ld]  median μs = %.1f  (workspace %llu B)\n",
               (long)B, (long)S, (long)N, (long)D, median_of(times),
               (unsigned long long)wsz);
    }

    // ---- Cast F16 → F32 at seq=4352 shape ----
    {
        std::vector<double> times;
        for (int i = 0; i < WARMUP + ITERS; ++i) {
            aclTensor*     src  = make_4d(d_f16_dst, B, S, N, D, ACL_FLOAT16);
            aclTensor*     dst  = make_4d(d_f32_src, B, S, N, D, ACL_FLOAT);
            uint64_t       need = 0;
            aclOpExecutor* exec = nullptr;
            ACL_CHECK_NN(aclnnCastGetWorkspaceSize(src, ACL_FLOAT, dst, &need, &exec));
            ensure_ws(need);
            auto t0 = std::chrono::high_resolution_clock::now();
            ACL_CHECK_NN(aclnnCast(need > 0 ? ws : nullptr, need, exec, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= WARMUP) {
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            aclDestroyTensor(src);
            aclDestroyTensor(dst);
        }
        printf("aclnnCast F16->F32  @ [%ld,%ld,%ld,%ld]  median μs = %.1f  (workspace %llu B)\n",
               (long)B, (long)S, (long)N, (long)D, median_of(times),
               (unsigned long long)wsz);
    }

    // ---- Permute [B,N,S,D] -> [B,S,N,D] (ggml's transpose12 materialized) ----
    {
        std::vector<double> times;
        for (int i = 0; i < WARMUP + ITERS; ++i) {
            // src laid out as [B, N, S, D] logical view
            int64_t src_shape[4]   = {B, N, S, D};
            int64_t src_strides[4] = {N * S * D, S * D, D, 1};
            aclTensor* src = aclCreateTensor(src_shape, 4, ACL_FLOAT16,
                                             src_strides, 0, ACL_FORMAT_ND,
                                             src_shape, 4, d_f16_dst);
            // dst laid out as [B, S, N, D]
            int64_t dst_shape[4]   = {B, S, N, D};
            int64_t dst_strides[4] = {S * N * D, N * D, D, 1};
            aclTensor* dst = aclCreateTensor(dst_shape, 4, ACL_FLOAT16,
                                             dst_strides, 0, ACL_FORMAT_ND,
                                             dst_shape, 4, d_f16_alt);
            int64_t perm[4] = {0, 2, 1, 3};  // [B,N,S,D] -> [B,S,N,D]
            aclIntArray* perm_arr = aclCreateIntArray(perm, 4);
            uint64_t       need = 0;
            aclOpExecutor* exec = nullptr;
            ACL_CHECK_NN(aclnnPermuteGetWorkspaceSize(src, perm_arr, dst, &need, &exec));
            ensure_ws(need);
            auto t0 = std::chrono::high_resolution_clock::now();
            ACL_CHECK_NN(aclnnPermute(need > 0 ? ws : nullptr, need, exec, stream));
            ACL_CHECK(aclrtSynchronizeStream(stream));
            auto t1 = std::chrono::high_resolution_clock::now();
            if (i >= WARMUP) {
                times.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
            }
            aclDestroyTensor(src);
            aclDestroyTensor(dst);
            aclDestroyIntArray(perm_arr);
        }
        printf("aclnnPermute BNSD->BSND F16 @ [%ld,%ld,%ld,%ld]  median μs = %.1f  (workspace %llu B)\n",
               (long)B, (long)S, (long)N, (long)D, median_of(times),
               (unsigned long long)wsz);
    }

    if (ws) aclrtFree(ws);
    aclrtFree(d_f32_src);
    aclrtFree(d_f16_dst);
    aclrtFree(d_f16_alt);

    ACL_CHECK(aclrtDestroyStream(stream));
    ACL_CHECK(aclrtResetDevice(0));
    ACL_CHECK(aclFinalize());
    printf("DONE\n");
    return 0;
}
