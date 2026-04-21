// SPDX-License-Identifier: Apache-2.0
// =============================================================================
// fused_attn_sublayer.cpp — AscendC custom kernel for Path C W4.1.
// W4.1.2v rewrite: replaces scalar FMA loops on LocalTensor with AscendC
// vector primitives (Cast / Muls / Mul / Adds / Duplicate / Exp /
// ReduceSum / DataCopy). The scalar version hung the watchdog because the
// vector pipeline sat idle while 2 048 × 1 024 FMAs ran one lane at a
// time; the rewrite keeps the pipeline hot and uses EnQue / DeQue to
// correctly sync DMA (MTE2) against vector (V) stages.
//
// Kernel signature + arg pack are unchanged so the W4.1.2 harness
// (test_fused_attn_diff.cpp, commit aa91ed9e) still works and the
// auto-generated aclrtlaunch_fused_attn_sublayer host stub is compatible.
//
// -----------------------------------------------------------------------------
// Op chain collapsed into this kernel (stock aclnn chain being replaced):
//     RmsNorm(cur, in_ln_gamma)                 [1 dispatch]
//     Q = W8Mm(normed, Wq_i8, Wq_scale)         [1]
//     K = W8Mm(normed, Wk_i8, Wk_scale)         [1]
//     V = W8Mm(normed, Wv_i8, Wv_scale)         [1]
//     RmsNorm(Q, q_norm),  RmsNorm(K, k_norm)   [2]
//     RoPE(Q), RoPE(K)                          [2]
//     memcpy V → v_cache[pos]                   [pure copy]
//     FusedInferAttentionScoreV2(Q, Kc, Vc)     [1]
//     O = W8Mm(attn_out, Wo_i8, Wo_scale)       [1]
//     residual += O                             [1]
//   Total: 12 aclnn dispatches → 1 aclrtlaunch.
//
// -----------------------------------------------------------------------------
// Shapes (baked in — match cp_cann_engine.h 187-198)
// -----------------------------------------------------------------------------
//   cp_hidden = 1024   (d_model)
//   q_dim     = 2048   (n_heads=16 × head_dim=128)
//   kv_dim    = 1024   (n_kv=8     × head_dim=128)
//   head_dim  = 128
//   MAX_SEQ   = 17
//
// -----------------------------------------------------------------------------
// Precision / reduction primitives
// -----------------------------------------------------------------------------
//   Storage: F16 throughout. Reductions: F32 (Cast after Mul, ReduceSum
//   in F32, scalar-promote the tiny final divisions).
//   Softmax: (x - max) in F32, Exp() primitive, ReduceSum sum, divide.
//
//   Reductions use the Level-2 ReduceSum(dst, src, workBuf, count)
//   interface (kernel_operator_vec_reduce_intf.h). The workBuf must hold
//   ceil(count/64)+1 F32 partials — sized at 128 F32 = 512 B, covers
//   up to count=8 192.
//
// -----------------------------------------------------------------------------
// Pipe synchronisation (this is the anti-hang contract)
// -----------------------------------------------------------------------------
//   DataCopy(UB←GM) runs on the MTE2 pipe. Vector ops (Mul / Cast /
//   Muls / Add / ReduceSum / Exp / Duplicate) run on the V pipe.
//   Each GM→UB DataCopy is followed by que.EnQue(tensor); que.DeQue<T>()
//   before the first vector op reads it. The DeQue'd read-only handle is
//   what the vector op consumes. Symmetric pattern for GM←UB.
//   FreeTensor releases the slot for reuse by the next Alloc.
// =============================================================================

#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t kCpHidden = 1024;
constexpr int32_t kQDim     = 2048;
constexpr int32_t kKvDim    = 1024;
constexpr int32_t kHeadDim  = 128;
constexpr int32_t kNHeads   = 16;
constexpr int32_t kNKv      = 8;
constexpr int32_t kGroup    = kNHeads / kNKv;   // GQA group = 2

constexpr int32_t kMaxSeq   = 17;

constexpr int32_t kPipeBufCount = 1;

class KernelFusedAttnSublayer {
public:
    __aicore__ inline KernelFusedAttnSublayer() {}

    __aicore__ inline void Init(
        GM_ADDR residual, GM_ADDR normed, GM_ADDR in_ln_gamma,
        GM_ADDR wq_i8, GM_ADDR wq_scale,
        GM_ADDR wk_i8, GM_ADDR wk_scale,
        GM_ADDR wv_i8, GM_ADDR wv_scale,
        GM_ADDR wo_i8, GM_ADDR wo_scale,
        GM_ADDR q_norm_f16, GM_ADDR k_norm_f16,
        GM_ADDR rope_cos, GM_ADDR rope_sin,
        GM_ADDR k_cache, GM_ADDR v_cache,
        GM_ADDR o_out, GM_ADDR scratch_q, GM_ADDR scratch_scores,
        uint32_t seq_len, uint32_t eps_bits, uint32_t opts)
    {
        seqLen_ = seq_len;
        opts_   = opts;
        float eps_f32;
        __builtin_memcpy(&eps_f32, &eps_bits, sizeof(float));
        epsF32_ = eps_f32;

        residualGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(residual),
                                     kCpHidden);
        normedGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(normed),
                                     kCpHidden);
        inLnGm_.SetGlobalBuffer    (reinterpret_cast<__gm__ half*>(in_ln_gamma),
                                     kCpHidden);
        wqI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wq_i8),
                                     (uint32_t)kQDim * kCpHidden);
        wqScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wq_scale),
                                     kQDim);
        wkI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wk_i8),
                                     (uint32_t)kKvDim * kCpHidden);
        wkScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wk_scale),
                                     kKvDim);
        wvI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wv_i8),
                                     (uint32_t)kKvDim * kCpHidden);
        wvScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wv_scale),
                                     kKvDim);
        woI8Gm_.SetGlobalBuffer    (reinterpret_cast<__gm__ int8_t*>(wo_i8),
                                     (uint32_t)kCpHidden * kQDim);
        woScaleGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(wo_scale),
                                     kCpHidden);
        qNormGm_.SetGlobalBuffer   (reinterpret_cast<__gm__ half*>(q_norm_f16),
                                     kHeadDim);
        kNormGm_.SetGlobalBuffer   (reinterpret_cast<__gm__ half*>(k_norm_f16),
                                     kHeadDim);
        ropeCosGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(rope_cos),
                                     kHeadDim);
        ropeSinGm_.SetGlobalBuffer (reinterpret_cast<__gm__ half*>(rope_sin),
                                     kHeadDim);
        kCacheGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(k_cache),
                                     (uint32_t)kMaxSeq * kKvDim);
        vCacheGm_.SetGlobalBuffer  (reinterpret_cast<__gm__ half*>(v_cache),
                                     (uint32_t)kMaxSeq * kKvDim);
        oOutGm_.SetGlobalBuffer    (reinterpret_cast<__gm__ half*>(o_out),
                                     kCpHidden);
        (void)scratch_q;
        (void)scratch_scores;

        // UB queues. Sizes chosen so the biggest single op (kQDim F32 =
        // 8 KB) fits; separate queues per role keep dep-tracking clean.
        pipe_.InitBuffer(normedQ_,   kPipeBufCount, kCpHidden * sizeof(half));   // normed F16
        pipe_.InitBuffer(nF32Q_,     kPipeBufCount, kQDim     * sizeof(float));  // F32 scratch for norm + attn
        pipe_.InitBuffer(wRowI8Q_,   kPipeBufCount, kQDim     * sizeof(int8_t)); // i8 weight row
        pipe_.InitBuffer(wRowF16Q_,  kPipeBufCount, kQDim     * sizeof(half));   // f16 dequant
        pipe_.InitBuffer(qF16Q_,     kPipeBufCount, kQDim     * sizeof(half));   // Q F16 output
        pipe_.InitBuffer(kF16Q_,     kPipeBufCount, kKvDim    * sizeof(half));   // K F16 output
        pipe_.InitBuffer(vF16Q_,     kPipeBufCount, kKvDim    * sizeof(half));   // V F16 output
        pipe_.InitBuffer(gammaQ_,    kPipeBufCount, kQDim     * sizeof(half));   // gamma/scale staging
        pipe_.InitBuffer(scratchF32Q_,kPipeBufCount,kQDim     * sizeof(float));  // F32 scratch (prod, reduce src)
        pipe_.InitBuffer(workF32Q_,  kPipeBufCount, 128       * sizeof(float));  // ReduceSum work
        pipe_.InitBuffer(oF16Q_,     kPipeBufCount, kCpHidden * sizeof(half));   // O projection output
        pipe_.InitBuffer(residOutQ_, kPipeBufCount, kCpHidden * sizeof(half));   // residual out write-back
        pipe_.InitBuffer(kvRowQ_,    kPipeBufCount, kKvDim    * sizeof(half));   // K/V cache row read
    }

    // ------------------------------------------------------------
    // Process — one block, one iteration per launch.
    // ------------------------------------------------------------
    __aicore__ inline void Process() {
        // ---- 1. normed = RmsNorm(residual, in_ln_gamma) OR copy pre-computed normed
        LocalTensor<half> normed = LoadOrComputeNormed_();

        // ---- 2. Q/K/V projections via W8 matmul (vector path)
        LocalTensor<half> qF16 = qF16Q_.AllocTensor<half>();
        W8MatmulVec_(normed, wqI8Gm_, wqScaleGm_, qF16, kQDim,  kCpHidden);

        LocalTensor<half> kF16 = kF16Q_.AllocTensor<half>();
        W8MatmulVec_(normed, wkI8Gm_, wkScaleGm_, kF16, kKvDim, kCpHidden);

        LocalTensor<half> vF16 = vF16Q_.AllocTensor<half>();
        W8MatmulVec_(normed, wvI8Gm_, wvScaleGm_, vF16, kKvDim, kCpHidden);

        normedQ_.FreeTensor(normed);

        // ---- 3. Per-head RmsNorm on Q and K
        PerHeadRmsNorm_(qF16, qNormGm_, kNHeads);
        PerHeadRmsNorm_(kF16, kNormGm_, kNKv);

        // ---- 4. NEOX RoPE on Q and K
        ApplyRopeToQK_(qF16, kF16);

        // ---- 5. Append K/V into the cache at slot = seqLen_ - 1
        AppendKVCache_(kF16, vF16);

        // NB: AppendKVCache_ already FreeTensor'd kF16 / vF16 (see comment
        // inside that helper). The caller must not Free them again.

        // ---- 6. Attention — writes output over qF16 in-place.
        ComputeAttention_(qF16);

        // ---- 7. O projection W8Mm
        LocalTensor<half> oF16 = oF16Q_.AllocTensor<half>();
        W8MatmulVec_(qF16, woI8Gm_, woScaleGm_, oF16, kCpHidden, kQDim);
        qF16Q_.FreeTensor(qF16);

        // ---- 8. Emit o_out, update residual += O
        oF16Q_.EnQue(oF16);
        {
            LocalTensor<half> oOut = oF16Q_.DeQue<half>();
            DataCopy(oOutGm_, oOut, kCpHidden);
            oF16Q_.FreeTensor(oOut);
        }
        // oF16 is GM-written; pull it back via the residual add path.
        // Easiest: re-alloc and re-read O (it's now in GM), then add to resid.
        AddResidualAndWriteBack_();
    }

private:
    // ------------------------------------------------------------
    // LoadOrComputeNormed_ — returns UB tensor with RmsNorm(residual, gamma)
    // applied (or the pre-computed `normed` if fusion_active bit is set).
    // ------------------------------------------------------------
    __aicore__ inline LocalTensor<half> LoadOrComputeNormed_() {
        LocalTensor<half> normed = normedQ_.AllocTensor<half>();
        const bool fusion_active = (opts_ & 1u) != 0u;
        if (fusion_active) {
            DataCopy(normed, normedGm_, kCpHidden);
            normedQ_.EnQue(normed);
            LocalTensor<half> ro = normedQ_.DeQue<half>();
            // Hand ownership back to caller. (We consumed the only slot so
            // just return the DeQue'd handle — pipe flow unaffected.)
            return ro;
        }

        // Compute path: load residual, load gamma, run RmsNormInPlace_.
        DataCopy(normed, residualGm_, kCpHidden);
        normedQ_.EnQue(normed);
        LocalTensor<half> normedRo = normedQ_.DeQue<half>();

        LocalTensor<half> gamma = gammaQ_.AllocTensor<half>();
        DataCopy(gamma, inLnGm_, kCpHidden);
        gammaQ_.EnQue(gamma);
        LocalTensor<half> gammaRo = gammaQ_.DeQue<half>();

        RmsNormInPlace_(normedRo, gammaRo, kCpHidden);

        gammaQ_.FreeTensor(gammaRo);
        return normedRo;
    }

    // ------------------------------------------------------------
    // ReduceSumF32_ — scalar reduce of count F32 elements into dst[0].
    // Wraps the Level-2 ReduceSum(dst, src, workBuf, count) interface
    // with a persistent work buffer. Returns the value as a float.
    // ------------------------------------------------------------
    __aicore__ inline float ReduceSumF32_(LocalTensor<float>& dst,
                                           LocalTensor<float>& src,
                                           int32_t count)
    {
        LocalTensor<float> work = workF32Q_.AllocTensor<float>();
        ReduceSum<float>(dst, src, work, count);
        workF32Q_.FreeTensor(work);
        return dst.GetValue(0);
    }

    // ------------------------------------------------------------
    // RmsNormInPlace_
    //
    //   y = x * (1 / sqrt(mean(x^2) + eps)) * gamma
    //
    // Vectorised path:
    //   nF32  = Cast<f32>(x)            vector
    //   sqF32 = nF32 * nF32             vector
    //   sum   = ReduceSum(sqF32)        level-2
    //   rstd  = 1/sqrt(sum/count + eps) scalar once
    //   nF32  = Muls(nF32, rstd)        vector
    //   x     = Cast<f16>(nF32)         vector
    //   x     = Mul(x, gamma)           vector
    // ------------------------------------------------------------
    __aicore__ inline void RmsNormInPlace_(LocalTensor<half>& x,
                                            LocalTensor<half>& gamma,
                                            int32_t len)
    {
        LocalTensor<float> nF32  = nF32Q_.AllocTensor<float>();
        LocalTensor<float> sqF32 = scratchF32Q_.AllocTensor<float>();

        Cast(nF32, x, RoundMode::CAST_NONE, len);
        Mul (sqF32, nF32, nF32, len);

        float sum_sq  = ReduceSumF32_(sqF32, sqF32, len);
        float mean_sq = sum_sq / (float)len;
        float rstd    = 1.0f / sqrt(mean_sq + epsF32_);

        Muls(nF32, nF32, rstd, len);
        Cast(x, nF32, RoundMode::CAST_NONE, len);
        Mul (x, x, gamma, len);

        scratchF32Q_.FreeTensor(sqF32);
        nF32Q_.FreeTensor(nF32);
    }

    // ------------------------------------------------------------
    // W8MatmulVec_  —  y[r] = Σ_c x[c] * (w_i8[r,c] * scale[r])
    //
    // Pre-stage scale all-at-once (fits in gammaQ_), then per-row:
    //   wRowI8 = DataCopy(w_i8[r,:])    MTE2
    //   EnQue(wRowI8Q_); DeQue<i8>()   — sync MTE2 → V
    //   wRowF16= Cast<f16>(wRowI8)      vector
    //   Muls(wRowF16, scale[r])         vector (per-row scalar)
    //   Mul (prodF16 = wRowF16 * x)     vector
    //   Cast(prodF32, prodF16)          vector
    //   y[r]   = ReduceSum(prodF32)     level-2
    // ------------------------------------------------------------
    __aicore__ inline void W8MatmulVec_(LocalTensor<half>& x,
                                         GlobalTensor<int8_t>& wI8Gm,
                                         GlobalTensor<half>& wScaleGm,
                                         LocalTensor<half>& y,
                                         int32_t out_n,
                                         int32_t in_k)
    {
        // Pre-stage all scales.
        LocalTensor<half> scaleAlloc = gammaQ_.AllocTensor<half>();
        DataCopy(scaleAlloc, wScaleGm, out_n);
        gammaQ_.EnQue(scaleAlloc);
        LocalTensor<half> scaleAll = gammaQ_.DeQue<half>();

        // Per-row working buffers — allocated once, reused per iteration.
        LocalTensor<half>   wRowF16   = wRowF16Q_.AllocTensor<half>();
        LocalTensor<float>  prodF32   = scratchF32Q_.AllocTensor<float>();
        LocalTensor<float>  reduceDst = nF32Q_.AllocTensor<float>();

        for (int32_t r = 0; r < out_n; ++r) {
            // Stream one int8 row from GM into UB with full MTE2 → V sync.
            LocalTensor<int8_t> wRowI8 = wRowI8Q_.AllocTensor<int8_t>();
            DataCopy(wRowI8, wI8Gm[(uint32_t)r * (uint32_t)in_k], in_k);
            wRowI8Q_.EnQue(wRowI8);
            LocalTensor<int8_t> wRowI8Ro = wRowI8Q_.DeQue<int8_t>();

            Cast(wRowF16, wRowI8Ro, RoundMode::CAST_NONE, in_k);
            wRowI8Q_.FreeTensor(wRowI8Ro);

            half scale_r = scaleAll.GetValue(r);
            Muls(wRowF16, wRowF16, scale_r, in_k);

            Mul (wRowF16, wRowF16, x, in_k);
            Cast(prodF32, wRowF16, RoundMode::CAST_NONE, in_k);

            float acc = ReduceSumF32_(reduceDst, prodF32, in_k);
            y.SetValue(r, (half)acc);
        }

        nF32Q_.FreeTensor(reduceDst);
        scratchF32Q_.FreeTensor(prodF32);
        wRowF16Q_.FreeTensor(wRowF16);
        gammaQ_.FreeTensor(scaleAll);
    }

    // ------------------------------------------------------------
    // PerHeadRmsNorm_ — apply per-head RmsNorm over a [n, head_dim] tensor
    // using a single gamma[head_dim] vector loaded once into UB.
    // ------------------------------------------------------------
    __aicore__ inline void PerHeadRmsNorm_(LocalTensor<half>& x,
                                            GlobalTensor<half>& gammaGm,
                                            int32_t n_heads)
    {
        LocalTensor<half> gammaAlloc = gammaQ_.AllocTensor<half>();
        DataCopy(gammaAlloc, gammaGm, kHeadDim);
        gammaQ_.EnQue(gammaAlloc);
        LocalTensor<half> gamma = gammaQ_.DeQue<half>();

        for (int32_t h = 0; h < n_heads; ++h) {
            LocalTensor<half> x_h = x[h * kHeadDim];
            RmsNormInPlace_(x_h, gamma, kHeadDim);
        }

        gammaQ_.FreeTensor(gamma);
    }

    // ------------------------------------------------------------
    // ApplyRopeToQK_ — load cos/sin into UB once, then rotate each
    // Q head + each K kv-head via NEOX rotate_half.
    //
    //   out_lo = x_lo * c - x_hi * s
    //   out_hi = x_lo * s + x_hi * c
    //
    // 4 temp slices into wRowF16Q_ (128 * 4 = 512 F16 = 1 KB) avoid the
    // lo/hi alias hazard.
    // ------------------------------------------------------------
    __aicore__ inline void ApplyRopeToQK_(LocalTensor<half>& qF16,
                                            LocalTensor<half>& kF16)
    {
        LocalTensor<half> cosAlloc = gammaQ_.AllocTensor<half>();
        DataCopy(cosAlloc, ropeCosGm_, kHeadDim);
        gammaQ_.EnQue(cosAlloc);
        LocalTensor<half> cos_t = gammaQ_.DeQue<half>();

        LocalTensor<half> sinAlloc = kvRowQ_.AllocTensor<half>();
        DataCopy(sinAlloc, ropeSinGm_, kHeadDim);
        kvRowQ_.EnQue(sinAlloc);
        LocalTensor<half> sin_t = kvRowQ_.DeQue<half>();

        ApplyRopeAllHeads_(qF16, cos_t, sin_t, kNHeads);
        ApplyRopeAllHeads_(kF16, cos_t, sin_t, kNKv);

        kvRowQ_.FreeTensor(sin_t);
        gammaQ_.FreeTensor(cos_t);
    }

    __aicore__ inline void ApplyRopeAllHeads_(LocalTensor<half>& x,
                                                LocalTensor<half>& cos_t,
                                                LocalTensor<half>& sin_t,
                                                int32_t n_heads)
    {
        constexpr int32_t kHalf = kHeadDim / 2;

        LocalTensor<half> tmp = wRowF16Q_.AllocTensor<half>();
        LocalTensor<half> tmp_xlo_c = tmp[0];
        LocalTensor<half> tmp_xhi_s = tmp[kHalf];
        LocalTensor<half> tmp_xlo_s = tmp[2 * kHalf];
        LocalTensor<half> tmp_xhi_c = tmp[3 * kHalf];

        for (int32_t h = 0; h < n_heads; ++h) {
            int32_t base = h * kHeadDim;
            LocalTensor<half> x_lo = x[base];
            LocalTensor<half> x_hi = x[base + kHalf];

            LocalTensor<half> c = cos_t[0];
            LocalTensor<half> s = sin_t[0];

            Mul(tmp_xlo_c, x_lo, c, kHalf);
            Mul(tmp_xhi_s, x_hi, s, kHalf);
            Mul(tmp_xlo_s, x_lo, s, kHalf);
            Mul(tmp_xhi_c, x_hi, c, kHalf);

            Sub(x_lo, tmp_xlo_c, tmp_xhi_s, kHalf);
            Add(x_hi, tmp_xlo_s, tmp_xhi_c, kHalf);
        }

        wRowF16Q_.FreeTensor(tmp);
    }

    // ------------------------------------------------------------
    // AppendKVCache_ — DataCopy K/V into cache at slot = seqLen_ - 1.
    // Both k and v are in UB (K just went through RoPE vector ops, V came
    // straight out of the W8 matmul). The EnQue + DeQue round-trip on
    // each VECOUT queue (kF16Q_ / vF16Q_) provides the V → MTE3 sync the
    // GM write needs. Without it, the DMA reads partial vector output.
    // ------------------------------------------------------------
    __aicore__ inline void AppendKVCache_(LocalTensor<half>& k,
                                           LocalTensor<half>& v)
    {
        const uint32_t off = (seqLen_ - 1) * (uint32_t)kKvDim;
        // K: EnQue the VECOUT buffer, DeQue for MTE3 access.
        kF16Q_.EnQue(k);
        LocalTensor<half> kRo = kF16Q_.DeQue<half>();
        DataCopy(kCacheGm_[off], kRo, kKvDim);
        // We keep kRo live because the caller (Process) still wants to
        // use `kF16` after this — but because we returned ownership via
        // the DeQue, the original `k` handle is now `kRo`. Rebind the
        // caller's reference by updating the argument through the
        // pointer-equivalent behaviour: the caller will continue using
        // the same LocalTensor handle since AllocTensor returns a UB
        // address wrapper and the backing memory is unchanged.
        //
        // However AscendC's queue API wants strict Alloc → (write) →
        // EnQue → DeQue → (read) → Free order. After this EnQue/DeQue,
        // the slot semantics say the "producer-side handle" is closed.
        // To keep things clean, we FreeTensor here — the caller's
        // subsequent FreeTensor on the original handle would double-free.
        //
        // Simplest fix: caller doesn't use `k` / `v` after this call.
        // `Process` calls ComputeAttention_ next which reads K/V from
        // the cache (GM), not from UB k/v. So it's safe to free the UB
        // copy here. Adjust the free in `Process` by removing the
        // redundant FreeTensor for k / v.
        kF16Q_.FreeTensor(kRo);

        vF16Q_.EnQue(v);
        LocalTensor<half> vRo = vF16Q_.DeQue<half>();
        DataCopy(vCacheGm_[off], vRo, kKvDim);
        vF16Q_.FreeTensor(vRo);
    }

    // ------------------------------------------------------------
    // ComputeAttention_ — GQA scaled-dot-product attention.
    // Writes output over `q` in-place. Accumulator kept in F32 to
    // avoid F16 precision loss across kMaxSeq=17 positions.
    // ------------------------------------------------------------
    __aicore__ inline void ComputeAttention_(LocalTensor<half>& q)
    {
        const float inv_sqrt_d = 1.0f / sqrt((float)kHeadDim);

        float scores[kNHeads][kMaxSeq];
        float max_score[kNHeads];
        for (int32_t h = 0; h < kNHeads; ++h) max_score[h] = -3.4e38f;

        LocalTensor<half>  prodF16  = wRowF16Q_.AllocTensor<half>();
        LocalTensor<float> prodF32  = scratchF32Q_.AllocTensor<float>();
        LocalTensor<float> attn     = nF32Q_.AllocTensor<float>();
        Duplicate(attn, 0.0f, kQDim);

        // ---- Score computation ----
        for (uint32_t t = 0; t < seqLen_; ++t) {
            LocalTensor<half> kRowAlloc = kvRowQ_.AllocTensor<half>();
            DataCopy(kRowAlloc, kCacheGm_[t * (uint32_t)kKvDim], kKvDim);
            kvRowQ_.EnQue(kRowAlloc);
            LocalTensor<half> kRow = kvRowQ_.DeQue<half>();

            for (int32_t h = 0; h < kNHeads; ++h) {
                int32_t kv_h   = h / kGroup;
                int32_t q_base = h    * kHeadDim;
                int32_t k_base = kv_h * kHeadDim;

                LocalTensor<half>  q_h    = q     [q_base];
                LocalTensor<half>  k_h    = kRow  [k_base];
                LocalTensor<half>  dst16  = prodF16[0];
                LocalTensor<float> dst32  = prodF32[0];

                Mul (dst16, q_h, k_h, kHeadDim);
                Cast(dst32, dst16, RoundMode::CAST_NONE, kHeadDim);
                float s = ReduceSumF32_(dst32, dst32, kHeadDim) * inv_sqrt_d;
                scores[h][t] = s;
                if (s > max_score[h]) max_score[h] = s;
            }

            kvRowQ_.FreeTensor(kRow);
        }

        // ---- Softmax (per-head) ----
        for (int32_t h = 0; h < kNHeads; ++h) {
            // Stage scores[h][0..seqLen_) into prodF32 (F32), subtract max.
            for (uint32_t t = 0; t < seqLen_; ++t) {
                prodF32.SetValue((int32_t)t, scores[h][t] - max_score[h]);
            }
            Exp(prodF32, prodF32, (int32_t)seqLen_);

            // Sum. Note ReduceSumF32_ writes prodF32[0] with the sum —
            // losing one Exp value. Re-stage after the reduce.
            float sum_exp = ReduceSumF32_(prodF32, prodF32, (int32_t)seqLen_);
            float inv_sum = 1.0f / sum_exp;

            // Re-stage Exp results, normalise, store back into scores.
            for (uint32_t t = 0; t < seqLen_; ++t) {
                prodF32.SetValue((int32_t)t, scores[h][t] - max_score[h]);
            }
            Exp(prodF32, prodF32, (int32_t)seqLen_);
            for (uint32_t t = 0; t < seqLen_; ++t) {
                scores[h][t] = prodF32.GetValue((int32_t)t) * inv_sum;
            }
        }

        // ---- Weighted-V accumulation ----
        for (uint32_t t = 0; t < seqLen_; ++t) {
            LocalTensor<half> vRowAlloc = kvRowQ_.AllocTensor<half>();
            DataCopy(vRowAlloc, vCacheGm_[t * (uint32_t)kKvDim], kKvDim);
            kvRowQ_.EnQue(vRowAlloc);
            LocalTensor<half> vRow = kvRowQ_.DeQue<half>();

            for (int32_t h = 0; h < kNHeads; ++h) {
                int32_t kv_h   = h / kGroup;
                int32_t v_base = kv_h * kHeadDim;
                int32_t o_base = h    * kHeadDim;

                LocalTensor<half>  v_h     = vRow  [v_base];
                LocalTensor<half>  scaled  = prodF16[0];
                LocalTensor<float> scaledF = prodF32[0];
                LocalTensor<float> attn_h  = attn  [o_base];

                half s_h = (half)scores[h][t];
                Muls(scaled, v_h, s_h, kHeadDim);
                Cast(scaledF, scaled, RoundMode::CAST_NONE, kHeadDim);
                Add (attn_h, attn_h, scaledF, kHeadDim);
            }

            kvRowQ_.FreeTensor(vRow);
        }

        // ---- Cast accumulator back to F16 over q[] in-place ----
        Cast(q, attn, RoundMode::CAST_NONE, kQDim);

        nF32Q_.FreeTensor(attn);
        scratchF32Q_.FreeTensor(prodF32);
        wRowF16Q_.FreeTensor(prodF16);
    }

    // ------------------------------------------------------------
    // AddResidualAndWriteBack_ — resid = resid + o_out (in GM).
    //
    // o_out was just written to GM by the kernel (before this call).
    // We re-read both resid and o_out from GM, sum in UB, write back.
    // (Cheap: 2 KB * 2 reads + 2 KB write per frame.)
    // ------------------------------------------------------------
    __aicore__ inline void AddResidualAndWriteBack_() {
        // Stage resid (GM → UB) with MTE2 → V sync.
        LocalTensor<half> residAlloc = residOutQ_.AllocTensor<half>();
        DataCopy(residAlloc, residualGm_, kCpHidden);
        residOutQ_.EnQue(residAlloc);
        LocalTensor<half> resid = residOutQ_.DeQue<half>();

        // Stage o_out (GM → UB) with MTE2 → V sync.
        LocalTensor<half> oAlloc = oF16Q_.AllocTensor<half>();
        DataCopy(oAlloc, oOutGm_, kCpHidden);
        oF16Q_.EnQue(oAlloc);
        LocalTensor<half> oOut = oF16Q_.DeQue<half>();

        // Vector add.
        Add(resid, resid, oOut, kCpHidden);
        oF16Q_.FreeTensor(oOut);

        // V → MTE3 sync: round-trip the VECOUT queue before GM write.
        // residOutQ_ is positioned VECOUT so the En/De chains the sync.
        // (resid handle was obtained via DeQue above; re-EnQue is a
        // producer-to-MTE3 handshake.)
        residOutQ_.EnQue(resid);
        LocalTensor<half> residOut = residOutQ_.DeQue<half>();
        DataCopy(residualGm_, residOut, kCpHidden);
        residOutQ_.FreeTensor(residOut);
    }

    // ---------- pipe / queue state ----------
    TPipe pipe_;
    TQue<QuePosition::VECIN,  kPipeBufCount> normedQ_;
    TQue<QuePosition::VECIN,  kPipeBufCount> nF32Q_;
    TQue<QuePosition::VECIN,  kPipeBufCount> wRowI8Q_;
    TQue<QuePosition::VECIN,  kPipeBufCount> wRowF16Q_;
    TQue<QuePosition::VECOUT, kPipeBufCount> qF16Q_;
    TQue<QuePosition::VECOUT, kPipeBufCount> kF16Q_;
    TQue<QuePosition::VECOUT, kPipeBufCount> vF16Q_;
    TQue<QuePosition::VECIN,  kPipeBufCount> gammaQ_;
    TQue<QuePosition::VECIN,  kPipeBufCount> scratchF32Q_;
    TQue<QuePosition::VECIN,  kPipeBufCount> workF32Q_;
    TQue<QuePosition::VECOUT, kPipeBufCount> oF16Q_;
    TQue<QuePosition::VECOUT, kPipeBufCount> residOutQ_;
    TQue<QuePosition::VECIN,  kPipeBufCount> kvRowQ_;

    // ---------- GM views ----------
    GlobalTensor<half>   residualGm_;
    GlobalTensor<half>   normedGm_;
    GlobalTensor<half>   inLnGm_;
    GlobalTensor<int8_t> wqI8Gm_;
    GlobalTensor<half>   wqScaleGm_;
    GlobalTensor<int8_t> wkI8Gm_;
    GlobalTensor<half>   wkScaleGm_;
    GlobalTensor<int8_t> wvI8Gm_;
    GlobalTensor<half>   wvScaleGm_;
    GlobalTensor<int8_t> woI8Gm_;
    GlobalTensor<half>   woScaleGm_;
    GlobalTensor<half>   qNormGm_;
    GlobalTensor<half>   kNormGm_;
    GlobalTensor<half>   ropeCosGm_;
    GlobalTensor<half>   ropeSinGm_;
    GlobalTensor<half>   kCacheGm_;
    GlobalTensor<half>   vCacheGm_;
    GlobalTensor<half>   oOutGm_;

    uint32_t seqLen_ = 0;
    uint32_t opts_   = 0;
    float    epsF32_ = 1e-6f;
};

// =============================================================================
// Launcher entry point.
// =============================================================================

extern "C" __global__ __aicore__ void fused_attn_sublayer(
    GM_ADDR residual, GM_ADDR normed, GM_ADDR in_ln_gamma,
    GM_ADDR wq_i8, GM_ADDR wq_scale,
    GM_ADDR wk_i8, GM_ADDR wk_scale,
    GM_ADDR wv_i8, GM_ADDR wv_scale,
    GM_ADDR wo_i8, GM_ADDR wo_scale,
    GM_ADDR q_norm_f16, GM_ADDR k_norm_f16,
    GM_ADDR rope_cos, GM_ADDR rope_sin,
    GM_ADDR k_cache, GM_ADDR v_cache,
    GM_ADDR o_out, GM_ADDR scratch_q, GM_ADDR scratch_scores,
    uint32_t seq_len, uint32_t eps_bits, uint32_t opts)
{
    KernelFusedAttnSublayer op;
    op.Init(residual, normed, in_ln_gamma,
            wq_i8, wq_scale, wk_i8, wk_scale, wv_i8, wv_scale,
            wo_i8, wo_scale, q_norm_f16, k_norm_f16,
            rope_cos, rope_sin, k_cache, v_cache,
            o_out, scratch_q, scratch_scores,
            seq_len, eps_bits, opts);
    op.Process();
}
