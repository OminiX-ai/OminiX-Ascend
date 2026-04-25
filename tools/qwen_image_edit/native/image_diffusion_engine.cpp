// ============================================================================
// ImageDiffusionEngine — Phase 2.1 body (Q4-resident load path).
//
// Phase 1  (a50d0174): scaffold (constructor / dtor / device open).
// Phase 2  (0f860c5b): preload-dequant to F16 — RED on ac03, OOM at 26.3 GiB
//                      mid-upload (docs/qie_q2_p2_smoke.md). The published
//                      Qwen-Image-Edit-2509-Q4_0 GGUF contains 1933 tensors
//                      totalling 40.86 GB F16-equiv, not the ~13 GB the
//                      contract originally budgeted.
// Phase 2.1 (this file after Q2.1 rewrite): keep Q4_0 tensors COMPRESSED on
//                      device. Each big Linear weight becomes a pair:
//                        - INT4 packed buffer, shape [K, N] K-contiguous
//                          (K*N/2 bytes), per `docs/qie_q2_q4resident_probe.md`
//                          "transposeB-style view" expected by WQBMMv3.
//                        - F16 per-group scale buffer, shape [K/32, N], with
//                          group size 32 = Q4_0 block size exactly.
//                      Forward-path matmul (Phase 2) dispatches via
//                      aclnnWeightQuantBatchMatmulV3 with
//                      antiquantGroupSize=32, same dispatch shape as TTS's
//                      w8_matmul_ but per-group instead of per-channel.
//
// Offset compensation (probe doc §"First-run trap", Option 1 — nibble XOR):
//   GGUF Q4_0 stores each nibble as an unsigned bias-8 integer in [0..15];
//   the dequant formula is `(u - 8) * d`. WQBMMv3 with
//   antiquantOffsetOptional=nullptr interprets the nibble as SIGNED
//   two's-complement in [-8, +7]. Repacking does `s = u - 8`, encoded as
//   `s & 0x0f`, equivalent to `u ^ 0x08`. The probe verified this path
//   (cos_sim 0.999 vs CPU dequant reference); we must match that encoding
//   here or we get the -0.56 cos_sim regression the probe documented.
//
// HBM budget (probe doc §"Notes for Phase 1 scoping"):
//   W4 weights: 40.86 GiB / 8 = 5.11 GiB resident
//   Scales:     40.86 GiB / 16 = ~1.27 GiB worst case (F16 one per 32 elts)
//               — but we only emit scales for tensors that arrive as Q4_0;
//               non-Q4 tensors stay single-F16-buffer (no scale) so true
//               scale total is strictly the DiT matmul subset
//   Norms/biases F16: ~80 MiB
//   Scratch: ~2 GiB
//   Expected peak: < 9 GiB (Phase 1 smoke gate).
//
// Non-Q4 tensors (biases, 1D RMSNorm gammas, any test GGUF that ships
// weights as F16/F32 instead of Q4_0) fall through to the existing
// dequant-to-F16 upload path. The resulting buffer lives in the same
// `_w_q4` pointer as its Q4 sibling would; its `_scale` companion stays
// null so forward-path dispatch can detect F16-fallback via scale==null
// and route to plain aclnnMm.
//
// Tensor name convention: diffusers-style (unchanged from Phase 2)
//   "transformer_blocks.<i>.attn.to_q.weight"
//   "transformer_blocks.<i>.img_mlp.net.0.proj.weight"
//   ...
// LayerNorm affine=false on img/txt_norm1/2 per qwen_image.hpp — tolerated.
// ============================================================================

#include "image_diffusion_engine.h"

#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

namespace ominix_qie {

// ---------------------------------------------------------------------------
// Logging + ACL error handling — same shape as TalkerCannEngine macros so
// ac03 smoke logs line up between engines.
// ---------------------------------------------------------------------------
#define QIE_LOG(fmt, ...) \
    fprintf(stderr, "[qie_native] " fmt "\n", ##__VA_ARGS__)

#define QIE_ACL_CHECK(expr)                                                 \
    do {                                                                     \
        aclError _err = (expr);                                              \
        if (_err != 0) {                                                     \
            QIE_LOG("ACL call failed at %s:%d err=%d (%s)",                  \
                    __FILE__, __LINE__, (int)_err,                           \
                    g_cann.aclGetRecentErrMsg                                \
                        ? g_cann.aclGetRecentErrMsg() : "<n/a>");            \
            return false;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// F32 <-> F16 (arm64 hardware-native __fp16). Same primitive
// TalkerCannEngine uses — ac03 is aarch64.
// ---------------------------------------------------------------------------
namespace {

inline uint16_t fp32_to_fp16(float x) {
    __fp16 h = (__fp16)x;
    uint16_t out;
    std::memcpy(&out, &h, sizeof(out));
    return out;
}

// ---------------------------------------------------------------------------
// GGUF tensor names for the Qwen-Image-2511 DiT carry a
// `model.diffusion_model.` prefix in the original (pre-`name_conversion`)
// exporter — see tools/ominix_diffusion/src/model.cpp:1057. Some re-exports
// strip the prefix. We look up both forms so the native engine is robust
// to either GGUF vintage.
// ---------------------------------------------------------------------------
ggml_tensor *get_gguf_tensor_flex(ggml_context *ggml_ctx, const char *name) {
    ggml_tensor *t = ggml_get_tensor(ggml_ctx, name);
    if (t) return t;
    char buf[192];
    snprintf(buf, sizeof(buf), "model.diffusion_model.%s", name);
    return ggml_get_tensor(ggml_ctx, buf);
}

// ---------------------------------------------------------------------------
// Read a GGUF tensor as host F32 using ggml's type-traits to_float. This is
// the "dequant once" step of the preload-dequant fix — Q4_1 / Q5_K / Q8_0 /
// F16 / F32 all route through here and emerge as F32 on host. Caller
// immediately bit-converts to F16 and uploads (matmul) or keeps F32
// (norm gammas).
// ---------------------------------------------------------------------------
bool load_gguf_tensor_f32(ggml_context *ggml_ctx,
                          const char *name,
                          size_t expected_elems,
                          std::vector<float> &out_host) {
    ggml_tensor *t = get_gguf_tensor_flex(ggml_ctx, name);
    if (!t) {
        return false;  // caller decides whether missing is fatal
    }
    size_t n = ggml_nelements(t);
    if (expected_elems > 0 && n != expected_elems) {
        QIE_LOG("%s: elem-count mismatch expected=%zu got=%zu",
                name, expected_elems, n);
        return false;
    }
    out_host.assign(n, 0.0f);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(out_host.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < n; ++i) out_host[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            QIE_LOG("%s: unsupported dtype %d (no to_float trait)",
                    name, (int)t->type);
            return false;
        }
        tr->to_float(t->data, out_host.data(), (int64_t)n);
    }
    return true;
}

// ---------------------------------------------------------------------------
// Upload host F32 as F16 to a freshly allocated device buffer. Primary
// matmul-weight path. Bumps the stats counters for the Phase-2 receipts line.
// ---------------------------------------------------------------------------
bool upload_f32_as_f16(const float *host, size_t n, void *&dev,
                      size_t &stats_bytes, int64_t &stats_count) {
    if (!host || n == 0) return false;
    std::vector<uint16_t> f16buf(n);
    for (size_t i = 0; i < n; ++i) f16buf[i] = fp32_to_fp16(host[i]);
    const size_t bytes = n * sizeof(uint16_t);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        QIE_LOG("aclrtMalloc(%zu) failed err=%d", bytes, (int)err);
        dev = nullptr;
        return false;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, f16buf.data(), bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        QIE_LOG("aclrtMemcpy(H2D, %zu) failed err=%d", bytes, (int)err);
        g_cann.aclrtFree(dev); dev = nullptr;
        return false;
    }
    stats_bytes += bytes;
    stats_count += 1;
    return true;
}

bool upload_f32(const float *host, size_t n, void *&dev,
                size_t &stats_bytes, int64_t &stats_count) {
    if (!host || n == 0) return false;
    const size_t bytes = n * sizeof(float);
    aclError err = g_cann.aclrtMalloc(&dev, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        QIE_LOG("aclrtMalloc(%zu) failed err=%d", bytes, (int)err);
        dev = nullptr;
        return false;
    }
    err = g_cann.aclrtMemcpy(dev, bytes, host, bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        QIE_LOG("aclrtMemcpy(H2D, %zu) failed err=%d", bytes, (int)err);
        g_cann.aclrtFree(dev); dev = nullptr;
        return false;
    }
    stats_bytes += bytes;
    stats_count += 1;
    return true;
}

// ---------------------------------------------------------------------------
// Combined "fetch F32 + upload F16" and "fetch F32 + upload F32" — used
// everywhere in init_from_gguf below.
// ---------------------------------------------------------------------------
bool dequant_upload_f16(ggml_context *ggml_ctx, const char *name,
                       size_t expected_elems, void *&dev,
                       size_t &stats_bytes, int64_t &stats_count,
                       double &dequant_ms) {
    std::vector<float> host;
    auto t0 = std::chrono::steady_clock::now();
    if (!load_gguf_tensor_f32(ggml_ctx, name, expected_elems, host))
        return false;
    auto t1 = std::chrono::steady_clock::now();
    dequant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return upload_f32_as_f16(host.data(), host.size(), dev,
                             stats_bytes, stats_count);
}

bool dequant_upload_f32(ggml_context *ggml_ctx, const char *name,
                       size_t expected_elems, void *&dev,
                       size_t &stats_bytes, int64_t &stats_count,
                       double &dequant_ms) {
    std::vector<float> host;
    auto t0 = std::chrono::steady_clock::now();
    if (!load_gguf_tensor_f32(ggml_ctx, name, expected_elems, host))
        return false;
    auto t1 = std::chrono::steady_clock::now();
    dequant_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();
    return upload_f32(host.data(), host.size(), dev,
                      stats_bytes, stats_count);
}

// Tolerant F16 upload: missing tensor → returns true with dev=nullptr. Used
// for LayerNorm affine-off gammas/betas (qwen_image.hpp:205-213).
bool try_dequant_upload_f16(ggml_context *ggml_ctx, const char *name,
                            size_t expected_elems, void *&dev,
                            size_t &stats_bytes, int64_t &stats_count,
                            double &dequant_ms) {
    if (!get_gguf_tensor_flex(ggml_ctx, name)) {
        dev = nullptr;
        return true;
    }
    return dequant_upload_f16(ggml_ctx, name, expected_elems, dev,
                              stats_bytes, stats_count, dequant_ms);
}

// ---------------------------------------------------------------------------
// Q2.1: Re-tile a Q4_0 GGUF tensor into WQBMMv3's weight-layout convention.
//
// GGUF Q4_0 row layout (per `block_q4_0` in ggml/src/ggml-common.h:170-174
// and `dequantize_row_q4_0` in ggml/src/ggml-quants.c:307-325):
//   A weight matrix of shape [N=ne[1], K=ne[0]] is stored row-by-row, one row
//   per output channel. Each row of K elements is split into K/32 blocks; each
//   block occupies 18 bytes — 2 bytes ggml_half scale `d` followed by 16 bytes
//   of nibble quants `qs[0..15]`. Inside a block, `qs[j]` low nibble holds
//   element at intra-block index `j` (range 0..15) and `qs[j]` high nibble
//   holds the element at `j + 16`. Each nibble is an UNSIGNED bias-8 integer
//   `u ∈ [0..15]`, dequantised as `(u - 8) * d`.
//
// WQBMMv3 weight view (per `docs/qie_q2_q4resident_probe.md` §"Probe spec"):
//   shape [K, N], strides (1, K). Storage is `K*N` INT4 elements packed two
//   per byte. For weight `w[k, n]`, the nibble's linear index is
//   `lin = n * K + k`; `lin` even → low nibble of byte `lin/2`; `lin` odd →
//   high nibble. Each nibble is SIGNED two's-complement 4-bit in [-8, +7] —
//   symmetric quant with `antiquantOffsetOptional=nullptr`.
//
// Scale view (per probe): shape [K/32, N], strides (N, 1) row-major. One F16
// scale per (block_index, output_column).
//
// This helper reads the GGUF tensor's raw bytes (NO ggml dequant pass), emits
// two host buffers in the layouts above, and uploads both to fresh NPU
// device allocations. The offset compensation (`u ^ 0x08`) is applied at
// repack time — the probe RED'd its first run using the bias-8 encoding
// directly with `antiquantOffsetOptional=nullptr` and got cos_sim = -0.556.
// ---------------------------------------------------------------------------
bool repack_q4_0_upload(ggml_tensor *t,
                        int64_t expected_K, int64_t expected_N,
                        void *&w_dev, void *&scale_dev,
                        size_t &stats_w_bytes, size_t &stats_scale_bytes,
                        int64_t &stats_q4_tensors,
                        int64_t &stats_count,
                        double &repack_ms) {
    // Q4_0 per ggml-common.h:170
    constexpr int64_t QK4_0 = 32;
    if (t->type != GGML_TYPE_Q4_0) return false;

    // ggml Q4_0 tensor layout: ne[0] is the blocked (K) dim — it's the dim
    // `ggml_row_size` requires to be divisible by 32. ne[1] is the outer (N)
    // dim stored row-by-row. Reject anything that doesn't match this contract.
    const int64_t K = t->ne[0];
    const int64_t N = t->ne[1];
    if (K != expected_K || N != expected_N) {
        QIE_LOG("%s: Q4_0 shape mismatch got [K=%lld, N=%lld] "
                "expected [K=%lld, N=%lld]",
                t->name, (long long)K, (long long)N,
                (long long)expected_K, (long long)expected_N);
        return false;
    }
    if (K % QK4_0 != 0) {
        QIE_LOG("%s: Q4_0 K=%lld not divisible by 32", t->name, (long long)K);
        return false;
    }
    const int64_t BLK = K / QK4_0;             // blocks per row
    const size_t  BLK_BYTES = 2 /*scale*/ + 16 /*qs*/;

    auto t0 = std::chrono::steady_clock::now();

    // Host output: packed nibbles K*N/2 bytes + F16 scales [BLK, N].
    std::vector<uint8_t>  out_w((size_t)K * N / 2, 0);
    std::vector<uint16_t> out_s((size_t)BLK * N, 0);

    const uint8_t *src = (const uint8_t *)t->data;

    // Re-tile row-by-row. For each output-channel n:
    //   for each block b:
    //     scale_f16 = row[b].d
    //     for each intra-block pair j (0..15):
    //       u_lo = qs[j] & 0x0f    → element k0 = b*32 + j
    //       u_hi = qs[j] >> 4      → element k1 = b*32 + j + 16
    //       signed s_lo = u_lo - 8 → packed byte at (n*K + k0)/2
    //       signed s_hi = u_hi - 8 → packed byte at (n*K + k1)/2
    //
    // We apply `u ^ 0x08` (≡ subtract 8 mod 16) to flip the bias-8 encoding
    // into signed two's-complement 4-bit.
    for (int64_t n = 0; n < N; ++n) {
        const uint8_t *row_src = src + (size_t)n * BLK * BLK_BYTES;
        uint8_t       *row_dst = out_w.data();       // packed output base
        const size_t   n_base_nib = (size_t)n * K;   // linear index of w[0, n]

        for (int64_t b = 0; b < BLK; ++b) {
            const uint8_t *blk = row_src + (size_t)b * BLK_BYTES;

            // Scale: little-endian ggml_half, two bytes at the start of block.
            uint16_t d;
            std::memcpy(&d, blk, sizeof(uint16_t));
            out_s[(size_t)b * N + n] = d;

            const uint8_t *qs = blk + 2;
            const size_t block_nib_base = n_base_nib + (size_t)b * QK4_0;
            for (int64_t j = 0; j < QK4_0 / 2; ++j) {
                const uint8_t byte = qs[j];
                // `u ^ 0x08`: bias-8 unsigned → signed two's-complement nibble.
                const uint8_t s_lo = (uint8_t)((byte & 0x0f) ^ 0x08);
                const uint8_t s_hi = (uint8_t)(((byte >> 4) & 0x0f) ^ 0x08);

                // Element k0 = b*32 + j → linear index `block_nib_base + j`.
                const size_t lin_lo = block_nib_base + (size_t)j;
                // Element k1 = b*32 + j + 16.
                const size_t lin_hi = block_nib_base + (size_t)j + (QK4_0 / 2);

                auto write_nib = [&](size_t lin, uint8_t nib) {
                    uint8_t &b_out = row_dst[lin / 2];
                    if ((lin & 1u) == 0u) {
                        b_out = (uint8_t)((b_out & 0xf0) | (nib & 0x0fu));
                    } else {
                        b_out = (uint8_t)((b_out & 0x0f) | (uint8_t)(nib << 4));
                    }
                };
                write_nib(lin_lo, s_lo);
                write_nib(lin_hi, s_hi);
            }
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    repack_ms += std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Upload packed nibbles.
    const size_t w_bytes = out_w.size();
    aclError err = g_cann.aclrtMalloc(&w_dev, w_bytes,
                                       ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        QIE_LOG("%s: aclrtMalloc(q4 %zu) failed err=%d",
                t->name, w_bytes, (int)err);
        w_dev = nullptr;
        return false;
    }
    err = g_cann.aclrtMemcpy(w_dev, w_bytes, out_w.data(), w_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        QIE_LOG("%s: aclrtMemcpy(q4 H2D %zu) failed err=%d",
                t->name, w_bytes, (int)err);
        g_cann.aclrtFree(w_dev); w_dev = nullptr;
        return false;
    }

    // Upload per-group scales.
    const size_t s_bytes = out_s.size() * sizeof(uint16_t);
    err = g_cann.aclrtMalloc(&scale_dev, s_bytes,
                              ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        QIE_LOG("%s: aclrtMalloc(scale %zu) failed err=%d",
                t->name, s_bytes, (int)err);
        g_cann.aclrtFree(w_dev); w_dev = nullptr;
        scale_dev = nullptr;
        return false;
    }
    err = g_cann.aclrtMemcpy(scale_dev, s_bytes, out_s.data(), s_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE);
    if (err != 0) {
        QIE_LOG("%s: aclrtMemcpy(scale H2D %zu) failed err=%d",
                t->name, s_bytes, (int)err);
        g_cann.aclrtFree(w_dev);     w_dev = nullptr;
        g_cann.aclrtFree(scale_dev); scale_dev = nullptr;
        return false;
    }

    stats_w_bytes     += w_bytes;
    stats_scale_bytes += s_bytes;
    stats_q4_tensors  += 1;
    stats_count       += 1;
    return true;
}

// ---------------------------------------------------------------------------
// Q2.1 matmul-weight entry point: if the GGUF tensor is Q4_0, keep it
// compressed on device via repack_q4_0_upload(); otherwise fall back to the
// existing dequant-to-F16 upload. Forward-path branches on `scale_dev != null`
// to decide WQBMMv3 vs aclnnMm dispatch.
//
// Shape convention: caller passes logical (K=in_features, N=out_features).
// Q4_0 tensors pack as [K, N] K-contiguous; F16 fallback keeps the original
// N*K element count — the forward path handles dtype-specific layout itself.
// ---------------------------------------------------------------------------
bool load_matmul_weight_upload(ggml_context *ggml_ctx, const char *name,
                               int64_t K, int64_t N,
                               void *&w_dev, void *&scale_dev,
                               DiTInitStats &stats,
                               double &load_ms) {
    ggml_tensor *t = get_gguf_tensor_flex(ggml_ctx, name);
    if (!t) {
        QIE_LOG("%s: not found in GGUF", name);
        return false;
    }
    if (t->type == GGML_TYPE_Q4_0) {
        return repack_q4_0_upload(t, K, N, w_dev, scale_dev,
                                   stats.q4_weight_bytes,
                                   stats.q4_scale_bytes,
                                   stats.q4_tensors,
                                   stats.tensors_uploaded,
                                   load_ms);
    }
    // Fallback: dequant-to-F16 path. `scale_dev` stays null so forward can
    // branch on scale-null → aclnnMm / scale-non-null → WQBMMv3.
    scale_dev = nullptr;
    const size_t expected = (size_t)K * N;
    stats.f16_fallback_tensors += 1;
    return dequant_upload_f16(ggml_ctx, name, expected, w_dev,
                              stats.f16_weight_bytes,
                              stats.tensors_uploaded, load_ms);
}

// ---------------------------------------------------------------------------
// 3D axial RoPE pre-compute (Q0.5.3 verdict: retire V2 packed layout; MLX
// numbers prefer pre-compute tables).
//
// Layout (matches Qwen::Rope::apply_rope contract at rope.hpp:603):
//   pe[pos, d_pair, row, col]  with shape [seq, head_dim/2, 2, 2]
//   pe[pos, d_pair, 0, :] = [ cos(θ), -sin(θ) ]
//   pe[pos, d_pair, 1, :] = [ sin(θ),  cos(θ) ]
// where θ = position_on_axis × 1 / theta^(2·d_pair_on_axis / axis_dim).
//
// Axial assignment: d_pair indexes the full head_dim/2. First
// rope_axes_temporal/2 pairs use the temporal position; next
// rope_axes_h/2 pairs use the H position; last rope_axes_w/2 use the W
// position. With axes = {16, 56, 56} that's {0..7 → T, 8..35 → H, 36..63 → W}
// → 64 pairs = head_dim/2 for head_dim=128. ✓ contract.
//
// Ref-latents are concatenated onto the img stream at model entry
// (qwen_image.hpp:454-459) with the same (t=ref_index, h, w) id scheme
// `Rope::gen_refs_ids`. Our table only needs to span the MAX position we
// ever index — we lay out [0, max_total_seq) in row-major order of
// (position_id) and trust the caller to slice by actual seq length.
//
// For the txt stream (`gen_qwen_image_ids`): txt_ids use t=h=w=same
// linspace value. The apply_rope kernel uses the same pe table indexed
// by sequence position — the txt positions happen to be "diagonal"
// (all three axes equal) so per-pair angle is identity-safe-ish but
// NOT exactly cos=1/sin=0. To preserve reference parity we actually
// compute the correct angles here.
//
// Phase 2 builds the tables for a fixed worst-case layout:
//   context_len = max_txt_seq (256)
//   h = w = sqrt(max_img_seq) assumed (4096 → 64x64 patch grid)
//   no ref_latents at build time — the Q1 baseline at 256×256 uses a
//   small patch grid and the ref is concatenated onto the img stream
//   (so the pe index for ref tokens extends past img_tokens).
// Phase 3 NOTE-TO-AGENT: Qwen-Image-Edit actually concatenates ref latents
// onto the img stream at `forward` entry (qwen_image.hpp:454-459). The
// ref tokens use `gen_refs_ids` with `index=1,2,...` on the temporal axis
// — so pe index for ref-tokens needs its own entries beyond the `img`
// block. Phase 2 skips this: the Q1 integration smoke only has a single
// ref latent of the same resolution as the edit target, giving total pos
// = ctx_len + 2*img_tokens. Adjust the RoPE size / layout when the
// session-rebuild hook lands in Phase 3. Track this via a Q2.5-ref-rope
// TODO — see final report.
// ---------------------------------------------------------------------------
// Phase 4.1 NOTE: this helper now additionally emits two flat F16 tables
// (`cos_f16_out`, `sin_f16_out`) of shape [total_pos, head_dim/2]. These
// feed the on-device interleaved-RoPE path directly (no strided-view gymnastics
// over the packed pe [seq, hd/2, 2, 2] layout). If callers pass null for
// the flat outputs we skip the separated emission.
void compute_qwen_rope_pe_host(const ImageDiffusionConfig &cfg,
                               std::vector<uint16_t> &pe_f16_out,
                               int64_t &total_pos_out,
                               std::vector<uint16_t> *cos_f16_out = nullptr,
                               std::vector<uint16_t> *sin_f16_out = nullptr) {
    const int axes_t = cfg.rope_axes_temporal;  // 16
    const int axes_h = cfg.rope_axes_h;         // 56
    const int axes_w = cfg.rope_axes_w;         // 56
    const int head_dim = cfg.head_dim;          // 128
    const int half = head_dim / 2;              // 64 pairs
    (void)half;
    assert((axes_t + axes_h + axes_w) == head_dim &&
           "axes_dim sum must match head_dim");

    // Patch grid corresponding to max_img_seq. For the default 4096 tokens
    // we pick h=w=sqrt(4096)=64 which gives 64*64=4096 patches. Caller can
    // override via Q2.5+ session rebuild hook.
    int h_len = (int)std::lround(std::sqrt((double)cfg.max_img_seq));
    int w_len = h_len;
    while (h_len * w_len < cfg.max_img_seq) { ++h_len; }
    const int img_tokens = h_len * w_len;
    const int ctx_len   = cfg.max_txt_seq;
    const int txt_start = std::max(h_len, w_len);  // Qwen-Image txt id start

    total_pos_out = (int64_t)ctx_len + img_tokens;
    pe_f16_out.assign((size_t)total_pos_out * head_dim / 2 * 2 * 2, 0);
    if (cos_f16_out) cos_f16_out->assign((size_t)total_pos_out * head_dim / 2, 0);
    if (sin_f16_out) sin_f16_out->assign((size_t)total_pos_out * head_dim / 2, 0);

    auto pe_set = [&](int64_t pos, int64_t dpair,
                      float cos_v, float sin_v) {
        const size_t base =
            ((size_t)pos * head_dim / 2 + (size_t)dpair) * 4;
        pe_f16_out[base + 0] = fp32_to_fp16(cos_v);
        pe_f16_out[base + 1] = fp32_to_fp16(-sin_v);
        pe_f16_out[base + 2] = fp32_to_fp16(sin_v);
        pe_f16_out[base + 3] = fp32_to_fp16(cos_v);
        if (cos_f16_out) {
            const size_t flat = (size_t)pos * head_dim / 2 + (size_t)dpair;
            (*cos_f16_out)[flat] = fp32_to_fp16(cos_v);
        }
        if (sin_f16_out) {
            const size_t flat = (size_t)pos * head_dim / 2 + (size_t)dpair;
            (*sin_f16_out)[flat] = fp32_to_fp16(sin_v);
        }
    };

    // Per-axis 1/theta^(2d / axis_dim) scale via the `linspace(0, 2-2/axis_dim)`
    // / theta^{·} convention (rope.hpp:51-56). Each axis contributes
    // axis_dim/2 pairs sequentially in the head_dim/2 packing.
    auto axis_omega = [&](int axis_dim, std::vector<float> &omega) {
        // linspace(0, (d-2)/d, d/2) followed by 1 / theta^scale — matches
        // Rope::rope / Rope::linspace in rope.hpp lines 44-56.
        const int half_axis = axis_dim / 2;
        omega.assign(half_axis, 0.0f);
        if (half_axis == 0) return;
        if (half_axis == 1) { omega[0] = 1.0f; return; }
        const float end_scale = (axis_dim - 2.0f) / (float)axis_dim;
        for (int i = 0; i < half_axis; ++i) {
            const float scale = end_scale * (float)i / (float)(half_axis - 1);
            omega[i] = 1.0f / std::pow((float)cfg.rope_theta, scale);
        }
    };
    std::vector<float> omega_t, omega_h, omega_w;
    axis_omega(axes_t, omega_t);
    axis_omega(axes_h, omega_h);
    axis_omega(axes_w, omega_w);

    // Fill txt positions [0 .. ctx_len) — positions are diagonal (t=h=w).
    for (int i = 0; i < ctx_len; ++i) {
        const float p = (float)(txt_start + i);
        int64_t pos = i;
        int64_t dp = 0;
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

    // Fill img positions: (t=0, h=row_id, w=col_id) per `gen_flux_img_ids`
    // and `gen_qwen_image_ids` in rope.hpp. scale_rope=true in the qwen
    // image path → h_start=-h_len/2, w_start=-w_len/2.
    const int h_start = -h_len / 2;
    const int w_start = -w_len / 2;
    for (int r = 0; r < h_len; ++r) {
        const float h_id = (float)(h_start + r);
        for (int c = 0; c < w_len; ++c) {
            const float w_id = (float)(w_start + c);
            const int64_t pos = (int64_t)ctx_len + r * w_len + c;
            if (pos >= total_pos_out) break;
            int64_t dp = 0;
            const float t_id = 0.0f;  // non-ref tokens use t=0
            for (int j = 0; j < (int)omega_t.size(); ++j, ++dp) {
                float a = t_id * omega_t[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
            for (int j = 0; j < (int)omega_h.size(); ++j, ++dp) {
                float a = h_id * omega_h[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
            for (int j = 0; j < (int)omega_w.size(); ++j, ++dp) {
                float a = w_id * omega_w[j];
                pe_set(pos, dp, std::cos(a), std::sin(a));
            }
        }
    }
}

}  // namespace

// ---------------------------------------------------------------------------
// dtor — unchanged from Phase 1; just ensures `ready_` now matters.
// ---------------------------------------------------------------------------
ImageDiffusionEngine::~ImageDiffusionEngine() {
    if (!cp_cann_load_symbols()) return;

    auto free_dev = [](void *&p) {
        if (p) { g_cann.aclrtFree(p); p = nullptr; }
    };

    for (auto &lw : layer_w_) {
        // Attention img-side: (w_q4, scale, b) per projection.
        free_dev(lw.to_q_w_q4);      free_dev(lw.to_q_scale);      free_dev(lw.to_q_b);
        free_dev(lw.to_k_w_q4);      free_dev(lw.to_k_scale);      free_dev(lw.to_k_b);
        free_dev(lw.to_v_w_q4);      free_dev(lw.to_v_scale);      free_dev(lw.to_v_b);
        free_dev(lw.to_out_0_w_q4);  free_dev(lw.to_out_0_scale);  free_dev(lw.to_out_0_b);
        // Attention txt-side.
        free_dev(lw.add_q_w_q4);     free_dev(lw.add_q_scale);     free_dev(lw.add_q_b);
        free_dev(lw.add_k_w_q4);     free_dev(lw.add_k_scale);     free_dev(lw.add_k_b);
        free_dev(lw.add_v_w_q4);     free_dev(lw.add_v_scale);     free_dev(lw.add_v_b);
        free_dev(lw.to_add_out_w_q4); free_dev(lw.to_add_out_scale); free_dev(lw.to_add_out_b);
        // Norm gammas (1D, no scale pair).
        free_dev(lw.norm_q_w);       free_dev(lw.norm_k_w);
        free_dev(lw.norm_added_q_w); free_dev(lw.norm_added_k_w);
        free_dev(lw.img_norm1_w);    free_dev(lw.img_norm1_b);
        free_dev(lw.img_norm2_w);    free_dev(lw.img_norm2_b);
        free_dev(lw.txt_norm1_w);    free_dev(lw.txt_norm1_b);
        free_dev(lw.txt_norm2_w);    free_dev(lw.txt_norm2_b);
        // Modulation heads.
        free_dev(lw.img_mod_w_q4);   free_dev(lw.img_mod_scale);   free_dev(lw.img_mod_b);
        free_dev(lw.txt_mod_w_q4);   free_dev(lw.txt_mod_scale);   free_dev(lw.txt_mod_b);
        // FFN up/down.
        free_dev(lw.img_ff_up_w_q4);   free_dev(lw.img_ff_up_scale);   free_dev(lw.img_ff_up_b);
        free_dev(lw.img_ff_down_w_q4); free_dev(lw.img_ff_down_scale); free_dev(lw.img_ff_down_b);
        free_dev(lw.txt_ff_up_w_q4);   free_dev(lw.txt_ff_up_scale);   free_dev(lw.txt_ff_up_b);
        free_dev(lw.txt_ff_down_w_q4); free_dev(lw.txt_ff_down_scale); free_dev(lw.txt_ff_down_b);
    }
    layer_w_.clear();

    free_dev(global_w_.time_linear1_w_q4);   free_dev(global_w_.time_linear1_scale);
    free_dev(global_w_.time_linear1_b);
    free_dev(global_w_.time_linear2_w_q4);   free_dev(global_w_.time_linear2_scale);
    free_dev(global_w_.time_linear2_b);
    free_dev(global_w_.img_in_w_q4);         free_dev(global_w_.img_in_scale);
    free_dev(global_w_.img_in_b);
    free_dev(global_w_.txt_in_w_q4);         free_dev(global_w_.txt_in_scale);
    free_dev(global_w_.txt_in_b);
    free_dev(global_w_.txt_norm_w);
    free_dev(global_w_.norm_out_linear_w_q4); free_dev(global_w_.norm_out_linear_scale);
    free_dev(global_w_.norm_out_linear_b);
    free_dev(global_w_.proj_out_w_q4);        free_dev(global_w_.proj_out_scale);
    free_dev(global_w_.proj_out_b);
    free_dev(global_w_.rope_pe_dev);
    free_dev(global_w_.rope_cos_dev);
    free_dev(global_w_.rope_sin_dev);

    free_dev(scratch_q_dev_);    free_dev(scratch_k_dev_);
    free_dev(scratch_v_dev_);    free_dev(scratch_attn_dev_);
    free_dev(scratch_mlp_dev_);  free_dev(scratch_mod_dev_);
    free_dev(scratch_rope_a_dev_); free_dev(scratch_rope_b_dev_);
    free_dev(scratch_rope_c_dev_);
    free_dev(scratch_rope_cos_bcast_dev_);
    free_dev(scratch_rope_sin_bcast_dev_);
    free_dev(scratch_rope_cos_full_dev_);
    free_dev(scratch_rope_sin_full_dev_);
    free_dev(rstd_dev_);
    free_dev(scratch_img_norm_dev_);  free_dev(scratch_txt_norm_dev_);
    free_dev(scratch_img_out_dev_);   free_dev(scratch_txt_out_dev_);
    free_dev(mean_dev_);              free_dev(ln_rstd_dev_);
    free_dev(scratch_img_hidden_f16_dev_);
    free_dev(scratch_txt_hidden_f16_dev_);
    free_dev(scratch_residual_tmp_f32_dev_);
    free_dev(img_hidden_cond_dev_);   free_dev(img_hidden_uncond_dev_);
    free_dev(txt_hidden_cond_dev_);   free_dev(txt_hidden_uncond_dev_);
    free_dev(workspace_dev_);
    // Q2.4.5.4c BF16 plumbing scratch (lazy; may be null if env not set).
    free_dev(scratch_bf16_scale_dev_);    scratch_bf16_scale_bytes_ = 0;
    free_dev(scratch_bf16_bias_dev_);     scratch_bf16_bias_bytes_  = 0;
    free_dev(scratch_bf16_src_f32_dev_);  scratch_bf16_src_f32_bytes_ = 0;

    if (primary_stream_) {
        g_cann.aclrtDestroyStream(primary_stream_);
        primary_stream_ = nullptr;
    }
    compute_stream_ = nullptr;
    ready_ = false;
}

// ---------------------------------------------------------------------------
// Phase 2 init: open device, parse GGUF, dequant + upload every weight
// listed in DiTLayerWeights + DiTGlobalWeights, precompute RoPE tables,
// allocate scratch buffers. Returns true with ready_=true on full success.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::init_from_gguf(const std::string &gguf_path,
                                           const ImageDiffusionConfig &cfg,
                                           int device) {
    auto t_init0 = std::chrono::steady_clock::now();
    if (!cp_cann_load_symbols()) {
        QIE_LOG("symbol load failed; engine disabled");
        return false;
    }

    cfg_    = cfg;
    device_ = device;
    QIE_ACL_CHECK(g_cann.aclrtSetDevice(device_));
    QIE_ACL_CHECK(g_cann.aclrtCreateStream(&primary_stream_));
    compute_stream_ = primary_stream_;

    layer_w_.clear();
    layer_w_.resize(cfg_.num_layers);

    // ------------------------------------------------------------------
    // GGUF open — standard ggml path, same as TalkerCannEngine.
    // ------------------------------------------------------------------
    ggml_context *ggml_ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ggml_ctx;
    gguf_context *gguf_ctx = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gguf_ctx || !ggml_ctx) {
        QIE_LOG("failed to load GGUF: %s", gguf_path.c_str());
        return false;
    }

    const int64_t H       = cfg_.hidden_size;               // 3072
    const int64_t HEAD_D  = cfg_.head_dim;                  // 128
    const int64_t FF_DIM  = (int64_t)H * cfg_.ff_mult;      // 12288
    const int64_t JD      = cfg_.joint_attention_dim;       // 3584
    const int64_t PATCH_IN = cfg_.in_channels;              // 64
    const int64_t PATCH_OUT = (int64_t)cfg_.patch_size *
                              cfg_.patch_size * cfg_.out_channels;  // 64

    double dequant_ms = 0.0;
    size_t &f16b = stats_.f16_weight_bytes;
    size_t &f32b = stats_.f32_weight_bytes;
    int64_t &tc  = stats_.tensors_uploaded;

    // ------------------------------------------------------------------
    // Per-layer weights. Tensor names match the diffusers convention the
    // CPU reference path consumes — see qwen_image.hpp block construction
    // + name_conversion.cpp (no remapping: GGUF produced via the
    // ominix_diffusion exporter keeps diffusers names).
    //
    // Q2.1 Q4-resident: matmul weights route through
    // load_matmul_weight_upload, which keeps Q4_0 tensors compressed on
    // device (INT4 buffer + F16 per-group-32 scale) or falls back to an
    // F16-dequant upload for non-Q4 source tensors. 1D tensors (biases,
    // RMSNorm gammas) keep the original dequant_upload_{f16,f32} path.
    //
    // Logical weight shape for every Linear is (K=in_features, N=out_features).
    // ------------------------------------------------------------------
    char name[128];
#define TNAME(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)

    for (int il = 0; il < cfg_.num_layers; ++il) {
        auto &lw = layer_w_[il];

        // --- attn.to_{q,k,v} projections + biases (img side) ---
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_q.weight", il),
                H, H, lw.to_q_w_q4, lw.to_q_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_q.bias", il),
                (size_t)H, lw.to_q_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_k.weight", il),
                H, H, lw.to_k_w_q4, lw.to_k_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_k.bias", il),
                (size_t)H, lw.to_k_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_v.weight", il),
                H, H, lw.to_v_w_q4, lw.to_v_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_v.bias", il),
                (size_t)H, lw.to_v_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_out.0.weight", il),
                H, H, lw.to_out_0_w_q4, lw.to_out_0_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_out.0.bias", il),
                (size_t)H, lw.to_out_0_b, f16b, tc, dequant_ms)) goto fail;

        // --- attn.add_{q,k,v}_proj + to_add_out (txt side) ---
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_q_proj.weight", il),
                H, H, lw.add_q_w_q4, lw.add_q_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_q_proj.bias", il),
                (size_t)H, lw.add_q_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_k_proj.weight", il),
                H, H, lw.add_k_w_q4, lw.add_k_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_k_proj.bias", il),
                (size_t)H, lw.add_k_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_v_proj.weight", il),
                H, H, lw.add_v_w_q4, lw.add_v_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_v_proj.bias", il),
                (size_t)H, lw.add_v_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_add_out.weight", il),
                H, H, lw.to_add_out_w_q4, lw.to_add_out_scale, stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_add_out.bias", il),
                (size_t)H, lw.to_add_out_b, f16b, tc, dequant_ms)) goto fail;

        // --- Q/K RMSNorm gammas (per-head_dim, F32 for aclnnRmsNorm) ---
        if (!dequant_upload_f32(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.norm_q.weight", il),
                (size_t)HEAD_D, lw.norm_q_w, f32b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f32(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.norm_k.weight", il),
                (size_t)HEAD_D, lw.norm_k_w, f32b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f32(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.norm_added_q.weight", il),
                (size_t)HEAD_D, lw.norm_added_q_w, f32b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f32(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.norm_added_k.weight", il),
                (size_t)HEAD_D, lw.norm_added_k_w, f32b, tc, dequant_ms)) goto fail;

        // --- Block-level LayerNorm gammas/betas (may be absent: affine=false) ---
        // qwen_image.hpp:205-213 constructs these with affine=false, so
        // typically the GGUF has NO tensor for them. We try-load anyway in
        // case an exporter ever flips affine on — forward path branches on
        // nullptr to skip the multiply-add.
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_norm1.weight", il),
                (size_t)H, lw.img_norm1_w, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_norm1.bias", il),
                (size_t)H, lw.img_norm1_b, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_norm2.weight", il),
                (size_t)H, lw.img_norm2_w, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_norm2.bias", il),
                (size_t)H, lw.img_norm2_b, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_norm1.weight", il),
                (size_t)H, lw.txt_norm1_w, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_norm1.bias", il),
                (size_t)H, lw.txt_norm1_b, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_norm2.weight", il),
                (size_t)H, lw.txt_norm2_w, f16b, tc, dequant_ms)) goto fail;
        if (!try_dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_norm2.bias", il),
                (size_t)H, lw.txt_norm2_b, f16b, tc, dequant_ms)) goto fail;

        // --- img_mod.1 / txt_mod.1 modulation heads (hidden → 6·hidden) ---
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mod.1.weight", il),
                H, 6 * H, lw.img_mod_w_q4, lw.img_mod_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mod.1.bias", il),
                (size_t)(6 * H), lw.img_mod_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mod.1.weight", il),
                H, 6 * H, lw.txt_mod_w_q4, lw.txt_mod_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mod.1.bias", il),
                (size_t)(6 * H), lw.txt_mod_b, f16b, tc, dequant_ms)) goto fail;

        // --- FFN (img_mlp + txt_mlp). FeedForward.net = [Linear, GELU,
        // Linear] so .net.0.proj.* (up: K=hidden, N=ff_dim) and .net.2.*
        // (down: K=ff_dim, N=hidden). NOT SwiGLU. GELU exact (not GELU-tanh)
        // per contract §Q2 / Part-2 §3.8.
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.0.proj.weight", il),
                H, FF_DIM, lw.img_ff_up_w_q4, lw.img_ff_up_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.0.proj.bias", il),
                (size_t)FF_DIM, lw.img_ff_up_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.2.weight", il),
                FF_DIM, H, lw.img_ff_down_w_q4, lw.img_ff_down_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.2.bias", il),
                (size_t)H, lw.img_ff_down_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.0.proj.weight", il),
                H, FF_DIM, lw.txt_ff_up_w_q4, lw.txt_ff_up_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.0.proj.bias", il),
                (size_t)FF_DIM, lw.txt_ff_up_b, f16b, tc, dequant_ms)) goto fail;
        if (!load_matmul_weight_upload(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.2.weight", il),
                FF_DIM, H, lw.txt_ff_down_w_q4, lw.txt_ff_down_scale,
                stats_, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.2.bias", il),
                (size_t)H, lw.txt_ff_down_b, f16b, tc, dequant_ms)) goto fail;
    }

    // ------------------------------------------------------------------
    // Global (non-per-layer) weights.
    // ------------------------------------------------------------------
    // time_text_embed.timestep_embedder.linear_1: Linear(256, hidden).
    //   Note: with K=256 and group=32, K/32=8 scale groups per column.
    if (!load_matmul_weight_upload(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_1.weight",
            256, H, global_w_.time_linear1_w_q4, global_w_.time_linear1_scale,
            stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_1.bias",
            (size_t)H, global_w_.time_linear1_b, f16b, tc, dequant_ms)) goto fail;
    if (!load_matmul_weight_upload(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_2.weight",
            H, H, global_w_.time_linear2_w_q4, global_w_.time_linear2_scale,
            stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_2.bias",
            (size_t)H, global_w_.time_linear2_b, f16b, tc, dequant_ms)) goto fail;

    // img_in / txt_in projections.
    //   img_in: Linear(patch_in=64, hidden)  → K=64 → only 2 scale groups/col.
    //   txt_in: Linear(joint_dim=3584, hidden).
    if (!load_matmul_weight_upload(ggml_ctx, "img_in.weight",
            PATCH_IN, H, global_w_.img_in_w_q4, global_w_.img_in_scale,
            stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "img_in.bias",
            (size_t)H, global_w_.img_in_b, f16b, tc, dequant_ms)) goto fail;
    if (!load_matmul_weight_upload(ggml_ctx, "txt_in.weight",
            JD, H, global_w_.txt_in_w_q4, global_w_.txt_in_scale,
            stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "txt_in.bias",
            (size_t)H, global_w_.txt_in_b, f16b, tc, dequant_ms)) goto fail;

    // txt_norm RMSNorm over joint_attention_dim (1D, stays F32).
    if (!dequant_upload_f32(ggml_ctx, "txt_norm.weight",
            (size_t)JD, global_w_.txt_norm_w, f32b, tc, dequant_ms)) goto fail;

    // norm_out.linear (AdaLayerNormContinuous): Linear(hidden, 2·hidden).
    if (!load_matmul_weight_upload(ggml_ctx, "norm_out.linear.weight",
            H, 2 * H, global_w_.norm_out_linear_w_q4,
            global_w_.norm_out_linear_scale, stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "norm_out.linear.bias",
            (size_t)(2 * H), global_w_.norm_out_linear_b,
            f16b, tc, dequant_ms)) goto fail;

    // proj_out: Linear(hidden, patch_size² · out_channels).
    if (!load_matmul_weight_upload(ggml_ctx, "proj_out.weight",
            H, PATCH_OUT, global_w_.proj_out_w_q4, global_w_.proj_out_scale,
            stats_, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "proj_out.bias",
            (size_t)PATCH_OUT, global_w_.proj_out_b,
            f16b, tc, dequant_ms)) goto fail;

#undef TNAME

    // GGUF/ggml context no longer needed.
    gguf_free(gguf_ctx); gguf_ctx = nullptr;
    ggml_free(ggml_ctx); ggml_ctx = nullptr;

    // ------------------------------------------------------------------
    // Pre-compute 3D axial RoPE tables (Q0.5.3 MUST-HAVE).
    // ------------------------------------------------------------------
    if (cfg_.precompute_rope) {
        std::vector<uint16_t> pe_host, cos_host, sin_host;
        int64_t total_pos = 0;
        auto t_r0 = std::chrono::steady_clock::now();
        compute_qwen_rope_pe_host(cfg_, pe_host, total_pos,
                                   &cos_host, &sin_host);
        auto t_r1 = std::chrono::steady_clock::now();
        stats_.dequant_wall_ms +=
            std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();

        const size_t pe_bytes  = pe_host.size()  * sizeof(uint16_t);
        const size_t cos_bytes = cos_host.size() * sizeof(uint16_t);
        const size_t sin_bytes = sin_host.size() * sizeof(uint16_t);
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_pe_dev, pe_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_pe_dev, pe_bytes,
                                          pe_host.data(), pe_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        // Phase 4.1 on-device RoPE tables (flat cos + sin).
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_cos_dev, cos_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_cos_dev, cos_bytes,
                                          cos_host.data(), cos_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_sin_dev, sin_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_sin_dev, sin_bytes,
                                          sin_host.data(), sin_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        global_w_.rope_total_pos = total_pos;
        stats_.rope_bytes = pe_bytes + cos_bytes + sin_bytes;
        QIE_LOG("rope pe: pos=%lld head_dim/2=%lld bytes=%zu "
                "(layout=[seq, hd/2, 2, 2] F16) + cos/sin flat %zu+%zu B",
                (long long)total_pos, (long long)(HEAD_D / 2), pe_bytes,
                cos_bytes, sin_bytes);

        // Pre-broadcast cos/sin over NH: explicit [total_pos, NH, half] tiles
        // to avoid ACL stride-0 broadcast numerical bugs in aclnnMul.
        {
            const int64_t half = cfg_.head_dim / 2;
            const int64_t NH = cfg_.num_heads;
            std::vector<uint16_t> cos_bcast((size_t)total_pos * NH * half, 0);
            std::vector<uint16_t> sin_bcast((size_t)total_pos * NH * half, 0);
            for (int64_t p = 0; p < total_pos; ++p) {
                for (int64_t h = 0; h < NH; ++h) {
                    const size_t src_off = (size_t)p * half;
                    const size_t dst_off =
                        ((size_t)p * NH + (size_t)h) * half;
                    std::memcpy(&cos_bcast[dst_off], &cos_host[src_off],
                                 (size_t)half * sizeof(uint16_t));
                    std::memcpy(&sin_bcast[dst_off], &sin_host[src_off],
                                 (size_t)half * sizeof(uint16_t));
                }
            }
            const size_t bc_bytes = cos_bcast.size() * sizeof(uint16_t);
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_cos_bcast_dev_,
                                              bc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_cos_bcast_dev_, bc_bytes,
                                              cos_bcast.data(), bc_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_sin_bcast_dev_,
                                              bc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_sin_bcast_dev_, bc_bytes,
                                              sin_bcast.data(), bc_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            stats_.rope_bytes += 2 * bc_bytes;
        }

        // Build "full" cos/sin tables for aclnnRotaryPositionEmbedding
        // interleave-mode dispatch. Shape [total_pos, head_dim] with each
        // pair's cos/sin duplicated across both elements of the pair.
        {
            const int64_t HD = cfg_.head_dim;
            const int64_t half = HD / 2;
            std::vector<uint16_t> cos_full((size_t)total_pos * HD, 0);
            std::vector<uint16_t> sin_full((size_t)total_pos * HD, 0);
            for (int64_t p = 0; p < total_pos; ++p) {
                for (int64_t dp = 0; dp < half; ++dp) {
                    uint16_t c = cos_host[(size_t)p * half + dp];
                    uint16_t s = sin_host[(size_t)p * half + dp];
                    cos_full[(size_t)p * HD + 2*dp + 0] = c;
                    cos_full[(size_t)p * HD + 2*dp + 1] = c;
                    sin_full[(size_t)p * HD + 2*dp + 0] = s;
                    sin_full[(size_t)p * HD + 2*dp + 1] = s;
                }
            }
            const size_t cf_bytes = cos_full.size() * sizeof(uint16_t);
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_cos_full_dev_,
                                              cf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_cos_full_dev_, cf_bytes,
                                              cos_full.data(), cf_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_sin_full_dev_,
                                              cf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_sin_full_dev_, cf_bytes,
                                              sin_full.data(), cf_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            stats_.rope_bytes += 2 * cf_bytes;
        }
    }

    // ------------------------------------------------------------------
    // Scratch allocations sized for the worst case. Phase 3 will populate,
    // Phase 2 just reserves the address space so subsequent alloc / reuse
    // plans are deterministic.
    //
    // Per-step scratch (F16):
    //   scratch_q/k/v : [seq_total × hidden]
    //   scratch_attn  : [seq_total × hidden]
    //   scratch_mlp   : [seq_total × ff_dim]
    //   scratch_mod   : [6 × hidden] per stream (img + txt), duplicated
    //   rstd          : [heads × seq_total] F32
    //
    // CFG duplicates land in the denoise path (Phase 4). We pre-reserve
    // the pointers so the Phase-2 HBM receipt reflects the engine's
    // steady-state peak rather than its init-only peak.
    // ------------------------------------------------------------------
    // Scope-block around scratch alloc: the `goto fail` branches inside this
    // block must not cross the outer-scope initializers that follow. Keeping
    // the block self-contained avoids clang's "jump over variable
    // initializer" error.
    {
        const int64_t SEQ = (int64_t)cfg_.max_img_seq + cfg_.max_txt_seq;
        const size_t F16 = sizeof(uint16_t);
        auto try_alloc = [&](void **ptr, size_t bytes) -> bool {
            aclError err = g_cann.aclrtMalloc(ptr, bytes,
                                               ACL_MEM_MALLOC_HUGE_FIRST);
            if (err != 0) {
                QIE_LOG("aclrtMalloc(%zu) failed err=%d", bytes, (int)err);
                *ptr = nullptr;
                return false;
            }
            stats_.scratch_bytes += bytes;
            return true;
        };
        if (!try_alloc(&scratch_q_dev_,    (size_t)SEQ * H * F16))  goto fail;
        if (!try_alloc(&scratch_k_dev_,    (size_t)SEQ * H * F16))  goto fail;
        if (!try_alloc(&scratch_v_dev_,    (size_t)SEQ * H * F16))  goto fail;
        if (!try_alloc(&scratch_attn_dev_, (size_t)SEQ * H * F16))  goto fail;
        if (!try_alloc(&scratch_mlp_dev_,  (size_t)SEQ * FF_DIM * F16)) goto fail;
        if (!try_alloc(&scratch_mod_dev_,  (size_t)12 * H * F16))      goto fail;
        if (!try_alloc(&rstd_dev_,         (size_t)cfg_.num_heads
                                             * SEQ * sizeof(float)))    goto fail;

        // Q2.3 per-stream intermediates.
        if (!try_alloc(&scratch_img_norm_dev_,
                       (size_t)cfg_.max_img_seq * H * F16)) goto fail;
        if (!try_alloc(&scratch_txt_norm_dev_,
                       (size_t)cfg_.max_txt_seq * H * F16)) goto fail;
        if (!try_alloc(&scratch_img_out_dev_,
                       (size_t)cfg_.max_img_seq * H * F16)) goto fail;
        if (!try_alloc(&scratch_txt_out_dev_,
                       (size_t)cfg_.max_txt_seq * H * F16)) goto fail;
        // LayerNorm mean/rstd are F32 per row (rows = B*seq).
        if (!try_alloc(&mean_dev_,    (size_t)SEQ * sizeof(float))) goto fail;
        if (!try_alloc(&ln_rstd_dev_, (size_t)SEQ * sizeof(float))) goto fail;

        // Phase 4.4c F32-residual scratch:
        //   scratch_img_hidden_f16 / scratch_txt_hidden_f16 — F16 mirror of
        //     the caller's F32 residual, populated by a Cast at LayerNorm1/2
        //     entry (reads F32 residual → writes F16).
        //   scratch_residual_tmp_f32 — F32 tmp, sized for max(img,txt), used
        //     by gated_residual_add_f32_ to hold Cast(src*gate, F32) before
        //     the += into the F32 residual stream.
        if (!try_alloc(&scratch_img_hidden_f16_dev_,
                       (size_t)cfg_.max_img_seq * H * F16)) goto fail;
        if (!try_alloc(&scratch_txt_hidden_f16_dev_,
                       (size_t)cfg_.max_txt_seq * H * F16)) goto fail;
        {
            const int64_t max_stream_seq =
                std::max(cfg_.max_img_seq, cfg_.max_txt_seq);
            if (!try_alloc(&scratch_residual_tmp_f32_dev_,
                           (size_t)max_stream_seq * H * sizeof(float)))
                goto fail;
        }

        // Phase 4.1 on-device RoPE scratch (three [seq, NH, head_dim/2] F16
        // buffers for the interleaved-rotation dispatch).
        {
            const int64_t NH_ = cfg_.num_heads;
            const int64_t HALF_HD = cfg_.head_dim / 2;
            const size_t rope_half_bytes =
                (size_t)SEQ * NH_ * HALF_HD * F16;
            if (!try_alloc(&scratch_rope_a_dev_, rope_half_bytes)) goto fail;
            if (!try_alloc(&scratch_rope_b_dev_, rope_half_bytes)) goto fail;
            if (!try_alloc(&scratch_rope_c_dev_, rope_half_bytes)) goto fail;
        }

        // Small workspace seed — ops grow it via ensure_workspace_.
        if (!try_alloc(&workspace_dev_, 4 * 1024 * 1024)) goto fail;
        workspace_size_ = 4 * 1024 * 1024;
    }

    {
        auto t_init1 = std::chrono::steady_clock::now();
        stats_.load_wall_ms =
            std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();

        const size_t total_bytes =
            stats_.q4_weight_bytes + stats_.q4_scale_bytes +
            stats_.f16_weight_bytes + stats_.f32_weight_bytes +
            stats_.rope_bytes + stats_.scratch_bytes;

        QIE_LOG("Phase 2.1 init OK: device=%d gguf=%s",
                device_, gguf_path.c_str());
        QIE_LOG("  tensors uploaded:   %lld "
                "(Q4-resident=%lld, F16-fallback=%lld)",
                (long long)stats_.tensors_uploaded,
                (long long)stats_.q4_tensors,
                (long long)stats_.f16_fallback_tensors);
        QIE_LOG("  Q4 weight bytes:    %zu (%.2f GiB)", stats_.q4_weight_bytes,
                stats_.q4_weight_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  Q4 scale  bytes:    %zu (%.2f GiB)", stats_.q4_scale_bytes,
                stats_.q4_scale_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  F16 weight bytes:   %zu (%.2f GiB)  "
                "[biases + F16-fallback weights]", stats_.f16_weight_bytes,
                stats_.f16_weight_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  F32 weight bytes:   %zu (%.2f MiB)  "
                "[RMSNorm gammas]", stats_.f32_weight_bytes,
                stats_.f32_weight_bytes / (1024.0 * 1024.0));
        QIE_LOG("  RoPE pe bytes:      %zu (%.2f MiB)", stats_.rope_bytes,
                stats_.rope_bytes / (1024.0 * 1024.0));
        QIE_LOG("  Scratch bytes:      %zu (%.2f GiB)", stats_.scratch_bytes,
                stats_.scratch_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  Peak init HBM:      %zu (%.2f GiB)  "
                "[Q2.1 smoke gate: < 9 GiB]", total_bytes,
                total_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  Dequant/repack wall: %.1f ms", stats_.dequant_wall_ms);
        QIE_LOG("  Total init wall:    %.1f ms", stats_.load_wall_ms);
    }

    // Mark ready. forward/denoise still stub (Phase 3/4) — callers that
    // hit those paths will get an explicit "scaffold" log.
    ready_ = true;
    return true;

fail:
    if (gguf_ctx) gguf_free(gguf_ctx);
    if (ggml_ctx) ggml_free(ggml_ctx);
    QIE_LOG("init_from_gguf FAILED partway through weight upload; see log above");
    return false;
}

// ---------------------------------------------------------------------------
// forward — dispatch all DiT blocks in sequence. Phase 3 smoke entry point.
// Phase 4 wires the wider denoising loop around this.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::forward(void *img_hidden_dev, int64_t img_seq,
                                     void *txt_hidden_dev, int64_t txt_seq,
                                     void *t_emb_dev,
                                     void *pe_dev) {
    if (!ready_) {
        QIE_LOG("forward: engine not ready");
        return false;
    }
    for (int il = 0; il < cfg_.num_layers; ++il) {
        if (!forward_block_(layer_w_[il],
                            img_hidden_dev, img_seq,
                            txt_hidden_dev, txt_seq,
                            t_emb_dev, pe_dev)) {
            QIE_LOG("forward: block %d returned error", il);
            return false;
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// denoise — full 20-step Euler-flow loop (Phase 4 will fill this body).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::denoise(const float * /*initial_noise*/,
                                     int64_t N, int64_t C, int64_t H, int64_t W,
                                     const float * /*cond_emb*/,
                                     int64_t cond_seq, int64_t cond_dim,
                                     const float * /*uncond_emb*/,
                                     const float * /*ref_latents*/,
                                     int64_t ref_N, int64_t ref_C,
                                     int64_t ref_H, int64_t ref_W,
                                     float * /*out_latents*/) {
    if (!ready_) {
        QIE_LOG("denoise: engine not ready");
        return false;
    }
    QIE_LOG("denoise: scaffold Phase 2 — loop wires up in Phase 4. "
            "latent=[%lld,%lld,%lld,%lld] cond=[%lld,%lld] "
            "ref=[%lld,%lld,%lld,%lld]",
            (long long)N, (long long)C, (long long)H, (long long)W,
            (long long)cond_seq, (long long)cond_dim,
            (long long)ref_N, (long long)ref_C,
            (long long)ref_H, (long long)ref_W);
    return false;
}

// ---------------------------------------------------------------------------
// Internal helpers.
// ---------------------------------------------------------------------------
void ImageDiffusionEngine::alloc_dev_(void **ptr, size_t bytes) {
    if (!cp_cann_load_symbols()) {
        *ptr = nullptr;
        return;
    }
    aclError err = g_cann.aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (err != 0) {
        QIE_LOG("aclrtMalloc(%zu) failed err=%d", bytes, (int)err);
        *ptr = nullptr;
    }
}

void ImageDiffusionEngine::ensure_workspace_(size_t bytes) {
    if (bytes <= workspace_size_) return;
    if (workspace_dev_) {
        g_cann.aclrtFree(workspace_dev_);
        workspace_dev_ = nullptr;
    }
    alloc_dev_(&workspace_dev_, bytes);
    workspace_size_ = bytes;
}

namespace {
// Q2.4.5.4c: lazy device-buffer grow helper. Allocates / reallocates
// `*ptr` so it holds at least `bytes` bytes. `cur_bytes` is updated to
// the new size on success. Used by dispatch_matmul_'s BF16 path for
// per-tensor scale/bias cast scratch and by gated_residual_add_f32_bf16src_
// for its F32 staging buffer.
inline bool ensure_dev_grow_(void **ptr, size_t *cur_bytes, size_t bytes) {
    if (bytes <= *cur_bytes) return true;
    if (*ptr) {
        g_cann.aclrtFree(*ptr);
        *ptr = nullptr;
        *cur_bytes = 0;
    }
    aclError ae = g_cann.aclrtMalloc(ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (ae != 0 || !*ptr) {
        QIE_LOG("ensure_dev_grow_: aclrtMalloc(%zu) failed status=%d",
                bytes, (int)ae);
        return false;
    }
    *cur_bytes = bytes;
    return true;
}
}  // namespace

void ImageDiffusionEngine::build_rope_tables_() {
    // The meat of the table build lives in the namespace-scope
    // compute_qwen_rope_pe_host() used by init_from_gguf directly. Phase
    // 3+ may re-upload on resolution change — this method is retained
    // for future reuse (e.g. a session.rebuild_rope_at(h, w, ref_h) hook)
    // but Phase 2 does everything in init.
    QIE_LOG("build_rope_tables_: subsumed by init_from_gguf in Phase 2");
}

void ImageDiffusionEngine::build_time_emb_(float timestep, void *out_dev) {
    // Sinusoidal timestep embedding — host-side F32 compute, upload as F16
    // [256]. Standard diffusion formula (matches
    // ominix_diffusion's `timestep_embedding`):
    //   for i in [0, half):
    //       freqs[i] = exp(-log(max_period) * i / half)
    //   emb[2i]     = cos(timestep * freqs[i])
    //   emb[2i + 1] = sin(timestep * freqs[i])   (pair interleaved)
    // We emit 256 dims matching time_text_embed.timestep_embedder's input
    // width (see qwen_image.hpp). Caller is expected to chain `time_linear{1,2}`
    // to project to `hidden`. For the Phase 4.3 smoke the caller pre-computes
    // a single-step synthetic t_emb on host and skips the linear projection,
    // so this function is exposed for future Phase-4-production consumers.
    if (!out_dev) return;
    const int DIM = 256;
    const int half = DIM / 2;
    const float max_period = 10000.0f;
    std::vector<uint16_t> emb((size_t)DIM, 0);
    for (int i = 0; i < half; ++i) {
        float freq = std::exp(-std::log(max_period) * (float)i / (float)half);
        float arg  = timestep * freq;
        emb[(size_t)(2 * i)    ] = fp32_to_fp16(std::cos(arg));
        emb[(size_t)(2 * i + 1)] = fp32_to_fp16(std::sin(arg));
    }
    g_cann.aclrtMemcpy(out_dev, (size_t)DIM * sizeof(uint16_t),
                         emb.data(), (size_t)DIM * sizeof(uint16_t),
                         ACL_MEMCPY_HOST_TO_DEVICE);
}

// ============================================================================
// Q2.3 Phase 3 — forward_block_ and its op dispatch helpers.
//
// Implements one DiT block on NPU. Every site below maps 1:1 to a line in the
// CPU reference at `tools/ominix_diffusion/src/qwen_image.hpp:251-315`.
//
// Ordering:
//   1. Modulation: img_mod.1(silu(t_emb)) split into 6 chunks × 2 streams
//   2. LayerNorm1 (affine-off) + modulate(scale1, shift1) → img_mod, txt_mod
//   3. QKV projections per stream → reshape to [B, seq, n_head, head_dim]
//   4. Q/K RMSNorm per stream (head-wise gamma)
//   5. Apply 3D-axial RoPE on Q, K per stream (using pre-computed pe table)
//   6. Concat txt || img along seq dim → joint [B, seq_txt+seq_img, N, D]
//   7. aclnnFusedInferAttentionScoreV2 at joint seq
//   8. Split attn output back into img / txt chunks
//   9. Output projections to_out.0 (img) and to_add_out (txt)
//  10. Gated residual add: img += attn_out_img * gate1_img  (and txt)
//  11. LayerNorm2 + modulate(scale2, shift2)
//  12. FFN per stream: Linear(H→ff_dim) → GELU-tanh → Linear(ff_dim→H)
//  13. Gated residual add: img += ffn_out_img * gate2_img  (and txt)
// ============================================================================

namespace {

// Local helpers for tensor construction in the forward path. Keep them
// namespace-scoped so we don't pollute the engine class with friend boilerplate.
inline aclTensor *tensor_nd_f16(void *dev, int ndim,
                                 const int64_t *shape,
                                 const int64_t *strides) {
    int64_t storage = 1;
    for (int i = 0; i < ndim; ++i) storage *= shape[i];
    return g_cann.aclCreateTensor(shape, (uint64_t)ndim, ACL_FLOAT16,
                                   strides, 0, ACL_FORMAT_ND,
                                   &storage, 1, dev);
}

inline aclTensor *tensor_nd_f32(void *dev, int ndim,
                                 const int64_t *shape,
                                 const int64_t *strides) {
    int64_t storage = 1;
    for (int i = 0; i < ndim; ++i) storage *= shape[i];
    return g_cann.aclCreateTensor(shape, (uint64_t)ndim, ACL_FLOAT,
                                   strides, 0, ACL_FORMAT_ND,
                                   &storage, 1, dev);
}

// Q2.4.5.4c: BF16 tensor builder. Same byte-size as F16 (2 bytes/elem) so
// existing F16-sized scratch buffers fit BF16 tensors of the same shape.
inline aclTensor *tensor_nd_bf16(void *dev, int ndim,
                                  const int64_t *shape,
                                  const int64_t *strides) {
    int64_t storage = 1;
    for (int i = 0; i < ndim; ++i) storage *= shape[i];
    return g_cann.aclCreateTensor(shape, (uint64_t)ndim, ACL_BF16,
                                   strides, 0, ACL_FORMAT_ND,
                                   &storage, 1, dev);
}

// Row-major contiguous stride helper.
inline void make_contig_strides(int ndim, const int64_t *shape,
                                 int64_t *out_strides) {
    int64_t s = 1;
    for (int i = ndim - 1; i >= 0; --i) {
        out_strides[i] = s;
        s *= shape[i];
    }
}

inline aclScalar *make_f16_scalar_local(float v) {
    uint16_t b = fp32_to_fp16(v);
    return g_cann.aclCreateScalar(&b, ACL_FLOAT16);
}

}  // namespace

// ---------------------------------------------------------------------------
// Q2.3: matmul dispatch. Activation `x_f16` [M, K]; weight logical [K, N];
// output `y_f16` [M, N]. Weight physical layout depends on path:
//   Q4-resident: `weight_dev` is INT4 packed [K*N/2 bytes], `weight_scale_dev`
//                is F16 [K/32, N]; antiquantGroupSize=32 per Q2.1 probe.
//   F16-fallback: `weight_dev` holds the F16 weight in GGUF's physical [N, K]
//                 row-major layout (aka `[K, N]` viewed with strides (1, K)).
//                 We build a transposed view and dispatch aclnnMm.
// If bias_f16_dev != nullptr, aclnnAdd is chained after the matmul to add
// bias broadcast over M (bias shape [N]).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::dispatch_matmul_(void *x_f16_dev, void *weight_dev,
                                              void *weight_scale_dev,
                                              void *bias_f16_dev,
                                              int64_t M, int64_t K, int64_t N,
                                              void *y_dev,
                                              aclDataType out_dtype) {
    if (!x_f16_dev || !weight_dev || !y_dev) {
        QIE_LOG("dispatch_matmul_: null buffer (x=%p w=%p y=%p)",
                x_f16_dev, weight_dev, y_dev);
        return false;
    }
    // Q2.4.5.4c: BF16 output dispatch.
    //   WQBMMv3 (Q4 + F16 scale) — cast scale F16→BF16 inline; BF16 output.
    //   aclnnMm (F16 fallback)  — pre-cast input F16→BF16 and weight
    //                              F16→BF16 to scratch buffers, then run
    //                              aclnnMm with BF16/BF16/BF16 + cubeMathType
    //                              ALLOW_FP32_DOWN_PRECISION (1) for F32
    //                              accumulator. Weight cast is the expensive
    //                              part (75 MB per FF→H matmul); amortised
    //                              by caching a per-pointer BF16 mirror in
    //                              scratch_bf16_weight_cache_ (a single-slot
    //                              LRU keyed on weight_dev pointer; ff_down
    //                              has 120 unique weights × 75 MB = 9 GB —
    //                              too large to cache all, so under-budget
    //                              we re-cast on every call. Optimisation
    //                              for a follow-up: pre-convert all ff_down
    //                              weights at init_from_gguf time when env
    //                              is set, like ggml-cann's GGML_CANN_QUANT_BF16.
    const bool use_bf16 = (out_dtype == ACL_BF16);

    // Q2.4.4b precision knobs. Cached once: these control the matmul
    // accumulator/numerical path. Defaults preserve the Phase 4.2 synthetic
    // smoke behaviour (inner_precise=1 for WQBMMv3 → HIGH_PERFORMANCE F16
    // accumulator; cube_math_type=0 for aclnnMm → KEEP_DTYPE F16 accumulator).
    // Real-weight Phase 4.4 forward NaNs on the defaults because F16
    // accumulation overflows once per-channel |x @ W| grows past F16 max
    // (~65504) — set
    //   QIE_MATMUL_INNER_PRECISE=0  → WQBMMv3 HIGH_PRECISION (F32 accum)
    //   QIE_MATMUL_CUBE_MATH=1      → aclnnMm ALLOW_FP32_DOWN_PRECISION
    //                                  (F32 accum, F16 inputs preserved)
    // to route through the F32-accumulator paths.
    static int s_inner_precise = -1;
    static int s_cube_math     = -1;
    if (s_inner_precise < 0) {
        const char *v = std::getenv("QIE_MATMUL_INNER_PRECISE");
        s_inner_precise = v ? std::atoi(v) : 1;
        QIE_LOG("dispatch_matmul_: QIE_MATMUL_INNER_PRECISE=%d (WQBMMv3 "
                "innerPrecise; 0=HIGH_PRECISION/F32-accum, "
                "1=HIGH_PERFORMANCE/F16-accum)", s_inner_precise);
    }
    if (s_cube_math < 0) {
        const char *v = std::getenv("QIE_MATMUL_CUBE_MATH");
        s_cube_math = v ? std::atoi(v) : 0;
        QIE_LOG("dispatch_matmul_: QIE_MATMUL_CUBE_MATH=%d (aclnnMm "
                "cubeMathType; 0=KEEP_DTYPE, 1=ALLOW_FP32_DOWN_PRECISION, "
                "2=USE_FP16, 3=USE_HF32)", s_cube_math);
    }

    // Build activation tensor view [M, K] contig.
    int64_t x_shape[2]   = {M, K};
    int64_t x_strides[2] = {K, 1};
    aclTensor *t_x = tensor_nd_f16(x_f16_dev, 2, x_shape, x_strides);

    // Build output tensor view [M, N] contig. Q2.4.5.4c: BF16 when
    // env-gated. Same byte-size as F16 (2 bytes/elem) so the caller's
    // buffer sizing is unchanged.
    int64_t y_shape[2]   = {M, N};
    int64_t y_strides[2] = {N, 1};
    aclTensor *t_y = use_bf16 ? tensor_nd_bf16(y_dev, 2, y_shape, y_strides)
                              : tensor_nd_f16 (y_dev, 2, y_shape, y_strides);

    // Q2.4.5.4c/d BF16 plumbing — pre-cast input F16→BF16 to local scratch.
    // Both matmul paths need this when out_dtype=BF16:
    //   - aclnnMm fallback: BF16/BF16/BF16 with cubeMathType=1.
    //   - WQBMMv3 (Q4 path): activation must match scale/output dtype
    //     (CANN spec — confirmed empirically: WQBMMv3 returns 161002
    //     "wrong dtype combo" when t_x stays F16 while t_scale and t_y
    //     are BF16).
    // Weight pre-cast (F16→BF16) is exclusive to the aclnnMm path; the
    // WQBMMv3 path keeps the INT4-packed weight verbatim and uses the
    // already-cast BF16 scale tile.
    void *bf16_x_dev = nullptr;   // pre-cast input scratch (M*K BF16)
    void *bf16_w_dev = nullptr;   // pre-cast weight scratch (K*N BF16) — aclnnMm only
    if (use_bf16) {
        const size_t x_bytes = (size_t)M * (size_t)K * 2;
        // Reuse scratch_bf16_scale_dev_ for the input cast. For the
        // aclnnMm path scratch_bf16_scale_dev_ is otherwise unused; for
        // the WQBMMv3 path it ALSO holds the cast scale tile, but the
        // scale cast happens AFTER the input cast (input is consumed in
        // the matmul launch which is sequenced after the scale cast),
        // and we need the input buffer to remain valid through the
        // matmul launch. Sequencing: input lives in scratch_bf16_scale_dev_
        // first; for WQBMMv3 the scale is cast into scratch_bf16_bias_dev_
        // (tail buffer) instead — we re-route the scale buffer below.
        // For simplicity we use a dedicated pair of buffers per path.
        //
        // The aclnnMm path uses (scratch_bf16_scale_dev_, scratch_bf16_src_f32_dev_)
        // for (input_x, weight_w) since neither original consumer fires.
        // The WQBMMv3 path uses (scratch_bf16_src_f32_dev_, scratch_bf16_scale_dev_)
        // for (input_x, scale) — input goes into the larger src_f32 slot
        // and scale into the smaller scale slot. Sized for the worst
        // input (M·K BF16 ≤ FF·H bytes per call) and worst scale tile
        // (K/32 · N BF16). M·K can exceed the scale-tile size at the
        // attn-out site (M = max(seq) = 4096, K = H = 3072 → 24 MiB)
        // vs scale tile (K/32 · N = 96 · H = 288 K elems = 0.6 MiB) so
        // src_f32 must hold the input. Both buffers grow lazily.
        if (weight_scale_dev != nullptr) {
            // WQBMMv3 path layout.
            if (!ensure_dev_grow_(&scratch_bf16_src_f32_dev_,
                                   &scratch_bf16_src_f32_bytes_, x_bytes)) {
                g_cann.aclDestroyTensor(t_x); g_cann.aclDestroyTensor(t_y);
                return false;
            }
            bf16_x_dev = scratch_bf16_src_f32_dev_;
        } else {
            // aclnnMm path layout — input + weight both need BF16 scratch.
            const size_t w_bytes = (size_t)K * (size_t)N * 2;
            if (!ensure_dev_grow_(&scratch_bf16_scale_dev_,
                                   &scratch_bf16_scale_bytes_, x_bytes)) {
                g_cann.aclDestroyTensor(t_x); g_cann.aclDestroyTensor(t_y);
                return false;
            }
            if (!ensure_dev_grow_(&scratch_bf16_src_f32_dev_,
                                   &scratch_bf16_src_f32_bytes_, w_bytes)) {
                g_cann.aclDestroyTensor(t_x); g_cann.aclDestroyTensor(t_y);
                return false;
            }
            bf16_x_dev = scratch_bf16_scale_dev_;
            bf16_w_dev = scratch_bf16_src_f32_dev_;
        }

        // Cast input F16 → BF16. Flat 1-D view (cast is shape-agnostic).
        {
            int64_t f_shape[1]   = {(int64_t)(M * K)};
            int64_t f_strides[1] = {1};
            aclTensor *t_xf16 = tensor_nd_f16 (x_f16_dev, 1, f_shape, f_strides);
            aclTensor *t_xbf16 = tensor_nd_bf16(bf16_x_dev, 1, f_shape, f_strides);
            uint64_t cw = 0; aclOpExecutor *cex = nullptr;
            aclnnStatus cs_x = g_cann.aclnnCastGetWorkspaceSize(
                t_xf16, ACL_BF16, t_xbf16, &cw, &cex);
            if (cs_x == 0) {
                ensure_workspace_(cw);
                cs_x = g_cann.aclnnCast(cw > 0 ? workspace_dev_ : nullptr,
                                          cw, cex, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_xf16);
            g_cann.aclDestroyTensor(t_xbf16);
            if (cs_x != 0) {
                QIE_LOG("dispatch_matmul_: input F16->BF16 cast status=%d",
                        (int)cs_x);
                g_cann.aclDestroyTensor(t_x); g_cann.aclDestroyTensor(t_y);
                return false;
            }
        }
        // Cast weight F16 → BF16 (aclnnMm path only).
        if (weight_scale_dev == nullptr) {
            int64_t f_shape[1]   = {(int64_t)(K * N)};
            int64_t f_strides[1] = {1};
            aclTensor *t_wf16 = tensor_nd_f16 (weight_dev, 1, f_shape, f_strides);
            aclTensor *t_wbf16 = tensor_nd_bf16(bf16_w_dev, 1, f_shape, f_strides);
            uint64_t cw = 0; aclOpExecutor *cex = nullptr;
            aclnnStatus cs_w = g_cann.aclnnCastGetWorkspaceSize(
                t_wf16, ACL_BF16, t_wbf16, &cw, &cex);
            if (cs_w == 0) {
                ensure_workspace_(cw);
                cs_w = g_cann.aclnnCast(cw > 0 ? workspace_dev_ : nullptr,
                                          cw, cex, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_wf16);
            g_cann.aclDestroyTensor(t_wbf16);
            if (cs_w != 0) {
                QIE_LOG("dispatch_matmul_: weight F16->BF16 cast status=%d",
                        (int)cs_w);
                g_cann.aclDestroyTensor(t_x); g_cann.aclDestroyTensor(t_y);
                return false;
            }
        }
    }

    aclnnStatus s = 0;
    if (weight_scale_dev != nullptr) {
        // --- Q4 path: WQBMMv3 with antiquantGroupSize=32 -------------------
        // Weight logical shape [K, N] with strides (1, K): byte `n*K+k` lo
        // nibble = element (k, n). Scale logical shape [K/32, N].
        int64_t w_shape[2]   = {K, N};
        int64_t w_strides[2] = {1, K};
        int64_t w_storage    = K * N;
        aclTensor *t_w = g_cann.aclCreateTensor(
            w_shape, 2, ACL_INT4, w_strides, 0, ACL_FORMAT_ND,
            &w_storage, 1, weight_dev);

        int64_t s_shape[2]   = {K / 32, N};
        int64_t s_strides[2] = {N, 1};
        // Q2.4.5.4c: when output is BF16, WQBMMv3 requires the scale tensor
        // in matching dtype (precedent: ggml-cann backend's GGML_CANN_QUANT_BF16
        // path; see ggml/src/ggml-cann/aclnn_ops.cpp:2670-2686). Cast F16 →
        // BF16 lazily into scratch_bf16_scale_dev_ on each call (the cast is
        // a single Vec op on K/32 × N elements, cheap relative to the matmul).
        aclTensor *t_scale = nullptr;
        if (use_bf16) {
            const size_t scale_elems = (size_t)(K / 32) * (size_t)N;
            const size_t scale_bytes = scale_elems * 2;  // BF16 = 2 bytes
            if (!ensure_dev_grow_(&scratch_bf16_scale_dev_,
                                   &scratch_bf16_scale_bytes_,
                                   scale_bytes)) {
                g_cann.aclDestroyTensor(t_x);
                g_cann.aclDestroyTensor(t_y);
                g_cann.aclDestroyTensor(t_w);
                return false;
            }
            int64_t flat_shape[1]   = {(int64_t)scale_elems};
            int64_t flat_strides[1] = {1};
            aclTensor *t_scale_f16 =
                tensor_nd_f16(weight_scale_dev, 1, flat_shape, flat_strides);
            aclTensor *t_scale_bf16 =
                tensor_nd_bf16(scratch_bf16_scale_dev_, 1, flat_shape,
                                flat_strides);
            uint64_t cs_ws = 0;
            aclOpExecutor *cs_exec = nullptr;
            aclnnStatus cs_s = g_cann.aclnnCastGetWorkspaceSize(
                t_scale_f16, ACL_BF16, t_scale_bf16, &cs_ws, &cs_exec);
            if (cs_s == 0) {
                ensure_workspace_(cs_ws);
                cs_s = g_cann.aclnnCast(cs_ws > 0 ? workspace_dev_ : nullptr,
                                          cs_ws, cs_exec, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_scale_f16);
            if (cs_s != 0) {
                QIE_LOG("dispatch_matmul_: scale F16->BF16 cast status=%d",
                        (int)cs_s);
                g_cann.aclDestroyTensor(t_scale_bf16);
                g_cann.aclDestroyTensor(t_x);
                g_cann.aclDestroyTensor(t_y);
                g_cann.aclDestroyTensor(t_w);
                return false;
            }
            // Re-view the BF16 scale buffer in [K/32, N] layout for WQBMMv3.
            g_cann.aclDestroyTensor(t_scale_bf16);
            t_scale = tensor_nd_bf16(scratch_bf16_scale_dev_, 2, s_shape,
                                       s_strides);
        } else {
            t_scale = tensor_nd_f16(weight_scale_dev, 2, s_shape, s_strides);
        }

        // Q2.4.5.4d: under BF16 output the activation must also be BF16
        // (WQBMMv3 dtype-strict — F16 input vs BF16 scale/output returns
        // status=161002 "wrong dtype combo"). Use the pre-cast input
        // built earlier (`bf16_x_dev` in scratch_bf16_src_f32_dev_).
        aclTensor *t_x_local_bf16 = nullptr;
        aclTensor *t_x_eff = t_x;
        if (use_bf16) {
            int64_t x_shape_local[2]   = {M, K};
            int64_t x_strides_local[2] = {K, 1};
            t_x_local_bf16 = tensor_nd_bf16(bf16_x_dev, 2, x_shape_local,
                                              x_strides_local);
            t_x_eff = t_x_local_bf16;
        }

        uint64_t ws_needed = 0;
        aclOpExecutor *exec = nullptr;
        if (g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize &&
            g_cann.aclnnWeightQuantBatchMatmulV3) {
            s = g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
                t_x_eff, t_w, t_scale,
                /*antiquantOffsetOptional*/ nullptr,
                /*quantScaleOptional*/      nullptr,
                /*quantOffsetOptional*/     nullptr,
                /*biasOptional*/            nullptr,  // apply bias after
                /*antiquantGroupSize*/      32,
                /*innerPrecise*/            s_inner_precise,
                t_y, &ws_needed, &exec);
            if (s == 0) {
                ensure_workspace_(ws_needed);
                void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
                s = g_cann.aclnnWeightQuantBatchMatmulV3(ws, ws_needed, exec,
                                                          compute_stream_);
                if (s != 0) {
                    QIE_LOG("dispatch_matmul_: WQBMMv3 launch status=%d",
                            (int)s);
                }
            } else {
                QIE_LOG("dispatch_matmul_: WQBMMv3 workspace status=%d",
                        (int)s);
            }
        } else {
            QIE_LOG("dispatch_matmul_: WQBMMv3 symbol missing");
            s = -1;
        }
        g_cann.aclDestroyTensor(t_w);
        g_cann.aclDestroyTensor(t_scale);
        if (t_x_local_bf16) g_cann.aclDestroyTensor(t_x_local_bf16);
        if (s != 0) {
            g_cann.aclDestroyTensor(t_x);
            g_cann.aclDestroyTensor(t_y);
            return false;
        }
    } else {
        // --- F16 fallback path: plain aclnnMm ------------------------------
        // Weight stored as GGUF [N, K] row-major (K contig). Transposed view
        // into [K, N] with strides (1, K).
        // Q2.4.5.4c: under BF16 output we already pre-cast input + weight
        // F16 → BF16 above (bf16_x_dev / bf16_w_dev). Use BF16 views for
        // the matmul inputs in that case; the output tensor is already
        // BF16 by t_y dtype branch.
        int64_t w_shape[2]   = {K, N};
        int64_t w_strides[2] = {1, K};
        int64_t w_storage    = K * N;
        aclTensor *t_w = nullptr;
        aclTensor *t_x_eff = t_x;  // alias unless we need to re-view as BF16
        aclTensor *t_x_local_bf16 = nullptr;
        if (use_bf16) {
            // Build BF16 views over the pre-cast scratch buffers.
            int64_t x_shape_local[2]   = {M, K};
            int64_t x_strides_local[2] = {K, 1};
            t_x_local_bf16 = tensor_nd_bf16(bf16_x_dev, 2, x_shape_local,
                                              x_strides_local);
            t_x_eff = t_x_local_bf16;
            t_w = g_cann.aclCreateTensor(
                w_shape, 2, ACL_BF16, w_strides, 0, ACL_FORMAT_ND,
                &w_storage, 1, bf16_w_dev);
        } else {
            t_w = g_cann.aclCreateTensor(
                w_shape, 2, ACL_FLOAT16, w_strides, 0, ACL_FORMAT_ND,
                &w_storage, 1, weight_dev);
        }

        // Q2.4.5.4c: under BF16 force cubeMathType=1
        // (ALLOW_FP32_DOWN_PRECISION → F32 accumulator) so the inner sum
        // can hold magnitudes > BF16 range; it down-casts to BF16 on store.
        const int8_t cube_math_eff =
            use_bf16 ? (int8_t)1 : (int8_t)s_cube_math;

        uint64_t ws_needed = 0;
        aclOpExecutor *exec = nullptr;
        s = g_cann.aclnnMmGetWorkspaceSize(t_x_eff, t_w, t_y,
                                             cube_math_eff,
                                             &ws_needed, &exec);
        if (s != 0) {
            QIE_LOG("dispatch_matmul_: Mm workspace status=%d", (int)s);
        } else {
            ensure_workspace_(ws_needed);
            void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
            s = g_cann.aclnnMm(ws, ws_needed, exec, compute_stream_);
            if (s != 0) {
                QIE_LOG("dispatch_matmul_: Mm launch status=%d", (int)s);
            }
        }
        g_cann.aclDestroyTensor(t_w);
        if (t_x_local_bf16) g_cann.aclDestroyTensor(t_x_local_bf16);
        if (s != 0) {
            g_cann.aclDestroyTensor(t_x);
            g_cann.aclDestroyTensor(t_y);
            return false;
        }
    }

    // Chain bias add if present: y += bias broadcast over M.
    if (bias_f16_dev) {
        int64_t b_shape[2]   = {1, N};
        int64_t b_strides[2] = {N, 1};
        // Q2.4.5.4c: when output is BF16 the bias must also be BF16
        // (aclnnInplaceAdd dtype-strict). Cast F16 → BF16 lazily into
        // scratch_bf16_bias_dev_ on each call.
        aclTensor *t_b = nullptr;
        if (use_bf16) {
            const size_t bias_bytes = (size_t)N * 2;  // BF16 1D [N]
            if (!ensure_dev_grow_(&scratch_bf16_bias_dev_,
                                   &scratch_bf16_bias_bytes_,
                                   bias_bytes)) {
                g_cann.aclDestroyTensor(t_x);
                g_cann.aclDestroyTensor(t_y);
                return false;
            }
            int64_t f_shape[1]   = {N};
            int64_t f_strides[1] = {1};
            aclTensor *t_b_f16 =
                tensor_nd_f16(bias_f16_dev, 1, f_shape, f_strides);
            aclTensor *t_b_bf16_flat =
                tensor_nd_bf16(scratch_bf16_bias_dev_, 1, f_shape, f_strides);
            uint64_t cb_ws = 0;
            aclOpExecutor *cb_exec = nullptr;
            aclnnStatus cb_s = g_cann.aclnnCastGetWorkspaceSize(
                t_b_f16, ACL_BF16, t_b_bf16_flat, &cb_ws, &cb_exec);
            if (cb_s == 0) {
                ensure_workspace_(cb_ws);
                cb_s = g_cann.aclnnCast(cb_ws > 0 ? workspace_dev_ : nullptr,
                                          cb_ws, cb_exec, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_b_f16);
            g_cann.aclDestroyTensor(t_b_bf16_flat);
            if (cb_s != 0) {
                QIE_LOG("dispatch_matmul_: bias F16->BF16 cast status=%d",
                        (int)cb_s);
                g_cann.aclDestroyTensor(t_x);
                g_cann.aclDestroyTensor(t_y);
                return false;
            }
            t_b = tensor_nd_bf16(scratch_bf16_bias_dev_, 2, b_shape, b_strides);
        } else {
            t_b = tensor_nd_f16(bias_f16_dev, 2, b_shape, b_strides);
        }

        // alpha scalar dtype: F16 path keeps the historical F16 alpha (the
        // safe pre-Q2.4.5.4c default). BF16 path uses F32 alpha which CANN
        // ops broadly accept regardless of tensor dtype (avoids needing a
        // BF16-typed scalar variant whose support across aclnn versions is
        // less proven than F16/F32).
        aclScalar *alpha = nullptr;
        if (use_bf16) {
            float one_f32 = 1.0f;
            alpha = g_cann.aclCreateScalar(&one_f32, ACL_FLOAT);
        } else {
            uint16_t one = fp32_to_fp16(1.0f);
            alpha = g_cann.aclCreateScalar(&one, ACL_FLOAT16);
        }
        uint64_t ws_needed = 0;
        aclOpExecutor *exec = nullptr;
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_y, t_b, alpha,
                                                     &ws_needed, &exec);
        if (s != 0) {
            QIE_LOG("dispatch_matmul_: InplaceAdd workspace status=%d",
                    (int)s);
        } else {
            ensure_workspace_(ws_needed);
            void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
            s = g_cann.aclnnInplaceAdd(ws, ws_needed, exec, compute_stream_);
            if (s != 0) {
                QIE_LOG("dispatch_matmul_: InplaceAdd launch status=%d",
                        (int)s);
            }
        }
        g_cann.aclDestroyScalar(alpha);
        g_cann.aclDestroyTensor(t_b);
    }

    g_cann.aclDestroyTensor(t_x);
    g_cann.aclDestroyTensor(t_y);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.3: modulate — x = x * (1 + scale) + shift, scale/shift [B, hidden]
// broadcast over seq. Implemented as two ops:
//   aclnnMul   y_tmp = x * scale_reshape     (broadcast)
//   aclnnInplaceAdd  x += y_tmp              (bias-style add)
//   aclnnInplaceAdd  x += shift_reshape      (bias-style add, broadcast)
// The broadcast works because aclnnMul / aclnnAdd follow numpy-style rules:
// shape [B, seq, H] op shape [B, 1, H] → [B, seq, H].
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::modulate_(void *x_f16_dev,
                                       const void *scale_f16_dev,
                                       const void *shift_f16_dev,
                                       int64_t B, int64_t seq, int64_t hidden) {
    if (!x_f16_dev || !scale_f16_dev || !shift_f16_dev) return false;
    if (!g_cann.aclnnMul || !g_cann.aclnnMulGetWorkspaceSize) {
        QIE_LOG("modulate_: aclnnMul symbol missing");
        return false;
    }

    // Build tensors. scratch_mlp_dev_ is sized [SEQ, FF_DIM] F16 worst case —
    // ample for a temporary [B, seq, H]. We borrow its head here.
    int64_t x_shape[3]   = {B, seq, hidden};
    int64_t x_strides[3];
    make_contig_strides(3, x_shape, x_strides);
    int64_t b_shape[3]   = {B, 1, hidden};
    int64_t b_strides[3] = {hidden, hidden, 1};

    aclTensor *t_x = tensor_nd_f16(x_f16_dev, 3, x_shape, x_strides);
    aclTensor *t_scale = tensor_nd_f16(const_cast<void *>(scale_f16_dev),
                                         3, b_shape, b_strides);
    aclTensor *t_shift = tensor_nd_f16(const_cast<void *>(shift_f16_dev),
                                         3, b_shape, b_strides);
    // tmp = x * scale — needs a temporary buffer. Use scratch_mlp_dev_ head.
    aclTensor *t_tmp = tensor_nd_f16(scratch_mlp_dev_, 3, x_shape, x_strides);

    aclnnStatus s = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;

    s = g_cann.aclnnMulGetWorkspaceSize(t_x, t_scale, t_tmp, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnMul(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                             compute_stream_);
    }
    if (s != 0) QIE_LOG("modulate_: Mul status=%d", (int)s);

    // x += tmp
    if (s == 0) {
        aclScalar *alpha = make_f16_scalar_local(1.0f);
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x, t_tmp, alpha,
                                                     &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnInplaceAdd(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
        }
        g_cann.aclDestroyScalar(alpha);
        if (s != 0) QIE_LOG("modulate_: InplaceAdd(x+=tmp) status=%d", (int)s);
    }

    // x += shift
    if (s == 0) {
        aclScalar *alpha = make_f16_scalar_local(1.0f);
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x, t_shift, alpha,
                                                     &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnInplaceAdd(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
        }
        g_cann.aclDestroyScalar(alpha);
        if (s != 0) QIE_LOG("modulate_: InplaceAdd(x+=shift) status=%d",
                            (int)s);
    }

    g_cann.aclDestroyTensor(t_x);
    g_cann.aclDestroyTensor(t_scale);
    g_cann.aclDestroyTensor(t_shift);
    g_cann.aclDestroyTensor(t_tmp);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.3: gated residual add. x += src * gate, gate [B, hidden] broadcast over
// seq. Two ops: aclnnMul(src, gate, tmp) then aclnnInplaceAdd(x, tmp).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::gated_residual_add_(void *x_f16_dev,
                                                 const void *src_f16_dev,
                                                 const void *gate_f16_dev,
                                                 int64_t B, int64_t seq,
                                                 int64_t hidden) {
    if (!x_f16_dev || !src_f16_dev || !gate_f16_dev) return false;
    if (!g_cann.aclnnMul) { QIE_LOG("gate_add_: no Mul"); return false; }

    int64_t x_shape[3]   = {B, seq, hidden};
    int64_t x_strides[3];
    make_contig_strides(3, x_shape, x_strides);
    int64_t g_shape[3]   = {B, 1, hidden};
    int64_t g_strides[3] = {hidden, hidden, 1};

    aclTensor *t_x   = tensor_nd_f16(x_f16_dev, 3, x_shape, x_strides);
    aclTensor *t_src = tensor_nd_f16(const_cast<void *>(src_f16_dev),
                                       3, x_shape, x_strides);
    aclTensor *t_g   = tensor_nd_f16(const_cast<void *>(gate_f16_dev),
                                       3, g_shape, g_strides);
    aclTensor *t_tmp = tensor_nd_f16(scratch_mlp_dev_, 3, x_shape, x_strides);

    aclnnStatus s = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    s = g_cann.aclnnMulGetWorkspaceSize(t_src, t_g, t_tmp, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnMul(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                             compute_stream_);
    }
    if (s != 0) QIE_LOG("gate_add_: Mul status=%d", (int)s);

    if (s == 0) {
        aclScalar *alpha = make_f16_scalar_local(1.0f);
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x, t_tmp, alpha,
                                                     &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnInplaceAdd(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
        }
        g_cann.aclDestroyScalar(alpha);
        if (s != 0) QIE_LOG("gate_add_: InplaceAdd status=%d", (int)s);
    }

    g_cann.aclDestroyTensor(t_x);
    g_cann.aclDestroyTensor(t_src);
    g_cann.aclDestroyTensor(t_g);
    g_cann.aclDestroyTensor(t_tmp);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Phase 4.4c: generic aclnnCast wrapper. `in` is viewed as a flat 1-D tensor
// of `n` elements; same for `out`. Cast follows CANN's standard dtype matrix
// (F16 ↔ F32 both safe). Used to shuttle between the F32 residual accumulator
// and the F16 matmul/LayerNorm entry/exit points.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::cast_f32_to_f16_(const void *in_f32_dev,
                                               void *out_f16_dev,
                                               int64_t n) {
    if (!g_cann.aclnnCast || !g_cann.aclnnCastGetWorkspaceSize) {
        QIE_LOG("cast_f32_to_f16_: aclnnCast symbol missing");
        return false;
    }
    if (!in_f32_dev || !out_f16_dev) {
        QIE_LOG("cast_f32_to_f16_: null buffer (in=%p out=%p)", in_f32_dev,
                out_f16_dev);
        return false;
    }
    int64_t shape[1]   = {n};
    int64_t strides[1] = {1};
    aclTensor *t_in  = tensor_nd_f32(const_cast<void *>(in_f32_dev),
                                        1, shape, strides);
    aclTensor *t_out = tensor_nd_f16(out_f16_dev, 1, shape, strides);

    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    // ACL_FLOAT16 is the target dtype enum. The Cast API takes the destination
    // dtype as a plain int matching the aclDataType enum.
    aclnnStatus s = g_cann.aclnnCastGetWorkspaceSize(
        t_in, /*dtype=*/ ACL_FLOAT16, t_out, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                              compute_stream_);
    }
    g_cann.aclDestroyTensor(t_in);
    g_cann.aclDestroyTensor(t_out);
    if (s != 0) QIE_LOG("cast_f32_to_f16_: status=%d", (int)s);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Phase 4.4c: gated residual add into an F32 accumulator.
//
//   x_f32 += Cast(src_f16 * gate_f16, F32)
//
// `src_f16_dev` is [B, seq, hidden] F16; `gate_f16_dev` is [B, 1, hidden] F16
// broadcast along seq; `x_f32_dev` is [B, seq, hidden] F32.
//
// Staging:
//   scratch_mlp_dev_                (F16, sized [max_seq, ff_dim]) → tmp_f16
//     Large enough: hidden ≤ ff_dim so [seq, hidden] fits.
//   scratch_residual_tmp_f32_dev_   (F32, sized [max(img,txt), hidden]) →
//     tmp_f32
//
// Op trio: aclnnMul → aclnnCast(F16→F32) → aclnnInplaceAdd(F32+=F32).
// This is the F16-overflow fix for the real-GGUF forward path
// (see docs/qie_q2_phase4_smoke.md §4.4c).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::gated_residual_add_f32_(void *x_f32_dev,
                                                     const void *src_f16_dev,
                                                     const void *gate_f16_dev,
                                                     int64_t B, int64_t seq,
                                                     int64_t hidden) {
    if (!x_f32_dev || !src_f16_dev || !gate_f16_dev) {
        QIE_LOG("gate_add_f32_: null buffer (x=%p src=%p gate=%p)",
                x_f32_dev, src_f16_dev, gate_f16_dev);
        return false;
    }
    if (!g_cann.aclnnMul || !g_cann.aclnnCast ||
        !g_cann.aclnnInplaceAdd) {
        QIE_LOG("gate_add_f32_: missing aclnn symbols "
                "(Mul=%p Cast=%p InplaceAdd=%p)",
                (void *)g_cann.aclnnMul,
                (void *)g_cann.aclnnCast,
                (void *)g_cann.aclnnInplaceAdd);
        return false;
    }
    if (!scratch_residual_tmp_f32_dev_) {
        QIE_LOG("gate_add_f32_: scratch_residual_tmp_f32_dev_ not allocated");
        return false;
    }

    int64_t x_shape[3]   = {B, seq, hidden};
    int64_t x_strides[3];
    make_contig_strides(3, x_shape, x_strides);
    int64_t g_shape[3]   = {B, 1, hidden};
    int64_t g_strides[3] = {hidden, hidden, 1};

    aclTensor *t_x_f32  = tensor_nd_f32(x_f32_dev, 3, x_shape, x_strides);
    aclTensor *t_src_f16 = tensor_nd_f16(const_cast<void *>(src_f16_dev),
                                            3, x_shape, x_strides);
    aclTensor *t_g_f16   = tensor_nd_f16(const_cast<void *>(gate_f16_dev),
                                            3, g_shape, g_strides);
    // tmp_f16 sits in scratch_mlp_dev_ (oversized for any per-stream [seq,H]).
    aclTensor *t_tmp_f16 = tensor_nd_f16(scratch_mlp_dev_,
                                            3, x_shape, x_strides);
    aclTensor *t_tmp_f32 = tensor_nd_f32(scratch_residual_tmp_f32_dev_,
                                            3, x_shape, x_strides);

    aclnnStatus s = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;

    // 1. tmp_f16 = src_f16 * gate_f16 (broadcast over seq).
    s = g_cann.aclnnMulGetWorkspaceSize(t_src_f16, t_g_f16, t_tmp_f16,
                                           &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnMul(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                             compute_stream_);
    }
    if (s != 0) {
        QIE_LOG("gate_add_f32_: Mul status=%d", (int)s);
    }

    // 2. tmp_f32 = Cast(tmp_f16, F32).
    if (s == 0) {
        s = g_cann.aclnnCastGetWorkspaceSize(t_tmp_f16, ACL_FLOAT,
                                                t_tmp_f32, &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                   compute_stream_);
        }
        if (s != 0) QIE_LOG("gate_add_f32_: Cast(F16->F32) status=%d",
                              (int)s);
    }

    // 3. x_f32 += tmp_f32 (InplaceAdd with alpha=1.0 F32).
    if (s == 0) {
        float one = 1.0f;
        aclScalar *alpha =
            g_cann.aclCreateScalar(&one, ACL_FLOAT);
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x_f32, t_tmp_f32,
                                                      alpha, &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnInplaceAdd(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
        }
        if (alpha) g_cann.aclDestroyScalar(alpha);
        if (s != 0) QIE_LOG("gate_add_f32_: InplaceAdd(F32) status=%d",
                              (int)s);
    }

    g_cann.aclDestroyTensor(t_x_f32);
    g_cann.aclDestroyTensor(t_src_f16);
    g_cann.aclDestroyTensor(t_g_f16);
    g_cann.aclDestroyTensor(t_tmp_f16);
    g_cann.aclDestroyTensor(t_tmp_f32);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.4.5.4c: BF16-source variant of gated_residual_add_f32_. Used when the
// upstream matmul (FFN-down under QIE_FFN_DOWN_BF16=1) emits BF16 to escape
// F16's 65504 saturation. Computes (all on device):
//   src_f32  = Cast(src_bf16, F32)        — exact (BF16 range == F32 range)
//   gate_f32 = Cast(gate_f16, F32)        — exact
//   tmp_f32  = src_f32 * gate_f32_bcast   — F32 mul, no overflow path
//   x_f32   += tmp_f32                    — F32 inplace add
// Buffers:
//   - scratch_bf16_src_f32_dev_  (F32 [B, seq, hidden]) for src_f32
//   - scratch_residual_tmp_f32_dev_ (F32 [B, seq, hidden]) for tmp_f32 +
//     for the gate F32 broadcast intermediate (the gate cast lives in the
//     head of scratch_mlp_dev_ since aclnnMul wants its second operand
//     persistently mapped during the launch — we use a small lazy F32
//     gate-cast buffer carved from scratch_residual_tmp_f32_dev_'s tail
//     IF the F32 tmp doesn't span the full [seq,H] region; for safety we
//     just use scratch_bf16_src_f32_dev_ as src, scratch_residual_tmp_f32_
//     as tmp, and a tiny separate tail buffer for the gate F32. To avoid
//     introducing a fourth buffer, we sequence: cast src→f32 first, then
//     cast gate→f32 over scratch_mlp_dev_ (which is scratch space — never
//     read after gate use), then mul into scratch_residual_tmp_f32_dev_,
//     then InplaceAdd into x_f32_dev). The aclnnMul second-operand stride
//     pattern is broadcast-friendly so [B,1,hidden] gate × [B,seq,hidden]
//     src works without materializing the full broadcast.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::gated_residual_add_f32_bf16src_(
        void *x_f32_dev,
        const void *src_bf16_dev,
        const void *gate_f16_dev,
        int64_t B, int64_t seq, int64_t hidden) {
    if (!x_f32_dev || !src_bf16_dev || !gate_f16_dev) {
        QIE_LOG("gate_add_f32_bf16src_: null buffer (x=%p src=%p gate=%p)",
                x_f32_dev, (void *)src_bf16_dev, (void *)gate_f16_dev);
        return false;
    }
    if (!g_cann.aclnnMul || !g_cann.aclnnCast || !g_cann.aclnnInplaceAdd) {
        QIE_LOG("gate_add_f32_bf16src_: missing aclnn symbols");
        return false;
    }
    if (!scratch_residual_tmp_f32_dev_) {
        QIE_LOG("gate_add_f32_bf16src_: scratch_residual_tmp_f32_dev_ null");
        return false;
    }

    // Lazy-allocate F32 src buffer sized [B, seq, hidden].
    const size_t src_f32_bytes = (size_t)B * seq * hidden * sizeof(float);
    if (!ensure_dev_grow_(&scratch_bf16_src_f32_dev_,
                           &scratch_bf16_src_f32_bytes_,
                           src_f32_bytes)) {
        return false;
    }

    int64_t x_shape[3]   = {B, seq, hidden};
    int64_t x_strides[3];
    make_contig_strides(3, x_shape, x_strides);
    int64_t g_shape[3]   = {B, 1, hidden};
    int64_t g_strides[3] = {hidden, hidden, 1};

    aclTensor *t_x_f32   = tensor_nd_f32(x_f32_dev, 3, x_shape, x_strides);
    aclTensor *t_src_bf16 = tensor_nd_bf16(const_cast<void *>(src_bf16_dev),
                                             3, x_shape, x_strides);
    aclTensor *t_src_f32 = tensor_nd_f32(scratch_bf16_src_f32_dev_,
                                            3, x_shape, x_strides);
    aclTensor *t_g_f16   = tensor_nd_f16(const_cast<void *>(gate_f16_dev),
                                            3, g_shape, g_strides);
    // gate F32 broadcast tensor: tiny ([B, 1, hidden]) — reuse the head of
    // scratch_residual_tmp_f32_dev_. After the F32 mul writes to a different
    // region of the same buffer (we use scratch_mlp_dev_ for that), the gate
    // is already consumed.
    // Carve [B,1,hidden] F32 from scratch_residual_tmp_f32_dev_ tail. The
    // tmp F32 ([B, seq, hidden]) sits at the head — safe because tail offset
    // is past the head's reserved range. But the buffer is sized exactly
    // for [max_seq, hidden] F32 — no slack. So we use scratch_mlp_dev_
    // (sized [SEQ, FF_DIM] F16 = SEQ*FF_DIM*2 bytes ≈ 4×SEQ×H bytes; F32
    // gate fits trivially).
    aclTensor *t_g_f32   = tensor_nd_f32(scratch_mlp_dev_,
                                            3, g_shape, g_strides);
    aclTensor *t_tmp_f32 = tensor_nd_f32(scratch_residual_tmp_f32_dev_,
                                            3, x_shape, x_strides);

    aclnnStatus s = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;

    // 1. src_f32 = Cast(src_bf16, F32). Exact, no overflow.
    s = g_cann.aclnnCastGetWorkspaceSize(t_src_bf16, ACL_FLOAT, t_src_f32,
                                           &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                               compute_stream_);
    }
    if (s != 0) QIE_LOG("gate_add_f32_bf16src_: Cast(BF16->F32) status=%d",
                          (int)s);

    // 2. gate_f32 = Cast(gate_f16, F32).
    if (s == 0) {
        s = g_cann.aclnnCastGetWorkspaceSize(t_g_f16, ACL_FLOAT, t_g_f32,
                                               &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                   compute_stream_);
        }
        if (s != 0)
            QIE_LOG("gate_add_f32_bf16src_: Cast(gate F16->F32) status=%d",
                    (int)s);
    }

    // 3. tmp_f32 = src_f32 * gate_f32_bcast.
    if (s == 0) {
        s = g_cann.aclnnMulGetWorkspaceSize(t_src_f32, t_g_f32, t_tmp_f32,
                                               &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnMul(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                  compute_stream_);
        }
        if (s != 0) QIE_LOG("gate_add_f32_bf16src_: Mul(F32) status=%d",
                              (int)s);
    }

    // 4. x_f32 += tmp_f32.
    if (s == 0) {
        float one = 1.0f;
        aclScalar *alpha = g_cann.aclCreateScalar(&one, ACL_FLOAT);
        s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x_f32, t_tmp_f32, alpha,
                                                      &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnInplaceAdd(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
        }
        if (alpha) g_cann.aclDestroyScalar(alpha);
        if (s != 0)
            QIE_LOG("gate_add_f32_bf16src_: InplaceAdd(F32) status=%d", (int)s);
    }

    g_cann.aclDestroyTensor(t_x_f32);
    g_cann.aclDestroyTensor(t_src_bf16);
    g_cann.aclDestroyTensor(t_src_f32);
    g_cann.aclDestroyTensor(t_g_f16);
    g_cann.aclDestroyTensor(t_g_f32);
    g_cann.aclDestroyTensor(t_tmp_f32);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.3: affine-off LayerNorm via aclnnLayerNorm(gamma=null, beta=null).
// Input / output F16 [B, seq, hidden]. Normalize over the last dim.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::layer_norm_(void *x_f16_dev, void *out_f16_dev,
                                         int64_t B, int64_t seq,
                                         int64_t hidden) {
    if (!g_cann.aclnnLayerNorm || !g_cann.aclnnLayerNormGetWorkspaceSize) {
        QIE_LOG("layer_norm_: aclnnLayerNorm symbol missing");
        return false;
    }
    if (!g_cann.aclCreateIntArray) {
        QIE_LOG("layer_norm_: aclCreateIntArray symbol missing");
        return false;
    }

    int64_t shape[3]   = {B, seq, hidden};
    int64_t strides[3];
    make_contig_strides(3, shape, strides);
    aclTensor *t_in  = tensor_nd_f16(x_f16_dev,    3, shape, strides);
    aclTensor *t_out = tensor_nd_f16(out_f16_dev,  3, shape, strides);

    // Normalized shape = [hidden].
    int64_t norm_shape_arr[1] = {hidden};
    aclIntArray *norm_shape = g_cann.aclCreateIntArray(norm_shape_arr, 1);

    // mean/rstd tensors: shape [B, seq, 1] F32. Stored contig in
    // mean_dev_ / ln_rstd_dev_ (sized for SEQ rows).
    int64_t mr_shape[3]   = {B, seq, 1};
    int64_t mr_strides[3];
    make_contig_strides(3, mr_shape, mr_strides);
    aclTensor *t_mean = tensor_nd_f32(mean_dev_,    3, mr_shape, mr_strides);
    aclTensor *t_rstd = tensor_nd_f32(ln_rstd_dev_, 3, mr_shape, mr_strides);

    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnLayerNormGetWorkspaceSize(
        t_in, norm_shape,
        /*weight*/ nullptr, /*bias*/ nullptr,
        (double)cfg_.layernorm_eps,
        t_out, t_mean, t_rstd,
        &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnLayerNorm(ws > 0 ? workspace_dev_ : nullptr,
                                    ws, exec, compute_stream_);
    }
    if (s != 0) QIE_LOG("layer_norm_: status=%d", (int)s);

    g_cann.aclDestroyIntArray(norm_shape);
    g_cann.aclDestroyTensor(t_in);
    g_cann.aclDestroyTensor(t_out);
    g_cann.aclDestroyTensor(t_mean);
    g_cann.aclDestroyTensor(t_rstd);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Phase 4.4c: F32 → F16 LayerNorm. Runs aclnnLayerNorm in F32 (input/output)
// so the normalization denominator `sqrt(var + eps)` does not see Inf when
// the F32 residual magnitude has grown past the F16 range (observed at
// depth N ≥ ~35 on real Q4_0 weights). The normalized output has std ≈ 1
// regardless of input scale, so down-casting to F16 afterward is safe.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::layer_norm_f32_to_f16_(const void *x_f32_dev,
                                                     void *out_f16_dev,
                                                     int64_t B, int64_t seq,
                                                     int64_t hidden) {
    if (!g_cann.aclnnLayerNorm || !g_cann.aclnnLayerNormGetWorkspaceSize) {
        QIE_LOG("layer_norm_f32_to_f16_: aclnnLayerNorm symbol missing");
        return false;
    }
    if (!g_cann.aclnnCast || !g_cann.aclnnCastGetWorkspaceSize) {
        QIE_LOG("layer_norm_f32_to_f16_: aclnnCast symbol missing");
        return false;
    }
    if (!g_cann.aclCreateIntArray) {
        QIE_LOG("layer_norm_f32_to_f16_: aclCreateIntArray symbol missing");
        return false;
    }
    if (!scratch_residual_tmp_f32_dev_) {
        QIE_LOG("layer_norm_f32_to_f16_: scratch_residual_tmp_f32_dev_ null");
        return false;
    }

    int64_t shape[3]   = {B, seq, hidden};
    int64_t strides[3];
    make_contig_strides(3, shape, strides);
    aclTensor *t_in   = tensor_nd_f32(const_cast<void *>(x_f32_dev),
                                        3, shape, strides);
    aclTensor *t_tmp  = tensor_nd_f32(scratch_residual_tmp_f32_dev_,
                                        3, shape, strides);

    // Normalized shape = [hidden].
    int64_t norm_shape_arr[1] = {hidden};
    aclIntArray *norm_shape = g_cann.aclCreateIntArray(norm_shape_arr, 1);

    // mean/rstd tensors: F32 [B, seq, 1] — reused from the F16 LayerNorm
    // path (aclnnLayerNorm mandates F32 regardless of input dtype).
    int64_t mr_shape[3]   = {B, seq, 1};
    int64_t mr_strides[3];
    make_contig_strides(3, mr_shape, mr_strides);
    aclTensor *t_mean = tensor_nd_f32(mean_dev_,    3, mr_shape, mr_strides);
    aclTensor *t_rstd = tensor_nd_f32(ln_rstd_dev_, 3, mr_shape, mr_strides);

    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnLayerNormGetWorkspaceSize(
        t_in, norm_shape,
        /*weight*/ nullptr, /*bias*/ nullptr,
        (double)cfg_.layernorm_eps,
        t_tmp, t_mean, t_rstd,
        &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnLayerNorm(ws > 0 ? workspace_dev_ : nullptr,
                                    ws, exec, compute_stream_);
    }
    if (s != 0) QIE_LOG("layer_norm_f32_to_f16_: LayerNorm(F32) status=%d",
                          (int)s);

    g_cann.aclDestroyIntArray(norm_shape);
    g_cann.aclDestroyTensor(t_in);
    g_cann.aclDestroyTensor(t_tmp);
    g_cann.aclDestroyTensor(t_mean);
    g_cann.aclDestroyTensor(t_rstd);

    if (s != 0) return false;

    // Cast the normalized F32 output → F16 for downstream matmul/modulate.
    // After normalization values are bounded ~1σ so the F16 cast is safe.
    return cast_f32_to_f16_(scratch_residual_tmp_f32_dev_, out_f16_dev,
                              B * seq * hidden);
}

// ---------------------------------------------------------------------------
// Phase 4.5 Step 4: RMSNorm over last dim `inner`, F32 in → F16 out. Used
// for the global `txt_norm` (RMSNorm over joint_attention_dim=3584) on the
// raw text-encoder conditioning before `txt_in` matmul. F32-in is required
// because the host-dumped txt conditioning arrives F32 and we keep the
// numerically sensitive normalization in F32 (joint_dim=3584 is large
// enough that F16 accumulator overflow is plausible). The bounded ~1σ
// output casts cleanly to F16 for the subsequent Q4_0 dispatch_matmul_.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::rms_norm_row_f32_to_f16_(const void *x_f32_dev,
                                                      void *out_f16_dev,
                                                      const void *gamma_f32_dev,
                                                      int64_t rows,
                                                      int64_t inner) {
    if (!g_cann.aclnnRmsNorm || !g_cann.aclnnRmsNormGetWorkspaceSize) {
        QIE_LOG("rms_norm_row_f32_to_f16_: aclnnRmsNorm symbol missing");
        return false;
    }
    if (!g_cann.aclnnCast || !g_cann.aclnnCastGetWorkspaceSize) {
        QIE_LOG("rms_norm_row_f32_to_f16_: aclnnCast symbol missing");
        return false;
    }
    if (!scratch_residual_tmp_f32_dev_) {
        QIE_LOG("rms_norm_row_f32_to_f16_: scratch_residual_tmp_f32_dev_ null");
        return false;
    }
    if (!gamma_f32_dev) {
        QIE_LOG("rms_norm_row_f32_to_f16_: null gamma");
        return false;
    }

    // Input / output F32 shape [rows, inner].
    int64_t x_shape[2]   = {rows, inner};
    int64_t x_strides[2] = {inner, 1};
    aclTensor *t_in  = tensor_nd_f32(const_cast<void *>(x_f32_dev),
                                       2, x_shape, x_strides);
    aclTensor *t_out = tensor_nd_f32(scratch_residual_tmp_f32_dev_,
                                       2, x_shape, x_strides);

    int64_t g_shape[1]   = {inner};
    int64_t g_strides[1] = {1};
    aclTensor *t_g = tensor_nd_f32(const_cast<void *>(gamma_f32_dev),
                                     1, g_shape, g_strides);

    // rstd [rows, 1] F32 — rstd_dev_ is sized for num_heads*SEQ so it's
    // plenty for `rows = txt_seq` in the worst case.
    int64_t r_shape[2]   = {rows, 1};
    int64_t r_strides[2] = {1, 1};
    aclTensor *t_rstd = tensor_nd_f32(rstd_dev_, 2, r_shape, r_strides);

    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnRmsNormGetWorkspaceSize(
        t_in, t_g, (double)cfg_.rms_norm_eps,
        t_out, t_rstd, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnRmsNorm(ws > 0 ? workspace_dev_ : nullptr,
                                  ws, exec, compute_stream_);
    }
    if (s != 0) QIE_LOG("rms_norm_row_f32_to_f16_: RmsNorm(F32) status=%d",
                          (int)s);

    g_cann.aclDestroyTensor(t_in);
    g_cann.aclDestroyTensor(t_out);
    g_cann.aclDestroyTensor(t_g);
    g_cann.aclDestroyTensor(t_rstd);

    if (s != 0) return false;

    return cast_f32_to_f16_(scratch_residual_tmp_f32_dev_, out_f16_dev,
                              rows * inner);
}

// ---------------------------------------------------------------------------
// Q2.3: RMSNorm across the head_dim axis. Input is [rows, head_dim] F16;
// gamma is F32 [head_dim]. rstd_dev_ reused (F32 [rows]).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::rms_norm_head_(void *x_f16_dev, void *out_f16_dev,
                                             void *gamma_f32_dev,
                                             int64_t rows, int64_t head_dim) {
    if (!g_cann.aclnnRmsNorm || !gamma_f32_dev) {
        QIE_LOG("rms_norm_head_: missing symbol or gamma (gamma=%p)",
                gamma_f32_dev);
        return false;
    }

    int64_t x_shape[2]   = {rows, head_dim};
    int64_t x_strides[2] = {head_dim, 1};
    aclTensor *t_in  = tensor_nd_f16(x_f16_dev,   2, x_shape, x_strides);
    aclTensor *t_out = tensor_nd_f16(out_f16_dev, 2, x_shape, x_strides);

    int64_t g_shape[1]   = {head_dim};
    int64_t g_strides[1] = {1};
    aclTensor *t_g = tensor_nd_f32(gamma_f32_dev, 1, g_shape, g_strides);

    int64_t r_shape[2]   = {rows, 1};
    int64_t r_strides[2] = {1, 1};
    aclTensor *t_rstd = tensor_nd_f32(rstd_dev_, 2, r_shape, r_strides);

    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnRmsNormGetWorkspaceSize(
        t_in, t_g, (double)cfg_.rms_norm_eps, t_out, t_rstd, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnRmsNorm(ws > 0 ? workspace_dev_ : nullptr,
                                  ws, exec, compute_stream_);
    }
    if (s != 0) QIE_LOG("rms_norm_head_: status=%d", (int)s);
    g_cann.aclDestroyTensor(t_in);
    g_cann.aclDestroyTensor(t_out);
    g_cann.aclDestroyTensor(t_g);
    g_cann.aclDestroyTensor(t_rstd);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.4.1 (Phase 4.1): apply 3D-axial RoPE to a Q or K tensor.
//
// The CPU reference (`Rope::apply_rope` in rope.hpp:603) implements the
// rotation as a repeat-then-pair-mul-and-add over two pe slices:
//   pe = [cos, -sin; sin, cos]
//   x_pair_0 = repeat(x[..., 0], 2)   // x0 x0
//   x_pair_1 = repeat(x[..., 1], 2)   // x1 x1
//   out = x_pair_0 * pe[0] + x_pair_1 * pe[1]
// where `pe[0] = [cos, -sin]` and `pe[1] = [sin, cos]`, producing:
//   y_even = x_even * cos + x_odd  * sin        (output at offsets 2*dp)
//   y_odd  = x_odd  * cos - x_even * sin        (output at offsets 2*dp+1)
//
// Phase 3 used a host round-trip (D2H → rotate in F32 → H2D) — bit-accurate
// but catastrophic on PCIe traffic (~96 GiB per image at seq=4352 × 60
// blocks × 20 steps × 2 CFG). Phase 4.1 dispatches this on-device via
// aclnnMul + aclnnAdd with strided views over separate cos/sin tables.
//
// `pe_row_offset` selects the starting row in the cos/sin tables: 0 for the
// txt stream, ctx_len for the img stream per `compute_qwen_rope_pe_host`.
//
// The `pe_f16_dev` argument is retained for signature-compat with the Phase 3
// smoke harness; the on-device path indexes `global_w_.rope_cos_dev` and
// `rope_sin_dev` (populated by the same compute_qwen_rope_pe_host call).
//
// Fallback to host-side round-trip is available via `QIE_ROPE_HOST=1` env
// var — used only by parity probes.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::apply_rope_(void *x_f16_dev,
                                         const void *pe_f16_dev,
                                         int64_t pe_row_offset,
                                         int64_t B, int64_t seq,
                                         int64_t n_heads, int64_t head_dim) {
    // Phase 4.1 STATUS: on-device dispatch is in-flight but the CANN
    // aclnnRotaryPositionEmbedding mode=2 (interleave) parity is still RED
    // at the time of this landing (see docs/qie_q2_phase4_smoke.md §4.1).
    // Default path remains the Phase 3 host round-trip so the block-forward
    // smoke + upstream Phase 4.2 plumbing stay GREEN. Opt-in to the on-device
    // experimental path via QIE_ROPE_DEVICE=1 — used by the Q2.4.1 probe
    // during bring-up.
    static const bool force_device = [] {
        const char *v = std::getenv("QIE_ROPE_DEVICE");
        return v && *v && v[0] != '0';
    }();
    if (force_device) {
        (void)pe_f16_dev;
        return apply_rope_on_device_(x_f16_dev, pe_row_offset,
                                       B, seq, n_heads, head_dim);
    }
    return apply_rope_host_(x_f16_dev, pe_f16_dev, pe_row_offset,
                              B, seq, n_heads, head_dim);
}

// Phase 3 reference — preserved for parity testing.
bool ImageDiffusionEngine::apply_rope_host_(void *x_f16_dev,
                                              const void *pe_f16_dev,
                                              int64_t pe_row_offset,
                                              int64_t B, int64_t seq,
                                              int64_t n_heads, int64_t head_dim) {
    if (!x_f16_dev || !pe_f16_dev) return false;
    const int64_t half = head_dim / 2;
    const size_t  n_elt = (size_t)B * seq * n_heads * head_dim;

    // Host round-trip: D2H → rotate in F32 → H2D. Bit-accurate reference
    // for Phase 3 smoke. Replace with an on-device kernel in Phase 3.1.
    std::vector<uint16_t> x_host(n_elt);
    std::vector<uint16_t> pe_host((size_t)(pe_row_offset + seq) *
                                   (size_t)half * 2 * 2);

    aclError err = g_cann.aclrtSynchronizeStream(compute_stream_);
    if (err != 0) return false;
    err = g_cann.aclrtMemcpy(x_host.data(), x_host.size() * sizeof(uint16_t),
                              x_f16_dev, x_host.size() * sizeof(uint16_t),
                              ACL_MEMCPY_DEVICE_TO_HOST);
    if (err != 0) return false;
    err = g_cann.aclrtMemcpy(pe_host.data(),
                              pe_host.size() * sizeof(uint16_t),
                              const_cast<void *>(pe_f16_dev),
                              pe_host.size() * sizeof(uint16_t),
                              ACL_MEMCPY_DEVICE_TO_HOST);
    if (err != 0) return false;

    // x layout [B, seq, n_heads, head_dim] contig. For each element pair
    // (d = 2*dp, d+1) at position (b, s, h): rotate using
    // pe[pe_row_offset + s, dp, :, :].
    auto f16_to_f32 = [](uint16_t bits) -> float {
        __fp16 h; std::memcpy(&h, &bits, sizeof(h)); return (float)h;
    };
    auto f32_to_f16 = [](float v) -> uint16_t {
        __fp16 h = (__fp16)v;
        uint16_t out; std::memcpy(&out, &h, sizeof(out));
        return out;
    };

    for (int64_t b = 0; b < B; ++b) {
        for (int64_t s = 0; s < seq; ++s) {
            const size_t pe_base =
                (size_t)(pe_row_offset + s) * half * 4;  // 4 = 2·2
            for (int64_t h = 0; h < n_heads; ++h) {
                const size_t row_base =
                    (((size_t)b * seq + s) * n_heads + h) * head_dim;
                for (int64_t dp = 0; dp < half; ++dp) {
                    const size_t pe_here = pe_base + (size_t)dp * 4;
                    // pe layout: [cos, -sin, sin, cos]
                    const float pe00 = f16_to_f32(pe_host[pe_here + 0]);
                    const float pe01 = f16_to_f32(pe_host[pe_here + 1]);
                    const float pe10 = f16_to_f32(pe_host[pe_here + 2]);
                    const float pe11 = f16_to_f32(pe_host[pe_here + 3]);
                    // x layout: interleaved pairs (x0, x1) at positions
                    // (2*dp, 2*dp+1).
                    const size_t i0 = row_base + (size_t)(2 * dp);
                    const size_t i1 = row_base + (size_t)(2 * dp + 1);
                    const float x0 = f16_to_f32(x_host[i0]);
                    const float x1 = f16_to_f32(x_host[i1]);
                    // Match rope.hpp:623-636 exactly:
                    //   x_0 (broadcast) has shape [d/2, 2] filled with x0 x0
                    //   x_1 (broadcast) has shape [d/2, 2] filled with x1 x1
                    //   out = x_0 * pe[0]  +  x_1 * pe[1]
                    // where pe[0] = (cos, -sin), pe[1] = (sin, cos). So
                    //   out[0] = x0 * cos + x1 * sin
                    //   out[1] = x0 * (-sin) + x1 * cos
                    x_host[i0] = f32_to_f16(x0 * pe00 + x1 * pe10);
                    x_host[i1] = f32_to_f16(x0 * pe01 + x1 * pe11);
                }
            }
        }
    }

    err = g_cann.aclrtMemcpy(x_f16_dev, x_host.size() * sizeof(uint16_t),
                              x_host.data(), x_host.size() * sizeof(uint16_t),
                              ACL_MEMCPY_HOST_TO_DEVICE);
    return err == 0;
}

// ---------------------------------------------------------------------------
// Phase 4.1: on-device interleaved RoPE.
//
// x has layout [B, seq, NH, head_dim] F16 contiguous. Half head_dim = half.
// Logically we need:
//   y_even[b,s,h,dp] = x_even[b,s,h,dp] * cos[s,dp] + x_odd[b,s,h,dp] * sin[s,dp]
//   y_odd [b,s,h,dp] = x_odd [b,s,h,dp] * cos[s,dp] - x_even[b,s,h,dp] * sin[s,dp]
// where x_even = x[..., 2*dp]  and  x_odd = x[..., 2*dp+1], interleaved.
//
// Phase 3 used a D2H→F32→H2D round-trip (~ host ms per call) as a bit-exact
// baseline. Phase 4.1 dispatches on-device via:
//   (1) 4× aclnnMul with strided-input views + broadcast cos/sin
//   (2) 2× aclnnAdd (alpha-weighted) into contiguous scratch
//   (3) 2× aclnnInplaceCopy to scatter the contiguous scratch back into the
//       strided even/odd views of x (cannot rely on aclnnAdd accepting a
//       strided OUTPUT tensor — empirical result: it does not scatter).
//
// Scratch usage (`scratch_rope_{a,b,c}_dev_` each `[B, seq, NH, half]` F16):
//   C ← x_even * cos
//   A ← x_odd  * sin
//   C ← C + A               (y_even; A and C free)
//   A ← x_odd  * cos
//   B ← x_even * sin        (must read x_even BEFORE it gets scatter-overwritten)
//   A ← A + (-1)*B          (y_odd; B free)
//   scatter-copy C → x_even_view  (in-place copy to stride-2 positions)
//   scatter-copy A → x_odd_view
//
// Scatter order matters: copy y_even first (reads stride-0 positions of x
// which are x_even that we no longer need), then copy y_odd. Since we
// captured both x_even*sin and x_odd*cos before scattering, neither copy
// depends on live x values anymore.
//
// 4 muls + 2 adds + 2 copies = 8 dispatches — all short, no host round-trip.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::apply_rope_on_device_(void *x_f16_dev,
                                                   int64_t pe_row_offset,
                                                   int64_t B, int64_t seq,
                                                   int64_t n_heads,
                                                   int64_t head_dim) {
    if (!x_f16_dev) return false;
    if (!g_cann.aclnnRotaryPositionEmbedding ||
        !g_cann.aclnnRotaryPositionEmbeddingGetWorkspaceSize) {
        QIE_LOG("apply_rope_on_device_: aclnnRotaryPositionEmbedding missing");
        return false;
    }
    // Default backend: "rope" (use aclnnRotaryPositionEmbedding). Fallback to
    // "manual" (hand-rolled 4Mul+2Add+2Copy) via env var — used during the
    // Q2.4.1 bring-up while the interleave-mode op is being debugged.
    const char *backend = std::getenv("QIE_ROPE_BACKEND");
    if (backend && std::strcmp(backend, "manual") == 0) {
        return apply_rope_manual_(x_f16_dev, pe_row_offset, B, seq,
                                     n_heads, head_dim);
    }

    const int64_t HD   = head_dim;
    const int64_t half = HD / 2;

    if (!scratch_rope_cos_full_dev_ || !scratch_rope_sin_full_dev_) {
        QIE_LOG("apply_rope_on_device_: full cos/sin tables not allocated");
        return false;
    }
    if (!global_w_.rope_cos_dev || !global_w_.rope_sin_dev) {
        QIE_LOG("apply_rope_on_device_: half cos/sin tables not allocated");
        return false;
    }

    // x view: [B, seq, NH, HD] contig. Output goes in-place (out = x).
    const int64_t x_shape[4]   = {B, seq, n_heads, HD};
    int64_t       x_strides[4];
    make_contig_strides(4, x_shape, x_strides);
    const int64_t x_storage[1] = {B * seq * n_heads * HD};
    aclTensor *t_x  = g_cann.aclCreateTensor(x_shape, 4, ACL_FLOAT16,
                                               x_strides, 0, ACL_FORMAT_ND,
                                               x_storage, 1, x_f16_dev);
    aclTensor *t_x_out = g_cann.aclCreateTensor(x_shape, 4, ACL_FLOAT16,
                                                   x_strides, 0, ACL_FORMAT_ND,
                                                   x_storage, 1, x_f16_dev);

    // Mode + cos/sin layout selection via env. Defaults to mode=2 + full-HD
    // cos/sin (duplicated per-pair). QIE_ROPE_COS_LAYOUT=half points at the
    // half-sized cos/sin (one entry per pair) for ops that expect that shape.
    int64_t mode = 2;
    if (const char *m = std::getenv("QIE_ROPE_MODE")) mode = std::atoll(m);
    const bool cos_half = [] {
        const char *c = std::getenv("QIE_ROPE_COS_LAYOUT");
        return c && std::strcmp(c, "half") == 0;
    }();

    const int64_t cs_last = cos_half ? half : HD;
    void *cos_base = cos_half ? global_w_.rope_cos_dev
                                : scratch_rope_cos_full_dev_;
    void *sin_base = cos_half ? global_w_.rope_sin_dev
                                : scratch_rope_sin_full_dev_;
    const int64_t cs_shape[4]   = {1, seq, 1, cs_last};
    const int64_t cs_strides[4] = {0, cs_last, 0, 1};
    const int64_t cs_storage[1] = {seq * cs_last};
    auto make_cs_view = [&](void *base) -> aclTensor * {
        uint8_t *sliced = (uint8_t *)base +
                           (size_t)pe_row_offset * cs_last *
                           sizeof(uint16_t);
        return g_cann.aclCreateTensor(cs_shape, 4, ACL_FLOAT16,
                                         cs_strides, 0, ACL_FORMAT_ND,
                                         cs_storage, 1, sliced);
    };
    aclTensor *t_cos = make_cs_view(cos_base);
    aclTensor *t_sin = make_cs_view(sin_base);

    uint64_t ws = 0; aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnRotaryPositionEmbeddingGetWorkspaceSize(
        t_x, t_cos, t_sin, mode, t_x_out, &ws, &exec);
    if (s == 0) {
        ensure_workspace_(ws);
        s = g_cann.aclnnRotaryPositionEmbedding(
            ws > 0 ? workspace_dev_ : nullptr, ws, exec, compute_stream_);
    }
    if (s != 0) {
        QIE_LOG("apply_rope_on_device_: mode=%lld cs_half=%d status=%d",
                (long long)mode, (int)cos_half, (int)s);
    }

    g_cann.aclDestroyTensor(t_x);
    g_cann.aclDestroyTensor(t_x_out);
    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Stashed Q2.4.1 hand-rolled 4Mul+2Add RoPE (kept as a fallback path in case
// aclnnRotaryPositionEmbedding's interleave mode proves unusable on a given
// CANN version). Selected via env QIE_ROPE_BACKEND=manual.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::apply_rope_manual_(void *x_f16_dev,
                                                 int64_t pe_row_offset,
                                                 int64_t B, int64_t seq,
                                                 int64_t n_heads,
                                                 int64_t head_dim) {
    if (!x_f16_dev) return false;
    if (!global_w_.rope_cos_dev || !global_w_.rope_sin_dev) {
        QIE_LOG("apply_rope_manual_: cos/sin tables not allocated");
        return false;
    }
    if (!g_cann.aclnnMul || !g_cann.aclnnAdd || !g_cann.aclnnInplaceCopy) {
        return false;
    }
    if (!scratch_rope_a_dev_ || !scratch_rope_b_dev_ || !scratch_rope_c_dev_) {
        return false;
    }

    const int64_t half = head_dim / 2;
    const int64_t HD   = head_dim;

    const int64_t x_storage[1] = {B * seq * n_heads * HD};
    const int64_t half_shape[4]   = {B, seq, n_heads, half};
    const int64_t x_strides_stride2[4] = {
        seq * n_heads * HD,
        n_heads * HD,
        HD,
        2
    };
    auto make_x_view = [&](int64_t elem_off) -> aclTensor * {
        uint8_t *sliced = (uint8_t *)x_f16_dev +
                           (size_t)elem_off * sizeof(uint16_t);
        return g_cann.aclCreateTensor(half_shape, 4, ACL_FLOAT16,
                                         x_strides_stride2, 0, ACL_FORMAT_ND,
                                         x_storage, 1, sliced);
    };
    aclTensor *t_x_even = make_x_view(0);
    aclTensor *t_x_odd  = make_x_view(1);

    // cos/sin views over the NH-pre-broadcast tiles. Shape matches A's
    // contig shape [B=1, seq, NH, half] exactly — no broadcasting needed at
    // aclnnMul dispatch. Slice each base pointer to pe_row_offset * NH * half
    // elements via byte arithmetic so storage descriptor starts at row 0.
    if (!scratch_rope_cos_bcast_dev_ || !scratch_rope_sin_bcast_dev_) {
        QIE_LOG("apply_rope_on_device_: pre-broadcast cos/sin tiles missing "
                "(cos=%p sin=%p)",
                scratch_rope_cos_bcast_dev_, scratch_rope_sin_bcast_dev_);
        return false;
    }
    const int64_t cs_shape[4]   = {1, seq, n_heads, half};
    int64_t cs_strides[4];
    make_contig_strides(4, cs_shape, cs_strides);
    const int64_t cs_storage[1] = {seq * n_heads * half};
    auto make_cs_view = [&](void *base) -> aclTensor * {
        uint8_t *sliced = (uint8_t *)base +
                           (size_t)pe_row_offset * n_heads * half *
                           sizeof(uint16_t);
        return g_cann.aclCreateTensor(cs_shape, 4, ACL_FLOAT16,
                                         cs_strides, 0, ACL_FORMAT_ND,
                                         cs_storage, 1, sliced);
    };
    aclTensor *t_cos = make_cs_view(scratch_rope_cos_bcast_dev_);
    aclTensor *t_sin = make_cs_view(scratch_rope_sin_bcast_dev_);

    // Scratch contiguous [B, seq, NH, half] F16.
    const int64_t scratch_storage[1] = {B * seq * n_heads * half};
    int64_t scratch_strides[4];
    make_contig_strides(4, half_shape, scratch_strides);
    auto make_scratch = [&](void *base) -> aclTensor * {
        return g_cann.aclCreateTensor(half_shape, 4, ACL_FLOAT16,
                                         scratch_strides, 0, ACL_FORMAT_ND,
                                         scratch_storage, 1, base);
    };
    aclTensor *t_A = make_scratch(scratch_rope_a_dev_);
    aclTensor *t_B = make_scratch(scratch_rope_b_dev_);
    aclTensor *t_C = make_scratch(scratch_rope_c_dev_);
    // 4th scratch buffer carved from scratch_mlp_dev_ (oversized 107 MiB at
    // worst case; apply_rope_ runs outside modulate_/gated_residual_add_ so
    // no simultaneous use). Pre-initialized here so it is in scope for every
    // goto-done path below.
    aclTensor *t_D = make_scratch(scratch_mlp_dev_);

    aclnnStatus s = 0;
    uint64_t ws = 0;
    aclOpExecutor *exec = nullptr;

    auto do_mul = [&](aclTensor *a, aclTensor *b, aclTensor *out) -> aclnnStatus {
        ws = 0; exec = nullptr;
        aclnnStatus st = g_cann.aclnnMulGetWorkspaceSize(a, b, out, &ws, &exec);
        if (st != 0) return st;
        ensure_workspace_(ws);
        return g_cann.aclnnMul(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                 compute_stream_);
    };
    auto do_add = [&](aclTensor *x, aclTensor *y, float alpha,
                      aclTensor *out) -> aclnnStatus {
        ws = 0; exec = nullptr;
        aclScalar *a = make_f16_scalar_local(alpha);
        aclnnStatus st = g_cann.aclnnAddGetWorkspaceSize(x, y, a, out,
                                                           &ws, &exec);
        if (st != 0) { g_cann.aclDestroyScalar(a); return st; }
        ensure_workspace_(ws);
        st = g_cann.aclnnAdd(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                compute_stream_);
        g_cann.aclDestroyScalar(a);
        return st;
    };
    auto do_copy = [&](aclTensor *dst, const aclTensor *src) -> aclnnStatus {
        ws = 0; exec = nullptr;
        aclnnStatus st = g_cann.aclnnInplaceCopyGetWorkspaceSize(dst, src,
                                                                    &ws, &exec);
        if (st != 0) return st;
        ensure_workspace_(ws);
        return g_cann.aclnnInplaceCopy(ws > 0 ? workspace_dev_ : nullptr, ws,
                                          exec, compute_stream_);
    };

    // Phase 4.1 approach: gather x_even / x_odd into contiguous scratches
    // first, run 4 muls + 2 adds on contiguous tensors, then scatter
    // y_even / y_odd back to x via InplaceCopy. Four scratch slots
    // (A=x_even, B=x_odd, C=y_even, D=y_odd / intermediate) are used.
    //
    // 1. A = gather x_even  (contig = strided copy)
    s = do_copy(t_A, t_x_even);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Copy(gather x_even) status=%d",
                           (int)s); goto done; }
    // 2. B = gather x_odd
    s = do_copy(t_B, t_x_odd);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Copy(gather x_odd) status=%d",
                           (int)s); goto done; }
    // 3. C = A * cos                  (x_even * cos)
    s = do_mul(t_A, t_cos, t_C);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Mul(A*cos) status=%d",
                           (int)s); goto done; }
    // 4. D = B * sin                  (x_odd * sin)
    s = do_mul(t_B, t_sin, t_D);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Mul(B*sin) status=%d",
                           (int)s); goto done; }
    // 5. C = C + D                    (y_even = x_even*cos + x_odd*sin)
    s = do_add(t_C, t_D, 1.0f, t_C);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Add(y_even) status=%d",
                           (int)s); goto done; }

    // 6. D = B * cos                  (x_odd * cos)
    s = do_mul(t_B, t_cos, t_D);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Mul(B*cos) status=%d",
                           (int)s); goto done; }
    // 7. A = A * sin                  (x_even * sin; overwrite A — we already consumed it)
    s = do_mul(t_A, t_sin, t_A);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Mul(A*sin) status=%d",
                           (int)s); goto done; }
    // 8. D = D + (-1)*A               (y_odd = x_odd*cos - x_even*sin)
    s = do_add(t_D, t_A, -1.0f, t_D);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Add(y_odd) status=%d",
                           (int)s); goto done; }

    // 9. Scatter y_even back: x_even_view ← C
    s = do_copy(t_x_even, t_C);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Copy(scatter y_even) status=%d",
                           (int)s); goto done; }
    // 10. Scatter y_odd back: x_odd_view ← D
    s = do_copy(t_x_odd, t_D);
    if (s != 0) { QIE_LOG("apply_rope_on_device_: Copy(scatter y_odd) status=%d",
                           (int)s); goto done; }

done:
    g_cann.aclDestroyTensor(t_x_even);
    g_cann.aclDestroyTensor(t_x_odd);
    g_cann.aclDestroyTensor(t_cos);
    g_cann.aclDestroyTensor(t_sin);
    g_cann.aclDestroyTensor(t_A);
    g_cann.aclDestroyTensor(t_B);
    g_cann.aclDestroyTensor(t_C);
    g_cann.aclDestroyTensor(t_D);
    return s == 0;
}

// ---------------------------------------------------------------------------
// Q2.3 MAIN: one transformer-block forward on NPU.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::forward_block_(const DiTLayerWeights &lw,
                                             void *img_hidden, int64_t img_seq,
                                             void *txt_hidden, int64_t txt_seq,
                                             void *t_emb,
                                             void *pe) {
    if (!ready_) return false;
    const int64_t B    = 1;
    const int64_t H    = cfg_.hidden_size;        // 3072
    const int64_t HD   = cfg_.head_dim;           // 128
    const int64_t NH   = cfg_.num_heads;          // 24
    const int64_t FF   = (int64_t)H * cfg_.ff_mult;  // 12288
    const int64_t seq_total = img_seq + txt_seq;

    // Phase 4.5.4b intra-block NaN bisect. Gated by
    // QIE_DEBUG_INTRA_BLOCK0=1 — scans only the FIRST invocation of
    // forward_block_ in the process lifetime (i.e., block 0, step 0).
    static int  s_intra = -1;
    static int  s_intra_calls = 0;
    if (s_intra < 0) {
        const char *v = std::getenv("QIE_DEBUG_INTRA_BLOCK0");
        s_intra = v ? std::atoi(v) : 0;
    }
    const bool do_intra = s_intra && (s_intra_calls == 0);
    s_intra_calls++;

    // Q2.4.5.4c BF16 plumbing for FFN-down. Cached once (one std::getenv per
    // process). Defaults OFF so the synthetic-weight Step 1 denoise_loop_test
    // path (cos_sim=1.0 against CPU reference) is byte-identical to before.
    // QIE_FFN_DOWN_BF16=1 → emit BF16 from the ff_down WQBMMv3 dispatch and
    // route the gated-residual #2 add through gated_residual_add_f32_bf16src_
    // (cast BF16 src → F32 before the gate-mul, escaping F16 65504 saturation).
    //
    // Q2.4.5.4d Step 7 widening (this commit):
    // QIE_ALL_BF16=1 → in addition to ff_down, also emit BF16 from the
    // attention-output projections (`to_add_out` for txt and `to_out_0` for
    // img — block residual #1 contributors) and route gated-residual #1
    // through the same BF16-src variant. §5.5.4 receipt 2 surfaced a leak
    // at block 1 IMG once block 0 ff_down was BF16-clean: with the F32
    // residual stream now safely holding 7.2M magnitudes, block 1's
    // post-LN/post-modulate inputs are still bounded but the per-channel
    // matmul outputs for attn-out + ff-down can both saturate at F16 max
    // once the residual contribution accumulates. Both contributors must
    // be BF16 to keep the residual additive chain overflow-free. Other
    // matmul callsites (Q/K/V projections, FFN-up, modulation linear,
    // norm_out / proj_out / time_linear / img_in / txt_in) are NOT
    // promoted under QIE_ALL_BF16 because their downstream consumers
    // (RMSNorm + RoPE + FIA, GELU, modulate, Cast/SiLU, CFG-compose) are
    // F16-strict and the magnitudes there are bounded by the upstream
    // LN/SiLU normalization (max ≤ ~3700 observed at block 0, far below
    // F16's 65504). The leak surface is solely the residual contributors.
    static int s_ffn_down_bf16 = -1;
    static int s_all_bf16      = -1;
    if (s_ffn_down_bf16 < 0) {
        const char *v_specific = std::getenv("QIE_FFN_DOWN_BF16");
        const char *v_all      = std::getenv("QIE_ALL_BF16");
        int specific = v_specific ? std::atoi(v_specific) : 0;
        int all      = v_all      ? std::atoi(v_all)      : 0;
        s_all_bf16      = all;
        s_ffn_down_bf16 = specific || all;
        QIE_LOG("forward_block_: QIE_FFN_DOWN_BF16=%d QIE_ALL_BF16=%d "
                "(ff_down BF16 + bf16src #2 always under either; "
                "attn-out BF16 + bf16src #1 only under ALL)",
                s_ffn_down_bf16, s_all_bf16);
    }
    const bool ffn_down_bf16 = s_ffn_down_bf16 != 0;
    const bool attn_out_bf16 = s_all_bf16 != 0;

    // Q2.4.5.4c: probe is dtype-aware via a 3-state enum-like overload set.
    // The legacy bool overload (`is_f16`) is preserved verbatim; a new int
    // overload accepts {0=F32, 1=F16, 2=BF16}.
    enum ProbeDtype { PROBE_F32 = 0, PROBE_F16 = 1, PROBE_BF16 = 2 };
    auto intra_probe_dt = [&](const char *label, void *dev,
                                int64_t n_elts, ProbeDtype dt) -> void {
        if (!do_intra) return;
        g_cann.aclrtSynchronizeStream(compute_stream_);
        const size_t bpe = (dt == PROBE_F32) ? 4 : 2;
        std::vector<uint8_t> host((size_t)n_elts * bpe);
        aclError me = g_cann.aclrtMemcpy(host.data(), host.size(), dev,
                                           host.size(),
                                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (me != 0) return;
        double sum_abs = 0.0, max_abs = 0.0;
        int64_t nanc = 0, infc = 0;
        for (int64_t i = 0; i < n_elts; ++i) {
            float v;
            if (dt == PROBE_F16) {
                __fp16 hh;
                std::memcpy(&hh, host.data() + (size_t)i * 2, 2);
                v = (float)hh;
            } else if (dt == PROBE_BF16) {
                // BF16 = upper 16 bits of an F32 (little-endian: low bytes
                // are zero'd lsbs). Reconstruct by left-shifting into the
                // upper half of an F32 bit pattern.
                uint16_t bh;
                std::memcpy(&bh, host.data() + (size_t)i * 2, 2);
                uint32_t bits = ((uint32_t)bh) << 16;
                std::memcpy(&v, &bits, 4);
            } else {
                std::memcpy(&v, host.data() + (size_t)i * 4, 4);
            }
            if (std::isnan(v)) { nanc++; continue; }
            if (std::isinf(v)) { infc++; continue; }
            double a = std::fabs((double)v);
            sum_abs += a;
            if (a > max_abs) max_abs = a;
        }
        int64_t valid = n_elts - nanc - infc;
        double mean_abs = valid > 0 ? sum_abs / (double)valid : 0.0;
        const char *dt_str = (dt == PROBE_F32)  ? "F32" :
                              (dt == PROBE_F16) ? "F16" : "BF16";
        QIE_LOG("intra_b0[%s]: n=%lld %s mean_abs=%.4g max_abs=%.4g "
                "NaN=%lld Inf=%lld",
                label, (long long)n_elts, dt_str,
                mean_abs, max_abs, (long long)nanc, (long long)infc);
    };
    // Backward-compat wrapper for the existing bool-flavoured callers.
    auto intra_probe = [&](const char *label, void *dev,
                             int64_t n_elts, bool is_f16) -> void {
        intra_probe_dt(label, dev, n_elts,
                        is_f16 ? PROBE_F16 : PROBE_F32);
    };

    intra_probe("00_img_hidden_in", img_hidden, img_seq * H, false);
    intra_probe("00_txt_hidden_in", txt_hidden, txt_seq * H, false);
    intra_probe("00_t_emb_in", t_emb, H, true);

    // ------------------------------------------------------------------
    // 1. Modulation: mod_params = img_mod.1(silu(t_emb))  (and txt side)
    //    scratch_mod_dev_ holds [12, H] F16 — first 6 img, next 6 txt.
    //    First compute silu(t_emb) in-place on scratch_q_dev_[:H] (reuse).
    // ------------------------------------------------------------------
    // silu(t_emb) — t_emb is [B, H] F16. Write to scratch_q_dev_.
    {
        int64_t shape[2]   = {B, H};
        int64_t strides[2] = {H, 1};
        aclTensor *t_in  = tensor_nd_f16(t_emb,          2, shape, strides);
        aclTensor *t_out = tensor_nd_f16(scratch_q_dev_, 2, shape, strides);
        uint64_t ws = 0; aclOpExecutor *exec = nullptr;
        aclnnStatus s = g_cann.aclnnSiluGetWorkspaceSize(t_in, t_out, &ws,
                                                          &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnSilu(ws > 0 ? workspace_dev_ : nullptr, ws, exec,
                                   compute_stream_);
        }
        g_cann.aclDestroyTensor(t_in);
        g_cann.aclDestroyTensor(t_out);
        if (s != 0) { QIE_LOG("block: silu(t_emb) status=%d", (int)s);
                       return false; }
    }
    intra_probe("01_silu_t_emb", scratch_q_dev_, H, true);

    // img_mod_params = img_mod.1 Linear → scratch_mod_dev_[0 .. 6H)
    if (!dispatch_matmul_(scratch_q_dev_, lw.img_mod_w_q4, lw.img_mod_scale,
                          lw.img_mod_b, B, H, 6 * H, scratch_mod_dev_))
        return false;
    intra_probe("02_img_mod_out", scratch_mod_dev_, 6 * H, true);
    // txt_mod_params = txt_mod.1 Linear → scratch_mod_dev_[6H .. 12H)
    if (!dispatch_matmul_(scratch_q_dev_, lw.txt_mod_w_q4, lw.txt_mod_scale,
                          lw.txt_mod_b, B, H, 6 * H,
                          (uint8_t *)scratch_mod_dev_ + (size_t)6 * H *
                                                          sizeof(uint16_t)))
        return false;
    intra_probe("03_txt_mod_out",
                (uint8_t *)scratch_mod_dev_ + (size_t)6 * H * sizeof(uint16_t),
                6 * H, true);

    // Chunk pointers (6 × H each) — scale1, shift1, gate1, scale2, shift2, gate2.
    auto mod_chunk = [&](void *base, int which) {
        return (uint8_t *)base + (size_t)which * H * sizeof(uint16_t);
    };
    void *img_scale1 = mod_chunk(scratch_mod_dev_, 0);
    void *img_shift1 = mod_chunk(scratch_mod_dev_, 1);
    void *img_gate1  = mod_chunk(scratch_mod_dev_, 2);
    void *img_scale2 = mod_chunk(scratch_mod_dev_, 3);
    void *img_shift2 = mod_chunk(scratch_mod_dev_, 4);
    void *img_gate2  = mod_chunk(scratch_mod_dev_, 5);
    void *txt_base   = (uint8_t *)scratch_mod_dev_ +
                       (size_t)6 * H * sizeof(uint16_t);
    void *txt_scale1 = mod_chunk(txt_base, 0);
    void *txt_shift1 = mod_chunk(txt_base, 1);
    void *txt_gate1  = mod_chunk(txt_base, 2);
    void *txt_scale2 = mod_chunk(txt_base, 3);
    void *txt_shift2 = mod_chunk(txt_base, 4);
    void *txt_gate2  = mod_chunk(txt_base, 5);

    // ------------------------------------------------------------------
    // 2. LayerNorm1 on img + txt, then modulate(scale1, shift1).
    //    Phase 4.4c: residual (img_hidden / txt_hidden) is F32 on-device.
    //    layer_norm_f32_to_f16_ runs LayerNorm entirely in F32 (so the
    //    normalization sees no Inf from F32→F16 down-cast of large-magnitude
    //    residuals at deep layers) then casts the bounded ~1σ output to F16
    //    for matmul/modulate consumption.
    // ------------------------------------------------------------------
    if (!layer_norm_f32_to_f16_(img_hidden, scratch_img_norm_dev_,
                                  B, img_seq, H))
        return false;
    intra_probe("04_img_LN1", scratch_img_norm_dev_, img_seq * H, true);
    if (!modulate_(scratch_img_norm_dev_, img_scale1, img_shift1,
                   B, img_seq, H)) return false;
    intra_probe("05_img_mod1", scratch_img_norm_dev_, img_seq * H, true);
    if (!layer_norm_f32_to_f16_(txt_hidden, scratch_txt_norm_dev_,
                                  B, txt_seq, H))
        return false;
    intra_probe("06_txt_LN1", scratch_txt_norm_dev_, txt_seq * H, true);
    if (!modulate_(scratch_txt_norm_dev_, txt_scale1, txt_shift1,
                   B, txt_seq, H)) return false;
    intra_probe("07_txt_mod1", scratch_txt_norm_dev_, txt_seq * H, true);

    // ------------------------------------------------------------------
    // 3. QKV projections. img → to_q / to_k / to_v, txt → add_q/k/v.
    //    The joint attention buffer layout is `txt || img` along seq dim
    //    (matches qwen_image.hpp:160: ggml_concat(txt, img, 2)).
    //    scratch_q/k/v_dev_ hold [seq_total, H] F16, with txt occupying the
    //    first txt_seq rows and img the following img_seq rows.
    // ------------------------------------------------------------------
    auto offset_rows = [&](void *base, int64_t rows) {
        return (uint8_t *)base + (size_t)rows * H * sizeof(uint16_t);
    };
    // txt QKV (rows 0 .. txt_seq) — head of scratch buffers.
    if (!dispatch_matmul_(scratch_txt_norm_dev_, lw.add_q_w_q4, lw.add_q_scale,
                          lw.add_q_b, txt_seq, H, H, scratch_q_dev_))
        return false;
    if (!dispatch_matmul_(scratch_txt_norm_dev_, lw.add_k_w_q4, lw.add_k_scale,
                          lw.add_k_b, txt_seq, H, H, scratch_k_dev_))
        return false;
    if (!dispatch_matmul_(scratch_txt_norm_dev_, lw.add_v_w_q4, lw.add_v_scale,
                          lw.add_v_b, txt_seq, H, H, scratch_v_dev_))
        return false;
    // img QKV (rows txt_seq .. seq_total).
    if (!dispatch_matmul_(scratch_img_norm_dev_, lw.to_q_w_q4, lw.to_q_scale,
                          lw.to_q_b, img_seq, H, H,
                          offset_rows(scratch_q_dev_, txt_seq))) return false;
    if (!dispatch_matmul_(scratch_img_norm_dev_, lw.to_k_w_q4, lw.to_k_scale,
                          lw.to_k_b, img_seq, H, H,
                          offset_rows(scratch_k_dev_, txt_seq))) return false;
    if (!dispatch_matmul_(scratch_img_norm_dev_, lw.to_v_w_q4, lw.to_v_scale,
                          lw.to_v_b, img_seq, H, H,
                          offset_rows(scratch_v_dev_, txt_seq))) return false;
    intra_probe("08_img_Q", offset_rows(scratch_q_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("08_img_K", offset_rows(scratch_k_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("08_img_V", offset_rows(scratch_v_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("08_txt_Q", scratch_q_dev_, txt_seq * H, true);
    intra_probe("08_txt_K", scratch_k_dev_, txt_seq * H, true);
    intra_probe("08_txt_V", scratch_v_dev_, txt_seq * H, true);

    // ------------------------------------------------------------------
    // 4. RMSNorm on Q / K (both streams). Since Q/K have shape
    //    [seq, n_heads, head_dim] and RMSNorm is over head_dim, we view
    //    each as [seq * n_heads, head_dim].
    //    Reference: qwen_image.hpp:147-148, 157-158.
    // ------------------------------------------------------------------
    // img Q/K (rows = img_seq * n_heads).
    {
        void *img_q = offset_rows(scratch_q_dev_, txt_seq);
        void *img_k = offset_rows(scratch_k_dev_, txt_seq);
        if (!rms_norm_head_(img_q, img_q, lw.norm_q_w,
                            img_seq * NH, HD)) return false;
        if (!rms_norm_head_(img_k, img_k, lw.norm_k_w,
                            img_seq * NH, HD)) return false;
    }
    // txt Q/K (rows = txt_seq * n_heads) — head of scratch buffers.
    {
        if (!rms_norm_head_(scratch_q_dev_, scratch_q_dev_,
                            lw.norm_added_q_w, txt_seq * NH, HD)) return false;
        if (!rms_norm_head_(scratch_k_dev_, scratch_k_dev_,
                            lw.norm_added_k_w, txt_seq * NH, HD)) return false;
    }
    intra_probe("09_img_Q_rmsnorm", offset_rows(scratch_q_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("09_img_K_rmsnorm", offset_rows(scratch_k_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("09_txt_Q_rmsnorm", scratch_q_dev_, txt_seq * H, true);
    intra_probe("09_txt_K_rmsnorm", scratch_k_dev_, txt_seq * H, true);

    // ------------------------------------------------------------------
    // 5. RoPE on Q, K. pe index layout (from compute_qwen_rope_pe_host):
    //    rows [0 .. ctx_len) are txt, rows [ctx_len .. ctx_len + img_tokens)
    //    are img. ctx_len = cfg_.max_txt_seq.
    // ------------------------------------------------------------------
    {
        // txt stream: pe offset 0.
        if (!apply_rope_(scratch_q_dev_, pe, 0, B, txt_seq, NH, HD))
            return false;
        if (!apply_rope_(scratch_k_dev_, pe, 0, B, txt_seq, NH, HD))
            return false;
        // img stream: pe offset = cfg_.max_txt_seq (the ctx_len used at
        // pe build time). For the Phase 3 smoke we run with txt_seq ==
        // max_txt_seq so the offset equals ctx_len exactly. Higher-phase
        // work may need session rebuild (see compute_qwen_rope_pe_host
        // NOTE-TO-AGENT).
        const int64_t img_pe_off = cfg_.max_txt_seq;
        if (!apply_rope_(offset_rows(scratch_q_dev_, txt_seq),
                         pe, img_pe_off, B, img_seq, NH, HD)) return false;
        if (!apply_rope_(offset_rows(scratch_k_dev_, txt_seq),
                         pe, img_pe_off, B, img_seq, NH, HD)) return false;
    }
    intra_probe("10_img_Q_rope", offset_rows(scratch_q_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("10_img_K_rope", offset_rows(scratch_k_dev_, txt_seq),
                img_seq * H, true);
    intra_probe("10_txt_Q_rope", scratch_q_dev_, txt_seq * H, true);
    intra_probe("10_txt_K_rope", scratch_k_dev_, txt_seq * H, true);

    // ------------------------------------------------------------------
    // 6. Joint attention via aclnnFusedInferAttentionScoreV2.
    //    Layout BSND = [B=1, seq_total, NH, HD]. Q/K/V scratch are already
    //    laid out as txt || img along seq. V is [seq_total, H] = BSND.
    // ------------------------------------------------------------------
    {
        int64_t qkv_shape[4]   = {B, seq_total, NH, HD};
        int64_t qkv_strides[4];
        make_contig_strides(4, qkv_shape, qkv_strides);
        aclTensor *t_q = tensor_nd_f16(scratch_q_dev_,    4, qkv_shape, qkv_strides);
        aclTensor *t_k = tensor_nd_f16(scratch_k_dev_,    4, qkv_shape, qkv_strides);
        aclTensor *t_v = tensor_nd_f16(scratch_v_dev_,    4, qkv_shape, qkv_strides);
        aclTensor *t_o = tensor_nd_f16(scratch_attn_dev_, 4, qkv_shape, qkv_strides);
        aclTensorList *t_k_list = g_cann.aclCreateTensorList(&t_k, 1);
        aclTensorList *t_v_list = g_cann.aclCreateTensorList(&t_v, 1);
        char layout[5] = {'B','S','N','D',0};
        double scale = 1.0 / std::sqrt((double)HD);
        uint64_t ws = 0; aclOpExecutor *exec = nullptr;
        aclnnStatus s = g_cann.aclnnFusedInferAttentionScoreV2GetWorkspaceSize(
            t_q, t_k_list, t_v_list,
            nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
            (int64_t)NH, scale,
            (int64_t)65535, (int64_t)65535,
            layout, (int64_t)NH, (int64_t)0, (int64_t)0,
            (int64_t)0, (int64_t)0, false,
            (int64_t)0, (int64_t)0,
            t_o, nullptr, &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnFusedInferAttentionScoreV2(
                ws > 0 ? workspace_dev_ : nullptr, ws, exec, compute_stream_);
        }
        g_cann.aclDestroyTensorList(t_k_list);
        g_cann.aclDestroyTensorList(t_v_list);
        g_cann.aclDestroyTensor(t_q);
        g_cann.aclDestroyTensor(t_o);
        if (s != 0) {
            QIE_LOG("block: FIAv2 status=%d (seq_total=%lld NH=%lld HD=%lld)",
                    (int)s, (long long)seq_total, (long long)NH, (long long)HD);
            return false;
        }
    }
    intra_probe("11_attn_out_txt", scratch_attn_dev_, txt_seq * H, true);
    intra_probe("11_attn_out_img", offset_rows(scratch_attn_dev_, txt_seq),
                img_seq * H, true);

    // ------------------------------------------------------------------
    // 7. Output projections.
    //    txt attn-out view: scratch_attn_dev_[0 .. txt_seq*H] → to_add_out
    //    img attn-out view: scratch_attn_dev_[txt_seq*H .. end] → to_out.0
    //    The output projections write to scratch_img_out_dev_ /
    //    scratch_txt_out_dev_ so the subsequent gated residual is a pure
    //    add on top of the original img_hidden / txt_hidden buffers.
    // ------------------------------------------------------------------
    // Q2.4.5.4d Step 7 widening: under QIE_ALL_BF16 the attn-output
    // projections emit BF16 (matches the ff_down treatment in §5.5.4) so
    // the residual #1 contributor is overflow-immune. Same scaffold —
    // dispatch_matmul_ pre-casts inputs/weights and emits BF16 with F32
    // accumulator (cubeMathType=1). When OFF, F16 path is preserved
    // byte-identical.
    if (!dispatch_matmul_(scratch_attn_dev_, lw.to_add_out_w_q4,
                          lw.to_add_out_scale, lw.to_add_out_b,
                          txt_seq, H, H, scratch_txt_out_dev_,
                          attn_out_bf16 ? ACL_BF16 : ACL_FLOAT16)) return false;
    if (!dispatch_matmul_(offset_rows(scratch_attn_dev_, txt_seq),
                          lw.to_out_0_w_q4, lw.to_out_0_scale, lw.to_out_0_b,
                          img_seq, H, H, scratch_img_out_dev_,
                          attn_out_bf16 ? ACL_BF16 : ACL_FLOAT16)) return false;
    intra_probe_dt("12_to_add_out", scratch_txt_out_dev_, txt_seq * H,
                    attn_out_bf16 ? PROBE_BF16 : PROBE_F16);
    intra_probe_dt("12_to_out_0",   scratch_img_out_dev_, img_seq * H,
                    attn_out_bf16 ? PROBE_BF16 : PROBE_F16);

    // ------------------------------------------------------------------
    // 8. Gated residual add: img += attn_out_img * gate1_img (and txt).
    //    Phase 4.4c: residual is F32; gated_residual_add_f32_ casts the
    //    (src_f16 * gate_f16) product up to F32 before the accumulator add.
    //    Q2.4.5.4d: under QIE_ALL_BF16 the matmul source is BF16, so route
    //    through the BF16-src variant (cast BF16→F32 before gate-mul).
    // ------------------------------------------------------------------
    if (attn_out_bf16) {
        if (!gated_residual_add_f32_bf16src_(img_hidden, scratch_img_out_dev_,
                                              img_gate1, B, img_seq, H))
            return false;
        if (!gated_residual_add_f32_bf16src_(txt_hidden, scratch_txt_out_dev_,
                                              txt_gate1, B, txt_seq, H))
            return false;
    } else {
        if (!gated_residual_add_f32_(img_hidden, scratch_img_out_dev_,
                                       img_gate1, B, img_seq, H)) return false;
        if (!gated_residual_add_f32_(txt_hidden, scratch_txt_out_dev_,
                                       txt_gate1, B, txt_seq, H)) return false;
    }
    intra_probe("13_img_resid1", img_hidden, img_seq * H, false);
    intra_probe("13_txt_resid1", txt_hidden, txt_seq * H, false);

    // ------------------------------------------------------------------
    // 9. LayerNorm2 + modulate(scale2, shift2) — Phase 4.4c F32-in path
    //    (see step 2 for the full-F32 LN rationale).
    // ------------------------------------------------------------------
    if (!layer_norm_f32_to_f16_(img_hidden, scratch_img_norm_dev_,
                                  B, img_seq, H))
        return false;
    intra_probe("14_img_LN2", scratch_img_norm_dev_, img_seq * H, true);
    if (!modulate_(scratch_img_norm_dev_, img_scale2, img_shift2,
                   B, img_seq, H)) return false;
    intra_probe("15_img_mod2", scratch_img_norm_dev_, img_seq * H, true);
    if (!layer_norm_f32_to_f16_(txt_hidden, scratch_txt_norm_dev_,
                                  B, txt_seq, H))
        return false;
    intra_probe("16_txt_LN2", scratch_txt_norm_dev_, txt_seq * H, true);
    if (!modulate_(scratch_txt_norm_dev_, txt_scale2, txt_shift2,
                   B, txt_seq, H)) return false;
    intra_probe("17_txt_mod2", scratch_txt_norm_dev_, txt_seq * H, true);

    // ------------------------------------------------------------------
    // 10. FFN per stream: Linear(H→FF) → GELU-tanh → Linear(FF→H).
    //     scratch_mlp_dev_ [SEQ, FF] holds the post-up-projection
    //     activation (enough for either stream; streams run sequentially).
    // ------------------------------------------------------------------
    auto gelu_activate = [&](void *buf_dev, int64_t rows) -> bool {
        int64_t shape[2]   = {rows, FF};
        int64_t strides[2] = {FF, 1};
        aclTensor *t_in  = tensor_nd_f16(buf_dev, 2, shape, strides);
        aclTensor *t_out = tensor_nd_f16(buf_dev, 2, shape, strides);
        uint64_t ws = 0; aclOpExecutor *exec = nullptr;
        aclnnStatus s = 0;
        // Prefer GELU V2 with approximate="tanh" (1) to match the ggml CPU
        // reference (ggml_gelu uses tanh approx).
        if (g_cann.has_gelu_v2()) {
            s = g_cann.aclnnGeluV2GetWorkspaceSize(t_in, /*approximate=*/1,
                                                     t_out, &ws, &exec);
            if (s == 0) {
                ensure_workspace_(ws);
                s = g_cann.aclnnGeluV2(ws > 0 ? workspace_dev_ : nullptr,
                                         ws, exec, compute_stream_);
            }
        } else if (g_cann.aclnnGelu &&
                   g_cann.aclnnGeluGetWorkspaceSize) {
            // Fallback: exact erf. Drifts vs CPU ref by ~1e-3 cos_sim.
            s = g_cann.aclnnGeluGetWorkspaceSize(t_in, t_out, &ws, &exec);
            if (s == 0) {
                ensure_workspace_(ws);
                s = g_cann.aclnnGelu(ws > 0 ? workspace_dev_ : nullptr,
                                       ws, exec, compute_stream_);
            }
        } else {
            QIE_LOG("block: no Gelu symbol resolved");
            s = -1;
        }
        g_cann.aclDestroyTensor(t_in);
        g_cann.aclDestroyTensor(t_out);
        return s == 0;
    };

    // img FFN.
    if (!dispatch_matmul_(scratch_img_norm_dev_, lw.img_ff_up_w_q4,
                          lw.img_ff_up_scale, lw.img_ff_up_b,
                          img_seq, H, FF, scratch_mlp_dev_)) return false;
    intra_probe("18_img_ff_up", scratch_mlp_dev_, img_seq * FF, true);
    if (!gelu_activate(scratch_mlp_dev_, img_seq)) return false;
    intra_probe("19_img_gelu", scratch_mlp_dev_, img_seq * FF, true);
    // Q2.4.5.4c: ff_down output dtype gated by QIE_FFN_DOWN_BF16 (or its
    // superset QIE_ALL_BF16). F16 by default (Step 1 synthetic regression
    // invariant); BF16 escapes the ~65504 saturation observed in §5.5.3
    // bisect on real Q4 weights.
    if (!dispatch_matmul_(scratch_mlp_dev_, lw.img_ff_down_w_q4,
                          lw.img_ff_down_scale, lw.img_ff_down_b,
                          img_seq, FF, H, scratch_img_out_dev_,
                          ffn_down_bf16 ? ACL_BF16 : ACL_FLOAT16)) return false;
    intra_probe_dt("20_img_ff_down", scratch_img_out_dev_, img_seq * H,
                    ffn_down_bf16 ? PROBE_BF16 : PROBE_F16);

    // txt FFN.
    if (!dispatch_matmul_(scratch_txt_norm_dev_, lw.txt_ff_up_w_q4,
                          lw.txt_ff_up_scale, lw.txt_ff_up_b,
                          txt_seq, H, FF, scratch_mlp_dev_)) return false;
    intra_probe("21_txt_ff_up", scratch_mlp_dev_, txt_seq * FF, true);
    if (!gelu_activate(scratch_mlp_dev_, txt_seq)) return false;
    intra_probe("22_txt_gelu", scratch_mlp_dev_, txt_seq * FF, true);
    if (!dispatch_matmul_(scratch_mlp_dev_, lw.txt_ff_down_w_q4,
                          lw.txt_ff_down_scale, lw.txt_ff_down_b,
                          txt_seq, FF, H, scratch_txt_out_dev_,
                          ffn_down_bf16 ? ACL_BF16 : ACL_FLOAT16)) return false;
    intra_probe_dt("23_txt_ff_down", scratch_txt_out_dev_, txt_seq * H,
                    ffn_down_bf16 ? PROBE_BF16 : PROBE_F16);

    // ------------------------------------------------------------------
    // 11. Gated residual add #2. Phase 4.4c: F32 accumulator (see step 8).
    // Q2.4.5.4c: when ff_down emitted BF16, route through the BF16-src
    // variant so the gate-mul happens in F32 (no F16 65504 saturation).
    // ------------------------------------------------------------------
    if (ffn_down_bf16) {
        if (!gated_residual_add_f32_bf16src_(img_hidden, scratch_img_out_dev_,
                                              img_gate2, B, img_seq, H))
            return false;
        if (!gated_residual_add_f32_bf16src_(txt_hidden, scratch_txt_out_dev_,
                                              txt_gate2, B, txt_seq, H))
            return false;
    } else {
        if (!gated_residual_add_f32_(img_hidden, scratch_img_out_dev_,
                                       img_gate2, B, img_seq, H)) return false;
        if (!gated_residual_add_f32_(txt_hidden, scratch_txt_out_dev_,
                                       txt_gate2, B, txt_seq, H)) return false;
    }
    intra_probe("24_img_resid2", img_hidden, img_seq * H, false);
    intra_probe("24_txt_resid2", txt_hidden, txt_seq * H, false);

    return true;
}

void ImageDiffusionEngine::scheduler_step_(void * /*latent_dev*/,
                                             const void * /*model_out_dev*/,
                                             int step_idx) {
    (void)step_idx;
    // Phase 4.3 note: the in-place axpy `x += dt * eps` is provided via the
    // `scheduler_step_test()` public test hook below (it takes explicit dt
    // and element count, which matches the smoke probe's needs without
    // needing to thread the full sigma schedule / C-H-W layout through this
    // helper). A production `denoise()` body would call that same axpy
    // sequence. See tools/ominix_diffusion/src/denoiser.hpp:831-865 for the
    // Euler-flow reference formula:
    //     d  = (x - denoised) / sigma          (for denoised-prediction model)
    //     dt = sigmas[i+1] - sigmas[i]
    //     x  = x + d * dt
    // For Qwen-Image flow-matching the model predicts velocity directly, so
    // the step is simply `x += dt * eps` — no division by sigma.
}

// ---------------------------------------------------------------------------
// Phase 4.3 scheduler-step test hook — see denoise_loop_test below for the
// full Euler loop. Does a single in-place `x += dt * eps` via
// aclnnInplaceAdd(alpha=dt). Supports n_elts > INT32_MAX by flattening to a
// 1-D tensor; callers pass total element count.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::scheduler_step_test(void *x_f16_dev,
                                                 const void *eps_f16_dev,
                                                 int64_t n_elts, float dt) {
    // Phase 4.4c: x and eps are F32 on-device (the latent/residual-stream
    // promotion cascaded here from denoise_loop_test). Parameter names kept
    // for source compatibility.
    if (!ready_) {
        QIE_LOG("scheduler_step_test: engine not ready");
        return false;
    }
    if (!x_f16_dev || !eps_f16_dev || n_elts <= 0) {
        QIE_LOG("scheduler_step_test: bad args");
        return false;
    }
    if (!g_cann.aclnnInplaceAdd || !g_cann.aclnnInplaceAddGetWorkspaceSize) {
        QIE_LOG("scheduler_step_test: aclnnInplaceAdd symbol missing");
        return false;
    }
    int64_t shape[1]   = { n_elts };
    int64_t strides[1] = { 1 };
    int64_t storage    = n_elts;
    aclTensor *t_x = g_cann.aclCreateTensor(shape, 1, ACL_FLOAT, strides,
                                             0, ACL_FORMAT_ND, &storage, 1,
                                             x_f16_dev);
    aclTensor *t_e = g_cann.aclCreateTensor(shape, 1, ACL_FLOAT, strides,
                                             0, ACL_FORMAT_ND, &storage, 1,
                                             (void *)eps_f16_dev);
    float dt_f32 = dt;
    aclScalar *alpha = g_cann.aclCreateScalar(&dt_f32, ACL_FLOAT);

    uint64_t ws_needed = 0;
    aclOpExecutor *exec = nullptr;
    aclnnStatus s = g_cann.aclnnInplaceAddGetWorkspaceSize(t_x, t_e, alpha,
                                                             &ws_needed, &exec);
    bool ok = (s == 0);
    if (ok) {
        ensure_workspace_(ws_needed);
        void *ws = ws_needed > 0 ? workspace_dev_ : nullptr;
        s = g_cann.aclnnInplaceAdd(ws, ws_needed, exec, compute_stream_);
        ok = (s == 0);
        if (!ok) QIE_LOG("scheduler_step_test: aclnnInplaceAdd launch err=%d",
                          (int)s);
    } else {
        QIE_LOG("scheduler_step_test: workspace err=%d", (int)s);
    }

    g_cann.aclDestroyScalar(alpha);
    g_cann.aclDestroyTensor(t_x);
    g_cann.aclDestroyTensor(t_e);
    return ok;
}

// ---------------------------------------------------------------------------
// Phase 4.3 Euler-flow 20-step denoise loop — test hook. Operates on
// already-resident activation buffers (no patchify / no VAE / no text
// encoder). For a production denoise() body we would wrap this with
// patchify(initial_noise) on entry and unpatchify on exit.
//
// Per-step algorithm (flow-matching convention):
//   x_copy       <- x                          (snapshot, D2D memcpy)
//   txt_c_copy   <- txt_hidden_cond            (snapshot)
//   eps_cond     <- forward(x_copy, t_emb, txt_c_copy)
//                  (forward is in-place on img; x_copy is now eps_cond)
//   x_copy2      <- x                          (snapshot, D2D memcpy)
//   txt_u_copy   <- txt_hidden_uncond          (snapshot)
//   eps_uncond   <- forward(x_copy2, t_emb, txt_u_copy)
//                  (forward is in-place on img; x_copy2 is now eps_uncond)
//   eps          <- eps_uncond + cfg*(eps_cond - eps_uncond)
//                  expressed as:
//                    eps_cond  -= eps_uncond                  (aclnnInplaceAdd, alpha=-1)
//                    eps_uncond += cfg * eps_cond             (aclnnInplaceAdd, alpha=cfg)
//                  so eps_uncond now holds the CFG-composed eps.
//   dt           = sigmas[s+1] - sigmas[s]
//   x            += dt * eps_uncond            (aclnnInplaceAdd, alpha=dt)
//
// When cfg_scale == 1.0 the CFG composition simplifies to `eps = eps_cond`
// (since eps_cond + 1.0*(eps_cond - eps_cond) = eps_cond). We still run two
// forward passes in the smoke (simpler code; single-forward path is a 4.4
// optimisation knob).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::denoise_loop_test(void *x_f16_dev, int64_t img_seq,
                                               void *txt_hidden_cond_f16_dev,
                                               void *txt_hidden_uncond_f16_dev,
                                               int64_t txt_seq,
                                               void *t_emb_f16_dev,
                                               void *pe_f16_dev,
                                               const float *sigmas,
                                               int n_steps,
                                               float cfg_scale,
                                               double *per_step_ms) {
    // Phase 4.4c: `x_f16_dev`, `txt_hidden_*_f16_dev` are now F32 buffers
    // (residual-stream promotion — see forward_block_). Parameter names are
    // preserved for source compatibility with the 4.3 probe; ownership
    // semantics are unchanged. Scheduler step and CFG composition use F32
    // scalars / F32 InplaceAdd accordingly.
    if (!ready_) {
        QIE_LOG("denoise_loop_test: engine not ready");
        return false;
    }
    if (!x_f16_dev || !txt_hidden_cond_f16_dev ||
        !txt_hidden_uncond_f16_dev || !t_emb_f16_dev || !sigmas || n_steps < 1) {
        QIE_LOG("denoise_loop_test: bad args");
        return false;
    }

    const int64_t H    = cfg_.hidden_size;
    const size_t  F32  = sizeof(float);
    const size_t  img_bytes = (size_t)img_seq * H * F32;
    const size_t  txt_bytes = (size_t)txt_seq * H * F32;

    // Temporary D2D staging buffers for:
    //   * latent snapshot (so the second forward can re-read x)
    //   * per-pass img/txt working buffers (forward is in-place)
    //   * eps_cond storage (img buffer from the cond pass)
    // We need two "img working" buffers (one per CFG pass) and two "txt
    // working" buffers. All sized [seq, H] F16.
    void *x_snap_dev      = nullptr;
    void *img_work_c_dev  = nullptr;  // cond pass working latent → becomes eps_cond
    void *img_work_u_dev  = nullptr;  // uncond pass working latent → becomes eps_uncond
    void *txt_work_c_dev  = nullptr;  // cond pass working txt (destroyed in-place)
    void *txt_work_u_dev  = nullptr;  // uncond pass working txt (destroyed in-place)

    auto alloc_dev = [&](void **p, size_t n) {
        aclError e = g_cann.aclrtMalloc(p, n, ACL_MEM_MALLOC_HUGE_FIRST);
        if (e != 0) { QIE_LOG("denoise_loop_test: aclrtMalloc(%zu) err=%d",
                               n, (int)e); *p = nullptr; return false; }
        return true;
    };
    bool alloc_ok = alloc_dev(&x_snap_dev,     img_bytes) &&
                    alloc_dev(&img_work_c_dev, img_bytes) &&
                    alloc_dev(&img_work_u_dev, img_bytes) &&
                    alloc_dev(&txt_work_c_dev, txt_bytes) &&
                    alloc_dev(&txt_work_u_dev, txt_bytes);
    auto free_all = [&]() {
        if (x_snap_dev)     g_cann.aclrtFree(x_snap_dev);
        if (img_work_c_dev) g_cann.aclrtFree(img_work_c_dev);
        if (img_work_u_dev) g_cann.aclrtFree(img_work_u_dev);
        if (txt_work_c_dev) g_cann.aclrtFree(txt_work_c_dev);
        if (txt_work_u_dev) g_cann.aclrtFree(txt_work_u_dev);
    };
    if (!alloc_ok) { free_all(); return false; }

    const bool run_uncond = (cfg_scale != 1.0f);

    for (int step = 0; step < n_steps; ++step) {
        auto t_step_0 = std::chrono::steady_clock::now();

        // Snapshot x into x_snap_dev and also into both working buffers for
        // the two forward passes. forward_all_blocks_test updates img_hidden
        // in-place, so each pass needs its own clone.
        aclError me = g_cann.aclrtMemcpy(x_snap_dev, img_bytes,
                                           x_f16_dev, img_bytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (me != 0) { QIE_LOG("denoise_loop_test: snap x err=%d", (int)me);
                       free_all(); return false; }
        me = g_cann.aclrtMemcpy(img_work_c_dev, img_bytes,
                                 x_snap_dev, img_bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (me != 0) { free_all(); return false; }
        me = g_cann.aclrtMemcpy(txt_work_c_dev, txt_bytes,
                                 txt_hidden_cond_f16_dev, txt_bytes,
                                 ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (me != 0) { free_all(); return false; }

        // Pass 1: cond. img_work_c_dev becomes eps_cond after forward.
        if (!forward_all_blocks_test(img_work_c_dev, img_seq,
                                      txt_work_c_dev, txt_seq,
                                      t_emb_f16_dev, pe_f16_dev,
                                      nullptr, 0)) {
            QIE_LOG("denoise_loop_test: cond forward failed at step %d", step);
            free_all(); return false;
        }

        void *eps_composed_dev = img_work_c_dev;  // default if !run_uncond

        if (run_uncond) {
            me = g_cann.aclrtMemcpy(img_work_u_dev, img_bytes,
                                     x_snap_dev, img_bytes,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (me != 0) { free_all(); return false; }
            me = g_cann.aclrtMemcpy(txt_work_u_dev, txt_bytes,
                                     txt_hidden_uncond_f16_dev, txt_bytes,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (me != 0) { free_all(); return false; }

            // Pass 2: uncond. img_work_u_dev becomes eps_uncond.
            if (!forward_all_blocks_test(img_work_u_dev, img_seq,
                                          txt_work_u_dev, txt_seq,
                                          t_emb_f16_dev, pe_f16_dev,
                                          nullptr, 0)) {
                QIE_LOG("denoise_loop_test: uncond forward failed at step %d",
                        step);
                free_all(); return false;
            }

            // CFG: eps = eps_uncond + cfg * (eps_cond - eps_uncond)
            // Decompose into two in-place add-with-alpha dispatches (no
            // reliance on aclnnMuls self-aliasing):
            //   1) eps_cond -= eps_uncond                      (alpha = -1)
            //      → img_work_c_dev now holds (eps_cond - eps_uncond)
            //   2) eps_uncond += cfg * (eps_cond - eps_uncond) (alpha = cfg)
            //      → img_work_u_dev now holds eps = eps_uncond + cfg*(Δ)
            int64_t nE = (int64_t)img_seq * H;
            int64_t shape1[1]   = { nE };
            int64_t strides1[1] = { 1 };
            int64_t storage1    = nE;

            // Phase 4.4c: CFG tensors are F32 (residual stream promoted).
            aclTensor *t_u = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, img_work_u_dev);
            aclTensor *t_c = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, img_work_c_dev);

            // Step 1: img_work_c_dev += (-1) * img_work_u_dev  (F32)
            float neg_one_f32 = -1.0f;
            aclScalar *alpha_neg1 = g_cann.aclCreateScalar(&neg_one_f32,
                                                              ACL_FLOAT);
            uint64_t ws1 = 0;
            aclOpExecutor *ex1 = nullptr;
            aclnnStatus st = g_cann.aclnnInplaceAddGetWorkspaceSize(t_c, t_u,
                                                                      alpha_neg1,
                                                                      &ws1, &ex1);
            if (st != 0) { QIE_LOG("denoise_loop_test: InplaceAdd(-1) ws err=%d",
                                    (int)st);
                           g_cann.aclDestroyScalar(alpha_neg1);
                           g_cann.aclDestroyTensor(t_u);
                           g_cann.aclDestroyTensor(t_c);
                           free_all(); return false; }
            ensure_workspace_(ws1);
            st = g_cann.aclnnInplaceAdd(ws1 > 0 ? workspace_dev_ : nullptr, ws1,
                                          ex1, compute_stream_);
            g_cann.aclDestroyScalar(alpha_neg1);
            if (st != 0) { QIE_LOG("denoise_loop_test: InplaceAdd(-1) launch err=%d",
                                    (int)st);
                           g_cann.aclDestroyTensor(t_u);
                           g_cann.aclDestroyTensor(t_c);
                           free_all(); return false; }

            // Step 2: img_work_u_dev += cfg * img_work_c_dev (which is Δ) — F32
            float cfg_f32 = cfg_scale;
            aclScalar *sc_cfg = g_cann.aclCreateScalar(&cfg_f32, ACL_FLOAT);
            uint64_t ws2 = 0;
            aclOpExecutor *ex2 = nullptr;
            st = g_cann.aclnnInplaceAddGetWorkspaceSize(t_u, t_c, sc_cfg,
                                                          &ws2, &ex2);
            if (st != 0) { QIE_LOG("denoise_loop_test: InplaceAdd(cfg) ws err=%d",
                                    (int)st);
                           g_cann.aclDestroyScalar(sc_cfg);
                           g_cann.aclDestroyTensor(t_u);
                           g_cann.aclDestroyTensor(t_c);
                           free_all(); return false; }
            ensure_workspace_(ws2);
            st = g_cann.aclnnInplaceAdd(ws2 > 0 ? workspace_dev_ : nullptr,
                                         ws2, ex2, compute_stream_);
            g_cann.aclDestroyScalar(sc_cfg);
            g_cann.aclDestroyTensor(t_u);
            g_cann.aclDestroyTensor(t_c);
            if (st != 0) { QIE_LOG("denoise_loop_test: InplaceAdd(cfg) launch err=%d",
                                    (int)st);
                           free_all(); return false; }

            eps_composed_dev = img_work_u_dev;
        }

        // Scheduler step: x += dt * eps. dt is signed — for a decreasing
        // sigma schedule (typical flow matching) dt < 0.
        float dt = sigmas[step + 1] - sigmas[step];
        if (!scheduler_step_test(x_f16_dev, eps_composed_dev,
                                  (int64_t)img_seq * H, dt)) {
            QIE_LOG("denoise_loop_test: scheduler_step failed at step %d",
                    step);
            free_all(); return false;
        }

        if (per_step_ms) {
            aclError se = g_cann.aclrtSynchronizeStream(compute_stream_);
            if (se != 0) {
                QIE_LOG("denoise_loop_test: sync after step %d err=%d",
                        step, (int)se);
                free_all(); return false;
            }
            auto t_step_1 = std::chrono::steady_clock::now();
            per_step_ms[step] =
                std::chrono::duration<double, std::milli>(t_step_1 - t_step_0)
                    .count();
        }
    }

    free_all();
    return true;
}

// ============================================================================
// Q2.3 Phase 3 test-only hooks. NOT part of the steady-state engine API.
// ============================================================================

bool ImageDiffusionEngine::forward_block_test(int il,
                                                void *img_hidden, int64_t img_seq,
                                                void *txt_hidden, int64_t txt_seq,
                                                void *t_emb, void *pe) {
    if (!ready_) return false;
    if (il < 0 || il >= (int)layer_w_.size()) return false;
    return forward_block_(layer_w_[il], img_hidden, img_seq,
                          txt_hidden, txt_seq, t_emb, pe);
}

// ---------------------------------------------------------------------------
// Phase 4.2: chain all populated DiT blocks in sequence. The `forward()`
// production entry point already does this (see line ~1215); this test hook
// exists so smoke probes can (a) scope the run to the first `n_blocks`
// layers when diagnosing layer-by-layer divergence and (b) capture per-block
// wall-clock without having to thread timing knobs through the production
// signature. Each sample synchronises the compute stream so the reported ms
// reflects the actual NPU work for that block (plus the stream sync — at
// seq=96 that sync is the dominant contributor; keep that caveat in mind
// when reading the receipt).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::forward_all_blocks_test(void *img_hidden,
                                                     int64_t img_seq,
                                                     void *txt_hidden,
                                                     int64_t txt_seq,
                                                     void *t_emb,
                                                     void *pe,
                                                     double *per_block_ms,
                                                     int n_blocks) {
    if (!ready_) {
        QIE_LOG("forward_all_blocks_test: engine not ready");
        return false;
    }
    const int L = (n_blocks <= 0) ? (int)layer_w_.size()
                                    : std::min(n_blocks, (int)layer_w_.size());

    // Phase 4.5.4b NaN bisect: per-block F32 residual scan (img + txt).
    // Gated by QIE_DEBUG_PER_BLOCK_NAN=1. Scans ONLY on the first call
    // (guarded by static latch) to bound cost — typical step-0 forward,
    // not every subsequent denoise step.
    static int  s_per_block_nan = -1;
    static bool s_per_block_fired = false;
    if (s_per_block_nan < 0) {
        const char *v = std::getenv("QIE_DEBUG_PER_BLOCK_NAN");
        s_per_block_nan = v ? std::atoi(v) : 0;
    }
    const bool do_scan = s_per_block_nan && !s_per_block_fired;

    auto scan_block_residual = [&](const char *stream, void *dev_f32,
                                     int64_t n_elts, int block) -> void {
        g_cann.aclrtSynchronizeStream(compute_stream_);
        std::vector<float> host((size_t)n_elts);
        aclError me = g_cann.aclrtMemcpy(host.data(),
                                           host.size() * sizeof(float), dev_f32,
                                           host.size() * sizeof(float),
                                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (me != 0) return;
        double sum_abs = 0.0, max_abs = 0.0;
        int64_t nanc = 0, infc = 0;
        for (int64_t i = 0; i < n_elts; ++i) {
            float v = host[(size_t)i];
            if (std::isnan(v)) { nanc++; continue; }
            if (std::isinf(v)) { infc++; continue; }
            double a = std::fabs((double)v);
            sum_abs += a;
            if (a > max_abs) max_abs = a;
        }
        int64_t valid = n_elts - nanc - infc;
        double mean_abs = valid > 0 ? sum_abs / (double)valid : 0.0;
        QIE_LOG("per_block_nan[b%02d/%s]: mean_abs=%.4g max_abs=%.4g "
                "NaN=%lld Inf=%lld",
                block, stream, mean_abs, max_abs,
                (long long)nanc, (long long)infc);
    };

    for (int il = 0; il < L; ++il) {
        auto t0 = std::chrono::steady_clock::now();
        if (!forward_block_(layer_w_[il],
                            img_hidden, img_seq,
                            txt_hidden, txt_seq,
                            t_emb, pe)) {
            QIE_LOG("forward_all_blocks_test: block %d returned error", il);
            return false;
        }
        if (do_scan) {
            const int64_t H = cfg_.hidden_size;
            scan_block_residual("img", img_hidden, img_seq * H, il);
            scan_block_residual("txt", txt_hidden, txt_seq * H, il);
        }
        if (per_block_ms) {
            // Only sync when the caller asked for per-block timing: the
            // stream sync is expensive relative to the queued aclnn ops and
            // we don't want to penalise the no-timing path.
            aclError err = g_cann.aclrtSynchronizeStream(compute_stream_);
            if (err != 0) {
                QIE_LOG("forward_all_blocks_test: sync after block %d err=%d",
                        il, (int)err);
                return false;
            }
            auto t1 = std::chrono::steady_clock::now();
            per_block_ms[il] =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
        }
    }
    if (do_scan) s_per_block_fired = true;
    return true;
}

DiTLayerWeights *ImageDiffusionEngine::mutable_layer_weights(int il) {
    if (il < 0 || il >= (int)layer_w_.size()) return nullptr;
    return &layer_w_[il];
}

bool ImageDiffusionEngine::init_for_smoke(const ImageDiffusionConfig &cfg,
                                            int device) {
    if (!cp_cann_load_symbols()) {
        QIE_LOG("init_for_smoke: symbol load failed");
        return false;
    }
    cfg_    = cfg;
    device_ = device;
    QIE_ACL_CHECK(g_cann.aclrtSetDevice(device_));
    QIE_ACL_CHECK(g_cann.aclrtCreateStream(&primary_stream_));
    compute_stream_ = primary_stream_;

    layer_w_.clear();
    layer_w_.resize(cfg_.num_layers);

    // Pre-compute RoPE table (same as init_from_gguf).
    const int64_t HEAD_D = cfg_.head_dim;
    (void)HEAD_D;
    if (cfg_.precompute_rope) {
        std::vector<uint16_t> pe_host, cos_host, sin_host;
        int64_t total_pos = 0;
        compute_qwen_rope_pe_host(cfg_, pe_host, total_pos,
                                   &cos_host, &sin_host);
        const size_t pe_bytes  = pe_host.size()  * sizeof(uint16_t);
        const size_t cos_bytes = cos_host.size() * sizeof(uint16_t);
        const size_t sin_bytes = sin_host.size() * sizeof(uint16_t);
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_pe_dev, pe_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_pe_dev, pe_bytes,
                                          pe_host.data(), pe_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_cos_dev, cos_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_cos_dev, cos_bytes,
                                          cos_host.data(), cos_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_sin_dev, sin_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_sin_dev, sin_bytes,
                                          sin_host.data(), sin_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        global_w_.rope_total_pos = total_pos;
        stats_.rope_bytes = pe_bytes + cos_bytes + sin_bytes;

        // Pre-broadcast cos/sin over NH: explicit [1, total_pos, NH, half]
        // tiles to avoid ACL stride-0 broadcast numerical bugs observed in
        // Q2.4.1 RoPE smoke. Cost: total_pos * NH * half * F16 each.
        {
            const int64_t half = cfg_.head_dim / 2;
            const int64_t NH = cfg_.num_heads;
            std::vector<uint16_t> cos_bcast((size_t)total_pos * NH * half, 0);
            std::vector<uint16_t> sin_bcast((size_t)total_pos * NH * half, 0);
            for (int64_t p = 0; p < total_pos; ++p) {
                for (int64_t h = 0; h < NH; ++h) {
                    const size_t src_off = (size_t)p * half;
                    const size_t dst_off =
                        ((size_t)p * NH + (size_t)h) * half;
                    std::memcpy(&cos_bcast[dst_off], &cos_host[src_off],
                                 (size_t)half * sizeof(uint16_t));
                    std::memcpy(&sin_bcast[dst_off], &sin_host[src_off],
                                 (size_t)half * sizeof(uint16_t));
                }
            }
            const size_t bc_bytes = cos_bcast.size() * sizeof(uint16_t);
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_cos_bcast_dev_,
                                              bc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_cos_bcast_dev_, bc_bytes,
                                              cos_bcast.data(), bc_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_sin_bcast_dev_,
                                              bc_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_sin_bcast_dev_, bc_bytes,
                                              sin_bcast.data(), bc_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            stats_.rope_bytes += 2 * bc_bytes;
        }

        // Build "full" cos/sin tables for aclnnRotaryPositionEmbedding
        // interleave-mode dispatch. Shape [total_pos, head_dim] with each
        // pair's cos/sin duplicated across both elements of the pair.
        {
            const int64_t HD = cfg_.head_dim;
            const int64_t half = HD / 2;
            std::vector<uint16_t> cos_full((size_t)total_pos * HD, 0);
            std::vector<uint16_t> sin_full((size_t)total_pos * HD, 0);
            for (int64_t p = 0; p < total_pos; ++p) {
                for (int64_t dp = 0; dp < half; ++dp) {
                    uint16_t c = cos_host[(size_t)p * half + dp];
                    uint16_t s = sin_host[(size_t)p * half + dp];
                    cos_full[(size_t)p * HD + 2*dp + 0] = c;
                    cos_full[(size_t)p * HD + 2*dp + 1] = c;
                    sin_full[(size_t)p * HD + 2*dp + 0] = s;
                    sin_full[(size_t)p * HD + 2*dp + 1] = s;
                }
            }
            const size_t cf_bytes = cos_full.size() * sizeof(uint16_t);
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_cos_full_dev_,
                                              cf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_cos_full_dev_, cf_bytes,
                                              cos_full.data(), cf_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            QIE_ACL_CHECK(g_cann.aclrtMalloc(&scratch_rope_sin_full_dev_,
                                              cf_bytes, ACL_MEM_MALLOC_HUGE_FIRST));
            QIE_ACL_CHECK(g_cann.aclrtMemcpy(scratch_rope_sin_full_dev_, cf_bytes,
                                              sin_full.data(), cf_bytes,
                                              ACL_MEMCPY_HOST_TO_DEVICE));
            stats_.rope_bytes += 2 * cf_bytes;
        }
    }

    // Scratch allocations — same layout as init_from_gguf.
    const int64_t H      = cfg_.hidden_size;
    const int64_t FF_DIM = (int64_t)H * cfg_.ff_mult;
    const int64_t SEQ    = (int64_t)cfg_.max_img_seq + cfg_.max_txt_seq;
    const size_t F16     = sizeof(uint16_t);
    auto try_alloc = [&](void **ptr, size_t bytes) -> bool {
        aclError err = g_cann.aclrtMalloc(ptr, bytes,
                                           ACL_MEM_MALLOC_HUGE_FIRST);
        if (err != 0) { QIE_LOG("init_for_smoke: aclrtMalloc(%zu) err=%d",
                                 bytes, (int)err); *ptr = nullptr;
                          return false; }
        stats_.scratch_bytes += bytes;
        return true;
    };
    if (!try_alloc(&scratch_q_dev_,    (size_t)SEQ * H * F16))  return false;
    if (!try_alloc(&scratch_k_dev_,    (size_t)SEQ * H * F16))  return false;
    if (!try_alloc(&scratch_v_dev_,    (size_t)SEQ * H * F16))  return false;
    if (!try_alloc(&scratch_attn_dev_, (size_t)SEQ * H * F16))  return false;
    if (!try_alloc(&scratch_mlp_dev_,  (size_t)SEQ * FF_DIM * F16)) return false;
    if (!try_alloc(&scratch_mod_dev_,  (size_t)12 * H * F16))      return false;
    if (!try_alloc(&rstd_dev_,         (size_t)cfg_.num_heads
                                         * SEQ * sizeof(float)))    return false;
    if (!try_alloc(&scratch_img_norm_dev_,
                   (size_t)cfg_.max_img_seq * H * F16)) return false;
    if (!try_alloc(&scratch_txt_norm_dev_,
                   (size_t)cfg_.max_txt_seq * H * F16)) return false;
    if (!try_alloc(&scratch_img_out_dev_,
                   (size_t)cfg_.max_img_seq * H * F16)) return false;
    if (!try_alloc(&scratch_txt_out_dev_,
                   (size_t)cfg_.max_txt_seq * H * F16)) return false;
    if (!try_alloc(&mean_dev_,    (size_t)SEQ * sizeof(float))) return false;
    if (!try_alloc(&ln_rstd_dev_, (size_t)SEQ * sizeof(float))) return false;

    // Phase 4.4c F32-residual scratch (mirrors init_from_gguf). Without
    // these, layer_norm_f32_to_f16_ and gated_residual_add_f32_ RED at
    // block 0 because scratch_residual_tmp_f32_dev_ is null.
    if (!try_alloc(&scratch_img_hidden_f16_dev_,
                   (size_t)cfg_.max_img_seq * H * F16)) return false;
    if (!try_alloc(&scratch_txt_hidden_f16_dev_,
                   (size_t)cfg_.max_txt_seq * H * F16)) return false;
    {
        const int64_t max_stream_seq =
            std::max(cfg_.max_img_seq, cfg_.max_txt_seq);
        if (!try_alloc(&scratch_residual_tmp_f32_dev_,
                       (size_t)max_stream_seq * H * sizeof(float)))
            return false;
    }

    // Phase 4.1 on-device RoPE scratch — three [seq_max, NH, head_dim/2] F16.
    {
        const int64_t NH_ = cfg_.num_heads;
        const int64_t HALF_HD = cfg_.head_dim / 2;
        const size_t rope_half_bytes =
            (size_t)SEQ * NH_ * HALF_HD * F16;
        if (!try_alloc(&scratch_rope_a_dev_, rope_half_bytes)) return false;
        if (!try_alloc(&scratch_rope_b_dev_, rope_half_bytes)) return false;
        if (!try_alloc(&scratch_rope_c_dev_, rope_half_bytes)) return false;
    }

    if (!try_alloc(&workspace_dev_, 4 * 1024 * 1024)) return false;
    workspace_size_ = 4 * 1024 * 1024;

    ready_ = true;
    QIE_LOG("init_for_smoke: OK device=%d hidden=%lld heads=%lld head_dim=%lld "
            "img_seq=%lld txt_seq=%lld (NO WEIGHTS LOADED — caller must "
            "populate layer_w_ via mutable_layer_weights)",
            device_, (long long)H, (long long)cfg_.num_heads,
            (long long)cfg_.head_dim,
            (long long)cfg_.max_img_seq, (long long)cfg_.max_txt_seq);
    return true;
}

// ============================================================================
// Phase 4.5 Step 4: denoise_full — first real end-to-end cat-edit PNG via
// the native engine.
//
// Wraps the 60-block loop (forward_block_) with the five previously missing
// projection / embedding paths: time_linear{1,2}, img_in, txt_in + txt_norm,
// and norm_out + proj_out (AdaLayerNormContinuous head). Per the §5.5 shape
// plan at 256×256:
//
//   init_latent        [32, 32, 16, 1]         F32   (Qwen-Image VAE layout)
//   ref_latent_0       [32, 32, 16, 1]         F32   (optional)
//   cond_c_crossattn   [3584, 214, 1, 1]       F32
//   uncond_c_crossattn [3584, 214, 1, 1]       F32   (optional)
//
// Semantics — faithful to the CPU Euler sampler (denoiser.hpp L831-866):
//   For each step s = 0 .. n_steps-1:
//     sigma    = sigmas[s]
//     denoised = DiT_full_forward(x, sigma)        # shape matches x
//     d        = (x - denoised) / sigma
//     x       += d * (sigmas[s+1] - sigmas[s])
// The "full DiT forward" includes norm_out + proj_out — so these must run
// every step, not once at the end. We match that: host-side x[W_lat,H_lat,
// C_lat,B] is re-patchified per step, pushed through img_in → blocks →
// norm_out + proj_out → unpatchified back to x's shape, and the Euler step
// runs on host over the full unpatched volume.
//
// Host-side preprocessing (once per request):
//   * pad_and_patchify(latent, patch=2): reshape `[W_lat, H_lat, C, B]` →
//     `[B, H_lat/2 * W_lat/2, 2*2*C]` — 16 × patch² = 64-dim per token.
//     At 32×32 latent this produces 256 tokens. Concat init|ref along seq
//     yields `img_seq = 512` tokens. Ref is patchified once; init is
//     re-patchified each step from evolving x_host.
//   * Text conditioning: uploaded F32 on-device once, RMSNorm+txt_in run
//     once (before the loop) to produce per-stream F32 residual that is
//     D2D-copied into the per-step txt working buffer (forward destroys
//     it in-place).
//
// Per-step NPU pipeline:
//   1. t = sigma_s * 1000 → sinusoidal [256] on host (matching
//      ggml_compute_forward_timestep_embedding_f32: cos[0..half),
//      sin[half..dim)). Upload F32 [1, 256], cast F32→F16 into scratch,
//      Q4 Linear time_linear1 [256 → 3072], SiLU in-place, Q4 Linear
//      time_linear2 [3072 → 3072]. Output is F16 [1, 3072] t_emb_s.
//   2. Patchify x_host → concat with ref → F16 upload → img_in Q4 Linear
//      [IN_CH=64 → H=3072] → F32 via Cast. Sets img_res_c_f32 (and
//      img_res_u_f32 if CFG — same projection, differs only in txt cond).
//   3. D2D txt_res_c → txt_work_c. forward_all_blocks_test on (img_res_c,
//      txt_work_c, t_emb_s). At exit, img_res_c_f32 holds the 60-block
//      output hidden stream for cond.
//   4. Compute AdaLN (scale, shift) with this step's t_emb_s:
//         silu = SiLU(t_emb_s)
//         emb  = Linear(silu) → split → scale [1,H], shift [1,H]
//      LayerNorm(img_res_c_f32) → modulate → proj_out Q4 Linear [H → 64]
//      → F16 [img_seq, 64] cond noise prediction.
//   5. (run_uncond) Same steps 3-4 with uncond txt stream. CFG compose:
//         eps = eps_uncond + cfg*(eps_cond - eps_uncond)
//      (F16 InplaceAdd).
//   6. D2H the first init_img_tokens × PATCH_OUT elements (drop ref
//      tokens — they're carried during forward for attention mixing but
//      not used in the Euler update). Host F16→F32, unpatchify to
//      denoised_host [W_lat, H_lat, out_channels=16, B].
//   7. Host Euler: x_host[j] += (x_host[j] - denoised_host[j]) / sigma *
//                                 (sigmas[s+1] - sigma).
//
// Scratch footprint per request (at 256×256 production shape):
//   ~30 MiB on top of the ~8 GiB resident weights. Comfortably fits in
//   the 32 GiB HBM budget.
// ============================================================================

// Host-side helpers for denoise_full.
namespace {

// Host patchify: input `latent` is laid out Qwen-Image VAE convention
// `[ne0=W_lat, ne1=H_lat, ne2=C_lat, ne3=B]` row-major in ne0. Equivalent
// of DiT::pad_and_patchify(patch_size=2) with no padding required when
// W_lat and H_lat are already multiples of patch.
// Output buffer `out` is written as F32 `[B, seq, patch²*C_lat]` row-major:
//   seq = (H_lat/patch) * (W_lat/patch), out layout per-token is
//   `[ch, py, px]` flattened (C-contiguous over patch² × C).
// Matches the CPU reference logic from common_dit.hpp::pad_and_patchify
// (convert to tokens in row-major patch sweep).
static void host_patchify_latent(const float *latent,
                                  int64_t W_lat, int64_t H_lat,
                                  int64_t C_lat, int64_t B,
                                  int patch,
                                  std::vector<float> &out,
                                  int64_t &seq_out,
                                  int64_t &token_dim_out) {
    const int64_t H_tok = H_lat / patch;
    const int64_t W_tok = W_lat / patch;
    seq_out = H_tok * W_tok;
    token_dim_out = (int64_t)patch * patch * C_lat;
    out.assign((size_t)B * seq_out * token_dim_out, 0.0f);

    // `latent[w, h, c, b]` index in row-major with ne0=W_lat fastest:
    //   lat_idx = ((b * C_lat + c) * H_lat + h) * W_lat + w
    // Token (ty, tx) covers the patch block w in [tx*patch .. tx*patch+patch),
    // h in [ty*patch .. ty*patch+patch). The per-token layout matches
    // diffusers' convention: channel-major then py, px.
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t ty = 0; ty < H_tok; ++ty) {
            for (int64_t tx = 0; tx < W_tok; ++tx) {
                const int64_t tok = ty * W_tok + tx;
                float *dst = &out[((size_t)b * seq_out + tok) * token_dim_out];
                int64_t k = 0;
                for (int64_t c = 0; c < C_lat; ++c) {
                    for (int64_t py = 0; py < patch; ++py) {
                        for (int64_t px = 0; px < patch; ++px) {
                            const int64_t h = ty * patch + py;
                            const int64_t w = tx * patch + px;
                            const size_t lat_idx =
                                (((size_t)b * C_lat + c) * H_lat + h) *
                                    W_lat + w;
                            dst[k++] = latent[lat_idx];
                        }
                    }
                }
            }
        }
    }
}

// Inverse of host_patchify_latent: given flat F32 `[B, seq, patch²*C_lat]`
// in the same layout produced above, scatter back to `[W_lat, H_lat, C_lat, B]`.
static void host_unpatchify_latent(const float *tokens,
                                    int64_t W_lat, int64_t H_lat,
                                    int64_t C_lat, int64_t B,
                                    int patch,
                                    float *out_latent) {
    const int64_t H_tok = H_lat / patch;
    const int64_t W_tok = W_lat / patch;
    const int64_t seq   = H_tok * W_tok;
    const int64_t token_dim = (int64_t)patch * patch * C_lat;
    std::memset(out_latent, 0,
                 (size_t)B * C_lat * H_lat * W_lat * sizeof(float));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t ty = 0; ty < H_tok; ++ty) {
            for (int64_t tx = 0; tx < W_tok; ++tx) {
                const int64_t tok = ty * W_tok + tx;
                const float *src = &tokens[((size_t)b * seq + tok) * token_dim];
                int64_t k = 0;
                for (int64_t c = 0; c < C_lat; ++c) {
                    for (int64_t py = 0; py < patch; ++py) {
                        for (int64_t px = 0; px < patch; ++px) {
                            const int64_t h = ty * patch + py;
                            const int64_t w = tx * patch + px;
                            const size_t lat_idx =
                                (((size_t)b * C_lat + c) * H_lat + h) *
                                    W_lat + w;
                            out_latent[lat_idx] = src[k++];
                        }
                    }
                }
            }
        }
    }
}

// Host sinusoidal timestep embedding — matches the ggml CPU reference
// `ggml_compute_forward_timestep_embedding_f32` layout: dim=256 total,
// `[cos(arg_0), cos(arg_1), ..., cos(arg_half-1), sin(arg_0), ...,
//   sin(arg_half-1)]` for `arg_j = timestep * exp(-log(max_period)*j/half)`.
// This is IMPORTANT — the engine's earlier `build_time_emb_` used the
// interleaved convention which doesn't match the trained time_linear1
// weight layout.
static void host_timestep_embedding_f32(float timestep, int dim,
                                          int max_period,
                                          std::vector<float> &out) {
    out.assign((size_t)dim, 0.0f);
    const int half = dim / 2;
    for (int j = 0; j < half; ++j) {
        float freq = std::exp(-std::log((float)max_period) * (float)j /
                               (float)half);
        float arg = timestep * freq;
        out[(size_t)j] = std::cos(arg);
        out[(size_t)j + half] = std::sin(arg);
    }
    if (dim % 2 != 0) {
        out[(size_t)2 * half] = 0.0f;
    }
}

// Read whole file into a vector. Returns file size; -1 on error.
static long slurp_file(const std::string &path, std::vector<uint8_t> &out) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) return -1;
    std::fseek(f, 0, SEEK_END);
    long fsize = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (fsize <= 0) { std::fclose(f); return -1; }
    out.assign((size_t)fsize, 0);
    size_t nread = std::fread(out.data(), 1, (size_t)fsize, f);
    std::fclose(f);
    if (nread != (size_t)fsize) return -1;
    return fsize;
}

}  // namespace

bool ImageDiffusionEngine::init_from_dump(const std::string &dump_dir,
                                            int64_t W_lat, int64_t H_lat,
                                            int64_t C_lat, int64_t B,
                                            int64_t joint_dim,
                                            std::vector<float> &initial_latent_out,
                                            std::vector<float> &ref_latent_out,
                                            std::vector<float> &txt_cond_out,
                                            std::vector<float> &txt_uncond_out,
                                            int64_t &txt_seq_out,
                                            bool &has_ref_out,
                                            bool &has_uncond_out) {
    initial_latent_out.clear();
    ref_latent_out.clear();
    txt_cond_out.clear();
    txt_uncond_out.clear();
    has_ref_out = false;
    has_uncond_out = false;
    txt_seq_out = 0;

    const size_t latent_elems = (size_t)W_lat * H_lat * C_lat * B;
    const size_t latent_bytes = latent_elems * sizeof(float);

    // Required: init_latent.f32.bin
    {
        std::vector<uint8_t> raw;
        long fsize = slurp_file(dump_dir + "/init_latent.f32.bin", raw);
        if (fsize < 0) {
            QIE_LOG("init_from_dump: missing %s/init_latent.f32.bin",
                    dump_dir.c_str());
            return false;
        }
        if ((size_t)fsize != latent_bytes) {
            QIE_LOG("init_from_dump: init_latent size %ld != expected %zu "
                    "(W=%lld H=%lld C=%lld B=%lld)",
                    fsize, latent_bytes, (long long)W_lat, (long long)H_lat,
                    (long long)C_lat, (long long)B);
            return false;
        }
        initial_latent_out.assign(latent_elems, 0.0f);
        std::memcpy(initial_latent_out.data(), raw.data(), latent_bytes);
    }

    // Optional: ref_latent_0.f32.bin
    {
        std::vector<uint8_t> raw;
        long fsize = slurp_file(dump_dir + "/ref_latent_0.f32.bin", raw);
        if (fsize >= 0) {
            if ((size_t)fsize != latent_bytes) {
                QIE_LOG("init_from_dump: ref_latent_0 size %ld != expected %zu",
                        fsize, latent_bytes);
                return false;
            }
            ref_latent_out.assign(latent_elems, 0.0f);
            std::memcpy(ref_latent_out.data(), raw.data(), latent_bytes);
            has_ref_out = true;
        }
    }

    // Required: cond_c_crossattn.f32.bin. Size = joint_dim * txt_seq * 4 B.
    {
        std::vector<uint8_t> raw;
        long fsize = slurp_file(dump_dir + "/cond_c_crossattn.f32.bin", raw);
        if (fsize < 0) {
            QIE_LOG("init_from_dump: missing %s/cond_c_crossattn.f32.bin",
                    dump_dir.c_str());
            return false;
        }
        if ((size_t)fsize % (joint_dim * sizeof(float)) != 0) {
            QIE_LOG("init_from_dump: cond_c_crossattn size %ld not multiple "
                    "of joint_dim*4 (joint_dim=%lld)",
                    fsize, (long long)joint_dim);
            return false;
        }
        const size_t elems = (size_t)fsize / sizeof(float);
        txt_seq_out = (int64_t)(elems / (size_t)joint_dim);
        txt_cond_out.assign(elems, 0.0f);
        std::memcpy(txt_cond_out.data(), raw.data(), (size_t)fsize);
    }

    // Optional: uncond_c_crossattn.f32.bin
    {
        std::vector<uint8_t> raw;
        long fsize = slurp_file(dump_dir + "/uncond_c_crossattn.f32.bin", raw);
        if (fsize >= 0) {
            if ((size_t)fsize != txt_cond_out.size() * sizeof(float)) {
                QIE_LOG("init_from_dump: uncond_c_crossattn size %ld != cond "
                        "size %zu — skipping uncond (will force cfg=1.0)",
                        fsize, txt_cond_out.size() * sizeof(float));
            } else {
                txt_uncond_out.assign(txt_cond_out.size(), 0.0f);
                std::memcpy(txt_uncond_out.data(), raw.data(), (size_t)fsize);
                has_uncond_out = true;
            }
        }
    }

    QIE_LOG("init_from_dump OK: dir=%s lat=[%lld,%lld,%lld,%lld] "
            "txt=[%lld x %lld] has_ref=%d has_uncond=%d",
            dump_dir.c_str(),
            (long long)W_lat, (long long)H_lat, (long long)C_lat, (long long)B,
            (long long)joint_dim, (long long)txt_seq_out,
            (int)has_ref_out, (int)has_uncond_out);
    return true;
}

bool ImageDiffusionEngine::denoise_full(const float *initial_latent,
                                          const float *ref_latent,
                                          int64_t W_lat, int64_t H_lat,
                                          int64_t C_lat, int64_t B,
                                          const float *txt_cond,
                                          const float *txt_uncond,
                                          int64_t txt_seq, int64_t joint_dim,
                                          const float *sigmas, int n_steps,
                                          float cfg_scale,
                                          float *out_latent,
                                          double *per_step_ms) {
    if (!ready_) {
        QIE_LOG("denoise_full: engine not ready");
        return false;
    }
    if (!initial_latent || !txt_cond || !sigmas || !out_latent ||
        n_steps < 1 || B != 1) {
        QIE_LOG("denoise_full: bad args (B must be 1 for Step 4)");
        return false;
    }
    const int PATCH = cfg_.patch_size;
    if (W_lat % PATCH != 0 || H_lat % PATCH != 0) {
        QIE_LOG("denoise_full: W_lat/H_lat must be multiples of patch=%d "
                "(got %lld x %lld)",
                PATCH, (long long)W_lat, (long long)H_lat);
        return false;
    }
    if (joint_dim != cfg_.joint_attention_dim) {
        QIE_LOG("denoise_full: joint_dim mismatch got=%lld cfg=%d",
                (long long)joint_dim, cfg_.joint_attention_dim);
        return false;
    }
    if (C_lat * PATCH * PATCH != cfg_.in_channels) {
        QIE_LOG("denoise_full: C_lat*patch² != in_channels (got %lld != %d)",
                (long long)C_lat * PATCH * PATCH, cfg_.in_channels);
        return false;
    }

    const int64_t H = cfg_.hidden_size;
    const int64_t IN_CH = cfg_.in_channels;            // patch² * C_lat = 64
    const int64_t PATCH_OUT = (int64_t)PATCH * PATCH * cfg_.out_channels;  // 64
    const int64_t init_img_tokens = (H_lat / PATCH) * (W_lat / PATCH);
    const int64_t ref_img_tokens  = ref_latent ? init_img_tokens : 0;
    const int64_t img_seq         = init_img_tokens + ref_img_tokens;

    if (img_seq > cfg_.max_img_seq) {
        QIE_LOG("denoise_full: img_seq=%lld > max_img_seq=%d",
                (long long)img_seq, cfg_.max_img_seq);
        return false;
    }
    if (txt_seq > cfg_.max_txt_seq) {
        QIE_LOG("denoise_full: txt_seq=%lld > max_txt_seq=%d — "
                "continuing but RoPE table was built for max_txt_seq; "
                "numerics may drift (§5.5 pe-rebuild TODO)",
                (long long)txt_seq, cfg_.max_txt_seq);
    }

    const bool run_uncond = (cfg_scale != 1.0f) && txt_uncond != nullptr;

    QIE_LOG("denoise_full: W_lat=%lld H_lat=%lld C_lat=%lld B=%lld "
            "img_tokens=%lld+%lld=%lld txt_seq=%lld joint_dim=%lld "
            "n_steps=%d cfg=%.2f run_uncond=%d",
            (long long)W_lat, (long long)H_lat, (long long)C_lat, (long long)B,
            (long long)init_img_tokens, (long long)ref_img_tokens,
            (long long)img_seq, (long long)txt_seq, (long long)joint_dim,
            n_steps, cfg_scale, (int)run_uncond);

    // ------------------------------------------------------------------
    // Host-side patchify of the REF latent: ref is static across steps
    // (it's the VAE-encoded conditioning image). We patchify it once on
    // host, then upload once and re-use every step. The INIT latent is
    // re-patchified each step since `x` evolves with Euler updates.
    // ------------------------------------------------------------------
    std::vector<float> ref_tokens_f32;  // [B, init_img_tokens, IN_CH] or empty
    if (ref_latent) {
        int64_t rs = 0, rtd = 0;
        host_patchify_latent(ref_latent, W_lat, H_lat, C_lat, B, PATCH,
                              ref_tokens_f32, rs, rtd);
        if (rs != init_img_tokens || rtd != IN_CH) {
            QIE_LOG("denoise_full: ref patchify shape mismatch");
            return false;
        }
    }

    // `x_host` is the evolving latent `[W_lat, H_lat, C_lat, B]` F32. We
    // operate Euler on host (per-step update volume is tiny, ~64 KiB at
    // 256×256) to match the CPU reference's sample_euler exactly:
    //   for each step:
    //     denoised = DiT_full_forward(x, sigma_s)        // [W_lat,H_lat,C_lat,B]
    //     d = (x - denoised) / sigma_s
    //     x += d * (sigmas[s+1] - sigmas[s])
    // This matches tools/ominix_diffusion/src/denoiser.hpp EULER_SAMPLE_METHOD.
    std::vector<float> x_host((size_t)W_lat * H_lat * C_lat * B, 0.0f);
    std::memcpy(x_host.data(), initial_latent, x_host.size() * sizeof(float));

    // ------------------------------------------------------------------
    // Device allocation for denoise_full. All buffers freed in a single
    // cleanup lambda at the end.
    // ------------------------------------------------------------------
    struct DevBuf { void *p = nullptr; };
    std::vector<DevBuf> owned;
    auto alloc_bytes = [&](size_t bytes) -> void * {
        void *p = nullptr;
        aclError e = g_cann.aclrtMalloc(&p, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
        if (e != 0) { QIE_LOG("denoise_full: aclrtMalloc(%zu) err=%d",
                                bytes, (int)e); return nullptr; }
        owned.push_back({p});
        return p;
    };
    auto free_owned = [&]() {
        for (auto &b : owned) if (b.p) g_cann.aclrtFree(b.p);
        owned.clear();
    };

    const size_t F32 = sizeof(float);
    const size_t F16 = sizeof(uint16_t);

    // --- activations ---
    // img_in input: F16 [img_seq, IN_CH] — patchified concat(x, ref) per step.
    void *img_in_in_f16  = alloc_bytes((size_t)img_seq * IN_CH * F16);
    // img residual (F32) — RE-SEEDED per step from the F32 img_in projection.
    // Two copies for cond/uncond working buffers.
    void *img_res_c_f32  = alloc_bytes((size_t)img_seq * H * F32);
    void *img_res_u_f32  = run_uncond
        ? alloc_bytes((size_t)img_seq * H * F32) : nullptr;
    // txt activations: cond/uncond F16 staging (after txt_norm+txt_in) +
    // F32 residual (input to forward) + working copies (destroyed in-place).
    void *txt_in_out_c_f16 = alloc_bytes((size_t)txt_seq * H * F16);
    void *txt_res_c_f32    = alloc_bytes((size_t)txt_seq * H * F32);
    void *txt_work_c_f32   = alloc_bytes((size_t)txt_seq * H * F32);
    void *txt_in_out_u_f16 = run_uncond
        ? alloc_bytes((size_t)txt_seq * H * F16) : nullptr;
    void *txt_res_u_f32    = run_uncond
        ? alloc_bytes((size_t)txt_seq * H * F32) : nullptr;
    void *txt_work_u_f32   = run_uncond
        ? alloc_bytes((size_t)txt_seq * H * F32) : nullptr;

    // Text conditioning F32 on device (cond + uncond).
    void *txt_cond_f32_dev   = alloc_bytes((size_t)txt_seq * joint_dim * F32);
    void *txt_uncond_f32_dev = run_uncond
        ? alloc_bytes((size_t)txt_seq * joint_dim * F32) : nullptr;

    // t_emb staging.
    void *t_emb_in_f32    = alloc_bytes(256 * F32);  // sinusoidal input host→dev
    void *t_emb_in_f16    = alloc_bytes(256 * F16);
    void *t_emb_mid_f16   = alloc_bytes((size_t)H * F16);  // time_linear1 out
    void *t_emb_out_f16   = alloc_bytes((size_t)H * F16);  // time_linear2 out

    // norm_out + proj_out scratch.
    void *adaln_silu_f16  = alloc_bytes((size_t)H * F16);
    void *adaln_emb_f16   = alloc_bytes((size_t)2 * H * F16);
    void *proj_out_in_f16  = alloc_bytes((size_t)img_seq * H * F16);
    void *proj_out_out_f16 = alloc_bytes((size_t)img_seq * PATCH_OUT * F16);

    // Per-stream txt_norm F16 landing zone (joint_dim > H, so we can't
    // reuse scratch_txt_norm_dev_ which is sized for hidden).
    void *txt_norm_out_c_f16 =
        alloc_bytes((size_t)txt_seq * joint_dim * F16);
    void *txt_norm_out_u_f16 = run_uncond
        ? alloc_bytes((size_t)txt_seq * joint_dim * F16) : nullptr;

    if (!img_in_in_f16 || !img_res_c_f32 ||
        (run_uncond && !img_res_u_f32) ||
        !txt_in_out_c_f16 || !txt_res_c_f32 || !txt_work_c_f32 ||
        (run_uncond && (!txt_in_out_u_f16 || !txt_res_u_f32 ||
                         !txt_work_u_f32 || !txt_uncond_f32_dev ||
                         !txt_norm_out_u_f16)) ||
        !txt_cond_f32_dev || !txt_norm_out_c_f16 ||
        !t_emb_in_f32 || !t_emb_in_f16 || !t_emb_mid_f16 || !t_emb_out_f16 ||
        !adaln_silu_f16 || !adaln_emb_f16 ||
        !proj_out_in_f16 || !proj_out_out_f16) {
        free_owned();
        return false;
    }

    auto sync_ok = [&](const char *where) -> bool {
        aclError e = g_cann.aclrtSynchronizeStream(compute_stream_);
        if (e != 0) {
            QIE_LOG("denoise_full: sync after %s err=%d", where, (int)e);
            return false;
        }
        return true;
    };

    auto h2d_f32 = [&](void *dst, const float *src, size_t n) -> bool {
        aclError e = g_cann.aclrtMemcpy(dst, n * F32, src, n * F32,
                                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (e != 0) { QIE_LOG("denoise_full: H2D(F32, %zu) err=%d",
                                n, (int)e); return false; }
        return true;
    };
    auto h2d_f16 = [&](void *dst, const uint16_t *src, size_t n) -> bool {
        aclError e = g_cann.aclrtMemcpy(dst, n * F16, src, n * F16,
                                          ACL_MEMCPY_HOST_TO_DEVICE);
        if (e != 0) { QIE_LOG("denoise_full: H2D(F16, %zu) err=%d",
                                n, (int)e); return false; }
        return true;
    };

    // Inline F16→F32 cast on a flat [rows, cols] F16 buffer → F32 buffer.
    auto cast_f16_to_f32 = [&](void *in_f16, void *out_f32,
                                 int64_t rows, int64_t cols) -> bool {
        int64_t sh[2] = {rows, cols};
        int64_t st[2] = {cols, 1};
        aclTensor *t_in  = tensor_nd_f16(in_f16,  2, sh, st);
        aclTensor *t_out = tensor_nd_f32(out_f32, 2, sh, st);
        uint64_t ws = 0; aclOpExecutor *exec = nullptr;
        aclnnStatus s = g_cann.aclnnCastGetWorkspaceSize(
            t_in, ACL_FLOAT, t_out, &ws, &exec);
        if (s == 0) {
            ensure_workspace_(ws);
            s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr,
                                   ws, exec, compute_stream_);
        }
        g_cann.aclDestroyTensor(t_in);
        g_cann.aclDestroyTensor(t_out);
        return s == 0;
    };

    // Phase 4.5.4b NaN bisect: on step 0 only, copy a device buffer back to
    // host, compute stats (mean-abs, max-abs, NaN count, Inf count), log.
    // Gated by QIE_DEBUG_NAN_BISECT=1. Keep the D2H cheap (<= 2 MiB at
    // 256×256 production shape) so we don't perturb per-step wall much.
    static int s_nan_bisect = -1;
    if (s_nan_bisect < 0) {
        const char *v = std::getenv("QIE_DEBUG_NAN_BISECT");
        s_nan_bisect = v ? std::atoi(v) : 0;
    }
    auto probe_stats = [&](const char *label, void *dev,
                             int64_t n_elts, bool is_f16,
                             int step) -> void {
        if (!s_nan_bisect || step != 0) return;
        // Sync to force the preceding op to commit before D2H.
        g_cann.aclrtSynchronizeStream(compute_stream_);
        const size_t bpe = is_f16 ? 2 : 4;
        std::vector<uint8_t> host((size_t)n_elts * bpe);
        aclError me = g_cann.aclrtMemcpy(host.data(), host.size(), dev,
                                           host.size(),
                                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (me != 0) {
            QIE_LOG("probe_stats[%s]: D2H err=%d", label, (int)me);
            return;
        }
        double sum_abs = 0.0, max_abs = 0.0;
        int64_t nanc = 0, infc = 0;
        for (int64_t i = 0; i < n_elts; ++i) {
            float v;
            if (is_f16) {
                __fp16 hh;
                std::memcpy(&hh, host.data() + (size_t)i * 2, 2);
                v = (float)hh;
            } else {
                std::memcpy(&v, host.data() + (size_t)i * 4, 4);
            }
            if (std::isnan(v)) { nanc++; continue; }
            if (std::isinf(v)) { infc++; continue; }
            double a = std::fabs((double)v);
            sum_abs += a;
            if (a > max_abs) max_abs = a;
        }
        int64_t valid = n_elts - nanc - infc;
        double mean_abs = valid > 0 ? sum_abs / (double)valid : 0.0;
        QIE_LOG("probe_stats[%s]: n=%lld dtype=%s mean_abs=%.4g "
                "max_abs=%.4g NaN=%lld Inf=%lld",
                label, (long long)n_elts,
                is_f16 ? "F16" : "F32", mean_abs, max_abs,
                (long long)nanc, (long long)infc);
    };

    // ------------------------------------------------------------------
    // Upload text conditioning + run one-shot txt_norm + txt_in for cond
    // and uncond. These don't change across steps — results park in
    // txt_res_{c,u}_f32 and are D2D-copied into txt_work_{c,u}_f32 per
    // step (forward destroys in-place).
    // ------------------------------------------------------------------
    if (!h2d_f32(txt_cond_f32_dev, txt_cond,
                  (size_t)txt_seq * joint_dim)) {
        free_owned(); return false;
    }
    if (run_uncond) {
        if (!h2d_f32(txt_uncond_f32_dev, txt_uncond,
                      (size_t)txt_seq * joint_dim)) {
            free_owned(); return false;
        }
    }

    // txt_norm (RMSNorm) + txt_in (cond).
    probe_stats("pre_txt_norm.cond", txt_cond_f32_dev,
                txt_seq * joint_dim, /*f16*/false, 0);
    if (!rms_norm_row_f32_to_f16_(txt_cond_f32_dev, txt_norm_out_c_f16,
                                    global_w_.txt_norm_w,
                                    txt_seq, joint_dim)) {
        QIE_LOG("denoise_full: txt_norm(cond) failed");
        free_owned(); return false;
    }
    probe_stats("post_txt_norm.cond", txt_norm_out_c_f16,
                txt_seq * joint_dim, /*f16*/true, 0);
    if (!dispatch_matmul_(txt_norm_out_c_f16, global_w_.txt_in_w_q4,
                           global_w_.txt_in_scale, global_w_.txt_in_b,
                           txt_seq, joint_dim, H, txt_in_out_c_f16)) {
        QIE_LOG("denoise_full: txt_in matmul(cond) failed");
        free_owned(); return false;
    }
    probe_stats("post_txt_in.cond", txt_in_out_c_f16,
                txt_seq * H, /*f16*/true, 0);
    if (!cast_f16_to_f32(txt_in_out_c_f16, txt_res_c_f32, txt_seq, H)) {
        QIE_LOG("denoise_full: txt(cond) F16→F32 err");
        free_owned(); return false;
    }
    probe_stats("txt_res_c_f32", txt_res_c_f32,
                txt_seq * H, /*f16*/false, 0);

    if (run_uncond) {
        if (!rms_norm_row_f32_to_f16_(txt_uncond_f32_dev, txt_norm_out_u_f16,
                                        global_w_.txt_norm_w,
                                        txt_seq, joint_dim)) {
            QIE_LOG("denoise_full: txt_norm(uncond) failed");
            free_owned(); return false;
        }
        if (!dispatch_matmul_(txt_norm_out_u_f16, global_w_.txt_in_w_q4,
                               global_w_.txt_in_scale, global_w_.txt_in_b,
                               txt_seq, joint_dim, H, txt_in_out_u_f16)) {
            QIE_LOG("denoise_full: txt_in matmul(uncond) failed");
            free_owned(); return false;
        }
        if (!cast_f16_to_f32(txt_in_out_u_f16, txt_res_u_f32, txt_seq, H)) {
            QIE_LOG("denoise_full: txt(uncond) F16→F32 err");
            free_owned(); return false;
        }
    }

    if (!sync_ok("txt conditioning setup")) { free_owned(); return false; }

    // ------------------------------------------------------------------
    // Per-step denoising loop — FULL per-step forward (re-patchify x →
    // img_in → blocks → norm_out + proj_out → unpatchify → Euler on host).
    //
    // Reference: tools/ominix_diffusion/src/denoiser.hpp line 831-866
    // (EULER_SAMPLE_METHOD) — for each step the sampler calls the model
    // wrapper `model(x, sigma, i+1)` which returns `denoised` at the same
    // shape as `x`, then does:
    //     d = (x - denoised) / sigma
    //     x += d * (sigmas[s+1] - sigmas[s])
    //
    // Our per-step body mirrors exactly this, with the full DiT forward
    // on-device.
    // ------------------------------------------------------------------
    void *pe_dev = global_w_.rope_pe_dev;

    // Per-step host buffers. Reused across steps via resize-only-if-grow.
    std::vector<float>    concat_tokens_f32((size_t)img_seq * IN_CH, 0.0f);
    std::vector<uint16_t> img_in_f16((size_t)img_seq * IN_CH, 0);
    std::vector<uint16_t> out_tokens_f16((size_t)init_img_tokens * PATCH_OUT,
                                          0);
    std::vector<float>    out_tokens_f32(out_tokens_f16.size(), 0.0f);
    std::vector<float>    denoised_host((size_t)W_lat * H_lat *
                                         cfg_.out_channels * B, 0.0f);

    for (int step = 0; step < n_steps; ++step) {
        auto t_step_0 = std::chrono::steady_clock::now();

        const float sigma = sigmas[step];

        // --- (1) Timestep embedding for this step. ---
        // Qwen-Image sigma_to_t = sigma * 1000.0.
        const float t_val = sigma * 1000.0f;
        std::vector<float> t_sinu_f32;
        host_timestep_embedding_f32(t_val, 256, 10000, t_sinu_f32);
        if (!h2d_f32(t_emb_in_f32, t_sinu_f32.data(), t_sinu_f32.size())) {
            free_owned(); return false;
        }
        // F32 → F16 Cast.
        {
            int64_t sh[2] = {1, 256};
            int64_t st[2] = {256, 1};
            aclTensor *t_in  = tensor_nd_f32(t_emb_in_f32, 2, sh, st);
            aclTensor *t_out = tensor_nd_f16(t_emb_in_f16, 2, sh, st);
            uint64_t ws = 0; aclOpExecutor *exec = nullptr;
            aclnnStatus s = g_cann.aclnnCastGetWorkspaceSize(
                t_in, ACL_FLOAT16, t_out, &ws, &exec);
            if (s == 0) {
                ensure_workspace_(ws);
                s = g_cann.aclnnCast(ws > 0 ? workspace_dev_ : nullptr,
                                       ws, exec, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_in);
            g_cann.aclDestroyTensor(t_out);
            if (s != 0) { QIE_LOG("denoise_full: t_emb F32→F16 err=%d",
                                    (int)s); free_owned(); return false; }
        }
        // time_linear1 + SiLU + time_linear2.
        probe_stats("pre_time_linear1", t_emb_in_f16,
                    256, /*f16*/true, step);
        if (!dispatch_matmul_(t_emb_in_f16,
                               global_w_.time_linear1_w_q4,
                               global_w_.time_linear1_scale,
                               global_w_.time_linear1_b,
                               1, 256, H, t_emb_mid_f16)) {
            QIE_LOG("denoise_full: time_linear1 failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_time_linear1", t_emb_mid_f16, H,
                    /*f16*/true, step);
        {
            int64_t sh[2] = {1, H};
            int64_t st[2] = {H, 1};
            aclTensor *t_in  = tensor_nd_f16(t_emb_mid_f16, 2, sh, st);
            aclTensor *t_out = tensor_nd_f16(t_emb_mid_f16, 2, sh, st);
            uint64_t ws = 0; aclOpExecutor *exec = nullptr;
            aclnnStatus s = g_cann.aclnnSiluGetWorkspaceSize(
                t_in, t_out, &ws, &exec);
            if (s == 0) {
                ensure_workspace_(ws);
                s = g_cann.aclnnSilu(ws > 0 ? workspace_dev_ : nullptr,
                                       ws, exec, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_in);
            g_cann.aclDestroyTensor(t_out);
            if (s != 0) { QIE_LOG("denoise_full: SiLU(t_emb) err=%d",
                                    (int)s); free_owned(); return false; }
        }
        probe_stats("post_time_silu", t_emb_mid_f16, H,
                    /*f16*/true, step);
        if (!dispatch_matmul_(t_emb_mid_f16,
                               global_w_.time_linear2_w_q4,
                               global_w_.time_linear2_scale,
                               global_w_.time_linear2_b,
                               1, H, H, t_emb_out_f16)) {
            QIE_LOG("denoise_full: time_linear2 failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_time_linear2", t_emb_out_f16, H,
                    /*f16*/true, step);

        // --- (2) Host patchify(x) → concat with ref → upload as F16. ---
        {
            std::vector<float> init_tokens;
            int64_t rs = 0, rtd = 0;
            host_patchify_latent(x_host.data(), W_lat, H_lat, C_lat, B,
                                   PATCH, init_tokens, rs, rtd);
            if (rs != init_img_tokens || rtd != IN_CH) {
                QIE_LOG("denoise_full: patchify(x) shape mismatch");
                free_owned(); return false;
            }
            std::memcpy(concat_tokens_f32.data(), init_tokens.data(),
                         init_tokens.size() * sizeof(float));
            if (ref_latent) {
                std::memcpy(concat_tokens_f32.data() +
                                (size_t)init_img_tokens * IN_CH,
                             ref_tokens_f32.data(),
                             ref_tokens_f32.size() * sizeof(float));
            }
            for (size_t i = 0; i < img_in_f16.size(); ++i) {
                img_in_f16[i] = fp32_to_fp16(concat_tokens_f32[i]);
            }
            if (!h2d_f16(img_in_in_f16, img_in_f16.data(),
                          img_in_f16.size())) {
                free_owned(); return false;
            }
        }

        // --- (3) img_in → F16 [img_seq, H] → Cast F32 into img_res_c_f32. ---
        probe_stats("pre_img_in", img_in_in_f16, img_seq * IN_CH,
                    /*f16*/true, step);
        if (!dispatch_matmul_(img_in_in_f16,
                               global_w_.img_in_w_q4, global_w_.img_in_scale,
                               global_w_.img_in_b,
                               img_seq, IN_CH, H, scratch_img_norm_dev_)) {
            QIE_LOG("denoise_full: img_in matmul failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_img_in", scratch_img_norm_dev_, img_seq * H,
                    /*f16*/true, step);
        if (!cast_f16_to_f32(scratch_img_norm_dev_, img_res_c_f32,
                               img_seq, H)) {
            QIE_LOG("denoise_full: img_in F16→F32 err step=%d", step);
            free_owned(); return false;
        }
        probe_stats("img_res_c_f32.seed", img_res_c_f32, img_seq * H,
                    /*f16*/false, step);
        // If CFG on, also copy img_res_c → img_res_u for the uncond pass
        // (both passes consume the same patch-projected latent — only the
        // text conditioning differs).
        if (run_uncond) {
            aclError me = g_cann.aclrtMemcpy(img_res_u_f32,
                                               (size_t)img_seq * H * F32,
                                               img_res_c_f32,
                                               (size_t)img_seq * H * F32,
                                               ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (me != 0) { QIE_LOG("denoise_full: D2D img_res cond→uncond "
                                     "err=%d step=%d", (int)me, step);
                             free_owned(); return false; }
        }

        // --- (4) cond pass: txt_res_c → txt_work_c, run 60 blocks. ---
        aclError me = g_cann.aclrtMemcpy(txt_work_c_f32,
                                           (size_t)txt_seq * H * F32,
                                           txt_res_c_f32,
                                           (size_t)txt_seq * H * F32,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE);
        if (me != 0) { QIE_LOG("denoise_full: D2D txt_res_c→work err=%d",
                                 (int)me); free_owned(); return false; }

        if (!forward_all_blocks_test(img_res_c_f32, img_seq,
                                       txt_work_c_f32, txt_seq,
                                       t_emb_out_f16, pe_dev,
                                       nullptr, 0)) {
            QIE_LOG("denoise_full: cond forward failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_blocks.img_res_c", img_res_c_f32, img_seq * H,
                    /*f16*/false, step);
        probe_stats("post_blocks.txt_work_c", txt_work_c_f32, txt_seq * H,
                    /*f16*/false, step);

        // --- (5) norm_out + proj_out on the cond img residual. ---
        //
        // Compute AdaLN (scale, shift) ONCE per step using this step's
        // t_emb. Apply the normalize+modulate pipeline on whichever img
        // residual we're about to project (cond first; uncond after if
        // running CFG).
        //
        // SiLU(t_emb_out) → adaln_silu.
        {
            int64_t sh[2] = {1, H};
            int64_t st[2] = {H, 1};
            aclTensor *t_in  = tensor_nd_f16(t_emb_out_f16, 2, sh, st);
            aclTensor *t_out = tensor_nd_f16(adaln_silu_f16, 2, sh, st);
            uint64_t ws = 0; aclOpExecutor *exec = nullptr;
            aclnnStatus s = g_cann.aclnnSiluGetWorkspaceSize(t_in, t_out,
                                                               &ws, &exec);
            if (s == 0) {
                ensure_workspace_(ws);
                s = g_cann.aclnnSilu(ws > 0 ? workspace_dev_ : nullptr,
                                       ws, exec, compute_stream_);
            }
            g_cann.aclDestroyTensor(t_in);
            g_cann.aclDestroyTensor(t_out);
            if (s != 0) { QIE_LOG("denoise_full: SiLU(norm_out t_emb) err=%d "
                                    "step=%d", (int)s, step);
                            free_owned(); return false; }
        }
        probe_stats("post_adaln_silu", adaln_silu_f16, H,
                    /*f16*/true, step);
        // norm_out.linear: [1, H] × [H, 2H] → [1, 2H] F16.
        if (!dispatch_matmul_(adaln_silu_f16,
                               global_w_.norm_out_linear_w_q4,
                               global_w_.norm_out_linear_scale,
                               global_w_.norm_out_linear_b,
                               1, H, 2 * H, adaln_emb_f16)) {
            QIE_LOG("denoise_full: norm_out Linear failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_norm_out.linear", adaln_emb_f16, 2 * H,
                    /*f16*/true, step);
        void *adaln_scale = adaln_emb_f16;
        void *adaln_shift = (uint8_t *)adaln_emb_f16 + (size_t)H * F16;

        // cond: LayerNorm(img_res_c_f32) → modulate → proj_out → F16 out.
        if (!layer_norm_f32_to_f16_(img_res_c_f32, proj_out_in_f16,
                                      /*B*/1, img_seq, H)) {
            QIE_LOG("denoise_full: norm_out LN(cond) failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_norm_out.LN", proj_out_in_f16, img_seq * H,
                    /*f16*/true, step);
        if (!modulate_(proj_out_in_f16, adaln_scale, adaln_shift,
                        /*B*/1, img_seq, H)) {
            QIE_LOG("denoise_full: norm_out modulate(cond) failed step=%d",
                    step);
            free_owned(); return false;
        }
        probe_stats("post_norm_out.modulate", proj_out_in_f16, img_seq * H,
                    /*f16*/true, step);
        if (!dispatch_matmul_(proj_out_in_f16,
                               global_w_.proj_out_w_q4,
                               global_w_.proj_out_scale,
                               global_w_.proj_out_b,
                               img_seq, H, PATCH_OUT, proj_out_out_f16)) {
            QIE_LOG("denoise_full: proj_out(cond) failed step=%d", step);
            free_owned(); return false;
        }
        probe_stats("post_proj_out", proj_out_out_f16, img_seq * PATCH_OUT,
                    /*f16*/true, step);

        // --- (6) Optional uncond pass. ---
        // Output of cond proj_out is F16 [img_seq, PATCH_OUT] in
        // proj_out_out_f16. For CFG we need to compose cond + uncond
        // predictions — but the composition happens in the SAME F16
        // [img_seq, PATCH_OUT] output space (not the hidden residual).
        // Strategy: after the uncond forward+proj_out, materialise its
        // output in-place over proj_out_out_f16 with a temporary scratch.
        void *eps_out_f16 = proj_out_out_f16;  // default for non-CFG
        void *cond_proj_out_f16_saved = nullptr;

        if (run_uncond) {
            // Save cond proj_out before running uncond (uncond reuses the
            // same proj_out_in/out buffers).
            cond_proj_out_f16_saved =
                alloc_bytes((size_t)img_seq * PATCH_OUT * F16);
            if (!cond_proj_out_f16_saved) { free_owned(); return false; }
            aclError m2 = g_cann.aclrtMemcpy(
                cond_proj_out_f16_saved,
                (size_t)img_seq * PATCH_OUT * F16,
                proj_out_out_f16,
                (size_t)img_seq * PATCH_OUT * F16,
                ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (m2 != 0) {
                QIE_LOG("denoise_full: D2D cond_proj_out save err=%d step=%d",
                        (int)m2, step);
                free_owned(); return false;
            }

            // uncond 60-block forward on img_res_u + txt_res_u.
            me = g_cann.aclrtMemcpy(txt_work_u_f32,
                                     (size_t)txt_seq * H * F32,
                                     txt_res_u_f32,
                                     (size_t)txt_seq * H * F32,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE);
            if (me != 0) { QIE_LOG("denoise_full: D2D txt_res_u→work err=%d",
                                     (int)me); free_owned(); return false; }
            if (!forward_all_blocks_test(img_res_u_f32, img_seq,
                                           txt_work_u_f32, txt_seq,
                                           t_emb_out_f16, pe_dev,
                                           nullptr, 0)) {
                QIE_LOG("denoise_full: uncond forward failed step=%d", step);
                free_owned(); return false;
            }
            // uncond norm_out + proj_out (re-uses same AdaLN scale/shift).
            if (!layer_norm_f32_to_f16_(img_res_u_f32, proj_out_in_f16,
                                          /*B*/1, img_seq, H)) {
                QIE_LOG("denoise_full: norm_out LN(uncond) failed step=%d",
                        step);
                free_owned(); return false;
            }
            if (!modulate_(proj_out_in_f16, adaln_scale, adaln_shift,
                            /*B*/1, img_seq, H)) {
                QIE_LOG("denoise_full: norm_out modulate(uncond) failed "
                        "step=%d", step);
                free_owned(); return false;
            }
            if (!dispatch_matmul_(proj_out_in_f16,
                                   global_w_.proj_out_w_q4,
                                   global_w_.proj_out_scale,
                                   global_w_.proj_out_b,
                                   img_seq, H, PATCH_OUT,
                                   proj_out_out_f16)) {
                QIE_LOG("denoise_full: proj_out(uncond) failed step=%d", step);
                free_owned(); return false;
            }
            // At this point: proj_out_out_f16 = eps_uncond,
            //                cond_proj_out_f16_saved = eps_cond.
            // Compose eps = eps_uncond + cfg*(eps_cond - eps_uncond), all F16.
            int64_t nE = img_seq * PATCH_OUT;
            int64_t shape1[1]   = { nE };
            int64_t strides1[1] = { 1 };
            int64_t storage1    = nE;
            aclTensor *t_u = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT16, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, proj_out_out_f16);
            aclTensor *t_c = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT16, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, cond_proj_out_f16_saved);

            // cond -= uncond (alpha=-1 F16 scalar).
            uint16_t neg_one_f16 = fp32_to_fp16(-1.0f);
            aclScalar *alpha_neg1 = g_cann.aclCreateScalar(&neg_one_f16,
                                                             ACL_FLOAT16);
            uint64_t ws1 = 0; aclOpExecutor *ex1 = nullptr;
            aclnnStatus st = g_cann.aclnnInplaceAddGetWorkspaceSize(
                t_c, t_u, alpha_neg1, &ws1, &ex1);
            if (st == 0) {
                ensure_workspace_(ws1);
                st = g_cann.aclnnInplaceAdd(ws1 > 0 ? workspace_dev_ : nullptr,
                                               ws1, ex1, compute_stream_);
            }
            g_cann.aclDestroyScalar(alpha_neg1);
            if (st != 0) { QIE_LOG("denoise_full: CFG sub err=%d step=%d",
                                     (int)st, step);
                             g_cann.aclDestroyTensor(t_u);
                             g_cann.aclDestroyTensor(t_c);
                             free_owned(); return false; }

            // uncond += cfg * (cond - uncond)
            uint16_t cfg_f16 = fp32_to_fp16(cfg_scale);
            aclScalar *sc_cfg = g_cann.aclCreateScalar(&cfg_f16, ACL_FLOAT16);
            uint64_t ws2 = 0; aclOpExecutor *ex2 = nullptr;
            st = g_cann.aclnnInplaceAddGetWorkspaceSize(t_u, t_c, sc_cfg,
                                                          &ws2, &ex2);
            if (st == 0) {
                ensure_workspace_(ws2);
                st = g_cann.aclnnInplaceAdd(ws2 > 0 ? workspace_dev_ : nullptr,
                                               ws2, ex2, compute_stream_);
            }
            g_cann.aclDestroyScalar(sc_cfg);
            g_cann.aclDestroyTensor(t_u);
            g_cann.aclDestroyTensor(t_c);
            if (st != 0) { QIE_LOG("denoise_full: CFG add err=%d step=%d",
                                     (int)st, step);
                             free_owned(); return false; }

            // Free the per-step saved cond proj_out buffer.
            g_cann.aclrtFree(cond_proj_out_f16_saved);
            // Remove from owned list so free_owned doesn't double-free.
            for (auto it = owned.begin(); it != owned.end(); ++it) {
                if (it->p == cond_proj_out_f16_saved) {
                    owned.erase(it); break;
                }
            }
            cond_proj_out_f16_saved = nullptr;

            eps_out_f16 = proj_out_out_f16;  // holds composed CFG eps
        }

        if (!sync_ok("step compute")) { free_owned(); return false; }

        // --- (7) D2H the FIRST init_img_tokens * PATCH_OUT elements. ---
        aclError m3 = g_cann.aclrtMemcpy(out_tokens_f16.data(),
                                           out_tokens_f16.size() * F16,
                                           eps_out_f16,
                                           out_tokens_f16.size() * F16,
                                           ACL_MEMCPY_DEVICE_TO_HOST);
        if (m3 != 0) {
            QIE_LOG("denoise_full: D2H proj_out step=%d err=%d",
                    step, (int)m3);
            free_owned(); return false;
        }
        for (size_t i = 0; i < out_tokens_f16.size(); ++i) {
            __fp16 hh;
            std::memcpy(&hh, &out_tokens_f16[i], sizeof(hh));
            out_tokens_f32[i] = (float)hh;
        }
        // Unpatchify to `denoised_host` [W_lat, H_lat, C_out=16, B].
        host_unpatchify_latent(out_tokens_f32.data(),
                                W_lat, H_lat, cfg_.out_channels, B,
                                PATCH, denoised_host.data());

        // --- (8) Euler step on host. ---
        // Qwen-Image (flow-matching) Euler reference:
        //     d[j] = (x[j] - denoised[j]) / sigma
        //     x[j] += d[j] * (sigmas[s+1] - sigmas[s])
        // Handle sigma==0 defensively (shouldn't happen during loop since
        // sigmas[0..n_steps-1] are all > 0; sigmas[n_steps] is the terminal
        // zero boundary that's only read as `dt`).
        float dt = sigmas[step + 1] - sigma;
        if (sigma > 0.0f) {
            const size_t nelt = x_host.size();
            for (size_t j = 0; j < nelt; ++j) {
                float d = (x_host[j] - denoised_host[j]) / sigma;
                x_host[j] += d * dt;
            }
        } else {
            // Degenerate: sigma=0, treat as no-op.
        }

        if (per_step_ms) {
            auto t_step_1 = std::chrono::steady_clock::now();
            per_step_ms[step] =
                std::chrono::duration<double, std::milli>(t_step_1 - t_step_0)
                    .count();
        }
    }

    // Final latent = x_host. Copy into caller's out buffer.
    std::memcpy(out_latent, x_host.data(),
                 x_host.size() * sizeof(float));

    free_owned();
    return true;
}

}  // namespace ominix_qie
