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
    free_dev(img_hidden_cond_dev_);   free_dev(img_hidden_uncond_dev_);
    free_dev(txt_hidden_cond_dev_);   free_dev(txt_hidden_uncond_dev_);
    free_dev(workspace_dev_);

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
                                              void *y_f16_dev) {
    if (!x_f16_dev || !weight_dev || !y_f16_dev) {
        QIE_LOG("dispatch_matmul_: null buffer (x=%p w=%p y=%p)",
                x_f16_dev, weight_dev, y_f16_dev);
        return false;
    }

    // Build activation tensor view [M, K] contig.
    int64_t x_shape[2]   = {M, K};
    int64_t x_strides[2] = {K, 1};
    aclTensor *t_x = tensor_nd_f16(x_f16_dev, 2, x_shape, x_strides);

    // Build output tensor view [M, N] contig.
    int64_t y_shape[2]   = {M, N};
    int64_t y_strides[2] = {N, 1};
    aclTensor *t_y = tensor_nd_f16(y_f16_dev, 2, y_shape, y_strides);

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
        aclTensor *t_scale = tensor_nd_f16(weight_scale_dev, 2, s_shape, s_strides);

        uint64_t ws_needed = 0;
        aclOpExecutor *exec = nullptr;
        if (g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize &&
            g_cann.aclnnWeightQuantBatchMatmulV3) {
            s = g_cann.aclnnWeightQuantBatchMatmulV3GetWorkspaceSize(
                t_x, t_w, t_scale,
                /*antiquantOffsetOptional*/ nullptr,
                /*quantScaleOptional*/      nullptr,
                /*quantOffsetOptional*/     nullptr,
                /*biasOptional*/            nullptr,  // apply bias after
                /*antiquantGroupSize*/      32,
                /*innerPrecise*/            1,
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
        if (s != 0) {
            g_cann.aclDestroyTensor(t_x);
            g_cann.aclDestroyTensor(t_y);
            return false;
        }
    } else {
        // --- F16 fallback path: plain aclnnMm ------------------------------
        // Weight stored as GGUF [N, K] row-major (K contig). Transposed view
        // into [K, N] with strides (1, K).
        int64_t w_shape[2]   = {K, N};
        int64_t w_strides[2] = {1, K};
        int64_t w_storage    = K * N;
        aclTensor *t_w = g_cann.aclCreateTensor(
            w_shape, 2, ACL_FLOAT16, w_strides, 0, ACL_FORMAT_ND,
            &w_storage, 1, weight_dev);

        uint64_t ws_needed = 0;
        aclOpExecutor *exec = nullptr;
        s = g_cann.aclnnMmGetWorkspaceSize(t_x, t_w, t_y, (int8_t)0,
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
        aclTensor *t_b = tensor_nd_f16(bias_f16_dev, 2, b_shape, b_strides);

        aclScalar *alpha = nullptr;
        {
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

    // img_mod_params = img_mod.1 Linear → scratch_mod_dev_[0 .. 6H)
    if (!dispatch_matmul_(scratch_q_dev_, lw.img_mod_w_q4, lw.img_mod_scale,
                          lw.img_mod_b, B, H, 6 * H, scratch_mod_dev_))
        return false;
    // txt_mod_params = txt_mod.1 Linear → scratch_mod_dev_[6H .. 12H)
    if (!dispatch_matmul_(scratch_q_dev_, lw.txt_mod_w_q4, lw.txt_mod_scale,
                          lw.txt_mod_b, B, H, 6 * H,
                          (uint8_t *)scratch_mod_dev_ + (size_t)6 * H *
                                                          sizeof(uint16_t)))
        return false;

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
    //    Outputs written to scratch_img_norm_dev_ / scratch_txt_norm_dev_.
    // ------------------------------------------------------------------
    if (!layer_norm_(img_hidden, scratch_img_norm_dev_, B, img_seq, H))
        return false;
    if (!modulate_(scratch_img_norm_dev_, img_scale1, img_shift1,
                   B, img_seq, H)) return false;
    if (!layer_norm_(txt_hidden, scratch_txt_norm_dev_, B, txt_seq, H))
        return false;
    if (!modulate_(scratch_txt_norm_dev_, txt_scale1, txt_shift1,
                   B, txt_seq, H)) return false;

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

    // ------------------------------------------------------------------
    // 7. Output projections.
    //    txt attn-out view: scratch_attn_dev_[0 .. txt_seq*H] → to_add_out
    //    img attn-out view: scratch_attn_dev_[txt_seq*H .. end] → to_out.0
    //    The output projections write to scratch_img_out_dev_ /
    //    scratch_txt_out_dev_ so the subsequent gated residual is a pure
    //    add on top of the original img_hidden / txt_hidden buffers.
    // ------------------------------------------------------------------
    if (!dispatch_matmul_(scratch_attn_dev_, lw.to_add_out_w_q4,
                          lw.to_add_out_scale, lw.to_add_out_b,
                          txt_seq, H, H, scratch_txt_out_dev_)) return false;
    if (!dispatch_matmul_(offset_rows(scratch_attn_dev_, txt_seq),
                          lw.to_out_0_w_q4, lw.to_out_0_scale, lw.to_out_0_b,
                          img_seq, H, H, scratch_img_out_dev_)) return false;

    // ------------------------------------------------------------------
    // 8. Gated residual add: img += attn_out_img * gate1_img (and txt).
    // ------------------------------------------------------------------
    if (!gated_residual_add_(img_hidden, scratch_img_out_dev_,
                             img_gate1, B, img_seq, H)) return false;
    if (!gated_residual_add_(txt_hidden, scratch_txt_out_dev_,
                             txt_gate1, B, txt_seq, H)) return false;

    // ------------------------------------------------------------------
    // 9. LayerNorm2 + modulate(scale2, shift2) — output to scratch_*_norm.
    // ------------------------------------------------------------------
    if (!layer_norm_(img_hidden, scratch_img_norm_dev_, B, img_seq, H))
        return false;
    if (!modulate_(scratch_img_norm_dev_, img_scale2, img_shift2,
                   B, img_seq, H)) return false;
    if (!layer_norm_(txt_hidden, scratch_txt_norm_dev_, B, txt_seq, H))
        return false;
    if (!modulate_(scratch_txt_norm_dev_, txt_scale2, txt_shift2,
                   B, txt_seq, H)) return false;

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
    if (!gelu_activate(scratch_mlp_dev_, img_seq)) return false;
    if (!dispatch_matmul_(scratch_mlp_dev_, lw.img_ff_down_w_q4,
                          lw.img_ff_down_scale, lw.img_ff_down_b,
                          img_seq, FF, H, scratch_img_out_dev_)) return false;

    // txt FFN.
    if (!dispatch_matmul_(scratch_txt_norm_dev_, lw.txt_ff_up_w_q4,
                          lw.txt_ff_up_scale, lw.txt_ff_up_b,
                          txt_seq, H, FF, scratch_mlp_dev_)) return false;
    if (!gelu_activate(scratch_mlp_dev_, txt_seq)) return false;
    if (!dispatch_matmul_(scratch_mlp_dev_, lw.txt_ff_down_w_q4,
                          lw.txt_ff_down_scale, lw.txt_ff_down_b,
                          txt_seq, FF, H, scratch_txt_out_dev_)) return false;

    // ------------------------------------------------------------------
    // 11. Gated residual add #2.
    // ------------------------------------------------------------------
    if (!gated_residual_add_(img_hidden, scratch_img_out_dev_,
                             img_gate2, B, img_seq, H)) return false;
    if (!gated_residual_add_(txt_hidden, scratch_txt_out_dev_,
                             txt_gate2, B, txt_seq, H)) return false;

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
    aclTensor *t_x = g_cann.aclCreateTensor(shape, 1, ACL_FLOAT16, strides,
                                             0, ACL_FORMAT_ND, &storage, 1,
                                             x_f16_dev);
    aclTensor *t_e = g_cann.aclCreateTensor(shape, 1, ACL_FLOAT16, strides,
                                             0, ACL_FORMAT_ND, &storage, 1,
                                             (void *)eps_f16_dev);
    uint16_t dt_f16 = fp32_to_fp16(dt);
    aclScalar *alpha = g_cann.aclCreateScalar(&dt_f16, ACL_FLOAT16);

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
    const size_t  F16  = sizeof(uint16_t);
    const size_t  img_bytes = (size_t)img_seq * H * F16;
    const size_t  txt_bytes = (size_t)txt_seq * H * F16;

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

            aclTensor *t_u = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT16, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, img_work_u_dev);
            aclTensor *t_c = g_cann.aclCreateTensor(
                shape1, 1, ACL_FLOAT16, strides1, 0, ACL_FORMAT_ND,
                &storage1, 1, img_work_c_dev);

            // Step 1: img_work_c_dev += (-1) * img_work_u_dev
            uint16_t neg_one_f16 = fp32_to_fp16(-1.0f);
            aclScalar *alpha_neg1 = g_cann.aclCreateScalar(&neg_one_f16,
                                                              ACL_FLOAT16);
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

            // Step 2: img_work_u_dev += cfg * img_work_c_dev (which is Δ)
            uint16_t cfg16 = fp32_to_fp16(cfg_scale);
            aclScalar *sc_cfg = g_cann.aclCreateScalar(&cfg16, ACL_FLOAT16);
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
    for (int il = 0; il < L; ++il) {
        auto t0 = std::chrono::steady_clock::now();
        if (!forward_block_(layer_w_[il],
                            img_hidden, img_seq,
                            txt_hidden, txt_seq,
                            t_emb, pe)) {
            QIE_LOG("forward_all_blocks_test: block %d returned error", il);
            return false;
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

}  // namespace ominix_qie
