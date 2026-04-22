// ============================================================================
// ImageDiffusionEngine — Phase 2 body.
//
// Phase 1 (a50d0174) shipped the scaffold (constructor / dtor / device open).
// Phase 2 fills `init_from_gguf` with real weight traversal + preload-dequant
// to F16 + H2D upload for every DiT tensor the reference path in
// tools/ominix_diffusion/src/qwen_image.hpp touches, plus the 3D-axial RoPE
// pre-compute (Q0.5.3 MUST-HAVE). `forward` / `denoise` still stub out —
// Phase 3 / 4 fill those.
//
// Preload-dequant rationale (primary NaN fix hypothesis per Q1 baseline):
//   Q4_1 / Q5_K on-disk weights are dequantised ONCE at session start via
//   ggml's type-traits `to_float`, then re-packed as F16 and uploaded to
//   resident NPU buffers. Per-step forward then reads from resident F16 —
//   no per-step D2H/H2D round-trip through ggml-cann. HBM budget per
//   contract §Q2:  ~13 GB F16 DiT weights (up from 5 GB Q4 on disk) + 7 GB
//   activations/workspace = 20 GB total → fits 32 GB with margin.
//
// Tensor name convention: diffusers-style
//   "transformer_blocks.<i>.attn.to_q.weight"
//   "transformer_blocks.<i>.img_mod.1.weight"
//   "transformer_blocks.<i>.img_mlp.net.0.proj.weight"
//   ...
// Same names the ominix_diffusion CPU path uses — see qwen_image.hpp and
// name_conversion.cpp. We do NOT use the llama-style "blk.<i>.*" scheme.
//
// LayerNorm affine=false on img_norm1/2 and txt_norm1/2 per qwen_image.hpp:
// we try-upload gamma/beta but tolerate their absence (nullptr kept).
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
void compute_qwen_rope_pe_host(const ImageDiffusionConfig &cfg,
                               std::vector<uint16_t> &pe_f16_out,
                               int64_t &total_pos_out) {
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

    auto pe_set = [&](int64_t pos, int64_t dpair,
                      float cos_v, float sin_v) {
        const size_t base =
            ((size_t)pos * head_dim / 2 + (size_t)dpair) * 4;
        pe_f16_out[base + 0] = fp32_to_fp16(cos_v);
        pe_f16_out[base + 1] = fp32_to_fp16(-sin_v);
        pe_f16_out[base + 2] = fp32_to_fp16(sin_v);
        pe_f16_out[base + 3] = fp32_to_fp16(cos_v);
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
        free_dev(lw.to_q_w);         free_dev(lw.to_q_b);
        free_dev(lw.to_k_w);         free_dev(lw.to_k_b);
        free_dev(lw.to_v_w);         free_dev(lw.to_v_b);
        free_dev(lw.to_out_0_w);     free_dev(lw.to_out_0_b);
        free_dev(lw.add_q_w);        free_dev(lw.add_q_b);
        free_dev(lw.add_k_w);        free_dev(lw.add_k_b);
        free_dev(lw.add_v_w);        free_dev(lw.add_v_b);
        free_dev(lw.to_add_out_w);   free_dev(lw.to_add_out_b);
        free_dev(lw.norm_q_w);       free_dev(lw.norm_k_w);
        free_dev(lw.norm_added_q_w); free_dev(lw.norm_added_k_w);
        free_dev(lw.img_norm1_w);    free_dev(lw.img_norm1_b);
        free_dev(lw.img_norm2_w);    free_dev(lw.img_norm2_b);
        free_dev(lw.txt_norm1_w);    free_dev(lw.txt_norm1_b);
        free_dev(lw.txt_norm2_w);    free_dev(lw.txt_norm2_b);
        free_dev(lw.img_mod_w);      free_dev(lw.img_mod_b);
        free_dev(lw.txt_mod_w);      free_dev(lw.txt_mod_b);
        free_dev(lw.img_ff_up_w);    free_dev(lw.img_ff_up_b);
        free_dev(lw.img_ff_down_w);  free_dev(lw.img_ff_down_b);
        free_dev(lw.txt_ff_up_w);    free_dev(lw.txt_ff_up_b);
        free_dev(lw.txt_ff_down_w);  free_dev(lw.txt_ff_down_b);
        free_dev(lw.to_q_w_i4);      free_dev(lw.to_q_scale_f16);
    }
    layer_w_.clear();

    free_dev(global_w_.time_linear1_w);    free_dev(global_w_.time_linear1_b);
    free_dev(global_w_.time_linear2_w);    free_dev(global_w_.time_linear2_b);
    free_dev(global_w_.img_in_w);          free_dev(global_w_.img_in_b);
    free_dev(global_w_.txt_in_w);          free_dev(global_w_.txt_in_b);
    free_dev(global_w_.txt_norm_w);
    free_dev(global_w_.norm_out_linear_w); free_dev(global_w_.norm_out_linear_b);
    free_dev(global_w_.proj_out_w);        free_dev(global_w_.proj_out_b);
    free_dev(global_w_.rope_pe_dev);

    free_dev(scratch_q_dev_);    free_dev(scratch_k_dev_);
    free_dev(scratch_v_dev_);    free_dev(scratch_attn_dev_);
    free_dev(scratch_mlp_dev_);  free_dev(scratch_mod_dev_);
    free_dev(rstd_dev_);
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
    // NOTE on `to_out.0` vs `to_out_0`: qwen_image.hpp:103 registers it as
    // "to_out.0" (with a literal dot), matching HF diffusers. GGUF exports
    // reproduce the dot. `img_mod.1` and `txt_mod.1` are similar — the
    // leading `img_mod.0` / `txt_mod.0` slots are `nn.SiLU` activations
    // (no params).
    // ------------------------------------------------------------------
    char name[128];
#define TNAME(fmt, ...) (snprintf(name, sizeof(name), fmt, __VA_ARGS__), name)

    for (int il = 0; il < cfg_.num_layers; ++il) {
        auto &lw = layer_w_[il];

        // --- attn.to_{q,k,v} projections + biases (img side) ---
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_q.weight", il),
                (size_t)H * H, lw.to_q_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_q.bias", il),
                (size_t)H, lw.to_q_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_k.weight", il),
                (size_t)H * H, lw.to_k_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_k.bias", il),
                (size_t)H, lw.to_k_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_v.weight", il),
                (size_t)H * H, lw.to_v_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_v.bias", il),
                (size_t)H, lw.to_v_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_out.0.weight", il),
                (size_t)H * H, lw.to_out_0_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_out.0.bias", il),
                (size_t)H, lw.to_out_0_b, f16b, tc, dequant_ms)) goto fail;

        // --- attn.add_{q,k,v}_proj + to_add_out (txt side) ---
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_q_proj.weight", il),
                (size_t)H * H, lw.add_q_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_q_proj.bias", il),
                (size_t)H, lw.add_q_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_k_proj.weight", il),
                (size_t)H * H, lw.add_k_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_k_proj.bias", il),
                (size_t)H, lw.add_k_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_v_proj.weight", il),
                (size_t)H * H, lw.add_v_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.add_v_proj.bias", il),
                (size_t)H, lw.add_v_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.attn.to_add_out.weight", il),
                (size_t)H * H, lw.to_add_out_w, f16b, tc, dequant_ms)) goto fail;
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
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mod.1.weight", il),
                (size_t)(6 * H) * H, lw.img_mod_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mod.1.bias", il),
                (size_t)(6 * H), lw.img_mod_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mod.1.weight", il),
                (size_t)(6 * H) * H, lw.txt_mod_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mod.1.bias", il),
                (size_t)(6 * H), lw.txt_mod_b, f16b, tc, dequant_ms)) goto fail;

        // --- FFN (img_mlp + txt_mlp). FeedForward.net = [Linear, GELU,
        // Linear] so .net.0.proj.* (up) and .net.2.* (down). NOT SwiGLU.
        // GELU exact (not GELU-tanh) per contract §Q2 / Part-2 §3.8.
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.0.proj.weight", il),
                (size_t)FF_DIM * H, lw.img_ff_up_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.0.proj.bias", il),
                (size_t)FF_DIM, lw.img_ff_up_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.2.weight", il),
                (size_t)H * FF_DIM, lw.img_ff_down_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.img_mlp.net.2.bias", il),
                (size_t)H, lw.img_ff_down_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.0.proj.weight", il),
                (size_t)FF_DIM * H, lw.txt_ff_up_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.0.proj.bias", il),
                (size_t)FF_DIM, lw.txt_ff_up_b, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.2.weight", il),
                (size_t)H * FF_DIM, lw.txt_ff_down_w, f16b, tc, dequant_ms)) goto fail;
        if (!dequant_upload_f16(ggml_ctx,
                TNAME("transformer_blocks.%d.txt_mlp.net.2.bias", il),
                (size_t)H, lw.txt_ff_down_b, f16b, tc, dequant_ms)) goto fail;
    }

    // ------------------------------------------------------------------
    // Global (non-per-layer) weights.
    // ------------------------------------------------------------------
    // time_text_embed.timestep_embedder.linear_{1,2}
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_1.weight",
            (size_t)H * 256, global_w_.time_linear1_w, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_1.bias",
            (size_t)H, global_w_.time_linear1_b, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_2.weight",
            (size_t)H * H, global_w_.time_linear2_w, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx,
            "time_text_embed.timestep_embedder.linear_2.bias",
            (size_t)H, global_w_.time_linear2_b, f16b, tc, dequant_ms)) goto fail;

    // img_in / txt_in projections.
    if (!dequant_upload_f16(ggml_ctx, "img_in.weight",
            (size_t)H * PATCH_IN, global_w_.img_in_w, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "img_in.bias",
            (size_t)H, global_w_.img_in_b, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "txt_in.weight",
            (size_t)H * JD, global_w_.txt_in_w, f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "txt_in.bias",
            (size_t)H, global_w_.txt_in_b, f16b, tc, dequant_ms)) goto fail;

    // txt_norm RMSNorm over joint_attention_dim.
    if (!dequant_upload_f32(ggml_ctx, "txt_norm.weight",
            (size_t)JD, global_w_.txt_norm_w, f32b, tc, dequant_ms)) goto fail;

    // norm_out.linear (AdaLayerNormContinuous): Linear(hidden, 2·hidden).
    if (!dequant_upload_f16(ggml_ctx, "norm_out.linear.weight",
            (size_t)(2 * H) * H, global_w_.norm_out_linear_w,
            f16b, tc, dequant_ms)) goto fail;
    if (!dequant_upload_f16(ggml_ctx, "norm_out.linear.bias",
            (size_t)(2 * H), global_w_.norm_out_linear_b,
            f16b, tc, dequant_ms)) goto fail;

    // proj_out: Linear(hidden, patch_size² · out_channels).
    if (!dequant_upload_f16(ggml_ctx, "proj_out.weight",
            (size_t)PATCH_OUT * H, global_w_.proj_out_w,
            f16b, tc, dequant_ms)) goto fail;
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
        std::vector<uint16_t> pe_host;
        int64_t total_pos = 0;
        auto t_r0 = std::chrono::steady_clock::now();
        compute_qwen_rope_pe_host(cfg_, pe_host, total_pos);
        auto t_r1 = std::chrono::steady_clock::now();
        stats_.dequant_wall_ms +=
            std::chrono::duration<double, std::milli>(t_r1 - t_r0).count();

        const size_t pe_bytes = pe_host.size() * sizeof(uint16_t);
        QIE_ACL_CHECK(g_cann.aclrtMalloc(&global_w_.rope_pe_dev, pe_bytes,
                                          ACL_MEM_MALLOC_HUGE_FIRST));
        QIE_ACL_CHECK(g_cann.aclrtMemcpy(global_w_.rope_pe_dev, pe_bytes,
                                          pe_host.data(), pe_bytes,
                                          ACL_MEMCPY_HOST_TO_DEVICE));
        stats_.rope_bytes = pe_bytes;
        QIE_LOG("rope pe: pos=%lld head_dim/2=%lld bytes=%zu "
                "(layout=[seq, hd/2, 2, 2] F16)",
                (long long)total_pos, (long long)(HEAD_D / 2), pe_bytes);
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

        // Small workspace seed — ops grow it via ensure_workspace_.
        if (!try_alloc(&workspace_dev_, 4 * 1024 * 1024)) goto fail;
        workspace_size_ = 4 * 1024 * 1024;
    }

    {
        auto t_init1 = std::chrono::steady_clock::now();
        stats_.load_wall_ms =
            std::chrono::duration<double, std::milli>(t_init1 - t_init0).count();

        const size_t total_bytes =
            stats_.f16_weight_bytes + stats_.f32_weight_bytes +
            stats_.rope_bytes + stats_.scratch_bytes;

        QIE_LOG("Phase 2 init OK: device=%d gguf=%s",
                device_, gguf_path.c_str());
        QIE_LOG("  tensors uploaded: %lld", (long long)stats_.tensors_uploaded);
        QIE_LOG("  F16 weight bytes: %zu (%.2f GiB)", stats_.f16_weight_bytes,
                stats_.f16_weight_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  F32 weight bytes: %zu (%.2f MiB)", stats_.f32_weight_bytes,
                stats_.f32_weight_bytes / (1024.0 * 1024.0));
        QIE_LOG("  RoPE pe bytes:    %zu (%.2f MiB)", stats_.rope_bytes,
                stats_.rope_bytes / (1024.0 * 1024.0));
        QIE_LOG("  Scratch bytes:    %zu (%.2f GiB)", stats_.scratch_bytes,
                stats_.scratch_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  Peak init HBM:    %zu (%.2f GiB)", total_bytes,
                total_bytes / (1024.0 * 1024.0 * 1024.0));
        QIE_LOG("  Dequant wall:     %.1f ms", stats_.dequant_wall_ms);
        QIE_LOG("  Total init wall:  %.1f ms", stats_.load_wall_ms);
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
// forward — single DiT step (Phase 3 will fill this body).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::forward(void * /*img_hidden_dev*/, int64_t img_seq,
                                     void * /*txt_hidden_dev*/, int64_t txt_seq,
                                     void * /*t_emb_dev*/,
                                     void * /*pe_dev*/) {
    if (!ready_) {
        QIE_LOG("forward: engine not ready");
        return false;
    }
    QIE_LOG("forward: scaffold Phase 2 — body not wired yet "
            "(Phase 3 adds 60-block dispatch). img_seq=%lld txt_seq=%lld",
            (long long)img_seq, (long long)txt_seq);
    return false;
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
    (void)timestep; (void)out_dev;
    // Phase 4 body: sinusoidal timestep encoding into 256 dims, uploaded
    // as F16 to `out_dev`. Caller then runs time_linear{1,2} to project
    // to `hidden`.
}

void ImageDiffusionEngine::forward_block_(const DiTLayerWeights & /*lw*/,
                                            void * /*img_hidden*/,
                                            int64_t img_seq,
                                            void * /*txt_hidden*/,
                                            int64_t txt_seq,
                                            void * /*t_emb*/,
                                            void * /*pe*/) {
    (void)img_seq; (void)txt_seq;
    // Phase 3 body. Sequence documented inline in the header —
    // image_diffusion_engine.h lines ~260-310.
}

void ImageDiffusionEngine::scheduler_step_(void * /*latent_dev*/,
                                             const void * /*model_out_dev*/,
                                             int step_idx) {
    (void)step_idx;
    // Phase 4 body: Euler-flow step per
    // tools/ominix_diffusion/src/denoiser.hpp:831-865.
}

}  // namespace ominix_qie
