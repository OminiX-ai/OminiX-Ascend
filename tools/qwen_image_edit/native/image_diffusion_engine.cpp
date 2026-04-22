// ============================================================================
// ImageDiffusionEngine — Phase 1 skeleton.
//
// Goal of this file is to compile clean on ac03 (and on the Mac cross-check,
// gated on ASCEND_TOOLKIT_HOME) so Phase 2/3/4 can land as pure fills with
// zero churn to the public interface. Every method that would touch GGUF
// or issue aclnn ops simply logs a one-liner and returns a known-safe value.
//
// Once Q1 (ggml-cann unblock) lands, Phase 2 fills init_from_gguf, Phase 3
// fills forward_block_, and Phase 4 wires denoise + scheduler_step_.
// ============================================================================

#include "image_diffusion_engine.h"

#include <cstdio>
#include <cstring>

namespace ominix_qie {

// ---------------------------------------------------------------------------
// Minimal logging wrapper — prefixed so ac03 smoke logs are grep-friendly.
// ---------------------------------------------------------------------------
#define QIE_LOG(fmt, ...) \
    fprintf(stderr, "[qie_native] " fmt "\n", ##__VA_ARGS__)

// ---------------------------------------------------------------------------
// ACL error-check macro — mirrors TalkerCannEngine's ACL_CHECK_RET. Phase
// 1 only invokes this on the device-open + stream-create paths.
// ---------------------------------------------------------------------------
#define QIE_ACL_CHECK(expr)                                                 \
    do {                                                                     \
        aclError _err = (expr);                                              \
        if (_err != 0) {                                                     \
            QIE_LOG("ACL call failed at %s:%d err=%d",                       \
                    __FILE__, __LINE__, (int)_err);                          \
            return false;                                                    \
        }                                                                    \
    } while (0)

// ---------------------------------------------------------------------------
// dtor — releases every device buffer this engine owns.
// ---------------------------------------------------------------------------
ImageDiffusionEngine::~ImageDiffusionEngine() {
    if (!cp_cann_load_symbols()) {
        // Symbols never loaded (Phase 1 scaffold may be compiled-in but
        // never initialised on non-Ascend CI). Nothing to free.
        return;
    }

    auto free_dev = [](void *&p) {
        if (p) { g_cann.aclrtFree(p); p = nullptr; }
    };

    // Per-layer weight buffers.
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

    // Global weights.
    free_dev(global_w_.time_linear1_w);    free_dev(global_w_.time_linear1_b);
    free_dev(global_w_.time_linear2_w);    free_dev(global_w_.time_linear2_b);
    free_dev(global_w_.img_in_w);          free_dev(global_w_.img_in_b);
    free_dev(global_w_.txt_in_w);          free_dev(global_w_.txt_in_b);
    free_dev(global_w_.txt_norm_w);
    free_dev(global_w_.norm_out_linear_w); free_dev(global_w_.norm_out_linear_b);
    free_dev(global_w_.proj_out_w);        free_dev(global_w_.proj_out_b);
    free_dev(global_w_.rope_pe_dev);

    // Scratch.
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
// Phase 1 init: resolve CANN symbols, open the device, create one stream.
// No GGUF parse, no weight upload — that's Phase 2. Returns true so downstream
// integration tests can exercise the scaffold shape.
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::init_from_gguf(const std::string &gguf_path,
                                           const ImageDiffusionConfig &cfg,
                                           int device) {
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

    QIE_LOG("Phase 1 scaffold: device=%d gguf=%s layers=%d hidden=%d "
            "heads=%d head_dim=%d ff_mult=%d seq=%d+%d rope_axes=%d,%d,%d "
            "precompute_rope=%s use_q4=%s",
            device_, gguf_path.c_str(),
            cfg_.num_layers, cfg_.hidden_size, cfg_.num_heads,
            cfg_.head_dim, cfg_.ff_mult,
            cfg_.max_img_seq, cfg_.max_txt_seq,
            cfg_.rope_axes_temporal, cfg_.rope_axes_h, cfg_.rope_axes_w,
            cfg_.precompute_rope ? "on" : "off",
            cfg_.use_q4_weights ? "on" : "off");
    QIE_LOG("Phase 1 scaffold: GGUF parse + weight upload are Phase 2 work "
            "(gated on Q1.1 ggml-cann unblock for weight-name audit). "
            "init returning ready=false");

    // Intentionally leave ready_=false so any caller that tries to run
    // `forward` or `denoise` before Phase 2/3/4 land gets a clear error
    // rather than a silently-incorrect result.
    return true;
}

// ---------------------------------------------------------------------------
// forward — single DiT step (scaffold returns false).
// ---------------------------------------------------------------------------
bool ImageDiffusionEngine::forward(void * /*img_hidden_dev*/, int64_t img_seq,
                                     void * /*txt_hidden_dev*/, int64_t txt_seq,
                                     void * /*t_emb_dev*/,
                                     void * /*pe_dev*/) {
    if (!ready_) {
        QIE_LOG("forward: engine not ready (scaffold Phase 1 only); "
                "img_seq=%lld txt_seq=%lld",
                (long long)img_seq, (long long)txt_seq);
        return false;
    }
    // Phase 3 body: iterate 60 transformer blocks, dispatching
    //   forward_block_(layer_w_[i], img, img_seq, txt, txt_seq, t_emb, pe)
    // on each.
    return false;
}

// ---------------------------------------------------------------------------
// denoise — full 20-step Euler-flow loop (scaffold returns false).
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
        QIE_LOG("denoise: engine not ready (scaffold Phase 1 only); "
                "latent=[%lld,%lld,%lld,%lld] cond=[%lld,%lld] "
                "ref=[%lld,%lld,%lld,%lld]",
                (long long)N, (long long)C, (long long)H, (long long)W,
                (long long)cond_seq, (long long)cond_dim,
                (long long)ref_N, (long long)ref_C,
                (long long)ref_H, (long long)ref_W);
        return false;
    }
    // Phase 4 body: for each of cfg_.num_inference_steps steps:
    //   (a) compute t_emb via build_time_emb_ + time_linear{1,2}.
    //   (b) run `forward()` on the cond pass.
    //   (c) if cfg_scale != 1.0, run `forward()` on the uncond pass.
    //   (d) combine via CFG: v = uncond + cfg_scale * (cond - uncond).
    //   (e) scheduler_step_(latent, v, step_idx).
    // Then apply norm_out + proj_out + unpatchify to produce `out_latents`.
    return false;
}

// ---------------------------------------------------------------------------
// Internal helpers — Phase 1 stubs.
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
    // Phase 2/3 — 3D axial RoPE pre-compute.
    //
    // Pseudo-code for Phase 3:
    //   - For each position p in [0, max_img_seq + max_txt_seq):
    //       decompose p into (t, h, w) indices
    //       for dim_pair d in [0, head_dim/2):
    //           pick axis from cfg_.rope_axes_{temporal,h,w} based on d
    //           freq = 1 / rope_theta^(2d/head_dim)
    //           angle = pos_on_axis * freq
    //           pe[p, d, :, :] = [[cos, -sin], [sin, cos]]
    //   - Txt positions (p >= max_img_seq): zero rotation (cos=1, sin=0).
    //   - Upload as F16 [seq · head_dim/2 · 2 · 2] contiguous to
    //     global_w_.rope_pe_dev.
    //
    // Retires RoPE-V2 packed layout per Q0.5.3 verdict: pre-computed tables
    // win by +10-25% per step per MLX measurement.
    QIE_LOG("build_rope_tables_: stub — Phase 2/3 fills this");
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
    // Phase 3 body sequence (matches qwen_image.hpp:251-314):
    //
    //   (1) img_mod = SiLU(t_emb); img_mod = img_mod @ img_mod_w + img_mod_b
    //       Chunk img_mod into 6 pieces: (scale1, shift1, gate1, scale2,
    //                                      shift2, gate2).
    //       Same for txt_mod.
    //
    //   (2) img_normed  = LayerNorm(img_hidden, affine=false)
    //       img_modulated = img_normed * (1 + scale1) + shift1
    //       Same for txt.
    //
    //   (3) QKV projections (img): Q = img_modulated @ to_q_w + to_q_b
    //                              K = img_modulated @ to_k_w + to_k_b
    //                              V = img_modulated @ to_v_w + to_v_b
    //       reshape Q/K/V to [N, img_seq, num_heads, head_dim]
    //       RMSNorm Q via norm_q_w gammas; RMSNorm K via norm_k_w
    //       Repeat for txt (add_{q,k,v}_proj + norm_added_{q,k}).
    //
    //   (4) Concat txt-then-img along seq dim to get joint Q/K/V:
    //       Q_joint: [N, txt_seq + img_seq, num_heads, head_dim]
    //
    //   (5) RoPE apply on Q_joint, K_joint using `pe` tables.
    //
    //   (6) Attention:
    //        aclnnFusedInferAttentionScoreV2(Q_joint, K_joint, V_joint)
    //        -> attn: [N, txt_seq + img_seq, num_heads * head_dim]
    //       Split back into txt_attn / img_attn along seq dim.
    //
    //   (7) Output projections:
    //        img_attn = img_attn @ to_out_0_w + to_out_0_b
    //        txt_attn = txt_attn @ to_add_out_w + to_add_out_b
    //
    //   (8) Residual + gate1:
    //        img_hidden += img_attn * gate1
    //        txt_hidden += txt_attn * gate1
    //
    //   (9) LayerNorm2 + scale2/shift2:
    //        img_norm2 = LN(img_hidden, affine=false)
    //        img_modulated2 = img_norm2 * (1 + scale2) + shift2
    //        Same for txt.
    //
    //  (10) FFN: GELU-activated, NOT SwiGLU:
    //        img_ff = img_ff_up_w @ img_modulated2 + img_ff_up_b
    //        img_ff = GELU(img_ff)
    //        img_ff = img_ff_down_w @ img_ff + img_ff_down_b
    //        Same for txt.
    //
    //  (11) Residual + gate2:
    //        img_hidden += img_ff * gate2
    //        txt_hidden += txt_ff * gate2
    QIE_LOG("forward_block_: stub — Phase 3 fills this");
}

void ImageDiffusionEngine::scheduler_step_(void * /*latent_dev*/,
                                             const void * /*model_out_dev*/,
                                             int step_idx) {
    (void)step_idx;
    // Phase 4 body: Euler-flow scheduler step. Port of
    // denoiser.hpp (stable-diffusion.cpp Euler-flow). Given sigma(step_idx),
    // sigma(step_idx + 1), and the model's predicted velocity in
    // model_out_dev, update latent_dev in-place:
    //   latent = latent + (sigma_next - sigma_cur) * velocity
}

}  // namespace ominix_qie
