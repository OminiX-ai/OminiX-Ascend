#pragma once
// ============================================================================
// ImageDiffusionEngine — native Qwen-Image-Edit-2511 DiT on Ascend NPU.
//
// Scaffold for contract QIE §Q2 (tools/qwen_image_edit/native bring-up).
// This is the Phase-1 skeleton: class declaration + constructor / teardown +
// init-from-GGUF + single-block forward + full denoising loop entry points
// are declared; bodies in .cpp are stubbed (no compute yet) so the engine
// compiles cleanly on ac03 without waiting for Q1 (ggml-cann backend bug
// fixes) to land. Phase 2 fills `init_from_gguf` with real weight upload;
// Phase 3 fills `forward_block_`; Phase 4 wires `denoise`.
//
// Architecture reference (audited by Q0.5 + feasibility agents):
//   - 60 transformer blocks × 24 heads × head_dim=128, joint txt+img attn
//   - Hidden = 3072, joint_attention_dim (txt side) = 3584
//   - QwenImageAttention: RMSNorm on Q/K + added_Q/added_K; Linear projections
//     on both img and txt streams (to_q/k/v + add_{q,k,v}_proj + to_out.0 +
//     to_add_out). Bias on in-projections, bias on out-projections.
//   - QwenImageTransformerBlock: img_mod.1 + txt_mod.1 (Linear dim → 6·dim,
//     bias ON). img_norm1/2 and txt_norm1/2 are LayerNorm with affine=false
//     (no learnable gamma/beta — scale/shift comes from timestep modulation).
//     img_mlp / txt_mlp are FeedForward(dim → 4·dim, GELU, Linear) — NOT
//     SwiGLU, NOT MoE.
//   - Global: time_text_embed (sinusoidal 256 → Linear → SiLU → Linear →
//     hidden_size), txt_norm (RMSNorm on joint_attention_dim), img_in + txt_in
//     (Linear input projections), norm_out (AdaLayerNormContinuous) + proj_out.
//   - 3D axial RoPE: `axes_dim = {16, 56, 56}` (temporal/h/w), passed as
//     `pe: [seq_len, d_head/2, 2, 2]` — NOT standard 1D RoPE. See
//     tools/ominix_diffusion/src/rope.hpp for the reference layout. RoPE
//     tables are PRE-COMPUTED at session init (Q0.5.3 verdict: retire
//     RoPE-V2 packed layout; pre-compute wins on MLX parity numbers).
//   - Ref-image latents are patchified and concatenated onto the img token
//     stream at model entry (qwen_image.hpp:454-459).
//
// Precision scheme (matches TalkerCannEngine / Qwen3 conventions):
//   - F32: I/O staging at engine boundary; RmsNorm gammas (q/k/added_q/
//     added_k norms); LayerNorm is affine-off so no gamma/beta upload
//     required for img_norm{1,2}/txt_norm{1,2}; norm_out (AdaLN) does
//     compute its shift/scale from `t_emb` via a Linear head.
//   - F16: matmuls, residual adds, RoPE, attention, FFN activations.
//   - Attention: `aclnnFusedInferAttentionScoreV2` at seq ≈ 4096 img tokens
//     + 256 txt tokens ≈ 4352. Q0.5.2 verdict: LIKELY GREEN at this shape,
//     confirm at Q3 runtime probe. Scaffold falls back to plain aclnnMm(QK),
//     softmax, aclnnMm(V) triad if FIAv2 is unavailable or shape-rejected
//     (Q3's job to switch to FIAv2).
//
// Weight quantization:
//   - F16 (baseline) — always supported.
//   - Q4_K antiquant via `aclnnWeightQuantBatchMatmulV3` antiquantGroupSize=32
//     (K-quant block size). Contingent on Q1.1 landing Q4_K in ggml-cann;
//     until then the engine uploads F16 and quantization is a Phase 4/late-
//     Phase 2 task.
//
// Resources (concurrency note):
//   - ImageDiffusionEngine weights load in ~18-20 GiB HBM when Q4-quantized
//     (~60 GiB as F16). Ac03 910B4 has 32 GiB HBM total; A4b's TTS prefill
//     path co-tenant wants ~14 GiB. Smoke runs MUST take the shared cooperative
//     lock at `/tmp/ac03_hbm_lock` before `init_from_gguf` — see
//     `main_native.cpp` for the lock-wrap entry point.
//
// Persistent-engine pattern: `init_from_gguf()` runs once per process;
// `denoise(...)` runs per edit request. No weight reload between requests.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "../../qwen_tts/cp_cann_symbols.h"   // g_cann (CANN dlsym handle)

namespace ominix_qie {

// ---------------------------------------------------------------------------
// Config mirrors `Qwen::QwenImageParams` in
// tools/ominix_diffusion/src/qwen_image.hpp (the reference CPU implementation
// our ggml-cann path builds against). Defaults match Qwen-Image-Edit-2511.
// ---------------------------------------------------------------------------
struct ImageDiffusionConfig {
    // --- DiT architecture ---
    int   num_layers          = 60;     // transformer_blocks count
    int   num_heads           = 24;     // attention heads per block
    int   head_dim            = 128;    // per-head dim
    int   hidden_size         = 3072;   // num_heads * head_dim
    int   ff_mult             = 4;      // FFN intermediate = hidden * mult = 12288
    int   patch_size          = 2;      // 2x2 patch embed on the latent grid
    int   in_channels         = 64;     // patch channels (= 4 VAE * 2 * 2 * 2)
    int   out_channels        = 16;     // latent channels out
    int   joint_attention_dim = 3584;   // text stream's hidden BEFORE txt_in
    float rms_norm_eps        = 1e-6f;
    float layernorm_eps       = 1e-6f;
    bool  zero_cond_t         = false;  // Qwen-Image-Edit uses zero_cond_t=false

    // --- 3D axial RoPE (axes_dim = {temporal, h, w}) ---
    int   rope_axes_temporal  = 16;
    int   rope_axes_h         = 56;
    int   rope_axes_w         = 56;
    int   rope_theta          = 10000;

    // --- Scheduler ---
    int   num_inference_steps = 20;      // Euler-flow step count (default)
    float cfg_scale           = 4.0f;    // classifier-free guidance weight

    // --- Runtime budget ---
    // Max joint sequence length. Q0.5 probe fixes this at 4352 (4096 img
    // patches at 1024x1024 latent resolution + 256 txt tokens). If a future
    // resolution bumps img side, grow MAX_SEQ accordingly — attention
    // scratch is sized for this worst case up front.
    int   max_img_seq         = 4096;
    int   max_txt_seq         = 256;

    // --- Weight storage ---
    // F16 default. Q4 flag flips weight upload to int4 quantized + F16 scale
    // layout and routes matmul call sites through aclnnWeightQuantBatchMatmul.
    // Gated at init on `g_cann.has_w8_quant()` (same symbol covers V2/V3
    // dispatch for K-quants) AND Q1.1 landing the CANN backend fix.
    bool  use_q4_weights      = false;

    // --- Pre-computed RoPE (Q0.5.3 retire-V2 verdict) ---
    // When true, init precomputes the full cos/sin tables for every
    // {temporal, h, w} position in [0, max_img_seq + max_txt_seq) up front
    // and uploads them as F16 [seq, head_dim/2, 2, 2]. Every denoising
    // step then reads those tables directly — no per-step RoPE tensor
    // rebuild. MLX measured +10-25% per-step on this path; must-have per
    // contract §Q2.
    bool  precompute_rope     = true;
};

// ---------------------------------------------------------------------------
// Per-layer weight handles. All device pointers are null until
// `init_from_gguf` runs. F16 unless otherwise noted. Layout on device
// matches the [out, in] convention `aclnnMm` consumes — see
// TalkerCannEngine::LayerWeights for the same pattern.
// ---------------------------------------------------------------------------
struct DiTLayerWeights {
    // --- Attention projections (img side) --------------------------------
    void *to_q_w       = nullptr;  // F16 [hidden, hidden]        + bias
    void *to_q_b       = nullptr;  // F16 [hidden]
    void *to_k_w       = nullptr;  // F16 [hidden, hidden]        + bias
    void *to_k_b       = nullptr;  // F16 [hidden]
    void *to_v_w       = nullptr;  // F16 [hidden, hidden]        + bias
    void *to_v_b       = nullptr;  // F16 [hidden]
    void *to_out_0_w   = nullptr;  // F16 [hidden, hidden]        + bias
    void *to_out_0_b   = nullptr;  // F16 [hidden]

    // --- Attention projections (txt side, "add_*") -----------------------
    void *add_q_w      = nullptr;  // F16 [hidden, hidden]        + bias
    void *add_q_b      = nullptr;
    void *add_k_w      = nullptr;
    void *add_k_b      = nullptr;
    void *add_v_w      = nullptr;
    void *add_v_b      = nullptr;
    void *to_add_out_w = nullptr;  // F16 [hidden, hidden]        + bias
    void *to_add_out_b = nullptr;

    // --- RMSNorm gammas (Q/K-norm sites) --------------------------------
    // Per-head_dim gamma; input is post-projection [..., head_dim].
    void *norm_q_w       = nullptr;  // F32 [head_dim]
    void *norm_k_w       = nullptr;  // F32 [head_dim]
    void *norm_added_q_w = nullptr;  // F32 [head_dim]
    void *norm_added_k_w = nullptr;  // F32 [head_dim]

    // --- LayerNorm gammas/betas for block norm1/norm2 -------------------
    // Qwen-Image TransformerBlock uses `affine=false` on these — per
    // qwen_image.hpp:205-213. We leave these fields null; init_from_gguf
    // skips them if the GGUF has no entry, and the forward path runs
    // the "normalize only, no gamma/beta" LayerNorm variant. If a future
    // Qwen-Image revision flips affine on, populate these and have the
    // forward path multiply/add.
    void *img_norm1_w = nullptr;  // F32 [hidden] (may stay null)
    void *img_norm1_b = nullptr;  // F32 [hidden] (may stay null)
    void *img_norm2_w = nullptr;
    void *img_norm2_b = nullptr;
    void *txt_norm1_w = nullptr;
    void *txt_norm1_b = nullptr;
    void *txt_norm2_w = nullptr;
    void *txt_norm2_b = nullptr;

    // --- Timestep modulation heads (img_mod.1 / txt_mod.1) --------------
    // Both are Linear(dim → 6·dim, bias=true). We chunk the output into
    // 6 pieces per stream per block inside `forward_block_`.
    void *img_mod_w = nullptr;   // F16 [6·hidden, hidden] + bias
    void *img_mod_b = nullptr;   // F16 [6·hidden]
    void *txt_mod_w = nullptr;   // F16 [6·hidden, hidden] + bias
    void *txt_mod_b = nullptr;

    // --- FFN (GELU, NOT SwiGLU, NOT MoE) --------------------------------
    // FeedForward(dim → mult·dim, GELU, Linear(mult·dim → dim)). In
    // diffusers / stable-diffusion.cpp naming this is stored under
    // `ff.net.0.proj.{weight,bias}` + `ff.net.2.{weight,bias}`. `net.1`
    // is the GELU activation (no params), `net.3` is a Dropout (no
    // params).
    void *img_ff_up_w   = nullptr;  // F16 [ff_dim, hidden] + bias
    void *img_ff_up_b   = nullptr;  // F16 [ff_dim]
    void *img_ff_down_w = nullptr;  // F16 [hidden, ff_dim] + bias
    void *img_ff_down_b = nullptr;  // F16 [hidden]
    void *txt_ff_up_w   = nullptr;
    void *txt_ff_up_b   = nullptr;
    void *txt_ff_down_w = nullptr;
    void *txt_ff_down_b = nullptr;

    // --- Q4 quant slots --------------------------------------------------
    // Populated only when `cfg.use_q4_weights` AND CANN has the antiquant
    // path at init. Mirrors TalkerCannEngine's A16W8 pattern but with
    // group_size=32 instead of per-output-channel. Phase 2/late or Phase 4.
    // For the scaffold they stay null.
    void *to_q_w_i4        = nullptr;
    void *to_q_scale_f16   = nullptr;
    // ... (full set added in Phase 2 once Q1.1 confirms Q4_K path works.)
};

// ---------------------------------------------------------------------------
// Global (non-per-layer) weight handles.
// ---------------------------------------------------------------------------
struct DiTGlobalWeights {
    // time_text_embed.timestep_embedder.linear_{1,2}
    void *time_linear1_w = nullptr;  // F16 [hidden, 256] + bias
    void *time_linear1_b = nullptr;
    void *time_linear2_w = nullptr;  // F16 [hidden, hidden] + bias
    void *time_linear2_b = nullptr;

    // img_in, txt_in — input projections onto `hidden`.
    void *img_in_w = nullptr;  // F16 [hidden, in_channels · patch_size²]
    void *img_in_b = nullptr;
    void *txt_in_w = nullptr;  // F16 [hidden, joint_attention_dim]
    void *txt_in_b = nullptr;

    // txt_norm: RMSNorm over joint_attention_dim (F32 gamma).
    void *txt_norm_w = nullptr;   // F32 [joint_attention_dim]

    // norm_out = AdaLayerNormContinuous(hidden, hidden, affine=false, eps=1e-6).
    //   .norm   : LayerNorm(hidden, affine=false) — no params uploaded.
    //   .linear : Linear(hidden, 2·hidden, bias=true) — emits (scale, shift)
    //             from SiLU(t_emb).
    void *norm_out_linear_w = nullptr;  // F16 [2·hidden, hidden] + bias
    void *norm_out_linear_b = nullptr;

    // Final patch-level projection out to pixel-space channels.
    //   proj_out : Linear(hidden, patch_size² · out_channels, bias=true).
    void *proj_out_w = nullptr;  // F16 [ps²·out_ch, hidden] + bias
    void *proj_out_b = nullptr;

    // Pre-computed 3D axial RoPE tables (cos/sin). Populated by
    // `build_rope_tables_` during init when `cfg.precompute_rope`.
    // Layout matches `Qwen::Rope::apply_rope` / `Rope::attention` contract:
    //   pe: [seq, head_dim/2, 2, 2]  (F16)
    // where seq = max_img_seq + max_txt_seq. For joint attention the txt
    // block gets zero-rotation (cos=1, sin=0) in the upstream reference;
    // populate the tables accordingly so no runtime branch is needed.
    void *rope_pe_dev = nullptr;  // F16 [seq · head_dim/2 · 2 · 2]
};

// ---------------------------------------------------------------------------
// ImageDiffusionEngine — persistent handle. One instance = one DiT loaded
// on one NPU device.
//
// Thread safety: a single instance is NOT concurrently callable. Call
// `denoise` serially. An orchestrator that wants two concurrent sessions
// creates two engines on two devices (ac03 has CANN0/CANN1 presented).
// ---------------------------------------------------------------------------
class ImageDiffusionEngine {
public:
    ImageDiffusionEngine() = default;
    ~ImageDiffusionEngine();

    // One-time init: loads DiT weights from a GGUF produced by the
    // ominix_diffusion exporter (same format stable-diffusion.cpp reads).
    // `device` is the ACL device ID (0 or 1 on ac03).
    //
    // Phase 1: returns true on symbol-load + ACL-device-open; leaves every
    //          weight pointer null (no GGUF traversal).
    // Phase 2: parses the GGUF, uploads every tensor listed in DiTLayerWeights
    //          + DiTGlobalWeights, prints tensor count + peak HBM.
    bool init_from_gguf(const std::string &gguf_path,
                        const ImageDiffusionConfig &cfg,
                        int device = 0);

    // Full denoising: run `cfg_.num_inference_steps` Euler-flow steps of
    // joint-attention DiT forward on `initial_noise` conditioned on `cond_emb`
    // (text-encoder features) and `ref_latents` (VAE-encoded input image).
    // Output is written to `out_latents` in F32 layout
    //   [N, out_channels, H, W]  — same as `initial_noise`.
    //
    // CFG: this method runs cond and uncond passes sequentially per step
    // (scaffold contract; batched CFG is Q4 scope per Q0.5.1 symmetry
    // verdict). `cfg_scale=1.0` disables CFG and runs a single pass per step.
    //
    // Phase 4: implemented. Before then, returns false with a log line.
    bool denoise(const float *initial_noise,
                 int64_t N, int64_t C, int64_t H, int64_t W,
                 const float *cond_emb,
                 int64_t cond_seq, int64_t cond_dim,
                 const float *uncond_emb,
                 const float *ref_latents,
                 int64_t ref_N, int64_t ref_C, int64_t ref_H, int64_t ref_W,
                 float *out_latents);

    // One DiT step (exposed for unit probing & parity with ggml-cann).
    // All tensors are on NPU; caller manages device pointers.
    //   img_hidden        F16 [N, img_seq, hidden]   — in/out
    //   txt_hidden        F16 [N, txt_seq, hidden]   — in/out
    //   t_emb             F16 [N, hidden]            — timestep MLP output
    //   pe                F16 [img_seq+txt_seq, hd/2, 2, 2]  — rope tables
    //
    // Phase 3: implemented block-by-block. Before then this logs a
    // "scaffold: forward not wired" message and returns false.
    bool forward(void *img_hidden_dev, int64_t img_seq,
                 void *txt_hidden_dev, int64_t txt_seq,
                 void *t_emb_dev,
                 void *pe_dev);

    // Accessors.
    bool is_ready()   const { return ready_; }
    int  device_id()  const { return device_; }
    const ImageDiffusionConfig &config() const { return cfg_; }

private:
    // --- State ---
    bool  ready_                 = false;
    int   device_                = 0;
    ImageDiffusionConfig cfg_{};
    aclrtStream primary_stream_  = nullptr;   // owned
    aclrtStream compute_stream_  = nullptr;   // aliases primary_ by default

    // --- Weights ---
    std::vector<DiTLayerWeights> layer_w_;   // size = cfg_.num_layers
    DiTGlobalWeights             global_w_{};

    // --- Intermediate scratch (single-request; no request-level concurrency) ---
    // Sized at init for the worst case (max_img_seq + max_txt_seq at
    // cfg_.hidden_size). Reused every step.
    void *scratch_q_dev_    = nullptr;  // F16 [seq, hidden]
    void *scratch_k_dev_    = nullptr;  // F16 [seq, hidden]
    void *scratch_v_dev_    = nullptr;  // F16 [seq, hidden]
    void *scratch_attn_dev_ = nullptr;  // F16 [seq, hidden]
    void *scratch_mlp_dev_  = nullptr;  // F16 [seq, ff_dim]
    void *scratch_mod_dev_  = nullptr;  // F16 [6 · hidden] per stream, per block
    void *rstd_dev_         = nullptr;  // F32 [max_heads · seq] (RMSNorm rstd)

    // CFG duplicates — cond + uncond pass share weights, need separate
    // activation scratch. Phase 4 wires these.
    void *img_hidden_cond_dev_   = nullptr;
    void *img_hidden_uncond_dev_ = nullptr;
    void *txt_hidden_cond_dev_   = nullptr;
    void *txt_hidden_uncond_dev_ = nullptr;

    // aclnn workspace (grows on demand).
    void  *workspace_dev_ = nullptr;
    size_t workspace_size_ = 0;

    // --- Helpers (all stubbed in Phase 1) ----------------------------------
    void alloc_dev_(void **ptr, size_t bytes);
    void ensure_workspace_(size_t bytes);

    // Build the 3D axial RoPE tables for {temporal, h, w} into
    // `global_w_.rope_pe_dev` (layout: [seq, head_dim/2, 2, 2] F16). Uses
    // cfg_.rope_axes_{temporal,h,w} and rope_theta.
    void build_rope_tables_();

    // Populate the timestep sinusoidal 256-dim embedding for `timestep`
    // into `out_dev` (F16 [256]). Followed by `time_linear{1,2}` on device
    // to get the `hidden`-dim `t_emb` used by every block's modulation.
    void build_time_emb_(float timestep, void *out_dev);

    // Run one full transformer block on NPU.
    //   lw          — per-layer weights (already uploaded)
    //   img_hidden  — F16 [N, img_seq, hidden]   in-place update
    //   txt_hidden  — F16 [N, txt_seq, hidden]   in-place update
    //   t_emb       — F16 [N, hidden]
    //   pe          — RoPE cos/sin tables on device
    //   img_seq, txt_seq — actual current sequence lengths
    //
    // Phase 3 target: ~15 aclnn op dispatches per block (6 matmul + 4 norm +
    // RoPE + attention + 2 FFN matmuls + residual adds + gates).
    void forward_block_(const DiTLayerWeights &lw,
                        void *img_hidden, int64_t img_seq,
                        void *txt_hidden, int64_t txt_seq,
                        void *t_emb,
                        void *pe);

    // One Euler-flow scheduler step: given the DiT's predicted velocity
    // `model_out` (NPU buffer), the current latent, and the Euler-flow
    // sigma/t schedule, in-place updates `latent`. Port of the stable-
    // diffusion.cpp Euler-flow kernel. See denoiser.hpp in
    // tools/ominix_diffusion/src/ for the reference. Phase 4.
    void scheduler_step_(void *latent_dev, const void *model_out_dev,
                         int step_idx);
};

}  // namespace ominix_qie
