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
// Phase 2 runtime context. The engine tracks a few counters across init so
// Phase-2 smoke can print a receipts blob (tensor count + peak HBM) without
// piping through a whole stats struct. These are read-only accessors for
// the driver binary.
// ---------------------------------------------------------------------------
struct DiTInitStats {
    int64_t tensors_uploaded  = 0;
    // F16 bytes covers small 1D tensors (biases, modulation biases) that
    // never route through Q4-resident packing — biases are always small.
    size_t  f16_weight_bytes  = 0;
    size_t  f32_weight_bytes  = 0;
    // Q2.1 Q4-resident load path: the big Linear-weight matrices are kept
    // on device in INT4 + per-group-32 F16 scale form instead of expanding
    // to F16. These counters surface the two contributions separately so
    // smoke receipts can compare against the probe doc's budget
    // (W4 ≈ 5.1 GiB, scale ≈ 0.6 GiB at full DiT).
    size_t  q4_weight_bytes   = 0;
    size_t  q4_scale_bytes    = 0;
    int64_t q4_tensors        = 0;
    int64_t f16_fallback_tensors = 0;  // non-Q4 GGUF tensors that fell back
                                        // through the F16 upload path
    size_t  scratch_bytes     = 0;
    size_t  rope_bytes        = 0;
    double  load_wall_ms      = 0.0;
    double  dequant_wall_ms   = 0.0;
};

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
// Q2.1 Q4-RESIDENT: every matmul-weight slot is a PAIR — a packed INT4
// buffer (shape [K, N] contiguous-K, `K*N/2` bytes) and a per-group F16
// scale buffer (shape [K/32, N], 2 bytes per scale entry). The pair is
// designed to feed `aclnnWeightQuantBatchMatmulV3` with
// `antiquantGroupSize=32` per `docs/qie_q2_q4resident_probe.md`. GGUF
// tensors that arrive as F16/F32 (not Q4_0) fall back to an F16-only
// upload (see load_matmul_weight_upload in image_diffusion_engine.cpp):
// in that fallback the `_w_q4` pointer holds the F16 weight buffer
// directly and `_scale` stays null so forward-path dispatch can
// branch on scale==null → use aclnnMm, scale!=null → use WQBMMv3.
//
// Biases and RMSNorm gammas remain single-pointer (always small 1D).
struct DiTLayerWeights {
    // --- Attention projections (img side) --------------------------------
    // Each projection's weight matrix is `[hidden, hidden]` logically;
    // stored as INT4 `[K=hidden, N=hidden]` K-contiguous + F16 scale
    // `[K/32, N]`. Bias stays F16 `[hidden]`.
    void *to_q_w_q4      = nullptr;  // INT4 packed  (K*N/2 bytes)
    void *to_q_scale     = nullptr;  // F16 [K/32, N]
    void *to_q_b         = nullptr;  // F16 [hidden]
    void *to_k_w_q4      = nullptr;
    void *to_k_scale     = nullptr;
    void *to_k_b         = nullptr;
    void *to_v_w_q4      = nullptr;
    void *to_v_scale     = nullptr;
    void *to_v_b         = nullptr;
    void *to_out_0_w_q4  = nullptr;
    void *to_out_0_scale = nullptr;
    void *to_out_0_b     = nullptr;

    // --- Attention projections (txt side, "add_*") -----------------------
    void *add_q_w_q4      = nullptr;
    void *add_q_scale     = nullptr;
    void *add_q_b         = nullptr;
    void *add_k_w_q4      = nullptr;
    void *add_k_scale     = nullptr;
    void *add_k_b         = nullptr;
    void *add_v_w_q4      = nullptr;
    void *add_v_scale     = nullptr;
    void *add_v_b         = nullptr;
    void *to_add_out_w_q4 = nullptr;
    void *to_add_out_scale = nullptr;
    void *to_add_out_b    = nullptr;

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
    // the "normalize only, no gamma/beta" LayerNorm variant.
    void *img_norm1_w = nullptr;  // F32 [hidden] (may stay null)
    void *img_norm1_b = nullptr;  // F32 [hidden] (may stay null)
    void *img_norm2_w = nullptr;
    void *img_norm2_b = nullptr;
    void *txt_norm1_w = nullptr;
    void *txt_norm1_b = nullptr;
    void *txt_norm2_w = nullptr;
    void *txt_norm2_b = nullptr;

    // --- Timestep modulation heads (img_mod.1 / txt_mod.1) --------------
    // Linear(hidden → 6·hidden, bias=true).
    void *img_mod_w_q4   = nullptr;  // INT4 [K=hidden, N=6·hidden]
    void *img_mod_scale  = nullptr;  // F16 [K/32, N]
    void *img_mod_b      = nullptr;  // F16 [6·hidden]
    void *txt_mod_w_q4   = nullptr;
    void *txt_mod_scale  = nullptr;
    void *txt_mod_b      = nullptr;

    // --- FFN (GELU, NOT SwiGLU, NOT MoE) --------------------------------
    // FeedForward(dim → mult·dim, GELU, Linear(mult·dim → dim)). The
    // "up" Linear has K=hidden, N=ff_dim; "down" has K=ff_dim, N=hidden.
    void *img_ff_up_w_q4    = nullptr;  // INT4 [K=hidden, N=ff_dim]
    void *img_ff_up_scale   = nullptr;
    void *img_ff_up_b       = nullptr;  // F16 [ff_dim]
    void *img_ff_down_w_q4  = nullptr;  // INT4 [K=ff_dim, N=hidden]
    void *img_ff_down_scale = nullptr;
    void *img_ff_down_b     = nullptr;  // F16 [hidden]
    void *txt_ff_up_w_q4    = nullptr;
    void *txt_ff_up_scale   = nullptr;
    void *txt_ff_up_b       = nullptr;
    void *txt_ff_down_w_q4  = nullptr;
    void *txt_ff_down_scale = nullptr;
    void *txt_ff_down_b     = nullptr;
};

// ---------------------------------------------------------------------------
// Global (non-per-layer) weight handles.
// ---------------------------------------------------------------------------
// Same Q4-resident pairing convention as DiTLayerWeights (see comment above).
struct DiTGlobalWeights {
    // time_text_embed.timestep_embedder.linear_{1,2}
    void *time_linear1_w_q4 = nullptr;  // INT4 [K=256, N=hidden] (or F16 fallback)
    void *time_linear1_scale = nullptr;
    void *time_linear1_b    = nullptr;
    void *time_linear2_w_q4 = nullptr;  // INT4 [K=hidden, N=hidden]
    void *time_linear2_scale = nullptr;
    void *time_linear2_b    = nullptr;

    // img_in, txt_in — input projections onto `hidden`.
    void *img_in_w_q4 = nullptr;  // INT4 [K=in_channels·patch_size², N=hidden]
    void *img_in_scale = nullptr;
    void *img_in_b    = nullptr;
    void *txt_in_w_q4 = nullptr;  // INT4 [K=joint_attention_dim, N=hidden]
    void *txt_in_scale = nullptr;
    void *txt_in_b    = nullptr;

    // txt_norm: RMSNorm over joint_attention_dim (F32 gamma, 1D — stays F32).
    void *txt_norm_w = nullptr;   // F32 [joint_attention_dim]

    // norm_out = AdaLayerNormContinuous.
    //   .linear : Linear(hidden, 2·hidden, bias=true).
    void *norm_out_linear_w_q4 = nullptr;  // INT4 [K=hidden, N=2·hidden]
    void *norm_out_linear_scale = nullptr;
    void *norm_out_linear_b    = nullptr;

    // proj_out : Linear(hidden, patch_size² · out_channels, bias=true).
    void *proj_out_w_q4 = nullptr;  // INT4 [K=hidden, N=ps²·out_ch]
    void *proj_out_scale = nullptr;
    void *proj_out_b    = nullptr;

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

    // Test-only variant: open the device, allocate scratch + RoPE tables,
    // but DO NOT load any weights. The caller's probe is expected to
    // populate `layer_w_[il]` via `mutable_layer_weights(il)` with its
    // own synthetic device buffers. Used by the Q2.3 Phase 3 smoke (which
    // generates random F16 weights on-host and uploads them directly,
    // avoiding the ~18 GiB full-GGUF resident cost on ac03).
    bool init_for_smoke(const ImageDiffusionConfig &cfg, int device = 0);

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
    const DiTInitStats &stats() const { return stats_; }

    // Test-only hook: dispatch one block (by layer index) for smoke probes.
    // Phase 3 gate uses this to isolate a single block forward vs CPU
    // reference without running the full 60-block stack. Private
    // implementation is called via forward(), but this hook lets a
    // probe run block 0 in isolation.
    bool forward_block_test(int il,
                             void *img_hidden, int64_t img_seq,
                             void *txt_hidden, int64_t txt_seq,
                             void *t_emb,
                             void *pe);

    // Test-only hook: mutable access to DiTLayerWeights[il] so a probe can
    // populate synthetic weights without a real GGUF. Returns nullptr if
    // `il` is out of range or the engine has not allocated its layer
    // vector yet.
    DiTLayerWeights *mutable_layer_weights(int il);

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
    //
    // Q2.3 Phase 3 refines the scratch layout: within a block we need
    //   - separate Q/K/V staging for the joint (img+txt) attention input
    //     so concat is a pair of memcpys into a common buffer
    //   - a normed+modulated activation buffer per stream (img, txt)
    //   - an attention-output buffer big enough for joint seq
    //   - an FFN-intermediate buffer at [seq, ff_dim] per stream
    // Shared buffers (single-request, no intra-block concurrency) are fine.
    void *scratch_q_dev_    = nullptr;  // F16 [seq_total, hidden]  (txt||img)
    void *scratch_k_dev_    = nullptr;  // F16 [seq_total, hidden]
    void *scratch_v_dev_    = nullptr;  // F16 [seq_total, hidden]
    void *scratch_attn_dev_ = nullptr;  // F16 [seq_total, hidden]  (attn out)
    void *scratch_mlp_dev_  = nullptr;  // F16 [max_seq,   ff_dim]
    void *scratch_mod_dev_  = nullptr;  // F16 [12 · hidden] (6 img + 6 txt)
    void *rstd_dev_         = nullptr;  // F32 [heads · seq] (RMSNorm rstd)

    // Q2.3 additional scratch for block intermediates. These are small
    // (multiples of hidden × max_seq) and cheap to reserve up-front.
    void *scratch_img_norm_dev_ = nullptr;  // F16 [img_seq, hidden]
    void *scratch_txt_norm_dev_ = nullptr;  // F16 [txt_seq, hidden]
    void *scratch_img_out_dev_  = nullptr;  // F16 [img_seq, hidden] (attn/ffn out per-stream)
    void *scratch_txt_out_dev_  = nullptr;  // F16 [txt_seq, hidden]
    void *mean_dev_             = nullptr;  // F32 [max_seq]  (LayerNorm mean)
    void *ln_rstd_dev_          = nullptr;  // F32 [max_seq]  (LayerNorm rstd)
    // Affine-off LayerNorm helpers: since aclnnLayerNorm takes optional
    // gamma/beta we can pass nullptr. Reserved for future probe fallbacks
    // that need explicit constant tensors.

    // CFG duplicates — cond + uncond pass share weights, need separate
    // activation scratch. Phase 4 wires these.
    void *img_hidden_cond_dev_   = nullptr;
    void *img_hidden_uncond_dev_ = nullptr;
    void *txt_hidden_cond_dev_   = nullptr;
    void *txt_hidden_uncond_dev_ = nullptr;

    // aclnn workspace (grows on demand).
    void  *workspace_dev_ = nullptr;
    size_t workspace_size_ = 0;

    // --- Phase 2 bookkeeping ---
    DiTInitStats stats_{};

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
    // Returns true on success (every dispatch returned 0). False aborts
    // the outer `forward` / `denoise` caller with a logged error.
    bool forward_block_(const DiTLayerWeights &lw,
                        void *img_hidden, int64_t img_seq,
                        void *txt_hidden, int64_t txt_seq,
                        void *t_emb,
                        void *pe);

    // Matmul dispatch wrapper (Q2.3). Branches on `weight_scale_dev`:
    //   non-null → aclnnWeightQuantBatchMatmulV3 (Q4 packed + F16 scale,
    //              antiquantGroupSize=32 per Q2.1 probe)
    //   null      → aclnnMm (F16 fallback: Q4_1/Q5_K/BF16 source upload)
    //
    // Activation `x_f16` has logical shape [M, K]; weight has logical shape
    // [K, N]; output `y_f16` has logical shape [M, N]. M can be seq_len ×
    // batch collapsed. `bias_f16_dev` is applied post-matmul if non-null
    // (F16 1D [N]). Returns false on any aclnn status != 0.
    bool dispatch_matmul_(void *x_f16_dev, void *weight_dev,
                          void *weight_scale_dev, void *bias_f16_dev,
                          int64_t M, int64_t K, int64_t N,
                          void *y_f16_dev);

    // Apply element-wise `x = x * (1 + scale) + shift` with scale/shift
    // broadcasting over the seq dim. `x_f16_dev` is [B, seq, hidden];
    // scale/shift are F16 [B, hidden]. In-place on x. Phase 3 uses B=1.
    bool modulate_(void *x_f16_dev, const void *scale_f16_dev,
                   const void *shift_f16_dev,
                   int64_t B, int64_t seq, int64_t hidden);

    // Apply `x = x + src * gate` with gate broadcasting over seq.
    // `x_f16_dev` and `src_f16_dev` are [B, seq, hidden]; gate is F16
    // [B, hidden]. In-place on x (residual add with gated residual).
    bool gated_residual_add_(void *x_f16_dev, const void *src_f16_dev,
                             const void *gate_f16_dev,
                             int64_t B, int64_t seq, int64_t hidden);

    // aclnnLayerNorm dispatch: out = LayerNorm(x, normalizedShape=[hidden],
    // gamma=null, beta=null, eps=cfg_.layernorm_eps). Input/output are F16
    // [B, seq, hidden].
    bool layer_norm_(void *x_f16_dev, void *out_f16_dev,
                     int64_t B, int64_t seq, int64_t hidden);

    // aclnnRmsNorm dispatch over the last dim `head_dim`. Input/output are
    // F16 [B, seq, n_heads, head_dim]; gamma is F32 [head_dim]. For QIE Q2.3
    // we reshape to `[B * seq * n_heads, head_dim]` as required by the op.
    bool rms_norm_head_(void *x_f16_dev, void *out_f16_dev, void *gamma_f32_dev,
                        int64_t rows, int64_t head_dim);

    // Apply 3D-axial RoPE to a Q or K tensor using the pre-computed pe
    // table. `x_f16_dev` has layout [B, seq, n_heads, head_dim] F16.
    // `pe_f16_dev` layout matches `Rope::apply_rope` expectations:
    // [seq_pe_max, head_dim/2, 2, 2]. `pe_row_offset` is the starting row
    // inside pe to apply (0 for txt stream, ctx_len for img stream).
    // Output is in-place on x.
    bool apply_rope_(void *x_f16_dev,
                     const void *pe_f16_dev, int64_t pe_row_offset,
                     int64_t B, int64_t seq, int64_t n_heads, int64_t head_dim);

    // One Euler-flow scheduler step: given the DiT's predicted velocity
    // `model_out` (NPU buffer), the current latent, and the Euler-flow
    // sigma/t schedule, in-place updates `latent`. Port of the stable-
    // diffusion.cpp Euler-flow kernel. See denoiser.hpp in
    // tools/ominix_diffusion/src/ for the reference. Phase 4.
    void scheduler_step_(void *latent_dev, const void *model_out_dev,
                         int step_idx);
};

}  // namespace ominix_qie
