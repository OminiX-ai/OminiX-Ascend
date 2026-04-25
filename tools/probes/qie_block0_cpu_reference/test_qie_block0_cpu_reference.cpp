// ============================================================================
// Q2.4.5.4i Step 2 — Block-0 CPU reference (ggml CPU backend, F32) forward
// for native-vs-CPU discrimination.
//
// Mission (per docs/qie_q2_phase4_smoke.md §5.5.8 hand-off):
//   Native engine produces block-0 output with mean_abs ≈ 1.1e5, max_abs ≈ 7.4e6
//   under QIE_FFN_DOWN_BF16=1 — and final post-60-block latent is all-NaN.
//   Two open angles:
//     (1) GGUF row-permutation in AdaLN modulation chunks
//     (2) Algorithm error somewhere in the native block forward
//
//   This probe is the discriminator. It runs ONE block-0 forward through the
//   ggml CPU backend (F32 end-to-end, no Ascend) on the SAME inputs the
//   native engine consumed. If the CPU output magnitudes are also large
//   (>1e4), then the native engine is correct and the trained Q4 weights +
//   AdaLN binding are inherently large-magnitude → strategic pivot.
//   If the CPU output is healthy (<1e2), then the native engine has an
//   algorithm error in modulation/attn/ffn ordering → bug-hunt continues.
//
// I/O contract:
//   Inputs from /tmp/qie_block0_inputs/ (or $QIE_BLOCK0_INPUTS_DIR):
//     00_img.f32       [img_seq, H]       F32 — produced by native dump
//     00_txt.f32       [txt_seq, H]       F32 — produced by native dump
//     00_t_emb.f32     [H]                F32 — produced by native dump
//   GGUF: $QIE_Q45_GGUF or default Qwen-Image-Edit-2509-Q4_0.gguf path.
//   Shapes: $QIE_Q45_IMG_SEQ + $QIE_Q45_TXT_SEQ control activation shape.
//
// Outputs to $QIE_BLOCK0_OUTPUTS_DIR (default /tmp/qie_block0_outputs/):
//     cpu_24_img_resid2.f32   [img_seq, H]  F32
//     cpu_24_txt_resid2.f32   [txt_seq, H]  F32
//
// Build on ac03:
//   cd tools/probes/qie_block0_cpu_reference && bash build_and_run.sh
// ============================================================================

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

// ominix_diffusion includes — pull in QwenImageTransformerBlock + ModelLoader.
#include "tools/ominix_diffusion/src/model.h"
#include "tools/ominix_diffusion/src/ggml_extend.hpp"
#include "tools/ominix_diffusion/src/qwen_image.hpp"
#include "tools/ominix_diffusion/src/rope.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <string>
#include <vector>
#include <map>
#include <memory>

// ---------------------------------------------------------------------------
// Filesystem + binary I/O helpers.
// ---------------------------------------------------------------------------
static bool read_f32_bin(const std::string &path, std::vector<float> &out,
                          size_t expected_n) {
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) {
        std::fprintf(stderr, "[block0_cpu] open %s failed (errno=%d)\n",
                     path.c_str(), errno);
        return false;
    }
    std::fseek(f, 0, SEEK_END);
    long sz = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    if (sz < 0 || (size_t)sz != expected_n * sizeof(float)) {
        std::fprintf(stderr, "[block0_cpu] %s: size %ld != expected %zu*4 = %zu\n",
                     path.c_str(), sz, expected_n, expected_n * sizeof(float));
        std::fclose(f);
        return false;
    }
    out.resize(expected_n);
    size_t got = std::fread(out.data(), sizeof(float), expected_n, f);
    std::fclose(f);
    if (got != expected_n) {
        std::fprintf(stderr, "[block0_cpu] %s: short read %zu/%zu\n",
                     path.c_str(), got, expected_n);
        return false;
    }
    return true;
}

static bool write_f32_bin(const std::string &path,
                           const float *data, size_t n) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (!f) {
        std::fprintf(stderr, "[block0_cpu] open %s for write failed\n",
                     path.c_str());
        return false;
    }
    size_t wrote = std::fwrite(data, sizeof(float), n, f);
    std::fclose(f);
    if (wrote != n) {
        std::fprintf(stderr, "[block0_cpu] %s: short write %zu/%zu\n",
                     path.c_str(), wrote, n);
        return false;
    }
    return true;
}

// ---------------------------------------------------------------------------
// Stats helper.
// ---------------------------------------------------------------------------
struct Stats {
    double mean_abs;
    double max_abs;
    int64_t nan_c;
    int64_t inf_c;
};
static Stats compute_stats(const float *p, size_t n) {
    Stats s = {0.0, 0.0, 0, 0};
    double sum_abs = 0.0;
    int64_t valid = 0;
    for (size_t i = 0; i < n; ++i) {
        float v = p[i];
        if (std::isnan(v)) { s.nan_c++; continue; }
        if (std::isinf(v)) { s.inf_c++; continue; }
        double a = std::fabs((double)v);
        sum_abs += a;
        if (a > s.max_abs) s.max_abs = a;
        valid++;
    }
    s.mean_abs = valid > 0 ? sum_abs / (double)valid : 0.0;
    return s;
}

// ---------------------------------------------------------------------------
// Helper that re-exposes GGMLBlock's protected `blocks` map publicly. We
// reinterpret the existing QwenImageModel through this derived view to
// reach into transformer_blocks.0 without modifying upstream headers.
// ---------------------------------------------------------------------------
struct PublicQwenImageModel : public Qwen::QwenImageModel {
    using Qwen::QwenImageModel::blocks;
};

struct PublicQwenImageTransformerBlock : public Qwen::QwenImageTransformerBlock {
    using GGMLBlock::blocks;
};

// Q2.4.5.4k: expose JointAttention internals (blocks + dim_head) so the
// inline-replicate path in compute_block0 can emit per-substep dumps.
struct PublicQwenImageAttention : public Qwen::QwenImageAttention {
    using GGMLBlock::blocks;
    using Qwen::QwenImageAttention::dim_head;
};

// ---------------------------------------------------------------------------
// Block-0 CPU runner. Wraps a single QwenImageTransformerBlock and runs a
// forward graph on the CPU backend in F32. Inputs (img, txt, t_emb, pe) are
// supplied as raw float arrays; outputs (img', txt') are written into caller
// buffers via GGMLRunner::compute's output_ctx.
// ---------------------------------------------------------------------------
struct Block0CpuRunner : public Qwen::QwenImageRunner {
    int64_t img_seq;
    int64_t txt_seq;
    std::vector<float> img_in;
    std::vector<float> txt_in;
    std::vector<float> t_emb_in;
    std::vector<float> pe_in;

    Block0CpuRunner(ggml_backend_t backend,
                     const String2TensorStorage &tsm,
                     int64_t img_seq, int64_t txt_seq)
        : Qwen::QwenImageRunner(backend, /*offload_params_to_cpu*/ false,
                                       tsm, "model.diffusion_model",
                                       VERSION_QWEN_IMAGE,
                                       /*zero_cond_t*/ false),
          img_seq(img_seq), txt_seq(txt_seq) {}

    std::string get_desc() override { return "qwen_image_block0_cpu"; }

    // Build a graph that runs ONLY transformer_blocks.0.forward on the
    // supplied (img, txt, t_emb, pe). Returns a graph whose final node is
    // the concatenated [img_out, txt_out] for D2H readout.
    struct ggml_cgraph *build_graph_block0() {
        struct ggml_cgraph *gf = new_graph_custom(Qwen::QWEN_IMAGE_GRAPH_SIZE);

        const int64_t H = qwen_image_params.num_attention_heads *
                          qwen_image_params.attention_head_dim;

        // Allocate the input tensors on the runtime ctx + register host data.
        auto img = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, H, img_seq, 1);
        auto txt = ggml_new_tensor_3d(compute_ctx, GGML_TYPE_F32, H, txt_seq, 1);
        // t_emb is [B, H] in the QwenImageTransformerBlock contract.
        auto t_emb = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, H, 1);

        // pe is laid out as [2, 2, axes_dim_sum/2, pos_len] in qwen_image.hpp:702.
        const int axes_dim_sum = qwen_image_params.axes_dim_sum;
        const int pos_len = (int)(pe_in.size() / axes_dim_sum / 2);
        auto pe = ggml_new_tensor_4d(compute_ctx, GGML_TYPE_F32,
                                       2, 2, axes_dim_sum / 2, pos_len);

        ggml_set_name(img,   "in_img");
        ggml_set_name(txt,   "in_txt");
        ggml_set_name(t_emb, "in_t_emb");
        ggml_set_name(pe,    "in_pe");

        set_backend_tensor_data(img,   img_in.data());
        set_backend_tensor_data(txt,   txt_in.data());
        set_backend_tensor_data(t_emb, t_emb_in.data());
        set_backend_tensor_data(pe,    pe_in.data());

        // Replicate Qwen::QwenImageTransformerBlock::forward INLINE so we
        // can intercept each substep with ggml_set_name + forward_expand.
        // The substep names match the native engine's `dump_tensor_*` files
        // so the comparator can do a parallel substep bisect.
        // Source: qwen_image.hpp:255-340 (verbatim, with named taps).
        auto &qim_pub = reinterpret_cast<PublicQwenImageModel&>(qwen_image);
        auto &block_pub = reinterpret_cast<PublicQwenImageTransformerBlock&>(
            *qim_pub.blocks["transformer_blocks.0"]);

        auto img_mod_1 = std::dynamic_pointer_cast<Linear>(block_pub.blocks["img_mod.1"]);
        auto img_norm1 = std::dynamic_pointer_cast<LayerNorm>(block_pub.blocks["img_norm1"]);
        auto img_norm2 = std::dynamic_pointer_cast<LayerNorm>(block_pub.blocks["img_norm2"]);
        auto img_mlp   = std::dynamic_pointer_cast<FeedForward>(block_pub.blocks["img_mlp"]);

        auto txt_mod_1 = std::dynamic_pointer_cast<Linear>(block_pub.blocks["txt_mod.1"]);
        auto txt_norm1 = std::dynamic_pointer_cast<LayerNorm>(block_pub.blocks["txt_norm1"]);
        auto txt_norm2 = std::dynamic_pointer_cast<LayerNorm>(block_pub.blocks["txt_norm2"]);
        auto txt_mlp   = std::dynamic_pointer_cast<FeedForward>(block_pub.blocks["txt_mlp"]);

        auto attn = std::dynamic_pointer_cast<Qwen::QwenImageAttention>(block_pub.blocks["attn"]);

        auto runner_ctx_obj = get_context();
        GGMLRunnerContext *ctx = &runner_ctx_obj;

        // Modulation chains.
        auto img_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
        img_mod_params         = img_mod_1->forward(ctx, img_mod_params);
        auto img_mod_param_vec = block_pub.get_mod_params_vec(ctx->ggml_ctx, img_mod_params, nullptr);

        auto txt_mod_params    = ggml_silu(ctx->ggml_ctx, t_emb);
        txt_mod_params         = txt_mod_1->forward(ctx, txt_mod_params);
        auto txt_mod_param_vec = block_pub.get_mod_params_vec(ctx->ggml_ctx, txt_mod_params, nullptr);

        // LN1 + mod1 substeps.
        auto img_normed    = img_norm1->forward(ctx, img);
        auto img_modulated = Flux::modulate(ctx->ggml_ctx, img_normed,
                                              img_mod_param_vec[0],
                                              img_mod_param_vec[1], false);
        auto img_gate1     = img_mod_param_vec[2];

        auto txt_normed    = txt_norm1->forward(ctx, txt);
        auto txt_modulated = Flux::modulate(ctx->ggml_ctx, txt_normed,
                                              txt_mod_param_vec[0],
                                              txt_mod_param_vec[1]);
        auto txt_gate1     = txt_mod_param_vec[2];

        ggml_set_name(img_normed,    "cpu_04_img_LN1");
        ggml_set_name(img_modulated, "cpu_05_img_mod1");
        ggml_set_name(txt_normed,    "cpu_06_txt_LN1");
        ggml_set_name(txt_modulated, "cpu_07_txt_mod1");
        // ggml_set_output prevents the gallocr from reusing this tensor's
        // memory for downstream ops — required so we can read intermediate
        // values back after compute() returns.
        ggml_set_output(img_normed);
        ggml_set_output(img_modulated);
        ggml_set_output(txt_normed);
        ggml_set_output(txt_modulated);
        ggml_build_forward_expand(gf, img_normed);
        ggml_build_forward_expand(gf, img_modulated);
        ggml_build_forward_expand(gf, txt_normed);
        ggml_build_forward_expand(gf, txt_modulated);

        // Gate reshape (matches qwen_image.hpp:309-316 N=1 branch).
        img_gate1 = ggml_reshape_4d(ctx->ggml_ctx, img_gate1,
                                     img_gate1->ne[0], 1,
                                     img_gate1->ne[1], img_gate1->ne[2]);
        txt_gate1 = ggml_reshape_4d(ctx->ggml_ctx, txt_gate1,
                                     txt_gate1->ne[0], 1,
                                     txt_gate1->ne[1], txt_gate1->ne[2]);

        // Attention — inline-replicate Qwen::QwenImageAttention::forward
        // (qwen_image.hpp:113-191) to expose pre-projection (08/09/10/11pre)
        // intermediates for the Q2.4.5.4k bisect. Using-redeclares cast.
        auto &attn_pub = reinterpret_cast<PublicQwenImageAttention&>(*attn);
        const int dim_head_attn = attn_pub.dim_head;

        auto a_norm_q       = std::dynamic_pointer_cast<UnaryBlock>(attn_pub.blocks["norm_q"]);
        auto a_norm_k       = std::dynamic_pointer_cast<UnaryBlock>(attn_pub.blocks["norm_k"]);
        auto a_to_q         = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["to_q"]);
        auto a_to_k         = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["to_k"]);
        auto a_to_v         = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["to_v"]);
        auto a_to_out_0     = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["to_out.0"]);
        auto a_norm_added_q = std::dynamic_pointer_cast<UnaryBlock>(attn_pub.blocks["norm_added_q"]);
        auto a_norm_added_k = std::dynamic_pointer_cast<UnaryBlock>(attn_pub.blocks["norm_added_k"]);
        auto a_add_q_proj   = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["add_q_proj"]);
        auto a_add_k_proj   = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["add_k_proj"]);
        auto a_add_v_proj   = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["add_v_proj"]);
        auto a_to_add_out   = std::dynamic_pointer_cast<Linear>(attn_pub.blocks["to_add_out"]);

        const int64_t Nb         = img_modulated->ne[2];
        const int64_t n_img_tok  = img_modulated->ne[1];
        const int64_t n_txt_tok  = txt_modulated->ne[1];

        auto cpu_img_q = a_to_q->forward(ctx, img_modulated);
        const int64_t cpu_num_heads = cpu_img_q->ne[0] / dim_head_attn;
        cpu_img_q      = ggml_reshape_4d(ctx->ggml_ctx, cpu_img_q, dim_head_attn, cpu_num_heads, n_img_tok, Nb);
        auto cpu_img_k = a_to_k->forward(ctx, img_modulated);
        cpu_img_k      = ggml_reshape_4d(ctx->ggml_ctx, cpu_img_k, dim_head_attn, cpu_num_heads, n_img_tok, Nb);
        auto cpu_img_v = a_to_v->forward(ctx, img_modulated);
        cpu_img_v      = ggml_reshape_4d(ctx->ggml_ctx, cpu_img_v, dim_head_attn, cpu_num_heads, n_img_tok, Nb);

        ggml_set_name(cpu_img_q, "cpu_08_img_Q");
        ggml_set_name(cpu_img_k, "cpu_08_img_K");
        ggml_set_name(cpu_img_v, "cpu_08_img_V");
        ggml_set_output(cpu_img_q);
        ggml_set_output(cpu_img_k);
        ggml_set_output(cpu_img_v);
        ggml_build_forward_expand(gf, cpu_img_q);
        ggml_build_forward_expand(gf, cpu_img_k);
        ggml_build_forward_expand(gf, cpu_img_v);

        cpu_img_q = a_norm_q->forward(ctx, cpu_img_q);
        cpu_img_k = a_norm_k->forward(ctx, cpu_img_k);

        ggml_set_name(cpu_img_q, "cpu_09_img_Q_rmsnorm");
        ggml_set_name(cpu_img_k, "cpu_09_img_K_rmsnorm");
        ggml_set_output(cpu_img_q);
        ggml_set_output(cpu_img_k);
        ggml_build_forward_expand(gf, cpu_img_q);
        ggml_build_forward_expand(gf, cpu_img_k);

        auto cpu_txt_q = a_add_q_proj->forward(ctx, txt_modulated);
        cpu_txt_q      = ggml_reshape_4d(ctx->ggml_ctx, cpu_txt_q, dim_head_attn, cpu_num_heads, n_txt_tok, Nb);
        auto cpu_txt_k = a_add_k_proj->forward(ctx, txt_modulated);
        cpu_txt_k      = ggml_reshape_4d(ctx->ggml_ctx, cpu_txt_k, dim_head_attn, cpu_num_heads, n_txt_tok, Nb);
        auto cpu_txt_v = a_add_v_proj->forward(ctx, txt_modulated);
        cpu_txt_v      = ggml_reshape_4d(ctx->ggml_ctx, cpu_txt_v, dim_head_attn, cpu_num_heads, n_txt_tok, Nb);

        ggml_set_name(cpu_txt_q, "cpu_08_txt_Q");
        ggml_set_name(cpu_txt_k, "cpu_08_txt_K");
        ggml_set_name(cpu_txt_v, "cpu_08_txt_V");
        ggml_set_output(cpu_txt_q);
        ggml_set_output(cpu_txt_k);
        ggml_set_output(cpu_txt_v);
        ggml_build_forward_expand(gf, cpu_txt_q);
        ggml_build_forward_expand(gf, cpu_txt_k);
        ggml_build_forward_expand(gf, cpu_txt_v);

        cpu_txt_q = a_norm_added_q->forward(ctx, cpu_txt_q);
        cpu_txt_k = a_norm_added_k->forward(ctx, cpu_txt_k);

        ggml_set_name(cpu_txt_q, "cpu_09_txt_Q_rmsnorm");
        ggml_set_name(cpu_txt_k, "cpu_09_txt_K_rmsnorm");
        ggml_set_output(cpu_txt_q);
        ggml_set_output(cpu_txt_k);
        ggml_build_forward_expand(gf, cpu_txt_q);
        ggml_build_forward_expand(gf, cpu_txt_k);

        auto cpu_q_joint = ggml_concat(ctx->ggml_ctx, cpu_txt_q, cpu_img_q, 2);
        auto cpu_k_joint = ggml_concat(ctx->ggml_ctx, cpu_txt_k, cpu_img_k, 2);
        auto cpu_v_joint = ggml_concat(ctx->ggml_ctx, cpu_txt_v, cpu_img_v, 2);

        // Pre-projection FIA-equivalent attention output (mirrors native's
        // 11_attn_out which is the FIA output BEFORE to_out_0/to_add_out).
        auto cpu_attn_pre = Rope::attention(ctx, cpu_q_joint, cpu_k_joint,
                                              cpu_v_joint, pe,
                                              /*mask*/ nullptr, 1.0f / 128.f);
        auto cpu_txt_attn_pre = ggml_view_3d(
            ctx->ggml_ctx, cpu_attn_pre,
            cpu_attn_pre->ne[0], n_txt_tok, cpu_attn_pre->ne[2],
            cpu_attn_pre->nb[1], cpu_attn_pre->nb[2], 0);
        auto cpu_img_attn_pre = ggml_view_3d(
            ctx->ggml_ctx, cpu_attn_pre,
            cpu_attn_pre->ne[0], n_img_tok, cpu_attn_pre->ne[2],
            cpu_attn_pre->nb[1], cpu_attn_pre->nb[2],
            n_txt_tok * cpu_attn_pre->nb[1]);
        cpu_img_attn_pre = ggml_cont(ctx->ggml_ctx, cpu_img_attn_pre);
        cpu_txt_attn_pre = ggml_cont(ctx->ggml_ctx, cpu_txt_attn_pre);
        ggml_set_name(cpu_img_attn_pre, "cpu_11pre_attn_out_img");
        ggml_set_name(cpu_txt_attn_pre, "cpu_11pre_attn_out_txt");
        ggml_set_output(cpu_img_attn_pre);
        ggml_set_output(cpu_txt_attn_pre);
        ggml_build_forward_expand(gf, cpu_img_attn_pre);
        ggml_build_forward_expand(gf, cpu_txt_attn_pre);

        // Final post-projection outputs (mirrors native's 12_to_*).
        auto img_attn_output = a_to_out_0->forward(ctx, cpu_img_attn_pre);
        auto txt_attn_output = a_to_add_out->forward(ctx, cpu_txt_attn_pre);
        ggml_set_name(img_attn_output, "cpu_12_to_out_0");
        ggml_set_name(txt_attn_output, "cpu_12_to_add_out");
        ggml_set_output(img_attn_output);
        ggml_set_output(txt_attn_output);
        ggml_build_forward_expand(gf, img_attn_output);
        ggml_build_forward_expand(gf, txt_attn_output);

        // Residual #1.
        auto img1 = ggml_add(ctx->ggml_ctx, img,
                              ggml_mul(ctx->ggml_ctx, img_attn_output, img_gate1));
        auto txt1 = ggml_add(ctx->ggml_ctx, txt,
                              ggml_mul(ctx->ggml_ctx, txt_attn_output, txt_gate1));
        ggml_set_name(img1, "cpu_13_img_resid1");
        ggml_set_name(txt1, "cpu_13_txt_resid1");
        ggml_set_output(img1);
        ggml_set_output(txt1);
        ggml_build_forward_expand(gf, img1);
        ggml_build_forward_expand(gf, txt1);

        // LN2 + mod2 substeps.
        auto img_normed2    = img_norm2->forward(ctx, img1);
        auto img_modulated2 = Flux::modulate(ctx->ggml_ctx, img_normed2,
                                               img_mod_param_vec[3],
                                               img_mod_param_vec[4], false);
        auto img_gate2      = img_mod_param_vec[5];

        auto txt_normed2    = txt_norm2->forward(ctx, txt1);
        auto txt_modulated2 = Flux::modulate(ctx->ggml_ctx, txt_normed2,
                                               txt_mod_param_vec[3],
                                               txt_mod_param_vec[4]);
        auto txt_gate2      = txt_mod_param_vec[5];

        ggml_set_name(img_normed2,    "cpu_14_img_LN2");
        ggml_set_name(img_modulated2, "cpu_15_img_mod2");
        ggml_set_name(txt_normed2,    "cpu_16_txt_LN2");
        ggml_set_name(txt_modulated2, "cpu_17_txt_mod2");
        ggml_set_output(img_normed2);
        ggml_set_output(img_modulated2);
        ggml_set_output(txt_normed2);
        ggml_set_output(txt_modulated2);
        ggml_build_forward_expand(gf, img_normed2);
        ggml_build_forward_expand(gf, img_modulated2);
        ggml_build_forward_expand(gf, txt_normed2);
        ggml_build_forward_expand(gf, txt_modulated2);

        img_gate2 = ggml_reshape_4d(ctx->ggml_ctx, img_gate2,
                                     img_gate2->ne[0], 1,
                                     img_gate2->ne[1], img_gate2->ne[2]);
        txt_gate2 = ggml_reshape_4d(ctx->ggml_ctx, txt_gate2,
                                     txt_gate2->ne[0], 1,
                                     txt_gate2->ne[1], txt_gate2->ne[2]);

        // FFN substeps (post-down only).
        auto img_mlp_out = img_mlp->forward(ctx, img_modulated2);
        auto txt_mlp_out = txt_mlp->forward(ctx, txt_modulated2);
        ggml_set_name(img_mlp_out, "cpu_20_img_ff_down");
        ggml_set_name(txt_mlp_out, "cpu_23_txt_ff_down");
        ggml_set_output(img_mlp_out);
        ggml_set_output(txt_mlp_out);
        ggml_build_forward_expand(gf, img_mlp_out);
        ggml_build_forward_expand(gf, txt_mlp_out);

        // Residual #2.
        auto img_out = ggml_add(ctx->ggml_ctx, img1,
                                  ggml_mul(ctx->ggml_ctx, img_mlp_out, img_gate2));
        auto txt_out = ggml_add(ctx->ggml_ctx, txt1,
                                  ggml_mul(ctx->ggml_ctx, txt_mlp_out, txt_gate2));
        ggml_set_name(img_out, "img_out");
        ggml_set_name(txt_out, "txt_out");
        ggml_build_forward_expand(gf, img_out);
        ggml_build_forward_expand(gf, txt_out);
        return gf;
    }

    // Compute and write outputs into provided host buffers. Returns false
    // on any error.
    bool compute_block0(int n_threads,
                          std::vector<float> &img_out,
                          std::vector<float> &txt_out) {
        const int64_t H = qwen_image_params.num_attention_heads *
                          qwen_image_params.attention_head_dim;

        struct ggml_init_params params = {};
        params.mem_size = (size_t)512 * 1024 * 1024;  // 512 MiB host work ctx
        params.mem_buffer = nullptr;
        params.no_alloc = false;
        struct ggml_context *work_ctx = ggml_init(params);
        if (!work_ctx) {
            std::fprintf(stderr, "[block0_cpu] ggml_init(work) failed\n");
            return false;
        }

        auto get_graph = [this]() -> struct ggml_cgraph * {
            return this->build_graph_block0();
        };

        // We want both img_out and txt_out. GGMLRunner::compute returns only
        // a single output. So instead pre-tile into a single concat and
        // post-split here. Simpler: run twice with different "output" picks.
        // Even simpler: monkey-patch the output via graph node lookup.
        struct ggml_tensor *out_concat = nullptr;
        // Trick: after compute, both 'img_out' and 'txt_out' nodes live in
        // compute_ctx; we copy out via ggml_backend_tensor_get on the
        // backend tensors. But GGMLRunner::compute is the only path that
        // sets up the gallocr / copy_data_to_backend flow. Easiest: invoke
        // GGMLRunner::compute with output=nullptr (just run the graph),
        // then read both outputs via the public `get_first_stage_tensor`
        // helper or directly via tensor names.
        if (!GGMLRunner::compute(get_graph, n_threads, false,
                                   /*output*/ nullptr, work_ctx)) {
            std::fprintf(stderr, "[block0_cpu] runner compute failed\n");
            ggml_free(work_ctx);
            return false;
        }

        // After compute: locate img_out / txt_out by name in compute_ctx
        // (compute_ctx is reset on next compute, so we read NOW).
        // GGMLRunner::get_compute_graph overrides the LAST graph node's name
        // to final_result_name. We expand img_out first then txt_out, so
        // txt_out is the last node and now lives under final_result_name.
        struct ggml_tensor *img_t = ggml_get_tensor(compute_ctx, "img_out");
        struct ggml_tensor *txt_t = ggml_get_tensor(compute_ctx,
            final_result_name.c_str());
        if (!img_t || !txt_t) {
            std::fprintf(stderr, "[block0_cpu] output tensors not found "
                                  "(img=%p txt=%p)\n", (void*)img_t, (void*)txt_t);
            ggml_free(work_ctx);
            return false;
        }
        img_out.resize((size_t)img_seq * H);
        txt_out.resize((size_t)txt_seq * H);
        ggml_backend_tensor_get(img_t, img_out.data(), 0,
                                  img_out.size() * sizeof(float));
        ggml_backend_tensor_get(txt_t, txt_out.data(), 0,
                                  txt_out.size() * sizeof(float));

        // Substep dumps (Q2.4.5.4j extension) — written to
        // $QIE_BLOCK0_OUTPUTS_DIR/cpu_<step>.f32, paired with the native
        // engine's QIE_DUMP_BLOCK0_DIR substep dumps for substep-by-substep
        // bisect via compare_block0.py.
        const char *dump_env = std::getenv("QIE_BLOCK0_OUTPUTS_DIR");
        if (dump_env) {
            std::string out_dir(dump_env);
            auto dump_named = [&](const char *cpu_name,
                                    const char *file_name,
                                    int64_t expected_n) {
                struct ggml_tensor *t = ggml_get_tensor(compute_ctx, cpu_name);
                if (!t) {
                    std::fprintf(stderr, "[block0_cpu] substep '%s' not in compute_ctx\n",
                                 cpu_name);
                    return;
                }
                int64_t n = ggml_nelements(t);
                if (n != expected_n) {
                    std::fprintf(stderr, "[block0_cpu] substep '%s' size %lld != expected %lld\n",
                                 cpu_name, (long long)n, (long long)expected_n);
                }
                std::vector<float> buf((size_t)n);
                ggml_backend_tensor_get(t, buf.data(), 0,
                                         buf.size() * sizeof(float));
                std::string path = out_dir + "/" + file_name;
                FILE *fp = std::fopen(path.c_str(), "wb");
                if (!fp) {
                    std::fprintf(stderr, "[block0_cpu] open %s for write failed\n",
                                 path.c_str());
                    return;
                }
                std::fwrite(buf.data(), sizeof(float), buf.size(), fp);
                std::fclose(fp);
            };
            // Substep ↔ filename map (matches native engine dumps).
            const int64_t img_n = (int64_t)img_seq * H;
            const int64_t txt_n = (int64_t)txt_seq * H;
            const int64_t HD = (int64_t)128;
            const int64_t NH = H / HD;
            const int64_t img_qkv_n = (int64_t)img_seq * NH * HD;  // == img_n
            const int64_t txt_qkv_n = (int64_t)txt_seq * NH * HD;  // == txt_n
            (void)img_qkv_n; (void)txt_qkv_n;
            dump_named("cpu_04_img_LN1",     "cpu_04_img_LN1.f32",     img_n);
            dump_named("cpu_05_img_mod1",    "cpu_05_img_mod1.f32",    img_n);
            dump_named("cpu_06_txt_LN1",     "cpu_06_txt_LN1.f32",     txt_n);
            dump_named("cpu_07_txt_mod1",    "cpu_07_txt_mod1.f32",    txt_n);
            // Q2.4.5.4k attention internals — mirrors native intra_probe sites
            // 08/09/10 (which the existing native dump scaffold already writes
            // when QIE_DUMP_BLOCK0_DIR is set, but only for 04/05/06/07 + 11+
            // — extend native side similarly to enable a like-for-like compare
            // here).
            dump_named("cpu_08_img_Q",       "cpu_08_img_Q.f32",       img_n);
            dump_named("cpu_08_img_K",       "cpu_08_img_K.f32",       img_n);
            dump_named("cpu_08_img_V",       "cpu_08_img_V.f32",       img_n);
            dump_named("cpu_08_txt_Q",       "cpu_08_txt_Q.f32",       txt_n);
            dump_named("cpu_08_txt_K",       "cpu_08_txt_K.f32",       txt_n);
            dump_named("cpu_08_txt_V",       "cpu_08_txt_V.f32",       txt_n);
            dump_named("cpu_09_img_Q_rmsnorm", "cpu_09_img_Q_rmsnorm.f32", img_n);
            dump_named("cpu_09_img_K_rmsnorm", "cpu_09_img_K_rmsnorm.f32", img_n);
            dump_named("cpu_09_txt_Q_rmsnorm", "cpu_09_txt_Q_rmsnorm.f32", txt_n);
            dump_named("cpu_09_txt_K_rmsnorm", "cpu_09_txt_K_rmsnorm.f32", txt_n);
            // Pre-projection FIA-equivalent attention output (mirrors native's
            // 11_attn_out which is FIA out, BEFORE to_out_0/to_add_out).
            dump_named("cpu_11pre_attn_out_img", "cpu_11pre_attn_out_img.f32", img_n);
            dump_named("cpu_11pre_attn_out_txt", "cpu_11pre_attn_out_txt.f32", txt_n);
            // Backward-compat aliases: prior runs compared
            // native_11_attn_out (pre-proj) against cpu_11_attn_out
            // (post-proj) — a category mismatch. Re-emit cpu_11_attn_out_*
            // as the pre-proj tensor so the comparator's 11_attn_out row is
            // semantically correct against native's 11_attn_out FIA dump.
            dump_named("cpu_11pre_attn_out_img", "cpu_11_attn_out_img.f32", img_n);
            dump_named("cpu_11pre_attn_out_txt", "cpu_11_attn_out_txt.f32", txt_n);
            dump_named("cpu_12_to_out_0",    "cpu_12_to_out_0.f32",    img_n);
            dump_named("cpu_12_to_add_out",  "cpu_12_to_add_out.f32",  txt_n);
            dump_named("cpu_13_img_resid1",  "cpu_13_img_resid1.f32",  img_n);
            dump_named("cpu_13_txt_resid1",  "cpu_13_txt_resid1.f32",  txt_n);
            dump_named("cpu_14_img_LN2",     "cpu_14_img_LN2.f32",     img_n);
            dump_named("cpu_15_img_mod2",    "cpu_15_img_mod2.f32",    img_n);
            dump_named("cpu_16_txt_LN2",     "cpu_16_txt_LN2.f32",     txt_n);
            dump_named("cpu_17_txt_mod2",    "cpu_17_txt_mod2.f32",    txt_n);
            dump_named("cpu_20_img_ff_down", "cpu_20_img_ff_down.f32", img_n);
            dump_named("cpu_23_txt_ff_down", "cpu_23_txt_ff_down.f32", txt_n);
        }

        ggml_free(work_ctx);
        return true;
    }
};

// ---------------------------------------------------------------------------
// Main.
// ---------------------------------------------------------------------------
int main(int argc, char **argv) {
    (void)argc; (void)argv;
    setvbuf(stdout, nullptr, _IOLBF, 0);

    // ---- Inputs ----
    const char *gguf_env = std::getenv("QIE_Q45_GGUF");
    std::string gguf_path = gguf_env
        ? std::string(gguf_env)
        : std::string("/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf");

    const char *in_dir_env = std::getenv("QIE_BLOCK0_INPUTS_DIR");
    std::string in_dir = in_dir_env ? std::string(in_dir_env)
                                    : std::string("/tmp/qie_block0_inputs");

    const char *out_dir_env = std::getenv("QIE_BLOCK0_OUTPUTS_DIR");
    std::string out_dir = out_dir_env ? std::string(out_dir_env)
                                      : std::string("/tmp/qie_block0_outputs");

    int64_t img_seq = 64;
    int64_t txt_seq = 32;
    if (const char *e = std::getenv("QIE_Q45_BIG"); e && e[0] != '0') {
        img_seq = 256; txt_seq = 64;
    }
    if (const char *e = std::getenv("QIE_Q45_IMG_SEQ")) img_seq = std::atoi(e);
    if (const char *e = std::getenv("QIE_Q45_TXT_SEQ")) txt_seq = std::atoi(e);

    int n_threads = 8;
    if (const char *e = std::getenv("QIE_BLOCK0_THREADS")) {
        int k = std::atoi(e);
        if (k > 0 && k <= 64) n_threads = k;
    }

    const int64_t H = 24 * 128;  // num_attention_heads * head_dim
    const int axes_dim_sum = 128;
    const int patch_size = 2;

    std::printf("[block0_cpu] gguf=%s\n", gguf_path.c_str());
    std::printf("[block0_cpu] in_dir=%s\n", in_dir.c_str());
    std::printf("[block0_cpu] out_dir=%s\n", out_dir.c_str());
    std::printf("[block0_cpu] img_seq=%lld txt_seq=%lld H=%lld n_threads=%d\n",
                (long long)img_seq, (long long)txt_seq, (long long)H, n_threads);

    // ---- Load activation dumps ----
    std::vector<float> img_in, txt_in, t_emb_in;
    if (!read_f32_bin(in_dir + "/00_img.f32",   img_in,   (size_t)img_seq * H)) return 2;
    if (!read_f32_bin(in_dir + "/00_txt.f32",   txt_in,   (size_t)txt_seq * H)) return 2;
    if (!read_f32_bin(in_dir + "/00_t_emb.f32", t_emb_in, (size_t)H))           return 2;

    auto s_img = compute_stats(img_in.data(),   img_in.size());
    auto s_txt = compute_stats(txt_in.data(),   txt_in.size());
    auto s_t   = compute_stats(t_emb_in.data(), t_emb_in.size());
    std::printf("[block0_cpu] in_img:  mean_abs=%.4g max_abs=%.4g NaN=%lld Inf=%lld\n",
                s_img.mean_abs, s_img.max_abs, (long long)s_img.nan_c, (long long)s_img.inf_c);
    std::printf("[block0_cpu] in_txt:  mean_abs=%.4g max_abs=%.4g NaN=%lld Inf=%lld\n",
                s_txt.mean_abs, s_txt.max_abs, (long long)s_txt.nan_c, (long long)s_txt.inf_c);
    std::printf("[block0_cpu] in_t_emb: mean_abs=%.4g max_abs=%.4g NaN=%lld Inf=%lld\n",
                s_t.mean_abs, s_t.max_abs, (long long)s_t.nan_c, (long long)s_t.inf_c);

    // ---- Load GGUF ----
    auto t_load0 = std::chrono::steady_clock::now();
    ModelLoader model_loader;
    // Match QwenImageRunner::load_from_file_and_test convention: prefix
    // "model.diffusion_model." is added by init_from_file_and_convert_name
    // so the resulting tensor names match the runner's expected paths
    // (transformer_blocks.0.* maps to model.diffusion_model.transformer_blocks.0.*).
    if (!model_loader.init_from_file_and_convert_name(gguf_path,
                                                       "model.diffusion_model.")) {
        std::fprintf(stderr, "[block0_cpu] init_from_file failed: %s\n",
                     gguf_path.c_str());
        return 2;
    }
    auto t_load1 = std::chrono::steady_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t_load1 - t_load0).count();
    std::printf("[block0_cpu] gguf parse OK (%.1f ms)\n", load_ms);

    // CPU runner: ggml_backend_cpu_init(); F32 throughout.
    ggml_backend_t backend = ggml_backend_cpu_init();
    if (!backend) {
        std::fprintf(stderr, "[block0_cpu] ggml_backend_cpu_init failed\n");
        return 2;
    }

    auto &tsm = model_loader.get_tensor_storage_map();
    Block0CpuRunner runner(backend, tsm, img_seq, txt_seq);
    runner.qwen_image_params.num_layers = 60;  // re-pin (constructor scans tsm)

    if (!runner.alloc_params_buffer()) {
        std::fprintf(stderr, "[block0_cpu] alloc_params_buffer failed\n");
        return 2;
    }
    std::map<std::string, ggml_tensor *> tensors;
    runner.get_param_tensors(tensors, "model.diffusion_model");

    // Tensor count receipt.
    std::printf("[block0_cpu] runner expects %zu tensors\n", tensors.size());

    if (!model_loader.load_tensors(tensors)) {
        std::fprintf(stderr, "[block0_cpu] load_tensors failed\n");
        return 2;
    }
    auto t_load2 = std::chrono::steady_clock::now();
    double load_full_ms = std::chrono::duration<double, std::milli>(t_load2 - t_load0).count();
    std::printf("[block0_cpu] gguf load + alloc + dequant OK (total %.1f ms)\n",
                load_full_ms);

    // ---- Build pe vector (matches QwenImageRunner::build_graph) ----
    // h, w must satisfy h*w/patch_size^2 = img_seq for the small smoke
    // (img_seq=64 → h=w=16, img_seq=256 → h=w=32). Caller can override
    // by setting QIE_BLOCK0_PE_H + QIE_BLOCK0_PE_W.
    int pe_h = 0, pe_w = 0;
    if (const char *eh = std::getenv("QIE_BLOCK0_PE_H")) pe_h = std::atoi(eh);
    if (const char *ew = std::getenv("QIE_BLOCK0_PE_W")) pe_w = std::atoi(ew);
    if (pe_h <= 0 || pe_w <= 0) {
        // Default: square grid (img_seq must be a perfect square in
        // patch units).
        int patch_tokens = (int)img_seq;
        int side = (int)std::lround(std::sqrt((double)patch_tokens));
        while (side * side < patch_tokens) ++side;
        pe_h = side * patch_size;  // pre-patch h = side * 2
        pe_w = side * patch_size;
    }
    std::printf("[block0_cpu] pe_h=%d pe_w=%d patch=%d txt_ctx=%lld\n",
                pe_h, pe_w, patch_size, (long long)txt_seq);
    runner.pe_in = Rope::gen_qwen_image_pe(pe_h, pe_w, patch_size,
                                             /*bs*/ 1,
                                             /*context_len*/ (int)txt_seq,
                                             /*ref_latents*/ {},
                                             /*increase_ref_index*/ false,
                                             /*theta*/ 10000,
                                             /*circular_h*/ false,
                                             /*circular_w*/ false,
                                             /*axes_dim*/ {16, 56, 56});
    int pe_pos_len = (int)(runner.pe_in.size() / axes_dim_sum / 2);
    std::printf("[block0_cpu] pe_vec size=%zu floats (pos_len=%d)\n",
                runner.pe_in.size(), pe_pos_len);

    runner.img_in   = std::move(img_in);
    runner.txt_in   = std::move(txt_in);
    runner.t_emb_in = std::move(t_emb_in);

    // ---- Run block-0 forward ----
    std::vector<float> img_out, txt_out;
    auto t_run0 = std::chrono::steady_clock::now();
    if (!runner.compute_block0(n_threads, img_out, txt_out)) {
        std::fprintf(stderr, "[block0_cpu] compute_block0 failed\n");
        ggml_backend_free(backend);
        return 2;
    }
    auto t_run1 = std::chrono::steady_clock::now();
    double run_ms = std::chrono::duration<double, std::milli>(t_run1 - t_run0).count();
    std::printf("[block0_cpu] compute_block0 OK (%.1f ms)\n", run_ms);

    // ---- Stats + save ----
    auto s_img_out = compute_stats(img_out.data(), img_out.size());
    auto s_txt_out = compute_stats(txt_out.data(), txt_out.size());
    std::printf("\n=== CPU reference block-0 output ===\n");
    std::printf("cpu_24_img_resid2: mean_abs=%.4g max_abs=%.4g NaN=%lld Inf=%lld\n",
                s_img_out.mean_abs, s_img_out.max_abs,
                (long long)s_img_out.nan_c, (long long)s_img_out.inf_c);
    std::printf("cpu_24_txt_resid2: mean_abs=%.4g max_abs=%.4g NaN=%lld Inf=%lld\n",
                s_txt_out.mean_abs, s_txt_out.max_abs,
                (long long)s_txt_out.nan_c, (long long)s_txt_out.inf_c);

    // mkdir -p out_dir.
    std::string mkdir_cmd = "mkdir -p \"" + out_dir + "\"";
    int rc = std::system(mkdir_cmd.c_str());
    (void)rc;
    if (!write_f32_bin(out_dir + "/cpu_24_img_resid2.f32",
                        img_out.data(), img_out.size())) return 2;
    if (!write_f32_bin(out_dir + "/cpu_24_txt_resid2.f32",
                        txt_out.data(), txt_out.size())) return 2;
    std::printf("\n[block0_cpu] outputs written to %s\n", out_dir.c_str());

    ggml_backend_free(backend);
    return 0;
}
