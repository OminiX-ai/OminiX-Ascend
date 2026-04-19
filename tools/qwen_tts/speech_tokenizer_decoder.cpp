#include "speech_tokenizer_decoder.h"
#include "ggml.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

// ============================================================================
// load_hparams
// ============================================================================

bool SpeechTokenizerDecoderModel::load_hparams(const ModelLoader &loader) {
    loader.get_u32("codebook_size", config.codebook_size);
    loader.get_u32("hidden_size", config.hidden_size);
    loader.get_u32("num_hidden_layers", config.num_hidden_layers);
    loader.get_u32("num_attention_heads", config.num_attention_heads);
    loader.get_u32("num_key_value_heads", config.num_key_value_heads);
    loader.get_u32("num_quantizers", config.num_quantizers);
    loader.get_u32("decoder_dim", config.decoder_dim);
    loader.get_u32("sliding_window", config.sliding_window);
    loader.get_u32("latent_dim", config.latent_dim, false);
    loader.get_f32("rope_theta", config.rope_theta);
    loader.get_f32("rms_norm_eps", config.rms_norm_eps);
    loader.get_u32("output_sample_rate", config.output_sample_rate, false);
    loader.get_u32("decode_upsample_rate", config.decode_upsample_rate, false);

    // head_dim derived from attention projection shapes
    // q_proj maps hidden_size → num_heads * head_dim
    // We'll derive it after loading tensors; use default 64 for now
    config.head_dim = 64;

    printf("[decoder] codebook_size=%d hidden=%d layers=%d heads=%d/%d "
           "quantizers=%d decoder_dim=%d window=%d\n",
           config.codebook_size, config.hidden_size, config.num_hidden_layers,
           config.num_attention_heads, config.num_key_value_heads,
           config.num_quantizers, config.decoder_dim, config.sliding_window);
    return true;
}

// ============================================================================
// reset_input_shape
// ============================================================================

void SpeechTokenizerDecoderModel::reset_input_shape() {
    // codes input: [num_quantizers * time_steps] flattened as i32
    // We'll reshape in build_graph. Default: 16 quantizers × 100 time steps
    input_shapes_ = {
        {"codes", {config.num_quantizers * 100}},
    };
}

// ============================================================================
// get_tensors_to_load (271 tensors)
// ============================================================================

std::vector<ggml_tensor *>
SpeechTokenizerDecoderModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    auto load_rvq_group = [&](RVQGroup &grp, const std::string &prefix,
                              int n_layers) {
        grp.input_proj_w =
            get_tensor(ctx, (prefix + ".input_proj.weight").c_str(), tensors);
        grp.output_proj_w =
            get_tensor(ctx, (prefix + ".output_proj.weight").c_str(), tensors);
        grp.codebooks.resize(n_layers);
        for (int i = 0; i < n_layers; i++) {
            std::string lp = prefix + ".vq.layers." + std::to_string(i) +
                             "._codebook.";
            grp.codebooks[i].cluster_usage =
                get_tensor(ctx, (lp + "cluster_usage").c_str(), tensors);
            grp.codebooks[i].embedding_sum =
                get_tensor(ctx, (lp + "embedding_sum").c_str(), tensors);
        }
    };

    // RVQ quantizer
    load_rvq_group(rvq_first_, "quantizer.rvq_first", 1);
    load_rvq_group(rvq_rest_, "quantizer.rvq_rest", config.num_quantizers - 1);

    // Pre-conv
    pre_conv_w_ = get_tensor(ctx, "pre_conv.conv.weight", tensors);
    pre_conv_b_ = get_tensor(ctx, "pre_conv.conv.bias", tensors);

    // Pre-transformer
    pt_input_proj_w_ =
        get_tensor(ctx, "pre_transformer.input_proj.weight", tensors);
    pt_input_proj_b_ =
        get_tensor(ctx, "pre_transformer.input_proj.bias", tensors);
    pt_output_proj_w_ =
        get_tensor(ctx, "pre_transformer.output_proj.weight", tensors);
    pt_output_proj_b_ =
        get_tensor(ctx, "pre_transformer.output_proj.bias", tensors);
    pt_norm_w_ = get_tensor(ctx, "pre_transformer.norm.weight", tensors);

    pt_layers_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string lp = "pre_transformer.layers." + std::to_string(i) + ".";
        auto &l = pt_layers_[i];
        l.q_proj_w =
            get_tensor(ctx, (lp + "self_attn.q_proj.weight").c_str(), tensors);
        l.k_proj_w =
            get_tensor(ctx, (lp + "self_attn.k_proj.weight").c_str(), tensors);
        l.v_proj_w =
            get_tensor(ctx, (lp + "self_attn.v_proj.weight").c_str(), tensors);
        l.o_proj_w =
            get_tensor(ctx, (lp + "self_attn.o_proj.weight").c_str(), tensors);
        l.gate_proj_w =
            get_tensor(ctx, (lp + "mlp.gate_proj.weight").c_str(), tensors);
        l.up_proj_w =
            get_tensor(ctx, (lp + "mlp.up_proj.weight").c_str(), tensors);
        l.down_proj_w =
            get_tensor(ctx, (lp + "mlp.down_proj.weight").c_str(), tensors);
        l.input_layernorm_w =
            get_tensor(ctx, (lp + "input_layernorm.weight").c_str(), tensors);
        l.post_attention_layernorm_w = get_tensor(
            ctx, (lp + "post_attention_layernorm.weight").c_str(), tensors);
        l.self_attn_layer_scale = get_tensor(
            ctx, (lp + "self_attn_layer_scale.scale").c_str(), tensors);
        l.mlp_layer_scale =
            get_tensor(ctx, (lp + "mlp_layer_scale.scale").c_str(), tensors);
    }

    // Upsample blocks
    for (int i = 0; i < 2; i++) {
        std::string bp = "upsample." + std::to_string(i) + ".";
        auto &blk = upsample_blocks_[i];
        blk.conv_w = get_tensor(ctx, (bp + "0.conv.weight").c_str(), tensors);
        blk.conv_b = get_tensor(ctx, (bp + "0.conv.bias").c_str(), tensors);
        auto &cn = blk.convnext;
        cn.gamma = get_tensor(ctx, (bp + "1.gamma").c_str(), tensors);
        cn.dwconv_w =
            get_tensor(ctx, (bp + "1.dwconv.conv.weight").c_str(), tensors);
        cn.dwconv_b =
            get_tensor(ctx, (bp + "1.dwconv.conv.bias").c_str(), tensors);
        cn.norm_w = get_tensor(ctx, (bp + "1.norm.weight").c_str(), tensors);
        cn.norm_b = get_tensor(ctx, (bp + "1.norm.bias").c_str(), tensors);
        cn.pwconv1_w =
            get_tensor(ctx, (bp + "1.pwconv1.weight").c_str(), tensors);
        cn.pwconv1_b =
            get_tensor(ctx, (bp + "1.pwconv1.bias").c_str(), tensors);
        cn.pwconv2_w =
            get_tensor(ctx, (bp + "1.pwconv2.weight").c_str(), tensors);
        cn.pwconv2_b =
            get_tensor(ctx, (bp + "1.pwconv2.bias").c_str(), tensors);
    }

    // Vocoder: initial conv
    voc_init_conv_w_ = get_tensor(ctx, "decoder.0.conv.weight", tensors);
    voc_init_conv_b_ = get_tensor(ctx, "decoder.0.conv.bias", tensors);

    // Vocoder: 4 decoder blocks (decoder.1 - decoder.4)
    for (int i = 0; i < 4; i++) {
        std::string bp = "decoder." + std::to_string(i + 1) + ".block.";
        auto &blk = voc_blocks_[i];
        blk.snake.alpha = get_tensor(ctx, (bp + "0.alpha").c_str(), tensors);
        blk.snake.beta = get_tensor(ctx, (bp + "0.beta").c_str(), tensors);
        blk.transconv_w =
            get_tensor(ctx, (bp + "1.conv.weight").c_str(), tensors);
        blk.transconv_b =
            get_tensor(ctx, (bp + "1.conv.bias").c_str(), tensors);
        // 3 residual units (block.2, block.3, block.4)
        for (int j = 0; j < 3; j++) {
            std::string rp = bp + std::to_string(j + 2) + ".";
            auto &ru = blk.res_units[j];
            ru.act1.alpha =
                get_tensor(ctx, (rp + "act1.alpha").c_str(), tensors);
            ru.act1.beta =
                get_tensor(ctx, (rp + "act1.beta").c_str(), tensors);
            ru.conv1_w =
                get_tensor(ctx, (rp + "conv1.conv.weight").c_str(), tensors);
            ru.conv1_b =
                get_tensor(ctx, (rp + "conv1.conv.bias").c_str(), tensors);
            ru.act2.alpha =
                get_tensor(ctx, (rp + "act2.alpha").c_str(), tensors);
            ru.act2.beta =
                get_tensor(ctx, (rp + "act2.beta").c_str(), tensors);
            ru.conv2_w =
                get_tensor(ctx, (rp + "conv2.conv.weight").c_str(), tensors);
            ru.conv2_b =
                get_tensor(ctx, (rp + "conv2.conv.bias").c_str(), tensors);
        }
    }

    // Vocoder: final SnakeBeta + conv
    voc_final_snake_.alpha = get_tensor(ctx, "decoder.5.alpha", tensors);
    voc_final_snake_.beta = get_tensor(ctx, "decoder.5.beta", tensors);
    voc_final_conv_w_ = get_tensor(ctx, "decoder.6.conv.weight", tensors);
    voc_final_conv_b_ = get_tensor(ctx, "decoder.6.conv.bias", tensors);

    printf("[decoder] loaded %zu tensors\n", tensors.size());
    return tensors;
}

// ============================================================================
// Graph building helpers
// ============================================================================

// Causal Conv1d: left-pad by (kernel_size-1)*dilation, then conv with p=0
ggml_tensor *SpeechTokenizerDecoderModel::build_causal_conv1d(
    ggml_context *ctx0, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b,
    int d) {
    int kernel_size = (int)w->ne[0];
    int pad_left = (kernel_size - 1) * d;
    if (pad_left > 0) {
        // Zero-pad on the left (dim 0) using ggml_pad_ext
        x = ggml_pad_ext(ctx0, x, pad_left, 0, 0, 0, 0, 0, 0, 0);
    }
    return build_conv1d(ctx0, x, w, b, 1, 0, d);
}

// Depthwise causal Conv1d: efficient for groups=channels (e.g., ConvNeXt dwconv)
// x: [seq_len, C, 1], w: [K, 1, C] → output: [seq_len, C, 1]
// Uses shift-and-multiply approach: K iterations instead of C groups
ggml_tensor *SpeechTokenizerDecoderModel::build_depthwise_conv1d_causal(
    ggml_context *ctx0, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b) {
    int K = (int)w->ne[0];   // kernel size
    int C = (int)w->ne[2];   // channels
    int seq_len = (int)x->ne[0];

    // Causal left-padding
    if (K > 1) {
        x = ggml_pad_ext(ctx0, x, K - 1, 0, 0, 0, 0, 0, 0, 0);
    }
    // x: [seq_len + K-1, C, 1]

    // Cast weight to F32 if needed (F16 not supported in element-wise ops)
    if (w->type != GGML_TYPE_F32) {
        w = ggml_cast(ctx0, w, GGML_TYPE_F32);
    }
    // Reshape weight: [K, 1, C] → [K, C] then transpose to [C, K]
    ggml_tensor *w_2d = ggml_reshape_2d(ctx0, w, K, C);
    ggml_tensor *w_t = ggml_cont(ctx0, ggml_permute(ctx0, w_2d, 1, 0, 2, 3));
    // w_t: [C, K] — w_t[:, k] is contiguous for each k

    ggml_tensor *output = nullptr;
    for (int k = 0; k < K; k++) {
        // Shifted input: x[k:k+seq_len, :] → [seq_len, C]
        ggml_tensor *shifted = ggml_view_2d(ctx0, x, seq_len, C,
                                             x->nb[1],
                                             k * ggml_element_size(x));

        // Weight for kernel position k: w_t[:, k] → [C] contiguous
        ggml_tensor *w_k = ggml_view_1d(ctx0, w_t, C,
                                          k * C * ggml_element_size(w_t));
        // Reshape to [1, C] for broadcasting
        w_k = ggml_reshape_2d(ctx0, w_k, 1, C);

        // Element-wise multiply with broadcasting: [seq_len, C] * [1, C]
        ggml_tensor *term = ggml_mul(ctx0, shifted, w_k);

        if (output == nullptr) {
            output = term;
        } else {
            output = ggml_add(ctx0, output, term);
        }
    }

    // Add bias
    if (b) {
        ggml_tensor *b_r = ggml_reshape_2d(ctx0, b, 1, C);
        output = ggml_add(ctx0, output, b_r);
    }

    return ggml_reshape_3d(ctx0, output, seq_len, C, 1);
}

// Causal ConvTranspose1d: transposed conv then trim right by (kernel-stride)
ggml_tensor *SpeechTokenizerDecoderModel::build_causal_transconv1d(
    ggml_context *ctx0, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b,
    int stride) {
    int kernel_size = (int)w->ne[0];
    // ggml_conv_transpose_1d (via im2col) requires F16 kernel weights.
    if (w->type != GGML_TYPE_F16) {
        w = ggml_cast(ctx0, w, GGML_TYPE_F16);
    }
    // ggml_conv_transpose_1d(ctx, w, x, stride, padding, dilation)
    // Use p=0, d=1 for the raw transposed conv
    ggml_tensor *out = ggml_conv_transpose_1d(ctx0, w, x, stride, 0, 1);
    if (b) {
        ggml_tensor *b_reshaped = ggml_reshape_2d(ctx0, b, 1, b->ne[0]);
        out = ggml_add(ctx0, out, b_reshaped);
    }
    // Trim right padding: remove (kernel_size - stride) samples from right
    int trim = kernel_size - stride;
    if (trim > 0) {
        int64_t out_len = out->ne[0] - trim;
        out = ggml_view_3d(ctx0, out, out_len, out->ne[1], out->ne[2],
                           out->nb[1], out->nb[2], 0);
        out = ggml_cont(ctx0, out);
    }
    return out;
}

// SnakeBeta: x + exp(-beta) * sin²(x * exp(alpha))
// x can be 2D [seq_len, channels] or 3D [seq_len, channels, 1]
ggml_tensor *SpeechTokenizerDecoderModel::build_snake_beta(
    ggml_context *ctx0, ggml_tensor *x, const SnakeBetaParams &params) {
    int channels = (int)params.alpha->ne[0];
    // Use 2D reshape [1, channels] — broadcasts with both 2D and 3D x
    ggml_tensor *alpha_exp =
        ggml_exp(ctx0, ggml_reshape_2d(ctx0, params.alpha, 1, channels));

    // sin_part = sin(x * alpha_exp)
    ggml_tensor *sin_part = ggml_sin(ctx0, ggml_mul(ctx0, x, alpha_exp));
    // sin² = sin_part * sin_part
    ggml_tensor *sin_sq = ggml_mul(ctx0, sin_part, sin_part);

    // inv_beta = exp(-beta) = 1/exp(beta)
    ggml_tensor *neg_beta = ggml_neg(ctx0,
        ggml_reshape_2d(ctx0, params.beta, 1, channels));
    ggml_tensor *inv_beta = ggml_exp(ctx0, neg_beta);

    // result = x + sin² * inv_beta  (sin_sq is larger, must be first arg in ggml_mul)
    return ggml_add(ctx0, x, ggml_mul(ctx0, sin_sq, inv_beta));
}

// ConvNeXt block: dwconv → norm → pwconv1 → GELU → pwconv2 → gamma → residual
ggml_tensor *SpeechTokenizerDecoderModel::build_convnext(
    ggml_context *ctx0, ggml_tensor *x, const ConvNeXtBlock &blk) {
    ggml_tensor *residual = x;
    // Depthwise causal conv (groups=channels)
    ggml_tensor *cur = build_depthwise_conv1d_causal(ctx0, x, blk.dwconv_w, blk.dwconv_b);

    // Permute to [channels, seq_len] for LayerNorm (need channels in ne[0])
    // cur: [seq_len, channels, 1] → transpose to [channels, seq_len, 1]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // LayerNorm over channels dimension (ne[0])
    cur = build_norm(ctx0, cur, blk.norm_w, blk.norm_b, NORM_TYPE_NORMAL,
                     1e-5f);
    // pwconv1: Linear(dim, 4*dim)
    cur = build_linear(ctx0, cur, blk.pwconv1_w, blk.pwconv1_b);
    cur = ggml_gelu(ctx0, cur);
    // pwconv2: Linear(4*dim, dim)
    cur = build_linear(ctx0, cur, blk.pwconv2_w, blk.pwconv2_b);
    // gamma scaling
    cur = ggml_mul(ctx0, cur, blk.gamma);
    // Permute back to [seq_len, channels, 1]
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // Residual
    return ggml_add(ctx0, cur, residual);
}

// ============================================================================
// RVQ decode: codes[16, T] → embeddings[512, T]
// ============================================================================

ggml_tensor *SpeechTokenizerDecoderModel::build_rvq_decode(
    ggml_context *ctx0, ggml_tensor *codes) {
    // codes: [num_quantizers * T] i32 input, reshape to [T, num_quantizers]
    int n_q = config.num_quantizers;
    int T = (int)(codes->ne[0] / n_q);
    ggml_tensor *codes_2d = ggml_reshape_2d(ctx0, codes, T, n_q);

    auto decode_group = [&](RVQGroup &grp, int q_start, int q_end)
        -> ggml_tensor * {
        int codebook_dim = (int)grp.codebooks[0].embedding_sum->ne[0];
        ggml_tensor *sum = nullptr;

        for (int qi = q_start; qi < q_end; qi++) {
            int local_idx = qi - q_start;
            auto &cb = grp.codebooks[local_idx];

            // Compute normalized embedding: embedding_sum / cluster_usage
            // embedding_sum may be F16 — cast to F32 for binary ops
            ggml_tensor *emb_sum = cb.embedding_sum;
            if (emb_sum->type != GGML_TYPE_F32) {
                emb_sum = ggml_cast(ctx0, emb_sum, GGML_TYPE_F32);
            }
            // Reshape cluster_usage to [1, codebook_size] for broadcasting
            ggml_tensor *usage_2d = ggml_reshape_2d(ctx0, cb.cluster_usage,
                                                     1, cb.cluster_usage->ne[0]);
            ggml_tensor *embedding = ggml_div(ctx0, emb_sum, usage_2d);
            // embedding: [codebook_dim, codebook_size]

            // Extract codes for this quantizer: codes_2d[:, qi]
            // codes_2d: [T, n_q], we want column qi → [T]
            ggml_tensor *qi_codes = ggml_view_2d(ctx0, codes_2d, T, 1,
                                                  codes_2d->nb[1], qi * codes_2d->nb[1]);
            qi_codes = ggml_reshape_1d(ctx0, qi_codes, T);

            // Lookup: embedding[:, codes] → [codebook_dim, T]
            ggml_tensor *quantized = ggml_get_rows(ctx0, embedding, qi_codes);
            // ggml_get_rows returns [codebook_dim, T]

            if (sum == nullptr) {
                sum = quantized;
            } else {
                sum = ggml_add(ctx0, sum, quantized);
            }
        }
        return sum; // [codebook_dim, T]
    };

    // Decode rvq_first (semantic, quantizer 0)
    ggml_tensor *first_sum = decode_group(rvq_first_, 0, 1);
    // first_sum: [codebook_dim(256), T] from ggml_get_rows
    // Conv1d expects [seq_len, in_channels, batch] → transpose to [T, 256, 1]
    first_sum = ggml_cont(ctx0, ggml_permute(ctx0, first_sum, 1, 0, 2, 3));
    first_sum = ggml_reshape_3d(ctx0, first_sum, first_sum->ne[0], first_sum->ne[1], 1);
    // output_proj: Conv1d(256→512, k=1)
    ggml_tensor *first_out = build_conv1d(ctx0, first_sum, rvq_first_.output_proj_w,
                                           nullptr, 1, 0, 1);

    // Decode rvq_rest (acoustic, quantizers 1-15)
    ggml_tensor *rest_sum = decode_group(rvq_rest_, 1, n_q);
    // rest_sum: [codebook_dim(256), T] → transpose to [T, 256, 1]
    rest_sum = ggml_cont(ctx0, ggml_permute(ctx0, rest_sum, 1, 0, 2, 3));
    rest_sum = ggml_reshape_3d(ctx0, rest_sum, rest_sum->ne[0], rest_sum->ne[1], 1);
    ggml_tensor *rest_out = build_conv1d(ctx0, rest_sum, rvq_rest_.output_proj_w,
                                          nullptr, 1, 0, 1);

    // Sum both groups → [T, 512, 1]
    // Note: conv1d output is [out_len, out_channels, batch]
    return ggml_add(ctx0, first_out, rest_out);
}

// ============================================================================
// Pre-transformer: input_proj → 8 layers → norm → output_proj
// ============================================================================

ggml_tensor *SpeechTokenizerDecoderModel::build_pre_transformer(
    ggml_context *ctx0, ggml_tensor *x) {
    // x: [seq_len, latent_dim(1024), 1] from pre_conv
    // Transpose to [latent_dim, seq_len] for linear ops
    int seq_len = (int)x->ne[0];
    int latent_dim = (int)x->ne[1];
    ggml_tensor *cur = ggml_reshape_2d(ctx0, x, seq_len, latent_dim);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // cur: [latent_dim, seq_len]

    // input_proj: Linear(latent_dim → hidden_size)
    cur = build_linear(ctx0, cur, pt_input_proj_w_, pt_input_proj_b_);
    // cur: [hidden_size, seq_len]

    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    int head_dim = config.head_dim;
    int hidden = config.hidden_size;
    float kq_scale = 1.0f / sqrtf((float)head_dim);

    // Build sliding window causal mask
    // mask shape: [seq_len, seq_len] with -inf for masked positions
    ggml_tensor *kq_mask =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // Position IDs for RoPE
    ggml_tensor *pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "dec_pos");
    ggml_set_input(pos);

    // Transformer layers
    for (int il = 0; il < config.num_hidden_layers; il++) {
        auto &layer = pt_layers_[il];
        ggml_tensor *residual = cur;

        // RMSNorm
        cur = build_norm(ctx0, cur, layer.input_layernorm_w, nullptr,
                         NORM_TYPE_RMS, config.rms_norm_eps, 1, il);

        // Self-attention: Q, K, V projections
        ggml_tensor *q = ggml_mul_mat(ctx0, layer.q_proj_w, cur);
        ggml_tensor *k = ggml_mul_mat(ctx0, layer.k_proj_w, cur);
        ggml_tensor *v = ggml_mul_mat(ctx0, layer.v_proj_w, cur);

        // Reshape for multi-head: [head_dim, n_heads, seq_len]
        q = ggml_reshape_3d(ctx0, q, head_dim, n_heads, seq_len);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_kv_heads, seq_len);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_kv_heads, seq_len);

        // Apply RoPE (NEOX-style: split-half rotation)
        q = ggml_rope_ext(ctx0, q, pos, nullptr, head_dim,
                           GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                           1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, head_dim,
                           GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                           1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Attention
        ggml_tensor *attn_out = build_attn(ctx0, layer.o_proj_w, nullptr,
                                            q, k, v, kq_mask, kq_scale, il);

        // Layer scale + residual
        attn_out = ggml_mul(ctx0, attn_out, layer.self_attn_layer_scale);
        cur = ggml_add(ctx0, residual, attn_out);

        // MLP
        residual = cur;
        cur = build_norm(ctx0, cur, layer.post_attention_layernorm_w, nullptr,
                         NORM_TYPE_RMS, config.rms_norm_eps, 1, il);

        // SwiGLU FFN
        ggml_tensor *ffn_out = build_ffn(ctx0, cur, layer.up_proj_w, nullptr,
                                          layer.gate_proj_w, nullptr,
                                          layer.down_proj_w, nullptr,
                                          FFN_SILU, il);

        // Layer scale + residual
        ffn_out = ggml_mul(ctx0, ffn_out, layer.mlp_layer_scale);
        cur = ggml_add(ctx0, residual, ffn_out);
    }

    // Final norm
    cur = build_norm(ctx0, cur, pt_norm_w_, nullptr, NORM_TYPE_RMS,
                     config.rms_norm_eps);

    // output_proj: Linear(hidden_size → latent_dim)
    cur = build_linear(ctx0, cur, pt_output_proj_w_, pt_output_proj_b_);
    // cur: [latent_dim, seq_len]

    // Transpose back to [seq_len, latent_dim, 1] for conv operations
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_3d(ctx0, cur, seq_len, latent_dim, 1);
    return cur;
}

// ============================================================================
// Vocoder: initial conv → 4 blocks (SnakeBeta+TransConv+ResUnits) → final
// ============================================================================

ggml_tensor *SpeechTokenizerDecoderModel::build_vocoder(
    ggml_context *ctx0, ggml_tensor *x) {
    // x: [seq_len, 1024, 1]
    // Initial causal conv: 1024 → 1536, kernel=7
    ggml_tensor *cur = build_causal_conv1d(ctx0, x, voc_init_conv_w_,
                                            voc_init_conv_b_);

    int dilations[3] = {1, 3, 9};

    // 4 decoder blocks
    for (int i = 0; i < 4; i++) {
        auto &blk = voc_blocks_[i];
        int stride = config.upsample_rates[i];

        // SnakeBeta activation
        cur = build_snake_beta(ctx0, cur, blk.snake);

        // Causal transposed conv (upsample)
        cur = build_causal_transconv1d(ctx0, cur, blk.transconv_w,
                                        blk.transconv_b, stride);

        // 3 residual units
        for (int j = 0; j < 3; j++) {
            auto &ru = blk.res_units[j];
            ggml_tensor *res = cur;
            cur = build_snake_beta(ctx0, cur, ru.act1);
            cur = build_causal_conv1d(ctx0, cur, ru.conv1_w, ru.conv1_b,
                                       dilations[j]);
            cur = build_snake_beta(ctx0, cur, ru.act2);
            cur = build_causal_conv1d(ctx0, cur, ru.conv2_w, ru.conv2_b);
            cur = ggml_add(ctx0, cur, res);
        }
    }

    // Final SnakeBeta + causal conv → 1 channel
    cur = build_snake_beta(ctx0, cur, voc_final_snake_);
    cur = build_causal_conv1d(ctx0, cur, voc_final_conv_w_, voc_final_conv_b_);

    // Tanh activation on output
    cur = ggml_tanh(ctx0, cur);
    return cur;
}

// ============================================================================
// Main build_graph: RVQ → pre_conv → pre_transformer → upsample → vocoder
// ============================================================================

std::vector<ggml_tensor *>
SpeechTokenizerDecoderModel::build_graph(ggml_context *ctx0) {
    std::vector<int> codes_shape = input_shapes_["codes"];
    int total_codes = codes_shape[0]; // num_quantizers * T

    // Input: flattened codes [num_quantizers * T] as i32
    ggml_tensor *codes =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, total_codes);
    ggml_set_name(codes, "codes");
    ggml_set_input(codes);

    // Step 1: RVQ decode → [T, 512, 1]
    ggml_tensor *cur = build_rvq_decode(ctx0, codes);

    // Step 2: Pre-conv (causal Conv1d 512→1024, k=3)
    cur = build_causal_conv1d(ctx0, cur, pre_conv_w_, pre_conv_b_);
    // cur: [T, 1024, 1]

    // Step 3: Pre-transformer (8 layers sliding window attention)
    cur = build_pre_transformer(ctx0, cur);
    // cur: [T, 1024, 1]

    // Debug early exit: stop after pre-transformer
    if (debug_stop_after_ == 1) {
        cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
        ggml_set_name(cur, "audio_output");
        ggml_set_output(cur);
        return {cur};
    }

    // Step 4: Upsample (2 blocks, each 2× → total 4×)
    {
        auto &blk = upsample_blocks_[0];
        cur = build_causal_transconv1d(ctx0, cur, blk.conv_w, blk.conv_b,
                                        config.upsampling_ratios[0]);
        if (debug_stop_after_ == 20) {  // after upsample0 transconv only
            cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
            ggml_set_name(cur, "audio_output"); ggml_set_output(cur); return {cur};
        }
        cur = build_convnext(ctx0, cur, blk.convnext);
        if (debug_stop_after_ == 21) {  // after upsample0 full
            cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
            ggml_set_name(cur, "audio_output"); ggml_set_output(cur); return {cur};
        }
    }
    {
        auto &blk = upsample_blocks_[1];
        cur = build_causal_transconv1d(ctx0, cur, blk.conv_w, blk.conv_b,
                                        config.upsampling_ratios[1]);
        if (debug_stop_after_ == 22) {  // after upsample1 transconv only
            cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
            ggml_set_name(cur, "audio_output"); ggml_set_output(cur); return {cur};
        }
        cur = build_convnext(ctx0, cur, blk.convnext);
    }
    // cur: [T*4, 1024, 1]

    if (debug_stop_after_ == 2) {
        cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
        ggml_set_name(cur, "audio_output");
        ggml_set_output(cur);
        return {cur};
    }

    // Step 5: Vocoder - split into sub-steps for debug
    // 5a: Initial causal conv: 1024 → 1536, kernel=7
    cur = build_causal_conv1d(ctx0, cur, voc_init_conv_w_, voc_init_conv_b_);

    if (debug_stop_after_ == 3) {
        cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
        ggml_set_name(cur, "audio_output");
        ggml_set_output(cur);
        return {cur};
    }

    // 5b: 4 vocoder blocks
    int dilations[3] = {1, 3, 9};
    for (int i = 0; i < 4; i++) {
        auto &blk = voc_blocks_[i];
        int stride = config.upsample_rates[i];
        cur = build_snake_beta(ctx0, cur, blk.snake);
        cur = build_causal_transconv1d(ctx0, cur, blk.transconv_w,
                                        blk.transconv_b, stride);
        for (int j = 0; j < 3; j++) {
            auto &ru = blk.res_units[j];
            ggml_tensor *res = cur;
            cur = build_snake_beta(ctx0, cur, ru.act1);
            cur = build_causal_conv1d(ctx0, cur, ru.conv1_w, ru.conv1_b,
                                       dilations[j]);
            cur = build_snake_beta(ctx0, cur, ru.act2);
            cur = build_causal_conv1d(ctx0, cur, ru.conv2_w, ru.conv2_b);
            cur = ggml_add(ctx0, cur, res);
        }

        if (debug_stop_after_ == 4 + i) {
            cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
            ggml_set_name(cur, "audio_output");
            ggml_set_output(cur);
            return {cur};
        }
    }

    // 5c: Final SnakeBeta + causal conv → 1 channel + tanh
    cur = build_snake_beta(ctx0, cur, voc_final_snake_);
    cur = build_causal_conv1d(ctx0, cur, voc_final_conv_w_, voc_final_conv_b_);
    cur = ggml_tanh(ctx0, cur);

    // Flatten to 1D output
    cur = ggml_reshape_1d(ctx0, cur, ggml_nelements(cur));
    ggml_set_name(cur, "audio_output");
    ggml_set_output(cur);
    return {cur};
}

// ============================================================================
// High-level interface
// ============================================================================

bool SpeechTokenizerDecoder::load(const std::string &model_path,
                                   const ContextParams &params) {
    session_ = std::make_unique<InferenceSession<SpeechTokenizerDecoderModel>>(
        model_path, params);
    config_ = session_->get_model().config;
    printf("[decoder] model loaded, sample_rate=%d upsample_rate=%d\n",
           config_.output_sample_rate, config_.decode_upsample_rate);
    return true;
}

bool SpeechTokenizerDecoder::load(const std::string &model_path,
                                   const ContextParams &npu_params,
                                   const ContextParams &cpu_params) {
    // Primary: CANN (fast, but fails for >99 total frames due to CANN buffer limit)
    session_ = std::make_unique<InferenceSession<SpeechTokenizerDecoderModel>>(
        model_path, npu_params);
    config_ = session_->get_model().config;

    // CPU fallback for long sequences
    cpu_session_ = std::make_unique<InferenceSession<SpeechTokenizerDecoderModel>>(
        model_path, cpu_params);

    split_mode_ = true;
    printf("[decoder] CANN+CPU fallback mode (CANN limit: %d frames), sample_rate=%d\n",
           CANN_MAX_FRAMES, config_.output_sample_rate);
    return true;
}

bool SpeechTokenizerDecoder::decode(
    const std::vector<std::vector<int>> &codes,
    std::vector<float> &audio) {
    if (codes.empty() || codes[0].empty()) {
        printf("[decoder] empty codes input\n");
        return false;
    }

    int n_q = (int)codes.size();
    int T = (int)codes[0].size();

    if (n_q != config_.num_quantizers) {
        printf("[decoder] expected %d quantizers, got %d\n",
               config_.num_quantizers, n_q);
        return false;
    }

    // Flatten codes: [q0_t0, q0_t1, ..., q0_tT, q1_t0, ..., q15_tT]
    std::vector<int> flat_codes(n_q * T);
    for (int q = 0; q < n_q; q++) {
        for (int t = 0; t < T; t++) {
            flat_codes[q * T + t] = codes[q][t];
        }
    }

    if (split_mode_ && cpu_session_) {
        // Use chunked decoding for long sequences
        if (T > CANN_MAX_FRAMES) {
            return decode_chunked(codes, audio);
        }

        // Short sequence: CANN for fast decode (27x faster), CPU fallback if fails
        // CANN max_diff vs CPU ~0.25 (F16/F32 precision across 480x upsample chain)
        bool use_cann = true;
        const char *backend = "CANN";

        session_->set_input_shape({{"codes", {n_q * T}}});
        session_->set_input("codes", flat_codes);
        if (session_->run(audio)) {
            // Runtime validation: check for corrupted output (RMS ~1.0 = all clipped)
            float sum_sq = 0;
            int n_check = std::min((int)audio.size(), 4800); // check first 200ms
            for (int i = 0; i < n_check; i++) sum_sq += audio[i] * audio[i];
            float rms = sqrtf(sum_sq / std::max(n_check, 1));
            if (rms > 0.95f) {
                printf("[decoder] CANN output corrupt (RMS=%.2f), retrying on CPU\n", rms);
                use_cann = false;
            }
        } else {
            printf("[decoder] CANN failed, falling back to CPU\n");
            use_cann = false;
        }

        if (!use_cann) {
            cpu_session_->set_input_shape({{"codes", {n_q * T}}});
            cpu_session_->set_input("codes", flat_codes);
            if (!cpu_session_->run(audio)) {
                printf("[decoder] CPU inference failed\n");
                return false;
            }
            backend = "CPU";
        }

        printf("[decoder] decoded %d frames → %zu samples (%s)\n",
               T, audio.size(), backend);
        return true;
    }

    // === Single session mode (backward compat) ===
    // Update input shape (internally rebuilds graph)
    session_->set_input_shape({{"codes", {n_q * T}}});
    session_->set_input("codes", flat_codes);

    // Run inference
    if (!session_->run(audio)) {
        printf("[decoder] inference failed\n");
        return false;
    }

    printf("[decoder] decoded %d frames → %zu audio samples\n", T,
           audio.size());
    return true;
}

// ============================================================================
// Track L (M6.4) public chunk API — thin wrapper over decode_single_chunk
// with CANN→CPU fallback so a frame-streaming producer can decode incremental
// chunks without touching decoder internals.
// ============================================================================

bool SpeechTokenizerDecoder::forward_chunk(
    const std::vector<std::vector<int>> &chunk_codes,
    std::vector<float> &chunk_audio,
    bool prefer_cann) {
    if (chunk_codes.empty() || chunk_codes[0].empty()) {
        chunk_audio.clear();
        return true;  // empty chunk is not an error
    }
    int n_q = (int)chunk_codes.size();
    if (n_q != config_.num_quantizers) {
        printf("[decoder] forward_chunk: expected %d quantizers, got %d\n",
               config_.num_quantizers, n_q);
        return false;
    }

    // Try CANN first (fast path) if preferred and session exists.
    if (prefer_cann && session_) {
        if (decode_single_chunk(chunk_codes, chunk_audio, /*use_cann=*/true)) {
            return true;
        }
        // CANN failed — fall through to CPU if available.
        if (split_mode_ && cpu_session_) {
            printf("[decoder] forward_chunk: CANN failed, falling back to CPU\n");
        }
    }
    // CPU path (either requested directly, or CANN fallback).
    if (split_mode_ && cpu_session_) {
        return decode_single_chunk(chunk_codes, chunk_audio, /*use_cann=*/false);
    }
    // No CPU fallback available — single-session mode uses whatever session_ is.
    if (!prefer_cann && session_) {
        return decode_single_chunk(chunk_codes, chunk_audio, /*use_cann=*/true);
    }
    return false;
}

// ============================================================================
// Single chunk decoding helper
// ============================================================================

bool SpeechTokenizerDecoder::decode_single_chunk(
    const std::vector<std::vector<int>> &codes,
    std::vector<float> &audio,
    bool use_cann) {

    int n_q = (int)codes.size();
    int T = (int)codes[0].size();

    // Flatten codes
    std::vector<int> flat_codes(n_q * T);
    for (int q = 0; q < n_q; q++) {
        for (int t = 0; t < T; t++) {
            flat_codes[q * T + t] = codes[q][t];
        }
    }

    auto *sess = use_cann ? session_.get() : cpu_session_.get();
    sess->set_input_shape({{"codes", {n_q * T}}});
    sess->set_input("codes", flat_codes);

    if (!sess->run(audio)) {
        return false;
    }

    // Runtime validation for CANN output
    if (use_cann) {
        float sum_sq = 0;
        int n_check = std::min((int)audio.size(), 4800);
        for (int i = 0; i < n_check; i++) sum_sq += audio[i] * audio[i];
        float rms = sqrtf(sum_sq / std::max(n_check, 1));
        if (rms > 0.95f) {
            printf("[decoder] CANN output corrupt (RMS=%.2f), chunk failed\n", rms);
            return false;
        }
    }

    return true;
}

// ============================================================================
// Chunked decoding for long sequences (>99 frames)
// ============================================================================

bool SpeechTokenizerDecoder::decode_chunked(
    const std::vector<std::vector<int>> &codes,
    std::vector<float> &audio) {

    int n_q = (int)codes.size();
    int T = (int)codes[0].size();

    if (T <= CANN_MAX_FRAMES) {
        // Short sequence: use single-shot decoding
        return decode_single_chunk(codes, audio, true);
    }

    // Long sequence: split into chunks with overlap for context.
    // Each chunk provides full sliding_window context for its interior frames.
    // Stitching: keep only the NEW frames from each chunk (hard-cut, no crossfade).
    int step = CHUNK_SIZE - OVERLAP_FRAMES;
    printf("[decoder] chunked mode: %d frames, chunk=%d, overlap=%d, step=%d\n",
           T, CHUNK_SIZE, OVERLAP_FRAMES, step);

    audio.clear();
    int upsample = config_.decode_upsample_rate;
    int chunk_count = 0;

    for (int start = 0; start < T; start += step) {
        int end = std::min(start + CHUNK_SIZE, T);

        // Extract chunk codes
        std::vector<std::vector<int>> chunk_codes(n_q);
        for (int q = 0; q < n_q; q++) {
            chunk_codes[q].assign(
                codes[q].begin() + start,
                codes[q].begin() + end
            );
        }

        // Decode chunk with CANN
        std::vector<float> chunk_audio;
        bool success = decode_single_chunk(chunk_codes, chunk_audio, true);

        if (!success) {
            printf("[decoder] CANN chunk %d failed, falling back to CPU\n", chunk_count);
            return decode_single_chunk(codes, audio, false);
        }

        // Hard-cut stitching: only keep the frames that are "new" in this chunk.
        // First chunk: keep all. Subsequent: skip overlap prefix.
        int skip_frames = (start > 0) ? OVERLAP_FRAMES : 0;
        int skip_samples = skip_frames * upsample;
        int keep_samples = (int)chunk_audio.size() - skip_samples;

        if (keep_samples <= 0) break;  // no new frames in this chunk

        {
            audio.insert(audio.end(),
                         chunk_audio.begin() + skip_samples,
                         chunk_audio.end());
        }

        chunk_count++;
        printf("[decoder]   chunk %d: frames [%d,%d), keep [%d,%d) -> %d samples (CANN)\n",
               chunk_count, start, end, start + skip_frames, end, keep_samples);
    }

    printf("[decoder] chunked decode complete: %d chunks → %zu samples\n",
           chunk_count, audio.size());
    return true;
}
