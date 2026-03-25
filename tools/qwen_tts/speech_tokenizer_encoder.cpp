#include "speech_tokenizer_encoder.h"
#include "ggml.h"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>

// ============================================================================
// SpeechTokenizerEncoderModel: load_hparams
// ============================================================================

bool SpeechTokenizerEncoderModel::load_hparams(const ModelLoader &loader) {
    printf("[encoder] loading hparams...\n");

    loader.get_u32("encoder.num_filters",          config.num_filters);
    loader.get_u32("encoder.hidden_size",           config.hidden_size);
    loader.get_u32("encoder.num_hidden_layers",     config.num_hidden_layers);
    loader.get_u32("encoder.num_attention_heads",   config.num_attention_heads);
    loader.get_u32("encoder.num_key_value_heads",   config.num_key_value_heads);
    loader.get_u32("encoder.codebook_size",         config.codebook_size);
    loader.get_u32("encoder.codebook_dim",          config.codebook_dim);
    loader.get_u32("encoder.num_quantizers",        config.num_quantizers);
    loader.get_u32("encoder_valid_num_quantizers",  config.valid_num_quantizers);
    loader.get_u32("input_sample_rate",             config.input_sample_rate);
    loader.get_u32("encode_downsample_rate",        config.encode_downsample_rate);

    config.head_dim = config.hidden_size / config.num_attention_heads;
    config.intermediate_size = config.hidden_size * 4;  // 512 * 4 = 2048

    printf("[encoder] hidden=%d layers=%d heads=%d/%d codebook=%dx%d "
           "quantizers=%d(valid=%d) downsample=%d\n",
           config.hidden_size, config.num_hidden_layers,
           config.num_attention_heads, config.num_key_value_heads,
           config.codebook_size, config.codebook_dim,
           config.num_quantizers, config.valid_num_quantizers,
           config.encode_downsample_rate);
    return true;
}

void SpeechTokenizerEncoderModel::reset_input_shape() {
    input_shapes_ = {
        {"audio", {24000}},  // default 1 second of audio
    };
}

// ============================================================================
// SpeechTokenizerEncoderModel: get_tensors_to_load
// ============================================================================

std::vector<ggml_tensor *>
SpeechTokenizerEncoderModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    // Helper: look up tensor by name in the ctx_meta populated by gguf_init_from_file
    auto gt = [&](const std::string &name) -> ggml_tensor * {
        return get_tensor(ctx, name, tensors);
    };

    // ---- Conv encoder layers ----
    // layers.0: initial Conv1d(1, 64, k=7)
    init_conv_w_ = gt("encoder.layers.0.conv.weight");
    init_conv_b_ = gt("encoder.layers.0.conv.bias");

    // 4 downsample blocks: ResNet layers [1,4,7,10], Downsample conv layers [3,6,9,12]
    int resnet_layer_ids[4] = {1, 4, 7, 10};
    int ds_conv_layer_ids[4] = {3, 6, 9, 12};

    for (int i = 0; i < 4; i++) {
        auto &blk = down_blocks_[i];
        int rl = resnet_layer_ids[i];
        int dl = ds_conv_layer_ids[i];
        char name[128];

        snprintf(name, sizeof(name), "encoder.layers.%d.block.1.conv.weight", rl);
        blk.resnet.conv1_w = gt(name);
        snprintf(name, sizeof(name), "encoder.layers.%d.block.1.conv.bias", rl);
        blk.resnet.conv1_b = gt(name);

        snprintf(name, sizeof(name), "encoder.layers.%d.block.3.conv.weight", rl);
        blk.resnet.conv2_w = gt(name);
        snprintf(name, sizeof(name), "encoder.layers.%d.block.3.conv.bias", rl);
        blk.resnet.conv2_b = gt(name);

        snprintf(name, sizeof(name), "encoder.layers.%d.conv.weight", dl);
        blk.ds_conv_w = gt(name);
        snprintf(name, sizeof(name), "encoder.layers.%d.conv.bias", dl);
        blk.ds_conv_b = gt(name);
    }

    // layers.14: final Conv1d(1024, 512, k=3)
    final_conv_w_ = gt("encoder.layers.14.conv.weight");
    final_conv_b_ = gt("encoder.layers.14.conv.bias");

    // ---- Encoder transformer (8 layers) ----
    tf_layers_.resize(config.num_hidden_layers);
    for (int il = 0; il < config.num_hidden_layers; il++) {
        auto &layer = tf_layers_[il];
        char name[128];

        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.self_attn.q_proj.weight", il);
        layer.q_proj_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.self_attn.k_proj.weight", il);
        layer.k_proj_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.self_attn.v_proj.weight", il);
        layer.v_proj_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.self_attn.o_proj.weight", il);
        layer.o_proj_w = gt(name);

        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.mlp.fc1.weight", il);
        layer.fc1_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.mlp.fc2.weight", il);
        layer.fc2_w = gt(name);

        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.input_layernorm.weight", il);
        layer.input_layernorm_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.input_layernorm.bias", il);
        layer.input_layernorm_b = gt(name);

        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.post_attention_layernorm.weight", il);
        layer.post_attn_layernorm_w = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.post_attention_layernorm.bias", il);
        layer.post_attn_layernorm_b = gt(name);

        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.self_attn_layer_scale.scale", il);
        layer.self_attn_layer_scale = gt(name);
        snprintf(name, sizeof(name), "encoder_transformer.layers.%d.mlp_layer_scale.scale", il);
        layer.mlp_layer_scale = gt(name);
    }

    // ---- Post-transformer downsample ----
    downsample_conv_w_ = gt("downsample.conv.weight");

    // ---- RVQ quantizer ----
    // Semantic RVQ: 1 codebook
    rvq_semantic_.input_proj_w = gt("quantizer.semantic_residual_vector_quantizer.input_proj.weight");
    rvq_semantic_.output_proj_w = gt("quantizer.semantic_residual_vector_quantizer.output_proj.weight");
    rvq_semantic_.codebooks.resize(1);
    {
        auto &cb = rvq_semantic_.codebooks[0];
        cb.embed_sum = gt("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.embed_sum");
        cb.cluster_usage = gt("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.cluster_usage");
        gt("quantizer.semantic_residual_vector_quantizer.layers.0.codebook.initialized");
    }

    // Acoustic RVQ: num_quantizers-1 codebooks (31 in GGUF)
    int n_acoustic = config.num_quantizers - 1;
    rvq_acoustic_.input_proj_w = gt("quantizer.acoustic_residual_vector_quantizer.input_proj.weight");
    rvq_acoustic_.output_proj_w = gt("quantizer.acoustic_residual_vector_quantizer.output_proj.weight");
    rvq_acoustic_.codebooks.resize(n_acoustic);
    for (int i = 0; i < n_acoustic; i++) {
        auto &cb = rvq_acoustic_.codebooks[i];
        char name[128];

        snprintf(name, sizeof(name), "quantizer.acoustic_residual_vector_quantizer.layers.%d.codebook.embed_sum", i);
        cb.embed_sum = gt(name);
        snprintf(name, sizeof(name), "quantizer.acoustic_residual_vector_quantizer.layers.%d.codebook.cluster_usage", i);
        cb.cluster_usage = gt(name);
        snprintf(name, sizeof(name), "quantizer.acoustic_residual_vector_quantizer.layers.%d.codebook.initialized", i);
        gt(name);
    }

    printf("[encoder] prepared %zu tensors to load\n", tensors.size());
    return tensors;
}

// ============================================================================
// Graph building: causal Conv1d
// ============================================================================

ggml_tensor *SpeechTokenizerEncoderModel::build_causal_conv1d(
    ggml_context *ctx0, ggml_tensor *x, ggml_tensor *w, ggml_tensor *b,
    int stride, int dilation) {
    int kernel_size = (int)w->ne[0];
    // Effective kernel size with dilation
    int kernel_size_eff = (kernel_size - 1) * dilation + 1;
    // Causal padding: pad_left = kernel_size_eff - stride (matches Qwen3 CausalConvNet)
    int pad_left = kernel_size_eff - stride;

    // Compute extra right-padding to ensure output length = ceil(input_len / stride)
    int input_len = (int)x->ne[0];
    int n_frames_numer = input_len - kernel_size_eff + pad_left;
    int n_frames = n_frames_numer / stride + 1;
    // Round up: if n_frames_numer is not divisible by stride, we need one more frame
    if (n_frames_numer % stride != 0) n_frames++;
    int ideal_length = (n_frames - 1) * stride + kernel_size_eff - pad_left;
    int pad_right = ideal_length - input_len;
    if (pad_right < 0) pad_right = 0;

    if (pad_left > 0 || pad_right > 0) {
        x = ggml_pad_ext(ctx0, x, pad_left, pad_right, 0, 0, 0, 0, 0, 0);
    }
    return build_conv1d(ctx0, x, w, b, stride, 0, dilation);
}

// ============================================================================
// Graph building: ResNet block
// ============================================================================

ggml_tensor *SpeechTokenizerEncoderModel::build_resnet_block(
    ggml_context *ctx0, ggml_tensor *x, const EncoderResNetBlock &blk,
    int dilation) {
    ggml_tensor *residual = x;

    // ELU → Conv1d(ch→ch/2, k=3, dilation)
    ggml_tensor *cur = ggml_elu(ctx0, x);
    cur = build_causal_conv1d(ctx0, cur, blk.conv1_w, blk.conv1_b, 1, dilation);

    // ELU → Conv1d(ch/2→ch, k=1)
    cur = ggml_elu(ctx0, cur);
    cur = build_causal_conv1d(ctx0, cur, blk.conv2_w, blk.conv2_b, 1, 1);

    // Residual connection (identity shortcut - no conv_shortcut in Mimi encoder)
    return ggml_add(ctx0, cur, residual);
}

// ============================================================================
// Graph building: Conv encoder (downsample 960×)
// ============================================================================

ggml_tensor *SpeechTokenizerEncoderModel::build_conv_encoder(
    ggml_context *ctx0, ggml_tensor *x) {
    // x: [seq_len, 1, 1]

    // Initial conv: Conv1d(1→64, k=7)
    ggml_tensor *cur = build_causal_conv1d(ctx0, x, init_conv_w_, init_conv_b_);
    // cur: [seq_len, 64, 1]

    int dilations[4] = {1, 1, 1, 1};

    // 4 downsample blocks
    for (int i = 0; i < 4; i++) {
        auto &blk = down_blocks_[i];
        int stride = config.downsample_strides[i];

        // ResNet block
        cur = build_resnet_block(ctx0, cur, blk.resnet, dilations[i]);

        // ELU + downsample Conv1d
        cur = ggml_elu(ctx0, cur);
        cur = build_causal_conv1d(ctx0, cur, blk.ds_conv_w, blk.ds_conv_b, stride, 1);
    }
    // cur: [seq_len/960, 1024, 1]

    // Final: ELU + Conv1d(1024→512, k=3)
    cur = ggml_elu(ctx0, cur);
    cur = build_causal_conv1d(ctx0, cur, final_conv_w_, final_conv_b_);
    // cur: [seq_len/960, 512, 1]

    return cur;
}

// ============================================================================
// Graph building: Encoder transformer (8 layers)
// ============================================================================

ggml_tensor *SpeechTokenizerEncoderModel::build_transformer(
    ggml_context *ctx0, ggml_tensor *x) {
    // x: [seq_len, hidden(512), 1] from conv encoder
    int seq_len = (int)x->ne[0];
    int hidden = config.hidden_size;

    // Reshape to [hidden, seq_len] for linear ops
    ggml_tensor *cur = ggml_reshape_2d(ctx0, x, seq_len, hidden);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // cur: [hidden, seq_len]

    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    int head_dim = config.head_dim;
    float kq_scale = 1.0f / sqrtf((float)head_dim);

    // Causal mask for attention (SDPA uses is_causal=True when no explicit mask)
    // mask shape: [seq_len, seq_len], -inf for future positions
    ggml_tensor *kq_mask =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // Position ids for RoPE
    ggml_tensor *pos = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    // Transformer layers
    for (int il = 0; il < config.num_hidden_layers; il++) {
        auto &layer = tf_layers_[il];
        ggml_tensor *residual = cur;

        // Pre-norm: LayerNorm (with bias)
        cur = build_norm(ctx0, cur, layer.input_layernorm_w,
                         layer.input_layernorm_b, NORM_TYPE_NORMAL,
                         config.norm_eps, 1, il);

        // QKV projections
        ggml_tensor *q = ggml_mul_mat(ctx0, layer.q_proj_w, cur);
        ggml_tensor *k = ggml_mul_mat(ctx0, layer.k_proj_w, cur);
        ggml_tensor *v = ggml_mul_mat(ctx0, layer.v_proj_w, cur);
        q = ggml_reshape_3d(ctx0, q, head_dim, n_heads, seq_len);
        k = ggml_reshape_3d(ctx0, k, head_dim, n_kv_heads, seq_len);
        v = ggml_reshape_3d(ctx0, v, head_dim, n_kv_heads, seq_len);

        // RoPE (NeoX-style split-half rotation)
        q = ggml_rope_ext(ctx0, q, pos, nullptr, head_dim,
                           GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                           1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx0, k, pos, nullptr, head_dim,
                           GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                           1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // Self-attention with causal mask (always applied)
        ggml_tensor *attn_out = build_attn(ctx0, layer.o_proj_w, nullptr,
                                            q, k, v, kq_mask, kq_scale, il);

        // LayerScale + residual
        attn_out = ggml_mul(ctx0, attn_out, layer.self_attn_layer_scale);
        cur = ggml_add(ctx0, residual, attn_out);

        // MLP block
        residual = cur;
        cur = build_norm(ctx0, cur, layer.post_attn_layernorm_w,
                         layer.post_attn_layernorm_b, NORM_TYPE_NORMAL,
                         config.norm_eps, 1, il);

        // fc1 → GELU(erf) → fc2
        ggml_tensor *ffn_out = build_ffn(ctx0, cur,
                                          layer.fc1_w, nullptr,
                                          nullptr, nullptr,
                                          layer.fc2_w, nullptr,
                                          FFN_GELU_ERF, il);

        // LayerScale + residual
        ffn_out = ggml_mul(ctx0, ffn_out, layer.mlp_layer_scale);
        cur = ggml_add(ctx0, residual, ffn_out);

    }

    // No final norm in encoder transformer

    // Transpose back to [seq_len, hidden, 1] for conv
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    cur = ggml_reshape_3d(ctx0, cur, seq_len, hidden, 1);

    return cur;
}

// ============================================================================
// Graph building: RVQ encode (nearest neighbor quantization)
// ============================================================================

// NOTE: build_rvq_encode is not used - RVQ quantization is done on CPU
// in SpeechTokenizerEncoder::encode() because it requires iterative
// residual subtraction which can't be expressed as a single ggml graph.

// ============================================================================
// Main build_graph
// ============================================================================

std::vector<ggml_tensor *>
SpeechTokenizerEncoderModel::build_graph(ggml_context *ctx0) {
    std::vector<int> audio_shape = input_shapes_["audio"];
    int audio_len = audio_shape[0];

    // Input: raw audio samples [audio_len]
    ggml_tensor *audio =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_F32, audio_len);
    ggml_set_name(audio, "audio");
    ggml_set_input(audio);

    // Reshape to [audio_len, 1, 1] for Conv1d
    ggml_tensor *cur = ggml_reshape_3d(ctx0, audio, audio_len, 1, 1);

    // Step 1: Conv encoder (÷960)
    cur = build_conv_encoder(ctx0, cur);
    // cur: [T, 512, 1] where T = audio_len/960

    // Step 2: Transformer (8 layers)
    cur = build_transformer(ctx0, cur);
    // cur: [T, 512, 1]

    // Step 3: Post-transformer downsample Conv1d(512→512, k=4, s=2)
    cur = build_causal_conv1d(ctx0, cur, downsample_conv_w_, nullptr, 2, 1);
    // cur: [T/2, 512, 1] where T/2 = audio_len/1920

    // Reshape to [hidden, seq_len] for output
    int out_len = (int)cur->ne[0];
    cur = ggml_reshape_2d(ctx0, cur, out_len, config.hidden_size);
    cur = ggml_cont(ctx0, ggml_permute(ctx0, cur, 1, 0, 2, 3));
    // cur: [hidden(512), out_len]

    ggml_set_name(cur, "encoder_output");
    ggml_set_output(cur);
    return {cur};
}

// ============================================================================
// High-level interface
// ============================================================================

// Download tensor data from backend to CPU float vector
static void download_tensor_f32(ggml_tensor *t, std::vector<float> &out) {
    int n = (int)ggml_nelements(t);
    if (t->type == GGML_TYPE_F32) {
        out.resize(n);
        ggml_backend_tensor_get(t, out.data(), 0, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        std::vector<ggml_fp16_t> tmp(n);
        ggml_backend_tensor_get(t, tmp.data(), 0, n * sizeof(ggml_fp16_t));
        out.resize(n);
        for (int i = 0; i < n; i++) {
            out[i] = ggml_fp16_to_fp32(tmp[i]);
        }
    } else {
        printf("[encoder] unsupported tensor type for download: %d\n", t->type);
        out.resize(n, 0.0f);
    }
}

bool SpeechTokenizerEncoder::load(const std::string &model_path,
                                   const ContextParams &params) {
    session_ = std::make_unique<InferenceSession<SpeechTokenizerEncoderModel>>(
        model_path, params);
    config_ = session_->get_model().config;

    // Pre-download and normalize codebook weights for CPU-side RVQ
    prepare_codebooks();

    printf("[encoder] model loaded, sample_rate=%d downsample_rate=%d "
           "quantizers=%d\n",
           config_.input_sample_rate, config_.encode_downsample_rate,
           config_.valid_num_quantizers);
    return true;
}

void SpeechTokenizerEncoder::prepare_codebooks() {
    auto &model = session_->get_model();
    int cb_dim = config_.codebook_dim;
    int cb_size = config_.codebook_size;
    int n_q = config_.valid_num_quantizers;

    // Download projection weights (Conv1d k=1 → effectively linear)
    download_tensor_f32(model.rvq_semantic_.input_proj_w, sem_input_proj_);
    download_tensor_f32(model.rvq_semantic_.output_proj_w, sem_output_proj_);
    download_tensor_f32(model.rvq_acoustic_.input_proj_w, acou_input_proj_);
    download_tensor_f32(model.rvq_acoustic_.output_proj_w, acou_output_proj_);

    // Download and normalize codebooks: embed = embed_sum / cluster_usage
    auto normalize_codebook = [&](EncoderRVQCodebook &cb,
                                   std::vector<float> &out) {
        std::vector<float> embed_sum, usage;
        download_tensor_f32(cb.embed_sum, embed_sum);
        download_tensor_f32(cb.cluster_usage, usage);
        out.resize(cb_size * cb_dim);
        for (int k = 0; k < cb_size; k++) {
            float u = usage[k];
            if (u < 1e-7f) u = 1.0f;
            for (int d = 0; d < cb_dim; d++) {
                // embed_sum: [dim, size] col-major → data[d + k * dim]
                out[k * cb_dim + d] = embed_sum[d + k * cb_dim] / u;
            }
        }
    };

    // Semantic codebook (1)
    codebooks_.resize(n_q);
    normalize_codebook(model.rvq_semantic_.codebooks[0], codebooks_[0]);

    // Acoustic codebooks (1..n_q-1)
    for (int q = 0; q < n_q - 1; q++) {
        normalize_codebook(model.rvq_acoustic_.codebooks[q], codebooks_[q + 1]);
    }

    printf("[encoder] prepared %d codebooks (%dx%d)\n", n_q, cb_size, cb_dim);

    // Debug: dump first codebook for alignment verification
    {
        FILE *f = fopen("logs/cpp_sem_codebook.bin", "wb");
        if (f) {
            fwrite(&cb_size, 4, 1, f);
            fwrite(&cb_dim, 4, 1, f);
            fwrite(codebooks_[0].data(), sizeof(float), cb_size * cb_dim, f);
            fclose(f);
            printf("[encoder] dumped semantic codebook: %dx%d\n", cb_size, cb_dim);
            printf("[encoder] cb[0][:5]: %.6f %.6f %.6f %.6f %.6f\n",
                   codebooks_[0][0], codebooks_[0][1], codebooks_[0][2],
                   codebooks_[0][3], codebooks_[0][4]);
            printf("[encoder] cb[2047][:5]: %.6f %.6f %.6f %.6f %.6f\n",
                   codebooks_[0][2047*cb_dim+0], codebooks_[0][2047*cb_dim+1],
                   codebooks_[0][2047*cb_dim+2], codebooks_[0][2047*cb_dim+3],
                   codebooks_[0][2047*cb_dim+4]);
        }
        // Also dump input projection
        f = fopen("logs/cpp_sem_input_proj.bin", "wb");
        if (f) {
            int in_ch = config_.hidden_size, out_ch = cb_dim;
            fwrite(&in_ch, 4, 1, f);
            fwrite(&out_ch, 4, 1, f);
            fwrite(sem_input_proj_.data(), sizeof(float), in_ch * out_ch, f);
            fclose(f);
            printf("[encoder] dumped sem_input_proj: %dx%d\n", in_ch, out_ch);
        }
    }
}

bool SpeechTokenizerEncoder::encode(
    const std::vector<float> &audio,
    std::vector<std::vector<int>> &codes,
    std::vector<float> *hidden_out_ptr) {

    if (audio.empty()) {
        printf("[encoder] empty audio input\n");
        return false;
    }

    int audio_len = (int)audio.size();
    int n_q = config_.valid_num_quantizers;  // 16
    int hidden = config_.hidden_size;
    int cb_dim = config_.codebook_dim;
    int cb_size = config_.codebook_size;

    // Step 1: Run conv encoder + transformer + downsample through ggml
    // set_input_shape triggers alloc_compute_meta which rebuilds and allocates the graph
    // Do NOT call alloc_graph() again — it would double-build, creating duplicate
    // tensor objects where set_input populates the old ones while compute uses the new ones
    session_->set_input_shape({{"audio", {audio_len}}});
    session_->set_input("audio", audio);

    // Compute the transformer seq_len (after conv encoder ÷960, before downsample)
    // Conv encoder: audio_len → T_conv = ceil-ish of audio_len / 960
    // The exact value depends on causal padding; we calculate from the graph
    // For the mask/pos, we need the seq_len used inside build_transformer
    // Since build_graph computes it dynamically, we estimate:
    int T_conv = audio_len;
    for (int i = 0; i < 4; i++) {
        int stride = config_.downsample_strides[i];
        // causal conv with stride: output = ceil(input / stride)
        // but with left padding, it's: (input + pad_left) / stride
        // Actually for causal conv with stride s, kernel k:
        // pad_left = k - 1, output_len = (input + pad_left - k) / s + 1 = input / s
        // Wait, more precisely: floor((input + pad - k) / s) + 1
        // With pad_left = (k-1)*d and d=1: floor((input + k - 1 - k) / s) + 1 = floor((input-1)/s) + 1
        T_conv = (T_conv - 1) / stride + 1;
    }
    // After final conv (k=3, s=1): T stays same
    int tf_seq_len = T_conv;

    // Fill causal mask (SDPA uses is_causal=True in the Python model)
    // Also apply sliding window for long sequences
    // ggml soft_max_ext expects mask[j, i] where j=key_pos, i=query_pos
    // Convention: 0.0 = attend, -inf = mask out
    int sw = config_.sliding_window;
    {
        std::vector<float> mask(tf_seq_len * tf_seq_len);
        for (int i = 0; i < tf_seq_len; i++) {       // query position
            for (int j = 0; j < tf_seq_len; j++) {   // key position
                bool causal_ok = (j <= i);            // key must be at or before query
                bool sw_ok = (abs(i - j) <= sw);      // within sliding window
                mask[j + i * tf_seq_len] = (causal_ok && sw_ok) ? 0.0f : -INFINITY;
            }
        }
        session_->set_input("kq_mask", mask);
    }

    // Fill position IDs for RoPE
    std::vector<int32_t> pos(tf_seq_len);
    for (int i = 0; i < tf_seq_len; i++) pos[i] = i;
    session_->set_input("pos", pos);

    // Run inference (single output: encoder_output [hidden, T])
    std::vector<float> hidden_out;
    if (!session_->run(hidden_out)) {
        printf("[encoder] inference failed\n");
        return false;
    }

    int T = (int)hidden_out.size() / hidden;
    printf("[encoder] output: hidden=%d, T=%d\n", hidden, T);

    // Optionally return hidden states for verification
    if (hidden_out_ptr) {
        *hidden_out_ptr = hidden_out;
    }

    // Step 2: RVQ quantization on CPU
    // The ggml output tensor has ne[0]=hidden(512), ne[1]=T (after final permute+cont)
    // Data layout: hidden_out[h + t * hidden] — already row-major [T, hidden]
    // No transpose needed.

    codes.resize(n_q);
    for (int q = 0; q < n_q; q++) codes[q].resize(T);

    // Working buffers — hidden_out is already [T, hidden] row-major
    std::vector<float> residual = hidden_out;

    std::vector<float> projected(T * cb_dim);
    std::vector<float> quantized_cb(T * cb_dim);
    std::vector<float> quantized_hidden(T * hidden);

    // Matmul helper: out[t,o] = sum_i(W[i,o] * in[t,i])
    // Conv1d weight [1, in_ch, out_ch] in ggml: ne[0]=1, ne[1]=in_ch, ne[2]=out_ch
    // Flat index for element (k=0, i, o) = i + o * in_ch
    auto matmul = [](const float *W, const float *in, float *out,
                     int T, int in_ch, int out_ch) {
        for (int t = 0; t < T; t++) {
            for (int o = 0; o < out_ch; o++) {
                float sum = 0.0f;
                for (int i = 0; i < in_ch; i++) {
                    sum += W[i + o * in_ch] * in[t * in_ch + i];
                }
                out[t * out_ch + o] = sum;
            }
        }
    };

    // Nearest neighbor search
    auto nearest_neighbor = [](const float *query, const float *codebook,
                                int dim, int size) -> int {
        int best = 0;
        float best_dist = 1e30f;
        for (int k = 0; k < size; k++) {
            float dist = 0.0f;
            const float *cb_k = codebook + k * dim;
            for (int d = 0; d < dim; d++) {
                float diff = query[d] - cb_k[d];
                dist += diff * diff;
            }
            if (dist < best_dist) {
                best_dist = dist;
                best = k;
            }
        }
        return best;
    };

    // Quantize: semantic (q=0)
    {
        matmul(sem_input_proj_.data(), residual.data(), projected.data(),
               T, hidden, cb_dim);

        // Debug: dump projected vectors for comparison
        {
            FILE *f = fopen("logs/cpp_sem_projected.bin", "wb");
            if (f) {
                fwrite(&T, 4, 1, f);
                fwrite(&cb_dim, 4, 1, f);
                fwrite(projected.data(), sizeof(float), T * cb_dim, f);
                fclose(f);
                printf("[encoder] dumped projected (%dx%d), first 5: %.4f %.4f %.4f %.4f %.4f\n",
                       T, cb_dim, projected[0], projected[1], projected[2], projected[3], projected[4]);
            }
        }

        const float *cb = codebooks_[0].data();
        for (int t = 0; t < T; t++) {
            int idx = nearest_neighbor(projected.data() + t * cb_dim, cb, cb_dim, cb_size);
            codes[0][t] = idx;
            for (int d = 0; d < cb_dim; d++)
                quantized_cb[t * cb_dim + d] = cb[idx * cb_dim + d];
        }

        matmul(sem_output_proj_.data(), quantized_cb.data(),
               quantized_hidden.data(), T, cb_dim, hidden);

        for (int i = 0; i < hidden * T; i++)
            residual[i] -= quantized_hidden[i];
    }

    // Quantize: acoustic (q=1..15)
    // Acoustic RVQ works entirely in codebook dimension (256):
    // 1. Project residual from hidden (512) to cb_dim (256) ONCE
    // 2. Iteratively quantize and subtract in cb_dim space
    // (No output_proj round-trip per layer — unlike semantic)
    {
        // Project residual to cb_dim space
        std::vector<float> acou_residual(T * cb_dim);
        matmul(acou_input_proj_.data(), residual.data(), acou_residual.data(),
               T, hidden, cb_dim);

        for (int q = 0; q < n_q - 1; q++) {
            const float *cb = codebooks_[q + 1].data();
            for (int t = 0; t < T; t++) {
                int idx = nearest_neighbor(acou_residual.data() + t * cb_dim,
                                           cb, cb_dim, cb_size);
                codes[q + 1][t] = idx;
                // Subtract quantized vector from residual in cb_dim space
                const float *cb_entry = cb + idx * cb_dim;
                float *res_t = acou_residual.data() + t * cb_dim;
                for (int d = 0; d < cb_dim; d++)
                    res_t[d] -= cb_entry[d];
            }
        }
    }

    printf("[encoder] encoded %d frames, %d quantizers\n", T, n_q);
    return true;
}
