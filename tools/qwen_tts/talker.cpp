#include "talker.h"
#include "ggml.h"
#ifdef OMINIX_HAS_CANN
#include "ggml-cann.h"
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#ifdef __aarch64__
#include <arm_neon.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

// Thread count for CP parallel matvec (set at model load)
static int cp_n_threads = 1;

// ============================================================================
// Sampling utilities
// ============================================================================

static std::mt19937 &get_rng() {
    static std::mt19937 rng(42);
    return rng;
}

void set_sampling_seed(uint32_t seed) {
    get_rng().seed(seed);
}

// Apply repetition penalty to logits for previously generated tokens
static void apply_repetition_penalty(float *logits, int vocab_size,
                                      const std::vector<int> &prev_tokens,
                                      float penalty) {
    if (penalty == 1.0f) return;
    for (int tok : prev_tokens) {
        if (tok < 0 || tok >= vocab_size) continue;
        if (logits[tok] > 0) {
            logits[tok] /= penalty;
        } else {
            logits[tok] *= penalty;
        }
    }
}

// Suppress special tokens (>= 2048) except EOS
static void suppress_special_tokens(float *logits, int vocab_size, int eos_id) {
    for (int i = 2048; i < vocab_size; i++) {
        if (i != eos_id) {
            logits[i] = -INFINITY;
        }
    }
}

// Sample from logits with temperature, top-k, top-p
static int sample_token(float *logits, int vocab_size,
                         float temperature, int top_k, float top_p,
                         bool do_sample) {
    if (!do_sample || temperature <= 0.0f) {
        // Greedy
        return (int)(std::max_element(logits, logits + vocab_size) - logits);
    }

    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }

    // Build (token, logit) pairs for sorting
    std::vector<std::pair<float, int>> candidates;
    candidates.reserve(vocab_size);
    for (int i = 0; i < vocab_size; i++) {
        if (logits[i] > -1e9f) {
            candidates.push_back({logits[i], i});
        }
    }

    // Sort descending by logit
    std::sort(candidates.begin(), candidates.end(),
              [](auto &a, auto &b) { return a.first > b.first; });

    // Top-K filter
    if (top_k > 0 && top_k < (int)candidates.size()) {
        candidates.resize(top_k);
    }

    // Softmax
    float max_logit = candidates[0].first;
    float sum = 0.0f;
    for (auto &c : candidates) {
        c.first = expf(c.first - max_logit);
        sum += c.first;
    }
    for (auto &c : candidates) {
        c.first /= sum;
    }

    // Top-P (nucleus) filter
    if (top_p < 1.0f) {
        float cumsum = 0.0f;
        int cutoff = (int)candidates.size();
        for (int i = 0; i < (int)candidates.size(); i++) {
            cumsum += candidates[i].first;
            if (cumsum >= top_p) {
                cutoff = i + 1;
                break;
            }
        }
        candidates.resize(cutoff);
        // Renormalize
        sum = 0.0f;
        for (auto &c : candidates) sum += c.first;
        for (auto &c : candidates) c.first /= sum;
    }

    // Sample from distribution
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(get_rng());
    float cumsum = 0.0f;
    for (auto &c : candidates) {
        cumsum += c.first;
        if (r <= cumsum) return c.second;
    }
    return candidates.back().second;
}

// ============================================================================
// CodePredictorModel: load_hparams
// ============================================================================

bool CodePredictorModel::load_hparams(const ModelLoader &loader) {
    loader.get_u32("hidden_size", config.hidden_size);
    loader.get_u32("num_hidden_layers", config.num_hidden_layers);
    loader.get_u32("num_attention_heads", config.num_attention_heads);
    loader.get_u32("num_key_value_heads", config.num_key_value_heads);
    loader.get_u32("intermediate_size", config.intermediate_size);
    loader.get_u32("head_dim", config.head_dim);
    loader.get_u32("vocab_size", config.vocab_size);
    loader.get_u32("num_code_groups", config.num_code_groups);
    loader.get_u32("talker_hidden_size", config.talker_hidden_size);
    loader.get_f32("rope_theta", config.rope_theta);
    loader.get_f32("rms_norm_eps", config.rms_norm_eps);

    printf("[code_predictor] hidden=%d layers=%d heads=%d/%d vocab=%d "
           "groups=%d talker_hidden=%d\n",
           config.hidden_size, config.num_hidden_layers,
           config.num_attention_heads, config.num_key_value_heads,
           config.vocab_size, config.num_code_groups,
           config.talker_hidden_size);
    return true;
}

void CodePredictorModel::reset_input_shape() {
    input_shapes_ = {
        {"inputs_embeds", {config.talker_hidden_size, 2}},  // [talker_hidden, seq_len]
    };
}

// ============================================================================
// CodePredictorModel: get_tensors_to_load (88 tensors)
// ============================================================================

std::vector<ggml_tensor *>
CodePredictorModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    // Projection from Talker hidden to CP hidden
    small_to_mtp_proj_w_ =
        get_tensor(ctx, "small_to_mtp_projection.weight", tensors);
    small_to_mtp_proj_b_ =
        get_tensor(ctx, "small_to_mtp_projection.bias", tensors);

    // Transformer layers
    layers_.resize(config.num_hidden_layers);
    for (int i = 0; i < config.num_hidden_layers; i++) {
        std::string lp = "model.layers." + std::to_string(i) + ".";
        auto &l = layers_[i];
        l.q_proj_w =
            get_tensor(ctx, (lp + "self_attn.q_proj.weight").c_str(), tensors);
        l.k_proj_w =
            get_tensor(ctx, (lp + "self_attn.k_proj.weight").c_str(), tensors);
        l.v_proj_w =
            get_tensor(ctx, (lp + "self_attn.v_proj.weight").c_str(), tensors);
        l.o_proj_w =
            get_tensor(ctx, (lp + "self_attn.o_proj.weight").c_str(), tensors);
        l.q_norm_w =
            get_tensor(ctx, (lp + "self_attn.q_norm.weight").c_str(), tensors);
        l.k_norm_w =
            get_tensor(ctx, (lp + "self_attn.k_norm.weight").c_str(), tensors);
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
    }

    // Final norm
    norm_w_ = get_tensor(ctx, "model.norm.weight", tensors);

    // Per-group codec embeddings and LM heads (groups 1-15)
    int n_groups = config.num_code_groups - 1;  // 15
    codec_embeddings_.resize(n_groups);
    lm_heads_.resize(n_groups);
    for (int i = 0; i < n_groups; i++) {
        std::string idx = std::to_string(i);
        codec_embeddings_[i] = get_tensor(
            ctx, ("model.codec_embedding." + idx + ".weight").c_str(), tensors);
        lm_heads_[i] = get_tensor(
            ctx, ("lm_head." + idx + ".weight").c_str(), tensors);
    }

    printf("[code_predictor] loaded %zu tensors\n", tensors.size());
    return tensors;
}

// ============================================================================
// CodePredictorModel: build_cp_layer (single transformer layer)
// ============================================================================

ggml_tensor *CodePredictorModel::build_cp_layer(
    ggml_context *ctx0, ggml_tensor *x, ggml_tensor *kq_mask,
    ggml_tensor *pos, int il) {
    auto &layer = layers_[il];
    int n_heads = config.num_attention_heads;
    int n_kv_heads = config.num_key_value_heads;
    int head_dim = config.head_dim;
    float kq_scale = 1.0f / sqrtf((float)head_dim);
    int seq_len = (int)x->ne[1];

    ggml_tensor *residual = x;

    // RMSNorm
    ggml_tensor *cur = build_norm(ctx0, x, layer.input_layernorm_w, nullptr,
                                   NORM_TYPE_RMS, config.rms_norm_eps, 1, il);

    // Q, K, V projections
    ggml_tensor *q = ggml_mul_mat(ctx0, layer.q_proj_w, cur);
    ggml_tensor *k = ggml_mul_mat(ctx0, layer.k_proj_w, cur);
    ggml_tensor *v = ggml_mul_mat(ctx0, layer.v_proj_w, cur);

    // Reshape for multi-head
    q = ggml_reshape_3d(ctx0, q, head_dim, n_heads, seq_len);
    k = ggml_reshape_3d(ctx0, k, head_dim, n_kv_heads, seq_len);
    v = ggml_reshape_3d(ctx0, v, head_dim, n_kv_heads, seq_len);

    // QK norm (Qwen3 style)
    q = build_norm(ctx0, q, layer.q_norm_w, nullptr, NORM_TYPE_RMS,
                   config.rms_norm_eps, 1, il);
    k = build_norm(ctx0, k, layer.k_norm_w, nullptr, NORM_TYPE_RMS,
                   config.rms_norm_eps, 1, il);

    // RoPE (Qwen3 uses NEOX-style: split-half rotation)
    q = ggml_rope_ext(ctx0, q, pos, nullptr, head_dim,
                       GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    k = ggml_rope_ext(ctx0, k, pos, nullptr, head_dim,
                       GGML_ROPE_TYPE_NEOX, 0, config.rope_theta,
                       1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

    // Attention
    ggml_tensor *attn_out = build_attn(ctx0, layer.o_proj_w, nullptr,
                                        q, k, v, kq_mask, kq_scale, il);
    cur = ggml_add(ctx0, residual, attn_out);

    // MLP
    residual = cur;
    cur = build_norm(ctx0, cur, layer.post_attention_layernorm_w, nullptr,
                     NORM_TYPE_RMS, config.rms_norm_eps, 1, il);
    cur = build_ffn(ctx0, cur, layer.up_proj_w, nullptr,
                    layer.gate_proj_w, nullptr,
                    layer.down_proj_w, nullptr, FFN_SILU, il);
    cur = ggml_add(ctx0, residual, cur);
    return cur;
}

// ============================================================================
// CodePredictorModel: build_graph
// Predicts tokens for ONE group at a time
// Input: hidden_states from Talker + previous group's tokens
// Output: logits for the current group [vocab_size, seq_len]
// ============================================================================

std::vector<ggml_tensor *>
CodePredictorModel::build_graph(ggml_context *ctx0) {
    auto ie_shape = input_shapes_["inputs_embeds"];
    int talker_hidden = ie_shape[0];
    int seq_len = ie_shape[1];

    // Input: pre-built embeddings in Talker space [talker_hidden, seq_len]
    // Positions: [talker_hs, g0_emb, g1_emb, ..., g(n-1)_emb]
    ggml_tensor *inputs_embeds =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, talker_hidden, seq_len);
    ggml_set_name(inputs_embeds, "inputs_embeds");
    ggml_set_input(inputs_embeds);

    // Project all positions: Talker space (2048) → CP hidden (1024)
    ggml_tensor *cur = build_linear(ctx0, inputs_embeds,
                                     small_to_mtp_proj_w_,
                                     small_to_mtp_proj_b_);
    // cur: [cp_hidden, seq_len]

    // Causal mask
    ggml_tensor *kq_mask =
        ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, seq_len, seq_len);
    ggml_set_name(kq_mask, "kq_mask");
    ggml_set_input(kq_mask);

    // Position IDs for RoPE [seq_len]
    ggml_tensor *pos =
        ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, seq_len);
    ggml_set_name(pos, "pos");
    ggml_set_input(pos);

    // Transformer layers
    for (int il = 0; il < config.num_hidden_layers; il++) {
        cur = build_cp_layer(ctx0, cur, kq_mask, pos, il);
    }

    // Final norm
    cur = build_norm(ctx0, cur, norm_w_, nullptr, NORM_TYPE_RMS,
                     config.rms_norm_eps);

    // LM head for current group
    ggml_tensor *logits = ggml_mul_mat(ctx0, lm_heads_[current_group_], cur);
    // logits: [vocab_size, seq_len]

    ggml_set_name(logits, "logits");
    ggml_set_output(logits);
    return {logits};
}

// ============================================================================
// TalkerEmbeddingModel: load_hparams
// ============================================================================

bool TalkerEmbeddingModel::load_hparams(const ModelLoader &loader) {
    loader.get_u32("hidden_size", config.hidden_size);
    loader.get_u32("vocab_size", config.vocab_size);
    loader.get_u32("text_vocab_size", config.text_vocab_size);
    loader.get_u32("num_code_groups", config.num_code_groups);
    loader.get_u32("codec_bos_id", config.codec_bos_id);
    loader.get_u32("codec_eos_token_id", config.codec_eos_token_id);
    loader.get_u32("codec_pad_id", config.codec_pad_id);
    loader.get_f32("rms_norm_eps", config.rms_norm_eps);

    // Parse language IDs from JSON
    std::string lang_json;
    loader.get_string("codec_language_ids_json", lang_json, false);
    if (!lang_json.empty()) {
        // Simple JSON parsing for {"key": value, ...}
        // Format: {"english": 2151, "chinese": 2152, ...}
        size_t pos = 0;
        while ((pos = lang_json.find('"', pos)) != std::string::npos) {
            size_t key_start = pos + 1;
            size_t key_end = lang_json.find('"', key_start);
            if (key_end == std::string::npos) break;
            std::string key = lang_json.substr(key_start, key_end - key_start);
            size_t colon = lang_json.find(':', key_end);
            if (colon == std::string::npos) break;
            int val = std::atoi(lang_json.c_str() + colon + 1);
            config.language_ids[key] = val;
            pos = lang_json.find(',', colon);
            if (pos == std::string::npos) break;
            pos++;
        }
        printf("[talker_embed] loaded %zu language IDs\n",
               config.language_ids.size());
    }

    printf("[talker_embed] hidden=%d codec_vocab=%d text_vocab=%d groups=%d\n",
           config.hidden_size, config.vocab_size, config.text_vocab_size,
           config.num_code_groups);
    return true;
}

void TalkerEmbeddingModel::reset_input_shape() {
    // Dummy shape - not used for graph building
    input_shapes_ = {{"dummy", {1}}};
}

std::vector<ggml_tensor *>
TalkerEmbeddingModel::get_tensors_to_load(ggml_context *ctx) {
    std::vector<ggml_tensor *> tensors;

    text_embedding_w =
        get_tensor(ctx, "model.text_embedding.weight", tensors);
    codec_embedding_w =
        get_tensor(ctx, "model.codec_embedding.weight", tensors);
    text_proj_fc1_w =
        get_tensor(ctx, "text_projection.linear_fc1.weight", tensors);
    text_proj_fc1_b =
        get_tensor(ctx, "text_projection.linear_fc1.bias", tensors);
    text_proj_fc2_w =
        get_tensor(ctx, "text_projection.linear_fc2.weight", tensors);
    text_proj_fc2_b =
        get_tensor(ctx, "text_projection.linear_fc2.bias", tensors);
    codec_head_w = get_tensor(ctx, "codec_head.weight", tensors);
    norm_w = get_tensor(ctx, "model.norm.weight", tensors);

    printf("[talker_embed] loaded %zu tensors\n", tensors.size());
    return tensors;
}

std::vector<ggml_tensor *>
TalkerEmbeddingModel::build_graph(ggml_context *ctx0) {
    // This model doesn't have a standard graph - tensors are used directly
    // by the TalkerLLM class for embedding computation
    return {};
}

// ============================================================================
// TalkerLLM: destructor
// ============================================================================

TalkerLLM::~TalkerLLM() {
    if (cp_llama_ctx_) {
        llama_free(cp_llama_ctx_);
        cp_llama_ctx_ = nullptr;
    }
    if (cp_llama_model_) {
        llama_model_free(cp_llama_model_);
        cp_llama_model_ = nullptr;
    }
    if (llama_ctx_) {
        llama_free(llama_ctx_);
        llama_ctx_ = nullptr;
    }
    if (llama_model_) {
        llama_model_free(llama_model_);
        llama_model_ = nullptr;
    }
}

// ============================================================================
// TalkerLLM: load_model
// ============================================================================

bool TalkerLLM::load_model(const std::string &talker_llama_path,
                             const std::string &talker_embed_path,
                             const std::string &code_predictor_path,
                             int n_threads,
                             int n_gpu_layers,
                             const std::string &cp_llama_path) {
    // 1. Load custom embedding/head tensors from original Talker GGUF
    ContextParams embed_params;
    embed_params.n_threads = n_threads;
    embed_session_ =
        std::make_unique<InferenceSession<TalkerEmbeddingModel>>(
            talker_embed_path, embed_params);
    talker_config_ = embed_session_->get_model().config;

    // 2. Load Talker backbone via llama.cpp (llama-compatible GGUF)
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = n_gpu_layers;

    llama_model_ = llama_model_load_from_file(talker_llama_path.c_str(),
                                               model_params);
    if (!llama_model_) {
        printf("[talker] failed to load llama.cpp backbone from %s\n",
               talker_llama_path.c_str());
        return false;
    }

    n_embd_ = llama_model_n_embd(llama_model_);
    printf("[talker] llama.cpp backbone: n_embd=%d\n", n_embd_);

    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = 4096;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;
    ctx_params.embeddings = true;   // Get hidden states, not logits
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

    llama_ctx_ = llama_init_from_model(llama_model_, ctx_params);
    if (!llama_ctx_) {
        printf("[talker] failed to create llama context\n");
        return false;
    }
    printf("[talker] llama.cpp backbone loaded\n");

    // 3. Load Code Predictor
    ContextParams cp_params;
    cp_params.n_threads = n_threads;
    cp_session_ =
        std::make_unique<InferenceSession<CodePredictorModel>>(
            code_predictor_path, cp_params);
    cp_config_ = cp_session_->get_model().config;

    // Set thread count for CP parallel matvec
    cp_n_threads = n_threads;

    // 4. Optionally load CP transformer via llama.cpp (for NPU acceleration)
    if (!cp_llama_path.empty()) {
        printf("[talker] loading CP llama.cpp backend from %s\n",
               cp_llama_path.c_str());

#ifdef OMINIX_HAS_CANN
        // The CP rollout produces 15 distinct ACL graph shapes per frame
        // (1× 2-token prefill at pos 0+1, then 14× single-token decodes
        // at pos 2..15). The default ggml-cann graph LRU cache holds only
        // 12 entries, which would force a recapture of 3 shapes per frame
        // and erase the speedup. Bump the cache via env var so all 15
        // captured graphs survive between frames. The env var is read by
        // the ggml-cann context constructor below.
        if (!getenv("GGML_CANN_GRAPH_CACHE_CAPACITY")) {
            setenv("GGML_CANN_GRAPH_CACHE_CAPACITY", "32", 0);
        }
        // ADD+RMS_NORM operator fusion reduces kernel launch overhead
        // in transformer blocks (documented in LLM_CANN_OPTIMIZATIONS.md).
        if (!getenv("GGML_CANN_OPERATOR_FUSION")) {
            setenv("GGML_CANN_OPERATOR_FUSION", "on", 0);
        }
#endif

        llama_model_params cp_model_params = llama_model_default_params();
        cp_model_params.n_gpu_layers = n_gpu_layers;

        cp_llama_model_ = llama_model_load_from_file(cp_llama_path.c_str(),
                                                       cp_model_params);
        if (!cp_llama_model_) {
            printf("[talker] WARN: failed to load CP llama model, falling back to CPU\n");
        } else {
            int cp_n_embd = llama_model_n_embd(cp_llama_model_);
            printf("[talker] CP llama.cpp: n_embd=%d\n", cp_n_embd);

            llama_context_params cp_ctx_params = llama_context_default_params();
            cp_ctx_params.n_ctx = 32;         // CP max seq = 17
            cp_ctx_params.n_threads = n_threads;
            cp_ctx_params.n_threads_batch = n_threads;
            cp_ctx_params.embeddings = true;  // Get hidden states, not logits
            cp_ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;

            cp_llama_ctx_ = llama_init_from_model(cp_llama_model_, cp_ctx_params);
            if (!cp_llama_ctx_) {
                printf("[talker] WARN: failed to create CP llama context, falling back to CPU\n");
                llama_model_free(cp_llama_model_);
                cp_llama_model_ = nullptr;
            } else {
                cp_use_llama_ = true;
                printf("[talker] CP llama.cpp backend loaded (NPU-accelerated)\n");
            }
        }
    }

    printf("[talker] all components loaded (hidden=%d, groups=%d, cp_threads=%d, cp_llama=%s)\n",
           n_embd_, talker_config_.num_code_groups, cp_n_threads,
           cp_use_llama_ ? "yes" : "no");
    return true;
}

// ============================================================================
// Embedding helpers: raw tensor data access
// ============================================================================

// Helper: read one row from a 2D ggml tensor (handles f16→f32 conversion)
static void read_embedding_row(ggml_tensor *t, int row, float *out, int dim) {
    if (t->type == GGML_TYPE_F32) {
        const float *data = (const float *)t->data;
        memcpy(out, data + (size_t)row * dim, dim * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *data = (const ggml_fp16_t *)t->data;
        const ggml_fp16_t *src = data + (size_t)row * dim;
        for (int i = 0; i < dim; i++) {
            out[i] = ggml_fp16_to_fp32(src[i]);
        }
    } else {
        printf("[talker] unsupported tensor type %d\n", t->type);
        memset(out, 0, dim * sizeof(float));
    }
}

void TalkerLLM::lookup_text_embedding(int token_id, float *out) {
    auto &m = embed_session_->get_model();
    // text_embedding_w: ggml [hidden, text_vocab] col-major
    // row = token_id, each row has hidden_size floats
    read_embedding_row(m.text_embedding_w, token_id, out, n_embd_);
}

void TalkerLLM::lookup_codec_embedding(int token_id, float *out) {
    auto &m = embed_session_->get_model();
    read_embedding_row(m.codec_embedding_w, token_id, out, n_embd_);
}

// Forward declaration
static void cp_matvec_f32(const float *W, const float *b,
                            const float *x, float *out, int nout, int nin);

// text_projection: fc1 (SiLU) → fc2
// fc1: [out, in] + bias, fc2: [out, in] + bias
// GGUF layout after dimension reversal: ne[0]=in_features, ne[1]=out_features
// Data: W[out][in] = data[out * dim + in]
// So: output[o] = sum_i data[o * dim + i] * input[i]
void TalkerLLM::apply_text_projection(const float *in, float *out) {
    init_head_f32_weights();
    int dim = n_embd_;
    std::vector<float> tmp(dim);
    cp_matvec_f32(tp_fc1_w_.data(), tp_fc1_b_.empty() ? nullptr : tp_fc1_b_.data(),
                   in, tmp.data(), dim, dim);
    for (int o = 0; o < dim; o++) {
        float x = tmp[o];
        tmp[o] = x / (1.0f + expf(-x));
    }
    cp_matvec_f32(tp_fc2_w_.data(), tp_fc2_b_.empty() ? nullptr : tp_fc2_b_.data(),
                   tmp.data(), out, dim, dim);
}

// codec_head: logits = W * hidden (no bias)
// W: [hidden, vocab] in ggml col-major
void TalkerLLM::apply_codec_head(const float *hidden, float *logits) {
    init_head_f32_weights();
    cp_matvec_f32(codec_head_w_.data(), nullptr, hidden, logits,
                   talker_config_.vocab_size, n_embd_);
}

// ============================================================================
// TalkerLLM: Helper functions for embedding construction
// ============================================================================

void TalkerLLM::lookup_text_projected(int token_id, float *out) {
    std::vector<float> tmp(n_embd_);
    lookup_text_embedding(token_id, tmp.data());
    apply_text_projection(tmp.data(), out);
}

void TalkerLLM::cache_tts_embeddings() {
    if (tts_embeds_cached_) return;
    int dim = n_embd_;
    tts_bos_embed_.resize(dim);
    tts_eos_embed_.resize(dim);
    tts_pad_embed_.resize(dim);
    lookup_text_projected(tts_bos_token_id, tts_bos_embed_.data());
    lookup_text_projected(tts_eos_token_id, tts_eos_embed_.data());
    lookup_text_projected(tts_pad_token_id, tts_pad_embed_.data());
    tts_embeds_cached_ = true;
}

void TalkerLLM::compute_ref_frame_embedding(
    const std::vector<std::vector<int>> &ref_codes,
    int frame_idx, float *out) {

    int dim = n_embd_;
    // Group 0: Talker's codec embedding
    lookup_codec_embedding(ref_codes[0][frame_idx], out);

    // Groups 1-15: Code Predictor's per-group codec embeddings
    auto &cp_model = cp_session_->get_model();
    std::vector<float> tmp(dim);
    int n_groups = std::min((int)ref_codes.size(), (int)cp_model.codec_embeddings_.size() + 1);
    for (int g = 1; g < n_groups; g++) {
        read_embedding_row(cp_model.codec_embeddings_[g - 1],
                            ref_codes[g][frame_idx], tmp.data(), dim);
        for (int i = 0; i < dim; i++) out[i] += tmp[i];
    }
}

// ============================================================================
// TalkerLLM: build_input_embeddings (streaming ICL mode)
//
// Python reference (modeling_qwen3_tts.py, generate_icl_prompt, non_streaming_mode=False):
//   Every position = text_component + codec_component (element-wise add)
//
// Sequence layout:
//   Section 1: Role prefix (3 pos) — text_proj(text_emb(im_start, assistant, \n))
//   Section 2: Mixed prefix (N-1 pos) — [tts_pad×(N-2), tts_bos] + codec_prefix[:-1]
//   Section 3: ICL interleaved (max(text_lens, codec_lens) pos)
//     - text_embed = text_proj(ref_text ++ target_text) + tts_eos  [text_lens]
//     - codec_embed = codec_bos + sum_of_groups(ref_code)          [codec_lens]
//     - if text < codec: pad text with tts_pad, add element-wise
//     - if text >= codec: use first codec_lens, rest go to trailing
//   trailing_text_hidden: remaining text tokens (or tts_pad if text <= codec)
// ============================================================================

bool TalkerLLM::build_input_embeddings(
    const std::vector<int> &ref_text_tokens,
    const std::vector<int> &target_text_tokens,
    const std::vector<float> &spk_embedding,
    const std::vector<std::vector<int>> &ref_codes,
    const std::string &language,
    std::vector<float> &embeddings,
    int &seq_len,
    bool xvec_mode) {

    int dim = n_embd_;
    auto &cfg = talker_config_;

    // Cache TTS special embeddings
    cache_tts_embeddings();

    // --- Build codec prefix token list ---
    std::vector<int> codec_prefix_ids;
    auto lang_it = cfg.language_ids.find(language);
    int language_id = -1;
    if (lang_it != cfg.language_ids.end()) {
        language_id = lang_it->second;
        codec_prefix_ids.push_back(cfg.codec_think_id);      // 2154
        codec_prefix_ids.push_back(cfg.codec_think_bos_id);  // 2156
        codec_prefix_ids.push_back(language_id);
        codec_prefix_ids.push_back(cfg.codec_think_eos_id);  // 2157
    } else {
        codec_prefix_ids.push_back(cfg.codec_nothink_id);    // 2155
        codec_prefix_ids.push_back(cfg.codec_think_bos_id);  // 2156
        codec_prefix_ids.push_back(cfg.codec_think_eos_id);  // 2157
    }
    int spk_idx = spk_embedding.empty() ? -1 : (int)codec_prefix_ids.size();
    if (!spk_embedding.empty()) codec_prefix_ids.push_back(-1);
    codec_prefix_ids.push_back(cfg.codec_pad_id);   // 2148
    codec_prefix_ids.push_back(cfg.codec_bos_id);   // 2149

    int N = (int)codec_prefix_ids.size();

    // --- Role prefix ---
    int role_ids[] = {151644, 77091, 198};
    int role_len = 3;

    // Cache role prefix embeddings (same 3 tokens for every sentence)
    if (!role_embeds_cached_) {
        cached_role_embeds_.resize((size_t)role_len * dim);
        for (int i = 0; i < role_len; i++)
            lookup_text_projected(role_ids[i],
                                   cached_role_embeds_.data() + (size_t)i * dim);
        role_embeds_cached_ = true;
    }

    // ========================================================================
    // x-vector-only branch (Python: modeling_qwen3_tts.py:2188 else-branch)
    //
    // Layout:
    //   Section 1: role prefix       (3 pos)         text only
    //   Section 2: mixed prefix      (N-1 pos)       same as ICL (spk inserted)
    //   Section 3: 1 extra position  (1 pos)         text_proj(target_text[0]) + codec_bos
    //
    // trailing_text_hidden = text_proj(target_text[1:]) + tts_eos
    //   length = (target_text_len - 1) + 1 = target_text_len
    // ========================================================================
    if (xvec_mode) {
        if (target_text_tokens.empty()) {
            printf("[talker] xvec mode: target_text_tokens is empty\n");
            return false;
        }
        int target_text_len_x = (int)target_text_tokens.size();

        // Build trailing_text_: target_text[1:] + tts_eos
        int trailing_n = target_text_len_x;  // (n-1) + 1
        trailing_text_.assign((size_t)trailing_n * dim, 0.0f);
        for (int i = 1; i < target_text_len_x; i++) {
            lookup_text_projected(target_text_tokens[i],
                                   trailing_text_.data() + (size_t)(i - 1) * dim);
        }
        memcpy(trailing_text_.data() + (size_t)(target_text_len_x - 1) * dim,
               tts_eos_embed_.data(), dim * sizeof(float));
        trailing_text_len_ = trailing_n;

        // Total prefill = role + (N-1) + 1
        seq_len = role_len + (N - 1) + 1;
        embeddings.assign((size_t)seq_len * dim, 0.0f);

        int pos = 0;
        std::vector<float> tmp_codec(dim);

        // Section 1: role prefix (cached)
        memcpy(embeddings.data(), cached_role_embeds_.data(),
               (size_t)role_len * dim * sizeof(float));
        pos += role_len;

        // Section 2: mixed prefix (text=tts_pad×(N-2)|tts_bos, codec=codec_prefix[:-1] with spk)
        for (int i = 0; i < N - 1; i++) {
            float *dst = embeddings.data() + (size_t)pos * dim;
            const float *text_src = (i < N - 2) ? tts_pad_embed_.data()
                                                 : tts_bos_embed_.data();
            memcpy(dst, text_src, dim * sizeof(float));

            if (i == spk_idx) {
                for (int j = 0; j < dim; j++) dst[j] += spk_embedding[j];
            } else {
                lookup_codec_embedding(codec_prefix_ids[i], tmp_codec.data());
                for (int j = 0; j < dim; j++) dst[j] += tmp_codec[j];
            }
            pos++;
        }

        // Section 3: 1 extra position
        //   text_proj(target_text[0])  +  codec_bos  (= codec_prefix_ids[N-1])
        {
            float *dst = embeddings.data() + (size_t)pos * dim;
            lookup_text_projected(target_text_tokens[0], dst);
            lookup_codec_embedding(codec_prefix_ids[N - 1], tmp_codec.data());
            for (int j = 0; j < dim; j++) dst[j] += tmp_codec[j];
            pos++;
        }

        printf("[talker] built input embeddings (xvec): seq_len=%d "
               "(role=%d mixed_prefix=%d xtra=1, target_text=%d trailing=%d)\n",
               seq_len, role_len, N - 1, target_text_len_x, trailing_text_len_);
        return true;
    }

    // --- Build ICL text embeddings: text_proj(ref_text ++ target_text) + tts_eos ---
    int ref_text_len = (int)ref_text_tokens.size();
    int target_text_len = (int)target_text_tokens.size();
    int text_lens = ref_text_len + target_text_len + 1;  // +1 for tts_eos

    // Cache ref_text projected embeddings (same across all sentences)
    if (cached_ref_text_keys_ != ref_text_tokens) {
        cached_ref_text_proj_.resize((size_t)ref_text_len * dim);
        for (int i = 0; i < ref_text_len; i++)
            lookup_text_projected(ref_text_tokens[i],
                                   cached_ref_text_proj_.data() + (size_t)i * dim);
        cached_ref_text_keys_ = ref_text_tokens;
    }

    std::vector<float> text_embeds((size_t)text_lens * dim);
    memcpy(text_embeds.data(), cached_ref_text_proj_.data(),
           (size_t)ref_text_len * dim * sizeof(float));
    for (int i = 0; i < target_text_len; i++)
        lookup_text_projected(target_text_tokens[i],
                               text_embeds.data() + (size_t)(ref_text_len + i) * dim);
    memcpy(text_embeds.data() + (size_t)(text_lens - 1) * dim,
           tts_eos_embed_.data(), dim * sizeof(float));

    // --- Build ICL codec embeddings: codec_bos + sum_of_groups(ref_code) ---
    int ref_frames = ref_codes.empty() ? 0 : (int)ref_codes[0].size();
    int codec_lens = 1 + ref_frames;  // codec_bos + ref frames

    // Cache ref codec frame embeddings (16-group lookups × ref_frames, same across sentences)
    if (cached_codec_lens_ != codec_lens) {
        cached_codec_embeds_.resize((size_t)codec_lens * dim);
        lookup_codec_embedding(cfg.codec_bos_id, cached_codec_embeds_.data());
        for (int f = 0; f < ref_frames; f++)
            compute_ref_frame_embedding(ref_codes, f,
                                         cached_codec_embeds_.data() + (size_t)(1 + f) * dim);
        cached_codec_lens_ = codec_lens;
    }
    const std::vector<float> &codec_embeds = cached_codec_embeds_;

    // --- Streaming interleave (matches Python generate_icl_prompt) ---
    int icl_len = std::max(text_lens, codec_lens);

    // Build trailing_text_hidden for generation loop
    trailing_text_.clear();
    if (text_lens > codec_lens) {
        // text longer: first codec_lens positions interleaved, rest → trailing
        trailing_text_.assign(text_embeds.begin() + (size_t)codec_lens * dim,
                               text_embeds.end());
        // Truncate text_embeds to codec_lens
        text_lens = codec_lens;
        icl_len = codec_lens;
    } else {
        // text shorter or equal: trailing = tts_pad (single embed, repeated each step)
        trailing_text_ = tts_pad_embed_;  // dim floats
    }
    trailing_text_len_ = (text_lens <= codec_lens)
                             ? 0  // flag: use tts_pad for all steps
                             : (int)(trailing_text_.size() / dim);

    // --- Total sequence length ---
    seq_len = role_len + (N - 1) + icl_len;
    embeddings.resize((size_t)seq_len * dim, 0.0f);

    int pos = 0;
    std::vector<float> tmp_codec(dim);

    // Section 1: Role prefix (cached)
    memcpy(embeddings.data(), cached_role_embeds_.data(),
           (size_t)role_len * dim * sizeof(float));
    pos += role_len;

    // Section 2: Mixed prefix (N-1 positions)
    for (int i = 0; i < N - 1; i++) {
        float *dst = embeddings.data() + (size_t)pos * dim;
        const float *text_src = (i < N - 2) ? tts_pad_embed_.data()
                                             : tts_bos_embed_.data();
        memcpy(dst, text_src, dim * sizeof(float));

        if (i == spk_idx) {
            for (int j = 0; j < dim; j++) dst[j] += spk_embedding[j];
        } else {
            lookup_codec_embedding(codec_prefix_ids[i], tmp_codec.data());
            for (int j = 0; j < dim; j++) dst[j] += tmp_codec[j];
        }
        pos++;
    }

    // Section 3: ICL interleaved (icl_len positions)
    // text_embed[i] + codec_embed[i], padding shorter side
    lookup_codec_embedding(cfg.codec_pad_id, tmp_codec.data());  // for text-side padding

    for (int i = 0; i < icl_len; i++) {
        float *dst = embeddings.data() + (size_t)pos * dim;

        // Text component
        if (i < text_lens) {
            memcpy(dst, text_embeds.data() + (size_t)i * dim, dim * sizeof(float));
        } else {
            // Pad text side with tts_pad
            memcpy(dst, tts_pad_embed_.data(), dim * sizeof(float));
        }

        // Codec component (add)
        if (i < codec_lens) {
            const float *codec_src = codec_embeds.data() + (size_t)i * dim;
            for (int j = 0; j < dim; j++) dst[j] += codec_src[j];
        } else {
            // Pad codec side with codec_pad
            for (int j = 0; j < dim; j++) dst[j] += tmp_codec[j];
        }

        // Add tts_pad to codec positions (Python: codec_embed + tts_pad for all)
        // Wait — in Python streaming mode, the ICL section is:
        //   return text_embed + codec_embed   (no extra tts_pad addition)
        // The tts_pad addition only happens in Section 2 and during generation.
        // So we should NOT add tts_pad here. The above text+codec sum is correct.

        pos++;
    }

    printf("[talker] built input embeddings (streaming): seq_len=%d "
           "(role=%d mixed_prefix=%d icl=%d, text=%d codec=%d trailing=%d)\n",
           seq_len, role_len, N - 1, icl_len, text_lens, codec_lens,
           trailing_text_len_);

    return true;
}

// ============================================================================
// CP weight pre-conversion and KV-cached forward pass helpers
// ============================================================================

// Convert ggml tensor data to F32 vector
static std::vector<float> tensor_to_f32(ggml_tensor *t) {
    int n = (int)ggml_nelements(t);
    std::vector<float> out(n);
    if (t->type == GGML_TYPE_F32) {
        memcpy(out.data(), t->data, n * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (int i = 0; i < n; i++) out[i] = ggml_fp16_to_fp32(src[i]);
    }
    return out;
}

// Pre-convert all CP weights from F16 to F32 for fast matvec
void TalkerLLM::init_cp_f32_weights() {
    if (cp_f32_ready_) return;
    auto &m = cp_session_->get_model();
    auto &cfg = cp_config_;

    cp_f32_.proj_w = tensor_to_f32(m.small_to_mtp_proj_w_);
    cp_f32_.proj_b = tensor_to_f32(m.small_to_mtp_proj_b_);

    cp_f32_.layers.resize(cfg.num_hidden_layers);
    for (int il = 0; il < cfg.num_hidden_layers; il++) {
        auto &src = m.layers_[il];
        auto &dst = cp_f32_.layers[il];
        dst.q_proj_w = tensor_to_f32(src.q_proj_w);
        dst.k_proj_w = tensor_to_f32(src.k_proj_w);
        dst.v_proj_w = tensor_to_f32(src.v_proj_w);
        dst.o_proj_w = tensor_to_f32(src.o_proj_w);
        dst.q_norm_w = tensor_to_f32(src.q_norm_w);
        dst.k_norm_w = tensor_to_f32(src.k_norm_w);
        dst.gate_proj_w = tensor_to_f32(src.gate_proj_w);
        dst.up_proj_w = tensor_to_f32(src.up_proj_w);
        dst.down_proj_w = tensor_to_f32(src.down_proj_w);
        dst.input_ln_w = tensor_to_f32(src.input_layernorm_w);
        dst.post_ln_w = tensor_to_f32(src.post_attention_layernorm_w);
    }
    cp_f32_.norm_w = tensor_to_f32(m.norm_w_);

    int n_groups = cfg.num_code_groups - 1;
    cp_f32_.lm_head_w.resize(n_groups);
    for (int g = 0; g < n_groups; g++) {
        cp_f32_.lm_head_w[g] = tensor_to_f32(m.lm_heads_[g]);
    }

    cp_f32_ready_ = true;
    printf("[talker] pre-converted CP weights to F32\n");
}

void TalkerLLM::init_head_f32_weights() {
    if (head_f32_ready_) return;
    auto &m = embed_session_->get_model();

    tp_fc1_w_ = tensor_to_f32(m.text_proj_fc1_w);
    tp_fc2_w_ = tensor_to_f32(m.text_proj_fc2_w);
    if (m.text_proj_fc1_b) tp_fc1_b_ = tensor_to_f32(m.text_proj_fc1_b);
    if (m.text_proj_fc2_b) tp_fc2_b_ = tensor_to_f32(m.text_proj_fc2_b);
    codec_head_w_ = tensor_to_f32(m.codec_head_w);

    head_f32_ready_ = true;
    printf("[talker] pre-converted text_proj + codec_head to F32\n");
}

// F32 matrix-vector product with NEON + OpenMP: out[o] = sum_i W[o*in+i] * x[i] + b[o]
static void cp_matvec_f32(const float *W, const float *b,
                            const float *x, float *out, int nout, int nin) {
#ifdef _OPENMP
    #pragma omp parallel for num_threads(cp_n_threads) schedule(static)
#endif
    for (int o = 0; o < nout; o++) {
        const float *row = W + (size_t)o * nin;
#ifdef __aarch64__
        float32x4_t s0 = vdupq_n_f32(0), s1 = vdupq_n_f32(0);
        float32x4_t s2 = vdupq_n_f32(0), s3 = vdupq_n_f32(0);
        int i = 0;
        for (; i + 15 < nin; i += 16) {
            s0 = vfmaq_f32(s0, vld1q_f32(row + i),      vld1q_f32(x + i));
            s1 = vfmaq_f32(s1, vld1q_f32(row + i + 4),  vld1q_f32(x + i + 4));
            s2 = vfmaq_f32(s2, vld1q_f32(row + i + 8),  vld1q_f32(x + i + 8));
            s3 = vfmaq_f32(s3, vld1q_f32(row + i + 12), vld1q_f32(x + i + 12));
        }
        float result = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
        for (; i < nin; i++) result += row[i] * x[i];
#else
        float result = 0.0f;
        for (int i = 0; i < nin; i++) result += row[i] * x[i];
#endif
        out[o] = result;
    }
    if (b) {
        for (int o = 0; o < nout; o++) out[o] += b[o];
    }
}

// RMS normalization with F32 affine weight
static void cp_rms_norm_f32(const float *x, const float *w,
                              float *out, int dim, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) sum_sq += x[i] * x[i];
    float scale = 1.0f / sqrtf(sum_sq / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * scale * w[i];
}

// NEOX-style RoPE: split-half rotation
static void cp_rope_neox(float *vec, int head_dim, int pos, float theta) {
    int half = head_dim / 2;
    for (int i = 0; i < half; i++) {
        float freq = 1.0f / powf(theta, (float)(2 * i) / head_dim);
        float angle = pos * freq;
        float cos_a = cosf(angle);
        float sin_a = sinf(angle);
        float x0 = vec[i];
        float x1 = vec[i + half];
        vec[i]        = x0 * cos_a - x1 * sin_a;
        vec[i + half] = x1 * cos_a + x0 * sin_a;
    }
}

// Single-query GQA attention against KV cache
// q: [n_heads * head_dim], k_cache/v_cache: [max_seq * kv_dim]
static void cp_gqa_attention(const float *q,
                              const float *k_cache, const float *v_cache,
                              int seq_len, int n_heads, int n_kv_heads,
                              int head_dim, float scale,
                              float *out, float *score_buf) {
    int group_size = n_heads / n_kv_heads;
    int kv_dim = n_kv_heads * head_dim;

    for (int h = 0; h < n_heads; h++) {
        int kv_h = h / group_size;
        const float *q_h = q + h * head_dim;
        float *out_h = out + h * head_dim;

        // Compute attention scores
        float max_s = -INFINITY;
        for (int p = 0; p < seq_len; p++) {
            const float *k_p = k_cache + (size_t)p * kv_dim + kv_h * head_dim;
            float dot = 0.0f;
            for (int d = 0; d < head_dim; d++) dot += q_h[d] * k_p[d];
            score_buf[p] = dot * scale;
            if (score_buf[p] > max_s) max_s = score_buf[p];
        }

        // Softmax
        float sum = 0.0f;
        for (int p = 0; p < seq_len; p++) {
            score_buf[p] = expf(score_buf[p] - max_s);
            sum += score_buf[p];
        }
        float inv_sum = 1.0f / sum;
        for (int p = 0; p < seq_len; p++) score_buf[p] *= inv_sum;

        // Weighted sum of V
        memset(out_h, 0, head_dim * sizeof(float));
        for (int p = 0; p < seq_len; p++) {
            const float *v_p = v_cache + (size_t)p * kv_dim + kv_h * head_dim;
            float s = score_buf[p];
            for (int d = 0; d < head_dim; d++) out_h[d] += s * v_p[d];
        }
    }
}

// ============================================================================
// TalkerLLM: cp_forward_one_token
// Process one token through CP transformer, caching K/V at position pos
// Input: [talker_hidden], Output: [cp_hidden] (post final-norm)
// ============================================================================

void TalkerLLM::init_cp_work_bufs() {
    if (cp_bufs_.ready) return;
    auto &cfg = cp_config_;
    int ch = cfg.hidden_size;
    int q_dim = cfg.num_attention_heads * cfg.head_dim;
    int kv_dim = cfg.num_key_value_heads * cfg.head_dim;
    int inter = cfg.intermediate_size;
    cp_bufs_.cur.resize(ch);
    cp_bufs_.residual.resize(ch);
    cp_bufs_.normed.resize(ch);
    cp_bufs_.q.resize(q_dim);
    cp_bufs_.k.resize(kv_dim);
    cp_bufs_.v.resize(kv_dim);
    cp_bufs_.attn_out.resize(q_dim);
    cp_bufs_.o_out.resize(ch);
    cp_bufs_.gate.resize(inter);
    cp_bufs_.up.resize(inter);
    cp_bufs_.ffn_out.resize(ch);
    cp_bufs_.scores.resize(CP_MAX_SEQ);
    cp_bufs_.ready = true;
}

void TalkerLLM::cp_forward_one_token(const float *input_talker_space,
                                       int pos, float *hidden_out) {
    auto &cfg = cp_config_;
    int th = cfg.talker_hidden_size;   // 2048
    int ch = cfg.hidden_size;          // 1024
    int n_heads = cfg.num_attention_heads;   // 16
    int n_kv = cfg.num_key_value_heads;      // 8
    int hd = cfg.head_dim;                   // 128
    int q_dim = n_heads * hd;               // 2048
    int kv_dim = n_kv * hd;                 // 1024
    int inter = cfg.intermediate_size;       // 3072
    float eps = cfg.rms_norm_eps;
    float kq_scale = 1.0f / sqrtf((float)hd);

    // Use pre-allocated working buffers
    init_cp_work_bufs();
    float *cur = cp_bufs_.cur.data();
    float *residual = cp_bufs_.residual.data();
    float *normed = cp_bufs_.normed.data();
    float *q = cp_bufs_.q.data();
    float *k = cp_bufs_.k.data();
    float *v = cp_bufs_.v.data();
    float *attn_out = cp_bufs_.attn_out.data();
    float *o_out = cp_bufs_.o_out.data();
    float *gate = cp_bufs_.gate.data();
    float *up = cp_bufs_.up.data();
    float *ffn_out = cp_bufs_.ffn_out.data();
    float *scores = cp_bufs_.scores.data();

    // 1. Project from talker space to CP space (F32 weights)
    cp_matvec_f32(cp_f32_.proj_w.data(), cp_f32_.proj_b.data(),
                  input_talker_space, cur, ch, th);

    // 2. Transformer layers
    for (int il = 0; il < cfg.num_hidden_layers; il++) {
        auto &lw = cp_f32_.layers[il];

        memcpy(residual, cur, ch * sizeof(float));

        // Input LayerNorm
        cp_rms_norm_f32(cur, lw.input_ln_w.data(), normed, ch, eps);

        // Q/K/V projections
        cp_matvec_f32(lw.q_proj_w.data(), nullptr, normed, q, q_dim, ch);
        cp_matvec_f32(lw.k_proj_w.data(), nullptr, normed, k, kv_dim, ch);
        cp_matvec_f32(lw.v_proj_w.data(), nullptr, normed, v, kv_dim, ch);

        // QK norm (per-head, shared weight)
        for (int h = 0; h < n_heads; h++)
            cp_rms_norm_f32(&q[h * hd], lw.q_norm_w.data(), &q[h * hd], hd, eps);
        for (int h = 0; h < n_kv; h++)
            cp_rms_norm_f32(&k[h * hd], lw.k_norm_w.data(), &k[h * hd], hd, eps);

        // RoPE (NEOX style)
        for (int h = 0; h < n_heads; h++)
            cp_rope_neox(&q[h * hd], hd, pos, cfg.rope_theta);
        for (int h = 0; h < n_kv; h++)
            cp_rope_neox(&k[h * hd], hd, pos, cfg.rope_theta);

        // Store K/V in cache
        memcpy(&cp_k_cache_[il][(size_t)pos * kv_dim], k, kv_dim * sizeof(float));
        memcpy(&cp_v_cache_[il][(size_t)pos * kv_dim], v, kv_dim * sizeof(float));

        // Attention against all cached positions [0..pos]
        cp_gqa_attention(q, cp_k_cache_[il].data(), cp_v_cache_[il].data(),
                          pos + 1, n_heads, n_kv, hd, kq_scale, attn_out, scores);

        // O projection
        cp_matvec_f32(lw.o_proj_w.data(), nullptr, attn_out, o_out, ch, q_dim);

        for (int i = 0; i < ch; i++) cur[i] = residual[i] + o_out[i];

        // Post-attention LayerNorm + FFN
        memcpy(residual, cur, ch * sizeof(float));
        cp_rms_norm_f32(cur, lw.post_ln_w.data(), normed, ch, eps);

        // SwiGLU FFN: silu(gate) * up → down
        cp_matvec_f32(lw.gate_proj_w.data(), nullptr, normed, gate, inter, ch);
        cp_matvec_f32(lw.up_proj_w.data(), nullptr, normed, up, inter, ch);
        for (int i = 0; i < inter; i++) {
            float sigmoid = 1.0f / (1.0f + expf(-gate[i]));
            gate[i] = gate[i] * sigmoid * up[i];
        }
        cp_matvec_f32(lw.down_proj_w.data(), nullptr, gate, ffn_out, ch, inter);

        for (int i = 0; i < ch; i++) cur[i] = residual[i] + ffn_out[i];
    }

    // 3. Final norm
    cp_rms_norm_f32(cur, cp_f32_.norm_w.data(), hidden_out, ch, eps);
}

// ============================================================================
// TalkerLLM: predict_code_groups (KV-cached Code Predictor for groups 1-15)
// ============================================================================

bool TalkerLLM::predict_code_groups(
    const float *hidden_states, int /*seq_len*/,
    int group0_token, std::vector<int> &group_tokens,
    const TalkerSamplingParams &sampling) {

    int n_groups = cp_config_.num_code_groups - 1;  // 15
    int talker_hidden = cp_config_.talker_hidden_size;  // 2048
    int cp_hidden = cp_config_.hidden_size;  // 1024
    int vocab_size = cp_config_.vocab_size;  // 2048
    group_tokens.resize(n_groups);

    auto &cp_model = cp_session_->get_model();

    // Pre-convert weights on first call (needed for input_proj + lm_heads in both paths)
    init_cp_f32_weights();

    if (cp_use_llama_) {
        // ============================================================
        // NPU path: CP transformer via llama.cpp
        // input_proj and lm_heads remain on CPU (negligible compute)
        // ============================================================
        //
        // Force ACL graph mode ON for the duration of this function. The
        // cp_llama context is the only sub-model in the qwen_tts pipeline
        // whose shape pattern actually benefits from CANN graph capture:
        // each frame issues the same 15 unique shapes (1× 2-token prefill
        // at pos 0+1, then 14× single-token decodes at pos 2..15), and
        // those shapes repeat verbatim across all 50+ generation frames,
        // so cache hit rate after frame 0 is ~100%. The talker context
        // (different KV size every step → cache thrash) and the codec
        // decoder context (GatherV3 crash under capture, see Apr-2026
        // diagnostic) intentionally stay in eager mode.
#ifdef OMINIX_HAS_CANN
        ggml_backend_cann_set_thread_acl_graph_override(true, true);
#endif
        llama_memory_clear(llama_get_memory(cp_llama_ctx_), true);

        std::vector<float> projected(cp_hidden);
        std::vector<float> logits(vocab_size);
        std::vector<float> emb_buf(talker_hidden);

        // Positions 0+1: batch prefill (talker hidden + group0 embedding)
        cp_matvec_f32(cp_f32_.proj_w.data(), cp_f32_.proj_b.data(),
                      hidden_states, projected.data(), cp_hidden, talker_hidden);

        std::vector<float> projected1(cp_hidden);
        lookup_codec_embedding(group0_token, emb_buf.data());
        cp_matvec_f32(cp_f32_.proj_w.data(), cp_f32_.proj_b.data(),
                      emb_buf.data(), projected1.data(), cp_hidden, talker_hidden);

        {
            llama_batch batch = llama_batch_init(2, cp_hidden, 1);
            batch.n_tokens = 2;
            memcpy(batch.embd, projected.data(), cp_hidden * sizeof(float));
            memcpy(batch.embd + cp_hidden, projected1.data(), cp_hidden * sizeof(float));
            batch.pos[0] = 0;
            batch.pos[1] = 1;
            batch.n_seq_id[0] = 1;
            batch.n_seq_id[1] = 1;
            batch.seq_id[0][0] = 0;
            batch.seq_id[1][0] = 0;
            batch.logits[0] = 0;
            batch.logits[1] = 1;  // only need output from pos 1
            llama_decode(cp_llama_ctx_, batch);
            llama_batch_free(batch);
        }

        // Get hidden state after position 1
        float *llama_embd = llama_get_embeddings_ith(cp_llama_ctx_, -1);
        std::vector<float> cp_out(cp_hidden);
        memcpy(cp_out.data(), llama_embd, cp_hidden * sizeof(float));

        // Decode groups 1-15 autoregressively
        for (int g = 0; g < n_groups; g++) {
            // Apply lm_head on CPU
            cp_matvec_f32(cp_f32_.lm_head_w[g].data(), nullptr,
                          cp_out.data(), logits.data(), vocab_size, cp_hidden);

            int sampled = sample_token(logits.data(), vocab_size,
                                        sampling.cp_temperature, sampling.cp_top_k,
                                        sampling.cp_top_p, sampling.cp_do_sample);
            group_tokens[g] = sampled;

            // If not last group, feed this group's embedding for next iteration
            if (g < n_groups - 1) {
                read_embedding_row(cp_model.codec_embeddings_[g],
                                    sampled, emb_buf.data(), talker_hidden);
                cp_matvec_f32(cp_f32_.proj_w.data(), cp_f32_.proj_b.data(),
                              emb_buf.data(), projected.data(), cp_hidden, talker_hidden);

                llama_batch batch = llama_batch_init(1, cp_hidden, 1);
                batch.n_tokens = 1;
                memcpy(batch.embd, projected.data(), cp_hidden * sizeof(float));
                batch.pos[0] = g + 2;
                batch.n_seq_id[0] = 1;
                batch.seq_id[0][0] = 0;
                batch.logits[0] = 1;
                llama_decode(cp_llama_ctx_, batch);
                llama_batch_free(batch);

                llama_embd = llama_get_embeddings_ith(cp_llama_ctx_, -1);
                memcpy(cp_out.data(), llama_embd, cp_hidden * sizeof(float));
            }
        }

#ifdef OMINIX_HAS_CANN
        ggml_backend_cann_set_thread_acl_graph_override(false, false);
#endif
        return true;
    }

    // ============================================================
    // CPU path: original custom implementation
    // ============================================================

    // Initialize KV cache if needed
    int kv_dim = cp_config_.num_key_value_heads * cp_config_.head_dim;
    if (cp_k_cache_.empty()) {
        cp_k_cache_.resize(cp_config_.num_hidden_layers);
        cp_v_cache_.resize(cp_config_.num_hidden_layers);
        for (int i = 0; i < cp_config_.num_hidden_layers; i++) {
            cp_k_cache_[i].resize(CP_MAX_SEQ * kv_dim, 0.0f);
            cp_v_cache_[i].resize(CP_MAX_SEQ * kv_dim, 0.0f);
        }
    }

    // Clear KV cache for this new frame
    cp_cache_len_ = 0;

    // Position 0: process talker hidden state
    std::vector<float> cp_out(cp_hidden);
    cp_forward_one_token(hidden_states, 0, cp_out.data());

    // Position 1: process group 0 token embedding (Talker codec_embedding)
    std::vector<float> g0_emb(talker_hidden);
    lookup_codec_embedding(group0_token, g0_emb.data());
    cp_forward_one_token(g0_emb.data(), 1, cp_out.data());

    // Decode groups 1-15 autoregressively
    std::vector<float> logits(vocab_size);
    std::vector<float> group_emb(talker_hidden);

    for (int g = 0; g < n_groups; g++) {
        // Apply lm_head for this group to get logits (F32 pre-converted)
        cp_matvec_f32(cp_f32_.lm_head_w[g].data(), nullptr,
                      cp_out.data(), logits.data(), vocab_size, cp_hidden);

        // Sample
        int sampled = sample_token(logits.data(), vocab_size,
                                    sampling.cp_temperature, sampling.cp_top_k,
                                    sampling.cp_top_p, sampling.cp_do_sample);
        group_tokens[g] = sampled;

        // If not last group, process this group's embedding for next iteration
        if (g < n_groups - 1) {
            read_embedding_row(cp_model.codec_embeddings_[g],
                                sampled, group_emb.data(), talker_hidden);
            cp_forward_one_token(group_emb.data(), g + 2, cp_out.data());
        }
    }

    return true;
}

// ============================================================================
// TalkerLLM: compute_next_embedding
// Sum group 0 (Talker codec_emb) + groups 1-15 (CP codec_embs)
// ============================================================================

void TalkerLLM::compute_next_embedding(
    int group0_token, const std::vector<int> &group_tokens, float *out) {

    int dim = n_embd_;
    // Start with group 0 embedding from Talker
    lookup_codec_embedding(group0_token, out);

    // Add Code Predictor codec embeddings for groups 1-15
    auto &cp_model = cp_session_->get_model();
    std::vector<float> tmp(dim);
    int n_groups = (int)group_tokens.size();
    for (int g = 0; g < n_groups; g++) {
        read_embedding_row(cp_model.codec_embeddings_[g],
                            group_tokens[g], tmp.data(), dim);
        for (int i = 0; i < dim; i++) {
            out[i] += tmp[i];
        }
    }

    // Add tts_pad text component (every position = text + codec)
    for (int i = 0; i < dim; i++) {
        out[i] += tts_pad_embed_[i];
    }
}

// ============================================================================
// TalkerLLM: generate (full generation loop)
// ============================================================================

bool TalkerLLM::generate(
    const std::vector<int> &ref_text_tokens,
    const std::vector<int> &target_text_tokens,
    const std::vector<float> &spk_embedding,
    const std::vector<std::vector<int>> &ref_codes,
    const std::string &language,
    std::vector<std::vector<int>> &codec_tokens,
    int max_new_tokens,
    const TalkerSamplingParams &sampling,
    bool xvec_mode,
    FrameCallback frame_callback) {

    if (!llama_ctx_) {
        printf("[talker] llama.cpp backbone not loaded\n");
        return false;
    }

    int dim = n_embd_;
    int n_groups = talker_config_.num_code_groups;

    auto gen_t0 = std::chrono::high_resolution_clock::now();

    // Reset per-call state for the near-repeat loop detector.
    near_repeat_run_ = 0;

    // Ensure TTS embeddings are cached
    cache_tts_embeddings();

    // Initialize output: [n_groups][T]
    codec_tokens.resize(n_groups);

    // 1. Build prefill embeddings
    std::vector<float> prefill_embs;
    int prefill_len = 0;
    if (!build_input_embeddings(ref_text_tokens, target_text_tokens,
                                 spk_embedding, ref_codes,
                                 language, prefill_embs, prefill_len,
                                 xvec_mode)) {
        return false;
    }

    auto build_emb_t1 = std::chrono::high_resolution_clock::now();
    double build_emb_ms = std::chrono::duration<double, std::milli>(build_emb_t1 - gen_t0).count();

    // 2. Prefill: feed all embeddings to llama.cpp
    llama_memory_clear(llama_get_memory(llama_ctx_), true);

    llama_batch batch = llama_batch_init(prefill_len, dim, 1);
    batch.n_tokens = prefill_len;
    memcpy(batch.embd, prefill_embs.data(),
           (size_t)prefill_len * dim * sizeof(float));
    for (int i = 0; i < prefill_len; i++) {
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == prefill_len - 1) ? 1 : 0;
    }

    printf("[talker] prefill %d tokens (dim=%d)...\n", prefill_len, dim);
    auto prefill_t0 = std::chrono::high_resolution_clock::now();
    int decode_rc = llama_decode(llama_ctx_, batch);
    if (decode_rc != 0) {
        printf("[talker] prefill failed\n");
        llama_batch_free(batch);
        return false;
    }
    llama_batch_free(batch);
    auto prefill_t1 = std::chrono::high_resolution_clock::now();
    double talker_prefill_ms = std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();

    // 3. Get hidden states from last position
    float *embd = llama_get_embeddings_ith(llama_ctx_, -1);
    if (!embd) {
        printf("[talker] failed to get embeddings after prefill\n");
        return false;
    }

    // llama.cpp embeddings mode gives post-norm hidden states,
    // so codec_head can be applied directly.
    std::vector<float> hidden(dim);
    memcpy(hidden.data(), embd, dim * sizeof(float));

    // 4. Autoregressive decode loop
    int cur_pos = prefill_len;
    int vocab_size = talker_config_.vocab_size;
    std::vector<float> logits_buf(vocab_size);
    std::vector<float> next_emb(dim);
    std::vector<int> generated_g0;  // for repetition penalty

    printf("[talker] sampling: %s, temp=%.2f, top_k=%d, top_p=%.2f, rep_penalty=%.2f\n",
           sampling.do_sample ? "on" : "off (greedy)",
           sampling.temperature, sampling.top_k, sampling.top_p,
           sampling.repetition_penalty);

    double total_llm_ms = 0, total_cp_ms = 0, total_emb_ms = 0, total_head_ms = 0;
    double total_sample_ms = 0, total_trailing_ms = 0, total_loop_ms = 0;

    for (int step = 0; step < max_new_tokens; step++) {
        auto loop_t0 = std::chrono::high_resolution_clock::now();
        auto head_t0 = loop_t0;
        // 4a. Apply codec_head → group 0 logits
        apply_codec_head(hidden.data(), logits_buf.data());

        // 4b. Suppress special tokens (>= 2048 except EOS)
        suppress_special_tokens(logits_buf.data(), vocab_size,
                                 talker_config_.codec_eos_token_id);

        // 4c. Apply repetition penalty
        apply_repetition_penalty(logits_buf.data(), vocab_size,
                                  generated_g0, sampling.repetition_penalty);

        auto head_t1 = std::chrono::high_resolution_clock::now();
        total_head_ms += std::chrono::duration<double, std::milli>(head_t1 - head_t0).count();

        // 4d. Sample group 0
        auto sample_t0 = std::chrono::high_resolution_clock::now();
        int group0_token = sample_token(logits_buf.data(), vocab_size,
                                         sampling.temperature, sampling.top_k,
                                         sampling.top_p, sampling.do_sample);
        generated_g0.push_back(group0_token);
        auto sample_t1 = std::chrono::high_resolution_clock::now();
        total_sample_ms += std::chrono::duration<double, std::milli>(sample_t1 - sample_t0).count();

        // 4e. Check EOS
        if (group0_token == talker_config_.codec_eos_token_id) {
            printf("[talker] EOS at step %d\n", step);
            break;
        }

        // 4d. Handle special tokens vs actual codec tokens
        bool is_special = (group0_token >= 2048);

        if (!is_special) {
            // Actual codec token (0-2047): run Code Predictor for groups 1-15
            auto cp_t0 = std::chrono::high_resolution_clock::now();
            std::vector<int> group_tokens;
            if (!predict_code_groups(hidden.data(), 1, group0_token,
                                      group_tokens, sampling)) {
                printf("[talker] code predictor failed at step %d\n", step);
                return false;
            }

            // Store tokens
            codec_tokens[0].push_back(group0_token);
            for (int g = 0; g < (int)group_tokens.size(); g++) {
                codec_tokens[g + 1].push_back(group_tokens[g]);
            }

            auto cp_t1 = std::chrono::high_resolution_clock::now();
            total_cp_ms += std::chrono::duration<double, std::milli>(cp_t1 - cp_t0).count();

            // ----- Near-repeat / loop detector ----------------------------
            // Generic safety net for any sampling configuration: if the
            // talker drives itself into a state where consecutive codec
            // frames share most of their quantizer tokens, the audio for
            // those frames sounds like the same syllable looping forever.
            // Captured signature in one observed regression:
            //   frame N:   g0=1521 | 1443 1471 59 145 736 406 1034 632 ...
            //   frame N+1: g0=1095 | 1443 1471 59 145 736 406 1034 632 ...
            // 10+ of 16 quantizers identical between adjacent frames.
            //
            // Strategy: count consecutive frames whose total quantizer
            // match against the previous frame is ≥ 10. After 3 such
            // near-repeat frames in a row, declare a loop, drop the
            // looping frames, and force EOS. The threshold leaves room
            // for legitimate held syllables (which may share a few
            // quantizers) but catches the tight-repetition pattern.
            int n_so_far = (int) codec_tokens[0].size();
            if (n_so_far >= 2) {
                int last = n_so_far - 1;
                int matches = 0;
                int n_q = (int) codec_tokens.size();
                for (int q = 0; q < n_q; q++) {
                    if (codec_tokens[q][last] == codec_tokens[q][last - 1]) {
                        matches++;
                    }
                }
                if (matches >= 10) {
                    near_repeat_run_++;
                } else {
                    near_repeat_run_ = 0;
                }
                if (near_repeat_run_ >= 3) {
                    printf("[talker] near-repeat loop detected at step %d "
                           "(%d/%d quantizers match prev frame for "
                           "%d frames running) → forcing EOS\n",
                           step, matches, n_q, near_repeat_run_);
                    int drop = std::min(near_repeat_run_, n_so_far);
                    for (int q = 0; q < n_q; q++) {
                        codec_tokens[q].resize(n_so_far - drop);
                    }
                    near_repeat_run_ = 0;
                    break;
                }
            }

            // Streaming hook: notify caller that a new frame is ready.
            // Fired only for real codec frames (special tokens skip CP and
            // shouldn't enter the audio stream).
            if (frame_callback) {
                int frame_idx = (int)codec_tokens[0].size() - 1;
                frame_callback(frame_idx, codec_tokens);
            }

            // Compute next embedding (sum of all 16 group embeddings)
            auto emb_t0 = std::chrono::high_resolution_clock::now();
            compute_next_embedding(group0_token, group_tokens, next_emb.data());
            auto emb_t1 = std::chrono::high_resolution_clock::now();
            total_emb_ms += std::chrono::duration<double, std::milli>(emb_t1 - emb_t0).count();
        } else {
            // Special token (≥2048): skip Code Predictor, use Talker codec emb
            lookup_codec_embedding(group0_token, next_emb.data());
        }

        // Add trailing_text_hidden
        auto trailing_t0 = std::chrono::high_resolution_clock::now();
        {
            int gen_step = (int)codec_tokens[0].size();  // current generation step
            if (trailing_text_len_ > 0 && gen_step < trailing_text_len_) {
                // Remaining text tokens from ICL (text > codec case)
                const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
                for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
            } else {
                // Default: add tts_pad (text <= codec, or past trailing range)
                for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
            }
        }

        auto trailing_t1 = std::chrono::high_resolution_clock::now();
        total_trailing_ms += std::chrono::duration<double, std::milli>(trailing_t1 - trailing_t0).count();

        // 4e. Feed to llama.cpp
        auto llm_t0 = std::chrono::high_resolution_clock::now();
        llama_batch step_batch = llama_batch_init(1, dim, 1);
        step_batch.n_tokens = 1;
        memcpy(step_batch.embd, next_emb.data(), dim * sizeof(float));
        step_batch.pos[0] = cur_pos;
        step_batch.n_seq_id[0] = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0] = 1;

        if (llama_decode(llama_ctx_, step_batch) != 0) {
            printf("[talker] decode failed at step %d\n", step);
            llama_batch_free(step_batch);
            return false;
        }
        llama_batch_free(step_batch);
        cur_pos++;

        // 4f. Get new hidden states
        embd = llama_get_embeddings_ith(llama_ctx_, -1);
        if (!embd) {
            printf("[talker] failed to get embeddings at step %d\n", step);
            return false;
        }
        memcpy(hidden.data(), embd, dim * sizeof(float));
        auto llm_t1 = std::chrono::high_resolution_clock::now();
        total_llm_ms += std::chrono::duration<double, std::milli>(llm_t1 - llm_t0).count();
        total_loop_ms += std::chrono::duration<double, std::milli>(llm_t1 - loop_t0).count();

        if ((step + 1) % 50 == 0) {
            printf("[talker] step %d/%d, group0_token=%d, codec_frames=%zu\n",
                   step + 1, max_new_tokens, group0_token,
                   codec_tokens[0].size());
        }
    }

    printf("[talker] generated %zu codec frames\n", codec_tokens[0].size());

    // Dump first 5 frames for analysis
    {
        int n_dump = std::min((int)codec_tokens[0].size(), 5);
        for (int t = 0; t < n_dump; t++) {
            printf("[talker] frame %d: g0=%d |", t, codec_tokens[0][t]);
            for (size_t g = 1; g < codec_tokens.size(); g++)
                printf(" %d", codec_tokens[g][t]);
            printf("\n");
        }
    }

    printf("[talker] timing breakdown:\n");
    printf("[talker]   build_emb: %7.0f ms (build_input_embeddings)\n", build_emb_ms);
    printf("[talker]   prefill:   %7.0f ms\n", talker_prefill_ms);
    printf("[talker]   head:      %7.0f ms (codec_head + suppress + rep_penalty)\n", total_head_ms);
    printf("[talker]   sample:    %7.0f ms\n", total_sample_ms);
    printf("[talker]   CP:        %7.0f ms (predict_code_groups)\n", total_cp_ms);
    printf("[talker]   EMB:       %7.0f ms (compute_next_embedding)\n", total_emb_ms);
    printf("[talker]   trailing:  %7.0f ms (trailing_text_hidden)\n", total_trailing_ms);
    printf("[talker]   LLM:       %7.0f ms (llama_decode + get_embd)\n", total_llm_ms);
    printf("[talker]   loop_sum:  %7.0f ms (per-step wall clock)\n", total_loop_ms);
    double total_accounted = build_emb_ms + talker_prefill_ms + total_loop_ms;
    printf("[talker]   TOTAL:     %7.0f ms (build_emb + prefill + loop)\n", total_accounted);
    return true;
}
