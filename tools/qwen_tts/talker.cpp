#include "talker.h"
#include "tts_transformer.h"
#include "ggml.h"
#if defined(QWEN_TTS_HAS_CP_CANN)
#include "cp_cann_engine.h"
#include "talker_cann_engine.h"
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

    // Parse speaker IDs from JSON (CustomVoice model)
    std::string spk_json;
    loader.get_string("spk_ids_json", spk_json, false);
    if (!spk_json.empty()) {
        size_t pos = 0;
        while ((pos = spk_json.find('"', pos)) != std::string::npos) {
            size_t key_start = pos + 1;
            size_t key_end = spk_json.find('"', key_start);
            if (key_end == std::string::npos) break;
            std::string key = spk_json.substr(key_start, key_end - key_start);
            size_t colon = spk_json.find(':', key_end);
            if (colon == std::string::npos) break;
            int val = std::atoi(spk_json.c_str() + colon + 1);
            config.spk_ids[key] = val;
            pos = spk_json.find(',', colon);
            if (pos == std::string::npos) break;
            pos++;
        }
        printf("[talker_embed] loaded %zu speaker IDs (CustomVoice)\n",
               config.spk_ids.size());
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
#if defined(QWEN_TTS_HAS_CP_CANN)
    if (cp_cann_engine_) {
        delete cp_cann_engine_;
        cp_cann_engine_ = nullptr;
    }
    if (talker_cann_engine_) {
        delete talker_cann_engine_;
        talker_cann_engine_ = nullptr;
    }
#endif
#if defined(QWEN_TTS_LLAMA)
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
    if (talker_step_batch_ready_) {
        llama_batch_free(talker_step_batch_);
        talker_step_batch_ready_ = false;
    }
#endif
}

// ============================================================================
// TalkerLLM: load_model
// ============================================================================

bool TalkerLLM::load_model(const std::string &talker_llama_path,
                             const std::string &talker_embed_path,
                             const std::string &code_predictor_path,
                             int n_threads,
                             int n_gpu_layers,
                             const std::string &cp_llama_path,
                             bool use_cp_cann,
                             bool use_talker_cann) {
    // 1. Load custom embedding/head tensors from original Talker GGUF
    ContextParams embed_params;
    embed_params.n_threads = n_threads;
    embed_session_ =
        std::make_unique<InferenceSession<TalkerEmbeddingModel>>(
            talker_embed_path, embed_params);
    talker_config_ = embed_session_->get_model().config;

    // 2. Load Talker backbone via llama.cpp (llama-compatible GGUF)
#if defined(QWEN_TTS_LLAMA)
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
    ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

    llama_ctx_ = llama_init_from_model(llama_model_, ctx_params);
    if (!llama_ctx_) {
        printf("[talker] failed to create llama context\n");
        return false;
    }
    printf("[talker] llama.cpp backbone loaded\n");
#else
    // Built without QWEN_TTS_LLAMA — native Talker CANN engine is the only
    // backbone path. n_embd_ is populated from the embedding GGUF below.
    n_embd_ = talker_config_.hidden_size;
    printf("[talker] llama.cpp disabled at build time (QWEN_TTS_LLAMA=OFF); "
           "native path required\n");
    (void)n_gpu_layers;
#endif

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
#if defined(QWEN_TTS_LLAMA)
    if (!cp_llama_path.empty()) {
        printf("[talker] loading CP llama.cpp backend from %s\n",
               cp_llama_path.c_str());

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
            cp_ctx_params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;

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
#else
    if (!cp_llama_path.empty()) {
        printf("[talker] --cp_model ignored: built without QWEN_TTS_LLAMA "
               "(rebuild with -DQWEN_TTS_LLAMA=ON to enable llama.cpp CP path)\n");
    }
#endif

    // 5. Optionally enable native CP CANN engine (direct ACL ops).
    //    Takes precedence over llama.cpp CP path when both are requested.
#if defined(QWEN_TTS_HAS_CP_CANN)
    if (use_cp_cann) {
        init_cp_f32_weights();
        cp_cann_engine_ = new CpCannEngine();
        // Stretch S1: TALKER_W8_QUANT=1 env toggles A16W8 INT8 weight matmul
        // in the CP engine. Mirrors the existing TALKER_NZ_WEIGHTS pattern
        // inside CpCannEngine::init — we call the setter here so the flag is
        // live before init() runs (init re-checks the env and honours either
        // path). Empty / "0" values count as "off" to match the engine-side
        // gating.
        {
            const char *w8_env = getenv("TALKER_W8_QUANT");
            if (w8_env && w8_env[0] != '\0' && w8_env[0] != '0') {
                cp_cann_engine_->set_use_w8_weights(true);
            }
        }
        // CpWeightsF32 mirrors TalkerLLM::CPWeightsF32 field-for-field; this
        // avoids a deep copy but relies on the two structs staying in sync.
        const CpWeightsF32 &cp_view =
            *reinterpret_cast<const CpWeightsF32 *>(&cp_f32_);
        if (cp_cann_engine_->init(cp_view, cp_config_, 0)) {
            cp_use_cann_ = true;
            // Native CANN owns CP now; release the llama.cpp CP context to
            // avoid holding duplicate NPU KV buffers.
#if defined(QWEN_TTS_LLAMA)
            if (cp_llama_ctx_) {
                llama_free(cp_llama_ctx_);
                cp_llama_ctx_ = nullptr;
            }
            if (cp_llama_model_) {
                llama_model_free(cp_llama_model_);
                cp_llama_model_ = nullptr;
            }
#endif
            cp_use_llama_ = false;
        } else {
            printf("[talker] WARN: CP CANN engine init failed, "
                   "falling back to %s path\n",
                   cp_use_llama_ ? "llama.cpp" : "CPU");
            delete cp_cann_engine_;
            cp_cann_engine_ = nullptr;
        }
    }
#else
    (void)use_cp_cann;
#endif

    printf("[talker] all components loaded (hidden=%d, groups=%d, cp_threads=%d, cp=%s)\n",
           n_embd_, talker_config_.num_code_groups, cp_n_threads,
           cp_use_cann_ ? "cann-native" : (cp_use_llama_ ? "llama.cpp" : "cpu"));

    // 6. Optionally enable native Talker backbone engine.
    //    When active, it replaces llama.cpp for the main Talker transformer.
#if defined(QWEN_TTS_HAS_CP_CANN)
    if (use_talker_cann) {
        talker_cann_engine_ = new TalkerCannEngine();
        // Stretch S1: TALKER_W8_QUANT=1 env toggles A16W8 INT8 weight matmul
        // in the Talker backbone. Engine's init_from_gguf reads the same env
        // var internally; calling the setter here makes the opt-in explicit
        // at the call site for readability. Same empty/"0" = "off" semantics.
        {
            const char *w8_env = getenv("TALKER_W8_QUANT");
            if (w8_env && w8_env[0] != '\0' && w8_env[0] != '0') {
                talker_cann_engine_->set_use_w8_weights(true);
            }
        }
        if (talker_cann_engine_->init_from_gguf(talker_llama_path,
                                                  talker_config_, 0)) {
            talker_use_cann_ = true;
            printf("[talker] native TalkerCannEngine initialized\n");
        } else {
            printf("[talker] WARN: native TalkerCannEngine init failed, "
                   "falling back to llama.cpp Talker\n");
            delete talker_cann_engine_;
            talker_cann_engine_ = nullptr;
        }
    }
#else
    (void)use_talker_cann;
#endif

#if !defined(QWEN_TTS_LLAMA)
    // Built without llama.cpp: the native TalkerCannEngine is the only
    // backbone path. If the caller disabled it (e.g. --llama_fallback),
    // we have nothing to run.
    if (!talker_use_cann_) {
        printf("[talker] llama.cpp fallback not compiled in "
               "(build with -DQWEN_TTS_LLAMA=ON)\n");
        return false;
    }
#endif
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
    int &seq_len) {

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

    // --- Build ICL text embeddings: text_proj(ref_text ++ target_text) + tts_eos ---
    int ref_text_len = (int)ref_text_tokens.size();
    int target_text_len = (int)target_text_tokens.size();
    int text_lens = ref_text_len + target_text_len + 1;  // +1 for tts_eos

    std::vector<float> text_embeds(text_lens * dim);
    for (int i = 0; i < ref_text_len; i++)
        lookup_text_projected(ref_text_tokens[i], text_embeds.data() + (size_t)i * dim);
    for (int i = 0; i < target_text_len; i++)
        lookup_text_projected(target_text_tokens[i],
                               text_embeds.data() + (size_t)(ref_text_len + i) * dim);
    memcpy(text_embeds.data() + (size_t)(text_lens - 1) * dim,
           tts_eos_embed_.data(), dim * sizeof(float));

    // --- Build ICL codec embeddings: codec_bos + sum_of_groups(ref_code) ---
    int ref_frames = ref_codes.empty() ? 0 : (int)ref_codes[0].size();
    int codec_lens = 1 + ref_frames;  // codec_bos + ref frames

    std::vector<float> codec_embeds(codec_lens * dim);
    lookup_codec_embedding(cfg.codec_bos_id, codec_embeds.data());
    for (int f = 0; f < ref_frames; f++)
        compute_ref_frame_embedding(ref_codes, f,
                                     codec_embeds.data() + (size_t)(1 + f) * dim);

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

    // Section 1: Role prefix (text only)
    for (int i = 0; i < role_len; i++) {
        lookup_text_projected(role_ids[i],
                               embeddings.data() + (size_t)pos * dim);
        pos++;
    }

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
// Public C API accessors
// ============================================================================

void TalkerLLM::reset_cache_public() {
#if defined(QWEN_TTS_HAS_CP_CANN)
    if (talker_use_cann_ && talker_cann_engine_) {
        talker_cann_engine_->reset_kv_cache();
    }
#endif
#if defined(QWEN_TTS_LLAMA)
    if (llama_ctx_) {
        llama_memory_clear(llama_get_memory(llama_ctx_), true);
    }
#endif
    api_cur_pos_ = 0;
}

int TalkerLLM::forward_public(const float *input_embeds, int seq_len,
                               float *logits_out, float *hidden_out) {
    int dim = n_embd_;
    cache_tts_embeddings();

#if defined(QWEN_TTS_HAS_CP_CANN)
    // Native path: bypass llama.cpp entirely for the Talker backbone.
    // Prefill: seq_len > 1 → forward_prefill, advance cursor by seq_len.
    // Decode:  seq_len == 1 → forward_decode at current position.
    if (talker_use_cann_ && talker_cann_engine_) {
        std::vector<float> hidden(dim);
        if (seq_len == 1) {
            talker_cann_engine_->forward_decode(input_embeds, api_cur_pos_,
                                                 hidden.data());
            api_cur_pos_ += 1;
        } else {
            talker_cann_engine_->forward_prefill(input_embeds, seq_len,
                                                  api_cur_pos_,
                                                  hidden.data());
            api_cur_pos_ += seq_len;
        }
        if (hidden_out) {
            memcpy(hidden_out, hidden.data(), dim * sizeof(float));
        }
        if (logits_out) {
            apply_codec_head(hidden.data(), logits_out);
        }
        return 0;
    }
#endif

#if defined(QWEN_TTS_LLAMA)
    if (!llama_ctx_) return -1;

    llama_batch batch = llama_batch_init(seq_len, dim, 1);
    batch.n_tokens = seq_len;
    memcpy(batch.embd, input_embeds, (size_t)seq_len * dim * sizeof(float));

    // For MROPE: position tensor needs 4× entries [temporal, h, w, extra]
    // TTS uses temporal only: h=0, w=0, extra=0
    bool use_mrope = talker_config_.language_ids.size() > 0 &&
                     !talker_config_.language_ids.empty(); // crude check; real check is rope_sections
    // Actually just check if we have mrope sections
    // For safety, always zero-fill 4× positions when mrope is possible
    // llama_batch_init allocates pos for n_tokens, but llama.cpp may read 4*n_tokens
    // Reallocate pos to 4× size
    free(batch.pos);
    batch.pos = (llama_pos *)calloc(4 * seq_len, sizeof(llama_pos));  // calloc zeros all
    for (int i = 0; i < seq_len; i++) {
        batch.pos[i] = api_cur_pos_ + i;  // temporal positions
        // batch.pos[seq_len + i] = 0;    // h = 0 (already zeroed by calloc)
        // batch.pos[2*seq_len + i] = 0;  // w = 0
        // batch.pos[3*seq_len + i] = 0;  // extra = 0
    }
    for (int i = 0; i < seq_len; i++) {
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == seq_len - 1) ? 1 : 0;
    }

    int rc = llama_decode(llama_ctx_, batch);
    llama_batch_free(batch);
    if (rc != 0) return rc;

    api_cur_pos_ += seq_len;

    // Extract hidden state
    float *embd = llama_get_embeddings_ith(llama_ctx_, -1);
    if (!embd) return -1;

    if (hidden_out) {
        memcpy(hidden_out, embd, dim * sizeof(float));
    }

    // Apply codec head → logits
    if (logits_out) {
        apply_codec_head(embd, logits_out);
    }

    return 0;
#else
    (void)input_embeds;
    (void)seq_len;
    (void)logits_out;
    (void)hidden_out;
    printf("[talker] llama.cpp fallback not compiled in "
           "(build with -DQWEN_TTS_LLAMA=ON)\n");
    return -1;
#endif
}

// ============================================================================
// X-Vector voice clone prefill (matches MLX build_voice_clone_prefill_embedding)
// 10 positions: [role(3)] + [codec_overlay(4)] + [spk(1)] + [bos+pad(1)] + [first_text+bos(1)]
// ============================================================================

bool TalkerLLM::build_xvec_prefill(
    const std::vector<int> &target_text_tokens,
    const std::vector<float> &spk_embedding,
    const std::string &language,
    std::vector<float> &embeddings,
    int &seq_len) {

    int dim = n_embd_;
    auto &cfg = talker_config_;
    cache_tts_embeddings();

    // Resolve language ID
    auto lang_it = cfg.language_ids.find(language);
    if (lang_it == cfg.language_ids.end()) {
        printf("[talker] unknown language: %s\n", language.c_str());
        return false;
    }
    int lang_id = lang_it->second;

    // codec prefix: [think, think_bos, lang, think_eos] (4 tokens, voice_design style)
    std::vector<int> codec_prefix = {
        cfg.codec_think_id,
        cfg.codec_think_bos_id,
        lang_id,
        cfg.codec_think_eos_id,
    };

    seq_len = 10;  // role(3) + codec_overlay(4) + spk(1) + bos+pad(1) + first_text+bos(1)
    embeddings.resize((size_t)seq_len * dim, 0.0f);

    // Precompute trailing text for autoregressive loop
    // x-vector mode: trailing = text_tokens[1:] + tts_eos + tts_pad*...
    if (target_text_tokens.size() > 1) {
        trailing_text_len_ = (int)target_text_tokens.size(); // includes first token already consumed
        trailing_text_.resize((size_t)trailing_text_len_ * dim);
        for (int i = 1; i < (int)target_text_tokens.size(); i++) {
            lookup_text_projected(target_text_tokens[i],
                                   trailing_text_.data() + (size_t)(i - 1) * dim);
        }
        // Last trailing position: tts_eos
        memcpy(trailing_text_.data() + (size_t)(trailing_text_len_ - 1) * dim,
               tts_eos_embed_.data(), dim * sizeof(float));
    } else {
        trailing_text_len_ = 0;
    }

    int pos = 0;

    // --- Positions 0-2: Role prefix (text_proj only, NO codec) ---
    int role_ids[] = {im_start_token_id, 77091, 198};  // <|im_start|> assistant \n
    for (int i = 0; i < 3; i++) {
        lookup_text_projected(role_ids[i], embeddings.data() + (size_t)pos * dim);
        pos++;
    }

    // --- Positions 3-6: Codec prefix overlaid on tts_pad ---
    for (int i = 0; i < 4; i++) {
        float *dst = embeddings.data() + (size_t)pos * dim;
        // text_proj(tts_pad) + codec_embed(prefix[i])
        memcpy(dst, tts_pad_embed_.data(), dim * sizeof(float));
        // Add codec embedding
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(codec_prefix[i], codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    // --- Position 7: Speaker embedding + tts_pad ---
    {
        float *dst = embeddings.data() + (size_t)pos * dim;
        memcpy(dst, tts_pad_embed_.data(), dim * sizeof(float));
        // Add speaker embedding (element-wise, broadcasting if enc_dim == dim)
        int spk_dim = (int)spk_embedding.size();
        int copy_dim = std::min(spk_dim, dim);
        for (int j = 0; j < copy_dim; j++) dst[j] += spk_embedding[j];
        pos++;
    }

    // --- Position 8: tts_bos + codec_pad ---
    {
        float *dst = embeddings.data() + (size_t)pos * dim;
        memcpy(dst, tts_bos_embed_.data(), dim * sizeof(float));
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(cfg.codec_pad_id, codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    // --- Position 9: first_text_token + codec_bos ---
    {
        float *dst = embeddings.data() + (size_t)pos * dim;
        int first_text = target_text_tokens.empty() ? tts_pad_token_id : target_text_tokens[0];
        lookup_text_projected(first_text, dst);
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(cfg.codec_bos_id, codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    printf("[talker] x-vector prefill: seq_len=%d (role=3, codec=4, spk=1, bos=1, text=1)\n", seq_len);
    return true;
}

// ============================================================================
// CustomVoice prefill (built-in speakers — discrete spk_id token)
// 10 positions: [role(3)] + [codec_prefix(5) with spk_id] + [bos+pad(1)] + [first_text+bos(1)]
// ============================================================================

bool TalkerLLM::build_customvoice_prefill(
    const std::vector<int> &target_text_tokens,
    int spk_token_id,
    const std::string &language,
    std::vector<float> &embeddings,
    int &seq_len) {

    int dim = n_embd_;
    auto &cfg = talker_config_;
    cache_tts_embeddings();

    auto lang_it = cfg.language_ids.find(language);
    if (lang_it == cfg.language_ids.end()) {
        printf("[talker] unknown language: %s\n", language.c_str());
        return false;
    }
    int lang_id = lang_it->second;

    // codec prefix: [think, think_bos, lang, think_eos, spk_id] (5 tokens)
    std::vector<int> codec_prefix = {
        cfg.codec_think_id,
        cfg.codec_think_bos_id,
        lang_id,
        cfg.codec_think_eos_id,
        spk_token_id,
    };

    seq_len = 10;  // role(3) + codec_overlay(5) + bos+pad(1) + first_text+bos(1)
    embeddings.resize((size_t)seq_len * dim, 0.0f);

    // Precompute trailing text
    if (target_text_tokens.size() > 1) {
        trailing_text_len_ = (int)target_text_tokens.size();
        trailing_text_.resize((size_t)trailing_text_len_ * dim);
        for (int i = 1; i < (int)target_text_tokens.size(); i++) {
            lookup_text_projected(target_text_tokens[i],
                                   trailing_text_.data() + (size_t)(i - 1) * dim);
        }
        memcpy(trailing_text_.data() + (size_t)(trailing_text_len_ - 1) * dim,
               tts_eos_embed_.data(), dim * sizeof(float));
    } else {
        trailing_text_len_ = 0;
    }

    int pos = 0;

    // --- Positions 0-2: Role prefix ---
    int role_ids[] = {im_start_token_id, 77091, 198};
    for (int i = 0; i < 3; i++) {
        lookup_text_projected(role_ids[i], embeddings.data() + (size_t)pos * dim);
        pos++;
    }

    // --- Positions 3-7: Codec prefix (5 tokens) overlaid on tts_pad ---
    for (int i = 0; i < 5; i++) {
        float *dst = embeddings.data() + (size_t)pos * dim;
        memcpy(dst, tts_pad_embed_.data(), dim * sizeof(float));
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(codec_prefix[i], codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    // --- Position 8: tts_bos + codec_pad ---
    {
        float *dst = embeddings.data() + (size_t)pos * dim;
        memcpy(dst, tts_bos_embed_.data(), dim * sizeof(float));
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(cfg.codec_pad_id, codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    // --- Position 9: first_text_token + codec_bos ---
    {
        float *dst = embeddings.data() + (size_t)pos * dim;
        int first_text = target_text_tokens.empty() ? tts_pad_token_id : target_text_tokens[0];
        lookup_text_projected(first_text, dst);
        std::vector<float> codec_emb(dim);
        lookup_codec_embedding(cfg.codec_bos_id, codec_emb.data());
        for (int j = 0; j < dim; j++) dst[j] += codec_emb[j];
        pos++;
    }

    printf("[talker] customvoice prefill: seq_len=%d (role=3, codec=5, bos=1, text=1), spk_id=%d\n",
           seq_len, spk_token_id);
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

    // 2. Transformer layers (optionally limited by cp_active_layers_)
    int n_layers = cfg.num_hidden_layers;
    if (cp_active_layers_ > 0 && cp_active_layers_ < n_layers)
        n_layers = cp_active_layers_;
    for (int il = 0; il < n_layers; il++) {
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

    int n_groups_full = cp_config_.num_code_groups - 1;  // 15
    int n_groups = (sampling.cp_max_groups > 0 && sampling.cp_max_groups < n_groups_full)
                    ? sampling.cp_max_groups : n_groups_full;
    int talker_hidden = cp_config_.talker_hidden_size;  // 2048
    int cp_hidden = cp_config_.hidden_size;  // 1024
    int vocab_size = cp_config_.vocab_size;  // 2048

    // Set active CP layer count (0=all)
    cp_active_layers_ = sampling.cp_max_layers;
    // Zero-fill skipped groups — decoder sums contributions, zero = no effect.
    // Note: EOS may not fire when groups are skipped because the model wasn't
    // trained with partial codebooks. Use --max_tokens to limit output length.
    group_tokens.resize(n_groups_full, 0);

    auto &cp_model = cp_session_->get_model();

    // Pre-convert weights on first call (needed for input_proj + lm_heads in both paths)
    init_cp_f32_weights();

#if defined(QWEN_TTS_HAS_CP_CANN)
    if (cp_use_cann_ && cp_cann_engine_ && cp_cann_engine_->is_ready()) {
        // ============================================================
        // Native CANN path: CP transformer runs entirely on NPU via
        // direct ACL ops. input_proj + lm_heads stay on CPU (the engine
        // applies input_proj internally but we still compute logits here).
        //
        // M6.2 Track H: dispatch every `forward_one_token` as a launch+fetch
        // pair so the CP ops queue onto the engine's current stream
        // (the orchestrator may have swapped it to Talker's `stream_b_`
        // for cross-stream overlap) and the final event survives after
        // this function returns — the Talker forward_decode_launch that
        // follows fences against it via aclrtStreamWaitEvent.  In the
        // single-stream case the behaviour is identical to the previous
        // serial forward_one_token calls.
        // ============================================================
        auto &cp_model_cann = cp_session_->get_model();
        // Propagate --cp_layers to the native engine (CLI flag was only
        // wired into the llama.cpp path before).
        cp_cann_engine_->set_active_layers(sampling.cp_max_layers);
        cp_cann_engine_->reset_kv_cache();

        std::vector<float> cp_out(cp_hidden);

        // Position 0: talker hidden state (engine does input_proj internally)
        cp_cann_engine_->forward_one_token_launch(hidden_states, 0,
                                                   /*wait*/ nullptr);
        cp_cann_engine_->forward_one_token_fetch(cp_out.data());

        // Position 1: group-0 codec embedding in talker space
        std::vector<float> g0_emb(talker_hidden);
        lookup_codec_embedding(group0_token, g0_emb.data());
        cp_cann_engine_->forward_one_token_launch(g0_emb.data(), 1,
                                                   /*wait*/ nullptr);
        cp_cann_engine_->forward_one_token_fetch(cp_out.data());

        // Decode groups 1-15 autoregressively
        std::vector<float> logits(vocab_size);
        std::vector<float> group_emb(talker_hidden);

        for (int g = 0; g < n_groups; g++) {
            cp_matvec_f32(cp_f32_.lm_head_w[g].data(), nullptr,
                          cp_out.data(), logits.data(), vocab_size, cp_hidden);

            int sampled = sample_token(logits.data(), vocab_size,
                                       sampling.cp_temperature, sampling.cp_top_k,
                                       sampling.cp_top_p, sampling.cp_do_sample);
            group_tokens[g] = sampled;

            if (g < n_groups - 1) {
                read_embedding_row(cp_model_cann.codec_embeddings_[g],
                                   sampled, group_emb.data(), talker_hidden);
                // Launch the next CP forward; the fetch below serialises on
                // the forward_done_event_ recorded by the launch, so host
                // work above (lm_head matvec + sampling + embedding
                // lookup for group g+1) can overlap with the NPU chew of
                // group g's forward.
                cp_cann_engine_->forward_one_token_launch(group_emb.data(),
                                                           g + 2,
                                                           /*wait*/ nullptr);
                cp_cann_engine_->forward_one_token_fetch(cp_out.data());
            }
        }
        return true;
    }
#endif

    if (cp_use_llama_) {
#if defined(QWEN_TTS_LLAMA)
        // ============================================================
        // NPU path: CP transformer via llama.cpp
        // input_proj and lm_heads remain on CPU (NEON-accelerated)
        // ============================================================
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
        // Optimization: reuse a single batch object to avoid alloc/free per iteration
        llama_batch step_batch = llama_batch_init(1, cp_hidden, 1);
        step_batch.n_tokens = 1;
        step_batch.n_seq_id[0] = 1;
        step_batch.seq_id[0][0] = 0;
        step_batch.logits[0] = 1;

        for (int g = 0; g < n_groups; g++) {
            // Apply lm_head on CPU (NEON: [2048, 1024] × [1024] → [2048])
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

                memcpy(step_batch.embd, projected.data(), cp_hidden * sizeof(float));
                step_batch.pos[0] = g + 2;
                llama_decode(cp_llama_ctx_, step_batch);

                llama_embd = llama_get_embeddings_ith(cp_llama_ctx_, -1);
                memcpy(cp_out.data(), llama_embd, cp_hidden * sizeof(float));
            }
        }
        llama_batch_free(step_batch);

        return true;
#else
        printf("[talker] llama.cpp CP fallback not compiled in "
               "(build with -DQWEN_TTS_LLAMA=ON)\n");
        return false;
#endif
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
// TalkerLLM: ensure_talker_step_batch (pre-allocate reusable batch)
// ============================================================================

void TalkerLLM::ensure_talker_step_batch() {
#if defined(QWEN_TTS_LLAMA)
    if (talker_step_batch_ready_) return;
    int dim = n_embd_;
    talker_step_batch_ = llama_batch_init(1, dim, 1);
    talker_step_batch_.n_tokens = 1;
    talker_step_batch_.n_seq_id[0] = 1;
    talker_step_batch_.seq_id[0][0] = 0;
    talker_step_batch_.logits[0] = 1;
    talker_step_batch_ready_ = true;
#endif
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

    // NOTE: Do NOT add tts_pad here. The caller adds the text component
    // (trailing_text or tts_pad) separately, matching the Rust/MLX implementation:
    //   codec0_embed + codec_1_15_sum + text_embed
    // Adding tts_pad here AND in the caller causes double-addition corruption.
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
    const TalkerSamplingParams &sampling) {

    const bool use_native_talker =
#if defined(QWEN_TTS_HAS_CP_CANN)
        talker_use_cann_ && talker_cann_engine_ != nullptr;
#else
        false;
#endif
#if defined(QWEN_TTS_LLAMA)
    if (!use_native_talker && !llama_ctx_) {
        printf("[talker] neither llama.cpp backbone nor native Talker loaded\n");
        return false;
    }
#else
    if (!use_native_talker) {
        printf("[talker] llama.cpp fallback not compiled in "
               "(build with -DQWEN_TTS_LLAMA=ON)\n");
        return false;
    }
#endif

    int dim = n_embd_;
    int n_groups = talker_config_.num_code_groups;

    auto gen_t0 = std::chrono::high_resolution_clock::now();

    // Ensure TTS embeddings are cached
    cache_tts_embeddings();

    // Initialize output: [n_groups][T]
    codec_tokens.resize(n_groups);

    // 1. Build prefill embeddings
    std::vector<float> prefill_embs;
    int prefill_len = 0;
    if (!build_input_embeddings(ref_text_tokens, target_text_tokens,
                                 spk_embedding, ref_codes,
                                 language, prefill_embs, prefill_len)) {
        return false;
    }

    auto build_emb_t1 = std::chrono::high_resolution_clock::now();
    double build_emb_ms = std::chrono::duration<double, std::milli>(build_emb_t1 - gen_t0).count();

    // 2. Prefill: feed all embeddings through the Talker backbone.
    std::vector<float> hidden(dim);
    printf("[talker] prefill %d tokens (dim=%d)...\n", prefill_len, dim);
    auto prefill_t0 = std::chrono::high_resolution_clock::now();
#if defined(QWEN_TTS_HAS_CP_CANN)
    if (use_native_talker) {
        talker_cann_engine_->reset_kv_cache();
        talker_cann_engine_->forward_prefill(prefill_embs.data(), prefill_len,
                                              0, hidden.data());
    } else
#endif
    {
#if defined(QWEN_TTS_LLAMA)
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
        int decode_rc = llama_decode(llama_ctx_, batch);
        if (decode_rc != 0) {
            printf("[talker] prefill failed\n");
            llama_batch_free(batch);
            return false;
        }
        llama_batch_free(batch);

        float *embd_llama = llama_get_embeddings_ith(llama_ctx_, -1);
        if (!embd_llama) {
            printf("[talker] failed to get embeddings after prefill\n");
            return false;
        }
        memcpy(hidden.data(), embd_llama, dim * sizeof(float));
#else
        printf("[talker] llama.cpp fallback not compiled in "
               "(build with -DQWEN_TTS_LLAMA=ON)\n");
        return false;
#endif
    }
    auto prefill_t1 = std::chrono::high_resolution_clock::now();
    double talker_prefill_ms = std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();

    // Prefill hidden-state debug probe. Set TALKER_PREFILL_DUMP=<path> and the
    // first 16 floats + norm stats + full binary are written for a
    // llama.cpp-vs-native diff. Kept behind an env var so production is free.
    if (const char *dump = getenv("TALKER_PREFILL_DUMP")) {
        double sum = 0, sumsq = 0, amin = hidden[0], amax = hidden[0];
        for (int i = 0; i < dim; i++) {
            sum += hidden[i];
            sumsq += (double)hidden[i] * hidden[i];
            if (hidden[i] < amin) amin = hidden[i];
            if (hidden[i] > amax) amax = hidden[i];
        }
        double mean = sum / dim;
        double rms  = std::sqrt(sumsq / dim);
        printf("[talker][prefill-dump] dim=%d mean=%.6f rms=%.6f min=%.6f max=%.6f"
               " first16=[", dim, mean, rms, amin, amax);
        for (int i = 0; i < 16; i++) printf("%s%.4f", i ? " " : "", hidden[i]);
        printf("]\n");
        FILE *f = fopen(dump, "wb");
        if (f) {
            fwrite(hidden.data(), sizeof(float), (size_t)dim, f);
            fclose(f);
            printf("[talker][prefill-dump] wrote %s (%d floats)\n", dump, dim);
        }
    }

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

    // Pre-allocate reusable batch for per-step talker decode
    ensure_talker_step_batch();

    // ------------------------------------------------------------------
    // M6.2 Track J — speculative-embedding ICL loop.
    //
    // Layout: Talker runs on the Talker engine's primary stream (A), CP runs
    // on `stream_b_` (redirected once via `cp_engine->set_stream`).  Key
    // idea: we launch Talker[N+1] *before* CP[N] has produced groups 1..15
    // by using a provisional input embedding that contains only g0 + text
    // (groups 1..15 contributions set to zero).  In parallel on stream B,
    // CP[N] samples the 15 groups.  As soon as CP[N]'s final fetch lands on
    // the host, we build the delta = sum_{g=1..15} codec_embed_g(tok_g) and
    // queue an F32-delta Add onto Talker[N+1]'s F16 residual input — the
    // delta-add is fenced onto CP's `forward_done_event_` so it only runs
    // after CP's last group's device-side work drained.  The Talker layers
    // then consume the corrected input and produce hidden_{N+1}.
    //
    // This gives true Talker<->CP overlap: the 28-layer Talker body on
    // stream A runs while CP's remaining groups work on stream B (since
    // both are queued before the other finishes), bounded only by the
    // serial delta-add.  The numerical approximation is bit-exact in the
    // kernel arithmetic (we truly add delta = codec_emb_1..15 onto the
    // cast input before any layer runs), so the only source of drift is
    // the order of sums (F16 add of two F16 rvalues that previously were
    // F32-summed on host).  Track J's §8 note documents the ASR/DTW
    // delta; if ASR fails we fall back to k=1 (TALKER_SPECULATIVE=0).
    //
    // Graph capture is disabled on this path — cur_dev_'s contents at `pos`
    // now depend on the delta, not just `pos`, so a captured graph at pos
    // would be stale for the next utterance.  The non-speculative fallback
    // path still uses capture when enabled.
    // ------------------------------------------------------------------
#if defined(QWEN_TTS_HAS_CP_CANN)
    const bool pipeline_base = use_native_talker &&
                                cp_use_cann_ &&
                                cp_cann_engine_ &&
                                cp_cann_engine_->is_ready();

    // Speculative-embedding toggle. Benchmarks showed the speculative
    // variant is ~1 fps SLOWER than the Track H k=1 path on single-
    // utterance workloads (26.5 vs 27.7 fps on the long utt) because the
    // Cast+InplaceAdd+fence overhead exceeds the overlap win on our short
    // CP (cp_groups=8). Default OFF; opt in with `TALKER_SPECULATIVE=1`
    // for experiments on workloads with longer CP or larger max_tokens
    // where the overlap win is expected to dominate.
    bool pipeline_speculative = false;
    if (pipeline_base) {
        const char *spec = getenv("TALKER_SPECULATIVE");
        if (spec && spec[0] == '1' && spec[1] == '\0') {
            pipeline_speculative = true;
            printf("[talker] TALKER_SPECULATIVE=1 — speculative-embedding "
                   "pipeline enabled (experimental)\n");
        }
    }
    const bool pipeline_native = pipeline_base;  // either mode needs the swap

    // Redirect the CP engine onto Talker's stream_b_ so the two engines
    // hit *physically different* NPU streams.  Both engines cooperate via
    // `set_stream`; we restore CP to its own primary stream on exit.
    if (pipeline_native) {
        cp_cann_engine_->set_stream(talker_cann_engine_->get_stream_b());
    }
#else
    const bool pipeline_native = false;
    const bool pipeline_speculative = false;
#endif

    // Shared scratch for the speculative delta (groups 1..15 sum).
    std::vector<float> delta_emb(dim);

    // Tracks whether CP's forward_done_event_ has been recorded at least
    // once in this session — before the first CP call, the event is in
    // reset state and waiting on it may block forever (runtime-dependent).
    bool cp_ever_fired = false;

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

#if defined(QWEN_TTS_HAS_CP_CANN)
        // =========================================================
        // Speculative path: Talker[N+1] launches with provisional
        // next_emb (g0 + trailing, no groups 1..15).  CP[N] runs on
        // stream B.  Delta-add patches Talker's input after CP done.
        // =========================================================
        if (pipeline_speculative && !is_special) {
            // Build provisional embedding on host: g0's codec_emb + trailing.
            // This is everything compute_next_embedding would produce if
            // groups 1..15 all held token 0 AND codec_embedding(g,0)==0 —
            // which is generally NOT true, so we patch the real groups-1..15
            // contribution on-device via add_input_delta_f32 below.
            auto emb_t0 = std::chrono::high_resolution_clock::now();
            lookup_codec_embedding(group0_token, next_emb.data());
            {
                int gen_step = (int)codec_tokens[0].size();
                if (trailing_text_len_ > 0 && gen_step < trailing_text_len_) {
                    const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
                    for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
                } else {
                    for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
                }
            }
            auto emb_t1 = std::chrono::high_resolution_clock::now();
            total_trailing_ms += std::chrono::duration<double, std::milli>(emb_t1 - emb_t0).count();

            // Launch Talker[N+1] cast on stream A.  Fence on previous
            // CP's event if present (so stream A sees any in-flight KV
            // writes from CP[N-1] that weren't host-drained yet — in
            // practice the host already synchronised inside the previous
            // `predict_code_groups` fetch, but the cross-stream fence is
            // cheap).  Skip the fence on the first step of the utterance
            // (event is in reset state; wait-semantics runtime-dependent).
            auto llm_t0 = std::chrono::high_resolution_clock::now();
            aclrtEvent prev_cp_event = cp_ever_fired
                ? cp_cann_engine_->get_forward_done_event() : nullptr;
            talker_cann_engine_->forward_decode_launch_cast(
                next_emb.data(), cur_pos, prev_cp_event);

            // Run CP[N] on stream B.  This is a blocking host call —
            // predict_code_groups issues launch/fetch pairs and host-waits
            // for each group's output before sampling the next.  When it
            // returns, CP[N]'s final `forward_done_event_` has been recorded
            // on stream B and the CP tokens are all on the host.
            auto cp_t0 = std::chrono::high_resolution_clock::now();
            std::vector<int> group_tokens;
            if (!predict_code_groups(hidden.data(), 1, group0_token,
                                      group_tokens, sampling)) {
                printf("[talker] code predictor failed at step %d\n", step);
                return false;
            }
            cp_ever_fired = true;
            auto cp_t1 = std::chrono::high_resolution_clock::now();
            total_cp_ms += std::chrono::duration<double, std::milli>(cp_t1 - cp_t0).count();

            // Build the groups 1..15 delta on host.  compute_next_embedding
            // would emit (codec_0 + sum_g codec_g) so the delta is just
            // sum_g codec_g — no tts_pad here (that's already in next_emb).
            auto dlt_t0 = std::chrono::high_resolution_clock::now();
            {
                auto &cp_model_cann = cp_session_->get_model();
                std::fill(delta_emb.begin(), delta_emb.end(), 0.0f);
                std::vector<float> tmp(dim);
                int n_gt = (int)group_tokens.size();
                for (int g = 0; g < n_gt; g++) {
                    read_embedding_row(cp_model_cann.codec_embeddings_[g],
                                        group_tokens[g], tmp.data(), dim);
                    for (int j = 0; j < dim; j++) delta_emb[j] += tmp[j];
                }
            }
            auto dlt_t1 = std::chrono::high_resolution_clock::now();
            total_emb_ms += std::chrono::duration<double, std::milli>(dlt_t1 - dlt_t0).count();

            // Patch Talker[N+1]'s F16 input on stream A, fenced on CP[N]'s
            // final event.  `aclrtStreamWaitEvent(talker_stream, cp_event)`
            // makes the subsequent Cast+InplaceAdd observe CP's KV writes
            // without a host round-trip.
            aclrtEvent cp_done = cp_cann_engine_->get_forward_done_event();
            talker_cann_engine_->add_input_delta_f32(delta_emb.data(),
                                                     cp_done);

            // Kick off the 28 layers on stream A.  Fetch blocks on the
            // decode_done_event_ recorded after the final Cast-to-F32.
            talker_cann_engine_->forward_decode_launch_layers(cur_pos);
            talker_cann_engine_->forward_decode_fetch(hidden.data());
            cur_pos++;

            // Book-keeping: push tokens after the Talker[N+1] launch so
            // the next step's logic (which tests codec_tokens[0].size())
            // sees the right generation index.
            codec_tokens[0].push_back(group0_token);
            for (int g = 0; g < (int)group_tokens.size(); g++) {
                codec_tokens[g + 1].push_back(group_tokens[g]);
            }

            auto llm_t1 = std::chrono::high_resolution_clock::now();
            total_llm_ms += std::chrono::duration<double, std::milli>(llm_t1 - llm_t0).count();
            total_loop_ms += std::chrono::duration<double, std::milli>(llm_t1 - loop_t0).count();

            if ((step + 1) % 50 == 0) {
                printf("[talker] step %d/%d, group0_token=%d, codec_frames=%zu\n",
                       step + 1, max_new_tokens, group0_token,
                       codec_tokens[0].size());
            }
            continue;  // skip the non-speculative tail below.
        }
#endif

        // -------- k=1 / non-pipelined fallback (Track H behaviour) --------
        if (!is_special) {
            // Actual codec token (0-2047): run Code Predictor for groups 1-15.
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
                const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
                for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
            } else {
                for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
            }
        }

        auto trailing_t1 = std::chrono::high_resolution_clock::now();
        total_trailing_ms += std::chrono::duration<double, std::milli>(trailing_t1 - trailing_t0).count();

        // 4e. Feed to the Talker backbone for the next hidden state.
        auto llm_t0 = std::chrono::high_resolution_clock::now();
#if defined(QWEN_TTS_HAS_CP_CANN)
        if (pipeline_native) {
            aclrtEvent cp_event = cp_cann_engine_->get_forward_done_event();
            talker_cann_engine_->forward_decode_launch(next_emb.data(),
                                                        cur_pos, cp_event);
            talker_cann_engine_->forward_decode_fetch(hidden.data());
            cur_pos++;
        } else if (use_native_talker) {
            talker_cann_engine_->forward_decode(next_emb.data(), cur_pos,
                                                  hidden.data());
            cur_pos++;
        } else
#endif
        {
#if defined(QWEN_TTS_LLAMA)
            memcpy(talker_step_batch_.embd, next_emb.data(), dim * sizeof(float));
            talker_step_batch_.pos[0] = cur_pos;

            if (llama_decode(llama_ctx_, talker_step_batch_) != 0) {
                printf("[talker] decode failed at step %d\n", step);
                return false;
            }
            cur_pos++;

            float *embd_step = llama_get_embeddings_ith(llama_ctx_, -1);
            if (!embd_step) {
                printf("[talker] failed to get embeddings at step %d\n", step);
                return false;
            }
            memcpy(hidden.data(), embd_step, dim * sizeof(float));
#else
            printf("[talker] llama.cpp fallback not compiled in "
                   "(build with -DQWEN_TTS_LLAMA=ON)\n");
            return false;
#endif
        }
        auto llm_t1 = std::chrono::high_resolution_clock::now();
        total_llm_ms += std::chrono::duration<double, std::milli>(llm_t1 - llm_t0).count();
        total_loop_ms += std::chrono::duration<double, std::milli>(llm_t1 - loop_t0).count();

        if ((step + 1) % 50 == 0) {
            printf("[talker] step %d/%d, group0_token=%d, codec_frames=%zu\n",
                   step + 1, max_new_tokens, group0_token,
                   codec_tokens[0].size());
        }
    }

#if defined(QWEN_TTS_HAS_CP_CANN)
    // Restore CP to its own primary stream so callers outside `generate`
    // (e.g. a follow-up utterance that doesn't go through the pipelined
    // path) don't end up running CP ops on a stream the Talker engine
    // doesn't own anymore.
    if (pipeline_native) {
        cp_cann_engine_->set_stream(nullptr);
    }
#endif

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

// ============================================================================
// generate_xvec: x-vector voice clone (no ref audio codec)
// ============================================================================

bool TalkerLLM::generate_xvec(
    const std::vector<int> &target_text_tokens,
    const std::vector<float> &spk_embedding,
    const std::string &language,
    std::vector<std::vector<int>> &codec_tokens,
    int max_new_tokens,
    const TalkerSamplingParams &sampling) {

#if !defined(QWEN_TTS_LLAMA)
    (void)target_text_tokens; (void)spk_embedding; (void)language;
    (void)codec_tokens; (void)max_new_tokens; (void)sampling;
    printf("[talker] x-vec mode requires llama.cpp "
           "(build with -DQWEN_TTS_LLAMA=ON; native Talker does not yet implement "
           "MRoPE 4×pos)\n");
    return false;
#else
    if (!llama_ctx_) { printf("[talker] backbone not loaded\n"); return false; }
    int dim = n_embd_;
    int n_groups = talker_config_.num_code_groups;
    auto gen_t0 = std::chrono::high_resolution_clock::now();
    cache_tts_embeddings();
    codec_tokens.resize(n_groups);

    // Build x-vector prefill
    std::vector<float> prefill_embs;
    int prefill_len = 0;
    if (!build_xvec_prefill(target_text_tokens, spk_embedding, language,
                            prefill_embs, prefill_len)) return false;

    // Prefill
    llama_memory_clear(llama_get_memory(llama_ctx_), true);
    llama_batch batch = llama_batch_init(prefill_len, dim, 1);
    batch.n_tokens = prefill_len;
    memcpy(batch.embd, prefill_embs.data(), (size_t)prefill_len * dim * sizeof(float));
    // MRoPE: position tensor needs 4× entries [temporal, h, w, extra]
    // TTS uses temporal only: h=w=extra=0
    free(batch.pos);
    batch.pos = (llama_pos *)calloc(4 * prefill_len, sizeof(llama_pos));
    for (int i = 0; i < prefill_len; i++) {
        batch.pos[i] = i;  // temporal positions; h/w/extra = 0 (calloc)
        batch.n_seq_id[i] = 1;  batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == prefill_len - 1) ? 1 : 0;
    }
    printf("[talker] x-vec prefill %d tokens (MRoPE 4×pos)...\n", prefill_len);
    if (llama_decode(llama_ctx_, batch) != 0) { llama_batch_free(batch); return false; }
    llama_batch_free(batch);

    float *embd = llama_get_embeddings_ith(llama_ctx_, -1);
    if (!embd) return false;
    std::vector<float> hidden(dim);
    memcpy(hidden.data(), embd, dim * sizeof(float));

    // Autoregressive loop (same as ICL generate)
    int cur_pos = prefill_len;
    int vocab_size = talker_config_.vocab_size;
    std::vector<float> logits_buf(vocab_size), next_emb(dim);
    std::vector<int> generated_g0;

    for (int step = 0; step < max_new_tokens; step++) {
        apply_codec_head(hidden.data(), logits_buf.data());
        suppress_special_tokens(logits_buf.data(), vocab_size, talker_config_.codec_eos_token_id);
        apply_repetition_penalty(logits_buf.data(), vocab_size, generated_g0, sampling.repetition_penalty);

        int g0 = sample_token(logits_buf.data(), vocab_size, sampling.temperature,
                               sampling.top_k, sampling.top_p, sampling.do_sample);
        generated_g0.push_back(g0);
        if (g0 == talker_config_.codec_eos_token_id) { printf("[talker] EOS at step %d\n", step); break; }

        bool is_special = (g0 >= 2048);
        if (!is_special) {
            std::vector<int> group_tokens;
            if (!predict_code_groups(hidden.data(), 1, g0, group_tokens, sampling)) return false;
            codec_tokens[0].push_back(g0);
            for (int g = 0; g < (int)group_tokens.size(); g++) codec_tokens[g + 1].push_back(group_tokens[g]);
            compute_next_embedding(g0, group_tokens, next_emb.data());
        } else {
            lookup_codec_embedding(g0, next_emb.data());
        }

        // Add trailing text
        int gen_step = (int)codec_tokens[0].size();
        if (trailing_text_len_ > 0 && gen_step < trailing_text_len_) {
            const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
            for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
        } else {
            for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
        }

        // Forward one step (reuse pre-allocated batch)
        ensure_talker_step_batch();
        memcpy(talker_step_batch_.embd, next_emb.data(), dim * sizeof(float));
        talker_step_batch_.pos[0] = cur_pos;
        if (llama_decode(llama_ctx_, talker_step_batch_) != 0) return false;
        cur_pos++;
        embd = llama_get_embeddings_ith(llama_ctx_, -1);
        if (!embd) return false;
        memcpy(hidden.data(), embd, dim * sizeof(float));
    }

    auto gen_t1 = std::chrono::high_resolution_clock::now();
    printf("[talker] x-vec generate: %d frames in %.1f ms\n",
           (int)codec_tokens[0].size(),
           std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    return true;
#endif  // QWEN_TTS_LLAMA
}

// ============================================================================
// generate_customvoice: built-in speaker (discrete token, no speaker encoder)
// ============================================================================

bool TalkerLLM::generate_customvoice(
    const std::vector<int> &target_text_tokens,
    const std::string &speaker,
    const std::string &language,
    std::vector<std::vector<int>> &codec_tokens,
    int max_new_tokens,
    const TalkerSamplingParams &sampling) {

#if !defined(QWEN_TTS_LLAMA)
    (void)target_text_tokens; (void)speaker; (void)language;
    (void)codec_tokens; (void)max_new_tokens; (void)sampling;
    printf("[talker] customvoice mode requires llama.cpp "
           "(build with -DQWEN_TTS_LLAMA=ON; native Talker does not yet implement "
           "MRoPE 4×pos)\n");
    return false;
#else
    if (!llama_ctx_) { printf("[talker] backbone not loaded\n"); return false; }

    // Resolve speaker name to token ID
    auto spk_it = talker_config_.spk_ids.find(speaker);
    if (spk_it == talker_config_.spk_ids.end()) {
        printf("[talker] unknown speaker: %s. Available:", speaker.c_str());
        for (auto &p : talker_config_.spk_ids) printf(" %s(%d)", p.first.c_str(), p.second);
        printf("\n");
        return false;
    }
    int spk_token_id = spk_it->second;

    int dim = n_embd_;
    int n_groups = talker_config_.num_code_groups;
    auto gen_t0 = std::chrono::high_resolution_clock::now();
    cache_tts_embeddings();
    codec_tokens.resize(n_groups);

    // Build CustomVoice prefill
    std::vector<float> prefill_embs;
    int prefill_len = 0;
    if (!build_customvoice_prefill(target_text_tokens, spk_token_id, language,
                                   prefill_embs, prefill_len)) return false;

    // Prefill
    llama_memory_clear(llama_get_memory(llama_ctx_), true);
    llama_batch batch = llama_batch_init(prefill_len, dim, 1);
    batch.n_tokens = prefill_len;
    memcpy(batch.embd, prefill_embs.data(), (size_t)prefill_len * dim * sizeof(float));
    for (int i = 0; i < prefill_len; i++) {
        batch.pos[i] = i;  batch.n_seq_id[i] = 1;  batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == prefill_len - 1) ? 1 : 0;
    }
    printf("[talker] customvoice prefill %d tokens (speaker=%s, id=%d)...\n",
           prefill_len, speaker.c_str(), spk_token_id);
    if (llama_decode(llama_ctx_, batch) != 0) { llama_batch_free(batch); return false; }
    llama_batch_free(batch);

    float *embd = llama_get_embeddings_ith(llama_ctx_, -1);
    if (!embd) return false;
    std::vector<float> hidden(dim);
    memcpy(hidden.data(), embd, dim * sizeof(float));

    // Autoregressive loop (same as x-vec/ICL)
    int cur_pos = prefill_len;
    int vocab_size = talker_config_.vocab_size;
    std::vector<float> logits_buf(vocab_size), next_emb(dim);
    std::vector<int> generated_g0;

    for (int step = 0; step < max_new_tokens; step++) {
        apply_codec_head(hidden.data(), logits_buf.data());
        suppress_special_tokens(logits_buf.data(), vocab_size, talker_config_.codec_eos_token_id);
        apply_repetition_penalty(logits_buf.data(), vocab_size, generated_g0, sampling.repetition_penalty);

        int g0 = sample_token(logits_buf.data(), vocab_size, sampling.temperature,
                               sampling.top_k, sampling.top_p, sampling.do_sample);
        generated_g0.push_back(g0);
        if (g0 == talker_config_.codec_eos_token_id) { printf("[talker] EOS at step %d\n", step); break; }

        bool is_special = (g0 >= 2048);
        if (!is_special) {
            std::vector<int> group_tokens;
            if (!predict_code_groups(hidden.data(), 1, g0, group_tokens, sampling)) return false;
            codec_tokens[0].push_back(g0);
            for (int g = 0; g < (int)group_tokens.size(); g++) codec_tokens[g + 1].push_back(group_tokens[g]);
            compute_next_embedding(g0, group_tokens, next_emb.data());
        } else {
            lookup_codec_embedding(g0, next_emb.data());
        }

        int gen_step = (int)codec_tokens[0].size();
        if (trailing_text_len_ > 0 && gen_step < trailing_text_len_) {
            const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
            for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
        } else {
            for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
        }

        llama_batch sb = llama_batch_init(1, dim, 1);
        sb.n_tokens = 1;  memcpy(sb.embd, next_emb.data(), dim * sizeof(float));
        sb.pos[0] = cur_pos;  sb.n_seq_id[0] = 1;  sb.seq_id[0][0] = 0;  sb.logits[0] = 1;
        if (llama_decode(llama_ctx_, sb) != 0) { llama_batch_free(sb); return false; }
        llama_batch_free(sb);
        cur_pos++;
        embd = llama_get_embeddings_ith(llama_ctx_, -1);
        if (!embd) return false;
        memcpy(hidden.data(), embd, dim * sizeof(float));
    }

    auto gen_t1 = std::chrono::high_resolution_clock::now();
    printf("[talker] customvoice generate: %d frames in %.1f ms\n",
           (int)codec_tokens[0].size(),
           std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    return true;
#endif  // QWEN_TTS_LLAMA
}

// ============================================================================
// generate_xvec_standalone: x-vector using standalone transformer (correct MRoPE)
// ============================================================================

bool TalkerLLM::generate_xvec_standalone(
    const std::vector<int> &target_text_tokens,
    const std::vector<float> &spk_embedding,
    const std::string &language,
    std::vector<std::vector<int>> &codec_tokens,
    int max_new_tokens,
    const TalkerSamplingParams &sampling,
    TtsTransformer *tfm) {

    int dim = n_embd_;
    int n_groups = talker_config_.num_code_groups;
    auto gen_t0 = std::chrono::high_resolution_clock::now();
    cache_tts_embeddings();
    codec_tokens.resize(n_groups);

    // Build x-vector prefill (same as build_xvec_prefill)
    std::vector<float> prefill_embs;
    int prefill_len = 0;
    if (!build_xvec_prefill(target_text_tokens, spk_embedding, language,
                            prefill_embs, prefill_len)) return false;

    // Forward through standalone transformer (correct MRoPE!)
    tts_transformer_reset(tfm);
    std::vector<float> hidden(dim);
    printf("[standalone] prefill %d positions through transformer...\n", prefill_len);
    auto prefill_t0 = std::chrono::high_resolution_clock::now();
    int rc = tts_transformer_forward(tfm, prefill_embs.data(), prefill_len, hidden.data());
    if (rc != 0) {
        printf("[standalone] prefill failed: %d\n", rc);
        return false;
    }
    auto prefill_t1 = std::chrono::high_resolution_clock::now();
    double prefill_ms = std::chrono::duration<double, std::milli>(prefill_t1 - prefill_t0).count();
    printf("[standalone] prefill: %.0f ms\n", prefill_ms);

    // Apply codec head → logits
    int vocab_size = talker_config_.vocab_size;
    std::vector<float> logits(vocab_size);
    apply_codec_head(hidden.data(), logits.data());

    // Autoregressive loop (same as generate_xvec but using standalone transformer)
    std::vector<float> next_emb(dim);
    std::vector<int> generated_g0;

    printf("[standalone] generating (max %d tokens)...\n", max_new_tokens);
    for (int step = 0; step < max_new_tokens; step++) {
        // Suppress special tokens
        suppress_special_tokens(logits.data(), vocab_size, talker_config_.codec_eos_token_id);
        apply_repetition_penalty(logits.data(), vocab_size, generated_g0, sampling.repetition_penalty);

        int g0 = sample_token(logits.data(), vocab_size, sampling.temperature,
                               sampling.top_k, sampling.top_p, sampling.do_sample);
        generated_g0.push_back(g0);
        if (g0 == talker_config_.codec_eos_token_id) {
            printf("[standalone] EOS at step %d\n", step);
            break;
        }

        // Code predictor for groups 1-15
        std::vector<int> group_tokens;
        if (!predict_code_groups(hidden.data(), 1, g0, group_tokens, sampling)) return false;
        codec_tokens[0].push_back(g0);
        for (int g = 0; g < (int)group_tokens.size(); g++)
            codec_tokens[g + 1].push_back(group_tokens[g]);

        // Build next embedding: codec_sum + trailing_text or tts_pad
        compute_next_embedding(g0, group_tokens, next_emb.data());
        // Add trailing text
        int gen_step = (int)codec_tokens[0].size();
        if (trailing_text_len_ > 0 && gen_step < trailing_text_len_) {
            const float *txt = trailing_text_.data() + (size_t)gen_step * dim;
            for (int j = 0; j < dim; j++) next_emb[j] += txt[j];
        } else {
            for (int j = 0; j < dim; j++) next_emb[j] += tts_pad_embed_[j];
        }

        // Forward one step through standalone transformer
        rc = tts_transformer_forward(tfm, next_emb.data(), 1, hidden.data());
        if (rc != 0) { printf("[standalone] forward failed at step %d\n", step); return false; }
        apply_codec_head(hidden.data(), logits.data());
    }

    auto gen_t1 = std::chrono::high_resolution_clock::now();
    printf("[standalone] xvec generate: %d frames in %.1f ms\n",
           (int)codec_tokens[0].size(),
           std::chrono::duration<double, std::milli>(gen_t1 - gen_t0).count());
    return true;
}
