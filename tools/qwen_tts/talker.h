#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_defs.h"
#include "model_loader.h"
#include "infer_session.hpp"
#include "llama.h"
#include <memory>
#include <string>
#include <vector>
#include <map>

// ============================================================================
// Sampling Parameters (matching Python defaults)
// ============================================================================

// Set random seed for reproducible sampling
void set_sampling_seed(uint32_t seed);

struct TalkerSamplingParams {
    float temperature = 0.9f;
    int top_k = 50;
    float top_p = 1.0f;
    float repetition_penalty = 1.05f;
    bool do_sample = true;
    // CP sampling
    float cp_temperature = 0.9f;
    int cp_top_k = 50;
    float cp_top_p = 1.0f;
    bool cp_do_sample = true;
};

// ============================================================================
// Talker Configuration
// ============================================================================

struct TalkerConfig {
    int hidden_size = 2048;
    int num_hidden_layers = 28;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int intermediate_size = 6144;
    int head_dim = 128;
    int vocab_size = 3072;          // codec vocab
    int text_vocab_size = 151936;
    int text_hidden_size = 2048;
    int num_code_groups = 16;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
    int max_position_embeddings = 32768;
    // Special token IDs
    int codec_bos_id = 2149;
    int codec_eos_token_id = 2150;
    int codec_pad_id = 2148;
    int codec_think_id = 2154;
    int codec_nothink_id = 2155;
    int codec_think_bos_id = 2156;
    int codec_think_eos_id = 2157;
    // Language IDs (parsed from JSON)
    std::map<std::string, int> language_ids;
};

// ============================================================================
// Code Predictor Configuration
// ============================================================================

struct CodePredictorConfig {
    int hidden_size = 1024;
    int num_hidden_layers = 5;
    int num_attention_heads = 16;
    int num_key_value_heads = 8;
    int intermediate_size = 3072;
    int head_dim = 128;
    int vocab_size = 2048;
    int num_code_groups = 16;
    int talker_hidden_size = 2048;
    float rope_theta = 1000000.0f;
    float rms_norm_eps = 1e-6f;
};

// ============================================================================
// Code Predictor Model (ggml, 5-layer transformer)
// ============================================================================

struct CPTransformerLayer {
    ggml_tensor *q_proj_w = nullptr;
    ggml_tensor *k_proj_w = nullptr;
    ggml_tensor *v_proj_w = nullptr;
    ggml_tensor *o_proj_w = nullptr;
    ggml_tensor *q_norm_w = nullptr;   // [head_dim]
    ggml_tensor *k_norm_w = nullptr;   // [head_dim]
    ggml_tensor *gate_proj_w = nullptr;
    ggml_tensor *up_proj_w = nullptr;
    ggml_tensor *down_proj_w = nullptr;
    ggml_tensor *input_layernorm_w = nullptr;
    ggml_tensor *post_attention_layernorm_w = nullptr;
};

class CodePredictorModel : public BaseModel {
public:
    CodePredictorConfig config;

    bool load_hparams(const ModelLoader &loader) override;
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    void reset_input_shape() override;

private:
    // Projection from Talker hidden to CP hidden
    ggml_tensor *small_to_mtp_proj_w_ = nullptr;  // [talker_hidden, cp_hidden]
    ggml_tensor *small_to_mtp_proj_b_ = nullptr;

    // Transformer layers
    std::vector<CPTransformerLayer> layers_;
    ggml_tensor *norm_w_ = nullptr;

    // Per-group codec embeddings (groups 1-15) — in Talker hidden space
    std::vector<ggml_tensor *> codec_embeddings_;  // 15 × [talker_hidden, vocab_size]

    // Per-group LM heads (groups 1-15)
    std::vector<ggml_tensor *> lm_heads_;  // 15 × [cp_hidden, vocab_size]

    // Current group index for build_graph (0-14 for groups 1-15)
    int current_group_ = 0;

    // TalkerLLM needs access to codec_embeddings_ for compute_next_embedding
    friend class TalkerLLM;

    // Graph building
    ggml_tensor *build_cp_layer(ggml_context *ctx0, ggml_tensor *x,
                                 ggml_tensor *kq_mask, ggml_tensor *pos,
                                 int il);
};

// ============================================================================
// Talker Embedding Model (handles text/codec embeddings + text_projection)
// ============================================================================

class TalkerEmbeddingModel : public BaseModel {
public:
    TalkerConfig config;

    bool load_hparams(const ModelLoader &loader) override;
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    void reset_input_shape() override;

    // Public tensors for direct access
    ggml_tensor *text_embedding_w = nullptr;   // [hidden, text_vocab]
    ggml_tensor *codec_embedding_w = nullptr;  // [hidden, codec_vocab]
    ggml_tensor *text_proj_fc1_w = nullptr;
    ggml_tensor *text_proj_fc1_b = nullptr;
    ggml_tensor *text_proj_fc2_w = nullptr;
    ggml_tensor *text_proj_fc2_b = nullptr;
    ggml_tensor *codec_head_w = nullptr;       // [hidden, codec_vocab]
    ggml_tensor *norm_w = nullptr;             // final RMSNorm
};

// ============================================================================
// High-level Talker LLM interface
// ============================================================================

class TalkerLLM {
public:
    TalkerLLM() = default;
    ~TalkerLLM();

    // Load models:
    //   talker_llama_path: llama.cpp-compatible GGUF (backbone)
    //   talker_embed_path: original Talker GGUF (embeddings + heads)
    //   code_predictor_path: Code Predictor GGUF
    bool load_model(const std::string &talker_llama_path,
                    const std::string &talker_embed_path,
                    const std::string &code_predictor_path,
                    int n_threads = 4,
                    int n_gpu_layers = 0,
                    const std::string &cp_llama_path = "");

    // Generate codec tokens (ICL voice cloning mode)
    //   ref_text_tokens: tokenized ref text content (no role prefix/suffix)
    //   target_text_tokens: tokenized target text content (no role prefix/suffix)
    bool generate(const std::vector<int> &ref_text_tokens,
                  const std::vector<int> &target_text_tokens,
                  const std::vector<float> &spk_embedding,
                  const std::vector<std::vector<int>> &ref_codes,
                  const std::string &language,
                  std::vector<std::vector<int>> &codec_tokens,
                  int max_new_tokens = 2048,
                  const TalkerSamplingParams &sampling = TalkerSamplingParams());

    const TalkerConfig &get_config() const { return talker_config_; }

    // TTS special text token IDs (from Qwen3-TTS config)
    static constexpr int tts_bos_token_id = 151672;
    static constexpr int tts_eos_token_id = 151673;
    static constexpr int tts_pad_token_id = 151671;
    static constexpr int im_start_token_id = 151644;

    // Code Predictor for groups 1-15 (public for testing)
    bool predict_code_groups(const float *hidden_states,
                              int seq_len, int group0_token,
                              std::vector<int> &group_tokens,
                              const TalkerSamplingParams &sampling = TalkerSamplingParams());

private:
    TalkerConfig talker_config_;
    CodePredictorConfig cp_config_;

    // llama.cpp context for Talker backbone
    llama_model *llama_model_ = nullptr;
    llama_context *llama_ctx_ = nullptr;
    int n_embd_ = 0;

    // CP via llama.cpp (NPU acceleration)
    llama_model *cp_llama_model_ = nullptr;
    llama_context *cp_llama_ctx_ = nullptr;
    bool cp_use_llama_ = false;

    // Custom embedding/head tensors
    std::unique_ptr<InferenceSession<TalkerEmbeddingModel>> embed_session_;

    // Code Predictor
    std::unique_ptr<InferenceSession<CodePredictorModel>> cp_session_;

    // Cached TTS special text embeddings (text_proj(text_emb(token)))
    std::vector<float> tts_pad_embed_;   // [dim]
    std::vector<float> tts_bos_embed_;   // [dim]
    std::vector<float> tts_eos_embed_;   // [dim]

    // Pre-converted F32 weights for text_projection and codec_head
    std::vector<float> tp_fc1_w_, tp_fc1_b_, tp_fc2_w_, tp_fc2_b_;
    std::vector<float> codec_head_w_;
    bool head_f32_ready_ = false;
    void init_head_f32_weights();

    // Trailing text hidden for streaming ICL generation
    std::vector<float> trailing_text_;   // [trailing_text_len_ * dim] or [dim] if just tts_pad
    int trailing_text_len_ = 0;          // 0 = use tts_pad for all steps
    bool tts_embeds_cached_ = false;
    void cache_tts_embeddings();

    // Embedding helpers (operate on raw weight data)
    void lookup_text_embedding(int token_id, float *out);
    void lookup_codec_embedding(int token_id, float *out);
    void apply_text_projection(const float *in, float *out);
    void apply_codec_head(const float *hidden, float *logits);

    // Build prefill embedding sequence (non-streaming ICL mode)
    bool build_input_embeddings(const std::vector<int> &ref_text_tokens,
                                 const std::vector<int> &target_text_tokens,
                                 const std::vector<float> &spk_embedding,
                                 const std::vector<std::vector<int>> &ref_codes,
                                 const std::string &language,
                                 std::vector<float> &embeddings,
                                 int &seq_len);

    // Compute text_proj(text_emb(token_id)) into out
    void lookup_text_projected(int token_id, float *out);

    // Compute sum of all 16 group codec embeddings for a single ref frame
    void compute_ref_frame_embedding(const std::vector<std::vector<int>> &ref_codes,
                                      int frame_idx, float *out);

    // Sum all 16 group embeddings for next decode step + tts_pad
    void compute_next_embedding(int group0_token,
                                 const std::vector<int> &group_tokens,
                                 float *out);

    // CP KV cache for incremental code prediction (avoids full recompute)
    static constexpr int CP_MAX_SEQ = 17;  // 2 base + 15 groups
    std::vector<std::vector<float>> cp_k_cache_;  // [n_layers][CP_MAX_SEQ * kv_dim]
    std::vector<std::vector<float>> cp_v_cache_;  // [n_layers][CP_MAX_SEQ * kv_dim]
    int cp_cache_len_ = 0;

    // Pre-converted F32 copies of CP weights for fast matvec
    struct CPWeightsF32 {
        std::vector<float> proj_w, proj_b;
        struct Layer {
            std::vector<float> q_proj_w, k_proj_w, v_proj_w, o_proj_w;
            std::vector<float> q_norm_w, k_norm_w;
            std::vector<float> gate_proj_w, up_proj_w, down_proj_w;
            std::vector<float> input_ln_w, post_ln_w;
        };
        std::vector<Layer> layers;
        std::vector<float> norm_w;
        std::vector<std::vector<float>> lm_head_w;  // 15 × [vocab * cp_hidden]
    };
    CPWeightsF32 cp_f32_;
    bool cp_f32_ready_ = false;
    void init_cp_f32_weights();

    // Pre-allocated CP working buffers (avoid heap allocs per forward call)
    struct CPWorkBufs {
        std::vector<float> cur, residual, normed;
        std::vector<float> q, k, v;
        std::vector<float> attn_out, o_out;
        std::vector<float> gate, up, ffn_out;
        std::vector<float> scores;
        bool ready = false;
    };
    CPWorkBufs cp_bufs_;
    void init_cp_work_bufs();

    // Process one token through CP transformer with KV caching
    void cp_forward_one_token(const float *input_talker_space,
                               int pos, float *hidden_out);
};
