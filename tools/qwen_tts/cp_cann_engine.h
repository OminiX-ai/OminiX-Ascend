#pragma once
// ============================================================================
// CP CANN Engine: Direct ACL-based Code Predictor inference on Ascend NPU
//
// Replaces both llama.cpp (3ms/step overhead) and CPU (fast but no NPU)
// paths with pre-allocated NPU buffers and direct aclnn* op calls.
//
// All intermediate buffers are allocated once at init time.
// Forward pass issues only aclnn kernel launches -- zero allocation.
// ============================================================================

#include <acl/acl.h>
#include <string>
#include <vector>

struct CodePredictorConfig;

// Standalone mirror of TalkerLLM::CPWeightsF32 -- same layout, avoids
// reaching into TalkerLLM private scope.  Caller can reinterpret_cast
// or copy the fields from TalkerLLM::cp_f32_.
struct CpWeightsF32 {
    std::vector<float> proj_w, proj_b;
    struct Layer {
        std::vector<float> q_proj_w, k_proj_w, v_proj_w, o_proj_w;
        std::vector<float> q_norm_w, k_norm_w;
        std::vector<float> gate_proj_w, up_proj_w, down_proj_w;
        std::vector<float> input_ln_w, post_ln_w;
    };
    std::vector<Layer> layers;
    std::vector<float> norm_w;
    std::vector<std::vector<float>> lm_head_w;
};

class CpCannEngine {
public:
    CpCannEngine() = default;
    ~CpCannEngine();

    // Initialize: upload all weights to NPU, allocate work buffers.
    // `cp_f32` must already be populated (call init_cp_f32_weights first).
    // `device` is the ACL device ID (usually 0).
    bool init(const CpWeightsF32 &cp_f32, const CodePredictorConfig &cfg,
              int device = 0);

    // Process one token through the CP transformer.
    // input_talker_space: [talker_hidden] F32 on HOST
    // pos: KV cache position (0-based)
    // hidden_out: [cp_hidden] F32 on HOST
    void forward_one_token(const float *input_talker_space, int pos,
                           float *hidden_out);

    // Reset KV cache for a new frame
    void reset_kv_cache();

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int device_ = 0;
    aclrtStream stream_ = nullptr;

    // Model dimensions (cached from config)
    int talker_hidden_ = 0;  // 2048
    int cp_hidden_ = 0;      // 1024
    int n_heads_ = 0;        // 16
    int n_kv_ = 0;           // 8
    int head_dim_ = 0;       // 128
    int q_dim_ = 0;          // 2048
    int kv_dim_ = 0;         // 1024
    int inter_ = 0;          // 3072
    int n_layers_ = 0;       // 5
    float eps_ = 0.0f;
    float rope_theta_ = 0.0f;
    static constexpr int MAX_SEQ = 17;  // matches CP_MAX_SEQ

    // ---- NPU weight buffers (persistent, uploaded once) ----
    void *proj_w_dev_ = nullptr;   // [cp_hidden, talker_hidden]
    void *proj_b_dev_ = nullptr;   // [cp_hidden]

    struct LayerWeights {
        void *q_proj_w = nullptr;  // [q_dim, cp_hidden]
        void *k_proj_w = nullptr;  // [kv_dim, cp_hidden]
        void *v_proj_w = nullptr;  // [kv_dim, cp_hidden]
        void *o_proj_w = nullptr;  // [cp_hidden, q_dim]
        void *q_norm_w = nullptr;  // [head_dim]
        void *k_norm_w = nullptr;  // [head_dim]
        void *gate_proj_w = nullptr;  // [inter, cp_hidden]
        void *up_proj_w = nullptr;    // [inter, cp_hidden]
        void *down_proj_w = nullptr;  // [cp_hidden, inter]
        void *input_ln_w = nullptr;   // [cp_hidden]
        void *post_ln_w = nullptr;    // [cp_hidden]
    };
    std::vector<LayerWeights> layer_w_;
    void *final_norm_w_dev_ = nullptr;  // [cp_hidden]

    // ---- NPU intermediate buffers (pre-allocated, reused every forward) ----
    void *cur_dev_ = nullptr;       // [cp_hidden]
    void *residual_dev_ = nullptr;  // [cp_hidden]
    void *normed_dev_ = nullptr;    // [cp_hidden]
    void *q_dev_ = nullptr;         // [q_dim]
    void *k_dev_ = nullptr;         // [kv_dim]
    void *v_dev_ = nullptr;         // [kv_dim]
    void *attn_out_dev_ = nullptr;  // [q_dim]
    void *o_out_dev_ = nullptr;     // [cp_hidden]
    void *gate_dev_ = nullptr;      // [inter]
    void *up_dev_ = nullptr;        // [inter]
    void *ffn_out_dev_ = nullptr;   // [cp_hidden]

    // Attention intermediates
    void *scores_dev_ = nullptr;    // [n_heads, MAX_SEQ]
    void *attn_weights_dev_ = nullptr; // [n_heads, MAX_SEQ]

    // RoPE cos/sin table: precomputed for all positions [MAX_SEQ, head_dim/2]
    void *rope_cos_dev_ = nullptr;
    void *rope_sin_dev_ = nullptr;

    // RmsNorm rstd scratch: [1]
    void *rstd_dev_ = nullptr;

    // ---- KV cache on NPU [n_layers][MAX_SEQ * kv_dim] ----
    std::vector<void *> k_cache_dev_;  // per-layer K cache
    std::vector<void *> v_cache_dev_;  // per-layer V cache
    int kv_cache_len_ = 0;

    // ---- Host scratch buffers (pre-allocated, avoid heap alloc per forward) ----
    std::vector<float> q_host_;          // [q_dim]
    std::vector<float> k_host_;          // [kv_dim]
    std::vector<float> attn_out_host_;   // [q_dim]
    std::vector<float> k_cache_host_;    // [MAX_SEQ * kv_dim]  (reused per layer)
    std::vector<float> v_cache_host_;    // [MAX_SEQ * kv_dim]
    std::vector<float> score_buf_;       // [MAX_SEQ]

    // ---- Host-cached QK-norm weights (avoid device download per forward) ----
    struct LayerNormWeightsHost {
        std::vector<float> q_norm_w;  // [head_dim]
        std::vector<float> k_norm_w;  // [head_dim]
    };
    std::vector<LayerNormWeightsHost> qk_norm_host_;

    // aclnn workspace (reusable, grown as needed)
    void *workspace_dev_ = nullptr;
    size_t workspace_size_ = 0;

    // ---- Internal helpers ----
    void alloc_dev(void **ptr, size_t bytes);
    void upload(void *dev, const float *host, size_t n_floats);
    void download(float *host, const void *dev, size_t n_floats);
    void ensure_workspace(size_t needed);

    // Execute aclnn two-phase call, managing workspace
    // (Each op call site uses the CANN_OP macro below instead)

    // RoPE: apply NEOX-style rotation in-place on NPU
    // We precompute cos/sin tables, then do element-wise multiply+add
    void apply_rope_on_host(float *vec, int dim, int n_vecs, int pos);

    // Per-head RmsNorm on host (small dim=128, not worth NPU launch)
    void qk_norm_on_host(float *buf, int dim, int n_vecs,
                         const float *norm_w_host);
};
