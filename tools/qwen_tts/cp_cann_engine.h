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
#include <aclnn/acl_meta.h>   // aclScalar, aclTensor
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

    // Alternative init path: load weights directly from an MLX-style
    // safetensors file containing BF16 tensors with names like
    // `talker.code_predictor.small_to_mtp_projection.weight`, etc.
    // Use this to match MLX's numerical trajectory bit-for-bit — the F16
    // GGUF derived from the same pretrained model has subtly different
    // rounding that shows up as audio fragments on the CP path.
    bool init_from_safetensors(const std::string &path,
                                const CodePredictorConfig &cfg,
                                int device = 0);

    // Process one token through the CP transformer.
    // input_talker_space: [talker_hidden] F32 on HOST
    // pos: KV cache position (0-based)
    // hidden_out: [cp_hidden] F32 on HOST
    void forward_one_token(const float *input_talker_space, int pos,
                           float *hidden_out);

    // ---- M6.2 async split of forward_one_token ----------------------------
    // `_launch` uploads input and queues every CP op on `stream_` without
    // syncing and without doing the final D2H. An optional `wait_event`
    // (e.g. Talker's decode_done_event) is waited on by `stream_` before the
    // first queued op. After queuing, records `forward_done_event_` on
    // `stream_`.
    //
    // `_fetch` syncs the recorded event and downloads the F32 cp_hidden
    // state to the host.
    //
    // The original `forward_one_token` is now `{ launch; fetch; }` for
    // callers that don't want async.
    void forward_one_token_launch(const float *input_talker_space, int pos,
                                   aclrtEvent wait_event = nullptr);
    void forward_one_token_fetch(float *hidden_out);

    // Accessor for the event last recorded by `_launch` — lets an external
    // orchestrator wait on CP's completion from a different stream.
    aclrtEvent get_forward_done_event() const { return forward_done_event_; }

    // Reset KV cache for a new frame
    void reset_kv_cache();

    // Cap the number of transformer layers used in forward_one_token.
    // `n` must be in [1, num_hidden_layers]; 0 means "use all layers".
    // Lets the caller trade quality for throughput at runtime (mirrors
    // the --cp_layers CLI flag, which otherwise only affected the
    // llama.cpp CP path).
    void set_active_layers(int n) {
        active_layers_ = (n > 0 && n < n_layers_) ? n : 0;
    }

    // Pre-convert each F16 matmul weight buffer to CANN's FRACTAL_NZ layout
    // (M5.2). MUST be called before init / init_from_safetensors — the
    // conversion runs inline during weight upload. Per-layer Q/K/V/O/gate/up/
    // down projections get the aclnnTransMatmulWeight pass when enabled AND
    // g_cann.has_nz() is true. The F32 input projection (proj_w) is left in
    // ND because TransMatmulWeight only supports F16/INT8/BF16 weights. If
    // the runtime lacks the symbol we silently fall back to ND; default
    // (flag off) preserves pre-M5 behaviour bit-for-bit. Matmul call sites
    // still dispatch via plain aclnnMm — format is transparent to the op
    // API (see M5.1 audit).
    void set_use_nz_weights(bool enable) { use_nz_weights_ = enable; }
    bool use_nz_weights() const { return use_nz_weights_; }
    // True once the NZ pre-conversion actually ran on the weight buffers.
    bool nz_applied() const { return nz_applied_; }

    // ---- Multi-stream pipelining (M6.1) -----------------------------------
    // Engine owns two aclrtStream handles — `primary_stream_` (default) and
    // `stream_b_` (spare for multi-stream overlap). set_stream(s) swaps
    // which stream subsequent ops target; passing nullptr restores the
    // primary.
    //
    // Lifetime: both streams are created in init()/init_from_safetensors
    // and destroyed in the dtor. set_stream() does NOT take ownership of
    // the passed handle.
    aclrtStream get_stream()         const { return stream_; }
    aclrtStream get_stream_b()       const { return stream_b_; }
    aclrtStream get_primary_stream() const { return primary_stream_; }
    void set_stream(aclrtStream s) {
        stream_ = (s != nullptr) ? s : primary_stream_;
    }

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int device_ = 0;
    // `stream_` is the stream every op in this engine targets. By default
    // it points to `primary_stream_`; an orchestrator can swap it via
    // set_stream() to pipeline two engines on two physical NPU streams.
    aclrtStream stream_         = nullptr;
    aclrtStream primary_stream_ = nullptr;  // owned
    aclrtStream stream_b_       = nullptr;  // owned; used for M6 overlap

    // ---- M6.2 async completion event --------------------------------------
    // Recorded on `stream_` at the end of `forward_one_token_launch`.
    aclrtEvent  forward_done_event_ = nullptr;  // owned

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
    int active_layers_ = 0;  // 0=use all; <n_layers_ = truncate early
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

    // Attention intermediates (all device-side — v2 keeps everything on NPU).
    // scores_dev_ is F32 (for F32 softmax stability matching llama.cpp).
    // scores_f16_dev_ holds the F16 version used by the BMMs.
    void *scores_dev_ = nullptr;       // F32 [n_heads * MAX_SEQ]
    void *scores_f16_dev_ = nullptr;   // F16 [n_heads * MAX_SEQ]

    // RoPE cos/sin tables: precomputed per position [MAX_SEQ, head_dim] where
    // each row duplicates the half so the HF-style "rotate_half" formula maps
    // to aclnnRotaryPositionEmbedding(mode=0, NEOX).
    void *rope_cos_dev_ = nullptr;
    void *rope_sin_dev_ = nullptr;

    // RmsNorm rstd scratch: sized for the largest case (QK-norm over n_heads).
    void *rstd_dev_ = nullptr;

    // ---- KV cache on NPU [n_layers][MAX_SEQ * kv_dim] ----
    std::vector<void *> k_cache_dev_;
    std::vector<void *> v_cache_dev_;
    int kv_cache_len_ = 0;

    // Attention-scale scalar: 1/sqrt(head_dim). Two copies — F32 for use with
    // the F32 softmax path (applied pre-softmax) and F16 for legacy paths.
    aclScalar *attn_scale_ = nullptr;       // F32
    aclScalar *attn_scale_f16_ = nullptr;   // F16
    // Alpha=1.0 scalar reused by aclnnAdd / aclnnInplaceAdd.
    aclScalar *one_scalar_ = nullptr;

    // Boundary F32 staging buffers (I/O is F32; internal compute is F16).
    void *input_stage_f32_dev_ = nullptr;
    void *output_stage_f32_dev_ = nullptr;
    // F32 scratch for the input projection output; cast F32->F16 afterwards.
    // Also serves as the F32 residual ACCUMULATOR across all layers — F32
    // adds preserve the small per-layer deltas that F16 rounds away.
    void *proj_out_f32_dev_ = nullptr;
    // Per-sublayer F16 delta is cast into this buffer before the F32 accum Add.
    void *accum_scratch_f32_dev_ = nullptr;
    // F32 RmsNorm output buffer — RmsNorm runs F32 end-to-end, then we Cast
    // F32 -> F16 into normed_dev_ for the subsequent Mm.
    void *normed_f32_dev_ = nullptr;

    // aclnn workspace (reusable, grown as needed)
    void *workspace_dev_ = nullptr;
    size_t workspace_size_ = 0;

    // ---- Persistent aclTensor descriptors -------------------------------
    // Building an aclTensor isn't free (it allocates metadata and validates
    // shape/strides). The previous forward_one_token created ~300 of them per
    // decode, which shows up as a few ms/frame even though the underlying
    // buffers never change. Here we pre-create every handle whose shape + data
    // pointer is fixed for the lifetime of the engine and reuse it across
    // forwards. Only KV-cache views (seq_len-dependent + per-layer offset) and
    // per-pos RoPE row views remain dynamic.
    //
    // Buffer-view naming: <buffer>_<shape-tag>.  Shape tags: `col` = [N, 1],
    // `row` = [1, N], `flat` = [N], `gqa` = [n_kv, group, head_dim],
    // `heads` = [n_heads, head_dim], `kv` = [n_kv, head_dim], `4d` = [1,1,N,D]
    // for RoPE tensors.
    struct LayerTensors {
        aclTensor *q_proj, *k_proj, *v_proj, *o_proj;
        aclTensor *q_norm, *k_norm;
        aclTensor *gate_proj, *up_proj, *down_proj;
        aclTensor *input_ln, *post_ln;
    };
    struct Tensors {
        aclTensor *proj_w;
        aclTensor *proj_b;
        std::vector<LayerTensors> layer;
        aclTensor *final_norm;

        aclTensor *cur_row, *cur_flat;
        aclTensor *residual_flat;
        aclTensor *normed_row, *normed_col;
        aclTensor *q_col, *q_heads, *q_rope_4d, *q_gqa;
        aclTensor *k_col, *k_kv;
        aclTensor *v_col;
        aclTensor *attn_out_col, *attn_out_4d, *attn_out_gqa;
        aclTensor *o_out_col, *o_out_flat;
        aclTensor *gate_col, *gate_flat;
        aclTensor *up_col, *up_flat;
        aclTensor *ffn_out_col, *ffn_out_flat;
        aclTensor *rstd_11, *rstd_heads, *rstd_kv;
        // Boundary staging (F32) + cast targets (F16 views of existing bufs)
        aclTensor *input_f32, *input_f16;
        aclTensor *output_f16, *output_f32;
        // F32 projection path: proj_w/proj_b stay F32, output in F32 scratch,
        // then cast F32->F16 into cur_dev_.
        aclTensor *proj_w_f32, *proj_b_f32;
        aclTensor *proj_out_f32_col, *proj_out_f32_flat;
        aclTensor *cur_f16_flat_as_target;  // [cp_hidden] F16 view of cur_dev_
        // F32 accumulator scratch for residual adds
        aclTensor *accum_scratch_f32_flat;  // F32 [cp_hidden]
        // F32 views used by the RmsNorm-in-F32 path
        aclTensor *accum_f32_row_2d;        // F32 [1, cp_hidden] — RmsNorm IN
        aclTensor *normed_f32_row_2d;       // F32 [1, cp_hidden] — RmsNorm OUT
        aclTensor *normed_f32_flat;         // F32 [cp_hidden]    — Cast input
    };
    Tensors t_{};

    // ---- FRACTAL_NZ weight pre-conversion (M5.2) ----------------------------
    bool use_nz_weights_ = false;
    bool nz_applied_     = false;

    // ---- Internal helpers ----
    void alloc_dev(void **ptr, size_t bytes);
    void upload(void *dev, const float *host, size_t n_floats);
    void download(float *host, const void *dev, size_t n_floats);
    void ensure_workspace(size_t needed);
    // Apply aclnnTransMatmulWeight in place. No-op if runtime lacks has_nz()
    // or if use_nz_weights_ is false. See header comment on
    // set_use_nz_weights() for the full semantics.
    void nz_convert_weight_(void *weight_dev, int64_t rows, int64_t cols);

    // Create all persistent aclTensor descriptors after weights + buffers
    // have been allocated. Called once at the end of init().
    void build_persistent_tensors_();
    // Destroy all persistent descriptors (called from destructor).
    void destroy_persistent_tensors_();
};
