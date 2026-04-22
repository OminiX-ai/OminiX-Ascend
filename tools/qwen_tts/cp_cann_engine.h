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
                                   aclrtEvent wait_event = nullptr,
                                   bool use_async_memcpy = false,
                                   bool record_event = true);
    void forward_one_token_fetch(float *hidden_out);

    // ---- M3'new' batched prefill of CP positions 0+1 ----------------------
    // Queues two back-to-back CP forwards (pos 0 and pos 1) on `stream_`
    // without any host sync between them. Both input uploads and both
    // forward dispatches are submitted on `stream_` with stream-ordering
    // providing the H→D-write / device-read ordering for the shared
    // `input_stage_f32_dev_` buffer.
    //
    // Semantics match `forward_one_token_launch(input0, 0);
    //                  forward_one_token_launch(input1, 1)` with two
    // differences:
    //   1. No `forward_one_token_sync` needed between positions — pos 1
    //      begins queuing as soon as pos 0's ops are submitted.
    //   2. Only one `forward_done_event_` record (after pos 1). The
    //      caller can wait on that event to get pos 1's hidden state;
    //      pos 0's output is overwritten and not exposed (matches the
    //      talker.cpp contract — pos 0's hidden is never consumed).
    //
    // Typically gated by `TALKER_CP_POS_BATCH=1` at the caller.
    void forward_two_tokens_launch(const float *input0, const float *input1,
                                    aclrtEvent wait_event = nullptr);

    // W1: device-only sync after `_launch`. Blocks on `forward_done_event_`
    // (or `stream_` if no event) without doing the D2H of hidden_out. Use
    // this when the hidden state stays on NPU for a downstream op such as
    // `forward_lm_head`; skips the ~0.2 ms memcpy cost per group.
    void forward_one_token_sync();

    // Accessor for the event last recorded by `_launch` — lets an external
    // orchestrator wait on CP's completion from a different stream.
    aclrtEvent get_forward_done_event() const { return forward_done_event_; }

    // ---- W1 NPU lm_head port --------------------------------------------------
    // Dispatch the per-group lm_head matmul on NPU. Reads the F32 hidden
    // state currently sitting in `output_stage_f32_dev_` (the output buffer
    // written by `forward_one_token_launch`), casts to F16, multiplies by
    // `lm_head_w_dev_[group_idx]` (F16 [vocab, cp_hidden]), and writes F16
    // logits into an engine-owned device buffer. Shape-stable: vocab_size ×
    // cp_hidden for all 15 groups. `fetch_logits` downloads those logits to
    // a host F32 buffer (F16→F32 upconvert inline).
    //
    // Contract:
    //   - `forward_one_token_launch` must have produced a valid hidden in
    //     `output_stage_f32_dev_` before `forward_lm_head` is called.
    //   - W1 gates the whole port behind `TALKER_LM_HEAD_NPU=1`. When the
    //     env var is unset the caller MUST stay on the CPU `cp_matvec_f32`
    //     path and MUST NOT call these methods.
    //   - Not thread-safe across groups — dispatch + fetch are serial on
    //     `stream_`.
    void forward_lm_head(int group_idx);
    void fetch_logits(float *host_out);

    // True iff lm_head weights were uploaded to NPU during init. Caller
    // should check this before calling forward_lm_head / fetch_logits.
    bool has_lm_head() const { return lm_head_ready_; }
    int  lm_head_vocab_size() const { return lm_head_vocab_size_; }

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

    // ---- A16W8 weight quantization (Stretch S1) ---------------------------
    // Opt-in toggle. When true AND g_cann.has_w8_quant() at init AND
    // use_nz_weights_ is NOT set, init / init_from_safetensors calibrates
    // each F16 matmul weight to per-output-channel symmetric INT8 and stores
    // the INT8 weight + F16 scale buffers alongside the existing F16 weight
    // (F16 retained because forward_prefill stays on F16 this round).
    // forward_one_token then dispatches aclnnWeightQuantBatchMatmulV3/V2 at
    // decode matmul call sites. The F32 input projection (proj_w) remains
    // F32 — it's outside the matmul-quant scope. Mutually exclusive with
    // NZ (W8 wins if both env flags set).
    void set_use_w8_weights(bool enable) { use_w8_weights_ = enable; }
    bool use_w8_weights() const { return use_w8_weights_; }
    bool w8_applied() const { return w8_applied_; }

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
        void *input_ln_w = nullptr;   // [cp_hidden] F32
        void *post_ln_w = nullptr;    // [cp_hidden] F32

        // W3b fusion: F16 copies of the two LN gammas (aclnnAddRmsNorm
        // requires gamma dtype to match x1/x2, which is F16 here). Allocated
        // only when cp_fusion_applied_; null otherwise.
        void *input_ln_w_f16 = nullptr;
        void *post_ln_w_f16  = nullptr;

        // W4.1 AscendC path: F16 copies of the per-head RmsNorm gammas the
        // fused attn-sublayer kernel reads directly. Sized [head_dim] F16.
        // Allocated only when cp_ascendc_applied_; null otherwise.
        void *q_norm_w_f16   = nullptr;
        void *k_norm_w_f16   = nullptr;

        // A16W8 (Stretch S1): INT8 [out, in] + F16 per-output-channel scale,
        // allocated alongside the F16 buffers when w8_applied_. Null otherwise.
        void *q_proj_w_i8    = nullptr;
        void *k_proj_w_i8    = nullptr;
        void *v_proj_w_i8    = nullptr;
        void *o_proj_w_i8    = nullptr;
        void *gate_proj_w_i8 = nullptr;
        void *up_proj_w_i8   = nullptr;
        void *down_proj_w_i8 = nullptr;
        void *q_proj_scale    = nullptr;
        void *k_proj_scale    = nullptr;
        void *v_proj_scale    = nullptr;
        void *o_proj_scale    = nullptr;
        void *gate_proj_scale = nullptr;
        void *up_proj_scale   = nullptr;
        void *down_proj_scale = nullptr;

        // Phase B: gate||up concatenated INT8 weight + F16 scale, in the
        // layout aclnnFFNV3 expects as its weight1 argument.
        //   gate_up_w_i8: INT8 byte-concat of [gate_proj_w_i8, up_proj_w_i8]
        //                 → row-major shape [2*inter, cp_hidden]. Viewed with
        //                 strides [1, cp_hidden] this becomes the logical
        //                 [K1, N1] = [cp_hidden, 2*inter] tensor FFNV3 reads.
        //   gate_up_scale: F16 concat of [gate_proj_scale, up_proj_scale] →
        //                  shape [2*inter]. Matches antiquantScale1 [N1].
        // Allocated only when cp_ffn_v3_applied_. Null otherwise.
        void *gate_up_w_i8  = nullptr;
        void *gate_up_scale = nullptr;
    };
    std::vector<LayerWeights> layer_w_;
    void *final_norm_w_dev_ = nullptr;  // [cp_hidden] F32
    // W3b fusion: F16 copy of the final norm gamma, used by the last
    // layer's post-FFN AddRmsNorm. Null unless cp_fusion_applied_.
    void *final_norm_w_f16_dev_ = nullptr;

    // ---- W1: lm_head weights on NPU ---------------------------------------
    // F16, [vocab_size, cp_hidden] per group. Uploaded during init() /
    // init_from_safetensors() when `TALKER_LM_HEAD_NPU=1` is set. Each
    // group's weight is dispatched via aclnnMm into `logits_f16_dev_`.
    // `logits_stage_f32_dev_` is the F32 staging buffer used by
    // `fetch_logits` for the D2H downcast.
    std::vector<void *> lm_head_w_dev_;
    void *logits_f16_dev_        = nullptr;  // F16 [vocab_size]
    void *logits_stage_f32_dev_  = nullptr;  // F32 [vocab_size]
    bool  lm_head_ready_         = false;
    int   lm_head_vocab_size_    = 0;        // 2048 typically
    int   lm_head_n_groups_      = 0;        // 15 typically

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
        // W3b fusion: F16-dtype views over input_ln_w_f16 / post_ln_w_f16.
        // Only created when cp_fusion_applied_; null otherwise.
        aclTensor *input_ln_f16 = nullptr;
        aclTensor *post_ln_f16  = nullptr;
    };
    struct Tensors {
        aclTensor *proj_w;
        aclTensor *proj_b;
        std::vector<LayerTensors> layer;
        aclTensor *final_norm;
        // W3b fusion: F16-dtype view over final_norm_w_f16_dev_. Null
        // unless cp_fusion_applied_.
        aclTensor *final_norm_f16 = nullptr;

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

    // ---- A16W8 weight quantization state (Stretch S1) -----------------------
    bool use_w8_weights_ = false;
    bool w8_applied_     = false;

    // ---- CP kernel fusion state (W3b) ---------------------------------------
    // Gated by env var TALKER_CP_FUSION=1 (opt-in, same pattern as
    // TALKER_LM_HEAD_NPU). When `cp_fusion_enabled_` is true AND the runtime
    // exposes `aclnnAddRmsNorm`, the per-layer tail `Add(residual, output) +
    // RmsNorm(cur, gamma)` collapses into a single `aclnnAddRmsNorm` dispatch
    // — saving one aclnn launch per sublayer (2/layer × 5 layers × 17 forwards
    // = 170 dispatches/frame). `cp_fusion_applied_` records whether the
    // runtime actually enabled fusion after the capability check.
    bool cp_fusion_enabled_ = false;
    bool cp_fusion_applied_ = false;

    // ---- Path C W4.1 AscendC fused attention sublayer -----------------------
    // Gated by env var TALKER_CP_ASCENDC=1 (opt-in). When set AND the build
    // has QWEN_TTS_HAS_ASCENDC AND w8_applied_, forward_one_token_launch's
    // attention sublayer dispatches the single fused kernel
    // `aclrtlaunch_fused_attn_sublayer` instead of the stock aclnn chain
    // (RmsNorm → QKV-Mm → QK-norm → RoPE → V-copy → FIAS → O-Mm → +residual).
    // Unset (or on an ascendc-less build) = bit-identical to W1+W3b.
    bool cp_ascendc_enabled_ = false;
    bool cp_ascendc_applied_ = false;

    // ---- G2 aclGraph pos-keyed capture/replay cache -------------------------
    // Gated by env var TALKER_CP_ACLGRAPH=1 (opt-in, same pattern as W1/W3b).
    // When set AND g_cann.has_aclgraph() at init time, the engine captures
    // MAX_ACLGRAPH_POS+1 = 17 full forward-one-token graphs (one per
    // pos in [0, 16]) during init(). Each graph bakes the pos-specific
    // RoPE slice pointer, V-cache slot offset, and FIAv2 seq_len. Per-frame
    // replay = pointer-select + aclmdlRIExecuteAsync.
    //
    // Canonical benchmark path only: capture runs only if
    // cp_fusion_applied_ && w8_applied_ && !cp_ascendc_applied_. Otherwise
    // aclgraph stays disabled (eager path). This is a scope constraint —
    // other combinations stay on the stock eager path.
    //
    // D2D memcpys inside `forward_one_token_launch` (residual=cur, V→slot)
    // return err=507009 inside aclmdlRICapture on CANN 8.3.RC1 (G1 finding).
    // The capturable variant `forward_one_token_capturable_(pos)` replaces
    // these with aclnn-based Adds (aclnnAdd with alpha=1 over a zero buf),
    // and redirects V-proj output to write directly into the pos-specific
    // cache slot — both capturable.
    static constexpr int MAX_ACLGRAPH_POS = 16;
    bool aclgraph_enabled_ = false;
    bool aclgraph_applied_ = false;
    std::vector<aclmdlRI> aclgraph_graphs_;
    void *zero_f16_cp_dev_ = nullptr;  // F16 [cp_hidden] zeros — copy helper

    // ---- Phase A.1: aclnnInplaceAddRmsNorm drop-in (TALKER_CP_INPLACE_ADDRMSNORM)
    // When set AND g_cann.has_inplace_add_rms_norm(), the fused post-attn and
    // post-FFN tails dispatch aclnnInplaceAddRmsNorm and skip the trailing
    // residual-copy (saves ~75 aclnn copies/frame). Per the Phase A contract,
    // aclGraph capture is disabled while this flag is on until A.3 verifies
    // the inplace semantics compose with captured tensor descriptors.
    bool cp_inplace_ars_enabled_ = false;
    bool cp_inplace_ars_applied_ = false;

    // ---- Phase B: aclnnFFNV3 fused SwiGLU FFN (TALKER_CP_FFN_V3)
    // When set AND g_cann.has_ffn_v3() AND w8_applied_, the 5-op FFN chain
    // (gate-Mm + up-Mm + SiLU + mul + down-Mm) collapses into a single
    // aclnnFFNV3 call with activation="swiglu". Requires the gate||up
    // concatenated INT8 weight + F16 scale to be built at engine init (see
    // build_ffn_v3_weights_). aclGraph capture is conservatively disabled
    // when this flag is on — the FFN op-count changed (5→1) and the existing
    // pos-keyed graphs must re-capture with FFNV3 in place; we let A.3-style
    // composition verification handle that in a follow-up.
    bool cp_ffn_v3_enabled_ = false;
    bool cp_ffn_v3_applied_ = false;

    // ---- Internal helpers ----
    void alloc_dev(void **ptr, size_t bytes);
    void upload(void *dev, const float *host, size_t n_floats);
    void download(float *host, const void *dev, size_t n_floats);
    void ensure_workspace(size_t needed);
    // Apply aclnnTransMatmulWeight in place. No-op if runtime lacks has_nz()
    // or if use_nz_weights_ is false. See header comment on
    // set_use_nz_weights() for the full semantics.
    void nz_convert_weight_(void *weight_dev, int64_t rows, int64_t cols);

    // A16W8 calibration (Stretch S1). Allocates NEW device buffers for the
    // INT8 weight + F16 scale. Caller retains whatever F16 buffer exists —
    // prefill still uses it. Returns true on success; outputs untouched on
    // failure.
    bool w8_calibrate_weight_(const float *host_w, int64_t rows, int64_t cols,
                               void *&weight_i8_dev, void *&scale_dev);

    // A16W8 matmul dispatch. Prefers V3, falls back to V2. See the
    // TalkerCannEngine equivalent for the full shape contract.
    void w8_matmul_(const aclTensor *x, void *weight_dev, void *scale_dev,
                     int64_t out_n, int64_t in_k, const aclTensor *y);

    // Phase B: at engine init under cp_ffn_v3_applied_, walk every layer_w_
    // slot and build its gate||up concatenated INT8 weight blob +
    // F16 scale blob (see LayerWeights.gate_up_w_i8 / .gate_up_scale). This
    // is a device-to-device memcpy sequence — no host-side conversion. Must
    // run AFTER w8_calibrate_weight_ has filled gate_proj_w_i8, up_proj_w_i8,
    // gate_proj_scale, up_proj_scale on every layer.
    bool build_ffn_v3_weights_();

    // Phase B: fused SwiGLU FFN dispatch.
    //   x_row:       F16 [1, cp_hidden] — post-RmsNorm input
    //   y_row:       F16 [1, cp_hidden] — overwritten with down_proj output
    //   gate_up_i8:  LayerWeights.gate_up_w_i8   (device buf)
    //   gate_up_s:   LayerWeights.gate_up_scale  (device buf)
    //   down_i8:     LayerWeights.down_proj_w_i8 (device buf)
    //   down_s:      LayerWeights.down_proj_scale (device buf)
    void ffn_v3_swiglu_(const aclTensor *x_row,
                         void *gate_up_i8, void *gate_up_s,
                         void *down_i8,    void *down_s,
                         const aclTensor *y_row);

    // W1: upload per-group lm_head weights (F16) + allocate logits buffers.
    // No-op when already initialised. Caller provides the F32 host-side
    // weights in [n_groups][vocab_size * cp_hidden] row-major layout.
    bool init_lm_head_(const std::vector<std::vector<float>> &lm_head_w,
                        int vocab_size);

    // W3b: allocate F16 copies of the LN gammas via aclnnCast. No-op unless
    // cp_fusion_applied_. Must be called AFTER the F32 gamma buffers are
    // uploaded and BEFORE build_persistent_tensors_().
    void init_fusion_f16_gammas_();

    // W4.1 AscendC: allocate F16 copies of the per-head Q/K RmsNorm gammas,
    // and (for the rare ascendc-on-without-fusion case) the input_ln / final
    // layer-norm gammas too. No-op unless cp_ascendc_applied_.
    void init_ascendc_f16_gammas_();

    // Create all persistent aclTensor descriptors after weights + buffers
    // have been allocated. Called once at the end of init().
    void build_persistent_tensors_();
    // Destroy all persistent descriptors (called from destructor).
    void destroy_persistent_tensors_();

    // G2: capturable variant of forward_one_token_launch. Identical aclnn
    // dispatch chain to the W8+fusion path, but with all D2D memcpys replaced
    // by aclnn Adds and V-proj writing directly to v_cache slot. Used both at
    // capture time (inside CaptureBegin/End) and as the scoped replay fallback
    // for pos > MAX_ACLGRAPH_POS. Records `forward_done_event_` at the tail.
    void forward_one_token_capturable_(int pos);

    // G2: capture-at-init. No-op unless env gate set and canonical path. On
    // success, fills aclgraph_graphs_[0..MAX] and flips aclgraph_applied_=true.
    void capture_aclgraph_forwards_();
};
