#pragma once
// ============================================================================
// Talker CANN Engine: Direct ACL-based Qwen3-TTS Talker backbone on Ascend NPU
//
// Mirrors CpCannEngine but targets the 28-layer Talker backbone instead of
// the 5-layer Code Predictor. Loads weights from the standard llama-style
// GGUF produced by `export_talker_llama.py`.
//
// Caller supplies float embeddings directly (embedding lookup / text_projection
// happens upstream). The engine performs:
//   input_embeds (F32 on host) -> F16 transformer compute -> F32 hidden out.
//
// Precision scheme (matches ggml-cann's Qwen3 convention):
//   - F32: I/O staging at engine boundary; RmsNorm gammas (all norm weights).
//   - F16: matmuls, residual adds, RoPE, attention, FFN, KV cache.
//   - Attention: aclnnFusedInferAttentionScoreV2 (the op ggml-cann uses) —
//     layout=BSND, numKeyValueHeads for GQA, innerPrecise=0 for decode
//     (S=1) and innerPrecise=2 with a causal mask for prefill (S>1).
//
// All intermediate buffers allocated once at init. Per-forward path issues
// only aclnn kernel launches — no allocation on the hot path.
// ============================================================================

#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <string>
#include <vector>
#include <cstdint>

#include "cp_cann_symbols.h"

struct TalkerConfig;

class TalkerCannEngine {
public:
    TalkerCannEngine() = default;
    ~TalkerCannEngine();

    // Load the Talker backbone from a llama-style GGUF (F16 matmul weights,
    // F32 norm gammas). `device` is the ACL device ID (usually 0).
    bool init_from_gguf(const std::string &gguf_path, const TalkerConfig &cfg,
                        int device = 0);

    // Single-token decode path (S=1). Uses KV cache at `pos`.
    //   input_embed: [n_embd] F32 on HOST
    //   hidden_out:  [n_embd] F32 on HOST (post final RmsNorm)
    void forward_decode(const float *input_embed, int pos, float *hidden_out);

    // ---- M6.2 async split of forward_decode -------------------------------
    // `_launch` uploads the F32 input embedding and queues every kernel launch
    // for the 28-layer decode path onto `stream_`, without syncing and without
    // doing the final D2H of `output_stage_f32_dev_`. An optional `wait_event`
    // is waited on by `stream_` before the first queued op — use this to make
    // Talker[N+1]'s launch depend on CP[N]'s completion when pipelining on
    // two streams. After queuing, it records `decode_done_event_` on
    // `stream_` so another stream can fence against Talker's hidden output.
    //
    // `_fetch` syncs the recorded event and downloads the F32 hidden state to
    // the host. It's safe to call `_fetch` from any thread once `_launch`
    // returned; the event ordering is the only synchronization required.
    //
    // The original `forward_decode` is now `{ launch; fetch; }` internally
    // so callers that don't want async get the same behaviour as before.
    void forward_decode_launch(const float *input_embed, int pos,
                                aclrtEvent wait_event = nullptr);
    void forward_decode_fetch(float *hidden_out);

    // ---- M6.2 Track J: speculative-embedding split ------------------------
    // `_launch_cast` uploads the F32 input embedding and queues the initial
    // F32->F16 Cast into `cur_dev_` on `stream_`. Disables aclGraph capture
    // for the caller's session because graph capture spans the entire decode
    // body; the split halves are incompatible with a single captured block.
    // Returns immediately (async on `stream_`).
    //
    // `add_input_delta_f32` uploads a host F32 `[n_embd]` delta, casts it to
    // F16, and issues an in-flight `aclnnInplaceAdd` onto `cur_dev_`. When
    // `wait_event` is non-null, `stream_` fences on it BEFORE the upload
    // (so the caller can stall the delta-add until e.g. CP[N] on a sibling
    // stream finishes and its last group's contribution is known on host).
    //
    // `_launch_layers` runs the 28-layer transformer body + final RmsNorm +
    // Cast-to-F32 on `stream_`, then records `decode_done_event_` as before.
    // `forward_decode_fetch` stays unchanged — D2H the F32 hidden.
    //
    // Callers that want the un-split behaviour keep calling
    // `forward_decode_launch` (still in place as a convenience that calls
    // cast + layers back-to-back).
    //
    // Precondition on `add_input_delta_f32`: must be called between a
    // `_launch_cast` and its paired `_launch_layers` on the same `pos`. The
    // delta buffer on the host is read synchronously inside the H2D memcpy
    // so the caller may free/rewrite it as soon as the call returns.
    void forward_decode_launch_cast(const float *input_embed, int pos,
                                     aclrtEvent wait_event = nullptr);
    void add_input_delta_f32(const float *delta,
                              aclrtEvent wait_event = nullptr);
    void forward_decode_launch_layers(int pos);

    // Batched prefill (S>1). Appends `seq_len` tokens to the KV cache
    // starting at `start_pos`. Only the LAST position's hidden state is
    // returned (TalkerLLM only needs that for next-token sampling).
    //   input_embeds:    [seq_len, n_embd] F32 on HOST
    //   last_hidden_out: [n_embd] F32 on HOST
    void forward_prefill(const float *input_embeds, int seq_len, int start_pos,
                          float *last_hidden_out);

    // Reset KV cache position (caller should do this between utterances).
    void reset_kv_cache();

    // Set RoPE position speed factor (EOS steering — MLX parity). >1.0 makes
    // the model's internal clock run faster. Rebuilds cos/sin tables on
    // change; cheap on subsequent forwards with unchanged factor.
    void set_rope_speed_factor(float factor);

    // Pre-convert each matmul weight buffer to CANN's FRACTAL_NZ layout
    // (M5.2). MUST be called before init_from_gguf — the conversion is
    // applied inline during weight upload. When enabled AND the runtime
    // resolves the required symbol (CannSyms::has_nz()), every per-layer
    // Q/K/V/O/gate/up/down projection weight is passed through
    // aclnnTransMatmulWeight after upload so the hardware-preferred layout
    // is baked in. If the symbol is unavailable we silently fall back to
    // ND; if init runs with this false (the default), nothing changes from
    // the pre-M5 behaviour. Matmul call sites still dispatch via plain
    // aclnnMm — the format is transparent to the op API (see M5.1 audit).
    void set_use_nz_weights(bool enable) { use_nz_weights_ = enable; }
    bool use_nz_weights() const { return use_nz_weights_; }
    // True once the NZ pre-conversion actually ran on the weight buffers
    // (i.e., use_nz_weights_ was on AND g_cann.has_nz() resolved at init).
    bool nz_applied() const { return nz_applied_; }

    // ---- A16W8 weight quantization (Stretch S1) ---------------------------
    // Opt-in toggle. When true AND g_cann.has_w8_quant() at init AND
    // use_nz_weights_ is NOT set (W8 and NZ are mutually exclusive in S1),
    // init_from_gguf calibrates each F16 matmul weight to per-output-channel
    // symmetric INT8: scale_c = max(|W[c,:]|) / 127, zero = 0, then stores
    // weight_i8 [out, in] + scale_f16 [out] + zero_f16 [out] on device
    // (replacing the F16 buffer — saves ~50% weight memory). Matmul call
    // sites in forward_decode + run_decode_ops_ then dispatch
    // aclnnWeightQuantBatchMatmulV3 (or V2 as fallback) instead of plain
    // aclnnMm. `forward_prefill` intentionally stays on F16 (ND/NZ) this
    // round so we keep one safe fallback path on the critical prefill step.
    void set_use_w8_weights(bool enable) { use_w8_weights_ = enable; }
    bool use_w8_weights() const { return use_w8_weights_; }
    // True once W8 calibration actually ran on the weight buffers (i.e.
    // use_w8_weights_ && g_cann.has_w8_quant() && !use_nz_weights_).
    bool w8_applied() const { return w8_applied_; }

    // ---- Multi-stream pipelining (M6.1) -----------------------------------
    // The engine owns TWO aclrtStream handles — the primary `stream_` (used
    // by every op by default) and a secondary `stream_b_` that the
    // orchestrator (TalkerLLM::generate) can borrow to run a second engine
    // on so Talker[N+1] overlaps CP[N]. `set_stream(s)` swaps which stream
    // subsequent ops target; passing nullptr restores the primary.
    //
    // Lifetime: both streams are created in init_from_gguf and destroyed in
    // the dtor. set_stream() does NOT take ownership of the passed stream.
    aclrtStream get_stream()         const { return stream_; }
    aclrtStream get_stream_b()       const { return stream_b_; }
    aclrtStream get_primary_stream() const { return primary_stream_; }
    void set_stream(aclrtStream s) {
        stream_ = (s != nullptr) ? s : primary_stream_;
    }

    // Accessor for the event last recorded by `forward_decode_launch`.
    aclrtEvent get_decode_done_event() const { return decode_done_event_; }

    bool is_ready() const { return ready_; }

private:
    bool ready_ = false;
    int device_ = 0;
    // `stream_` is the stream every op in this engine targets. By default it
    // points to `primary_stream_`; an orchestrator can swap it to
    // `stream_b_` (or an externally-owned stream from another engine) via
    // set_stream() to pipeline two engines on two physical NPU streams.
    aclrtStream stream_         = nullptr;
    aclrtStream primary_stream_ = nullptr;  // owned
    aclrtStream stream_b_       = nullptr;  // owned; used for multi-stream overlap

    // ---- M6.2 async decode event ------------------------------------------
    // Recorded on `stream_` by `forward_decode_launch` after the final Cast
    // op. `forward_decode_fetch` waits on this event before issuing the D2H
    // copy. Other engines can also `aclrtStreamWaitEvent` on this handle to
    // make their stream's next op depend on Talker's hidden output being
    // ready, without round-tripping through the host.
    aclrtEvent  decode_done_event_ = nullptr;  // owned

    // Model dimensions (cached from config)
    int n_embd_   = 0;   // 2048
    int n_heads_  = 0;   // 16
    int n_kv_     = 0;   // 8
    int head_dim_ = 0;   // 128
    int q_dim_    = 0;   // n_heads * head_dim = 2048
    int kv_dim_   = 0;   // n_kv * head_dim    = 1024
    int inter_    = 0;   // 6144
    int n_layers_ = 0;   // 28
    float eps_        = 0.0f;
    float rope_theta_ = 0.0f;
    float rope_speed_factor_ = 1.0f;

    // Talker sequence budget — Talker may prefill up to ~100 text tokens plus
    // generate a few thousand codec frames. 4096 is a conservative cap that
    // fits 28 * 4096 * 1024 * 2 bytes * 2 (K+V) ≈ 460 MB of KV cache.
    static constexpr int MAX_SEQ = 4096;
    // Max prefill batch size handled by the preallocated staging buffers.
    // If a caller prefills more than this in one shot, we fall back to
    // chunked prefill (each chunk ≤ MAX_PREFILL).
    static constexpr int MAX_PREFILL = 512;

    // ---- NPU weight buffers (persistent, uploaded once) ----
    // The F16 *_proj_w slots are always populated (used by forward_prefill +
    // decode's F16/NZ fallback). When `w8_applied_` is true we ALSO keep a
    // parallel INT8 buffer (and matching F16 scale) in the *_proj_w_i8 /
    // *_proj_scale slots; decode matmul call sites dispatch
    // aclnnWeightQuantBatchMatmul on those. Prefill stays on the F16 buffers
    // this round per the Stretch S1 contract ("don't touch forward_prefill
    // body"), which means W8 net memory is F16 + INT8 + scales in S1.
    struct LayerWeights {
        void *q_proj_w = nullptr;    // F16 [q_dim, n_embd]
        void *k_proj_w = nullptr;    // F16 [kv_dim, n_embd]
        void *v_proj_w = nullptr;    // F16 [kv_dim, n_embd]
        void *o_proj_w = nullptr;    // F16 [n_embd, q_dim]
        void *q_norm_w = nullptr;    // F32 [head_dim]
        void *k_norm_w = nullptr;    // F32 [head_dim]
        void *gate_proj_w = nullptr; // F16 [inter, n_embd]
        void *up_proj_w   = nullptr; // F16 [inter, n_embd]
        void *down_proj_w = nullptr; // F16 [n_embd, inter]
        void *input_ln_w  = nullptr; // F32 [n_embd] (attn_norm)
        void *post_ln_w   = nullptr; // F32 [n_embd] (ffn_norm)

        // A16W8 (S1): INT8 weight + F16 per-output-channel scale. Null
        // unless w8_applied_ is true. Symmetric quant (zero = 0) so we pass
        // nullptr for antiquantOffset — no separate zero buffer needed.
        void *q_proj_w_i8    = nullptr;  // INT8 [q_dim,  n_embd]
        void *k_proj_w_i8    = nullptr;  // INT8 [kv_dim, n_embd]
        void *v_proj_w_i8    = nullptr;  // INT8 [kv_dim, n_embd]
        void *o_proj_w_i8    = nullptr;  // INT8 [n_embd, q_dim]
        void *gate_proj_w_i8 = nullptr;  // INT8 [inter,  n_embd]
        void *up_proj_w_i8   = nullptr;  // INT8 [inter,  n_embd]
        void *down_proj_w_i8 = nullptr;  // INT8 [n_embd, inter]
        void *q_proj_scale    = nullptr; // F16 [q_dim]
        void *k_proj_scale    = nullptr; // F16 [kv_dim]
        void *v_proj_scale    = nullptr; // F16 [kv_dim]
        void *o_proj_scale    = nullptr; // F16 [n_embd]
        void *gate_proj_scale = nullptr; // F16 [inter]
        void *up_proj_scale   = nullptr; // F16 [inter]
        void *down_proj_scale = nullptr; // F16 [n_embd]
    };
    std::vector<LayerWeights> layer_w_;
    void *final_norm_w_dev_ = nullptr;  // F32 [n_embd] (output_norm.weight)

    // ---- NPU intermediate buffers (single-token path) ----
    void *cur_dev_      = nullptr;  // F16 [n_embd]
    void *residual_dev_ = nullptr;  // F16 [n_embd]
    void *normed_dev_   = nullptr;  // F16 [n_embd]
    void *q_dev_        = nullptr;  // F16 [q_dim]
    void *k_dev_        = nullptr;  // F16 [kv_dim]
    void *v_dev_        = nullptr;  // F16 [kv_dim]
    void *attn_out_dev_ = nullptr;  // F16 [q_dim]
    void *o_out_dev_    = nullptr;  // F16 [n_embd]
    void *gate_dev_     = nullptr;  // F16 [inter]
    void *up_dev_       = nullptr;  // F16 [inter]
    void *ffn_out_dev_  = nullptr;  // F16 [n_embd]

    // Prefill staging (batched: seq_len rows of each). Sized for MAX_PREFILL.
    void *cur_batch_dev_      = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *residual_batch_dev_ = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *normed_batch_dev_   = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *q_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, q_dim]
    void *k_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, kv_dim]
    void *v_batch_dev_        = nullptr;  // F16 [MAX_PREFILL, kv_dim]
    void *attn_out_batch_dev_ = nullptr;  // F16 [MAX_PREFILL, q_dim]
    void *o_out_batch_dev_    = nullptr;  // F16 [MAX_PREFILL, n_embd]
    void *gate_batch_dev_     = nullptr;  // F16 [MAX_PREFILL, inter]
    void *up_batch_dev_       = nullptr;  // F16 [MAX_PREFILL, inter]
    void *ffn_out_batch_dev_  = nullptr;  // F16 [MAX_PREFILL, n_embd]

    // Causal attention mask for prefill: we build a contiguous F16
    // [seq_len, seq_len_total] buffer per prefill call and upload it to
    // this scratch (sized for the worst case). FIAS's pseShift path rejects
    // strided/padded masks for some tiling keys, so we stick to a packed
    // [1, n_heads, S_q, S_kv] layout with stride[heads]=0 broadcast.
    void *causal_mask_dev_ = nullptr;  // F16 [MAX_PREFILL * MAX_SEQ]

    // RoPE cos/sin tables: precomputed per position [MAX_SEQ, head_dim] with
    // halves duplicated so NEOX-mode RotaryPositionEmbedding maps to the
    // HuggingFace/MLX rotate_half formula.
    void *rope_cos_dev_ = nullptr;  // F16
    void *rope_sin_dev_ = nullptr;  // F16

    // RmsNorm rstd scratch: sized for the largest case (prefill QK-norm over
    // MAX_PREFILL rows × n_heads heads).
    void *rstd_dev_ = nullptr;  // F32

    // ---- KV cache on NPU [n_layers][MAX_SEQ * kv_dim] (F16) ----
    std::vector<void *> k_cache_dev_;
    std::vector<void *> v_cache_dev_;
    int kv_cache_len_ = 0;

    // Boundary staging buffers (F32).
    void *input_stage_f32_dev_  = nullptr;  // F32 [MAX_PREFILL * n_embd]
    void *output_stage_f32_dev_ = nullptr;  // F32 [n_embd]

    // M6.2 Track J: F16 staging for the speculative delta-add path.
    // `add_input_delta_f32` uploads an F32 delta here (host->device), then
    // issues a Cast to F16 into `delta_stage_f16_dev_` and an InplaceAdd
    // onto `cur_dev_` — all on `stream_`.
    void *delta_stage_f32_dev_ = nullptr;   // F32 [n_embd]
    void *delta_stage_f16_dev_ = nullptr;   // F16 [n_embd]

    // Scalars
    aclScalar *one_scalar_f16_ = nullptr;   // F16 1.0 (for F16 Add alpha)

    // aclnn workspace (grows on demand)
    void *workspace_dev_  = nullptr;
    size_t workspace_size_ = 0;

    // Host-side copies of the full cos/sin tables so set_rope_speed_factor
    // can rebuild the device tables without recomputing trig from scratch.
    std::vector<float> cos_host_;  // [MAX_SEQ * head_dim]
    std::vector<float> sin_host_;

    // ---- aclGraph capture/replay cache (M4) --------------------------------
    // One graph per position slot. First call at pos=p runs eagerly AND
    // captures the kernel stream into decode_graphs_[p]. Subsequent calls at
    // the same pos replay the captured graph (saves per-kernel launch
    // overhead, which dominates at 28 layers × ~10 ops/layer). Because KV
    // cache, RoPE row, and workspace addresses are all tied to `pos`, two
    // calls at the same pos map to identical kernel arguments — only the
    // input embedding (input_stage_f32_dev_) and output (output_stage_f32_dev_)
    // vary between calls. Those are host-memcpy'd outside the captured region.
    //
    // Vector sized to MAX_SEQ up front; entries are nullptr until captured.
    // `pos` increments monotonically during a utterance, so the cache fills
    // in order. reset_kv_cache() does NOT invalidate graphs — the captured
    // ops only depend on pos, KV cache addresses, and weight addresses, all
    // of which are stable across utterances (the KV cache is overwritten in
    // place, but the address is the same). This means the graph cache
    // amortizes across every utterance in a session.
    std::vector<aclmdlRI> decode_graphs_;
    bool graph_enabled_ = false;  // Set at init based on env var + symbol
                                   // availability. When false, forward_decode
                                   // runs its original eager path unmodified.

    // ---- FRACTAL_NZ weight pre-conversion (M5.2) ----------------------------
    // Caller-controlled opt-in. When true AND g_cann.has_nz() resolves at
    // init, init_from_gguf runs aclnnTransMatmulWeight on each matmul weight
    // buffer right after the F16 upload. The ND layout path (default) stays
    // bit-identical to pre-M5 behavior. M5.3 (switching call sites to
    // *WeightNz variants) is out of scope for the agent that landed this —
    // see the comment on set_use_nz_weights() above.
    bool use_nz_weights_ = false;
    bool nz_applied_     = false;  // true once weights have been converted.

    // ---- A16W8 weight quantization state (Stretch S1) -----------------------
    // Opt-in flag + post-init truth bit; see set_use_w8_weights() above for
    // the full gating semantics. The F16 weight buffer is dropped and replaced
    // by INT8 weight + F16 scales when calibration runs (saving ~50% weight
    // memory). Mutually exclusive with nz_applied_: if both flags were
    // requested we take W8 (opt-in for the Stretch track) and warn.
    bool use_w8_weights_ = false;
    bool w8_applied_     = false;

    // ---- Internal helpers ----
    void alloc_dev_(void **ptr, size_t bytes);
    void ensure_workspace_(size_t needed);
    void upload_(void *dev, const void *host, size_t bytes);
    void build_rope_tables_();   // populate cos/sin and upload as F16.
    void build_causal_mask_();   // fill causal_mask_dev_ once at init.

    // Apply aclnnTransMatmulWeight to one [out, in] F16 matmul weight buffer
    // in place. Called from init_from_gguf for every projection weight when
    // use_nz_weights_ && g_cann.has_nz(). Safe to call with unsupported
    // runtime: returns without touching the buffer (caller must have already
    // gated on has_nz()).
    void nz_convert_weight_(void *weight_dev, int64_t rows, int64_t cols);

    // A16W8 calibration: given a host F32 weight buffer [rows, cols] (row-
    // major, [out, in]), compute per-output-channel symmetric INT8
    // quantization. Allocates NEW device buffers at `weight_i8_dev` (INT8
    // [rows, cols]) and `scale_dev` (F16 [rows]). The original F16
    // `weight_f16_dev` is left untouched — prefill still reads from it, so
    // S1 pays the double-storage cost. `rows = out_features`,
    // `cols = in_features`. Returns true on success.
    bool w8_calibrate_weight_(const float *host_w, int64_t rows, int64_t cols,
                               void *&weight_i8_dev, void *&scale_dev);

    // Dispatch one W8 matmul: y = x @ dequant(weight, scale).
    //   x          [M, K] F16 activation (ND)
    //   weight     [N, K] INT8 weight ([out, in], row-major). The V3 op
    //              expects mat2 to be transposable — it handles the transpose
    //              internally given a [K, N]-shaped view with strides (1, K)
    //              and the same N rows-major buffer.
    //   scale      [N]    F16 antiquant scale (per output channel)
    //   y          [M, N] F16 output (ND)
    // Prefers V3 when available, falls back to V2.
    void w8_matmul_(const aclTensor *x, void *weight_dev, void *scale_dev,
                     int64_t out_n, int64_t in_k, const aclTensor *y);

    // Core decode kernel sequence — what used to be the body of forward_decode
    // between the input-upload/cast and the final-output readback. Broken out
    // so capture mode can record exactly these ops while the surrounding H2D
    // and D2H transfers stay on the eager path. Assumes `cur_dev_` already
    // holds the F16 input and leaves the final (post-norm) result in
    // `normed_dev_` ready to be cast to F32 and downloaded.
    void run_decode_ops_(int pos);

    // Same as run_decode_ops_ but WITHOUT the initial F32->F16 Cast at the
    // top. Used by the speculative split path (`_launch_cast` does the cast
    // explicitly, then `add_input_delta_f32` mutates cur_dev_, then
    // `_launch_layers` calls this to run the 28-layer body). Keeping
    // `run_decode_ops_` separate preserves the aclGraph capture contract on
    // the non-speculative path.
    void run_decode_body_(int pos);

    // Internal: F32 cast of input_stage_f32_dev_ -> cur_dev_ (F16) on stream_.
    // Factored out so `_launch_cast` and `run_decode_ops_` share the same
    // op call site.
    void cast_input_f32_to_f16_();
};
