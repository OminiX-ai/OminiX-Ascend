#pragma once

#include "build_graph.h"
#include "ctx_manager.h"
#include "ggml.h"
#include "model_defs.h"
#include "model_loader.h"
#include "infer_session.hpp"
#include <memory>
#include <string>
#include <vector>

// Speech Tokenizer Decoder for Qwen3-TTS
// Converts codec tokens (16 quantizers × T) to 24kHz audio waveform
//
// Architecture:
//   RVQ decode: codes[16,T] → [512,T]
//   pre_conv: Conv1d(512→1024, k=3, causal)
//   pre_transformer: input_proj(1024→512) → 8 layers (sliding window=72) → output_proj(512→1024)
//   upsample: 2 blocks (ratio 2,2) with ConvTranspose1d + ConvNeXt → 4× upsample
//   vocoder: Conv1d(1024→1536) → 4 blocks (rates 8,5,4,3) with SnakeBeta → Conv1d(96→1)
//   Total upsample: 4 × 480 = 1920×

struct DecoderConfig {
    int codebook_size = 2048;
    int codebook_dim = 256;
    int hidden_size = 512;      // transformer hidden dim
    int latent_dim = 1024;      // pre_conv output / upsample dim
    int num_hidden_layers = 8;
    int num_attention_heads = 16;
    int num_key_value_heads = 16;
    int head_dim = 64;
    int num_quantizers = 16;
    int decoder_dim = 1536;
    int sliding_window = 72;
    float rope_theta = 10000.0f;
    float rms_norm_eps = 1e-5f;
    int upsample_rates[4] = {8, 5, 4, 3};
    int upsampling_ratios[2] = {2, 2};
    int output_sample_rate = 24000;
    int decode_upsample_rate = 1920;
};

// RVQ codebook for one quantizer layer
struct RVQCodebook {
    ggml_tensor *cluster_usage = nullptr;  // [2048]
    ggml_tensor *embedding_sum = nullptr;  // [256, 2048]
};

// One RVQ group (rvq_first or rvq_rest)
struct RVQGroup {
    ggml_tensor *input_proj_w = nullptr;   // [1, 512, 256] (unused in decode)
    ggml_tensor *output_proj_w = nullptr;  // [1, 256, 512]
    std::vector<RVQCodebook> codebooks;
};

// Pre-transformer layer (sliding window attention + SwiGLU MLP + layer scale)
struct PreTransformerLayer {
    ggml_tensor *q_proj_w = nullptr;    // [hidden, n_heads*head_dim]
    ggml_tensor *k_proj_w = nullptr;
    ggml_tensor *v_proj_w = nullptr;
    ggml_tensor *o_proj_w = nullptr;    // [n_heads*head_dim, hidden]
    ggml_tensor *gate_proj_w = nullptr;
    ggml_tensor *up_proj_w = nullptr;
    ggml_tensor *down_proj_w = nullptr;
    ggml_tensor *input_layernorm_w = nullptr;       // [hidden]
    ggml_tensor *post_attention_layernorm_w = nullptr;
    ggml_tensor *self_attn_layer_scale = nullptr;   // [hidden]
    ggml_tensor *mlp_layer_scale = nullptr;         // [hidden]
};

// ConvNeXt block for upsample stage
struct ConvNeXtBlock {
    ggml_tensor *gamma = nullptr;          // [dim]
    ggml_tensor *dwconv_w = nullptr;       // [7, 1, dim] depthwise
    ggml_tensor *dwconv_b = nullptr;       // [dim]
    ggml_tensor *norm_w = nullptr;         // [dim]
    ggml_tensor *norm_b = nullptr;         // [dim]
    ggml_tensor *pwconv1_w = nullptr;      // [dim, 4*dim]
    ggml_tensor *pwconv1_b = nullptr;      // [4*dim]
    ggml_tensor *pwconv2_w = nullptr;      // [4*dim, dim]
    ggml_tensor *pwconv2_b = nullptr;      // [dim]
};

// Upsample block: ConvTranspose1d + ConvNeXt
struct UpsampleBlock {
    ggml_tensor *conv_w = nullptr;  // transposed conv weight
    ggml_tensor *conv_b = nullptr;
    ConvNeXtBlock convnext;
};

// SnakeBeta activation parameters
struct SnakeBetaParams {
    ggml_tensor *alpha = nullptr;
    ggml_tensor *beta = nullptr;
};

// Vocoder residual unit: SnakeBeta → Conv1d(k=7,dilation) → SnakeBeta → Conv1d(k=1)
struct VocoderResUnit {
    SnakeBetaParams act1;
    ggml_tensor *conv1_w = nullptr;
    ggml_tensor *conv1_b = nullptr;
    SnakeBetaParams act2;
    ggml_tensor *conv2_w = nullptr;
    ggml_tensor *conv2_b = nullptr;
};

// Vocoder decoder block: SnakeBeta → ConvTranspose1d → 3 residual units
struct VocoderBlock {
    SnakeBetaParams snake;
    ggml_tensor *transconv_w = nullptr;
    ggml_tensor *transconv_b = nullptr;
    VocoderResUnit res_units[3];  // dilations: 1, 3, 9
};

class SpeechTokenizerDecoderModel : public BaseModel {
public:
    DecoderConfig config;
    int debug_stop_after_ = 0;  // 0=full, 1=pre_transformer, 2=upsample, 3=voc_init, 4-7=voc_block[0-3]

    bool load_hparams(const ModelLoader &loader) override;
    std::vector<ggml_tensor *> get_tensors_to_load(ggml_context *ctx) override;
    std::vector<ggml_tensor *> build_graph(ggml_context *ctx0) override;
    void reset_input_shape() override;

private:
    // RVQ quantizer
    RVQGroup rvq_first_;   // 1 codebook (semantic)
    RVQGroup rvq_rest_;    // 15 codebooks (acoustic)

    // Pre-conv (causal Conv1d 512→1024, k=3)
    ggml_tensor *pre_conv_w_ = nullptr;
    ggml_tensor *pre_conv_b_ = nullptr;

    // Pre-transformer
    ggml_tensor *pt_input_proj_w_ = nullptr;   // [1024, 512]
    ggml_tensor *pt_input_proj_b_ = nullptr;
    ggml_tensor *pt_output_proj_w_ = nullptr;  // [512, 1024]
    ggml_tensor *pt_output_proj_b_ = nullptr;
    ggml_tensor *pt_norm_w_ = nullptr;         // [512]
    std::vector<PreTransformerLayer> pt_layers_;

    // Upsample (2 blocks)
    UpsampleBlock upsample_blocks_[2];

    // Vocoder
    ggml_tensor *voc_init_conv_w_ = nullptr;   // [7, 1024, 1536]
    ggml_tensor *voc_init_conv_b_ = nullptr;
    VocoderBlock voc_blocks_[4];               // rates: 8, 5, 4, 3
    SnakeBetaParams voc_final_snake_;          // [96]
    ggml_tensor *voc_final_conv_w_ = nullptr;  // [7, 96, 1]
    ggml_tensor *voc_final_conv_b_ = nullptr;

    // Graph building helpers
    ggml_tensor *build_rvq_decode(ggml_context *ctx0, ggml_tensor *codes);
    ggml_tensor *build_causal_conv1d(ggml_context *ctx0, ggml_tensor *x,
                                      ggml_tensor *w, ggml_tensor *b, int d = 1);
    ggml_tensor *build_depthwise_conv1d_causal(ggml_context *ctx0, ggml_tensor *x,
                                                ggml_tensor *w, ggml_tensor *b);
    ggml_tensor *build_causal_transconv1d(ggml_context *ctx0, ggml_tensor *x,
                                           ggml_tensor *w, ggml_tensor *b, int stride);
    ggml_tensor *build_snake_beta(ggml_context *ctx0, ggml_tensor *x,
                                   const SnakeBetaParams &params);
    ggml_tensor *build_convnext(ggml_context *ctx0, ggml_tensor *x,
                                 const ConvNeXtBlock &blk);
    ggml_tensor *build_pre_transformer(ggml_context *ctx0, ggml_tensor *x);
    ggml_tensor *build_vocoder(ggml_context *ctx0, ggml_tensor *x);
};

// High-level Speech Tokenizer Decoder interface
class SpeechTokenizerDecoder {
public:
    SpeechTokenizerDecoder() = default;
    ~SpeechTokenizerDecoder() = default;

    // Single-session mode (CPU only, backward compat)
    bool load(const std::string &model_path, const ContextParams &params);
    // Split mode: phase1 (NPU) + phase2 (CPU)
    bool load(const std::string &model_path,
              const ContextParams &npu_params,
              const ContextParams &cpu_params);

    // codes: [num_quantizers][time_steps], audio: output waveform
    bool decode(const std::vector<std::vector<int>> &codes,
                std::vector<float> &audio);

    // Track L (M6.4) chunk geometry — exposed so callers can align their own
    // frame-streaming schedule to the decoder's chunk window without having
    // to duplicate the constants.
    static int chunk_size() { return CHUNK_SIZE; }
    static int overlap_frames() { return OVERLAP_FRAMES; }
    static int chunk_step() { return CHUNK_SIZE - OVERLAP_FRAMES; }
    static int cann_max_frames() { return CANN_MAX_FRAMES; }

    // Track L (M6.4) per-chunk decode: runs a single chunk through the CANN
    // (or CPU fallback) session and returns the audio samples that belong to
    // this chunk *before* hard-cut stitching is applied. The caller is
    // responsible for trimming `skip_samples = skip_frames * upsample_rate`
    // from the front of the returned buffer when stitching multiple chunks
    // together (see decode_chunked() for the reference stitching pattern).
    // Returns false if both CANN and CPU sessions fail.
    //
    // This API is the primitive a frame-streaming producer (e.g. a future
    // version of TalkerLLM::generate() that yields codec frames as they are
    // sampled) can drive to overlap codec decode with the generation loop.
    // Today's generate() still invokes decode() on the fully-gathered code
    // sequence; decode() internally calls forward_chunk() via decode_chunked
    // when the sequence is longer than cann_max_frames().
    bool forward_chunk(const std::vector<std::vector<int>> &chunk_codes,
                       std::vector<float> &chunk_audio,
                       bool prefer_cann = true);

    int upsample_rate() const { return config_.decode_upsample_rate; }

private:
    DecoderConfig config_;
    bool split_mode_ = false;
    // Primary session (CANN or CPU)
    std::unique_ptr<InferenceSession<SpeechTokenizerDecoderModel>> session_;
    // CPU fallback session for long sequences (CANN fails at >100 total frames)
    std::unique_ptr<InferenceSession<SpeechTokenizerDecoderModel>> cpu_session_;
    static constexpr int CANN_MAX_FRAMES = 99;  // CANN decoder correct ≤99, fails ~100+ (CANN 8.5.0)
    static constexpr int CHUNK_SIZE = 96;       // chunk size per CANN pass
    static constexpr int OVERLAP_FRAMES = 72;   // must >= sliding_window (72) for correct attention context

    // Chunked decoding for long sequences (>99 frames)
    bool decode_chunked(const std::vector<std::vector<int>> &codes,
                        std::vector<float> &audio);

    // Single chunk decoding helper
    bool decode_single_chunk(const std::vector<std::vector<int>> &codes,
                             std::vector<float> &audio,
                             bool use_cann);
};
