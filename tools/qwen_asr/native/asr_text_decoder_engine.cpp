// ============================================================================
// AsrTextDecoderCannEngine implementation — see header for overview.
//
// The engine is a thin wrapper around TalkerCannEngine. The 28-layer backbone
// is reused verbatim (same ops, same dtype conventions, same RoPE). What this
// file adds:
//   - Host-side text-token embedding lookup from `token_embd.weight`.
//   - NPU lm_head port (Cast F32→F16, Mm [vocab,h] × [h,1], Cast F16→F32,
//     D2H) modelled on CpCannEngine::{init_lm_head_,forward_lm_head,
//     fetch_logits}. ASR has one head (vs CP's 15 codebook groups) and
//     vocab=151936 (vs CP's 2048) — larger buffers, otherwise identical.
//   - End-to-end `transcribe()`: mel → audio encoder → 3-phase prefill →
//     greedy decode loop with on-NPU lm_head → vocab.json detokenize.
// ============================================================================

#include "asr_text_decoder_engine.h"

#include "audio_io.h"
#include "ggml.h"
#include "ggml-cpp.h"
#include "gguf.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <sstream>

// ---------------------------------------------------------------------------
// Local F32↔F16 helpers. The upload path casts host F32 → F16 bitwise because
// tools/qwen_tts's `upload_tensor_f16` helper is file-local (anonymous ns) and
// not linkable from here.
// ---------------------------------------------------------------------------

static inline uint16_t fp32_to_fp16_local(float f) {
    __fp16 h = (__fp16)f;
    uint16_t bits;
    std::memcpy(&bits, &h, sizeof(bits));
    return bits;
}

static inline float fp16_to_fp32_local(uint16_t bits) {
    __fp16 h;
    std::memcpy(&h, &bits, sizeof(h));
    return (float)h;
}

// ---------------------------------------------------------------------------
// Minimal tensor helpers. We can't reuse the ones in talker_cann_engine.cpp
// (file-local anonymous ns), so reproduce the relevant subset here.
// ---------------------------------------------------------------------------

static aclTensor *asr_make_tensor(void *buf, int64_t rank,
                                   const int64_t *shape, const int64_t *strides,
                                   aclDataType dtype) {
    int64_t storage_len = 0;
    if (rank > 0) {
        int64_t max_off = 0;
        for (int64_t i = 0; i < rank; ++i) {
            if (shape[i] > 0) max_off += (shape[i] - 1) * strides[i];
        }
        storage_len = max_off + 1;
    }
    return g_cann.aclCreateTensor(shape, rank, dtype, strides, 0,
                                   ACL_FORMAT_ND, &storage_len, 1, buf);
}

static aclTensor *asr_tensor_1d(void *buf, int64_t n, aclDataType dtype) {
    int64_t shape[1]   = {n};
    int64_t strides[1] = {1};
    return asr_make_tensor(buf, 1, shape, strides, dtype);
}

static aclTensor *asr_tensor_2d(void *buf, int64_t d0, int64_t d1,
                                 aclDataType dtype) {
    int64_t shape[2]   = {d0, d1};
    int64_t strides[2] = {d1, 1};
    return asr_make_tensor(buf, 2, shape, strides, dtype);
}

#define ASR_ACL_CHECK(expr)                                                    \
    do {                                                                        \
        aclError _err = (expr);                                                 \
        if (_err != 0) {                                                        \
            fprintf(stderr, "[asr_native] ACL call failed (%s:%d) err=%d\n",   \
                    __FILE__, __LINE__, (int)_err);                             \
            std::abort();                                                       \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// dtor — release NPU buffers owned by the lm_head side-car.
// ---------------------------------------------------------------------------

AsrTextDecoderCannEngine::~AsrTextDecoderCannEngine() {
    if (!cp_cann_load_symbols()) return;  // already unloaded
    auto free_dev = [](void *&p) {
        if (p) { g_cann.aclrtFree(p); p = nullptr; }
    };
    free_dev(lm_head_w_dev_);
    free_dev(lm_head_h_f16_dev_);
    free_dev(lm_head_h_f32_dev_);
    free_dev(logits_f16_dev_);
    free_dev(logits_f32_dev_);
    free_dev(lm_workspace_dev_);
    lm_head_ready_ = false;
    ready_ = false;
}

// ---------------------------------------------------------------------------
// Load `token_embd.weight` from the decoder GGUF and upload the tied lm_head
// weight to the NPU.
//
// Qwen3-ASR has `tie_word_embeddings=true`. export_decoder_llama.py writes
// both `token_embd.weight` and `output.weight` (which may or may not have
// identical byte patterns depending on PyTorch tie-handling), but we prefer
// `token_embd.weight` since (a) it's guaranteed present and (b) the host-side
// table is reused for per-token embedding lookup during prefill/decode.
// ---------------------------------------------------------------------------

bool AsrTextDecoderCannEngine::load_embedding_and_lm_head_(const std::string &gguf_path) {
    ggml_context *ctx = nullptr;
    gguf_init_params params;
    params.no_alloc = false;
    params.ctx      = &ctx;
    gguf_context *gg = gguf_init_from_file(gguf_path.c_str(), params);
    if (!gg || !ctx) {
        fprintf(stderr, "[asr_native] failed to open GGUF %s\n",
                gguf_path.c_str());
        return false;
    }

    ggml_tensor *t = ggml_get_tensor(ctx, "token_embd.weight");
    if (!t) {
        fprintf(stderr, "[asr_native] missing token_embd.weight\n");
        gguf_free(gg); ggml_free(ctx);
        return false;
    }

    const size_t expected = (size_t)vocab_size_ * n_embd_;
    const size_t nelems   = ggml_nelements(t);
    if (nelems != expected) {
        fprintf(stderr, "[asr_native] token_embd.weight has %zu elems, expected %zu\n",
                nelems, expected);
        gguf_free(gg); ggml_free(ctx);
        return false;
    }

    // Host-side F32 table (reused for per-token embedding lookup).
    // A4: Q8_0 (and any block-quant format) is dequantized via ggml's type
    // traits — matches llama.cpp's CPU dequantize path bit-for-bit.
    tok_embd_host_.resize(expected);
    if (t->type == GGML_TYPE_F32) {
        std::memcpy(tok_embd_host_.data(), t->data, expected * sizeof(float));
    } else if (t->type == GGML_TYPE_F16) {
        const ggml_fp16_t *src = (const ggml_fp16_t *)t->data;
        for (size_t i = 0; i < expected; ++i)
            tok_embd_host_[i] = ggml_fp16_to_fp32(src[i]);
    } else {
        const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
        if (!tr || !tr->to_float) {
            fprintf(stderr,
                    "[asr_native] token_embd.weight unsupported type %d "
                    "(no to_float trait)\n", (int)t->type);
            gguf_free(gg); ggml_free(ctx);
            return false;
        }
        tr->to_float(t->data, tok_embd_host_.data(), (int64_t)expected);
    }

    // Upload as F16 lm_head weight [vocab, n_embd].
    std::vector<uint16_t> f16buf(expected);
    for (size_t i = 0; i < expected; ++i)
        f16buf[i] = fp32_to_fp16_local(tok_embd_host_[i]);

    ASR_ACL_CHECK(g_cann.aclrtMalloc(&lm_head_w_dev_,
                                      expected * sizeof(uint16_t),
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    ASR_ACL_CHECK(g_cann.aclrtMemcpy(
        lm_head_w_dev_, expected * sizeof(uint16_t),
        f16buf.data(), expected * sizeof(uint16_t),
        ACL_MEMCPY_HOST_TO_DEVICE));

    // Hidden + logits staging buffers.
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&lm_head_h_f32_dev_,
                                      (size_t)n_embd_ * sizeof(float),
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&lm_head_h_f16_dev_,
                                      (size_t)n_embd_ * sizeof(uint16_t),
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&logits_f16_dev_,
                                      (size_t)vocab_size_ * sizeof(uint16_t),
                                      ACL_MEM_MALLOC_HUGE_FIRST));
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&logits_f32_dev_,
                                      (size_t)vocab_size_ * sizeof(float),
                                      ACL_MEM_MALLOC_HUGE_FIRST));

    // Seed workspace (grows on demand in ensure_lm_workspace_).
    lm_workspace_size_ = 4 * 1024 * 1024;
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&lm_workspace_dev_,
                                      lm_workspace_size_,
                                      ACL_MEM_MALLOC_HUGE_FIRST));

    // Borrow the talker's primary stream so serialization with decode is
    // free (same stream ⇒ ops are in-order, no explicit fence needed).
    lm_stream_ = talker_.get_primary_stream();

    lm_head_ready_ = true;
    printf("[asr_native] lm_head uploaded (F16 [vocab=%d, n_embd=%d], %.1f MB)\n",
           vocab_size_, n_embd_,
           (double)(expected * sizeof(uint16_t)) / 1048576.0);

    gguf_free(gg);
    ggml_free(ctx);
    return true;
}

void AsrTextDecoderCannEngine::ensure_lm_workspace_(size_t bytes) {
    if (bytes <= lm_workspace_size_) return;
    if (lm_workspace_dev_) {
        g_cann.aclrtFree(lm_workspace_dev_);
        lm_workspace_dev_ = nullptr;
    }
    lm_workspace_size_ = bytes;
    ASR_ACL_CHECK(g_cann.aclrtMalloc(&lm_workspace_dev_, lm_workspace_size_,
                                      ACL_MEM_MALLOC_HUGE_FIRST));
}

// ---------------------------------------------------------------------------
// init — load audio encoder, text decoder, tokenizer; upload lm_head.
// ---------------------------------------------------------------------------

bool AsrTextDecoderCannEngine::init(const AsrConfig &cfg) {
    if (ready_) return true;
    cfg_ = cfg;

    // 1. Mel filters (optional but recommended for parity).
    if (!cfg.mel_filters_path.empty()) {
        if (!mel_.load_mel_filterbank(cfg.mel_filters_path)) {
            printf("[asr_native] mel filter load failed, using default filterbank\n");
        } else {
            printf("[asr_native] mel filter loaded from %s\n",
                   cfg.mel_filters_path.c_str());
        }
    }

    // 2. Tokenizer.
    tokenizer_ = std::make_unique<BpeTokenizer>();
    if (!tokenizer_->load(cfg.vocab_path, cfg.merges_path)) {
        fprintf(stderr, "[asr_native] tokenizer load failed\n");
        return false;
    }
    // Reverse vocab for detokenization.
    init_reverse_vocab_(cfg.vocab_path);

    // 3. Audio encoder. Must be loaded on CANN (or CPU) — baseline uses CANN0
    //    when gpu_layers > 0, which is our steady-state config.
    if (!audio_encoder_.load(cfg.audio_encoder_path, cfg.device, cfg.n_threads)) {
        fprintf(stderr, "[asr_native] audio encoder load failed on %s; "
                "trying CPU\n", cfg.device.c_str());
        if (!audio_encoder_.load(cfg.audio_encoder_path, "CPU", cfg.n_threads))
            return false;
    }

    // 4. Text decoder via TalkerCannEngine. Configure for Qwen3-ASR-1.7B
    //    (same shape as Qwen3-TTS Talker except `vocab_size` denotes the
    //    text vocabulary).
    talker_cfg_.hidden_size          = 2048;
    talker_cfg_.num_hidden_layers    = 28;
    talker_cfg_.num_attention_heads  = 16;
    talker_cfg_.num_key_value_heads  = 8;
    talker_cfg_.intermediate_size    = 6144;
    talker_cfg_.head_dim             = 128;
    talker_cfg_.rope_theta           = 1000000.0f;
    talker_cfg_.rms_norm_eps         = 1e-6f;
    talker_cfg_.max_position_embeddings = 32768;
    // The engine reads only the shape fields above; vocab entries are
    // irrelevant here (no embedding lookup inside the engine itself).

    n_embd_     = talker_cfg_.hidden_size;
    vocab_size_ = 151936;

    if (!talker_.init_from_gguf(cfg.decoder_path, talker_cfg_, /*device=*/0)) {
        fprintf(stderr, "[asr_native] TalkerCannEngine init_from_gguf failed\n");
        return false;
    }

    // 5. Load tok_embd + upload lm_head (tied).
    if (!load_embedding_and_lm_head_(cfg.decoder_path)) return false;

    ready_ = true;
    printf("[asr_native] ready (28L x 2048, vocab=%d, audio encoder on %s)\n",
           vocab_size_, cfg.device.c_str());
    return true;
}

// ---------------------------------------------------------------------------
// Build the prompt segments (baseline parity with QwenASR::build_prompt_segments).
// ---------------------------------------------------------------------------

void AsrTextDecoderCannEngine::build_prompt_segments_(std::vector<int> &pre,
                                                       std::vector<int> &post) {
    pre.clear(); post.clear();
    auto nl = tokenizer_->encode("\n");

    pre.push_back(cfg_.im_start_id);
    auto sys = tokenizer_->encode("system\n");
    pre.insert(pre.end(), sys.begin(), sys.end());
    pre.push_back(cfg_.im_end_id);
    pre.insert(pre.end(), nl.begin(), nl.end());
    pre.push_back(cfg_.im_start_id);
    auto user = tokenizer_->encode("user\n");
    pre.insert(pre.end(), user.begin(), user.end());
    pre.push_back(cfg_.audio_start_id);

    post.push_back(cfg_.audio_end_id);
    post.push_back(cfg_.im_end_id);
    post.insert(post.end(), nl.begin(), nl.end());
    post.push_back(cfg_.im_start_id);
    auto asst = tokenizer_->encode("assistant\n");
    post.insert(post.end(), asst.begin(), asst.end());
}

// ---------------------------------------------------------------------------
// Lookup tokens → F32 [N, n_embd] on host from tok_embd_host_.
// ---------------------------------------------------------------------------

std::vector<float> AsrTextDecoderCannEngine::embed_tokens_(const std::vector<int> &tokens) {
    std::vector<float> out((size_t)tokens.size() * n_embd_);
    for (size_t i = 0; i < tokens.size(); ++i) {
        int id = tokens[i];
        if (id < 0 || id >= vocab_size_) {
            fprintf(stderr, "[asr_native] token id %d out of range\n", id);
            std::memset(&out[i * n_embd_], 0, (size_t)n_embd_ * sizeof(float));
            continue;
        }
        std::memcpy(&out[i * n_embd_],
                     &tok_embd_host_[(size_t)id * n_embd_],
                     (size_t)n_embd_ * sizeof(float));
    }
    return out;
}

// ---------------------------------------------------------------------------
// forward_lm_head_ — upload hidden, run Cast+Mm+Cast on NPU, D2H logits.
// Mirrors CpCannEngine::{forward_lm_head, fetch_logits} but operates on a
// host-provided hidden (talker_.forward_decode already D2H'd it) so we have
// to H2D it first. Future optimisation: keep hidden on NPU and fence across
// streams (W1 pattern), avoiding the extra round-trip. For A1b-v2 we prefer
// the simpler host-staging path — matches the llama.cpp baseline's logic
// and keeps the code trivial to reason about.
// ---------------------------------------------------------------------------

void AsrTextDecoderCannEngine::forward_lm_head_(const float *hidden_f32,
                                                 std::vector<float> &logits_out) {
    // H2D the F32 hidden [n_embd] into a pinned device buffer.
    ASR_ACL_CHECK(g_cann.aclrtMemcpyAsync(
        lm_head_h_f32_dev_, (size_t)n_embd_ * sizeof(float),
        hidden_f32,         (size_t)n_embd_ * sizeof(float),
        ACL_MEMCPY_HOST_TO_DEVICE, lm_stream_));

    // 1. Cast F32 hidden -> F16.
    {
        aclTensor *t_h_f32 = asr_tensor_1d(lm_head_h_f32_dev_, n_embd_, ACL_FLOAT);
        aclTensor *t_h_f16 = asr_tensor_1d(lm_head_h_f16_dev_, n_embd_, ACL_FLOAT16);
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        ASR_ACL_CHECK(g_cann.aclnnCastGetWorkspaceSize(
            t_h_f32, ACL_FLOAT16, t_h_f16, &ws, &exec));
        ensure_lm_workspace_(ws);
        ASR_ACL_CHECK(g_cann.aclnnCast(lm_workspace_dev_, ws, exec, lm_stream_));
        g_cann.aclDestroyTensor(t_h_f32);
        g_cann.aclDestroyTensor(t_h_f16);
    }

    // 2. Mm: logits[vocab] = W[vocab, n_embd] @ h[n_embd, 1].
    {
        aclTensor *t_w = asr_tensor_2d(lm_head_w_dev_, vocab_size_, n_embd_, ACL_FLOAT16);
        aclTensor *t_x = asr_tensor_2d(lm_head_h_f16_dev_, n_embd_, 1,       ACL_FLOAT16);
        aclTensor *t_y = asr_tensor_2d(logits_f16_dev_,    vocab_size_, 1,   ACL_FLOAT16);
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        ASR_ACL_CHECK(g_cann.aclnnMmGetWorkspaceSize(
            t_w, t_x, t_y, /*cubeMathType=*/0, &ws, &exec));
        ensure_lm_workspace_(ws);
        ASR_ACL_CHECK(g_cann.aclnnMm(lm_workspace_dev_, ws, exec, lm_stream_));
        g_cann.aclDestroyTensor(t_w);
        g_cann.aclDestroyTensor(t_x);
        g_cann.aclDestroyTensor(t_y);
    }

    // 3. Cast F16 logits -> F32 staging.
    {
        aclTensor *t_l_f16 = asr_tensor_1d(logits_f16_dev_, vocab_size_, ACL_FLOAT16);
        aclTensor *t_l_f32 = asr_tensor_1d(logits_f32_dev_, vocab_size_, ACL_FLOAT);
        uint64_t ws = 0;
        aclOpExecutor *exec = nullptr;
        ASR_ACL_CHECK(g_cann.aclnnCastGetWorkspaceSize(
            t_l_f16, ACL_FLOAT, t_l_f32, &ws, &exec));
        ensure_lm_workspace_(ws);
        ASR_ACL_CHECK(g_cann.aclnnCast(lm_workspace_dev_, ws, exec, lm_stream_));
        g_cann.aclDestroyTensor(t_l_f16);
        g_cann.aclDestroyTensor(t_l_f32);
    }

    // 4. D2H + sync.
    ASR_ACL_CHECK(g_cann.aclrtSynchronizeStream(lm_stream_));
    logits_out.resize(vocab_size_);
    ASR_ACL_CHECK(g_cann.aclrtMemcpy(
        logits_out.data(),  (size_t)vocab_size_ * sizeof(float),
        logits_f32_dev_,    (size_t)vocab_size_ * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST));
}

// ---------------------------------------------------------------------------
// transcribe_from_features — the core driver. Takes pre-computed audio
// features (the encoder's [num_frames, 2048] F32 output) and produces text.
//
// Flow:
//   Phase 1: prefill pre-tokens (embedded on host).
//   Phase 2: prefill audio embeddings (already F32 [N, 2048]).
//   Phase 3: prefill post-tokens (this one returns a hidden state).
//   Phase 4: sample from logits, decode_one, loop until EOS / im_end.
//
// KV cache positions:
//   pre:  [0,               pre_len)
//   audio:[pre_len,          pre_len + audio_len)
//   post: [pre_len+audio_len,pre_len + audio_len + post_len)
//   gen:  [..., ..+1), one per step
// ---------------------------------------------------------------------------

bool AsrTextDecoderCannEngine::transcribe_from_features(
    const std::vector<float> &audio_features, int num_audio_frames,
    std::string &text_out) {
    if (!ready_) {
        fprintf(stderr, "[asr_native] not ready\n");
        return false;
    }
    auto t0 = std::chrono::high_resolution_clock::now();

    // Reset KV cache between utterances.
    talker_.reset_kv_cache();

    std::vector<int> pre_tokens, post_tokens;
    build_prompt_segments_(pre_tokens, post_tokens);

    const int pre_len   = (int)pre_tokens.size();
    const int audio_len = num_audio_frames;
    const int post_len  = (int)post_tokens.size();
    const int total_len = pre_len + audio_len + post_len;
    printf("[asr_native] prompt: %d (pre=%d + audio=%d + post=%d)\n",
           total_len, pre_len, audio_len, post_len);

    // Phase 1: pre-tokens.
    std::vector<float> pre_embeds = embed_tokens_(pre_tokens);
    talker_.forward_prefill(pre_embeds.data(), pre_len,
                              /*start_pos=*/0,
                              /*last_hidden_out=*/nullptr);
    auto t1 = std::chrono::high_resolution_clock::now();

    // Phase 2: audio features (already F32 [N, 2048]).
    talker_.forward_prefill(audio_features.data(), audio_len,
                              /*start_pos=*/pre_len,
                              /*last_hidden_out=*/nullptr);
    auto t2 = std::chrono::high_resolution_clock::now();

    // Phase 3: post-tokens — we want the last hidden back.
    std::vector<float> post_embeds = embed_tokens_(post_tokens);
    std::vector<float> last_hidden((size_t)n_embd_, 0.0f);
    talker_.forward_prefill(post_embeds.data(), post_len,
                              /*start_pos=*/pre_len + audio_len,
                              last_hidden.data());
    auto t3 = std::chrono::high_resolution_clock::now();

    // Phase 4: autoregressive greedy decode using NPU lm_head.
    std::vector<int> gen_tokens;
    int pos = pre_len + audio_len + post_len;  // next position to write.
    std::vector<float> logits;
    std::vector<float> next_embed((size_t)n_embd_, 0.0f);

    forward_lm_head_(last_hidden.data(), logits);
    for (int step = 0; step < cfg_.max_new_tokens; ++step) {
        // Greedy argmax.
        int best = 0;
        float best_val = logits[0];
        for (int i = 1; i < vocab_size_; ++i) {
            if (logits[i] > best_val) { best_val = logits[i]; best = i; }
        }

        if (best == cfg_.im_end_id || best == cfg_.endoftext_id) break;
        gen_tokens.push_back(best);

        // Embed this token and run one decode step.
        std::memcpy(next_embed.data(),
                     &tok_embd_host_[(size_t)best * n_embd_],
                     (size_t)n_embd_ * sizeof(float));

        std::vector<float> hidden((size_t)n_embd_, 0.0f);
        talker_.forward_decode(next_embed.data(), pos, hidden.data());
        pos++;

        forward_lm_head_(hidden.data(), logits);
    }
    auto t4 = std::chrono::high_resolution_clock::now();

    text_out = decode_tokens_(gen_tokens);
    timing_.gen_tokens = (int)gen_tokens.size();
    timing_.pre_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing_.audio_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing_.post_ms  = std::chrono::duration<double, std::milli>(t3 - t2).count();
    timing_.gen_ms   = std::chrono::duration<double, std::milli>(t4 - t3).count();
    timing_.dec_ms   = std::chrono::duration<double, std::milli>(t4 - t0).count();

    printf("[asr_native] gen %zu tokens | pre=%.0fms audio=%.0fms post=%.0fms gen=%.0fms\n",
           gen_tokens.size(),
           timing_.pre_ms, timing_.audio_ms, timing_.post_ms, timing_.gen_ms);
    return true;
}

// ---------------------------------------------------------------------------
// transcribe(samples) / transcribe_file — wrap mel + encoder + decoder.
// ---------------------------------------------------------------------------

bool AsrTextDecoderCannEngine::transcribe(const std::vector<float> &audio_16k,
                                           std::string &text_out) {
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> mel;
    int mel_T = 0;
    mel_.compute(audio_16k, mel, mel_T);
    auto t1 = std::chrono::high_resolution_clock::now();

    std::vector<float> audio_features;
    int num_audio_frames = 0;
    audio_encoder_.encode(mel, mel_T, audio_features, num_audio_frames);
    auto t2 = std::chrono::high_resolution_clock::now();

    bool ok = transcribe_from_features(audio_features, num_audio_frames, text_out);
    auto t3 = std::chrono::high_resolution_clock::now();

    timing_.mel_ms   = std::chrono::duration<double, std::milli>(t1 - t0).count();
    timing_.enc_ms   = std::chrono::duration<double, std::milli>(t2 - t1).count();
    timing_.dec_ms   = std::chrono::duration<double, std::milli>(t3 - t2).count();
    timing_.total_ms = std::chrono::duration<double, std::milli>(t3 - t0).count();

    printf("[asr_native] timing: mel=%.0fms, encoder=%.0fms, decoder=%.0fms, total=%.0fms\n",
           timing_.mel_ms, timing_.enc_ms, timing_.dec_ms, timing_.total_ms);
    return ok;
}

bool AsrTextDecoderCannEngine::transcribe_file(const std::string &audio_path,
                                                std::string &text_out) {
    std::vector<float> samples;
    if (!audio_io::load_audio(audio_path, 16000, samples)) {
        fprintf(stderr, "[asr_native] failed to load audio %s\n", audio_path.c_str());
        return false;
    }
    printf("[asr_native] audio: %zu samples (%.2fs)\n",
           samples.size(), (float)samples.size() / 16000.f);
    return transcribe(samples, text_out);
}

// ---------------------------------------------------------------------------
// Detokenization (byte-level BPE, mirrors QwenASR::init_reverse_vocab +
// bpe_unicode_to_bytes + decode_tokens exactly so transcripts are comparable
// to the A1a baseline byte-for-byte).
// ---------------------------------------------------------------------------

void AsrTextDecoderCannEngine::init_reverse_vocab_(const std::string &vocab_path) {
    std::memset(unicode_to_byte_, 0, sizeof(unicode_to_byte_));
    int n = 0;
    for (int b = 0; b < 256; ++b) {
        int cp = ((b>=33&&b<=126)||(b>=161&&b<=172)||(b>=174&&b<=255)) ? b : 256+n++;
        if (cp < 512) unicode_to_byte_[cp] = (uint8_t)b;
    }
    std::ifstream f(vocab_path);
    if (!f.is_open()) return;
    std::stringstream ss; ss << f.rdbuf(); f.close();
    std::string content = ss.str();

    size_t p = 0;
    while (p < content.size()) {
        size_t q1 = content.find('"', p);
        if (q1 == std::string::npos) break;
        size_t q2 = q1 + 1;
        while (q2 < content.size()) {
            if (content[q2] == '\\') { q2 += 2; continue; }
            if (content[q2] == '"') break;
            q2++;
        }
        if (q2 >= content.size()) break;

        std::string key_raw = content.substr(q1 + 1, q2 - q1 - 1);
        std::string key;
        for (size_t k = 0; k < key_raw.size(); ++k) {
            if (key_raw[k] == '\\' && k + 1 < key_raw.size()) {
                char e = key_raw[k + 1];
                if      (e == '"')  { key += '"';  k++; }
                else if (e == '\\') { key += '\\'; k++; }
                else if (e == 'n')  { key += '\n'; k++; }
                else if (e == 't')  { key += '\t'; k++; }
                else if (e == 'u' && k + 5 < key_raw.size()) {
                    uint32_t cp = (uint32_t)strtol(key_raw.substr(k+2, 4).c_str(), nullptr, 16);
                    if (cp < 0x80) key += (char)cp;
                    else if (cp < 0x800) {
                        key += (char)(0xC0 | (cp >> 6));
                        key += (char)(0x80 | (cp & 0x3F));
                    } else {
                        key += (char)(0xE0 | (cp >> 12));
                        key += (char)(0x80 | ((cp >> 6) & 0x3F));
                        key += (char)(0x80 | (cp & 0x3F));
                    }
                    k += 5;
                } else { key += key_raw[k]; }
            } else { key += key_raw[k]; }
        }
        size_t colon = content.find(':', q2 + 1);
        if (colon == std::string::npos) break;
        size_t ns = content.find_first_of("0123456789", colon + 1);
        if (ns == std::string::npos) break;
        size_t ne = content.find_first_not_of("0123456789", ns);
        if (ne == std::string::npos) ne = content.size();

        int id = atoi(content.substr(ns, ne - ns).c_str());
        id_to_bytes_[id] = bpe_unicode_to_bytes_(key);
        p = ne;
    }
    printf("[asr_native] vocab loaded: %zu entries\n", id_to_bytes_.size());
}

std::string AsrTextDecoderCannEngine::bpe_unicode_to_bytes_(const std::string &s) {
    std::string r;
    size_t p = 0;
    while (p < s.size()) {
        uint32_t cp = 0;
        uint8_t c = (uint8_t)s[p];
        int len = (c < 0x80) ? 1 : (c < 0xE0) ? 2 : (c < 0xF0) ? 3 : 4;
        cp = (len == 1) ? c : (len == 2) ? (c & 0x1F) : (len == 3) ? (c & 0x0F) : (c & 0x07);
        for (int i = 1; i < len && p + i < s.size(); ++i)
            cp = (cp << 6) | ((uint8_t)s[p + i] & 0x3F);
        p += len;
        if (cp < 512) r += (char)unicode_to_byte_[cp];
        else if (cp < 0x80) r += (char)cp;
        else if (cp < 0x800) {
            r += (char)(0xC0 | (cp >> 6));
            r += (char)(0x80 | (cp & 0x3F));
        } else if (cp < 0x10000) {
            r += (char)(0xE0 | (cp >> 12));
            r += (char)(0x80 | ((cp >> 6) & 0x3F));
            r += (char)(0x80 | (cp & 0x3F));
        } else {
            r += (char)(0xF0 | (cp >> 18));
            r += (char)(0x80 | ((cp >> 12) & 0x3F));
            r += (char)(0x80 | ((cp >> 6) & 0x3F));
            r += (char)(0x80 | (cp & 0x3F));
        }
    }
    return r;
}

std::string AsrTextDecoderCannEngine::decode_tokens_(const std::vector<int> &ids) {
    std::string r;
    for (int id : ids) {
        auto it = id_to_bytes_.find(id);
        if (it != id_to_bytes_.end()) r += it->second;
    }
    return r;
}
