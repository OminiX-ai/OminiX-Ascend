/**
 * qwen_tts_api.cpp — C API implementation wrapping existing C++ TTS primitives.
 *
 * Delegates to TalkerLLM (backbone + embeddings + code predictor),
 * SpeechTokenizerDecoder (vocoder), and SpeakerEncoder (ECAPA-TDNN).
 */

#include "qwen_tts_api.h"
#include "qwen_tts.h"
#include "talker.h"
#include "speaker_encoder.h"
#include "speech_tokenizer_encoder.h"
#include "speech_tokenizer_decoder.h"
#include "bpe_tokenizer.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

struct qwen_tts_ctx {
    TalkerLLM talker;
    SpeechTokenizerDecoder decoder;
    SpeakerEncoder speaker_encoder;
    SpeechTokenizerEncoder tokenizer_encoder;
    BpeTokenizer bpe_tokenizer;
    bool has_speaker_encoder;
    bool has_tokenizer_encoder;
    int hidden_size;
    int vocab_size;

    // --- High-level synthesize() path (B5) --------------------------------
    // Loader config captured at qwen_tts_load() so we can lazily construct
    // a QwenTTS wrapper on the first qwen_tts_synthesize() call without
    // re-plumbing the load arguments. The primitives above stay authoritative
    // for the fine-grained ABI (embed / forward / predict_codes / ...); the
    // QwenTTS instance below is a parallel, one-shot synthesis surface that
    // reuses the existing generation logic in qwen_tts.cpp verbatim.
    std::string load_model_dir;
    std::string load_tokenizer_dir;
    std::string load_talker_override;
    std::string load_cp_override;
    int         load_n_gpu_layers = 0;
    int         load_n_threads    = 0;
    std::unique_ptr<QwenTTS> synth;
    std::mutex  synth_mu;  // serialize synth init + generate (engine is !thread-safe per-handle)
};

extern "C" {

qwen_tts_ctx_t* qwen_tts_load(
    const char* model_dir,
    const char* tokenizer_dir,
    const char* talker_override,
    const char* cp_override,
    int n_gpu_layers,
    int n_threads
) {
    auto* ctx = new (std::nothrow) qwen_tts_ctx();
    if (!ctx) return nullptr;

    std::string mdir(model_dir);
    std::string tdir = tokenizer_dir ? std::string(tokenizer_dir) : mdir;
    if (mdir.back() != '/') mdir += '/';
    if (tdir.back() != '/') tdir += '/';

    // Capture load config so qwen_tts_synthesize() can lazily build a QwenTTS
    // with the same settings without re-plumbing args through the ABI.
    ctx->load_model_dir       = mdir;
    ctx->load_tokenizer_dir   = tdir;
    ctx->load_talker_override = talker_override ? talker_override : "";
    ctx->load_cp_override     = cp_override     ? cp_override     : "";
    ctx->load_n_gpu_layers    = n_gpu_layers;
    ctx->load_n_threads       = n_threads;

    // Load BPE tokenizer
    if (!ctx->bpe_tokenizer.load(tdir + "vocab.json", tdir + "merges.txt")) {
        fprintf(stderr, "[qwen_tts_api] failed to load BPE tokenizer\n");
        delete ctx;
        return nullptr;
    }

    // Load talker LLM. Default to the F16/F32 `qwen_tts_talker_llama.gguf`
    // because the native TalkerCannEngine (CP-CANN path) rejects Q-quantized
    // weights with "unsupported dtype". Q8_0 is only viable via the llama.cpp
    // fallback (QWEN_TTS_LLAMA=ON), which is off by default in the API build.
    // Honour an explicit caller override without second-guessing.
    std::string talker_llama = talker_override
        ? std::string(talker_override)
        : mdir + "qwen_tts_talker_llama.gguf";

    std::string cp_path = cp_override
        ? std::string(cp_override)
        : mdir + "qwen_tts_cp_llama.gguf";
    {
        FILE* f = fopen(cp_path.c_str(), "rb");
        if (!f) cp_path.clear();
        else fclose(f);
    }

    // Decide whether to activate the native CP-CANN / Talker-CANN paths.
    // Built-in when QWEN_TTS_HAS_CP_CANN is defined at compile time AND the
    // caller asked for NPU layers. Without llama.cpp linked, the native path
    // is the ONLY viable forward pass — load_model() returns false otherwise.
#ifdef QWEN_TTS_HAS_CP_CANN
    const bool use_cp_cann     = (n_gpu_layers > 0);
    const bool use_talker_cann = (n_gpu_layers > 0);
#else
    const bool use_cp_cann     = false;
    const bool use_talker_cann = false;
#endif

    if (!ctx->talker.load_model(
        talker_llama,
        mdir + "qwen_tts_talker.gguf",
        mdir + "qwen_tts_code_predictor.gguf",
        n_threads, n_gpu_layers, cp_path,
        use_cp_cann, use_talker_cann)) {
        fprintf(stderr, "[qwen_tts_api] failed to load talker\n");
        delete ctx;
        return nullptr;
    }

    ctx->hidden_size = ctx->talker.get_config().hidden_size;
    ctx->vocab_size = ctx->talker.get_config().vocab_size;

    // Load speech tokenizer decoder
    std::string dec_gguf = mdir + "qwen_tts_tokenizer_dec.gguf";
    {
        FILE* f = fopen(dec_gguf.c_str(), "rb");
        if (f) {
            fclose(f);
            ContextParams dec_params;
            dec_params.device_name = (n_gpu_layers > 0) ? "CANN0" : "CPU";
            dec_params.n_threads = n_threads;
            dec_params.max_nodes = 65536;
            if (ctx->decoder.load(dec_gguf, dec_params)) {
                printf("[qwen_tts_api] decoder loaded\n");
            }
        }
    }

    // Load speaker encoder (optional — Base model only)
    std::string spk_gguf = mdir + "qwen_tts_speaker_encoder.gguf";
    {
        FILE* f = fopen(spk_gguf.c_str(), "rb");
        if (f) {
            fclose(f);
            ContextParams spk_params;
            spk_params.device_name = "CPU"; // CANN 2.7x slower
            spk_params.n_threads = n_threads;
            ctx->has_speaker_encoder = ctx->speaker_encoder.load(spk_gguf, spk_params);
            if (ctx->has_speaker_encoder)
                printf("[qwen_tts_api] speaker encoder loaded\n");
        }
    }

    // Load speech tokenizer encoder (optional — for ICL clone)
    std::string enc_gguf = mdir + "qwen_tts_tokenizer_enc.gguf";
    {
        FILE* f = fopen(enc_gguf.c_str(), "rb");
        if (f) {
            fclose(f);
            ContextParams enc_params;
            enc_params.device_name = (n_gpu_layers > 0) ? "CANN0" : "CPU";
            enc_params.n_threads = n_threads;
            enc_params.max_nodes = 8192;
            ctx->has_tokenizer_encoder = ctx->tokenizer_encoder.load(enc_gguf, enc_params);
            if (ctx->has_tokenizer_encoder)
                printf("[qwen_tts_api] tokenizer encoder loaded\n");
        }
    }

    printf("[qwen_tts_api] loaded: hidden=%d vocab=%d spk=%d enc=%d\n",
           ctx->hidden_size, ctx->vocab_size,
           ctx->has_speaker_encoder, ctx->has_tokenizer_encoder);
    return ctx;
}

void qwen_tts_free(qwen_tts_ctx_t* ctx) {
    if (ctx) delete ctx;
}

int qwen_tts_hidden_size(const qwen_tts_ctx_t* ctx) {
    return ctx ? ctx->hidden_size : 0;
}

int qwen_tts_vocab_size(const qwen_tts_ctx_t* ctx) {
    return ctx ? ctx->vocab_size : 0;
}

int qwen_tts_has_speaker_encoder(const qwen_tts_ctx_t* ctx) {
    return ctx ? ctx->has_speaker_encoder : 0;
}

/* ========================================================================== */
/* Embedding operations                                                       */
/* ========================================================================== */

void qwen_tts_text_embed(qwen_tts_ctx_t* ctx, uint32_t token_id, float* out) {
    if (!ctx) return;
    // Delegates to TalkerLLM's lookup_text_projected
    // (which does text_embed → text_projection)
    ctx->talker.cache_tts_embeddings_public();
    ctx->talker.lookup_text_projected_public(token_id, out);
}

void qwen_tts_codec_embed(qwen_tts_ctx_t* ctx, uint32_t codec_token, float* out) {
    if (!ctx) return;
    ctx->talker.lookup_codec_embedding_public(codec_token, out);
}

void qwen_tts_codec_head(qwen_tts_ctx_t* ctx, const float* hidden, float* logits_out) {
    if (!ctx) return;
    ctx->talker.apply_codec_head_public(hidden, logits_out);
}

void qwen_tts_generation_embed(
    qwen_tts_ctx_t* ctx,
    const float* text_embed,
    const uint32_t* prev_codes,
    float* out
) {
    if (!ctx) return;
    int dim = ctx->hidden_size;

    // Sum codec embeddings for all 16 groups
    memset(out, 0, dim * sizeof(float));
    float tmp[4096]; // max hidden_size
    for (int g = 0; g < 16; g++) {
        ctx->talker.lookup_codec_embedding_public(prev_codes[g], tmp);
        for (int j = 0; j < dim; j++) out[j] += tmp[j];
    }

    // Add text embedding
    for (int j = 0; j < dim; j++) out[j] += text_embed[j];
}

/* ========================================================================== */
/* Transformer forward                                                        */
/* ========================================================================== */

void qwen_tts_reset_cache(qwen_tts_ctx_t* ctx) {
    if (!ctx) return;
    // Reset llama.cpp KV cache via the talker's internal context
    // This needs a public method on TalkerLLM
    ctx->talker.reset_cache_public();
}

int qwen_tts_forward(
    qwen_tts_ctx_t* ctx,
    const float* input_embeds,
    int seq_len,
    float* logits_out,
    float* hidden_out
) {
    if (!ctx) return -1;
    return ctx->talker.forward_public(input_embeds, seq_len, logits_out, hidden_out);
}

/* ========================================================================== */
/* Code prediction                                                            */
/* ========================================================================== */

int qwen_tts_predict_codes(
    qwen_tts_ctx_t* ctx,
    const float* hidden,
    uint32_t code0,
    uint32_t* codes_out
) {
    if (!ctx) return -1;
    std::vector<int> group_tokens;
    TalkerSamplingParams sampling;
    if (!ctx->talker.predict_code_groups(hidden, 1, (int)code0, group_tokens, sampling)) {
        return -1;
    }
    for (size_t i = 0; i < group_tokens.size() && i < 15; i++) {
        codes_out[i] = (uint32_t)group_tokens[i];
    }
    return 0;
}

/* ========================================================================== */
/* Speech decoder                                                             */
/* ========================================================================== */

int qwen_tts_decode_audio(
    qwen_tts_ctx_t* ctx,
    const uint32_t* codes,
    int n_frames,
    int n_groups,
    float* audio_out,
    int* n_samples_out
) {
    if (!ctx) return -1;

    // Convert flat codes to vector<vector<int>> format
    std::vector<std::vector<int>> code_vecs(n_groups);
    for (int g = 0; g < n_groups; g++) {
        code_vecs[g].resize(n_frames);
        for (int f = 0; f < n_frames; f++) {
            code_vecs[g][f] = (int)codes[f * n_groups + g];
        }
    }

    std::vector<float> audio;
    if (!ctx->decoder.decode(code_vecs, audio)) {
        return -1;
    }

    if (audio_out && !audio.empty()) {
        memcpy(audio_out, audio.data(), audio.size() * sizeof(float));
    }
    if (n_samples_out) {
        *n_samples_out = (int)audio.size();
    }
    return 0;
}

/* ========================================================================== */
/* Speaker encoder                                                            */
/* ========================================================================== */

int qwen_tts_extract_speaker(
    qwen_tts_ctx_t* ctx,
    const float* audio,
    int n_samples,
    int sample_rate,
    float* embedding_out
) {
    if (!ctx || !ctx->has_speaker_encoder) return -1;

    std::vector<float> audio_vec(audio, audio + n_samples);
    std::vector<float> embedding;
    if (!ctx->speaker_encoder.extract(audio_vec, sample_rate, embedding)) {
        return -1;
    }

    if (embedding_out && !embedding.empty()) {
        memcpy(embedding_out, embedding.data(), embedding.size() * sizeof(float));
    }
    return 0;
}

/* ========================================================================== */
/* High-level one-shot synthesis (B5)                                         */
/* ========================================================================== */

// Fill a QwenTTSParams struct the way QwenTTS::generate* expects.
// `defaults` semantics mirror the struct doc in qwen_tts_api.h:
//   seed                -> unused by the C++ path (RNG is set inside
//                          TalkerSamplingParams/engine); we keep it in the
//                          struct for future wiring but don't plumb here.
//   max_tokens == 0     -> 2048
//   temperature == 0    -> 0.9f
//   top_k == 0          -> 50 ; top_k == -1 -> disabled (INT_MAX sentinel)
//   top_p == 0          -> 1.0f
//   repetition_penalty == 0 -> 1.05f
//   cp_groups == 0      -> all 15
//   cp_layers == 0      -> all 5
//   greedy != 0         -> do_sample=false (argmax)
static void translate_synth_params(
    const qwen_tts_ctx* ctx,
    const qwen_tts_synth_params_t* in,
    QwenTTSParams& out)
{
    out.model_dir     = ctx->load_model_dir;
    out.tokenizer_dir = ctx->load_tokenizer_dir;
    out.talker_model  = ctx->load_talker_override;
    out.cp_model      = ctx->load_cp_override;
    out.n_threads     = ctx->load_n_threads;
    out.n_gpu_layers  = ctx->load_n_gpu_layers;
    // native CANN paths ON when GPU layers > 0 (matches QwenTTS::load default)
    out.cp_cann       = true;
    out.native_talker = true;

    out.text        = in->text        ? in->text        : "";
    out.ref_audio   = in->ref_audio_path ? in->ref_audio_path : "";
    out.ref_text    = in->ref_text    ? in->ref_text    : "";
    out.ref_lang    = in->ref_lang    ? in->ref_lang    : "English";
    out.target_lang = in->target_lang ? in->target_lang : "English";
    out.mode        = in->mode        ? in->mode        : "icl";
    out.speaker     = in->speaker     ? in->speaker     : "";

    out.max_new_tokens = in->max_tokens > 0 ? in->max_tokens : 2048;

    TalkerSamplingParams s; // defaults match Python reference (0.9 / 50 / 1.0 / 1.05)
    if (in->temperature != 0.0f)        s.temperature        = in->temperature;
    if (in->top_p != 0.0f)              s.top_p              = in->top_p;
    if (in->repetition_penalty != 0.0f) s.repetition_penalty = in->repetition_penalty;
    if (in->top_k == -1)                s.top_k              = 0; // disabled in talker.cpp's sampler
    else if (in->top_k > 0)             s.top_k              = in->top_k;
    // else: keep default 50

    if (in->greedy != 0) {
        s.do_sample    = false;
        s.cp_do_sample = false;
    }

    // Mirror CP sampling hyperparams onto the CP branch (Python reference
    // uses the same temperature / top_k / top_p for both sampler passes).
    s.cp_temperature = s.temperature;
    s.cp_top_k       = s.top_k;
    s.cp_top_p       = s.top_p;

    if (in->cp_groups > 0) s.cp_max_groups = in->cp_groups;
    if (in->cp_layers > 0) s.cp_max_layers = in->cp_layers;

    out.sampling = s;
}

int qwen_tts_synthesize(
    qwen_tts_ctx_t* ctx,
    const qwen_tts_synth_params_t* in,
    float** pcm_out,
    int* n_samples_out
) {
    // Zero outputs up front so the error contract holds for every early return.
    if (pcm_out)       *pcm_out = nullptr;
    if (n_samples_out) *n_samples_out = 0;

    if (!ctx || !in || !pcm_out || !n_samples_out) return -1;
    if (!in->text || !in->mode) return -1;

    std::lock_guard<std::mutex> guard(ctx->synth_mu);

    // Lazy-init the QwenTTS wrapper on first call. Reuses the same model dir,
    // tokenizer dir, and GGUF overrides passed to qwen_tts_load(). This
    // duplicates the Talker/decoder load vs. the primitive ctx above — a
    // deliberate tradeoff to keep the primitive ABI bit-for-bit unchanged
    // (see contract §5 B5). Callers using synthesize() only can ignore the
    // primitive components; callers using primitives only never pay this cost.
    if (!ctx->synth) {
        auto synth = std::make_unique<QwenTTS>();
        QwenTTSParams load_params;
        load_params.model_dir     = ctx->load_model_dir;
        load_params.tokenizer_dir = ctx->load_tokenizer_dir;
        load_params.talker_model  = ctx->load_talker_override;
        load_params.cp_model      = ctx->load_cp_override;
        load_params.n_threads     = ctx->load_n_threads;
        load_params.n_gpu_layers  = ctx->load_n_gpu_layers;
        load_params.cp_cann       = true;
        load_params.native_talker = true;
        if (!synth->load(load_params)) {
            fprintf(stderr, "[qwen_tts_api] QwenTTS::load failed\n");
            return -3;
        }
        ctx->synth = std::move(synth);
    }

    QwenTTSParams gen_params;
    translate_synth_params(ctx, in, gen_params);

    std::string mode = in->mode;
    std::vector<float> audio;
    bool ok = false;
    if (mode == "icl") {
        ok = ctx->synth->generate(gen_params, audio);
    } else if (mode == "xvec") {
        ok = ctx->synth->generate_xvec(gen_params, audio);
    } else if (mode == "customvoice") {
        ok = ctx->synth->generate_customvoice(gen_params, audio);
    } else {
        fprintf(stderr, "[qwen_tts_api] unknown mode: %s\n", mode.c_str());
        return -2;
    }

    if (!ok) return -3;

    const size_t n = audio.size();
    if (n == 0) {
        // Engine returned success but no samples — treat as generation failure
        // so the caller doesn't have to special-case a NULL buffer with n==0.
        return -3;
    }

    float* buf = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!buf) return -3;
    std::memcpy(buf, audio.data(), n * sizeof(float));

    *pcm_out       = buf;
    *n_samples_out = static_cast<int>(n);
    return 0;
}

void qwen_tts_pcm_free(float* pcm) {
    // Pair for the malloc() inside qwen_tts_synthesize(). Safe on NULL.
    std::free(pcm);
}

} // extern "C"
