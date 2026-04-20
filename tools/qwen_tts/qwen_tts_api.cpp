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
#include <cstring>
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

} // extern "C"
