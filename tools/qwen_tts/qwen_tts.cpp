#include "qwen_tts.h"
#include "audio_io.h"
#include <chrono>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <cmath>
#include <thread>
#include <cctype>

namespace {
// Split target_text on sentence boundaries so each sentence can run
// through the talker independently. Necessary because Qwen3-TTS was
// trained on relatively short utterances; given a long input it
// compresses speech to roughly half the natural rate (162 ms/token vs
// 305 ms/token measured on a 100-word English paragraph), causing
// slurred / garbled speech in the second half. Per-sentence runs each
// get fresh talker state and stay inside the model's training
// distribution.
//
// Boundaries:
//   English: `. ! ?` followed by whitespace and an uppercase letter (or
//     non-ASCII / EOT). Avoids splitting "Mr. Smith", "1.5", "U.S.A.".
//   Chinese: `。！？` (3-byte UTF-8 forms).
//   Always: em-dash (—, U+2014) and en-dash (–, U+2013) — these mark
//     parenthetical clauses and the talker handles cleaner sentence-
//     shaped fragments much better than parenthetical structure.
//   Always: `;` (semicolon).
//   `:` followed by whitespace, regardless of next-char case (so
//     "ratio 3:2" / "11:30" don't split because they have no space, but
//     "snapshot: unpretentious" does).
//
// For clause-level splits (em-dash, semicolon, colon) the dash / punct
// is dropped and a `.` is appended to the preceding fragment via
// ensure_terminal_punct() so the model gets a clear EOS signal.
//
// Always returns at least one element.
static bool is_zh_sentence_end(const std::string &t, size_t i) {
    if (i + 2 >= t.size()) return false;
    unsigned char a = (unsigned char) t[i];
    unsigned char b = (unsigned char) t[i + 1];
    unsigned char c = (unsigned char) t[i + 2];
    if (a == 0xE3 && b == 0x80 && c == 0x82) return true;            // 。
    if (a == 0xEF && b == 0xBC && (c == 0x81 || c == 0x9F)) return true;  // ！？
    return false;
}

// 3-byte UTF-8 dash characters that act as clause boundaries.
//   — em-dash  U+2014 → E2 80 94
//   – en-dash  U+2013 → E2 80 93
static bool is_dash_clause_break(const std::string &t, size_t i) {
    if (i + 2 >= t.size()) return false;
    unsigned char a = (unsigned char) t[i];
    unsigned char b = (unsigned char) t[i + 1];
    unsigned char c = (unsigned char) t[i + 2];
    return a == 0xE2 && b == 0x80 && (c == 0x94 || c == 0x93);
}

// Append `.` to fragments that don't already end with terminal
// punctuation, so the talker has a clear EOS signal. Empirically a
// fragment like "She wears a modern, cute dress in bright, soft colors"
// (em-dash split, no period) over-generates because the model has been
// trained on sentence-shaped inputs that end in `. ! ?`.
static void ensure_terminal_punct(std::string &s) {
    if (s.empty()) return;
    char back = s.back();
    if (back == '.' || back == '!' || back == '?') return;
    if (s.size() >= 3) {
        unsigned char a = (unsigned char) s[s.size() - 3];
        unsigned char b = (unsigned char) s[s.size() - 2];
        unsigned char c = (unsigned char) s[s.size() - 1];
        if (a == 0xE3 && b == 0x80 && c == 0x82) return;  // 。
        if (a == 0xEF && b == 0xBC && (c == 0x81 || c == 0x9F)) return;  // ！？
    }
    if (back == ',') s.pop_back();  // drop dangling comma before adding period
    s.push_back('.');
}

static std::vector<std::string> split_sentences(const std::string &text) {
    std::vector<std::string> out;
    auto trim = [](std::string s) {
        while (!s.empty() && std::isspace((unsigned char) s.front())) s.erase(s.begin());
        while (!s.empty() && std::isspace((unsigned char) s.back()))  s.pop_back();
        return s;
    };
    auto push_natural = [&](std::string s) {
        s = trim(std::move(s));
        if (!s.empty()) out.push_back(std::move(s));
    };
    auto push_fragment = [&](std::string s) {
        s = trim(std::move(s));
        if (s.empty()) return;
        ensure_terminal_punct(s);
        out.push_back(std::move(s));
    };

    size_t start = 0;
    size_t i = 0;
    while (i < text.size()) {
        char c = text[i];
        bool en_end     = (c == '.' || c == '!' || c == '?');
        bool semicolon  = (c == ';');
        bool colon      = (c == ':');
        bool zh_end     = is_zh_sentence_end(text, i);
        bool dash_break = is_dash_clause_break(text, i);

        if (en_end) {
            size_t j = i + 1;
            if (j == text.size()) {
                push_natural(text.substr(start, j - start));
                start = j;
                i = j;
                continue;
            }
            if (!std::isspace((unsigned char) text[j])) {
                i = j;
                continue;
            }
            size_t k = j;
            while (k < text.size() && std::isspace((unsigned char) text[k])) k++;
            if (k == text.size()) {
                push_natural(text.substr(start, j - start));
                start = j;
                i = k;
                continue;
            }
            unsigned char nc = (unsigned char) text[k];
            if ((nc >= 'A' && nc <= 'Z') || nc >= 0x80) {
                push_natural(text.substr(start, j - start));
                start = j;
                i = k;
                continue;
            }
            i = j;
            continue;
        }
        if (colon) {
            size_t j = i + 1;
            if (j == text.size() || std::isspace((unsigned char) text[j])) {
                push_fragment(text.substr(start, i - start));
                start = j;
                i = j;
                continue;
            }
            i = j;
            continue;
        }
        if (semicolon) {
            push_fragment(text.substr(start, i - start));
            start = i + 1;
            i = i + 1;
            continue;
        }
        if (dash_break) {
            push_fragment(text.substr(start, i - start));
            start = i + 3;
            i = i + 3;
            continue;
        }
        if (zh_end) {
            size_t j = i + 3;
            push_natural(text.substr(start, j - start));
            start = j;
            i = j;
            continue;
        }
        i++;
    }
    if (start < text.size()) {
        push_natural(text.substr(start));
    }
    if (out.empty()) out.push_back(text);
    return out;
}
}  // namespace

// ============================================================================
// Load all model components
// ============================================================================

bool QwenTTS::load(const QwenTTSParams& params) {
    params_ = params;
    auto t0 = std::chrono::high_resolution_clock::now();

    std::string model_dir = params.model_dir;
    // Ensure trailing slash
    if (!model_dir.empty() && model_dir.back() != '/') model_dir += '/';

    std::string tokenizer_dir = params.tokenizer_dir;
    if (tokenizer_dir.empty()) tokenizer_dir = model_dir;
    if (!tokenizer_dir.empty() && tokenizer_dir.back() != '/') tokenizer_dir += '/';

    // NPU policy: only Talker LLM + Code Predictor benefit from CANN.
    // Speaker Encoder / Tokenizer Encoder: CANN 2-3x slower (kernel launch overhead).
    // Tokenizer Decoder: CANN produces incorrect output (Conv/SnakeBeta ops).
    std::string cpu_device = "CPU";

    printf("=== Loading QwenTTS models ===\n");
    printf("  Model dir: %s\n", model_dir.c_str());
    printf("  Tokenizer dir: %s\n", tokenizer_dir.c_str());
    printf("  NPU offload: Talker+CP (n_gpu_layers=%d)\n", params.n_gpu_layers);

    // 1. BPE Tokenizer
    printf("\n[1/5] Loading text tokenizer...\n");
    if (!tokenizer_.load(tokenizer_dir + "vocab.json",
                          tokenizer_dir + "merges.txt")) {
        printf("FAIL: cannot load BPE tokenizer\n");
        return false;
    }

    // 2. Speaker Encoder (CPU: CANN 2.7x slower due to kernel launch overhead)
    printf("\n[2/5] Loading speaker encoder...\n");
    ContextParams spk_params;
    spk_params.device_name = cpu_device;
    spk_params.n_threads = params.n_threads;
    if (!speaker_encoder_.load(model_dir + "qwen_tts_speaker_encoder.gguf",
                                spk_params)) {
        printf("FAIL: cannot load speaker encoder\n");
        return false;
    }

    // 3. Speech Tokenizer Encoder (testing CANN)
    printf("\n[3/5] Loading speech tokenizer encoder...\n");
    ContextParams enc_params;
    enc_params.device_name = (params.n_gpu_layers > 0) ? "CANN0" : cpu_device;
    enc_params.n_threads = params.n_threads;
    enc_params.max_nodes = 8192;
    if (!tokenizer_encoder_.load(model_dir + "qwen_tts_tokenizer_enc.gguf",
                                  enc_params)) {
        printf("FAIL: cannot load tokenizer encoder\n");
        return false;
    }

    // 4. Talker LLM (3 GGUF files)
    printf("\n[4/5] Loading Talker LLM...\n");
    std::string talker_gguf = params.talker_model.empty()
        ? model_dir + "qwen_tts_talker_llama.gguf"
        : params.talker_model;
    if (!talker_.load_model(talker_gguf,
                             model_dir + "qwen_tts_talker.gguf",
                             model_dir + "qwen_tts_code_predictor.gguf",
                             params.n_threads,
                             params.n_gpu_layers,
                             params.cp_model)) {
        printf("FAIL: cannot load Talker LLM\n");
        return false;
    }

    // 5. Speech Tokenizer Decoder
    // CANN 27x faster (0.45s vs 11s) but fails for >99 total frames.
    // Use CANN with CPU fallback for long sequences.
    printf("\n[5/5] Loading speech tokenizer decoder...\n");
    if (params.n_gpu_layers > 0) {
        ContextParams dec_npu;
        dec_npu.device_name = "CANN0";
        dec_npu.n_threads = params.n_threads;
        dec_npu.max_nodes = 65536;
        ContextParams dec_cpu;
        dec_cpu.device_name = cpu_device;
        dec_cpu.n_threads = params.n_threads;
        dec_cpu.max_nodes = 65536;
        if (!tokenizer_decoder_.load(model_dir + "qwen_tts_tokenizer_dec.gguf",
                                      dec_npu, dec_cpu)) {
            printf("FAIL: cannot load tokenizer decoder\n");
            return false;
        }
    } else {
        ContextParams dec_params;
        dec_params.device_name = cpu_device;
        dec_params.n_threads = params.n_threads;
        dec_params.max_nodes = 65536;
        if (!tokenizer_decoder_.load(model_dir + "qwen_tts_tokenizer_dec.gguf",
                                      dec_params)) {
            printf("FAIL: cannot load tokenizer decoder\n");
            return false;
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("\n=== All models loaded in %.1f seconds ===\n", dt);

    loaded_ = true;
    return true;
}

// ============================================================================
// Tokenize text for TTS — returns ref_text and target_text tokens separately
// ============================================================================

void QwenTTS::tokenize_tts_text(const std::string &ref_text,
                                 const std::string &target_text,
                                 std::vector<int> &ref_text_tokens,
                                 std::vector<int> &target_text_tokens) const {
    // Python reference tokenizes two separate strings:
    //   ref_ids = tokenize("<|im_start|>assistant\n{ref_text}<|im_end|>\n")
    //   input_ids = tokenize("<|im_start|>assistant\n{target_text}<|im_end|>\n<|im_start|>assistant\n")
    //
    // Then extracts pure content tokens:
    //   ref_text_tokens = ref_ids[3:-2]    (strip role prefix + im_end + \n)
    //   target_text_tokens = input_ids[3:-5] (strip role prefix + im_end\n + im_start assistant\n)

    auto ref_ids = tokenizer_.encode(
        "<|im_start|>assistant\n" + ref_text + "<|im_end|>\n");
    auto target_ids = tokenizer_.encode(
        "<|im_start|>assistant\n" + target_text + "<|im_end|>\n<|im_start|>assistant\n");

    // Extract content tokens (strip role prefix and suffix special tokens)
    if (ref_ids.size() > 5) {
        ref_text_tokens.assign(ref_ids.begin() + 3, ref_ids.end() - 2);
    } else {
        ref_text_tokens.clear();
    }
    if (target_ids.size() > 8) {
        target_text_tokens.assign(target_ids.begin() + 3, target_ids.end() - 5);
    } else {
        target_text_tokens.clear();
    }

    printf("[tokenize] ref_text: %zu tokens, target_text: %zu tokens\n",
           ref_text_tokens.size(), target_text_tokens.size());
}

// ============================================================================
// End-to-end voice clone generation
// ============================================================================

bool QwenTTS::generate(const QwenTTSParams& params, std::vector<float>& audio_out) {
    if (!loaded_) {
        printf("[qwen_tts] models not loaded\n");
        return false;
    }

    bool xvec_mode = !params.xvec.empty();

    printf("\n=== %s Generation ===\n", xvec_mode ? "X-Vector" : "Voice Clone");
    if (xvec_mode) {
        printf("  X-vector: %s\n", params.xvec.c_str());
    } else {
        printf("  Ref audio: %s\n", params.ref_audio.c_str());
        printf("  Ref text: %s\n", params.ref_text.c_str());
    }
    printf("  Target text: %s\n", params.text.c_str());
    printf("  Language: %s\n", params.target_lang.c_str());

    auto total_t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> spk_embedding;
    std::vector<std::vector<int>> ref_codes;
    std::vector<float> encoder_hidden;
    bool used_cache = false;
    cached_ref_text_.clear();

    // ----- xvec mode: load standalone speaker embedding -----
    // Format: magic "QXVC" (4B) | version u32 | spk_dim u32 | float32[spk_dim]
    if (xvec_mode) {
        FILE *fx = fopen(params.xvec.c_str(), "rb");
        if (!fx) {
            printf("FAIL: cannot open xvec file: %s\n", params.xvec.c_str());
            return false;
        }
        char magic[4] = {0};
        uint32_t version = 0, spk_dim = 0;
        if (fread(magic, 1, 4, fx) != 4 ||
            fread(&version, 4, 1, fx) != 1 ||
            fread(&spk_dim, 4, 1, fx) != 1) {
            printf("FAIL: short read on xvec header\n");
            fclose(fx); return false;
        }
        if (memcmp(magic, "QXVC", 4) != 0) {
            printf("FAIL: bad xvec magic (got %.4s, expected QXVC)\n", magic);
            fclose(fx); return false;
        }
        if (version != 1) {
            printf("FAIL: unsupported xvec version %u (expected 1)\n", version);
            fclose(fx); return false;
        }
        if (spk_dim == 0 || spk_dim > 65536) {
            printf("FAIL: invalid xvec spk_dim %u\n", spk_dim);
            fclose(fx); return false;
        }
        spk_embedding.resize(spk_dim);
        if (fread(spk_embedding.data(), sizeof(float), spk_dim, fx) != spk_dim) {
            printf("FAIL: short read on xvec data\n");
            fclose(fx); return false;
        }
        fclose(fx);
        printf("\n--- Loaded xvec: %s (spk_dim=%u) ---\n",
               params.xvec.c_str(), spk_dim);
    }

    // Try loading from ref_cache file
    if (!xvec_mode && !params.ref_cache.empty()) {
        FILE *fc = fopen(params.ref_cache.c_str(), "rb");
        if (fc) {
            // Load cached ref_codes + spk_embedding
            int nq, nf, spk_dim;
            if (fread(&nq, 4, 1, fc) == 1 && fread(&nf, 4, 1, fc) == 1) {
                ref_codes.resize(nq);
                for (int q = 0; q < nq; q++) {
                    ref_codes[q].resize(nf);
                    fread(ref_codes[q].data(), sizeof(int), nf, fc);
                }
                if (fread(&spk_dim, 4, 1, fc) == 1) {
                    spk_embedding.resize(spk_dim);
                    fread(spk_embedding.data(), sizeof(float), spk_dim, fc);
                }
                // Read cached ref_text (if present)
                int ref_text_len = 0;
                if (fread(&ref_text_len, 4, 1, fc) == 1 && ref_text_len > 0) {
                    std::string cached_ref_text(ref_text_len, '\0');
                    fread(&cached_ref_text[0], 1, ref_text_len, fc);
                    cached_ref_text_ = cached_ref_text;
                    printf("\n--- Loaded ref cache: %s (%dx%d codes, spk=%d, ref_text=%d chars) ---\n",
                           params.ref_cache.c_str(), nq, nf, spk_dim, ref_text_len);
                } else {
                    printf("\n--- Loaded ref cache: %s (%dx%d codes, spk=%d, no ref_text) ---\n",
                           params.ref_cache.c_str(), nq, nf, spk_dim);
                }
                used_cache = true;
            }
            fclose(fc);
        }
    }

    double spk_time = 0, enc_time = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    auto t1 = t0;

    if (!used_cache && !xvec_mode) {
        // Need ref_audio + ref_text for encoding
        if (params.ref_audio.empty() || params.ref_text.empty()) {
            printf("FAIL: --ref_audio and --ref_text required (no ref_cache found)\n");
            return false;
        }

    // Step 1: Load reference audio
    printf("\n--- Step 1: Load reference audio ---\n");
    std::vector<float> ref_audio;
    if (!audio_io::load_audio(params.ref_audio, 24000, ref_audio)) {
        printf("FAIL: cannot load reference audio: %s\n", params.ref_audio.c_str());
        return false;
    }
    printf("  Loaded %zu samples (%.2f sec at 24kHz)\n",
           ref_audio.size(), ref_audio.size() / 24000.0f);

    if (params.profiling) {
        FILE *f = fopen("logs/cpp_ref_audio_input.bin", "wb");
        if (f) {
            int n = (int)ref_audio.size();
            fwrite(&n, 4, 1, f);
            fwrite(ref_audio.data(), sizeof(float), n, f);
            fclose(f);
            printf("  [debug] dumped ref_audio input (%d samples)\n", n);
            printf("  [debug] first 10: ");
            for (int i = 0; i < 10 && i < n; i++) printf("%.6f ", ref_audio[i]);
            printf("\n");
        }
    }

    // Step 2+3: Speaker embedding + Audio encoding (parallel)
    printf("\n--- Step 2+3: Speaker + Encoder (parallel) ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    bool spk_ok = false, enc_ok = false;

    // Run speaker encoder in a separate thread
    std::thread spk_thread([&]() {
        spk_ok = speaker_encoder_.extract(ref_audio, 24000, spk_embedding);
    });

    // Run audio encoder in main thread
    enc_ok = tokenizer_encoder_.encode(ref_audio, ref_codes, &encoder_hidden);

    spk_thread.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(t1 - t0).count();

    if (!spk_ok) { printf("FAIL: speaker embedding extraction failed\n"); return false; }
    if (!enc_ok) { printf("FAIL: audio encoding failed\n"); return false; }

    int n_ref_frames = ref_codes.empty() ? 0 : (int)ref_codes[0].size();
    double spk_time = parallel_time;  // for timing report compatibility
    double enc_time = parallel_time;
    printf("  Speaker embedding: %zu dims\n", spk_embedding.size());
    printf("  Ref codes: %d quantizers x %d frames\n",
           (int)ref_codes.size(), n_ref_frames);
    printf("  Parallel time: %.2f sec\n", parallel_time);

    // Debug: dump ref_codes for round-trip testing
    if (params.profiling) {
        FILE *f = fopen("logs/cpp_ref_codes.bin", "wb");
        if (f) {
            int nq = (int)ref_codes.size(), nf = n_ref_frames;
            fwrite(&nq, 4, 1, f);
            fwrite(&nf, 4, 1, f);
            for (int q = 0; q < nq; q++)
                for (int t = 0; t < nf; t++) {
                    int v = ref_codes[q][t];
                    fwrite(&v, 4, 1, f);
                }
            fclose(f);
            printf("  [debug] dumped ref_codes to logs/cpp_ref_codes.bin\n");
        }
    }  // profiling

    if (params.profiling && !encoder_hidden.empty()) {
        FILE *f = fopen("logs/cpp_encoder_hidden.bin", "wb");
        if (f) {
            int n = (int)encoder_hidden.size();
            fwrite(&n, 4, 1, f);
            fwrite(encoder_hidden.data(), sizeof(float), n, f);
            fclose(f);
            printf("  [debug] dumped encoder_hidden (%d floats, hidden=%d, T=%d)\n",
                   n, 512, n / 512);
        }
    }

    // Save ref cache if requested
    if (!used_cache && !params.ref_cache.empty()) {
        FILE *fc = fopen(params.ref_cache.c_str(), "wb");
        if (fc) {
            int nq = (int)ref_codes.size();
            int nf = ref_codes.empty() ? 0 : (int)ref_codes[0].size();
            int spk_dim = (int)spk_embedding.size();
            fwrite(&nq, 4, 1, fc);
            fwrite(&nf, 4, 1, fc);
            for (int q = 0; q < nq; q++)
                fwrite(ref_codes[q].data(), sizeof(int), nf, fc);
            fwrite(&spk_dim, 4, 1, fc);
            fwrite(spk_embedding.data(), sizeof(float), spk_dim, fc);
            // Save ref_text
            int ref_text_len = (int)params.ref_text.size();
            fwrite(&ref_text_len, 4, 1, fc);
            fwrite(params.ref_text.data(), 1, ref_text_len, fc);
            fclose(fc);
            printf("  Saved ref cache: %s (ref_text=%d chars)\n",
                   params.ref_cache.c_str(), ref_text_len);
        }
    }

    }  // end if (!used_cache)

    // Step 4: split target_text on sentence boundaries when chunking is
    // enabled. Qwen3-TTS was trained on relatively short utterances and
    // compresses long inputs to ~half the natural speech rate (162 ms/
    // token vs 305 ms/token measured), causing slurring / garbling in
    // the second half of long paragraphs. Looping the talker per
    // sentence keeps each call inside the model's training distribution.
    std::string effective_ref_text = xvec_mode
        ? std::string()
        : (cached_ref_text_.empty() ? params.ref_text : cached_ref_text_);

    std::vector<std::string> sentences;
    if (params.auto_sentence_chunk) {
        sentences = split_sentences(params.text);
    } else {
        sentences = {params.text};
    }
    printf("\n--- Step 4: Tokenize text (%zu sentence%s) ---\n",
           sentences.size(), sentences.size() == 1 ? "" : "s");
    if (sentences.size() > 1) {
        for (size_t i = 0; i < sentences.size(); i++) {
            printf("  [%zu] %s\n", i + 1, sentences[i].c_str());
        }
    }

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double>(prefill_end - total_t0).count();

    // Convert language string to lowercase for matching
    std::string lang = params.target_lang;
    for (auto &c : lang) c = tolower(c);

    // Streaming state shared across all sentences. codec_tokens is reused
    // (cleared) per sentence so the per-frame callback always observes
    // the current sentence's accumulated frames; decoded_until resets to
    // 0 at the top of each sentence.
    bool streaming = (params.stream_chunk_frames > 0 && params.stream_callback);
    int  chunk_frames  = streaming ? params.stream_chunk_frames : 0;
    constexpr int kStreamWarmup = 72;  // matches OVERLAP_FRAMES inside the decoder
    int  upsample_per_frame = 1920;    // 24kHz * 0.08s; matches DecoderConfig.decode_upsample_rate
    std::vector<std::vector<int>> codec_tokens;
    int decoded_until = 0;
    double stream_decode_time = 0;
    int    stream_chunks_emitted = 0;
    bool   stream_failed = false;
    audio_out.clear();

    // Per-chunk decode with a left-context warmup window so the codec
    // decoder's pre-transformer sliding-window attention (window=72) and
    // causal conv stacks see realistic context across chunk boundaries
    // within a sentence. Cross-sentence boundaries decode in isolation
    // (negligible click measured: max boundary jump ~1.6% of full range).
    auto emit_chunk = [&](int from, int to, bool is_final) {
        if (to <= from) {
            if (is_final && params.stream_callback) {
                params.stream_callback(nullptr, 0, true);
            }
            return true;
        }
        int warmup_start = std::max(0, from - kStreamWarmup);
        std::vector<std::vector<int>> chunk_codes(codec_tokens.size());
        for (size_t q = 0; q < codec_tokens.size(); q++) {
            chunk_codes[q].assign(codec_tokens[q].begin() + warmup_start,
                                   codec_tokens[q].begin() + to);
        }
        std::vector<float> chunk_audio;
        auto dec_t0 = std::chrono::high_resolution_clock::now();
        if (!tokenizer_decoder_.decode(chunk_codes, chunk_audio)) {
            printf("FAIL: streaming decode failed at frames [%d,%d)\n", from, to);
            return false;
        }
        auto dec_t1 = std::chrono::high_resolution_clock::now();
        stream_decode_time += std::chrono::duration<double>(dec_t1 - dec_t0).count();
        stream_chunks_emitted++;
        int warmup_samples = (from - warmup_start) * upsample_per_frame;
        if (warmup_samples > (int) chunk_audio.size()) {
            warmup_samples = (int) chunk_audio.size();
        }
        const float *kept_begin = chunk_audio.data() + warmup_samples;
        size_t       kept_count = chunk_audio.size() - warmup_samples;
        audio_out.insert(audio_out.end(), kept_begin, kept_begin + kept_count);
        if (params.stream_callback) {
            params.stream_callback(kept_begin, kept_count, is_final);
        }
        return true;
    };

    TalkerLLM::FrameCallback frame_cb = nullptr;
    if (streaming) {
        frame_cb = [&](int frame_idx,
                       const std::vector<std::vector<int>> &cur_tokens) {
            (void) cur_tokens;
            if (stream_failed) return;
            int n_now = frame_idx + 1;
            while (n_now - decoded_until >= chunk_frames) {
                int next = decoded_until + chunk_frames;
                if (!emit_chunk(decoded_until, next, /*is_final=*/false)) {
                    stream_failed = true;
                    return;
                }
                decoded_until = next;
            }
        };
    }

    // Per-sentence aggregates for the final timing report.
    double generate_time = 0;
    double decode_time   = 0;
    int    n_gen_frames  = 0;

    // === Sentence loop ============================================
    for (size_t s_idx = 0; s_idx < sentences.size(); s_idx++) {
        const std::string &sentence = sentences[s_idx];
        bool is_last_sentence = (s_idx + 1 == sentences.size());

        if (sentences.size() > 1) {
            printf("\n=== Sentence %zu/%zu ===\n", s_idx + 1, sentences.size());
        }

        // Tokenize this sentence (ref_text shared across all sentences).
        std::vector<int> ref_text_tokens, target_text_tokens;
        tokenize_tts_text(effective_ref_text, sentence,
                          ref_text_tokens, target_text_tokens);

        if (params.profiling && s_idx == 0) {
            FILE *f = fopen("logs/cpp_ref_text_tokens.bin", "wb");
            if (f) {
                int n = (int)ref_text_tokens.size();
                fwrite(&n, 4, 1, f);
                fwrite(ref_text_tokens.data(), sizeof(int), n, f);
                fclose(f);
            }
            f = fopen("logs/cpp_target_text_tokens.bin", "wb");
            if (f) {
                int n = (int)target_text_tokens.size();
                fwrite(&n, 4, 1, f);
                fwrite(target_text_tokens.data(), sizeof(int), n, f);
                fclose(f);
            }
            printf("  [debug] dumped text tokens (ref=%zu, tgt=%zu)\n",
                   ref_text_tokens.size(), target_text_tokens.size());
        }

        // Auto max_tokens cap (per sentence). Heuristic upper bound on
        // the codec frames the talker may emit, derived from the
        // tokenized target text length: ~8 frames per English BPE token,
        // ~6 per Chinese, 30-frame floor. Generic safety net that stops
        // a failing-to-EOS talker from running to 2048 frames. Disabled
        // when the user passes --max_tokens explicitly.
        int effective_max_tokens = params.max_new_tokens;
        if (params.auto_max_tokens && !target_text_tokens.empty()) {
            std::string lang_lc = params.target_lang;
            for (auto &c : lang_lc) c = (char) tolower((unsigned char) c);
            int per_token = (lang_lc == "chinese") ? 6 : 8;
            int auto_cap  = std::max(30, (int) target_text_tokens.size() * per_token);
            if (auto_cap < effective_max_tokens) {
                printf("  [auto-cap] %zu target tokens (lang=%s) → max_new_tokens=%d "
                       "(~%.1fs of audio); pass --max_tokens to override\n",
                       target_text_tokens.size(), params.target_lang.c_str(),
                       auto_cap, auto_cap * 0.08f);
                effective_max_tokens = auto_cap;
            }
        }

        // Step 5: talker generation for this sentence.
        printf("\n--- Step 5: Generate codec tokens ---\n");
        if (streaming && s_idx == 0) {
            printf("  [streaming] chunk_frames=%d\n", params.stream_chunk_frames);
        }
        codec_tokens.clear();
        decoded_until = 0;

        auto t_gen0 = std::chrono::high_resolution_clock::now();
        if (!talker_.generate(ref_text_tokens, target_text_tokens,
                               spk_embedding, ref_codes, lang,
                               codec_tokens, effective_max_tokens,
                               params.sampling, xvec_mode, frame_cb)) {
            printf("FAIL: codec generation failed\n");
            return false;
        }
        if (stream_failed) {
            return false;
        }
        auto t_gen1 = std::chrono::high_resolution_clock::now();
        generate_time += std::chrono::duration<double>(t_gen1 - t_gen0).count();

        int sent_frames = codec_tokens.empty() ? 0 : (int)codec_tokens[0].size();
        n_gen_frames += sent_frames;

        if (params.profiling) {
            FILE *f = fopen("logs/cpp_codec_tokens.bin",
                            s_idx == 0 ? "wb" : "ab");
            if (f) {
                int nq = (int)codec_tokens.size();
                int nf = sent_frames;
                fwrite(&nq, 4, 1, f);
                fwrite(&nf, 4, 1, f);
                for (int q = 0; q < nq; q++)
                    fwrite(codec_tokens[q].data(), sizeof(int), nf, f);
                fclose(f);
                printf("  [debug] dumped codec_tokens (%dx%d)\n", nq, nf);
            }
        }

        printf("  Generated %d codec frames (%.2f sec, %.1f frames/sec)\n",
               sent_frames,
               std::chrono::duration<double>(t_gen1 - t_gen0).count(),
               sent_frames / std::max(
                   std::chrono::duration<double>(t_gen1 - t_gen0).count(), 1e-6));

        // Step 6: decode this sentence's frames into audio.
        printf("\n--- Step 6: Decode to audio ---\n");
        auto t_dec0 = std::chrono::high_resolution_clock::now();
        if (streaming) {
            // Flush this sentence's leftover frames. is_final fires only
            // on the last partial chunk of the last sentence.
            if (decoded_until < sent_frames) {
                bool fire_final = is_last_sentence;
                if (!emit_chunk(decoded_until, sent_frames, fire_final)) {
                    return false;
                }
                decoded_until = sent_frames;
            } else if (is_last_sentence && params.stream_callback) {
                params.stream_callback(nullptr, 0, true);
            }
            // decode_time is updated inside emit_chunk (stream_decode_time).
        } else {
            std::vector<float> sentence_audio;
            if (!tokenizer_decoder_.decode(codec_tokens, sentence_audio)) {
                printf("FAIL: audio decoding failed\n");
                return false;
            }
            auto t_dec1 = std::chrono::high_resolution_clock::now();
            decode_time += std::chrono::duration<double>(t_dec1 - t_dec0).count();
            printf("  Decoded %zu samples (%.2f sec)\n",
                   sentence_audio.size(),
                   std::chrono::duration<double>(t_dec1 - t_dec0).count());
            audio_out.insert(audio_out.end(),
                             sentence_audio.begin(), sentence_audio.end());
        }
    }

    if (streaming) {
        decode_time = stream_decode_time;
        printf("\n  Streaming totals: %zu samples in %d chunk(s) "
               "(%.2f sec decode, ratio %.1fx)\n",
               audio_out.size(), stream_chunks_emitted, decode_time,
               audio_out.size() / 24000.0 / std::max(decode_time, 1e-6));
    }

    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_t1 - total_t0).count();
    double target_duration = audio_out.size() / 24000.0;
    printf("\n=== Generation complete ===\n");
    printf("  Output: %zu samples (%.2f sec at 24kHz)\n",
           audio_out.size(), target_duration);
    printf("  Timing breakdown:\n");
    if (xvec_mode) {
        printf("    X-Vector:    loaded (encoder skipped, no ICL)\n");
    } else if (used_cache) {
        printf("    Ref Cache:   loaded (encoder skipped)\n");
    } else {
        printf("    Speaker+Enc: %.2f sec (parallel)\n", spk_time);
    }
    printf("    Prefill tot: %.2f sec\n", prefill_time);
    printf("    Generate:    %.2f sec\n", generate_time);
    printf("    Decode:      %.2f sec\n", decode_time);
    printf("    Total:       %.2f sec\n", total_time);
    printf("  Inference RTF: %.2fx (generate+decode / audio)\n",
           (generate_time + decode_time) / target_duration);
    printf("  Total RTF:     %.2fx (end-to-end / audio)\n",
           total_time / target_duration);

    return true;
}

// ============================================================================
// xvec extraction tool
//
// Loads a wav file, runs the speaker encoder once, and writes the resulting
// 2048-dim speaker embedding to a standalone .xvec file.
//
// File format:
//   offset 0:  magic "QXVC" (4 bytes)
//   offset 4:  version  u32   (= 1)
//   offset 8:  spk_dim  u32
//   offset 12: spk_emb  float32[spk_dim]
// ============================================================================
bool QwenTTS::extract_xvec(const std::string &wav_path,
                            const std::string &out_xvec_path) {
    if (!loaded_) {
        printf("[qwen_tts] models not loaded\n");
        return false;
    }
    printf("\n=== X-Vector Extraction ===\n");
    printf("  Input wav: %s\n", wav_path.c_str());
    printf("  Output xvec: %s\n", out_xvec_path.c_str());

    std::vector<float> ref_audio;
    if (!audio_io::load_audio(wav_path, 24000, ref_audio)) {
        printf("FAIL: cannot load audio: %s\n", wav_path.c_str());
        return false;
    }
    printf("  Loaded %zu samples (%.2f sec at 24kHz)\n",
           ref_audio.size(), ref_audio.size() / 24000.0f);

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> spk_embedding;
    if (!speaker_encoder_.extract(ref_audio, 24000, spk_embedding)) {
        printf("FAIL: speaker embedding extraction failed\n");
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();
    printf("  Extracted %zu-dim speaker embedding in %.2f sec\n",
           spk_embedding.size(), dt);

    FILE *f = fopen(out_xvec_path.c_str(), "wb");
    if (!f) {
        printf("FAIL: cannot open output file for write: %s\n",
               out_xvec_path.c_str());
        return false;
    }
    const char magic[4] = {'Q', 'X', 'V', 'C'};
    uint32_t version = 1;
    uint32_t spk_dim = (uint32_t)spk_embedding.size();
    fwrite(magic, 1, 4, f);
    fwrite(&version, 4, 1, f);
    fwrite(&spk_dim, 4, 1, f);
    fwrite(spk_embedding.data(), sizeof(float), spk_dim, f);
    fclose(f);

    long sz = 12 + (long)spk_dim * 4;
    printf("  Saved xvec: %s (%ld bytes, magic=QXVC v=%u dim=%u)\n",
           out_xvec_path.c_str(), sz, version, spk_dim);
    return true;
}
