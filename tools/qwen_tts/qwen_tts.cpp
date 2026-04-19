#include "qwen_tts.h"
#include "audio_io.h"
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <numeric>
#include <cmath>
#include <queue>
#include <thread>

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
    //    Skip for CustomVoice models which have built-in speaker embeddings
    std::string spk_enc_path = model_dir + "qwen_tts_speaker_encoder.gguf";
    bool has_speaker_encoder = false;
    {
        FILE *f = fopen(spk_enc_path.c_str(), "rb");
        if (f) { fclose(f); has_speaker_encoder = true; }
    }
    if (has_speaker_encoder) {
        printf("\n[2/5] Loading speaker encoder...\n");
        ContextParams spk_params;
        spk_params.device_name = cpu_device;
        spk_params.n_threads = params.n_threads;
        if (!speaker_encoder_.load(spk_enc_path, spk_params)) {
            printf("FAIL: cannot load speaker encoder\n");
            return false;
        }
    } else {
        printf("\n[2/5] Skipping speaker encoder (CustomVoice model)\n");
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
                             params.cp_model,
                             params.cp_cann,
                             params.native_talker)) {
        printf("FAIL: cannot load Talker LLM\n");
        return false;
    }

    // 5. Speech Tokenizer Decoder
    // CANN 27x faster (0.45s vs 11s) but fails for >99 total frames.
    // Use CANN with CPU fallback for long sequences.
    //
    // Track R (2026-04-17): probe CANN0 registration BEFORE handing "CANN0"
    // to the decoder's ContextManager. Without this probe, a missing
    // libggml-cann.so in the executable dir (or a silently-failing aclInit)
    // causes ContextManager to log `create_backend: ERROR: backend CANN0
    // not found` and fall back to CPU — but SpeechTokenizerDecoder::load()
    // still sets split_mode_=true with session_ pointing to a CPU backend,
    // which silently mislabels CPU decode as "CANN" in Step 6 logs and
    // inside forward_chunk()'s prefer_cann=true path (exactly what Track L
    // M6.4 hit). Probing up front lets us pick the honest single-session
    // CPU path when CANN0 is unavailable, so ASR / throughput behavior
    // matches the `n_gpu_layers==0` case deterministically.
    printf("\n[5/5] Loading speech tokenizer decoder...\n");

    auto probe_cann0_available = [&]() -> bool {
        if (params.n_gpu_layers <= 0) return false;
        // Force dynamic backend discovery (idempotent — ContextManager also
        // calls this, but doing it here lets us check presence before we
        // commit to a split-mode load).
        ggml_backend_load_all();
        ggml_backend_dev_t dev = ggml_backend_dev_by_name("CANN0");
        if (dev) return true;
        // CANN0 is missing. Print a single-block diagnostic so the root
        // cause is obvious from the build log (exec dir search path,
        // registered backends, and relevant env vars).
        fprintf(stderr,
                "[qwen_tts] Track R: CANN0 backend NOT registered for decoder "
                "ggml context.\n"
                "  Registered backends (%zu):\n",
                ggml_backend_reg_count());
        for (size_t i = 0; i < ggml_backend_reg_count(); ++i) {
            ggml_backend_reg_t reg = ggml_backend_reg_get(i);
            fprintf(stderr, "    - reg[%zu]: %s\n", i,
                    ggml_backend_reg_name(reg));
        }
        fprintf(stderr, "  Registered devices (%zu):\n",
                ggml_backend_dev_count());
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            fprintf(stderr, "    - dev[%zu]: %s (%s)\n", i,
                    ggml_backend_dev_name(d),
                    ggml_backend_dev_description(d));
        }
        const char *env_keys[] = {"LD_LIBRARY_PATH", "ASCEND_TOOLKIT_HOME",
                                   "ASCEND_OPP_PATH", "ASCEND_HOME_PATH",
                                   "ASCEND_CUSTOM_OPP_PATH",
                                   "ASCEND_RT_VISIBLE_DEVICES",
                                   "GGML_BACKEND_PATH"};
        fprintf(stderr, "  Env (decoder load thread):\n");
        for (const char *k : env_keys) {
            const char *v = std::getenv(k);
            fprintf(stderr, "    %s=%s\n", k, v ? v : "(unset)");
        }
        fprintf(stderr,
                "  Likely cause: libggml-cann.so is not in the executable "
                "dir or the CWD, so ggml_backend_load_all() silently "
                "skipped it (talker / CP still work because they dlopen "
                "$ASCEND_TOOLKIT_HOME/lib64/libascendcl.so directly via "
                "cp_cann_symbols.cpp, bypassing the ggml backend "
                "registry).\n"
                "  Falling back to single-session CPU decoder (honest, "
                "deterministic, bit-identical to n_gpu_layers==0 path).\n");
        return false;
    };

    const bool use_cann_decoder = probe_cann0_available();
    if (use_cann_decoder) {
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

    // Load standalone transformer (correct MRoPE for xvec/customvoice modes)
    std::string talker_llama = params.talker_model.empty()
        ? model_dir + "qwen_tts_talker_llama.gguf" : params.talker_model;
    standalone_tfm_ = tts_transformer_load(talker_llama.c_str(), params.n_threads);
    if (standalone_tfm_) {
        printf("[qwen_tts] standalone transformer loaded (correct MRoPE)\n");
    } else {
        printf("[qwen_tts] WARNING: standalone transformer failed to load, xvec/customvoice may produce distorted audio\n");
    }

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

    printf("\n=== Voice Clone Generation ===\n");
    printf("  Ref audio: %s\n", params.ref_audio.c_str());
    printf("  Ref text: %s\n", params.ref_text.c_str());
    printf("  Target text: %s\n", params.text.c_str());
    printf("  Language: %s\n", params.target_lang.c_str());

    auto total_t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> spk_embedding;
    std::vector<std::vector<int>> ref_codes;
    std::vector<float> encoder_hidden;
    bool used_cache = false;
    cached_ref_text_.clear();

    // Try loading from ref_cache file
    if (!params.ref_cache.empty()) {
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

    if (!used_cache) {
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
    //
    // Track K / M6.3 note: the two encoders live on disjoint resources:
    //   - speaker encoder: CPU (GGML CPU backend — see `load()`; CANN is 2.7× slower here)
    //   - tokenizer encoder: NPU (CANN0 when n_gpu_layers>0)
    //
    // Because they sit on different execution units, they already overlap
    // trivially via std::thread without needing distinct aclrtStreams. A
    // secondary NPU stream buys nothing when only one of the two halves is
    // actually on the NPU. Adding per-encoder timing below to quantify who
    // dominates the "Parallel time" wall-clock and how much overlap is real.
    printf("\n--- Step 2+3: Speaker + Encoder (parallel) ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    bool spk_ok = false, enc_ok = false;
    double spk_only_ms = 0.0, enc_only_ms = 0.0;

    // Run speaker encoder in a separate thread
    std::thread spk_thread([&]() {
        auto s0 = std::chrono::high_resolution_clock::now();
        spk_ok = speaker_encoder_.extract(ref_audio, 24000, spk_embedding);
        auto s1 = std::chrono::high_resolution_clock::now();
        spk_only_ms = std::chrono::duration<double, std::milli>(s1 - s0).count();
    });

    // Run audio encoder in main thread
    auto e0 = std::chrono::high_resolution_clock::now();
    enc_ok = tokenizer_encoder_.encode(ref_audio, ref_codes, &encoder_hidden);
    auto e1 = std::chrono::high_resolution_clock::now();
    enc_only_ms = std::chrono::duration<double, std::milli>(e1 - e0).count();

    spk_thread.join();

    auto t1 = std::chrono::high_resolution_clock::now();
    double parallel_time = std::chrono::duration<double>(t1 - t0).count();
    printf("  [Track K] speaker_encoder (CPU): %.0f ms\n", spk_only_ms);
    printf("  [Track K] tokenizer_encoder (NPU): %.0f ms\n", enc_only_ms);
    printf("  [Track K] overlap efficiency: %.1f%% (ideal=100%% if perfectly parallel)\n",
           100.0 * std::max(spk_only_ms, enc_only_ms) / (parallel_time * 1000.0));

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

    int n_ref_frames = ref_codes.empty() ? 0 : (int)ref_codes[0].size();

    // Step 4: Tokenize text
    printf("\n--- Step 4: Tokenize text ---\n");
    std::vector<int> ref_text_tokens, target_text_tokens;
    tokenize_tts_text(cached_ref_text_.empty() ? params.ref_text : cached_ref_text_,
                       params.text,
                       ref_text_tokens, target_text_tokens);

    auto prefill_end = std::chrono::high_resolution_clock::now();
    double prefill_time = std::chrono::duration<double>(prefill_end - total_t0).count();

    if (params.profiling) {
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

    // Step 5: Generate codec tokens with Talker LLM
    printf("\n--- Step 5: Generate codec tokens ---\n");
    t0 = std::chrono::high_resolution_clock::now();

    // Convert language string to lowercase for matching
    std::string lang = params.target_lang;
    for (auto &c : lang) c = tolower(c);

    std::vector<std::vector<int>> codec_tokens;
    if (!talker_.generate(ref_text_tokens, target_text_tokens,
                           spk_embedding, ref_codes, lang,
                           codec_tokens, params.max_new_tokens,
                           params.sampling)) {
        printf("FAIL: codec generation failed\n");
        return false;
    }
    t1 = std::chrono::high_resolution_clock::now();
    int n_gen_frames = codec_tokens.empty() ? 0 : (int)codec_tokens[0].size();
    double generate_time = std::chrono::duration<double>(t1 - t0).count();

    if (params.profiling) {
        FILE *f = fopen("logs/cpp_codec_tokens.bin", "wb");
        if (f) {
            int nq = (int)codec_tokens.size();
            int nf = codec_tokens.empty() ? 0 : (int)codec_tokens[0].size();
            fwrite(&nq, 4, 1, f);
            fwrite(&nf, 4, 1, f);
            for (int q = 0; q < nq; q++)
                fwrite(codec_tokens[q].data(), sizeof(int), nf, f);
            fclose(f);
            printf("  [debug] dumped codec_tokens (%dx%d)\n", nq, nf);
        }
    }

    printf("  Generated %d codec frames (%.2f sec, %.1f frames/sec)\n",
           n_gen_frames, generate_time,
           n_gen_frames / generate_time);

    // Step 6: Decode codec tokens to audio
    printf("\n--- Step 6: Decode to audio ---\n");
    t0 = std::chrono::high_resolution_clock::now();

    // Concatenate ref_codes + generated codec_tokens for ICL-mode decoding
    // (decoder expects the full sequence including reference)
    std::vector<std::vector<int>> full_codes(codec_tokens.size());
    for (size_t q = 0; q < full_codes.size(); q++) {
        full_codes[q].reserve(n_ref_frames + n_gen_frames);
        if (q < ref_codes.size()) {
            full_codes[q].insert(full_codes[q].end(),
                                  ref_codes[q].begin(), ref_codes[q].end());
        }
        full_codes[q].insert(full_codes[q].end(),
                              codec_tokens[q].begin(), codec_tokens[q].end());
    }

    // Track L (M6.4) — chunk-driven decode orchestration.
    //
    // Structural blocker for "overlap with Step 5": TalkerLLM::generate()
    // is monolithic and out-of-scope for this track (strict file ownership
    // per §5 M6.4 — cannot edit talker.cpp or talker_cann_engine.*). So
    // chunk 0 cannot start decoding until all codec frames are ready; the
    // frame-streaming producer the contract envisions would require a
    // yield-per-frame hook that lives in files Track L cannot touch.
    //
    // What we CAN do is drive Step 6 through the public
    // SpeechTokenizerDecoder::forward_chunk() primitive added under this
    // track, so that a future track (with Talker streaming in scope) can
    // interleave forward_chunk() calls with codec frames as they arrive
    // without needing any further changes here. With TTS_DECODER_PIPELINE=1
    // we also spin a background worker thread that runs forward_chunk()
    // jobs pulled from a queue — the scaffolding a streaming producer
    // would feed. Today, because ACL contexts are thread-affine on the
    // 910B4 runtime (CANN 8.5) and the decoder session was created on the
    // main thread, calling forward_chunk() from a worker thread costs
    // context-switch overhead (~3× slower per chunk in practice). So the
    // default path runs the pipeline in-thread and produces bit-identical
    // output to decoder.decode(). The thread-pool variant is kept behind
    // TTS_DECODER_PIPELINE=2 for future work.
    const char *pipe_env = std::getenv("TTS_DECODER_PIPELINE");
    const int pipe_mode = (pipe_env && pipe_env[0] >= '1' && pipe_env[0] <= '9')
                           ? (pipe_env[0] - '0') : 0;

    std::vector<float> full_audio;
    int total_T = full_codes.empty() ? 0 : (int)full_codes[0].size();
    bool did_pipeline = false;

    if (pipe_mode >= 1 && total_T > SpeechTokenizerDecoder::cann_max_frames()) {
        // Chunk-driven decode: iterate the same chunk geometry
        // decode_chunked() uses, but route through the public
        // forward_chunk() API so a streaming producer could drive it.
        const int chunk_size = SpeechTokenizerDecoder::chunk_size();
        const int overlap    = SpeechTokenizerDecoder::overlap_frames();
        const int step       = SpeechTokenizerDecoder::chunk_step();
        const int upsample   = tokenizer_decoder_.upsample_rate();
        const int n_q        = (int)full_codes.size();

        auto pipe_t0 = std::chrono::high_resolution_clock::now();
        int chunk_count = 0;
        bool chunk_failed = false;

        auto run_chunk_inline = [&](int idx, int start, int end) -> bool {
            std::vector<std::vector<int>> chunk_codes(n_q);
            for (int q = 0; q < n_q; q++) {
                chunk_codes[q].assign(full_codes[q].begin() + start,
                                      full_codes[q].begin() + end);
            }
            std::vector<float> chunk_audio;
            if (!tokenizer_decoder_.forward_chunk(chunk_codes, chunk_audio,
                                                   /*prefer_cann=*/true)) {
                return false;
            }
            int skip_frames  = (start > 0) ? overlap : 0;
            int skip_samples = skip_frames * upsample;
            int keep_samples = (int)chunk_audio.size() - skip_samples;
            if (keep_samples > 0) {
                full_audio.insert(full_audio.end(),
                                  chunk_audio.begin() + skip_samples,
                                  chunk_audio.end());
            }
            printf("[decoder/pipeline]   chunk %d: frames [%d,%d), "
                   "keep %d samples\n", idx, start, end, keep_samples);
            return true;
        };

        if (pipe_mode == 1) {
            // Mode 1: in-thread chunk iteration via forward_chunk().
            // Same kernel launches as decode_chunked(), same ACL context.
            // Bit-identical audio to decode().
            for (int start = 0, idx = 0;
                 start < total_T;
                 start += step, idx++) {
                int end = std::min(start + chunk_size, total_T);
                if (start > 0 && (end - start) <= overlap) break;
                if (!run_chunk_inline(idx, start, end)) {
                    chunk_failed = true;
                    break;
                }
                chunk_count++;
            }
        } else {
            // Mode 2+: thread-pool variant — submit chunks to a worker.
            // Held behind TTS_DECODER_PIPELINE=2 because CANN context is
            // thread-affine; this path is slower today but is the shape
            // a streaming producer would feed.
            struct DecodeJob { int idx, start, end;
                               std::vector<std::vector<int>> codes;
                               bool sentinel = false; };
            struct DecodeResult { int idx, start, end;
                                  std::vector<float> audio; bool ok = false; };
            std::queue<DecodeJob> in_q;
            std::queue<DecodeResult> out_q;
            std::mutex in_mtx, out_mtx;
            std::condition_variable in_cv, out_cv;
            std::thread worker([&]() {
                while (true) {
                    DecodeJob job;
                    {
                        std::unique_lock<std::mutex> lk(in_mtx);
                        in_cv.wait(lk, [&]{ return !in_q.empty(); });
                        job = std::move(in_q.front()); in_q.pop();
                    }
                    if (job.sentinel) break;
                    DecodeResult r; r.idx = job.idx;
                    r.start = job.start; r.end = job.end;
                    r.ok = tokenizer_decoder_.forward_chunk(
                        job.codes, r.audio, /*prefer_cann=*/true);
                    { std::lock_guard<std::mutex> lk(out_mtx);
                      out_q.push(std::move(r)); }
                    out_cv.notify_one();
                }
            });
            int produced = 0, consumed = 0;
            const int max_inflight = 2;
            auto drain = [&]() -> bool {
                DecodeResult r;
                { std::unique_lock<std::mutex> lk(out_mtx);
                  out_cv.wait(lk, [&]{ return !out_q.empty(); });
                  r = std::move(out_q.front()); out_q.pop(); }
                if (!r.ok) { chunk_failed = true; return false; }
                int skip_frames = (r.start > 0) ? overlap : 0;
                int skip_samples = skip_frames * upsample;
                int keep_samples = (int)r.audio.size() - skip_samples;
                if (keep_samples > 0) {
                    full_audio.insert(full_audio.end(),
                        r.audio.begin() + skip_samples, r.audio.end());
                }
                printf("[decoder/pipeline-thr]   chunk %d: [%d,%d), keep %d\n",
                       r.idx, r.start, r.end, keep_samples);
                chunk_count++;
                return true;
            };
            for (int start = 0; start < total_T; start += step) {
                int end = std::min(start + chunk_size, total_T);
                if (start > 0 && (end - start) <= overlap) break;
                while ((produced - consumed) >= max_inflight) {
                    if (!drain()) break;
                    consumed++;
                }
                if (chunk_failed) break;
                DecodeJob j; j.idx = produced; j.start = start; j.end = end;
                j.codes.resize(n_q);
                for (int q = 0; q < n_q; q++)
                    j.codes[q].assign(full_codes[q].begin() + start,
                                      full_codes[q].begin() + end);
                { std::lock_guard<std::mutex> lk(in_mtx);
                  in_q.push(std::move(j)); }
                in_cv.notify_one();
                produced++;
            }
            while (!chunk_failed && consumed < produced) {
                if (!drain()) break;
                consumed++;
            }
            { DecodeJob s; s.sentinel = true;
              std::lock_guard<std::mutex> lk(in_mtx);
              in_q.push(std::move(s)); }
            in_cv.notify_one();
            worker.join();
        }

        if (chunk_failed) {
            printf("[decoder/pipeline] chunk failed — falling back to "
                   "decoder.decode() for correctness\n");
            full_audio.clear();
        } else {
            auto pipe_t1 = std::chrono::high_resolution_clock::now();
            double pipe_s = std::chrono::duration<double>(pipe_t1 - pipe_t0).count();
            printf("[decoder/pipeline] mode=%d: %d chunks → %zu samples in %.2f sec\n",
                   pipe_mode, chunk_count, full_audio.size(), pipe_s);
            did_pipeline = true;
        }
    }

    if (!did_pipeline) {
        if (!tokenizer_decoder_.decode(full_codes, full_audio)) {
            printf("FAIL: audio decoding failed\n");
            return false;
        }
    }
    t1 = std::chrono::high_resolution_clock::now();
    double decode_time = std::chrono::duration<double>(t1 - t0).count();
    printf("  Decoded %zu samples (%.2f sec, ratio %.1fx)%s\n",
           full_audio.size(), decode_time,
           full_audio.size() / 24000.0 / decode_time,
           did_pipeline ? " [pipelined]" : "");

    // Step 7: Remove reference audio portion
    // full_audio = decoder([ref_codes || gen_codes])
    // Only ref_codes correspond to the original ref audio; gen_codes are target-text only.
    // Cut proportionally: cut = n_ref_frames / total_frames * audio_length
    // (matches Python: cut = int(ref_len / total_len * wav.shape[0]))
    int total_frames = n_ref_frames + n_gen_frames;
    int cut = (int)((long long)n_ref_frames * (long long)full_audio.size() / std::max(total_frames, 1));
    printf("  Cutting first %d samples (%.2f sec) -- ref audio portion\n",
           cut, cut / 24000.0f);
    if (getenv("QWEN_TTS_KEEP_REF") != nullptr) {
        printf("  QWEN_TTS_KEEP_REF set — emitting full decoded audio "
               "(ref + target), for decoder round-trip check.\n");
        audio_out = std::move(full_audio);
    } else if (cut > 0 && cut < (int)full_audio.size()) {
        audio_out.assign(full_audio.begin() + cut, full_audio.end());
    } else {
        audio_out = std::move(full_audio);
    }

    // Fade-in mask for the decoder boundary artifact at the ref/target cut
    // point. Originally landed as e6eb3929 (50 ms linear); later restored
    // at 120 ms linear (1b88df98). User still reported a prefix click —
    // waveform analysis showed a noise burst at 50-100 ms that a 120 ms
    // linear fade only attenuates to ~0.6×. Cubic fade over 200 ms knocks
    // the first 100 ms down to 0.125× (inaudible) while still reaching full
    // amplitude by 200 ms (speech onset usually starts around 180-250 ms
    // post-cut, so the audible onset is not softened).
    {
        int fade_ms = 200;
        int fade_samples = std::min(
            (int)(fade_ms * 24 /* samples per ms at 24kHz */),
            (int)audio_out.size());
        for (int i = 0; i < fade_samples; i++) {
            float t = (float)i / fade_samples;   // 0..1
            audio_out[i] *= t * t * t;            // cubic ease-in
        }
    }

    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_t1 - total_t0).count();
    double target_duration = audio_out.size() / 24000.0;
    printf("\n=== Generation complete ===\n");
    printf("  Output: %zu samples (%.2f sec at 24kHz)\n",
           audio_out.size(), target_duration);
    printf("  Timing breakdown:\n");
    if (used_cache) {
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
// X-vector voice clone (no ref audio codec, just speaker embedding)
// ============================================================================

bool QwenTTS::generate_xvec(const QwenTTSParams& params, std::vector<float>& audio_out) {
    if (!loaded_) return false;

    printf("\n=== X-Vector Voice Clone ===\n");
    printf("  Ref audio: %s\n", params.ref_audio.c_str());
    printf("  Target text: %s\n", params.text.c_str());
    printf("  Language: %s\n", params.target_lang.c_str());

    // Lowercase language for map lookup (keys are lowercase in GGUF config)
    std::string lang = params.target_lang;
    for (auto &c : lang) c = tolower(c);

    // Extract speaker embedding from ref audio
    std::vector<float> ref_audio;
    if (!audio_io::load_audio(params.ref_audio, 24000, ref_audio)) {
        printf("FAIL: cannot load ref audio\n");
        return false;
    }
    std::vector<float> spk_embedding;
    if (!speaker_encoder_.extract(ref_audio, 24000, spk_embedding)) {
        printf("FAIL: speaker embedding extraction failed\n");
        return false;
    }
    printf("  Speaker embedding: %zu dims\n", spk_embedding.size());

    // Tokenize target text
    std::vector<int> ref_text_tokens, target_text_tokens;
    tokenize_tts_text("", params.text, ref_text_tokens, target_text_tokens);

    // Generate codec tokens via x-vector mode
    std::vector<std::vector<int>> codec_tokens;
    // Use llama.cpp backbone with MRoPE (dimension_sections in GGUF enables
    // automatic MRoPE via use_mrope() → CANN-accelerated rope_multi kernel).
    // The standalone transformer is no longer needed when GGUF has MRoPE metadata.
    {
        printf("  Using llama.cpp backbone (MRoPE via GGUF metadata)\n");
        if (!talker_.generate_xvec(target_text_tokens, spk_embedding,
                                    lang, codec_tokens,
                                    params.max_new_tokens, params.sampling)) {
            printf("FAIL: generation failed\n");
            return false;
        }
    }

    int n_gen_frames = codec_tokens.empty() ? 0 : (int)codec_tokens[0].size();
    printf("  Generated %d codec frames\n", n_gen_frames);

    // Decode to audio
    std::vector<float> full_audio;
    if (!tokenizer_decoder_.decode(codec_tokens, full_audio)) {
        printf("FAIL: audio decoding failed\n");
        return false;
    }

    // No ref audio to cut (x-vector mode doesn't include ref in codec)
    audio_out = std::move(full_audio);
    printf("  Output: %zu samples (%.2f sec at 24kHz)\n",
           audio_out.size(), audio_out.size() / 24000.0f);
    return true;
}

// ============================================================================
// CustomVoice (built-in speaker, no ref audio needed)
// ============================================================================

bool QwenTTS::generate_customvoice(const QwenTTSParams& params, std::vector<float>& audio_out) {
    if (!loaded_) return false;

    printf("\n=== CustomVoice Generation ===\n");
    printf("  Speaker: %s\n", params.speaker.c_str());
    printf("  Target text: %s\n", params.text.c_str());
    printf("  Language: %s\n", params.target_lang.c_str());

    // Lowercase language for map lookup
    std::string lang = params.target_lang;
    for (auto &c : lang) c = tolower(c);

    // Tokenize target text
    std::vector<int> ref_text_tokens, target_text_tokens;
    tokenize_tts_text("", params.text, ref_text_tokens, target_text_tokens);

    // Generate codec tokens via customvoice mode
    std::vector<std::vector<int>> codec_tokens;
    if (!talker_.generate_customvoice(target_text_tokens, params.speaker,
                                      lang, codec_tokens,
                                      params.max_new_tokens, params.sampling)) {
        printf("FAIL: generation failed\n");
        return false;
    }

    int n_gen_frames = codec_tokens.empty() ? 0 : (int)codec_tokens[0].size();
    printf("  Generated %d codec frames\n", n_gen_frames);

    // Decode to audio
    std::vector<float> full_audio;
    if (!tokenizer_decoder_.decode(codec_tokens, full_audio)) {
        printf("FAIL: audio decoding failed\n");
        return false;
    }

    audio_out = std::move(full_audio);
    printf("  Output: %zu samples (%.2f sec at 24kHz)\n",
           audio_out.size(), audio_out.size() / 24000.0f);
    return true;
}
