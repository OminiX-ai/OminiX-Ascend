#include "qwen_tts.h"
#include "audio_io.h"
#include <chrono>
#include <cstdio>
#include <numeric>
#include <cmath>

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

    // 3. Speech Tokenizer Encoder (CPU: CANN 2.3x slower due to kernel launch overhead)
    printf("\n[3/5] Loading speech tokenizer encoder...\n");
    ContextParams enc_params;
    enc_params.device_name = cpu_device;
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

    printf("\n=== Voice Clone Generation ===\n");
    printf("  Ref audio: %s\n", params.ref_audio.c_str());
    printf("  Ref text: %s\n", params.ref_text.c_str());
    printf("  Target text: %s\n", params.text.c_str());
    printf("  Language: %s\n", params.target_lang.c_str());

    auto total_t0 = std::chrono::high_resolution_clock::now();

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

    // Step 2: Extract speaker embedding
    printf("\n--- Step 2: Extract speaker embedding ---\n");
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> spk_embedding;
    if (!speaker_encoder_.extract(ref_audio, 24000, spk_embedding)) {
        printf("FAIL: speaker embedding extraction failed\n");
        return false;
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("  Speaker embedding: %zu dims (%.2f sec)\n",
           spk_embedding.size(),
           std::chrono::duration<double>(t1 - t0).count());

    if (params.profiling) {
        FILE *f = fopen("logs/cpp_spk_embedding.bin", "wb");
        if (f) {
            int dim = (int)spk_embedding.size();
            fwrite(&dim, 4, 1, f);
            fwrite(spk_embedding.data(), sizeof(float), dim, f);
            fclose(f);
            printf("  [debug] dumped spk_embedding (%d dims)\n", dim);
        }
    }

    // Step 3: Encode reference audio to codec tokens
    printf("\n--- Step 3: Encode reference audio ---\n");
    t0 = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<int>> ref_codes;
    std::vector<float> encoder_hidden;
    if (!tokenizer_encoder_.encode(ref_audio, ref_codes, &encoder_hidden)) {
        printf("FAIL: audio encoding failed\n");
        return false;
    }
    t1 = std::chrono::high_resolution_clock::now();
    int n_ref_frames = ref_codes.empty() ? 0 : (int)ref_codes[0].size();
    printf("  Ref codes: %d quantizers x %d frames (%.2f sec)\n",
           (int)ref_codes.size(), n_ref_frames,
           std::chrono::duration<double>(t1 - t0).count());

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

    // Step 4: Tokenize text
    printf("\n--- Step 4: Tokenize text ---\n");
    std::vector<int> ref_text_tokens, target_text_tokens;
    tokenize_tts_text(params.ref_text, params.text,
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

    std::vector<float> full_audio;
    if (!tokenizer_decoder_.decode(full_codes, full_audio)) {
        printf("FAIL: audio decoding failed\n");
        return false;
    }
    t1 = std::chrono::high_resolution_clock::now();
    double decode_time = std::chrono::duration<double>(t1 - t0).count();
    printf("  Decoded %zu samples (%.2f sec, ratio %.1fx)\n",
           full_audio.size(), decode_time,
           full_audio.size() / 24000.0 / decode_time);

    // Step 7: Remove reference audio portion
    // full_audio = decoder([ref_codes || gen_codes])
    // Only ref_codes correspond to the original ref audio; gen_codes are target-text only.
    // Cut proportionally: cut = n_ref_frames / total_frames * audio_length
    // (matches Python: cut = int(ref_len / total_len * wav.shape[0]))
    int total_frames = n_ref_frames + n_gen_frames;
    int cut = (int)((long long)n_ref_frames * (long long)full_audio.size() / std::max(total_frames, 1));
    printf("  Cutting first %d samples (%.2f sec) -- ref audio portion\n",
           cut, cut / 24000.0f);
    if (cut > 0 && cut < (int)full_audio.size()) {
        audio_out.assign(full_audio.begin() + cut, full_audio.end());
    } else {
        audio_out = std::move(full_audio);
    }



    auto total_t1 = std::chrono::high_resolution_clock::now();
    double total_time = std::chrono::duration<double>(total_t1 - total_t0).count();
    double target_duration = audio_out.size() / 24000.0;
    printf("\n=== Generation complete ===\n");
    printf("  Output: %zu samples (%.2f sec at 24kHz)\n",
           audio_out.size(), target_duration);
    printf("  Timing breakdown:\n");
    printf("    Prefill:   %.2f sec\n", prefill_time);
    printf("    Generate:  %.2f sec\n", generate_time);
    printf("    Decode:    %.2f sec\n", decode_time);
    printf("    Total:     %.2f sec\n", total_time);
    printf("  Inference RTF: %.2fx (generate+decode / audio)\n",
           (generate_time + decode_time) / target_duration);
    printf("  Total RTF:     %.2fx (end-to-end / audio)\n",
           total_time / target_duration);

    return true;
}
