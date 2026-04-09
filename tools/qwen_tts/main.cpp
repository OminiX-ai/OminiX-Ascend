#include "qwen_tts.h"
#include "audio_io.h"
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sys/stat.h>

// ---------------------------------------------------------------------------
// Built-in voices: voices.json schema
//   {
//     "voices": [
//       {"id": "ellen", "lang": "en", "desc": "...", "cache": "ellen.bin"},
//       ...
//     ]
//   }
// Each "cache" path is resolved relative to voices_dir.
// ---------------------------------------------------------------------------

static std::string default_voices_dir() {
    // Search order: $CWD/tools/qwen_tts/data/voices, then data/voices
    struct stat st;
    const char* candidates[] = {
        "tools/qwen_tts/data/voices",
        "data/voices",
        nullptr,
    };
    for (int i = 0; candidates[i]; i++) {
        if (stat(candidates[i], &st) == 0 && S_ISDIR(st.st_mode)) return candidates[i];
    }
    return "tools/qwen_tts/data/voices";
}

static bool load_voices_json(const std::string& voices_dir, nlohmann::json& out) {
    std::string path = voices_dir + "/voices.json";
    std::ifstream f(path);
    if (!f.is_open()) {
        fprintf(stderr, "Error: cannot open %s\n", path.c_str());
        return false;
    }
    try { f >> out; } catch (const std::exception& e) {
        fprintf(stderr, "Error: failed to parse %s: %s\n", path.c_str(), e.what());
        return false;
    }
    return true;
}

static int list_voices(const std::string& voices_dir) {
    nlohmann::json j;
    if (!load_voices_json(voices_dir, j)) return 1;
    const auto& voices = j.value("voices", nlohmann::json::array());
    printf("Built-in voices (from %s/voices.json):\n", voices_dir.c_str());
    printf("  %-20s %-6s %s\n", "ID", "LANG", "DESCRIPTION");
    printf("  %-20s %-6s %s\n", "--", "----", "-----------");
    for (const auto& v : voices) {
        std::string id   = v.value("id",   "");
        std::string lang = v.value("lang", "");
        std::string desc = v.value("desc", "");
        printf("  %-20s %-6s %s\n", id.c_str(), lang.c_str(), desc.c_str());
    }
    if (voices.empty()) {
        printf("  (no voices defined — run scripts/bake_voices.sh to populate)\n");
    }
    return 0;
}

static bool resolve_voice(const std::string& voices_dir, const std::string& id,
                          std::string& out_cache_path) {
    nlohmann::json j;
    if (!load_voices_json(voices_dir, j)) return false;
    for (const auto& v : j.value("voices", nlohmann::json::array())) {
        if (v.value("id", "") == id) {
            std::string cache = v.value("cache", "");
            if (cache.empty()) {
                fprintf(stderr, "Error: voice '%s' has no 'cache' field\n", id.c_str());
                return false;
            }
            out_cache_path = voices_dir + "/" + cache;
            struct stat st;
            if (stat(out_cache_path.c_str(), &st) != 0) {
                fprintf(stderr, "Error: voice cache file not found: %s\n", out_cache_path.c_str());
                return false;
            }
            return true;
        }
    }
    fprintf(stderr, "Error: unknown voice id '%s' (use --list_voices to see available)\n", id.c_str());
    return false;
}

static void print_usage(const char* prog) {
    printf("Qwen3-TTS Voice Clone\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Required:\n");
    printf("  -m, --model_dir <path>     GGUF model directory\n");
    printf("  -t, --text <text>          Target text to synthesize\n");
    printf("  -r, --ref_audio <path>     Reference audio file (WAV, 24kHz)\n");
    printf("  --ref_text <text>          Reference audio transcript\n");
    printf("  (or) --voice <id>          Use a built-in voice (see --list_voices)\n");
    printf("  (or) --ref_cache <path>    Use a pre-computed speaker cache file\n");
    printf("  (or) --xvec <path>         Use a pre-extracted x-vector (.xvec file)\n");
    printf("\nTool modes:\n");
    printf("  --xvec_extract <wav>       Extract speaker embedding from wav into\n");
    printf("                             --xvec_out <file>, then exit (no synthesis)\n");
    printf("\nOptional:\n");
    printf("  --voices_dir <path>        Directory with voices.json (default: tools/qwen_tts/data/voices)\n");
    printf("  --list_voices              List built-in voices and exit\n");
    printf("  --xvec_out <path>          Output path for --xvec_extract (default: voice.xvec)\n");
    printf("  --tokenizer_dir <path>     Tokenizer directory (vocab.json + merges.txt)\n");
    printf("                             Default: same as model_dir\n");
    printf("  --target_lang <lang>       Target language (English/Chinese, default: English)\n");
    printf("  --ref_lang <lang>          Reference language (default: English)\n");
    printf("  -o, --output <path>        Output audio file (default: output.wav)\n");
    printf("  --ref_cache <path>         Pre-computed ref cache file (.bin)\n");
    printf("                             If file exists: load and skip encoder\n");
    printf("                             If not exists + ref_audio given: encode and save\n");
    printf("  --talker_model <path>      Override Talker GGUF file (for quantized models)\n");
    printf("  --cp_model <path>          Override CP llama GGUF (for NPU acceleration)\n");
    printf("  -d, --device <device>      Compute device (CPU, default: CPU)\n");
    printf("  -n, --n_threads <num>      Thread count (default: 8)\n");
    printf("  --n_gpu_layers <num>       Layers to offload to GPU/NPU (default: 0)\n");
    printf("  --max_tokens <num>         Max generated codec frames. When omitted,\n");
    printf("                             a per-text auto cap is applied (~8 frames per\n");
    printf("                             English BPE token, ~6 per Chinese, min 30) so\n");
    printf("                             the talker can't run away to the 2048 default.\n");
    printf("  --temperature <float>      Sampling temperature (default: 0.9)\n");
    printf("  --top_k <int>              Top-K sampling (default: 50, 0=disabled)\n");
    printf("  --top_p <float>            Top-P nucleus sampling (default: 1.0)\n");
    printf("  --repetition_penalty <f>   Repetition penalty (default: 1.05)\n");
    printf("  --greedy                   Disable sampling (use greedy decoding)\n");
    printf("  --seed <int>               Random seed for sampling (default: 42)\n");
    printf("  -p, --profiling            Enable profiling\n");
    printf("  --stream <chunk_frames>    Streaming mode: emit audio in chunks of\n");
    printf("                             N codec frames (~40ms per frame). Suggested\n");
    printf("                             N=8 (~320ms chunks). 0=off (default).\n");
    printf("  -h, --help                 Show this help\n");
    printf("\nExample:\n");
    printf("  %s -m gguf/ --tokenizer_dir /path/to/Qwen3-TTS/ \\\n", prog);
    printf("     -r ref.wav --ref_text \"Hello\" -t \"How are you?\" -o out.wav\n");
}

int main(int argc, char** argv) {
    QwenTTSParams params;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if ((arg == "-m" || arg == "--model_dir") && i + 1 < argc) {
            params.model_dir = argv[++i];
        } else if (arg == "--tokenizer_dir" && i + 1 < argc) {
            params.tokenizer_dir = argv[++i];
        } else if ((arg == "-t" || arg == "--text") && i + 1 < argc) {
            params.text = argv[++i];
        } else if (arg == "--target_lang" && i + 1 < argc) {
            params.target_lang = argv[++i];
        } else if ((arg == "-r" || arg == "--ref_audio") && i + 1 < argc) {
            params.ref_audio = argv[++i];
        } else if (arg == "--ref_text" && i + 1 < argc) {
            params.ref_text = argv[++i];
        } else if (arg == "--ref_lang" && i + 1 < argc) {
            params.ref_lang = argv[++i];
        } else if ((arg == "-o" || arg == "--output") && i + 1 < argc) {
            params.output = argv[++i];
        } else if (arg == "--talker_model" && i + 1 < argc) {
            params.talker_model = argv[++i];
        } else if (arg == "--cp_model" && i + 1 < argc) {
            params.cp_model = argv[++i];
        } else if (arg == "--ref_cache" && i + 1 < argc) {
            params.ref_cache = argv[++i];
        } else if (arg == "--voice" && i + 1 < argc) {
            params.voice = argv[++i];
        } else if (arg == "--voices_dir" && i + 1 < argc) {
            params.voices_dir = argv[++i];
        } else if (arg == "--list_voices") {
            std::string vdir = params.voices_dir.empty() ? default_voices_dir() : params.voices_dir;
            return list_voices(vdir);
        } else if (arg == "--xvec" && i + 1 < argc) {
            params.xvec = argv[++i];
        } else if (arg == "--xvec_extract" && i + 1 < argc) {
            params.xvec_extract = argv[++i];
        } else if (arg == "--xvec_out" && i + 1 < argc) {
            params.xvec_out = argv[++i];
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            params.device = argv[++i];
        } else if ((arg == "-n" || arg == "--n_threads") && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--n_gpu_layers" && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            params.max_new_tokens = std::atoi(argv[++i]);
            params.auto_max_tokens = false;  // user-set: skip the auto cap
        } else if (arg == "--temperature" && i + 1 < argc) {
            params.sampling.temperature = std::atof(argv[++i]);
        } else if (arg == "--top_k" && i + 1 < argc) {
            params.sampling.top_k = std::atoi(argv[++i]);
        } else if (arg == "--top_p" && i + 1 < argc) {
            params.sampling.top_p = std::atof(argv[++i]);
        } else if (arg == "--repetition_penalty" && i + 1 < argc) {
            params.sampling.repetition_penalty = std::atof(argv[++i]);
        } else if (arg == "--greedy") {
            params.sampling.do_sample = false;
            params.sampling.cp_do_sample = false;
        } else if (arg == "--seed" && i + 1 < argc) {
            set_sampling_seed(std::atoi(argv[++i]));
        } else if (arg == "-p" || arg == "--profiling") {
            params.profiling = true;
        } else if (arg == "--stream" && i + 1 < argc) {
            params.stream_chunk_frames = std::atoi(argv[++i]);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    // ----- Tool mode: --xvec_extract -----
    // Loads only what we need (speaker encoder), runs extract_xvec(), exits.
    if (!params.xvec_extract.empty()) {
        if (params.model_dir.empty()) {
            fprintf(stderr, "Error: --model_dir is required for --xvec_extract\n");
            return 1;
        }
        if (params.xvec_out.empty()) {
            params.xvec_out = "voice.xvec";
            fprintf(stderr, "[xvec_extract] --xvec_out not specified, using default: %s\n",
                    params.xvec_out.c_str());
        }
        // Stub a target text to satisfy load() — extract path doesn't use it.
        if (params.text.empty()) params.text = ".";
        QwenTTS tts;
        if (!tts.load(params)) {
            fprintf(stderr, "Failed to load models\n");
            return 1;
        }
        return tts.extract_xvec(params.xvec_extract, params.xvec_out) ? 0 : 1;
    }

    // Resolve --voice → --ref_cache
    if (!params.voice.empty()) {
        std::string vdir = params.voices_dir.empty() ? default_voices_dir() : params.voices_dir;
        std::string cache_path;
        if (!resolve_voice(vdir, params.voice, cache_path)) return 1;
        if (!params.ref_cache.empty() && params.ref_cache != cache_path) {
            fprintf(stderr, "Error: --voice and --ref_cache are mutually exclusive\n");
            return 1;
        }
        params.ref_cache = cache_path;
        printf("[voice] using built-in voice '%s' -> %s\n",
               params.voice.c_str(), cache_path.c_str());
    }

    bool has_xvec      = !params.xvec.empty();
    bool has_ref_cache = !params.ref_cache.empty();
    bool has_ref_audio = !params.ref_audio.empty() && !params.ref_text.empty();

    // xvec is mutually exclusive with ICL inputs
    if (has_xvec && (has_ref_cache || has_ref_audio)) {
        fprintf(stderr, "Error: --xvec is mutually exclusive with --voice/--ref_cache/--ref_audio (xvec mode is x-vector-only, no ICL)\n");
        return 1;
    }

    if (params.model_dir.empty() || params.text.empty() ||
        (!has_xvec && !has_ref_cache && !has_ref_audio)) {
        fprintf(stderr, "Error: --model_dir, --text, and one of (--voice | --ref_cache | --ref_audio + --ref_text | --xvec) are required\n\n");
        print_usage(argv[0]);
        return 1;
    }

    QwenTTS tts;
    if (!tts.load(params)) {
        fprintf(stderr, "Failed to load models\n");
        return 1;
    }

    // Warmup run (same input, discard output) to trigger lazy kernel compilation
    {
        printf("\n=== Warmup run ===\n");
        QwenTTSParams warmup_params = params;
        warmup_params.max_new_tokens = 5;  // minimal generation
        warmup_params.profiling = false;
        warmup_params.stream_chunk_frames = 0;  // never stream the warmup
        warmup_params.stream_callback = nullptr;
        std::vector<float> warmup_audio;
        tts.generate(warmup_params, warmup_audio);
        printf("=== Warmup complete ===\n\n");
    }

    // If --stream is set, install a chunk callback that just measures
    // first-byte and per-chunk latency. The audio is still accumulated
    // into audio_out (by the streaming path inside QwenTTS::generate) and
    // saved as a single wav file at the end.
    auto wall_start = std::chrono::high_resolution_clock::now();
    int  stream_chunks_seen   = 0;
    bool stream_first_emitted = false;
    double stream_first_byte_ms = 0;
    if (params.stream_chunk_frames > 0) {
        params.stream_callback = [&](const float * /*samples*/,
                                      size_t n_samples,
                                      bool   is_final) {
            if (!stream_first_emitted && n_samples > 0) {
                auto now = std::chrono::high_resolution_clock::now();
                stream_first_byte_ms = std::chrono::duration<double, std::milli>(
                                          now - wall_start).count();
                stream_first_emitted = true;
                printf("[stream] first audio chunk: %.0f ms after generate() "
                       "(%.2f sec of audio)\n",
                       stream_first_byte_ms, n_samples / 24000.0);
            }
            if (n_samples > 0) {
                stream_chunks_seen++;
                printf("[stream] chunk %d: %zu samples (%.2f sec)%s\n",
                       stream_chunks_seen, n_samples, n_samples / 24000.0,
                       is_final ? " [final]" : "");
            } else if (is_final) {
                printf("[stream] end-of-stream marker\n");
            }
        };
    }

    wall_start = std::chrono::high_resolution_clock::now();
    std::vector<float> audio_out;
    if (!tts.generate(params, audio_out)) {
        fprintf(stderr, "Failed to generate audio\n");
        return 1;
    }
    if (params.stream_chunk_frames > 0 && stream_first_emitted) {
        auto now = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(
                              now - wall_start).count();
        printf("[stream] first-byte=%.0f ms, total=%.0f ms, %d chunks, "
               "%.2fx speedup over wait-for-full\n",
               stream_first_byte_ms, total_ms, stream_chunks_seen,
               total_ms / stream_first_byte_ms);
    }

    // Save output audio
    if (!audio_out.empty()) {
        if (audio_io::save_wav(params.output, audio_out, 24000)) {
            printf("Saved output to %s (%zu samples, %.2f sec)\n",
                   params.output.c_str(), audio_out.size(),
                   audio_out.size() / 24000.0f);
        } else {
            fprintf(stderr, "Failed to save output audio\n");
            return 1;
        }
    }

    return 0;
}
