#include "qwen_tts.h"
#include "audio_io.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

static void print_usage(const char* prog) {
    printf("Qwen3-TTS Voice Clone\n\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Required:\n");
    printf("  -m, --model_dir <path>     GGUF model directory\n");
    printf("  -t, --text <text>          Target text to synthesize\n");
    printf("  -r, --ref_audio <path>     Reference audio file (WAV, 24kHz)\n");
    printf("  --ref_text <text>          Reference audio transcript\n");
    printf("\nOptional:\n");
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
    printf("  --max_tokens <num>         Max generated codec frames (default: 2048)\n");
    printf("  --temperature <float>      Sampling temperature (default: 0.9)\n");
    printf("  --top_k <int>              Top-K sampling (default: 50, 0=disabled)\n");
    printf("  --top_p <float>            Top-P nucleus sampling (default: 1.0)\n");
    printf("  --repetition_penalty <f>   Repetition penalty (default: 1.05)\n");
    printf("  --greedy                   Disable sampling (use greedy decoding)\n");
    printf("  --seed <int>               Random seed for sampling (default: 42)\n");
    printf("  -p, --profiling            Enable profiling\n");
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
        } else if ((arg == "-d" || arg == "--device") && i + 1 < argc) {
            params.device = argv[++i];
        } else if ((arg == "-n" || arg == "--n_threads") && i + 1 < argc) {
            params.n_threads = std::atoi(argv[++i]);
        } else if (arg == "--n_gpu_layers" && i + 1 < argc) {
            params.n_gpu_layers = std::atoi(argv[++i]);
        } else if (arg == "--max_tokens" && i + 1 < argc) {
            params.max_new_tokens = std::atoi(argv[++i]);
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
        } else {
            fprintf(stderr, "Unknown argument: %s\n", argv[i]);
            print_usage(argv[0]);
            return 1;
        }
    }

    bool has_ref_cache = !params.ref_cache.empty();
    bool has_ref_audio = !params.ref_audio.empty() && !params.ref_text.empty();
    if (params.model_dir.empty() || params.text.empty() ||
        (!has_ref_cache && !has_ref_audio)) {
        fprintf(stderr, "Error: --model_dir, --text, and (--ref_audio + --ref_text or --ref_cache) are required\n\n");
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
        std::vector<float> warmup_audio;
        tts.generate(warmup_params, warmup_audio);
        printf("=== Warmup complete ===\n\n");
    }

    std::vector<float> audio_out;
    if (!tts.generate(params, audio_out)) {
        fprintf(stderr, "Failed to generate audio\n");
        return 1;
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
