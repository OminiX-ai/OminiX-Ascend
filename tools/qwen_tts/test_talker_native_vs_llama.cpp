// Contract item M1.6: validate TalkerCannEngine against the llama.cpp path.
//
// Given a fixed random input embedding, run it through both:
//   (a) llama.cpp talker backbone (via llama_decode with batch.embd)
//   (b) the new TalkerCannEngine::forward_decode
// and compare final hidden states.
//
// Pass criteria (contract M1.6): max absolute element error < 0.01 on
// output hidden [n_embd], and cosine similarity > 0.999.
//
// Usage:
//   test_talker_native_vs_llama <talker_llama_gguf_path> [--n_threads N]

#include "llama.h"
#include "talker_cann_engine.h"
#include "talker.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

// Deterministic input: Gaussian noise with a fixed seed so both paths see the
// same bytes. The actual values matter less than reproducibility.
std::vector<float> make_fixed_input(int n_embd, uint64_t seed) {
    std::vector<float> v(n_embd);
    std::mt19937_64 rng(seed);
    std::normal_distribution<float> dist(0.0f, 0.25f);
    for (int i = 0; i < n_embd; ++i) v[i] = dist(rng);
    return v;
}

int run_llama(const std::string &gguf, const std::vector<float> &input_embed,
               int n_embd, std::vector<float> &hidden_out) {
    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 29;
    llama_model *model = llama_model_load_from_file(gguf.c_str(), mp);
    if (!model) {
        fprintf(stderr, "failed to load llama talker gguf: %s\n", gguf.c_str());
        return 1;
    }
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 256;
    cp.embeddings = true;
    cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "failed to init llama ctx\n");
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_init(1, n_embd, 1);
    batch.n_tokens = 1;
    std::memcpy(batch.embd, input_embed.data(), n_embd * sizeof(float));
    batch.pos[0] = 0;
    batch.n_seq_id[0] = 1;
    batch.seq_id[0][0] = 0;
    batch.logits[0] = 1;

    if (llama_decode(ctx, batch) != 0) {
        fprintf(stderr, "llama_decode failed\n");
        llama_batch_free(batch);
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    const float *h = llama_get_embeddings_ith(ctx, -1);
    hidden_out.assign(h, h + n_embd);

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

int run_native(const std::string &gguf, const std::vector<float> &input_embed,
                int n_embd, std::vector<float> &hidden_out) {
#if !defined(QWEN_TTS_HAS_CP_CANN)
    fprintf(stderr, "Build without QWEN_TTS_HAS_CP_CANN — skipping.\n");
    return 1;
#else
    TalkerCannEngine eng;
    TalkerConfig cfg;  // defaults from talker.h (Qwen3 1.7B)
    if (!eng.init_from_gguf(gguf, cfg, /*device=*/0)) {
        fprintf(stderr, "TalkerCannEngine init failed\n");
        return 1;
    }
    hidden_out.assign(n_embd, 0.0f);
    eng.forward_decode(input_embed.data(), /*pos=*/0, hidden_out.data());
    return 0;
#endif
}

void report(const std::vector<float> &a, const std::vector<float> &b) {
    size_t n = std::min(a.size(), b.size());
    if (n == 0) { printf("empty\n"); return; }
    float max_abs = 0, sum_sq = 0, dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n; ++i) {
        float d = std::fabs(a[i] - b[i]);
        if (d > max_abs) max_abs = d;
        sum_sq += d * d;
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    float rmse = std::sqrt(sum_sq / (float)n);
    float cos = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0f;
    printf("  n=%zu max_abs=%.4e rmse=%.4e cosine_sim=%.6f\n", n, max_abs, rmse, cos);
    bool pass_abs = max_abs < 0.01f;
    bool pass_cos = cos > 0.999f;
    printf("  gate: max_abs<0.01 %s | cos>0.999 %s | %s\n",
           pass_abs ? "PASS" : "FAIL",
           pass_cos ? "PASS" : "FAIL",
           (pass_abs && pass_cos) ? "OVERALL PASS" : "OVERALL FAIL");
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <talker_llama_gguf> [--n_threads N]\n", argv[0]);
        return 2;
    }
    std::string gguf = argv[1];
    int n_embd = 2048;  // from TalkerConfig default

    auto input = make_fixed_input(n_embd, 0x5eed5eed5eed5eedULL);

    std::vector<float> h_llama, h_native;
    printf("[1/2] running llama.cpp path ...\n");
    if (run_llama(gguf, input, n_embd, h_llama) != 0) return 1;
    printf("[2/2] running native TalkerCannEngine ...\n");
    if (run_native(gguf, input, n_embd, h_native) != 0) return 1;

    printf("\nllama hidden RMS: %.4f, native RMS: %.4f\n",
           std::sqrt([&]{ float s=0; for (float v:h_llama) s += v*v; return s/h_llama.size(); }()),
           std::sqrt([&]{ float s=0; for (float v:h_native) s += v*v; return s/h_native.size(); }()));
    printf("\ndiff:\n");
    report(h_llama, h_native);
    return 0;
}
