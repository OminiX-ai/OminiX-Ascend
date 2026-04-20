/* test_api_smoke.c — Ascend API Bridge Contract §5 B1.4.
 *
 * Minimal C (not C++) smoke test for libqwen_tts_api.so.
 * Loads + frees the handle 10 times in a loop to verify:
 *   (a) dlopen-compatibility of the .so,
 *   (b) no crashes across repeated context lifetimes,
 *   (c) no obvious NPU/GPU memory leak (compare npu-smi before/after).
 *
 * Build: (from build-85-cann-on/)
 *   gcc -std=c11 -O2 -I../tools/qwen_tts ../tools/qwen_tts/test_api_smoke.c \
 *       -Wl,-rpath,$PWD/bin -Lbin -lqwen_tts_api -o bin/test_api_smoke
 *
 * Run:
 *   MODEL_DIR=/path/to/gguf/dir ./bin/test_api_smoke
 */
#define _POSIX_C_SOURCE 200809L
#include "qwen_tts_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define N_ITER 10

int main(int argc, char** argv) {
    const char* model_dir = getenv("MODEL_DIR");
    if (!model_dir && argc > 1) model_dir = argv[1];
    if (!model_dir) {
        fprintf(stderr, "usage: MODEL_DIR=/path/to/gguf %s    (or pass as argv[1])\n", argv[0]);
        return 2;
    }

    const int n_gpu_layers = 29;  /* all layers to NPU for 1.7B */
    const int n_threads    = 4;

    printf("[smoke] libqwen_tts_api load/free loop; model_dir=%s iters=%d\n",
           model_dir, N_ITER);

    int failures = 0;
    for (int i = 0; i < N_ITER; i++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);

        qwen_tts_ctx_t* ctx = qwen_tts_load(
            model_dir,
            NULL,          /* tokenizer_dir = same as model_dir */
            NULL,          /* talker_override */
            NULL,          /* cp_override */
            n_gpu_layers,
            n_threads);
        if (!ctx) {
            fprintf(stderr, "[smoke] iter %d: qwen_tts_load returned NULL\n", i);
            failures++;
            continue;
        }

        int hidden = qwen_tts_hidden_size(ctx);
        int vocab  = qwen_tts_vocab_size(ctx);
        int has_spk = qwen_tts_has_speaker_encoder(ctx);

        qwen_tts_free(ctx);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;

        printf("[smoke] iter %d OK hidden=%d vocab=%d spk=%d %.2fs\n",
               i, hidden, vocab, has_spk, dt);
    }

    if (failures) {
        fprintf(stderr, "[smoke] FAIL: %d/%d iters failed\n", failures, N_ITER);
        return 1;
    }
    printf("[smoke] PASS: %d/%d load+free cycles, no crash\n", N_ITER, N_ITER);
    return 0;
}
