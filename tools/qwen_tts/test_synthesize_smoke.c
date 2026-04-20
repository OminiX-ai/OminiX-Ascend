/*
 * test_synthesize_smoke.c — B5 end-to-end smoke test for qwen_tts_synthesize.
 *
 * Loads the TTS stack via qwen_tts_load, runs a single ICL synthesis against
 * the canonical mayun_ref (Chinese), writes the resulting PCM as a 24kHz mono
 * 16-bit WAV, releases the buffer via qwen_tts_pcm_free, and tears down
 * the context. Exits 0 on success.
 *
 * Usage:
 *   test_synthesize_smoke <model_dir> <ref_audios_dir> <out_wav>
 *
 * Invoked from contract §5 B5.4 verification; see build-85-cann-on/bin/.
 */

#include "qwen_tts_api.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Build a 24kHz mono 16-bit PCM WAV on disk from a float PCM buffer. */
static int write_wav(const char* path, const float* pcm, int n_samples, int sample_rate) {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[smoke] cannot open %s for write\n", path);
        return -1;
    }

    const uint16_t channels       = 1;
    const uint16_t bits_per_sample = 16;
    const uint32_t byte_rate       = (uint32_t)sample_rate * channels * bits_per_sample / 8;
    const uint16_t block_align     = channels * bits_per_sample / 8;
    const uint32_t data_bytes      = (uint32_t)n_samples * block_align;
    const uint32_t riff_bytes      = 36 + data_bytes;

    fwrite("RIFF", 1, 4, f);
    fwrite(&riff_bytes, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    uint32_t fmt_size = 16;
    uint16_t pcm_fmt  = 1;
    uint32_t sr       = (uint32_t)sample_rate;
    fwrite(&fmt_size, 4, 1, f);
    fwrite(&pcm_fmt, 2, 1, f);
    fwrite(&channels, 2, 1, f);
    fwrite(&sr, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&data_bytes, 4, 1, f);

    /* Clamp + convert f32 [-1, 1] → i16. */
    for (int i = 0; i < n_samples; i++) {
        float v = pcm[i];
        if (v >  1.0f) v =  1.0f;
        if (v < -1.0f) v = -1.0f;
        int16_t s = (int16_t)(v * 32767.0f);
        fwrite(&s, 2, 1, f);
    }

    fclose(f);
    return 0;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <model_dir> <ref_audios_dir> <out_wav>\n", argv[0]);
        return 2;
    }
    const char* model_dir      = argv[1];
    const char* ref_audios_dir = argv[2];
    const char* out_wav        = argv[3];

    char ref_audio_path[1024];
    snprintf(ref_audio_path, sizeof(ref_audio_path), "%s/mayun_ref.wav", ref_audios_dir);

    /* mayun_ref.txt line 1 = language, line 2 = transcript (see data/ref_audios/). */
    const char* ref_text =
        "\xe8\xbf\x99\xe6\x98\xaf\xe6\x88\x91\xe4\xbb\xac\xe6\x9c\x80\xe5\xa4\xa7\xe7\x9a\x84"  /* 这是我们最大的 */
        "\xe5\xb8\x8c\xe6\x9c\x9b\xef\xbc\x8c\xe8\x83\xbd\xe6\x8b\x9b\xe8\x81\x98\xe5\xbe\x97"  /* 希望，能招聘得 */
        "\xe5\x88\xb0\xe4\xba\xba\xe3\x80\x82\xe5\x9c\xa8\xe4\xbb\x8a\xe5\xa4\xa9\xe9\x98\xbf"  /* 到人。在今天阿 */
        "\xe9\x87\x8c\xe5\xb7\xb4\xe5\xb7\xb4\xe5\x85\xac\xe5\x8f\xb8\xe5\x86\x85\xe9\x83\xa8"  /* 里巴巴公司内部 */
        "\xef\xbc\x8c\xe6\x88\x91\xe8\x87\xaa\xe5\xb7\xb1\xe8\xbf\x99\xe4\xb9\x88\xe8\xa7\x89"  /* ，我自己这么觉 */
        "\xe5\xbe\x97\xef\xbc\x8c\xe4\xba\xba\xe6\x89\x8d\xe6\xa2\xaf\xe9\x98\x9f\xe7\x9a\x84"  /* 得，人才梯队的 */
        "\xe5\xbb\xba\xe8\xae\xbe\xe9\x9d\x9e\xe5\xb8\xb8\xe4\xb9\x8b\xe5\xa5\xbd\xe3\x80\x82"; /* 建设非常之好。 */

    /* Short trivial target — "大家好，今天天气真不错。" */
    const char* target_text =
        "\xe5\xa4\xa7\xe5\xae\xb6\xe5\xa5\xbd\xef\xbc\x8c"      /* 大家好， */
        "\xe4\xbb\x8a\xe5\xa4\xa9\xe5\xa4\xa9\xe6\xb0\x94"      /* 今天天气 */
        "\xe7\x9c\x9f\xe4\xb8\x8d\xe9\x94\x99\xe3\x80\x82";     /* 真不错。 */

    printf("[smoke] loading model_dir=%s\n", model_dir);
    qwen_tts_ctx_t* ctx = qwen_tts_load(
        model_dir,
        NULL,    /* tokenizer_dir = model_dir */
        NULL,    /* talker_override auto */
        NULL,    /* cp_override auto */
        29,      /* n_gpu_layers: all 29 Talker layers on NPU (matches qwen_tts CLI) */
        8        /* n_threads */
    );
    if (!ctx) {
        fprintf(stderr, "[smoke] qwen_tts_load failed\n");
        return 1;
    }
    printf("[smoke] loaded: hidden=%d vocab=%d spk=%d\n",
           qwen_tts_hidden_size(ctx),
           qwen_tts_vocab_size(ctx),
           qwen_tts_has_speaker_encoder(ctx));

    qwen_tts_synth_params_t p;
    memset(&p, 0, sizeof(p));
    p.text           = target_text;
    p.ref_audio_path = ref_audio_path;
    p.ref_text       = ref_text;
    p.ref_lang       = "Chinese";
    p.target_lang    = "Chinese";
    p.mode           = "icl";
    p.speaker        = NULL;
    /* seed/max_tokens/temperature/top_k/top_p/repetition_penalty/cp_groups/
     * cp_layers/greedy all default via the 0-sentinel. */

    float* pcm    = NULL;
    int n_samples = 0;
    printf("[smoke] calling qwen_tts_synthesize (mode=%s, ref=%s)\n",
           p.mode, p.ref_audio_path);
    int rc = qwen_tts_synthesize(ctx, &p, &pcm, &n_samples);
    if (rc != 0) {
        fprintf(stderr, "[smoke] synthesize returned %d (expected 0)\n", rc);
        qwen_tts_pcm_free(pcm);
        qwen_tts_free(ctx);
        return 1;
    }
    if (!pcm || n_samples <= 0) {
        fprintf(stderr, "[smoke] synthesize returned success but pcm=%p n=%d\n",
                (void*)pcm, n_samples);
        qwen_tts_pcm_free(pcm);
        qwen_tts_free(ctx);
        return 1;
    }

    printf("[smoke] pcm: %d samples, %.3f sec at 24kHz\n",
           n_samples, (double)n_samples / 24000.0);

    if (write_wav(out_wav, pcm, n_samples, 24000) != 0) {
        qwen_tts_pcm_free(pcm);
        qwen_tts_free(ctx);
        return 1;
    }
    printf("[smoke] wrote %s (%d samples)\n", out_wav, n_samples);

    qwen_tts_pcm_free(pcm);
    qwen_tts_free(ctx);
    printf("[smoke] PASS\n");
    return 0;
}
