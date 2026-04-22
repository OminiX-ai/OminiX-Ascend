// ============================================================================
// qwen_image_edit_native — Phase 1 driver for ImageDiffusionEngine.
//
// Minimal CLI: opens the engine against a DiT GGUF and exits. Once Phase 3/4
// land, this grows `--image`, `--prompt`, `--ref-image` flags to run an
// end-to-end edit task. For now its job is to prove the scaffold compiles +
// links on ac03 and the class can construct + destruct cleanly.
//
// Cooperative HBM lock: on ac03 this binary co-tenants with A4b's TTS prefill
// smoke (~14 GiB). Before any GGUF load, the driver probes `/tmp/ac03_hbm_lock`
// and waits if present. When enabled via --hbm-lock, the driver also creates
// its OWN lock file for the duration of the run so A4b reciprocates. Contract:
// see tools/qwen_image_edit/native/AGENTS.md (Phase 1 sets this up; A4b is
// expected to honor the same lock path from its side).
// ============================================================================

#include "image_diffusion_engine.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>

#include <sys/stat.h>
#include <unistd.h>

static void usage(const char *prog) {
    printf("Usage: %s [options]\n\n", prog);
    printf("Phase 1 options:\n");
    printf("  --gguf FILE          DiT weights GGUF\n");
    printf("  --device N           ACL device ID (default 0)\n");
    printf("  --steps N            Denoising steps (default 20)\n");
    printf("  --cfg SCALE          Classifier-free guidance scale (default 4.0)\n");
    printf("  --seq-img N          Max image token seq (default 4096)\n");
    printf("  --seq-txt N          Max text token seq (default 256)\n");
    printf("  --q4                 Enable Q4 weight quantization (Phase 2 late)\n");
    printf("  --hbm-lock           Take/release /tmp/ac03_hbm_lock around init\n");
    printf("  --wait-hbm-lock-s N  Poll seconds when lock is held (default 5)\n");
    printf("  -h, --help           Show this help\n");
}

static bool file_exists(const char *path) {
    struct stat st;
    return ::stat(path, &st) == 0;
}

static bool acquire_hbm_lock(const char *path) {
    // O_EXCL | O_CREAT would be cleaner but requires <fcntl.h>; for Phase 1
    // the coarser existence probe is fine — the harness doesn't race two
    // instances of the same binary, only ImageDiffusionEngine vs TalkerCannEngine.
    if (file_exists(path)) return false;
    FILE *f = fopen(path, "w");
    if (!f) return false;
    fprintf(f, "qie_native pid=%d\n", (int)getpid());
    fclose(f);
    return true;
}

static void release_hbm_lock(const char *path) {
    ::unlink(path);
}

int main(int argc, char *argv[]) {
    std::string gguf_path;
    int   device       = 0;
    int   steps        = 20;
    float cfg_scale    = 4.0f;
    int   seq_img      = 4096;
    int   seq_txt      = 256;
    bool  use_q4       = false;
    bool  use_hbm_lock = false;
    int   lock_wait_s  = 5;

    for (int i = 1; i < argc; ++i) {
        auto S = [&](int k){ return std::string(argv[k]); };
        if      (strcmp(argv[i], "--gguf") == 0       && i+1<argc) gguf_path = S(++i);
        else if (strcmp(argv[i], "--device") == 0     && i+1<argc) device    = atoi(argv[++i]);
        else if (strcmp(argv[i], "--steps") == 0      && i+1<argc) steps     = atoi(argv[++i]);
        else if (strcmp(argv[i], "--cfg") == 0        && i+1<argc) cfg_scale = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seq-img") == 0    && i+1<argc) seq_img   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seq-txt") == 0    && i+1<argc) seq_txt   = atoi(argv[++i]);
        else if (strcmp(argv[i], "--q4") == 0)                     use_q4    = true;
        else if (strcmp(argv[i], "--hbm-lock") == 0)               use_hbm_lock = true;
        else if (strcmp(argv[i], "--wait-hbm-lock-s") == 0 && i+1<argc)
            lock_wait_s = atoi(argv[++i]);
        else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            usage(argv[0]); return 0;
        } else {
            fprintf(stderr, "Unknown arg: %s\n", argv[i]);
            usage(argv[0]); return 2;
        }
    }

    if (gguf_path.empty()) {
        fprintf(stderr, "error: --gguf is required\n");
        usage(argv[0]); return 2;
    }

    const char *lock_path = "/tmp/ac03_hbm_lock";
    if (use_hbm_lock) {
        // Poll-sleep while A4b holds the lock. Phase 1 is coarse: a single
        // existence probe + sleep. A4b is expected to do the same on its side.
        int waited = 0;
        while (file_exists(lock_path)) {
            printf("[qie_native] waiting for HBM lock (held by peer); sleeping %ds\n",
                   lock_wait_s);
            std::this_thread::sleep_for(std::chrono::seconds(lock_wait_s));
            waited += lock_wait_s;
            if (waited >= 600) {
                fprintf(stderr, "[qie_native] gave up waiting for HBM lock after %ds\n",
                        waited);
                return 1;
            }
        }
        if (!acquire_hbm_lock(lock_path)) {
            fprintf(stderr, "[qie_native] failed to acquire HBM lock (race?)\n");
            return 1;
        }
        printf("[qie_native] acquired HBM lock %s\n", lock_path);
    }

    ominix_qie::ImageDiffusionConfig cfg;
    cfg.num_inference_steps = steps;
    cfg.cfg_scale           = cfg_scale;
    cfg.max_img_seq         = seq_img;
    cfg.max_txt_seq         = seq_txt;
    cfg.use_q4_weights      = use_q4;

    int rc = 0;
    {
        ominix_qie::ImageDiffusionEngine engine;
        auto t0 = std::chrono::steady_clock::now();
        if (!engine.init_from_gguf(gguf_path, cfg, device)) {
            fprintf(stderr, "[qie_native] init_from_gguf failed\n");
            rc = 1;
        } else {
            auto t1 = std::chrono::steady_clock::now();
            double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            printf("[qie_native] init_from_gguf returned ready=%s (%.1f ms). "
                   "Phase 1 scaffold — nothing else to do yet.\n",
                   engine.is_ready() ? "true" : "false", ms);
        }
    }  // engine dtor runs here

    if (use_hbm_lock) {
        release_hbm_lock(lock_path);
        printf("[qie_native] released HBM lock %s\n", lock_path);
    }
    return rc;
}
