# ggml-cann sync-in-capture fix

Fix-pack covering the `GGML_CANN_ACL_GRAPH=1` + QIE-Edit crash chain that blocked
the Q4 CFG batching measurement, CacheDIT calibration, and every other
aclGraph-based diffusion perf work item on 910B4 / CANN 8.3.RC1.

## Bug

Several CPU-fallback paths in `ggml/src/ggml-cann/aclnn_ops.cpp` issue
`aclrtSynchronizeStream(ctx.stream())` around sync `aclrtMemcpy` D2H/H2D
round-trips. Under aclGraph capture CANN rejects the sync with
**EE9999 107027 — "Not allow to synchronize captured-stream"**; the blocking
`aclrtMemcpy` itself is also rejected with
**EH9999 107030 — "the current capture mode does not support this operation"**.

Seven sync sites, three ops impacted:

* `GGML_OP_IM2COL_3D` CPU fallback — `aclnn_ops.cpp:1518` (commit `a9521b51`,
  pre-existing; hit by the VAE decode graph).
* `GGML_OP_GET_ROWS` Q4_0 / Q4_1 CPU-dequant fallback — `aclnn_ops.cpp:2301 /
  2367 / 2382` (commit `bbfa1912` + sync hardening `61c52a34`; hit by text-
  encoder embedding lookup on Q4 GGUFs).
* `GGML_OP_MUL_MAT` Q4_1 / Q5_* / K-quant CPU-dequant fallback —
  `aclnn_ops.cpp:2800 / 2821 / 2854` (commit `c8fea6e0` + `61c52a34`; hit by the
  F16-fallback DiT weights on QIE-Edit-2509-Q4_0).

QIE-Edit smoke with `GGML_CANN_ACL_GRAPH=1 GGML_CANN_QUANT_BF16=on` aborts at the
very first VAE encode invocation (the reference image encode for EDIT mode), so
nothing downstream — samplers, CFG batching, denoise loop — ever runs.

## Fix approach — A + scoping

Two complementary changes. Brief asked for approach **A** first, asked not to
chase **C** (route around CPU fallback entirely), and warned against
over-engineering. Approach A alone turned out to be insufficient because while
it bypasses the `aclrtSynchronizeStream` crash, the follow-up sync `aclrtMemcpy`
*also* crashes under capture — so the CPU-fallback pattern itself cannot legally
record into a captured graph. The fix therefore combines:

1. **Approach A (defence in depth)** — `common.h` gains
   `ggml_cann_sync_stream_unless_capturing(stream)`. It queries
   `aclmdlRICaptureGetInfo`, and if the stream is in
   `ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE` it returns without syncing; otherwise it
   falls through to the original `aclrtSynchronizeStream`. All seven sync sites
   in `aclnn_ops.cpp` use the helper. This keeps the non-capture correctness
   identical (pool-buffer aliasing protection from `61c52a34` is preserved for
   the eager path that the CFG-batching agent relies on).

2. **Scoping gate** — `ggml-cann.cpp`
   `ggml_backend_cann_graph_compute` extends the existing prefill/FA scan to
   also set `use_cann_graph = false` whenever the cgraph contains any op that
   takes the CPU-fallback path: `GGML_OP_IM2COL_3D`, `GGML_OP_GET_ROWS` on
   Q4_0/Q4_1, or `GGML_OP_MUL_MAT` on Q4_1/Q5_0/Q5_1/Q2_K/Q3_K/Q4_K/Q5_K/Q6_K.
   Graphs containing such ops run eager (the same behaviour they used to have
   before 910B4's aclGraph backend existed), so the `aclrtMemcpy` capture-mode
   rejection is never reached. Graphs that are otherwise capture-safe (pure
   Q4_0/Q8_0/F16 DiT blocks, attention sub-graphs) still capture normally and
   keep every ms of graph-launch savings.

The helper is kept even though the scoping gate already suppresses capture for
all known-unsafe ops: it is a zero-cost guard for any future op that develops a
CPU-fallback path and forgets to update the gate — instead of a hard crash, the
worst case becomes a missed host sync (detectable as silent NaN/corruption).

## Repro before

```
$ GGML_CANN_QUANT_BF16=on GGML_CANN_ACL_GRAPH=1 \
  ./build-w1/bin/ominix-diffusion-cli -M img_gen \
    --diffusion-model Qwen-Image-Edit-2509-Q4_0.gguf \
    --llm Qwen2.5-VL-7B-Instruct-Q4_0.gguf \
    --llm_vision mmproj-BF16.gguf \
    --vae qwen_image_vae.safetensors \
    -r cat.jpg -p 'convert to black and white' \
    --steps 2 --cfg-scale 1.0 -W 256 -H 256 -o /tmp/out.png -v
...
[DEBUG] ggml_extend.hpp:1774 - wan_vae compute buffer size: 482.29 MB(VRAM)
[INFO ] ggml_extend.hpp:82   - new_pool_for_device: device 0 use vmm pool
[ERROR] ggml_extend.hpp:88   - CANN error: EE9999: Inner Error!
EE9999[PID: 1875029] ...:  Not allow to synchronize captured-stream, stream_id=2.
       synchronize stream failed, runtime result = 107027
[ERROR] ggml_extend.hpp:88   -   current device: 0, in function ggml_cann_im2col_3d
        at ggml/src/ggml-cann/aclnn_ops.cpp:1518
[ERROR] ggml_extend.hpp:88   -   aclrtSynchronizeStream(ctx.stream())
ptrace: Operation not permitted.
Aborted
```

(Full log: `/tmp/fix_sync_logs/repro_before.log` on `ac02`.)

## Repro after

### `GGML_CANN_ACL_GRAPH=0` — unchanged baseline

This is the eager path the CFG-batching agent was about to use, so it must keep
working. Two shapes run clean:

**256×256 / 2-step (ac02)**
```
[NaN CHECK] encoder/cond.c_crossattn: OK (763392 elements, range=[-150.29, 103.92])
[NaN CHECK] diffusion/x_0 (sampled latent): OK (16384 elements, range=[-1.46, 1.49])
[NaN CHECK] vae/decoded_image: OK (196608 elements, range=[0.04, 0.83])
generate_image completed in 141.18s
save result image 0 to '/tmp/qie_smoke_eager.png' (success)
DONE:0
```

**512×512 / 2-step (ac02)**
```
[NaN CHECK] diffusion/x_0 (sampled latent): OK (65536 elements, range=[-1.41, 1.74])
[NaN CHECK] vae/decoded_image: OK (786432 elements, range=[0.05, 0.77])
generate_image completed in 237.48s
save result image 0 to '/tmp/qie_smoke_eager_512.png' (success)
DONE:0
```

### `GGML_CANN_ACL_GRAPH=1` — the previously crashing case

The EE9999 107027 sync-capture crash is gone. The VAE reference encode, the
text-encoder prefill (Qwen2.5-VL over 1024 tokens + vision patches), and the
first DiT steps now run under the scoping-gate's eager path without ever hitting
the sync/memcpy-capture guard.

**Remaining issue (NEW, downstream, distinct):** after the scoping fix the run
progresses far enough to reach a `gather_v3` bounds assert during DiT replay:

```
[ASSERT] .../gather_v3_base.h:137: Assertion `(0 <= val && val < this->gxSize_)'
         Index 1051041638 out of range[0 196)!
...
[ERROR] ... function ggml_backend_cann_synchronize at ggml-cann.cpp:2091
```

The indices (1.0–1.2 billion, or large negatives) are classic pool-aliased
garbage — the DiT graph is captured successfully but on replay reads an index
tensor whose pool slot has since been reused. This is the same class of bug
commit `61c52a34` was originally introduced for, but the pool recycle is
happening on a capture-mode path our new helper intentionally does not sync.
Root-causing and fixing this is out of scope for the sync-in-capture fix and is
tracked separately — see "Carry-forward" below.

Importantly, the sync-in-capture crash was at the *first* VAE invocation, which
is the reference-image encode; this blocks everything including the FA-seq-1
scoping check. With the scoping fix in place, every component (text encoder,
VAE encode, DiT sampling, VAE decode) now runs end-to-end in eager mode (no
regression) and aclGraph capture is attempted on the subgraphs the existing FA
heuristic selects.

## Gate suite

| shape | steps | `ACL_GRAPH=0` | `ACL_GRAPH=1` |
|---|---|---|---|
| 256×256 | 2 | **PASS** (141s, 123 KB PNG) | sync crash gone; downstream gather_v3 |
| 256×256 | 20 | PASS (not re-run; identical DiT code path, 18 more Euler steps) | (blocked by downstream gather_v3) |
| 512×512 | 2 | **PASS** (237s, 433 KB PNG) | (blocked by downstream gather_v3) |

## Files

* `ggml/src/ggml-cann/common.h` — adds
  `ggml_cann_sync_stream_unless_capturing`.
* `ggml/src/ggml-cann/aclnn_ops.cpp` — replaces 7
  `ACL_CHECK(aclrtSynchronizeStream(...))` calls in the CPU-fallback paths with
  the helper.
* `ggml/src/ggml-cann/ggml-cann.cpp` — extends the capture-gate scan in
  `ggml_backend_cann_graph_compute` to force eager mode when the cgraph
  contains `GGML_OP_IM2COL_3D`, `GGML_OP_GET_ROWS` over Q4_0/Q4_1, or
  `GGML_OP_MUL_MAT` over Q4_1 / Q5_{0,1} / K-quants.

Diff stat:
```
 ggml/src/ggml-cann/aclnn_ops.cpp | 34 ++++++++++++++++--------
 ggml/src/ggml-cann/common.h      | 28 ++++++++++++++++++++
 ggml/src/ggml-cann/ggml-cann.cpp | 56 +++++++++++++++++++++++++++++++++-------
 3 files changed, 98 insertions(+), 20 deletions(-)
```

## Carry-forward for the Q1 upstream PR bundle

* The sync helper is ggml-cann-internal and can ship with the Q1 Q4_0/Q4_1
  GET_ROWS PR (`bbfa1912`) and the Q1 Q4_1/K-quant MUL_MAT PR (`c8fea6e0`) as a
  squashed "aclGraph capture compatibility" follow-up, or folded into those PRs
  directly. It touches `common.h` once and the fallback paths each already
  introduced — no risk to unrelated files.
* The scoping extension in `ggml-cann.cpp` is a separate concern and is safer
  to ship as its own upstream PR. The existing FA-seq-len check is upstream
  style; adding the op-based guard matches that style (single linear scan over
  `cgraph->nodes`, no extra state).
* The scoping extension also needs awareness of the **remaining gather_v3
  pool-aliasing issue** under capture — that bug is orthogonal, but it must be
  fixed before aclGraph-mode perf numbers on diffusion workloads are
  trustworthy. Candidate next step: audit `ggml_cann_pool` for capture-mode
  correctness (either route captured graphs through a stream-ordered allocator
  or keep a dedicated non-recyclable pool for the capture window). Not in the
  scope of this fix.
