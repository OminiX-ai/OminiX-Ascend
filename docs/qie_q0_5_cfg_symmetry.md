# Q0.5.1 — CFG cond/uncond shape-symmetry probe (edit mode)

**Agent**: QIE-Q0.5
**Date**: 2026-04-22
**Mode**: source-read only (no NPU execution; Q0 backend still RED so runtime probe is Q1-gated).
**Question**: In QIE edit mode, does `ref_latents` concat apply symmetrically to BOTH cond and uncond forwards, or only cond? Asymmetric → Q4 CFG batching is a 1-week refactor, not drop-in.

## TL;DR

**VERDICT: SYMMETRIC.** `ref_latents` is carried on `DiffusionParams` and passed into **every** `work_diffusion_model->compute(...)` call in the step loop (cond, uncond, img_cond, skip). Token counts for cond and uncond are byte-identical per step. Q4 CFG batching is **drop-in** from the ref-image-shape axis (other risks — workspace doubling and modulation broadcast — remain, but they are batch=2 overhead, not refactor blockers).

## Source evidence

### 1. ref_latents is set once, then shared across all forwards

`stable-diffusion.cpp:2115-2123` — the shared `diffusion_params` bundle is set before any cond/uncond branch:

```cpp
diffusion_params.x                  = noised_input;
diffusion_params.timesteps          = timesteps;
diffusion_params.guidance           = guidance_tensor;
diffusion_params.ref_latents        = ref_latents;            // <-- set once
diffusion_params.increase_ref_index = increase_ref_index;
diffusion_params.controls           = controls;
diffusion_params.control_strength   = control_strength;
diffusion_params.vace_context       = vace_context;
diffusion_params.vace_strength      = vace_strength;
```

### 2. Cond forward uses the shared bundle

`stable-diffusion.cpp:2125-2147`:

```cpp
if (start_merge_step == -1 || step <= start_merge_step) {
    diffusion_params.context  = cond.c_crossattn;   // only swap c_crossattn / y
    diffusion_params.c_concat = cond.c_concat;
    diffusion_params.y        = cond.c_vector;
    ...
}
...
if (!work_diffusion_model->compute(n_threads, diffusion_params, active_output)) { ... }
```

Only `context`, `c_concat`, `y` are swapped between the cond and uncond branches — `ref_latents` stays in place.

### 3. Uncond forward uses the same `diffusion_params`

`stable-diffusion.cpp:2154-2177`:

```cpp
if (has_unconditioned) {
    ...
    diffusion_params.controls = controls;
    diffusion_params.context  = uncond.c_crossattn;
    diffusion_params.c_concat = uncond.c_concat;
    diffusion_params.y        = uncond.c_vector;
    ...
    if (!work_diffusion_model->compute(n_threads, diffusion_params, &out_uncond)) { ... }
}
```

Again only `context` / `c_concat` / `y` change. `diffusion_params.ref_latents` still references the same vector of ref-image latent tensors set on line 2118.

### 4. The QwenImage DiffusionModel impl pipes ref_latents into every call

`diffusion_model.hpp:439-451`:

```cpp
bool compute(int n_threads,
             DiffusionParams diffusion_params,
             struct ggml_tensor** output, ...) override {
    return qwen_image.compute(n_threads,
                              diffusion_params.x,
                              diffusion_params.timesteps,
                              diffusion_params.context,
                              diffusion_params.ref_latents,   // <-- unconditional
                              true,  // increase_ref_index
                              output, ...);
}
```

There is no cond/uncond branch inside `compute`. Both forwards receive the same `ref_latents` vector.

### 5. Concat happens inside `QwenImageModel::forward`, driven only by `ref_latents.size()`

`qwen_image.hpp:454-459`:

```cpp
if (ref_latents.size() > 0) {
    for (ggml_tensor* ref : ref_latents) {
        ref = DiT::pad_and_patchify(ctx, ref, params.patch_size, params.patch_size);
        img = ggml_concat(ctx->ggml_ctx, img, ref, 1);
    }
}
```

No cond/uncond parameter. Concat happens unconditionally on the img token stream every forward call.

### 6. `gen_qwen_image_pe` also receives the same `ref_latents` list

`qwen_image.hpp:544-554` — per-step RoPE positional embedding generation takes `ref_latents` and produces a `pe_vec` whose length is driven by `img_tokens + sum(ref_tokens) + context_len`. Because `ref_latents` is identical across cond/uncond, `pe_vec` is identical too — same `pos_len`, same tensor shape `[2, 2, 64, pos_len]`.

## Shape-symmetry conclusion

Per step, both forwards see:
- identical `img` input token count: `h_len * w_len + Σ(ref_h_len * ref_w_len)`
- identical `pe` layout
- identical `context` token count for the **text-cross path** — this is where cond and uncond legitimately differ (`cond.c_crossattn` vs `uncond.c_crossattn`), but both are produced by the same text encoder on the same tokenizer prefix length, so shape matches. Only the **content** of the embedding differs (prompt vs empty/negative prompt).

The only shape axis that varies in principle is the text-token sequence length (`context->ne[1]`). In practice, the QIE inference pipeline pads both prompts to the same length (see `conditioner.hpp` QwenImageEditPlusPipeline path), so `cond.c_crossattn.ne[1] == uncond.c_crossattn.ne[1]` at runtime.

**Confirm at Q1-green**: a 5-line `LOG_DEBUG("cond ctx=%ld, uncond ctx=%ld", cond.c_crossattn->ne[1], uncond.c_crossattn->ne[1])` instrumentation on ac02 once the backend is unblocked. Cheap sanity check, not a refactor.

## Verdict

**SYMMETRIC** on the ref-image / image-token axis. Q4 CFG batching is drop-in from the shape-compatibility axis:
- Both forwards process identical img+ref token counts.
- Both forwards process the same `pe` tensor (can be shared).
- Text-cross sequence lengths match in practice under the same prompt-padding policy.

The Q4 batching plumbing is what you'd expect for any DiT — extend batch dimension in block matmuls, broadcast timestep embedding, and linear-combine the two output halves in the CFG epilogue. Estimate: **3-5 engineer-days** (contract Q4 bucket is 1 week), no asymmetry-refactor tax.

## Residual risks (pre-Q1 runtime cannot close these)

1. **Text-cross padding asymmetry.** If the prompt tokenizer path ever yields different padded lengths for cond vs uncond (e.g. uncond = empty → shorter), Q4 must pad to common length. 5 lines of host code; not a refactor.
2. **Modulation tensor broadcast.** Timestep embedding produced by `time_text_embed->forward(ctx, timestep)` is shape `[batch, embed]`. At batch=1 it's the same for cond and uncond; at batch=2 it needs to be duplicated or per-batch. Per-block modulation indices (`modulate_index`, zero_cond_t path at line 564-586) also need batch-expansion. Standard DiT batching plumbing.
3. **Workspace doubling.** At batch=2, activation HBM roughly doubles per step. With Q4 weights at 18-20 GB on 910B4 (32 GB HBM), the 12 GB margin absorbs this — but verify with `aclGetUsedMem` at Q4 landing.
4. **`increase_ref_index=true` hardcoded at `diffusion_model.hpp:448`.** Cond and uncond both get `increase_ref_index=true`, so ref positional indices are byte-identical across the two calls. Good.

## Recommendation for PM

- Mark Q0.5.1 **GREEN** on the shape axis. Q4 scope stays at **1 week (3-5 engineer-days core + integration overhead)**, not widened.
- Runtime-confirm the text-cross padding line once at Q1-green via `LOG_DEBUG` — 10-minute task, no agent dispatch needed.
- Keep the residual risks on Q4's checklist but do not gate dispatch on them.
