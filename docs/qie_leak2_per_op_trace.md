# QIE Leak #2 — Per-Op Trace (Step 1 + 2 + 3 of mission workplan)

Per-op residual-stream + delta-tensor magnitudes for the 60-block QIE DiT
forward, measured on ac02 with `OMINIX_QIE_TRACE_OPS=1`. Goal: identify
which op compounds first to the F16-overflow signature documented in the
Q2.4.4b bisect (`docs/qie_q2_phase4_smoke.md` §4.4b).

## Methodology
- Trace helper: `qie_trace(t, label)` in `tools/ominix_diffusion/src/qwen_image.hpp`
- Each call inserts `cast(F32) → cont → abs → sum` into the graph and
  records `sum_abs` as a graph output. Mean-abs = sum_abs / n_elements.
- Cast-to-F32 first ensures the trace shows the F32-correct magnitude;
  raw on-device storage may differ if the backend silently demotes.
- 13 sites per block × 60 blocks = 780 trace points per build_graph.
- Step-gated: only step 0 traced by default (`OMINIX_QIE_TRACE_STEPS=1`).

## Run configuration
- Shape: 256×256, 20 steps, cfg=1.0, Q4_0 weights
- Backend: ggml-cann on Ascend 910B4 (ac02), with Path C #3 backend fix
  (`3acc62aa`) honoring GGML_PREC_F32 hints.
- Two runs:
  1. Baseline: `OMINIX_QIE_F32_RESIDUAL` unset.
  2. Comparison: `OMINIX_QIE_F32_RESIDUAL=1` (Leak #2 cast inserts active).

## Findings — Step 0 trace, baseline (cast OFF)

```
blk      img_norm1   img_mod1   img_attn   img_resid1   img_norm2   img_mlp     img_resid2
b00      0.655       1.26       7.23       5.3          0.445       2.61e+03    9.61e+04
b01      0.231       5.71       1.66e+03   1.11e+05     0.247       971         1.14e+05
b02      0.243       6.47       1.16e+03   1.18e+05     0.253       553         1.20e+05
b15      0.369       13.7       880        1.86e+05     0.369       284         1.86e+05
b30      0.479       33.9       1.20e+03   2.58e+05     0.478       428         2.50e+05
b45      0.459       38.7       1.31e+03   2.90e+05     0.468       305         2.78e+05
b59      0.465       31.6       642        6.32e+05     0.434       3.29e+04    1.89e+06
```

(Full table: `/tmp/qie_trace_baseline_step0.log`, parsed via
`/tmp/parse_trace.py`.)

### Observations
1. **LayerNorm strips magnitude** (img_norm1, img_norm2 stay 0.2–0.6
   across all 60 blocks). LayerNorm itself is NOT the leak.
2. **Modulation (Flux::modulate) is linear-bounded** — img_mod1 grows
   1.26 → 46 by b50s. Modulation output magnitude tracks the
   modulation parameter scale.
3. **Attention output is bounded** to ~700–2000 across all blocks
   (post-Linear with normalised QK).
4. **Residual streams grow linearly with depth** — img_resid1
   crosses F16 limit (65504) at **block 1** with 1.11e5 mean-abs.
   img_resid2 crosses F16 at **block 0** with 9.61e4 mean-abs.
5. **FFN output (img_mlp) is mostly bounded** to ~200–500 except a
   b00 transient (2614) and b59 spike (3.29e4). The MLP itself is
   NOT the primary leak — its OUTPUT * gate2 adds to the residual,
   and the residual is what compounds.

### Pattern verdict (Pattern A from mission workplan)
> max_abs grows steadily through residual-add → leak is residual stream
> type (already tried, didn't work — so NOT this)

This is the trace pattern. Residual-stream growth IS the leak.

## Findings — Step 0 trace, OMINIX_QIE_F32_RESIDUAL=1

Magnitudes are **byte-identical to baseline** (sub-1% noise per cell).
The cast-at-residual-boundary patch from `qie-leak2-f32-residual` does
NOT reduce the F32-correct magnitudes — it cannot, because the cast
preserves numerical values; it only changes the storage type.

This proves: the trace measures F32-correct magnitudes, and the F32
storage cast does not affect what the model computes. The actual
storage-time saturation (which is what produces NaN at 20 steps per
b708890f) is in a CANN-backend internal accumulator that the
graph-level cast does not influence.

## Next-step proposals

### Proposal 1 — backend op widening (preferred, mirrors native 4.4d)
Promote three op classes to F32 storage in ggml-cann:
- `GGML_OP_ADD` when both inputs are F32-tagged (residual add)
- `GGML_OP_MUL` when both inputs are F32-tagged (gate × delta)
- `GGML_OP_NORM` (LayerNorm) — variance compute in F32, not F16
This is the analog of native engine 4.4d's broader F32 widening.
Implementation: in `ggml-cann/aclnn_ops.cpp`, ensure these ops use
F32 dispatch when GGML_PREC_F32 is hinted; today they may still use
F16 internally for performance.

### Proposal 2 — graph-level F32 storage type tag
Add a `ggml_set_storage_type(t, GGML_TYPE_F32)` API that forces the
allocator to reserve an F32-sized buffer for `t`. Currently
`ggml_cast(t, F32)` produces a CPY op whose output has F32 ne but the
backend may still pick F16 buffer if the next consumer is F16-typed.

### Proposal 3 — vendor escalation
The CANN op spec
(`/usr/local/Ascend/ascend-toolkit/.../aic-ascend910b-ops-info.json`)
documents WQBMMv2 (which v3 dispatches to) as F16/BF16-only output
on 910b. If our residual is fundamentally bottlenecked on the matmul
output dtype, the only path is BF16 conversion (`GGML_CANN_QUANT_BF16=on`
already in flight) — a separate workstream.

## Trace reproduction
```bash
ssh ac02
cd ~/work/OminiX-Ascend-leak2
# Baseline trace (no cast):
OMINIX_QIE_TRACE_OPS=1 ./build-leak2/bin/ominix-diffusion-cli ...
# With Leak #2 cast (no measurable effect):
OMINIX_QIE_TRACE_OPS=1 OMINIX_QIE_F32_RESIDUAL=1 ./build-leak2/bin/...
# Logs: /tmp/qie_trace_baseline_step0.log,
#       /tmp/qie_trace_f32resid_20step_256.log
# Parse: python3 /tmp/parse_trace.py <log>
```
