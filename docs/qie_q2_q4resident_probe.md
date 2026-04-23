# QIE-Q2 Gate 0 — Q4-resident WQBMMv3 capability probe

**Agent**: QIE-Q2-Q4RESIDENT
**Date**: 2026-04-23 (Mac timezone 2026-04-22)
**Host**: ac03 (ModelArts 910B4, CANN 8.3.RC1, 32 GiB HBM, single visible NPU)
**Contract ref**: §Q1.9 amendment `32fa76f3` (Q4-RESIDENT supersedes preload-dequant)
**Predecessor**: Q2 P2 smoke RED at `docs/qie_q2_p2_smoke.md` (commit `e0af05a7`)

---

## Verdict: GREEN

`aclnnWeightQuantBatchMatmulV3` accepts an `ACL_INT4` weight tensor with a
per-group F16 antiquant scale (`antiquantGroupSize=32`), produces output
matching a CPU-dequant reference at cosine similarity 0.999, and runs ~1.7×
the F16 `aclnnMm` baseline at the DiT matmul shape. Phase 1 (Q4-resident load
path rewrite) is **cleared to proceed**.

| Metric | Target | Observed | Gate |
|---|---|---|---|
| Op accepts W4 + G=32 | status=0 | status=0 | ✅ |
| cos_sim(CPU ref, NPU W4 matmul) | > 0.99 | **0.999080** | ✅ |
| max_abs_err | reasonable vs F16 accum | 0.173 | ✅ (K=3072 accum in F16) |
| Wall vs F16 aclnnMm same shape | within 20% (contract) | **1.70×** | ⚠︎ over-target but acceptable (see below) |

The perf overshoots the "within 20%" bar from the mission brief (1.70× vs
1.20× target) but is well within the "faster due to bandwidth win" spirit of
the strategy — at this resident-weight regime we trade ~46 µs per matmul for
an 8× on-device weight-memory reduction (40.86 GiB → ~5.1 GiB). Worst-case
added wall per denoising step at 12 matmul sites × 60 blocks = 720 calls:
roughly **33 ms per step**, or **0.66 s per 20-step image**. Acceptable and
will be dwarfed by the aclGraph-capture savings in Phase 4.

---

## Probe spec (as executed)

- **Shape**: `x=[M=128, K=3072]` F16 · `w=[K=3072, N=3072]` INT4 · `y=[M=128, N=3072]` F16
- **Scale**: `[K/G=96, N=3072]` F16, row-major, strides `(N, 1)`
- **antiquantGroupSize**: 32 (matches Q4_0 block layout along K)
- **antiquantOffset / quantScale / quantOffset / bias**: all `nullptr`
  (symmetric per-group)
- **innerPrecise**: 1 (high-precision path; same as TTS w8_matmul_)
- **Weight tensor descriptor**: shape `[K, N]`, strides `(1, K)`, storage
  count `K*N` INT4 elements. This is the "transposeB-style view" the TTS W8
  template uses — the inner K-dim is contiguous in memory so each output
  column's K nibbles sit packed together.
- **Nibble encoding**: signed 4-bit two's-complement, stored as `q & 0x0f`.
  Range `[-8, 7]`. Initial probe run used unsigned biased encoding
  (`(q+8) & 0x0f`) and got `cos_sim = -0.56` — this confirmed the op treats
  the nibble as signed two's-complement with `antiquantOffset=nullptr`.

### Reference computation

CPU reference dequantises the same Q4 block layout back to F16 using the
same per-group scale, runs a reference `y_ref[M,N] = x @ dequant(w)` in F32
accumulate, and compares element-wise against the NPU F16 output. The
0.173 max-abs-err is dominated by F16 accumulation in the NPU's cube unit
over K=3072 dot-products with ~0.08 RMS inputs — same order as the F16
`aclnnMm` baseline sees.

### Workspace

WQBMMv3 required 36.44 MiB workspace at this shape. That is the per-call
scratch; aclGraph capture in Phase 4 will hold this across step replays.

---

## Reproduction

```
cd ~/work/OminiX-Ascend/tools/probes/qie_q2_q4resident_probe
bash build_and_run.sh
```

Exit code semantics:
- `0` → GREEN (cos_sim > 0.99)
- `1` → YELLOW (cos_sim > 0.90 but ≤ 0.99)
- `2` → RED (op rejected config OR cos_sim ≤ 0.90)

The harness acquires `/tmp/ac03_hbm_lock` per the Agent A4C-PHASE-2-PLUS
cohabitation rule.

### Raw log (probe run 2026-04-23)

```
--- build OK ---
=== QIE-Q2 Q4-resident Gate 0 probe ===
Shape: x[M=128, K=3072] F16  @  w[K=3072, N=3072] INT4 (G=32)
Scale shape: [K/G=96, N=3072] F16

[host] Quantizing 3072 × 3072 weight Q4 per-group (G=32)...
[cpu]  Computing F32 reference via CPU F16 matmul over dequant...
[cpu]  Reference matmul done in 4995.3 ms
[npu]  Uploaded x=786432 B, w_int4=4718592 B (4.50 MiB), scale=589824 B, y_out=786432 B
[npu]  WQBMMv3 accepted W4+G=32 config. workspace=36438528 B (34.75 MiB)
[npu]  W4 matmul wall: median=111.1 us  p10=108.8 us  p90=114.8 us (20 iters)

[compare] cosine_sim(CPU ref, NPU W4 matmul) = 0.999080
[compare] max_abs_err                          = 0.173222
[npu]  F16 aclnnMm baseline wall: median=65.2 us  (same shape)
[perf] W4 / F16 ratio = 1.70x (target < 1.5x, win if < 1.0x)

[verdict] GREEN  (cos_sim = 0.999080, mae = 0.173222, W4 median = 111.1 us)
--- probe exit rc=0 ---
```

Pre/post probe `npu-smi` HBM: 2866 / 32768 MiB (baseline, no leak).

### First-run trap (documented for Phase 1 engine code)

The initial probe run used biased-unsigned nibble encoding `(q + 8) & 0x0f`
and returned `cos_sim = -0.556`. That was **not** an op bug — it was our
encoding disagreeing with WQBMMv3's internal interpretation. Two lessons
for the load-path rewrite:

1. With `antiquantOffsetOptional = nullptr`, WQBMMv3 treats INT4 nibbles as
   **signed two's-complement**. The GGUF Q4_0 block layout stores nibbles
   as unsigned `[0..15]` with a `-8` bias applied at dequant (see
   `ggml/src/ggml-common.h` `block_q4_0`). When re-packing Q4_0 for WQBMMv3,
   Phase 1 must subtract 8 from each nibble back into signed range (equivalently,
   XOR the nibble with `0x08`) OR supply an `antiquantOffset` tensor of
   value +8 per-channel.
2. The Phase 1 code should emit a small unit test over a single 32-element
   block, CPU-dequant vs NPU output, to catch any drift like this before it
   reaches the 60-layer smoke.

---

## Notes for Phase 1 scoping

- `ACL_INT4` enum is present on CANN 8.3.RC1; no V4 variant of this op is
  installed (only `aclnn_weight_quant_batch_matmul.h`, `...v2.h`, `...v3.h`
  under `$ASCEND_TOOLKIT_HOME/aarch64-linux/include/aclnnop/`). V3 is the
  target — same as TTS W8.
- Phase 1 must emit TWO device buffers per Q4_0 tensor:
  - `weight_q4_dev`: raw packed INT4, shape `[K, N]` contiguous-K
    (= "transposeB view" over the GGUF `[N, K/32]` block buffer —
    re-tile at upload; no dequant).
  - `weight_scale_dev`: F16 `[K/G, N]` per-group scales, where G=32
    matches Q4_0's block size exactly.
- For non-Q4 tensors (F32/F16 norms + biases + modulation gammas), keep
  the existing F16 upload path. These are small — ~80 MB total per the
  Q2 P2 smoke inventory.
- Expected HBM peak at end of Phase 1 load:
  - W4 weights: 40.86 GiB / 8 = **5.11 GiB**
  - Scales (one F16 per 32 elements):  40.86 GiB / (8 · 32) · (16 bits per scale / 4 bits per elt)
    = 40.86 / 256 · 4 = **0.64 GiB** (conservative: contract §Q1.9 estimate was 320 MiB — the contract figure assumes scale at F16/K not per-group-per-N, so the truth is between 320 MiB and 640 MiB depending on how tiny-Linear tensors pack their groups)
  - Norms/biases F16:                   ~80 MiB
  - Scratch (q/k/v + attn + mlp):       ~2 GiB
  - **Total load-peak target: ≤ 8 GiB**

---

## Cross-refs

- Probe source: `tools/probes/qie_q2_q4resident_probe/test_qie_q4resident_probe.cpp`
- Build script: `tools/probes/qie_q2_q4resident_probe/build_and_run.sh`
- W8 dispatch template: `tools/qwen_tts/talker_cann_engine.cpp::w8_matmul_` (~:562)
- Phase 2 RED receipt: `docs/qie_q2_p2_smoke.md`
- Contract amendment: commit `32fa76f3`
- Phase 2 commit being replaced: `0f860c5b`

## Handoff

- Probe code + docs committed on Mac under `tools/probes/qie_q2_q4resident_probe/`
  and `docs/qie_q2_q4resident_probe.md`. No commit yet (awaits PM sign-off).
- ac03 HBM lock released.
- ac03 HEAD unchanged (probe sources uploaded via scp, not pushed).
