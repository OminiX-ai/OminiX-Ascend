# QIE-Q2 Phase 2 — ac03 smoke receipt

**Author**: Agent AC03-HARVESTER
**Date**: 2026-04-23
**Host**: ac03 (modelarts 910B4, CANN 8.3.RC1, 32 GiB HBM, single NPU)
**Scope**: First-run ac03 load of the Phase 2 scaffold (commit `0f860c5b`
`feat(qwen_image_edit): Q2.2 GGUF parse + preload-dequant weight upload
+ 3D-axial RoPE precompute`) against the real Q4_0 GGUF.

---

## Build verdict: GREEN

Clean rebuild with `cmake --build build-w1 --target qwen_image_edit_native
--clean-first` on ac03 returns exit 0 with no QIE-side warnings. The
`__fp16` cast in `fp32_to_fp16()` compiles fine under the CMakeLists'
`-march=armv8.2-a+fp16` guard — no `ggml_fp32_to_fp16` helper fallback
needed. `qwen_image_edit_native` binary: 77 KiB, dynamically linked
against `libggml-cann.so.0`, `libascendcl.so`, and the usual
toolkit-lib chain.

No patch required. The build-fix branch anticipated in the mission
brief is moot.

---

## Smoke load verdict: **RED**

### Environment

- Binary: `build-w1/bin/qwen_image_edit_native` (2026-04-23 build).
- GGUF: `/home/ma-user/work/qie_weights/Qwen-Image-Edit-2509-Q4_0.gguf`,
  11.93 GB, md5 `e6eea975692f73d992c2b016dfa78beb` (scp'd from ac02
  `/home/ma-user/qie_q0v2/weights/…`; md5-verified).
- Invocation: `./qwen_image_edit_native --gguf … --device 0`
  (with and without `GGML_CANN_QUANT_BF16=on`; same failure).
- HBM baseline pre-run: 2864 / 32768 MiB (driver + idle).
- Free HBM budget: ~29.9 GiB.

### Observed failure

```
[qie_native] aclrtMalloc(18874368) failed err=207001
[qie_native] init_from_gguf FAILED partway through weight upload; see log above
[qie_native] init_from_gguf failed
```

The engine logs only the *failing* malloc. `npu-smi` during the run
showed HBM climbing to **29214 / 32768 MiB** (≈26.3 GiB in engine
allocations, plus driver baseline) before the 18 MiB failing call.
So this is **not a first-alloc device-handle issue** — preceding
uploads succeeded. The engine is running out of HBM during bulk
weight upload.

### Receipts table (expected vs actual)

| Metric | Contract expected | Observed | Delta |
|---|---|---|---|
| Tensor count | ~60 blocks × ~15 = ~900 + 13 globals | _never printed (OOM)_ | — |
| F16 weights bytes | ~13 GiB | ~**26.3 GiB allocated before OOM; full load would be 40.86 GiB** | +210% |
| Scratch bytes | ~2 GiB | _never reached_ | — |
| RoPE pe bytes | ~2 MiB | _never reached_ | — |
| Peak init HBM | ~15–17 GiB | **>29.9 GiB (HBM exhausted mid-upload)** | +100% |
| Dequant wall | < 120 s | _never reached_ | — |
| Total init | < 180 s | _never reached_ | — |

### Root cause: contract F16 budget is wrong for this GGUF

Independent GGUF inspection (`gguf-py.GGUFReader`, see below) shows
`Qwen-Image-Edit-2509-Q4_0.gguf` contains **1933 tensors totalling
40.86 GB F16-equiv** once dequantised, not ~13 GiB. Breakdown:

| Pattern | Copies | F16 total |
|---|---|---|
| `transformer_blocks.X.img_mod.1.weight` | 60 | 6.80 GB |
| `transformer_blocks.X.txt_mod.1.weight` | 60 | 6.80 GB |
| `transformer_blocks.X.img_mlp.net.0.proj.weight` | 60 | 4.53 GB |
| `transformer_blocks.X.img_mlp.net.2.weight` | 60 | 4.53 GB |
| `transformer_blocks.X.txt_mlp.net.0.proj.weight` | 60 | 4.53 GB |
| `transformer_blocks.X.txt_mlp.net.2.weight` | 60 | 4.53 GB |
| `transformer_blocks.X.attn.*_proj.weight` (8 per block) | 60×8 | 9.06 GB |
| (biases + norms + globals + etc.) | — | 0.08 GB |
| **Total** | **1933** | **40.86 GB** |

The contract amendment's "~13 GiB F16" assumed attention projections
only (~9 GB) plus the small globals, omitting the separate `img_mod`,
`txt_mod`, `img_mlp`, and `txt_mlp` Linear stacks. The actual DiT
published in Qwen-Image-Edit-2509 fully dequantised does NOT fit a
single 32 GiB 910B4.

No missing-tensor list can be produced because the engine aborts
before completing the inventory sweep.

---

## Decision surface (not in AC03-HARVESTER scope)

Four credible recoveries for the next QIE agent to evaluate. Listed in
ROI order per the contract's "preload-dequant NaN-safe" primary
constraint:

1. **Stay Q4 resident**: abandon preload-to-F16, upload Q4_0 tensors
   directly to NPU and dequant-on-use at step boundary (original
   ggml-cann path). ~11.9 GiB resident, fits. Re-opens the Q1-baseline
   NaN issue Q2 was built to sidestep; may require the Q1 gather_v3
   probe verdict (`docs/qwen_image_edit/q1_baseline/*`) to be re-read.
2. **Dequant a subset to F16**: keep `img_mlp` + `txt_mlp` (18 GB of
   the 40 GB) in Q4_0, dequant-on-use — only attn projections and
   modulation Linears go to F16. Target footprint ≈ 22 GB F16 resident,
   fits. Cost: the engine code must grow a tensor-class tag + a
   dequant-on-use path for the "kept-Q4" slots.
3. **A16W8 in-HBM**: static symmetric-INT8 quant of the big Linears
   as per the CP-side W8 playbook (A4/A4b), halving F16 footprint.
   ~20 GB resident, fits. New calibration pass required.
4. **Two-NPU split**: sharded load across 2×910B4. Ac03 has only 1
   NPU visible inside the container (`/dev/davinci7`, logical device 0,
   `NPU-VISIBLE-DEVICES=7`), so this is not actionable on ac03 —
   requires a different host.

A4c's "Phase 3 = fill forward_block_ dispatch" is orthogonal to this
HBM-budget miss — the scaffold's Phases 3-5 can still land on a
smaller synthetic / subset GGUF, but the real-weights HBM gate does
not pass without one of (1)-(4).

---

## Final verdict: RED

Build GREEN, smoke RED. Phase 2 cannot complete its stated receipts on
ac03 against the published `Qwen-Image-Edit-2509-Q4_0.gguf` because
the HBM budget assumed in the contract amendment (§Q2, ~13 GiB F16) is
roughly 3× too low for the real tensor count.

No code fix is applied — this is not a build bug. The fix is a contract
update + engine-design choice per Decision Surface above, which is
above AC03-HARVESTER's scope.

Artifacts:
- `docs/qie_q2_p2_smoke.md` (this file)
- `/tmp/qie_q2_p2_smoke.log` on Mac (3-line failure log)
- No `/tmp/qie_q2_p2_buildfix.patch` — build did not need fixing.

## State hand-off

- ac03 HBM lock released (`/tmp/ac03_hbm_lock` absent).
- A4c Phase 1 working-tree delta stashed:
  `stash@{0}: On main: A4c Phase 1 patch (RED verdict; unstash if
  resuming Phase 1+ work)`.
- ac03 HEAD = `0f860c5b` (fork main, unchanged).
