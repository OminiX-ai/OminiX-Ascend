# FIA V3/V4 Probe Verdict

**Agent**: FIA-V34-probe
**Host**: ac01 (CANN 8.3.RC1)
**Date**: 2026-04-21
**Scope**: Standalone probe only (no engine wiring). Shape = our prod CP attn: Q `[1,1,16,128]`, K/V `[1,seq_len,8,128]`, F16, BSND, GQA 2:1.

## Header signature diff (V2 → V3 → V4 → VX)

All variants ship the same `GetWorkspaceSize` + `Launch` two-phase contract. Diffs are **strictly additive optional tensors + two int64 knobs**. Nothing deprecated, no layout-enum changes, no MoE/`expertTokens` gating, no `@note` dtype matrices in-header.

| Group | V2 | V3 adds | V4 adds (on top of V3) | VX adds (on top of V4) |
|---|---|---|---|---|
| Core Q/K/V + mask + seqlens | identical | — | — | — |
| Quant scales (deq/quant, antiq) | identical | — | — | — |
| `blockTable`, `query/kvPaddingSize` | identical | — | — | — |
| Shared-prefix K/V + len | identical | — | — | — |
| **MLA / decoupled RoPE** | absent | `queryRopeOptional`, `keyRopeOptional`, `keyRopeAntiquantScaleOptional` (3 optional tensors, pre-`numHeads`) | same | same |
| **Per-call Q dequant + softmax sink** | absent | absent | `dequantScaleQueryOptional`, `learnableSinkOptional` (2 optional tensors) | same |
| **Start-idx / pse-type** | absent | absent | absent | `qStartIdxOptional`, `kvStartIdxOptional`, `pseType` |
| New int64 knobs | `numHeads, scaleValue, preTokens, nextTokens, inputLayout, numKeyValueHeads, sparseMode, innerPrecise, blockSize, antiquantMode, softmaxLseFlag, key/valueAntiquantMode` | — | `+ queryQuantMode` | `+ pseType` |

Our production V2 call passes `nullptr` for every optional tensor already, so the extra V3/V4 slots degenerate cleanly by passing `nullptr` + `0` for the new scalar.

## Standalone correctness + wall

Build: `g++ -std=c++17 -O2 -lascendcl -lopapi -lnnopbase`. 5 warmup + 50 timed iter, median μs, fresh executor per iter (CANN one-shot semantics). Harness: `/tmp/fia_v34_probe/test_fia_v34.cpp`.

### pos=5 (seq_len=6)

| Variant | max_abs_diff vs V2 | max_rel_diff | Wall μs (median/50) | Verdict |
|---|---|---|---|---|
| V2 (ref) | 0.000000 | 0.000000 | 76.24 | baseline |
| V3 | 0.000000 | 0.000000 | 75.95 (1.00×) | **GREEN** |
| V4 | 0.000000 | 0.000000 | 76.01 (1.00×) | **GREEN** |

### pos=15 (seq_len=16, max pos-keyed aclGraph cache slot)

| Variant | max_abs_diff vs V2 | max_rel_diff | Wall μs (median/50) | Verdict |
|---|---|---|---|---|
| V2 (ref) | 0.000000 | 0.000000 | 76.32 | baseline |
| V3 | 0.000000 | 0.000000 | 76.48 (1.00×) | **GREEN** |
| V4 | 0.000000 | 0.000000 | 76.47 (1.00×) | **GREEN** |

**Byte-identical** F16 output across all 2048 elements at both positions. `first_div_idx = -1` (no differing element). Wall time within sub-percent noise.

## Ship candidates

Both V3 and V4 are wireable as drop-in replacements for FIAv2 on our shape. Required delta vs current call site (`cp_cann_engine.cpp:2943`):

- **V3**: insert `nullptr, nullptr, nullptr` before `numHeads` (for `queryRope`, `keyRope`, `keyRopeAntiquantScale`). Change `aclnnFusedInferAttentionScoreV2*` symbol → `...V3*`. No new pointer in `g_cann` dispatch table beyond symbol load.
- **V4**: V3 delta + `nullptr, nullptr` for `dequantScaleQuery`/`learnableSink`, + `(int64_t)0` for `queryQuantMode` after `valueAntiquantMode`. Same symbol-load pattern.
- No new headers needed for the numeric path — current op-dispatch loader just needs the new `dlsym` entry.
- No dtype, layout, or seq_len constraints violated in either variant at our shape.

## Recommendation for PM

**No wiring probe justified right now on pure-prod-shape grounds.** V3 and V4 are numerically identical to V2 (byte-for-byte, not just within-ulp) and perform within measurement noise at both decode positions we care about. There is no latency or correctness upside to a migration for the current CP attn workload.

**Where V3/V4 become interesting is feature-gated, not perf-gated:**
- **V3** unlocks MLA / decoupled-RoPE attention via `queryRope`/`keyRope` tensors. If we later explore DeepSeek-style models or want MLA on Qwen variants, V3 is the entry point — and this probe confirms V3 is a superset of V2 at zero cost to existing callers.
- **V4** adds `dequantScaleQuery` + `learnableSink` + `queryQuantMode`. The learnable sink is the GPT-OSS / Qwen3-Next softmax-sink primitive; `queryQuantMode` gives us per-token Q-quant for A16W8-pseudo paths. Relevant if we pursue full Q-quant decode (a16w8 Q branch) or adopt sink-attention architectures.
- **VX** further adds `qStartIdx`/`kvStartIdx`/`pseType` — speculative-decode / chunked-prefill surface. Flag in future brief if we pursue spec-decode.

**Deck flag**: we are not trapped on V2. The ACL op family is forward-compatible and our V2 call pattern lifts unchanged to V3/V4. Migration is a one-line risk whenever a new feature requires it; no preemptive move needed. This directly de-risks any future feature-request ask that depends on MLA or sink attention — we already know the runtime accepts it on our hardware at CANN 8.3.RC1.
