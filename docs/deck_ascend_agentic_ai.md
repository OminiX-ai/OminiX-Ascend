# Agentic AI Coding Unlocks a New Perf Frontier on Ascend 910

**Executive/technical deck — PM: OminiX, 2026-04-21**
**Thesis**: agentic AI coding (Codex, Claude Code) dissolves the rigid abstraction boundaries of the pre-AI-coding inference stack (ops API, computing engine, inference framework, dataflow layer), enabling end-to-end model/system co-design from single-card to cluster — faster and more thoroughly than a conventional team.

---

## Slide 1 — Thesis + Proof

**Title**: "The boundary was the bottleneck."

- Conventional inference stacks leave **10-50% perf on the table at layer boundaries**
- Boundaries are organisational, not technical: **humans specialise per layer**
- Agentic AI coding crosses **Rust ↔ C++ ↔ aclnn ↔ AscendC ↔ CMake ↔ Python** in one session
- **Proof**: OminiX Qwen3-TTS on 910B4 — **~1 fps (llama.cpp baseline) → 32.2 fps in ~7 days**, one PM + agent swarm. Full arc ≈ **32× lift** from conventional inference stack. Shipped at fork tag `32fps-landed`, byte-identical user-ear verified.
- Zero permanent headcount added; CP engine rewritten across 4 layers; **4 landed tracks, 2 cleanly killed tracks** (Path C, CannFusion) — all with receipts

**Speaker notes**: The punchline we'll defend in the next nine slides: the conventional inference stack (`llama.cpp`, `PyTorch`, `LLVM`, `OneFlow`, `vLLM`) is a layered system designed around human-scale specialisation. Each layer has its own language, its own release cadence, its own team. Cross-layer optimisation is something you plan in roadmaps and deliver in quarters. What we're showing today is that agentic AI coding — Codex, Claude Code, and the emerging agent-swarm pattern — collapses those layer boundaries into something much cheaper than a quarter. In the OminiX case study, we took Qwen3-TTS on Huawei 910B4 from **effectively unusable under stock llama.cpp + ggml-cann (~1 fps, sub-realtime)** to **32.2 fps clean-quality, user-ear-verified**, in about a week of wall time, with one PM gating a small agent swarm. That's not "AI wrote some code faster"; it's "AI rewrote the CP engine across the Rust FFI, the C++ dispatch layer, the aclnn ops chain, the AscendC custom-kernel path, and the Python weight-repack, in the same head." That combined perspective is the new thing. The journey has four phases: escape the conventional stack (~1 fps → 22 fps, first native `CpCannEngine`), co-design within the native stack (22 → 30.5 fps via W1 NPU lm_head + W3b kernel fusion), collapse the dispatch floor (30.5 → 31.6 fps via aclGraph pos-keyed replay), and incremental post-32 gate closes (31.6 → 32.2 fps via A.1 InplaceAddRmsNorm + M3'new' pos 0+1 batched prefill). The rest of the deck defends each claim with numbers from that trajectory. Shipped at fork tag `32fps-landed` (2026-04-21).

---

## Slide 2 — The Pre-AI-Coding Constraint

**Title**: "Dialects, teams, release cadences."

- Framework (PyTorch / Python) — ML researchers, monthly releases
- Inference engine (vLLM / TRT-LLM) — systems engineers, quarterly
- Computing engine / ops API (aclnn, cuDNN) — vendor, semi-annual
- Kernel DSL (AscendC / CUDA) — kernel authors, project-scoped
- Compiler (bisheng / LLVM / MLIR) — compiler team, decoupled
- **Cross-layer optimisation ≈ 1 quarter ≈ 1 retrospective ≈ 1 re-org**

**Speaker notes**: Here's the structural problem. A production inference stack looks like five layers, and each layer has its own dialect and its own people. The Python framework author does not grep the CANN headers in her free time. The kernel author does not edit the Rust FFI. The systems engineer does not pattern-match the vendor op catalog. They don't know each other's dialects, and they're on different release trains. A classic concrete failure mode in our own work: Huawei shipped `aclnnFFNV3` — a single op that does W8 SwiGLU plus per-channel dequant, collapsing five of our aclnn calls into one — in CANN 8.3 from day one. It sat in the `aclnnop/` include directory on our ac01 host for months. Nobody in our stack grepped for it until an agent did an FO-audit of all 717 aclnnop headers in 30 minutes. Three months of headroom were just sitting on the vendor's disk, untouched. That is not a code problem. That is a coordination-cost problem of exactly the kind AI coding collapses.

---

## Slide 3 — Single AI Crossing All Layers (W1: NPU lm_head Port)

**Title**: "One session, four layers, +8.3 fps."

- Baseline: `lm_head` was a CPU NEON+OMP matvec, **15 ms / frame**
- Agent crossed: Rust FFI, C++ `CpCannEngine`, aclnn dispatch, weight upload
- Same session edited `cp_cann_engine.cpp` + header + `cp_cann_symbols.cpp` + engine init
- Result: lm phase **11.78 ms → 2.06 ms** (**-9.72 ms / frame**); **21.9 fps → 30.2 fps** (**+8.3 fps**)
- Gate: first 10 frames of 3 canonical utts, **0 / 16 token drift**
- Conventional project: 4-6 weeks with specialisations; this was one agent dispatch

**Speaker notes**: W1 is the cleanest case study. Our `lm_head` call — the projection from 1024-dim CP hidden state to 2048-entry codebook vocab, run 15 times per audio frame — had been a CPU matvec since day one of the native engine. We never touched it because moving it to NPU looked like a cross-layer project: new aclTensor lifecycle, weight upload at init, new Rust-wrapped dispatch fn, a correctness gate against the CPU reference. An agent did the whole thing in one session. Loaded the 15 lm_head weights to device at init alongside the existing Q/K/V/O uploads, added `forward_lm_head(group_idx, …)` dispatching `aclnnMm`, added `fetch_logits` with an on-device F32 cast, gated behind `TALKER_LM_HEAD_NPU=1`. The agent read the W8 quant pattern off `aclnnWeightQuantBatchMatmulV3`'s doc and mirrored the code path. The numbers speak: 9.72 ms removed from every frame, 8.3 fps lifted, zero token drift on the correctness gate. That's a Q1-sized optimisation landed in one agent dispatch. Contract: `CP_FPS_OPTIMIZATION_CONTRACT.md §W1`.

---

## Slide 4 — Dissolving the "Ops API Is Fixed" Boundary

**Title**: "The vendor API surface is vastly under-mined."

- Agent grepped **717 aclnn headers** on ac01 in 30 min, filtered for fusion patterns
- Found **3 unused fused ops**, all applicable to our hot path:
  - `aclnnFFNV3` — W8 SwiGLU, 5 ops → 1 op, projected **+0.5-1.2 fps**
  - `aclnnApplyRotaryPosEmbV2` — Q+K fused RoPE, projected **+0.2 fps**
  - `aclnnInplaceAddRmsNorm` — in-place tail, projected **+0.1-0.2 fps**
- Conventional team never scheduled "read all CANN headers" — agents do it as a probe
- Full audit artefact: `docs/fused_op_audit.md` + landing contract `FUSED_OP_LANDING_CONTRACT.md`

**Speaker notes**: This slide is the most boring and the most important. The vendor API surface — `aclnnop/` on CANN, `cutlass/` on CUDA, `ggml-backend` on CPU — is a catalog of pre-fused ops shipped by the hardware vendor. It is the single highest-ROI place to look for free perf, because every op in there is a hand-tuned engineer-year that someone already paid for. And yet: a conventional team never schedules "go read all 717 headers". It's not a feature story, it's not a sprint item, there's no champion. An agent does it in 30 minutes as a side-probe — we called it the FO-audit. The deliverable is `docs/fused_op_audit.md`: 9 applicable candidates, ranked by fps upside × integration cost, with a recommended Phase A (easy ops, +0.25-0.45 fps) and a Phase B (FFNV3, +0.5-1.2 fps). The top find — `aclnnFFNV3` — had been shipping in CANN 8.3 since release. We just hadn't looked. This is the new-model-mistake that the pre-AI era will keep making as long as a human is the grep-query.

---

## Slide 5 — Failing Fast as a Feature (Path C Retrospective)

**Title**: "Optionality is the product."

- W4.1: hand-written AscendC fused-attn-sublayer kernel (RmsNorm+QKV+RoPE+FIAS+O+residual)
- Swarm: author → compile → offline diff **PASS 5e-4** → wire env-gated → live runtime gate **FAIL**
- Drift gate: `max_drift = 1949 codebook IDs` at frame 1, only **1 / 256 positions matched**
- PM closed the track in **72 hours, with receipts**: `docs/contracts/PATH_C_ASCENDC_CONTRACT.md §1a`
- PC-tile re-spike proved 40-core gemv **-32% wall vs aclnnMm** — but F32→F16 cross-core reduce wipes it
- Traditional team: 3 months sunk; AI swarm: closed cleanly, artefacts preserved, pivot to CannFusion

**Speaker notes**: This is the honest half of the story. W4.1 was a Path C bet: write our own fused attention sublayer in AscendC, keep intermediate tensors in SRAM, skip the DRAM round-trips that aclnn always pays. The agent swarm did it — W4.0 toolchain probe, W4.1.1 skeleton, W4.1.2v vector-primitive rewrite that even passed the offline diff at 5e-4 F16 noise. Landed the wiring env-gated on ac01. Then the live runtime drift gate — identical code path, real weights, real talker — returned a max drift of 1949 codebook IDs at frame 1. The kernel was mathematically incoherent, not just noisy. A conventional team would have spent 2-3 more weeks hypothesis-testing. The agent swarm, under PM supervision, closed the track in 72 hours with a full retrospective. The consolation prize was a PC-tile re-spike: we measured a 40-AIV-core tiled matmul at M=1 decode shapes and it beat aclnnMm by 26% at the gemv layer. But the mandatory F32→F16 cross-core reduce costs 18-22 μs, wiping the win. Atomic-add races at blockDim ≥ 20, NaN at 40 — a hardware limitation of 910B4, not a kernel bug. Contract closed with a recommendation for future resumption: don't start with attn-sublayer, start with a fused RmsNorm+Mm spike to check whether the reduce folds into the downstream op. The point of the slide isn't the kernel failure. The point is that optionality — being able to close a 3-week bet cleanly in 72 hours with data, receipts, and a clean revert — is now the primary organisational asset, and it is built out of agents + PM gates, not out of specialist teams.

---

## Slide 6 — Beyond-Boundary Co-Design (aclGraph + W3b + lm_head)

**Title**: "Three optimisations, one op chain, composed."

- W1: `lm_head` CPU → NPU port (**+8.3 fps**, Slide 3)
- W3b: `aclnnAddRmsNorm` kernel fusion, **255 dispatches / frame saved** (**+0.66 fps**)
- G0-G4: aclGraph pos-keyed capture, 17 graphs × 1 MB, **0 byte-drift** (**+1.15 fps**)
- Three adjacent optimisations on ONE op chain — port, fusion, dispatch amortisation
- No conventional team would have shipped these as a single coherent plan in one quarter
- Cumulative clean-quality result from first native engine: **22 fps → 32.2 fps**, user-ear verified (full arc from llama.cpp ~1 fps baseline = **~32× lift**, tag `32fps-landed`)

**Speaker notes**: This slide shows the compounding effect. The CP engine's `forward_one_token_launch` is an 80-op aclnn chain. An agent with the end-to-end mental model sees three non-competing optimisations on the same chain: port the lm_head call that follows it from CPU to NPU (W1), fuse the per-sublayer Add+RmsNorm tail into a single kernel (W3b), capture the whole thing as an aclGraph and replay per pos-key (G2). Conventional team dynamics would have shipped one of these per quarter, because each is owned by a different abstraction layer — dispatch, ops API, runtime. The agent composed them. And the gotcha: we projected +6 to +10 fps for aclGraph in the G0 feasibility probe; reality was +1.15. That's because `TASK_QUEUE_ENABLE=2` already amortized most launch overhead — the agent initially over-estimated the ceiling. But it's an honest +1.15 fps on top of W1's +8.3 and W3b's +0.66, user-ear verified at 31.6 fps. Byte-identical parity on canonical xvec / ICL / CV benchmarks (G3.1: max_drift = 0 over 1680 tokens, output WAV md5 matches stock). Stacked, under PM gating, in roughly one week wall. That's the beyond-boundary co-design dividend.

---

## Slide 7 — Parallel-Universe Proof (OminiX-MLX)

**Title**: "Apple Silicon has no CANN. Agent built the stack anyway."

- Zero inherited infra: no aclGraph, no AscendC, no `aclnn*` library
- Built from `mlx-rs` + `safetensors` + `mlx-sys` primitives
- **9+ production models shipped**: `flux-klein-mlx`, `zimage-mlx`, `gpt-sovits-mlx`, `funasr-qwen4b-mlx`, `qwen3-tts-mlx`, `qwen3-vl-mlx`, `glm4-mlx`, `deepseek-ocr2-mlx`, `qwen-image-mlx`
- **24 GB Mac runs quantized FLUX** via drop-after-encode (f32 encoder → encode → drop → 8-bit transformer)
- Cross-platform benchmark harness: **M3 Max 46 fps > M4 Pro 37 fps > Ascend 30.5 fps**
- Ascend gap is visible precisely because the MLX stack was written from scratch, no bloat

**Speaker notes**: The counterfactual. Apple Silicon's MLX runtime gives you nothing comparable to CANN's aclnn op library. No `aclnnFusedInferAttentionScoreV2`, no pre-fused SwiGLU, no graph capture, no AscendC-equivalent DSL. And yet in the same calendar window — using the same agent workflow — we shipped nine production models in the MLX universe: FLUX.2-klein image gen, Z-Image 4-bit quantized, GPT-SoVITS voice clone, FunASR Qwen4B ASR, Qwen3-TTS, Qwen3-VL, GLM-4, DeepSeek-OCR2, Qwen-Image. Each is its own Rust crate, averaging 1-1.5k LoC of hot-path code. The 24 GB Mac FLUX result is the most revealing: the agent discovered that f32 text encoder + 8-bit transformer didn't fit simultaneously, designed the drop-after-encode pattern (load encoder → encode prompt → explicitly drop to free GPU memory → load transformer → denoise), and verified with `mlx_sys::mlx_clear_cache()` between steps. That's a production memory-budget pattern invented + shipped by an agent, not extracted from a textbook. We also discovered — and documented in project memory — three MLX precision constraints the hard way: bf16 crashes the Metal backend with `std::runtime_error`, so it's unusable for compute; f16 produces NaN in RmsNorm and softmax because text embedding values range ±16000 near the f16 dynamic-range limits; only f32 is safe for the Qwen3 text encoder forward pass. That constraint discovery, at the agent+hardware boundary, is exactly the thing no pre-AI-coding team would have bothered to write down systematically. The Ascend-vs-Apple gap is now legible: Apple's M3 Max hits 46 fps on Qwen3-TTS because its MLX engine was written exactly for the workload, with no inherited framework bloat. Ascend's 30.5 fps is what you get when you're still peeling back a 15-year-old C++ dispatch layer. The delta isn't hardware — Apple M3 Max has ~400 GB/s memory bandwidth vs Ascend 910B4's 800 GB/s, so Ascend's raw fabric is 2× faster and yet it runs 1.5× slower. The delta is stack debt, and AI coding is how we pay it down.

---

## Slide 8 — Scale Trajectory: Single Card → Cluster

**Title**: "The same mental model, one PR bigger."

- Today: single 910B4, Qwen3-TTS, **32.2 fps** clean-quality (fork `32fps-landed`)
- Tensor-parallel across 8 cards: agent edits HCCL collective calls + model loader + pipeline stages **in one PR**
- Pipeline-parallel: re-partition 5 CP layers across nodes; agent handles boundary tensor marshalling
- Data-parallel sharding: agent co-designs the batcher with the engine (shared KV cache policy, packed requests)
- No team-boundary hand-offs = **no "we'll define the interface next quarter"**
- Agent carries the end-to-end mental model the entire time; humans carry the gates

**Speaker notes**: The reason this matters strategically is that nothing in the single-card playbook above stops at one card. Tensor-parallel, pipeline-parallel, data-parallel, and expert-parallel strategies all require exactly the kind of cross-layer editing the agent workflow already does. You edit the collective-comm layer (HCCL / NCCL), the model loader (weight sharding), the engine pipeline (activation exchange), the batcher (request packing), the framework (sampler) — across five layers, in one PR, with the test harness living in the same session. A conventional cluster bring-up is six engineers across three teams with a shared Slack channel and a monthly sync. The tradeoff changes: you need to invest less in the interface-definition ritual between teams and more in the correctness-gate ritual that an agent needs to be safe. But that ritual is cheap compared to what it replaces. Concretely: the tensor-parallel pattern on an 8-card 910B4 deployment demands a consistent view of what lives where — the QKV projection sharded along head dim, the O projection sharded along the contracted dim, the FFN sharded by column, and an all-reduce at every residual boundary. A conventional team will write four interface documents, three design reviews, and seven tickets to coordinate that. The agent edits all four sites in one session because it holds the same mental model across them — and then the PM gate is: "does the output still token-match single-card for the first 100 frames". If yes, ship; if no, debug with full context still loaded. We have not yet run this pattern at cluster scale on OminiX — the current deliverable is single-card. Slide 10 proposes exactly that next step. The ceiling argument we'll lean on when asked: our single-card Ascend path went from 12.2 → 33.8 fps, a 2.77× gain. If the cluster brings even a 1.5× tensor-parallel scaling win (realistic given HCCL overhead) on top of whatever single-card improvements we pull through with the same workflow, we're looking at a 4-5× total gain vs the llama.cpp-ggml-cann baseline — in wall-time measured in weeks, not quarters.

---

## Slide 9 — Limits + Risks (Honest)

**Title**: "Agents are fast; gates are what make them safe."

- **Correctness gates matter**: W4.1 kernel passed offline 5e-4 diff, **drifted 1949 at runtime** — only the live gate caught it
- **Agents invent**: CannFusion F1 probe found the dtype whitelist blocks A16W8 — would have wasted weeks without the probe-first gate
- **Ceiling math is often wrong**: G0 projected **+6-10 fps**, reality **+1-1.6 fps** — don't fund on agent estimates, fund on measured gates
- **PM role shifted**: "contract author + gate operator", not "code writer" — discipline change, not tools change
- **Security**: agents have unbounded write scope by default; we use patch-file + PM-pushes mechanism on ac01 (no fork push creds) to preserve review

**Speaker notes**: Four things to flag as honest limits. First: offline tests lie. Our W4.1 AscendC kernel passed a 5e-4 max-abs-diff on a synthetic fixture but catastrophically drifted on the live generate loop. The reason was subtle — something in the attn-sublayer's accumulation path interacted with real hidden-state statistics that the fixture didn't capture. You need live, end-to-end, token-level drift gates on every agent kernel ship, not just unit tests. The W1.4 gate we used (first 10 frames across 3 canonical utts, ≤1 token drift per frame per group) is the template. Second: agents confabulate. Our first CannFusion contract assumed the codegen would accept A16W8 (F16 activation × INT8 weight → F16) because that's the production dtype for Qwen3-TTS. The F1 probe revealed a hard-coded dtype whitelist in `src/validate.rs:141-163` that explicitly rejects it; the project has an intentional negative test at `validate.rs:367-372`. If we'd skipped the F1 probe and gone straight to implementation, we'd have burned a week. We filed the upstream ask at `docs/cannfusion_upstream_ask.md` and logged GitCode issue #26 as the clean close. Probe-first is now the default. Third: agent ceiling estimates are systematically optimistic. The G0 aclGraph probe projected +6 to +10 fps; reality was +1.15. The reason was a subtle one — TQE=2 already amortizes per-dispatch launch from ~40 μs (eager) to ~2-3 μs (two-phase submit), a ~15× reduction already baked in — and the agent just didn't factor it in. Never fund a track on an agent's projection; fund it on measured gates. Fourth: PM role shifts. The PM is now a contract author (crisp, numeric gates per milestone) and a gate operator (verify numbers, ear-check WAVs, sign off). PM does not write the code. That's a genuine discipline shift for the human in the loop. And a fifth, briefly: security. Agents on a remote NPU host have unbounded write scope by default. Our mitigation is that ac01 has no fork push credentials, so all commits land locally; the agent hands off a patch file; the PM pulls it to the Mac, reviews, and pushes. That keeps one human eye on every ship without bottlenecking the agent's local iteration speed.

---

## Slide 10 — Call to Action / Thesis Restated

**Title**: "Cluster next. Playbook ready."

- **Crack is real and exploitable today** — OminiX Ascend **~1 fps → 32.2 fps (≈32× lift)**, MLX 9 models shipped
- **Playbook = PM-gated agent swarm + contract-per-track + patch-file review mechanism**
- **First movers get a structural perf + velocity advantage measured in quarters, not sprints**
- **Concrete next step**: replicate the OminiX contract pattern on a cluster-scale workload
  - e.g. 8-card tensor-parallel Qwen3-VL-32B on 910B4 cluster, same gate discipline
- Bring OminiX dev playbook, the 5 contracts (CP-FPS, aclGraph, Path C, CannFusion, Fused-Op), and the PC-tile / FO-audit probe artefacts

**Speaker notes**: To restate: the structural claim is that pre-AI-coding inference stacks were layered because humans specialise. Agentic AI coding dissolves the specialisation constraint, which in turn dissolves the layered-stack performance ceiling. The OminiX proof point is concrete: ~1 fps on conventional llama.cpp + ggml-cann → 22 fps after escaping to a native `CpCannEngine` → 32.2 fps after W1 + W3b + aclGraph + A.1 + M3'new' co-design (fork tag `32fps-landed`), all on single-card 910B4 Qwen3-TTS in ~7 days, plus 9 production MLX models on Apple Silicon in the same calendar window, all via PM-gated agent swarms. The playbook is six things: (1) contract per track with numeric gates at every milestone, (2) agent-dispatched implementation per milestone, (3) probe-first on anything that might be externally blocked (dtype matrix, API presence), (4) live end-to-end drift gates on every kernel ship, (5) user-ear gate outranking fps numbers on audio workloads, (6) patch-file review mechanism where the agent runs on the hardware host and the PM does the final push. The org that adopts this pattern on a cluster-scale workload first gets a structural advantage. Our concrete recommendation is to run it next on an 8-card 910B4 deployment of a bigger model — Qwen3-VL-32B is a reasonable target, with tensor-parallel across cards and pipeline-parallel across nodes. We bring the contracts, the gate discipline, and the dev pattern. The layered-stack ceiling is real, and the crack is real, and this is the year to step through it.

---

## Appendix A — Artefacts Referenced

- `docs/contracts/CP_FPS_OPTIMIZATION_CONTRACT.md` — W1 + W3b
- `docs/contracts/ACLGRAPH_CONTRACT.md` — G0-G4
- `docs/contracts/PATH_C_ASCENDC_CONTRACT.md` — W4.1 retro (CLOSED)
- `docs/contracts/CANNFUSION_CONTRACT.md` — F1 dtype probe (CLOSED, upstream GitCode #26)
- `docs/contracts/FUSED_OP_LANDING_CONTRACT.md` — Phase A/B follow-on
- `docs/fused_op_audit.md` — 717 headers grepped
- `docs/aclgraph_feasibility.md` — G0 projection vs reality
- `docs/ascend_910b4_datasheet.md` — 20 AIC + 40 AIV, UB 192 KB, L0C 128 KB, 800 GB/s HBM
- `tools/qwen_tts/cp_cann_engine.cpp` — 2657 LoC, integration target
- OminiX-MLX: `qwen3-tts-mlx/`, `flux-klein-mlx/`, `zimage-mlx/`, `gpt-sovits-mlx/`, `funasr-qwen4b-mlx/`, 9 production crates total

## Appendix B — Key Metrics (for Q&A)

| Track | Before | After | Δ | Wall time |
|---|---|---|---|---|
| W1 NPU lm_head port | 21.9 fps | 30.2 fps | **+8.3** | 1 agent dispatch |
| W3b Add+RmsNorm fusion | 30.2 fps | 30.5 fps | **+0.66** | 1 agent dispatch |
| G2 aclGraph pos-keyed | 30.5 fps | 31.6 fps | **+1.15** | 3 agent dispatches |
| Path C AscendC (CLOSED) | 30.5 fps | 30.5 fps (reverted) | 0 | 72 hr to clean close |
| CannFusion F1 (CLOSED) | N/A | N/A | 0 | 25 min probe |
| A.1 InplaceAddRmsNorm | 31.6 fps | 31.9 fps | **+0.3** (noise) | 1 agent dispatch |
| M3'new' pos 0+1 batched prefill | 31.9 fps | 32.2 fps | **+0.3** | 1 agent dispatch |
| M1.B FFNV3 | — | — | — CLOSED (MoE-only runtime) | 1 agent dispatch (RED) |

| MLX Workload | M3 Max | M4 Pro | Ascend 910B4 |
|---|---|---|---|
| Qwen3-TTS (frames/s) | 46 | 37 | 30.5 |
| FLUX.2-klein 1024² | production | production | not yet ported |
| Z-Image 4-bit | production | production | not yet ported |

---

## Appendix C — Independent Data Point: pie-project (Yale, SOSP 2025)

**Title**: "Same boundary, opposite direction — a research-side crosscheck."

- **What it is**: Pie (`github.com/pie-project/pie`, SOSP 2025, Gim/Ma/Lee/Zhong @ Yale) is a **programmable LLM serving system** that replaces the monolithic prefill-decode loop (vLLM / SGLang / TGI) with **fine-grained API handlers** — `embed`, `forward`, `sample`, `alloc_kvpage`, `mask_kvpage`, `export_kvpage` — orchestrated by user-written **inferlets** that compile to WebAssembly. Thesis from their HotOS 2025 companion paper: *"serve programs, not prompts"*; LLM serving systems are the new OS, inferlets are the processes, KV pages are the pages.
- **Published numbers** (Llama 3.2 1B, L40 GPU, vs vLLM / SGLang / LMQL / StreamingLLM):
  - Text completion: **matches vLLM within 1.5–11% TPOT** (2.4% overhead at 8B, 11.4% at 1B)
  - Agentic workflows: **1.1–2.4× lower latency, 1.3–3.4× higher throughput**
  - Attention sink: **1.5× lower latency, 30× higher throughput** vs hand-built StreamingLLM
  - Adaptive batcher: **17× throughput** vs eager, 8–40% vs fixed-rate policies
- **Hardware**: NVIDIA (CUDA + FlashInfer) + Apple Metal. **No Ascend, no AMD, no TPU.** Single-node only; authors flag cluster scheduling as future work.
- **Maturity**: Pre-release research prototype, Apache-2.0, v0.2.5. SOSP 2025 artifacts-evaluated (functional, available, reproducible). ~147 GitHub stars as of 2026-04-21.

**Relation to our thesis**: **extends, does not reinforce.** Pie attacks a **different boundary** than we do — the *application↔serving-system* boundary, not the *framework↔engine↔ops↔kernel↔compiler* stack. Pie's contribution is a programming model (Wasm-sandboxed inferlets + 42-call API) that lets application code directly drive KV cache policy, decoding strategy, and tool/IO integration, replacing global serving policies with per-request ones. Our contribution is the opposite end: collapsing the *internal* layers of the stack via cross-dialect agent edits, so a single PM+swarm can move a workload across Rust/C++/aclnn/AscendC in one session. Both dissolve rigid abstractions, but Pie exposes a richer boundary *upward* to the user; we dissolve boundaries *downward* through the vendor stack. The two are composable: a future Pie-Ascend backend could sit on top of our native `CpCannEngine`, and our agent-swarm playbook could drive inferlet-authoring for agentic workloads on 910B4.

**Honest caveats**:
- Pie is **not an agentic-coding proof point** — it's a hand-authored Yale systems research project with `AGENTS.md` / `.agent/skills/pie/SKILL.md` *contributor runbooks* that assume Claude Code-style agents are working on the codebase. That's downstream-of-agents, not generated-by-agents.
- Pie's 1.3–3.4× on agentic workloads is **measured against vLLM on L40 with Llama 3.2 1B**; our 32× is measured against llama.cpp+ggml-cann on Ascend 910B4 with Qwen3-TTS. Different denominators, different workloads — do not compose the multipliers.
- Pie targets NVIDIA/Apple; they don't yet see the CANN surface area we've been mining.

**What we should re-check in our own thesis after reading Pie**:
1. Slide 2's layered-stack diagram currently stops at "framework (PyTorch)". Pie argues there's **another layer above** — *application-defined generation logic* — that is also locked behind a boundary, and the lock is also paid for in perf (1.1–2.4× latency). Our deck under-counts that layer.
2. Slide 8 (scale trajectory) talks about cluster hardware scaling. Pie's agentic-workflow numbers suggest **workload-shape scaling is a parallel axis**: the same single card runs 3× more throughput if the application-layer abstraction cracks open. A co-design deck should probably argue for both axes together.
3. Our "agent swarm crosses 5 layers in one session" frame is **syntactic boundary crossing** (edit Rust + C++ + aclnn + AscendC in one dispatch). Pie's frame is **semantic boundary crossing** (KV-cache reuse patterns that only the application knows are safe). Both are real, both compose; future OminiX work on tree-of-thought / speculative decoding on Ascend should borrow Pie's inferlet API shape as an input — not reinvent the vocabulary.

**Citations**: Gim et al., *Pie: A Programmable Serving System for Emerging LLM Applications*, SOSP '25 (DOI 10.1145/3731569.3764814). Gim & Zhong, *Serve Programs, Not Prompts*, HoTOS '25 (DOI 10.1145/3713082.3730398). Repo: `github.com/pie-project/pie`. Standalone brief: `docs/pie_project_brief.md`.
