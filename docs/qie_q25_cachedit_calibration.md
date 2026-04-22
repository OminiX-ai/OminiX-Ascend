# QIE Q2.5 CacheDIT Calibration — Pre-flight verdict (BLOCKED)

**Agent**: QIE-Q2.5-CACHEDIT
**Host**: ac01 (reachable, NPU 2 idle, CANN 8.3.RC1) — no compute consumed.
**Date**: 2026-04-22
**Status**: **BLOCKED on Q2 precondition**. Calibration cannot run meaningfully today. No ac01 wall burned; no weights copied; no scratch directories created. Writing this receipt instead so the block is documented and the next agent does not repeat the pre-flight.

## Why BLOCKED (three independent reasons, any one sufficient)

### 1. The locked 20-task suite (512×512 / 20-step) NaNs today on the ggml-cann path

Per `docs/qie_q1_baseline.md` §"Known regression — step-count / sequence-length NaN":

| W × H | Steps | Result |
|---|---|---|
| 256×256 | 2 | OK |
| 256×256 | ≥3 | NaN at `diffusion/x_0` |
| 512×512 | any | NaN |

The eye-gate suite (`docs/qie_eye_gate_suite.md` §Scoring harness) is locked at `--steps 20 --resolution 512`. **Every one of the 20 tasks would produce a blank / NaN image on the current ggml-cann backend.** The agent brief's Phase 1 ("If any task NaNs, log + skip") assumes sparse failure; actual expected failure rate is 20/20. There is no remaining calibration set.

The NaN is a Q2 problem (per Q1 doc §"This is NOT blocking Q1...It IS the first thing Q2 needs to tackle before meaningful optimization"). It is not a lever a Q2.5 calibration agent can move — it needs either: (a) a native-engine forward in FP32 / BF16 accumulators (Q2 landing), (b) a CPU-dequant-residency fix (hypothesis H1 in Q1 doc), or (c) an FIA / attention-overflow fix at seq≈2048. All three are kernel work, explicitly **out of Q2.5 scope** ("No kernel code changes. This is calibration + eye-gate, not engineering" — agent brief §Constraints).

### 2. Q2 has not landed — the "Q2 no-cache baseline" referenced by the Q2.5 gate does not exist

Contract §Q2.5 gate wording: **"+25-50% end-to-end wall vs Q2 no-cache baseline with eye-gate PASS on 20/20 tasks"** (`docs/contracts/QWEN_IMAGE_EDIT_2511_CONTRACT.md:144`). Contract timeline puts Q2.5 after Q2 native-engine and Q2 aclGraph (`:235`).

Current Q2 state:
- `a50d0174 feat(qwen_image_edit): Q2 Phase 1 — native ImageDiffusionEngine scaffold` (scaffold only)
- No byte-parity gate cleared (`Q1.7` Q2 smoke TODO).
- No RoPE pre-compute at session init (`Q1.8` TODO).
- No aclGraph step-keyed capture (Q2 second sub-contract, TODO).

The agent brief's Phase 1 substitutes "Q1 ggml-cann path" for "Q2 no-cache". That substitution is not in the contract. Two problems with honoring it:
- (a) Calibration of thresholds against the ggml-cann path produces a number that will not port to Q2 native (different attention, different dispatch, different precision mix). The Pareto point at the Q1 path is **advisory at best, not the locked Q8 default the gate demands**.
- (b) Per the suite doc §Baseline output: "Each subsequent milestone compares to this baseline, not to the prior milestone" — that Q2 baseline image set is required as the FROZEN referent for Q2.5/Q3/.../Q8 diffs. Substituting a Q1-path baseline would break the comparison chain for every downstream milestone **and** the suite lock prohibits mid-contract re-baselining.

### 3. The 20 suite source images are not locked

Suite doc open item (`qie_eye_gate_suite.md:175-179`): "**Source image acquisition and sha256 locking.** 1-2 hours agent wall... Commit URLs and hashes to METADATA.json **before Q2 baseline capture**."

Current state: `tools/ominix_diffusion/testdata/` does not exist on Mac or on ac01 source tree. Only `/home/ma-user/qie_q0v2/test/cat.jpg` on ac02 — one image, not 20. Running the suite today requires either:
- Ad-hoc fetching 19 more images per the suite's provenance recommendations (Wikimedia / Pexels / Unsplash), hashing them, committing. That's a Q1 / Q2 open item that another agent was assigned (see suite doc checklist).
- Proceeding with 1 task × 4 cache modes × 4 thresholds = 16 runs on a single cat image. Not a calibration — a coincidence.

Touching the source-image set outside the Q1 / Q2 agent's scope would step on the suite-lock contract and poison sha256 provenance for every downstream milestone.

## What I checked before halting

- SSH to ac01: reachable, `910B4`, CANN 8.3.RC1, NPU 2 idle (HBM 2842 / 32768 MB, no processes).
- ac01 source tree: `/home/ma-user/work/OminiX-Ascend-w1/` present at `build-w1/`.
- ac01 CLI: `build-w1/bin/ominix-diffusion-cli` exists; `--cache-mode`, `--cache-option`, `--cache-preset`, `--scm-mask` all wired (help-output confirmed, 4 modes selectable).
- ac01 weights: **absent**. Would need 17 GB scp from ac02 (`qie_q0v2/weights/{mmproj-BF16.gguf, Qwen2.5-VL-7B-Instruct-Q4_0.gguf, Qwen-Image-Edit-2509-Q4_0.gguf, split_files/vae/qwen_image_vae.safetensors}`).
- ac02 state: per agent brief, running Q3 FIAv2 probe; I did not disturb it other than `ls` on weights + test dir.
- `stable-diffusion.cpp:1761-1864`: confirmed 4-mode wiring (`EasyCache`, `UCache`, `DBCache`, `TaylorSeer` / `CacheDIT` combined) with `sd_cache_params_t` plumbed through the sample loop. The calibration lever is real and non-destructive; the gating problem is upstream of it.
- Contract + suite lock dates: both are `2026-04-22`, same day as this pre-flight; no newer amendment exists that would relax the conditions above.

## Minimum-viable unblock path (for the next agent — not this agent's scope)

Ordered by who-should-do-what, lightest-first:

1. **Q1 / Q2 agent** (owns the suite source images): fetch + sha256-lock the 19 non-cat images per `qie_eye_gate_suite.md` provenance. 1-2 hours. Commit under `tools/ominix_diffusion/testdata/qie_eye_gate/sources/` with `METADATA.json`. Without this, no calibration run can hash-compare across milestones.
2. **Q2 agent** (native engine): land byte-parity at 512×512 / 20 steps without NaN. Per the Q1 doc's hypotheses H1-H3, the most likely single-lever fix is **weight residency at buffer load time** (H1), eliminating the per-call CPU dequant D2H/H2D jitter that compounds across steps. Until this lands, no 512×512 suite run produces valid output.
3. **Q2 agent** (baseline capture): run the 20-task suite once at 512×512 / 20 / no-cache, produce `baseline_q2/` under `testdata/qie_eye_gate/`. That is the FROZEN referent for everything below Q8.
4. **Q2 aclGraph** (optional but in-contract): land step-keyed capture, produce a second timing baseline (quality identical to Q2 native).
5. **This agent (Q2.5) re-runs**: now there is a `Q2 no-cache baseline` with valid 20/20 images, and threshold sweeps can happen without blank-image false PASS.

Agent-wall estimate for 1+2+3: 2-4 weeks (dominated by #2). That is already in the contract timeline; Q2.5 is intentionally scheduled after Q2 lands.

## Interim-ok work this agent could do if re-tasked (does NOT clear the gate)

Offered in case the PM wants a sliver of forward progress while Q2 proceeds. None of these hit the Q2.5 gate; they are ramp.

- **A**: Cache-mode plumbing smoke on ac01 @ 256×256 / 2-step / `cat.jpg` (the only config that currently produces a valid image). One run per mode × default thresholds = 4 runs × ~150 s each ≈ 10 minutes wall. Exercises that the cache modes init + don't crash on QIE dispatch; does NOT validate thresholds, does NOT touch eye-gate. Output: a receipt that the 4 `--cache-mode` branches are not silently broken at QIE-shape. Would go under this doc as §"2-step dispatch smoke" addendum; would not touch the contract Q2.5 checklist.
- **B**: Pre-commit the Q2.5 timing-harness shell script (`tools/ominix_diffusion/scripts/run_eye_gate.sh` per suite doc §Scoring harness) so Q2 agent + Q2.5 re-run are unblocked on harness. No NPU needed. ~1 hour agent wall. Adds value independent of Q2 landing.
- **C**: Draft the per-mode `--cache-option` grid (DBCache `Fn=`, `Bn=`, `threshold=`, `warmup=`; EasyCache `threshold=`, `start=`, `end=`; UCache similar) as a YAML matrix so Q2.5 re-run can sweep mechanically. ~1 hour. Paired with (B) cleanly.

**Recommendation**: do **none** of A/B/C without explicit PM green-light. The contract asks for a calibration result, not harness prep, and silently extending scope poisons the deliverable shape. Returning BLOCKED cleanly is the more honest outcome.

## Compliance receipt

- **Host rule**: ac01 ONLY requested. Zero compute used. SSH reached only to verify reachability + absence of weights + presence of CLI.
- **ac02 / ac03**: read-only `ls` on ac02 weights; zero touch on ac03.
- **Budget**: 0 of 7-10 day budget consumed. Entire budget remains available for the re-run after Q2 clears.
- **Hard-kill rule**: N/A — no threshold runs attempted, no eye-gate scored.
- **Contract amendment needed**: **no**. Q2.5 gate wording already names `Q2 no-cache baseline` as the referent; honoring that is just being punctual, not redefining scope.

## Sign-off

Halting here until Q2 byte-parity + baseline-capture lands. No patch to Mac diff, no commit, no weights scp'd.
