# QIE Q1 — Upstream PRs to ggml-org/llama.cpp

**Agent**: UPSTREAM-PR
**Date**: 2026-04-22
**Status**: **BLOCKED — NOT OPENED**. Branches pushed to `ymote/llama.cpp`; PRs not filed pending human author to adopt.

## Summary

Three ggml-cann backend fixes from the OminiX-Ascend fork were cherry-picked onto clean branches of `ggml-org/llama.cpp` master (`0d0764dfd`) and pushed to `ymote/llama.cpp`. PRs were NOT opened due to an upstream AI-contribution policy blocker discovered during Phase 4 review of `CONTRIBUTING.md` / `AGENTS.md`.

## Commits upstreamed (branch state on `ymote/llama.cpp`)

| Source (OminiX-Ascend) | Target branch (ymote/llama.cpp) | Files | Status |
|---|---|---|---|
| `bbfa1912` fix(ggml-cann): Q4_0/Q4_1 GET_ROWS | `fix-ggml-cann-get-rows-q4` | `aclnn_ops.cpp` +104 / `ggml-cann.cpp` +10 | PUSHED, PR NOT OPEN |
| `c8fea6e0` fix(ggml-cann): Q4_1/Q5_*/K-quant MUL_MAT fallback | `fix-ggml-cann-mul-mat-k-quant` | `aclnn_ops.cpp` +89 / `ggml-cann.cpp` +18 | PUSHED, PR NOT OPEN |
| `61c52a34` fix(ggml-cann): stream sync around pool buffers | `fix-ggml-cann-stream-sync` | `aclnn_ops.cpp` +23/-2 (stacked on both above) | PUSHED, PR NOT OPEN |

Branches on `ymote/llama.cpp`:
- https://github.com/ymote/llama.cpp/tree/fix-ggml-cann-get-rows-q4
- https://github.com/ymote/llama.cpp/tree/fix-ggml-cann-mul-mat-k-quant
- https://github.com/ymote/llama.cpp/tree/fix-ggml-cann-stream-sync

## BLOCKER: upstream AI-contribution policy

`ggml-org/llama.cpp` updated its contribution policy to explicitly prohibit AI-authored PRs. Quoting `AGENTS.md`:

> The following will result in **immediate PR closure**:
> - **AI-written PR descriptions or commit messages** — these are typically recognizable and waste reviewer time
> - **AI-generated responses to reviewer comments**
> - Implementing features without understanding the codebase

And from `CONTRIBUTING.md`:

> Maintainers reserve the right to decline review or close pull requests for any reason, particularly under any of the following conditions:
> - The contributor fails to adhere to this contributing guide or **the AI policy**.

The three source commits on `ymote/OminiX-Ascend` were authored by `Agent A2-reopen` with AI-drafted commit messages. The upstream PR bodies the mission asked for would also be AI-drafted. Filing them as-is risks:

1. Immediate closure of all 3 PRs on first reviewer glance.
2. The `ymote` account flagged for repeat violations (CONTRIBUTING.md warns of permanent bans).
3. Reputational damage that makes future legitimate contributions harder.

Additionally, CONTRIBUTING.md §Pull requests says **"If you are a new contributor, limit your open PRs to 1"** — the `ymote` account has no prior contribution history on `ggml-org/llama.cpp`, so 3 simultaneous PRs would also violate this.

## What is NOT blocked

The patches themselves are:
- Clean-applying on upstream master (two of three hunks land with zero conflicts; the third needed a 10-line manual placement in `ggml_backend_cann_supports_op` because upstream's Q8_0 case has BF16 gated on `ASCEND_310P` where our fork did not).
- Net-new code — neither the Q4 GET_ROWS branches nor the K-quant CPU-dequant fallback nor the `aclrtSynchronizeStream` guards exist on upstream master.
- Fleet-value: fix the same class of abort that closed issues `ggml-org/llama.cpp#15759` and `#9979` hit.

Technical review of the patches against upstream at `0d0764dfd`:
- `fix-ggml-cann-get-rows-q4`: compiles (no new headers required; `std::vector` and `ggml_fp32_to_fp16_row` are already included transitively).
- `fix-ggml-cann-mul-mat-k-quant`: same, applies at hunk offsets −578/−604/−602/+40 (file drifted but no content conflict).
- `fix-ggml-cann-stream-sync`: applies cleanly on top of the stacked branches.

## Recommended next steps (PM decision needed)

**Option A — human author adopts and rewrites (preferred).** Hand the three branches to a human contributor willing to:

1. Read through each patch and understand it end-to-end (enough to defend every line to a maintainer without AI help).
2. Rewrite the commit messages in their own words — short, direct, no AI-tell.
3. Run `ci/run.sh` locally (or at least `test-backend-ops`) and paste the output in the PR body.
4. File as **one** PR initially (the contributor's first), probably `fix-ggml-cann-get-rows-q4` since it's the smallest and most self-contained; add the other two as follow-ups after the first merges. This respects the "limit to 1" rule.
5. Disclose AI-assist (per AGENTS.md permitted-usage rules) only if the human genuinely used AI for pattern-completion or formatting, not for authoring.

**Option B — file as-is and accept closure risk.** Not recommended. Burns the account reputation for three small fixes.

**Option C — keep the fix in the fork only.** Acceptable for QIE Q1 delivery since the fork already ships the code. Revisit upstreaming when a human maintainer has bandwidth.

## Review-latency signal

Searched `ggml-org/llama.cpp` for ggml-cann PR turnaround (for future planning):
- `leo-pony` (CANN CODEOWNER) is the primary reviewer for Ascend NPU changes.
- Closed issues touching Q4_0 on CANN (`#9979`, `#15759`) went stale rather than being fixed upstream — signal that ggml-cann Q-quant coverage is a neglected corner of the tree and PRs may sit for a while unless the reviewer is explicitly pinged.
- Recent merged CANN PRs can be queried via `gh pr list --repo ggml-org/llama.cpp --label "Ascend NPU" --state merged`.

## Local artefacts

- Clean upstream clone: `/Users/yuechen/home/ymote-llama-cpp/`
- Patches: `/tmp/upstream-patches/*.patch` (format-patched from the 3 source commits)
- Source fork remote: `fork https://github.com/ymote/OminiX-Ascend.git`
- Test receipt for PR bodies: `docs/qie_q1_baseline.md` (910B4, CANN 8.3.RC1, Q4_0 QIE, 2-step 256×256, valid cat PNG, 145s wall)

---

## DECISION 2026-04-23: SKIP UPSTREAM

PM opted to keep the 3 fixes fork-only rather than attempt upstream filing.
Rationale: CONTRIBUTING.md AI-authored-PR prohibition creates reputation
risk on `ymote` account with uncertain reviewer-tolerance payoff. Branches
remain pushed at `ymote/llama.cpp` for future human-authored submission
if scope changes.
