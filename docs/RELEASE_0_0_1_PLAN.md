# Release 0.0.1: Requirements and Current Status

**Repository:** vllm-project/vllm-torchtpu
**Status:** Proposal — nothing below is in effect yet
**Last updated:** 2026-09-21

---

> [!NOTE] Relationship to other documents
> [PRE_PUBLIC_DEV_GUIDE.md](PRE_PUBLIC_DEV_GUIDE.md) governs `main` during the pre-public
> phase. This document covers only the additional rules for a release branch, and reuses
> that guide's mechanisms (`ready` label, approval gate, guarded merge) rather than
> inventing new ones.

---

## 1. Requirements

As agreed:

1. **Cut `release/0.0.1` from a relatively stable `origin/main` commit.** Target the
   2026-09-19 or 2026-09-20 nightly build, if it looks good. Known bugs are acceptable;
   fixes can be imported later.
2. **Blocking issues are fixed on `origin/main` first, then cherry-picked back** to
   `release/0.0.1` by the bug owner.
3. **Non-blocking issues are not backported**, to keep divergence from `main` minimal.
4. **`release/0.0.1` keeps CI capability.** A cherry-pick may be merged only after the
   cherry-pick is complete and CI passes, and only with approval from a predefined list
   of people.

Two implementation notes on requirement 4, from the discussion:

- The main engineering task is enabling presubmit CI for the branch.
- "Merging is only allowed after approval" can be as simple as: the PR may be merged only
  when someone on a predefined list has approved it.

---

## 2. How tpu-inference does this

There is no release runbook in `vllm-project/tpu-inference`; the following was recovered
from its branches, its GitHub history, and the `vllm` Buildkite org.

### 2.1 Branch and tag shape

Sixteen branches named `releases/vX.Y.Z`, each cut from a `main` commit and never merged
back. Only `main` is protected — the same free-plan limitation we have. The release tag
sits on the **branch head**, not on `main`.

| Branch | Commits ahead of cut | Behind `main` today | What is on it |
| --- | --- | --- | --- |
| `releases/v0.12.0` (first) | 2 | 2832 | "set up v0.12.0 release" + support-matrix refresh |
| `releases/v0.18.0` | 23 | 1718 | Cherry-picks plus 8 WIP commits pushed straight to the branch |
| `releases/v0.23.0` | 8 | 793 | 7 cherry-pick PRs + a docs commit |
| `releases/v0.29.0` (latest) | 2 | 75 | Docs only |

### 2.2 The cut commit

Their first release cut (`814459ca`, "set up v0.12.0 release") changes exactly one thing:
`docker/Dockerfile` stops floating on vLLM `main` and hard-pins `git checkout v0.12.0`.
No version file is bumped — `setup.py` reads the version from `VLLM_VERSION_OVERRIDE`,
which the release workflow derives from the git tag.

### 2.3 Cherry-pick discipline

They converged on this over about three releases. `releases/v0.18.0` still has raw commits
pushed to the branch (`keep`, `pre-commit`, `version with bfloat16`). From `v0.23.0` on,
every change is a PR titled:

```
[releases/v0.23.0] Cherry-pick #2931: Fix num tokens in mmlu test for continue decode
```

with a body that links the `main` PR and states what it fixes. Fix-on-main-first is the
norm. The exceptions are release-only test-threshold relaxations, which have no `main`
equivalent to cherry-pick from.

### 2.4 CI on the release branch

- **Presubmit needs no configuration.** `tpu-inference-ci` has
  `branch_configuration: 'main'`, yet every PR builds, because a Buildkite branch filter
  does not apply to pull-request builds. PRs into `releases/*` get full CI for free.
- **Nightlies on the release branch are triggered manually through the API** by the
  release manager. Build #39 on `releases/v0.29.0` is `source=api`, `NIGHTLY=1`. There is
  no Buildkite schedule for any release branch.
- **Commit-back steps had to be gated.** Support matrices, verified commit hashes, and the
  LKG bump all push back to the repo with a `GITHUB_PAT` that is only provisioned for
  `main` and release tags, so they failed on release-branch nightlies. tpu-inference PR
  #2941 gates those steps to `(NIGHTLY && branch == main) || tag`, and
  `update_tokamax_pin.sh` refuses to push when `BUILDKITE_BRANCH != TARGET_BRANCH`.

### 2.5 Merge gating

Nothing enforces it. `check_ready_label.yml` (a `ready` label) is their only merge-adjacent
check and it is not a required check. Who may approve a cherry-pick is convention.

### 2.6 One thing not to copy

Their `release.yml` aborts unless the pushed tag is reachable from `origin/main`. No
release-branch tag can satisfy that: their tags point at commits like "fix docs for
v0.29.0 release" that exist only on the branch.

---

## 3. Current status in this repository

### 3.1 Already works, no change needed

| Capability | Where | Note |
| --- | --- | --- |
| Presubmit CI on PRs into any branch | `vllm-torchtpu-ci` Buildkite pipeline | `build_pull_requests: true`; the `main` branch filter does not apply to PR builds |
| vLLM pin freezes on cut | `pyproject.toml`, read by `.buildkite/scripts/configs/pipeline_config.sh:45` | Branching freezes the pin automatically |
| `ready` label gate | `.buildkite/scripts/bootstrap.sh:86` | Applies to PRs against any base branch, so cherry-pick PRs need the label too |
| Version from git tag | `pyproject.toml` `[tool.setuptools_scm]` | A `v0.0.1` tag produces the version; no file to bump |
| Approval reporting | `.github/workflows/approval-gate.yml` | Reads CODEOWNERS from the PR's **base** branch |

### 3.2 Gaps

| # | Gap | Detail | Fix |
| --- | --- | --- | --- |
| 1 | Post-merge builds on the release branch | `vllm-torchtpu-ci` has `branch_configuration: 'main'`, so a push to `release/0.0.1` creates no build | Add `release/*` to the branch filter. Buildkite setting only, no repo change |
| 2 | Nightly on the release branch | The only schedule is `branch=main, NIGHTLY=1` | Add a schedule on `release/0.0.1`, or trigger manually with `bk build create -b release/0.0.1 -e NIGHTLY=1` as tpu-inference does |
| 3 | Commit-back guard | `.buildkite/scripts/update_lkg_version.sh:24,83` pushes to `TARGET_BRANCH` (default `main`) with no check that the build ran on that branch | Add tpu-inference's guard. Low urgency: `pipeline_promote.yml` is uploaded only by the `vllm-torchtpu-integration` pipeline (`bootstrap.sh:173`), which must not be pointed at the release branch anyway |
| 4 | Predefined approver list | No branch protection on this plan (see PRE_PUBLIC_DEV_GUIDE.md §1) | See §4 below — a per-branch CODEOWNERS gives this with no new code |
| 5 | No release artifact path | No `release.yml` in this repo; `.github/scripts/determine_release_vars.sh` is an orphan from the tpu-inference sync with nothing calling it. `publish_nightly_images.sh` pushes `nightly-YYYYMMDD` and `latest` to Artifact Registry, with no release-tag variant | Decide what 0.0.1 ships (§6) |

### 3.3 Candidate cut commits

Scheduled nightlies on `main` have ended red every night since 2026-09-12, but most of
that is a small, stable set of jobs. Each nightly also reports 8 `broken` jobs — the
`(PR)`-only unit-test lanes and `Presubmit: disagg-single-host`, skipped by design on a
nightly build. Those are not failures and are excluded below.

| Date | Build | Commit | Passed | Real failures (exit 1) |
| --- | --- | --- | --- | --- |
| 2026-09-19 | #1768 | `43fd95e8` | 33 | 5 |
| 2026-09-20 | #1812 | `fcf2d133` | 29 | 9 |
| 2026-09-21 | #1845 | `a7ba5ac2` | 32 | 5 |

Failing on all three nights, so pre-existing rather than introduced by either candidate:

- `Unit Tests - Speculative Decoding (Nightly)`
- `Perf: Nightly glm-5.2-fp8-mp-dp16-ep (TPU v7x-16)`
- `Perf/Eval: Nightly deepseek-v4-pro-dp16-ep (TPU v7x-16)` and its xprof capture
- `Perf/Eval: Nightly Kimi-K2.6 Real (TPU v7x-8)`

2026-09-20 adds four more on top of that set — `Unit Tests - General (Nightly)`,
`qwen3.5-397b-fp8-tp1-dp8-ep`, and both `deepseek-v4-flash` jobs — so **2026-09-19
(`43fd95e8`) is the better candidate**: 33 passing, and its 5 failures are the standing
set rather than anything new.

"Some bugs are acceptable" is a reasonable bar for 0.0.1, and 5 known failures is a
defensible starting point. What is needed before the cut is a decision on each of the 5:
accepted as a documented 0.0.1 limitation, or fix-on-main-then-cherry-pick. Waiting for a
fully green nightly is not realistic while the kube lane migration is in flight.

---

## 4. Proposed merge gate for `release/0.0.1`

CODEOWNERS is evaluated per branch, and `approval-gate.yml` reads it from the PR's **base**
branch. So committing a one-line `.github/CODEOWNERS` to `release/0.0.1`:

```
* @<release-manager-1> @<release-manager-2> @<release-manager-3>
```

gives, with no new code:

- The `CODEOWNERS Approval` check reports exactly "a release manager approved this".
- `AUTO_REQUEST_REVIEWERS` assigns those people when the cherry-pick PR is opened.
- `scripts/find_owners.py --request` names them.

Two settings to change for the release branch specifically:

- `MISSING_APPROVAL_CONCLUSION: failure`, so an unapproved cherry-pick is red rather than
  grey. The reason it is `neutral` on `main` — every PR is unapproved for its whole review
  window — matters less on a branch that takes a handful of PRs.
- Keep the `approval-gate-override` label available, but treat any use on the release
  branch as requiring a written justification.

This is still advisory, like everything else in the pre-public phase: nothing disables the
merge button. The enforcement is the same as on `main` — the guarded merge command in
PRE_PUBLIC_DEV_GUIDE.md §3, and revocation of write access for repeat violations.

---

## 5. Proposed process

### 5.1 Cutting the branch

1. Pick the cut commit from a nightly (§3.3) and record which failures are accepted.
2. `git branch release/0.0.1 <sha> && git push origin release/0.0.1`.
3. Add `release/*` to the `vllm-torchtpu-ci` branch filter (gap 1) and set up the nightly
   trigger (gap 2).
4. Land the branch's `.github/CODEOWNERS` and the `MISSING_APPROVAL_CONCLUSION` change
   (§4) as the first PR into the branch, which also smoke-tests presubmit on it.
5. Verify: open a throwaway PR into `release/0.0.1`, confirm Buildkite builds it and the
   approval gate reports the release-manager list.

### 5.2 Backporting a blocking fix

1. The bug owner lands the fix on `main` through the normal flow.
2. The same owner opens a cherry-pick PR against `release/0.0.1`, titled
   `[release/0.0.1] Cherry-pick #<N>: <subject>`, with a body linking PR #N and stating
   why the bug is blocking.
3. Apply the `ready` label so presubmit runs.
4. Merge only when CI is green and a release manager has approved.

Release-only changes with no `main` equivalent (test-threshold relaxations, for example)
follow the same PR flow, with the body saying why it is release-only.

### 5.3 What does not get backported

Non-blocking bugs, features, refactors, and dependency bumps. "Blocking" means a
`release/0.0.1` user cannot run their workload and has no workaround.

---

## 6. Open decisions

1. **What does 0.0.1 ship?** A tagged Docker image in Artifact Registry, a PyPI wheel, or
   just the branch and tag. This determines whether gap 5 is work or a no-op.
2. **Who is on the release-manager list?** (§4)
3. **Branch naming.** tpu-inference uses `releases/vX.Y.Z`. We said `release/0.0.1`. Only
   matters for shared tooling; there is none today.
4. **Which of the 5 standing nightly failures are accepted for 0.0.1?** (§3.3)
5. **How long does the branch live** — until 0.0.2 is cut, or a fixed window?
