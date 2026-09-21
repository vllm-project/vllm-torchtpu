# Pre-Public Development Guide

**Repository:** vllm-project/vllm-torchtpu
**Effective Period:** July 2026 – Public Launch (Estimated Oct 2026)
**Target Audience:** All Onboarded Developers, Core Maintainers, and Reviewers
**Primary Tooling Standard:** GitHub CLI (`gh`) + Git

---

> [!NOTE] Relationship to Standard Documentation
> For all general contribution guidelines, bug reporting procedures, project directory structures, and general testing policies, please refer directly to [CONTRIBUTING.md](../CONTRIBUTING.md).
>
> This document is **EXCLUSIVELY** a temporary, specialized governance supplement for the pre-public phase to handle missing GitHub repo branch rulesets due to private repo on github free account.

---

## Pre-Merge Verification Checklist

Every developer merging code **MUST** verify:

- [ ] Review approval verified: `latestReviews` has at least 1 `APPROVED` for `<PR_NUMBER>` ✅
- [ ] Discussions checked: All comment threads marked as resolved for `<PR_NUMBER>` ✅
- [ ] `gh pr checks <PR_NUMBER>` returns **ALL GREEN** ✅ (Automated by the Guarded Merge command)
- [ ] `ready` label applied, so presubmit CI actually ran (a PR that never got the label has no Buildkite result to be green)
- [ ] DCO Sign-off verified (`-s` or UI override via "Set DCO to PASS")
- [ ] All PR discussion comments resolved
- [ ] Post-merge validation plan ready (Inspect Buildkite Nightly dashboard)

---

## 1. Why This Guide Exists (Background & Rationale)

Welcome to the `vllm-project/vllm-torchtpu` pre-public development phase! Please read the rationale below regarding why this temporary governance guide is required:

### 🔍 The Situation & Challenge

- **Transition Phase:** We are currently operating in a private organizational repository phase preparing for our official Public Launch (Estimated October 2026).
- **Missing Automated Rulesets:** Because this repository is currently a private organizational repository without GitHub Team subscription, GitHub **DOES NOT** automatically enforce Branch Protection Rules. Specifically:
  - ❌ GitHub will **NOT** block developers from accidentally running `git push origin main`.
  - ❌ GitHub will **NOT** lock the "Merge" button while CI tests are still running or failing.
  - ❌ GitHub will **NOT** show the UI "Update branch" button on outdated PRs.
- **Open Read/Write Access Policy:** To foster active collaboration, we will open Read/Write permissions to all onboarded developers. However, this creates significant risks of build breakage, unstable main branch state, and PR stalls if self-discipline is not maintained.

### Purpose of This Document

Until the repository goes public (at which point standard automated GitHub Branch Protections will be activated and standard [CONTRIBUTING.md](../CONTRIBUTING.md) will take full effect), this document serves as our binding Developer Code of Conduct and Standard Operating Procedure (SOP).

Compliance relies on every developer following these procedures strictly.

> [!WARNING] Access Policy Enforcement
> Because automated branch rulesets are disabled, compliance is monitored manually. Contributors who repeatedly violate these rules (e.g., pushing directly to `main` or merging before CI passes) will have their write permissions revoked and downgraded back to Read-Only access.

---

## 2. One-Time Setup & Prerequisites

Before submitting code, ensure you have `gh` CLI authenticated:

```bash
# Login to GitHub CLI (select github.com and HTTPS or SSH)
gh auth login

# Verify authentication status
gh auth status
```

> [!TIP] Formatting & Linting
> For pre-commit formatting and linting setup, directory structures, and code style rules, refer directly to [CONTRIBUTING.md](../CONTRIBUTING.md).

---

## 3. Code Submission Flow

All contributions during this pre-public phase must strictly follow this 4-step submission lifecycle:

> [!NOTE] PR Target Specification
> Suggest explicitly specifying the target `<PR_NUMBER>` in all `gh` CLI commands (e.g., `gh pr checks 35`, `gh pr view 35`) to avoid ambiguity. If not provided, it will find the `PR_NUMBER` linked to the current branch.

### Phase 1: Create PR

1. **Create Branch & Set Direct Push Guard:** Never commit or push directly to `main`.

   ```bash
   # Switch to main and pull latest changes
   git checkout main && git pull origin main

   # Create feature branch (e.g. username/feature-name)
   git checkout -b username/feature-name

   # Guard against accidental direct push to origin/main locally
   git config branch.main.pushRemote no_push
   ```

2. **Commit with DCO Sign-off (-s):**

   ```bash
   git commit -s -m "feat(kernel): add experimental TPU attention layer"
   ```

#### 💡 Handling DCO Failures
- **CLI Fix (Recommended):** Amend latest commit and force push:

  ```bash
  git commit --amend --signoff --no-edit
  git push origin HEAD --force-with-lease
  ```

- **UI Override (Temporary Pre-Public Option):** During this pre-public stage, repository collaborators can also manually click the DCO check details in the GitHub PR UI and click "Set DCO to PASS" to manually override and pass the check when necessary.
1. **Find Your Reviewers:** GitHub does not auto-request CODEOWNERS on this private repo, so pick reviewers yourself. `scripts/find_owners.py` maps your changed files onto [.github/CODEOWNERS](https://github.com/vllm-project/vllm-torchtpu/blob/main/.github/CODEOWNERS) and prints the smallest set of owners covering every area you touched:

   ```bash
   scripts/find_owners.py             # owners of the current branch's changes
   scripts/find_owners.py --pr 123    # owners of an existing PR
   scripts/find_owners.py --handles   # just a comma-separated handle list
   ```

   Once the PR exists, `--request` requests review from those owners for you instead of you retyping the handles. It uses the current branch's PR unless you name one with `--pr`:

   ```bash
   scripts/find_owners.py --request
   scripts/find_owners.py --pr 123 --request
   ```

1. **Push & Open PR with Assignees & Reviewers:** Because GitHub UI only allows assigning a pending PR to one reviewer for this private repo, you can assign multiple reviewers via `--assignee`, or add multiple reviewers using comma-separated handles:

   ```bash
   git push origin HEAD
   gh pr create --fill --assignee "$(scripts/find_owners.py --handles)"

   # Or name them yourself
   gh pr create --fill --assignee reviewer1,reviewer2
   gh pr edit --add-reviewer reviewer3,reviewer4
   ```

1. **Start CI with the `ready` Label:** Opening a PR does **NOT** start the test suite. Buildkite runs a short bootstrap job that stops unless the PR carries the `ready` label, so work-in-progress branches no longer occupy TPU agents:

   ```bash
   gh pr edit <PR_NUMBER> --add-label ready
   ```

   Applying the label starts a build on its own — no need to push a new commit for it.

### Phase 2: Wait for Test Pass & Review Approval

During this pre-public phase, GitHub does not automatically block merges or show status updates in the usual web UI ways. Pay attention to key differences and rules to watch out for:

- **Missing `ready` Label Fails Fast:** If `buildkite/vllm-torchtpu-ci/pr` fails within seconds reporting `Missing 'ready' label`, that is the presubmit gate, not a test failure. Add the label (see Phase 1) and a fresh build starts automatically.
- **Watch CI Execution (No Automated Merge Blocking):** GitHub will not disable the merge button while CI is running or failing. Developers must manually monitor CI progress until all checks complete green.
  (Refer to [Section 5: Quick Reference Cheat Sheet](#5-quick-reference-cheat-sheet-gh-commands-links) for `gh pr checks <PR_NUMBER> --watch --fail-fast`)
  You can also check the Buildkite dashboard from the UI: [Buildkite TPU Commons](https://screenshot.googleplex.com/36Qsd6vKhWSjEmb)
- **Verify Reviewer Approval:** Ensure at least one reviewer has explicitly approved the PR before merging.
  (Refer to [Section 5: Quick Reference Cheat Sheet](#5-quick-reference-cheat-sheet-gh-commands-links) for approval verification command)
  The **CODEOWNERS Approval** check reports whether every area you touched has an owner's approval — see [Approval Gate (Advisory)](#approval-gate-advisory) below.
- **Check Unresolved Discussion Comments:** Verify that all review comment threads have been addressed and marked as resolved.
  (Refer to [Section 5: Quick Reference Cheat Sheet](#5-quick-reference-cheat-sheet-gh-commands-links) for GraphQL thread query)
- **Update Outdated PR (Missing Web UI Button):** Because the "Update branch" web button is disabled/missing on outdated PRs in this repository setup, update your branch directly via GitHub CLI if main has moved ahead.
  (Refer to [Section 5: Quick Reference Cheat Sheet](#5-quick-reference-cheat-sheet-gh-commands-links) for `gh pr update-branch <PR_NUMBER>`)

### Phase 3: Merge the PR

Once review approval is confirmed **AND** `gh pr checks` returns `PASS` (All Green) for `<PR_NUMBER>`, execute the safe guarded merge command:

```bash
# Verify approval AND CI status on target PR_NUMBER; only merge if both pass
[ "$(gh pr view <PR_NUMBER> --json latestReviews --jq '[.latestReviews[] | select(.state == "APPROVED")] | length')" -gt 0 ] && gh pr checks <PR_NUMBER> && gh pr merge <PR_NUMBER> --squash --delete-branch
```

> [!NOTE] Chaining
> Chaining `[ approved ] && gh pr checks && gh pr merge` guarantees that the merge will abort if the PR has no approvals **OR** if CI checks fail.

### Phase 4: Post Merge Validation & Nightly Build Inspection

After the PR is merged into `main`:

1. **Inspect Nightly & Scheduled Build Results (Buildkite):**
   To verify that your change runs successfully in scheduled nightly TPU benchmarks:
   🔗 **Buildkite Dashboard URL:**
   [Buildkite Scheduled Builds](https://buildkite.com/vllm/vllm-torchtpu-ci/builds?branch=main&query=Scheduled+build)
2. **Verify Main Branch CI Runs (GitHub CLI):** Confirm that recent workflow runs on `main` remain green and stable:

   ```bash
   gh run list --branch main --limit 5
   ```

3. **Clean Up Local Environment:** Sync your local workspace and delete obsolete feature branches:

   ```bash
   git checkout main
   git pull origin main
   git branch -d username/feature-name
   git remote prune origin

   ```

---

## 4. CI Policy

### Pure CPU unit tests

Explicitly mark newly authored pure CPU unit tests with `pytestmark = pytest.mark.cpu_test`
at module scope, or `@pytest.mark.cpu_test` on individual tests in mixed modules.
CI uses this marker to run these tests on CPU queues and exclude them from TPU queues.
Unmarked tests remain eligible for TPU queues.

### Presubmit Requires the `ready` Label

- **Policy:** Presubmit pipelines only run on a PR that carries the `ready` label. Without it, the bootstrap job stops in about ten seconds and no test, build, or benchmark steps are uploaded.
- **Rationale:** Previously every push to a PR branch launched the full suite, so a branch being iterated on could hold TPU agents for a whole test cycle before anyone intended to review it. The label makes that an explicit, deliberate step.
- **Applying it:** `gh pr edit <PR_NUMBER> --add-label ready`. Any onboarded developer (Write access or above) can apply it — this is a signal of intent, not a permission boundary.
- **Re-running:** A label change triggers a build by itself, so you do not need an empty commit to kick CI off. Pushing a new commit to a PR that already has the label rebuilds as usual.
- **Not affected by the gate:**
  - Post-merge builds on `main`, scheduled nightlies, and manual Buildkite builds — these never look at labels.
  - Docs-only PRs — these already short-circuit earlier and need no label.
  - `pre-commit` and `DCO`, which are GitHub checks and run regardless.

> [!NOTE] The Label Does Not Gate Merging
> Because this repository has no branch rulesets (see Section 1), the `ready` label controls only whether CI **runs**. It does not block the merge button. The Pre-Merge Verification Checklist is still what protects `main`.

### Approval Gate (Advisory)

- **Policy:** The `CODEOWNERS Approval` check reports whether every area a PR touches has at least one approval from an owner of that area in [.github/CODEOWNERS](../.github/CODEOWNERS). It resolves ownership with the same matcher as `scripts/find_owners.py`, so its verdict always agrees with what that command tells you to do. (The `Run Approval Gate` check next to it is the job that produces the verdict; it is green whenever the gate managed to run.)
- **Rationale:** GitHub applies CODEOWNERS automatically only on plans this repository is not on (see Section 1), so an approval from someone who happens to be available today does not mean the people who own the code have seen it. The gate makes that gap visible instead of leaving it to the reviewer's memory.
- **It is advisory, not enforcing.** An unmet gate is reported **grey, not red**, so it stays out of the "some checks were not successful" banner and is never confused with a real test failure. Every PR is unmet for the whole window between opening it and being reviewed, and a red X across that window would just train everyone to ignore the check. Treat it as a line item on the Pre-Merge Verification Checklist. Making it block, once branch rulesets become available, takes two changes: set `MISSING_APPROVAL_CONCLUSION: failure` in the workflow, and mark the check required.
- **Reading the result:** The check's details have a table of one row per owned area, who approved it, and who could. An unmet gate names the specific areas that are missing an approval and the handles that would satisfy each one — `scripts/find_owners.py --request` asks exactly those people. The check re-runs on every push and on every review, so approving a PR turns it green without anyone re-triggering CI.
- **Reviewers are requested for you.** When a PR is opened (or marked ready for review), the gate requests review from the smallest set of owners that covers everything the PR touches — the same set `scripts/find_owners.py --request` would pick, and never the author. It does this only on those two events: re-requesting someone whose review is still pending is a no-op, but re-requesting someone who has already reviewed re-adds them and notifies them again, so doing it on every push would nag the whole reviewer list. Add anyone else yourself; the gate never removes a reviewer. Set `AUTO_REQUEST_REVIEWERS: 'false'` in the workflow to turn it off.
- **A broken CODEOWNERS disables assignment.** Before resolving owners the gate asks GitHub to validate the base branch's CODEOWNERS via `/codeowners/errors`. A handle that does not exist, or a user without write access, can neither approve nor receive a review request, and GitHub rejects an entire `requestReviewers` batch if any single entry is bad. When there are errors the gate reports them on the check and skips assignment, rather than silently failing to assign on every PR until someone notices.
- **What counts:**
  - Ownership is read from the **base branch's** CODEOWNERS, matching how GitHub evaluates it: a PR does not get to grant itself ownership of a directory until it merges. The resolver itself runs from the PR, so a change to `scripts/find_owners.py` is exercised by the gate before it lands.
  - Self-approvals are ignored. If you are an area's **only** owner, any other reviewer's approval clears it, so the PR does not deadlock.
  - A stale approval — one submitted against an older head commit — still counts, and is reported as a warning. This is deliberate: the DCO fix in Phase 1 (`git commit --amend --signoff` plus a force push) rewrites the head SHA, and dismissing approvals on that would be noise. Flip `STRICT_STALE_APPROVALS` in the workflow to change this.
- **Override:** Add the `approval-gate-override` label to bypass the gate in an emergency (for example, reverting a broken `main` when the owner is unreachable). Every use is logged as a warning on the run; explain it in the PR.

### Addressing Existing Regression in main

- **Policy:** Should the `main` branch currently exhibit failures (e.g., in nightly TPU benchmarks or post-submit runs), merging remains permissible provided your PR passes all required presubmit validations.
- **Rationale:**
  - **Unblocked Development Speed:** Engineering velocity must not be hindered by pre-existing, unrelated build breakages in `main`.
  - **Workflow Consistency:** This approach mirrors modern CI/CD practices where excessive "merge friction" is mitigated by allowing merges without constant head synchronization.

### Emergency Reversion Policy (Revert-on-Breakage)

- **Policy:** If a merged contribution is identified as the root cause of a `main` branch failure (detected via integration suites or nightly benchmarks), the mandatory immediate action is a PR rollback.
- **Why Rollbacks are Mandatory (No "Fix-Forward"):**
  - **Maintenance Efficiency:** Core maintainers lack the capacity to troubleshoot or patch downstream regressions introduced by incoming PRs.
  - **Rapid Stability Recovery:** Reversion is the most efficient mechanism to restore a green `main` state and unblock the broader development team.
  - **Avoid Error Propagation:** Attempting to "fix-forward" often results in secondary regressions or masks the underlying issue, complicating future root-cause analysis.
  - **Accountability:** The original author is responsible for debugging the regression in their local environment before submitting a sanitized PR.

---

## 5. FAQ & Best Practices

### Question: How do I skip CI and merge a PR when necessary?

Running CI checks before merging is strongly recommended. If you encounter bugs or issues with the CI/CD pipeline, please file an issue on [GitHub Issues](https://github.com/vllm-project/vllm-torchtpu/issues/new?assignees=maxwillzq).

In urgent situations where you must unblock yourself immediately:
- Prefix your PR title with `[skip-ci]`. So our automation daily email will know that and not flag it.
- Detail the exact reason for skipping CI in the PR description.

> [!NOTE]
> Please use this bypass sparingly and only when strictly necessary. We may reach out offline for additional context.

> [!TIP] `[skip-ci]` PR Title vs `[skip ci]` Commit Message
> These are two different mechanisms. The `[skip-ci]` **PR title** prefix above is for our daily email automation. A `[skip ci]` (or `[ci skip]`) marker in the **commit message** is understood natively by both Buildkite and GitHub Actions, so no build is created at all — including `pre-commit`. Note that a squash merge can carry that marker into `main`, which would also skip the post-merge build; edit the subject at merge time if you do not want that.
> Simply leaving the `ready` label off is now the lightest way to keep a PR from consuming CI.

---

## 6. Quick Reference Cheat Sheet (gh Commands & Links)

| Objective | Command / Direct Link |
| :--- | :--- |
| **Phase 1: Find Reviewers (CODEOWNERS)** | `scripts/find_owners.py` (or `--pr <PR_NUMBER>`) |
| **Phase 1: Request Those Reviewers** | `scripts/find_owners.py --request` (or `--pr <PR_NUMBER> --request`) |
| **Phase 1: Create PR & Assign** | `gh pr create --fill --assignee "$(scripts/find_owners.py --handles)"` |
| **Phase 1: Add Multiple Reviewers** | `gh pr edit <PR_NUMBER> --add-reviewer reviewer3,reviewer4` |
| **Phase 1: Start CI (`ready` label)** | `gh pr edit <PR_NUMBER> --add-label ready` |
| **Phase 2: Check PR Approval** | `gh pr view <PR_NUMBER> --json latestReviews --jq 'if ([.latestReviews[] \| select(.state == "APPROVED")] \| length > 0) then "APPROVED ✅" else "NOT APPROVED ❌" end'` |
| **Phase 2: Check Resolved Discussions** | `gh api graphql -F owner='vllm-project' -F repo='vllm-torchtpu' -F pr=<PR_NUMBER> -f query='query($owner: String!, $repo: String!, $pr: Int!) { repository(owner: $owner, name: $repo) { pullRequest(number: $pr) { reviewThreads(first: 50) { nodes { isResolved } } } } }' --jq 'if ([.data.repository.pullRequest.reviewThreads.nodes[] \| select(.isResolved == false)] \| length == 0) then "RESOLVED ✅" else "UNRESOLVED ❌" end'` |
| **Phase 2: Watch CI (Fail Fast)** | `gh pr checks <PR_NUMBER> --watch --fail-fast` |
| **Phase 2: Update Outdated Branch** | `gh pr update-branch <PR_NUMBER>` |
| **Phase 3: Guarded Merge (Approved + Green)** | `[ "$(gh pr view <PR_NUMBER> --json latestReviews \| jq '[.latestReviews[] \| select(.state == "APPROVED")] \| length')" -gt 0 ] && gh pr checks <PR_NUMBER> && gh pr merge <PR_NUMBER> --squash --delete-branch` |
| **Phase 4: Main Branch CI Check (CLI)** | `gh run list --branch main --limit 5` |
| **Phase 4: Buildkite Nightly URL** | [Buildkite Scheduled Builds](https://buildkite.com/vllm/vllm-torchtpu-ci/builds?branch=main&query=Scheduled+build) |
| **Fix Missing DCO (CLI)** | `git commit --amend --signoff --no-edit && git push origin HEAD --force-with-lease` |
| **Fix Missing DCO (UI Override)** | Click DCO check details in GitHub PR UI -> Click "Set DCO to PASS" |

---

No access to the repo? Fill [this Google Form](https://docs.google.com/forms/d/e/1FAIpQLSftqPghWScccs-3bDQNCEqPg2sLvIyUKS4GjVAWzNj9m5QV-w/viewform?usp=header) to request it.
