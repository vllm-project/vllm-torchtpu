# Agent Instructions for vLLM TPU

## Repository standards

Every change must follow these documents. Reviewers, including the review bot,
check PRs against them.

- `docs/developers_guide/coding_guidance.md` (R-1.x through R-8.x coding style
  and hygiene)
- `docs/developers_guide/optimization_best_practices.md` (TPU performance and
  memory invariants)
- `docs/developers_guide/type_checking.md` (pyrefly and ruff checks: running
  them locally before a PR, fixing new errors, baseline rules)

## Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm-torchtpu --comments
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

## Small, single-purpose PRs

Follow [Google's Small CLs guide](https://google.github.io/eng-practices/review/developer/small-cls.html)
and `R-7.1`–`R-7.3` in `docs/developers_guide/coding_guidance.md`. Small,
focused PRs are faster to review thoroughly, make bugs easier to catch, cause
fewer merge conflicts, and are safer to revert.

Before opening a PR, inspect the diff size and scope:

```bash
git diff --stat origin/main...HEAD
```

- **Size and cohesion (`R-7.1`–`R-7.3`)**: Aim for one self-contained logical
  change within `R-7.1`'s soft target (**≤ 300 changed lines**, excluding
  baselines and lockfiles).
  - Separate mechanical or behavior-preserving refactors from behavioral
    changes (`R-7.2`), and omit unrelated cleanups or drive-by edits (`R-7.3`).
  - Break larger end-to-end work into independently testable, reviewable steps
    (for example, landing a self-contained building block with its tests before
    broader integration). Pure deletions, atomic renames/moves with their
    caller updates, tiny dependent follow-ups, and cohesive changes that cannot
    be split cleanly may stay in one PR.
- **Stacked PRs workflow**: Each PR in a stack must build and pass tests on its
  own without relying on code added in later PRs, and link the other PRs in the
  stack.
  - **With push access to `vllm-project/vllm-torchtpu`**: Push each branch to
    the main repo and set `--base` on PR 2 to PR 1's branch so PR 2's diff shows
    only PR 2's changes:

    ```bash
    # PR 1 (base change off origin/main)
    git checkout -b feat/part-1 origin/main
    git commit -s -m "..."
    git push -u origin feat/part-1
    gh pr create --base main --head feat/part-1

    # PR 2 (stacked on PR 1)
    git checkout -b feat/part-2 feat/part-1
    git commit -s -m "..."
    git push -u origin feat/part-2
    gh pr create --base feat/part-1 --head feat/part-2
    ```

  - **From a fork**: GitHub requires `--base` to be a branch in
    `vllm-project/vllm-torchtpu`, so open PR 2 with `--base main` and state
    `Depends on #<PR 1>` at the top of its description (its diff will include
    PR 1's commits until PR 1 merges).
  - **Updating a stack**:

    ```bash
    # 1. When PR 1 gains new commits: rebase PR 2 onto PR 1
    git checkout feat/part-2
    git rebase feat/part-1

    # 2. When rebasing the unmerged stack onto newer origin/main (run from the top branch)
    git fetch origin main
    git checkout feat/part-2
    git rebase --update-refs origin/main

    # 3. After PR 1 is squash-merged into main: drop PR 1's commits and retarget PR 2
    git fetch origin main
    git rebase --onto origin/main feat/part-1 feat/part-2
    gh pr edit <pr2_number> --base main
    ```

## Development workflow

### Running pre-commit

Install both hook types in the active project virtual environment:

```bash
pre-commit install --hook-type pre-commit --hook-type commit-msg
```

Run the relevant checks before proposing a PR:

```bash
# Run all pre-commit hooks on staged files:
pre-commit run

# Run on all files:
pre-commit run --all-files

# Run a specific hook:
pre-commit run ruff-check --all-files
```

### DCO sign-off

Every commit must include a Developer Certificate of Origin sign-off. Create
signed commits with `git commit --signoff` (or `git commit -s`) and verify that
the commit message contains this trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```
