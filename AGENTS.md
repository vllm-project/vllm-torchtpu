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
and `R-7.1`–`R-7.4` in `docs/developers_guide/coding_guidance.md`. Before
opening a PR, inspect the diff size and file spread:

```bash
git diff --stat origin/main...HEAD
```

- **Scope and split boundaries (`R-7.1`, `R-7.2`)**: Aim for one self-contained
  logical change within `R-7.1`'s soft target (**≤ 300 changed lines**,
  excluding baselines and lockfiles). Pure deletions, atomic renames/moves with
  their caller updates, and tiny dependent follow-ups may stay in one PR; when a
  diff exceeds ~300 changed lines and spans multiple layers, split it along
  `R-7.1`/`R-7.2`:
  1. **Refactor first, feature second (`R-7.2`)**: Land mechanical refactors,
     signature changes, or helper moves in their own behavior-preserving PR
     before the feature or optimization PR that uses them.
  2. **Kernel vs. integration (`R-7.1`)**: Land a new or modified Pallas kernel
     under `src/vllm_torchtpu/kernels/` with its standalone kernel unit test in
     PR 1; wire it into `layers/`, `runner/`, or `models/` in PR 2.
  3. **Shared plumbing vs. model bring-up (`R-7.1`)**: Land shared changes in
     `runner/`, `layers/`, or `quantization/` first; land model-specific files
     under `models/` in a follow-up PR.
- **No test bloat (`R-8.3`, `R-8.8`)**: Do not clone near-identical test
  functions; use `@pytest.mark.parametrize` and shared helpers, keep each test
  focused on one behavior on the smallest hardware that exercises it (`R-8.5`),
  and if the test diff dwarfs the code change, cut scope (`R-8.8`).
- **Stacked PRs workflow**: Each PR in a stack must build and pass tests on its
  own without relying on code added in later PRs, and link the other PRs in the
  stack.
  - **With push access to `vllm-project/vllm-torchtpu`**: Push each branch to
    the main repo and set `--base` on PR 2 to PR 1's branch so PR 2's diff shows
    only PR 2's changes:

    ```bash
    # PR 1 (base change off origin/main)
    git checkout -b feat/part-1-refactor origin/main
    git commit -s -m "..."
    git push -u origin feat/part-1-refactor
    gh pr create --base main --head feat/part-1-refactor

    # PR 2 (stacked on PR 1)
    git checkout -b feat/part-2-wiring feat/part-1-refactor
    git commit -s -m "..."
    git push -u origin feat/part-2-wiring
    gh pr create --base feat/part-1-refactor --head feat/part-2-wiring
    ```

  - **From a fork**: GitHub requires `--base` to be a branch in
    `vllm-project/vllm-torchtpu`, so open PR 2 with `--base main` and state
    `Depends on #<PR 1>` at the top of its description (its diff will include
    PR 1's commits until PR 1 merges).
  - **Updating a stack**:

    ```bash
    # 1. When PR 1 gains new commits: rebase PR 2 onto PR 1
    git checkout feat/part-2-wiring
    git rebase feat/part-1-refactor

    # 2. When rebasing the unmerged stack onto newer origin/main (run from the top branch)
    git fetch origin main
    git checkout feat/part-2-wiring
    git rebase --update-refs origin/main

    # 3. After PR 1 is squash-merged into main: drop PR 1's commits and retarget PR 2
    git fetch origin main
    git rebase --onto origin/main feat/part-1-refactor feat/part-2-wiring
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
