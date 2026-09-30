# Agent Instructions for vLLM TPU

## Repository standards

- `docs/coding_guidance.md` (R-1.x through R-8.x coding style and hygiene)
- `docs/optimization_best_practices.md` (TPU performance and memory invariants)

## Duplicate-work checks

Before proposing a PR, run these checks:

```bash
gh issue view <issue_number> --repo vllm-project/vllm-torchtpu --comments
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<issue_number> in:body"
gh pr list --repo vllm-project/vllm-torchtpu --state open --search "<short area keywords>"
```

- If an open PR already addresses the same fix, do not open another.
- If your approach is materially different, explain the difference in the issue.

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

Run ruff and pyrefly through pre-commit, not from `PATH`: a `ruff` or
`pyrefly` on `PATH` may differ from the pinned version and report different
results.

### Type checking

Two pyrefly checks gate every PR:

- The `pyrefly` pre-commit hook runs without third-party packages, so every
  import of torch, jax or vllm is `Any`. Its baseline is
  `.pyrefly-baseline.json`.
- Buildkite's "Type Check - Full Environment" runs `tools/typecheck_full.sh`
  in the CI image, where calls into torch, jax and vllm are checked against
  their real signatures. Its baseline is `.pyrefly-baseline-full.json`.

Before proposing a PR that changes `src/`, run both:

```bash
pre-commit run --all-files
bash tools/typecheck_full.sh
```

Other package versions report different errors, so `tools/typecheck_full.sh`
reruns itself inside the nightly `torchtpu-vllm-dev` image when started outside
a container. It needs Docker and a one-time
`gcloud auth configure-docker us-central1-docker.pkg.dev`, but no TPU.

A local pass does not guarantee that Buildkite passes. Buildkite checks the
image it builds from your PR's commit, and the nightly image can differ from
it:

- The nightly image is built from `main` once a day, so it is usually up to
  a day behind `main`, and further behind when a nightly build fails. The
  script pulls it on every run and prints its build date.
- It does not include dependency changes made in your PR.
- Dependencies without a pinned version can resolve to a newer release in a
  later build.

If Buildkite's check fails but the local run passes, rerun locally in the
exact image Buildkite used. The ":docker: Build and Push CI Image" step
prints its tag:

```bash
TYPECHECK_IMAGE=us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu:<commit>-<vllm-ref> \
  bash tools/typecheck_full.sh
```

To fix a new pyrefly error:

- Fix the code. Do not add new errors to either baseline.
- Keep runtime behavior unchanged. Do not add `try`/`except` or fallback
  values to silence an error. Common fixes:
  - `assert x is not None` directly before the use, in the same branch, when
    setup guarantees the value.
  - Narrow into a local variable and assert before storing it on `self`, so
    the attribute keeps the narrow type.
  - `typing.cast` where vLLM declares a looser type than the value the code
    receives.
  - Initialize a variable before a conditional block instead of relying on
    every branch to set it.
- Use an own-line `# pyrefly: ignore[<error-code>]` only for false positives,
  such as narrowing an inherited attribute (`bad-override-mutable-attribute`).
  Always name the error code.

Regenerate the baselines only to remove entries for errors your change fixed.
A stale entry does not fail the checks, but it can hide a later error with the
same message in the same file. Regenerating records every current error, so
first make both checks pass without it. Then regenerate and run pre-commit
(pyrefly writes the files without a trailing newline):

```bash
uvx pyrefly@1.3.0 check --update-baseline
bash tools/typecheck_full.sh --update-baseline
```

The baseline diff must only delete entries. An added entry means your change
introduced that error; fix it instead. The one exception is a moved or renamed
file, whose entries move with it.

### DCO sign-off

Every commit must include a Developer Certificate of Origin sign-off. Create
signed commits with `git commit --signoff` (or `git commit -s`) and verify that
the commit message contains this trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```
