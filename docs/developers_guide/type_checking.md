# Type checking

Two pyrefly checks gate every PR:

- The `pyrefly` pre-commit hook runs without third-party packages, so every
  import of torch, jax or vllm is `Any`. Its baseline is
  `.pyrefly-baseline.json`.
- Buildkite's "Type Check - Full Environment" runs `tools/typecheck_full.sh`
  in the CI image, where calls into torch, jax and vllm are checked against
  their real signatures. Its baseline is `.pyrefly-baseline-full.json`.

Run ruff and pyrefly through pre-commit, not from `PATH`: a `ruff` or
`pyrefly` on `PATH` may differ from the pinned version and report different
results.

## Running the checks locally

Before proposing a PR that changes `src/`, run both:

```bash
pre-commit run --all-files
bash tools/typecheck_full.sh
```

Other package versions report different errors, so `tools/typecheck_full.sh`
reruns itself inside the nightly `vllm-torchtpu-dev` image when started outside
a container. It needs Docker, read access to the image's Artifact Registry
project (`cloud-ullm-inference-ci-cd`) and a one-time
`gcloud auth configure-docker us-central1-docker.pkg.dev`, but no TPU.

Without access to that registry, build the same image locally and point the
script at it:

```bash
./docker/build_image.sh --target dev -t vllm-torchtpu-dev-local
TYPECHECK_IMAGE=vllm-torchtpu-dev-local bash tools/typecheck_full.sh
```

The build needs only the torch-tpu registry access that installing
`vllm-torchtpu` already requires (see the README). The script uses a local
image as it is, so rebuild it after `main` or your PR changes dependencies.

## When Buildkite disagrees with a local run

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

## Fixing a new error

When either check reports a new error:

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

## Regenerating the baselines

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
