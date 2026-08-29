# Buildkite CI/CD Pipelines

CI for this repo runs on Buildkite: pipeline
[`vllm-torchtpu-ci`](https://buildkite.com/tpu-commons/vllm-torchtpu-ci) in the
`tpu-commons` organization. Builds trigger on branch pushes and pull requests,
plus a nightly scheduled build on `main` at 2:00 AM America/Los_Angeles with
`NIGHTLY=1`. Pull requests from forks do not trigger builds.

## How a build works

The pipeline's only fixed step runs [`scripts/bootstrap.sh`](scripts/bootstrap.sh),
which:

1. Determines the job priority for the build type (post-merge on `main` >
   pre-merge PR > default; constants in
   [`scripts/configs/pipeline_config.sh`](scripts/configs/pipeline_config.sh)).
   That priority applies to the build and unit-test pipelines; integration and
   perf always get their own fixed, lower priorities (3 and 2). Nightly builds
   drop everything to 0 (lowest).
2. Uploads the four pipeline YAMLs below, injecting those priorities.
   They are uploaded in reverse order so they display as Build → Tests →
   Integration → Perf.

| File | What it runs | Queue |
|---|---|---|
| [`pipeline_build.yml`](pipeline_build.yml) | Builds the CI Docker image ([`../docker/build_image.sh`](../docker/build_image.sh) `--target dev`), tags it with the build commit, and pushes it to Artifact Registry. Note the image builds FROM the floating base `torch-tpu-base:nightly-latest`, so the same commit can produce different images across days — the first thing to suspect when CI goes red with no code change. All other jobs depend on this step. | `cpu_64_core` |
| [`pipeline_tests.yml`](pipeline_tests.yml) | Unit tests, split into speculative-decoding tests and everything else. Non-nightly builds (PRs, post-merge, manual) run `pytest -m "not nightly"`; nightly builds run the full set. | `tpu_v7x_2_queue` / `tpu_v7x_8_queue` |
| [`pipeline_integration.yml`](pipeline_integration.yml) | Integration cases (e.g. qwen3.5 P4D2 disaggregated correctness). | `tpu_v7x_8_queue` |
| [`pipeline_perf.yml`](pipeline_perf.yml) | Perf + eval flows ([`../scripts/vllm/benchmarking/run_eval_flow.sh`](../scripts/vllm/benchmarking/run_eval_flow.sh), lm-eval/evalplus). Small configs run on every build; large models (480B, 397B) are nightly-only. Results are uploaded as Buildkite artifacts under `perf_eval_results/**`. | `tpu_v7x_2_queue` / `tpu_v7x_8_queue` |

Steps gated with `if: build.env("NIGHTLY") == "1"` never run on regular PR
builds; the Buildkite API reports them with state `broken` (its term for steps
filtered out by an `if:` condition — distinct from the `skipped` state used
for `skip:` attributes and superseded builds). The UI shows them grayed out.
Either way it is expected, not a failure.

## Test execution on TPU agents

Every TPU step goes through [`scripts/run_in_docker.sh`](scripts/run_in_docker.sh),
which:

- Pulls the commit-tagged image from
  `us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu-ci/vllm-torchtpu`.
- Runs the given command in a privileged container with host networking,
  mounting `/mnt/disks/persist/models` as the HF cache and
  `/mnt/disks/persist/perf_eval_results` for results.
- Sources `HF_TOKEN` from GCP Secret Manager (`bm-agent-hf-token`) on first use.
- Copies results back into the workspace for artifact upload and removes the
  pulled image afterwards.

## Finding the build for a PR

Builds are keyed by branch. Filter builds by the PR's head branch and match the
commit SHA to the PR HEAD:

```bash
curl -s -H "Authorization: Bearer $BUILDKITE_API_TOKEN" \
  "https://api.buildkite.com/v2/organizations/tpu-commons/pipelines/vllm-torchtpu-ci/builds?branch=<HEAD_REF>&per_page=5"
```
