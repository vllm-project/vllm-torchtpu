# Buildkite pipelines (epic/dsv41)

CI runs on the Kueue fleet: `bootstrap.sh` (`bootstrap.py`) uploads
`pipeline_build.yml` and `pipeline_dsv41.yml`, and every TPU step runs in a pod
through `.buildkite/kubernetes/run.sh` or `run_multihost.sh`. The `kubernetes/`
tree and the launcher scripts match `main`; keep them that way so syncs from
`main` stay mechanical.

This branch pins an older vLLM and targets DeepSeek-V4.1 alone, so it does not
run `main`'s `cpu`, `tests`, `integration`, `perf` or `verify` lanes. Every
build that runs TPU uploads the image build and the `dsv41` lane in one stage;
the dsv41 steps wait on `build_docker` themselves. `OMNI_IMAGE_SKIP` is always
set, so the `-omni` image is never built. There is no integration pipeline, LKG
promotion or release publishing on this branch.

PR builds still need the `ready` label. These labels apply on a PR:

- `oncall-fix` raises job and Kueue workload priority without changing what runs.
- `skip-tpu-ci` uploads only the image build. (Merged-PR label lookup only
  happens on `main` post-merge builds, so a push to `epic/dsv41` always runs
  the lane.)
- `run-full-ci` runs the same lanes with `NIGHTLY=1`.
- `run-perf-eval` is accepted and adds nothing: the dsv41 lane carries its own
  benchmark step.

`skip-tpu-ci`, `run-perf-eval` and `run-full-ci` are mutually exclusive;
`bootstrap.py` fails the build if more than one is set. Merge-queue and `/ci`
builds (`CI_RUN_TPU`, `CI_PRECHECK`, `CI_EXTRAS`) select suites the same way as
on `main`; a precheck runs the image build only. Scheduled builds with
`NIGHTLY=1` and manual builds with `CI_SUITE=full` run the lane with
`NIGHTLY=1`; `CI_SUITE=default` runs it without.

`pipeline_dev.yml` and `pipeline_dynamic.yml` are the `vllm-torchtpu-dev`
pipeline's files (`bootstrap_dev.sh`), unchanged from `main`.
