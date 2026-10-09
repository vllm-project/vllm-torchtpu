# Buildkite pipelines

`pipeline_tests.yml` runs unmarked unit tests in `tests/` on `cpu_64_core` by
default and runs tests marked `tpu_test` (or `multichip`) on the TPU queues.
PR steps exclude `nightly` tests; nightly steps include them. Both use the CI
image built by `pipeline_build.yml`.
The CPU steps set `CPU_ONLY=1` to run Docker without privileged mode or device
mounts and select the JAX CPU backend. Host model caches and test results
use `$HOME/.cache/vllm-torchtpu-ci` on CPU agents and `/mnt/disks/persist`
on TPU agents.

`pipeline_perf.yml` (`pipeline_perf_kube.yml` on kube) holds the perf/eval
steps. Post-merge builds do not run them, and PR builds run them only when the
PR has the `run-perf-eval` label; adding that label to a PR that already has
`ready` starts a build that includes them. The 02:00 PT nightly runs
all of them. Between nightlies, schedules on `vllm-torchtpu-ci` that set
`CI_LANES=perf` run the perf lane alone; without `NIGHTLY` that is only the
steps not gated on it. The schedules keep clear of the hours the nightly's
perf steps take. The integration pipeline also runs the lane, because its LKG
promotion waits on two of its steps.

The `vllm-torchtpu-nightly-image` pipeline runs `pipeline_nightly_image.yml`,
which builds the ci, dev and prod images with
`scripts/publish_nightly_images.sh`. It has no git triggers; its schedules pick
the branch. A build of `main` pushes `nightly-YYYYMMDD` and `latest`, and a
build of `release/X.Y.Z` pushes `YYYYMMDD-X.Y.Z` and `X.Y.Z` (see
`docs/getting_started/docker.md`). To start nightly images for a new release
branch, add a schedule on that pipeline with the branch set to `release/X.Y.Z`.

Unit tests run on CPU by default. For a test file that requires a single-host
TPU, add `pytestmark = pytest.mark.tpu_test` after its imports (or
`pytestmark = pytest.mark.multichip` for tests requiring a multi-chip slice).
Use `@pytest.mark.tpu_test` for individual TPU tests in mixed files. Module
imports must work on CPU because pytest collects tests before applying marker
filters. Mocks that use `torch.tpu` must supply that object on CPU hosts. Tests
that take the `device` fixture also need `tpu_test`: `--use-tpu` defaults to on,
so the fixture asks for `torch.device("tpu")` and fails on a CPU host.

`nightly` only decides whether a test runs on PRs; it says nothing about
hardware. A nightly test that needs a TPU still carries `tpu_test` or
`multichip`, otherwise the CPU nightly step collects it and it fails there.

Run the default CPU unit tests locally with the project's test dependencies installed:

```bash
JAX_PLATFORMS=cpu pytest tests/ -m "not tpu_test and not multichip and not nightly"
```
