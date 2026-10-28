# Buildkite pipelines

CI runs on the Kueue fleet: `bootstrap.sh` uploads the `pipeline_<lane>.yml`
lanes (`tests`, `integration`, `perf`, `verify`), and every step runs in a pod
through `.buildkite/kubernetes/run.sh`.

`pipeline_tests.yml` runs unmarked unit tests in `tests/` in CPU pods and
tests marked `tpu_test` (or `multichip`) on TPU v7x slices (1x1x1 and 2x2x1).
PR steps exclude `nightly` tests; nightly steps include them. Both use the CI
image built by `pipeline_build.yml`. The CPU steps set `JAX_PLATFORMS=cpu` so
JAX does not look for a TPU.

`pipeline_perf.yml` holds the perf/eval
steps. Post-merge builds do not run them, and PR builds run them only when the
PR has the `run-perf-eval` label (or `run-full-ci`, which sets `NIGHTLY=1` and
runs the full nightly suite across every lane: `tests`, `integration`, `perf`
and `verify`); adding either label to a PR that already has
`ready` starts a build that includes them. The 02:00 PT nightly runs
all of them. Between nightlies, schedules on `vllm-torchtpu-ci` that set
`CI_LANES=perf` run the perf lane alone; without `NIGHTLY` that is only the
steps not gated on it. The schedules keep clear of the hours the nightly's
perf steps take. The integration pipeline also runs the lane, because its LKG
promotion waits on two of its steps.

A PR labeled `skip-tpu-ci` (it still needs `ready`) runs only the `tests` lane
and the image build, and the lane's `Unit Tests on TPU` group is skipped through
`SKIP_TPU_CI`, so what runs is the type check and the CPU unit tests. The
post-merge build of such a PR does the same: `bootstrap.sh` reads the merged
PR's labels from GitHub. Keep TPU steps of the `tests` lane inside that group.

Releases and nightly images are published by the `vllm-torchtpu-release`
pipeline in the TPU Release cluster; [`release/README.md`](release/README.md)
has how to cut a release. Its nightly step runs
`scripts/publish_nightly_images.sh` on `main`, pushing the `ci`, `dev` and
`prod` images to GAR as `nightly-YYYYMMDD` and `latest`, and the `dev` image to
Docker Hub ([`vllm/vllm-torchtpu`](https://hub.docker.com/r/vllm/vllm-torchtpu/tags))
as `nightly-YYYYMMDD-<commit>-<vllm>` and `nightly`. A release publishes the
`prod` image as `vX.Y.Z`, and `latest` if the approver chooses (see
`docs/getting_started/docker.md`). `pipeline_nightly_image.yml` is the same
nightly step for the `vllm-torchtpu-nightly-image` pipeline, which runs until
the new pipeline's schedule takes over.

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
