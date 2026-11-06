# Buildkite pipelines

CI runs on the Kueue fleet: `bootstrap.sh` uploads the `pipeline_<lane>.yml`
lanes (`tests`, `integration`, `perf`, `verify`), and every step runs in a pod
through `.buildkite/kubernetes/run.sh`.

The `tests` lane is two files. `pipeline_tests_cpu.yml` runs the type check and
unmarked unit tests in `tests/` in CPU pods, and `pipeline_tests.yml` runs
tests marked `tpu_test` (or `multichip`) on TPU v7x slices (1x1x1 and 2x2x1)
and, side by side, the same tests on TPU v6e slices (1x1 and 2x4): bootstrap
uploads it once per generation in `TPU_GENERATION_FILES`. Every step's label
starts with its generation (`v6e ...`, `v7x ...`), v6e's keys carry a `tpu6e_`
prefix while v7x's keep their names, and the v6e steps soft-fail until the
tests that fail rather than skip on v6e skip there. A new TPU step's label
starts with `${TPU_GENERATION:-v7x}`, so each generation gets its own GitHub
status, and its key with `${TPU_KEY_PREFIX:-}`, or the second generation's
upload is rejected for a duplicate key.
PR steps exclude `nightly` tests; nightly steps include them. Both use the CI
image built by `pipeline_build.yml`. The CPU steps set `JAX_PLATFORMS=cpu` so
JAX does not look for a TPU.

The `integration` lane, `pipeline_integration.yml`, goes up the same way, once
per generation in `TPU_GENERATION_FILES`, with the same names and soft-fail
on v6e: its eight-core steps take a v6e 2x4 slice where v7x's take a 2x2x1.
Steps that run only on v7x as written - those that go through
`run_multihost.sh`, whose topology and JobSets assume 4-chip v7x hosts, and
those that set `TPU_ACCELERATOR_TYPE=tpu7x` - carry an `if` comparing
`TPU_GENERATION` with `v7x`, so v6e's upload leaves them out.

A build started by hand or over the API can run one generation alone: with
`TPU_VERSION=tpu6e` or `TPU_VERSION=tpu7x`, each file in `TPU_GENERATION_FILES`
goes up for that generation only, with its slice shapes, so the build sets no
`KUBE_SHAPE_*`; on a nightly, the verify lane's `upload_models_and_features.sh`
sends up only that generation's group. Without it every generation runs. A
lane's file follows `TPU_VERSION` once it is in `TPU_GENERATION_FILES`; until
then it goes up as it is. A nightly started by hand leaves out the nightly
report, which needs every generation's steps. Integration, scheduled and tag
builds refuse `TPU_VERSION` - LKG promotion must not pass on one generation,
and a schedule or a release checks them all - and so does a build with no lane
that follows it.

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

The perf lane goes up once per generation too. A v6e 2x4 slice holds 256 GB of
HBM to a v7x-8's 768 GB, so v6e runs only what fits and needs neither fp4 nor
`run_multihost.sh`: a General suite of the four configs that fit, qwen3-0.6b,
and the nightly general suite. Every other step carries an `if` comparing
`TPU_GENERATION` with `v7x`. The steps set `TPU_ACCELERATOR_TYPE` from the
generation, since the server reads the chip from it and the results upload
names the device after it. The perf baselines were all measured on v7x, so
v6e's steps set `SKIP_PERF_CHECK=1` and report their numbers without that
gate; the accuracy gate still applies.

A PR labeled `skip-tpu-ci` (it still needs `ready`) runs only the `tests` lane
and the image build, and the lane's `Unit Tests on TPU` groups are skipped
through `SKIP_TPU_CI`, so what runs is the type check and the CPU unit tests.
The post-merge build of such a PR does the same: `bootstrap.sh` reads the
merged PR's labels from GitHub. Keep TPU steps of the `tests` lane inside that
group, and its CPU steps in `pipeline_tests_cpu.yml`.

Releases and nightly images are published by the `vllm-torchtpu-release`
pipeline in the TPU Release cluster; [`release/README.md`](https://github.com/vllm-project/vllm-torchtpu/blob/main/.buildkite/release/README.md)
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
