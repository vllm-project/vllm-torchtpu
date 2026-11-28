# Buildkite pipelines

CI runs on the Kueue fleet: `bootstrap.sh` (`bootstrap.py`) uploads the
`pipeline_<lane>.yml` lanes (`cpu`, `tests`, `integration`, `perf`, `verify`),
and every step runs in a pod through `.buildkite/kubernetes/run.sh`.

## Pipeline catalog

| Pipeline | Trigger / Schedule | Suite / Env | Purpose |
| --- | --- | --- | --- |
| `vllm-torchtpu-ci` | PRs, merge queue, post-merge `main`, every-4h `CI_SUITE=perf`, daily 02:00 PT `NIGHTLY=1` | PR two-stage suites, `default`, `perf`, `full` | Core unit, integration, perf/eval, and model-matrix tests |
| `vllm-torchtpu-canary-vllm` | Daily 16:00 PT | `vllm-canary` (auto-selected by slug) | Tests upstream vLLM `HEAD` and promotes LKG |
| `vllm-torchtpu-canary-torchtpu` | Daily 02:00 UTC | `default` + `torch-tpu` leg (auto-selected by slug); schedule adds `NIGHTLY=1` | Daily `torch-tpu` wheel canary |
| `vllm-torchtpu-canary-tpu-sync` | Daily + API trigger from `tpu-sync` golden workflow | `default` + `tpu-sync` leg (auto-selected by slug) | Daily and on-demand `tpu-sync-torch` wheel canary |
| `vllm-torchtpu-release` | Scheduled `main` (nightly images) and manual `releases/vX.Y.Z` branches (TPU Release cluster) | `release/pipeline.yml` | Publishes nightly `ci`/`dev`/`prod` images on `main` and cuts official releases on `releases/vX.Y.Z` |
| `vllm-torchtpu-dev` | Manual / experimental | `bootstrap_dev.sh` | Developer sandbox |

**Wheel-canary trigger inputs and precedence:** each wheel-canary build layers a
single package leg (`torch-tpu` or `tpu-sync`). `bootstrap.py` (`bootstrap.sh`
and `bootstrap_dev.sh`) resolves, normalizes, and validates the leg once and
publishes canonical `torch-tpu` or `tpu-sync` to Buildkite meta-data
(`TORCHTPU_NIGHTLY_LEG`), which `setup_docker_env.sh` reads:

1. `TORCHTPU_NIGHTLY_LEG` (`torch-tpu` or `tpu-sync`; also accepts `torch_tpu`,
   `tpu_sync`, and `tpu-sync-torch`): explicit leg override; beats the pipeline
   slug's default leg and starts a canary build on its own (without
   `TORCHTPU_NIGHTLY=1`).
2. Pipeline slug default: `vllm-torchtpu-canary-torchtpu` selects `torch-tpu`;
   `vllm-torchtpu-canary-tpu-sync` selects `tpu-sync`.
3. `TORCH_TPU_VERSION_OVERRIDE` / `TPU_SYNC_TORCH_VERSION_OVERRIDE`: pins the
   wheel version for `torch-tpu` or `tpu-sync-torch` instead of querying the
   Artifact Registry index. Setting one without an explicit or slug leg (and
   without `TORCHTPU_NIGHTLY=1`) starts a canary build for that package's leg;
   setting both, or setting an override for the other leg, fails `bootstrap.py`
   before any steps upload.
4. `TORCHTPU_NIGHTLY=1`: legacy boolean canary flag; defaults the leg to
   `torch-tpu` when none of the inputs above selects a leg.

**Migration / legacy names:** legacy canary slugs `vllm-torchtpu-integration`
(aliased to `vllm-torchtpu-canary-vllm`, plus `CI_SUITE=integration` ->
`vllm-canary`) and `vllm-torchtpu-torchtpu-nightly` (replaced by
`vllm-torchtpu-canary-torchtpu` and `vllm-torchtpu-canary-tpu-sync`) remain
supported during cut-over. `vllm-torchtpu-ci` is unchanged in this phase, and
`.buildkite/release/ci_gate.py` continues to gate releases on
`buildkite/vllm-torchtpu-ci` nightly statuses. `vllm-torchtpu-nightly-image`
(`pipeline_nightly_image.yml`) runs alongside `vllm-torchtpu-release` until the
nightly-image cut-over in `.buildkite/release/README.md` completes.

## Lanes, labels, and scheduling

`pipeline_cpu.yml` runs the full `pyrefly` type check and unmarked unit tests in
`tests/` in CPU pods (with `JAX_PLATFORMS=cpu` so JAX does not look for a TPU).
`pipeline_tests.yml` runs tests marked `tpu_test` (or `multichip`) on TPU v7x
slices (1x1x1 and 2x2x1). PR steps exclude `nightly` tests; nightly steps
include them. Both use the CI image built by `pipeline_build.yml`. On PR builds,
`bootstrap.py` uploads stage 0 (`pipeline_build.yml` and `pipeline_cpu.yml`)
followed by a `wait` and an `upload-stages` step that uploads the TPU lanes
(`tests`, `integration`, `verify`, plus `perf` with `run-perf-eval` or
`run-full-ci`) only after every stage-0 step (both image builds, including
`-omni`, and the CPU checks) passes. Retrying a failed stage-0 step in Buildkite
releases the `wait` once the retry passes and runs the uploader step. The
`oncall-fix` label raises job and Kueue workload priority without changing the
suite. A `NIGHTLY=1` build on `main` runs at the pre-merge job priority and in
the fleet's `nightly` Kueue class, so its steps queue level with pull requests,
in order of age. The torch-tpu and tpu-sync canaries and a `NIGHTLY=1` build off
`main` stay unclassed, at the lowest job priority.

By default, the `tests` lane's TPU steps run only on TPU v7x. A build picks
other chips with one of these extras in `CI_EXTRAS` (set it in the Buildkite UI
or API, alone or with another extra, e.g.
`CI_EXTRAS=perf-eval,tpu-all`):

| `CI_EXTRAS` | The `tests` lane's TPU steps run on |
|---|---|
| neither extra | v7x |
| `tpu-all` | v6e and v7x, each step twice, side by side |
| `tpu-v6e` | v6e alone, not alongside v7x |

A build names one of the two at most. `tpu-v6e` is refused on a build that
runs the nightly report (`NIGHTLY=1` or a tag), which waits on the v7x
steps. A merge-queue, release or vLLM-canary build refuses either one, since a
failing v6e step would block the merge, the release or the LKG promotion. A
build that sets `KUBE_SHAPE_SINGLE` or `KUBE_SHAPE_MULTI` is rejected: the
generation brings its shapes.

The v6e steps do not pass yet, and they are not `soft_fail`, so a build that
runs them is expected to fail: for one, the `tests_omni` tests are not
qualified for v6e's 32 GB of HBM. No build has to set either extra. Use them on
a manual build to see where v6e stands.

These extras choose the chips for the `tests` lane only. A build with
`NIGHTLY=1` also runs the nightly model and feature tests for both chips (the
"TPU v6e nightly Tests" and "TPU v7x nightly Tests" groups) and the nightly
report, whatever `CI_EXTRAS` is set to.

In the Buildkite UI, the name of each of the `tests` lane's TPU steps starts
with the chip it runs on (`v6e ...` or `v7x ...`).

A TPU step in `pipeline_tests.yml` is written once, with no generation in its
`label` or `key`. For each generation, `bootstrap.py` reads the file through
`buildkite-agent pipeline upload --dry-run`, puts the generation's name (`v6e`,
`v7x`) before every label and group name, and the name and an underscore
(`v6e_`, `v7x_`) before every key the file defines (and before a `depends_on`
that names one), then uploads the result. If that fails, the upload fails; a
file never goes up unstamped. A step outside the file that waits on one of its
steps names the prefixed key, as the nightly report in `pipeline_verify.yml`
does. A new generation is an entry in `tpu_generations.py` (`TpuGeneration`)
and no change to a step.

`pipeline_perf.yml` holds the perf/eval
steps. Post-merge builds do not run them, and PR builds run them only when the
PR has the `run-perf-eval` label (or `run-full-ci`, which sets `NIGHTLY=1` and
runs the full nightly suite across every lane: `cpu`, `tests`, `integration`,
`perf` and `verify`); adding either label to a PR that already has
`ready` starts a build that includes them. The 02:00 PT nightly and release
builds (`releases/vX.Y.Z` branches and release tags) select the `full` suite with
`NIGHTLY=1` and run all of them across every lane. Between nightlies, schedules on
`vllm-torchtpu-ci` that set `CI_SUITE=perf` run the perf lane alone; without
`NIGHTLY` that is only the steps not gated on it. The schedules keep clear of
the hours the nightly's perf steps take. The `vllm-torchtpu-canary-vllm`
pipeline also runs the lane, because its LKG promotion waits on two of its
steps.

A PR labeled `skip-tpu-ci` (it still needs `ready`) uploads only the `cpu` lane
and the image build (skipping the `-omni` image and every TPU lane), so what
runs is the image build, the type check, and the CPU unit tests. The post-merge
build of such a PR does the same: `bootstrap.py` reads the merged PR's labels
from GitHub (ignoring `run-perf-eval` and `run-full-ci` post-merge). On a PR,
the suite labels `skip-tpu-ci`, `run-perf-eval`, and `run-full-ci` are mutually
exclusive (`bootstrap.py` fails the build if more than one is set; `ready` and
`oncall-fix` are orthogonal).

GitHub merge-queue builds (`BUILDKITE_MERGE_QUEUE_BASE_BRANCH`) and `/ci run`
builds (`CI_RUN_TPU=1`, with optional `CI_EXTRAS=perf-eval,nightly-suite`) use
the same two-stage PR suites; merge-queue builds run at `post-merge` priority
(11, uncapped), read optional `perf-eval`, `nightly-suite`, or `precheck-only`
extras from the PR's `/ci merge` bot marker bound to the PR head SHA (and
fail if they can't read it), and always run the `pr+nightly-suite` gate on
`releases/vX.Y.Z` branches. `/ci precheck` builds (`CI_PRECHECK=1`) and PR
pushes when `CI_PR_PUSH_CPU_ONLY=1` select `pr+cpu-only` (stage 0 only, no TPU).
Merge-queue and `/ci` builds ignore PR labels. Once `CI_PR_PUSH_CPU_ONLY=1`,
PR pushes and post-merge builds ignore them too.

Merge-queue builds, and after cutover PR and `/ci` builds, also run `pre-commit`
on the checkout (`pipeline_precommit.yml`, a `cpu` step next to the image build;
docs-only changes run only it). The Buildkite status then covers the static
checks, so the required checks can be Buildkite and `DCO` alone. Those are apps
that report however a PR entered the queue, whereas a GitHub Actions check
doesn't run when the `/ci` workflow's own token enqueued the PR.

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

Each nightly and release image carries the commit it was built from in the
`org.opencontainers.image.revision` label, which maps a tag that names no
commit, such as a GAR `nightly-YYYYMMDD`, to one. An image built before the
label was added has none: that covers tags pushed earlier, and releases from
branches cut earlier, since a release branch runs the tooling it was cut with.
Read the label without pulling the image (an image without it prints nothing):

```bash
docker buildx imagetools inspect <image>:<tag> \
  --format '{{ index .Image.Config.Labels "org.opencontainers.image.revision" }}'
```

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
