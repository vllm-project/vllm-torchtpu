# vllm-torchtpu-release

One Buildkite pipeline publishes everything vllm-torchtpu ships: the nightly
images, and each release's Docker image and two wheels (`vllm-torchtpu-core`
and the combined `vllm-torchtpu`). It runs only in the **TPU Release** cluster,
on the `release-vllm-torchtpu` queue. That queue's VM runs as its own service
account (`ci-release`), refuses jobs from any other pipeline, and is the only
agent that can read the Docker Hub token or mint a PyPI token. Once the
cut-over below is done, no CI agent can.

The pipeline's settings upload [`pipeline.yml`](pipeline.yml) from the build's
own checkout, and only for builds of `main` and `releases/v*`. A release branch
runs the release tooling (this directory and `.buildkite/scripts`) it was cut
with; a tooling fix a release needs goes to main and is cherry-picked to the
branch, or committed to the branch directly for one-off debugging.

## Cut a release

1. Get a full CI run of the release commit: a **vllm-torchtpu-ci** build of
   `releases/vX.Y.Z` with `NIGHTLY=1` (or a PR with the `run-full-ci` label).
   It has to pass; the release checks.
2. Start a **New Build** of **vllm-torchtpu-release** on `releases/vX.Y.Z`, at
   that commit. For a pre-release set `RELEASE_VERSION=vX.Y.Zrc1`.
3. The build then, without anyone's help:
   - **preflight**: settles the version (branch or `RELEASE_VERSION`), checks
     the commit's vllm-torchtpu-ci statuses on GitHub (passed, with nightly
     steps), and checks PyPI doesn't have the version yet;
   - **builds once**: the ci, dev and prod images, staged in GAR as
     `vllm-torchtpu-<target>:<version>` and `:<date>-<version>`, and both
     wheels;
   - **checks**: the wheels resolve together from pypi.org for CPython 3.12
     on x86_64 Linux.
4. **Approve** the publish block. It asks whether to move `latest` (never for a
   pre-release) and whether an existing Docker Hub tag may be replaced.
5. The build publishes what it built and checked:
   - the staged prod image, copied by digest to Docker Hub as
     `vllm/vllm-torchtpu:vX.Y.Z` (and `latest` if chosen);
   - `vllm-torchtpu-core`, then `vllm-torchtpu`, to PyPI with Trusted
     Publishing (`.buildkite/scripts/publish_wheel.sh`);
   - then checks, without credentials, that an anonymous pull of the tag is the
     staged digest and that `vllm-torchtpu==X.Y.Z` resolves from pypi.org.

Build env knobs, for when something is broken and you know why:

| Env | Effect |
|---|---|
| `RELEASE_VERSION=vX.Y.Zrc1` | publish a pre-release under that version |
| `RELEASE_SKIP_CI_GATE=1` | don't check vllm-torchtpu-ci (annotated as a warning) |
| `RELEASE_APPROVER_TEAMS=a,b` | only these Buildkite teams can approve |

## Nightly images

A schedule on `main` runs the `nightly_images` step:
`.buildkite/scripts/publish_nightly_images.sh` pushes the ci, dev and prod
images to GAR and the dev image to Docker Hub as `nightly`.

## Pipeline settings (Buildkite, made by hand)

- Cluster **TPU Release**; repository
  `git@github.com:vllm-project/vllm-torchtpu.git` (the VM checks out with the
  cluster's `VLLM_TORCHTPU_DEPLOY_KEY`).
- GitHub: no builds on push or pull request (manual and scheduled only).
- Schedule: `main`, daily, for the nightly images.
- Steps:

  ```yaml
  steps:
    - label: ":pipeline: Upload the release pipeline"
      branches: "main releases/v*"
      agents:
        queue: release-vllm-torchtpu
      command: buildkite-agent pipeline upload .buildkite/release/pipeline.yml
  ```

## One-time setup outside this repo

- ci-infra: the pipeline's ID in `allowed_pipeline_ids` of
  `ci_release_vllm_torchtpu` (`release_agents.tf`), so the release VM accepts
  its jobs.
- Buildkite: the TPU Release cluster secrets `DOCKERHUB_TOKEN_VLLM_TORCHTPU`
  and `VLLM_TORCHTPU_DEPLOY_KEY` readable by this pipeline.
- PyPI: a Google Trusted Publisher for
  `ci-release@cloud-ullm-inference-ci-cd.iam.gserviceaccount.com` on both
  `vllm-torchtpu-core` and `vllm-torchtpu`.

## Cut-over from vllm-torchtpu-nightly-image

Until this is done both pipelines run, and the old one can still publish a
release and holds a Docker Hub token in the CI cluster. Both nightly steps
push the same tags (`nightly-YYYYMMDD` and `latest` in GAR, `nightly` on
Docker Hub), so while both are scheduled on `main` the tags point at
whichever finished last. To compare them, give the new pipeline's schedule
scratch targets (`NIGHTLY_REGISTRY`, `DOCKERHUB_REPO`) or schedule only one.

1. Add the `main` nightly schedule to vllm-torchtpu-release.
2. Once its nightlies match the old pipeline's, disable the old pipeline's
   `main` schedule. Its `glm-agentx` schedule can stay: that branch has its
   own copy of the nightly script, which pushes to GAR only.
3. Remove the `approve_release` block and the release arm from
   `pipeline_nightly_image.yml` and `scripts/publish_nightly_images.sh`.
4. Delete the Docker Hub token from the CI cluster.
