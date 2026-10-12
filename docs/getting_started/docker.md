# Docker images

This folder contains the Dockerfile used to build the main image variants for vllm-torchtpu.

## Pre-built Nightly Images

Pre-built Docker images are automatically built nightly and hosted on Google Artifact Registry. You can pull them using standard `docker pull` commands after authenticating.

### Authentication

To access the images, you must first configure Docker to authenticate with Google Artifact Registry:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev
```

### Pulling Images

You can pull the following pre-built nightly image targets using the `:latest` tag for the most recent build:

* **CI Image** (`ci:latest` equivalent):

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-ci:latest
  ```

* **Developer Image** (`dev:latest` equivalent):

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev:latest
  ```

* **Production Image** (`prod:latest` equivalent):

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-prod:latest
  ```

#### Date-Specific Tags

To pull an image for a specific build date, use the `nightly-YYYYMMDD` tag format:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-<target>:nightly-YYYYMMDD
```

For example, to pull the Developer Image from May 23, 2026:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-dev:nightly-20260523
```

#### Release Branch Tags

Each scheduled `release/X.Y.Z` branch is also built nightly, into the same image names. `X.Y.Z` points at the most recent build of the branch head, and `YYYYMMDD-X.Y.Z` pins one day's build. `latest` and `nightly-YYYYMMDD` always come from `main`.

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-<target>:0.0.1
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/vllm-torchtpu-<target>:20260930-0.0.1
```

## Image targets

### `ci:latest`
CI image target built from `--target ci`.

Use this image for CI and test infrastructure. It includes extra tooling on top of the base runtime, such as:

* `git` and a C/C++ toolchain (`build-essential`, `cmake`, `ninja-build`)
* `uv`
* `google-cloud-cli` (without `gsutil` and `bq`; use `gcloud storage`)
* `gcsfuse`
* `lsof`
* `psmisc` / `fuser`

> [!NOTE]
> This image does not install the `vllm-torchtpu` project itself, only the dependencies and tools needed for CI.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target ci -t ci:latest
```

### `dev:latest`
Developer image target built from `--target dev`.

This image is based on the `ci` variant, copies the repository into the image, and installs the project in editable mode with the `vllm`/`test` dependency groups and the `benchmarking` and `runai` extras:

* dependencies: `--group vllm --group test --extra benchmarking --extra runai`
* `evalscope` in its own environment under `/opt/evalscope`, on `PATH` as `evalscope`
* intended for local development, debugging, and running tests

Every package the `prod` image ships has the same version here, so tests run against what `prod` ships. The build fails if the test or benchmarking dependencies would change one; `docker/uv-constraints.txt` holds the caps that prevent it.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target dev -t dev:latest
```

### `prod:latest`
Production image target built from `--target prod`.

This image is intended for regular runtime usage and is the best target for production-style execution. It contains:

* the runtime dependencies, including the `runai` extra for `--load-format runai_streamer`
* a standard (non-editable) install of the project
* the repository's `examples/` under `/workspace/vllm-torchtpu/examples`

It does not contain the compiler toolchain, `git`, `uv`, the CI tools, the test and benchmarking dependencies, or the rest of the repository. To install more packages in a derived image, use `pip`.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target prod -t prod:latest
```

## Useful build arguments

The Dockerfile also supports a few build arguments that may be useful in advanced workflows:

* `VLLM_SOURCE`: when set, the `dev` image installs `vllm` in editable mode from the copied local source tree

Example using the helper script with build arguments:

```bash
./docker/build_image.sh \
  --target dev \
  -s /path/to/vllm/source \
  -t dev:latest
```
