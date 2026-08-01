# Docker images

This folder contains the Dockerfile used to build the main image variants for torchtpu-vllm.

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
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-ci:latest
  ```

* **Developer Image** (`dev:latest` equivalent):

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev:latest
  ```

* **Production Image** (`prod:latest` equivalent):

  ```bash
  docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-prod:latest
  ```

#### Date-Specific Tags

To pull an image for a specific build date, use the `nightly-YYYYMMDD` tag format:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-<target>:nightly-YYYYMMDD
```

For example, to pull the Developer Image from May 23, 2026:

```bash
docker pull us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu/torchtpu-vllm-dev:nightly-20260523
```

## Image targets

### `ci:latest`
CI image target built from `--target ci`.

Use this image for CI and test infrastructure. It includes extra tooling on top of the base runtime, such as:

* `git`
* `google-cloud-cli`
* `gcsfuse`
* `lsof`
* `psmisc` / `fuser`

> [!NOTE]
> This image does not install the `torchtpu-vllm` project itself, only the dependencies and tools needed for CI.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target ci -t ci:latest
```

### `dev:latest`
Developer image target built from `--target dev`.

This image is based on the `ci` variant, copies the repository into the image, and installs the project in editable mode with test and benchmarking extras:

* editable install: `-e ".[test,benchmarking]"`
* intended for local development, debugging, and running tests

Example build using the standard helper script:

```bash
./docker/build_image.sh --target dev -t dev:latest
```

### `prod:latest`
Production image target built from `--target prod`.

This image is intended for regular runtime usage. It performs a standard install of the project and is the best target for production-style execution.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target prod -t prod:latest
```

## Useful build arguments

The Dockerfile also supports a few build arguments that may be useful in advanced workflows:

* `BASE_IMAGE`: base runtime image
* `VLLM_SOURCE`: when set, installs `vllm` from the copied local source tree

Example using the helper script with build arguments:

```bash
./docker/build_image.sh \
  --target prod \
  -t prod:latest
```
