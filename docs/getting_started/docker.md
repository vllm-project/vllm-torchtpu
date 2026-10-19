# Docker images

This folder contains the Dockerfile used to build the main image variants for vllm-torchtpu.

## Pre-built Images on Docker Hub

Official pre-built `vllm-torchtpu` Docker images are publicly available on Docker Hub at [`vllm/vllm-torchtpu`](https://hub.docker.com/r/vllm/vllm-torchtpu/tags). No registry authentication is required.

### Release Images

Release images are built from the `--target prod` Dockerfile stage and contain a standard runtime installation of `vllm` and `vllm-torchtpu`.

* **Latest Stable Release** (`:latest`):

  ```bash
  docker pull vllm/vllm-torchtpu:latest
  ```

* **Pinned Version Release** (`:vX.Y.Z`, e.g., `v0.29.0`):

  ```bash
  docker pull vllm/vllm-torchtpu:v0.29.0
  ```

### Nightly Images

Nightly images are built automatically every night from the `main` branch (`--target dev` stage, which includes the repository source, test dependencies, and benchmarking tools):

* **Latest Nightly Build** (`:nightly`):

  ```bash
  docker pull vllm/vllm-torchtpu:nightly
  ```

* **Date- and Commit-Pinned Nightly Tags** (`:nightly-YYYYMMDD-<commit>-<vllm-version>`):

  ```bash
  docker pull vllm/vllm-torchtpu:nightly-20261019-88534b8-v0.31.0
  ```

Browse all available release and nightly tags at [`https://hub.docker.com/r/vllm/vllm-torchtpu/tags`](https://hub.docker.com/r/vllm/vllm-torchtpu/tags).

### Running the Container on a Cloud TPU VM

Include `--privileged`, `--net=host`, and `--shm-size 150gb` so the container can access the `/dev/vfio/*` TPU devices and shared host memory:

```bash
sudo docker run -it --rm \
  --name $USER-vllm \
  --privileged \
  --net=host \
  -v /dev/shm:/dev/shm \
  --shm-size 150gb \
  -p 8000:8000 \
  --entrypoint /bin/bash \
  vllm/vllm-torchtpu:latest
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
> This image does not install the `vllm-torchtpu` project itself, only the dependencies and tools needed for CI.

Example build using the standard helper script:

```bash
./docker/build_image.sh --target ci -t ci:latest
```

### `dev:latest`
Developer image target built from `--target dev`.

This image is based on the `ci` variant, copies the repository into the image, and installs the project in editable mode with the `vllm`/`test` dependency groups and the `benchmarking` extra:

* dependencies: `--group vllm --group test --extra benchmarking`
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

## vllm-router

The `ci` and `dev` targets carry [vllm-router](https://github.com/vllm-project/router) at `/usr/local/bin/vllm-router`, built from the commit in the `VLLM_ROUTER_COMMIT` build arg. The disaggregated benchmark clients start it in prefill/decode mode when a serve config sets `AGENTX_PROXY=router`; the default stays the toy proxy in `examples/disagg`.

## Useful build arguments

The Dockerfile also supports a few build arguments that may be useful in advanced workflows:

* `VLLM_SOURCE`: when set, installs `vllm` from the copied local source tree

Example using the helper script with build arguments:

```bash
./docker/build_image.sh \
  --target prod \
  -t prod:latest
```
