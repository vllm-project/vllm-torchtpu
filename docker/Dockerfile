# Dockerfile for vllm-torchtpu.
# This Dockerfile builds images for running vLLM on Google Cloud TPUs.
# It pre-installs all dependencies from pyproject.toml (including torch-tpu) in the base variant.

# Variant 1: The Runtime (Base Image with pre-installed dependencies)
# This variant provides the core runtime environment with all external dependencies
# (including torch-tpu) pre-installed.
#
# Python 3.12 is a hard constraint, not a preference: tpu-raiden-torch publishes
# cp312 wheels only. Moving off it needs raiden to ship the newer ABI first.
FROM python:3.12-slim-bookworm AS base

# curl for health checks, git so uv can clone the vllm source dependency, the
# toolchain for packages that build from source, and libgoogle-perftools4 for
# the tcmalloc allocator enabled below. build-essential already pulls in gcc,
# g++, make and libc6-dev.
RUN apt-get update && apt-get install -y curl git build-essential cmake ninja-build libgoogle-perftools4 && rm -rf /var/lib/apt/lists/*

# tcmalloc for every derived image. Set after the install above, so that
# intermediate steps do not fail preloading a library that is not there yet.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# The interpreter lives in /usr/local, so Debian's usual /usr/bin/python3 does
# not exist. Nothing in this repo needs it, but third-party tooling run inside
# the image may assume the Debian path.
RUN ln -sf "$(command -v python)" /usr/bin/python3 && \
    pip install --no-cache-dir uv

WORKDIR /workspace/vllm-torchtpu

# Copy pyproject.toml to extract dependencies and read settings, and the
# transitive-dependency snapshot that pins what torch-tpu-base used to supply.
COPY pyproject.toml docker/constraints.txt ./

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container before building.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Note: we **intentionally** set VLLM_TARGET_DEVICE="empty", see README.
ENV CXX=g++
ENV CC=gcc
ENV VLLM_TARGET_DEVICE=empty

# Install all dependencies from pyproject.toml. We mount the gcloud token secret
# to authenticate with the private torch-tpu registry, falling back to PyPI for other packages.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=secret,id=gcloud_token \
    set -eu; \
    TOKEN=""; \
    if [ -f /run/secrets/gcloud_token ]; then TOKEN="$(cat /run/secrets/gcloud_token)"; fi; \
    if [ -z "${TOKEN}" ]; then echo "Missing BuildKit secret gcloud_token for registry mode" >&2; exit 1; fi; \
    export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME=oauth2accesstoken; \
    export UV_INDEX_TORCH_TPU_REGISTRY_PASSWORD="${TOKEN}"; \
    uv pip install --system \
      --index https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      --constraint constraints.txt \
      -r pyproject.toml


# Variant 2: The CI Environment (CI/CD Variant)
# This variant builds on top of the runtime base, installing additional packages
# and testing tools needed for CI/CD pipelines.
FROM base AS ci

# Copy bktec binary from Buildkite Test Engine client image
COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec

# Install tools needed for CI (gcloud, lsof, fuser, compilers)
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# Start from an empty PYTHONPATH; the dev and prod stages set their own.
ENV PYTHONPATH=""

WORKDIR /workspace/vllm-torchtpu

# Mark mounted directory as safe for git and maintain backward compatibility
RUN git config --global --add safe.directory /workspace/vllm-torchtpu && \
    git config --global --add safe.directory /root/torchtpu-vllm && \
    mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm

# Variant 3: Developer Image (Dev Variant with test/benchmark tools)
# This variant contains all source code and is intended for active development.
# It installs testing and benchmarking tools in editable mode.
FROM ci AS dev

# Pre-install test and benchmarking dependencies before copying full source tree
# so that changes to folders like src/ or tests/ do not invalidate the
# dependency installation layer cache.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --constraint constraints.txt \
      -r pyproject.toml \
      --extra test \
      --extra benchmarking

COPY . .

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container after COPY . .
# so changes made during the base stage are not overwritten.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Scoped to the vllm_torchtpu dist: the unscoped SETUPTOOLS_SCM_PRETEND_VERSION
# also applies to any other scm-versioned package built in this step.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system --no-deps -e .

# Install vLLM from source if requested.
# If VLLM_SOURCE is provided, we build it in editable mode.
ARG VLLM_SOURCE
# Copy vllm source (might be empty if not provided)
COPY vllm /workspace/vllm-torchtpu/vllm

# Install vllm from source if requested, otherwise clean up dummy dir
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_SOURCE" ]; then \
      uv pip install --system --no-deps \
        -e /workspace/vllm-torchtpu/vllm ; \
    else \
      rm -rf /workspace/vllm-torchtpu/vllm ; \
    fi

ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]


# Variant 4: Production/Full Image (Production/Minimal Variant)
# This variant copies all source code and installs the package regularly
# without extra dev/testing tools, starting directly from the runtime base.
FROM base AS prod
WORKDIR /workspace/vllm-torchtpu
COPY . .

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container after COPY . .
# so changes made during the base stage are not overwritten.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Use --no-deps to preserve layer caching since dependencies were already resolved and installed in base.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system --no-deps .

RUN if [ -z "$VLLM_SOURCE" ] && [ -d /workspace/vllm-torchtpu/vllm ]; then \
      rm -rf /workspace/vllm-torchtpu/vllm ; \
    fi

RUN mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm
ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]
