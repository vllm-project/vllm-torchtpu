# Dockerfile for torchtpu-vllm.
# This Dockerfile builds images for running vLLM on Google Cloud TPUs.
# It pre-installs all dependencies from pyproject.toml (including torch-tpu) in the base variant.

ARG BASE_IMAGE="us-docker.pkg.dev/ml-oss-artifacts-transient/torch-tpu-docker-container/torch-tpu-base:nightly-latest"

# Variant 1: The Runtime (Base Image with pre-installed dependencies)
# This variant provides the core runtime environment with all external dependencies
# (including torch-tpu) pre-installed.
FROM $BASE_IMAGE AS base

# Install curl for health checks (minimal for base)
RUN apt-get update && apt-get install -y curl git build-essential gcc g++ cmake ninja-build && rm -rf /var/lib/apt/lists/*

WORKDIR /root/torchtpu-vllm

# Copy pyproject.toml to extract dependencies and read settings
COPY pyproject.toml .

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
    export CXX=g++; \
    export CC=gcc; \
    export VLLM_TARGET_DEVICE=empty; \
    uv pip install --system \
      --index https://us-python.pkg.dev/ml-oss-artifacts-transient/torch-tpu-virtual-registry/simple/ \
      --extra-index-url https://pypi.org/simple \
      --index-strategy unsafe-best-match \
      -r pyproject.toml


# Variant 2: The CI Environment (CI/CD Variant)
# This variant builds on top of the runtime base, installing additional packages
# and testing tools needed for CI/CD pipelines.
FROM base AS ci

# Copy bktec binary from Buildkite Test Engine client image
COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec

# Install tools needed for CI (git, gcloud, lsof, fuser)
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg git lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# Clear PYTHONPATH inherited from base image if any
ENV PYTHONPATH=""

WORKDIR /root/torchtpu-vllm

# Mark mounted directory as safe for git
RUN git config --global --add safe.directory /root/torchtpu-vllm
RUN mkdir -p /workspace && ln -sfnT /root/torchtpu-vllm /workspace/torchtpu-vllm

# Note: we **intentionally** set VLLM_TARGET_DEVICE="empty", see README.
ENV VLLM_TARGET_DEVICE="empty"

# Variant 3: Developer Image (Dev Variant with test/benchmark tools)
# This variant contains all source code and is intended for active development.
# It installs testing and benchmarking tools in editable mode.
FROM ci AS dev
COPY . .
# Scoped to the vllm_torchtpu dist: the unscoped SETUPTOOLS_SCM_PRETEND_VERSION
# also applies to any other scm-versioned package built in this step. With a
# cold uv cache, resolving ".[...]" rebuilds metadata for the vllm git
# dependency, which then gets stamped 0.0.0 and reinstalled over the correctly
# versioned wheel from the base stage.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system -e ".[test,benchmarking]"

# Install vLLM from source if requested.
# If VLLM_SOURCE is provided, we build it in editable mode.
ARG VLLM_SOURCE
# Copy vllm source (might be empty if not provided)
COPY vllm /root/torchtpu-vllm/vllm

# Install vllm from source if requested.
# If VLLM_SOURCE is provided, we build it in editable mode.
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_SOURCE" ]; then \
      uv pip install --system --no-deps \
        -e /root/torchtpu-vllm/vllm ; \
    fi

ENV PYTHONPATH="/root/torchtpu-vllm/src:/root/torchtpu-vllm"
CMD ["/bin/bash"]


# Variant 4: Production/Full Image (Production/Minimal Variant)
# This variant copies all source code and installs the package regularly
# without extra dev/testing tools, starting directly from the runtime base.
FROM base AS prod
WORKDIR /root/torchtpu-vllm
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU=0.0.0 \
    uv pip install --system .

RUN mkdir -p /workspace && ln -sfnT /root/torchtpu-vllm /workspace/torchtpu-vllm
ENV PYTHONPATH="/root/torchtpu-vllm/src:/root/torchtpu-vllm"
CMD ["/bin/bash"]
