# Dockerfile for vllm-torchtpu.
# This Dockerfile builds images for running vLLM on Google Cloud TPUs.
# It pre-installs all dependencies from pyproject.toml (including torch-tpu) in the base variant.
#
# Stages:
#   runtime  Python plus the few OS packages vLLM needs at run time.
#   base     runtime + build toolchain + uv + every runtime Python dependency.
#   ci       base + CI tooling (gcloud, gcsfuse, Buildkite CLIs).
#   dev      ci + test/benchmarking dependencies + the repo, installed editable.
#   omni     dev + vllm-omni.
#   prod     runtime + base's installed Python packages + vllm-torchtpu-core.
#
# prod starts from runtime, not base, and copies only the installed Python
# packages out of base, so nothing from base's toolchain or from the ci/dev
# stages reaches it.

# Variant 1: The Runtime (OS layer shared by every image)
#
# Python 3.12 is a hard constraint, not a preference: tpu-raiden-torch publishes
# cp312 wheels only. Moving off it needs raiden to ship the newer ABI first.
FROM python:3.12-slim-bookworm AS runtime

# curl for health checks and libgoogle-perftools4 for the tcmalloc allocator
# enabled below. Build tools belong in the base stage, not here: this is
# everything prod adds to the Python image.
RUN apt-get update && apt-get install -y --no-install-recommends curl libgoogle-perftools4 && rm -rf /var/lib/apt/lists/*

# tcmalloc for every derived image. Set after the install above, so that
# intermediate steps do not fail preloading a library that is not there yet.
ENV LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

# The interpreter lives in /usr/local, so Debian's usual /usr/bin/python3 does
# not exist. Nothing in this repo needs it, but third-party tooling run inside
# the image may assume the Debian path.
RUN ln -sf "$(command -v python)" /usr/bin/python3

# Note: we **intentionally** set VLLM_TARGET_DEVICE="empty", see README.
ENV VLLM_TARGET_DEVICE=empty


# Variant 2: The Base (Runtime plus pre-installed dependencies)
# This variant installs all external dependencies (including torch-tpu). The
# ci, dev and omni variants build on it; prod copies its installed packages.
FROM runtime AS base

# git so uv can clone the vllm source dependency, and the toolchain for
# packages that build from source. build-essential already pulls in gcc, g++,
# make and libc6-dev.
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential cmake ninja-build && rm -rf /var/lib/apt/lists/*

# uv goes in /usr/bin rather than /usr/local, so prod's copy of /usr/local
# leaves it behind.
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /uvx /usr/bin/

WORKDIR /workspace/vllm-torchtpu

# Copy pyproject.toml to extract dependencies and read settings, plus the
# excludes and constraints that every image install uses.
COPY pyproject.toml .
COPY docker/uv-excludes.txt docker/uv-constraints.txt docker/

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container before building.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

ENV CXX=g++
ENV CC=gcc

# Install all dependencies from pyproject.toml. We mount the gcloud token secret
# to authenticate with the private torch-tpu registry, falling back to PyPI for other packages.
# vLLM comes from the `vllm` dependency group (see pyproject.toml), and the
# `runai` extra lets prod stream gs:// checkpoints.
# docker/uv-excludes.txt keeps GPU-only packages out of the image, and
# docker/uv-constraints.txt holds the caps that keep prod on the versions the
# dev image (and so CI) runs; see the check in the dev stage.
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
      --excludes docker/uv-excludes.txt \
      -c docker/uv-constraints.txt \
      -r pyproject.toml \
      --group vllm \
      --extra runai


# Variant 3: The CI Environment (CI/CD Variant)
# This variant builds on top of the runtime base, installing additional packages
# and testing tools needed for CI/CD pipelines.
FROM base AS ci

# Copy bktec binary from Buildkite Test Engine client image
COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec

# The agent CLI, so a pod can upload its own results before it is deleted.
COPY --from=buildkite/agent:3 /usr/local/bin/buildkite-agent /usr/local/bin/buildkite-agent

# Install tools needed for CI (gcloud, gcsfuse, lsof, fuser). CI calls
# `gcloud storage`, never gsutil or bq, so those are removed, along with the
# ~290 MB of precompiled bytecode the package ships; gcloud recompiles what it
# imports on first use. Their commands go too: left behind, `command -v gsutil`
# finds a wrapper with nothing to run.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates gnupg lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y --no-install-recommends google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/lib/google-cloud-sdk/platform/gsutil /usr/lib/google-cloud-sdk/platform/bq && \
    rm -f /usr/bin/gsutil /usr/bin/bq /usr/lib/google-cloud-sdk/bin/gsutil /usr/lib/google-cloud-sdk/bin/bq && \
    find /usr/lib/google-cloud-sdk -name __pycache__ -type d -prune -exec rm -rf {} +

# Start from an empty PYTHONPATH; the dev stage sets its own.
ENV PYTHONPATH=""

WORKDIR /workspace/vllm-torchtpu

# Mark mounted directory as safe for git and maintain backward compatibility
RUN git config --global --add safe.directory /workspace/vllm-torchtpu && \
    git config --global --add safe.directory /root/torchtpu-vllm && \
    mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm

# Variant 4: Developer Image (Dev Variant with test/benchmark tools)
# This variant contains all source code and is intended for active development.
# It installs testing and benchmarking tools in editable mode.
FROM ci AS dev

# Pre-install test and benchmarking dependencies before copying full source tree
# so that changes to folders like src/ or tests/ do not invalidate the
# dependency installation layer cache.
# --group vllm keeps vLLM (already installed in base) in this resolution, so
# the test/benchmarking deps cannot move shared packages outside vLLM's pins.
#
# CI runs this image, so it must hold the same versions of the packages prod
# ships. The step fails if the install changed any package base installed;
# cap the package in docker/uv-constraints.txt to fix it.
RUN --mount=type=cache,target=/root/.cache/uv \
    set -eu; \
    uv pip freeze --system > /tmp/runtime-freeze.txt; \
    uv pip install --system \
      --excludes docker/uv-excludes.txt \
      -c docker/uv-constraints.txt \
      -r pyproject.toml \
      --group vllm \
      --group test \
      --extra benchmarking \
      --extra runai; \
    changed="$(uv pip freeze --system | grep -vxFf - /tmp/runtime-freeze.txt || true)"; \
    rm /tmp/runtime-freeze.txt; \
    if [ -n "$changed" ]; then \
      echo "The test/benchmarking install changed packages that prod ships:" >&2; \
      echo "$changed" >&2; \
      echo "Cap them in docker/uv-constraints.txt so prod and CI run the same versions." >&2; \
      exit 1; \
    fi

# evalscope runs only as a CLI against the server's HTTP endpoint, and its
# litellm dependency caps openai<3 and filelock<4. It gets its own venv on top
# of the system packages: pip installs just evalscope's own dependencies and
# the older versions it needs there, and the system environment keeps the
# versions prod ships. This uses pip because uv ignores
# --system-site-packages when it installs; the venv's pip is upgraded first
# because `--group` needs pip 25.1.
RUN --mount=type=cache,target=/root/.cache/pip \
    python -m venv --system-site-packages /opt/evalscope && \
    /opt/evalscope/bin/pip install --upgrade pip && \
    /opt/evalscope/bin/pip install --group evalscope && \
    ln -s /opt/evalscope/bin/evalscope /usr/local/bin/evalscope

COPY . .

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container after COPY . .
# so changes made during the base stage are not overwritten.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Scoped to the vllm-torchtpu-core dist: the unscoped SETUPTOOLS_SCM_PRETEND_VERSION
# also applies to any other scm-versioned package built in this step.
RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 \
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


# Variant 5: Omni Image (Omni Variant with dev + omni dependencies)
FROM dev AS omni

# --group vllm keeps vLLM's source pin in this resolution, so vllm-omni cannot
# replace it with a PyPI vllm.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      --excludes docker/uv-excludes.txt \
      -r pyproject.toml \
      --group vllm \
      --extra omni


# Variant 6: Production Image (Production/Minimal Variant)
# This variant starts from the runtime stage and takes only base's installed
# Python packages and console scripts, then installs the project itself. The
# toolchain, uv, the CI tools and the test/benchmarking dependencies stay out.
FROM runtime AS prod

COPY --from=base /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=base /usr/local/bin /usr/local/bin

WORKDIR /workspace/vllm-torchtpu

# The repo and uv are bind-mounted, so only the installed package lands in the
# image. The repo mount is rw so setuptools can write build/ and egg-info,
# which are discarded with the mount. --no-deps because base already installed
# the dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,from=base,source=/usr/bin/uv,target=/usr/bin/uv \
    --mount=type=bind,source=.,target=/tmp/vllm-torchtpu,rw \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 \
    uv pip install --system --no-deps /tmp/vllm-torchtpu

# The examples are the one part of the repo prod users run directly.
COPY examples examples

RUN ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm
CMD ["/bin/bash"]
