# Dockerfile for vllm-torchtpu.
# This Dockerfile builds images for running vLLM on Google Cloud TPUs.
# It pre-installs all dependencies from pyproject.toml (including torch-tpu) in the base variant.

# vllm-router (vllm-project/router), the prefill/decode router the
# disaggregated benchmarks put in front of an engine pair. Built from source
# in a stage of its own; the pin is a build arg so a run can try another
# commit without touching this file.
FROM python:3.12-slim-bookworm AS router-build

# pkg-config and libssl-dev: hf-hub links the system OpenSSL.
RUN apt-get update && apt-get install -y curl git build-essential pkg-config libssl-dev && rm -rf /var/lib/apt/lists/*

ENV RUSTUP_HOME=/opt/rustup CARGO_HOME=/opt/cargo PATH=/opt/cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --no-modify-path --profile minimal --default-toolchain 1.95

ARG VLLM_ROUTER_COMMIT=5d3c3af9a41dec2d5ecff412024b7a743763472d
WORKDIR /build/router
RUN git clone --quiet --filter=blob:none https://github.com/vllm-project/router.git . && \
    git checkout --quiet "${VLLM_ROUTER_COMMIT}"

# The soft open-file limit of a BuildKit RUN step (1024) is too low for one
# rustc per core.
RUN --mount=type=cache,target=/opt/cargo/registry \
    --mount=type=cache,target=/opt/cargo/git \
    ulimit -n "$(ulimit -Hn)" && \
    cargo build --release --locked --bin vllm-router && \
    install -m 0755 target/release/vllm-router /usr/local/bin/vllm-router && \
    vllm-router --version

# Split dependency manifests by track/stage before installing packages so edits
# to unrelated pyproject.toml sections (for example tool.ruff, tool.pyrefly,
# benchmarking extras, or omni pins) never invalidate cached lower layers.
FROM python:3.12-slim-bookworm AS dep-manifests
WORKDIR /manifests
COPY pyproject.toml .
RUN python3 - <<'PY'
import json
import pathlib
import tomllib

raw = pathlib.Path("pyproject.toml").read_text()
data = tomllib.loads(raw)
proj = data["project"]
opt = proj.get("optional-dependencies", {})
groups = data.get("dependency-groups", {})
uv_idx = raw.find("[[tool.uv.index]]")
uv_tail = raw[uv_idx:] if uv_idx != -1 else ""


def fmt_list(items: list[str]) -> str:
    return "[\n" + "".join(f"    {json.dumps(x)},\n" for x in items) + "]"


def write_manifest(
    filename: str,
    *,
    extras: dict[str, list[str]] | None = None,
    dep_groups: dict[str, list[str]] | None = None,
) -> None:
    lines = [
        "[project]",
        f'name = {json.dumps(proj["name"])}',
        'version = "0.0.0"',
        f'requires-python = {json.dumps(proj["requires-python"])}',
        f'dependencies = {fmt_list(proj["dependencies"])}',
    ]
    if extras:
        lines.append("\n[project.optional-dependencies]")
        for k, v in extras.items():
            lines.append(f"{k} = {fmt_list(v)}")
    if dep_groups:
        lines.append("\n[dependency-groups]")
        for k, v in dep_groups.items():
            lines.append(f"{k} = {fmt_list(v)}")
    if uv_tail:
        lines.append("\n" + uv_tail.rstrip())
    pathlib.Path(filename).write_text("\n".join(lines) + "\n")


write_manifest("runtime.toml")
write_manifest("base.toml", dep_groups={"vllm": groups["vllm"]})
write_manifest(
    "dev.toml",
    extras={"benchmarking": opt["benchmarking"]},
    dep_groups={"vllm": groups["vllm"], "test": groups["test"]},
)
write_manifest(
    "omni.toml",
    extras={"omni": opt["omni"]},
    dep_groups={
        "vllm": groups["vllm"],
        "omni-git": groups["omni-git"],
        "test": groups["test"],
    },
)
PY

# Shared TPU runtime layer (torch, torch_tpu, tpu-sync-torch, jax, libtpu, ray).
# Installed before VLLM_COMMIT_HASH so upstream vLLM canary runs and vLLM pin
# bumps reuse the multi-GB TPU runtime cache.
#
# Python 3.12 is a hard constraint, not a preference: tpu-raiden-torch publishes
# cp312 wheels only. Moving off it needs raiden to ship the newer ABI first.
FROM python:3.12-slim-bookworm AS tpu-runtime

# curl for health checks, git so uv can clone source dependencies, the
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

# Note: we **intentionally** set VLLM_TARGET_DEVICE="empty", see README.
ENV CXX=g++
ENV CC=gcc
ENV VLLM_TARGET_DEVICE=empty
ENV UV_TORCH_BACKEND=cpu

COPY --from=dep-manifests /manifests/runtime.toml /tmp/runtime.toml
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system -r /tmp/runtime.toml && \
    rm -f /tmp/runtime.toml

# Variant 1: The Runtime (Base Image with pre-installed dependencies + vLLM)
# This variant provides the core runtime environment with all external dependencies
# (including torch-tpu and vLLM) pre-installed.
FROM tpu-runtime AS base

WORKDIR /workspace/vllm-torchtpu

# Copy base dependency manifest (core dependencies + vllm group only, so edits
# to omni/benchmarking/lint sections do not invalidate base or ci).
COPY --from=dep-manifests /manifests/base.toml pyproject.toml

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container before building.
ARG VLLM_COMMIT_HASH
RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

# Install all dependencies from pyproject.toml.
# vLLM comes from the `vllm` dependency group (see pyproject.toml).
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --system \
      -r pyproject.toml \
      --group vllm


# Variant 2: The CI Environment (CI/CD Variant)
# This variant builds on top of the runtime base, installing additional packages
# and testing tools needed for CI/CD pipelines.
FROM base AS ci

# Copy bktec binary from Buildkite Test Engine client image
COPY --from=buildkite/test-engine-client /usr/local/bin/bktec /usr/local/bin/bktec

# The agent CLI, so a pod can upload its own results before it is deleted.
COPY --from=buildkite/agent:3 /usr/local/bin/buildkite-agent /usr/local/bin/buildkite-agent

# Install tools needed for CI (gcloud, lsof, fuser, compilers)
RUN apt-get update && apt-get install -y apt-transport-https ca-certificates gnupg lsof psmisc lsb-release && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/cloud.google.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee -a /etc/apt/sources.list.d/gcsfuse.list && \
    apt-get update && apt-get install -y google-cloud-cli gcsfuse && \
    rm -rf /var/lib/apt/lists/*

# The P/D router the disaggregated benchmark clients start when the serve
# config asks for it (AGENTX_PROXY=router). A client-side tool, so ci and
# dev only; prod serves and does not route.
COPY --from=router-build /usr/local/bin/vllm-router /usr/local/bin/vllm-router

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

# Pre-install test and benchmarking dependencies from the narrow dev manifest
# before copying the full source tree so changes to src/, tests/, or omni pins
# do not invalidate the dependency installation layer cache.
# --group vllm keeps vLLM (already installed in base) in this resolution, so
# the test/benchmarking deps cannot move shared packages outside vLLM's pins.
COPY --from=dep-manifests /manifests/dev.toml /tmp/dev.toml
ARG VLLM_COMMIT_HASH
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" /tmp/dev.toml; \
    fi && \
    uv pip install --system \
      -r /tmp/dev.toml \
      --group vllm \
      --group test \
      --extra benchmarking && \
    rm -f /tmp/dev.toml

COPY . .

# If VLLM_COMMIT_HASH is passed (e.g., testing upstream HEAD in CI),
# dynamically update pyproject.toml inside the container after COPY . .
# so changes made during the base stage are not overwritten.
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


# Variant 4: Omni Image (Omni CI/Dev Variant branched directly from ci)
# Branches from ci rather than dev so Omni builds do not pull benchmarking
# packages and repo source edits (COPY . .) do not bust the vllm-omni layer.
FROM ci AS omni

# vllm-omni depends on opencv-python (not -headless), whose cv2 links libGL and
# glib; without them multi-rank diffusion workers crash at startup.
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

# Pre-install omni + test dependencies before copying the repository tree so
# edits under src/ or tests_omni/ reuse the cached vllm-omni build layer.
# --group vllm keeps vLLM's source pin in this resolution, so vllm-omni cannot
# replace it with a PyPI vllm. --group omni-git supplies the omni extra's
# vllm-omni from its git tag until the release is on PyPI (see pyproject.toml).
COPY --from=dep-manifests /manifests/omni.toml /tmp/omni.toml
ARG VLLM_COMMIT_HASH
RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" /tmp/omni.toml; \
    fi && \
    uv pip install --system \
      -r /tmp/omni.toml \
      --group vllm \
      --group test \
      --extra omni \
      --group omni-git && \
    rm -f /tmp/omni.toml

COPY . .

RUN if [ -n "$VLLM_COMMIT_HASH" ]; then \
      sed -i -E "s|(git\+https://github\.com/vllm-project/vllm\.git@)[a-zA-Z0-9.-]+|\1${VLLM_COMMIT_HASH}|g" pyproject.toml; \
    fi

RUN --mount=type=cache,target=/root/.cache/uv \
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 \
    uv pip install --system --no-deps -e .

ARG VLLM_SOURCE
COPY vllm /workspace/vllm-torchtpu/vllm

RUN --mount=type=cache,target=/root/.cache/uv \
    if [ -n "$VLLM_SOURCE" ]; then \
      uv pip install --system --no-deps \
        -e /workspace/vllm-torchtpu/vllm ; \
    else \
      rm -rf /workspace/vllm-torchtpu/vllm ; \
    fi

ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]


# Variant 5: Production/Full Image (Production/Minimal Variant)
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
    SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 \
    uv pip install --system --no-deps .

RUN if [ -z "$VLLM_SOURCE" ] && [ -d /workspace/vllm-torchtpu/vllm ]; then \
      rm -rf /workspace/vllm-torchtpu/vllm ; \
    fi

RUN mkdir -p /root /workspace && \
    ln -sfnT /workspace/vllm-torchtpu /root/torchtpu-vllm && \
    ln -sfnT /workspace/vllm-torchtpu /workspace/torchtpu-vllm
ENV PYTHONPATH="/workspace/vllm-torchtpu/src:/workspace/vllm-torchtpu"
CMD ["/bin/bash"]
