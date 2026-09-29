#!/bin/bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ------------------------------------------------------
# Build the combined `vllm-torchtpu` wheel.
#
# The output wheel IS vLLM, with its distribution renamed to `vllm-torchtpu`
# and a dependency on `vllm-torchtpu-core` (this repo's plugin) added. Wheels
# do not nest: the core plugin is fetched separately by pip at install time.
#
# This is the torchtpu counterpart of
# tpu-inference/.buildkite/scripts/build_vllm_tpu.sh. The two differ in one
# structural way that drives most of the logic below:
#
#   tpu-inference  VLLM_TARGET_DEVICE=tpu   -> setup.py reads requirements/tpu.txt
#   vllm-torchtpu  VLLM_TARGET_DEVICE=empty -> setup.py reads requirements/common.txt
#
# requirements/tpu.txt is read only by TPU builds, so tpu-inference can (and
# does) keep a committed `tpu-inference==X` placeholder there and simply sed it
# to the right version. requirements/common.txt is read by EVERY vLLM build --
# cuda, rocm, cpu, xpu -- so no placeholder can ever be committed there and the
# injection must be strictly temporary. Hence the trap-based revert below.
#
# Usage:
#   1. PyPI build -- depend on a published vllm-torchtpu-core release:
#        ./build_vllm_torchtpu.sh <core-version> <wheel-version> [vllm-ref]
#   2. Local build -- depend on a locally built core wheel:
#        ./build_vllm_torchtpu.sh --local <wheel-version> [vllm-ref]
#
# Examples:
#   ./build_vllm_torchtpu.sh 0.29.0rc1 0.29.0rc1 v0.29.0
#   ./build_vllm_torchtpu.sh --local 0.29.0rc1
#
# Environment:
#   VLLM_REPO  vLLM git remote to clone (default: upstream vllm-project/vllm).
#              Set this to build against a fork.
#   CORE_DIST_DIR  Where --local looks for the core wheel
#                  (default: <repo-root>/dist).
#
# Output:
#   vllm/dist/vllm_torchtpu-<wheel-version>-*.whl
# ------------------------------------------------------

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .buildkite/scripts -> .buildkite -> repo root
TORCHTPU_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
CORE_DIST_DIR="${CORE_DIST_DIR:-${TORCHTPU_ROOT}/dist}"

usage() {
    echo "Usage: $0 <core-version> <wheel-version> [vllm-ref]"
    echo "       $0 --local <wheel-version> [vllm-ref]"
    echo ""
    echo "  <core-version>   Published vllm-torchtpu-core version to depend on."
    echo "  <wheel-version>  Version stamped onto the produced vllm-torchtpu wheel."
    echo "  [vllm-ref]       vLLM branch, tag or commit to build (default: main)."
    echo ""
    echo "  --local          Depend on a locally built core wheel found in"
    echo "                   ${CORE_DIST_DIR} instead of a published version."
}

# --- Argument parsing & validation -------------------------------------------
WHL_PATH=""
if [[ "$1" == "--local" ]]; then
    shift

    if [ "$#" -lt 1 ]; then
        usage
        exit 1
    fi

    WHL_PATH=$(find "${CORE_DIST_DIR}" -name "vllm_torchtpu_core-*.whl" 2>/dev/null | head -n 1)
    if [ -z "${WHL_PATH}" ]; then
        echo "ERROR: no vllm_torchtpu_core-*.whl found in ${CORE_DIST_DIR}." >&2
        echo "Build the core wheel first, from the repo root:" >&2
        echo "  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 uv build --wheel" >&2
        exit 1
    fi

    # A PEP 508 direct reference must be an absolute file:// URL. A bare or
    # relative path parses fine in a requirements.txt but is rejected in
    # install_requires, which is where setup.py puts everything it reads out of
    # common.txt.
    WHL_PATH="$(cd "$(dirname "${WHL_PATH}")" && pwd)/$(basename "${WHL_PATH}")"

    CORE_VERSION=$(basename "$WHL_PATH" | cut -d'-' -f2)
    WHEEL_VERSION=$1
    VLLM_REF=${2:-"main"}
else
    if [ "$#" -lt 2 ]; then
        usage
        exit 1
    fi

    CORE_VERSION=$1
    WHEEL_VERSION=$2
    VLLM_REF=${3:-"main"}
fi

echo "--- Starting vllm-torchtpu wheel build ---"
echo "Mode                   : $([[ "${WHL_PATH}" ]] && echo "Local wheel" || echo "PyPI")"
echo "vllm-torchtpu-core ver : ${CORE_VERSION}"
echo "vllm-torchtpu version  : ${WHEEL_VERSION}"
echo "vLLM repo              : ${VLLM_REPO}"
echo "vLLM ref               : ${VLLM_REF}"

# --- Step 1: Clone vLLM ------------------------------------------------------
REPO_DIR="vllm"

if [ -d "$REPO_DIR" ]; then
    echo "Repository '$REPO_DIR' already exists. Skipping clone."
else
    echo "Cloning vLLM from ${VLLM_REPO}..."
    git clone "${VLLM_REPO}" "${REPO_DIR}"
fi
cd "${REPO_DIR}"

echo "Checking out vLLM ref: ${VLLM_REF}..."
if ! git checkout "${VLLM_REF}"; then
    echo "ERROR: failed to checkout '${VLLM_REF}'." >&2
    exit 1
fi
git pull --ff-only || echo "Warning: could not pull (detached HEAD or tag); continuing."

# The renamer lives in the vLLM tree, so the checked-out ref must contain it.
# Fail here with a clear message rather than deep inside the build.
BUILD_SCRIPT="tools/vllm-torchtpu/build.sh"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: ${BUILD_SCRIPT} not found in vLLM ref '${VLLM_REF}'." >&2
    echo "       That script must be committed to the vLLM branch being built." >&2
    exit 1
fi

# --- Step 2: Inject the vllm-torchtpu-core dependency ------------------------
# VLLM_TARGET_DEVICE=empty makes setup.py read common.txt, so that is the only
# file the dependency can land in for this build.
REQUIREMENTS_FILE="requirements/common.txt"
BACKUP_FILE="${REQUIREMENTS_FILE}.bak"

if [[ "${WHL_PATH}" ]]; then
    REQUIRED_LINE="vllm-torchtpu-core @ file://${WHL_PATH}"
else
    REQUIRED_LINE="vllm-torchtpu-core==${CORE_VERSION}"
fi

# --- Cleanup -----------------------------------------------------------------
# common.txt is shared by every vLLM platform, so leaving it modified would
# silently poison any later build in this checkout. Registered BEFORE the first
# mutation so an interrupt mid-write still restores. tpu-inference reverts
# linearly at the end of its script instead; with `set -e` that means a failed
# build leaves the file dirty, which is tolerable for tpu.txt but not here.
cleanup() {
    if [ -f "$BACKUP_FILE" ]; then
        echo "Reverting ${REQUIREMENTS_FILE}..."
        mv -f "$BACKUP_FILE" "$REQUIREMENTS_FILE"
    fi
}
trap cleanup EXIT HUP INT QUIT PIPE TERM

echo "Injecting '${REQUIRED_LINE}' into ${REQUIREMENTS_FILE}..."
cp "$REQUIREMENTS_FILE" "$BACKUP_FILE"

# A file without a trailing newline would glue our entry onto the last
# requirement, producing one corrupt line instead of two valid ones.
if [ -n "$(tail -c 1 "$REQUIREMENTS_FILE")" ]; then
    echo "" >> "$REQUIREMENTS_FILE"
fi

# Drop any pre-existing entry before appending. Unlike tpu-inference we cannot
# assume a committed placeholder to sed in place, and --local writes a
# `name @ file://...` form that a `^vllm-torchtpu-core==` grep would miss --
# repeat runs would then append duplicates.
if grep -qE '^vllm-torchtpu-core([[:space:]]|==|@|$)' "$REQUIREMENTS_FILE"; then
    echo "(Existing vllm-torchtpu-core entry found. Replacing.)"
    grep -vE '^vllm-torchtpu-core([[:space:]]|==|@|$)' "$REQUIREMENTS_FILE" > "${REQUIREMENTS_FILE}.tmp"
    mv "${REQUIREMENTS_FILE}.tmp" "$REQUIREMENTS_FILE"
fi
echo "$REQUIRED_LINE" >> "$REQUIREMENTS_FILE"

# --- Step 3: Build -----------------------------------------------------------
echo "Ensuring 'build' package is installed..."
pip install build

echo "Executing the vLLM torchtpu build script..."
bash "$BUILD_SCRIPT" "${WHEEL_VERSION}"

echo "--- Build complete! ---"
ls -lh dist/ || true
echo "The wheel file can be found in the 'vllm/dist' directory."

# cleanup() also runs via the EXIT trap; calling it here just makes the revert
# visible in the log before the success message scrolls past.
trap - EXIT HUP INT QUIT PIPE TERM
cleanup
echo "Cleanup complete. Script finished."
