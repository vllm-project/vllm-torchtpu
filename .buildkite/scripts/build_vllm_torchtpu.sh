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
# vllm-torchtpu builds vLLM with VLLM_TARGET_DEVICE=empty (TPU support comes
# from the vllm-torchtpu-core plugin, not vLLM's in-tree TPU code). With that
# target, setup.py takes install_requires from requirements/common.txt, so the
# vllm-torchtpu-core dependency has to be added there.
#
# common.txt is read by EVERY vLLM build (cuda, rocm, cpu, xpu, ...), so the
# entry can never be committed to vLLM; it is injected only for this build and
# reverted afterwards. Hence the trap-based revert below.
#
# Prerequisite:
#   The vLLM ref being built must contain tools/vllm-torchtpu/build.sh (the
#   counterpart of upstream's tools/vllm-tpu/build.sh). Upstream vLLM does not
#   ship it yet, so until it lands point VLLM_REPO / [vllm-ref] at a fork or
#   branch that does. The script fails fast with a clear message otherwise.
#
# Usage:
#   1. PyPI build -- depend on a published vllm-torchtpu-core release:
#        ./build_vllm_torchtpu.sh <core-version> <wheel-version> [vllm-ref]
#   2. Local build -- depend on a locally built core wheel:
#        ./build_vllm_torchtpu.sh --local <wheel-version> [vllm-ref]
#
#   [vllm-ref] defaults to the vLLM ref pinned in this repo's pyproject.toml,
#   i.e. the vLLM version vllm-torchtpu-core is tested against.
#
# Examples:
#   ./build_vllm_torchtpu.sh 0.29.0rc1 0.29.0rc1 v0.29.0
#   ./build_vllm_torchtpu.sh --local 0.29.0rc1
#
# Environment:
#   VLLM_REPO      vLLM git remote to clone (default: upstream vllm-project/vllm).
#                  Set this to build against a fork.
#   VLLM_SRC_DIR   Dedicated vLLM checkout used for the build
#                  (default: <repo-root>/build/vllm-torchtpu-src, gitignored).
#                  Never point this at a checkout you develop in: the script
#                  checks out other refs and temporarily edits files in it.
#   CORE_DIST_DIR  Where --local looks for the core wheel
#                  (default: <repo-root>/dist).
#   CORE_WHEEL     Explicit path to the core wheel for --local. Required when
#                  CORE_DIST_DIR holds more than one core wheel.
#
# Output:
#   ${VLLM_SRC_DIR}/dist/vllm_torchtpu-<wheel-version>-*.whl
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# .buildkite/scripts -> .buildkite -> repo root
TORCHTPU_ROOT="$(cd "${SCRIPT_DIR}/../../" && pwd)"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
VLLM_SRC_DIR="${VLLM_SRC_DIR:-${TORCHTPU_ROOT}/build/vllm-torchtpu-src}"
CORE_DIST_DIR="${CORE_DIST_DIR:-${TORCHTPU_ROOT}/dist}"

# The vLLM ref pinned by the `vllm` dependency group in pyproject.toml. Used as the default
# [vllm-ref] so the combined wheel ships core on the vLLM it is tested against.
PINNED_VLLM_REF="$(sed -nE \
    '\|.*"vllm @ git\+https://github\.com/vllm-project/vllm\.git@([^"#]+)".*|{s||\1|p;q;}' \
    "${TORCHTPU_ROOT}/pyproject.toml")"
DEFAULT_VLLM_REF="${PINNED_VLLM_REF}"

usage() {
    echo "Usage: $0 <core-version> <wheel-version> [vllm-ref]"
    echo "       $0 --local <wheel-version> [vllm-ref]"
    echo ""
    echo "  <core-version>   Published vllm-torchtpu-core version to depend on."
    echo "  <wheel-version>  Version stamped onto the produced vllm-torchtpu wheel."
    echo "  [vllm-ref]       vLLM branch, tag or commit to build"
    echo "                   (default: ${DEFAULT_VLLM_REF:-<not found>}, pinned in pyproject.toml)."
    echo ""
    echo "  --local          Depend on a locally built core wheel: \$CORE_WHEEL if"
    echo "                   set, otherwise the single core wheel in ${CORE_DIST_DIR}."
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# --- Argument parsing & validation -------------------------------------------
WHL_PATH=""
if [[ "${1:-}" == "--local" ]]; then
    shift

    if [ "$#" -lt 1 ]; then
        usage
        exit 1
    fi

    if [ -n "${CORE_WHEEL:-}" ]; then
        [ -f "${CORE_WHEEL}" ] || die "CORE_WHEEL='${CORE_WHEEL}' does not exist."
        WHL_PATH="${CORE_WHEEL}"
    else
        shopt -s nullglob
        CORE_WHEELS=("${CORE_DIST_DIR}"/vllm_torchtpu_core-*.whl)
        shopt -u nullglob

        if [ "${#CORE_WHEELS[@]}" -eq 0 ]; then
            echo "ERROR: no vllm_torchtpu_core-*.whl found in ${CORE_DIST_DIR}." >&2
            echo "Build the core wheel first, from the repo root:" >&2
            echo "  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE=0.0.0 uv build --wheel" >&2
            exit 1
        fi
        # dist/ is gitignored and accumulates wheels across builds. Picking
        # one arbitrarily could bake a stale core into the combined wheel.
        if [ "${#CORE_WHEELS[@]}" -gt 1 ]; then
            echo "ERROR: multiple core wheels found in ${CORE_DIST_DIR}:" >&2
            printf '  %s\n' "${CORE_WHEELS[@]}" >&2
            echo "Set CORE_WHEEL=<path> or remove the stale wheels." >&2
            exit 1
        fi
        WHL_PATH="${CORE_WHEELS[0]}"
    fi

    # A PEP 508 direct reference must be an absolute file:// URL. A bare or
    # relative path parses fine in a requirements.txt but is rejected in
    # install_requires, which is where setup.py puts everything it reads out of
    # common.txt.
    WHL_PATH="$(cd "$(dirname "${WHL_PATH}")" && pwd)/$(basename "${WHL_PATH}")"

    CORE_VERSION=$(basename "$WHL_PATH" | cut -d'-' -f2)
    WHEEL_VERSION=$1
    VLLM_REF=${2:-"${DEFAULT_VLLM_REF}"}
else
    if [ "$#" -lt 2 ]; then
        usage
        exit 1
    fi

    CORE_VERSION=$1
    WHEEL_VERSION=$2
    VLLM_REF=${3:-"${DEFAULT_VLLM_REF}"}
fi

[ -n "${VLLM_REF}" ] || die "no vLLM ref pinned in ${TORCHTPU_ROOT}/pyproject.toml" \
    "(expected a 'vllm @ git+https://github.com/vllm-project/vllm.git@<ref>' entry);" \
    "pass [vllm-ref] explicitly."

echo "--- Starting vllm-torchtpu wheel build ---"
echo "Mode                   : $([[ "${WHL_PATH}" ]] && echo "Local wheel (${WHL_PATH})" || echo "PyPI")"
echo "vllm-torchtpu-core ver : ${CORE_VERSION}"
echo "vllm-torchtpu version  : ${WHEEL_VERSION}"
echo "vLLM repo              : ${VLLM_REPO}"
echo "vLLM ref               : ${VLLM_REF}"
echo "vLLM source dir        : ${VLLM_SRC_DIR}"

# --- Step 1: Prepare the vLLM checkout ----------------------------------------
if [ -d "${VLLM_SRC_DIR}/.git" ]; then
    echo "Reusing vLLM checkout at ${VLLM_SRC_DIR}."
    # Refuse to clobber local edits (including a common.txt left modified by a
    # previous run that was SIGKILLed before its cleanup could run).
    if [ -n "$(git -C "${VLLM_SRC_DIR}" status --porcelain --untracked-files=no)" ]; then
        echo "ERROR: ${VLLM_SRC_DIR} has uncommitted changes:" >&2
        git -C "${VLLM_SRC_DIR}" status --short --untracked-files=no >&2
        echo "Revert them (git -C ${VLLM_SRC_DIR} checkout -- .) or use another VLLM_SRC_DIR." >&2
        exit 1
    fi
    git -C "${VLLM_SRC_DIR}" remote set-url origin "${VLLM_REPO}"
    echo "Fetching ${VLLM_REPO}..."
    git -C "${VLLM_SRC_DIR}" fetch --tags --force --prune origin
elif [ -e "${VLLM_SRC_DIR}" ]; then
    die "${VLLM_SRC_DIR} exists but is not a git checkout."
else
    echo "Cloning vLLM from ${VLLM_REPO} into ${VLLM_SRC_DIR}..."
    mkdir -p "$(dirname "${VLLM_SRC_DIR}")"
    git clone "${VLLM_REPO}" "${VLLM_SRC_DIR}"
fi
VLLM_SRC_DIR="$(cd "${VLLM_SRC_DIR}" && pwd)"
cd "${VLLM_SRC_DIR}"

# Prefer the freshly fetched remote branch so a stale local branch of the same
# name is never built; fall back to tags and commit SHAs.
if git rev-parse --verify --quiet "refs/remotes/origin/${VLLM_REF}^{commit}" >/dev/null; then
    CHECKOUT_TARGET="origin/${VLLM_REF}"
else
    CHECKOUT_TARGET="${VLLM_REF}"
fi
echo "Checking out vLLM ref: ${CHECKOUT_TARGET}..."
git checkout --detach "${CHECKOUT_TARGET}" || die "failed to checkout '${VLLM_REF}'."

# The renamer lives in the vLLM tree, so the checked-out ref must contain it.
# Fail here with a clear message rather than deep inside the build.
BUILD_SCRIPT="tools/vllm-torchtpu/build.sh"
if [ ! -f "$BUILD_SCRIPT" ]; then
    echo "ERROR: ${BUILD_SCRIPT} not found in vLLM ref '${VLLM_REF}' of ${VLLM_REPO}." >&2
    echo "       Upstream vLLM does not ship it yet; set VLLM_REPO and/or" >&2
    echo "       [vllm-ref] to a fork or branch that contains it." >&2
    exit 1
fi

# --- Step 2: Inject the vllm-torchtpu-core dependency ------------------------
# VLLM_TARGET_DEVICE=empty makes setup.py read common.txt, so that is the only
# file the dependency can land in for this build.
REQUIREMENTS_FILE="${VLLM_SRC_DIR}/requirements/common.txt"
BACKUP_FILE="${REQUIREMENTS_FILE}.bak"
TMP_FILE="${REQUIREMENTS_FILE}.tmp"

if [[ "${WHL_PATH}" ]]; then
    REQUIRED_LINE="vllm-torchtpu-core @ file://${WHL_PATH}"
    echo "WARNING: --local embeds '${REQUIRED_LINE}' in the wheel metadata." >&2
    echo "         The resulting wheel is for local testing only: it will not" >&2
    echo "         install on other machines and PyPI rejects direct references." >&2
else
    REQUIRED_LINE="vllm-torchtpu-core==${CORE_VERSION}"
fi

# --- Cleanup -----------------------------------------------------------------
# common.txt is shared by every vLLM platform, so leaving it modified would
# silently poison any later build in this checkout. Registered BEFORE the first
# mutation so an interrupt or failed build mid-way still restores it.
cleanup() {
    rm -f "$TMP_FILE"
    if [ -f "$BACKUP_FILE" ]; then
        echo "Reverting ${REQUIREMENTS_FILE}..."
        mv -f "$BACKUP_FILE" "$REQUIREMENTS_FILE"
    fi
}
trap cleanup EXIT
# Exit explicitly on signals so the script stops instead of resuming with the
# next command; the EXIT trap then performs the revert exactly once.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 131' QUIT
trap 'exit 141' PIPE
trap 'exit 143' TERM

echo "Injecting '${REQUIRED_LINE}' into ${REQUIREMENTS_FILE}..."
cp "$REQUIREMENTS_FILE" "$BACKUP_FILE"

# A file without a trailing newline would glue our entry onto the last
# requirement, producing one corrupt line instead of two valid ones.
if [ -n "$(tail -c 1 "$REQUIREMENTS_FILE")" ]; then
    echo "" >> "$REQUIREMENTS_FILE"
fi

# Drop any pre-existing entry before appending, so repeat runs do not add
# duplicates. The pattern matches both the `==` (PyPI) and `@ file://...`
# (--local) forms.
if grep -qE '^vllm-torchtpu-core([[:space:]]|==|@|$)' "$REQUIREMENTS_FILE"; then
    echo "(Existing vllm-torchtpu-core entry found. Replacing.)"
    grep -vE '^vllm-torchtpu-core([[:space:]]|==|@|$)' "$REQUIREMENTS_FILE" > "$TMP_FILE"
    mv "$TMP_FILE" "$REQUIREMENTS_FILE"
fi
echo "$REQUIRED_LINE" >> "$REQUIREMENTS_FILE"

# --- Step 3: Build -----------------------------------------------------------
# Install into the interpreter that will run the build, not whichever `pip`
# happens to be first on PATH.
echo "Ensuring 'build' package is installed..."
python3 -m pip install --quiet build

echo "Executing the vLLM torchtpu build script..."
bash "$BUILD_SCRIPT" "${WHEEL_VERSION}"

echo "--- Build complete! ---"
ls -lh "${VLLM_SRC_DIR}/dist/" || true
echo "The wheel file can be found in ${VLLM_SRC_DIR}/dist."
