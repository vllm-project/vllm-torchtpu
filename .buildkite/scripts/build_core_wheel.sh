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
# Build the vllm-torchtpu-core release wheel.
#
# Version:
#   - Official release (default): the release branch name without
#     "releases/v" (e.g. releases/vX.Y.Z -> X.Y.Z). Must be exactly
#     releases/vX.Y.Z (RELEASE_BRANCH_REGEX in configs/pipeline_config.sh).
#   - Pre-release (e.g. rc): set RELEASE_VERSION when starting the build
#     (Buildkite "New Build" -> Environment Variables), e.g.
#     RELEASE_VERSION=v0.29.0rc1. Used as-is; a leading "v" is dropped.
# Either way it must be a valid PEP 440 version that PyPI accepts (no local
# "+..." part), or the script exits 1. The version is forced onto
# setuptools-scm, so a shallow CI checkout without tags still gets it.
#
# Output:
#   dist/<core>-<version>-py3-none-any.whl, uploaded as a Buildkite artifact.
#   Build meta-data CORE_RELEASE_VERSION / CORE_WHEEL_NAME for later steps.
#
# CI only: runs inside a Buildkite release build (see bootstrap.sh).
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
WHEEL_UTILS="${SCRIPT_DIR}/wheel_utils.py"
# Provides RELEASE_BRANCH_REGEX.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v buildkite-agent >/dev/null 2>&1 \
    || die "buildkite-agent not found: this script only runs in Buildkite."

# --- Resolve the release version ---------------------------------------------
echo "--- :label: Resolving release version"
if [[ -n "${RELEASE_VERSION:-}" ]]; then
    VERSION="${RELEASE_VERSION#v}"
    VERSION_SOURCE="RELEASE_VERSION=${RELEASE_VERSION}"
    echo "Pre-release: version ${VERSION} from ${VERSION_SOURCE}"
else
    BRANCH="${BUILDKITE_BRANCH:-}"
    [[ "${BRANCH}" =~ ${RELEASE_BRANCH_REGEX} ]] \
        || die "not a release branch: '${BRANCH}' is not releases/vX.Y.Z" \
               "For an rc, set RELEASE_VERSION."
    VERSION="${BRANCH#releases/v}"
    VERSION_SOURCE="branch ${BRANCH}"
    echo "Official release: version ${VERSION} from ${VERSION_SOURCE}"
fi

# --- Private build venv ------------------------------------------------------
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/core-build.XXXXXX")"
trap 'rm -rf "${WORK_DIR}"' EXIT

# CI agents lack python3-venv (no ensurepip), so create the venv without pip
# and bootstrap pip with PyPA's get-pip.py.
"${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
    || die "could not create a venv with ${PYTHON}."
PY="${WORK_DIR}/venv/bin/python"
curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
    https://bootstrap.pypa.io/get-pip.py \
    || die "could not download get-pip.py from bootstrap.pypa.io."
"$PY" "${WORK_DIR}/get-pip.py" --quiet
"$PY" -m pip install --quiet build twine packaging \
    "setuptools>=61.0.0" wheel "setuptools_scm[toml]>=8.0"

# Exit 1 unless the version is valid PEP 440 and accepted by PyPI (e.g. a
# mistyped branch like releases/v0.29.x or releases/v0.29.0+abc); prints it normalized.
VERSION="$("$PY" "$WHEEL_UTILS" validate_version "$VERSION")" \
    || die "version from ${VERSION_SOURCE} is not publishable to PyPI."
echo "Release version: ${VERSION}"

# Stop before building if this version is already on PyPI (versions can't be
# reused). PUBLISH_PYPI comes from the step's env in pipeline_release_pypi.yml.
[[ -n "${PUBLISH_PYPI:-}" ]] || die "PyPI project (PUBLISH_PYPI) is not set."
INDEX_BASE="${PYPI_BASE_URL:-https://pypi.org}"
"$PY" "$WHEEL_UTILS" check_not_published "${INDEX_BASE%/}" "$PUBLISH_PYPI" "$VERSION"

# --- Build -------------------------------------------------------------------
echo "--- :package: Building wheel"
cd "${REPO_ROOT}"
rm -rf dist
SETUPTOOLS_SCM_PRETEND_VERSION="${VERSION}" "$PY" -m build --wheel --no-isolation

shopt -s nullglob
WHEELS=(dist/*.whl)
shopt -u nullglob
[[ "${#WHEELS[@]}" -eq 1 ]] \
    || die "expected exactly one wheel in dist/, found ${#WHEELS[@]}."
WHEEL="${WHEELS[0]}"

# The file name must carry exactly the release version; anything else means
# the version override did not take effect.
"$PY" "$WHEEL_UTILS" verify_built_wheel "$WHEEL" "$VERSION"

echo "--- :mag: Checking metadata"
"$PY" -m twine check --strict "$WHEEL"

# --- Hand off to later steps -------------------------------------------------
buildkite-agent artifact upload "$WHEEL"
buildkite-agent meta-data set "CORE_RELEASE_VERSION" "$VERSION"
buildkite-agent meta-data set "CORE_WHEEL_NAME" "$(basename "$WHEEL")"

echo "Built ${WHEEL}"
