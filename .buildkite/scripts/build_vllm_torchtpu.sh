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
# Build the combined `vllm-torchtpu` wheel and sdist from upstream vLLM.
#
# The result is vLLM renamed to `vllm-torchtpu`, with a pip dependency on
# `vllm-torchtpu-core` (this repo). The core is not bundled; pip installs it.
#
# Steps (all on a temp clone of vLLM, deleted on exit; nothing is committed):
#   1. Shallow-clone vLLM at the ref pinned in this repo's pyproject.toml
#      (`vllm @ git+...@<ref>`), the same ref CI tests against.
#   2. Rename the package "vllm" -> "vllm-torchtpu" in pyproject.toml and in
#      vLLM's own importlib.metadata lookups (else they fail at runtime).
#   3. Add vllm-torchtpu-core==<version> to requirements/common.txt, which is
#      where setup.py reads dependencies from when VLLM_TARGET_DEVICE=empty.
#   4. Build the wheel and sdist into OUTPUT_DIR.
#   5. Retag the wheel linux_<arch> -> manylinux_2_28_x86_64, since PyPI
#      rejects linux_<arch>. Refused if the wheel contains native binaries.
#   6. `twine check --strict` the wheel and sdist (metadata PyPI would reject).
#
# Usage:
#   ./build_vllm_torchtpu.sh <version>
#
#   <version>    vllm-torchtpu-core version to pin (must be on PyPI); also
#                used as the vllm-torchtpu version.
#
# Examples:
#   ./build_vllm_torchtpu.sh X.Y.Z        # official release
#   ./build_vllm_torchtpu.sh X.Y.ZrcN     # pre-release
#
# Requirements (checked up front):
#   - Linux (vLLM forces a CPU build on macOS). No TPU needed.
#   - git, curl, Python 3.12, and network access to github.com, PyPI,
#     bootstrap.pypa.io and download.pytorch.org.
#   - No Rust toolchain (cargo/rustc) on PATH, or vLLM compiles native
#     extensions and the wheel can't be retagged.
#   Build dependencies go into a private temp venv, so the caller's packages
#   (e.g. TPU torch on a TPU VM) are never touched.
#
# Building a wheel from the sdist later needs the same VLLM_TARGET_DEVICE=empty
# and VLLM_VERSION_OVERRIDE as below; otherwise vLLM builds its CUDA target.
#
# Environment:
#   VLLM_REPO          vLLM git remote (default: upstream vllm-project/vllm).
#   VLLM_REF_OVERRIDE  Dev only: build this vLLM branch, tag or full SHA
#                      instead of the pyproject.toml pin. Never set it for a
#                      release; the result was not tested with this core.
#   OUTPUT_DIR         Output directory (default: <repo-root>/dist). Old
#                      vllm_torchtpu-* files are removed; core files are kept.
#   KEEP_VLLM_SRC      Set to 1 to keep the patched clone for debugging.
#   PYTHON             Interpreter for the build venv (default: python3).
#   PIP_INDEX_URL      Standard pip setting; used by the core-version check.
#
# Output:
#   ${OUTPUT_DIR}/vllm_torchtpu-<version>-cpXY-cpXY-manylinux_2_28_x86_64.whl
#   ${OUTPUT_DIR}/vllm_torchtpu-<version>.tar.gz
#   (cpXY = the Python that ran the build, e.g. cp312)
# ------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCHTPU_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# Provides get_vllm_commit_hash (the vLLM pin in pyproject.toml).
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

VLLM_REPO="${VLLM_REPO:-https://github.com/vllm-project/vllm.git}"
OUTPUT_DIR="${OUTPUT_DIR:-${TORCHTPU_ROOT}/dist}"
KEEP_VLLM_SRC="${KEEP_VLLM_SRC:-0}"
PYTHON="${PYTHON:-python3}"
# Use own build deps under requirements/build/tpu.txt
BUILD_REQUIREMENTS_FILE="${TORCHTPU_ROOT}/requirements/build/tpu.txt"

# Linux x86_64 with glibc >= 2.28, the same baseline as upstream vLLM.
WHEEL_PLATFORM_TAG="manylinux_2_28_x86_64"

# vLLM files that look up their own package name via importlib.metadata.
METADATA_FILES=(
    "vllm/entrypoints/cli/main.py"
    "vllm/entrypoints/cli/run_batch.py"
    "vllm/utils/__init__.py"
    "vllm/platforms/__init__.py"
    "vllm/triton_utils/importing.py"
    "vllm/utils/import_utils.py"
    "vllm/snapshot/runtime.py"
)

usage() {
    echo "Usage: $0 <version>"
    echo ""
    echo "  <version>   Release version for both packages. The vllm-torchtpu wheel"
    echo "              gets this version and pins vllm-torchtpu-core==<version>,"
    echo "              which must already be published, e.g. X.Y.ZrcN -> both X.Y.ZrcN."
    echo ""
    echo "vLLM is built at the ref pinned in pyproject.toml (tag or full commit SHA)"
    echo "(VLLM_REF_OVERRIDE=<ref> for dev builds only)."
    echo ""
    echo "Examples: $0 X.Y.Z        # official release"
    echo "          $0 X.Y.ZrcN     # pre-release"
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "$2"
}

# --- Arguments ---------------------------------------------------------------
if [[ "$#" -ne 1 ]]; then
    usage
    exit 1
fi

CORE_VERSION=$1
WHEEL_VERSION="${CORE_VERSION}"
[[ -n "${CORE_VERSION}" ]] || die "CORE_VERSION must not be empty."

# The vLLM ref CI tested: the pin in pyproject.toml, unless overridden.
PINNED_VLLM_REF="$(get_vllm_commit_hash)" \
    || die "could not read the vLLM pin from ${TORCHTPU_ROOT}/pyproject.toml."
VLLM_REF="${VLLM_REF_OVERRIDE:-${PINNED_VLLM_REF}}"
if [[ "${VLLM_REF}" != "${PINNED_VLLM_REF}" ]]; then
    echo "WARNING: building vLLM ${VLLM_REF} (VLLM_REF_OVERRIDE), not the" \
         "pyproject.toml pin ${PINNED_VLLM_REF}. Do not publish this wheel." >&2
fi

# Make it absolute; the build runs from inside the temp clone.
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd "${OUTPUT_DIR}" && pwd)"

# --- Preflight checks (fail fast, before the long build) ---------------------
[[ "$(uname -s)" == "Linux" ]] \
    || die "must run on Linux (found $(uname -s)); vLLM forces a CPU build" \
           "on macOS. Use a Linux machine, TPU VM or container."
require_cmd git "git not found (apt-get install git)."
require_cmd curl "curl not found (apt-get install curl)."
require_cmd "${PYTHON}" "${PYTHON} not found; set PYTHON=<interpreter>."
"${PYTHON}" -c 'import sys; sys.exit(not ((3, 12) <= sys.version_info < (3, 13)))' \
    || die "Python 3.12 required (vllm-torchtpu-core requires >=3.12,<3.13)," \
           "found $("${PYTHON}" -V 2>&1)."
for tool in cargo rustc; do
    if command -v "$tool" >/dev/null 2>&1; then
        die "${tool} found at $(command -v "$tool"). With Rust on PATH vLLM" \
            "builds native extensions and the wheel cannot be retagged" \
            "${WHEEL_PLATFORM_TAG}. Run without Rust on PATH."
    fi
done

# --- Temp workspace and cleanup ----------------------------------------------
# Set up before cloning so the clone is always removed, even on failure.
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/vllm-torchtpu-build.XXXXXX")"
VLLM_SRC_DIR="${WORK_DIR}/vllm"
cleanup() {
    if [[ "${KEEP_VLLM_SRC}" == "1" ]]; then
        echo "KEEP_VLLM_SRC=1: patched vLLM source kept at ${VLLM_SRC_DIR}"
    else
        rm -rf "${WORK_DIR}"
    fi
}
trap cleanup EXIT
# Stop on signals instead of continuing; the EXIT trap then cleans up once.
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 131' QUIT
trap 'exit 141' PIPE
trap 'exit 143' TERM

# Private build venv. Created without pip (CI agents lack python3-venv), then
# pip is installed from get-pip.py.
echo "Creating build venv with $("${PYTHON}" -V 2>&1)..."
"${PYTHON}" -m venv --without-pip "${WORK_DIR}/venv" \
    || die "could not create a venv with ${PYTHON}."
PY="${WORK_DIR}/venv/bin/python"
curl -fsSL --retry 3 -o "${WORK_DIR}/get-pip.py" \
    https://bootstrap.pypa.io/get-pip.py \
    || die "could not download get-pip.py from bootstrap.pypa.io."
"$PY" "${WORK_DIR}/get-pip.py" --quiet

# The pinned core must already be on PyPI, or the release can't be installed.
# --ignore-requires-python: we only check the file exists.
# TODO: TEMP (CI test): the core is not on PyPI yet, so the check is disabled.
# Restore it before merging.
# echo "Checking that vllm-torchtpu-core==${CORE_VERSION} is published on PYPI..."
# "$PY" -m pip download --quiet --no-deps --no-cache-dir \
#     --ignore-requires-python --only-binary :all: \
#     --dest "${WORK_DIR}/core-check" \
#     "vllm-torchtpu-core==${CORE_VERSION}" \
#     || die "vllm-torchtpu-core==${CORE_VERSION} is not available on the package" \
#            "index. Upload the core release first, then build the wrapper."

echo "--- Starting vllm-torchtpu wheel build ---"
echo "vllm-torchtpu-core ver : ${CORE_VERSION}"
echo "vllm-torchtpu version  : ${WHEEL_VERSION}"
echo "vLLM repo              : ${VLLM_REPO}"
echo "vLLM ref               : ${VLLM_REF}"
echo "Output dir             : ${OUTPUT_DIR}"

# --- Step 1: Shallow-clone vLLM ----------------------------------------------
# init + fetch (not `clone --branch`) so full commit SHAs work too.
echo "Fetching ${VLLM_REF} from ${VLLM_REPO} (shallow)..."
git init --quiet "${VLLM_SRC_DIR}"
cd "${VLLM_SRC_DIR}"
git remote add origin "${VLLM_REPO}"
git fetch --quiet --depth 1 origin "${VLLM_REF}" \
    || die "failed to fetch '${VLLM_REF}' from ${VLLM_REPO}" \
           "(short SHAs are not supported; use a branch, tag or full SHA)."
git checkout --quiet --detach FETCH_HEAD
echo "vLLM commit            : $(git rev-parse HEAD)"

PYPROJECT_FILE="pyproject.toml"
REQUIREMENTS_FILE="requirements/common.txt"
for f in "$PYPROJECT_FILE" "$REQUIREMENTS_FILE"; do
    [[ -f "$f" ]] || die "${f} not found in vLLM ref '${VLLM_REF}'."
done

# --- Step 2: Rename the package to vllm-torchtpu -----------------------------
# Replace only the first `name = "vllm"` line (the [project] name).
# awk instead of sed -i so it works on both GNU and BSD.
echo "Renaming distribution in ${PYPROJECT_FILE} to vllm-torchtpu..."
awk '!done && /^name = "vllm"[[:space:]]*$/ { print "name = \"vllm-torchtpu\""; done = 1; next } { print }' \
    "$PYPROJECT_FILE" > "${PYPROJECT_FILE}.tmp"
mv "${PYPROJECT_FILE}.tmp" "$PYPROJECT_FILE"
grep -q '^name = "vllm-torchtpu"' "$PYPROJECT_FILE" \
    || die "could not find 'name = \"vllm\"' in ${PYPROJECT_FILE}."

# Rewrite version('vllm') / metadata('vllm') to use 'vllm-torchtpu'
# (both quote styles). `-i.bak` + rm works with GNU and BSD sed.
echo "Repointing importlib.metadata lookups of 'vllm' to 'vllm-torchtpu'..."
for f in "${METADATA_FILES[@]}"; do
    if [[ ! -f "$f" ]]; then
        echo "WARNING: ${f} not found in this vLLM ref; skipping." >&2
        continue
    fi
    sed -i.bak \
        -e "s/importlib.metadata.version(\(['\"]\)vllm\1)/importlib.metadata.version(\1vllm-torchtpu\1)/g" \
        -e "s/importlib.metadata.metadata(\(['\"]\)vllm\1)/importlib.metadata.metadata(\1vllm-torchtpu\1)/g" \
        -e "s/version(\(['\"]\)vllm\1)/version(\1vllm-torchtpu\1)/g" \
        "$f"
    rm -f "${f}.bak"
done

# Fail on lookups in files missing from METADATA_FILES (new upstream code):
# unpatched, they raise PackageNotFoundError at runtime.
REMAINING="$(grep -rnE "(metadata|version)\((['\"])vllm\2\)" vllm --include='*.py' || true)"
if [[ -n "$REMAINING" ]]; then
    echo "ERROR: unpatched importlib.metadata lookups of 'vllm' remain:" >&2
    echo "$REMAINING" >&2
    echo "         Add these files to METADATA_FILES in $0." >&2
    exit 1
fi

# --- Step 3: Pin vllm-torchtpu-core ------------------------------------------
REQUIRED_LINE="vllm-torchtpu-core==${CORE_VERSION}"
echo "Adding '${REQUIRED_LINE}' to ${REQUIREMENTS_FILE}..."
# Add a newline first if the file lacks one, so the lines don't merge.
if [[ -n "$(tail -c 1 "$REQUIREMENTS_FILE")" ]]; then
    echo "" >> "$REQUIREMENTS_FILE"
fi
echo "$REQUIRED_LINE" >> "$REQUIREMENTS_FILE"

# --- Step 4: Build -----------------------------------------------------------
# setup.py imports torch, setuptools-rust and setuptools-scm, so they must be
# installed for a --no-isolation build. Our requirements/build/tpu.txt is just
# build tools + CPU torch; `build` fails if it no longer fits vLLM's [build-system].
echo "Installing build dependencies..."
"$PY" -m pip install -r "$BUILD_REQUIREMENTS_FILE" build wheel twine

# Remove old wrapper files only; `vllm_torchtpu-` never matches core files
# (`vllm_torchtpu_core-`).
rm -f "${OUTPUT_DIR}"/vllm_torchtpu-*.whl "${OUTPUT_DIR}"/vllm_torchtpu-*.tar.gz

echo "Building vllm-torchtpu sdist and wheel into ${OUTPUT_DIR}..."
VLLM_TARGET_DEVICE=empty \
VLLM_VERSION_OVERRIDE="${WHEEL_VERSION}" \
SETUPTOOLS_SCM_PRETEND_VERSION="${WHEEL_VERSION}" \
    "$PY" -m build --sdist --wheel --no-isolation --outdir "${OUTPUT_DIR}"

shopt -s nullglob
BUILT_SDISTS=("${OUTPUT_DIR}"/vllm_torchtpu-*.tar.gz)
BUILT_WHEELS=("${OUTPUT_DIR}"/vllm_torchtpu-*.whl)
shopt -u nullglob
[[ "${#BUILT_SDISTS[@]}" -eq 1 ]] \
    || die "expected one vllm_torchtpu-*.tar.gz in ${OUTPUT_DIR}," \
           "found ${#BUILT_SDISTS[@]}."
[[ "${#BUILT_WHEELS[@]}" -eq 1 ]] \
    || die "expected one vllm_torchtpu-*.whl in ${OUTPUT_DIR}," \
           "found ${#BUILT_WHEELS[@]}."
BUILT_SDIST="${BUILT_SDISTS[0]}"
BUILT_WHEEL="${BUILT_WHEELS[0]}"

# --- Step 5: Retag the wheel as manylinux ------------------------------------
# Only safe for a pure-Python wheel, so refuse if any native binary is inside.
BINARIES="$("$PY" - "$BUILT_WHEEL" <<'EOF'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
print("\n".join(n for n in names
                if n.endswith((".so", ".pyd", ".dylib")) or n.endswith("/vllm-rs")))
EOF
)"
if [[ -n "$BINARIES" ]]; then
    rm -f "$BUILT_WHEEL"
    echo "ERROR: the built wheel contains native binaries:" >&2
    echo "$BINARIES" >&2
    echo "       Refusing to retag it as ${WHEEL_PLATFORM_TAG} (wheel removed)." >&2
    echo "       Build without a Rust toolchain on PATH." >&2
    exit 1
fi

# Only the platform tag changes; the Python/ABI tags (e.g. cp312-cp312) stay.
case "$(basename "$BUILT_WHEEL")" in
    *-"${WHEEL_PLATFORM_TAG}".whl)
        echo "Wheel is already ${WHEEL_PLATFORM_TAG}."
        FINAL_WHEEL="$BUILT_WHEEL" ;;
    *)
        echo "Retagging $(basename "$BUILT_WHEEL") to ${WHEEL_PLATFORM_TAG}..."
        # Writes the new wheel next to the old one, prints its name, and
        # --remove deletes the old one.
        NEW_NAME="$("$PY" -m wheel tags --remove \
            --platform-tag "${WHEEL_PLATFORM_TAG}" "$BUILT_WHEEL")"
        FINAL_WHEEL="${OUTPUT_DIR}/${NEW_NAME}" ;;
esac
[[ -f "$FINAL_WHEEL" ]] || die "retagged wheel not found at ${FINAL_WHEEL}."

# --- Step 6: Check metadata --------------------------------------------------
# Same check as the core wheel: catch metadata PyPI would reject before upload.
echo "--- Checking metadata ---"
"$PY" -m twine check --strict "$FINAL_WHEEL" "$BUILT_SDIST"

echo "--- Build complete! ---"
ls -lh "$FINAL_WHEEL" "$BUILT_SDIST"
echo "Wheel: ${FINAL_WHEEL}"
echo "Sdist: ${BUILT_SDIST}"
