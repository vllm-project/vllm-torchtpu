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
# shellcheck shell=bash

# Shared by the vllm-torchtpu-release steps. Sourced, not run.

# Where the staged images go before anyone approves, and where approved images
# are copied to. GAR is internal; Docker Hub is what users pull.
RELEASE_GAR="${RELEASE_GAR:-us-central1-docker.pkg.dev/cloud-ullm-inference-ci-cd/vllm-torchtpu}"
RELEASE_DOCKERHUB_REPO="${RELEASE_DOCKERHUB_REPO:-vllm/vllm-torchtpu}"
RELEASE_DOCKERHUB_USERNAME="${RELEASE_DOCKERHUB_USERNAME:-vllm}"

# Normalized X.Y.Z public PEP 440 version: PyPI rejects local (+...) parts,
# and setuptools normalizes away leading zeros.
_N="(0|[1-9][0-9]*)"
RELEASE_BRANCH_REGEX="^releases/v(${_N}\.${_N}\.${_N})$"
PEP440_PUBLIC_REGEX="^${_N}\.${_N}\.${_N}((a|b|rc)${_N})?(\.post${_N})?(\.dev${_N})?$"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# The full SHA of the commit the checkout at <repo_root> holds, which is what
# its images are built from. BUILDKITE_COMMIT is the fallback for a checkout
# without git metadata, and is not trusted first because it can read "HEAD"
# when a build is created without a commit. git is asked only when the
# checkout has its own .git: otherwise it would climb into an enclosing
# repository and report that one's HEAD.
image_revision() {
    local repo_root="$1" revision=""
    if [[ -e "${repo_root}/.git" ]]; then
        revision="$(git -C "${repo_root}" rev-parse HEAD 2>/dev/null || true)"
    fi
    if [[ ! "${revision}" =~ ^[0-9a-f]{40}$ ]]; then
        revision="${BUILDKITE_COMMIT:-}"
    fi
    [[ "${revision}" =~ ^[0-9a-f]{40}$ ]] \
        || die "cannot tell which commit the images are built from: no git HEAD in ${repo_root} and BUILDKITE_COMMIT='${BUILDKITE_COMMIT:-}' is not a full commit SHA."
    echo "${revision}"
}

# Look up Buildkite meta-data <key>. Prints the stored value and returns 0 when
# the key is set, returns 100 when nothing is stored, and dies on any other
# agent failure.
_read_release_metadata() {
    local key="$1" rc=0
    buildkite-agent meta-data exists "${key}" || rc=$?
    case "${rc}" in
        0)
            buildkite-agent meta-data get "${key}" \
                || die "could not read meta-data ${key}."
            ;;
        100) return 100 ;;
        *) die "could not read meta-data ${key} (exit ${rc})." ;;
    esac
}

# The commit the image build checks against preflight.sh: the one preflight
# gated on CI and stored in meta-data. Only build_images.sh checks this today;
# the wheel build steps still use their own checkouts. A checkout at <repo_root>
# holding another commit fails here, before any image is built. Running a step
# standalone, with nothing stored, uses the checkout's commit.
release_revision() {
    local repo_root="$1" checkout gated rc=0
    # errexit is off inside the caller's $(...), so a failure is passed on by hand.
    checkout="$(image_revision "${repo_root}")" || exit 1
    gated="$(_read_release_metadata release-revision)" || rc=$?
    case "${rc}" in
        0)
            [[ "${checkout}" == "${gated}" ]] \
                || die "this step's checkout is at ${checkout}, but preflight gated the release on ${gated}."
            ;;
        100) ;;
        *) exit "${rc}" ;;
    esac
    echo "${checkout}"
}

# The `docker build --label` value that records <revision> on an image.
image_revision_label() {
    echo "org.opencontainers.image.revision=$1"
}

is_valid_release_version() {
    [[ "$1" =~ ${PEP440_PUBLIC_REGEX} ]]
}

# A pre-release never moves latest on Docker Hub.
is_prerelease() {
    [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

# Resolve and validate the release version from RELEASE_VERSION (an rc, e.g.
# v0.30.0rc1) if set, else from BUILDKITE_BRANCH (releases/vX.Y.Z -> X.Y.Z).
resolve_release_version() {
    local version source
    if [[ -n "${RELEASE_VERSION:-}" ]]; then
        version="${RELEASE_VERSION#v}"
        source="RELEASE_VERSION=${RELEASE_VERSION}"
    else
        [[ "${BUILDKITE_BRANCH:-}" =~ ${RELEASE_BRANCH_REGEX} ]] \
            || die "not a release branch: '${BUILDKITE_BRANCH:-}'."
        version="${BASH_REMATCH[1]}"
        source="branch ${BUILDKITE_BRANCH}"
    fi
    is_valid_release_version "${version}" \
        || die "version '${version}' from ${source} is not a public PEP 440 version."
    echo "Release version: ${version} (from ${source})" >&2
    echo "${version}"
}

# The version every step of the build agrees on, set once by preflight.sh
# (or resolved on the spot when running a step standalone).
release_version() {
    local rc=0
    _read_release_metadata release-version || rc=$?
    case "${rc}" in
        0) ;;
        100) resolve_release_version ;;
        *) exit "${rc}" ;;
    esac
}

# Fail if <project>==<version> is already on PyPI (or <index_base>).
check_not_published() {
    local project="$1" version="$2" index_base="${3:-https://pypi.org}"
    local url="${index_base%/}/pypi/${project}/${version}/json" code
    code="$(curl -sL -o /dev/null -w '%{http_code}' --max-time 30 "${url}")" \
        || die "could not reach ${url}."
    case "${code}" in
        404) echo "${project} ${version}: not on ${index_base%/} yet" ;;
        200) die "${project} ${version} is already on ${index_base%/}; a release can't replace it." ;;
        *) die "${index_base%/} answered ${code} for ${project} ${version}." ;;
    esac
}

# Install uv into <dir> and print its binary path.
install_uv() {
    local dir="$1"
    curl -LsSf --retry 3 https://astral.sh/uv/install.sh \
        | env UV_INSTALL_DIR="${dir}" UV_NO_MODIFY_PATH=1 sh >/dev/null \
        || die "could not install uv into ${dir}."
    [[ -x "${dir}/uv" ]] || die "uv install finished but ${dir}/uv is missing."
    echo "${dir}/uv"
}

# Resolve vllm-torchtpu==<version> for the supported platform (CPython 3.12,
# x86_64 Linux, glibc >= 2.31) into <out_file>, passing any extra uv flags.
resolve_vllm_torchtpu() {
    local uv_bin="$1" version="$2" out_file="$3"
    shift 3
    local req_file
    req_file="$(dirname "${out_file}")/req.in"
    echo "vllm-torchtpu==${version}" > "${req_file}"
    "${uv_bin}" pip compile "${req_file}" --no-config --quiet \
        --python-version 3.12 --python-platform x86_64-manylinux_2_31 \
        --index-url https://pypi.org/simple \
        "$@" -o "${out_file}"
}

# Log in to Docker Hub inside a private <config_dir> and unset the secret
# token so no child process inherits it.
dockerhub_login() {
    local config_dir="$1" username="${2:-${RELEASE_DOCKERHUB_USERNAME}}"
    { set +x; } 2>/dev/null # Never trace the token into the build log.
    : "${DOCKERHUB_TOKEN_VLLM_TORCHTPU:?DOCKERHUB_TOKEN_VLLM_TORCHTPU must be set}"
    printf '%s' "${DOCKERHUB_TOKEN_VLLM_TORCHTPU}" \
        | DOCKER_CONFIG="${config_dir}" docker login -u "${username}" --password-stdin >/dev/null \
        || die "Docker Hub login failed for ${username}."
    unset DOCKERHUB_TOKEN_VLLM_TORCHTPU
}

# The digest part of <repo>@sha256:<hex>.
digest_of() {
    echo "${1##*@}"
}
