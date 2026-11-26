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

# Set the version of a test wheel build the way preflight.sh sets a release's:
# in the release-version meta-data, where release_version (lib.sh) reads it, so
# every wheel step of the build uses one version. The first command of the
# wheel lane (.buildkite/pipeline_build_wheels.yml).
#
# The version is 0.0.0+g<commit>: a local version, which PyPI rejects, and not a
# pre-release, so the core's exact pin resolves. "g" keeps the local part from
# being all digits, which would drop its leading zeros.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

# This checkout's commit, wherever the script is called from.
COMMIT="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)" \
    || die "could not read this checkout's commit."
VERSION="0.0.0+g${COMMIT:0:12}"

# A retried step finds its own version. Any other version belongs to a release
# (preflight.sh), which this script must not replace.
rc=0
buildkite-agent meta-data exists release-version || rc=$?
case "${rc}" in
    0)
        EXISTING="$(buildkite-agent meta-data get release-version)"
        [[ "${EXISTING}" == "${VERSION}" ]] \
            || die "release-version is already ${EXISTING}; not a test wheel build."
        ;;
    100) buildkite-agent meta-data set release-version "${VERSION}" ;;
    *) die "could not read meta-data release-version (exit ${rc})." ;;
esac
echo "Test wheel version: ${VERSION}"
