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

die() {
    echo "ERROR: $*" >&2
    exit 1
}

# The version every step of the build agrees on, set once by preflight.sh.
release_version() {
    buildkite-agent meta-data get release-version
}

# A pre-release never moves latest on Docker Hub.
is_prerelease() {
    [[ "$1" =~ (a|b|rc|dev)[0-9]+ ]]
}

# The digest part of <repo>@sha256:<hex>.
digest_of() {
    echo "${1##*@}"
}
