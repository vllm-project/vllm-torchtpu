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

# Build the vllm-torchtpu wheel and sdist at the build's version
# (release_version in lib.sh) and upload them as artifacts. One script for the
# release (pipeline.yml) and the wheel lane (../pipeline_build_wheels.yml), so
# the lane tests the step the release runs.
#
# vLLM is built at the pyproject.toml pin, or at VLLM_REF_OVERRIDE
# (build_vllm_torchtpu.sh). The release never sets it; the wheel lane sets it to
# the ref the CI image is built from.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/lib.sh"

VERSION="$(release_version)"
# The artifact paths below are relative to the repository root, where
# build_vllm_torchtpu.sh writes dist/.
cd "${REPO_ROOT}"
bash .buildkite/scripts/build_vllm_torchtpu.sh "${VERSION}"
buildkite-agent artifact upload "dist/vllm_torchtpu-*.whl"
buildkite-agent artifact upload "dist/vllm_torchtpu-*.tar.gz"
