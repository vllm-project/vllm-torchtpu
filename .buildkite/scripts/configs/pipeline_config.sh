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

# Priority constants for pipeline jobs.
# Post-merge > Pre-merge > Integration pipeline > Benchmark > Other/Default > Nightly
export PRIORITY_POST_MERGE=11
export PRIORITY_PRE_MERGE=6
export PRIORITY_INTEGRATION=4
export PRIORITY_BENCHMARK=3
export PRIORITY_DEFAULT=2
export PRIORITY_NIGHTLY=1
export PRIORITY_KERNEL_TUNING=-9

# Implemented dynamic job prioritization by injecting integers during upload
upload_with_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=$2
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-$PRIORITY_DEFAULT}"
  {
    echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
    cat "$yaml_file";
  } | buildkite-agent pipeline upload
}

get_vllm_commit_hash() {
  # Extract the pinned vLLM commit directly from pyproject.toml; fail loud if missing
  local config_dir repo_root pyproject_path commit_hash=""
  config_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${config_dir}/../../.." && pwd)"

  pyproject_path="${repo_root}/pyproject.toml"

  commit_hash="$(sed -nE 's/.*vllm @ git\+https:\/\/github\.com\/vllm-project\/vllm\.git@([a-zA-Z0-9.-]+).*/\1/p' "${pyproject_path}" | head -n 1)"

  if [ -z "${commit_hash:-}" ]; then
    echo "ERROR: vLLM commit hash is missing from pyproject.toml. Cannot proceed without a pinned hash." >&2
    exit 1
  fi
  echo "$commit_hash"
}
