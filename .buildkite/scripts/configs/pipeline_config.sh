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
# On-call fix > Post-merge > Pre-merge > Integration pipeline > Benchmark > Other/Default > Nightly
export PRIORITY_ONCALL_FIX=12
export PRIORITY_POST_MERGE=11
export PRIORITY_PRE_MERGE=6
export PRIORITY_INTEGRATION=4
export PRIORITY_BENCHMARK=3
export PRIORITY_DEFAULT=2
export PRIORITY_NIGHTLY=1
export PRIORITY_KERNEL_TUNING=-9

# How many TPU jobs on the kube fleet one author's pull-request builds may have
# in flight at once, waiting in Kueue or running. The rest wait in Buildkite as
# `limited`, holding no agent pod and no place in Kueue's queue, so one author's
# burst of PRs does not queue ahead of everyone else's. A PR build runs about
# 19, so this is two PRs at a time; 40 capped 6 of 45 authors in the week to
# 10-16. Nightly, post-merge and release builds are not PR builds and have no
# cap.
export PR_TPU_JOBS_PER_AUTHOR=40

# Implemented dynamic job prioritization by injecting integers during upload.
# Delegates to bootstrap.py so priority injection, NIGHTLY substitution, and
# PR TPU concurrency capping have a single implementation.
upload_with_priority() {
  local yaml_file=$1
  local job_priority=${2:-$PRIORITY_DEFAULT}
  python3 -u "$(dirname "${BASH_SOURCE[0]}")/../bootstrap.py" upload "$yaml_file" "$job_priority"
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
