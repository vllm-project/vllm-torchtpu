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

# Implemented dynamic job prioritization by injecting integers during upload
upload_with_priority() {
  local yaml_file=$1
  local JOB_PRIORITY=$2
  local capped
  local filter=(cat)
  echo "--- :pipeline: Uploading $yaml_file with priority ${JOB_PRIORITY:-$PRIORITY_DEFAULT}"
  # build.env("NIGHTLY") is evaluated by Buildkite against the build's initial
  # environment, so a NIGHTLY=1 set by bootstrap.sh (for the run-full-ci label)
  # never reaches it unless substituted before upload.
  if [[ "${NIGHTLY:-0}" == "1" ]]; then
    filter=(sed 's/build\.env("NIGHTLY")/"1"/g')
  fi
  # A pull-request build's TPU steps go in its author's concurrency group (set
  # by bootstrap.sh). The file is read once by the agent, so that the group is
  # added to the steps it would upload, anchors and interpolation resolved,
  # then uploaded as is. If any of that fails the file goes up uncapped rather
  # than failing the build.
  if [[ -n "${PR_TPU_CONCURRENCY_GROUP:-}" ]]; then
    if capped=$({
        echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
        "${filter[@]}" "$yaml_file";
      } | buildkite-agent pipeline upload --dry-run --format json \
        | python3 "$(dirname "${BASH_SOURCE[0]}")/../cap_pr_tpu_concurrency.py" \
            "$PR_TPU_CONCURRENCY_GROUP" "$PR_TPU_JOBS_PER_AUTHOR"); then
      buildkite-agent pipeline upload --no-interpolation <<< "$capped"
      return
    fi
    echo "WARNING: could not put $yaml_file in ${PR_TPU_CONCURRENCY_GROUP}; uploading it uncapped" >&2
  fi
  {
    echo "priority: ${JOB_PRIORITY:-$PRIORITY_DEFAULT}";
    "${filter[@]}" "$yaml_file";
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
