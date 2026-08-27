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

set -euo pipefail

# Resolve the absolute directory path of the current script.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the shared pipeline config file.
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/configs/pipeline_config.sh"

determine_job_priority() {
  local priority=""
  echo "--- Determining job priority" >&2
  if [[ "${NIGHTLY:-0}" == "1" ]]; then
    # Nightly build (Lowest priority)
    priority="$PRIORITY_NIGHTLY"
    echo "Build type: Nightly - Priority: $priority" >&2
  elif [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
    # Pre-merge PR tests
    priority="$PRIORITY_PRE_MERGE"
    echo "Build type: Pre-merge (PR #${BUILDKITE_PULL_REQUEST}) - Priority: $priority" >&2
  elif [[ "${BUILDKITE_BRANCH:-}" == "main" && "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
    # Post-merge tests on main (Highest priority)
    priority="$PRIORITY_POST_MERGE"
    echo "Build type: Post-merge (Main branch) - Priority: $priority" >&2
  else
    # Default priority for other branches or manual builds
    priority="$PRIORITY_DEFAULT"
    echo "Build type: General - Priority: $priority" >&2
  fi

  echo "$priority"
}

JOB_PRIORITY=$(determine_job_priority)
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"

# --- Check for explicit skip-ci or non-code/documentation-only changes ---
echo "--- :git: Checking if CI build should be skipped"

if [[ "${BUILDKITE_MESSAGE:-}" =~ \[skip[[:space:]]ci\] || "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]skip\] ]]; then
  echo "Commit message contains [skip ci]. Skipping build."
  exit 0
fi

if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  BASE_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-"main"}
  echo "PR detected. Target branch: ${BASE_BRANCH}"

  git fetch origin "${BASE_BRANCH}" --depth=20 --quiet || echo "Base fetch failed"
  git fetch origin "${BUILDKITE_COMMIT:-HEAD}" --depth=20 --quiet || true

  FILES_CHANGED=$(git diff --name-only origin/"${BASE_BRANCH}"..."${BUILDKITE_COMMIT:-HEAD}" 2>/dev/null || true)
  if [[ -z "${FILES_CHANGED}" ]]; then
    FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "${BUILDKITE_COMMIT:-HEAD}")
  fi

  echo "Files changed:"
  echo "${FILES_CHANGED}"

  # Filter out files we want to skip builds for (docs, md, icons, CODEOWNERS, LICENSE)
  NON_SKIPPABLE_FILES=$(echo "${FILES_CHANGED}" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/|^\.github\/CODEOWNERS$|^LICENSE$)" || true)

  if [[ -z "${NON_SKIPPABLE_FILES}" && -n "${FILES_CHANGED}" ]]; then
    echo "Only documentation/non-code files changed. Skipping CI build."
    exit 0
  else
    echo "Code files changed. Proceeding with pipeline upload."
  fi
fi

echo "--- Starting Buildkite Bootstrap"

# Since Buildkite prepends uploaded steps (inserts in reverse order),
# uploading the test steps first and build steps last ensures they
# are displayed and queued in correct order: Build -> Tests.

echo "Uploading Perf and Eval Pipeline"
upload_with_priority .buildkite/pipeline_perf.yml "$JOB_PRIORITY"

echo "Uploading Integration Tests Pipeline"
upload_with_priority .buildkite/pipeline_integration.yml "$JOB_PRIORITY"

echo "Uploading Unit Tests Pipeline"
upload_with_priority .buildkite/pipeline_tests.yml "$JOB_PRIORITY"

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
