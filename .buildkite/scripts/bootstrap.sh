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

# Benchmark / Perf jobs should use PRIORITY_BENCHMARK (unless it is a nightly run).
PERF_PRIORITY="$PRIORITY_BENCHMARK"
if [[ "${NIGHTLY:-0}" == "1" ]]; then
  PERF_PRIORITY="$PRIORITY_NIGHTLY"
fi

# Integration jobs should use PRIORITY_INTEGRATION (unless it is a nightly run).
INTEGRATION_PRIORITY="$PRIORITY_INTEGRATION"
if [[ "${NIGHTLY:-0}" == "1" ]]; then
  INTEGRATION_PRIORITY="$PRIORITY_NIGHTLY"
fi

echo "--- Starting Buildkite Bootstrap"

# Since Buildkite prepends uploaded steps (inserts in reverse order),
# uploading the test steps first and build steps last ensures they
# are displayed and queued in correct order: Build -> Tests.

echo "Uploading Perf and Eval Pipeline"
upload_with_priority .buildkite/pipeline_perf.yml "$PERF_PRIORITY"

echo "Uploading Integration Tests Pipeline"
upload_with_priority .buildkite/pipeline_integration.yml "$INTEGRATION_PRIORITY"

echo "Uploading Unit Tests Pipeline"
upload_with_priority .buildkite/pipeline_tests.yml "$JOB_PRIORITY"

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
