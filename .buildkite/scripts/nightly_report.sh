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

# The report a nightly or release-tag build ends with, once its suites have
# run: the support matrix for each TPU generation, committed; a failure if any
# model or feature failed; and, on a main nightly where every model, feature
# and nightly tests-lane step passed, the vLLM / vllm-torchtpu commit pair
# recorded as verified.
#
# Only the scheduled nightly pushes: its schedule sets KUBE_OWNS_NIGHTLY=1, and
# a manual or validation build with NIGHTLY=1 leaves it unset so it never
# commits.
set -euo pipefail

SCRIPTS="$(dirname "${BASH_SOURCE[0]}")"

meta() {
  buildkite-agent meta-data get "$1" --default "$2"
}

# Every tests-lane step a nightly runs (every key but the (PR) ones') passed.
# A step the report did not wait for may have no outcome yet, which counts as
# not passed: a nightly test missing from the step's depends_on can stop a
# pair being recorded, never let one through.
tests_lane_passed() {
  local file=".buildkite/pipeline_tests.yml" key outcome passed=true
  while IFS= read -r key; do
    outcome=$(buildkite-agent step get outcome --step "${key}" 2>/dev/null || true)
    echo "tests lane ${key}: ${outcome:-no outcome}"
    if [[ "${outcome}" != "passed" ]]; then
      passed=false
    fi
  done < <(grep -oE '^[[:space:]]*key:[[:space:]]*"?[A-Za-z0-9_-]+' "${file}" | sed -E 's/.*key:[[:space:]]*"?//' | grep -v '_pr$')
  [[ "${passed}" == true ]]
}

if [[ "$(meta stop_support_matrix false)" == "true" ]]; then
  echo "--- 🛑 Uploading the models and features failed: no support matrices"
  exit 1
fi

for tpu in v6e v7x; do
  if [[ "$(meta "run_${tpu:0:2}_matrix" false)" == "true" ]]; then
    TPU_VERSION="${tpu}" bash "${SCRIPTS}/generate_support_matrices.sh"
  else
    echo "--- No ${tpu} test data: no ${tpu} matrix"
  fi
done

owns_nightly=true
if [[ "${KUBE_OWNS_NIGHTLY:-0}" != "1" ]]; then
  owns_nightly=false
  echo "--- Not the scheduled nightly (KUBE_OWNS_NIGHTLY unset): nothing is pushed"
fi
main_nightly=false
if [[ "${NIGHTLY:-0}" == "1" && "${BUILDKITE_BRANCH:-}" == "main" ]]; then
  main_nightly=true
fi
release_tag=false
if [[ "${BUILDKITE_TAG:-}" =~ ^v?[0-9]+(\.[a-zA-Z0-9]+)*([a-zA-Z]+[0-9]*)?$ ]]; then
  release_tag=true
fi

if [[ "${owns_nightly}" == true ]] && [[ "${main_nightly}" == true || "${release_tag}" == true ]]; then
  bash "${SCRIPTS}/commit_support_matrices.sh"
fi

matrix_passed=true
bash "${SCRIPTS}/notify_test_results.sh" || matrix_passed=false

if [[ "${owns_nightly}" == true && "${main_nightly}" == true && "${matrix_passed}" == true ]]; then
  echo "--- Checking the tests lane"
  if tests_lane_passed; then
    bash "${SCRIPTS}/commit_verified_commit_hashes.sh"
  else
    echo "Not every nightly tests-lane step passed: no verified commit pair"
  fi
fi

[[ "${matrix_passed}" == true ]]
