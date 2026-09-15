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

# Join path patterns into one alternation anchored at the start of a path.
path_regex() {
  local IFS='|'
  echo "^($*)"
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

# Suites that only exercise one subsystem are gated on the paths that
# subsystem lives in. A build with no pull request has no diff to gate on and
# runs everything.
RUN_SPEC_DECODE_TESTS=true
RUN_INTEGRATION_TESTS=true
RUN_PERF_EVAL=true

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

  # --- Decide which optional suites this diff needs ---
  echo "--- :mag: Selecting suites for this diff"

  # Paths a suite is sensitive to, one per entry, joined by path_regex into a
  # single anchored alternation for grep -E. Each entry is matched against the
  # start of a path, so a trailing / means "this directory" and a trailing $
  # means "this file exactly".

  # Speculative decoding gets its own PR steps on the v7x-2 and v7x-8 queues,
  # and they are the largest single consumer of both. Run them for changes to
  # the spec-decode module, the runner that drives it, the sampler it depends
  # on, its own tests, or the CI wiring that executes it. Changes elsewhere
  # reach the full matrix through the unconditional nightly suite.
  SPEC_DECODE_PATHS=(
    'src/vllm_torchtpu/spec_decode/'
    'src/vllm_torchtpu/runner/'
    'src/vllm_torchtpu/layers/vllm/sample/'
    'tests/spec_decode/'
    'tests/entrypoints/llm/test_speculative_decoding\.py$'
    '\.buildkite/pipeline_tests\.yml$'
    '\.buildkite/scripts/run_in_docker\.sh$'
  )

  # The integration and perf/eval suites both stand up a server and drive it
  # end to end, one for correctness and one for a number. Neither can be moved
  # by a diff that changes only unit tests, docs or CI metadata.
  NON_RUNTIME_PATHS=(
    'tests/'
    'docs/'
    '\.github/'
    '\.buildkite/'
    '.*\.md$'
  )

  # The files that define or launch each suite are the exception to that rule:
  # a diff touching one of them has to run the suite it changes.
  INTEGRATION_INPUT_PATHS=(
    '\.buildkite/pipeline_integration\.yml$'
    '\.buildkite/scripts/run_in_docker\.sh$'
    '\.buildkite/scripts/run_[a-z_]*multihost[a-z_]*\.sh$'
    'scripts/vllm/integration/'
    'examples/disagg/'
  )

  # scripts/vllm/benchmarking/ holds both the eval runners and the recorded
  # baselines the regression check compares against, so a change to either
  # side of that comparison runs the suite.
  PERF_INPUT_PATHS=(
    '\.buildkite/pipeline_perf\.yml$'
    '\.buildkite/scripts/run_in_docker\.sh$'
    '\.buildkite/scripts/run_[a-z_]*multihost[a-z_]*\.sh$'
    '\.buildkite/scripts/stage_model_snapshot\.sh$'
    'scripts/vllm/benchmarking/'
  )

  # The gates need a file list to reason about, and [ci full] in the commit
  # message overrides them for a change whose blast radius the paths do not
  # describe. Either way the build runs every suite.
  APPLY_SUITE_GATES=true
  if [[ -z "${FILES_CHANGED}" ]]; then
    APPLY_SUITE_GATES=false
    echo "No file list to gate on. Running every suite."
  elif [[ "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]full\] ]]; then
    APPLY_SUITE_GATES=false
    echo "Commit message contains [ci full]. Running every suite."
  fi

  if [[ "${APPLY_SUITE_GATES}" == "true" ]] \
    && ! grep -qE "$(path_regex "${SPEC_DECODE_PATHS[@]}")" <<< "${FILES_CHANGED}"; then
    RUN_SPEC_DECODE_TESTS=false
    echo "No speculative decoding paths changed. Skipping the spec-decode PR steps."
  fi

  if [[ "${APPLY_SUITE_GATES}" == "true" ]] \
    && ! grep -qvE "$(path_regex "${NON_RUNTIME_PATHS[@]}")" <<< "${FILES_CHANGED}"; then
    if ! grep -qE "$(path_regex "${INTEGRATION_INPUT_PATHS[@]}")" <<< "${FILES_CHANGED}"; then
      RUN_INTEGRATION_TESTS=false
      echo "Only tests, docs and CI metadata changed. Skipping the integration suites."
    fi
    if ! grep -qE "$(path_regex "${PERF_INPUT_PATHS[@]}")" <<< "${FILES_CHANGED}"; then
      RUN_PERF_EVAL=false
      echo "Only tests, docs and CI metadata changed. Skipping the perf/eval suites."
    fi
  fi

  # --- Require the 'ready' label before spending agent time on a PR ---
  echo "--- :label: Checking for the 'ready' label on PR #${BUILDKITE_PULL_REQUEST}"

  # Buildkite only populates this on pull_request webhook events. PR builds
  # triggered by a push arrive as GitHub 'push' events, so fall back to asking
  # the GitHub API. The repo is private, so that call has to be authenticated.
  PR_LABELS="${BUILDKITE_PULL_REQUEST_LABELS:-}"

  if [[ -z "${PR_LABELS}" ]]; then
    # Reuse the GitHub App credential helper the agents already use to clone this
    # repo (configured system-wide by ci-infra), so there is no extra secret,
    # IAM grant or token rotation to own here.
    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
      GITHUB_TOKEN=$(GIT_TERMINAL_PROMPT=0 git credential fill \
        <<< $'protocol=https\nhost=github.com\n' 2>/dev/null \
        | sed -n 's/^password=//p' || true)
    fi

    if [[ -z "${GITHUB_TOKEN:-}" ]]; then
      echo "Could not mint a GitHub token from the agent's git credential helper."
      echo "Cannot verify labels. Failing closed."
      exit 1
    fi

    # Give GitHub a moment to make a just-applied label readable.
    sleep 5

    PR_API_URL="https://api.github.com/repos/vllm-project/vllm-torchtpu/pulls/${BUILDKITE_PULL_REQUEST}"
    HTTP_CODE=$(curl -sS -o pr_details.json -w '%{http_code}' \
      -H "Authorization: Bearer ${GITHUB_TOKEN}" \
      -H "Accept: application/vnd.github+json" \
      "${PR_API_URL}" || echo "000")

    if [[ "${HTTP_CODE}" != "200" ]]; then
      echo "GitHub API returned HTTP ${HTTP_CODE} for ${PR_API_URL}."
      echo "Cannot verify labels. Failing closed."
      exit 1
    fi

    PR_LABELS=$(jq -r '[.labels[].name] | join(",")' < pr_details.json)
    rm -f pr_details.json
  fi

  echo "PR labels: ${PR_LABELS:-<none>}"

  # Match the whole label, so 'not-ready' or 'ready-for-review' don't count.
  if ! grep -qx "ready" <<< "${PR_LABELS//,/$'\n'}"; then
    echo "Missing 'ready' label on PR #${BUILDKITE_PULL_REQUEST}. Failing build."
    echo "Add the 'ready' label to the PR when it should run full CI."
    exit 1
  fi

  echo "Found 'ready' label. Proceeding with pipeline upload."
fi

echo "--- Starting Buildkite Bootstrap"

# Since Buildkite prepends uploaded steps (inserts in reverse order),
# uploading the test steps first and build steps last ensures they
# are displayed and queued in correct order: Build -> Tests.

if [[ "${RUN_PERF_EVAL}" == "true" ]]; then
  echo "Uploading Perf and Eval Pipeline"
  upload_with_priority .buildkite/pipeline_perf.yml "$JOB_PRIORITY"
fi

if [[ "${RUN_INTEGRATION_TESTS}" == "true" ]]; then
  echo "Uploading Integration Tests Pipeline"
  upload_with_priority .buildkite/pipeline_integration.yml "$JOB_PRIORITY"
fi

# Read by the spec-decode PR steps' `if` expressions, interpolated at upload.
export RUN_SPEC_DECODE_TESTS

echo "Uploading Unit Tests Pipeline"
upload_with_priority .buildkite/pipeline_tests.yml "$JOB_PRIORITY"

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
