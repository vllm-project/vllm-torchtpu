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

# Echoes the build's bare-metal job priority and its Kueue priority class (empty
# for none), in that order.
determine_job_priority() {
  local priority=""
  local workload_priority=""
  echo "--- Determining job priority" >&2
  if [[ "${NIGHTLY:-0}" == "1" ]]; then
    # Nightly build (Lowest priority). Unclassed on kube, level with the
    # tpu-inference nightlies it shares the TPU queues with, which record no
    # class either.
    priority="$PRIORITY_NIGHTLY"
    echo "Build type: Nightly - Priority: $priority" >&2
  elif [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
    local labels="${BUILDKITE_PULL_REQUEST_LABELS:-}"
    if grep -qx "oncall-fix" <<< "${labels//,/$'\n'}"; then
      # PR that fixes a CI or nightly breakage (Highest priority)
      priority="$PRIORITY_ONCALL_FIX"
      workload_priority="oncall-fix"
      echo "Build type: On-call fix (PR #${BUILDKITE_PULL_REQUEST}) - Priority: $priority" >&2
    else
      # Pre-merge PR tests
      priority="$PRIORITY_PRE_MERGE"
      workload_priority="pre-merge"
      echo "Build type: Pre-merge (PR #${BUILDKITE_PULL_REQUEST}) - Priority: $priority" >&2
    fi
  elif [[ "${BUILDKITE_BRANCH:-}" == "main" && "${BUILDKITE_PULL_REQUEST:-false}" == "false" ]]; then
    # Post-merge tests on main
    priority="$PRIORITY_POST_MERGE"
    workload_priority="post-merge"
    echo "Build type: Post-merge (Main branch) - Priority: $priority" >&2
  else
    # Default priority for other branches or manual builds
    priority="$PRIORITY_DEFAULT"
    echo "Build type: General - Priority: $priority" >&2
  fi

  echo "$priority $workload_priority"
}

job_priorities=$(determine_job_priority)
read -r JOB_PRIORITY WORKLOAD_PRIORITY <<< "$job_priorities"
export JOB_PRIORITY
buildkite-agent meta-data set "JOB_PRIORITY" "$JOB_PRIORITY"
# The kube launcher submits every workload of the build at this Kueue priority
# class (ci-infra launch.py). None leaves them unclassed, at 0.
if [[ -n "$WORKLOAD_PRIORITY" ]]; then
  buildkite-agent meta-data set "WORKLOAD_PRIORITY" "$WORKLOAD_PRIORITY"
fi
if [[ "$JOB_PRIORITY" == "$PRIORITY_ONCALL_FIX" ]]; then
  buildkite-agent annotate --style warning --context oncall-fix \
    "Runs at on-call priority (\`oncall-fix\` label): ahead of every other build, including main."
fi

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

  # Filter out files we want to skip builds for (docs, md, icons, CODEOWNERS, LICENSE,
  # .git-blame-ignore-revs, .pyrefly-baseline.json). The pre-commit GitHub Action
  # checks .pyrefly-baseline.json and no Buildkite step reads it.
  # .pyrefly-baseline-full.json stays out of this list: only the Buildkite type
  # check reads it.
  NON_SKIPPABLE_FILES=$(echo "${FILES_CHANGED}" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/|^\.github\/CODEOWNERS$|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)" || true)

  if [[ -z "${NON_SKIPPABLE_FILES}" && -n "${FILES_CHANGED}" ]]; then
    echo "Only documentation/non-code files changed. Skipping CI build."
    exit 0
  else
    echo "Code files changed. Proceeding with pipeline upload."
  fi
fi

# Decided before anything is uploaded, and exported so the pipeline files can
# interpolate ${CI_FLEET}; see ci_fleet.sh.
echo "--- :kubernetes: Choosing bare metal or kube"
# shellcheck source=/dev/null
source "${SCRIPT_DIR}/ci_fleet.sh"
resolve_ci_fleet
if [[ "${CI_FLEET}" == "kube" ]]; then
  buildkite-agent annotate --style info --context ci-fleet \
    "Runs on the kube fleet (${CI_FLEET_REASON})."
fi

echo "--- Starting Buildkite Bootstrap"

# Resolve and Publish Version Metadata
INTEGRATION_SLUG="${VLLM_INTEGRATION_PIPELINE_SLUG:-vllm-torchtpu-integration}"

if [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
  # Integration pipeline tests upstream HEAD
  VLLM_COMMIT_HASH=$(git ls-remote "https://github.com/vllm-project/vllm.git" HEAD | awk '{ print $1 }')
else
  # Standard builds test the pinned LKG version
  VLLM_COMMIT_HASH=$(get_vllm_commit_hash)
fi

# Support manual developer override via environment variable
VLLM_COMMIT_HASH="${VLLM_COMMIT_HASH_OVERRIDE:-$VLLM_COMMIT_HASH}"
buildkite-agent meta-data set "VLLM_COMMIT_HASH" "${VLLM_COMMIT_HASH}"
echo "Using vLLM commit: ${VLLM_COMMIT_HASH}"

# Since Buildkite prepends uploaded steps (inserts in reverse order),
# uploading the test steps first and build steps last ensures they
# are displayed and queued in correct order: Build -> Tests.

if [[ "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
  # Upload promote step FIRST so Buildkite places it LAST after test steps (Step 8)
  echo "Uploading Promotion Pipeline Fragment"
  upload_with_priority .buildkite/pipeline_promote.yml "$JOB_PRIORITY"
fi

# The suite's lanes, in upload order. Lane <lane> is
# .buildkite/pipeline_<lane>.yml on bare metal and pipeline_<lane>_kube.yml on
# kube; no other file is uploaded as a lane. CI_LANES picks a subset - a
# scheduled pipeline names its own lane. Without it a build runs every lane but
# perf, which has scheduled builds of its own; nightly builds still run perf,
# and so does the integration pipeline, whose LKG promotion waits on two perf
# steps.
ALL_LANES="verify perf integration tests"
default_lanes="verify integration tests"
if [[ "${NIGHTLY:-0}" == "1" || "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
  default_lanes="${ALL_LANES}"
fi
lanes="${CI_LANES:-${default_lanes}}"
for lane in ${lanes}; do
  if [[ " ${ALL_LANES} " != *" ${lane} "* ]]; then
    echo "ERROR: CI_LANES names '${lane}'; lanes are: ${ALL_LANES}" >&2
    exit 1
  fi
done
suffix=""
if [[ "${CI_FLEET}" == "kube" ]]; then
  suffix="_kube"
fi
missing=()
for lane in ${ALL_LANES}; do
  if [[ " ${lanes} " != *" ${lane} "* ]]; then
    continue
  fi
  file=".buildkite/pipeline_${lane}${suffix}.yml"
  if [[ ! -f "${file}" ]]; then
    missing+=("${file}")
    continue
  fi
  echo "Uploading ${file}"
  upload_with_priority "${file}" "$JOB_PRIORITY"
done
if [[ ${#missing[@]} -gt 0 ]]; then
  # A lane asked for by name has to exist; the whole suite on a target that
  # lacks a lane runs the rest and says what it left out.
  if [[ -n "${CI_LANES:-}" ]]; then
    echo "ERROR: CI_LANES asks for ${missing[*]}, which does not exist" >&2
    exit 1
  fi
  buildkite-agent annotate --style warning --context ci-fleet-gaps \
    "Not run on ${CI_FLEET}: no ${missing[*]} yet."
fi

# pipeline_build.yml skips the -omni image build when this is non-empty.
# Buildkite rejects the upload if a skip reason is over 70 characters.
OMNI_IMAGE_SKIP=""
if [[ " ${lanes} " != *" tests "* ]]; then
  OMNI_IMAGE_SKIP="Only the tests lane uses the -omni image; this build has none."
fi
export OMNI_IMAGE_SKIP

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
