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

# Echoes the build's Buildkite job priority and its Kueue priority class (empty
# for none), in that order.
determine_job_priority() {
  local priority=""
  local workload_priority=""
  echo "--- Determining job priority" >&2
  if [[ "${NIGHTLY:-0}" == "1" || "${TORCHTPU_NIGHTLY:-0}" == "1" ]]; then
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
# One concurrency group per author for the TPU steps of their PR builds; see
# PR_TPU_JOBS_PER_AUTHOR. Keyed on the commit author's email, which every
# webhook build carries: the Buildkite creator is missing whenever the GitHub
# account is not linked to a Buildkite user.
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  pr_author="${BUILDKITE_BUILD_AUTHOR_EMAIL:-${BUILDKITE_BUILD_CREATOR_EMAIL:-}}"
  if [[ -n "$pr_author" ]]; then
    export PR_TPU_CONCURRENCY_GROUP="tpu-pr/${pr_author,,}"
  fi
fi

# --- Check for explicit skip-ci or non-code/documentation-only changes ---
echo "--- :git: Checking if CI build should be skipped"

if [[ "${BUILDKITE_MESSAGE:-}" =~ \[skip[[:space:]]ci\] || "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]skip\] ]]; then
  echo "Commit message contains [skip ci]. Skipping build."
  exit 0
fi

# True when every path in $1 is documentation or another file no step reads
# (docs, md, icons, CODEOWNERS, LICENSE, .git-blame-ignore-revs,
# .pyrefly-baseline.json). The pre-commit GitHub Action checks
# .pyrefly-baseline.json and no Buildkite step reads it.
# .pyrefly-baseline-full.json stays out of this list: only the Buildkite type
# check reads it.
only_docs_changed() {
  local files="$1"
  local non_skippable
  non_skippable=$(echo "${files}" | grep -vE "(\.md$|\.ico$|\.png$|^README$|^docs\/|^\.github\/CODEOWNERS$|^LICENSE$|^\.git-blame-ignore-revs$|^\.pyrefly-baseline\.json$)" || true)
  [[ -n "${files}" && -z "${non_skippable}" ]]
}

FILES_CHANGED=""
DIFF_BASE=""
DIFF_HEAD="${BUILDKITE_COMMIT:-HEAD}"
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  BASE_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-"main"}
  echo "PR detected. Target branch: ${BASE_BRANCH}"

  git fetch origin "${BASE_BRANCH}" --depth=20 --quiet || echo "Base fetch failed"
  git fetch origin "${DIFF_HEAD}" --depth=20 --quiet || true

  DIFF_BASE=$(git merge-base "origin/${BASE_BRANCH}" "${DIFF_HEAD}" 2>/dev/null || echo "origin/${BASE_BRANCH}")
  FILES_CHANGED=$(git diff --name-only origin/"${BASE_BRANCH}"..."${DIFF_HEAD}" 2>/dev/null || true)
  if [[ -z "${FILES_CHANGED}" ]]; then
    DIFF_BASE="${DIFF_HEAD}^"
    FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "${DIFF_HEAD}")
  fi
elif [[ "${BUILDKITE_SOURCE:-}" != "schedule" ]]; then
  # Any other build is of a commit, which on main is the squash merge of one PR,
  # so its diff against its parent is the whole change. Not a scheduled build:
  # the nightly tests main as it stands, whatever its last commit touched.
  DIFF_BASE="${DIFF_HEAD}^"
  git fetch origin "${DIFF_HEAD}" --depth=2 --quiet || true
  FILES_CHANGED=$(git diff --name-only "${DIFF_BASE}" "${DIFF_HEAD}" 2>/dev/null || true)
fi

INTEGRATION_SLUG="${VLLM_INTEGRATION_PIPELINE_SLUG:-vllm-torchtpu-integration}"

if [[ -n "${FILES_CHANGED}" ]]; then
  echo "Files changed:"
  echo "${FILES_CHANGED}"

  if [[ "${NIGHTLY:-0}" != "1" && "${TORCHTPU_NIGHTLY:-0}" != "1" \
      && "${BUILDKITE_PIPELINE_SLUG:-}" != "${INTEGRATION_SLUG}" \
      && -z "${CI_LANES:-}" && -z "${BUILDKITE_TAG:-}" ]] \
      && only_docs_changed "${FILES_CHANGED}"; then
    pr_lbls="${BUILDKITE_PULL_REQUEST_LABELS:-}"
    if ! grep -qxE "run-full-ci|run-omni-full-ci|run-perf-eval" <<< "${pr_lbls//,/$'\n'}"; then
      echo "Only documentation/non-code files changed. Skipping CI build."
      exit 0
    fi
  fi
  echo "Code files changed. Proceeding with pipeline upload."
fi

echo "--- Starting Buildkite Bootstrap"

# Resolve and Publish Version Metadata
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

# Echoes the labels of the PR a post-merge build on main merged, comma-separated
# like BUILDKITE_PULL_REQUEST_LABELS. They do not reach the build, so they are
# read from GitHub by the PR number GitHub puts at the end of a squash commit's
# subject, with a token from the agent's GitHub App helper when it has one. Any
# failure echoes nothing and the build runs everything.
merged_pr_labels() {
  local pr token
  local auth=()
  pr=$(head -n1 <<< "${BUILDKITE_MESSAGE:-}" | sed -nE 's/.*\(#([0-9]+)\)$/\1/p')
  [[ -n "${pr}" ]] || return 0
  token=$(/etc/buildkite-agent/git-credential-github-app get 2>/dev/null \
    | sed -n 's/^password=//p' || true)
  if [[ -n "${token}" ]]; then
    auth=(-H "Authorization: token ${token}")
  fi
  curl -fsS --max-time 10 ${auth[@]+"${auth[@]}"} \
    "https://api.github.com/repos/vllm-project/vllm-torchtpu/issues/${pr}/labels" 2>/dev/null \
    | python3 -c 'import json, sys; print(",".join(l["name"] for l in json.load(sys.stdin)))' \
      2>/dev/null || true
}

pr_labels="${BUILDKITE_PULL_REQUEST_LABELS:-}"
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  effective_labels="${pr_labels}"
elif [[ "${BUILDKITE_BRANCH:-}" == "main" && "${BUILDKITE_SOURCE:-}" == "webhook" ]]; then
  effective_labels="$(merged_pr_labels)"
else
  effective_labels=""
fi

if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]] \
    && grep -qx "run-full-ci" <<< "${pr_labels//,/$'\n'}"; then
  export NIGHTLY=1
  buildkite-agent annotate --style info --context full-ci \
    "Runs the full (nightly) suite (\`run-full-ci\` label)."
fi

# Resolve track selection (core-only, omni-only, both, or skip-all), default
# lanes, and Buildkite step skip reasons (all <= 70 chars) via select_ci_tracks.py.
ALL_LANES="verify perf integration tests"
track_plan="$(python3 "${SCRIPT_DIR}/select_ci_tracks.py" \
  --repo-root "${SCRIPT_DIR}/../.." \
  --base-ref "${DIFF_BASE}" \
  --head-ref "${DIFF_HEAD}" \
  --labels "${effective_labels}" \
  --format shell <<< "${FILES_CHANGED}")"
eval "${track_plan}"
export CI_TRACK_SELECTION CORE_IMAGE_SKIP CORE_TESTS_SKIP OMNI_IMAGE_SKIP OMNI_TESTS_SKIP OMNI_FULL_CI NIGHTLY SKIP_TPU_CI

if [[ "${CI_TRACK_SELECTION}" == "skip-all" ]]; then
  echo "Only documentation/non-code files changed. Skipping CI build."
  exit 0
fi

echo "Selected CI track: ${CI_TRACK_SELECTION} (default lanes: ${TRACK_DEFAULT_LANES})"

if [[ "${SKIP_TPU_CI}" == "1" ]]; then
  buildkite-agent annotate --style info --context skip-tpu-ci \
    "TPU steps not run (\`skip-tpu-ci\` label); the CPU steps still run."
else
  if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]] \
      && grep -qx "run-perf-eval" <<< "${pr_labels//,/$'\n'}" && [[ -z "${CI_LANES:-}" ]]; then
    buildkite-agent annotate --style info --context perf-eval \
      "Runs the perf/eval lane (\`run-perf-eval\` label)."
  fi
  if [[ -n "${TRACK_ANNOTATION}" ]]; then
    buildkite-agent annotate --style info --context ci-track-selection \
      "${TRACK_ANNOTATION}"
  fi
fi

lanes="${CI_LANES:-${TRACK_DEFAULT_LANES}}"
for lane in ${lanes}; do
  if [[ " ${ALL_LANES} " != *" ${lane} "* ]]; then
    echo "ERROR: CI_LANES names '${lane}'; lanes are: ${ALL_LANES}" >&2
    exit 1
  fi
done
missing=()
for lane in ${ALL_LANES}; do
  if [[ " ${lanes} " != *" ${lane} "* ]]; then
    continue
  fi
  file=".buildkite/pipeline_${lane}.yml"
  if [[ ! -f "${file}" ]]; then
    missing+=("${file}")
    continue
  fi
  echo "Uploading ${file}"
  upload_with_priority "${file}" "$JOB_PRIORITY"
done
if [[ ${#missing[@]} -gt 0 ]]; then
  # A lane asked for by name has to exist; the whole suite runs the lanes it
  # has and says what it left out.
  if [[ -n "${CI_LANES:-}" ]]; then
    echo "ERROR: CI_LANES asks for ${missing[*]}, which does not exist" >&2
    exit 1
  fi
  buildkite-agent annotate --style warning --context lane-gaps \
    "Not run: no ${missing[*]} yet."
fi

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
