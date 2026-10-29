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

# Buildkite sets BUILDKITE_MERGE_QUEUE_BASE_BRANCH on native merge-queue builds.
MERGE_QUEUE_BASE="${BUILDKITE_MERGE_QUEUE_BASE_BRANCH:-}"

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
  elif [[ -n "${MERGE_QUEUE_BASE}" ]]; then
    priority="$PRIORITY_POST_MERGE"
    workload_priority="post-merge"
    echo "Build type: Merge queue (${MERGE_QUEUE_BASE}) - Priority: $priority" >&2
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
  elif [[ "${CI_PRECHECK:-0}" == "1" || "${CI_RUN_TPU:-0}" == "1" ]]; then
    priority="$PRIORITY_PRE_MERGE"
    workload_priority="pre-merge"
    echo "Build type: Pre-merge (/ci) - Priority: $priority" >&2
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

# Merge-queue builds and explicit `/ci precheck` / `/ci run` builds ignore `[skip ci]`.
if [[ -z "${MERGE_QUEUE_BASE}" && "${CI_PRECHECK:-0}" != "1" && "${CI_RUN_TPU:-0}" != "1" ]] \
    && [[ "${BUILDKITE_MESSAGE:-}" =~ \[skip[[:space:]]ci\] || "${BUILDKITE_MESSAGE:-}" =~ \[ci[[:space:]]skip\] ]]; then
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
if [[ "${CI_PRECHECK:-0}" == "1" || "${CI_RUN_TPU:-0}" == "1" ]]; then
  :
elif [[ -n "${MERGE_QUEUE_BASE}" ]]; then
  if [[ -n "${BUILDKITE_MERGE_QUEUE_BASE_COMMIT:-}" ]]; then
    mq_diff_base="${BUILDKITE_MERGE_QUEUE_BASE_COMMIT}"
    git fetch origin "${mq_diff_base}" --depth=20 --quiet || true
  else
    mq_diff_base="origin/${MERGE_QUEUE_BASE}"
    git fetch origin "+refs/heads/${MERGE_QUEUE_BASE}:refs/remotes/origin/${MERGE_QUEUE_BASE}" --depth=20 --quiet \
      || git fetch origin "${MERGE_QUEUE_BASE}" --depth=20 --quiet \
      || echo "Base fetch failed"
  fi
  git fetch origin "${BUILDKITE_COMMIT:-HEAD}" --depth=20 --quiet || true
  FILES_CHANGED=$(git diff --name-only "${mq_diff_base}"..."${BUILDKITE_COMMIT:-HEAD}" 2>/dev/null \
    || git diff --name-only "${mq_diff_base}" "${BUILDKITE_COMMIT:-HEAD}" 2>/dev/null || true)
elif [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  BASE_BRANCH=${BUILDKITE_PULL_REQUEST_BASE_BRANCH:-"main"}
  echo "PR detected. Target branch: ${BASE_BRANCH}"

  git fetch origin "${BASE_BRANCH}" --depth=20 --quiet || echo "Base fetch failed"
  git fetch origin "${BUILDKITE_COMMIT:-HEAD}" --depth=20 --quiet || true

  FILES_CHANGED=$(git diff --name-only origin/"${BASE_BRANCH}"..."${BUILDKITE_COMMIT:-HEAD}" 2>/dev/null || true)
  if [[ -z "${FILES_CHANGED}" ]]; then
    FILES_CHANGED=$(git diff-tree --no-commit-id --name-only -r -m "${BUILDKITE_COMMIT:-HEAD}")
  fi
elif [[ "${BUILDKITE_SOURCE:-}" != "schedule" ]]; then
  # Any other build is of a commit, which on main is the squash merge of one PR,
  # so its diff against its parent is the whole change. Not a scheduled build:
  # the nightly tests main as it stands, whatever its last commit touched.
  git fetch origin "${BUILDKITE_COMMIT:-HEAD}" --depth=2 --quiet || true
  FILES_CHANGED=$(git diff --name-only "${BUILDKITE_COMMIT:-HEAD}^" "${BUILDKITE_COMMIT:-HEAD}" 2>/dev/null || true)
fi

if [[ -n "${FILES_CHANGED}" ]]; then
  echo "Files changed:"
  echo "${FILES_CHANGED}"

  if only_docs_changed "${FILES_CHANGED}"; then
    if [[ -n "${MERGE_QUEUE_BASE}" ]]; then
      buildkite-agent annotate --style info --context docs-only \
        "Docs-only change; skipped CPU and TPU checks in the merge queue."
    fi
    echo "Only documentation/non-code files changed. Skipping CI build."
    exit 0
  else
    echo "Code files changed. Proceeding with pipeline upload."
  fi
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
# .buildkite/pipeline_<lane>.yml; no other file is uploaded as a lane. CI_LANES
# picks a subset - a scheduled pipeline names its own lane. Without it a build runs every lane but
# perf, which has scheduled builds of its own; nightly builds still run perf,
# and so does the integration pipeline, whose LKG promotion waits on two perf
# steps. A PR whose change needs perf/eval coverage asks for it with the
# run-perf-eval label, or the full nightly suite across every lane with
# run-full-ci; adding either label to a ready PR starts a build.
ALL_LANES="verify perf integration tests"
default_lanes="verify integration tests"
pr_labels="${BUILDKITE_PULL_REQUEST_LABELS:-}"

# Echoes `<extras>|<actor>` from the latest bot `/ci merge` marker on PR $1
# (both empty when unset or cleared). On failure, echoes the error and returns 1.
merge_queue_pr_extras() {
  local pr="$1" token page=1 resp parsed count found val actor latest="" latest_actor=""
  local auth=()
  token=$(/etc/buildkite-agent/git-credential-github-app get 2>/dev/null \
    | sed -n 's/^password=//p' || true)
  if [[ -n "${token}" ]]; then
    auth=(-H "Authorization: token ${token}")
  fi
  while true; do
    if ! resp=$(curl -fsS --max-time 10 ${auth[@]+"${auth[@]}"} \
      "https://api.github.com/repos/vllm-project/vllm-torchtpu/issues/${pr}/comments?per_page=100&page=${page}" 2>&1); then
      if [[ ${#auth[@]} -gt 0 ]] && resp=$(curl -fsS --max-time 10 \
        "https://api.github.com/repos/vllm-project/vllm-torchtpu/issues/${pr}/comments?per_page=100&page=${page}" 2>&1); then
        auth=()
      else
        echo "${resp:-HTTP request failed}" | tr '\r\n' '  ' | sed 's/  */ /g; s/^ //; s/ $//'
        return 1
      fi
    fi
    if ! parsed=$(python3 -c '
import json, re, sys
MARKER = re.compile(r"<!--\s*ci-merge-extras:\s*([^\n>]*?)\s*-->")
ACTOR = re.compile(r"<!--\s*ci-merge-actor:\s*([A-Za-z0-9_.\[\]-]+)\s*-->")
VALID = {"", "perf-eval", "nightly-suite", "precheck-only"}
items = json.load(sys.stdin)
if not isinstance(items, list):
    raise ValueError("unexpected GitHub comments payload")
found, val, actor = 0, "", ""
for c in items:
    if (c.get("user") or {}).get("type") != "Bot":
        continue
    body = c.get("body") or ""
    actor_m = ACTOR.search(body)
    c_actor = actor_m.group(1) if actor_m else ""
    for m in MARKER.finditer(body):
        parts = [p.strip() for p in m.group(1).split(",") if p.strip()]
        bad = [p for p in parts if p not in VALID]
        if bad:
            raise ValueError(f"unknown extra {bad[0]!r}")
        if "precheck-only" in parts and len(parts) > 1:
            raise ValueError("precheck-only cannot be combined with other extras")
        found, val = 1, ",".join(parts)
        actor = c_actor if val else ""
print(f"{len(items)}|{found}|{val}|{actor}")
' <<< "${resp}" 2>&1); then
      echo "${parsed:-invalid JSON}" | tail -n1 | tr '\r\n' '  ' | sed 's/  */ /g; s/^ //; s/ $//'
      return 1
    fi
    IFS='|' read -r count found val actor <<< "${parsed}"
    if [[ "${found}" == "1" ]]; then
      latest="${val}"
      latest_actor="${actor}"
    fi
    if (( count < 100 )); then
      break
    fi
    page=$((page + 1))
  done
  echo "${latest}|${latest_actor}"
}

EXTRA_PERF_EVAL=0
EXTRA_NIGHTLY_SUITE=0
MQ_PRECHECK_ONLY=0
MQ_PRECHECK_ACTOR=""
if [[ "${CI_PRECHECK:-0}" != "1" ]]; then
  if [[ "${MERGE_QUEUE_BASE}" == releases/* ]]; then
    EXTRA_NIGHTLY_SUITE=1
  fi
  if [[ -n "${MERGE_QUEUE_BASE}" ]]; then
    mq_pr=""
    if [[ "${BUILDKITE_PULL_REQUEST:-}" =~ ^[0-9]+$ ]]; then
      mq_pr="${BUILDKITE_PULL_REQUEST}"
    elif [[ "${BUILDKITE_BRANCH:-}" =~ ^gh-readonly-queue/.+/pr-([0-9]+)-[^/]+$ ]]; then
      mq_pr="${BASH_REMATCH[1]}"
    fi
    if [[ -n "${mq_pr}" ]]; then
      if mq_marker=$(merge_queue_pr_extras "${mq_pr}"); then
        IFS='|' read -r mq_extras mq_actor <<< "${mq_marker}"
        if [[ "${mq_extras}" == "precheck-only" ]]; then
          if [[ "${MERGE_QUEUE_BASE}" != releases/* ]]; then
            MQ_PRECHECK_ONLY=1
            MQ_PRECHECK_ACTOR="${mq_actor}"
          fi
        elif [[ -n "${mq_extras}" ]]; then
          CI_EXTRAS="${CI_EXTRAS:+${CI_EXTRAS},}${mq_extras}"
        fi
      else
        buildkite-agent annotate --style warning --context ci-merge-extras \
          "could not read /ci merge extras for PR #${mq_pr}: ${mq_marker:-request failed}; ran the default gate"
      fi
    fi
  fi
  if [[ "${MQ_PRECHECK_ONLY}" != "1" ]]; then
    if [[ -n "${CI_EXTRAS:-}" ]]; then
      IFS=',' read -ra ci_extras <<< "${CI_EXTRAS}"
      for extra in "${ci_extras[@]}"; do
        case "${extra}" in
          "") ;;
          perf-eval) EXTRA_PERF_EVAL=1 ;;
          nightly-suite) EXTRA_NIGHTLY_SUITE=1 ;;
          *)
            echo "ERROR: CI_EXTRAS names '${extra}'; valid extras are: perf-eval, nightly-suite" >&2
            exit 1
            ;;
        esac
      done
    fi

    if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]] \
        && grep -qx "run-full-ci" <<< "${pr_labels//,/$'\n'}"; then
      export NIGHTLY=1
      buildkite-agent annotate --style info --context full-ci \
        "Runs the full (nightly) suite (\`run-full-ci\` label)."
    elif [[ "${EXTRA_NIGHTLY_SUITE}" == "1" ]]; then
      export NIGHTLY=1
      buildkite-agent annotate --style info --context full-ci \
        "Runs the full (nightly) suite (\`nightly-suite\`)."
    fi
    if [[ "${NIGHTLY:-0}" == "1" || "${BUILDKITE_PIPELINE_SLUG:-}" == "${INTEGRATION_SLUG}" ]]; then
      default_lanes="${ALL_LANES}"
    elif [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]] \
        && grep -qx "run-perf-eval" <<< "${pr_labels//,/$'\n'}"; then
      default_lanes="${ALL_LANES}"
      if [[ -z "${CI_LANES:-}" ]]; then
        buildkite-agent annotate --style info --context perf-eval \
          "Runs the perf/eval lane (\`run-perf-eval\` label)."
      fi
    elif [[ "${EXTRA_PERF_EVAL}" == "1" ]]; then
      default_lanes="${ALL_LANES}"
      if [[ -z "${CI_LANES:-}" ]]; then
        buildkite-agent annotate --style info --context perf-eval \
          "Runs the perf/eval lane (\`perf-eval\`)."
      fi
    fi
  fi
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

# skip-tpu-ci keeps only the tests lane, and SKIP_TPU_CI leaves out its TPU group
# (pipeline_tests.yml reads it when the lane is uploaded), so the image build,
# the type check and the CPU unit tests still run, on the PR and again on its
# post-merge build. It overrides run-perf-eval and run-full-ci.
if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]]; then
  skip_labels="${pr_labels}"
elif [[ "${BUILDKITE_BRANCH:-}" == "main" && "${BUILDKITE_SOURCE:-}" == "webhook" ]]; then
  skip_labels="$(merged_pr_labels)"
else
  skip_labels=""
fi
if grep -qx "skip-tpu-ci" <<< "${skip_labels//,/$'\n'}"; then
  SKIP_TPU_CI=1
fi

# Decides whether this build uploads TPU steps (0 = yes, 1 = CPU steps only).
# Merge-queue builds run the full TPU gate unless `/ci merge precheck-only` was
# set (outside `releases/*`). On PR builds, setting pipeline env
# CI_PR_TPU_OPT_IN=1 keeps pushes CPU-only unless CI_RUN_TPU=1.
should_upload_tpu_steps() {
  if [[ "${CI_PRECHECK:-0}" == "1" || "${MQ_PRECHECK_ONLY:-0}" == "1" ]]; then
    return 1
  fi
  if [[ -n "${MERGE_QUEUE_BASE}" ]]; then
    return 0
  fi
  if [[ "${SKIP_TPU_CI:-0}" == "1" ]]; then
    return 1
  fi
  if [[ "${BUILDKITE_PULL_REQUEST:-false}" != "false" && -n "${BUILDKITE_PULL_REQUEST:-}" ]] \
      && [[ "${CI_PR_TPU_OPT_IN:-0}" == "1" && "${CI_RUN_TPU:-0}" != "1" ]]; then
    return 1
  fi
  return 0
}

if ! should_upload_tpu_steps; then
  skip_was_label="${SKIP_TPU_CI:-0}"
  SKIP_TPU_CI=1
  export SKIP_TPU_CI
  default_lanes="tests"
  if [[ "${CI_PRECHECK:-0}" == "1" || "${MQ_PRECHECK_ONLY:-0}" == "1" ]]; then
    if [[ "${MQ_PRECHECK_ONLY:-0}" == "1" ]]; then
      by_actor="${MQ_PRECHECK_ACTOR:+ by @${MQ_PRECHECK_ACTOR}}"
      buildkite-agent annotate --style info --context ci-merge-precheck-only \
        "TPU skipped: \`/ci merge precheck-only\`${by_actor}"
    else
      buildkite-agent annotate --style info --context ci-precheck \
        "TPU steps not run (\`/ci precheck\`); the CPU steps still run."
    fi
    if [[ -f .buildkite/pipeline_cpu.yml ]]; then
      export OMNI_IMAGE_SKIP="Only TPU steps use the -omni image; precheck skips them."
      echo "Uploading .buildkite/pipeline_cpu.yml"
      upload_with_priority .buildkite/pipeline_cpu.yml "$JOB_PRIORITY"
      echo "Uploading Build Pipeline"
      upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"
      echo "--- Buildkite Bootstrap Finished"
      exit 0
    fi
  elif [[ "${skip_was_label}" == "1" ]]; then
    buildkite-agent annotate --style info --context skip-tpu-ci \
      "TPU steps not run (\`skip-tpu-ci\` label); the CPU steps still run."
  else
    buildkite-agent annotate --style info --context pr-tpu-opt-in \
      "TPU steps opt-in on PR pushes; comment \`/ci run\` to run TPU tests."
  fi
else
  export SKIP_TPU_CI=0
fi
lanes="${CI_LANES:-${default_lanes}}"
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

# pipeline_build.yml skips the -omni image build when this is non-empty.
# Buildkite rejects the upload if a skip reason is over 70 characters.
OMNI_IMAGE_SKIP=""
if [[ " ${lanes} " != *" tests "* ]]; then
  OMNI_IMAGE_SKIP="Only the tests lane uses the -omni image; this build has none."
elif [[ "${SKIP_TPU_CI}" == "1" ]]; then
  OMNI_IMAGE_SKIP="Only TPU steps use the -omni image; skip-tpu-ci skips them."
fi
export OMNI_IMAGE_SKIP

echo "Uploading Build Pipeline"
upload_with_priority .buildkite/pipeline_build.yml "$JOB_PRIORITY"

echo "--- Buildkite Bootstrap Finished"
