#!/usr/bin/env bash
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

# Sourced by run.sh and run_multihost.sh; leaves WORKLOAD_IMAGE exported and
# the launcher's --env flags in `env_args`.

# Must name the tag setup_docker_env.sh pushes, so ask it rather than rebuild
# the name: it publishes what it pushed as CI_IMAGE_TAG, which is what every
# bare-metal consumer already reads. Composing the tag here means this file has
# to track every change to the scheme, and it did not - the tag carries the vLLM
# commit as well as ours, for registry isolation, so a bare commit SHA names an
# image that was never pushed. Nothing catches that: the build is green, and it
# surfaces twenty minutes later as ImagePullBackOff on a TPU node that scaled up
# to sit idle.
#
# A step may still name its own image, which the disagg lane does - it builds
# one of its own rather than using the CI image.
if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  WORKLOAD_IMAGE=$(buildkite-agent meta-data get "CI_IMAGE_TAG" --default "" \
    2>/dev/null || true)
fi
if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  # Falling back to :latest would run whatever happened to be pushed last and
  # report the answer as this commit's.
  echo "$0: no CI_IMAGE_TAG metadata and no WORKLOAD_IMAGE. The image build" \
       "step has to run before this one, or the step has to name an image." >&2
  exit 2
fi
export WORKLOAD_IMAGE

# Anything exported here must also be named in FORWARD below: the launcher
# forwards only what --env names.
export FORCE_COLOR="1"
export TQDM_MININTERVAL="30"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU="0.0.0"
export UV_INDEX_TORCH_TPU_REGISTRY_USERNAME="oauth2accesstoken"
export UV_NO_CACHE="1"

# Which fleet a benchmark row came from: two fleets write the same BigQuery
# table.
export CI_RUNNER="kube"

# Nothing reads a Spanner row for this fleet.
export SKIP_SPANNER_UPLOAD="1"

# Names the launcher cannot guess. A name the step has not set is skipped
# rather than injected empty, which several of these depend on.
FORWARD=(
  FORCE_COLOR TQDM_MININTERVAL SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU
  UV_INDEX_TORCH_TPU_REGISTRY_USERNAME UV_NO_CACHE
  CI_RUNNER SKIP_SPANNER_UPLOAD
  BENCHMARK_WARMUP_RUNS RUN_CODE_EVAL RUN_TYPE MODEL_PATH MODEL_URI
  SERVED_MODEL_NAME RUN_ROOT VLLM_ENGINE_READY_TIMEOUT_S
  TPU_ACCELERATOR_TYPE TPU_MULTIHOST_BACKEND TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
  USE_MOE_SPARSE_CORE P4D2_BIND_HOST PROXY_PORT
  # Set by bootstrap.sh; benchmark configs branch on it.
  NIGHTLY CREATED_BY
  # Carries the -m filter into bktec's collection step; see the sharded steps
  # in pipeline_tests_kube.yml.
  PYTEST_ADDOPTS
  # Unset on purpose: naming an unset variable is how the launcher is asked to
  # supply it from its env_secrets registry. HF_TOKEN is not here - every pod
  # inherits that one from the fleet pod defaults.
  BUILDKITE_ANALYTICS_TOKEN
  # What a slice's pods are asked to run; read by multihost_entry.sh.
  MULTIHOST_ARGS_B64
  # Names the directory whose contents the pod uploads as artifacts.
  ARTIFACTS_DIR
)

# The `tests` plugin mints these on the agent and their two-hour life starts
# before admission, node scale-up and image pull. Unset rather than
# deny-listed: the launcher prefers its own environment, so leaving them set
# would forward the stale token instead of reaching Secret Manager.
unset BUILDKITE_ANALYTICS_TOKEN BUILDKITE_TEST_ENGINE_API_ACCESS_TOKEN

# Lets bktec mint its own token in the pod: `oidc request-token` otherwise
# refuses without the agent's log redactor, which lives behind the denied Job
# API socket.
export BUILDKITE_AGENT_OIDC_REQUEST_TOKEN_SKIP_TOKEN_REDACTION=true

# The BUILDKITE_* this shell holds are swept by the launcher, the export above
# included. Only what it cannot see - a name left unset for Secret Manager, or
# one that is not BUILDKITE_* at all - has to be named here.
env_args=()
for _name in "${FORWARD[@]}"; do
  env_args+=(--env "$_name")
done
unset _name
