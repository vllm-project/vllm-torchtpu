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
# A step on another variant of the image - the -omni build - names it with
# DOCKER_IMAGE_SUFFIX, as run_in_docker.sh reads it on bare metal. A step may
# also name its own image, which the disagg lane does - it builds one of its
# own rather than using the CI image.
if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  WORKLOAD_IMAGE=$(buildkite-agent meta-data get "${CI_IMAGE_TAG_KEY:-CI_IMAGE_TAG${DOCKER_IMAGE_SUFFIX:-}}" \
    --default "" 2>/dev/null || true)
fi
if [[ -z "${WORKLOAD_IMAGE:-}" ]]; then
  # Falling back to :latest would run whatever happened to be pushed last and
  # report the answer as this commit's.
  echo "$0: no CI_IMAGE_TAG${DOCKER_IMAGE_SUFFIX:-} metadata and no WORKLOAD_IMAGE." \
       "The image build step has to run before this one, or the step has to" \
       "name an image." >&2
  exit 2
fi
export WORKLOAD_IMAGE

# The image the Kimi-K3 AgentX pair's engines run (kimi-k3-agentx-1p1d.yaml
# substitutes it; no pod reads it from here): the step's, else the one the build
# published from build_k3_engine_image.sh, else the build's own.
if [[ -z "${K3_ENGINE_IMAGE:-}" ]]; then
  K3_ENGINE_IMAGE=$(buildkite-agent meta-data get K3_ENGINE_IMAGE --default "" 2>/dev/null || true)
fi
export K3_ENGINE_IMAGE="${K3_ENGINE_IMAGE:-$WORKLOAD_IMAGE}"

# Whether that pair's pods get the fleet's caches (kimi-k3-agentx-1p1d.yaml
# substitutes it; no pod reads it from here): the step's, else `none` for a
# megakernel point and empty, which keeps them, for anything else. A megakernel
# boot needs the host memory the gcsfuse file cache would hold; see the
# manifest. Tested for set rather than empty, so a step can keep the caches by
# setting it empty.
if [[ -z "${K3_FLEET_CACHES+set}" ]]; then
  case "${K3_PD_CONFIG:-}" in
    kimi-k3-megakernel*/*) K3_FLEET_CACHES=none ;;
    *) K3_FLEET_CACHES="" ;;
  esac
fi
export K3_FLEET_CACHES

# Anything exported here must also be named in FORWARD below: the launcher
# forwards only what --env names.
export FORCE_COLOR="1"
export TQDM_MININTERVAL="30"
export SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE="0.0.0"
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
  # cpu on the cpu queue; unset on a TPU host, where JAX finds the chips.
  JAX_PLATFORMS
  FORCE_COLOR TQDM_MININTERVAL SETUPTOOLS_SCM_PRETEND_VERSION_FOR_VLLM_TORCHTPU_CORE
  UV_INDEX_TORCH_TPU_REGISTRY_USERNAME UV_NO_CACHE
  CI_RUNNER SKIP_SPANNER_UPLOAD
  BENCHMARK_WARMUP_RUNS RUN_CODE_EVAL RUN_LM_EVAL RUN_TYPE MODEL_PATH MODEL_URI
  EVAL_SOFT_FAIL
  # Lets a one-off build force an xprof capture; unset leaves the day rule in
  # benchmark_with_profile.sh to decide.
  CAPTURE_PROFILE
  # run_in_docker.sh turns CONTAINER_TMPDIR into these two. A DSv4 compile
  # writes ~40 GB of inductor cache per host and a capture run ~30 GB of temp
  # JSON, which is an evicted pod if it lands on ephemeral storage.
  TMPDIR RAY_TMPDIR
  # vLLM's compile caches default to ~/.cache/vllm, on the pod's ephemeral
  # storage. A step that compiles a large model points them at /cache/jax.
  VLLM_CACHE_ROOT VLLM_XLA_CACHE_PATH
  SERVED_MODEL_NAME RUN_ROOT VLLM_ENGINE_READY_TIMEOUT_S
  TPU_ACCELERATOR_TYPE TPU_MULTIHOST_BACKEND TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL
  USE_MOE_SPARSE_CORE P4D2_BIND_HOST PROXY_PORT
  # Set by bootstrap.sh; benchmark configs branch on it.
  NIGHTLY CREATED_BY
  # The P/D benchmark loads its results into BigQuery when this is set.
  BQ_UPLOAD
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
  # The Kimi-K3 AgentX pair's job switches, read by k3_agentx_client_pod.sh
  # and the megakernel recipe: the eval-only job, the eval smoke, and skipping
  # the post-replay eval.
  AGENTX_EVAL_JOB AGENTX_EVAL_SMOKE AGENTX_SKIP_POST_EVAL
  # The config both engines of the pair and its client read, when a step serves
  # another than the manifest's default (a Kimi-K3 megakernel recipe point).
  K3_PD_CONFIG
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
