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

#
# Launch a single Docker container running a given `vllm serve` command for
# the `mp` multihost distributed-executor backend.
#
# Unlike run_cluster.sh (which starts a Ray daemon and joins a cluster),
# there is no cluster daemon here: each host's container just runs its own
# independent `vllm serve` process, and vllm-torchtpu's mp-multihost
# bootstrap (src/vllm_torchtpu/distributed/tpu_mp_multihost.py) handles
# cross-host rendezvous directly between those processes. This script is
# the shared "run this command in a TPU-privileged container" primitive
# used by both the head (foreground `vllm serve`) and each worker
# (foreground `vllm serve --headless ...`).
#
# Usage:
#   run_cluster_mp.sh <docker_image> <path_to_huggingface_cache> \
#       <serve_command> [-e KEY=VAL ...]
#
# The container is named "node" and runs in the foreground (blocks until
# `vllm serve` exits); callers background this script and `docker rm -f
# node` to tear it down.

if [ $# -lt 3 ]; then
    echo "Usage: $0 docker_image path_to_hf_home serve_command [additional_args...]"
    exit 1
fi

DOCKER_IMAGE="$1"
PATH_TO_HF_HOME="$2"
SERVE_CMD="$3"
shift 3

ADDITIONAL_ARGS=("$@")

CONTAINER_NAME="node"

cleanup() {
    docker stop "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# --privileged: Grants extended privileges to the container for TPU exposure
# --network host: allows the head/worker vllm serve processes to reach each
# other directly via host networking (data-parallel-address/rpc-port).
# --shm-size=128G: shared memory; the torch_tpu compile cache lives in
# /dev/shm/torch_tpu_cache.
# -v HF_HOME: mounts the HuggingFace cache (tokenizer/config only here;
# weights stream from GCS via runai_streamer).
docker run \
    --privileged \
    --entrypoint /bin/bash \
    --network host \
    --shm-size=128G \
    --name "${CONTAINER_NAME}" \
    -v "${PATH_TO_HF_HOME}:/root/.cache/huggingface" \
    "${ADDITIONAL_ARGS[@]}" \
    "${DOCKER_IMAGE}" -c "${SERVE_CMD}"
