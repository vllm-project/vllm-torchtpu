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

# What every pod of a multi-host workload shares, sourced by multihost_entry.sh
# (Ray), mp_multihost_entry.sh and k3_1p1d.sh (vLLM's own multi-node DP). Each
# host runs one of those with the same arguments and decides its own role from
# JOB_COMPLETION_INDEX; how the hosts find each other is theirs, and the rest
# is here.

# pod_setup: what a pod needs before it starts anything.
pod_setup() {
  REPO_DIR="${REPO_DIR:-/workspace/vllm-torchtpu}"
  cd "$REPO_DIR" || { echo "$0: no such directory: $REPO_DIR" >&2; exit 2; }
  # Core dumps are gigabytes each and fill the node's ephemeral storage.
  ulimit -c 0
  # In a pod the metadata server describes the node, not the workload.
  export TPU_SKIP_MDS_QUERY=1
  # Everything under ARTIFACTS_DIR is uploaded by publish_results. The step
  # names it; unset means this run keeps nothing.
  if [ -n "${ARTIFACTS_DIR:-}" ]; then
    mkdir -p "$ARTIFACTS_DIR"
  fi
}

# use_mp_backend: leave the slice to src/vllm_torchtpu/distributed/tpu_mp_multihost.py,
# which builds it from a TCPStore rendezvous among the DP ranks and publishes
# one TORCH_TPU_SLICEBUILDER_ADDRESSES entry per chip. GKE hands every TPU pod
# the environment of a slice member; left set, the injected coordinator is a
# second, contrary answer to the same question. Unlike single_host_engine.sh
# this pins no topology: the run really does span hosts, and the backend
# computes TORCH_TPU_TOPOLOGY from the chip count.
use_mp_backend() {
  export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT=''
  export MEGASCALE_SLICE_ID='' TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT=''
  export TPU_WORKER_HOSTNAMES=''
  export TPU_MULTIHOST_BACKEND=mp
}

# resolve <name> <seconds>: the name's IPv4 address, waiting up to <seconds>
# for it. DNS for a headless service can lag the pod.
resolve() {
  local deadline=$((SECONDS + $2)) ip
  until ip=$(getent ahostsv4 "$1" | awk 'NR == 1 { print $1 }') && [ -n "$ip" ]; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "ERROR: $1 did not resolve within $(($2 / 60)) minutes" >&2
      return 1
    fi
    sleep 5
  done
  echo "$ip"
}

# head_listening <host> <port>: is something listening there?
#
# Bounded, because the probe has to be able to fail and /dev/tcp has no
# deadline of its own. A completed pod stays a ready endpoint of the headless
# service, so its name keeps resolving after the head exits - to an address
# that no longer answers and drops rather than refuses. An unbounded connect
# hangs there, the miss is never counted, and the worker waits out the
# deadline holding chips.
head_listening() {
  timeout 5 bash -c "exec 3<>/dev/tcp/$1/$2" 2>/dev/null
}

# wait_for_head_to_go <host> <port>: return once the head has stopped
# listening. This is how a worker leaves: nothing kills these pods, so one that
# waited on its own processes would hold its chips until the JobSet deadline.
#
# The multihost manifests end the Job when the head succeeds (successPolicy on
# index 0), and Kubernetes terminates the workers then; this is the fallback,
# for a head that fails or a manifest without that rule. Gone for 20s
# straight, not one miss: a DNS blip on the headless service looks identical
# to a closed port from here, and leaving is irreversible. Probing every 2s
# keeps that window but notices the end within ~20s; three misses 15s apart
# held the worker's chips ~70s after the head had exited.
wait_for_head_to_go() {
  local gone_since=""
  while true; do
    if head_listening "$1" "$2"; then
      gone_since=""
    elif [ -z "$gone_since" ]; then
      gone_since=$SECONDS
    elif [ $((SECONDS - gone_since)) -ge 20 ]; then
      return 0
    fi
    sleep 2
  done
}

# publish_results: upload everything under ARTIFACTS_DIR. Not `|| true`: these
# artifacts are the step's only output, so a lost upload has to fail the step.
# The caller decides how, since a run that already failed keeps its own exit
# code.
publish_results() {
  [ -n "${ARTIFACTS_DIR:-}" ] || return 0
  [ -n "$(ls -A "$ARTIFACTS_DIR" 2>/dev/null)" ] || return 0
  if ! buildkite-agent artifact upload "$ARTIFACTS_DIR/**/*"; then
    echo "ERROR: artifacts were produced but could not be uploaded"
    return 1
  fi
}

# mp_serve_args: fill SERVE_ARGS, the `vllm serve` arguments every host of an
# mp slice shares, from the benchmark config the caller has sourced over its
# own defaults, plus DP_LOCAL, NUM_HOSTS, DP_ADDRESS and DP_RPC_PORT. The
# config's variables mean what run_benchmarks.sh makes of them on bare metal.
mp_serve_args() {
  if [ "$((DP_LOCAL * NUM_HOSTS))" -ne "$DATA_PARALLELISM" ]; then
    echo "ERROR: DP_LOCAL ($DP_LOCAL) x NUM_HOSTS ($NUM_HOSTS) != DATA_PARALLELISM ($DATA_PARALLELISM)" >&2
    return 1
  fi

  local extra_args=""
  [ "$ENABLE_EP" = "true" ] && extra_args="--enable-expert-parallel"
  [ -n "$QUANTIZATION" ] && extra_args="$extra_args --quantization $QUANTIZATION"
  extra_args="$extra_args --attention-backend $ATTENTION_BACKEND"
  [ -n "$EXTRA_SERVE_ARGS" ] && extra_args="$extra_args $EXTRA_SERVE_ARGS"
  # Empty leaves the flag off, so vLLM serves the checkpoint's own window - what
  # a full-context benchmark wants, rather than a number copied into a config.
  # Every config that sets it behaves exactly as before.
  [ -n "${MAX_MODEL_LEN:-}" ] && extra_args="$extra_args --max-model-len=$MAX_MODEL_LEN"
  local prefix_caching_flag="--no-enable-prefix-caching"
  [ "$ENABLE_PREFIX_CACHING" = "true" ] && prefix_caching_flag="--enable-prefix-caching"
  local async_flag="--async-scheduling"
  [ "$ASYNC_SCHEDULING" = "false" ] && async_flag="--no-async-scheduling"
  local serve_target="$MODEL"
  if [ -n "$MODEL_URI" ]; then
    serve_target="$MODEL_URI"
    extra_args="$extra_args --load-format runai_streamer --served-model-name $MODEL"
  fi

  # Word-split on purpose, as run_benchmarks.sh: the config's JSON has no
  # spaces.
  # shellcheck disable=SC2206
  SERVE_ARGS=(
    "$serve_target"
    --tensor-parallel-size="$TENSOR_PARALLELISM"
    --data-parallel-size="$DATA_PARALLELISM"
    --data-parallel-size-local="$DP_LOCAL"
    --data-parallel-address="$DP_ADDRESS"
    --data-parallel-rpc-port="$DP_RPC_PORT"
    --max-num-batched-tokens="$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs="$MAX_NUM_SEQS"
    "$async_flag"
    "$prefix_caching_flag"
    --gpu-memory-utilization="$GPU_MEMORY_UTILIZATION"
    --kv-cache-dtype="$KV_CACHE_DTYPE"
    $extra_args
  )
}

# mp_headless_worker: the life of any host but index 0 in an mp slice. It joins
# the head's DP group --headless at its own start rank and leaves once the head
# has gone. Reads INDEX, DP_LOCAL, DP_ADDRESS, PORT, SERVER_READY_WAIT_MIN and
# SERVE_ARGS.
mp_headless_worker() {
  local start_rank=$((INDEX * DP_LOCAL)) serve_pid deadline
  echo "--- Starting headless host ${INDEX} at data-parallel-start-rank=${start_rank}"
  vllm serve "${SERVE_ARGS[@]}" --headless --data-parallel-start-rank="${start_rank}" &
  serve_pid=$!

  # Not waited on: --headless never returns. The head finishing is what ends
  # the run, so this watches for that and leaves.
  echo "host ${INDEX}: waiting for the head to serve"
  deadline=$((SECONDS + SERVER_READY_WAIT_MIN * 60))
  while [ "$SECONDS" -lt "$deadline" ]; do
    head_listening "$DP_ADDRESS" "$PORT" && break
    if ! kill -0 "$serve_pid" 2>/dev/null; then
      echo "ERROR: this host's server exited before the head came up"
      return 1
    fi
    sleep 10
  done

  echo "host ${INDEX}: head is up; waiting for it to finish"
  wait_for_head_to_go "$DP_ADDRESS" "$PORT"
  echo "host ${INDEX}: head is gone; exiting"
  kill "$serve_pid" 2>/dev/null || true
}
