#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Sourced by run_multihost.sh; reuse its image, slice, cleanup and artifacts.
# Variables come from the parent; HOME expands remotely and printf quotes args.
# shellcheck disable=SC2154,SC2016,SC2029

run_dsv41_dp16_multihost() {
  if [[ "$NUM_HOSTS" != 2 || "$#" != 2 || "$2" != /perf_eval_results/* ]]; then
    echo "Expected two hosts and arguments: <model_uri> /perf_eval_results/<name>" >&2
    return 1
  fi
  local model_uri="$1" results="$2" worker_ip="${WORKER_IPS_ARRAY[0]}"
  local serve_cmd remote_cmd head_pid worker_pid status=0 worker_status=0
  local host_out="$PERSIST_ROOT${results}/benchmark"
  local shm="${CONTAINER_SHM_SIZE:-256g}"
  local worker_env=(
    "${CONTAINER_ENV_COMMON[@]}"
    -e "BUILDKITE_COMMIT=${BUILDKITE_COMMIT:?}"
    -e "DSV41_HEAD_ADDRESS=$HEAD_INTERNAL_IP" -e DSV41_HOST_RANK=1
    --shm-size="$shm"
  )
  serve_cmd=$(printf '%q ' bash .buildkite/scripts/run_dsv41_benchmark.sh \
    decode_dp16 "$model_uri" "$results")
  # Artifacts are writable by the host agent, including on failure.
  serve_cmd="umask 000; $serve_cmd"

  ssh_retry "${SSH_USER}@${worker_ip}" \
    'mkdir -p ~/multihost ~/hf_home; gcloud auth configure-docker us-central1-docker.pkg.dev --quiet' || return 1
  ssh_retry "${SSH_USER}@${worker_ip}" \
    'cat > ~/multihost/run_cluster_mp.sh' < "$RUN_CLUSTER_MP" || return 1
  printf -v remote_cmd 'bash -s -- "$HOME/persist/perf_eval_results" %q' "$IMAGE_TAG"
  ssh_retry "${SSH_USER}@${worker_ip}" "$remote_cmd" \
    < "${SCRIPT_DIR}/reset_results_dir.sh" || return 1

  # Quote each Docker argument separately: command/env values contain spaces.
  printf -v remote_cmd 'bash ~/multihost/run_cluster_mp.sh %q "$HOME/hf_home" %q ' \
    "$IMAGE_TAG" "$serve_cmd"
  remote_cmd+="$(printf '%q ' "${worker_env[@]}")"
  remote_cmd+=' -v "$HOME/persist/perf_eval_results:/perf_eval_results"'
  # Never restart a disconnected worker inside a live collective world.
  # shellcheck disable=SC2029
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "$remote_cmd" \
    > perf_eval_results/worker-container.log 2>&1 &
  worker_pid=$!
  bash "$RUN_CLUSTER_MP" "$IMAGE_TAG" "$HOST_HF_HOME" "$serve_cmd" \
    "${CONTAINER_ENV_HEAD[@]}" --shm-size="$shm" \
    -e "DSV41_HEAD_ADDRESS=$HEAD_INTERNAL_IP" -e DSV41_HOST_RANK=0 \
    > >(tee "$MULTIHOST_RUN_LOG") 2>&1 &
  head_pid=$!

  local launched=0 probe_status=0 probe_failures=0
  while kill -0 "$head_pid" 2>/dev/null; do
    if [[ -f "$host_out/formal-complete" ]]; then break; fi
    if ! kill -0 "$worker_pid" 2>/dev/null; then
      echo "ERROR: DP16 worker exited before the head benchmark completed" >&2
      status=1
      docker stop -t 60 node >/dev/null 2>&1 || true
      break
    fi
    if [[ "$launched" == 0 && -f "${host_out%/benchmark}/ready-to-launch" ]]; then
      printf -v remote_cmd 'test -f "$HOME/persist"%q' "$results/worker/ready-to-launch"
      probe_status=0
      ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" "$remote_cmd" || probe_status=$?
      if [[ "$probe_status" == 0 ]]; then
        printf -v remote_cmd 'touch "$HOME/persist"%q' "$results/worker/launch-permitted"
        if ssh_retry "${SSH_USER}@${worker_ip}" "$remote_cmd" \
          && touch "${host_out%/benchmark}/launch-permitted"; then
          launched=1
        else
          status=1
          break
        fi
      elif [[ "$probe_status" == 255 ]]; then
        probe_failures=$((probe_failures + 1))
        echo "WARNING: DP16 readiness SSH failure ${probe_failures}/5" >&2
        if (( probe_failures >= 5 )); then
          status=1
          break
        fi
      elif [[ "$probe_status" == 1 ]]; then
        probe_failures=0
      else
        status=1
        break
      fi
    fi
    sleep 2
  done
  if [[ ! -f "$host_out/formal-complete" ]]; then status=1; fi
  if [[ "$status" == 0 ]]; then
    # Ask the Python owner to stop its server process group and flush status.
    printf -v remote_cmd 'docker exec node python -c %q %q' \
      'import os,signal,sys; from pathlib import Path; os.kill(int(Path(sys.argv[1]).read_text()), signal.SIGTERM)' \
      "$results/worker/benchmark/launcher.pid"
    ssh_retry "${SSH_USER}@${worker_ip}" "$remote_cmd" || status=1
  fi
  if [[ "$status" != 0 ]]; then
    ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
      'docker stop -t 60 node >/dev/null 2>&1 || true' || true
  fi
  local deadline=$((SECONDS + 120))
  while kill -0 "$worker_pid" 2>/dev/null && (( SECONDS < deadline )); do
    sleep 2
  done
  if kill -0 "$worker_pid" 2>/dev/null; then
    echo "ERROR: DP16 worker did not stop within 120 seconds" >&2
    status=1
    kill "$worker_pid" 2>/dev/null || true
  fi
  wait "$worker_pid" || worker_status=$?
  if [[ "$worker_status" != 0 ]]; then status=1; fi
  if [[ "$status" == 0 ]]; then
    touch "$host_out/shutdown-permitted" || status=1
  else
    docker stop -t 60 node >/dev/null 2>&1 || true
  fi
  wait "$head_pid" || status=$?

  # Both log sets survive container removal. A missing worker copy fails CI.
  # Do not restore metadata on the head container's existing root-owned dirs.
  printf -v remote_cmd 'tar -C "$HOME/persist"%q -cf - worker' "$results"
  ssh "${SSH_OPTS[@]}" "${SSH_USER}@${worker_ip}" \
    "$remote_cmd" \
    | tar -C "$PERSIST_ROOT$results" -xf - || status=1
  if [[ "$status" == 0 ]]; then
    docker run --rm --network none --entrypoint python \
      -v "$PERSIST_ROOT/perf_eval_results:/perf_eval_results" \
      "$IMAGE_TAG" scripts/dsv41_bench/server_steps/audit_multihost.py \
      "$results" || status=$?
  fi
  return "$status"
}
