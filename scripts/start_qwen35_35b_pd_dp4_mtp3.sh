#!/usr/bin/env bash
# shellcheck disable=SC2034,SC2154  # Variables are consumed by the EXIT trap.
set -euo pipefail

action="${1:-start}"
case "${action}" in
  start|stop|status) ;;
  *)
    echo "Usage: $0 [start|stop|status]" >&2
    exit 2
    ;;
esac

run_root="${RUN_ROOT:-/mnt/data/tpu-misc-results/qwen35-35b-pr291-pd-dp4-mtp3}"
model_path="${MODEL_PATH:-/mnt/data/hf_cache/hub/models--Qwen--Qwen3.5-35B-A3B-FP8/snapshots/9d1823d2dee688a6b25e77009dc727688c44936e}"
python_bin="${PYTHON_BIN:-/home/yanzecheng_yzc/miniconda3/envs/torch_tpu/bin/python}"
vllm_src="${VLLM_SRC:-/mnt/data/vllm-v0.26.0-service}"
vllm_torchtpu_src="${VLLM_TORCHTPU_SRC:-/mnt/data/vllm-torchtpu}"
controller_src="${TPU_RAIDEN_CONTROLLER_SRC:-/home/yanzecheng_yzc/miniconda3/envs/torch_tpu/lib/python3.12/site-packages}"
host_ip="${HOST_IP:-$(hostname -I | awk '{print $1}')}"
served_model_name="${SERVED_MODEL_NAME:-Qwen3.5-35B-A3B-FP8}"
run_id="${RUN_ID:-qwen35_35b_pr291_pd_dp4_mtp3}"
startup_timeout_s="${STARTUP_TIMEOUT_S:-1800}"
allow_dirty_vllm_torchtpu="${ALLOW_DIRTY_VLLM_TORCHTPU:-0}"

expected_vllm_commit=568afb3a13806beb53bb2e6bd518269357b237c0
expected_vllm_torchtpu_commit="${EXPECTED_VLLM_TORCHTPU_COMMIT:-aeb0fa48736ba72f4c7d9e813ecb89582c5f327d}"
runtime_dir="${run_root}/runtime"
controller_py="${runtime_dir}/controller.py"

declare -Ar api_ports=([prefill]=8400 [decode]=9400)
declare -Ar controller_ports=([prefill]=28200 [decode]=28201)
declare -Ar rank_offsets=([prefill]=0 [decode]=4)
declare -Ar max_num_seqs_by_role=([prefill]=8 [decode]=16)
declare -Ar max_num_batched_tokens_by_role=([prefill]=16384 [decode]=1024)
declare -Ar transfer_ports=([prefill]=19300 [decode]=19400)
declare -Ar side_channel_ports=([prefill]=19700 [decode]=19720)
declare -Ar kv_roles=([prefill]=kv_producer [decode]=kv_consumer)
declare -Ar tmp_tags=([prefill]=current-pr291-p [decode]=current-pr291-d)
declare -Ar compile_configs=(
  [prefill]='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[16,1024,4096,8192,16384],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
  [decode]='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[16,32,64],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
)

pid_is_alive() {
  local pid="$1"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${pid}" 2>/dev/null
}

read_pid_file() {
  local pid_file="$1"
  [[ -s "${pid_file}" ]] || return 1
  local pid
  read -r pid <"${pid_file}"
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  printf '%s\n' "${pid}"
}

session_has_live_processes() {
  local sid="$1"
  local state
  while IFS= read -r state; do
    state="${state//[[:space:]]/}"
    if [[ -n "${state}" && "${state}" != Z* ]]; then
      return 0
    fi
  done < <(ps -o stat= --sid "${sid}" 2>/dev/null || true)
  return 1
}

stop_pid_file() {
  local pid_file="$1"
  local label="$2"
  local pid=""
  pid="$(read_pid_file "${pid_file}" 2>/dev/null || true)"
  if [[ -z "${pid}" ]]; then
    rm -f "${pid_file}"
    return
  fi
  local sid="${pid}"
  local stop_session=0
  if pid_is_alive "${pid}"; then
    local expected_command command_line
    if [[ "${label}" == *"server" ]]; then
      expected_command="vllm.entrypoints.openai.api_server"
    else
      expected_command="${controller_py}"
    fi
    command_line="$(tr '\0' ' ' <"/proc/${pid}/cmdline")"
    if [[ "${command_line}" != *"${expected_command}"* ]]; then
      echo "refusing to stop ${label}: pid=${pid} command does not match ${expected_command}" >&2
      return 1
    fi
    sid="$(ps -o sid= -p "${pid}" | tr -d ' ')"
    [[ "${sid}" == "${pid}" ]] && stop_session=1
  elif session_has_live_processes "${sid}"; then
    # Every tracked component is started with setsid. Its workers can outlive
    # the session leader after a fatal startup error, so the leader PID also
    # remains the stable identifier for the orphaned session.
    stop_session=1
  else
    rm -f "${pid_file}"
    return
  fi

  if (( stop_session )); then
    kill -TERM -- "-${sid}" 2>/dev/null || true
  else
    kill -TERM "${pid}" 2>/dev/null || true
  fi
  for _ in $(seq 1 60); do
    if (( stop_session )); then
      session_has_live_processes "${sid}" || break
    else
      pid_is_alive "${pid}" || break
    fi
    sleep 0.5
  done
  if (( stop_session )) && session_has_live_processes "${sid}"; then
    kill -KILL -- "-${sid}" 2>/dev/null || true
  elif (( ! stop_session )) && pid_is_alive "${pid}"; then
    kill -KILL "${pid}" 2>/dev/null || true
  fi
  rm -f "${pid_file}"
  echo "stopped ${label} pid=${pid}"
}

stop_all() {
  stop_pid_file "${run_root}/prefill/server.pid" "prefill server"
  stop_pid_file "${run_root}/decode/server.pid" "decode server"
  stop_pid_file "${run_root}/prefill/controller.pid" "prefill controller"
  stop_pid_file "${run_root}/decode/controller.pid" "decode controller"
}

show_status() {
  local role component pid_file pid api_port health
  for role in prefill decode; do
    for component in controller server; do
      pid_file="${run_root}/${role}/${component}.pid"
      pid="$(read_pid_file "${pid_file}" 2>/dev/null || true)"
      if [[ -n "${pid}" ]] && pid_is_alive "${pid}"; then
        printf '%s_%s=running pid=%s\n' "${role}" "${component}" "${pid}"
      else
        printf '%s_%s=stopped\n' "${role}" "${component}"
      fi
    done
    api_port="${api_ports[${role}]}"
    health="not-ready"
    curl -fsS --max-time 2 "http://127.0.0.1:${api_port}/health" >/dev/null 2>&1 \
      && health="ready"
    printf '%s_health=%s url=http://127.0.0.1:%s/health\n' \
      "${role}" "${health}" "${api_port}"
  done
}

if [[ "${action}" == "stop" ]]; then
  stop_all
  exit 0
fi
if [[ "${action}" == "status" ]]; then
  show_status
  exit 0
fi

extract_embedded() {
  local begin_marker="$1"
  local end_marker="$2"
  local destination="$3"
  awk -v begin_marker="${begin_marker}" -v end_marker="${end_marker}" '
    $0 == begin_marker { copying = 1; next }
    $0 == end_marker { exit }
    copying { print }
  ' "$0" >"${destination}"
}

validate_environment() {
  [[ "${startup_timeout_s}" =~ ^[1-9][0-9]*$ ]] || {
    echo "STARTUP_TIMEOUT_S must be a positive integer" >&2
    exit 2
  }
  [[ "${allow_dirty_vllm_torchtpu}" == "0" ||
     "${allow_dirty_vllm_torchtpu}" == "1" ]] || {
    echo "ALLOW_DIRTY_VLLM_TORCHTPU must be 0 or 1" >&2
    exit 2
  }
  local required_command
  for required_command in awk curl git hostname nohup ps setsid; do
    command -v "${required_command}" >/dev/null || {
      echo "missing required command: ${required_command}" >&2
      exit 2
    }
  done
  [[ -x "${python_bin}" ]] || {
    echo "missing Python executable: ${python_bin}" >&2
    exit 2
  }
  [[ -d "${model_path}" ]] || {
    echo "missing model snapshot: ${model_path}" >&2
    exit 2
  }
  [[ -d "${controller_src}/tpu_raiden" ]] || {
    echo "missing tpu_raiden package under: ${controller_src}" >&2
    exit 2
  }
  local actual_vllm_commit actual_vllm_torchtpu_commit
  actual_vllm_commit="$(git -C "${vllm_src}" rev-parse HEAD)"
  actual_vllm_torchtpu_commit="$(git -C "${vllm_torchtpu_src}" rev-parse HEAD)"
  [[ "${actual_vllm_commit}" == "${expected_vllm_commit}" ]] || {
    echo "vLLM commit mismatch: expected=${expected_vllm_commit} actual=${actual_vllm_commit}" >&2
    exit 2
  }
  git -C "${vllm_torchtpu_src}" merge-base --is-ancestor \
    "${expected_vllm_torchtpu_commit}" "${actual_vllm_torchtpu_commit}" || {
    echo "vllm-torchtpu source does not contain required commit: required=${expected_vllm_torchtpu_commit} actual=${actual_vllm_torchtpu_commit}" >&2
    exit 2
  }
  [[ -z "$(git -C "${vllm_src}" status --porcelain --untracked-files=no)" ]] || {
    echo "vLLM source worktree has tracked modifications" >&2
    exit 2
  }
  if [[ -n "$(git -C "${vllm_torchtpu_src}" status --porcelain --untracked-files=no)" ]]; then
    [[ "${allow_dirty_vllm_torchtpu}" == "1" ]] || {
      echo "vllm-torchtpu source worktree has tracked modifications" >&2
      exit 2
    }
    echo "using explicitly allowed vllm-torchtpu tracked modifications" >&2
  fi
  local local_rank
  for local_rank in {0..7}; do
    [[ -e "/dev/vfio/${local_rank}" ]] || {
      echo "missing local TPU device /dev/vfio/${local_rank}" >&2
      exit 2
    }
  done
}

assert_not_running() {
  local pid_file pid
  for pid_file in \
    "${run_root}/prefill/server.pid" \
    "${run_root}/decode/server.pid" \
    "${run_root}/prefill/controller.pid" \
    "${run_root}/decode/controller.pid"; do
    pid="$(read_pid_file "${pid_file}" 2>/dev/null || true)"
    if [[ -n "${pid}" ]] && pid_is_alive "${pid}"; then
      echo "existing process found in ${pid_file}: pid=${pid}; run '$0 stop' first" >&2
      exit 2
    fi
    rm -f "${pid_file}"
  done
}

prepare_runtime() {
  mkdir -p "${runtime_dir}"
  extract_embedded '#__CONTROLLER_BEGIN__' '#__CONTROLLER_END__' \
    "${controller_py}"
  chmod 0755 "${controller_py}"
  "${python_bin}" -c \
    'from pathlib import Path; import sys; [compile(Path(p).read_text(), p, "exec") for p in sys.argv[1:]]' \
    "${controller_py}"
  if [[ "${allow_dirty_vllm_torchtpu}" == "1" ]]; then
    git -C "${vllm_torchtpu_src}" diff --binary \
      >"${runtime_dir}/vllm-torchtpu-local.patch"
  fi
}

start_controller() {
  local role="$1"
  local role_dir="${run_root}/${role}"
  local controller_port="${controller_ports[${role}]}"
  local ready_file="${role_dir}/controller.ready.json"
  rm -f "${ready_file}"
  mkdir -p "${role_dir}/logs"

  nohup setsid env \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${controller_src}:${vllm_torchtpu_src}/src:${vllm_src}" \
    "${python_bin}" "${controller_py}" \
    --port "${controller_port}" \
    --advertise-host "${host_ip}" \
    --ready-file "${ready_file}" \
    >"${role_dir}/logs/controller.log" 2>&1 &
  local controller_pid=$!
  printf '%s\n' "${controller_pid}" >"${role_dir}/controller.pid"

  for _ in $(seq 1 200); do
    [[ -s "${ready_file}" ]] && return
    if ! pid_is_alive "${controller_pid}"; then
      echo "${role} controller exited during startup" >&2
      tail -n 100 "${role_dir}/logs/controller.log" >&2
      return 1
    fi
    sleep 0.1
  done
  echo "${role} controller did not become ready" >&2
  return 1
}

start_server() {
  local role="$1"
  local role_dir="${run_root}/${role}"
  local api_port="${api_ports[${role}]}"
  local controller_port="${controller_ports[${role}]}"
  local rank_offset="${rank_offsets[${role}]}"
  local max_num_seqs="${max_num_seqs_by_role[${role}]}"
  local max_num_batched_tokens="${max_num_batched_tokens_by_role[${role}]}"
  local compile_config="${compile_configs[${role}]}"
  local transfer_port="${transfer_ports[${role}]}"
  local side_channel_port="${side_channel_ports[${role}]}"
  local kv_role="${kv_roles[${role}]}"
  local tmp_tag="${tmp_tags[${role}]}"

  local kv_transfer_config
  kv_transfer_config="{\"kv_connector\":\"TPURaidenConnector\",\"kv_connector_module_path\":\"vllm_torchtpu.distributed.kv_transfer.tpu_connector\",\"kv_role\":\"${kv_role}\",\"kv_port\":14679}"
  local tmp_dir="/tmp/q35-rd-${role}-${tmp_tag}"
  local namespace="${run_id//[^a-zA-Z0-9_]/_}_${role}"
  mkdir -p \
    "${role_dir}/cache/xdg" \
    "${role_dir}/cache/hf" \
    "${role_dir}/cache/triton" \
    "${role_dir}/logs" \
    "${role_dir}/tpu" \
    "${tmp_dir}"

  local server_command=(
    "${python_bin}" -m vllm.entrypoints.openai.api_server
    --model "${model_path}"
    --served-model-name "${served_model_name}"
    --trust-remote-code
    --seed 42
    --host 0.0.0.0
    --port "${api_port}"
    --tensor-parallel-size 1
    --data-parallel-size 4
    --data-parallel-size-local 4
    --prefill-context-parallel-size 1
    --enable-expert-parallel
    --disable-custom-all-reduce
    --language-model-only
    --kv-cache-dtype fp8
    --quantization fp8
    --max-model-len 73728
    --max-num-batched-tokens "${max_num_batched_tokens}"
    --max-num-seqs "${max_num_seqs}"
    --gpu-memory-utilization 0.80
    --async-scheduling
    --no-enable-prefix-caching
    --no-disable-hybrid-kv-cache-manager
    --attention-backend CUSTOM
    --mamba-cache-mode align
    --enable-prompt-tokens-details
    --no-enable-log-requests
    --limit-mm-per-prompt '{"image":0,"video":0}'
    --compilation-config "${compile_config}"
    --kv-transfer-config "${kv_transfer_config}"
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  )

  printf '%q ' "${server_command[@]}" >"${role_dir}/server.command"
  printf '\n' >>"${role_dir}/server.command"

  nohup setsid env \
    -u SPEC_WARMUP \
    -u TPU_VISIBLE_CHIPS \
    -u TPU_CHIPS_PER_HOST_BOUNDS \
    -u TPU_VLLM_GDN_SPEC_CKPT_BLOCKS \
    PYTHONNOUSERSITE=1 \
    PYTHONPATH="${vllm_torchtpu_src}/src:${vllm_src}" \
    HF_HOME="${role_dir}/cache/hf" \
    HUGGINGFACE_HUB_CACHE="${role_dir}/cache/hf/hub" \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    XDG_CACHE_HOME="${role_dir}/cache/xdg" \
    TRITON_CACHE_DIR="${role_dir}/cache/triton" \
    TMPDIR="${tmp_dir}" \
    TPU_LOG_DIR="${role_dir}/tpu" \
    JAX_PLATFORMS=tpu,cpu \
    PJRT_DEVICE=TPU \
    TPU_BACKEND_TYPE=jax \
    VLLM_TARGET_DEVICE=tpu \
    NEW_MODEL_DESIGN=0 \
    SKIP_JAX_PRECOMPILE=1 \
    VLLM_XLA_CHECK_RECOMPILATION=0 \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
    TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled \
    TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1 \
    TPU_USE_RAIDEN_KV_CACHE_MANAGER=1 \
    TPU_RAIDEN_QWEN35_ADMISSION=1 \
    TPU_KV_RESHARD_TRANSPORT=raiden \
    TPU_RAIDEN_TRANSFER_PARALLELISM=1 \
    TPU_RAIDEN_CONTROLLER_ADDRESS="${host_ip}:${controller_port}" \
    TPU_RAIDEN_JOB_NAME="${role}" \
    TPU_RAIDEN_ENGINE_ID="${run_id}-${role}-engine" \
    TPU_KV_TRANSFER_NAMESPACE="${namespace}" \
    TPU_KV_TRANSFER_PORT="${transfer_port}" \
    TPU_SIDE_CHANNEL_PORT="${side_channel_port}" \
    TPU_P2P_WAIT_PULL_TIMEOUT=600 \
    TPU_KV_SHM_POOL_GB=8 \
    USE_BATCHED_RPA_KERNEL=1 \
    RAGGED_GATED_DELTA_RULE_IMPL=chunked_kernel_v3_pd \
    USE_MOE_SPARSE_CORE=1 \
    RAGGED_GATHER_VERSION=v2 \
    RAGGED_GATHER_REDUCE_VERSION=v2 \
    ONEHOT_MOE_PERMUTE_THRESHOLD=0 \
    LIBTPU_INIT_ARGS=' --xla_tpu_use_dynamic_smem_negotiation=true --xla_tpu_scoped_vmem_limit_kib=65536' \
    DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0 \
    VLLM_HOST_IP="${host_ip}" \
    TPU_LOCAL_RANK_OFFSET="${rank_offset}" \
    DEBUG_TPU_LOCAL_RANK_OFFSET="${rank_offset}" \
    "${server_command[@]}" \
    >"${role_dir}/logs/server.log" 2>&1 &
  local server_pid=$!
  printf '%s\n' "${server_pid}" >"${role_dir}/server.pid"
  echo "started ${role} server pid=${server_pid} api_port=${api_port} rank_offset=${rank_offset}"
}

wait_for_both_services() {
  local deadline=$((SECONDS + startup_timeout_s))
  local prefill_ready=0
  local decode_ready=0
  local role pid api_port
  while (( SECONDS < deadline )); do
    for role in prefill decode; do
      pid="$(read_pid_file "${run_root}/${role}/server.pid")"
      if ! pid_is_alive "${pid}"; then
        echo "${role} server exited during startup" >&2
        tail -n 160 "${run_root}/${role}/logs/server.log" >&2
        return 1
      fi
      api_port="${api_ports[${role}]}"
      if curl -fsS --max-time 2 "http://127.0.0.1:${api_port}/health" \
        >/dev/null 2>&1; then
        if [[ "${role}" == "prefill" ]]; then
          prefill_ready=1
        else
          decode_ready=1
        fi
      fi
    done
    if (( prefill_ready && decode_ready )); then
      return 0
    fi
    sleep 5
  done
  echo "services did not become ready within ${startup_timeout_s}s" >&2
  return 1
}

validate_environment
assert_not_running
prepare_runtime

start_completed=0
trap 'status=$?; if (( ! start_completed )); then stop_all; fi; exit "${status}"' EXIT
trap 'exit 130' INT TERM

start_controller prefill
start_controller decode
start_server prefill
start_server decode
wait_for_both_services

start_completed=1
trap - EXIT INT TERM
echo "Prefill and Decode are ready on the same host ${host_ip}."
show_status
echo "Run the workload with:"
echo "  RUN_ROOT=${run_root} ${python_bin} scripts/test_qwen35_35b_pd_dp4_mtp3.py"

: <<'__EMBEDDED_PAYLOADS__'
#__CONTROLLER_BEGIN__
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
from pathlib import Path

from tpu_raiden.rpc.raiden_controller import (
    RaidenController,
    RaidenControllerServer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--advertise-host", required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        force=True,
    )
    stopping = threading.Event()
    signal.signal(signal.SIGINT, lambda _signum, _frame: stopping.set())
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stopping.set())

    controller = RaidenController(port=args.port)
    server = RaidenControllerServer(controller)
    active_port = server.start()

    host = args.advertise_host.strip().strip("[]")
    rendered_host = f"[{host}]" if ":" in host else host
    payload = {
        "address": f"{rendered_host}:{active_port}",
        "pid": os.getpid(),
        "port": active_port,
    }
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.ready_file.with_name(
        f".{args.ready_file.name}.{os.getpid()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.ready_file)
    print(json.dumps(payload, sort_keys=True), flush=True)

    try:
        while not stopping.wait(1.0):
            pass
    finally:
        server.stop()
        print(
            json.dumps(
                {
                    "event": "raiden_controller_stopped",
                    "pid": os.getpid(),
                    "port": active_port,
                    "address": payload["address"],
                },
                sort_keys=True,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
#__CONTROLLER_END__
__EMBEDDED_PAYLOADS__
