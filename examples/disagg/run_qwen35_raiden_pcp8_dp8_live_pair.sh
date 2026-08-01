#!/usr/bin/env bash
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Run the Qwen3.5-35B-A3B-FP8 live disaggregated pair proven by the Stage-3
# Raiden resharding E2E: PCP8/TP1 prefill on one 8-chip host and DP8/TP1
# decode on another. This file deliberately contains its controller, client,
# validation, remote launch, and cleanup logic. It does not import helper
# scripts or add source checkouts to PYTHONPATH.

set -euo pipefail

readonly DEFAULT_MODEL="Qwen/Qwen3.5-35B-A3B-FP8"
readonly DEFAULT_SERVED_MODEL="Qwen3.5-35B-A3B-FP8"
readonly DEFAULT_RUN_ROOT="/mnt/disk/qwen35-raiden-live-pair"
readonly DEFAULT_EXPECTED_VLLM_VERSION="0.26.0"
readonly DEFAULT_STARTUP_TIMEOUT=2400
readonly DEFAULT_REQUEST_TIMEOUT=1800
readonly DEFAULT_PREFILL_API_PORT=8400
readonly DEFAULT_DECODE_API_PORT=9400
readonly DEFAULT_PREFILL_CONTROLLER_PORT=27000
readonly DEFAULT_DECODE_CONTROLLER_PORT=28000
readonly DEFAULT_KV_PORT=14579
readonly DEFAULT_TRANSFER_PORT=9100
readonly DEFAULT_SIDE_CHANNEL_PORT=9600

usage() {
  cat <<'EOF'
Run a two-host Qwen3.5-35B-A3B-FP8 Raiden live-pair test.

Run this command on the decode host:

  run_qwen35_raiden_pcp8_dp8_live_pair.sh \
    --prefill-ssh USER@PREFILL_HOST \
    --prefill-host PREFILL_ROUTABLE_IP \
    --decode-host DECODE_ROUTABLE_IP \
    [options]

Required:
  --prefill-ssh TARGET     OpenSSH target used to launch the prefill role.
  --prefill-host HOST      Prefill address reachable from the decode host.
  --decode-host HOST       Decode address reachable from the prefill host.

Options:
  --model MODEL            HF model ID or absolute path under /mnt/disk.
                           Default: Qwen/Qwen3.5-35B-A3B-FP8
  --served-model-name NAME Default: Qwen3.5-35B-A3B-FP8
  --prefill-python PATH    Python in the installed prefill environment.
                           Default: python3
  --decode-python PATH     Python in the installed decode environment.
                           Default: python3
  --run-root PATH          Per-run logs/caches; must be under /mnt/disk.
                           Default: /mnt/disk/qwen35-raiden-live-pair
  --run-id ID              Stable run identifier (default: UTC timestamp).
  --startup-timeout SEC    Server compilation/startup timeout (default: 2400).
  --request-timeout SEC    Each HTTP request timeout (default: 1800).
  --ssh-option OPTION      Append one OpenSSH -o option; may be repeated.
  --preflight-only         Check both installed environments and free ports.
  --strict-output          Also require eight token-87 / "xxxxxxxx" output.
  -h, --help               Show this help.

Advanced port overrides:
  --prefill-api-port PORT          Default: 8400
  --decode-api-port PORT           Default: 9400
  --prefill-controller-port PORT   Default: 27000
  --decode-controller-port PORT    Default: 28000
  --kv-port PORT                   vLLM connector metadata port (14579)
  --transfer-port PORT             Raiden control-port base (9100)
  --side-channel-port PORT         Connector side-channel base (9600)

Both machines need an 8-chip TPU host and installed vllm, vllm-torchtpu,
torch-tpu, and the Torch tpu-raiden wheel. The routable host values are not
necessarily the SSH names. Raiden advertises dynamic worker endpoints, so the
two hosts need bidirectional TCP reachability, not just access to the fixed
API and controller ports.

The default pass criterion is transport correctness: FA KV-cache plus the
three conv and three SSM state groups must complete through Raiden. Generated
token equality is intentionally checked only with --strict-output.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

log() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

is_positive_integer() {
  [[ "$1" =~ ^[1-9][0-9]*$ ]]
}

validate_port() {
  local name="$1"
  local value="$2"
  [[ "${value}" =~ ^[0-9]+$ ]] || die "${name} must be an integer: ${value}"
  ((value >= 1 && value <= 65535)) || die "${name} is outside [1, 65535]: ${value}"
}

quote_command() {
  local rendered=""
  local value quoted
  for value in "$@"; do
    printf -v quoted '%q' "${value}"
    rendered+="${rendered:+ }${quoted}"
  done
  printf '%s' "${rendered}"
}

host_port() {
  local host="$1"
  local port="$2"
  if [[ "${host}" == *:* && "${host}" != \[*\] ]]; then
    printf '[%s]:%s' "${host}" "${port}"
  else
    printf '%s:%s' "${host}" "${port}"
  fi
}

http_host() {
  local host="$1"
  if [[ "${host}" == *:* && "${host}" != \[*\] ]]; then
    printf '[%s]' "${host}"
  else
    printf '%s' "${host}"
  fi
}

proc_start_ticks() {
  local pid="$1"
  local stat rest
  [[ -r "/proc/${pid}/stat" ]] || return 1
  IFS= read -r stat <"/proc/${pid}/stat" || return 1
  rest="${stat##*) }"
  # Fields in rest begin at proc(5) field 3; starttime is field 22.
  # shellcheck disable=SC2086
  set -- ${rest}
  [[ "$#" -ge 20 ]] || return 1
  printf '%s\n' "${20}"
}

pid_matches_start_ticks() {
  local pid="$1"
  local expected="$2"
  local actual
  [[ "${pid}" =~ ^[1-9][0-9]*$ && "${expected}" =~ ^[0-9]+$ ]] || return 1
  actual="$(proc_start_ticks "${pid}" 2>/dev/null)" || return 1
  [[ "${actual}" == "${expected}" ]]
}

claim_process_group() {
  local pid="$1"
  local expected_ticks="$2"
  local pgid="$3"
  local actual_pgid
  [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  ((pgid > 1)) || return 1
  pid_matches_start_ticks "${pid}" "${expected_ticks}" || return 1
  actual_pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
  [[ "${actual_pgid}" == "${pgid}" ]]
}

process_group_alive() {
  local pgid="$1"
  [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 1
  ((pgid > 1)) || return 1
  kill -0 -- "-${pgid}" 2>/dev/null
}

signal_owned_process_group() {
  local pgid="$1"
  local signal="${2:-TERM}"
  [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] || return 0
  ((pgid > 1)) || return 0
  kill -"${signal}" -- "-${pgid}" 2>/dev/null || true
}

stop_role_dir() {
  local role_dir="$1"
  local supervisor_pid="" supervisor_ticks="" supervisor_pgid=""
  local supervisor_owned=0
  local child pid ticks pgid
  local any_alive
  local -A owned_pgids=()

  if [[ -r "${role_dir}/supervisor.pid" \
      && -r "${role_dir}/supervisor.start_ticks" ]]; then
    IFS= read -r supervisor_pid <"${role_dir}/supervisor.pid" || true
    IFS= read -r supervisor_ticks <"${role_dir}/supervisor.start_ticks" || true
  fi
  if [[ -r "${role_dir}/supervisor.pgid" ]]; then
    IFS= read -r supervisor_pgid <"${role_dir}/supervisor.pgid" || true
  fi
  if claim_process_group "${supervisor_pid}" "${supervisor_ticks}" \
      "${supervisor_pgid}"; then
    supervisor_owned=1
  fi

  # Claim every group while its recorded leader PID/start-time identity still
  # matches. Once claimed, the PGID remains safe to signal even if the leader
  # exits first: a live process group keeps that PGID allocated until its last
  # member exits.
  for child in server controller; do
    [[ -r "${role_dir}/${child}.pid" ]] || continue
    [[ -r "${role_dir}/${child}.start_ticks" ]] || continue
    [[ -r "${role_dir}/${child}.pgid" ]] || continue
    IFS= read -r pid <"${role_dir}/${child}.pid" || continue
    IFS= read -r ticks <"${role_dir}/${child}.start_ticks" || continue
    IFS= read -r pgid <"${role_dir}/${child}.pgid" || continue
    if claim_process_group "${pid}" "${ticks}" "${pgid}"; then
      owned_pgids["${child}"]="${pgid}"
    fi
  done

  if ((supervisor_owned == 1)); then
    signal_owned_process_group "${supervisor_pgid}" TERM
  elif pid_matches_start_ticks "${supervisor_pid}" "${supervisor_ticks}"; then
    kill -TERM "${supervisor_pid}" 2>/dev/null || true
  fi

  # The supervisor normally sends these signals itself. Sending TERM to the
  # already-claimed groups here also covers an ungraceful supervisor death.
  for pgid in "${owned_pgids[@]}"; do
    signal_owned_process_group "${pgid}" TERM
  done

  for _ in $(seq 1 60); do
    any_alive=0
    if ((supervisor_owned == 1)); then
      process_group_alive "${supervisor_pgid}" && any_alive=1
    else
      pid_matches_start_ticks "${supervisor_pid}" "${supervisor_ticks}" \
        && any_alive=1
    fi
    for pgid in "${owned_pgids[@]}"; do
      process_group_alive "${pgid}" && any_alive=1
    done
    ((any_alive == 0)) && return 0
    sleep 0.5
  done

  if ((supervisor_owned == 1)) \
      && process_group_alive "${supervisor_pgid}"; then
    signal_owned_process_group "${supervisor_pgid}" KILL
  elif pid_matches_start_ticks "${supervisor_pid}" "${supervisor_ticks}"; then
    kill -KILL "${supervisor_pid}" 2>/dev/null || true
  fi
  for pgid in "${owned_pgids[@]}"; do
    process_group_alive "${pgid}" && signal_owned_process_group "${pgid}" KILL
  done

  for _ in $(seq 1 20); do
    any_alive=0
    if ((supervisor_owned == 1)); then
      process_group_alive "${supervisor_pgid}" && any_alive=1
    else
      pid_matches_start_ticks "${supervisor_pid}" "${supervisor_ticks}" \
        && any_alive=1
    fi
    for pgid in "${owned_pgids[@]}"; do
      process_group_alive "${pgid}" && any_alive=1
    done
    ((any_alive == 0)) && return 0
    sleep 0.1
  done
  echo "ERROR: role teardown left a supervisor or child process group alive: ${role_dir}" >&2
  return 1
}

check_role_environment() {
  local python_bin="$1"
  local role="$2"
  local role_dir="$3"
  local expected_vllm_version="$4"
  local api_port="$5"
  local controller_port="$6"
  local transfer_port="$7"
  local side_channel_port="$8"
  local model="$9"

  command -v setsid >/dev/null 2>&1 || die "setsid is required on the ${role} host"
  mkdir -p "${role_dir}"
  cd "${role_dir}"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  export HF_HOME=/mnt/disk
  export HUGGINGFACE_HUB_CACHE=/mnt/disk/hub

  if [[ "${model}" == /* ]]; then
    [[ "${model}" == /mnt/disk/* ]] || die "absolute model paths must be under /mnt/disk: ${model}"
    [[ -d "${model}" ]] || die "model path is absent on the ${role} host: ${model}"
  fi

  "${python_bin}" - "${role}" "${role_dir}" "${expected_vllm_version}" \
    "${api_port}" "${controller_port}" "${transfer_port}" \
    "${side_channel_port}" <<'PY'
import contextlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import socket
import sys

role, role_dir, expected, api_port, controller_port, transfer_port, side_port = sys.argv[1:]

# TPU/plugin imports emit useful diagnostics. Keep the report on stdout valid
# JSON by routing those diagnostics to the preflight stderr artifact.
with contextlib.redirect_stdout(sys.stderr):
    import torch
    import torch_tpu
    import vllm
    import vllm_torchtpu
    from tpu_raiden.api.torch.kv_cache_manager import _torch_impl
    from tpu_raiden.rpc.raiden_controller import RaidenController, RaidenControllerServer
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    _tpu_raiden_torch = _torch_impl()

actual = vllm.__version__
if actual != expected and not actual.startswith(expected + "+"):
    raise SystemExit(f"vLLM must be {expected} (a +local suffix is allowed), got {actual}")
if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is None:
    raise SystemExit("vLLM OpenAI API server entry point is unavailable")

role_path = pathlib.Path(role_dir)
probe = role_path / f".write-test-{os.getpid()}"
probe.write_text("ok\n", encoding="utf-8")
probe.unlink()

ports = [int(api_port), int(controller_port)]
# Stage-3 uses one even control port for each of eight source/destination
# ranks. Keep the legacy side-channel range clear as well.
ports.extend(int(transfer_port) + 2 * rank for rank in range(8))
ports.extend(int(side_port) + rank for rank in range(8))
if len(ports) != len(set(ports)):
    raise SystemExit(f"configured {role} ports overlap: {ports}")

sockets = []
try:
    for port in ports:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("0.0.0.0", port))
        sockets.append(sock)
except OSError as exc:
    raise SystemExit(f"{role} port {port} is unavailable: {exc}") from exc
finally:
    for sock in sockets:
        sock.close()

def dist_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

print(json.dumps({
    "role": role,
    "python": sys.executable,
    "python_version": sys.version.split()[0],
    "vllm_version": actual,
    "vllm_file": vllm.__file__,
    "vllm_torchtpu_file": vllm_torchtpu.__file__,
    "torch_version": torch.__version__,
    "torch_tpu_version": dist_version("torch-tpu"),
    "tpu_raiden_torch_version": dist_version("tpu-raiden-torch"),
    "raiden_native_file": _tpu_raiden_torch.__file__,
    "controller_import": f"{RaidenController.__module__}.{RaidenController.__name__}",
    "controller_server_import": f"{RaidenControllerServer.__module__}.{RaidenControllerServer.__name__}",
    "connector_import": f"{TPURaidenConnector.__module__}.{TPURaidenConnector.__name__}",
    "ports_checked": ports,
    "hf_home": os.environ["HF_HOME"],
}, indent=2, sort_keys=True))
PY
}

configure_role_environment() {
  local role="$1"
  local role_host="$2"
  local controller_port="$3"
  local run_id="$4"
  local role_dir="$5"
  local transfer_port="$6"
  local side_channel_port="$7"
  local controller_address
  local namespace="${run_id//[^a-zA-Z0-9_]/_}_${role}"

  controller_address="$(host_port "${role_host}" "${controller_port}")"

  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  export HF_HOME=/mnt/disk
  export HUGGINGFACE_HUB_CACHE=/mnt/disk/hub

  export TPU_RAIDEN_CONTROLLER_ADDRESS="${controller_address}"
  export JAX_PLATFORMS=tpu,cpu
  export PJRT_DEVICE=TPU
  export TPU_BACKEND_TYPE=jax
  export VLLM_TARGET_DEVICE=tpu
  export SKIP_JAX_PRECOMPILE=1
  export VLLM_XLA_CHECK_RECOMPILATION=0
  export USE_MOE_SPARSE_CORE=0

  # This is the single current layout gate.
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
  export TPU_RAIDEN_QWEN35_ADMISSION=1
  export TPU_KV_RESHARD_TRANSPORT=raiden
  export TPU_RAIDEN_TRANSFER_PARALLELISM=8

  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=65536 --xla_tpu_enable_latency_hiding_scheduler=false"

  export TPU_P2P_WAIT_PULL_TIMEOUT=600
  export TPU_KV_SHM_POOL_GB=8
  export TPU_KV_TRANSFER_PORT="${transfer_port}"
  export TPU_SIDE_CHANNEL_PORT="${side_channel_port}"
  export ONEHOT_MOE_PERMUTE_THRESHOLD=1024
  export TPU_RAGGED_GATHER_REDUCE_IMPL=fallback
  export TPU_RAGGED_GATHER_IMPL=fallback
  export DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH=0

  unset VLLM_DISABLE_COMPILE_CACHE VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
  unset VLLM_XLA_CACHE_PATH TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT
  export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled

  export TPU_LOG_DIR="${role_dir}/tpu"
  export XDG_CACHE_HOME="${role_dir}/cache"
  export TRITON_CACHE_DIR="${role_dir}/triton"
  export TMPDIR="/tmp/q35-raiden-${namespace}"
  mkdir -p "${TPU_LOG_DIR}" "${XDG_CACHE_HOME}" "${TRITON_CACHE_DIR}" "${TMPDIR}"

  export VLLM_HOST_IP="${role_host}"
  export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7
  export TPU_CHIPS_PER_HOST_BOUNDS=1,8,1
  unset DEBUG_TPU_LOCAL_RANK_OFFSET
  export TPU_KV_TRANSFER_NAMESPACE="${namespace}"
  export TPU_RAIDEN_JOB_NAME="${role}"
  export TPU_RAIDEN_ENGINE_ID="${run_id}-${role}-engine"
}

record_child() {
  local role_dir="$1"
  local name="$2"
  local pid="$3"
  local pgid ticks supervisor_pgid
  ticks="$(proc_start_ticks "${pid}")"
  supervisor_pgid="$(ps -o pgid= -p "${BASHPID}" | tr -d '[:space:]')"
  [[ "${ticks}" =~ ^[0-9]+$ && "${supervisor_pgid}" =~ ^[1-9][0-9]*$ ]] \
    || die "could not record ${name} process identity"

  # The background PID can be observed between fork(2) and setsid(2). Poll
  # until it enters its new group, while continuously proving this is still
  # the same process and not a reused PID.
  pgid=""
  for _ in $(seq 1 100); do
    pid_matches_start_ticks "${pid}" "${ticks}" \
      || die "${name} exited before entering its process group"
    pgid="$(ps -o pgid= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${pgid}" =~ ^[1-9][0-9]*$ ]] \
        && ((pgid > 1)) && [[ "${pgid}" != "${supervisor_pgid}" ]]; then
      break
    fi
    pgid=""
    sleep 0.01
  done
  [[ -n "${pgid}" ]] || die "${name} did not enter a separate process group"
  printf '%s\n' "${pid}" >"${role_dir}/${name}.pid"
  printf '%s\n' "${ticks}" >"${role_dir}/${name}.start_ticks"
  printf '%s\n' "${pgid}" >"${role_dir}/${name}.pgid"
}

run_role_server() {
  local role="$1"
  local python_bin="$2"
  local role_host="$3"
  local role_dir="$4"
  local run_id="$5"
  local model="$6"
  local served_model="$7"
  local api_port="$8"
  local controller_port="$9"
  local kv_port="${10}"
  local transfer_port="${11}"
  local side_channel_port="${12}"
  local expected_vllm_version="${13}"
  local controller_pid="" controller_ticks="" controller_pgid=""
  local server_pid="" server_ticks="" server_pgid=""
  local controller_address compilation_config
  local server_owned=0 controller_owned=0
  local supervisor_pgid
  local -a common_args role_args server_command

  umask 077
  mkdir -p "${role_dir}"
  cd "${role_dir}"

  # This nested handler is reached indirectly through the EXIT trap below.
  # shellcheck disable=SC2317
  role_cleanup() {
    local status="$?"
    local server_alive controller_alive
    trap - EXIT INT TERM HUP
    set +e
    if ((server_owned == 1)); then
      signal_owned_process_group "${server_pgid}" TERM
    fi
    if ((controller_owned == 1)); then
      signal_owned_process_group "${controller_pgid}" TERM
    fi
    for _ in $(seq 1 60); do
      server_alive=0
      controller_alive=0
      ((server_owned == 1)) && process_group_alive "${server_pgid}" && server_alive=1
      ((controller_owned == 1)) && process_group_alive "${controller_pgid}" && controller_alive=1
      ((server_alive == 0 && controller_alive == 0)) && break
      sleep 0.5
    done
    if ((server_owned == 1)) && process_group_alive "${server_pgid}"; then
      signal_owned_process_group "${server_pgid}" KILL
    fi
    if ((controller_owned == 1)) && process_group_alive "${controller_pgid}"; then
      signal_owned_process_group "${controller_pgid}" KILL
    fi
    for _ in $(seq 1 20); do
      server_alive=0
      controller_alive=0
      ((server_owned == 1)) && process_group_alive "${server_pgid}" && server_alive=1
      ((controller_owned == 1)) && process_group_alive "${controller_pgid}" && controller_alive=1
      ((server_alive == 0 && controller_alive == 0)) && break
      sleep 0.1
    done
    if ((server_alive == 1 || controller_alive == 1)); then
      echo "ERROR: ${role} cleanup left a child process group alive" >&2
      status=1
    fi
    rm -f "${role_dir}/supervisor.pid" "${role_dir}/supervisor.start_ticks"
    rm -f "${role_dir}/supervisor.pgid"
    exit "${status}"
  }
  trap role_cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM HUP

  printf '%s\n' "${BASHPID}" >"${role_dir}/supervisor.pid"
  proc_start_ticks "${BASHPID}" >"${role_dir}/supervisor.start_ticks"
  supervisor_pgid="$(ps -o pgid= -p "${BASHPID}" | tr -d '[:space:]')"
  [[ "${supervisor_pgid}" == "${BASHPID}" ]] \
    || die "internal serve role must run in its own setsid session"
  printf '%s\n' "${supervisor_pgid}" >"${role_dir}/supervisor.pgid"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  export HF_HOME=/mnt/disk
  export HUGGINGFACE_HUB_CACHE=/mnt/disk/hub

  rm -f "${role_dir}/controller.ready.json"
  setsid "${python_bin}" - --port "${controller_port}" \
    --advertise-host "${role_host}" --request-registry-ttl-s 600 \
    --ready-file "${role_dir}/controller.ready.json" \
    >"${role_dir}/controller.log" 2>&1 <<'PY' &
from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import threading
import time
from pathlib import Path

from tpu_raiden.rpc.raiden_controller import RaidenController, RaidenControllerServer

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, required=True)
parser.add_argument("--advertise-host", required=True)
parser.add_argument("--request-registry-ttl-s", type=float, default=600.0)
parser.add_argument("--ready-file", type=Path, required=True)
args = parser.parse_args()

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s",
                    force=True)
stopping = threading.Event()
signal.signal(signal.SIGINT, lambda _signum, _frame: stopping.set())
signal.signal(signal.SIGTERM, lambda _signum, _frame: stopping.set())

controller = RaidenController(port=args.port,
                              request_registry_ttl_s=args.request_registry_ttl_s)
server = RaidenControllerServer(controller)
server.start()
host = args.advertise_host.strip().strip("[]")
rendered_host = f"[{host}]" if ":" in host else host
payload = {
    "schema_version": 1,
    "event": "raiden_controller_ready",
    "pid": os.getpid(),
    "requested_port": args.port,
    "port": server.port,
    "advertise_host": host,
    "address": f"{rendered_host}:{server.port}",
    "request_registry_ttl_s": args.request_registry_ttl_s,
    "time_unix_s": time.time(),
}
args.ready_file.parent.mkdir(parents=True, exist_ok=True)
temporary = args.ready_file.with_name(f".{args.ready_file.name}.{os.getpid()}.tmp")
temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(args.ready_file)
print(json.dumps(payload, sort_keys=True), flush=True)
logging.info("Raiden controller INFO event logging enabled address=%s", payload["address"])
try:
    while not stopping.wait(1.0):
        pass
finally:
    server.stop()
    print(json.dumps({"event": "raiden_controller_stopped",
                      "pid": os.getpid(), "port": server.port,
                      "address": payload["address"]}, sort_keys=True), flush=True)
PY
  controller_pid=$!
  record_child "${role_dir}" controller "${controller_pid}"
  controller_ticks="$(<"${role_dir}/controller.start_ticks")"
  controller_pgid="$(<"${role_dir}/controller.pgid")"
  claim_process_group "${controller_pid}" "${controller_ticks}" \
    "${controller_pgid}" || die "could not claim controller process group"
  controller_owned=1

  for _ in $(seq 1 200); do
    [[ -s "${role_dir}/controller.ready.json" ]] && break
    pid_matches_start_ticks "${controller_pid}" "${controller_ticks}" \
      || die "${role} controller exited during startup; see ${role_dir}/controller.log"
    sleep 0.1
  done
  [[ -s "${role_dir}/controller.ready.json" ]] \
    || die "${role} controller did not become ready"

  configure_role_environment "${role}" "${role_host}" "${controller_port}" \
    "${run_id}" "${role_dir}" "${transfer_port}" "${side_channel_port}"
  controller_address="$(host_port "${role_host}" "${controller_port}")"
  [[ "${TPU_RAIDEN_CONTROLLER_ADDRESS}" == "${controller_address}" ]] \
    || die "internal controller address mismatch"

  compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
  common_args=(
    --model "${model}"
    --served-model-name "${served_model}"
    --trust-remote-code
    --seed 42
    --max-model-len 65536
    --enable-expert-parallel
    --disable-custom-all-reduce
    --gpu-memory-utilization 0.9
    --kv-cache-dtype fp8
    --language-model-only
    --max-num-batched-tokens 4096
    --max-num-seqs 2
    --no-disable-hybrid-kv-cache-manager
    --attention-backend CUSTOM
    --mamba-cache-mode align
    --enable-prompt-tokens-details
    --no-enable-log-requests
    --compilation-config "${compilation_config}"
    --no-enable-prefix-caching
    --no-async-scheduling
    --tensor-parallel-size 1
  )

  if [[ "${role}" == prefill ]]; then
    role_args=(
      --host 0.0.0.0
      --port "${api_port}"
      --num-gpu-blocks-override 16
      --prefill-context-parallel-size 8
      --cp-kv-cache-interleave-size 256
      --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":'"${kv_port}"'}'
    )
  elif [[ "${role}" == decode ]]; then
    role_args=(
      --host 127.0.0.1
      --port "${api_port}"
      --num-gpu-blocks-override 19
      --prefill-context-parallel-size 1
      --data-parallel-size 8
      --data-parallel-size-local 8
      --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_consumer","kv_port":'"${kv_port}"'}'
    )
  else
    die "unknown server role: ${role}"
  fi

  server_command=("${python_bin}" -m vllm.entrypoints.openai.api_server
                  "${common_args[@]}" "${role_args[@]}")
  quote_command "${server_command[@]}" >"${role_dir}/server.command"
  printf '\n' >>"${role_dir}/server.command"
  setsid "${server_command[@]}" >"${role_dir}/server.log" 2>&1 &
  server_pid=$!
  record_child "${role_dir}" server "${server_pid}"
  server_ticks="$(<"${role_dir}/server.start_ticks")"
  server_pgid="$(<"${role_dir}/server.pgid")"
  claim_process_group "${server_pid}" "${server_ticks}" "${server_pgid}" \
    || die "could not claim server process group"
  server_owned=1

  log "${role} supervisor started controller=${controller_pid} server=${server_pid}"
  while true; do
    if ! pid_matches_start_ticks "${server_pid}" "${server_ticks}"; then
      wait "${server_pid}" || status=$?
      die "${role} API server exited (status=${status:-0}); see ${role_dir}/server.log"
    fi
    if ! pid_matches_start_ticks "${controller_pid}" "${controller_ticks}"; then
      wait "${controller_pid}" || status=$?
      die "${role} controller exited (status=${status:-0}); see ${role_dir}/controller.log"
    fi
    sleep 2
  done
}

http_ready() {
  local python_bin="$1"
  local url="$2"
  "${python_bin}" - "${url}" >/dev/null 2>&1 <<'PY'
import sys
import urllib.request
with urllib.request.urlopen(sys.argv[1], timeout=2) as response:
    if response.status != 200:
        raise SystemExit(1)
PY
}

run_request() {
  local python_bin="$1"
  local prefill_url="$2"
  local decode_url="$3"
  local served_model="$4"
  local output_dir="$5"
  local timeout="$6"

  "${python_bin}" - "${prefill_url}" "${decode_url}" "${served_model}" \
    "${output_dir}" "${timeout}" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.error
import urllib.request
import uuid

prefill_url, decode_url, model, output_dir, timeout = sys.argv[1:]
output = pathlib.Path(output_dir)
timeout = float(timeout)
request_id = f"q35-raiden-e2e-{uuid.uuid4()}"

payload = {
    "add_special_tokens": False,
    "ignore_eos": True,
    "max_tokens": 8,
    "model": model,
    "n": 1,
    "prompt": [87] * 32768,
    "return_token_ids": True,
    "seed": 0,
    "stream": False,
    "temperature": 0.0,
}

def write_json(name, value):
    (output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                               encoding="utf-8")

def post(url, value):
    request = urllib.request.Request(
        url,
        data=json.dumps(value, separators=(",", ":")).encode("utf-8"),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            if response.status != 200:
                raise RuntimeError(f"{url} returned HTTP {response.status}: {body[:1000]!r}")
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(f"{url} returned HTTP {exc.code}: {body[:4000]!r}") from exc
    return json.loads(body)

write_json("request.json", payload)
(output / "request_id.txt").write_text(request_id + "\n", encoding="utf-8")
(output / "request_start_utc.txt").write_text(
    time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")

try:
    rendered = post(prefill_url + "/v1/completions/render", payload)
    write_json("render_response.json", rendered)
    if not isinstance(rendered, list) or not rendered:
        raise RuntimeError("completion render response was not a non-empty list")
    rendered_prompt = rendered[0].get("token_ids")
    if not isinstance(rendered_prompt, list) or not all(type(x) is int for x in rendered_prompt):
        raise RuntimeError("completion render response did not contain integer token_ids")

    routed = dict(payload)
    routed["prompt"] = rendered_prompt
    prefill_payload = dict(routed)
    prefill_payload["max_tokens"] = 1
    prefill_payload["stream"] = False
    prefill_response = post(prefill_url + "/v1/completions", prefill_payload)
    write_json("prefill_response.json", prefill_response)
    kv_params = prefill_response.get("kv_transfer_params")
    if not isinstance(kv_params, dict) or not kv_params:
        raise RuntimeError("prefill response had no non-empty kv_transfer_params")

    routed["kv_transfer_params"] = kv_params
    write_json("decode_request.json", routed)
    response = post(decode_url + "/v1/completions", routed)
    write_json("response.json", response)
except Exception as exc:
    write_json("request_error.json", {"error": type(exc).__name__, "message": str(exc)})
    raise
finally:
    (output / "request_end_utc.txt").write_text(
        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n", encoding="utf-8")
PY
}

validate_result() {
  local python_bin="$1"
  local output_dir="$2"
  local strict_output="$3"

  "${python_bin}" - "${output_dir}" "${strict_output}" <<'PY'
from __future__ import annotations

import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
strict_output = sys.argv[2] == "1"
checks = {}
details = {}
errors = []

def load_json(name):
    try:
        return json.loads((root / name).read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"could not read {name}: {exc}")
        return {}

def log_events(name):
    events = []
    try:
        lines = (root / name).read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception as exc:
        errors.append(f"could not read {name}: {exc}")
        return events
    for line in lines:
        start = line.find("{")
        if start < 0:
            continue
        try:
            value = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "event" in value:
            events.append(value)
    return events

prefill_response = load_json("prefill_response.json")
response = load_json("response.json")
prefill_events = log_events("prefill.server.log")
decode_events = log_events("decode.server.log")
prefill_controller_events = log_events("prefill.controller.log")
decode_controller_events = log_events("decode.controller.log")

kv_params = prefill_response.get("kv_transfer_params")
checks["prefill_returned_kv_transfer_params"] = isinstance(kv_params, dict) and bool(kv_params)
main_uuid = kv_params.get("uuid") if isinstance(kv_params, dict) else None
details["uuid"] = main_uuid

usage = response.get("usage") if isinstance(response, dict) else None
usage = usage if isinstance(usage, dict) else {}
prompt_details = usage.get("prompt_tokens_details")
prompt_details = prompt_details if isinstance(prompt_details, dict) else {}
checks["prompt_tokens_32768"] = usage.get("prompt_tokens") == 32768
checks["completion_tokens_8"] = usage.get("completion_tokens") == 8
checks["cached_tokens_32767"] = prompt_details.get("cached_tokens") == 32767

senders = [e for e in prefill_events
           if e.get("event") == "raiden_stage3_sender_complete"
           and e.get("uuid") == main_uuid and e.get("num_tokens") == 32767]
sender_ranks = sorted({e.get("transfer_rank") for e in senders
                       if type(e.get("transfer_rank")) is int})
checks["prefill_sender_ranks_0_through_7_complete"] = sender_ranks == list(range(8))
details["prefill_sender_ranks"] = sender_ranks

submitted = [e for e in decode_events
             if e.get("event") == "raiden_stage3_transfer_submitted"
             and e.get("uuid") == main_uuid and e.get("num_tokens") == 32767]
checks["decode_transfer_submitted"] = bool(submitted)
checks["decode_transfer_has_three_state_groups"] = any(
    e.get("state_group_count") == 3 for e in submitted)
checks["decode_receiver_armed_before_push"] = any(
    e.get("recv_armed_before_push") is True for e in submitted)

receives = [e for e in decode_events
            if e.get("event") == "raiden_stage3_receiver_complete"
            and e.get("uuid") == main_uuid and e.get("num_tokens") == 32767]
checks["decode_receiver_complete"] = bool(receives)

main_req_id = None
for event in submitted:
    if isinstance(event.get("req_id"), str):
        main_req_id = event["req_id"]
        break
details["source_request_id"] = main_req_id

# T3.1 sibling collapse: ONE transfer carries FA plus all six GDN state
# classes under the base request id — one arm, one dispatch, seven tags.
expected_tags = ["fa",
                 "gdn.conv.g0", "gdn.ssm.g0",
                 "gdn.conv.g1", "gdn.ssm.g1",
                 "gdn.conv.g2", "gdn.ssm.g2"]
required_ids = ({main_req_id} if isinstance(main_req_id, str) else set())
# P2 topology: the SOURCE controller plans, arms the decode workers
# directly, and dispatches; the decode controller is a metadata directory
# and emits no arm event. Receiver arming is attested by the source
# controller's dispatch event carrying the armed-before-dispatch flag.
dispatched = [e for e in prefill_controller_events
              if e.get("event") == "raiden_pool_reshard_senders_dispatched"
              and e.get("req_id") in required_ids]
armed_ids = {
    e.get("req_id")
    for e in dispatched
    if e.get("receiver_armed_before_sender_dispatch") is True
}
dispatch_by_id = {e.get("req_id"): e for e in dispatched}
checks["fa_conv_ssm_receivers_armed"] = bool(required_ids) and armed_ids == required_ids
checks["fa_conv_ssm_senders_dispatched"] = bool(required_ids) and set(dispatch_by_id) == required_ids

ordering_ok = bool(required_ids) and set(dispatch_by_id) == required_ids
tags_ok = ordering_ok
if ordering_ok:
    event = dispatch_by_id[main_req_id]
    ack = event.get("receiver_arm_ack_monotonic_ns")
    dispatch = event.get("sender_dispatch_monotonic_ns")
    if (event.get("receiver_armed_before_sender_dispatch") is not True
            or type(ack) is not int or type(dispatch) is not int or ack > dispatch):
        ordering_ok = False
    event_tags = event.get("transfer_pool_tags")
    if sorted(event_tags or []) != sorted(expected_tags):
        tags_ok = False
checks["all_receivers_armed_before_sender_dispatch"] = ordering_ok
checks["fa_conv_ssm_pool_tags_exact"] = tags_ok
details["receiver_armed_request_ids"] = sorted(x for x in armed_ids if isinstance(x, str))
details["sender_dispatched_request_ids"] = sorted(x for x in dispatch_by_id if isinstance(x, str))

if strict_output:
    choices = response.get("choices") if isinstance(response, dict) else None
    choice = choices[0] if isinstance(choices, list) and choices else {}
    checks["strict_completion_token_ids"] = choice.get("token_ids") == [87] * 8
    checks["strict_completion_text"] = choice.get("text") == "xxxxxxxx"

result = {
    "scope": "raiden_fa_conv_ssm_transport" + ("_and_strict_output" if strict_output else ""),
    "passed": not errors and all(checks.values()),
    "checks": checks,
    "details": details,
    "errors": errors,
    "usage": usage,
}
(root / "validation.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["passed"] else 1)
PY
}

INTERNAL_ROLE=""
ROLE_KIND=""
ROLE_DIR=""
ROLE_PYTHON=""
ROLE_HOST=""
PREFILL_SSH=""
PREFILL_HOST=""
DECODE_HOST=""
MODEL="${DEFAULT_MODEL}"
SERVED_MODEL="${DEFAULT_SERVED_MODEL}"
PREFILL_PYTHON=python3
DECODE_PYTHON=python3
RUN_ROOT="${DEFAULT_RUN_ROOT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EXPECTED_VLLM_VERSION="${DEFAULT_EXPECTED_VLLM_VERSION}"
STARTUP_TIMEOUT="${DEFAULT_STARTUP_TIMEOUT}"
REQUEST_TIMEOUT="${DEFAULT_REQUEST_TIMEOUT}"
PREFILL_API_PORT="${DEFAULT_PREFILL_API_PORT}"
DECODE_API_PORT="${DEFAULT_DECODE_API_PORT}"
PREFILL_CONTROLLER_PORT="${DEFAULT_PREFILL_CONTROLLER_PORT}"
DECODE_CONTROLLER_PORT="${DEFAULT_DECODE_CONTROLLER_PORT}"
KV_PORT="${DEFAULT_KV_PORT}"
TRANSFER_PORT="${DEFAULT_TRANSFER_PORT}"
SIDE_CHANNEL_PORT="${DEFAULT_SIDE_CHANNEL_PORT}"
PREFLIGHT_ONLY=0
STRICT_OUTPUT=0
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1
             -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

while (($#)); do
  case "$1" in
    --prefill-ssh) PREFILL_SSH="${2:?missing value for $1}"; shift 2 ;;
    --prefill-host) PREFILL_HOST="${2:?missing value for $1}"; shift 2 ;;
    --decode-host) DECODE_HOST="${2:?missing value for $1}"; shift 2 ;;
    --model) MODEL="${2:?missing value for $1}"; shift 2 ;;
    --served-model-name) SERVED_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --prefill-python) PREFILL_PYTHON="${2:?missing value for $1}"; shift 2 ;;
    --decode-python) DECODE_PYTHON="${2:?missing value for $1}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?missing value for $1}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for $1}"; shift 2 ;;
    --expected-vllm-version) EXPECTED_VLLM_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="${2:?missing value for $1}"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="${2:?missing value for $1}"; shift 2 ;;
    --prefill-api-port) PREFILL_API_PORT="${2:?missing value for $1}"; shift 2 ;;
    --decode-api-port) DECODE_API_PORT="${2:?missing value for $1}"; shift 2 ;;
    --prefill-controller-port) PREFILL_CONTROLLER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --decode-controller-port) DECODE_CONTROLLER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --kv-port) KV_PORT="${2:?missing value for $1}"; shift 2 ;;
    --transfer-port) TRANSFER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --side-channel-port) SIDE_CHANNEL_PORT="${2:?missing value for $1}"; shift 2 ;;
    --ssh-option) SSH_OPTIONS+=(-o "${2:?missing value for $1}"); shift 2 ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --strict-output) STRICT_OUTPUT=1; shift ;;
    --internal-role) INTERNAL_ROLE="${2:?missing value for $1}"; shift 2 ;;
    --role-kind) ROLE_KIND="${2:?missing value for $1}"; shift 2 ;;
    --role-dir) ROLE_DIR="${2:?missing value for $1}"; shift 2 ;;
    --role-python) ROLE_PYTHON="${2:?missing value for $1}"; shift 2 ;;
    --role-host) ROLE_HOST="${2:?missing value for $1}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

for port_name in PREFILL_API_PORT DECODE_API_PORT PREFILL_CONTROLLER_PORT \
                 DECODE_CONTROLLER_PORT KV_PORT TRANSFER_PORT SIDE_CHANNEL_PORT; do
  validate_port "${port_name}" "${!port_name}"
done
is_positive_integer "${STARTUP_TIMEOUT}" || die "startup timeout must be positive"
is_positive_integer "${REQUEST_TIMEOUT}" || die "request timeout must be positive"
[[ "${RUN_ID}" =~ ^[a-zA-Z0-9_.-]+$ ]] \
  || die "run ID may contain only letters, numbers, dot, underscore, and dash"
[[ "${RUN_ROOT}" == /mnt/disk || "${RUN_ROOT}" == /mnt/disk/* ]] \
  || die "run root must be under /mnt/disk: ${RUN_ROOT}"

if [[ -n "${INTERNAL_ROLE}" ]]; then
  [[ -n "${ROLE_DIR}" ]] || die "--role-dir is required for internal roles"
  case "${INTERNAL_ROLE}" in
    stop)
      stop_role_dir "${ROLE_DIR}"
      exit 0
      ;;
    preflight|serve)
      [[ "${ROLE_KIND}" == prefill || "${ROLE_KIND}" == decode ]] \
        || die "--role-kind must be prefill or decode"
      [[ -n "${ROLE_PYTHON}" ]] || die "--role-python is required"
      [[ -n "${ROLE_HOST}" ]] || die "--role-host is required"
      if [[ "${ROLE_KIND}" == prefill ]]; then
        role_api_port="${PREFILL_API_PORT}"
        role_controller_port="${PREFILL_CONTROLLER_PORT}"
      else
        role_api_port="${DECODE_API_PORT}"
        role_controller_port="${DECODE_CONTROLLER_PORT}"
      fi
      if [[ "${INTERNAL_ROLE}" == preflight ]]; then
        check_role_environment "${ROLE_PYTHON}" "${ROLE_KIND}" "${ROLE_DIR}" \
          "${EXPECTED_VLLM_VERSION}" "${role_api_port}" \
          "${role_controller_port}" "${TRANSFER_PORT}" \
          "${SIDE_CHANNEL_PORT}" "${MODEL}"
      else
        run_role_server "${ROLE_KIND}" "${ROLE_PYTHON}" "${ROLE_HOST}" \
          "${ROLE_DIR}" "${RUN_ID}" "${MODEL}" "${SERVED_MODEL}" \
          "${role_api_port}" "${role_controller_port}" "${KV_PORT}" \
          "${TRANSFER_PORT}" "${SIDE_CHANNEL_PORT}" \
          "${EXPECTED_VLLM_VERSION}"
      fi
      exit 0
      ;;
    *) die "unknown internal role: ${INTERNAL_ROLE}" ;;
  esac
fi

[[ -n "${PREFILL_SSH}" ]] || die "--prefill-ssh is required"
[[ -n "${PREFILL_HOST}" ]] || die "--prefill-host is required"
[[ -n "${DECODE_HOST}" ]] || die "--decode-host is required"
[[ "${PREFILL_HOST}" != localhost && "${PREFILL_HOST}" != 127.* ]] \
  || die "--prefill-host must be routable from the decode host"
[[ "${DECODE_HOST}" != localhost && "${DECODE_HOST}" != 127.* ]] \
  || die "--decode-host must be routable from the prefill host"

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
LOCAL_RUN_DIR="${RUN_ROOT}/${RUN_ID}"
DECODE_ROLE_DIR="${LOCAL_RUN_DIR}/decode"
PREFILL_ROLE_DIR="${RUN_ROOT}/${RUN_ID}/prefill"
RESULT_DIR="${LOCAL_RUN_DIR}/result"
mkdir -p "${DECODE_ROLE_DIR}" "${RESULT_DIR}"

remote_args_common=(
  --role-kind prefill
  --role-dir "${PREFILL_ROLE_DIR}"
  --role-python "${PREFILL_PYTHON}"
  --role-host "${PREFILL_HOST}"
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL}"
  --run-root "${RUN_ROOT}"
  --run-id "${RUN_ID}"
  --expected-vllm-version "${EXPECTED_VLLM_VERSION}"
  --prefill-api-port "${PREFILL_API_PORT}"
  --decode-api-port "${DECODE_API_PORT}"
  --prefill-controller-port "${PREFILL_CONTROLLER_PORT}"
  --decode-controller-port "${DECODE_CONTROLLER_PORT}"
  --kv-port "${KV_PORT}"
  --transfer-port "${TRANSFER_PORT}"
  --side-channel-port "${SIDE_CHANNEL_PORT}"
)
local_args_common=(
  --role-kind decode
  --role-dir "${DECODE_ROLE_DIR}"
  --role-python "${DECODE_PYTHON}"
  --role-host "${DECODE_HOST}"
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL}"
  --run-root "${RUN_ROOT}"
  --run-id "${RUN_ID}"
  --expected-vllm-version "${EXPECTED_VLLM_VERSION}"
  --prefill-api-port "${PREFILL_API_PORT}"
  --decode-api-port "${DECODE_API_PORT}"
  --prefill-controller-port "${PREFILL_CONTROLLER_PORT}"
  --decode-controller-port "${DECODE_CONTROLLER_PORT}"
  --kv-port "${KV_PORT}"
  --transfer-port "${TRANSFER_PORT}"
  --side-channel-port "${SIDE_CHANNEL_PORT}"
)

log "checking installed decode environment"
"${SCRIPT_PATH}" --internal-role preflight "${local_args_common[@]}" \
  >"${RESULT_DIR}/decode.preflight.json" \
  2>"${RESULT_DIR}/decode.preflight.stderr"

log "checking installed prefill environment through ${PREFILL_SSH}"
remote_preflight_command="$(quote_command bash -s -- --internal-role preflight \
  "${remote_args_common[@]}")"
# quote_command has already shell-escaped every argument for the remote shell.
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" "${PREFILL_SSH}" "${remote_preflight_command}" \
  <"${SCRIPT_PATH}" >"${RESULT_DIR}/prefill.preflight.json" \
  2>"${RESULT_DIR}/prefill.preflight.stderr"

if ((PREFLIGHT_ONLY)); then
  log "preflight passed on both hosts; results: ${RESULT_DIR}"
  exit 0
fi

remote_ssh_pid=""
decode_supervisor_pid=""
cleanup_started=0

fetch_remote_file() {
  local remote_path="$1"
  local local_path="$2"
  local command
  command="$(quote_command cat -- "${remote_path}")"
  ssh -n "${SSH_OPTIONS[@]}" "${PREFILL_SSH}" "${command}" >"${local_path}" 2>/dev/null
}

collect_remote_logs() {
  fetch_remote_file "${PREFILL_ROLE_DIR}/server.log" \
    "${RESULT_DIR}/prefill.server.log" || true
  fetch_remote_file "${PREFILL_ROLE_DIR}/controller.log" \
    "${RESULT_DIR}/prefill.controller.log" || true
  fetch_remote_file "${PREFILL_ROLE_DIR}/server.command" \
    "${RESULT_DIR}/prefill.server.command" || true
}

wait_child_bounded() {
  local pid="$1"
  local label="$2"
  local state
  for _ in $(seq 1 100); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${state}" == Z* || -z "${state}" ]]; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.1
  done
  kill -TERM "${pid}" 2>/dev/null || true
  for _ in $(seq 1 50); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${state}" == Z* || -z "${state}" ]]; then
      wait "${pid}" 2>/dev/null || true
      return 0
    fi
    sleep 0.1
  done
  kill -KILL "${pid}" 2>/dev/null || true
  sleep 0.1
  if kill -0 "${pid}" 2>/dev/null; then
    state="$(ps -o stat= -p "${pid}" 2>/dev/null | tr -d '[:space:]')"
    if [[ "${state}" != Z* && -n "${state}" ]]; then
      echo "ERROR: ${label} did not exit during bounded cleanup (pid=${pid})" >&2
      return 1
    fi
  fi
  wait "${pid}" 2>/dev/null || true
  return 0
}

orchestrator_cleanup() {
  local status="$?"
  local teardown_failed=0
  ((cleanup_started == 0)) || exit "${status}"
  cleanup_started=1
  trap - EXIT INT TERM HUP
  set +e

  collect_remote_logs
  if [[ -n "${decode_supervisor_pid}" ]]; then
    if ! "${SCRIPT_PATH}" --internal-role stop --role-dir "${DECODE_ROLE_DIR}" \
        >/dev/null 2>&1; then
      echo "ERROR: local decode role teardown failed" >&2
      teardown_failed=1
    fi
  fi
  if [[ -n "${remote_ssh_pid}" ]]; then
    remote_stop_command="$(quote_command bash -s -- --internal-role stop \
      --role-dir "${PREFILL_ROLE_DIR}")"
    # quote_command has already shell-escaped every argument for the remote shell.
    # shellcheck disable=SC2029
    if ! ssh "${SSH_OPTIONS[@]}" "${PREFILL_SSH}" "${remote_stop_command}" \
        <"${SCRIPT_PATH}" >/dev/null 2>&1; then
      echo "ERROR: remote prefill role teardown failed" >&2
      teardown_failed=1
    fi
  fi
  if [[ -n "${decode_supervisor_pid}" ]] \
      && ! wait_child_bounded "${decode_supervisor_pid}" "decode supervisor"; then
    teardown_failed=1
  fi
  if [[ -n "${remote_ssh_pid}" ]] \
      && ! wait_child_bounded "${remote_ssh_pid}" "prefill SSH client"; then
    teardown_failed=1
  fi
  collect_remote_logs

  if ((teardown_failed != 0 && status == 0)); then
    status=1
  fi

  if ((status != 0)); then
    echo "Live-pair test failed; logs are under ${LOCAL_RUN_DIR}" >&2
    for diagnostic in \
      "${DECODE_ROLE_DIR}/server.log" \
      "${DECODE_ROLE_DIR}/controller.log" \
      "${RESULT_DIR}/prefill.launch.log"; do
      if [[ -s "${diagnostic}" ]]; then
        echo "----- tail: ${diagnostic} -----" >&2
        tail -n 80 "${diagnostic}" >&2 || true
      fi
    done
  else
    log "PASS: live-pair validation is in ${RESULT_DIR}/validation.json"
  fi
  exit "${status}"
}
trap orchestrator_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

log "starting remote PCP8 prefill role"
remote_serve_command="$(quote_command setsid bash -s -- --internal-role serve \
  "${remote_args_common[@]}")"
# quote_command has already shell-escaped every argument for the remote shell.
# shellcheck disable=SC2029
ssh "${SSH_OPTIONS[@]}" "${PREFILL_SSH}" "${remote_serve_command}" \
  <"${SCRIPT_PATH}" >"${RESULT_DIR}/prefill.launch.log" 2>&1 &
remote_ssh_pid=$!

log "starting local DP8 decode role"
setsid "${SCRIPT_PATH}" --internal-role serve "${local_args_common[@]}" \
  >"${RESULT_DIR}/decode.launch.log" 2>&1 &
decode_supervisor_pid=$!

prefill_url="http://$(http_host "${PREFILL_HOST}"):${PREFILL_API_PORT}"
decode_url="http://127.0.0.1:${DECODE_API_PORT}"
deadline=$((SECONDS + STARTUP_TIMEOUT))
prefill_ready=0
decode_ready=0
while ((SECONDS < deadline)); do
  if ((prefill_ready == 0)) && http_ready "${DECODE_PYTHON}" "${prefill_url}/health"; then
    prefill_ready=1
    log "prefill API is healthy"
  fi
  if ((decode_ready == 0)) && http_ready "${DECODE_PYTHON}" "${decode_url}/health"; then
    decode_ready=1
    log "decode API is healthy"
  fi
  ((prefill_ready == 1 && decode_ready == 1)) && break
  kill -0 "${remote_ssh_pid}" 2>/dev/null \
    || die "remote prefill supervisor exited during startup"
  kill -0 "${decode_supervisor_pid}" 2>/dev/null \
    || die "local decode supervisor exited during startup"
  sleep 5
done
((prefill_ready == 1)) || die "prefill API did not become healthy within ${STARTUP_TIMEOUT}s"
((decode_ready == 1)) || die "decode API did not become healthy within ${STARTUP_TIMEOUT}s"

log "sending the 32K-token prefill/decode request"
run_request "${DECODE_PYTHON}" "${prefill_url}" "${decode_url}" \
  "${SERVED_MODEL}" "${RESULT_DIR}" "${REQUEST_TIMEOUT}"

# Structured Raiden events are flushed per line, but allow workers a brief
# interval to report terminal sender completion before collecting remote logs.
sleep 2
cp "${DECODE_ROLE_DIR}/server.log" "${RESULT_DIR}/decode.server.log"
cp "${DECODE_ROLE_DIR}/controller.log" "${RESULT_DIR}/decode.controller.log"
cp "${DECODE_ROLE_DIR}/server.command" "${RESULT_DIR}/decode.server.command"
collect_remote_logs

log "validating FA cache and conv/SSM state transport"
validate_result "${DECODE_PYTHON}" "${RESULT_DIR}" "${STRICT_OUTPUT}"
log "validation passed; stopping both managed roles"
