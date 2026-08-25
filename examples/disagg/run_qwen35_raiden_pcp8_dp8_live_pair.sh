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

# Reproduce the Qwen3.5-397B-A17B-FP8 golden Stage-3 Raiden resharding E2E:
# PCP8/TP1 prefill on one 8-chip host and DP8/TP1 decode on another. This file
# deliberately contains its controllers, accuracy oracle, benchmark client,
# analysis, remote launch, and cleanup logic. It does not import helper scripts
# or add source checkouts to PYTHONPATH.

set -euo pipefail

readonly DEFAULT_MODEL="Qwen/Qwen3.5-397B-A17B-FP8"
readonly DEFAULT_MODEL_REVISION="ea5b4f81096f3901c91dea97f81324302495781d"
readonly DEFAULT_SERVED_MODEL="Qwen3.5-397B-A17B-FP8"
readonly DEFAULT_RUN_ROOT="/mnt/disk/qwen35-raiden-live-pair"
readonly DEFAULT_EXPECTED_VLLM_VERSION="0.26.1rc0"
readonly DEFAULT_STARTUP_TIMEOUT=2400
readonly DEFAULT_REQUEST_TIMEOUT=1800
readonly DEFAULT_MAX_MODEL_LEN=65536
readonly DEFAULT_BLOCK_SIZE=8192
readonly DEFAULT_PREFILL_API_PORT=8400
readonly DEFAULT_DECODE_API_PORT=9400
readonly DEFAULT_PREFILL_CONTROLLER_PORT=27000
readonly DEFAULT_DECODE_CONTROLLER_PORT=28000
readonly DEFAULT_KV_PORT=14579
readonly DEFAULT_TRANSFER_PORT=9100
readonly DEFAULT_SIDE_CHANNEL_PORT=9600

usage() {
  cat <<'EOF'
Run the two-host Qwen3.5-397B-A17B-FP8 Raiden golden live-pair benchmark.

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
                           Default: Qwen/Qwen3.5-397B-A17B-FP8
  --model-revision REV     Revision used with an HF model ID. Default:
                           ea5b4f81096f3901c91dea97f81324302495781d
  --served-model-name NAME Default: Qwen3.5-397B-A17B-FP8
  --prefill-python PATH    Python in the installed prefill environment.
                           Default: python3
  --decode-python PATH     Python in the installed decode environment.
                           Default: python3
  --run-root PATH          Per-run logs/caches; must be under /mnt/disk.
                           Default: /mnt/disk/qwen35-raiden-live-pair
  --run-id ID              Stable run identifier (default: UTC timestamp).
  --startup-timeout SEC    Server compilation/startup timeout (default: 2400).
  --request-timeout SEC    Each HTTP request timeout (default: 1800).
  --max-model-len TOKENS   Server model length (default: 65536).
  --block-size TOKENS      Shared P/D page size (default: 8192). The explicit
                           r20 geometry is required after main's physical-slot
                           sizing change.
  --ssh-option OPTION      Append one OpenSSH -o option; may be repeated.
  --preflight-only         Check both installed environments and free ports.
  --no-performance-gate    Report golden metrics without enforcing the r20
                           acceptance bands. Client and oracle validation
                           remain mandatory.
  --keep-cache             Keep per-run compilation caches after teardown.
  -h, --help               Show this help.

Advanced port overrides:
  --prefill-api-port PORT          Default: 8400
  --decode-api-port PORT           Default: 9400
  --prefill-controller-port PORT   Default: 27000
  --decode-controller-port PORT    Default: 28000
  --kv-port PORT                   vLLM connector metadata port (14579)
  --transfer-port PORT             Raiden control-port base (9100)
  --side-channel-port PORT         Connector side-channel base (9600)

Both machines need an 8-chip TPU host, the exact model snapshot, and matching
installed vllm, vllm-torchtpu, torch-tpu, and Torch tpu-raiden environments.
The routable host values are not necessarily the SSH names. Raiden advertises
dynamic worker endpoints, so the hosts need bidirectional TCP reachability.

By default the script runs the two-request meaningful-data 32K accuracy
oracle, then one primer and 12 measured requests at each of 8192, 32768, and
65534 prompt tokens. The measured requests use deterministic 20-second
inter-arrivals (0.05 qps) and two decode tokens. The connector timing overlay
described in the golden runbook is required for transfer-E2E acceptance
metrics.
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
  local kv_port="$7"
  local transfer_port="$8"
  local side_channel_port="$9"
  local model="${10}"

  command -v setsid >/dev/null 2>&1 || die "setsid is required on the ${role} host"
  setsid --help 2>&1 | grep -q -- '--wait' \
    || die "setsid with --wait support is required on the ${role} host"
  mkdir -p "${role_dir}"
  cd "${role_dir}"
  export PYTHONNOUSERSITE=1
  unset PYTHONPATH
  export HF_HOME=/mnt/disk
  export HUGGINGFACE_HUB_CACHE=/mnt/disk/hub
  export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"
  export TPU_VISIBLE_CHIPS=0,1,2,3,4,5,6,7

  if [[ "${model}" == /* ]]; then
    [[ "${model}" == /mnt/disk/* ]] || die "absolute model paths must be under /mnt/disk: ${model}"
    [[ -d "${model}" ]] || die "model path is absent on the ${role} host: ${model}"
  fi

  "${python_bin}" - "${role}" "${role_dir}" "${expected_vllm_version}" \
    "${api_port}" "${controller_port}" "${kv_port}" "${transfer_port}" \
    "${side_channel_port}" "${model}" <<'PY'
import contextlib
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import pathlib
import socket
import sys

role, role_dir, expected, api_port, controller_port, kv_port, transfer_port, side_port, model = sys.argv[1:]

# TPU/plugin imports emit useful diagnostics. Keep the report on stdout valid
# JSON by routing those diagnostics to the preflight stderr artifact.
with contextlib.redirect_stdout(sys.stderr):
    import torch
    import torch_tpu
    import tpu_sync
    import vllm
    import vllm_torchtpu
    from tpu_sync.api.torch.kv_cache_manager import _torch_impl
    from tpu_sync.rpc.raiden_controller import RaidenController, RaidenControllerServer
    from vllm_torchtpu.distributed.kv_transfer.tpu_connector import TPURaidenConnector

    _tpu_raiden_torch = _torch_impl()
    # This is the runbook's whole-chain smoke test, not just an import test.
    torch.empty((1,), device="tpu").cpu()

actual = vllm.__version__
if actual != expected and not actual.startswith(expected + "+"):
    raise SystemExit(f"vLLM must be {expected} (a +local suffix is allowed), got {actual}")
if importlib.util.find_spec("vllm.entrypoints.openai.api_server") is None:
    raise SystemExit("vLLM OpenAI API server entry point is unavailable")

role_path = pathlib.Path(role_dir)
probe = role_path / f".write-test-{os.getpid()}"
probe.write_text("ok\n", encoding="utf-8")
probe.unlink()

ports = [int(api_port), int(controller_port), int(kv_port)]
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

def package_fingerprint(package, suffixes):
    digest = hashlib.sha256()
    roots = sorted(pathlib.Path(item).resolve() for item in package.__path__)
    files = []
    for root in roots:
        for path in root.rglob("*"):
            if path.is_file() and path.suffix in suffixes:
                files.append((root, path))
    for root, path in sorted(files, key=lambda item: str(item[1])):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(b"\0")
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()

connector_path = pathlib.Path(sys.modules[TPURaidenConnector.__module__].__file__)
connector_text = connector_path.read_text(encoding="utf-8")
premapped = int(os.environ.get("TPU_PREMAPPED_BUFFER_SIZE", "0"))
if premapped < 17179869184:
    raise SystemExit(
        "TPU_PREMAPPED_BUFFER_SIZE must be at least 17179869184 for golden "
        f"transfer performance, got {premapped}")

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
    "vllm_torchtpu_fingerprint": package_fingerprint(vllm_torchtpu, {".py", ".so"}),
    "tpu_raiden_fingerprint": package_fingerprint(tpu_sync, {".py", ".so"}),
    "has_timing_overlay": all(name in connector_text for name in (
        "controller_submit_ms", "reshard_e2e_latency_ms")),
    "tpu_premapped_buffer_size": premapped,
    "controller_import": f"{RaidenController.__module__}.{RaidenController.__name__}",
    "controller_server_import": f"{RaidenControllerServer.__module__}.{RaidenControllerServer.__name__}",
    "connector_import": f"{TPURaidenConnector.__module__}.{TPURaidenConnector.__name__}",
    "ports_checked": ports,
    "hf_home": os.environ["HF_HOME"],
    "model": model,
}, indent=2, sort_keys=True))
PY
}

validate_preflight_pair() {
  local python_bin="$1"
  local decode_report="$2"
  local prefill_report="$3"
  local output_path="$4"

  "${python_bin}" - "${decode_report}" "${prefill_report}" \
    "${output_path}" <<'PY'
import json
import pathlib
import sys

decode_path, prefill_path, output_path = sys.argv[1:]
decode = json.loads(pathlib.Path(decode_path).read_text())
prefill = json.loads(pathlib.Path(prefill_path).read_text())
matching_keys = (
    "python_version",
    "vllm_version",
    "torch_version",
    "torch_tpu_version",
    "tpu_raiden_torch_version",
    "vllm_torchtpu_fingerprint",
    "tpu_raiden_fingerprint",
)
mismatches = {
    key: {"decode": decode.get(key), "prefill": prefill.get(key)}
    for key in matching_keys
    if decode.get(key) != prefill.get(key)
}
timings_ok = (bool(decode.get("has_timing_overlay"))
              and bool(prefill.get("has_timing_overlay")))
result = {
    "passed": not mismatches and timings_ok,
    "matching_keys": list(matching_keys),
    "mismatches": mismatches,
    "timing_overlay_required": True,
    "timing_overlay_available": {
        "decode": bool(decode.get("has_timing_overlay")),
        "prefill": bool(prefill.get("has_timing_overlay")),
    },
}
pathlib.Path(output_path).write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n")
print(json.dumps(result, indent=2, sort_keys=True))
raise SystemExit(0 if result["passed"] else 1)
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
  export VLLM_XLA_CHECK_RECOMPILATION=0

  # Golden r14/r20 runtime profile: keep the latency-hiding scheduler, sparse
  # core MoE, and optimized ragged implementations at their defaults. Explicit
  # unsets prevent a caller's legacy r13 environment from leaking into a run.
  unset LIBTPU_INIT_ARGS USE_MOE_SPARSE_CORE ONEHOT_MOE_PERMUTE_THRESHOLD
  unset TPU_RAGGED_GATHER_REDUCE_IMPL TPU_RAGGED_GATHER_IMPL
  unset DP_SCHED_BATCH_PREFILL_MAX_ADMIT_PER_FLUSH
  export TPU_VLLM_SKIP_DYNAMIC_SMEM_NEGOTIATION_FLAG=1

  # This is the single current layout gate.
  export TPU_VLLM_ENABLE_UNIFIED_BLOCK_POOL=1
  export TPU_USE_RAIDEN_KV_CACHE_MANAGER=1
  export TPU_RAIDEN_QWEN35_ADMISSION=1
  export TPU_KV_RESHARD_TRANSPORT=raiden
  export TPU_RAIDEN_TRANSFER_PARALLELISM=8
  export TPU_GDN_CONV_QK_PAIR_LAYOUT=1

  export XLA_PYTHON_CLIENT_PREALLOCATE=false
  export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
  export TPU_PREMAPPED_BUFFER_SIZE="${TPU_PREMAPPED_BUFFER_SIZE:-17179869184}"

  export TPU_P2P_WAIT_PULL_TIMEOUT=600
  export TPU_KV_SHM_POOL_GB=8
  export TPU_KV_TRANSFER_PORT="${transfer_port}"
  export TPU_SIDE_CHANNEL_PORT="${side_channel_port}"

  unset VLLM_DISABLE_COMPILE_CACHE VLLM_CACHE_ROOT TORCHINDUCTOR_CACHE_DIR
  unset VLLM_XLA_CACHE_PATH TORCH_TPU_INTERNAL_TIER3_COMPILATION_CACHE_ROOT
  unset VLLM_XLA_CACHE_PATH TORCH_TPU_TIER3_COMPILATION_CACHE_ROOT

  export TORCH_TPU_INTERNAL_TIER2_COMPILATION_CACHE=disabled
  export TORCH_TPU_TIER2_COMPILATION_CACHE=disabled

  export TPU_LOG_DIR="${role_dir}/tpu"
  export XDG_CACHE_HOME="${role_dir}/cache"
  export TRITON_CACHE_DIR="${role_dir}/triton"
  export TMPDIR="/tmp/q397-raiden-${namespace}"
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
  local controller_address prefill_compilation_config decode_compilation_config
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
  # Differential-corpus recorder for the C++ reshard-controller port: the
  # controller records raw RPC inputs and worker dispatches.
  # A few hundred KB per request; kept with the run's measurement data.
  export TPU_RAIDEN_PLAN_DUMP_DIR="${role_dir}/plan_dump"
  mkdir -p "${TPU_RAIDEN_PLAN_DUMP_DIR}"
  if [[ "${RESHARD_CONTROLLER_IMPL}" == store ]]; then
    # Zero-sidecar store mode: no controller process at all. Each engine's
    # transfer-rank-0 worker hosts the reshard store in-process; every
    # participant derives per-engine addresses from these bases, staggered
    # by dp_rank (prefill: rank 0 only; decode DP8: ranks 0..7).
    export TPU_RAIDEN_RESHARD_IMPL=store
    export TPU_RAIDEN_ADVERTISE_HOST="${role_host}"
    export TPU_RAIDEN_RESHARD_PORT_BASE="${controller_port}"
    export TPU_RAIDEN_STORE_DISPATCH_PORT_BASE="$((controller_port + 100))"
  else
    export TPU_RAIDEN_RESHARD_IMPL=controller
    if [[ "${RESHARD_CONTROLLER_IMPL}" == cpp ]]; then
      [[ -x "${RESHARD_SIDECAR_BIN}" ]] \
        || die "--reshard-controller-impl cpp requires an executable --reshard-sidecar-bin (got: ${RESHARD_SIDECAR_BIN})"
      setsid "${RESHARD_SIDECAR_BIN}" --port "${controller_port}" \
        --advertise-host "${role_host}" --request-registry-ttl-s 600 \
        --ready-file "${role_dir}/controller.ready.json" \
        >"${role_dir}/controller.log" 2>&1 &
    else
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

from tpu_sync.rpc.raiden_controller import RaidenController, RaidenControllerServer

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
    fi
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
  fi

  configure_role_environment "${role}" "${role_host}" "${controller_port}" \
    "${run_id}" "${role_dir}" "${transfer_port}" "${side_channel_port}"
  controller_address="$(host_port "${role_host}" "${controller_port}")"
  [[ "${TPU_RAIDEN_CONTROLLER_ADDRESS}" == "${controller_address}" ]] \
    || die "internal controller address mismatch"

  # These shapes mirror tpu_benchmark_daily. PCP8 consumes the 32K prefill
  # budget as 4K tokens per rank; decode compiles its real small-step shapes.
  prefill_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[4096],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
  decode_compilation_config='{"backend":"vllm_torchtpu.compilation.tpu_compiler.TpuCompilerAdaptor","compile_sizes":[8,16,32,4352,4384],"inductor_compile_config":{"enable_auto_functionalized_v2":false,"size_asserts":false,"alignment_asserts":false,"scalar_asserts":false}}'
  common_args=(
    --model "${model}"
    --served-model-name "${served_model}"
    --trust-remote-code
    --seed 42
    --max-model-len "${MAX_MODEL_LEN}"
    --block-size "${BLOCK_SIZE}"
    --enable-expert-parallel
    --disable-custom-all-reduce
    --gpu-memory-utilization 0.9
    --kv-cache-dtype fp8
    --language-model-only
    --no-disable-hybrid-kv-cache-manager
    --attention-backend CUSTOM
    --mamba-cache-mode align
    --enable-prompt-tokens-details
    --no-enable-log-requests
    --no-enable-prefix-caching
    --no-async-scheduling
    --tensor-parallel-size 1
  )
  if [[ "${model}" != /* && -n "${MODEL_REVISION}" ]]; then
    common_args+=(--revision "${MODEL_REVISION}")
  fi

  if [[ "${role}" == prefill ]]; then
    role_args=(
      --host 0.0.0.0
      --port "${api_port}"
      --max-num-batched-tokens 32768
      --long-prefill-token-threshold 32768
      --max-num-seqs 8
      --compilation-config "${prefill_compilation_config}"
      --num-gpu-blocks-override 64
      --prefill-context-parallel-size 8
      --cp-kv-cache-interleave-size 256
      --kv-transfer-config '{"kv_connector":"TPURaidenConnector","kv_connector_module_path":"vllm_torchtpu.distributed.kv_transfer.tpu_connector","kv_role":"kv_producer","kv_port":'"${kv_port}"'}'
    )
  elif [[ "${role}" == decode ]]; then
    role_args=(
      --host 127.0.0.1
      --port "${api_port}"
      --max-num-batched-tokens 4384
      --max-num-seqs 32
      --compilation-config "${decode_compilation_config}"
      --num-gpu-blocks-override 64
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
    if ((controller_owned == 1)) \
        && ! pid_matches_start_ticks "${controller_pid}" "${controller_ticks}"; then
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

run_meaningful_oracle() {
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
import random
import sys
import time
import urllib.error
import urllib.request
import uuid

prefill_url, decode_url, model, output_dir, timeout = sys.argv[1:]
output = pathlib.Path(output_dir)
output.mkdir(parents=True, exist_ok=True)
timeout = float(timeout)

sentences = [
    "The harbor master logged every vessel that entered the bay before dawn.",
    "A slow rain settled over the terraced fields and softened the clay paths.",
    "Engineers rerouted the aqueduct after the eastern span showed hairline cracks.",
    "The archivist catalogued letters from the expedition of the previous winter.",
    "Merchants argued about grain tariffs under the arcades of the old exchange.",
    "A lighthouse keeper recorded wind speeds in a leather-bound ledger.",
    "The observatory's brass dome creaked as it rotated toward the meridian.",
    "Cartographers disputed the elevation of the northern pass for a decade.",
    "The mill wheel turned unevenly whenever the sluice gate silted up.",
    "Botanists pressed alpine flowers between sheets of absorbent paper.",
    "The night train carried mail sacks, timber samples, and two violins.",
    "Masons squared the granite blocks with chalk lines and iron mallets.",
    "A courier changed horses twice before reaching the river crossing.",
    "The foundry poured bell bronze only when the humidity stayed low.",
    "Surveyors drove numbered stakes along the proposed canal alignment.",
    "The librarian rebound the damaged atlas with linen thread and paste.",
    "Fishermen mended their nets on the seawall while the tide ran out.",
    "An apprentice glazier sorted panes by thickness in the workshop loft.",
    "The telegraph office kept a duplicate of every message for ninety days.",
    "Vintners rolled oak casks into the cellar before the first frost.",
]


def post(url, payload, request_id):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"{url} returned HTTP {exc.code}: {body[:8000]!r}") from exc


def render_tokens(text, request_id):
    rendered = post(
        f"{prefill_url.rstrip('/')}/v1/completions/render",
        {"model": model, "prompt": text, "add_special_tokens": False,
         "max_tokens": 1},
        request_id,
    )
    if not isinstance(rendered, list) or not rendered:
        raise RuntimeError("render response was not a non-empty list")
    token_ids = rendered[0].get("token_ids")
    if not isinstance(token_ids, list) or not all(type(item) is int for item in token_ids):
        raise RuntimeError("render response did not contain integer token_ids")
    return token_ids


def filler(rng, count):
    return " ".join(rng.choice(sentences) for _ in range(count))


def find_subsequence(haystack, needle):
    for index in range(len(haystack) - len(needle) + 1):
        if haystack[index:index + len(needle)] == needle:
            return index
    return -1


def build_prompt(target_tokens, needle_value, seed, request_id):
    rng = random.Random(seed)
    needle_text = (
        " The vault access code for project Aurora is "
        f"{needle_value}. Keep this code confidential. "
    )
    question_text = (
        "\n\nQuestion: What is the vault access code for project Aurora?\n"
        "Answer: The vault access code for project Aurora is"
    )
    question_tokens = render_tokens(question_text, request_id)
    body_budget = target_tokens - len(question_tokens)
    before_count = max(1, int(body_budget * 0.40 / 13))
    after_count = max(1, int(body_budget * 0.85 / 13))
    for _ in range(6):
        body_text = (
            filler(rng, before_count) + needle_text + filler(rng, after_count)
        )
        body_tokens = render_tokens(body_text, request_id)
        if len(body_tokens) >= body_budget:
            break
        after_count = int(after_count * 1.3) + 8
    else:
        raise RuntimeError("could not reach the oracle token budget")
    kept_body = body_tokens[:body_budget]
    needle_tokens = render_tokens(needle_text.strip(), request_id)
    position = find_subsequence(kept_body, needle_tokens)
    if position < 0:
        position = find_subsequence(kept_body, needle_tokens[2:-2])
    if position <= 0:
        raise RuntimeError("needle was not preserved in the exact-token prompt")
    prompt = kept_body + question_tokens
    if len(prompt) != target_tokens:
        raise RuntimeError(
            f"oracle prompt has {len(prompt)} tokens, expected {target_tokens}")
    return prompt, position


records = []
for index in range(2):
    request_id = f"q397-meaningful-{index}-{uuid.uuid4().hex[:12]}"
    needle_value = str(500000 + index * 119101 + 38291)
    prompt, needle_position = build_prompt(
        32768, needle_value, 774001 + index, request_id + "-render")
    base = {
        "add_special_tokens": False,
        "ignore_eos": True,
        "max_tokens": 32,
        "model": model,
        "n": 1,
        "prompt": prompt,
        "return_token_ids": True,
        "seed": 0,
        "stream": False,
        "temperature": 0.0,
    }

    # The decode-local control deliberately runs before transferred state has
    # touched the consumer pools. The producer must take exactly one decode
    # step so its live mamba slot remains at the h(N-1) transfer contract.
    start = time.time()
    control = post(
        f"{decode_url.rstrip('/')}/v1/completions", base,
        request_id + "-local-control")
    control_done = time.time()
    producer = post(
        f"{prefill_url.rstrip('/')}/v1/completions",
        dict(base, max_tokens=1), request_id)
    producer_done = time.time()
    kv_params = producer.get("kv_transfer_params")
    if not isinstance(kv_params, dict) or not kv_params:
        raise RuntimeError("oracle producer returned no kv_transfer_params")
    decode = post(
        f"{decode_url.rstrip('/')}/v1/completions",
        dict(base, kv_transfer_params=kv_params), request_id)
    decode_done = time.time()

    producer_choice = producer["choices"][0]
    decode_choice = decode["choices"][0]
    control_choice = control["choices"][0]
    decode_usage = decode.get("usage", {})
    cached_tokens = decode_usage.get("prompt_tokens_details", {}).get(
        "cached_tokens")
    record = {
        "request_id": request_id,
        "prompt_tokens": len(prompt),
        "needle_value": needle_value,
        "needle_token_pos": needle_position,
        "kv_num_tokens": kv_params.get("num_tokens"),
        "decode_cached_tokens": cached_tokens,
        "prefill_token_ids": producer_choice.get("token_ids", []),
        "decode_token_ids": decode_choice.get("token_ids", []),
        "control_token_ids": control_choice.get("token_ids", []),
        "prefill_text": producer_choice.get("text", ""),
        "decode_text": decode_choice.get("text", ""),
        "control_text": control_choice.get("text", ""),
        "control_s": round(control_done - start, 3),
        "prefill_s": round(producer_done - control_done, 3),
        "decode_s": round(decode_done - producer_done, 3),
    }
    record.update({
        "match_decode_vs_control":
            record["decode_token_ids"] == record["control_token_ids"],
        "match_decode_vs_prefill":
            record["decode_token_ids"][:len(record["prefill_token_ids"])]
            == record["prefill_token_ids"],
        "match_first_token_vs_prefill": bool(
            record["prefill_token_ids"] and record["decode_token_ids"]
            and record["prefill_token_ids"][0] == record["decode_token_ids"][0]),
        "needle_in_control": needle_value in record["control_text"],
        "needle_in_prefill": needle_value in record["prefill_text"],
        "needle_in_decode": needle_value in record["decode_text"],
        # Per the runbook, token equality is diagnostic; meaningful retrieval
        # through the transferred state is the accuracy gate.
        "oracle_passed": needle_value in record["decode_text"],
    })
    records.append(record)
    (output / "meaningful_oracle.json").write_text(
        json.dumps({"passed": all(r["oracle_passed"] for r in records),
                    "requests": records}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: record[key] for key in (
        "request_id", "needle_value", "kv_num_tokens", "decode_cached_tokens",
        "match_decode_vs_control", "match_first_token_vs_prefill",
        "needle_in_control", "needle_in_decode", "oracle_passed")},
        indent=2, sort_keys=True), flush=True)

passed = len(records) == 2 and all(record["oracle_passed"] for record in records)
(output / "meaningful_oracle.json").write_text(
    json.dumps({"passed": passed, "requests": records},
               indent=2, sort_keys=True) + "\n")
print(f"MEANINGFUL_ORACLE {'PASSED' if passed else 'FAILED'} "
      f"({sum(record['oracle_passed'] for record in records)}/2)", flush=True)
raise SystemExit(0 if passed else 1)
PY
}

run_benchmark_phase() {
  local python_bin="$1"
  local prefill_url="$2"
  local decode_url="$3"
  local served_model="$4"
  local output_dir="$5"
  local num_requests="$6"
  local request_rate="$7"
  local prompt_tokens="$8"
  local seed_base="$9"
  local tag="${10}"
  local timeout="${11}"
  local max_model_len="${12}"

  "${python_bin}" - "${prefill_url}" "${decode_url}" "${served_model}" \
    "${output_dir}" "${num_requests}" "${request_rate}" \
    "${prompt_tokens}" "${seed_base}" "${tag}" "${timeout}" \
    "${max_model_len}" <<'PY'
from __future__ import annotations

import concurrent.futures
import datetime
import json
import pathlib
import random
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid

(prefill_url, decode_url, model, output_dir, num_requests, request_rate,
 prompt_tokens, seed_base, tag, timeout, max_model_len) = sys.argv[1:]
output = pathlib.Path(output_dir)
num_requests = int(num_requests)
request_rate = float(request_rate)
prompt_tokens = int(prompt_tokens)
seed_base = int(seed_base)
timeout = float(timeout)
max_model_len = int(max_model_len)
max_tokens = 2
if num_requests <= 0 or request_rate <= 0:
    raise SystemExit("request count and rate must be positive")
if prompt_tokens + max_tokens > max_model_len:
    raise SystemExit("prompt plus decode length exceeds server max model length")

output.mkdir(parents=True, exist_ok=True)
records_path = output / "requests.jsonl"
records_path.write_text("", encoding="utf-8")
interval_s = 1.0 / request_rate
epoch_ns = time.perf_counter_ns()
output_lock = threading.Lock()

(output / "config.json").write_text(json.dumps({
    "schema_version": 3,
    "arrival_distribution": "uniform_deterministic",
    "prompt_construction": "unique_random_token_ids_no_common_prefix",
    "token_id_range": [256, 100000],
    "seed_base": seed_base,
    "tag": tag,
    "request_rate_qps": request_rate,
    "inter_arrival_s": interval_s,
    "num_requests": num_requests,
    "prompt_tokens": prompt_tokens,
    "transferred_prefix_tokens": prompt_tokens - 1,
    "max_decode_tokens": max_tokens,
    "server_max_model_len": max_model_len,
    "model": model,
    "prefill_url": prefill_url,
    "decode_url": decode_url,
    "run_start_utc": datetime.datetime.now(
        datetime.timezone.utc).isoformat(),
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def relative_ms(now_ns):
    return (now_ns - epoch_ns) / 1_000_000


def build_prompt(sequence):
    rng = random.Random(seed_base + sequence)
    token_ids = [rng.randrange(256, 100000) for _ in range(prompt_tokens)]
    token_ids[0] = 257 + sequence
    return token_ids


def post_json(url, payload, request_id):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    start_ns = time.perf_counter_ns()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"{url} returned HTTP {exc.code}: {body[:8000]!r}") from exc
    return json.loads(body), (time.perf_counter_ns() - start_ns) / 1_000_000


def post_stream(url, payload, request_id):
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "X-Request-Id": request_id},
        method="POST",
    )
    start_ns = time.perf_counter_ns()
    headers_ns = None
    first_data_ns = None
    first_token_ns = None
    done_ns = None
    chunks = []
    text_parts = []
    token_ids = []
    usage = None
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            headers_ns = time.perf_counter_ns()
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                now_ns = time.perf_counter_ns()
                if first_data_ns is None:
                    first_data_ns = now_ns
                if data == "[DONE]":
                    saw_done = True
                    done_ns = now_ns
                    break
                value = json.loads(data)
                if not isinstance(value, dict):
                    continue
                chunks.append(value)
                if isinstance(value.get("usage"), dict):
                    usage = value["usage"]
                choices = value.get("choices")
                choice = (choices[0] if isinstance(choices, list) and choices
                          and isinstance(choices[0], dict) else {})
                text = choice.get("text")
                ids = choice.get("token_ids")
                if first_token_ns is None and (
                        isinstance(text, str) and text
                        or isinstance(ids, list) and ids):
                    first_token_ns = now_ns
                if isinstance(text, str):
                    text_parts.append(text)
                if isinstance(ids, list):
                    token_ids.extend(item for item in ids if type(item) is int)
    except urllib.error.HTTPError as exc:
        body = exc.read()
        raise RuntimeError(
            f"{url} returned HTTP {exc.code}: {body[:8000]!r}") from exc
    end_ns = time.perf_counter_ns()
    if first_token_ns is None:
        first_token_ns = first_data_ns
    return {
        "decode_http_headers_ms": (
            (headers_ns - start_ns) / 1_000_000 if headers_ns else None),
        "decode_ttft_ms": (
            (first_token_ns - start_ns) / 1_000_000 if first_token_ns else None),
        "decode_http_total_ms": (end_ns - start_ns) / 1_000_000,
        "decode_stream_done_ms": (
            (done_ns - start_ns) / 1_000_000 if done_ns else None),
        "decode_stream_saw_done": saw_done,
        "decode_chunk_count": len(chunks),
        "decode_completion_text": "".join(text_parts),
        "decode_completion_token_ids": token_ids,
        "decode_usage": usage,
    }


def run_one(sequence):
    scheduled_ns = epoch_ns + int(sequence * interval_s * 1e9)
    remaining_s = (scheduled_ns - time.perf_counter_ns()) / 1e9
    if remaining_s > 0:
        time.sleep(remaining_s)
    launched_ns = time.perf_counter_ns()
    request_id = f"q397-noprefix-{tag}-{sequence:03d}-{uuid.uuid4().hex[:12]}"
    record = {
        "schema_version": 3,
        "sequence": sequence,
        "tag": tag,
        "request_id": request_id,
        "prompt_seed": seed_base + sequence,
        "scheduled_start_ms": relative_ms(scheduled_ns),
        "actual_start_ms": relative_ms(launched_ns),
        "schedule_lag_ms": (launched_ns - scheduled_ns) / 1_000_000,
        "prompt_tokens": prompt_tokens,
        "max_tokens": max_tokens,
        "error": None,
    }
    try:
        payload = {
            "add_special_tokens": False,
            "ignore_eos": True,
            "max_tokens": max_tokens,
            "model": model,
            "n": 1,
            "prompt": build_prompt(sequence),
            "return_token_ids": True,
            "seed": sequence,
            "stream": False,
            "temperature": 0.0,
        }
        rendered, render_ms = post_json(
            f"{prefill_url.rstrip('/')}/v1/completions/render",
            payload, request_id)
        rendered_prompt = (rendered[0].get("token_ids")
                           if isinstance(rendered, list) and rendered
                           and isinstance(rendered[0], dict) else None)
        if not isinstance(rendered_prompt, list):
            raise RuntimeError("render response did not contain token_ids")
        if len(rendered_prompt) != prompt_tokens:
            raise RuntimeError(
                f"render changed prompt length to {len(rendered_prompt)}")
        routed = dict(payload, prompt=rendered_prompt)
        # Producer max_tokens=1 is required by the live-slot mamba-state
        # contract. Top-5 is diagnostic and matches the recorded r20 client.
        producer_payload = dict(routed, max_tokens=1, logprobs=5)
        producer, prefill_ms = post_json(
            f"{prefill_url.rstrip('/')}/v1/completions",
            producer_payload, request_id)
        if not isinstance(producer, dict):
            raise RuntimeError("producer response was not an object")
        choices = producer.get("choices")
        choice = (choices[0] if isinstance(choices, list) and choices
                  and isinstance(choices[0], dict) else {})
        kv_params = producer.get("kv_transfer_params")
        if not isinstance(kv_params, dict) or not kv_params:
            raise RuntimeError("producer response had no kv_transfer_params")
        routed.update({
            "kv_transfer_params": kv_params,
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        decode_start_ns = time.perf_counter_ns()
        stream_result = post_stream(
            f"{decode_url.rstrip('/')}/v1/completions",
            routed, request_id)
        record.update({
            "render_http_ms": render_ms,
            "prefill_http_ms": prefill_ms,
            "prefill_completion_token_ids": choice.get("token_ids"),
            "prefill_completion_text": choice.get("text"),
            "decode_start_ms": relative_ms(decode_start_ns),
            "uuid": kv_params.get("uuid"),
            "source_req_id": kv_params.get("req_id"),
            **stream_result,
        })
        usage = stream_result.get("decode_usage")
        details = (usage.get("prompt_tokens_details")
                   if isinstance(usage, dict)
                   and isinstance(usage.get("prompt_tokens_details"), dict)
                   else {})
        validation = {
            "stream_done": stream_result["decode_stream_saw_done"],
            "prompt_tokens_exact": (
                isinstance(usage, dict)
                and usage.get("prompt_tokens") == prompt_tokens),
            "cached_tokens_exact": details.get("cached_tokens") == prompt_tokens - 1,
            "completion_tokens_exact": (
                isinstance(usage, dict)
                and usage.get("completion_tokens") == max_tokens),
            "has_uuid": type(kv_params.get("uuid")) is int,
            "has_source_req_id": isinstance(kv_params.get("req_id"), str),
        }
        producer_ids = choice.get("token_ids")
        decode_ids = stream_result["decode_completion_token_ids"]
        logprobs = choice.get("logprobs") or {}
        top_logprobs = logprobs.get("top_logprobs") or []
        top5 = (list(top_logprobs[0].keys())
                if top_logprobs and isinstance(top_logprobs[0], dict) else [])
        exact = bool(producer_ids and decode_ids
                     and producer_ids[0] == decode_ids[0])
        record["client_validation"] = validation
        record["semantic_oracle"] = {
            "first_decode_token_matches_prefill": exact,
            "first_decode_token_in_prefill_top5": exact or any(
                stream_result["decode_completion_text"].startswith(item)
                for item in top5 if item),
            "prefill_top5": top5,
        }
        if not all(validation.values()):
            raise RuntimeError(
                "client validation failed: " + json.dumps(validation,
                                                           sort_keys=True))
    except Exception as exc:  # pylint: disable=broad-except
        record["error"] = {"type": type(exc).__name__, "message": str(exc)}
    record["client_total_ms"] = (
        time.perf_counter_ns() - launched_ns) / 1_000_000
    with output_lock:
        with records_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True,
                                    separators=(",", ":")) + "\n")
            stream.flush()
        print(json.dumps(record, sort_keys=True), flush=True)
    return record


with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
    records = list(executor.map(run_one, range(num_requests)))
records.sort(key=lambda item: item["sequence"])
failed = [record for record in records if record["error"] is not None]
summary = {
    "schema_version": 3,
    "tag": tag,
    "num_requests": len(records),
    "num_succeeded": len(records) - len(failed),
    "num_failed": len(failed),
    "failed_sequences": [record["sequence"] for record in failed],
    "run_elapsed_s": (time.perf_counter_ns() - epoch_ns) / 1e9,
    "actual_inter_arrival_ms": [
        records[index]["actual_start_ms"] - records[index - 1]["actual_start_ms"]
        for index in range(1, len(records))
    ],
}
(output / "client_summary.json").write_text(
    json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary, sort_keys=True), flush=True)
raise SystemExit(1 if failed else 0)
PY
}

analyze_golden_benchmark() {
  local python_bin="$1"
  local run_dir="$2"
  local performance_gate="$3"

  "${python_bin}" - "${run_dir}" "${performance_gate}" <<'PY'
from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

run = pathlib.Path(sys.argv[1])
performance_gate = sys.argv[2] == "1"


def events(path):
    decoder = json.JSONDecoder()
    result = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        cursor = 0
        while True:
            start = line.find("{", cursor)
            if start < 0:
                break
            try:
                value, length = decoder.raw_decode(line[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            cursor = start + length
            if isinstance(value, dict) and isinstance(value.get("event"), str):
                result.append(value)
    return result


def percentile(values, quantile):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1,
                max(0, round(quantile * (len(ordered) - 1))))
    return ordered[index]


def mean(values):
    return statistics.mean(values) if values else None


def in_band(value, reference, fraction=0.10):
    return (value is not None
            and reference * (1 - fraction) <= value <= reference * (1 + fraction))


decode_events = events(run / "result" / "decode.server.log")
submitted = {
    event["uuid"]: event for event in decode_events
    if event.get("event") == "raiden_stage3_transfer_submitted"
    and "uuid" in event
}
completed = {
    event["uuid"]: event for event in decode_events
    if event.get("event") == "raiden_stage3_receiver_complete"
    and "uuid" in event
}
payload_bytes = {
    8192: 317875200,
    32768: 695362560,
    65534: 1198663680,
}
rows = {}
for length in (8192, 32768, 65534):
    path = run / "client" / f"measured-{length}" / "requests.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()
               if line.strip()]
    valid = [record for record in records
             if record.get("error") is None
             and all(record.get("client_validation", {}).values())]
    prefill = [record["prefill_http_ms"] for record in valid]
    decode_ttft = [record["decode_ttft_ms"] for record in valid
                   if record.get("decode_ttft_ms") is not None]
    decode_total = [record["decode_http_total_ms"] for record in valid]
    second_step = [
        record["decode_stream_done_ms"] - record["decode_ttft_ms"]
        for record in valid
        if record.get("decode_stream_done_ms") is not None
        and record.get("decode_ttft_ms") is not None
    ]
    server_ttft = [
        record["prefill_http_ms"] + record["decode_ttft_ms"]
        for record in valid if record.get("decode_ttft_ms") is not None
    ]
    transfer = []
    controller_submit = []
    for record in valid:
        request_uuid = record.get("uuid")
        complete = completed.get(request_uuid)
        submit = submitted.get(request_uuid)
        if complete and complete.get("reshard_e2e_latency_ms") is not None:
            transfer.append(float(complete["reshard_e2e_latency_ms"]))
        if submit and submit.get("controller_submit_ms") is not None:
            controller_submit.append(float(submit["controller_submit_ms"]))
    transfer_p50 = percentile(transfer, 0.5)
    rows[str(length)] = {
        "num_requests": len(records),
        "num_valid": len(valid),
        "prefill_mean_ms": mean(prefill),
        "decode_ttft_p50_ms": percentile(decode_ttft, 0.5),
        "decode_ttft_mean_ms": mean(decode_ttft),
        "second_token_step_mean_ms": mean(second_step),
        "server_ttft_p50_ms": percentile(server_ttft, 0.5),
        "decode_e2el_p50_ms": percentile(decode_total, 0.5),
        "transfer_e2e_p50_ms": transfer_p50,
        "transfer_e2e_p90_ms": percentile(transfer, 0.9),
        "transfer_event_count": len(transfer),
        "controller_submit_p50_ms": percentile(controller_submit, 0.5),
        "controller_submit_event_count": len(controller_submit),
        "payload_bytes": payload_bytes[length],
        "payload_bandwidth_gbps": (
            payload_bytes[length] / transfer_p50 / 1_000_000
            if transfer_p50 else None),
    }

client_checks = {
    f"length_{length}_has_12_valid_requests":
        rows[str(length)]["num_requests"] == 12
        and rows[str(length)]["num_valid"] == 12
    for length in (8192, 32768, 65534)
}
timing_checks = {
    f"length_{length}_has_12_transfer_timings":
        rows[str(length)]["transfer_event_count"] == 12
        and rows[str(length)]["controller_submit_event_count"] == 12
    for length in (8192, 32768, 65534)
}
performance_checks = {
    "8192_transfer_p50_within_10pct_r20":
        in_band(rows["8192"]["transfer_e2e_p50_ms"], 43.0),
    "32768_transfer_p50_within_10pct_r20":
        in_band(rows["32768"]["transfer_e2e_p50_ms"], 59.1),
    "65534_transfer_p50_within_10pct_r20":
        in_band(rows["65534"]["transfer_e2e_p50_ms"], 90.2),
    "65534_prefill_mean_within_10pct_r20":
        in_band(rows["65534"]["prefill_mean_ms"], 1514.0),
    "65534_server_ttft_p50_within_10pct_r20":
        in_band(rows["65534"]["server_ttft_p50_ms"], 1700.0),
    "65534_second_token_step_below_30ms":
        rows["65534"]["second_token_step_mean_ms"] is not None
        and rows["65534"]["second_token_step_mean_ms"] < 30.0,
}
for length in (8192, 32768, 65534):
    value = rows[str(length)]["controller_submit_p50_ms"]
    performance_checks[f"{length}_controller_submit_in_expected_range"] = (
        value is not None and 25.0 <= value <= 50.0)

gated_checks = {**client_checks, **timing_checks}
if performance_gate:
    gated_checks.update(performance_checks)
result = {
    "schema_version": 1,
    "passed": all(gated_checks.values()),
    "performance_gate_enabled": performance_gate,
    "checks": gated_checks,
    "performance_checks": performance_checks,
    "rows": rows,
}
(run / "result" / "benchmark_summary.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = []
for length in (8192, 32768, 65534):
    row = rows[str(length)]
    def show(value, digits=1):
        return "n/a" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"
    line = (
        f"len={length}: n={row['num_valid']}/12 | "
        f"prefill mean={show(row['prefill_mean_ms'], 0)} ms | "
        f"decode TTFT p50={show(row['decode_ttft_p50_ms'], 0)} ms | "
        f"2nd-token mean={show(row['second_token_step_mean_ms'])} ms | "
        f"server TTFT p50={show(row['server_ttft_p50_ms'], 0)} ms | "
        f"transfer p50={show(row['transfer_e2e_p50_ms'])} "
        f"p90={show(row['transfer_e2e_p90_ms'])} ms | "
        f"ctrl submit p50={show(row['controller_submit_p50_ms'])} ms | "
        f"BW={show(row['payload_bandwidth_gbps'], 2)} GB/s")
    lines.append(line)
    print(line)
lines.append(f"GOLDEN_BENCHMARK {'PASSED' if result['passed'] else 'FAILED'}")
print(lines[-1])
(run / "result" / "benchmark_summary.txt").write_text(
    "\n".join(lines) + "\n", encoding="utf-8")
raise SystemExit(0 if result["passed"] else 1)
PY
}

INTERNAL_ROLE=""
ROLE_KIND=""
ROLE_DIR=""
ROLE_PYTHON=""
ROLE_HOST=""
PREFILL_SSH=""
# Reshard-controller implementation switch: the
# role launcher selects the reshard hosting per host. "store" = the
# zero-sidecar mode (engine rank-0 workers host in-process stores);
# "python" = the inline
# RaidenController below (default, byte-identical to the golden lineage);
# "cpp" = the bazel-built reshard_sidecar binary at RESHARD_SIDECAR_BIN.
RESHARD_CONTROLLER_IMPL="${RESHARD_CONTROLLER_IMPL:-python}"
RESHARD_SIDECAR_BIN="${RESHARD_SIDECAR_BIN:-}"
# Perf-only escape hatch: run the benchmark phases on a pair whose accuracy
# oracle is known-failing (e.g. while a data-plane accuracy regression is
# being root-caused upstream). The summary carries an explicit marker; a
# skipped-oracle run is NEVER a golden accuracy verification.
SKIP_ACCURACY_ORACLE=0
PREFILL_HOST=""
DECODE_HOST=""
MODEL="${DEFAULT_MODEL}"
MODEL_REVISION="${DEFAULT_MODEL_REVISION}"
SERVED_MODEL="${DEFAULT_SERVED_MODEL}"
PREFILL_PYTHON=python3
DECODE_PYTHON=python3
RUN_ROOT="${DEFAULT_RUN_ROOT}"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
EXPECTED_VLLM_VERSION="${DEFAULT_EXPECTED_VLLM_VERSION}"
STARTUP_TIMEOUT="${DEFAULT_STARTUP_TIMEOUT}"
REQUEST_TIMEOUT="${DEFAULT_REQUEST_TIMEOUT}"
MAX_MODEL_LEN="${DEFAULT_MAX_MODEL_LEN}"
BLOCK_SIZE="${DEFAULT_BLOCK_SIZE}"
PREFILL_API_PORT="${DEFAULT_PREFILL_API_PORT}"
DECODE_API_PORT="${DEFAULT_DECODE_API_PORT}"
PREFILL_CONTROLLER_PORT="${DEFAULT_PREFILL_CONTROLLER_PORT}"
DECODE_CONTROLLER_PORT="${DEFAULT_DECODE_CONTROLLER_PORT}"
KV_PORT="${DEFAULT_KV_PORT}"
TRANSFER_PORT="${DEFAULT_TRANSFER_PORT}"
SIDE_CHANNEL_PORT="${DEFAULT_SIDE_CHANNEL_PORT}"
PREFLIGHT_ONLY=0
PERFORMANCE_GATE=1
KEEP_CACHE=0
SSH_OPTIONS=(-o BatchMode=yes -o ConnectTimeout=10 -o ConnectionAttempts=1
             -o ServerAliveInterval=30 -o ServerAliveCountMax=6)

while (($#)); do
  case "$1" in
    --prefill-ssh) PREFILL_SSH="${2:?missing value for $1}"; shift 2 ;;
    --prefill-host) PREFILL_HOST="${2:?missing value for $1}"; shift 2 ;;
    --decode-host) DECODE_HOST="${2:?missing value for $1}"; shift 2 ;;
    --model) MODEL="${2:?missing value for $1}"; shift 2 ;;
    --model-revision) MODEL_REVISION="${2:?missing value for $1}"; shift 2 ;;
    --served-model-name) SERVED_MODEL="${2:?missing value for $1}"; shift 2 ;;
    --prefill-python) PREFILL_PYTHON="${2:?missing value for $1}"; shift 2 ;;
    --decode-python) DECODE_PYTHON="${2:?missing value for $1}"; shift 2 ;;
    --run-root) RUN_ROOT="${2:?missing value for $1}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for $1}"; shift 2 ;;
    --expected-vllm-version) EXPECTED_VLLM_VERSION="${2:?missing value for $1}"; shift 2 ;;
    --startup-timeout) STARTUP_TIMEOUT="${2:?missing value for $1}"; shift 2 ;;
    --request-timeout) REQUEST_TIMEOUT="${2:?missing value for $1}"; shift 2 ;;
    --max-model-len) MAX_MODEL_LEN="${2:?missing value for $1}"; shift 2 ;;
    --block-size) BLOCK_SIZE="${2:?missing value for $1}"; shift 2 ;;
    --prefill-api-port) PREFILL_API_PORT="${2:?missing value for $1}"; shift 2 ;;
    --decode-api-port) DECODE_API_PORT="${2:?missing value for $1}"; shift 2 ;;
    --prefill-controller-port) PREFILL_CONTROLLER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --decode-controller-port) DECODE_CONTROLLER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --kv-port) KV_PORT="${2:?missing value for $1}"; shift 2 ;;
    --transfer-port) TRANSFER_PORT="${2:?missing value for $1}"; shift 2 ;;
    --side-channel-port) SIDE_CHANNEL_PORT="${2:?missing value for $1}"; shift 2 ;;
    --ssh-option) SSH_OPTIONS+=(-o "${2:?missing value for $1}"); shift 2 ;;
    --reshard-controller-impl) RESHARD_CONTROLLER_IMPL="${2:?missing value for $1}"; shift 2 ;;
    --reshard-sidecar-bin) RESHARD_SIDECAR_BIN="${2:?missing value for $1}"; shift 2 ;;
    --skip-accuracy-oracle) SKIP_ACCURACY_ORACLE=1; shift ;;
    --preflight-only) PREFLIGHT_ONLY=1; shift ;;
    --no-performance-gate) PERFORMANCE_GATE=0; shift ;;
    --keep-cache) KEEP_CACHE=1; shift ;;
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
is_positive_integer "${MAX_MODEL_LEN}" || die "max model length must be positive"
is_positive_integer "${BLOCK_SIZE}" || die "block size must be positive"
((BLOCK_SIZE <= MAX_MODEL_LEN)) \
  || die "block size cannot exceed max model length"
if ((MAX_MODEL_LEN < 65536)); then
  die "the golden benchmark requires --max-model-len at least 65536"
fi
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
          "${role_controller_port}" "${KV_PORT}" "${TRANSFER_PORT}" \
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
  --model-revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL}"
  --run-root "${RUN_ROOT}"
  --run-id "${RUN_ID}"
  --expected-vllm-version "${EXPECTED_VLLM_VERSION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --block-size "${BLOCK_SIZE}"
  --prefill-api-port "${PREFILL_API_PORT}"
  --decode-api-port "${DECODE_API_PORT}"
  --prefill-controller-port "${PREFILL_CONTROLLER_PORT}"
  --decode-controller-port "${DECODE_CONTROLLER_PORT}"
  --kv-port "${KV_PORT}"
  --transfer-port "${TRANSFER_PORT}"
  --side-channel-port "${SIDE_CHANNEL_PORT}"
  --reshard-controller-impl "${RESHARD_CONTROLLER_IMPL}"
  --reshard-sidecar-bin "${RESHARD_SIDECAR_BIN}"
)
local_args_common=(
  --role-kind decode
  --role-dir "${DECODE_ROLE_DIR}"
  --role-python "${DECODE_PYTHON}"
  --role-host "${DECODE_HOST}"
  --model "${MODEL}"
  --model-revision "${MODEL_REVISION}"
  --served-model-name "${SERVED_MODEL}"
  --run-root "${RUN_ROOT}"
  --run-id "${RUN_ID}"
  --expected-vllm-version "${EXPECTED_VLLM_VERSION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --block-size "${BLOCK_SIZE}"
  --prefill-api-port "${PREFILL_API_PORT}"
  --decode-api-port "${DECODE_API_PORT}"
  --prefill-controller-port "${PREFILL_CONTROLLER_PORT}"
  --decode-controller-port "${DECODE_CONTROLLER_PORT}"
  --kv-port "${KV_PORT}"
  --transfer-port "${TRANSFER_PORT}"
  --side-channel-port "${SIDE_CHANNEL_PORT}"
  --reshard-controller-impl "${RESHARD_CONTROLLER_IMPL}"
  --reshard-sidecar-bin "${RESHARD_SIDECAR_BIN}"
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

log "checking that both hosts use the same software lineage"
validate_preflight_pair "${DECODE_PYTHON}" \
  "${RESULT_DIR}/decode.preflight.json" \
  "${RESULT_DIR}/prefill.preflight.json" \
  "${RESULT_DIR}/pair.preflight.json"

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

  if ((KEEP_CACHE == 0)); then
    # These are exact role-local directories created from the already
    # validated run root/id; preserve client and result artifacts.
    if [[ "${DECODE_ROLE_DIR}" == "${RUN_ROOT}/${RUN_ID}/decode" ]]; then
      rm -rf -- "${DECODE_ROLE_DIR}/cache"
    else
      echo "ERROR: refusing unexpected local cache path: ${DECODE_ROLE_DIR}" >&2
      teardown_failed=1
    fi
    if [[ "${PREFILL_ROLE_DIR}" == "${RUN_ROOT}/${RUN_ID}/prefill" ]]; then
      remote_cache_command="$(quote_command rm -rf -- "${PREFILL_ROLE_DIR}/cache")"
      # shellcheck disable=SC2029
      ssh -n "${SSH_OPTIONS[@]}" "${PREFILL_SSH}" "${remote_cache_command}" \
        >/dev/null 2>&1 || teardown_failed=1
    else
      echo "ERROR: refusing unexpected remote cache path: ${PREFILL_ROLE_DIR}" >&2
      teardown_failed=1
    fi
  fi

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
    log "PASS: golden E2E results are in ${RESULT_DIR}/benchmark_summary.json"
  fi
  exit "${status}"
}
trap orchestrator_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

log "starting remote PCP8 prefill role"
# --wait keeps the SSH channel (and therefore its stdin) alive if setsid must
# fork. Without it, scripts larger than the pipe buffer can be truncated.
remote_serve_command="$(quote_command setsid --wait bash -s -- --internal-role serve \
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

ORACLE_DIR="${LOCAL_RUN_DIR}/client/meaningful-32k"
mkdir -p "${ORACLE_DIR}"
if ((SKIP_ACCURACY_ORACLE == 1)); then
  log "WARNING: --skip-accuracy-oracle set; running perf phases WITHOUT the accuracy gate"
  printf 'accuracy oracle skipped by --skip-accuracy-oracle\n' \
    >"${LOCAL_RUN_DIR}/client/ACCURACY_ORACLE_SKIPPED"
else
  log "running the two-request meaningful-data 32K accuracy oracle"
  if ! run_meaningful_oracle "${DECODE_PYTHON}" "${prefill_url}" \
      "${decode_url}" "${SERVED_MODEL}" "${ORACLE_DIR}" \
      "${REQUEST_TIMEOUT}" 2>&1 | tee "${ORACLE_DIR}/oracle.log"; then
    printf 'oracle failed; performance phases were not run\n' \
      >"${LOCAL_RUN_DIR}/client/BENCH_SKIPPED_ORACLE_FAILED"
    die "meaningful-data accuracy oracle failed"
  fi
fi

for benchmark_length in 8192 32768 65534; do
  primer_dir="${LOCAL_RUN_DIR}/client/primer-${benchmark_length}"
  measured_dir="${LOCAL_RUN_DIR}/client/measured-${benchmark_length}"
  mkdir -p "${primer_dir}" "${measured_dir}"
  log "running excluded primer at ${benchmark_length} prompt tokens"
  run_benchmark_phase "${DECODE_PYTHON}" "${prefill_url}" "${decode_url}" \
    "${SERVED_MODEL}" "${primer_dir}" 1 1 "${benchmark_length}" \
    "$((900000 + benchmark_length))" "primer-${benchmark_length}" \
    "${REQUEST_TIMEOUT}" "${MAX_MODEL_LEN}" \
    2>&1 | tee "${primer_dir}/client.log"
  log "running 12 measured requests at ${benchmark_length} tokens (0.05 qps)"
  run_benchmark_phase "${DECODE_PYTHON}" "${prefill_url}" "${decode_url}" \
    "${SERVED_MODEL}" "${measured_dir}" 12 0.05 "${benchmark_length}" \
    "$((100000 + benchmark_length))" "measured-${benchmark_length}" \
    "${REQUEST_TIMEOUT}" "${MAX_MODEL_LEN}" \
    2>&1 | tee "${measured_dir}/client.log"
done

# Re-snapshot logs after all client phases so UUID joins see terminal timing
# events from every measured request.
sleep 2
cp "${DECODE_ROLE_DIR}/server.log" "${RESULT_DIR}/decode.server.log"
# Store mode hosts the reshard plane in-process: no controller.log exists.
if [[ -f "${DECODE_ROLE_DIR}/controller.log" ]]; then
  cp "${DECODE_ROLE_DIR}/controller.log" "${RESULT_DIR}/decode.controller.log"
fi
cp "${DECODE_ROLE_DIR}/server.command" "${RESULT_DIR}/decode.server.command"
collect_remote_logs

log "analyzing client validation and golden r20 acceptance metrics"
analyze_golden_benchmark "${DECODE_PYTHON}" "${LOCAL_RUN_DIR}" \
  "${PERFORMANCE_GATE}" | tee "${RESULT_DIR}/benchmark_analysis.log"
log "golden accuracy and performance validation passed; stopping both roles"
