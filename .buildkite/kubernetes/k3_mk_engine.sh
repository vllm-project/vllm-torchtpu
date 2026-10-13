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

# One host of a Kimi-K3 megakernel 1P1D role, served as the TPU7x recipe doc
# serves it: `vllm serve --nnodes 4 --node-rank R` on each of the slice's four
# hosts, TP32 across them (kimi-k3-mk-tp32-1p1d-agentx.sh).
#
#   k3_mk_engine.sh prefill|decode
#
# It runs in the engine image (K3_ENGINE_IMAGE), which is built from another
# branch than this file - the megakernel integration branch - so it does not use
# that image's copy of the repo's scripts: k3_mk_submit.sh ships this file and
# the config to the pod (K3_MK_BUNDLE_B64), and only vLLM and the plugin come
# from the image.
#
# Environment, from manifests/workloads/kimi-k3-mk-agentx-1p1d.yaml: POD_IP,
# ROLE_HOST_PREFIX and ROLE_HOST_DOMAIN (host i of this role is
# <prefix>i.<domain>), K3_MK_POINT, K3_ENGINE_IMAGE. K3_MK_DRY_RUN=1 prints the
# command and environment and stops.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
role="${1:-}"
case "$role" in
  prefill|decode) ;;
  *) echo "usage: $0 prefill|decode" >&2; exit 2 ;;
esac
: "${POD_IP:?}" "${ROLE_HOST_PREFIX:?}" "${ROLE_HOST_DOMAIN:?}" "${K3_MK_POINT:?}"
log() { echo "[k3-mk-${role}] $*"; }

# resolve <name> <seconds>: the name's IPv4 address, waiting for DNS to catch
# up with the pod (as multihost_lib.sh's).
resolve() {
  local deadline=$((SECONDS + $2)) ip
  until ip=$(getent ahostsv4 "$1" | awk 'NR == 1 { print $1 }') && [ -n "$ip" ]; do
    [ "$SECONDS" -ge "$deadline" ] && { log "ERROR: $1 did not resolve"; return 1; }
    sleep 5
  done
  echo "$ip"
}

# ---- the doc's per-host variables ---------------------------------------------
NODE_RANK="${JOB_COMPLETION_INDEX:-0}"
NODE_IP="$POD_IP"
if [ "$NODE_RANK" = 0 ]; then
  HEAD_IP="$POD_IP"
else
  HEAD_IP=$(resolve "${ROLE_HOST_PREFIX}0.${ROLE_HOST_DOMAIN}" 1800) || exit 1
fi
PLUGIN_SRC="${PLUGIN_SRC:-/workspace/vllm-torchtpu/src}"
LOG_DIR="${TMPDIR:-/dev/shm}/k3mk/logs"
HF_HOME_DIR="${TMPDIR:-/dev/shm}/k3mk/hf-home"
# Compilation caches persist on the fleet's /cache/jax, keyed by the engine
# image: a cache filled by one toolchain must not be read by another.
CACHE_ROOT="/cache/jax/k3-mk/$(printf '%s' "${K3_ENGINE_IMAGE:-unknown}" | sha1sum | cut -c1-12)"
mkdir -p "$CACHE_ROOT"/{prefill,decode}/{vllm-runtime,vllm-config,torch_tpu_tier3} "$HF_HOME_DIR" "$LOG_DIR"

# ---- the role's environment and arguments: the doc's, from the config --------
export K3_MK_ENGINE=1 K3_1P1D_ROLE="$role"
# shellcheck source=scripts/vllm/benchmarking/configs/kimi-k3-mk-tp32-1p1d-agentx.sh
source "$HERE/kimi-k3-mk-tp32-1p1d-agentx.sh" || exit 2

# GKE gives every TPU pod a slice member's environment. With --nnodes > 1 the
# plugin builds the slice itself from a TCPStore rendezvous, and the injected
# coordinator would be a second, contrary answer (multihost_lib.sh
# use_mp_backend clears the same names).
export MEGASCALE_COORDINATOR_ADDRESS='' MEGASCALE_NUM_SLICES='' MEGASCALE_PORT=''
export MEGASCALE_SLICE_ID='' TPU_PROCESS_ADDRESSES='' TPU_PROCESS_PORT=''
export TPU_WORKER_HOSTNAMES=''

SERVE=(vllm serve "$MODEL_DIR" --port "$K3_MK_PORT" "${COMMON_ARGS[@]}" "${ROLE_ARGS[@]}")
# With --nnodes every host but node 0 runs headless: vLLM's multi-node docs
# (docs/serving/parallelism_scaling.md) and the plugin (tpu_mp_multihost.py) both
# launch the other hosts that way, and vLLM does not infer it. The doc's command
# omits it; without it hosts 1-3 would each start an API server and an engine
# core. Only node 0 serves, so only node 0 runs the Rust frontend.
[ "$NODE_RANK" = 0 ] || SERVE+=(--headless)
log "point $K3_MK_POINT ($DECODE_MODE), host $NODE_RANK/4, head $HEAD_IP, cache $CACHE_ROOT"
if [ "${K3_MK_DRY_RUN:-0}" = 1 ]; then
  printf '[cmd]'; printf ' %q' "${SERVE[@]}"; echo
  env | grep -E '^(TPU_|VLLM_|HF_|TORCH_TPU_|RAIDEN_|LIBTPU_|NPROC|ONEHOT_|XLA_|PYTHONPATH|MEGASCALE_)' | sort
  exit 0
fi

# What has to be in place before a long boot can succeed.
for p in "$MODEL_DIR/config.json" "$PRESHARDED_DIR/layout.json" "$PRESHARDED_DIR/prefill-sp-cache/layout.json"; do
  [ -f "$p" ] || { log "FATAL: $p is missing"; exit 2; }
done
if [ "$NODE_RANK" = 0 ] && [ ! -x "$VLLM_RUST_FRONTEND_PATH" ]; then
  log "FATAL: no Rust frontend at $VLLM_RUST_FRONTEND_PATH; the engine image predates" \
      "#1884 (vllm-rs in the image) - rebuild it from the branch head"
  exit 2
fi
log "vllm_torchtpu from $(python3 -c 'import vllm_torchtpu, os; print(os.path.dirname(vllm_torchtpu.__file__))' 2>&1 | tail -1);" \
    "megakernel: $(python3 -c 'import vllm_torchtpu.megakernel' >/dev/null 2>&1 && echo yes || echo NO)"
if [ "$DECODE_MODE" != dp2_dcp16 ]; then
  _d="$PRESHARDED_DIR/dspark-${K3_MK_DSPARK_REVISION:0:8}"
  if [ ! -f "$DSPARK_DIR/config.json" ] || [ ! -d "$_d" ]; then
    log "FATAL: $DSPARK_DIR or $_d is missing"
    exit 2
  fi
fi

ulimit -c 0
cd "$PLUGIN_SRC/.." || exit 2
exec "${SERVE[@]}"
