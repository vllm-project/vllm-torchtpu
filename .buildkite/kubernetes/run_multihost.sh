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

# Runs work that spans more than one TPU host.
#
#   .buildkite/kubernetes/run_multihost.sh --topology <AxBxC> [--backend ray|mp] <args...>
#   .buildkite/kubernetes/run_multihost.sh --manifest <file-name>
#
# The first form runs one workload across the hosts of a v7x slice, through the
# generic slice manifest for the backend. ray, the default, forms a Ray cluster
# and runs the given command on its head. mp uses vLLM's own multi-node data
# parallelism instead and takes a benchmark config rather than a command - the
# same split, and the same flag, as the bare-metal script of this name.
#
# --topology is the v7x slice shape; either manifest takes it and the host
# count derived from it. It has no default, so every step says which slice it
# takes.
#
# --manifest submits a file from manifests/workloads/ that carries its own
# command per role, for work that is not one command on a head: a prefill and
# a decode engine, two replicas. The manifest names its own hardware, so it
# takes no other flag.
set -euo pipefail

usage() {
  echo "usage: $0 --topology <AxBxC> [--backend ray|mp] <args...>" >&2
  echo "       $0 --manifest <file-name>" >&2
  exit 2
}

backend=""
topology=""
manifest_name=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --backend)  backend="${2:?--backend needs a value}"; shift 2 ;;
    --topology) topology="${2:?--topology needs a value}"; shift 2 ;;
    --manifest) manifest_name="${2:?--manifest needs a file name}"; shift 2 ;;
    *) break ;;
  esac
done

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${manifest_name}" ]]; then
  if [[ -n "${backend}${topology}" || $# -ne 0 ]]; then
    usage
  fi
  manifest="${HERE}/manifests/workloads/${manifest_name}"
  if [[ ! -f "${manifest}" ]]; then
    echo "$0: no such manifest: ${manifest}" >&2
    exit 2
  fi
else
  if [[ -z "${topology}" ]]; then
    usage
  fi
  # Four chips to a v7x host, the tpu7x-standard-4t the manifest asks for.
  if [[ ! "${topology}" =~ ^([0-9]+)x([0-9]+)x([0-9]+)$ ]]; then
    echo "$0: --topology '${topology}' is not AxBxC" >&2
    exit 2
  fi
  chips=$(( BASH_REMATCH[1] * BASH_REMATCH[2] * BASH_REMATCH[3] ))
  if (( chips % 4 != 0 || chips / 4 < 2 )); then
    echo "$0: --topology '${topology}' is ${chips} chips, not a multi-host slice of 4-chip hosts" >&2
    exit 2
  fi
  export MULTIHOST_TOPOLOGY="${topology}"
  export MULTIHOST_HOSTS=$(( chips / 4 ))

  case "${backend:-ray}" in
    ray) manifest="${HERE}/manifests/workloads/ray-multihost-slice.yaml" ;;
    mp)  manifest="${HERE}/manifests/workloads/mp-multihost-slice.yaml" ;;
    *)   echo "$0: unknown backend '${backend}', expected ray or mp" >&2; exit 2 ;;
  esac

  if [[ $# -lt 1 ]]; then
    usage
  fi

  # Passed by name rather than substituted into the manifest, where a quote or
  # brace would be a YAML parse error. NUL-delimited so the pod recovers the
  # argument vector instead of re-splitting a string.
  MULTIHOST_ARGS_B64="$(printf '%s\0' "$@" | base64 | tr -d '\n')"
  export MULTIHOST_ARGS_B64
fi

# Exports WORKLOAD_IMAGE and fills `env_args`.
# shellcheck source=.buildkite/kubernetes/common.sh
# shellcheck disable=SC1091
source "${HERE}/common.sh"

# shellcheck disable=SC2154  # env_args comes from common.sh
exec /opt/launcher/launch --manifest "${manifest}" "${env_args[@]}"
