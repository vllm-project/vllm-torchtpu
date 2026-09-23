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

# Runs work across the hosts of a TPU slice.
#
#   .buildkite/kubernetes/run_multihost.sh [--backend ray|mp] <args...>
#
# ray, the default, forms a Ray cluster and runs the given command on its head.
# mp uses vLLM's own multi-node data parallelism instead and takes a benchmark
# config rather than a command - the same split, and the same flag, as the
# bare-metal script of this name.
#
# The slice shape comes from the manifest's nodeSelector, not from a flag.
set -euo pipefail

backend="ray"
if [[ "${1:-}" == "--backend" ]]; then
  backend="${2:?--backend needs a value}"
  shift 2
fi

case "${backend}" in
  ray) manifest="ray-multihost-slice.yaml" ;;
  mp)  manifest="mp-multihost-slice.yaml" ;;
  *)   echo "$0: unknown backend '${backend}', expected ray or mp" >&2; exit 2 ;;
esac

if [[ $# -lt 1 ]]; then
  echo "usage: $0 [--backend ray|mp] <args...>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=.buildkite/kubernetes/common.sh
# shellcheck disable=SC1091
source "${HERE}/common.sh"

# Passed by name rather than substituted into the manifest, where a quote or
# brace would be a YAML parse error. NUL-delimited so the pod recovers the
# argument vector instead of re-splitting a string.
MULTIHOST_ARGS_B64="$(printf '%s\0' "$@" | base64 | tr -d '\n')"
export MULTIHOST_ARGS_B64

# shellcheck disable=SC2154  # env_args comes from common.sh
exec /opt/launcher/launch \
  --manifest "${HERE}/manifests/workloads/${manifest}" \
  "${env_args[@]}"
