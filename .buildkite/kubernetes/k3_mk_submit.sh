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

# Submits the Kimi-K3 megakernel AgentX pair (kimi-k3-mk-agentx-1p1d.yaml)
# through run_multihost.sh. The client is this build's image and this commit's
# AgentX client; the engines run K3_ENGINE_IMAGE. This sets what the manifest
# substitutes: the engine image, the point, and K3_MK_BUNDLE_B64 - the engine
# launcher and serve config from this checkout, since the engine image is built
# from another branch and has neither.
#
# Build env it reads (all optional):
#   K3_ENGINE_IMAGE  engine image; else build_k3_engine_image.sh's metadata
#   K3_MK_POINT      the recipe doc's point, 1-5 (6-7 experimental); default 1
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"

if [[ -z "${K3_ENGINE_IMAGE:-}" ]]; then
  K3_ENGINE_IMAGE=$(buildkite-agent meta-data get K3_ENGINE_IMAGE --default "" 2>/dev/null || true)
fi
if [[ -z "${K3_ENGINE_IMAGE}" ]]; then
  echo "$0: no engine image. The engine image step publishes K3_ENGINE_IMAGE;" \
       "this step has to depend on it, or the build has to set K3_ENGINE_IMAGE." >&2
  exit 2
fi
export K3_ENGINE_IMAGE

export K3_MK_POINT="${K3_MK_POINT:-1}"
case "${K3_MK_POINT}" in
  [1-7]) ;;
  *) echo "$0: K3_MK_POINT must be 1-7, not '${K3_MK_POINT}'" >&2; exit 2 ;;
esac

# Flat, so the launcher finds the config beside it.
K3_MK_BUNDLE_B64="$(tar -C "${REPO_ROOT}" -czf - --transform 's|.*/||' \
  .buildkite/kubernetes/k3_mk_engine.sh \
  scripts/vllm/benchmarking/configs/kimi-k3-mk-tp32-1p1d-agentx.sh | base64 -w0)"
export K3_MK_BUNDLE_B64

echo "--- Kimi-K3 megakernel AgentX: point ${K3_MK_POINT}, engine image ${K3_ENGINE_IMAGE}"
exec "${HERE}/run_multihost.sh" --manifest kimi-k3-mk-agentx-1p1d.yaml
