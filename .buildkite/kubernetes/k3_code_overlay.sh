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

# Sourced by k3_1p1d.sh and k3_agentx_client_pod.sh when K3_CODE_OVERLAY is
# set. /overlays/<K3_CODE_OVERLAY>/code.tar (the kimi-k31-* manifests mount
# gs://tpu-ci-models-60b4c3c7/k31-overlays at /overlays) holds COMMIT,
# src/vllm_torchtpu, tools/megakernel and configs/.

# apply_code_overlay engine|client: copy the configs into the repo; an engine
# also replaces the repo's vllm_torchtpu and tools/megakernel.
apply_code_overlay() {
  local what="$1" repo="${REPO_DIR:-/workspace/vllm-torchtpu}"
  local tarball="/overlays/${K3_CODE_OVERLAY}/code.tar"
  [[ -f "$tarball" ]] || { echo "[overlay] FATAL: no $tarball" >&2; return 1; }
  rm -rf /tmp/k3-code-overlay
  mkdir -p /tmp/k3-code-overlay && tar -xf "$tarball" -C /tmp/k3-code-overlay || return 1
  echo "[overlay] ${K3_CODE_OVERLAY}: engine code $(cat /tmp/k3-code-overlay/COMMIT)"
  cp -a /tmp/k3-code-overlay/configs/. "$repo/scripts/vllm/benchmarking/configs/" || return 1
  [[ "$what" == engine ]] || return 0
  # Drop the image's modules the overlay does not have, then copy it over.
  (cd "$repo/src" && find vllm_torchtpu -name '*.py' | while read -r f; do
    [[ -f "/tmp/k3-code-overlay/src/$f" ]] || rm -f "$f"
  done)
  cp -a /tmp/k3-code-overlay/src/vllm_torchtpu/. "$repo/src/vllm_torchtpu/" || return 1
  rm -rf "${repo:?}/tools/megakernel"
  mkdir -p "$repo/tools" && cp -a /tmp/k3-code-overlay/tools/megakernel "$repo/tools/megakernel"
}
