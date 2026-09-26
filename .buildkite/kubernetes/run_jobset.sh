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

# Submits a workload manifest and follows it.
#
#   .buildkite/kubernetes/run_jobset.sh <manifest-file-name>
#
# For a manifest that carries its own command per role, which is what separates
# these from run_multihost.sh: there the step supplies one command for the Ray
# head, here each role runs something different and the manifest says what.
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 <manifest-file-name>" >&2
  exit 2
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
manifest="${HERE}/manifests/workloads/$1"
if [[ ! -f "${manifest}" ]]; then
  echo "$0: no such manifest: ${manifest}" >&2
  exit 2
fi

# Exports WORKLOAD_IMAGE and fills `env_args`.
# shellcheck source=.buildkite/kubernetes/common.sh
# shellcheck disable=SC1091
source "${HERE}/common.sh"

# shellcheck disable=SC2154  # env_args comes from common.sh
exec /opt/launcher/launch --manifest "${manifest}" "${env_args[@]}"
