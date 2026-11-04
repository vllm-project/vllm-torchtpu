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
set -euo pipefail

MODEL_GCS_PATH="${1:?model GCS path is required}"
CACHE_HUB_ROOT="${2:?cache hub directory is required}"
MODEL_DIR="${MODEL_GCS_PATH%/}"
MODEL_DIR="${MODEL_DIR##*/}"

mkdir -p "${CACHE_HUB_ROOT}/${MODEL_DIR}"
gcloud storage rsync -r "${MODEL_GCS_PATH}" "${CACHE_HUB_ROOT}/${MODEL_DIR}"

test -d "${CACHE_HUB_ROOT}/${MODEL_DIR}"
