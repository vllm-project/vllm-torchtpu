#!/usr/bin/env bash
set -euo pipefail

MODEL_GCS_PATH="${1:?model GCS path is required}"
CACHE_HUB_ROOT="${2:-/mnt/disks/persist/models}"
MODEL_DIR="${MODEL_GCS_PATH%/}"
MODEL_DIR="${MODEL_DIR##*/}"

if [[ -d "${CACHE_HUB_ROOT}/${MODEL_DIR}" ]]; then
  echo "Model snapshot is already cached: ${CACHE_HUB_ROOT}/${MODEL_DIR}"
  exit 0
fi

mkdir -p "${CACHE_HUB_ROOT}"
gcloud storage cp -r "${MODEL_GCS_PATH}" "${CACHE_HUB_ROOT}/"

test -d "${CACHE_HUB_ROOT}/${MODEL_DIR}"
