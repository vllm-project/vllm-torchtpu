#!/usr/bin/env bash
set -euo pipefail

MODEL_GCS_PATH="${1:?model GCS path is required}"
CACHE_HUB_ROOT="${2:-/mnt/disks/persist/models/hub}"
MODEL_DIR="${MODEL_GCS_PATH%/}"
MODEL_DIR="${MODEL_DIR##*/}"

mkdir -p "${CACHE_HUB_ROOT}/${MODEL_DIR}"
gcloud storage rsync -r "${MODEL_GCS_PATH}" "${CACHE_HUB_ROOT}/${MODEL_DIR}"

test -d "${CACHE_HUB_ROOT}/${MODEL_DIR}"
