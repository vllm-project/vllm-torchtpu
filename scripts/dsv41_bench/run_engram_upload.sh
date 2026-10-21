#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -Eeuo pipefail
ulimit -c 0
export TMPDIR=/tmp/tpu_logs/tmp
export PIP_CACHE_DIR=/tmp/tpu_logs/pip-cache
export UV_CACHE_DIR=/tmp/tpu_logs/uv-cache
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR" "$UV_CACHE_DIR"
python3 -m venv /tmp/tpu_logs/engram-upload-venv
# shellcheck source=/dev/null
source /tmp/tpu_logs/engram-upload-venv/bin/activate
python -m pip install -q torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu
python - <<'PY'
import hashlib
from pathlib import Path
wheel = Path('/wheels/torch-tpu/torch_tpu-0.2.0rc1-cp312-cp312-manylinux_2_31_x86_64.whl')
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == '915bf99d453e33b46933ad9baa37af24b930dcafc5e015a75f9f2b7943087d38'
PY
python -m pip install -q \
  /wheels/torch-tpu/torch_tpu-0.2.0rc1-cp312-cp312-manylinux_2_31_x86_64.whl \
  tpu-sync-torch==0.0.1.dev20261012084241 \
  libtpu==0.0.50 jax==0.11.2 jaxlib==0.11.2 numpy==2.3.5
python -m pip freeze > pip-freeze.txt
python -m pip check > pip-check.txt
unset TORCH_DEVICE_BACKEND_AUTOLOAD
python bench_engram_upload.py --upload-source host_transfer.py \
  --engram-source engram.py --output metrics.json --profile-dir "${ARTIFACT_LOCAL_DIR:-.}/engram-upload-profile"
