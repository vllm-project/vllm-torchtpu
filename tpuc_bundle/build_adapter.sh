#!/bin/bash
# Builds the TPU-C MoE adapter `lib/deepsea_torch_bridge.so` (the torch_tpu <->
# PJRT C API bridge that registers torch.ops.deepsea_moe.*) inside the
# vllm-torchtpu container, against its libtorch / torch_tpu headers.
#
# Usage (in the container): bash tpuc_bundle/build_adapter.sh
# vllm_torchtpu.layers.adapter.tpuc_moe also runs this on first use when the
# .so is missing or older than its source.
set -euo pipefail

BUNDLE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"
PY_INC=$("${PYTHON}" -c "import sysconfig; print(sysconfig.get_path('include'))")
TORCH_INC=$("${PYTHON}" -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'include'))")

mkdir -p "${BUNDLE_DIR}/lib"
OUT="${BUNDLE_DIR}/lib/deepsea_torch_bridge.so"
TMP="${OUT}.tmp.$$"
echo "[tpuc_bundle] g++ ${BUNDLE_DIR}/src/deepsea_torch_bridge_gxx.cc -> ${OUT}"
g++ -O3 -std=c++20 -shared -fPIC -D_GLIBCXX_USE_CXX11_ABI=1 \
  -I"${BUNDLE_DIR}/src" \
  -I"${PY_INC}" \
  -I"${TORCH_INC}" \
  -I"${TORCH_INC}/torch/csrc/api/include" \
  "${BUNDLE_DIR}/src/deepsea_torch_bridge_gxx.cc" \
  -o "${TMP}"
chmod 755 "${TMP}"
mv -f "${TMP}" "${OUT}"
echo "[tpuc_bundle] built ${OUT}"
