#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
results="${1:?results directory is required}"
mkdir -p "$results"
# Keep these local TPU checks on the original 8-device host. Each process
# releases the TPU before the next check starts.
python -m pytest -q tests/kernels/fused_moe/test_fused_ep_moe_v2.py \
  -k row_padded_flat_scales --junitxml="$results/kernel-scales.xml"
