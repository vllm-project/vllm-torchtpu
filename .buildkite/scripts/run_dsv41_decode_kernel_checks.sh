#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -Eeuo pipefail
results="${1:?results directory is required}"
mkdir -p "$results"
# Keep these local TPU checks on the original 8-device host. Each process
# releases the TPU before the next check starts.
python -m pytest -q tests/kernels/test_dsv41_decode_index_scores.py \
  -k 'grouped_decode or write_flat_rows_with_ragged_lengths' \
  --junitxml="$results/kernel-scores.xml"
python -m pytest -q tests/models/vllm/test_dsv41_decode_indexer_route.py \
  -k 'decode_route_tpu_matches_existing or candidate_cache_matches_existing_with_fallback or basic_config or long_candidate_cache' \
  --junitxml="$results/kernel-routes.xml"
python -m pytest -q tests/kernels/fused_moe/test_fused_ep_moe_v2.py \
  -k row_padded_flat_scales --junitxml="$results/kernel-scales.xml"
python -m pytest -q \
  tests/kernels/experimental/push_fused_moe/test_presort.py \
  tests/kernels/experimental/push_fused_moe/test_routing.py \
  -k 'physical_slot and False' --junitxml="$results/kernel-moe-metadata.xml"
