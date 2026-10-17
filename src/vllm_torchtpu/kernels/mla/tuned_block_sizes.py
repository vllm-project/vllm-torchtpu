# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tiles for BF16 sliding-window attention on v7x."""

# 128 query/head rows use four 32-entry MRB regions (two QK, two PV),
# leaving room for the 64-entry KV permutation. Eight windows amortize DMA
# pipeline prologue while keeping enough independent grids for both cores.
SWA_QUERY_ROWS = 128
SWA_PIPELINE_GROUPS = 8

# Amortize the four plane DMAs across independent decode windows.
SWA_DECODE_GROUPS = 8
