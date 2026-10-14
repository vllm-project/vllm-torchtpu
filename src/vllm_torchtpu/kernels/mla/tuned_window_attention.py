# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared-window tiles measured on TPU v7x with JAX 0.10.2."""

# (query heads, head dim, window width): queries per compute tile.
# Sharing a 256-key KV tile across 64 queries avoids reloading the tile;
# the fused kernel stays below the default 32-MiB scoped VMEM budget.
QUERY_BLOCKS: dict[tuple[int, int, int], int] = {(64, 512, 128): 64}
