# SPDX-License-Identifier: Apache-2.0
"""Wo_b output tiles measured on TPU7, one token per device."""

WO_B_TILES: dict[tuple[int, int, int], int] = {(7, 8192, 4096): 1024}

# (TPU generation, tokens, groups, K, rank): groups per weight tile.
WO_A_GROUP_TILES: dict[tuple[int, int, int, int, int], int] = {(7, 1, 8, 4096, 1024): 2}

# Two 8 MiB weight buffers per stage, plus small activation/scale buffers.
# Measured full-window winner over N=512/2048 on TPU7 B1 DP8.
WO_OUTPUT_TILES = {(7, 1, 8192, 4096): 1024}
