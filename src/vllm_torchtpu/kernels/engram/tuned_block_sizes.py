# SPDX-License-Identifier: Apache-2.0
"""Engram fusion tiles; adapted from PR #1853 at bbefef5720ee."""

# One M=288 tile streams the 150 MiB WKV weight once, rather than nine
# M=32 tiles reading it nine times. K=256 retains the reference PR's order.
# N=256 bounds each FP32 accumulator tile; a whole D=5120 group spilled.
BLOCK_SIZES = {(288, 6144, 4, 5120): (288, 256, 256)}
VECTOR_TILE_M = 32


def get_block_sizes(
    tokens: int, width: int, streams: int, dim: int
) -> tuple[int, int, int]:
    """Return M/K/N tiles, or small tiles for operator tests."""
    return BLOCK_SIZES.get(
        (tokens, width, streams, dim), (32, min(256, width), min(256, dim))
    )
