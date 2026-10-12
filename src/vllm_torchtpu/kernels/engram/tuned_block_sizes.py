# SPDX-License-Identifier: Apache-2.0
"""Experimental offload fusion tiles measured on v7x."""

from vllm_torchtpu.logger import init_logger

logger = init_logger(__name__)

# (input width, stream count, hidden width) -> (token tile, contracting tile).
# Stream values first, then one key stream at a time. At the released shape,
# these tiles need a 5 MiB accumulator and a 1.25 MiB FP8 weight tile;
# the larger token tile reduces rereads of the full projection weight.
# K=256 improved measured latency and numerical agreement over K=128.
BLOCK_SIZES = {(6144, 4, 5120): (256, 256)}

# Keep unpacking and gate intermediates small while the MXU uses wider tiles.
VECTOR_TILE_M = 32


def get_block_sizes(tokens: int, width: int, streams: int, dim: int) -> tuple[int, int]:
    key = (width, streams, dim)
    if key not in BLOCK_SIZES:
        logger.warning_once("Engram fusion has no measured tile sizes for %s", key)
    tile_m, tile_k = BLOCK_SIZES.get(key, (32, 256))
    return min(tile_m, tokens), min(tile_k, width)
