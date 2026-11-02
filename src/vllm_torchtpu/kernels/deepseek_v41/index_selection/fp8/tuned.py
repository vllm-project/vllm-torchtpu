# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TPU v7 tiles and pipeline depths for the packed FP8 indexer."""

import math
from dataclasses import astuple, dataclass

import jax
from jax.experimental.pallas import tpu as pltpu

from .cache import PAGE_RECORDS, PAGE_ROWS


@dataclass(frozen=True)
class PagedMQAConfig:
    """Paged MQA tiles and pipeline depths."""

    query_tile: int = 8
    key_tile: int = 16384
    issue_depth: int = 4
    key_buffers: int = 4

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in astuple(self)):
            raise ValueError(
                "Paged MQA tiles and issue_depth must be positive integers"
            )
        if self.key_buffers < 2:
            raise ValueError("Paged MQA needs at least two K buffers to overlap DMA")


# TPU v7, H32/D128, transposed packed FP8 pages.
PACKED_FP8_MQA_CONFIG = PagedMQAConfig(6, 65536, 8, 4)
PACKED_FP8_PREPARED_MQA_CONFIG = PagedMQAConfig(2, 32768, 16, 4)

# v7 H32/D128: fully static prefixes avoid dynamic per-fragment addresses.
# A 4K step bounds inactive computation while retaining a long scheduling window.
PACKED_FP8_PREPARED_PREFIX_TOKENS = 4096

# Bound short-output padding to one tile; 4K is the smallest measured choice.
PACKED_FP8_DENSE_MIN_KEY_TILE = 4096


def paged_mqa_key_tile(width: int, page_size: int, cap: int) -> int:
    """Balance tiles at the device MXU-column granularity."""
    alignment = math.lcm(page_size, pltpu.get_tpu_info().mxu_column_size)
    # Tiny storage pages use padded owner rows; keep the same K-buffer budget.
    page_rows = (PAGE_ROWS * page_size + PAGE_RECORDS - 1) // PAGE_RECORDS
    cap = cap * PAGE_ROWS * page_size // (PAGE_RECORDS * page_rows)
    cap = max(alignment, cap // alignment * alignment)
    tiles = max(1, (width + cap - 1) // cap)
    tile = (width + tiles * alignment - 1) // (tiles * alignment) * alignment
    return max(min(cap, PACKED_FP8_DENSE_MIN_KEY_TILE), tile)


def paged_mqa_query_tile(query_size: int, heads: int) -> int:
    """Bound queries by the MXU footprint and head-reduction sublanes."""
    info = pltpu.get_tpu_info()
    return min(query_size, info.num_sublanes, max(1, info.mxu_column_size // heads))


def indexer_query_rows(tokens: int, query_tile: int) -> int:
    """Derive query chunks from query tiles and the SparseCore wave size."""
    if jax.default_backend() != "tpu":
        return tokens
    sc = pltpu.get_tpu_info().sparse_core
    if sc is None:
        return tokens
    wave = math.lcm(query_tile, sc.num_cores * sc.num_subcores)
    chunk = math.gcd(tokens, wave)
    return chunk if chunk % query_tile == 0 else tokens


def topk_reconstruction_threshold(k: int) -> int:
    """Bound the live candidate excess proportionally to the output width."""
    return min(8192, 4 * k)


# Rolling SparseCore gather: three buffers overlap reads and writes for 8-token blocks.
PAGED_MQA_GATHER_CONFIG = {(8, 1): (96, 3), (8, 2): (64, 3)}

# (candidate block, physical page, banks): pages per tile and pipeline buffers.
PAGED_MQA_FRAGMENT_GATHER_CONFIG = {(8, 128, 2): (64, 4)}
