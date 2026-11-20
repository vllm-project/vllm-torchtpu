# SPDX-License-Identifier: Apache-2.0
"""BF16 key reads and completed-group scheduling composed with StreamIndex."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.components import (
    StreamIndexComponents,
)
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.config import KVLayout
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.metadata import MetadataRef


@dataclass(frozen=True)
class BF16KeyReader:
    """Read native BF16 or its byte-carrier view without requantizing keys."""

    @property
    def packing(self) -> int:
        return 2

    def validate(self, cache: jax.Array, width: int, layout: KVLayout) -> None:
        if layout != KVLayout.HEAD_ALONG_SUBLANE or cache.ndim != 4:
            raise ValueError("BF16 index keys require a four-dimensional HEAD cache")
        if cache.dtype == jnp.bfloat16:
            if cache.shape[2:] != (2, width):
                raise ValueError(
                    "native BF16 index keys require [pages, rows, 2, width]"
                )
        elif cache.dtype in (jnp.uint8, jnp.float8_e4m3fn):
            if cache.shape[2] != 4 or cache.shape[-1] < width or cache.shape[-1] % 128:
                raise ValueError(
                    "BF16 carrier requires packing 4 and aligned storage width"
                )
        else:
            raise ValueError("BF16 index keys require native BF16 or an 8-bit carrier")

    def read(self, ref: Any, rows: int, width: int) -> tuple[jax.Array, jax.Array]:
        values = (
            ref[...] if ref.dtype == jnp.bfloat16 else ref.bitcast(jnp.bfloat16)[...]
        )
        keys = values.reshape(rows, ref.shape[-1])[:, :width]
        return keys, jnp.ones((1, rows), jnp.float32)

    def read_block(self, block_ref: Any, batch_idx: int, rows: int, width: int) -> jax.Array:
        """``read`` of ``block_ref[batch_idx]`` straight into the matmul's tiles.

        A key pair (rows 2i and 2i + 1) shares one 32-bit word per lane, and
        word row i of the block holds lane tiles ``[0, W // 128)`` of that pair
        back to back. Viewed as ``[rows // 2 * W // 128, 128]`` words, lane
        tile c of every pair is a sublane-strided slice, and its bf16 bitcast
        is the packed (16, 128) layout the q @ k matmul consumes, so no
        relayout runs. Same keys as ``read``.
        """
        lane_tiles = block_ref.shape[-1] // 128
        words = block_ref.bitcast(jnp.uint32)
        words = words.reshape(words.shape[0], words.shape[1] * lane_tiles, 128)
        tiles = [
            pltpu.bitcast(
                words[batch_idx, pl.ds(c, rows // 2, stride=lane_tiles), :],
                jnp.bfloat16,
            )
            for c in range(width // 128)
        ]
        return tiles[0] if len(tiles) == 1 else jnp.concatenate(tiles, axis=1)


@dataclass(frozen=True)
class QSAIndexSchedule:
    """Prefill excludes incomplete groups; expanded decode already bounds lengths."""

    causal_position_offset: int = 0

    def validate(self, compression_ratio: int) -> None:
        if not 0 <= self.causal_position_offset < compression_ratio:
            raise ValueError("causal position offset must be in [0, compression_ratio)")

    def positions(self, positions: jax.Array) -> jax.Array:
        return positions - self.causal_position_offset

    def visible_lengths(self, lengths: jax.Array) -> jax.Array:
        return jnp.maximum(lengths, 0)

    def chunk_metadata(self, meta: MetadataRef, chunk_idx: int) -> MetadataRef:
        num_chunks = meta.num_steps.shape[0]
        if num_chunks == 1:
            return meta
        window_size = meta.batch_tile_idx.shape[0] // num_chunks
        offset = jnp.sum(meta.num_steps[:chunk_idx])
        return MetadataRef.create(
            meta.num_steps[chunk_idx : chunk_idx + 1],
            lax.dynamic_slice_in_dim(meta.batch_tile_idx, offset, window_size),
            lax.dynamic_slice_in_dim(meta.bq_idx, offset, window_size),
            lax.dynamic_slice_in_dim(meta.bkv_idx, offset, window_size),
        )


def qsa_indexing(
    causal_position_offset: int = 0,
    per_token_scores: bool = False,
    table_expansion: tuple[int, int, int] = (1, 1, 0),
    coalesce_entries: bool = False,
    dense_step: bool = False,
    tail_skip: int = 0,
) -> StreamIndexComponents:
    return StreamIndexComponents(
        BF16KeyReader(),
        QSAIndexSchedule(causal_position_offset),
        per_token_scores,
        table_expansion,
        coalesce_entries,
        dense_step,
        tail_skip,
    )
