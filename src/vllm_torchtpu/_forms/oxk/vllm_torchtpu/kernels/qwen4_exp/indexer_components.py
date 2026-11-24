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
    """Read native BF16 or its byte-carrier view without requantizing keys.

    ``strided`` (q38 L2 D24, VLLM_TPU_QSA_INDEXER_STRIDED_KEYS): read the 8-bit
    carrier as 32-bit words instead of a BF16 view. Carrier word (r, lane) packs
    keys 2r and 2r + 1 at that lane in BF16's native two-rows-per-word order, so
    a sublane-strided load of the key lanes plus a word -> BF16 vreg bitcast
    gives the same [rows, width] keys without the sublane repack (vrot /
    vcombine) the BF16 view's reshape and lane slice cost. Same key bits.
    """

    strided: bool = False

    @property
    def packing(self) -> int:
        return 2

    @property
    def unit_scales(self) -> bool:
        """q38 PORT-BL IDX: ``read`` returns all-ones scales (BF16 keys are not
        quantized), so the dense step may drop the scale multiply (x * 1.0 is
        x, bit for bit)."""
        return True

    def validate(self, cache: jax.Array, width: int, layout: KVLayout) -> None:
        if layout != KVLayout.HEAD_ALONG_SUBLANE or cache.ndim != 4:
            raise ValueError("BF16 index keys require a four-dimensional HEAD cache")
        if cache.dtype == jnp.bfloat16:
            # Rows at least as wide as the query (wider: the query's lanes
            # are scored, as on the carrier; the key-lanes copies need it).
            if cache.shape[2] != 2 or cache.shape[-1] < width or cache.shape[-1] % 128:
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
        if (
            self.strided
            and ref.dtype != jnp.bfloat16
            and ref.ndim == 3
            and ref.shape[1] == 4
            and rows == 2 * ref.shape[0]
            and width % 128 == 0
            and ref.shape[-1] % 128 == 0
        ):
            tiles = ref.shape[-1] // 128
            # [R, 4, W] carrier bytes -> [R, 1, W] words -> [R * tiles, 128]:
            # row r's lane tiles back to back. Lane tile t of every row r is
            # flat row r * tiles + t: a sublane-strided load per key tile.
            words = ref.bitcast(jnp.uint32).reshape(-1, 128)
            chunks = [
                words[pl.ds(t, ref.shape[0], tiles)] for t in range(width // 128)
            ]
            packed = chunks[0] if len(chunks) == 1 else jnp.concatenate(chunks, axis=1)
            keys = pltpu.bitcast(packed, jnp.bfloat16)
            return keys, jnp.ones((1, rows), jnp.float32)
        values = (
            ref[...] if ref.dtype == jnp.bfloat16 else ref.bitcast(jnp.bfloat16)[...]
        )
        keys = values.reshape(rows, ref.shape[-1])[:, :width]
        return keys, jnp.ones((1, rows), jnp.float32)


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
    strided_keys: bool | None = None,
    key_lanes: int = 0,
    dense_step: bool = False,
    tail_skip: int = 0,
) -> StreamIndexComponents:
    """``key_lanes`` > 0: copy only that many lanes of each key row
    (``StreamIndexComponents.key_lanes``; VLLM_TPU_QSA_INDEXER_KEY_LANES). With
    ``strided_keys`` the word-strided reader reads that narrower window (one lane
    tile per row at 128: a contiguous word load, no stride).

    q38 PORT-BL IDX (ported from Benliang Wang, 2ce02bb1a): ``dense_step``
    (VLLM_TPU_QSA_INDEXER_DENSE_STEP) and ``tail_skip``
    (VLLM_TPU_QSA_INDEXER_TAIL_SKIP), see ``StreamIndexComponents``."""
    from vllm_torchtpu import envs

    if strided_keys is None:
        strided_keys = bool(envs.VLLM_TPU_QSA_INDEXER_STRIDED_KEYS)
    return StreamIndexComponents(
        # q38 L2 D24: VLLM_TPU_QSA_INDEXER_STRIDED_KEYS reads the carrier as
        # words (same key bits; kernel name suffix -sk). Off = the BF16 view.
        # ``strided_keys=False`` pins the BF16 view whatever the flag (prefill).
        BF16KeyReader(strided=True) if strided_keys else BF16KeyReader(),
        QSAIndexSchedule(causal_position_offset),
        per_token_scores,
        table_expansion,
        coalesce_entries,
        key_lanes,
        dense_step,
        tail_skip,
    )
