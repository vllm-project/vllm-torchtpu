# SPDX-License-Identifier: Apache-2.0
"""BF16 key reads and completed-group scheduling composed with StreamIndex."""

import os
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax
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


@dataclass(frozen=True)
class InRowKeyReader:
    """Keys in the second BF16 half of 1 KiB FP8 main-KV rows, one per row.

    QWEN4_EXP_INROW_COMPRESSED_K (``pooled_qsa.inrow_compressed_slot``): a
    token row's lane word holds K and V of the device's one KV head in bytes
    0 and 1 and a compressed key's BF16 value in bytes 2 and 3. A kernel page
    of ``rows`` keys is ``rows`` token rows, so the DMA carries twice the key
    bytes; the high half of each word, with the low half cleared, is the
    key's exact FP32 value.
    """

    @property
    def packing(self) -> int:
        return 1

    def validate(self, cache: jax.Array, width: int, layout: KVLayout) -> None:
        if layout != KVLayout.HEAD_ALONG_SUBLANE or cache.ndim != 4:
            raise ValueError("in-row index keys require a four-dimensional HEAD cache")
        if cache.dtype not in (jnp.uint8, jnp.float8_e4m3fn) or cache.shape[2] != 4:
            raise ValueError("in-row index keys require an 8-bit carrier of packing 4")
        if cache.shape[-1] < width or cache.shape[-1] % 128:
            raise ValueError("in-row index keys require an aligned storage width")

    def read(self, ref: Any, rows: int, width: int) -> tuple[jax.Array, jax.Array]:
        # VLLM_TPU_QSA_INROW_READ picks the lowering (trace-time): "u32" views
        # the rows as words and keeps their high halves; "bf16" views them as
        # BF16 pairs and takes the second.
        if os.environ.get("VLLM_TPU_QSA_INROW_READ", "u32") == "bf16":
            pairs = ref.bitcast(jnp.bfloat16)[...]
            keys = pairs.reshape(rows, 2, ref.shape[-1])[:, 1, :width]
        else:
            words = ref.bitcast(jnp.uint32)[...].reshape(rows, ref.shape[-1])
            high = words & jnp.uint32(0xFFFF0000)
            keys = pltpu.bitcast(high, jnp.float32).astype(jnp.bfloat16)[:, :width]
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
    inrow: bool = False,
) -> StreamIndexComponents:
    return StreamIndexComponents(
        InRowKeyReader() if inrow else BF16KeyReader(),
        QSAIndexSchedule(causal_position_offset),
        per_token_scores,
        table_expansion,
        coalesce_entries,
    )
