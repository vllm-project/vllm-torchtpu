# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BF16 key reads and completed-group scheduling composed with StreamIndex."""

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
from jax import lax

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
            if cache.shape[2] != 4 or cache.shape[-1] != width or cache.shape[-1] % 128:
                raise ValueError(
                    "BF16 carrier requires packing 4 and aligned storage width"
                )
        else:
            raise ValueError("BF16 index keys require native BF16 or an 8-bit carrier")

    def read(self, ref: Any, rows: int, width: int) -> tuple[jax.Array, jax.Array]:
        values = (
            ref[...] if ref.dtype == jnp.bfloat16 else ref.bitcast(jnp.bfloat16)[...]
        )
        keys = values.reshape(rows, width)
        return keys, jnp.ones((1, rows), jnp.float32)


@dataclass(frozen=True)
class QSAIndexSchedule:
    """Prefill excludes incomplete groups; expanded decode already bounds lengths."""

    causal_position_offset: int = 0

    @property
    def preserves_kv_extent(self) -> bool:
        return True

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


def qsa_indexing(causal_position_offset: int = 0) -> StreamIndexComponents:
    return StreamIndexComponents(
        BF16KeyReader(), QSAIndexSchedule(causal_position_offset)
    )
