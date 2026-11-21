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
"""Thin ``pallas.jax_op`` bridges for DeepSeek-V4.1 single-bank KV caches."""

import jax
import torch
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    DEFAULT_PAGE_TOKENS,
    ROPE_WORDS,
    ROW_WORDS,
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)


def _flatten_word_pool(pool: jax.Array) -> jax.Array:
    # Avoid PyTorch .view(-1, 128) on 3D root pools before custom-op entry:
    # passing view inputs to the compiled TPU backend triggers extra strided-view
    # copy/relayout overhead.
    # Avoid pool.reshape(-1, ROW_WORDS): preserve shape[2] so malformed 3D
    # last dimensions fail kernel validation instead of reinterpreting rows.
    if pool.ndim == 3:
        return pool.reshape(pool.shape[0] * pool.shape[1], pool.shape[2])
    return pool


def _write_sub_rows_jax(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> jax.Array:
    updated = write_sub_rows(
        _flatten_word_pool(cache_words),
        updates,
        slot_mapping,
        page_tokens=page_tokens,
    )
    return updated.reshape(cache_words.shape)


def _write_sub_rows_after_jax(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    page_tokens: int,
    after: jax.Array,
) -> jax.Array:
    updated = write_sub_rows(
        _flatten_word_pool(cache_words),
        updates,
        slot_mapping,
        after=after,
        page_tokens=page_tokens,
    )
    return updated.reshape(cache_words.shape)


def _pack_fp8_kv_rows_jax(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    return pack_fp8_kv_rows(latent)


def _gather_cache_rows_jax(
    pool: jax.Array,
    row_indices: jax.Array,
) -> jax.Array:
    return gather_cache_rows(_flatten_word_pool(pool), row_indices)


_write_sub_rows_op = pallas.jax_op(
    "pallas::dsv41_write_sub_rows",
    _write_sub_rows_jax,
    donate_argnums=(0,),
)


@_write_sub_rows_op.register_fake
def _fake_write_sub_rows(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> torch.Tensor:
    del updates, slot_mapping, page_tokens
    return torch.empty_like(cache_words)


_write_sub_rows_after_op = pallas.jax_op(
    "pallas::dsv41_write_sub_rows_after",
    _write_sub_rows_after_jax,
    donate_argnums=(0,),
)


@_write_sub_rows_after_op.register_fake
def _fake_write_sub_rows_after(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    page_tokens: int,
    after: torch.Tensor,
) -> torch.Tensor:
    del updates, slot_mapping, page_tokens, after
    return torch.empty_like(cache_words)


_pack_fp8_kv_rows_op = pallas.jax_op(
    "pallas::dsv41_pack_fp8_kv_rows",
    _pack_fp8_kv_rows_jax,
)


@_pack_fp8_kv_rows_op.register_fake
def _fake_pack_fp8_kv_rows(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = latent.shape[0]
    return (
        latent.new_empty((num_tokens, ROW_WORDS), dtype=torch.int32),
        latent.new_empty((num_tokens, ROPE_WORDS), dtype=torch.int32),
    )


_gather_cache_rows_op = pallas.jax_op(
    "pallas::dsv41_gather_cache_rows",
    _gather_cache_rows_jax,
)


@_gather_cache_rows_op.register_fake
def _fake_gather_cache_rows(
    pool: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    return pool.new_empty((row_indices.shape[0], ROW_WORDS), dtype=torch.int32)


def dsv41_write_sub_rows_in_place(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    after: torch.Tensor | None = None,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> None:
    """Write ``int32[T, 32|64|128]`` slices into ``int32[R, 128]`` or ``[N, P, 128]``.

    Mutates ``cache_words`` in place via a donated custom op (``argnum=0``) and
    ``cache_words.copy_(updated)`` so callers never reuse an invalidated donated
    handle; optional ``after`` orders prior reads.
    """
    if after is not None:
        updated = _write_sub_rows_after_op(
            cache_words,
            updates,
            slot_mapping,
            int(page_tokens),
            after,
        )
    else:
        updated = _write_sub_rows_op(
            cache_words,
            updates,
            slot_mapping,
            int(page_tokens),
        )
    cache_words.copy_(updated)


def dsv41_pack_fp8_kv_rows(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ``bf16[T, 512]`` latents into ``(int32[T, 128], int32[T, 32])`` words."""
    return _pack_fp8_kv_rows_op(latent)


def dsv41_gather_cache_rows(
    pool: torch.Tensor,
    row_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather ``int32[Q, 128]`` rows from a single-bank 2D or 3D ``int32`` pool."""
    return _gather_cache_rows_op(pool, row_indices)


__all__ = [
    "dsv41_gather_cache_rows",
    "dsv41_pack_fp8_kv_rows",
    "dsv41_write_sub_rows_in_place",
]
