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
"""Thin ``pallas.jax_op`` custom-op bridges for DeepSeek-V4.1 word-row KV caches."""

from typing import Any

import jax
import torch
from jax.sharding import PartitionSpec
from torch_tpu._internal import pallas

from vllm_torchtpu.kernels.experimental.deepseek_v41.cache import (
    ROPE_WORDS,
    ROW_WORDS,
    gather_cache_rows,
    pack_fp8_kv_rows,
    write_sub_rows,
)

DEFAULT_PAGE_TOKENS: int = 16
_REPLICATED_SPEC: Any = PartitionSpec()
_REPLICATED_PAIR_SPECS: Any = (_REPLICATED_SPEC, _REPLICATED_SPEC)
WRITE_SUB_ROWS_PARTITION_SPECS: Any = (
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
    None,
    None,
)
WRITE_SUB_ROWS_AFTER_PARTITION_SPECS: Any = (
    *WRITE_SUB_ROWS_PARTITION_SPECS,
    _REPLICATED_SPEC,
)
PACK_FP8_KV_PARTITION_SPECS: Any = (_REPLICATED_SPEC,)
GATHER_CACHE_ROWS_PARTITION_SPECS: Any = (
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
    _REPLICATED_SPEC,
)


def _write_sub_rows_jax(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    slot_offset: int = 0,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> jax.Array:
    """Forward word-row slice updates into ``write_sub_rows`` without ``after``."""
    return write_sub_rows(
        cache_words,
        updates,
        slot_mapping,
        slot_offset=slot_offset,
        page_tokens=page_tokens,
    )


def _write_sub_rows_after_jax(
    cache_words: jax.Array,
    updates: jax.Array,
    slot_mapping: jax.Array,
    slot_offset: int,
    page_tokens: int,
    after: jax.Array,
) -> jax.Array:
    """Forward word-row slice updates into ``write_sub_rows`` with ``after``."""
    return write_sub_rows(
        cache_words,
        updates,
        slot_mapping,
        slot_offset=slot_offset,
        after=after,
        page_tokens=page_tokens,
    )


def _pack_fp8_kv_rows_jax(latent: jax.Array) -> tuple[jax.Array, jax.Array]:
    """Forward ``bfloat16[num_tokens, 512]`` latents into ``pack_fp8_kv_rows``."""
    return pack_fp8_kv_rows(latent)


def _gather_cache_rows_jax(
    pool_lo: jax.Array,
    row_indices: jax.Array,
    pool_hi: jax.Array | None = None,
) -> jax.Array:
    """Forward 1-bank or 2-bank row gathers into ``gather_cache_rows``."""
    return gather_cache_rows(pool_lo, row_indices, pool_hi=pool_hi)


_write_sub_rows_op = pallas.jax_op(
    "pallas::dsv41_write_sub_rows",
    _write_sub_rows_jax,
    donate_argnums=(0,),
    input_partition_specs=WRITE_SUB_ROWS_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_SPEC,
)


@_write_sub_rows_op.register_fake
def _fake_write_sub_rows(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_offset: int = 0,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> torch.Tensor:
    """Return an abstract tensor matching ``cache_words`` for FakeTensor tracing."""
    return torch.empty_like(cache_words)


_write_sub_rows_after_op = pallas.jax_op(
    "pallas::dsv41_write_sub_rows_after",
    _write_sub_rows_after_jax,
    donate_argnums=(0,),
    input_partition_specs=WRITE_SUB_ROWS_AFTER_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_SPEC,
)


@_write_sub_rows_after_op.register_fake
def _fake_write_sub_rows_after(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_offset: int,
    page_tokens: int,
    after: torch.Tensor,
) -> torch.Tensor:
    """Return an abstract tensor matching ``cache_words`` for ``after`` tracing."""
    return torch.empty_like(cache_words)


_pack_fp8_kv_rows_op = pallas.jax_op(
    "pallas::dsv41_pack_fp8_kv_rows",
    _pack_fp8_kv_rows_jax,
    input_partition_specs=PACK_FP8_KV_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_PAIR_SPECS,
)


@_pack_fp8_kv_rows_op.register_fake
def _fake_pack_fp8_kv_rows(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return abstract ``(nope_words, rope_words)`` tensors for FakeTensor tracing."""
    num_tokens = latent.shape[0]
    return (
        latent.new_empty((num_tokens, ROW_WORDS), dtype=torch.int32),
        latent.new_empty((num_tokens, ROPE_WORDS), dtype=torch.int32),
    )


_gather_cache_rows_op = pallas.jax_op(
    "pallas::dsv41_gather_cache_rows",
    _gather_cache_rows_jax,
    input_partition_specs=GATHER_CACHE_ROWS_PARTITION_SPECS,
    output_partition_specs=_REPLICATED_SPEC,
)


@_gather_cache_rows_op.register_fake
def _fake_gather_cache_rows(
    pool_lo: torch.Tensor,
    row_indices: torch.Tensor,
    pool_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return an abstract ``int32[num_queries, 128]`` tensor for FakeTensor tracing."""
    return pool_lo.new_empty((row_indices.shape[0], ROW_WORDS), dtype=torch.int32)


def dsv41_write_sub_rows(
    cache_words: torch.Tensor,
    updates: torch.Tensor,
    slot_mapping: torch.Tensor,
    slot_offset: int = 0,
    after: torch.Tensor | None = None,
    page_tokens: int = DEFAULT_PAGE_TOKENS,
) -> torch.Tensor:
    """Write ``32``-, ``64``-, or ``128``-word slices into ``int32[num_rows, 128]``.

    Args:
        cache_words: Target word-row cache bank of shape ``(num_rows, 128)`` and
            dtype ``torch.int32``.
        updates: Sub-row update words of shape ``(num_tokens, slice_words)`` and
            dtype ``torch.int32`` where ``slice_words in {32, 64, 128}``.
        slot_mapping: Global slot indices of shape ``(num_tokens,)`` and dtype
            ``torch.int32`` (negative or out-of-window slots are ignored; duplicate
            in-window slots obey deterministic last-writer-wins).
        slot_offset: Starting global slot index owned by ``cache_words`` (``0`` for
            bank 0; ``rows_lo * (128 // slice_words)`` for bank 1).
        after: Optional tensor whose first 4-byte word is injected into unused SMEM
            metadata padding ``meta[-1]`` to enforce read-before-write ordering
            without copying ``cache_words``.
        page_tokens: Number of logical records per physical page (e.g. ``16`` for
            1x pages or ``8`` for 2x compressed pages) used for burst DMA coalescing.

    Returns:
        Updated ``torch.int32`` cache bank of shape ``(num_rows, 128)`` aliased to
        ``cache_words`` via ``donate_argnums=(0,)``.

    Data lifecycle:
        - Used: ``updates``, ``slot_mapping`` (read-only), ``slot_offset`` and
          ``page_tokens`` (static), optional ``after`` (ordering dependency forwarded
          into unused SMEM padding ``meta[-1]`` so XLA schedules prior reads before
          donated writes).
        - Changed (donated on TPU via ``donate_argnums=(0,)``): ``cache_words`` is
          aliased into the output buffer; persistent pool callers update in place
          via ``cache_words.copy_(dsv41_write_sub_rows(cache_words, ...))`` (matching
          ``deepseek_v4_compressor.py:624-631``) or rebind the returned tensor.
        - Created: updated ``int32[num_rows, 128]`` bank (last-writer-wins, OOB no-op).
    """
    if after is not None:
        return _write_sub_rows_after_op(
            cache_words,
            updates,
            slot_mapping,
            int(slot_offset),
            int(page_tokens),
            after,
        )
    return _write_sub_rows_op(
        cache_words,
        updates,
        slot_mapping,
        int(slot_offset),
        int(page_tokens),
    )


def dsv41_pack_fp8_kv_rows(
    latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize ``bfloat16[num_tokens, 512]`` latents into CSA cache word rows.

    Args:
        latent: Concatenated ``(kv_nope, k_rope)`` tensor of shape
            ``(num_tokens, 512)`` and dtype ``torch.bfloat16`` (first ``448`` dims
            NoPE, last ``64`` dims RoPE).

    Returns:
        Tuple ``(nope_words, rope_words)`` of ``torch.int32`` tensors with shapes
        ``(num_tokens, 128)`` and ``(num_tokens, 32)``.

    Data lifecycle:
        - Used: ``latent`` (read-only).
        - Changed: None.
        - Created: ``(nope_words, rope_words)`` of shapes ``(num_tokens, 128)`` and
          ``(num_tokens, 32)``.
    """
    return _pack_fp8_kv_rows_op(latent)


def dsv41_gather_cache_rows(
    pool_lo: torch.Tensor,
    row_indices: torch.Tensor,
    pool_hi: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gather ``128``-word rows from 1-bank or 2-bank HBM word-row pools.

    Args:
        pool_lo: Low HBM word-row bank of shape ``(rows_lo, 128)`` and dtype
            ``torch.int32`` covering row indices ``[0, rows_lo)``.
        row_indices: Row indices to gather of shape ``(num_queries,)`` and dtype
            ``torch.int32`` (negative or out-of-range indices produce zero rows).
        pool_hi: Optional high HBM word-row bank of shape ``(rows_hi, 128)`` and
            dtype ``torch.int32`` covering row indices ``[rows_lo, rows_lo + rows_hi)``.

    Returns:
        Gathered ``torch.int32`` rows of shape ``(num_queries, 128)``.

    Data lifecycle:
        - Used: ``pool_lo``, ``row_indices``, optional ``pool_hi`` (read-only).
        - Changed: None.
        - Created: ``int32[num_queries, 128]`` with OOB indices zero-filled.
    """
    return _gather_cache_rows_op(pool_lo, row_indices, pool_hi)


__all__ = [
    "dsv41_gather_cache_rows",
    "dsv41_pack_fp8_kv_rows",
    "dsv41_write_sub_rows",
]
