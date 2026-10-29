# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridges for stage-scoped candidate index-K reuse."""

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_key_layout import DIM, ROW_WORDS, SLOTS
from vllm_torchtpu.kernels.deepseek_v41.index_selection import (
    candidate,
    tuned_block_sizes,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.gather_index_keys import (
    gather_packed_candidate_keys,
    is_supported_by_gather_packed_candidate_keys,
)

from .cache import (
    IndexCacheRows,
    _cache_num_rows,
    gather_cache_page_words,
    split_halves,
)


class CandidateCache(NamedTuple):
    """Packed Index-K, union block IDs/counts and per-query membership."""

    keys: torch.Tensor
    union_block_ids: torch.Tensor
    membership: torch.Tensor
    union_block_count: torch.Tensor
    usable: torch.Tensor


_prepare_op = None
_prepare_2bank_op = None
_select_op = None
_select_2bank_op = None


def prepare_candidate_cache_jax(
    pool: jax.Array,
    records: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Gather request-local candidate blocks; capacity counts Index-K records."""
    if block_size <= 0 or block_size % SLOTS or capacity <= 0 or capacity % block_size:
        raise ValueError("Candidate capacity must contain whole even-sized blocks")
    groups = records.shape[0]
    key_shape = (groups, capacity // SLOTS, ROW_WORDS)
    queries = candidate.request_query_count(candidate_block_ids.shape[0], groups)
    if queries is None:
        # These shapes are concrete in JAX lowering, unlike Dynamo token rows.
        return (
            jnp.zeros(key_shape, jnp.int32),
            jnp.full((groups, capacity // block_size), -1, jnp.int32),
            jnp.zeros((groups, capacity // block_size), jnp.int32),
            jnp.zeros((groups,), jnp.int32),
            jnp.zeros((1,), jnp.bool_),
        )
    expected = jnp.repeat(jnp.arange(groups, dtype=jnp.int32), queries)
    regular = jnp.all((requests == expected) | (lengths == 0))
    logical_blocks = (records.shape[1] + block_size - 1) // block_size
    if logical_blocks > tuned_block_sizes.CANDIDATE_DENSE_UNION_MAX_BLOCKS:
        candidate_block_ids = jnp.where(lengths[:, None] > 0, candidate_block_ids, -1)
    meta = candidate.union_metadata(
        candidate_block_ids.reshape(groups, queries, -1),
        logical_blocks=logical_blocks,
        capacity_blocks=capacity // block_size,
    )
    starts, membership = candidate.union_record_starts(
        meta, records, block_size=block_size, pool_records=_cache_num_rows(pool) * SLOTS
    )
    usable = regular & ~jnp.any(meta.overflow)
    use_sparsecore_kernel = is_supported_by_gather_packed_candidate_keys(
        pool, starts, block_size
    )
    if use_sparsecore_kernel:
        starts = jnp.where(usable, starts, -1)

    def gather():
        keys = (
            gather_packed_candidate_keys(pool, starts, block_size)
            if use_sparsecore_kernel
            else gather_cache_page_words(pool, starts, block_size, DIM)
        )
        return keys.reshape(key_shape)

    keys = (
        gather()
        if use_sparsecore_kernel
        else jax.lax.cond(usable, gather, lambda: jnp.zeros(key_shape, jnp.int32))
    )
    return (
        keys,
        meta.union_block_ids,
        membership,
        meta.union_block_count,
        usable.reshape(1),
    )


def prepare_candidate_cache_2bank_jax(
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return prepare_candidate_cache_jax(
        cast(Any, (pool_lo, pool_hi)),
        records,
        candidate_block_ids,
        requests,
        lengths,
        block_size,
        capacity,
    )


def select_candidate_cache_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    union_block_ids: jax.Array,
    membership: jax.Array,
    usable: jax.Array,
    block_size: int,
    page_records: int,
    request_tile: int | None,
) -> jax.Array:
    from .indexer_impl import paged_indexer_jax

    groups = records.shape[0]
    queries = candidate.request_query_count(q.shape[0], groups)

    def cached():
        assert queries is not None
        return candidate.candidate_indexer(
            split_halves(q).reshape(groups, queries, *q.shape[1:]),
            weights.astype(jnp.float32).reshape(groups, queries, q.shape[1]),
            keys,
            union_block_ids,
            membership,
            lengths.reshape(groups, queries),
            block_size=block_size,
            request_tile=request_tile,
        )

    def fallback():
        return paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidate_block_ids,
            requests,
            512,
            block_size,
            False,
            True,
            True,
            batch_max,
            page_records,
        )[0]

    if queries is None:
        return fallback()

    # Preserve the old batch-wide short-context bypass. The fallback retains
    # ragged/mixed request ownership and all candidates on union overflow.
    use_cache = usable[0] & (batch_max.max() > 512)
    return jax.lax.cond(use_cache, cached, fallback)


def select_candidate_cache_2bank_jax(
    q: jax.Array,
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidate_block_ids: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    union_block_ids: jax.Array,
    membership: jax.Array,
    usable: jax.Array,
    block_size: int,
    page_records: int,
    request_tile: int | None,
) -> jax.Array:
    return select_candidate_cache_jax(
        q,
        cast(Any, (pool_lo, pool_hi)),
        records,
        weights,
        lengths,
        candidate_block_ids,
        requests,
        batch_max,
        keys,
        union_block_ids,
        membership,
        usable,
        block_size,
        page_records,
        request_tile,
    )


def initialize_candidate_cache() -> None:
    global _prepare_op, _prepare_2bank_op, _select_op, _select_2bank_op
    if _prepare_op is not None:
        return
    from jax.sharding import PartitionSpec as P
    from torch_tpu._internal import pallas

    from vllm_torchtpu.models.vllm.vllm_model_wrapper_context import (
        get_pallas_op_mesh,
    )

    # The pinned bridge annotation omits valid PartitionSpec and static None entries.
    _prepare_op = pallas.jax_op(
        "pallas::dsv41_candidate_prepare",
        prepare_candidate_cache_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 5 + (None,) * 2)
        ),
    )
    _prepare_2bank_op = pallas.jax_op(
        "pallas::dsv41_candidate_prepare_2bank",
        prepare_candidate_cache_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 6 + (None,) * 2)
        ),
    )

    def fake_prepare(
        pool: torch.Tensor,
        records: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        block_size: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        groups = records.shape[0]
        return (
            pool.new_empty((groups, capacity // SLOTS, ROW_WORDS), dtype=torch.int32),
            records.new_empty((groups, capacity // block_size)),
            records.new_empty((groups, capacity // block_size)),
            records.new_empty((groups,)),
            records.new_empty((1,), dtype=torch.bool),
        )

    def fake_prepare_2bank(
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        block_size: int,
        capacity: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return fake_prepare(
            pool_lo,
            records,
            candidate_block_ids,
            requests,
            lengths,
            block_size,
            capacity,
        )

    _prepare_op.register_fake(fake_prepare)
    _prepare_2bank_op.register_fake(fake_prepare_2bank)
    _select_op = pallas.jax_op(
        "pallas::dsv41_candidate_select",
        select_candidate_cache_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 12 + (None,) * 3)
        ),
    )
    _select_2bank_op = pallas.jax_op(
        "pallas::dsv41_candidate_select_2bank",
        select_candidate_cache_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 13 + (None,) * 3)
        ),
    )

    def fake_select(
        q: torch.Tensor,
        pool: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        union_block_ids: torch.Tensor,
        membership: torch.Tensor,
        usable: torch.Tensor,
        block_size: int,
        page_records: int,
        request_tile: int | None,
    ) -> torch.Tensor:
        return records.new_empty((q.shape[0], 512))

    def fake_select_2bank(
        q: torch.Tensor,
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidate_block_ids: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        union_block_ids: torch.Tensor,
        membership: torch.Tensor,
        usable: torch.Tensor,
        block_size: int,
        page_records: int,
        request_tile: int | None,
    ) -> torch.Tensor:
        return fake_select(
            q,
            pool_lo,
            records,
            weights,
            lengths,
            candidate_block_ids,
            requests,
            batch_max,
            keys,
            union_block_ids,
            membership,
            usable,
            block_size,
            page_records,
            request_tile,
        )

    _select_op.register_fake(fake_select)
    _select_2bank_op.register_fake(fake_select_2bank)


def prepare_candidate_cache(
    cache: IndexCacheRows,
    records: torch.Tensor,
    candidate_block_ids: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    *,
    block_size: int,
    capacity: int,
) -> CandidateCache:
    """Gather reusable Index-K in the native packed halfplane layout."""
    assert _prepare_op is not None and _prepare_2bank_op is not None
    args = (
        records.int(),
        candidate_block_ids.int(),
        requests.int(),
        lengths.flatten().int(),
        block_size,
        capacity,
    )
    result = (
        _prepare_2bank_op(cache.raw, cache.raw_hi, *args)
        if cache.raw_hi is not None
        else _prepare_op(cache.raw, *args)
    )
    keys, union_block_ids, membership, union_block_count, usable = result
    return CandidateCache(keys, union_block_ids, membership, union_block_count, usable)


def select_candidate_cache(
    q: torch.Tensor,
    cache: IndexCacheRows,
    records: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    candidate_block_ids: torch.Tensor,
    requests: torch.Tensor,
    batch_max: torch.Tensor,
    prepared: CandidateCache,
    *,
    block_size: int,
    page_records: int,
    request_tile: int | None,
) -> torch.Tensor:
    assert _select_op is not None and _select_2bank_op is not None
    args = (
        records.int(),
        weights,
        lengths.flatten().int(),
        candidate_block_ids.int(),
        requests.int(),
        batch_max.reshape(1).int(),
        prepared.keys,
        prepared.union_block_ids,
        prepared.membership,
        prepared.usable,
        block_size,
        page_records,
        request_tile,
    )
    return (
        _select_2bank_op(q, cache.raw, cache.raw_hi, *args)
        if cache.raw_hi is not None
        else _select_op(q, cache.raw, *args)
    )
