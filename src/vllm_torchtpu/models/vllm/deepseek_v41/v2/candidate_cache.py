# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch bridges for stage-scoped candidate index-K reuse."""

from typing import Any, NamedTuple, cast

import jax
import jax.numpy as jnp
import torch

from vllm_torchtpu.kernels.deepseek_v41.index_selection import candidate

from .cache import (
    PagedCacheRows,
    _cache_num_rows,
    gather_cache_page_words,
    split_halves,
)


class CandidateCache(NamedTuple):
    keys: torch.Tensor
    ids: torch.Tensor
    membership: torch.Tensor
    actual_count: torch.Tensor
    usable: torch.Tensor
    index_key_layout: str = "ordinary"


_prepare_op = None
_prepare_2bank_op = None
_select_op = None
_select_2bank_op = None


def prepare_candidate_cache_jax(
    pool: jax.Array,
    records: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
    packed: bool = False,
    halfplane: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    groups = records.shape[0]
    key_shape = (groups, capacity // 2 if packed else capacity, 128)
    key_dtype = jnp.int32 if packed else jnp.bfloat16
    if candidates.shape[0] % groups or not 1 <= candidates.shape[0] // groups <= 8:
        # These shapes are concrete in JAX lowering, unlike Dynamo token rows.
        return (
            jnp.zeros(key_shape, key_dtype),
            jnp.full((groups, capacity // block_size), -1, jnp.int32),
            jnp.zeros((groups, capacity // block_size), jnp.int32),
            jnp.zeros((groups,), jnp.int32),
            jnp.zeros((1,), jnp.bool_),
        )
    queries = candidates.shape[0] // groups
    expected = jnp.repeat(jnp.arange(groups, dtype=jnp.int32), queries)
    regular = jnp.all((requests == expected) | (lengths == 0))
    if records.shape[1] > 131072:
        candidates = jnp.where(lengths[:, None] > 0, candidates, -1)
    meta = candidate.union_metadata(
        candidates.reshape(groups, queries, -1),
        logical_blocks=(records.shape[1] + block_size - 1) // block_size,
        capacity_blocks=capacity // block_size,
    )
    starts, membership = candidate.union_record_starts(
        meta, records, block_size=block_size, pool_records=_cache_num_rows(pool) * 2
    )
    usable = regular & ~jnp.any(meta.overflow)

    def gather():
        words = gather_cache_page_words(pool, starts, block_size, 128)
        words = words.reshape(groups, -1, 128)
        return (
            words
            if packed
            else candidate.unpack_candidate_words(words, halfplane=halfplane)
        )

    keys = jax.lax.cond(
        usable,
        gather,
        lambda: jnp.zeros(key_shape, key_dtype),
    )
    return keys, meta.ids, membership, meta.actual_count, usable.reshape(1)


def prepare_candidate_cache_2bank_jax(
    pool_lo: jax.Array,
    pool_hi: jax.Array,
    records: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    lengths: jax.Array,
    block_size: int,
    capacity: int,
    packed: bool = False,
    halfplane: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    return prepare_candidate_cache_jax(
        cast(Any, (pool_lo, pool_hi)),
        records,
        candidates,
        requests,
        lengths,
        block_size,
        capacity,
        packed,
        halfplane=halfplane,
    )


def select_candidate_cache_jax(
    q: jax.Array,
    pool: jax.Array,
    records: jax.Array,
    weights: jax.Array,
    lengths: jax.Array,
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    usable: jax.Array,
    block_size: int,
    page_records: int,
    request_tile: int | None,
    halfplane: bool = False,
) -> jax.Array:
    from .attention_kernels import paged_indexer_jax

    groups = records.shape[0]
    queries = q.shape[0] // groups

    def cached():
        return candidate.candidate_indexer(
            split_halves(q).reshape(groups, queries, *q.shape[1:]),
            weights.astype(jnp.float32).reshape(groups, queries, q.shape[1]),
            keys,
            ids,
            membership,
            lengths.reshape(groups, queries),
            block_size=block_size,
            request_tile=request_tile,
            halfplane=halfplane,
        )

    def fallback():
        return paged_indexer_jax(
            q,
            pool,
            records,
            weights,
            lengths,
            candidates,
            requests,
            512,
            block_size,
            False,
            True,
            True,
            batch_max,
            page_records,
            halfplane=halfplane,
        )[0]

    if q.shape[0] % groups or not 1 <= queries <= 8:
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
    candidates: jax.Array,
    requests: jax.Array,
    batch_max: jax.Array,
    keys: jax.Array,
    ids: jax.Array,
    membership: jax.Array,
    usable: jax.Array,
    block_size: int,
    page_records: int,
    request_tile: int | None,
    halfplane: bool = False,
) -> jax.Array:
    return select_candidate_cache_jax(
        q,
        cast(Any, (pool_lo, pool_hi)),
        records,
        weights,
        lengths,
        candidates,
        requests,
        batch_max,
        keys,
        ids,
        membership,
        usable,
        block_size,
        page_records,
        request_tile,
        halfplane=halfplane,
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
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 5 + (None,) * 4)
        ),
    )
    _prepare_2bank_op = pallas.jax_op(
        "pallas::dsv41_candidate_prepare_2bank",
        prepare_candidate_cache_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 6 + (None,) * 4)
        ),
    )

    def fake_prepare(
        pool,
        records,
        candidates,
        requests,
        lengths,
        block_size,
        capacity,
        packed=False,
        halfplane=False,
    ):
        groups = records.shape[0]
        return (
            pool.new_empty(
                (groups, capacity // 2 if packed else capacity, 128),
                dtype=torch.int32 if packed else torch.bfloat16,
            ),
            records.new_empty((groups, capacity // block_size)),
            records.new_empty((groups, capacity // block_size)),
            records.new_empty((groups,)),
            records.new_empty((1,), dtype=torch.bool),
        )

    def fake_prepare_2bank(
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        candidates: torch.Tensor,
        requests: torch.Tensor,
        lengths: torch.Tensor,
        block_size: int,
        capacity: int,
        packed: bool = False,
        halfplane: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        return fake_prepare(
            pool_lo,
            records,
            candidates,
            requests,
            lengths,
            block_size,
            capacity,
            packed,
            halfplane,
        )

    _prepare_op.register_fake(fake_prepare)
    _prepare_2bank_op.register_fake(fake_prepare_2bank)
    _select_op = pallas.jax_op(
        "pallas::dsv41_candidate_select",
        select_candidate_cache_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 12 + (None,) * 4)
        ),
    )
    _select_2bank_op = pallas.jax_op(
        "pallas::dsv41_candidate_select_2bank",
        select_candidate_cache_2bank_jax,
        mesh=get_pallas_op_mesh(),
        input_partition_specs=cast(
            tuple[tuple[str, ...], ...], cast(Any, (P(),) * 13 + (None,) * 4)
        ),
    )

    def fake_select(
        q: torch.Tensor,
        pool: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidates: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        ids: torch.Tensor,
        membership: torch.Tensor,
        usable: torch.Tensor,
        block_size: int,
        page_records: int,
        request_tile: int | None,
        halfplane: bool = False,
    ) -> torch.Tensor:
        return records.new_empty((q.shape[0], 512))

    def fake_select_2bank(
        q: torch.Tensor,
        pool_lo: torch.Tensor,
        pool_hi: torch.Tensor,
        records: torch.Tensor,
        weights: torch.Tensor,
        lengths: torch.Tensor,
        candidates: torch.Tensor,
        requests: torch.Tensor,
        batch_max: torch.Tensor,
        keys: torch.Tensor,
        ids: torch.Tensor,
        membership: torch.Tensor,
        usable: torch.Tensor,
        block_size: int,
        page_records: int,
        request_tile: int | None,
        halfplane: bool = False,
    ) -> torch.Tensor:
        return fake_select(
            q,
            pool_lo,
            records,
            weights,
            lengths,
            candidates,
            requests,
            batch_max,
            keys,
            ids,
            membership,
            usable,
            block_size,
            page_records,
            request_tile,
            halfplane,
        )

    _select_op.register_fake(fake_select)
    _select_2bank_op.register_fake(fake_select_2bank)


def prepare_candidate_cache(
    cache: PagedCacheRows,
    records: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    lengths: torch.Tensor,
    *,
    block_size: int,
    capacity: int,
    packed: bool = False,
) -> CandidateCache:
    """Prepare once; packed mode avoids the separate BF16 unpack temporary."""
    assert _prepare_op is not None and _prepare_2bank_op is not None
    args = (
        records.int(),
        candidates.int(),
        requests.int(),
        lengths.flatten().int(),
        block_size,
        capacity,
        packed,
        cache.index_key_layout == "halfplane",
    )
    result = (
        _prepare_2bank_op(cache.raw, cache.raw_hi, *args)
        if cache.raw_hi is not None
        else _prepare_op(cache.raw, *args)
    )
    keys, ids, membership, actual_count, usable = result
    return CandidateCache(
        keys, ids, membership, actual_count, usable, cache.index_key_layout
    )


def select_candidate_cache(
    q: torch.Tensor,
    cache: PagedCacheRows,
    records: torch.Tensor,
    weights: torch.Tensor,
    lengths: torch.Tensor,
    candidates: torch.Tensor,
    requests: torch.Tensor,
    batch_max: torch.Tensor,
    prepared: CandidateCache,
    *,
    block_size: int,
    page_records: int,
    request_tile: int | None,
) -> torch.Tensor:
    assert _select_op is not None and _select_2bank_op is not None
    if prepared.index_key_layout != cache.index_key_layout:
        raise ValueError("Prepared Index-K cache layout differs from source cache")
    args = (
        records.int(),
        weights,
        lengths.flatten().int(),
        candidates.int(),
        requests.int(),
        batch_max.reshape(1).int(),
        prepared.keys,
        prepared.ids,
        prepared.membership,
        prepared.usable,
        block_size,
        page_records,
        request_tile,
        cache.index_key_layout == "halfplane",
    )
    return (
        _select_2bank_op(q, cache.raw, cache.raw_hi, *args)
        if cache.raw_hi is not None
        else _select_op(q, cache.raw, *args)
    )
