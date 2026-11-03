# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query grouping and packed head reductions shared by FP8 MQA kernels."""

from functools import partial, reduce
from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


class QueryTile(NamedTuple):
    start: jax.Array
    count: jax.Array
    page_row: jax.Array


def _next_query_tile(query, offsets, tile_size, end):
    start = query.start + query.count
    request = lax.while_loop(
        lambda r: (r < offsets.shape[0] - 2) & (start >= offsets[r + 1]),
        lambda r: r + 1,
        query.page_row,
    )
    return QueryTile(
        start,
        jnp.minimum(tile_size, jnp.minimum(offsets[request + 1], end) - start),
        request,
    )


def _head_groups(queries, heads, sublanes):
    start = 0
    while start < queries:
        count = 1 << ((queries - start).bit_length() - 1)
        copies = sublanes // count
        yield start, count, heads // (2 * copies), copies
        start += count


def _head_order(x, queries, sublanes):
    heads, width = x.shape[0] // queries, x.shape[-1]
    x = x.reshape(queries, heads, width)
    parts = []
    for start, count, groups, copies in _head_groups(queries, heads, sublanes):
        part = x[start : start + count]
        if x.dtype == jnp.float32:
            low = part[:, : heads // 2].transpose(1, 0, 2)
            high = part[:, heads // 2 :].transpose(1, 0, 2)
            part = pltpu.pack_elementwise((low, high), packed_dtype=jnp.bfloat16)
        else:
            part = part.reshape(count, 2, groups, copies, width)
            part = part.transpose(2, 3, 0, 1, 4)
        parts.append(part.reshape(-1, width))
    ordered = jnp.concatenate(parts)
    return pltpu.bitcast(ordered, jnp.bfloat16) if x.dtype == jnp.float32 else ordered


def _head_columns(queries, heads, sublanes):
    """Head index of each head-ordered row, matching `_head_order`.

    Built from iota arithmetic so Pallas kernels capture no constants.
    """
    j = lax.broadcasted_iota(jnp.int32, (1, queries * heads), 1)
    columns = jnp.zeros_like(j)
    for start, count, groups, copies in _head_groups(queries, heads, sublanes):
        # Part rows follow (group, copy, query, half) of a [count, 2, groups,
        # copies] block; the head is half * groups * copies + group * copies + copy.
        local = j - start * heads
        half, copy = local % 2, local // (2 * count) % copies
        group = local // (2 * count * copies)
        inside = (local >= 0) & (local < count * heads)
        head = half * (groups * copies) + group * copies + copy
        columns = jnp.where(inside, head, columns)
    return columns


def _weighted_selector(selector, weights, rows, *, sublanes):
    """Fold per-query head weights into the MXU head-reduction selector."""
    queries, heads = selector.shape[0], weights.shape[-1]
    columns = _head_columns(queries, heads, sublanes)
    spread = lax.broadcasted_iota(jnp.int32, (heads, queries * heads), 0) == columns
    # Row i holds query i's weights at its head-ordered product columns. The
    # one-hot BF16 pass is exact for the BF16 weights the selector keeps.
    rows = jnp.concatenate([weights[r][None, :] for r in rows])
    expanded = lax.dot(
        rows.astype(jnp.bfloat16),
        spread.astype(jnp.bfloat16),
        preferred_element_type=jnp.float32,
    )
    return (selector.astype(jnp.float32) * expanded).astype(jnp.bfloat16)


def _reduce_on_mxu(fragment, stage, count, *, every, drain=0):
    """Every `every`-th fragment, and the first `drain` drain stages, use the MXU."""
    if stage < count:
        return fragment % every == every - 1
    return stage - count < drain


def _reduce_heads(products, weights, selector, *, sublanes, use_mxu):
    """Weighted head sums; MXU fragments take weights from ``selector``."""
    if use_mxu:
        return lax.dot(selector, products, preferred_element_type=jnp.float32)
    unpack = partial(
        pltpu.unpack_elementwise, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32
    )
    products = products * weights
    queries, width = selector.shape[0], products.shape[-1]
    heads = products.shape[0] // queries
    values = None
    for start, count, groups, _ in _head_groups(queries, heads, sublanes):
        rows = products[start * heads : (start + count) * heads]
        # Sum head pairs in packed BF16, unpacking once per vector.
        total = reduce(jnp.add, list(rows.reshape(groups, 2 * sublanes, width)))
        packed = pltpu.bitcast(total, jnp.uint32)
        summed = unpack(packed, index=0) + unpack(packed, index=1)
        offset = sublanes // 2
        while offset >= count:
            summed = summed + pltpu.roll(summed, offset, axis=0)
            offset //= 2
        values = (
            summed
            if values is None
            else jnp.where(jnp.arange(sublanes)[:, None] < start, values, summed)
        )
    return cast(jax.Array, values)[:queries]


def _dma_rows(src, dst, sem, count, wait=False, src_start=0, dst_start=0):
    for local in range(min(src.shape[0], dst.shape[0])):

        @pl.when(local < count)
        def copy(local=local):
            source = src.at[pl.ds(src_start + local, 1)]
            target = dst.at[pl.ds(dst_start + local, 1)]
            completion = sem.at[local] if sem.shape else sem
            dma = pltpu.make_async_copy(source, target, completion)
            (dma.wait if wait else dma.start)()
