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


def _reduce_heads(products, selector, *, sublanes, use_mxu):
    unpack = partial(
        pltpu.unpack_elementwise, packed_dtype=jnp.bfloat16, unpacked_dtype=jnp.float32
    )
    if use_mxu:
        values = lax.dot(selector, products, preferred_element_type=jnp.float32)
    else:
        queries, width = selector.shape[0], products.shape[-1]
        heads = products.shape[0] // queries
        values = None
        for start, count, groups, _ in _head_groups(queries, heads, sublanes):
            packed = pltpu.bitcast(
                products[start * heads : (start + count) * heads], jnp.uint32
            ).reshape(groups, sublanes, width)
            summed = reduce(
                jnp.add,
                (unpack(packed[g], index=h) for g in range(groups) for h in range(2)),
            )
            offset = sublanes // 2
            while offset >= count:
                summed = summed + pltpu.roll(summed, offset, axis=0)
                offset //= 2
            values = (
                summed
                if values is None
                else jnp.where(jnp.arange(sublanes)[:, None] < start, values, summed)
            )
        values = cast(jax.Array, values)[:queries]
    return values.astype(jnp.bfloat16).astype(jnp.float32)


def _dma_rows(src, dst, sem, count, wait=False, src_start=0, dst_start=0):
    for local in range(min(src.shape[0], dst.shape[0])):

        @pl.when(local < count)
        def copy(local=local):
            source = src.at[pl.ds(src_start + local, 1)]
            target = dst.at[pl.ds(dst_start + local, 1)]
            completion = sem.at[local] if sem.shape else sem
            dma = pltpu.make_async_copy(source, target, completion)
            (dma.wait if wait else dma.start)()
