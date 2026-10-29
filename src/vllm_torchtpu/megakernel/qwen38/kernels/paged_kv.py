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
"""Decode attention's KV pages in vLLM's unified pools (the HND layout).

Each attention layer's pool is FP8 ``[pages, 2, D / 4, 4, page]``: kernel page
``k`` holds this rank's KV head for ``page`` tokens, the key at index 0 and the
value at 1, each the head's ``[D, page]`` transpose (vLLM's
``VLLM_KV_CACHE_LAYOUT=HND`` pages, which the prefill writes and the KV
transfer moves; a ``[4, page]`` slab is one packed 32-bit row). A vLLM block of
a request's block table is ``pages_per_block`` consecutive kernel pages, and
under CP8 global page ``p`` is local page ``p // 8`` of rank ``p % 8``.

A bank holds one block of one kind: :func:`bank_shape`.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu

FP8 = jnp.float8_e4m3fn


def bank_shape(pages_per_block, head_dim, page):
    return (pages_per_block, head_dim // 4, 4, page)


def transposed(page_ref, live=None):
    """A bank page (``[D / 4, 4, page]``) as the FP8 ``[D, page]`` it holds;
    with ``live`` (traced count), zero in the columns from ``live`` on (a pool
    slot past a request's tokens may hold any bytes, NaN patterns included,
    which a zero probability does not cancel)."""
    rows, packing, width = page_ref.shape
    if live is None:
        return page_ref[...].reshape(rows * packing, width)
    words = page_ref.bitcast(jnp.uint32).reshape(rows, width)[...]
    lanes = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    return tpu.bitcast(jnp.where(lanes < live, words, 0), FP8)


def set_column(page_ref, column, row):
    """Set token ``column`` of a bank page (``[D / 4, 4, page]``) to the FP8
    ``row`` ``[D]``, on the packed words."""
    rows, _, width = page_ref.shape
    words = page_ref.bitcast(jnp.uint32).reshape(rows, width)
    filled = tpu.bitcast(jnp.broadcast_to(row[:, None], (row.shape[0], width)), jnp.uint32)
    lanes = jax.lax.broadcasted_iota(jnp.int32, (rows, width), 1)
    words[...] = jnp.where(lanes == column, filled, words[...])


@dataclasses.dataclass(frozen=True)
class PoolPages:
    """Request ``row``'s pages of attention layer ``layer`` (traced) in ``pools``
    (one per attention layer); ``tables`` is the SMEM ``[rows, width]`` block
    table of the step (rank-local vLLM blocks)."""

    pools: tuple
    layer: jax.Array
    tables: jax.Array
    row: int | jax.Array
    pages_per_block: int

    @property
    def capacity_pages(self):
        return self.pages_per_block * self.tables.shape[-1]

    def first_page(self, block):
        return self.tables[self.row, block] * self.pages_per_block

    def block_copy(self, block, kind, destination, sem):
        """The copy of local block ``block``'s ``kind`` (0 key, 1 value) pages
        into ``destination`` (one :func:`bank_shape`)."""
        def source(pool):
            return pool.at[pl.ds(self.first_page(block), self.pages_per_block), kind]

        return _LayerCopy(self.pools, self.layer, source, destination, sem)


@dataclasses.dataclass(frozen=True)
class _LayerCopy:
    """A copy to or from the traced layer's pool: started under one ``pl.when``
    per pool, waited through any pool's descriptor (the byte count is the same)."""

    pools: tuple
    layer: jax.Array
    source: Callable[[Any], Any]
    destination: Any
    sem: Any

    def _copy(self, pool):
        return tpu.make_async_copy(self.source(pool), self.destination, self.sem)

    def start(self):
        for i, pool in enumerate(self.pools):
            pl.when(self.layer == i)(lambda pool=pool: self._copy(pool).start())

    def wait(self):
        self._copy(self.pools[0]).wait()


def commit_columns(
    pools, columns, positions, tables, *, pages_per_block, cp=8, axis="tp", interpret=False
):
    """Write the decode step's new key/value columns into the pools, in place
    (inside ``shard_map`` over ``axis``; rank-local operands).

    ``columns`` is FP8 ``[layers, batch, 2, D]`` (attention layer ``o``'s key
    and value of request ``b``, for ``pools[o]``); request ``b`` at
    ``positions[b]`` (negative: nothing) writes on the rank owning its page, in
    block ``tables[b, local_page // pages_per_block]``. Each layer's page is read,
    updated and written back; the layers' reads overlap. Non-communicating, so
    the pools alias through without a copy."""
    layers, batch = columns.shape[:2]
    page = pools[0].shape[-1]

    def kernel(positions_ref, tables_ref, columns_ref, *refs):
        outputs, tiles, sems = refs[layers : 2 * layers], refs[2 * layers], refs[2 * layers + 1]
        lane = jax.lax.axis_index(axis) % cp

        @pl.loop(0, batch)
        def commit(b):
            position = positions_ref[b]
            global_page = position // page

            @pl.when((position >= 0) & (global_page % cp == lane))
            def write():
                local = global_page // cp
                target = tables_ref[b, local // pages_per_block] * pages_per_block + local % pages_per_block

                def copy(o, read):
                    pair = (outputs[o].at[target], tiles.at[o])
                    return tpu.make_async_copy(*(pair if read else pair[::-1]), sems.at[o])

                for o in range(layers):
                    copy(o, True).start()
                for o in range(layers):
                    copy(o, True).wait()
                    for kind in range(2):
                        set_column(tiles.at[o, kind], position % page, columns_ref[o, b, kind])
                    copy(o, False).start()
                for o in range(layers):
                    copy(o, False).wait()

    return pl.pallas_call(
        kernel,
        out_shape=tuple(jax.ShapeDtypeStruct(p.shape, p.dtype) for p in pools),
        grid_spec=tpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            in_specs=[pl.BlockSpec(memory_space=tpu.VMEM)] + [pl.BlockSpec(memory_space=tpu.HBM)] * layers,
            out_specs=[pl.BlockSpec(memory_space=tpu.HBM)] * layers,
            scratch_shapes=(
                tpu.VMEM((layers, *pools[0].shape[1:]), pools[0].dtype),
                tpu.SemaphoreType.DMA((layers,)),
            ),
        ),
        input_output_aliases={3 + i: i for i in range(layers)},
        interpret=interpret,
        name="qwen_decode_commit_columns",
    )(positions.astype(jnp.int32), tables.astype(jnp.int32), columns, *pools)
