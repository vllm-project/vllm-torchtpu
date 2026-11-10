# SPDX-License-Identifier: Apache-2.0
"""SparseCore-pipelined QSA prefill over a bounded read-only page window.

Each prefill query attends only its own selected groups (``blocks``, plus the
partial tail group of a length that is not a multiple of ``ratio``) instead of
the union of a query block's selections:

1. ``plan_pages`` certifies the request's live page prefix and remaps its
   page table onto a staging window; ``stage_pages`` copies those pages (and a
   zero null page) out of the native pool, which stays read-only.
2. Per 64-query microchunk, the SparseCore gathers every query's selected KV
   rows into one of two HBM slots (``slab_gather``) while the TensorCore
   attends the other slot (``grouped``): batched QK/PV per query, FP32 online
   softmax, BF16 output and gate rounding as the TensorCore prefill kernel.

Main-KV decode context parallelism (``dcp=(rank, world)``): a table entry
holds ``world`` consecutive pages' worth of tokens and this rank owns the
tokens with ``(t // page) % world == rank``. ``dcp_selection`` moves each
query's owned groups to the front of its list and renames them to the local
page layout, so a rank gathers only its owned rows; the output is then the
FP32 normalized partial attention and its lse (``qsa_rpa.dcp_combine``).
"""

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.sparse_gqa_prefill import grouped, slab_gather
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    SC_CHUNK_QUERIES,
    SC_GATHER_MAX_CACHE_BYTES,
    SC_QUERY_GROUP,
    SC_TILE_K,
)


class PagePlan(NamedTuple):
    physical: jax.Array
    count: jax.Array
    table: jax.Array
    valid: jax.Array


def window_data_pages(cache, starts, table, workspace_bytes: int | None) -> int:
    """Bound allocation by table namespaces, caller memory policy and SC ABI."""
    page_bytes = math.prod(cache.shape[1:]) * jnp.dtype(cache.dtype).itemsize
    limit = SC_GATHER_MAX_CACHE_BYTES
    if workspace_bytes is not None:
        if workspace_bytes < 0:
            raise ValueError("Workspace bytes must be nonnegative")
        limit = min(limit, workspace_bytes)
    rows_per_page = cache.shape[1] * cache.shape[2]
    pages = min(limit // page_bytes, jnp.iinfo(jnp.int32).max // rows_per_page)
    return max(0, min(pages - 1, (starts.shape[0] - 1) * table.shape[1]))


def _query_geometry_ok(query, gate, kv_heads, ratio, starts):
    return not (
        query.ndim != 3
        or min(query.shape) <= 0
        or kv_heads <= 0
        or query.shape[1] % kv_heads
        or query.shape[2] % 128
        or gate.shape != query.shape
        or query.dtype != jnp.bfloat16
        or gate.dtype != jnp.bfloat16
        or ratio <= 0
        or 256 % ratio
        or (ratio * 32) % 128
        or starts.ndim != 1
        or starts.shape[0] < 2
    )


def is_supported(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    *,
    kv_heads: int,
    ratio: int,
    k_scale=1.0,
    v_scale=1.0,
    kv_head_offset: int = 0,
    workspace_bytes: int | None = None,
) -> bool:
    """Static eligibility: geometry, dtypes, unit scales and SC resources."""
    if not _query_geometry_ok(query, gate, kv_heads, ratio, starts):
        return False
    if (
        cache.ndim != 5
        or cache.shape[0] <= 0
        or cache.shape[1] <= 0
        or cache.shape[1] % 128
        or cache.shape[1] % ratio
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or cache.shape[-1] != query.shape[2]
        or blocks.ndim != 2
        or blocks.shape[0] != len(query)
        or blocks.shape[1] <= 0
        or lengths.shape != (len(query),)
        or table.ndim != 2
        or table.shape[0] != len(query)
        or table.shape[1] <= 0
        or any(x.dtype != jnp.int32 for x in (blocks, lengths, table, starts))
    ):
        return False
    if (
        not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or k_scale != 1.0
        or v_scale != 1.0
    ):
        return False
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    if cache.shape[3] != packing or kv_head_offset < 0:
        return False
    if 2 * (kv_heads + kv_head_offset) > cache.shape[2] * packing:
        return False
    info = pltpu.get_tpu_info()
    if info.sparse_core is None:
        return False
    # attention() narrows starts to the one request with tokens.
    narrowed = jax.ShapeDtypeStruct((2,), jnp.int32)
    data_pages = window_data_pages(cache, narrowed, table, workspace_bytes)
    if data_pages == 0:
        return False
    window = jax.ShapeDtypeStruct((data_pages + 1, *cache.shape[1:]), cache.dtype)
    padded = pl.cdiv(len(query), SC_CHUNK_QUERIES) * SC_CHUNK_QUERIES
    if not grouped._supports_whole_slab(
        jax.ShapeDtypeStruct((padded, blocks.shape[1]), blocks.dtype),
        jax.ShapeDtypeStruct((padded, table.shape[1]), table.dtype),
        window,
        query_group=SC_QUERY_GROUP,
        tile_k=SC_TILE_K,
        padding_rows=cache.shape[1] * cache.shape[2],
    ):
        return False
    # Attention program VMEM: two gathered tiles, FP32 state and BF16 I/O.
    head_group = pl.cdiv(query.shape[1] // kv_heads, 8) * 8
    rows = head_group * SC_QUERY_GROUP
    dim = query.shape[2]
    item = jnp.dtype(cache.dtype).itemsize
    working = (
        2 * SC_QUERY_GROUP * SC_TILE_K * cache.shape[2] * packing * dim * item
        + kv_heads * rows * (2 * 128 + dim) * 4
        + 2 * 2 * 2 * kv_heads * rows * dim * 4
        + 4 * SC_QUERY_GROUP * SC_TILE_K * dim * 4
    )
    return working <= info.vmem_capacity_bytes * 3 // 4


def plan_pages(
    blocks,
    lengths,
    table,
    starts,
    *,
    page_size,
    owner_pages,
    ratio,
    data_pages,
    dcp_world: int = 1,
) -> PagePlan:
    """Certify request-local live prefixes; table padding has no meaning.

    A table entry covers ``dcp_world * page_size`` tokens. ``blocks`` are the
    global selected groups; the plan certifies them against ``lengths``."""
    batch, columns = table.shape
    requests = starts.shape[0] - 1
    entry_tokens = page_size * dcp_world
    rows = jnp.arange(batch, dtype=jnp.int32)
    # Counts of completed requests handle repeated empty starts without reads.
    request = jnp.sum(rows[:, None] >= starts[None, 1:], axis=1, dtype=jnp.int32)
    safe_request = jnp.clip(request, 0, requests - 1)
    maximum = jax.ops.segment_max(lengths, safe_request, num_segments=requests)
    maximum = jnp.maximum(maximum, 0)
    pages = (maximum + entry_tokens - 1) // entry_tokens
    offsets = jnp.cumsum(pages) - pages
    representatives = table[jnp.clip(starts[:-1], 0, batch - 1)]
    logical = jnp.arange(columns, dtype=jnp.int32)
    live = logical[None, :] < pages[:, None]
    count = jnp.sum(pages, dtype=jnp.int32)
    selected = jnp.nonzero(live.reshape(-1), size=data_pages, fill_value=0)[0]
    physical = representatives.reshape(-1)[selected]
    row_live = (lengths[:, None] > 0) & (logical < pages[safe_request, None])
    consistent = jnp.all(
        jnp.where(row_live, table == representatives[safe_request], True)
    )
    physical_valid = jnp.all(
        jnp.where(live, (representatives >= 0) & (representatives < owner_pages), True)
    )
    slots = jnp.arange(blocks.shape[1])[None, :]
    active = slots < jnp.minimum(lengths[:, None] // ratio, blocks.shape[1])
    selected_valid = jnp.all(
        jnp.where(active, (blocks >= 0) & (blocks < lengths[:, None] // ratio), True)
    )
    valid = (
        jnp.all((starts >= 0) & (starts <= batch))
        & jnp.all(
            jnp.where((rows < starts[0]) | (rows >= starts[-1]), lengths == 0, True)
        )
        & jnp.all(jnp.diff(starts) >= 0)
        & jnp.all((lengths >= 0) & (lengths <= columns * entry_tokens))
        & (count <= data_pages)
        & consistent
        & physical_valid
        & selected_valid
    )
    remapped = jnp.where(row_live, offsets[safe_request, None] + logical + 1, 0)
    return PagePlan(physical, count, remapped, valid)


def dcp_selection(blocks, lengths, *, ratio, page_size, world, rank):
    """This rank's selection: owned live groups first, renamed to local groups.

    Global group ``g`` lies in global page ``p = g * ratio // page_size``; the
    rank owns it when ``p % world == rank`` and stores it in table entry
    ``p // world``, so its local group id is
    ``(p // world) * (page_size // ratio) + g % (page_size // ratio)``.
    Returns ``(local_blocks, (count, selected_len, tail))``."""
    budget = blocks.shape[1]
    per_page = page_size // ratio
    nonnegative = jnp.maximum(lengths, 0)
    complete = nonnegative // ratio
    count = jnp.minimum(complete, budget)
    rank = jnp.asarray(rank, jnp.int32).reshape(-1)[0]

    def owned(group):
        return (group // per_page) % world == rank

    def local(group):
        return (group // per_page) // world * per_page + group % per_page

    slots = jnp.arange(budget, dtype=jnp.int32)[None, :]
    mine = (slots < count[:, None]) & owned(blocks)
    order = jnp.argsort(~mine, axis=1, stable=True)
    moved = jnp.take_along_axis(blocks, order, axis=1)
    owned_count = jnp.sum(mine, axis=1, dtype=jnp.int32)
    local_blocks = jnp.where(
        slots < owned_count[:, None], local(jnp.maximum(moved, 0)), -1
    )
    tail_tokens = jnp.where(owned(complete), nonnegative % ratio, 0)
    fields = (owned_count, owned_count * ratio + tail_tokens, local(complete))
    return local_blocks, fields


def stage_pages(cache, physical, count):
    """Write null page and active raw pages; the remaining tail is unwritten."""
    shape = (physical.shape[0] + 1, *cache.shape[1:])
    words = cache.shape[2] * cache.shape[-1]

    def move(ids, active, source, target, scratch, sem):
        page = pl.program_id(0)
        raw = scratch.bitcast(jnp.int32).reshape(1, cache.shape[1], words)
        dst = target.bitcast(jnp.int32).reshape(shape[0], cache.shape[1], words)

        def write():
            dma = pltpu.make_async_copy(raw, dst.at[pl.ds(page, 1)], sem)
            dma.start()
            dma.wait()

        @pl.when(page == 0)
        def null():
            raw[...] = jnp.zeros(raw.shape, jnp.int32)
            write()

        @pl.when((page > 0) & (page <= active[0]))
        def copy():
            src = source.bitcast(jnp.int32).reshape(
                cache.shape[0], cache.shape[1], words
            )
            dma = pltpu.make_async_copy(src.at[pl.ds(ids[page - 1], 1)], raw, sem)
            dma.start()
            dma.wait()
            write()

    return pl.pallas_call(
        move,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(shape[0],),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((1, *cache.shape[1:]), cache.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        out_shape=jax.ShapeDtypeStruct(shape, cache.dtype),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel",)),
        name="qsa_owned_window_raw_stage",
    )(physical, count.reshape(1), cache)


def _lookahead_prefill(
    q,
    gate,
    blocks,
    fields,
    table,
    cache,
    *,
    chunk_size,
    kv_heads,
    ratio,
    padding_rows,
    kv_head_offset=0,
    return_lse=False,
):
    """Two HBM slots: the SC fills one microchunk while the TC attends the
    other; Ref effects order each slot's reuse after its read."""
    query = q
    count = jnp.int32(len(query) // chunk_size)
    assert len(query) // chunk_size > 0 and len(query) % chunk_size == 0
    info = pltpu.get_tpu_info().sparse_core
    rows_per_core = 4 * info.num_lanes
    cores = info.num_cores * info.num_subcores
    head_axes = cache.shape[2]
    tile_k = SC_TILE_K
    capacity = pl.cdiv(blocks.shape[1] * ratio + ratio - 1, tile_k) * tile_k
    granularity = math.lcm(rows_per_core * cores, head_axes)
    rows = pl.cdiv(chunk_size * capacity * head_axes, granularity) * granularity
    shape = (rows, *cache.shape[3:])
    slab = slab_gather.pack_query_metadata(blocks, fields, table)
    selected = fields[1]
    slot_a = jax.ref.new_ref(jnp.zeros(shape, cache.dtype), memory_space=pltpu.HBM)
    slot_b = jax.ref.new_ref(jnp.zeros(shape, cache.dtype), memory_space=pltpu.HBM)

    def write(slot, chunk):
        slab_gather._gather_from_slab_ref(
            slot,
            slab,
            cache,
            budget=blocks.shape[1],
            table_cols=table.shape[1],
            ratio=ratio,
            query_group=SC_QUERY_GROUP,
            tile_k=tile_k,
            padding_rows=padding_rows,
            query_base=chunk * chunk_size,
            chunk_queries=chunk_size,
        )

    def consume(slot, chunk, output):
        part = lambda x: lax.dynamic_slice_in_dim(x, chunk * chunk_size, chunk_size)
        y = grouped._attend_compact_ref(
            part(query),
            part(gate),
            part(selected),
            slot,
            kv_heads=kv_heads,
            ratio=ratio,
            query_group=SC_QUERY_GROUP,
            source_group=SC_QUERY_GROUP,
            tile_k=tile_k,
            capacity=capacity,
            budget=blocks.shape[1],
            head_axes=head_axes,
            kv_head_offset=kv_head_offset,
            return_lse=return_lse,
        )
        return jax.tree.map(
            lambda o, v: lax.dynamic_update_slice_in_dim(o, v, chunk * chunk_size, 0),
            output,
            y,
        )

    if return_lse:
        output = (
            jnp.zeros(query.shape, jnp.float32),
            jnp.full(query.shape[:2], -jnp.inf, jnp.float32),
        )
    else:
        output = jnp.zeros_like(query)
    write(slot_a, jnp.int32(0))

    def pair(p, output):
        write(slot_b, 2 * p + 1)
        output = consume(slot_a, 2 * p, output)
        write(slot_a, 2 * p + 2)
        return consume(slot_b, 2 * p + 1, output)

    output = lax.fori_loop(0, (count - 1) // 2, pair, output)

    def even_tail(output):
        write(slot_b, count - 1)
        output = consume(slot_a, count - 2, output)
        return consume(slot_b, count - 1, output)

    def odd_tail(output):
        return consume(slot_a, count - 1, output)

    return lax.cond(count % 2 == 0, even_tail, odd_tail, output)


def single_request_starts(starts, lengths):
    """``[first, end)`` of the only request with tokens, and whether exactly
    one request has tokens. A two-entry ``starts`` bounds the staging window
    to one table row however many (empty) requests the step carries."""
    real = jnp.diff(starts)
    nonempty = real > 0
    index = jnp.argmax(nonempty)
    narrowed = jnp.stack([starts[index], starts[index + 1]]).astype(jnp.int32)
    return narrowed, (jnp.sum(nonempty) == 1) & jnp.all(real >= 0)


def plan_window(
    blocks, lengths, table, cache, starts, *, ratio, dcp=None, workspace_bytes=None
) -> PagePlan:
    """``plan_pages`` for a two-entry ``starts`` (``single_request_starts``)."""
    assert starts.shape == (2,)
    return plan_pages(
        blocks,
        lengths,
        table,
        starts,
        page_size=cache.shape[1],
        owner_pages=cache.shape[0],
        ratio=ratio,
        data_pages=window_data_pages(cache, starts, table, workspace_bytes),
        dcp_world=1 if dcp is None else int(dcp[1]),
    )


def sc_prefill(
    query,
    gate,
    blocks,
    lengths,
    cache,
    plan: PagePlan,
    *,
    kv_heads: int,
    ratio: int,
    kv_head_offset: int = 0,
    dcp=None,
    return_lse: bool = False,
):
    """SC attention over a certified ``plan`` (``plan_window``).

    Returns the gated BF16 attention, or with ``return_lse`` the pair
    ``(out f32 normalized ungated, lse)``. Meaningful only if ``plan.valid``."""
    page_size = cache.shape[1]
    if dcp is None:
        local_blocks = blocks
        fields = slab_gather.selection_fields(
            lengths, ratio=ratio, budget=blocks.shape[1]
        )
    else:
        local_blocks, fields = dcp_selection(
            blocks,
            lengths,
            ratio=ratio,
            page_size=page_size,
            world=int(dcp[1]),
            rank=dcp[0],
        )
    window = stage_pages(cache, plan.physical, plan.count)
    padded = pl.cdiv(len(query), SC_CHUNK_QUERIES) * SC_CHUNK_QUERIES

    def pad(x, value=0):
        return jnp.pad(
            x,
            ((0, padded - len(query)), *((0, 0),) * (x.ndim - 1)),
            constant_values=value,
        )

    live = lengths > 0
    output = _lookahead_prefill(
        pad(query),
        pad(gate),
        pad(local_blocks, -1),
        tuple(pad(jnp.where(live, f, 0)) for f in fields),
        pad(plan.table),
        window,
        chunk_size=SC_CHUNK_QUERIES,
        kv_heads=kv_heads,
        ratio=ratio,
        padding_rows=page_size * cache.shape[2],
        kv_head_offset=kv_head_offset,
        return_lse=return_lse,
    )
    return jax.tree.map(lambda x: x[: len(query)], output)


def context_reaches(lengths, starts, threshold: int):
    """Whether the longest live causal length in ``[starts[0], starts[-1])``
    reaches ``threshold`` (inclusive)."""
    if threshold <= 0:
        return jnp.bool_(True)
    rows = jnp.arange(len(lengths), dtype=starts.dtype)
    inside = (lengths > 0) & (rows >= starts[0]) & (rows < starts[-1])
    return jnp.max(jnp.where(inside, lengths, 0), initial=0) >= threshold


def attention(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    starts,
    tensorcore,
    *,
    kv_heads: int,
    ratio: int,
    kv_head_offset: int = 0,
    dcp=None,
    return_lse: bool = False,
    min_context: int = 0,
):
    """Run SC prefill when certified, else ``tensorcore()``.

    ``tensorcore()`` returns the same output form (no cache). The cache is an
    operand of both branches and never returned, so the conditional does not
    copy it."""
    narrowed, single = single_request_starts(starts, lengths)
    plan = plan_window(blocks, lengths, table, cache, narrowed, ratio=ratio, dcp=dcp)
    use = single & plan.valid & context_reaches(lengths, starts, min_context)

    def sparse(_):
        return sc_prefill(
            query,
            gate,
            blocks,
            lengths,
            cache,
            plan,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_head_offset=kv_head_offset,
            dcp=dcp,
            return_lse=return_lse,
        )

    return lax.cond(use, sparse, lambda _: tensorcore(), None)
