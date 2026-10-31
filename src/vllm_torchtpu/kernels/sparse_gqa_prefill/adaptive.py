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
"""Bounded TC/SC prefill with request-local work accounting."""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as candidate
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import (
    _paired_working_set,
    get_sparse_prefill_tiles,
    prefill_vmem_limit,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.cost_model import (
    prefill_cost_model,
    sc_cost,
    tc_cost,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.grouped import sc_grouped_attention
from vllm_torchtpu.kernels.sparse_gqa_prefill.occupancy import active_counts
from vllm_torchtpu.kernels.sparse_gqa_prefill.selector import (
    select_sc,
    tc_selected_by_static_bound,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.tuned_block_sizes import (
    PREFILL_DISPATCH_CHUNK_QUERIES,
    sc_chunk_size,
)


def adaptive_prefill(
    q: jax.Array,
    gate: jax.Array,
    blocks: jax.Array,
    lengths: jax.Array,
    table: jax.Array,
    cache: jax.Array,
    starts: jax.Array,
    *,
    kv_heads: int,
    ratio: int,
) -> tuple[jax.Array, jax.Array]:
    """Attend with a bounded workspace; cache is returned unchanged.

    Request starts include padded slots. Inactive token rows have length zero
    and group IDs -1; each active row has a unique valid prefix of selected
    complete groups. KV scales are one. Callers gate all other configurations
    with is_supported_by_adaptive_prefill before entering this function.
    """
    if not is_supported_by_adaptive_prefill(
        q,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        kv_heads=kv_heads,
        ratio=ratio,
        k_scale=1.0,
        v_scale=1.0,
    ):
        raise NotImplementedError(
            "Adaptive prefill requires native packed BF16/FP8 KV, "
            "aligned dimensions, sufficient VMEM and SparseCore; "
            "use block_sparse_gqa_attention for other configurations"
        )
    arrays = (q, gate, blocks, lengths, table)
    query_block, kv_tile = get_sparse_prefill_tiles(
        q, cache, kv_heads=kv_heads, ratio=ratio
    )
    if (
        _paired_working_set(
            q.shape[1],
            kv_heads,
            q.shape[2],
            cache.shape[3],
            jnp.dtype(cache.dtype).itemsize,
            256,
            kv_tile,
        )
        > prefill_vmem_limit()
        or sc_chunk_size(cache, budget=blocks.shape[1], ratio=ratio) == 0
    ):
        return candidate.block_sparse_gqa_attention(
            *arrays,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            query_block=query_block,
            kv_tile=kv_tile,
        )
    model = prefill_cost_model(
        q, blocks, table, cache, kv_heads=kv_heads, ratio=ratio, kv_tile=kv_tile
    )
    if len(q) <= PREFILL_DISPATCH_CHUNK_QUERIES and len(q) % 256 == 0:
        return _one_chunk(arrays, cache, starts, model)

    full_chunks, tail = divmod(len(q), PREFILL_DISPATCH_CHUNK_QUERIES)
    num_tiles = (table.shape[1] * cache.shape[1] + kv_tile - 1) // kv_tile
    choices = _select_sc_for_full_chunks(
        blocks,
        lengths,
        starts,
        cache,
        full_chunks,
        num_tiles,
        _tc_plan(
            q,
            blocks,
            table,
            cache,
            PREFILL_DISPATCH_CHUNK_QUERIES,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
            request_capacity=len(starts) - 1,
        ),
        model,
    )
    # Carry only the final output. In particular, do not slice/materialize
    # all full-chunk Q/gate inputs before the TC/SC branch, or concatenate
    # separately allocated prefix/tail outputs. SC gathers one microchunk.
    output = jnp.zeros_like(q)

    def full(i, output):
        choose = jnp.array(False) if choices is None else choices[i]
        return _write_chunk(
            arrays,
            cache,
            starts,
            output,
            i * PREFILL_DISPATCH_CHUNK_QUERIES,
            PREFILL_DISPATCH_CHUNK_QUERIES,
            choose,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )

    def remainder(output):
        offset = full_chunks * PREFILL_DISPATCH_CHUNK_QUERIES
        choose = _select_sc_for_tail(
            blocks,
            lengths,
            starts,
            cache,
            offset,
            tail,
            num_tiles,
            _tc_plan(
                q,
                blocks,
                table,
                cache,
                tail,
                kv_heads=kv_heads,
                ratio=ratio,
                kv_tile=kv_tile,
                request_capacity=len(starts) - 1,
            ),
            model,
        )
        return _write_chunk(
            arrays,
            cache,
            starts,
            output,
            offset,
            tail,
            choose,
            tail=True,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )

    if full_chunks and tail:
        # Keep the output in one loop carry, including the remainder. A separate
        # prefix loop followed by a tail update can retain another full output
        # workspace across the tail's branches and BF16 rounding operations.
        def step(i, output):
            return lax.cond(
                i < full_chunks, lambda out: full(i, out), remainder, output
            )

        output = lax.fori_loop(0, full_chunks + 1, step, output)
    elif full_chunks:
        output = lax.fori_loop(0, full_chunks, full, output)
    else:
        output = remainder(output)
    return output, cache


def _select_sc_for_full_chunks(
    blocks, lengths, starts, cache, full_chunks, num_tiles, plan, model
):
    """Select full outer chunks without materializing query or gate chunks."""
    choices = None
    if full_chunks:

        def select_chunk(data):
            chunk_index, chunk_blocks, chunk_lengths = data
            return _select_sc_for_chunk(
                chunk_blocks,
                chunk_lengths,
                starts,
                chunk_index * PREFILL_DISPATCH_CHUNK_QUERIES,
                PREFILL_DISPATCH_CHUNK_QUERIES,
                num_tiles,
                cache.dtype,
                plan,
                model,
            )

        choices = lax.map(
            select_chunk,
            (
                jnp.arange(full_chunks),
                blocks[: full_chunks * PREFILL_DISPATCH_CHUNK_QUERIES].reshape(
                    full_chunks, PREFILL_DISPATCH_CHUNK_QUERIES, -1
                ),
                lengths[: full_chunks * PREFILL_DISPATCH_CHUNK_QUERIES].reshape(
                    full_chunks, PREFILL_DISPATCH_CHUNK_QUERIES
                ),
            ),
        )

    return choices


def _select_sc_for_tail(
    blocks, lengths, starts, cache, offset, tail, num_tiles, plan, model
):
    """Pad only tail metadata for selection, at the tail's original call site."""
    padded = (tail + 255) // 256 * 256
    ids = jnp.pad(blocks[offset:], ((0, padded - tail), (0, 0)), constant_values=-1)
    ls = jnp.pad(lengths[offset:], (0, padded - tail))
    return _select_sc_for_chunk(
        ids, ls, starts, offset, tail, num_tiles, cache.dtype, plan, model
    )


def _direct_tc(
    query,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    output, _ = candidate.block_sparse_gqa_attention(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        jnp.array([0, len(query)], jnp.int32),
        kv_heads=kv_heads,
        ratio=ratio,
        query_block=256,
        kv_tile=kv_tile,
    )
    return output


def _ragged_tc_write(
    arrays,
    cache,
    starts,
    output,
    offset,
    size,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    query, gate, blocks, lengths, table = arrays
    return candidate._ragged_prefill(
        query,
        gate,
        blocks,
        lengths,
        table,
        cache,
        starts,
        output,
        offset,
        size,
        ratio=ratio,
        kv_heads=kv_heads,
        query_block=256,
        kv_tile=kv_tile,
        k_scale=1.0,
        v_scale=1.0,
    )


def _nonempty_request_starts_are_q256_aligned(starts):
    """Ignore padded empty requests when checking Q256 start alignment."""
    return jnp.all((starts[:-1] % 256 == 0) | (jnp.diff(starts) == 0))


def _tc(
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
    kv_tile: int,
):
    aligned = _nonempty_request_starts_are_q256_aligned(starts)
    arrays = (query, gate, blocks, lengths, table)
    return lax.cond(
        aligned,
        lambda _: _direct_tc(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        lambda _: _ragged_tc_write(
            arrays,
            cache,
            starts,
            jnp.zeros_like(query),
            0,
            len(query),
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        None,
    )


def _tc_plan(
    query,
    blocks,
    table,
    cache,
    size,
    *,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
    request_capacity: int,
):
    num_tiles = (table.shape[1] * cache.shape[1] + kv_tile - 1) // kv_tile
    words = ((num_tiles * (kv_tile // (ratio * 32)) + 127) // 128) * 128
    return candidate._ragged_plan(
        size,
        query.shape[1],
        cache.shape[4],
        words,
        table.shape[1] * cache.shape[1] // ratio,
        table.shape[1],
        blocks.shape[1],
        256,
        request_capacity=request_capacity,
        query_dtype=query.dtype,
    )


def _select_sc_for_chunk(
    ids, lengths, starts, offset, size, num_tiles, dtype, plan, model
):
    sc_queries = (size + model.sc_chunk - 1) // model.sc_chunk * model.sc_chunk
    pack_tiles, source_rows, _ = plan
    # One packed call bounds both the aligned and request-local schedules.
    if (
        source_rows == size
        and candidate._request_tile_capacity(size, 256, len(starts) - 1) <= pack_tiles
        and tc_selected_by_static_bound(
            queries=pack_tiles * 256,
            num_tiles=num_tiles,
            model=model,
            sc_queries=sc_queries,
        )
    ):
        return jnp.array(False)
    padding = (-len(lengths)) % 256
    ids = jnp.pad(ids, ((0, padding), (0, 0)), constant_values=-1)
    lengths = jnp.pad(lengths, (0, padding))
    local = jnp.clip(starts - offset, 0, size)
    aligned = _nonempty_request_starts_are_q256_aligned(local)

    def ragged(_):
        target = sc_cost(model, lengths[:sc_queries]) * 1.02
        tiles = jnp.sum((jnp.diff(local) + 255) // 256, dtype=jnp.int32)
        calls_lower = (tiles + pack_tiles - 1) // pack_tiles
        live_prefix = jnp.concatenate(
            (jnp.zeros(1, jnp.int32), jnp.cumsum(lengths[:size] > 0, dtype=jnp.int32))
        )
        live_per_request = live_prefix[local[1:]] - live_prefix[local[:-1]]
        visits_lower = jnp.sum((live_per_request + 255) // 256, dtype=jnp.int32)
        lower = tc_cost(
            model,
            queries=calls_lower * pack_tiles * 256,
            num_tiles=num_tiles,
            tile_visits=visits_lower,
            launches=calls_lower,
        )

        def fine(_):
            first, ends, total = candidate._request_tiles(starts, offset, size, 256)
            calls = candidate._ragged_call_count(
                first, ends, total, source_rows, pack_tiles, offset + size
            )

            def request(req, visits):
                count = (local[req + 1] - local[req] + 255) // 256

                def tile(i, visits):
                    first = local[req] + i * 256
                    rows = min(256, len(lengths))
                    start = jnp.minimum(first, len(lengths) - rows)
                    indices = start + jnp.arange(rows)
                    live = (indices >= first) & (indices < local[req + 1])
                    ls = jnp.pad(
                        jnp.where(
                            live, lax.dynamic_slice_in_dim(lengths, start, rows), 0
                        ),
                        (0, 256 - rows),
                    )
                    bs = jnp.pad(
                        lax.dynamic_slice_in_dim(ids, start, rows),
                        ((0, 256 - rows), (0, 0)),
                        constant_values=-1,
                    )
                    return (
                        visits
                        + active_counts(
                            bs,
                            ls,
                            num_tiles=num_tiles,
                            ratio=model.ratio,
                            kv_tile=model.kv_tile,
                        ).sum()
                    )

                return lax.fori_loop(0, count, tile, visits)

            visits = lax.fori_loop(0, len(local) - 1, request, jnp.int32(0))
            return (
                tc_cost(
                    model,
                    queries=calls * pack_tiles * 256,
                    num_tiles=num_tiles,
                    tile_visits=visits,
                    launches=calls,
                )
                > target
            )

        return lax.cond(lower > target, lambda _: jnp.array(True), fine, None)

    return lax.cond(
        aligned,
        lambda _: select_sc(
            ids, lengths, num_tiles=num_tiles, model=model, sc_queries=sc_queries
        ),
        ragged,
        None,
    )


def _one_chunk(arrays, cache, starts, model):
    kv_heads, ratio, kv_tile = model.kv_heads, model.ratio, model.kv_tile
    q, gate, blocks, lengths, table = arrays
    num_tiles = (table.shape[1] * cache.shape[1] + kv_tile - 1) // kv_tile
    choose = _select_sc_for_chunk(
        blocks,
        lengths,
        starts,
        0,
        len(q),
        num_tiles,
        cache.dtype,
        _tc_plan(
            q,
            blocks,
            table,
            cache,
            len(q),
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
            request_capacity=len(starts) - 1,
        ),
        model,
    )

    def sc(_):
        output, _ = _sc_prefill(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            chunk_size=model.sc_chunk,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )
        return output

    return lax.cond(
        choose,
        sc,
        lambda _: _tc(
            q,
            gate,
            blocks,
            lengths,
            table,
            cache,
            starts,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        None,
    ), cache


def _write_chunk(
    arrays,
    cache,
    starts,
    output,
    offset,
    size,
    choose,
    *,
    tail=False,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    def tc(output):
        return _write_tc_chunk(
            arrays,
            cache,
            starts,
            output,
            offset,
            size,
            tail=tail,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )

    def sc(output):
        return _write_sc_chunk(
            arrays,
            cache,
            output,
            offset,
            size,
            tail=tail,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )

    return lax.cond(choose, sc, tc, output)


def _write_tc_chunk(
    arrays,
    cache,
    starts,
    output,
    offset,
    size,
    *,
    tail=False,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    """Write aligned TC work or request-local ragged tiles into the final output."""
    local = jnp.clip(starts - offset, 0, size)
    aligned = _nonempty_request_starts_are_q256_aligned(local)

    def direct(output):
        if tail:
            padding = (-size) % 256
            chunk_arrays = tuple(
                jnp.pad(
                    x[offset:],
                    ((0, padding),) + ((0, 0),) * (x.ndim - 1),
                    constant_values=-1 if i == 2 else 0,
                )
                for i, x in enumerate(arrays)
            )
        else:
            chunk_arrays = tuple(
                lax.dynamic_slice_in_dim(x, offset, size) for x in arrays
            )
        query, gate, blocks, lengths, table = chunk_arrays
        result = _direct_tc(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        )[:size]
        return lax.dynamic_update_slice_in_dim(output, result, offset, axis=0)

    # Align only one request-local Q256 tile directly into the final
    # output. Do not retain full-chunk Q/gate copies and a separate
    # full-chunk ragged output across the request loop.
    return lax.cond(
        aligned,
        direct,
        lambda out: _ragged_tc_write(
            arrays,
            cache,
            starts,
            out,
            offset,
            size,
            kv_heads=kv_heads,
            ratio=ratio,
            kv_tile=kv_tile,
        ),
        output,
    )


def _write_sc_chunk(
    arrays,
    cache,
    output,
    offset,
    size,
    *,
    tail=False,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    """Gather and write one SC microchunk at a time, including a padded tail."""
    chunk = sc_chunk_size(cache, budget=arrays[2].shape[1], ratio=ratio)

    def micro(i, output):
        start = offset + i * chunk
        query, gate, blocks, lengths, table = tuple(
            lax.dynamic_slice_in_dim(x, start, chunk) for x in arrays
        )
        result, _ = sc_grouped_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            query_group=8,
            tile_k=256,
        )
        return lax.dynamic_update_slice_in_dim(output, result, start, axis=0)

    if size // chunk:
        output = lax.fori_loop(0, size // chunk, micro, output)
    if tail and size % chunk:
        live = size % chunk
        start = offset + size - live
        query, gate, blocks, lengths, table = tuple(
            jnp.pad(
                x[start:],
                ((0, chunk - live),) + ((0, 0),) * (x.ndim - 1),
                constant_values=-1 if i == 2 else 0,
            )
            for i, x in enumerate(arrays)
        )
        result, _ = sc_grouped_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            cache,
            kv_heads=kv_heads,
            ratio=ratio,
            query_group=8,
            tile_k=256,
        )
        output = lax.dynamic_update_slice_in_dim(output, result[:live], start, axis=0)
    return output


def _sc_prefill(
    q,
    gate,
    blocks,
    lengths,
    table,
    cache,
    *,
    chunk_size,
    kv_heads: int,
    ratio: int,
    kv_tile: int,
):
    def step(kv, inputs):
        query, gate, blocks, lengths, table = inputs
        out, kv = sc_grouped_attention(
            query,
            gate,
            blocks,
            lengths,
            table,
            kv,
            kv_heads=kv_heads,
            ratio=ratio,
            query_group=8,
            tile_k=256,
        )
        return kv, out

    cache, output = lax.scan(
        step,
        cache,
        tuple(
            x.reshape(-1, chunk_size, *x.shape[1:])
            for x in (q, gate, blocks, lengths, table)
        ),
        unroll=1,
    )
    return output.reshape(q.shape), cache


def is_supported_by_adaptive_prefill(
    query: jax.Array | jax.ShapeDtypeStruct,
    gate: jax.Array | jax.ShapeDtypeStruct,
    blocks: jax.Array | jax.ShapeDtypeStruct,
    lengths: jax.Array | jax.ShapeDtypeStruct,
    table: jax.Array | jax.ShapeDtypeStruct,
    cache: jax.Array | jax.ShapeDtypeStruct,
    starts: jax.Array | jax.ShapeDtypeStruct,
    *,
    kv_heads: int,
    ratio: int,
    k_scale: float,
    v_scale: float,
) -> bool:
    """Static eligibility only; request metadata obeys qsa_attention's contract."""
    # Keep these stages consecutive: early failures must not inspect later inputs.
    if not _has_query_and_request_geometry(query, gate, kv_heads, ratio, starts):
        return False
    if not _has_supported_cache_and_index_metadata(
        query, blocks, lengths, table, cache, starts
    ):
        return False
    if cache.shape[1] % ratio:
        return False
    if not _has_unit_scales(k_scale, v_scale):
        return False
    packing = 32 // jax.dtypes.itemsize_bits(cache.dtype)
    if cache.shape[2:] != (
        (2 * kv_heads + packing - 1) // packing,
        packing,
        query.shape[2],
    ):
        return False
    info = pltpu.get_tpu_info()
    return (
        info.sparse_core is not None
        and _paired_working_set(
            query.shape[1],
            kv_heads,
            query.shape[2],
            packing,
            jax.dtypes.itemsize_bits(cache.dtype) // 8,
            8,
            256,
        )
        <= prefill_vmem_limit()
    )


def _has_query_and_request_geometry(query, gate, kv_heads, ratio, starts):
    """Check query geometry before request offsets."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
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
    ):
        return False
    return True


def _has_supported_cache_and_index_metadata(
    query, blocks, lengths, table, cache, starts
):
    """Check cache layout and budget before inspecting index metadata."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
        cache.ndim != 5
        or cache.shape[0] == 0
        or cache.shape[1] <= 0
        or cache.shape[1] % 128
        or cache.dtype not in (jnp.bfloat16, jnp.float8_e4m3fn)
        or cache.size * jax.dtypes.itemsize_bits(cache.dtype) // 8 > 16 * 2**30
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
    return True


def _has_unit_scales(k_scale, v_scale):
    """Check both scale types before comparing their values."""
    if (  # noqa: SIM103 - Keep one truth test per predicate.
        not isinstance(k_scale, (int, float))
        or not isinstance(v_scale, (int, float))
        or k_scale != 1.0
        or v_scale != 1.0
    ):
        return False
    return True
