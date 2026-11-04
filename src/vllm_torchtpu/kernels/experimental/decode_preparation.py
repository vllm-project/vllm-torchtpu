# SPDX-License-Identifier: Apache-2.0
"""Experimental decode preparation boundaries validated at T=288.

These functions keep communication and SparseCore producers outside the
TensorCore preparation call. Callers retain their general-shape paths.
"""

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def take_along(values, indices, axis):
    """Explicit batched gather; retain singleton dimensions for Mosaic."""
    batch = tuple(i for i in range(values.ndim) if i != axis)
    return lax.gather(
        values,
        indices[..., None],
        lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(axis,),
            start_index_map=(axis,),
            operand_batching_dims=batch,
            start_indices_batching_dims=batch,
        ),
        slice_sizes=(1,) * values.ndim,
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def prefix_sum(a, axis=None, dtype=None, out=None):
    """Integer-only inclusive scan with explicit shift/add dependencies."""
    assert out is None
    if axis is None:
        a, axis = a.reshape(-1), 0
    if dtype is not None:
        a = a.astype(dtype)
    assert jnp.issubdtype(a.dtype, jnp.integer)
    distance = 1
    while distance < a.shape[axis]:
        tail = lax.slice_in_dim(a, 0, a.shape[axis] - distance, axis=axis)
        pad_shape = list(a.shape)
        pad_shape[axis] = distance
        a = a + jnp.concatenate((jnp.zeros(pad_shape, a.dtype), tail), axis=axis)
        distance *= 2
    return a


def fuse_function(fn, args, *, name, interpret=False):
    """Real Pallas boundary, with the same externally visible array outputs."""
    out_shape = jax.eval_shape(fn, *args)
    leaves, tree = jax.tree.flatten(out_shape)

    def kernel(*refs):
        outputs = fn(*(r[...] for r in refs[: len(args)]))
        for ref, value in zip(refs[len(args) :], jax.tree.leaves(outputs)):
            ref[...] = value

    call = pl.pallas_call(
        kernel,
        out_shape=tuple(leaves),
        name=name,
        interpret=interpret,
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
        ),
    )
    return lambda *xs: jax.tree.unflatten(tree, call(*xs))


def metadata_matrix(table, starts, positions, minimum):
    """Keep explicit row/column dimensions through gather and reductions."""
    tokens, requests = positions.shape[1], table.shape[0]
    first = jnp.sum(
        jnp.where(
            jnp.arange(tokens)[None, :] == jnp.minimum(starts[:, :-1].T, tokens - 1),
            positions,
            0,
        ),
        axis=1,
        keepdims=True,
    )
    count = (starts[:, 1:] - starts[:, :-1]).T
    left = jnp.where(count > 0, jnp.maximum(first - 127, 0), 0)
    right = jnp.where(count > 0, first + count, 0)
    columns = (tokens + 128 + 63) // 64 + 1
    logical = left // 64 + jnp.arange(columns)[None, :]
    hit = (
        jnp.arange(table.shape[1])[None, None, :]
        == jnp.minimum(logical, table.shape[1] - 1)[:, :, None]
    )
    blocks = jnp.sum(jnp.where(hit, table[:, None, :], 0), axis=2)
    visible = (count > 0) & (logical < table.shape[1]) & (logical * 64 < right)
    metadata = jnp.concatenate(
        (left // 64 * 64, minimum, jnp.where(visible, blocks, 0)), axis=1
    )
    index = jnp.arange(tokens)[None, :]
    request = jnp.minimum(
        jnp.sum(index >= starts[:, 1:].T, axis=0, keepdims=True), requests - 1
    )
    page_hit = (
        logical[:, :, None]
        == jnp.minimum(positions // 64, table.shape[1] - 1)[:, None, :]
    )
    per_request = jnp.sum(jnp.where(page_hit, blocks[:, :, None], 0), axis=1)
    current = jnp.sum(
        jnp.where(jnp.arange(requests)[:, None] == request, per_request, 0),
        axis=0,
        keepdims=True,
    )
    records = jnp.where(index < starts[:, -1:], current * 64 + positions % 64, -1)
    return metadata, jnp.concatenate((left, right), axis=1), records


def pack_matrix(records, values, *, base, capacity, padded):
    n = values.shape[0]
    local = jnp.where(
        (max(base, 64) <= records.T) & (base + capacity > records.T),
        records.T - base,
        capacity,
    )
    idx = jnp.arange(n, dtype=jnp.int32)[:, None]
    ranks = jnp.sum(
        (local > local.T) | ((local == local.T) & (idx > idx.T)), axis=1, keepdims=True
    )
    permutation = idx == ranks.T
    ids = jnp.sum(jnp.where(permutation, local.T, 0), axis=1, keepdims=True)
    next_ids = jnp.concatenate((ids[1:], jnp.full((1, 1), capacity, jnp.int32)), axis=0)
    pairs = ids // 2
    boundary = (ids < capacity) & (
        jnp.concatenate(
            (jnp.ones((1, 1), jnp.int32), (pairs[1:] != pairs[:-1]).astype(jnp.int32)),
            axis=0,
        )
        != 0
    )
    even = ids % 2 == 0
    odd = (~even) | (next_ids // 2 == pairs)
    value_bits = lax.bitcast_convert_type(values, jnp.uint16).astype(jnp.uint32)
    bits = jnp.dot(
        permutation.astype(jnp.float32),
        value_bits.astype(jnp.float32),
        precision=lax.Precision.HIGHEST,
    ).astype(jnp.uint32)
    next_bits = jnp.concatenate((bits[1:], jnp.zeros((1, 512), jnp.uint32)), axis=0)
    low = jnp.where(even, bits, jnp.uint32(0))
    high = jnp.where(odd, jnp.where(even, next_bits, bits), jnp.uint32(0))
    words = lax.bitcast_convert_type(low | (high << 16), jnp.int32)
    masks = jnp.where(even, jnp.int32(65535), jnp.int32(0)) | jnp.where(
        odd, jnp.int32(-65536), jnp.int32(0)
    )
    return (
        jnp.pad(
            jnp.where(boundary, pairs, -1),
            ((0, padded - n), (0, 0)),
            constant_values=-1,
        ),
        jnp.broadcast_to(jnp.pad(masks, ((0, padded - n), (0, 0))), (padded, 512)),
        jnp.pad(words, ((0, padded - n), (0, 0))),
    )


def main_records_call(xs, *, page_size):
    selected, pages, starts = xs
    t, k = selected.shape
    requests, pps = pages.shape
    padded_pps = (pps + 127) // 128 * 128

    def body(sel, table, boundaries):
        token = jnp.arange(t, dtype=jnp.int32)
        req = jnp.sum(token[:, None] >= boundaries[None, 1:], axis=1)
        req = jnp.minimum(req, requests - 1)
        onehot = (req[:, None] == jnp.arange(requests)[None, :]).astype(jnp.bfloat16)
        table = jnp.pad(table, ((0, 0), (0, padded_pps - pps)))
        window = jnp.zeros((t, padded_pps), jnp.int32)
        # Four exact byte limbs preserve every INT32 page id, including -1.
        # 4*2*T*requests*padded_pps = 56.6 MFLOP for this shape.
        for shift in (0, 8, 16, 24):
            limb = ((table >> shift) & 255).astype(jnp.bfloat16)
            gathered = jnp.dot(onehot, limb, preferred_element_type=jnp.float32).astype(
                jnp.int32
            )
            window = window | (gathered << shift)
        valid = (sel >= 0) & (token[:, None] < boundaries[-1])
        logical = jnp.maximum(sel, 0)
        page = jnp.clip(logical // page_size, 0, pps - 1)
        physical = jnp.zeros_like(page)
        for c in range(padded_pps // 128):
            local = page - c * 128
            chunk = window[:, c * 128 : (c + 1) * 128]
            value = take_along(chunk, jnp.clip(local, 0, 127), axis=1)
            physical = jnp.where((local >= 0) & (local < 128), value, physical)
        records = jnp.where(valid, physical * page_size + logical % page_size, -1)
        return records, jnp.sum(valid, axis=1, keepdims=True, dtype=jnp.int32).T

    fused = fuse_function(body, xs, name="dsv41_main_records_matrix")

    def call(*xs):
        records, counts = fused(*xs)
        return records, counts.reshape(t)

    return call


def compact_records(records: jax.Array, total_records: int, tb: int):
    n = records.size

    def body(records):
        r = records.reshape(1, n)
        valid = (r >= 0) & (r < total_records)
        rank = prefix_sum(valid.astype(jnp.int32), axis=1) - 1
        match = valid & (rank == jnp.arange(n, dtype=jnp.int32)[:, None])
        token = jnp.sum(jnp.where(match, jnp.arange(n)[None, :], 0), axis=1)
        record = jnp.sum(jnp.where(match, r, 0), axis=1)
        boundary = (jnp.arange(n // tb + 1) * tb)[:, None]
        offsets = jnp.sum(
            valid & (jnp.arange(n)[None, :] < boundary), axis=1, dtype=jnp.int32
        )
        return jnp.stack((token, record), axis=1), offsets

    pairs, offsets = fuse_function(body, (records,), name="decode_compact_records")(
        records
    )
    return pairs.reshape(-1), offsets


def unpack_cache_words(words: jax.Array) -> jax.Array:
    chunks, n, width = words.shape
    tile = max(v for v in range(128, 4097, 128) if n % v == 0)

    def kernel(wr, out):
        value = wr[...]
        lo = lax.bitcast_convert_type(value << 16, jnp.float32).astype(jnp.bfloat16)
        hi = lax.bitcast_convert_type(value & jnp.int32(-65536), jnp.float32).astype(
            jnp.bfloat16
        )
        out[0, ...], out[1, ...] = lo, hi

    return pl.pallas_call(
        kernel,
        grid=(chunks, n // tile),
        in_specs=[pl.BlockSpec((1, tile, width), lambda c, i: (c, i, 0))],
        out_specs=pl.BlockSpec((2, 1, tile, width), lambda c, i: (c, 0, i, 0)),
        out_shape=jax.ShapeDtypeStruct((2 * chunks, 1, n, width), jnp.bfloat16),
        name="decode_cache_word_unpack",
    )(words).reshape(2 * chunks, n, width)


def _swa_capacities(banks):
    info = pltpu.get_tpu_info().sparse_core
    assert info is not None
    lanes_cores = info.num_lanes * info.num_cores * info.num_subcores
    return tuple(bank.shape[0] * bank.shape[1] for bank in banks), lanes_cores


def swa_pack(banks, records, values):
    capacities, lanes_cores = _swa_capacities(banks)
    padded = (values.shape[0] + lanes_cores - 1) // lanes_cores * lanes_cores

    def body(records, values):
        result, base = [], 0
        for capacity in capacities:
            result.append(
                pack_matrix(
                    records, values, base=base, capacity=capacity, padded=padded
                )
            )
            base += capacity
        return tuple(result)

    prepared = fuse_function(
        body, (records[None, :], values), name="decode_swa_sort_pack"
    )(records[None, :], values)
    return tuple((p[0].reshape(-1), *p[1:]) for p in prepared)


def swa_prepare_publish(banks, table, starts, positions, minimum, values):
    from vllm_torchtpu.kernels.mla.paged_swa import write_swa_pages

    # Shape dispatch belongs inside the JAX bridge. A Torch-side equality
    # specializes dimensions that the model explicitly marks dynamic.
    if positions.shape != (288,) or table.shape[0] != 48:
        from vllm_torchtpu.kernels.deepseek_v41.csa2.cache import prepare_swa_metadata

        metadata, bounds, records = prepare_swa_metadata(
            table,
            starts,
            positions,
            minimum,
            page_size=64,
            page_stride=64,
            row_offset=0,
        )
        pages = banks if len(banks) == 2 else banks[0]
        updated = write_swa_pages(pages, records, values)
        owners = updated if isinstance(updated, tuple) else (updated,)
        return (*owners, metadata, bounds, records)

    capacities, lanes_cores = _swa_capacities(banks)
    padded = (values.shape[0] + lanes_cores - 1) // lanes_cores * lanes_cores

    def prepare(table, starts, positions, minimum, values):
        metadata, bounds, records = metadata_matrix(table, starts, positions, minimum)
        packed, base = [], 0
        for capacity in capacities:
            packed.append(
                pack_matrix(
                    records, values, base=base, capacity=capacity, padded=padded
                )
            )
            base += capacity
        return metadata, bounds, records, tuple(packed)

    inputs = table, starts[None, :], positions[None, :], minimum[:, None], values
    metadata, bounds, records, packed = fuse_function(
        prepare, inputs, name="decode_swa_metadata_sort_pack"
    )(*inputs)
    records = records.reshape(-1)
    packed = tuple((p[0].reshape(-1), *p[1:]) for p in packed)
    pages = banks if len(banks) == 2 else banks[0]
    updated = write_swa_pages(pages, records, values, prepared=packed)
    owners = updated if isinstance(updated, tuple) else (updated,)
    return (*owners, metadata, bounds, records)


def supports_compressor_state_packed(
    tokens: int,
    table_shape: tuple[int, ...],
    capacity: int,
    page_stride: int,
    row_offset: int,
    window: int,
) -> bool:
    """Measured decode envelope; other geometries retain native metadata."""
    return (tokens, table_shape, capacity, page_stride, row_offset, window) == (
        288,
        (48, 8),
        16,
        16,
        0,
        128,
    )


def _compressor_state_addresses(table, starts, kv_starts, positions, minimum):
    capacity, stride, offset = 16, 16, 0
    tokens, requests = positions.size, table.shape[0]
    rows = jnp.arange(tokens, dtype=jnp.int32)
    r = jnp.arange(requests, dtype=jnp.int32)[:, None]

    def map_requests(boundaries):
        req = jnp.minimum(
            jnp.sum(rows[None, :] >= boundaries[1:, None], axis=0), requests - 1
        )
        return req[None, :] == r

    selected = map_requests(starts)

    def select(values):
        return jnp.sum(jnp.where(selected, values[:, None], 0), axis=0)

    first, end, blocks = select(starts[:-1]), select(starts[1:]), select(table[:, 0])
    has_prev = rows != first
    active = rows < starts[-1]
    slots = jnp.where(active, blocks * capacity + positions % capacity, -1)
    previous = jnp.where(active, blocks * capacity + (positions - 1) % capacity, -1)
    stored = (slots >= 0) & (rows >= end - capacity)
    replay_start = jnp.sum(
        jnp.where(map_requests(kv_starts), minimum[:, None], 0), axis=0
    )
    replay = (
        (replay_start > 0)
        & (positions >= replay_start)
        & (positions < replay_start + 128)
    )
    valid = (rows < kv_starts[-1]) & ((positions + 1) % 2 == 0) & ~replay

    def physical(slot):
        return offset + slot // capacity * stride + slot % capacity

    return (
        has_prev.astype(jnp.int32),
        valid.astype(jnp.int32),
        jnp.where(stored, physical(slots), -1),
        jnp.where(has_prev | (slots < 0), -1, physical(previous)),
    )


def compressor_state_packed(
    table: jax.Array,
    starts: jax.Array,
    kv_starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    *,
    total_records: int,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Decode flags and stable ring tables for one 288-token compressor tile."""
    args = (table, starts, kv_starts, positions, minimum)
    shapes = ((48, 8), (49,), (49,), (288,), (48,))
    if any(
        x.shape != shape or x.dtype != jnp.int32
        for x, shape in zip(args, shapes, strict=True)
    ):
        raise ValueError("Packed compressor metadata requires INT32 decode shapes")

    def compact(records):
        n = records.size
        r = records.reshape(1, n)
        valid = (r >= 0) & (r < total_records)
        rank = prefix_sum(valid.astype(jnp.int32), axis=1) - 1
        match = valid & (rank == jnp.arange(n, dtype=jnp.int32)[:, None])
        token = jnp.sum(jnp.where(match, jnp.arange(n)[None, :], 0), axis=1)
        record = jnp.sum(jnp.where(match, r, 0), axis=1)
        count = jnp.sum(valid, dtype=jnp.int32)
        return jnp.stack((token, record), axis=1), jnp.stack((jnp.int32(0), count))

    def kernel(*refs):
        prev, valid, write, read = _compressor_state_addresses(
            *(r[...] for r in refs[:5])
        )
        outputs = (prev, valid, *compact(write), *compact(read))
        for ref, value in zip(refs[5:], outputs, strict=True):
            ref[...] = value

    n = positions.shape[0]
    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(s, jnp.int32)
            for s in ((n,), (n,), (n, 2), (2,), (n, 2), (2,))
        ),
        name="dsv41_compressor_state_packed",
        compiler_params=pltpu.CompilerParams(
            vmem_limit_bytes=pltpu.get_tpu_info().vmem_capacity_bytes // 2
        ),
    )(*args)
    return (
        outputs[0][:, None],
        outputs[1][:, None],
        outputs[2].reshape(-1),
        outputs[3],
        outputs[4].reshape(-1),
        outputs[5],
    )


def compressor_addresses_packed(
    table: jax.Array,
    starts: jax.Array,
    kv_starts: jax.Array,
    positions: jax.Array,
    minimum: jax.Array,
    main: jax.Array,
    index: jax.Array,
    *,
    total_records: int,
) -> tuple[
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
    jax.Array,
]:
    """Decode state, stable ring tables and ratio-2 main/index addresses."""
    args = (table, starts, kv_starts, positions, minimum, main, index)
    shapes = ((48, 8), (49,), (49,), (288,), (48,), (48, 386), (48, 193))
    if any(
        x.shape != shape or x.dtype != jnp.int32
        for x, shape in zip(args, shapes, strict=True)
    ):
        raise ValueError("Packed compressor addresses require INT32 decode shapes")

    def compact(records):
        r = records.reshape(1, 288)
        valid = (r >= 0) & (r < total_records)
        rank = prefix_sum(valid.astype(jnp.int32), axis=1) - 1
        match = valid & (rank == jnp.arange(288, dtype=jnp.int32)[:, None])
        token = jnp.sum(jnp.where(match, jnp.arange(288)[None, :], 0), axis=1)
        record = jnp.sum(jnp.where(match, r, 0), axis=1)
        return jnp.stack((token, record), axis=1), jnp.stack(
            (jnp.int32(0), jnp.sum(valid, dtype=jnp.int32))
        )

    def kernel(*refs):
        state, ss, ks, pos, mins, mt, it = (r[...] for r in refs[:7])
        prev, valid, write, read = _compressor_state_addresses(state, ss, ks, pos, mins)
        rows = jnp.arange(288, dtype=jnp.int32)
        request = jnp.minimum(jnp.sum(rows[None, :] >= ks[1:, None], axis=0), 47)
        selected = request[None, :] == jnp.arange(48)[:, None]

        def slots(page_table, capacity):
            logical = pos // 2
            col = jnp.minimum(logical // capacity, page_table.shape[1] - 1)
            col = jnp.where(col < 0, col + page_table.shape[1], col)
            indices = jnp.broadcast_to((col % 128)[None, :], (48, 288))
            chunks = (page_table.shape[1] + 127) // 128
            padded = jnp.pad(
                page_table, ((0, 0), (0, chunks * 128 - page_table.shape[1]))
            )
            per_request = jnp.zeros((48, 288), jnp.int32)
            for chunk in range(chunks):
                tile = padded[:, chunk * 128 : (chunk + 1) * 128]
                gathered = take_along(tile, indices, axis=1)
                per_request += jnp.where(col[None, :] // 128 == chunk, gathered, 0)
            blocks = jnp.sum(jnp.where(selected, per_request, 0), axis=0)
            return jnp.where(rows < ks[-1], blocks * capacity + logical % capacity, -1)

        main_slots, index_slots = slots(mt, 128), slots(it, 256)
        records = jnp.where((valid != 0) & (main_slots >= 0), main_slots, -1)
        outputs = (prev, valid, *compact(write), *compact(read), records, index_slots)
        for ref, value in zip(refs[7:], outputs, strict=True):
            ref[...] = value

    outputs = pl.pallas_call(
        kernel,
        out_shape=tuple(
            jax.ShapeDtypeStruct(s, jnp.int32)
            for s in ((288,), (288,), (288, 2), (2,), (288, 2), (2,), (288,), (288,))
        ),
        name="dsv41_compressor_addresses_packed_c128_i256",
        compiler_params=pltpu.CompilerParams(vmem_limit_bytes=16 * 1024 * 1024),
    )(table, starts, kv_starts, positions, minimum, main, index)
    return tuple(
        x.reshape(s)
        for x, s in zip(
            outputs,
            ((288, 1), (288, 1), (576,), (2,), (576,), (2,), (288,), (288,)),
            strict=True,
        )
    )
