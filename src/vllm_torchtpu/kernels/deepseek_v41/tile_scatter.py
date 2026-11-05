# SPDX-License-Identifier: Apache-2.0
"""``table[slots] = values`` into a long 1-D int32 table, in place.

XLA's scatter costs in proportion to the table, not to the updates (measured:
288 updates into s32[30198976] take ~40 us, into s32[3326928] ~5 us). This
kernel reads each 128-entry tile that an update lands in, applies every update
of that tile in order, and writes the tile back, so its cost follows the
number of runs of updates landing in one tile (a decode step's tokens of one
request are one run). A tile is the smallest DMA it can issue: a 1-D HBM
slice must start and end on a 128-entry boundary.

Semantics equal ``table.at[where(slots >= 0, slots, size)].set(values,
mode="drop")`` as XLA runs it on TPU: negative and out-of-range slots are
dropped and, among duplicate slots, the last update wins (checked by
tests/kernels/test_dsv41_tile_scatter.py against the XLA scatter).
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

LANES = 128
_BLOCK_ROWS = 96
# The update loop is O(updates x distinct tiles); above this, XLA's
# table-sized pass is the cheaper of the two.
MAX_UPDATES = 1024


def _kernel(slots_ref, values_ref, *refs, size):
    body = size - size % LANES
    if body < size:
        tail_in, _table_in, table, tail, rows, starts_ref, count_ref, sem = refs
        for e in range(size - body):
            tail[e] = tail_in[e]
    else:
        _table_in, table, rows, starts_ref, count_ref, sem = refs
    num_updates = slots_ref.shape[0]

    def tile_copy(i, to_rows):
        hbm = table.at[pl.ds(pl.multiple_of(starts_ref[i], LANES), LANES)]
        src, dst = (hbm, rows.at[i]) if to_rows else (rows.at[i], hbm)
        return pltpu.make_async_copy(src, dst, sem)

    def gather(k, carry):
        # One read per run of updates in the same tile. A tile revisited later
        # is read again; every copy applies all of the tile's updates, so the
        # copies written back are identical.
        count, previous = carry
        slot = slots_ref[k]
        start = slot // LANES * LANES
        new = (slot >= 0) & (slot < body) & (start != previous)

        @pl.when(new)
        def _():
            starts_ref[count] = start
            tile_copy(count, to_rows=True).start()

        if body < size:

            @pl.when((slot >= body) & (slot < size))
            def _():
                tail[slot - body] = values_ref[k]

        return count + new.astype(jnp.int32), jnp.where(new, start, previous)

    count, _ = jax.lax.fori_loop(
        0, num_updates, gather, (jnp.int32(0), jnp.int32(-1)), unroll=8
    )
    count_ref[0] = count

    @pl.loop(0, count)
    def _(i):
        tile_copy(i, to_rows=True).wait()

    lane = jax.lax.broadcasted_iota(jnp.int32, (_BLOCK_ROWS, LANES), 1)
    row = jax.lax.broadcasted_iota(jnp.int32, (_BLOCK_ROWS, LANES), 0)
    for base in range(0, rows.shape[0], _BLOCK_ROWS):

        @pl.when(base < count_ref[0])
        def _(base=base):
            def tile_start(r, col):
                live = base + r < count_ref[0]
                start = jnp.where(live, starts_ref[base + r], -1)
                return jnp.where(row == r, start, col)

            starts = jax.lax.fori_loop(
                0, _BLOCK_ROWS, tile_start, jnp.full_like(lane, -1), unroll=8
            )
            entry = jnp.where(starts >= 0, starts + lane, -1)

            def apply(k, tile):
                # In update order, so the last duplicate wins. Rows past the
                # count match only dropped slots and are never written back.
                return jnp.where(entry == slots_ref[k], values_ref[k], tile)

            block_rows = rows[pl.ds(base, _BLOCK_ROWS), :]
            rows[pl.ds(base, _BLOCK_ROWS), :] = jax.lax.fori_loop(
                0, num_updates, apply, block_rows, unroll=8
            )

    @pl.loop(0, count_ref[0])
    def _(i):
        tile_copy(i, to_rows=False).start()

    @pl.loop(0, count_ref[0])
    def _(i):
        tile_copy(i, to_rows=False).wait()


def tile_scatter(table: jax.Array, slots: jax.Array, values: jax.Array) -> jax.Array:
    """``table[slots] = values`` (see the module docstring); ``table`` aliased."""
    if table.ndim != 1 or table.dtype != jnp.int32 or table.shape[0] < LANES:
        raise ValueError(f"tile_scatter needs a 1-D int32 table, got {table}")
    if values.shape != slots.shape or values.dtype != jnp.int32:
        raise ValueError(f"values {values.shape} must match slots {slots.shape}")
    size = table.shape[0]
    if slots.shape[0] > MAX_UPDATES:
        return table.at[jnp.where(slots >= 0, slots, size)].set(values, mode="drop")
    body = size - size % LANES
    num_rows = -(-slots.shape[0] // _BLOCK_ROWS) * _BLOCK_ROWS
    smem = pl.BlockSpec(memory_space=pltpu.SMEM)
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    table_out = jax.ShapeDtypeStruct(table.shape, table.dtype)
    scratch = [
        pltpu.VMEM((num_rows, LANES), jnp.int32),
        pltpu.SMEM((num_rows,), jnp.int32),
        pltpu.SMEM((1,), jnp.int32),
        pltpu.SemaphoreType.DMA(()),
    ]
    kernel = functools.partial(_kernel, size=size)
    if body == size:
        return pl.pallas_call(
            kernel,
            out_shape=table_out,
            in_specs=[smem, smem, hbm],
            out_specs=hbm,
            scratch_shapes=scratch,
            input_output_aliases={2: 0},
            name="tile_scatter",
        )(slots, values, table)
    # The partial last tile is shorter than the smallest DMA; it goes through
    # SMEM and back in place.
    table, tail = pl.pallas_call(
        kernel,
        out_shape=(table_out, jax.ShapeDtypeStruct((size - body,), jnp.int32)),
        in_specs=[smem, smem, smem, hbm],
        out_specs=(hbm, smem),
        scratch_shapes=scratch,
        input_output_aliases={3: 0},
        name="tile_scatter",
    )(slots, values, table[body:], table)
    return jax.lax.dynamic_update_slice(table, tail, (body,))
