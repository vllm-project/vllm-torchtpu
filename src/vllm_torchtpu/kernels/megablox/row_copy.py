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
"""Row copies between whole-tile rows and matmul rows for the ragged
all-to-all EP exchange.

The exchange moves each bf16 row of H elements as one [16, H / 16] block of
whole native tiles, so a row can land at any offset of a buffer. A matmul
needs rows on sublanes ([rows, H]). These kernels copy only the rows that are
needed and relayout them in VMEM.
"""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Sublanes of one exchange row: a bf16 row of H elements is one [16, H / 16]
# block of whole tiles.
ROW_SUBLANES = 16
# Rows per copy and per output DMA of `gather_rows` and `copy_send_rows`.
_GATHER_TILE = 64
# bf16 rows per native tile.
_ROWS_PER_TILE = 16


def row_copy_supported(dtype, hidden: int) -> bool:
    """Whether rows of this dtype and width travel as whole-tile blocks."""
    return jnp.dtype(dtype) == jnp.bfloat16 and hidden % (ROW_SUBLANES * 128) == 0


def _canonical_bits(w):
    """The one-hot permute's result for two packed bf16 values per uint32: +0
    for zeros, subnormals, infinities and NaNs (the permute's isfinite select
    and its matmul), every other value unchanged."""
    u = jnp.uint32
    # Per half: exponent field + 1, so that 255 wraps to 0; fields 2..255 keep.
    keep = (((w & u(0x7F807F80)) + u(0x00800080)) & u(0x7F007F00)) + u(0x7F007F00)
    keep = keep & u(0x80008000)
    # 0x8000 -> 0x0000FFFF and 0x80000000 -> 0xFFFF0000, in one subtraction.
    return w & (jnp.left_shift(keep, u(1)) - jnp.right_shift(keep, u(15)))


def _gather_kernel(
    index_ref,
    sizes_ref,
    filled_ref,
    src_ref,
    out_ref,
    rows_vmem,
    out_buf,
    copy_sem,
    out_sems,
    *,
    tile: int,
    num_rows: int,
):
    n = sizes_ref[0]
    for i in range(1, sizes_ref.shape[0]):
        n = n + sizes_ref[i]
    num_tiles = (n + tile - 1) // tile
    num_src = rows_vmem.shape[0]
    chunk = min(_GATHER_TILE, num_src)
    last = (num_rows - 1) // tile
    tail = num_rows - last * tile
    hidden = out_buf.shape[-1]

    # The source rows that hold routed tokens come in whole chunks (the last one
    # ends at the buffer's end), so every row is read from VMEM afterwards.
    def copy(k):
        start = jnp.minimum(k * chunk, num_src - chunk)
        return pltpu.make_async_copy(
            src_ref.at[pl.ds(start, chunk)],
            rows_vmem.at[pl.ds(start, chunk)],
            copy_sem.at[0],
        )

    pieces = jnp.where(num_tiles > 0, (filled_ref[0] + chunk - 1) // chunk, 0)
    lax.fori_loop(0, pieces, lambda k, c: (copy(k).start(), c)[1], 0)
    lax.fori_loop(0, pieces, lambda k, c: (copy(k).wait(), c)[1], 0)

    def store(t, slot, size):
        return pltpu.make_async_copy(
            out_buf.at[slot, pl.ds(0, size)],
            out_ref.at[pl.ds(t * tile, size)],
            out_sems.at[slot],
        )

    def store_op(t, slot, op):
        if tail == tile:
            getattr(store(t, slot, tile), op)()
            return

        @pl.when(t != last)
        def _():
            getattr(store(t, slot, tile), op)()

        @pl.when(t == last)
        def _():
            getattr(store(t, slot, tail), op)()

    u = jnp.uint32

    def rows_block(t, slot, c):
        base = t * tile + c * _ROWS_PER_TILE
        rows = []
        for j in range(_ROWS_PER_TILE):
            pos = base + j
            row = jnp.where(pos < n, index_ref[jnp.minimum(pos, num_rows - 1)], 0)
            rows.append(rows_vmem[row])
        x = jnp.stack(rows).reshape(_ROWS_PER_TILE, hidden)
        # Rows past the gathered ones come out as zeros. uint32 sublane j packs
        # bf16 rows 2j (low half) and 2j + 1 (high half).
        sublane = lax.broadcasted_iota(jnp.int32, (_ROWS_PER_TILE // 2, 1), 0)
        v = n - base
        keep = jnp.where(2 * sublane < v, u(0xFFFF), u(0))
        keep = keep | jnp.where(2 * sublane + 1 < v, u(0xFFFF0000), u(0))
        w = _canonical_bits(pltpu.bitcast(x, jnp.uint32)) & keep
        out_buf[slot, pl.ds(c * _ROWS_PER_TILE, _ROWS_PER_TILE)] = pltpu.bitcast(
            w, out_buf.dtype
        )

    def step(t, carry):
        slot = lax.rem(t, 2)

        @pl.when(t >= 2)
        def _():
            store_op(t - 2, slot, "wait")

        for c in range(tile // _ROWS_PER_TILE):
            pl.when(c * _ROWS_PER_TILE < n - t * tile)(
                functools.partial(rows_block, t, slot, c)
            )
        store_op(t, slot, "start")
        return carry

    lax.fori_loop(0, num_tiles, step, 0)

    @pl.when(num_tiles >= 2)
    def _():
        store_op(num_tiles - 2, lax.rem(num_tiles - 2, 2), "wait")

    @pl.when(num_tiles >= 1)
    def _():
        store_op(num_tiles - 1, lax.rem(num_tiles - 1, 2), "wait")


@functools.partial(jax.jit, static_argnames=("tile",))
def gather_rows(src, index, sizes, filled, *, tile: int = _GATHER_TILE):
    """Matmul rows of the routes' token rows: ``out[j] = src[index[j]]`` for
    ``j < sum(sizes)``, as `fused_moe_scatter_planned`'s one-hot permute
    computes them (zeros, subnormals and non-finite values become +0).

    ``src`` [R, 16, H / 16] bf16 holds one token row per major index, the
    gathered ones below ``filled[0]``; ``index`` [P] int32; ``sizes`` int32
    [G], whose sum is the number of rows to gather (the route plan's group
    sizes). Returns [P, H]: rows from ``sum(sizes)`` to the end of their
    16-row block are zeros, later rows are not written. P must be a multiple
    of 16. The kernel copies ``src[:filled]`` to VMEM in whole chunks and
    reads each row from there.
    """
    num_src, sub, lanes = src.shape
    hidden = sub * lanes
    (num_rows,) = index.shape
    if sub != ROW_SUBLANES or not row_copy_supported(src.dtype, hidden):
        raise ValueError(
            f"gather_rows takes bf16 [R, 16, H / 16] rows, got {src.shape}"
        )
    if num_rows % _ROWS_PER_TILE or tile % _ROWS_PER_TILE:
        raise ValueError(f"gather_rows needs whole 16-row tiles: {num_rows}, {tile}")
    tile = min(tile, num_rows)
    vmem = (num_src + 2 * tile) * hidden * src.dtype.itemsize
    vmem_limit = vmem + (8 << 20)
    if vmem_limit > pltpu.get_tpu_info().vmem_capacity_bytes * 0.9:
        raise ValueError(f"gather_rows: {num_src} source rows do not fit VMEM")
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    # The kernel copies the rows itself: XLA must not stage them in VMEM.
    src = pltpu.with_memory_space_constraint(src, pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_gather_kernel, tile=tile, num_rows=num_rows),
        out_shape=jax.ShapeDtypeStruct((num_rows, hidden), src.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            in_specs=[hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((num_src, sub, lanes), src.dtype),
                pltpu.VMEM((2, tile, hidden), src.dtype),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((2,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, vmem_limit_bytes=vmem_limit
        ),
        name=f"row_copy_gather-p_{num_rows}-r_{num_src}-h_{hidden}-t_{tile}",
    )(index, sizes, filled, src)


def _send_kernel(
    num_tokens, index_ref, count_ref, x_ref, out_ref, stage, rows_vmem, sem
):
    c = pl.program_id(0)
    sub = ROW_SUBLANES
    lanes = stage.shape[1]

    def relayout(block):
        # Token t's row is stage rows [16 t, 16 t + 16): whole tiles.
        num_rows = block.shape[0]
        block = block.reshape(num_rows, sub, lanes)
        start = pl.multiple_of(c * _ROWS_PER_TILE * sub, _ROWS_PER_TILE * sub)
        stage[pl.ds(start, num_rows * sub)] = block.reshape(num_rows * sub, lanes)

    whole, rest = divmod(num_tokens, _ROWS_PER_TILE)
    if not rest:
        relayout(x_ref[...])
    else:
        if whole:
            pl.when(c < whole)(lambda: relayout(x_ref[...]))
        # The last block holds only ``rest`` rows of the array. Where the
        # array is placed in VMEM the block is read in place, and the rows
        # after them would be read past the array's end.
        pl.when(c == whole)(lambda: relayout(x_ref[pl.ds(0, rest)]))

    @pl.when(c == pl.num_programs(0) - 1)
    def _():
        n = count_ref[0]

        def copy_row(r, carry):
            src = pl.multiple_of(index_ref[r] * sub, sub)
            dst = pl.multiple_of(r * sub, sub)
            rows_vmem[pl.ds(dst, sub)] = stage[pl.ds(src, sub)]
            return carry

        # The filled rows leave in whole chunks (the last one ends at the
        # buffer's end), each as soon as its rows are copied.
        rows = rows_vmem.shape[0] // sub
        chunk = min(_GATHER_TILE, rows)

        def copy(k):
            first = pl.multiple_of(jnp.minimum(k * chunk, rows - chunk) * sub, sub)
            return pltpu.make_async_copy(
                rows_vmem.at[pl.ds(first, chunk * sub)],
                out_ref.at[pl.ds(first, chunk * sub)],
                sem.at[0],
            )

        def copy_chunk(k, carry):
            lax.fori_loop(k * chunk, jnp.minimum((k + 1) * chunk, n), copy_row, 0)
            copy(k).start()
            return carry

        pieces = (n + chunk - 1) // chunk
        lax.fori_loop(0, pieces, copy_chunk, 0)
        lax.fori_loop(0, pieces, lambda k, z: (copy(k).wait(), z)[1], 0)


@functools.partial(jax.jit, static_argnames=("rows",))
def copy_send_rows(x, index, count, rows: int):
    """The send buffer of the ragged all-to-all: row ``r`` of the [rows, 16,
    H / 16] result is token ``index[r]``'s row of ``x`` [S, H] as one block of
    whole tiles, for ``r < count[0]``; later rows are not written. The kernel
    relayouts ``x`` in VMEM once and copies rows there.
    """
    num_tokens, hidden = x.shape
    if not row_copy_supported(x.dtype, hidden):
        raise ValueError(
            f"copy_send_rows takes bf16 rows of 2048k elements, got {x.shape}"
        )
    sub = ROW_SUBLANES
    lanes = hidden // sub
    num_chunks = pl.cdiv(num_tokens, _ROWS_PER_TILE)
    vmem = (num_chunks * _ROWS_PER_TILE + rows) * hidden * x.dtype.itemsize
    vmem_limit = vmem + (16 << 20)
    if vmem_limit > pltpu.get_tpu_info().vmem_capacity_bytes * 0.9:
        raise ValueError(f"copy_send_rows: {rows} rows do not fit VMEM")
    out = pl.pallas_call(
        functools.partial(_send_kernel, num_tokens),
        # In HBM, where the ragged all-to-all reads it.
        out_shape=pltpu.HBM((rows * sub, lanes), x.dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=2,
            grid=(num_chunks,),
            in_specs=[pl.BlockSpec((_ROWS_PER_TILE, hidden), lambda c, *_: (c, 0))],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_chunks * _ROWS_PER_TILE * sub, lanes), x.dtype),
                pltpu.VMEM((rows * sub, lanes), x.dtype),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem_limit,
        ),
        name=f"row_copy_send-s_{num_tokens}-r_{rows}-h_{hidden}",
    )(index, count, x)
    return out.reshape(rows, sub, lanes)
