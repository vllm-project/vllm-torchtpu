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


def row_copies_fit(num_tokens: int, rows: int, num_routes: int, hidden: int) -> bool:
    """Whether `copy_send_rows` (``num_tokens`` tokens into ``rows`` send rows)
    and `gather_rows` (``num_routes`` routes from ``rows`` receive rows) of
    bf16 rows fit VMEM."""
    cap = pltpu.get_tpu_info().vmem_capacity_bytes * 0.9
    send = (pl.cdiv(num_tokens, _ROWS_PER_TILE) * _ROWS_PER_TILE + rows) * hidden * 2
    tile = min(_GATHER_TILE, num_routes)
    gather = (rows + 2 * tile) * hidden * 2
    return send + (16 << 20) <= cap and gather + (8 << 20) <= cap


# An fp8 exchange row of H elements travels as uint32 words [8, H / 32]: whole
# native 32-bit tiles, which the row DMAs and the all-to-all move as they are.
# Word (s, l) holds the fp8 values of rows 4s..4s+3 of the row's [32, H / 32]
# view at lane l, low byte first: the order in which a packed bitcast of the
# words to fp8 reads them back.
FP8_ROW_SUBLANES = 8
_FP8_PER_WORD = 4


def fp8_rows_supported(hidden: int) -> bool:
    """Whether fp8 rows of this width travel as whole-tile word blocks."""
    return hidden % (FP8_ROW_SUBLANES * _FP8_PER_WORD * 128) == 0


def fp8_row_words(q):
    """fp8 rows [N, H] as their exchange words [N, 8, H / 32] uint32."""
    num, hidden = q.shape
    lanes = hidden // (FP8_ROW_SUBLANES * _FP8_PER_WORD)
    b = q.reshape(num, FP8_ROW_SUBLANES, _FP8_PER_WORD, lanes)
    return lax.bitcast_convert_type(jnp.swapaxes(b, 2, 3), jnp.uint32)


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


def fp8_row_value(words, scale):
    """An fp8 row (its words [8, H / 32] read back as fp8 [32, H / 32]) times
    its token's scale, as bf16."""
    fp8 = pltpu.bitcast(words, jnp.float8_e4m3fn)
    return (fp8.astype(jnp.float32) * scale).astype(jnp.bfloat16)


def matmul_rows(x, end, start=None):
    """16 bf16 rows [16, H] as `gather_rows` writes them: rows ``start`` (0
    when None) to ``end`` with the one-hot permute's bits (see
    `_canonical_bits`), the others zeros."""
    u = jnp.uint32
    # uint32 sublane j packs bf16 rows 2j (low half) and 2j + 1 (high half).
    sublane = lax.broadcasted_iota(jnp.int32, (_ROWS_PER_TILE // 2, 1), 0)
    lo, hi = 2 * sublane, 2 * sublane + 1
    keep_lo, keep_hi = lo < end, hi < end
    if start is not None:
        keep_lo = jnp.logical_and(keep_lo, start <= lo)
        keep_hi = jnp.logical_and(keep_hi, start <= hi)
    keep = jnp.where(keep_lo, u(0xFFFF), u(0)) | jnp.where(keep_hi, u(0xFFFF0000), u(0))
    w = _canonical_bits(pltpu.bitcast(x, jnp.uint32)) & keep
    return pltpu.bitcast(w, x.dtype)


def _gather_kernel(
    sizes_ref,
    filled_ref,
    *refs,
    tile: int,
    num_rows: int,
    scaled: bool,
):
    if scaled:
        scale_ref, *refs = refs
    else:
        scale_ref = None
    index_hbm, *refs, index_ref, index_sem = refs
    # The index (one word per route, whole-group sized) copies to SMEM while
    # the source rows copy to VMEM, instead of before the kernel starts.
    index_copy = pltpu.make_async_copy(index_hbm, index_ref, index_sem.at[0])
    index_copy.start()
    _gather_body(
        index_ref,
        sizes_ref,
        filled_ref,
        scale_ref,
        *refs,
        tile=tile,
        num_rows=num_rows,
        index_copy=index_copy,
    )


def _gather_body(
    index_ref,
    sizes_ref,
    filled_ref,
    scale_ref,
    src_ref,
    out_ref,
    rows_vmem,
    out_buf,
    copy_sem,
    out_sems,
    *,
    tile: int,
    num_rows: int,
    index_copy=None,
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
    with jax.named_scope("gather_copy_in"):
        lax.fori_loop(0, pieces, lambda k, c: (copy(k).start(), c)[1], 0)
        lax.fori_loop(0, pieces, lambda k, c: (copy(k).wait(), c)[1], 0)
        if index_copy is not None:
            index_copy.wait()

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

    def rows_block(t, slot, c):
        base = t * tile + c * _ROWS_PER_TILE
        rows = []
        for j in range(_ROWS_PER_TILE):
            pos = base + j
            row = jnp.where(pos < n, index_ref[jnp.minimum(pos, num_rows - 1)], 0)
            if scale_ref is None:
                rows.append(rows_vmem[row])
            else:
                rows.append(fp8_row_value(rows_vmem[row], scale_ref[row]))
        x = jnp.stack(rows).reshape(_ROWS_PER_TILE, hidden)
        # Rows past the gathered ones come out as zeros.
        out_buf[slot, pl.ds(c * _ROWS_PER_TILE, _ROWS_PER_TILE)] = matmul_rows(
            x, n - base
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

    with jax.named_scope("gather_rows"):
        lax.fori_loop(0, num_tiles, step, 0)

    @pl.when(num_tiles >= 2)
    def _():
        store_op(num_tiles - 2, lax.rem(num_tiles - 2, 2), "wait")

    @pl.when(num_tiles >= 1)
    def _():
        store_op(num_tiles - 1, lax.rem(num_tiles - 1, 2), "wait")


@functools.partial(jax.jit, static_argnames=("tile",))
def gather_rows(src, index, sizes, filled, scale=None, *, tile: int = _GATHER_TILE):
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

    ``scale`` [R] float32: ``src`` holds fp8 rows as their words [R, 8, H /
    32] (`fp8_row_words`) and the result is bf16 ``fp8 row index[j] *
    scale[index[j]]`` (the same zero, subnormal and non-finite handling).
    """
    num_src, sub, lanes = src.shape
    hidden = sub * lanes
    (num_rows,) = index.shape
    scaled = scale is not None
    if scaled:
        hidden *= _FP8_PER_WORD
        if (
            sub != FP8_ROW_SUBLANES
            or src.dtype != jnp.uint32
            or not fp8_rows_supported(hidden)
        ):
            raise ValueError(
                f"gather_rows takes fp8 row words uint32 [R, 8, H / 32], got "
                f"{src.dtype}{src.shape}"
            )
    elif sub != ROW_SUBLANES or not row_copy_supported(src.dtype, hidden):
        raise ValueError(
            f"gather_rows takes bf16 [R, 16, H / 16] rows, got {src.shape}"
        )
    out_dtype = jnp.bfloat16 if scaled else src.dtype
    if num_rows % _ROWS_PER_TILE or tile % _ROWS_PER_TILE:
        raise ValueError(f"gather_rows needs whole 16-row tiles: {num_rows}, {tile}")
    tile = min(tile, num_rows)
    vmem = (num_src * src.dtype.itemsize + 2 * tile * 2) * hidden
    vmem_limit = vmem + (8 << 20)
    if vmem_limit > pltpu.get_tpu_info().vmem_capacity_bytes * 0.9:
        raise ValueError(f"gather_rows: {num_src} source rows do not fit VMEM")
    hbm = pl.BlockSpec(memory_space=pltpu.HBM)
    # The kernel copies the rows itself: XLA must not stage them in VMEM.
    src = pltpu.with_memory_space_constraint(src, pltpu.HBM)
    prefetch = (sizes, filled) + ((scale.astype(jnp.float32),) if scaled else ())
    index = pltpu.with_memory_space_constraint(index.astype(jnp.int32), pltpu.HBM)
    return pl.pallas_call(
        functools.partial(_gather_kernel, tile=tile, num_rows=num_rows, scaled=scaled),
        # In HBM, where the expert kernel reads it. Left to XLA, the result
        # lands in VMEM and is copied to HBM after the kernel (a whole [P, H]
        # copy, 3.8 us at P = 640, H = 8192), though only the routed rows are
        # written.
        out_shape=pltpu.HBM((num_rows, hidden), out_dtype),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=len(prefetch),
            in_specs=[hbm, hbm],
            out_specs=hbm,
            scratch_shapes=[
                pltpu.VMEM((num_src, sub, lanes), src.dtype),
                pltpu.VMEM((2, tile, hidden), out_dtype),
                pltpu.SemaphoreType.DMA((1,)),
                pltpu.SemaphoreType.DMA((2,)),
                pltpu.SMEM((num_rows,), jnp.int32),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            disable_bounds_checks=True, vmem_limit_bytes=vmem_limit
        ),
        name=f"row_copy_gather-p_{num_rows}-r_{num_src}-h_{hidden}-t_{tile}"
        + ("-f8" if scaled else ""),
    )(*prefetch, index, src)


def _send_kernel(
    num_tokens, index_ref, count_ref, x_ref, out_ref, stage, rows_vmem, sem
):
    _send_body(
        num_tokens, index_ref, count_ref, None, x_ref, out_ref, stage, rows_vmem, sem
    )


def _send_fp8_kernel(
    num_tokens, index_ref, count_ref, inv_ref, x_ref, out_ref, stage, rows_vmem, sem
):
    _send_body(
        num_tokens, index_ref, count_ref, inv_ref, x_ref, out_ref, stage, rows_vmem, sem
    )


def _stage_rows(num_tokens, inv_ref, x_ref, stage):
    """Grid step ``c``'s block of token rows into ``stage``: token ``t``'s row
    as stage rows [sub t, sub t + sub), bf16 or (``inv_ref``) its fp8 words."""
    c = pl.program_id(0)
    fp8 = inv_ref is not None
    sub = FP8_ROW_SUBLANES if fp8 else ROW_SUBLANES
    lanes = stage.shape[1]

    def relayout(block):
        num_rows = block.shape[0]
        if fp8:
            # Each row times its token's inverse scale, as fp8 [32, H / 32],
            # packed into its words [8, H / 32] (`fp8_row_words`' order).
            x = block.astype(jnp.float32)
            x = jnp.where(jnp.isfinite(x), x, 0.0)
            x = x.reshape(num_rows, _FP8_PER_WORD * sub, lanes)
            first = c * _ROWS_PER_TILE
            x = jnp.stack(
                [
                    x[i] * inv_ref[jnp.minimum(first + i, num_tokens - 1)]
                    for i in range(num_rows)
                ]
            )
            block = pltpu.bitcast(x.astype(jnp.float8_e4m3fn), jnp.uint32)
        else:
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


def _send_body(
    num_tokens, index_ref, count_ref, inv_ref, x_ref, out_ref, stage, rows_vmem, sem
):
    sub = FP8_ROW_SUBLANES if inv_ref is not None else ROW_SUBLANES
    _stage_rows(num_tokens, inv_ref, x_ref, stage)

    @pl.when(pl.program_id(0) == pl.num_programs(0) - 1)
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


@functools.partial(jax.jit, static_argnames=("rows",))
def copy_send_rows_fp8(x, inv_scale, index, count, rows: int):
    """`copy_send_rows` for the fp8 exchange: row ``r`` of the [rows, 8, H /
    32] uint32 result is the words (`fp8_row_words`) of token ``index[r]``'s
    row of ``x`` [S, H] bf16 times ``inv_scale[index[r]]`` [S] float32, as
    fp8 e4m3 (non-finite values as zeros), for ``r < count[0]``; later rows
    are not written. The kernel quantizes and packs ``x`` in VMEM once and
    copies rows there."""
    num_tokens, hidden = x.shape
    if not (row_copy_supported(x.dtype, hidden) and fp8_rows_supported(hidden)):
        raise ValueError(f"copy_send_rows_fp8 takes bf16 rows, got {x.shape}")
    sub = FP8_ROW_SUBLANES
    lanes = hidden // (sub * _FP8_PER_WORD)
    num_chunks = pl.cdiv(num_tokens, _ROWS_PER_TILE)
    vmem = (num_chunks * _ROWS_PER_TILE + rows) * hidden + (16 << 20)
    if vmem > pltpu.get_tpu_info().vmem_capacity_bytes * 0.9:
        raise ValueError(f"copy_send_rows_fp8: {rows} rows do not fit VMEM")
    out = pl.pallas_call(
        functools.partial(_send_fp8_kernel, num_tokens),
        out_shape=pltpu.HBM((rows * sub, lanes), jnp.uint32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(num_chunks,),
            in_specs=[pl.BlockSpec((_ROWS_PER_TILE, hidden), lambda c, *_: (c, 0))],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_chunks * _ROWS_PER_TILE * sub, lanes), jnp.uint32),
                pltpu.VMEM((rows * sub, lanes), jnp.uint32),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=vmem,
        ),
        name=f"row_copy_send_fp8-s_{num_tokens}-r_{rows}-h_{hidden}",
    )(index, count, inv_scale.astype(jnp.float32), x)
    return out.reshape(rows, sub, lanes)


def _quantize_fp8_kernel(num_tokens, inv_ref, x_ref, out_ref, stage, sem):
    c = pl.program_id(0)
    _stage_rows(num_tokens, inv_ref, x_ref, stage)
    sub = FP8_ROW_SUBLANES
    whole, rest = divmod(num_tokens, _ROWS_PER_TILE)

    def copy(chunk, num_rows):
        first = pl.multiple_of(chunk * _ROWS_PER_TILE * sub, _ROWS_PER_TILE * sub)
        return pltpu.make_async_copy(
            stage.at[pl.ds(first, num_rows * sub)],
            out_ref.at[pl.ds(first, num_rows * sub)],
            sem.at[0],
        )

    # Each block's words leave as soon as they are staged.
    if whole:
        pl.when(c < whole)(lambda: copy(c, _ROWS_PER_TILE).start())
    if rest:
        pl.when(c == whole)(lambda: copy(whole, rest).start())

    @pl.when(c == pl.num_programs(0) - 1)
    def _():
        for _ in range(whole):
            copy(0, _ROWS_PER_TILE).wait()
        if rest:
            copy(whole, rest).wait()


@jax.jit
def fp8_source_rows(x, inv_scale):
    """Every token row of ``x`` [S, H] bf16 as the fp8 words `copy_send_rows_fp8`
    sends for it: row ``t`` of the [S, 8, H / 32] uint32 result is
    ``copy_send_rows_fp8``'s row for ``index = t``, in HBM, where DMAs read it.
    """
    num_tokens, hidden = x.shape
    if not (row_copy_supported(x.dtype, hidden) and fp8_rows_supported(hidden)):
        raise ValueError(f"fp8_source_rows takes bf16 rows, got {x.shape}")
    sub = FP8_ROW_SUBLANES
    lanes = hidden // (sub * _FP8_PER_WORD)
    num_chunks = pl.cdiv(num_tokens, _ROWS_PER_TILE)
    out = pl.pallas_call(
        functools.partial(_quantize_fp8_kernel, num_tokens),
        out_shape=pltpu.HBM((num_tokens * sub, lanes), jnp.uint32),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(num_chunks,),
            in_specs=[pl.BlockSpec((_ROWS_PER_TILE, hidden), lambda c, *_: (c, 0))],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[
                pltpu.VMEM((num_chunks * _ROWS_PER_TILE * sub, lanes), jnp.uint32),
                pltpu.SemaphoreType.DMA((1,)),
            ],
        ),
        compiler_params=pltpu.CompilerParams(
            dimension_semantics=("arbitrary",),
            disable_bounds_checks=True,
            vmem_limit_bytes=num_chunks * _ROWS_PER_TILE * hidden + (16 << 20),
        ),
        name=f"row_copy_fp8_source-s_{num_tokens}-h_{hidden}",
    )(inv_scale.astype(jnp.float32), x)
    return out.reshape(num_tokens, sub, lanes)
