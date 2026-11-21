# SPDX-License-Identifier: Apache-2.0
"""Copy contiguous row segments of a ``[rows, ...]`` array with DMAs."""

import functools

import jax
import jax.numpy as jnp
from jax import lax
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _pieces(src_ref, dst_ref, sem, src_row, dst_row, length, bits):
    """One DMA per set bit of ``length``: static sizes, dynamic offsets, and no
    row outside the segment is read or written."""
    pieces = []
    for k in reversed(range(bits)):
        size = 1 << k
        done = length & ~((size << 1) - 1)  # rows the larger pieces cover
        copy = pltpu.make_async_copy(
            src_ref.at[pl.ds(src_row + done, size)],
            dst_ref.at[pl.ds(dst_row + done, size)],
            sem,
        )
        pieces.append(((length >> k) & 1, copy))
    return pieces


def _kernel(src_row_ref, dst_row_ref, length_ref, src_ref, _, dst_ref, sem, *, bits):
    def each(action):
        def body(s, carry):
            for on, copy in _pieces(
                src_ref,
                dst_ref,
                sem,
                src_row_ref[s],
                dst_row_ref[s],
                length_ref[s],
                bits,
            ):
                pl.when(on == 1)(lambda copy=copy: action(copy))
            return carry

        lax.fori_loop(0, src_row_ref.shape[0], body, 0)

    # Every segment's DMAs in flight at once, then drained in the same order.
    each(lambda copy: copy.start())
    each(lambda copy: copy.wait())


def copy_row_segments(
    src, dst, src_row, dst_row, length, *, max_length, interpret=False
):
    """``dst`` with ``dst[dst_row[s] + j] = src[src_row[s] + j]`` for ``j < length[s]``.

    One kernel issues up to ``bit_length(max_length)`` DMAs per segment, all
    in flight on one semaphore (QSA prefill: a few dozen segments per call).
    Segments must lie inside both arrays and not overlap in ``dst``; rows no
    segment covers keep ``dst``'s values. HBM to HBM, so the row dimension must
    be untiled: ``src`` and ``dst`` need at least 3 dimensions.
    """
    if src.ndim < 3 or dst.shape[1:] != src.shape[1:] or dst.dtype != src.dtype:
        raise ValueError(
            f"row segments of {src.shape} {src.dtype} into {dst.shape} {dst.dtype}"
        )
    # No piece can exceed a segment, and no segment either array.
    bits = max(1, int(min(max_length, src.shape[0], dst.shape[0])).bit_length())
    return pl.pallas_call(
        functools.partial(_kernel, bits=bits),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(),
            in_specs=[
                pl.BlockSpec(memory_space=pltpu.HBM),
                pl.BlockSpec(memory_space=pltpu.HBM),
            ],
            out_specs=pl.BlockSpec(memory_space=pltpu.HBM),
            scratch_shapes=[pltpu.SemaphoreType.DMA],
        ),
        out_shape=jax.ShapeDtypeStruct(dst.shape, dst.dtype),
        input_output_aliases={4: 0},
        interpret=interpret,
        name="copy_row_segments",
    )(
        src_row.astype(jnp.int32),
        dst_row.astype(jnp.int32),
        length.astype(jnp.int32),
        src,
        dst,
    )


_ROW_TILE = 8  # rows per HBM tile of a 32-bit [rows, width] array


def _block_kernel(
    start_ref, shift_ref, count_ref, src_ref, out_ref, window, sem, *, fill
):
    b = pl.program_id(0)
    copy = pltpu.make_async_copy(
        src_ref.at[pl.ds(pl.multiple_of(start_ref[b], _ROW_TILE), window.shape[0])],
        window,
        sem,
    )
    copy.start()
    copy.wait()
    rows = out_ref.shape[0]
    shift = (window.shape[0] - shift_ref[b]) % window.shape[0]
    x = pltpu.roll(window[...], shift, 0)[:rows]
    keep = lax.broadcasted_iota(jnp.int32, x.shape, 0) < count_ref[b]
    out_ref[...] = jnp.where(keep, x, jnp.asarray(fill, x.dtype))


def align_row_blocks(src, first, count, *, block, fill, interpret=False):
    """``out[b * block + j] = src[first[b] + j]`` for ``j < count[b]``, else ``fill``.

    For a ``[rows]`` or ``[rows, width]`` array, whose row dimension is tiled
    in HBM so a DMA cannot start at any row: each block DMAs a tile-aligned
    window of ``block + 8`` rows into VMEM and rotates its first row into
    place. Needs ``first[b] + count[b] <= rows``. VMEM per step: the window
    plus two output blocks and the rotate / select temporaries, about
    5 x block x width x 4 bytes (5 MiB at 512 x 512), inside the default
    scoped limit for the QSA shapes.
    """
    vector = src.ndim == 1
    if vector:
        src = src[:, None]
    if src.ndim != 2 or src.dtype.itemsize != 4 or block % _ROW_TILE:
        raise ValueError(f"row blocks of {src.shape} by {block}")
    rows = max(-(-src.shape[0] // _ROW_TILE) * _ROW_TILE, block + _ROW_TILE)
    if rows != src.shape[0]:
        src = jnp.pad(src, ((0, rows - src.shape[0]), (0, 0)))
    window = block + _ROW_TILE
    # The window [start, start + window) holds rows [first, first + count).
    start = jnp.clip(first - first % _ROW_TILE, 0, rows - window)
    shift = jnp.clip(first - start, 0, window - 1)
    out = pl.pallas_call(
        functools.partial(_block_kernel, fill=fill),
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=3,
            grid=(first.shape[0],),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec((block, src.shape[1]), lambda b, *_: (b, 0)),
            scratch_shapes=[
                pltpu.VMEM((window, src.shape[1]), src.dtype),
                pltpu.SemaphoreType.DMA,
            ],
        ),
        compiler_params=pltpu.CompilerParams(dimension_semantics=("arbitrary",)),
        out_shape=jax.ShapeDtypeStruct(
            (first.shape[0] * block, src.shape[1]), src.dtype
        ),
        interpret=interpret,
        name="align_row_blocks",
    )(start.astype(jnp.int32), shift.astype(jnp.int32), count.astype(jnp.int32), src)
    return out[:, 0] if vector else out
