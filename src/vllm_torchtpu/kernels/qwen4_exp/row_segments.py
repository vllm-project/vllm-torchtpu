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
