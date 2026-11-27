# SPDX-License-Identifier: Apache-2.0
"""RoPE rows for the attention-preprocessing kernels.

The model's ``cos_sin_cache`` (the model's ``_tpu_rope_cache``) is
``[max_position, rotary_dim]`` FP32 with ``[cos | sin]`` packed side by side.
Every layer of one forward step uses the same positions, so the caller gathers
the ``[tokens, rotary_dim]`` rows once per step (``gather_rows``) and every
kernel reads them as an ordinary tiled input. An in-kernel DMA gather was
measured slower: about 0.1 us per contiguous run, so decode batches and
non-contiguous positions cost up to 2-3x the kernel itself.
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

_LANE = 128


def _quantize_rows(x):
    values = x.astype(jnp.float32)
    limit = float(jnp.finfo(jnp.float8_e4m3fn).max)
    scale = jnp.max(jnp.abs(values), axis=-1, keepdims=True) / limit
    scale = jnp.where(scale == 0, 1.0, scale)
    encoded = jnp.clip(values / scale, -limit, limit)
    return encoded.astype(jnp.float8_e4m3fn), scale


def _lane_gather(operand, indices):
    rank = operand.ndim
    lanes = indices.shape[0]
    coords = jnp.broadcast_to(indices, (*operand.shape[:-1], lanes))[..., None]
    batching = tuple(range(rank - 1))
    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(),
        collapsed_slice_dims=(rank - 1,),
        start_index_map=(rank - 1,),
        operand_batching_dims=batching,
        start_indices_batching_dims=batching,
    )
    return jax.lax.gather(
        operand,
        coords,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1,) * rank,
        unique_indices=False,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


# Measured on tpu7x for FP32 rows of a [98816, 64] table (all rows consumed):
# one gather of 240 rows 4.6 us, of 288 rows 13.3 us, of 512 rows 13.2 us, of
# 2048 rows 7.8 us; 288 rows as gathers of 256 + 32 rows 7.2 us.
_ROW_BLOCK = 256
_ROW_BLOCKS_UP_TO = 1024


def by_row_blocks(gather, rows: jax.Array, blocked: bool) -> jax.Array:
    """``gather(rows)``, issued as 256-row gathers for 256 < rows <= 1024."""
    count = rows.shape[0]
    if not blocked or count <= _ROW_BLOCK or count > _ROW_BLOCKS_UP_TO:
        return gather(rows)
    return jnp.concatenate(
        [gather(rows[i : i + _ROW_BLOCK]) for i in range(0, count, _ROW_BLOCK)]
    )


def _clamped_rows(cos_sin_cache, positions, blocked):
    clamped = jnp.clip(positions, 0, cos_sin_cache.shape[0] - 1)
    return by_row_blocks(
        lambda rows: jnp.take(cos_sin_cache, rows, axis=0), clamped, blocked
    )


def gather_rows(cos_sin_cache: jax.Array, positions: jax.Array) -> jax.Array:
    """``[tokens, rotary_dim]`` rows for ``positions``, clamped to the table."""
    return _clamped_rows(cos_sin_cache, positions, False)


def gather_rows_blocked(cos_sin_cache: jax.Array, positions: jax.Array) -> jax.Array:
    """``gather_rows`` issued through ``by_row_blocks``; bitwise equal."""
    return _clamped_rows(cos_sin_cache, positions, True)


def gather_paired_rows(pairs: jax.Array, positions: jax.Array) -> jax.Array:
    """``gather_rows`` from a table packed two rows per 128-lane row.

    ``pairs`` is ``[ceil(max_position / 2), 2 * rotary_dim]``: row ``i`` holds
    table rows ``2i`` and ``2i + 1`` side by side, an odd table padded with a
    copy of its last row (so clamping to the padded length is still exact).
    A 64-wide FP32 row fills half a lane block, and XLA lowers that gather
    about 10x slower than a full-lane one (11.2 -> 1.0 us at 288 tokens).
    """
    clamped = _clamp_paired(pairs, positions)
    return _select_half(jnp.take(pairs, clamped >> 1, axis=0), clamped)


# A row DMA costs about 11 ns to issue on tpu7x (paired 128-lane FP32 rows,
# random positions: 72 rows 2.0 us, 288 rows 4.2, 2048 rows 23.4), while a
# SparseCore gather pays ~21 us of latency at decode sizes (72 rows) and slows
# concurrent TensorCore DMAs (DSpark main_proj 29.4 -> 46.8 us). Row DMAs are
# used while the rows fit one SparseCore gather block (512 rows, ~6.8 us).
_DMA_ROWS_UP_TO = 512


def gather_paired_rows_device(pairs: jax.Array, positions: jax.Array) -> jax.Array:
    """``gather_paired_rows`` with the row fetch on the fastest exact engine.

    Up to ``_DMA_ROWS_UP_TO`` tokens (decode and draft steps) the TensorCore
    issues one DMA per pair row. Larger counts run as one SparseCore row-DMA
    program (``ragged_gather_v2``), an async offload the TensorCore does not
    wait on until the rows are consumed. Falls back to ``gather_paired_rows``
    off TPU. The even/odd half select stays an XLA fusion. Bit-exact.
    """
    if not _on_tpu():
        return gather_paired_rows(pairs, positions)
    clamped = _clamp_paired(pairs, positions)
    if positions.shape[0] <= _DMA_ROWS_UP_TO:
        rows = _dma_rows(pairs, clamped >> 1)
    elif _has_sparse_core():
        from vllm_torchtpu.kernels.sparse_core.ragged_gather_v2 import ragged_gather_v2

        rows = ragged_gather_v2(
            pairs,
            clamped >> 1,
            jnp.zeros((1,), jnp.int32),
            jnp.full((1,), positions.shape[0], jnp.int32),
        )
    else:
        return gather_paired_rows(pairs, positions)
    return _select_half(rows, clamped)


def _dma_rows_kernel(rows_ref, table_ref, out_ref, sem):
    def issue(t, carry):
        pltpu.make_async_copy(
            table_ref.at[pl.ds(rows_ref[t], 1)], out_ref.at[pl.ds(t, 1)], sem.at[0]
        ).start()
        return carry

    def wait(t, carry):
        pltpu.make_async_copy(
            table_ref.at[pl.ds(0, 1)], out_ref.at[pl.ds(0, 1)], sem.at[0]
        ).wait()
        return carry

    jax.lax.fori_loop(0, out_ref.shape[0], issue, 0, unroll=8)
    jax.lax.fori_loop(0, out_ref.shape[0], wait, 0, unroll=8)


def _dma_rows(table: jax.Array, rows: jax.Array) -> jax.Array:
    """``table[rows]`` as one TensorCore DMA per row; ``rows`` in bounds."""
    count, width = rows.shape[0], table.shape[1]
    return pl.pallas_call(
        _dma_rows_kernel,
        grid_spec=pltpu.PrefetchScalarGridSpec(
            num_scalar_prefetch=1,
            grid=(1,),
            in_specs=[pl.BlockSpec(memory_space=pltpu.HBM)],
            out_specs=pl.BlockSpec((count, width), lambda i, rows: (0, 0)),
            scratch_shapes=[pltpu.SemaphoreType.DMA((1,))],
        ),
        out_shape=jax.ShapeDtypeStruct((count, width), table.dtype),
        name="rope_row_dma",
    )(rows, table)


def _clamp_paired(pairs: jax.Array, positions: jax.Array) -> jax.Array:
    return jnp.clip(positions, 0, 2 * pairs.shape[0] - 1)


def _select_half(rows: jax.Array, clamped: jax.Array) -> jax.Array:
    """Table row ``clamped`` from its packed pair row."""
    half = rows.shape[1] // 2
    even = ((clamped & 1) == 0)[:, None]
    return jnp.where(even, rows[:, :half], rows[:, half:])


def _on_tpu() -> bool:
    return jax.devices()[0].platform == "tpu"


def _has_sparse_core() -> bool:
    # Check the device, not only the backend name: host-traced TPU branches
    # have no TPU device kind to query (same check as cache._sparse_core_rows).
    return _on_tpu() and pltpu.get_tpu_info().sparse_core is not None


def coefficients(cos_sin):
    """``[rows, rotary_dim]`` packed rows -> 128-lane cos and sign-folded sin.

    The rotated channels are the tail of the last lane block; NoPE lanes get
    the identity (cos 1, sin 0). Pair partners are lanes ``i`` and ``i ^ 1``;
    the GPT-J sign (negative on even lanes) is folded into sin, which is exact.
    """
    rotary_dim = cos_sin.shape[-1]
    half = rotary_dim // 2
    lane = jnp.arange(_LANE)
    channel = lane - (_LANE - rotary_dim)
    is_rot = channel >= 0
    freq = jnp.maximum(channel, 0) >> 1
    sign = jnp.where((lane & 1) == 0, -1.0, 1.0)
    cos = jnp.where(is_rot, _lane_gather(cos_sin, freq), 1.0)
    sin = jnp.where(is_rot, _lane_gather(cos_sin, half + freq), 0.0) * sign
    return cos, sin


def check_rope_rows(rope_rows, tokens, head_dim):
    if rope_rows.ndim != 2 or rope_rows.dtype != jnp.float32:
        raise ValueError("rope_rows must be FP32 [tokens, rotary_dim]")
    if rope_rows.shape[0] != tokens:
        raise ValueError("rope_rows needs one row per token")
    rotary_dim = rope_rows.shape[1]
    if rotary_dim % 2 or not 0 < rotary_dim <= _LANE or rotary_dim >= head_dim:
        raise ValueError("Rotary dim must be even, at most one lane block")
