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
