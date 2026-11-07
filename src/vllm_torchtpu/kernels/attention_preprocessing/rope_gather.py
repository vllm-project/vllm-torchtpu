# SPDX-License-Identifier: Apache-2.0
"""Blocked RoPE row gathers used by the CSA2 inverse rotation."""

import jax
import jax.numpy as jnp

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
