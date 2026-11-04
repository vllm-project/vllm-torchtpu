# SPDX-License-Identifier: Apache-2.0
"""Frozen production _compact before next-shape gate expansion.

Copied verbatim from attention_preprocessing/compress_rope.py, 2026-11-04.
Keep the TPU T288 branch: this is NOT an all-shapes scatter baseline.
"""

import jax
import jax.numpy as jnp


def _compact(records, total_records, tiles, tb):
    """Valid ``records`` as flat ``(token row, record)`` pairs in token order,
    plus each tile's ``[offsets[t], offsets[t+1])`` pair range.

    One scatter and no gather: on TPU each lowers to a SparseCore offload of
    several us. Out-of-range records drop.
    """
    if jax.default_backend() == "tpu" and records.shape == (288,) and tiles * tb == 288:
        from vllm_torchtpu.kernels.experimental.decode_preparation import (
            compact_records,
        )

        return compact_records(records, total_records, tb)
    n = records.shape[0]
    valid = (records >= 0) & (records < total_records)
    slots = jnp.where(valid, jnp.cumsum(valid, dtype=jnp.int32) - 1, n)
    pairs = jnp.stack((jnp.arange(n, dtype=jnp.int32), records.astype(jnp.int32)), 1)
    table = jnp.zeros((n, 2), jnp.int32).at[slots].set(pairs, mode="drop")
    per_tile = valid.reshape(tiles, tb).sum(axis=1, dtype=jnp.int32)
    offsets = jnp.concatenate((jnp.zeros((1,), jnp.int32), jnp.cumsum(per_tile)))
    return table.reshape(-1), offsets.astype(jnp.int32)
