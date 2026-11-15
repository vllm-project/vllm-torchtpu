# SPDX-License-Identifier: Apache-2.0
"""``jnp.searchsorted`` over short sorted integer boundaries, without a while loop.

On TPU ``jnp.searchsorted`` lowers to a while loop of gathers (one iteration per
bisection step, two u32 fusions each). The QSA prefill path maps every token row
to its request with it in each of the 12 QSA layers, and inside the per-layer
selection ``conditional`` CSE cannot share the result. A c32 xprof (j-7b72fb03)
shows 121 such fusions per step: 1.8 ms at 1024 rows, 0.9 ms at 256.

For a non-decreasing integer ``boundaries`` array, ``searchsorted(side="right")``
is the number of entries ``<= x`` and ``side="left"`` the number ``< x``. A
``[len(x), len(boundaries)]`` compare-and-sum gives the same int32 result
exactly, in a few vector ops. VLLM_TPU_SORTED_SEARCH_MASK (default off) selects
it; above ``MAX_MASK_ELEMENTS`` the search is kept.
"""

import jax.numpy as jnp

from vllm_torchtpu import envs

# Largest [queries, boundaries] mask (8192 rows x 512 boundaries).
MAX_MASK_ELEMENTS = 1 << 22


def searchsorted(boundaries, x, side="left"):
    """``jnp.searchsorted(boundaries, x, side=side)`` for sorted integer boundaries."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    if (
        not envs.VLLM_TPU_SORTED_SEARCH_MASK
        or boundaries.ndim != 1
        or not jnp.issubdtype(boundaries.dtype, jnp.integer)
        or not jnp.issubdtype(jnp.asarray(x).dtype, jnp.integer)
        or boundaries.shape[0] * max(jnp.size(x), 1) > MAX_MASK_ELEMENTS
    ):
        return jnp.searchsorted(boundaries, x, side=side)
    x = jnp.asarray(x)
    xs = x[..., None]
    hit = boundaries <= xs if side == "right" else boundaries < xs
    return jnp.sum(hit, axis=-1, dtype=jnp.int32)
