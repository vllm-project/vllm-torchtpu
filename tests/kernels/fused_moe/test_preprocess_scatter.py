# SPDX-License-Identifier: Apache-2.0
"""Single-device coverage for exact token/scale paired scatter."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.fused_moe.v2 import host

CAPACITY = 128


@pytest.mark.parametrize(
    "tokens,topk,experts",
    [(256, 4, 32), (255, 4, 32), (2304, 6, 384)],
)
def test_merged_scatter_matches_indexed_add(tokens, topk, experts):
    """Check small, partial-chunk and serving shapes, including duplicate rows."""
    from vllm_torchtpu.kernels.fused_moe.v2.scatter import (
        can_scatter_pair,
        scatter_pair,
    )

    if not jax.devices()[0].device_kind.startswith("TPU7"):
        pytest.skip("SparseCore scatter requires v7x")
    size = host.ragged_stride_bound(tokens, topk, experts, CAPACITY)
    scale_size = host.act_scale_slab_rows(size) * host.HIDDEN_LANE_BLOCK
    n = tokens * topk
    assert can_scatter_pair(n, size, scale_size)
    rng = np.random.default_rng(42)
    rows = rng.integers(0, scale_size + 128, n, dtype=np.int32)
    rows[:64] = 1
    scales = rng.integers(-(2**31), 2**31 - 1, n, dtype=np.int32)

    @jax.jit
    def reference(rows, scales):
        token_ids = jnp.arange(n, dtype=jnp.int32) // topk
        t = jnp.zeros(size, jnp.int32).at[rows].add(token_ids, mode="drop")
        s = jnp.zeros(scale_size, jnp.int32).at[rows].add(scales, mode="drop")
        return jnp.clip(t, 0, tokens - 1), s

    actual = jax.jit(lambda r, s: scatter_pair(r, s, size, scale_size, topk))(
        rows, scales
    )
    expected = reference(rows, scales)
    for result, ref in zip(actual, expected):
        np.testing.assert_array_equal(np.asarray(result), np.asarray(ref))
    assert not can_scatter_pair(1_000_000, size, scale_size)
