# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical coverage for the single-request packed prefill scorer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import indexer


@pytest.mark.cpu_test
@pytest.mark.parametrize("tiles", [1, 2, 3, 16, 17, 32])
def test_bounded_pipeline_preserves_tile_order_and_outputs(tiles):
    """Drain every tile once, including the paired-loop fallback above 16."""
    values = jnp.arange(tiles * 4, dtype=jnp.float32).reshape(tiles, 4)
    score_groups, selection_groups = [], []

    def run(values):
        def score(value, group=None):
            score_groups.append(group)
            return value * value

        def select(i, scores, group=None):
            selection_groups.append(group)
            return scores.max(), i

        def write(i, selected, output):
            return tuple(dst.at[i].set(src) for dst, src in zip(output, selected))

        return indexer.pipeline_score_topk_bounded(
            tiles,
            lambda i: values[i],
            score,
            select,
            write,
            (jnp.zeros(tiles), jnp.full(tiles, -1, jnp.int32)),
        )

    scores, order = jax.jit(run)(values)
    np.testing.assert_array_equal(scores, (values * values).max(axis=1))
    np.testing.assert_array_equal(order, jnp.arange(tiles))
    if tiles <= 16:
        assert score_groups[0] is None
        assert selection_groups[-1] is None
        assert score_groups[1:] == selection_groups[:-1]
        assert len(set(score_groups[1:])) == tiles - 1


@pytest.mark.cpu_test
def test_bounded_pipeline_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one tile"):
        indexer.pipeline_score_topk_bounded(0, None, None, None, None, None)
