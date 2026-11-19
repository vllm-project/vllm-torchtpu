# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Streaming top-k at the tuned 1M-token score width against an exact NumPy top-k."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import streaming_topk


@pytest.mark.parametrize("unordered", [False, True])
def test_wide_rows_select_the_exact_top_k(unordered):
    if jax.default_backend() != "tpu":
        pytest.skip("Streaming top-k runs on SparseCore")
    rows, width, k = 16, 1048576, 512
    rng = np.random.default_rng(1)
    # Distinct scores, so the top-k set is unique; ragged and empty rows.
    scores = rng.permutation(rows * width).reshape(rows, width).astype(np.float32)
    lengths = rng.integers(k + 1, 400_000, rows).astype(np.int32)
    lengths[:3] = (0, 300, width)
    selected = np.asarray(
        streaming_topk.streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=k,
            exclude_neg_inf=True,
            unordered=unordered,
            reconstruct_threshold=4 * k,
        )
    )
    for row, length in enumerate(lengths):
        ids = selected[row][selected[row] >= 0]
        expected = np.argsort(-scores[row, :length], kind="stable")[: min(k, length)]
        if unordered:
            ids = np.sort(ids)
        np.testing.assert_array_equal(ids, np.sort(expected))
