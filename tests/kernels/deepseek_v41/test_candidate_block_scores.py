# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Publication masks partial blocks and always retains the newest visible block."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import (
    candidate_block_scores,
)


@pytest.mark.parametrize("width", [98816, 49408])
def test_candidate_block_scores_partial_lengths(width):
    if jax.default_backend() != "tpu":
        pytest.skip("Exercises the tuned TensorCore publication kernel")
    rows, block = 96, 8
    rng = np.random.default_rng(826)
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    scores[::3] = -np.inf
    scores[1::3] = 0
    lengths = np.tile(
        [0, 1, 7, 8, 9, 127, 128, 129, width - 1, width, width + 1, -1], rows // 12
    ).astype(np.int32)
    expected = np.full((rows, width // block), -np.inf, np.float32)
    for row, length in enumerate(np.clip(lengths, 0, width)):
        for start in range(0, length, block):
            expected[row, start // block] = scores[
                row, start : min(start + block, length)
            ].max()
        if length:
            expected[row, (length - 1) // block] = np.inf
    actual = candidate_block_scores(
        jnp.asarray(scores), jnp.asarray(lengths), block_size=block
    )
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), expected.view(np.uint32)
    )
