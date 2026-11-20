# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Candidate block maxima: the tuned kernel against the masked reduction."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection import publication


def _reference(scores, lengths, block_size):
    rows, width = scores.shape
    visible = np.arange(width)[None, :] < lengths[:, None]
    blocks = np.where(visible, scores, -np.inf).reshape(rows, -1, block_size).max(-1)
    last = (lengths - 1) // block_size
    blocks[np.arange(blocks.shape[1])[None, :] == last[:, None]] = np.inf
    return blocks


@pytest.mark.parametrize("rows", [16, 128])
def test_block_maxima_ignore_columns_past_each_length(rows):
    if jax.default_backend() != "tpu":
        pytest.skip("The tuned block-maximum kernel requires TPU")
    width, block_size = 1048576, 8
    rng = np.random.default_rng(5)
    # Empty rows, a block boundary, the second window, the last column.
    lengths = rng.integers(1, 300_000, rows)
    lengths[:4] = (0, 8 * 4096, 32768 + 1, width)
    lengths[-2:] = 0
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    # Columns past a length hold whatever the scorer left there.
    scores[np.arange(width)[None, :] >= lengths[:, None]] = np.nan
    actual = publication.candidate_block_scores(
        jnp.asarray(scores), jnp.asarray(lengths, jnp.int32), block_size=block_size
    )
    np.testing.assert_array_equal(
        np.asarray(actual), _reference(scores, lengths, block_size)
    )


def test_visible_block_scores_match_masked_reduction():
    width, block_size, rows = 65536, 8, 8
    rng = np.random.default_rng(11)
    lengths = rng.integers(1, width, rows)
    lengths[:4] = (0, 8, 9, width)
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    scores[np.arange(width)[None, :] >= lengths[:, None]] = -np.inf
    maxima = scores.reshape(rows, -1, block_size).max(-1)
    # Blocks past a length may hold anything.
    blocks = np.arange(maxima.shape[1])[None, :]
    maxima[blocks >= -(-lengths[:, None] // block_size)] = np.nan
    actual = publication.visible_block_scores(
        jnp.asarray(maxima), jnp.asarray(lengths, jnp.int32), block_size=block_size
    )
    np.testing.assert_array_equal(
        np.asarray(actual), _reference(scores, lengths, block_size)
    )
