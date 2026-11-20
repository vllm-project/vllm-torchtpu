# SPDX-License-Identifier: Apache-2.0
"""The Pallas block-max kernel at every tuned shape against the jnp reference."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.deepseek_v41.index_selection import tuned_block_sizes
from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import (
    candidate_block_scores,
)

pytestmark = pytest.mark.tpu_test


def _lengths(rows, width, block_size, block_tile):
    """Empty, one token, partial and whole first blocks, the tile edges, a
    partial last block, the full width and past it; repeated over the rows."""
    tile = block_tile * block_size
    edge = width // tile * tile
    lengths = [0, 1, block_size - 1, block_size, block_size + 1, tile, tile + 1]
    lengths += [edge, width - block_size // 2, width - 1, width, width + 1]
    return np.resize(np.asarray(lengths, np.int32), rows)


def _reference(scores, lengths, block_size):
    rows, width = scores.shape
    blocks = -(-width // block_size)
    lengths = np.clip(lengths, 0, width)
    masked = np.where(np.arange(width)[None, :] < lengths[:, None], scores, -np.inf)
    padded = np.full((rows, blocks * block_size), -np.inf, np.float32)
    padded[:, :width] = masked
    reduced = padded.reshape(rows, blocks, block_size).max(-1)
    last = (lengths - 1) // block_size
    return np.where(np.arange(blocks)[None, :] == last[:, None], np.inf, reduced)


@pytest.mark.parametrize(
    "key",
    sorted(tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG),
    ids=lambda k: f"gen{k[0]}-rows{k[1]}-width{k[2]}-block{k[3]}",
)
def test_tuned_shape_takes_the_kernel_and_matches_the_reference(key):
    if jax.default_backend() != "tpu":
        pytest.skip("the block-max kernel needs a TPU")
    generation, rows, width, block_size = key
    if pltpu.get_tpu_info().generation != generation:
        pytest.skip(f"tuned for TPU generation {generation}")
    _, block_tile = tuned_block_sizes.PUBLISH_BLOCK_MAX_CONFIG[key]
    scores = jax.random.normal(jax.random.key(rows), (rows, width), jnp.float32)
    lengths = jnp.asarray(_lengths(rows, width, block_size, block_tile))

    lowered = candidate_block_scores.lower(scores, lengths, block_size=block_size)
    assert "indexer_candidate_block_max" in lowered.as_text()
    got = candidate_block_scores(scores, lengths, block_size=block_size)
    np.testing.assert_array_equal(
        np.asarray(got),
        _reference(np.asarray(scores), np.asarray(lengths), block_size),
    )
