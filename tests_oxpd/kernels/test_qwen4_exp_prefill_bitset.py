# SPDX-License-Identifier: Apache-2.0
"""Scatter-free selection bitset for QSA block-sparse prefill."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    _scatter_bitset,
    _selection_bitset,
)


def _expected(blocks, lengths, *, ratio, groups, words, query_block, width):
    bitset = _scatter_bitset(
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        ratio=ratio,
        groups=groups,
        words=words,
    )
    bitset = np.asarray(bitset).reshape(-1, query_block, words).transpose(0, 2, 1)
    return np.pad(bitset, ((0, 0), (0, 0), (0, width - query_block)))


def _actual(blocks, lengths, **kwargs):
    return np.asarray(
        jax.jit(functools.partial(_selection_bitset, interpret=True, **kwargs))(
            jnp.asarray(blocks), jnp.asarray(lengths)
        )
    )


def _topk(rng, tokens, k, groups, lengths, ratio):
    """Distinct completed groups per row with a -1 suffix, like the indexer."""
    blocks = np.full((tokens, k), -1, np.int32)
    for row, length in enumerate(lengths):
        complete = length // ratio
        count = min(k, complete)
        if count:
            blocks[row, :count] = rng.choice(complete, count, replace=False)
    return blocks


def _geometry(ratio, kv_tile, groups):
    num_kv_tiles = (groups * ratio + kv_tile - 1) // kv_tile
    words_per_tile = kv_tile // (ratio * 32)
    return ((num_kv_tiles * words_per_tile + 127) // 128) * 128


CASES = [
    # (tokens, k, groups, query_block, kv_tile)
    (64, 16, 600, 32, 128),
    (256, 40, 1000, 64, 128),
    (512, 31, 4100, 256, 128),
    (256, 64, 2048, 128, 256),
    (96, 8, 37, 8, 128),
]


@pytest.mark.parametrize("tokens,k,groups,query_block,kv_tile", CASES)
@pytest.mark.parametrize("seed", [0, 1])
def test_indexer_like_selections_match_scatter(
    tokens, k, groups, query_block, kv_tile, seed
):
    ratio = 4
    rng = np.random.default_rng(seed)
    words = _geometry(ratio, kv_tile, groups)
    width = (query_block + 127) // 128 * 128
    # Ragged lengths, padding rows (length 0), exact multiples of ratio and
    # tails on the last group.
    lengths = rng.integers(0, groups * ratio, tokens).astype(np.int32)
    lengths[rng.random(tokens) < 0.1] = 0
    multiple = rng.random(tokens) < 0.1
    lengths[multiple] = lengths[multiple] // ratio * ratio
    lengths[: min(tokens, 3)] = groups * ratio - np.arange(min(tokens, 3))
    blocks = _topk(rng, tokens, k, groups, lengths, ratio)
    kwargs = dict(
        ratio=ratio, groups=groups, words=words, query_block=query_block, width=width
    )
    np.testing.assert_array_equal(
        _actual(blocks, lengths, **kwargs), _expected(blocks, lengths, **kwargs)
    )


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_adversarial_selections_match_scatter(seed):
    """Duplicates, interior -1, out-of-range IDs and all-empty blocks."""
    ratio, groups, query_block, tokens, k = 4, 1500, 32, 160, 24
    rng = np.random.default_rng(seed)
    words = _geometry(ratio, 128, groups)
    blocks = rng.integers(-3, groups + 40, (tokens, k)).astype(np.int32)
    # Duplicates, including a tail-group duplicate and a whole repeated word.
    blocks[:, 1] = blocks[:, 0]
    blocks[5, :] = 7
    blocks[6, :] = np.arange(k) % 32 + 64
    lengths = rng.integers(0, groups * ratio + 9, tokens).astype(np.int32)
    blocks[9, 0] = lengths[9] // ratio
    # A query block with no valid selection at all.
    blocks[32:64] = -1
    lengths[32:64] = 0
    # Extreme groups.
    blocks[10, :2] = (0, groups - 1)
    kwargs = dict(
        ratio=ratio, groups=groups, words=words, query_block=query_block, width=128
    )
    np.testing.assert_array_equal(
        _actual(blocks, lengths, **kwargs), _expected(blocks, lengths, **kwargs)
    )


def test_word_carries_match_scatter_add():
    """Duplicate bits add with carry, exactly like the scatter-add."""
    ratio, groups, query_block = 4, 256, 8
    words = _geometry(ratio, 128, groups)
    blocks = np.full((8, 5), -1, np.int32)
    blocks[0] = (3, 3, 3, 31, 31)
    blocks[1] = (31, 31, 63, 63, 0)
    lengths = np.full((8,), 4 * 200 + 1, np.int32)
    kwargs = dict(
        ratio=ratio, groups=groups, words=words, query_block=query_block, width=128
    )
    actual = _actual(blocks, lengths, **kwargs)
    np.testing.assert_array_equal(actual, _expected(blocks, lengths, **kwargs))
    # Three copies of bit 3, and two of bit 31 that carry out of the word.
    assert actual[0, 0, 0] == 3 * (1 << 3)
    assert actual[0, 6, 0] == 1 << (200 % 32)


def test_empty_input():
    kwargs = dict(ratio=4, groups=64, words=128, query_block=8, width=128)
    blocks = np.full((16, 4), -1, np.int32)
    lengths = np.zeros((16,), np.int32)
    actual = _actual(blocks, lengths, **kwargs)
    assert actual.shape == (2, 128, 128)
    assert not actual.any()
