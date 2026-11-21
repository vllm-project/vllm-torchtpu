# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block-maximum top-k: ties, masked keys, short rows, forced and published blocks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.block_topk import (
    block_topk,
    default_blocks,
    is_supported,
)

K, BLOCK, PUBLISH = 512, 8, 2048
# Empty, at most k, fewer than the scanned blocks' keys (4224), and full rows.
LENGTHS = [0, 1, 7, 511, 512, 513, 4223, 4224, 4225, 5001, 8191, 20000, 49407, 49408]
CASES = [(32, 49408, d) for d in ("random", "relu", "signed_zero", "masked")]
# 1M-token contexts (ratio 2 and 1): rows stream from HBM; the widest is
# unsupported and must fall back.
CASES += [(32, 524288, "random"), (32, 1048576, "masked")]


def _scores(distribution, rows, width, rng):
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    if distribution == "relu":
        scores = np.maximum(scores - 1, 0)  # Most keys tie at zero.
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::3] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, (np.arange(width) // 256) % 3 == 1] = -np.inf
    return scores


@pytest.mark.parametrize("publish", [0, PUBLISH])
@pytest.mark.parametrize("rows,width,distribution", [*CASES, (2048, 8192, "random")])
def test_block_topk(rows, width, distribution, publish):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore top-k requires TPU")
    if publish and width // BLOCK < publish:
        pytest.skip("Publication needs more blocks than it publishes")
    if not is_supported(rows, width, K, BLOCK):
        pytest.skip("Block maxima of this width do not fit SparseCore VMEM")
    rng = np.random.default_rng(23)
    scores = _scores(distribution, rows, width, rng)
    fixed = rows == 32 and width == LENGTHS[-1]
    lengths = np.resize(LENGTHS, rows) if fixed else rng.integers(0, width + 1, rows)
    lengths = lengths.astype(np.int32)
    scores[np.arange(width)[None, :] >= lengths[:, None]] = -np.inf
    result = block_topk(
        jnp.asarray(scores),
        jnp.asarray(lengths),
        k=K,
        block_size=BLOCK,
        publish=publish,
    )
    selected = np.asarray(result[0] if publish else result)
    for row, length in enumerate(lengths):
        ids, visible = selected[row][selected[row] >= 0], scores[row, :length]
        rest = visible > -np.inf
        assert np.unique(ids).size == ids.size == min(K, rest.sum())
        assert (visible[ids] > -np.inf).all()
        rest[ids] = False
        if ids.size and rest.any():
            assert visible[ids].min() >= visible[rest].max()
        if publish:
            # The highest block maxima in radix order (+0 above -0), the newest
            # visible block forced first, lowest IDs on ties, ascending.
            blocks = scores[row].reshape(-1, BLOCK)
            maxima = blocks.max(-1)
            positive_zero = ((blocks == 0) & ~np.signbit(blocks)).any(-1)
            maxima[(maxima == 0) & positive_zero] = np.float32(0.0)
            maxima[(length - 1) // BLOCK] = np.inf
            blocks = np.flatnonzero(maxima[: -(-length // BLOCK)] > -np.inf)
            values = maxima[blocks]
            order = np.lexsort((blocks, np.signbit(values), -values))[:publish]
            expected = np.full(publish, -1, np.int32)
            expected[: order.size] = np.sort(blocks[order])
            np.testing.assert_array_equal(np.asarray(result[1])[row], expected)


def _block_maxima(scores, block=BLOCK):
    """Block maxima in radix order: a block holding +0 has maximum +0."""
    blocks = scores.reshape(scores.shape[0], -1, block)
    maxima = blocks.max(-1)
    positive_zero = ((blocks == 0) & ~np.signbit(blocks)).any(-1)
    maxima[(maxima == 0) & positive_zero] = np.float32(0.0)
    return maxima


@pytest.mark.parametrize("block,maxima_block", [(BLOCK, None), (2 * BLOCK, BLOCK)])
@pytest.mark.parametrize("publish", [0, PUBLISH])
@pytest.mark.parametrize("rows,width,distribution", CASES[:5])
def test_given_maxima_match_reduced(
    rows, width, distribution, publish, block, maxima_block
):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore top-k requires TPU")
    if not (
        is_supported(rows, width, K, block)
        and is_supported(
            rows, width, K, block, given_maxima=True, maxima_block=maxima_block
        )
    ):
        pytest.skip("Block maxima of this width do not fit SparseCore VMEM")
    rng = np.random.default_rng(29)
    scores = _scores(distribution, rows, width, rng)
    lengths = rng.integers(0, width + 1, rows).astype(np.int32)
    lengths[:4] = (0, 1, 513, width)
    scores[np.arange(width)[None, :] >= lengths[:, None]] = -np.inf
    # Given maxima may be finer than the blocks; block top-k folds them.
    given_block = maxima_block or block
    maxima = _block_maxima(scores, given_block)
    # Maxima of blocks past a row's length are never read.
    starts = np.arange(maxima.shape[1]) * given_block // block * block
    maxima[starts[None, :] >= lengths[:, None]] = np.nan

    def run(**extra):
        return block_topk(
            jnp.asarray(scores),
            jnp.asarray(lengths),
            k=K,
            block_size=block,
            publish=publish,
            **extra,
        )

    reduced = run()
    given = run(maxima=jnp.asarray(maxima), maxima_block=maxima_block)
    for a, b in zip(jax.tree.leaves(reduced), jax.tree.leaves(given)):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _ordered(values):
    """FP32 radix order as unsigned bits: +0 above -0, -inf lowest."""
    bits = values.astype(np.float32).view(np.uint32)
    sign = (bits.view(np.int32) >> 31).view(np.uint32)
    return (bits ^ (sign | np.uint32(1 << 31))).astype(np.int64)


def _top(bits, count):
    """Positions of the `count` largest finite bits, ties in position order."""
    finite = bits > _ordered(np.float32(-np.inf))
    order = np.lexsort((np.arange(bits.size), -bits))
    return np.sort(order[finite[order]][:count])


def _expected_selection(scores, maxima, length, block, publish):
    """The kernel's exact output: the top blocks by maximum, then the top keys
    among theirs, ties to the lowest IDs, ascending and -1 padded."""
    live = -(-length // block)
    bits = _ordered(maxima[:live])
    if publish and live:
        bits[live - 1] = _ordered(np.float32(np.inf))
    chosen = _top(bits, default_blocks(K, 16, forced=bool(publish)))
    ids = (chosen[:, None] * block + np.arange(block)).ravel()
    ids = ids[ids < length]
    selected = ids[_top(_ordered(scores[ids]), K)]
    expected = np.full(K, -1, np.int32)
    expected[: selected.size] = selected
    return expected


# Served decode tiles: 16 rows, one request's 6 verify rows live. Ratio-2
# layers (block 8 maxima) and 1M-token ratio-1 layers (block 16 folded from
# 8), at contexts of about 110k-370k tokens.
SERVED = [(524288, 8, length) for length in (55000, 112000, 185000)]
SERVED += [(1048576, 16, length) for length in (110000, 370000)]


@pytest.mark.parametrize("distribution", ["random", "relu", "masked", "narrow"])
@pytest.mark.parametrize("publish", [0, PUBLISH])
@pytest.mark.parametrize("width,block,length", SERVED)
def test_served_decode_matches_reference(width, block, length, publish, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore top-k requires TPU")
    maxima_block = BLOCK
    if publish and block != maxima_block:
        pytest.skip("Folded selection does not publish")
    if block != maxima_block and distribution == "relu":
        pytest.skip("Folding signed-zero maxima may keep either zero")
    rows, live = 16, 6
    rng = np.random.default_rng(length + publish)
    scores = np.full((rows, width), -np.inf, np.float32)
    lengths = np.zeros(rows, np.int32)
    lengths[:live] = length + np.arange(live)
    for row in range(live):
        n = lengths[row]
        if distribution == "narrow":
            # One top radix byte: every block survives its first pivot.
            scores[row, :n] = 1 + rng.random(n, dtype=np.float32)
        else:
            scores[row, :n] = _scores(distribution, 1, n, rng)[0]
    maxima = _block_maxima(scores, maxima_block)
    result = block_topk(
        jnp.asarray(scores),
        jnp.asarray(lengths),
        k=K,
        block_size=block,
        publish=publish,
        maxima=jnp.asarray(maxima),
        maxima_block=None if block == maxima_block else maxima_block,
    )
    selected = np.asarray(result[0] if publish else result)
    folded = maxima.reshape(rows, -1, block // maxima_block).max(-1)
    for row in range(rows):
        expected = _expected_selection(
            scores[row], folded[row], lengths[row], block, publish
        )
        np.testing.assert_array_equal(selected[row], expected)
