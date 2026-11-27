# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Full-row prefill streaming top-k: stable ties and finite padding contract."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.streaming_topk import (
    streaming_topk,
)


@pytest.mark.parametrize("block,threshold", [(8192, 8192), (16384, 32768)])
@pytest.mark.parametrize("width", [65536, 65408, 66048])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "masked",
    ],
)
def test_streaming_topk_contract(block, threshold, width, distribution):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(17)
    scores = rng.normal(size=(32, width)).astype(np.float32)
    if distribution == "ascending":
        scores[:] = np.arange(width, dtype=np.float32)
    elif distribution == "descending":
        scores[:] = -np.arange(width, dtype=np.float32)
    elif distribution == "equal":
        scores[:] = 1
    elif distribution == "cross_block_ties":
        # Equal Kth values survive repeated candidate compactions, with later
        # blocks also supplying larger winners that displace earlier ties.
        scores[:] = np.arange(width) // 4096 % 5
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = np.float32(-0.0)
    elif distribution == "masked":
        scores[:, ::3] = -np.inf
        scores[:4] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 16383, 50176, width], np.int32), 32
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            valid_lengths=jnp.asarray(lengths),
            k=512,
            block_size=block,
            reconstruct_threshold=threshold,
        )
    )
    for row, length in enumerate(lengths):
        ids = actual[row]
        ids = ids[(ids >= 0) & (scores[row, np.maximum(ids, 0)] > -np.inf)]
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        order = np.lexsort((eligible, ~keys))[:512]
        np.testing.assert_array_equal(np.sort(ids), np.sort(eligible[order]))
        assert np.unique(ids).size == ids.size


@pytest.mark.parametrize("width", [65536, 66048])
@pytest.mark.parametrize("row_start", [0, 3])
@pytest.mark.parametrize("distribution", ["ties", "subnormal", "infinity"])
def test_streaming_topk_preserves_order_and_padding(distribution, row_start, width):
    """HBM windows keep exact FP32 ties, ascending IDs, and a -1 suffix."""
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    k, rows = 512, 32
    if distribution == "ties":
        values = (np.arange(width) // 4096 % 5).astype(np.float32)
    else:
        bits = [0x80000000, 0, 0x80000001, 1, 0x807FFFFF, 0x007FFFFF]
        if distribution == "infinity":
            bits += [0xFF800000, 0x7F800000]
        values = np.resize(np.array(bits, np.uint32).view(np.float32), width)
    scores = np.broadcast_to(values, (rows, width)).copy()
    scores[:, ::7] = -np.inf
    lengths = np.resize(
        np.array([0, 1, 511, 512, 513, 8191, 8193, 65535, width], np.int32), rows
    )
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(
                np.concatenate((np.zeros((row_start, width), np.float32), scores))
            ),
            row_start=row_start,
            row_count=rows,
            valid_lengths=jnp.asarray(lengths),
            k=k,
            reconstruct_threshold=2048,
            exclude_neg_inf=True,
        )
    )
    expected = np.full((rows, k), -1, np.int32)
    for row, length in enumerate(lengths):
        eligible = np.flatnonzero((np.arange(width) < length) & (scores[row] > -np.inf))
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        selected = eligible[np.lexsort((eligible, ~keys))[:k]]
        expected[row, : len(selected)] = np.sort(selected)
    np.testing.assert_array_equal(actual, expected)
