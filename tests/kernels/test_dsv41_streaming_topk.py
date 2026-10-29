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
@pytest.mark.parametrize("width", [32768, 49408, 98816, 65536, 65408])
@pytest.mark.parametrize(
    "distribution",
    [
        "random",
        "ascending",
        "descending",
        "equal",
        "cross_block_ties",
        "signed_zero",
        "subnormal",
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
    elif distribution == "subnormal":
        patterns = np.array(
            [0, 1, 0x80000000, 0x80000001, 0x007FFFFF, 0x807FFFFF, 0x10000, 0x80010000],
            np.uint32,
        )
        scores[:] = np.resize(patterns, width).view(np.float32)
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


@pytest.mark.parametrize("distribution", ["random", "ties", "signed_zero", "subnormal"])
@pytest.mark.parametrize("start_request", [None, 2])
def test_unordered_topk_with_unsorted_block_mapping(distribution, start_request):
    if jax.default_backend() != "tpu":
        pytest.skip("SparseCore selection requires TPU")
    rng = np.random.default_rng(821)
    rows, width, block, k = 4, 16384, 8, 512
    mapping = np.stack([rng.permutation(width // block) for _ in range(rows)]).astype(
        np.int32
    ) + (1 << 25)
    logical = (mapping[..., None] * block + np.arange(block)).reshape(rows, width)
    scores = rng.normal(size=(rows, width)).astype(np.float32)
    if distribution == "ties":
        scores = np.round(scores)
    elif distribution == "signed_zero":
        scores[:] = 0
        scores[:, ::2] = -0.0
    elif distribution == "subnormal":
        scores[:] = np.resize(
            np.array([0, 1, 0x80000000, 0x80000001, 0x007FFFFF, 0x807FFFFF], np.uint32),
            width,
        ).view(np.float32)
    scores[:, ::5] = -np.inf
    lengths = np.array([[0], [511], [8193], [width]], np.int32)
    actual = np.asarray(
        streaming_topk(
            jnp.asarray(scores),
            logical_blocks=jnp.pad(
                jnp.asarray(mapping),
                ((start_request or 0, 0), (0, 0)),
                constant_values=-1,
            ),
            start_request=None if start_request is None else jnp.int32(start_request),
            valid_lengths=jnp.asarray(lengths),
            exclude_neg_inf=True,
            unordered=True,
            k=k,
        )
    )
    for row in range(rows):
        eligible = np.flatnonzero(
            (np.arange(width) < lengths[row, 0]) & (scores[row] > -np.inf)
        )
        bits = scores[row, eligible].view(np.uint32)
        keys = bits ^ np.where(bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000))
        chosen = actual[row][actual[row] >= 0]
        assert len(chosen) == min(k, len(eligible))
        assert len(np.unique(chosen)) == len(chosen)
        slots_by_id = {int(record): slot for slot, record in enumerate(logical[row])}
        slots = np.array([slots_by_id[int(record)] for record in chosen], np.int32)
        assert np.all(np.isin(slots, eligible))
        selected_bits = scores[row, slots].view(np.uint32)
        selected_keys = selected_bits ^ np.where(
            selected_bits >> 31, np.uint32(0xFFFFFFFF), np.uint32(0x80000000)
        )
        np.testing.assert_array_equal(np.sort(selected_keys), np.sort(keys)[-k:])
