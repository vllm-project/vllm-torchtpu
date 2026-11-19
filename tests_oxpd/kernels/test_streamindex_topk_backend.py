# SPDX-License-Identifier: Apache-2.0
"""Selection contract and backend injection through the real scoring pipeline."""

from dataclasses import dataclass
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4.sparsecore_topk import sparsecore_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import streamindex_topk
from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk.topk_backend import (
    SparseCoreTopK,
)


@dataclass(frozen=True)
class ReferenceTopK:
    """Test-only JAX selector; not a proposed TensorCore implementation."""

    overlaps_scoring: ClassVar[bool] = False

    def select(self, score_bits, k, row_lengths, *, scheduling_group_id, return_scores):
        assert scheduling_group_id is None
        scores = jax.lax.bitcast_convert_type(score_bits, jnp.float32)
        scores = jnp.where(
            jnp.arange(scores.shape[1])[None, :] < row_lengths[:, None],
            scores,
            -jnp.inf,
        )
        values, ids = jax.lax.top_k(scores, k)
        ids = jnp.where(values > -jnp.inf, ids, -1)
        bits = jax.lax.bitcast_convert_type(values, jnp.int32)
        return (ids, bits) if return_scores else ids


@pytest.mark.parametrize("return_scores", [False, True])
def test_reference_contract_masks_rows_and_pads(return_scores):
    scores = np.array([[2, 2, 1, 100], [1, 2, 3, 4]], np.float32)
    selected = ReferenceTopK().select(
        jnp.asarray(scores.view(np.int32)),
        3,
        jnp.array([2, 0], jnp.int32),
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    ids = selected[0] if return_scores else selected
    np.testing.assert_array_equal(ids, [[0, 1, -1], [-1, -1, -1]])
    if return_scores:
        np.testing.assert_array_equal(
            np.asarray(selected[1]).view(np.float32),
            [[2, 2, -np.inf], [-np.inf, -np.inf, -np.inf]],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("return_scores", [False, True])
def test_default_selector_preserves_sparsecore_results(return_scores):
    rng = np.random.default_rng(42)
    scores = rng.integers(-4, 5, (32, 512)).astype(np.float32)
    scores[:, 31] = -np.inf
    bits = jnp.asarray(scores.view(np.int32))
    lengths = jnp.array([0, 17, 128, 511] * 8, jnp.int32)
    expected = sparsecore_topk(bits, 128, lengths, return_scores=return_scores)
    actual = SparseCoreTopK().select(
        bits,
        128,
        lengths,
        scheduling_group_id=None,
        return_scores=return_scores,
    )
    for got, want in zip(jax.tree.leaves(actual), jax.tree.leaves(expected)):
        np.testing.assert_array_equal(got, want)
    ids = np.asarray(actual[0] if return_scores else actual)
    for row, length in enumerate(np.asarray(lengths)):
        live = ids[row][ids[row] >= 0]
        assert np.all(live < length)
        assert len(live) == len(np.unique(live))
        eligible = scores[row, :length]
        eligible = eligible[eligible > -np.inf]
        np.testing.assert_array_equal(
            np.sort(scores[row, live]),
            np.sort(eligible)[-128:],
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("chunk_tokens", [None, 128])
@pytest.mark.parametrize("return_scores", [False, True])
def test_backend_replacement_keeps_real_streamindex_scores(chunk_tokens, return_scores):
    # All scores are zero: varying causal row lengths exercise ties, masks,
    # short rows and padding without a tolerance hiding selection errors.
    q = jnp.zeros((256, 4, 128), jnp.bfloat16)
    weights = jnp.zeros((256, 4), jnp.float32)
    cache = jnp.zeros((4, 32, 4, 256), jnp.uint8)
    args = (
        q,
        weights,
        cache,
        jnp.array([256, 257]),
        jnp.array([0, 1, 2, 3]),
        jnp.array([0, 128, 256]),
        jnp.array([0, 0, 2]),
    )
    kwargs = dict(
        k=128,
        compression_ratio=2,
        num_kv_pages_per_block=2,
        num_queries_per_block=(1, 128, 128),
        chunk_tokens=chunk_tokens,
        return_scores=return_scores,
    )
    default = streamindex_topk(*args, **kwargs)
    alternate = streamindex_topk(*args, topk_backend=ReferenceTopK(), **kwargs)
    for result in (default, alternate):
        ids = np.asarray(result[0] if return_scores else result)
        positions = np.r_[np.arange(128, 256), np.arange(129, 257)]
        lengths = np.minimum(positions // 2 + 1, np.repeat([128, 128], 128))
        for row, length in enumerate(lengths):
            live = ids[row][ids[row] >= 0]
            assert len(live) == min(length, 128)
            assert len(live) == len(np.unique(live))
            assert np.all(live < length)
            assert np.all(ids[row, len(live) :] == -1)
        if return_scores:
            scores = np.asarray(result[1]).view(np.float32)
            np.testing.assert_array_equal(scores[ids >= 0], 0)
            np.testing.assert_array_equal(scores[ids < 0], -np.inf)


@pytest.mark.parametrize("bounded", [False, True])
def test_sparsecore_backend_forwards_bounded_copy(monkeypatch, bounded):
    from vllm_torchtpu.kernels.deepseek_v4.streamindex_topk import topk_backend

    seen = {}

    def capture(score_bits, k, **kwargs):
        seen.update(kwargs)
        return jnp.zeros((score_bits.shape[0], k), jnp.int32)

    monkeypatch.setattr(topk_backend, "sparsecore_topk", capture)
    SparseCoreTopK(bounded_copy=bounded).select(
        jnp.zeros((4, 128), jnp.int32),
        16,
        jnp.full((4,), 128, jnp.int32),
        scheduling_group_id=None,
        return_scores=False,
    )
    assert seen["bounded_copy"] is bounded
    assert SparseCoreTopK() == SparseCoreTopK(bounded_copy=False)
    assert hash(SparseCoreTopK(bounded_copy=True)) != hash(SparseCoreTopK())
