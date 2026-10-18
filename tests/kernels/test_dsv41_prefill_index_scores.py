# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical coverage for the single-request packed prefill scorer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41 import indexer
from vllm_torchtpu.kernels.deepseek_v41.index_selection.scores import (
    prefill_index_scores,
)


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


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", ["contiguous", "shuffled", "bank_boundary"])
@pytest.mark.parametrize("limit", [0, 73, 256])
@pytest.mark.parametrize("coalesce", [False, True])
@pytest.mark.parametrize("record_mask", [False, True])
def test_prefill_scores(layout, limit, coalesce, record_mask):
    rng = np.random.default_rng(71)
    tokens, heads, width, page = 16, 4, 256, 16
    keys = jnp.asarray(rng.normal(size=(width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(keys.reshape(width, 64, 2), jnp.int32)
    words = words.reshape(width // 2, 128)
    # This boundary lies in the middle of a 64-key DMA tile.
    pool = (words[:48], words[48:]) if layout == "bank_boundary" else words
    order = np.arange(width // page, dtype=np.int32)
    if layout == "shuffled":
        rng.shuffle(order)
    pages = (order * page)[None]
    pages[0, -1] = -1
    records = (order[:, None] * page + np.arange(page)).reshape(1, width)
    records[0, -page:] = -1
    records[0, 7:10] = -1  # Partial-page holes must also become -inf.
    q = jnp.asarray(rng.normal(size=(tokens, heads, 128)), jnp.bfloat16)
    w = jnp.asarray(rng.normal(size=(tokens, heads, 1)), jnp.float32)
    actual = prefill_index_scores(
        q,
        w,
        pool,
        jnp.asarray(pages),
        jnp.array([limit], jnp.int32),
        page_records=page,
        query_tile=8,
        key_tile=64,
        coalesce=coalesce,
        interpret=True,
        records=jnp.asarray(records, jnp.int32) if record_mask else None,
    )
    reordered = keys.reshape(width // page, page, 128)[order].reshape(width, 128)
    split = jnp.concatenate((reordered[:, ::2], reordered[:, 1::2]), axis=-1)
    logits = jnp.einsum("thd,kd->thk", q, split, preferred_element_type=jnp.float32)
    expected = (jnp.maximum(logits, 0) * w).sum(1)
    lengths = np.maximum(0, limit - np.arange(tokens))
    valid = (np.arange(width)[None] < lengths[:, None]) & np.repeat(
        pages >= 0, page, axis=1
    )
    if record_mask:
        valid &= records >= 0
        assert np.isneginf(np.asarray(actual)[:, records[0] < 0]).all()
    np.testing.assert_allclose(
        np.asarray(actual)[valid], np.asarray(expected)[valid], rtol=1e-5, atol=2e-5
    )
    assert np.isneginf(np.asarray(actual)[:, ((limit + 63) // 64) * 64 :]).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("limit", [0, 65535, 65537, 131072])
@pytest.mark.parametrize("record_mask", [False, True])
def test_prefill_score_windows(limit, record_mask):
    """Crossing the 64K VMEM window retains logical columns and zero-tail masking."""
    rng = np.random.default_rng(82)
    width, page = 131072, 128
    keys = jnp.asarray(rng.normal(size=(width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(keys.reshape(width, 64, 2), jnp.int32)
    words = words.reshape(width // 2, 128)
    order = rng.permutation(width // page)
    pages = jnp.asarray((order * page)[None], jnp.int32)
    records = (order[:, None] * page + np.arange(page)).reshape(1, width)
    records[0, [17, 65535, 65539, width - 1]] = -1
    q = jnp.asarray(rng.normal(size=(1, 1, 128)), jnp.bfloat16)
    weights = jnp.ones((1, 1, 1), jnp.float32)
    actual = prefill_index_scores(
        q,
        weights,
        (words[: width // 4], words[width // 4 :]),
        pages,
        jnp.array([limit], jnp.int32),
        page_records=page,
        query_tile=1,
        key_tile=2048,
        interpret=True,
        records=jnp.asarray(records, jnp.int32) if record_mask else None,
    )
    reordered = keys.reshape(-1, page, 128)[order].reshape(width, 128)
    split = jnp.concatenate((reordered[:, ::2], reordered[:, 1::2]), axis=-1)
    expected = jnp.maximum(q.reshape(1, 128).astype(jnp.float32) @ split.T, 0)
    if record_mask:
        expected = jnp.where(records >= 0, expected, -jnp.inf)
        assert np.isneginf(np.asarray(actual)[:, records[0] < 0]).all()
    np.testing.assert_allclose(
        np.asarray(actual)[:, :limit],
        np.asarray(expected)[:, :limit],
        rtol=1e-5,
        atol=2e-5,
    )
    assert np.isneginf(np.asarray(actual)[:, ((limit + 2047) // 2048) * 2048 :]).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize(
    "shape,dtype", [((2, 256), jnp.int32), ((1, 256), jnp.float32)]
)
def test_prefill_rejects_invalid_record_mask(shape, dtype):
    """Record mask shape/dtype errors fail before Pallas lowering."""
    with pytest.raises(ValueError, match="Records must be int32"):
        prefill_index_scores(
            jnp.zeros((8, 4, 128), jnp.bfloat16),
            jnp.ones((8, 4, 1), jnp.float32),
            jnp.zeros((128, 128), jnp.int32),
            jnp.zeros((1, 16), jnp.int32),
            jnp.array([256], jnp.int32),
            page_records=16,
            query_tile=8,
            key_tile=64,
            interpret=True,
            records=jnp.zeros(shape, dtype),
        )
