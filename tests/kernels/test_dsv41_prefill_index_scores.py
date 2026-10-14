# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Numerical coverage for the single-request packed prefill scorer."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.prefill_indexer.scores import (
    prefill_index_scores,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("layout", ["contiguous", "shuffled", "bank_boundary"])
@pytest.mark.parametrize("limit", [0, 73, 256])
@pytest.mark.parametrize("coalesce", [False, True])
def test_prefill_scores(layout, limit, coalesce):
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
    )
    reordered = keys.reshape(width // page, page, 128)[order].reshape(width, 128)
    split = jnp.concatenate((reordered[:, ::2], reordered[:, 1::2]), axis=-1)
    logits = jnp.einsum("thd,kd->thk", q, split, preferred_element_type=jnp.float32)
    expected = (jnp.maximum(logits, 0) * w).sum(1)
    lengths = np.maximum(0, limit - np.arange(tokens))
    valid = (np.arange(width)[None] < lengths[:, None]) & np.repeat(
        pages >= 0, page, axis=1
    )
    np.testing.assert_allclose(
        np.asarray(actual)[valid], np.asarray(expected)[valid], rtol=1e-5, atol=2e-5
    )
    assert np.isneginf(np.asarray(actual)[:, ((limit + 63) // 64) * 64 :]).all()


@pytest.mark.cpu_test
@pytest.mark.parametrize("limit", [0, 65535, 65537, 131072])
def test_prefill_score_windows(limit):
    """Crossing the 64K VMEM window retains logical columns and zero-tail masking."""
    rng = np.random.default_rng(82)
    width, page = 131072, 128
    keys = jnp.asarray(rng.normal(size=(width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(keys.reshape(width, 64, 2), jnp.int32)
    words = words.reshape(width // 2, 128)
    order = rng.permutation(width // page)
    pages = jnp.asarray((order * page)[None], jnp.int32)
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
    )
    reordered = keys.reshape(-1, page, 128)[order].reshape(width, 128)
    split = jnp.concatenate((reordered[:, ::2], reordered[:, 1::2]), axis=-1)
    expected = jnp.maximum(q.reshape(1, 128).astype(jnp.float32) @ split.T, 0)
    np.testing.assert_allclose(
        np.asarray(actual)[:, :limit],
        np.asarray(expected)[:, :limit],
        rtol=1e-5,
        atol=2e-5,
    )
    assert np.isneginf(np.asarray(actual)[:, ((limit + 2047) // 2048) * 2048 :]).all()
