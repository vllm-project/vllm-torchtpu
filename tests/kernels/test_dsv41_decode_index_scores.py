# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request isolation and ragged masking for grouped decode scoring."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v41.index_selection.decode import (
    grouped_decode_scores,
)


@pytest.mark.cpu_test
@pytest.mark.parametrize("banks", [1, 2])
@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("fuse_mask", [False, True])
@pytest.mark.parametrize("score_window", [128, 32768])
def test_grouped_decode_scores_preserve_requests_and_causal_lengths(
    banks, prefetch, fuse_mask, score_window
):
    rng = np.random.default_rng(47)
    requests, queries, heads, width, page = 3, 6, 4, 256, 16
    keys = jnp.asarray(rng.normal(size=(requests * width, 128)), jnp.bfloat16)
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32)
    words = words.reshape(-1, 128)
    pool = (words[:176], words[176:]) if banks == 2 else words
    order = rng.permutation(requests * width // page).reshape(requests, -1)
    pages = order.astype(np.int32) * page
    pages[0, 3] = -1
    q = jnp.asarray(rng.normal(size=(requests, queries, heads, 128)), jnp.bfloat16)
    weights = jnp.asarray(rng.normal(size=(requests, queries, heads, 1)), jnp.float32)
    lengths = jnp.array(
        [[256, 255, 128, 73, 0, 5], [0] * 6, [3, 64, 129, 230, 254, 256]], jnp.int32
    )
    result = grouped_decode_scores(
        q,
        weights,
        pool,
        jnp.asarray(pages),
        lengths,
        page_records=page,
        key_tile=64,
        prefetch=prefetch,
        interpret=True,
        fuse_mask=fuse_mask,
        score_window=score_window,
    ).reshape(requests, queries, width)
    logical = keys.reshape(-1, page, 128)[order].reshape(requests, width, 128)
    logical = jnp.concatenate((logical[..., ::2], logical[..., 1::2]), axis=-1)
    logits = jnp.einsum(
        "rqhd,rkd->rqhk", q, logical, preferred_element_type=jnp.float32
    )
    expected = (jnp.maximum(logits, 0) * weights).sum(2)
    valid = (jnp.arange(width)[None, None] < lengths[:, :, None]) & jnp.repeat(
        jnp.asarray(pages >= 0), page, axis=1
    )[:, None]
    expected = jnp.where(valid, expected, -jnp.inf)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=3e-5)


@pytest.mark.cpu_test
def test_grouped_decode_scores_reject_incompatible_query_weights():
    with pytest.raises(ValueError, match="request-major"):
        grouped_decode_scores(
            jnp.zeros((1, 6, 4, 128), jnp.bfloat16),
            jnp.zeros((1, 5, 4, 1), jnp.float32),
            jnp.zeros((128, 128), jnp.int32),
            jnp.zeros((1, 16), jnp.int32),
            jnp.zeros((1, 6), jnp.int32),
            page_records=16,
            key_tile=64,
            interpret=True,
        )


@pytest.mark.skipif(jax.default_backend() != "tpu", reason="TPU SMEM lowering")
@pytest.mark.parametrize("fuse_mask", [False, True])
def test_grouped_decode_page_table_beyond_128_entries(fuse_mask):
    requests, queries, width, page = 3, 6, 4096, 16
    values = jnp.arange(requests * width, dtype=jnp.float32).astype(jnp.bfloat16)
    keys = jnp.broadcast_to(values[:, None], (requests * width, 128))
    words = jax.lax.bitcast_convert_type(keys.reshape(-1, 64, 2), jnp.int32)
    pool = words.reshape(-1, 128)
    pages = (
        jnp.arange(requests * width // page, dtype=jnp.int32).reshape(requests, -1)
        * page
    )
    result = grouped_decode_scores(
        jnp.ones((requests, queries, 1, 128), jnp.bfloat16),
        jnp.ones((requests, queries, 1, 1), jnp.float32),
        pool,
        pages,
        jnp.full((requests, queries), width, jnp.int32),
        page_records=page,
        key_tile=256,
        fuse_mask=fuse_mask,
    ).reshape(requests, queries, width)
    expected = jnp.broadcast_to(
        values.reshape(requests, 1, width).astype(jnp.float32) * 128, result.shape
    )
    np.testing.assert_array_equal(result, expected)
