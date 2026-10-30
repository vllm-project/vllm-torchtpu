# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("starts", [[0, 128, 256], [0, 96, 256]])
def test_qsa_main_cache_universal_tiled_writer_matches_scatter(fp8, starts):
    from vllm_torchtpu.kernels.qwen4_exp.pooled_qsa import write_main_cache
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows

    rng = np.random.default_rng(88421)
    tokens, page_size = 256, 256
    dtype = jnp.float32 if fp8 else jnp.bfloat16
    key = jnp.asarray(
        rng.standard_normal((tokens, 2, 256), dtype=np.float32) * 0.1, dtype
    )
    value = jnp.asarray(
        rng.standard_normal((tokens, 2, 256), dtype=np.float32) * 0.1, dtype
    )
    rows = pack_rows(key, value, fp8=fp8)
    pool = jnp.zeros((5, page_size, *rows.shape[1:]), rows.dtype)
    starts = jnp.asarray(starts, jnp.int32)
    request = jnp.searchsorted(starts[1:], jnp.arange(tokens), side="right")
    logical = jnp.arange(tokens) - starts[request]
    page = 1 + request
    valid = jnp.ones((tokens,), bool)
    expected = pool.at[page, logical].set(rows)
    actual = jax.jit(write_main_cache)(pool, rows, page, logical, valid)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint8), np.asarray(expected).view(np.uint8)
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("query_block", [32, 64, 128, 256])
def test_qsa_block_sparse_matches_exact_mask_with_partial_tail_and_nan(
    fp8, query_block
):
    from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
        block_sparse_gqa_attention,
        block_sparse_gqa_prefill,
    )
    from vllm_torchtpu.kernels.qwen4_exp.qsa_rpa import pack_rows

    rng = np.random.default_rng(731904)
    tokens, context, page_size = query_block, 2304, 128
    query = jnp.asarray(
        rng.standard_normal((tokens, 24, 256), dtype=np.float32) * 0.1,
        dtype=jnp.bfloat16,
    )
    keys = rng.standard_normal((context, 2, 256), dtype=np.float32) * 0.1
    values = rng.standard_normal((context, 2, 256), dtype=np.float32) * 0.1
    keys[2052:] = np.nan
    values[2052:] = np.nan
    cache_dtype = jnp.float32 if fp8 else jnp.bfloat16
    cache = pack_rows(
        jnp.asarray(keys, cache_dtype), jnp.asarray(values, cache_dtype), fp8=fp8
    ).reshape(
        context // page_size, page_size, 2 if not fp8 else 1, 2 if not fp8 else 4, 256
    )
    probe = np.array([0, 1, 3, 4, 2047, 2048, 2049, 2051, 2052], np.int32)
    lengths = jnp.asarray(np.resize(probe, tokens))
    complete = np.minimum(np.asarray(lengths) // 4, 512)
    groups = np.arange(512, dtype=np.int32)[None, :]
    blocks = jnp.asarray(np.where(groups < complete[:, None], groups, -1))
    table = jnp.broadcast_to(
        jnp.arange(context // page_size, dtype=jnp.int32),
        (tokens, context // page_size),
    )
    native = np.asarray(cache, np.float32).reshape(context, 4, 256)
    native = np.where(np.isfinite(native), native, 0)
    expected = np.zeros(query.shape, np.float32)
    for row, length in enumerate(np.asarray(lengths)):
        completed = min(length // 4, 512)
        ids = np.concatenate(
            (np.arange(completed * 4), np.arange(length // 4 * 4, length))
        )
        if not len(ids):
            continue
        for head in range(24):
            kv_head = head // 12
            scores = (
                native[ids, 2 * kv_head] @ np.asarray(query[row, head], np.float32) / 16
            )
            probs = np.exp(scores - scores.max())
            expected[row, head] = (probs / probs.sum()) @ native[ids, 2 * kv_head + 1]
    actual = block_sparse_gqa_prefill(
        query,
        blocks,
        lengths,
        table,
        cache,
        ratio=4,
        kv_heads=2,
        query_block=query_block,
        bf16_io=query_block > 128,
    )
    actual = np.asarray(actual, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_allclose(
        actual,
        np.asarray(expected, np.float32),
        rtol=0.05,
        atol=0.03,
    )
    wrapped, _ = block_sparse_gqa_attention(
        query,
        jnp.zeros((tokens, 24, 1), jnp.float32),
        blocks,
        lengths,
        table,
        cache,
        jnp.asarray([0, tokens], jnp.int32),
        ratio=4,
        kv_heads=2,
        query_block=query_block,
    )
    gated_expected = (
        jnp.asarray(expected, jnp.bfloat16).astype(jnp.float32) * 0.5
    ).astype(jnp.bfloat16)
    np.testing.assert_allclose(
        np.asarray(wrapped, np.float32),
        np.asarray(gated_expected, np.float32),
        rtol=0.05,
        atol=0.03,
    )
