# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contiguous pair windows preserve dense scores across every block offset."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from tests.kernels.deepseek_v41.test_fp8_paged_mqa_logits import _pack_fp8_cache
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.cache import PackedFP8Layout
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.dense_logits import (
    fp8_paged_mqa_logits,
)
from vllm_torchtpu.kernels.deepseek_v41.index_selection.fp8.sparse_logits import (
    fp8_gathered_mqa_logits,
)


@pytest.mark.parametrize(
    "width,start,count,tokens,group_size",
    [
        (128, 0, 32, 32, 2),
        (8192, 1, 29, 32, 2),
        (16384, 0, 32, 32, 2),
        (128, 1, 5, 8, 1),
        (128, 0, 1, 1, 1),
    ],
)
def test_gathered_windows_match_dense_with_unaligned_offsets_and_causal_holes(
    width, start, count, tokens, group_size
):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires the TPU MQA kernels")
    rng = np.random.default_rng(821)
    torch.manual_seed(821)
    capacity = width * 2
    layout = PackedFP8Layout(capacity // 128)
    cache = _pack_fp8_cache(
        torch.randn((layout.pages, 128, 1, 128), dtype=torch.bfloat16), layout
    )
    q = jnp.asarray(rng.normal(size=(tokens, 32, 128))).astype(jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (tokens, 32)), jnp.float32)
    offsets = jnp.arange(0, tokens + 1, group_size, dtype=jnp.int32)
    pages = jnp.asarray(
        np.stack([rng.permutation(layout.pages) for _ in range(tokens // group_size)])
    )
    gathered = cache.reshape(layout.pages, 33, 128)[pages].reshape(-1, 128)
    begins = np.zeros(tokens, np.int32)
    if group_size == 2:
        begins[1::2] = np.arange(tokens // 2) * 8 + (width // 2 // 128) * 128
    splits = begins + width // 2
    ends1 = begins + np.minimum(np.arange(tokens) * 13, width // 2)
    ends2 = splits + width // 2 - np.arange(tokens) % 8
    windows = np.stack((begins, splits, ends1, ends2), axis=1)
    if tokens > 4:
        windows[4] = 0
    baseline = fp8_paged_mqa_logits(
        q,
        cache,
        weights,
        jnp.full((tokens,), capacity, jnp.int32),
        pages,
        query_start_loc=offsets,
        max_model_len=capacity,
        q_scale=jnp.ones_like(weights),
        query_size=group_size,
        cache_layout=layout,
        query_start=jnp.int32(start),
        query_count=count,
    )
    actual = fp8_gathered_mqa_logits(
        q,
        gathered,
        weights,
        jnp.asarray(windows),
        query_start_loc=offsets,
        output_width=width,
        query_start=jnp.int32(start),
        query_count=count,
    )
    columns = begins[start : start + count, None] + np.arange(width)
    section = windows[start : start + count]
    allowed = (columns < section[:, 2, None]) | (
        (columns >= section[:, 1, None]) & (columns < section[:, 3, None])
    )
    expected = np.where(
        allowed, np.take_along_axis(np.asarray(baseline), columns, axis=1), -np.inf
    )
    actual = np.asarray(actual)
    assert actual.shape == (count, width)
    np.testing.assert_array_equal(np.isfinite(actual), np.isfinite(expected))
    live = np.isfinite(expected)
    np.testing.assert_allclose(actual[live], expected[live], rtol=0.01, atol=0.01)
