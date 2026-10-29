# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused publication preserves paged logits and visible block maxima."""

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
from vllm_torchtpu.kernels.deepseek_v41.index_selection.publication import (
    candidate_block_scores,
)


@pytest.mark.parametrize("width", [4096, 98816])
@pytest.mark.parametrize("bank_count", [1, 2])
def test_fused_paged_publication(width, bank_count):
    if jax.default_backend() != "tpu":
        pytest.skip("Requires the TPU paged MQA kernel")
    torch.manual_seed(829)
    rng = np.random.default_rng(829)
    layout = PackedFP8Layout(2 * width // 128)
    cache = _pack_fp8_cache(
        torch.randn((layout.pages, 128, 1, 128), dtype=torch.bfloat16), layout
    )
    banks = tuple(jnp.split(cache, bank_count))
    pages = rng.permutation(layout.pages)[: width // 128].reshape(1, -1)
    pages[0, -2] = -1
    q = jnp.asarray(rng.normal(size=(6, 32, 128))).astype(jnp.float8_e4m3fn)
    weights = jnp.asarray(rng.uniform(0.01, 0.1, (6, 32)), jnp.float32)
    lengths = jnp.asarray(
        [width, 0, 1, 7, width // 2 - 1, width]
        if width == 4096
        else [width - i for i in range(6)],
        jnp.int32,
    )
    kwargs = dict(
        query_start_loc=jnp.array([0, 6], jnp.int32),
        max_model_len=width,
        q_scale=jnp.ones_like(weights),
        query_size=6,
        cache_layout=layout,
        query_start=1,
        query_count=5,
    )
    args = (q, banks, weights, lengths, jnp.asarray(pages, jnp.int32))
    expected = fp8_paged_mqa_logits(*args, **kwargs)
    actual, blocks = fp8_paged_mqa_logits(*args, candidate_block_size=8, **kwargs)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
    )
    reference = candidate_block_scores(expected, lengths[1:], block_size=8)
    np.testing.assert_array_equal(
        np.asarray(blocks).view(np.uint32), np.asarray(reference).view(np.uint32)
    )
