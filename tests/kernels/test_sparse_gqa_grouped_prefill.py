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
"""SparseCore gathers preserve native KV and masked output ownership."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.sparse_gqa_prefill.grouped import sc_grouped_attention


@pytest.mark.tpu_test
@pytest.mark.parametrize("fp8", [False, True])
def test_grouped_prefill_masks_poisoned_padding_and_preserves_cache(fp8):
    if pltpu.get_tpu_info().chip_version != pltpu.ChipVersion.TPU_7X:
        pytest.skip("SparseCore prefill is measured on TPU7x")
    rng = np.random.default_rng(1071)
    dtype = jnp.float8_e4m3fn if fp8 else jnp.bfloat16
    packing = 4 if fp8 else 2
    query = jnp.asarray(rng.normal(0, 0.4, (32, 24, 256)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, query.shape), jnp.bfloat16)
    lengths = np.arange(32, 64, dtype=np.int32)
    lengths[[0, 31]] = 0
    blocks = np.full((32, 16), -1, np.int32)
    for row, length in enumerate(lengths):
        count = length // 4
        blocks[row, :count] = rng.permutation(count)
    table = jnp.ones((32, 1), jnp.int32)
    cache = jnp.asarray(
        rng.normal(0, 0.3, (2, 1664, 4 // packing, packing, 256)), dtype
    )
    cache = cache.at[1, 64:].set(jnp.nan)
    before = np.asarray(cache).view(np.uint8).copy()
    query = query.at[31].set(jnp.nan)
    gate = gate.at[0].set(jnp.nan)
    args = (query, gate, jnp.asarray(blocks), jnp.asarray(lengths), table, cache)
    actual, returned = jax.jit(
        functools.partial(sc_grouped_attention, kv_heads=2, tile_k=256)
    )(*args)
    expected, _ = jax.jit(
        functools.partial(
            block_sparse_gqa_attention,
            kv_heads=2,
            ratio=4,
            query_block=32,
            kv_tile=128,
        )
    )(*args, jnp.array([0, 32], jnp.int32))
    actual, expected = np.asarray(actual, np.float32), np.asarray(expected, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[lengths == 0], 0)
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    np.testing.assert_array_equal(np.asarray(returned).view(np.uint8), before)
