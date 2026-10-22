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
"""Typed byte coverage, masked fallback and BF16 pair layout contracts."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp import pooled_qsa
from vllm_torchtpu.kernels.qwen4_exp.cache_regions import region_rows


def test_region_rounding_does_not_lose_integer_capacity_at_large_values():
    pool = jax.ShapeDtypeStruct((1, 128, 1, 2, 256), jnp.bfloat16)
    # No tensor allocation. IEEE division rounds this numerator downward.
    elements = 2**60 + 1
    expected_tokens = (2 * elements + 1023) // 1024
    assert region_rows(pool, elements, jnp.bfloat16) == (
        expected_tokens,
        expected_tokens * 2,
    )


@pytest.mark.parametrize("tokens", [1, 17, 127, 129])
def test_partial_main_scatter_respects_valid_even_for_in_range_page_ids(tokens):
    pool = jnp.full((3, 128, 1, 2, 128), -1, jnp.bfloat16)
    rows = jnp.ones((tokens, 1, 2, 128), jnp.bfloat16)
    page = jnp.ones((tokens,), jnp.int32)
    logical = jnp.arange(tokens, dtype=jnp.int32)
    valid = logical % 3 == 0
    # Run the actual universal scatter; the valid flag is part of its API,
    # independent of the caller's additional out-of-range page encoding.
    actual = pooled_qsa._write_main_cache_universal(pool, rows, page, logical, valid)
    expected = np.asarray(pool).copy()
    for row in range(tokens):
        if bool(valid[row]):
            expected[1, row % 128] = 1
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("ratio", [0, 3, 128, 256])
def test_compressed_pair_layout_rejects_unsupported_ratio_clearly(ratio):
    with pytest.raises(ValueError, match="BF16 groups"):
        pooled_qsa._compressed_rows(ratio)


@pytest.mark.parametrize("ratio", [1, 2, 4, 8, 16, 32, 64])
def test_compressed_pair_layout_follows_logical_block_and_word_packing(ratio):
    rows = pooled_qsa._compressed_rows(ratio)
    assert rows * ratio == 128 and rows % 2 == 0


def test_pair_fallback_scalar_budget_counts_metadata_not_values(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    # Eval-shape reaches the real non-tiled pair path with a wide VMEM operand.
    # This is tracing evidence, not runtime/cross-chip qualification.
    n, width = 129, 512
    result = jax.eval_shape(
        lambda pool, keys: pooled_qsa.write_compressed(
            pool,
            keys,
            jnp.arange(n) % 4 == 3,
            jnp.zeros(n, jnp.int32),
            jnp.arange(n),
            jnp.array([0, n]),
            jnp.array([0]),
            jnp.array([[1, 2]]),
            4,
        ),
        jax.ShapeDtypeStruct((3, 128, 1, 2, width), jnp.bfloat16),
        jax.ShapeDtypeStruct((n, width), jnp.bfloat16),
    )
    assert result.shape == (3, 128, 1, 2, width)
