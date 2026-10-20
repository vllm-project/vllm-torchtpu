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

"""Projection, gate-only APE and complete paged-state write regression tests."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.deepseek_v4 import csa_cache_layout
from vllm_torchtpu.kernels.deepseek_v4.compress_and_store import (
    proj_and_save_state,
    project_and_save_state_ref,
)

pytestmark = pytest.mark.tpu_test


@pytest.mark.parametrize("layout", ["has256", "slabs128", "words128"])
@pytest.mark.parametrize("position", [0, 3, 65536])
@pytest.mark.parametrize("invalid", [False, True])
def test_indexer_single_token_projection(layout, position, invalid):
    check_projection(1, 128, True, 4, layout, position, invalid)


@pytest.mark.parametrize(
    "tokens,head_dim,overlap,ratio",
    [(4, 128, True, 4), (129, 128, True, 4), (1, 512, True, 4), (1, 512, False, 128)],
)
def test_original_projection_shapes(tokens, head_dim, overlap, ratio):
    check_projection(tokens, head_dim, overlap, ratio, "slabs128", 7, False)


def check_projection(tokens, head_dim, overlap, ratio, layout, position, invalid):
    rng = np.random.default_rng(473)
    state_dim = 2 * (1 + int(overlap)) * head_dim
    lanes = 256 if layout == "has256" else 128
    rows_per_token = state_dim // lanes
    page_rows = 64
    pages = (tokens * rows_per_token + page_rows - 1) // page_rows + 2
    # Dyadic operands make the independent FP32 reference exact, including APE.
    hidden = jnp.asarray(rng.integers(-2, 3, (tokens, 4096)) / 16, jnp.bfloat16)
    weights = jnp.asarray(rng.integers(-2, 3, (4096, state_dim)) / 16, jnp.bfloat16)
    ape = jnp.asarray(rng.integers(-8, 9, (ratio, state_dim // 2)) / 32, jnp.float32)
    positions = jnp.asarray(position + np.arange(tokens), jnp.int32)
    # Last state ends exactly at the end of the last page. Unwritten bytes
    # carry sentinels, including the slack in Indexer's state pages.
    slots = pages * page_rows - (np.arange(tokens) + 1) * rows_per_token
    if invalid:
        slots[:] = -1
    elif tokens > 1:
        slots[1] = -1
    slots = jnp.asarray(slots, jnp.int32)
    # The reference unpacks all cache bytes as FP32 before scattering. Finite
    # sentinels avoid NaN canonicalization in untouched reference rows.
    sentinel = rng.normal(size=(pages, page_rows, lanes)).astype(np.float32)
    slabs = jnp.asarray(
        sentinel.view(np.uint8)
        .reshape(pages, page_rows, lanes, 4)
        .transpose(0, 1, 3, 2)
        .copy()
    )
    cache = csa_cache_layout.slabs_to_words(slabs) if layout == "words128" else slabs
    actual = proj_and_save_state.proj_and_save_state(
        hidden, weights, ape, positions, slots, cache, ratio
    )
    with jax.default_matmul_precision("highest"):
        expected = project_and_save_state_ref.ref_wkv_proj_and_save_state(
            hidden.astype(jnp.float32),
            weights.astype(jnp.float32),
            ape,
            positions,
            slots,
            slabs,
            1,
            head_dim,
            ratio,
            overlap,
        )
    if layout == "words128":
        actual = csa_cache_layout.words_to_slabs(actual)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


@pytest.mark.parametrize(
    "state_width,compress_ratio", [(256, 4), (1024, 4), (512, 128)]
)
@pytest.mark.parametrize("num_tokens", [1, 9])
@pytest.mark.parametrize("word_cache", [False, True])
@pytest.mark.parametrize("invalid_slots", [False, True])
def test_joint_projection_state_writes(
    state_width, compress_ratio, num_tokens, word_cache, invalid_slots
):
    rng = np.random.default_rng(51)
    hidden_size = 4096
    # Dyadic values make FP32 accumulation exact, including the gate APE.
    x = rng.integers(-4, 5, (num_tokens, hidden_size)).astype(np.float32) / 32
    weight = rng.integers(-4, 5, (hidden_size, 2 * state_width)).astype(np.float32) / 64
    ape = rng.integers(-32, 33, (compress_ratio, state_width)).astype(np.float32) / 64
    positions = np.arange(num_tokens, dtype=np.int32) + compress_ratio - 1
    rows_per_state = 2 * state_width // 128
    # Start at the end of page zero, then write into page one.
    slots = (
        256 - rows_per_state + np.arange(num_tokens, dtype=np.int32) * rows_per_state
    )
    if invalid_slots:
        slots[::2] = -1

    cache = rng.integers(0, 256, (2, 256, 4, 128), dtype=np.uint8)
    expected = cache.copy()
    kv = x @ weight[:, :state_width]
    gate = x @ weight[:, state_width:] + ape[positions % compress_ratio]
    packed = np.concatenate((kv, gate), axis=1)
    packed_bytes = (
        packed.reshape(num_tokens, rows_per_state, 128)
        .view(np.uint8)
        .reshape(num_tokens, rows_per_state, 128, 4)
        .transpose(0, 1, 3, 2)
    )
    for i, slot in enumerate(slots):
        if slot >= 0:
            page, row = divmod(int(slot), 256)
            expected[page, row : row + rows_per_state] = packed_bytes[i]

    device_cache = jnp.asarray(cache)
    if word_cache:
        device_cache = csa_cache_layout.slabs_to_words(device_cache)
    actual = proj_and_save_state.proj_and_save_state(
        jnp.asarray(x, jnp.bfloat16),
        jnp.asarray(weight, jnp.bfloat16),
        jnp.asarray(ape),
        jnp.asarray(positions),
        jnp.asarray(slots),
        device_cache,
        compress_ratio,
    )
    if word_cache:
        actual = csa_cache_layout.words_to_slabs(actual)
    np.testing.assert_array_equal(np.asarray(actual), expected)
