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
"""BF16 paired-page sparse prefill: selection, layout and numeric contracts."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.pallas import tpu as pltpu

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
    block_sparse_gqa_prefill,
)
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles


@pytest.fixture
def tpu_info(monkeypatch):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    return info


@pytest.mark.tpu_test
@pytest.mark.parametrize("page_size", [128, 1664])
@pytest.mark.parametrize("k_scale,v_scale", [(1.0, 1.0), (0.375, 2.5)])
def test_bf16_paired_prefill_preserves_ragged_requests_and_page_tails(
    page_size, k_scale, v_scale
):
    rng = np.random.default_rng(714)
    tokens, dim, budget = 513, 256, 16
    query = jnp.asarray(rng.normal(0, 0.5, (tokens, 24, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.normal(0, 2, query.shape), jnp.bfloat16)
    starts = jnp.array([0, 257, 257, 512], jnp.int32)
    lengths = np.concatenate(
        (np.arange(1, 258), np.arange(3 * page_size - 255, 3 * page_size), [0])
    ).astype(np.int32)
    blocks = np.full((tokens, budget), -1, np.int32)
    for row, length in enumerate(lengths):
        count = min(length // 4, budget)
        blocks[row, :count] = rng.choice(length // 4, count, replace=False)
    tables = np.stack((rng.permutation(3), rng.permutation(3) + 3)).astype(np.int32)
    owners = np.array([0] * 257 + [1] * 255 + [0])
    table = tables[owners]
    stored = np.asarray(
        jnp.asarray(rng.normal(0, 0.4, (6, page_size, 2, 2, dim)), jnp.bfloat16)
    ).copy()
    for owner, last in [(0, 257), (1, 3 * page_size - 1)]:
        for logical_page, physical in enumerate(tables[owner]):
            stored[
                physical, np.arange(page_size) + logical_page * page_size >= last
            ] = np.nan
    cache = jnp.asarray(stored)
    args = (
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
        starts,
    )
    common = dict(
        kv_heads=2, ratio=4, query_block=256, k_scale=k_scale, v_scale=v_scale
    )
    candidate, returned = jax.jit(
        functools.partial(block_sparse_gqa_attention, **common, kv_tile=256)
    )(*args)
    original, _ = jax.jit(
        functools.partial(block_sparse_gqa_attention, **common, kv_tile=128)
    )(*args)
    actual = np.asarray(candidate, np.float32)
    expected = np.asarray(original, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[-1], 0)
    np.testing.assert_array_equal(
        np.asarray(returned).view(np.uint16), stored.view(np.uint16)
    )
    assert np.linalg.norm(actual - expected) / np.linalg.norm(expected) < 0.01
    # Independent selected-token reference at request, group and page boundaries.
    for row in (0, 255, 256, 257, 511):
        length = int(lengths[row])
        count = min(length // 4, budget)
        selected = (blocks[row, :count, None] * 4 + np.arange(4)).reshape(-1)
        selected = np.concatenate((selected, np.arange(length // 4 * 4, length)))
        values = stored[table[row, selected // page_size], selected % page_size]
        values = values.reshape(-1, 4, dim).astype(np.float64)
        wanted = np.empty((24, dim), np.float64)
        for head in range(2):
            scores = np.asarray(query[row, head * 12 : (head + 1) * 12], np.float64)
            scores = scores @ values[:, 2 * head].T * (k_scale / 16)
            prob = np.exp(scores - scores.max(axis=1, keepdims=True))
            prob /= prob.sum(axis=1, keepdims=True)
            wanted[head * 12 : (head + 1) * 12] = (
                prob @ values[:, 2 * head + 1] * v_scale
            )
        weight = jnp.asarray(
            1 / (1 + np.exp(-np.asarray(gate[row], np.float64))), jnp.bfloat16
        )
        wanted = np.asarray(jnp.asarray(wanted, jnp.bfloat16), np.float32) * np.asarray(
            weight, np.float32
        )
        wanted = np.asarray(jnp.asarray(wanted, jnp.bfloat16), np.float32)
        assert np.linalg.norm(actual[row] - wanted) / np.linalg.norm(wanted) < 0.01


@pytest.mark.parametrize("page_size", [128, 1664])
@pytest.mark.parametrize("dtype,packing", [(jnp.bfloat16, 2), (jnp.float8_e4m3fn, 4)])
def test_measured_prefill_tiles_include_bf16_cache(tpu_info, page_size, dtype, packing):
    shape = jax.ShapeDtypeStruct
    query = shape((8192, 24, 256), jnp.bfloat16)
    cache = shape((3, page_size, 4 // packing, packing, 256), dtype)
    assert get_sparse_prefill_tiles(query, cache, kv_heads=2, ratio=4) == (256, 256)


@pytest.mark.parametrize(
    "tokens,page_size,chip",
    [
        (512, 129, pltpu.ChipVersion.TPU_7X),
        (8192, 129, pltpu.ChipVersion.TPU_7X),
        (8192, 128, pltpu.ChipVersion.TPU_V6E),
    ],
)
def test_bf16_prefill_unmeasured_shapes_keep_single_word_tiles(
    monkeypatch, tokens, page_size, chip
):
    monkeypatch.setattr(
        pltpu, "get_tpu_info", lambda: pltpu.get_tpu_info_for_chip(chip, 1)
    )
    shape = jax.ShapeDtypeStruct
    query = shape((tokens, 24, 256), jnp.bfloat16)
    cache = shape((3, page_size, 2, 2, 256), jnp.bfloat16)
    assert get_sparse_prefill_tiles(query, cache, kv_heads=2, ratio=4) == (256, 128)


@pytest.mark.parametrize(
    "query_block,page_size,head_axis,packing,dim,dtype",
    [
        (64, 128, 2, 2, 256, jnp.bfloat16),
        (256, 129, 2, 2, 256, jnp.bfloat16),
        (256, 128, 3, 2, 256, jnp.bfloat16),
        (256, 128, 2, 2, 384, jnp.bfloat16),
        (256, 128, 4, 1, 256, jnp.float32),
    ],
)
def test_paired_prefill_rejects_unvalidated_cache_geometry(
    tpu_info, query_block, page_size, head_axis, packing, dim, dtype
):
    shape = jax.ShapeDtypeStruct
    with pytest.raises(
        ValueError, match="Paired pages require|selection word or two mask segments"
    ):
        jax.eval_shape(
            functools.partial(
                block_sparse_gqa_prefill,
                ratio=4,
                kv_heads=2,
                query_block=query_block,
                kv_tile=256,
                bf16_io=True,
            ),
            shape((256, 24, 256), jnp.bfloat16),
            shape((256, 16), jnp.int32),
            shape((256,), jnp.int32),
            shape((256, 1), jnp.int32),
            shape((1, page_size, head_axis, packing, dim), dtype),
        )


@pytest.mark.tpu_test
@pytest.mark.parametrize("page_size", [128, 1664])
def test_selected_bf16_prefill_tiles_run_paired_kernel(page_size):
    tokens = 8192
    pages = (tokens + page_size - 1) // page_size
    query = jnp.zeros((tokens, 24, 256), jnp.bfloat16)
    lengths = jnp.arange(1, tokens + 1, dtype=jnp.int32)
    # Select the last complete group; partial groups remain implicitly visible.
    blocks = jnp.where(lengths[:, None] >= 4, lengths[:, None] // 4 - 1, -1)
    table = jnp.broadcast_to(jnp.arange(pages, dtype=jnp.int32), (tokens, pages))
    cache = jnp.broadcast_to(
        jnp.array([0, 1, 0, 2], jnp.bfloat16).reshape(1, 1, 2, 2, 1),
        (pages, page_size, 2, 2, 256),
    )
    args = (query, query, blocks, lengths, table, cache, jnp.array([0, tokens]))
    query_block, kv_tile = get_sparse_prefill_tiles(query, cache, kv_heads=2, ratio=4)
    compiled = (
        jax.jit(
            functools.partial(
                block_sparse_gqa_attention,
                kv_heads=2,
                ratio=4,
                query_block=query_block,
                kv_tile=kv_tile,
            )
        )
        .lower(*args)
        .compile()
    )
    assert "prefill_paired_pages" in compiled.as_text()
    actual, returned = compiled(*args)
    expected = jnp.broadcast_to(
        jnp.repeat(jnp.array([0.5, 1], jnp.bfloat16), 12)[None, :, None],
        query.shape,
    )
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(returned, cache)
