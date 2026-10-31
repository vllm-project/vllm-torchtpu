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
"""Certified metadata prefixes preserve causal selection and full fallback."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from vllm_torchtpu.kernels.qwen4_exp import block_sparse_gqa_prefill as tc


@pytest.mark.parametrize(
    "case,narrow",
    [
        ("causal", True),
        ("duplicate", True),
        ("invalid", True),
        ("idle", True),
        ("high_group", False),
        ("long_history", False),
        ("negative_length", False),
    ],
)
def test_prefix_dispatch_certifies_runtime_values(monkeypatch, case, narrow):
    def core(query, blocks, lengths, table, cache, bitset, **kwargs):
        return jnp.full_like(query, table.shape[1])

    monkeypatch.setattr(tc, "_prefill", core)
    query = jnp.zeros((256, 4, 128), jnp.float32)
    cache = jnp.zeros((12, 128, 1, 4, 128), jnp.float8_e4m3fn)
    table = jnp.zeros((256, 12), jnp.int32)
    blocks = jnp.broadcast_to(jnp.arange(16, dtype=jnp.int32), (256, 16))
    if case == "duplicate":
        blocks = jnp.zeros_like(blocks)
    lengths = jnp.full(256, 17, jnp.int32)
    if case == "invalid":
        blocks = blocks.at[:, 0].set(12 * 128 // 4)
    elif case == "idle":
        blocks = blocks.at[:].set(-1)
        lengths = lengths.at[:].set(0)
    elif case == "high_group":
        blocks = blocks.at[:, 0].set(256 // 4)
    elif case == "long_history":
        lengths = lengths.at[:].set(257)
    elif case == "negative_length":
        lengths = lengths.at[:].set(-1)
    run = jax.jit(
        functools.partial(
            tc.block_sparse_gqa_prefill.__wrapped__,
            ratio=4,
            kv_heads=2,
            query_block=32,
            kv_tile=128,
        )
    )
    actual = run(query, blocks, lengths, table, cache)
    np.testing.assert_array_equal(actual, 2 if narrow else table.shape[1])
    assert actual.shape == query.shape


@pytest.mark.parametrize(
    "page,ratio,tile", [(128, 4, 128), (640, 8, 256), (1664, 4, 256)]
)
def test_prefix_metadata_matches_independent_causal_sets(page, ratio, tile):
    tokens, columns, budget = 256, 158, 16
    pages = tc._metadata_prefix_pages(tokens, budget, ratio, page, columns, tile)
    safe_groups = pages * page // tile * tile // ratio
    groups = columns * page // ratio
    rng = np.random.default_rng(5167)
    lengths = rng.integers(0, safe_groups * ratio + 1, tokens, dtype=np.int32)
    blocks = rng.integers(-1, safe_groups, (tokens, budget), dtype=np.int32)
    blocks[:, 1] = blocks[:, 0]
    blocks[::7, 0] = groups
    blocks[::11, 0] = np.iinfo(np.int32).max
    assert bool(
        tc._metadata_prefix_fits(
            jnp.asarray(blocks),
            jnp.asarray(lengths),
            ratio=ratio,
            full_groups=groups,
            prefix_groups=safe_groups,
        )
    )
    words = (
        ((pages * page + tile - 1) // tile * (tile // (ratio * 32)) + 127) // 128 * 128
    )
    bits = np.asarray(
        tc._selection_bitset(
            jnp.asarray(blocks),
            jnp.asarray(lengths),
            ratio=ratio,
            groups=pages * page // ratio,
            words=words,
        )
    ).view(np.uint32)
    expected = np.zeros_like(bits)
    for row, length in enumerate(lengths):
        complete = int(length) // ratio
        selected = {
            int(group)
            for group in blocks[row, : min(complete, budget)]
            if length > 0 and 0 <= group < complete
        }
        if length > 0 and length % ratio:
            selected.add(complete)
        for group in selected:
            expected[row, group // 32] |= np.uint32(1) << np.uint32(group % 32)
    np.testing.assert_array_equal(bits, expected)


@pytest.mark.parametrize("length", [8193, 8320, -1])
def test_prefix_rejects_incomplete_dma_tiles(length):
    pages = tc._metadata_prefix_pages(8192, 512, 4, 1664, 158, 256)
    assert pages == 5
    assert not bool(
        tc._metadata_prefix_fits(
            jnp.array([[0]], jnp.int32),
            jnp.array([length], jnp.int32),
            ratio=4,
            full_groups=158 * 1664 // 4,
            prefix_groups=pages * 1664 // 256 * 256 // 4,
        )
    )


@pytest.mark.tpu_test
@pytest.mark.parametrize("dtype", [jnp.bfloat16, jnp.float8_e4m3fn])
@pytest.mark.parametrize("history", [0, 512])
def test_native_prefix_matches_selected_token_reference(dtype, history):
    rng = np.random.default_rng(718)
    tokens, heads, dim, page, columns, budget, ratio = 256, 4, 128, 128, 12, 16, 4
    packing = 2 if dtype == jnp.bfloat16 else 4
    query = jnp.asarray(rng.normal(0, 0.3, (tokens, heads, dim)), jnp.bfloat16)
    cache = jnp.asarray(
        rng.normal(0, 0.3, (columns + 1, page, 4 // packing, packing, dim)), dtype
    )
    stored = np.asarray(cache).copy()
    table = np.broadcast_to(
        rng.permutation(columns).astype(np.int32) + 1, (tokens, columns)
    )
    lengths = history + np.arange(1, tokens + 1, dtype=np.int32)
    blocks = np.full((tokens, budget), -1, np.int32)
    for row, length in enumerate(lengths):
        count = min(int(length) // ratio, budget)
        if count:
            blocks[row, :count] = rng.integers(0, length // ratio, count)
    blocks[:, 1] = blocks[:, 0]
    blocks[127, 0] = np.iinfo(np.int32).max
    kwargs = dict(
        ratio=ratio,
        kv_heads=2,
        query_block=32,
        kv_tile=128,
        bf16_io=True,
        k_scale=1.0,
        v_scale=1.0,
    )
    operands = (
        query,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
    )
    actual = np.asarray(tc.block_sparse_gqa_prefill(*operands, **kwargs), np.float32)
    original = jax.jit(
        lambda q, bs, ls, pt, pool: tc._prefill(q, bs, ls, pt, pool, None, **kwargs)
    )(*operands)
    np.testing.assert_array_equal(actual, np.asarray(original, np.float32))
    np.testing.assert_array_equal(
        np.asarray(cache).view(np.uint8), stored.view(np.uint8)
    )
    for row in (0, 3, 4, 31, 127, 128, 255):
        length = int(lengths[row])
        groups = sorted(
            {
                int(group)
                for group in blocks[row, : min(length // ratio, budget)]
                if 0 <= group < length // ratio
            }
        )
        selected = [group * ratio + token for group in groups for token in range(ratio)]
        selected.extend(range(length // ratio * ratio, length))
        if not selected:
            np.testing.assert_array_equal(actual[row], 0)
            continue
        selected = np.asarray(selected)
        values = (
            stored[table[row, selected // page], selected % page]
            .reshape(-1, 4, dim)
            .astype(np.float64)
        )
        wanted = np.empty((heads, dim), np.float64)
        for head in range(2):
            span = slice(head * 2, (head + 1) * 2)
            logits = (
                np.asarray(query[row, span], np.float64)
                @ values[:, 2 * head].T
                / dim**0.5
            )
            probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
            probabilities /= probabilities.sum(axis=1, keepdims=True)
            wanted[span] = probabilities @ values[:, 2 * head + 1]
        assert np.linalg.norm(actual[row] - wanted) / np.linalg.norm(wanted) < 0.01
