# SPDX-License-Identifier: Apache-2.0
"""QSA prefill tiles for tensor-parallel per-rank shapes (one KV head)."""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu
from test_qwen4_exp_paired_pages import _dense_selected_reference, pack_rows

from vllm_torchtpu.kernels.qwen4_exp.block_sparse_gqa_prefill import (
    block_sparse_gqa_attention,
)
from vllm_torchtpu.kernels.qwen4_exp.prefill_tiles import get_sparse_prefill_tiles


@pytest.fixture
def interpret(monkeypatch):
    original = pl.pallas_call
    monkeypatch.setattr(
        pl,
        "pallas_call",
        lambda *args, **kwargs: original(*args, **{**kwargs, "interpret": True}),
    )
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)


@pytest.mark.parametrize("query_heads", [3, 6, 12, 24])
@pytest.mark.parametrize("page_size", [128, 1664])
def test_tp_shard_tiles_are_paired(monkeypatch, query_heads, page_size):
    info = pltpu.get_tpu_info_for_chip(pltpu.ChipVersion.TPU_7X, 1)
    monkeypatch.setattr(pltpu, "get_tpu_info", lambda: info)
    kv_heads = 2 if query_heads == 24 else 1
    query = jax.ShapeDtypeStruct((8192, query_heads, 256), jnp.bfloat16)
    cache = jax.ShapeDtypeStruct((11, page_size, 1, 4, 256), jnp.float8_e4m3fn)
    query_block, kv_tile, query_tile = get_sparse_prefill_tiles(
        query, cache, kv_heads=kv_heads, ratio=4, with_query_tile=True
    )
    assert kv_tile >= 256
    assert query_block * query_heads <= 256 * 24
    if kv_heads == 2:
        assert (query_block, kv_tile, query_tile) == (256, 256, None)
        return
    assert query_block % query_tile == 0
    small = jax.ShapeDtypeStruct((200, query_heads, 256), jnp.bfloat16)
    query_block, _, query_tile = get_sparse_prefill_tiles(
        small, cache, kv_heads=1, ratio=4, with_query_tile=True
    )
    assert query_block == 256 and query_block % query_tile == 0


@pytest.mark.parametrize(
    "query_heads,query_block,kv_tile,query_tile",
    [
        (3, 512, 256, 512),  # two bitset parts, one 512-row subtile
        (3, 512, 512, 512),  # four selection words per KV tile
        (6, 512, 1024, 128),  # eight words, four subtiles
        (3, 512, 256, 256),  # two subtiles
        (3, 256, 256, None),
        (6, 512, 128, 256),  # unpaired, two parts and subtiles
        (3, 256, 128, None),  # previous default
    ],
)
@pytest.mark.parametrize("kv_head_offset", [0, 1])
def test_single_kv_head_matches_dense(
    interpret, query_heads, query_block, kv_tile, query_tile, kv_head_offset
):
    rng = np.random.default_rng(query_heads * 100 + query_block + kv_head_offset)
    ratio, page_size, pages, dim = 4, 128, 12, 256
    tokens = 1024
    # Two requests: a 600-token prefill on a 700-token context and a short
    # one, padded to the query block by the attention wrapper.
    starts = np.array([0, 600, 700], np.int32)
    contexts = np.array([700, 50], np.int32)
    request = np.searchsorted(starts[1:], np.arange(tokens), side="right")
    lengths = np.zeros(tokens, np.int32)
    live = request < 2
    lengths[live] = (
        contexts[request[live]] + np.arange(tokens)[live] - starts[request[live]] + 1
    )
    owners = np.array([rng.permutation(pages)[:pages], rng.permutation(pages)])
    table = owners[np.minimum(request, 1)].astype(np.int32)
    query = jnp.asarray(rng.normal(0, 0.3, (tokens, query_heads, dim)), jnp.bfloat16)
    gate = jnp.asarray(rng.uniform(-3, 3, query.shape), jnp.bfloat16)
    key = rng.normal(0, 0.3, (pages * page_size, 2, dim)).astype(np.float32)
    value = rng.normal(0, 0.3, key.shape).astype(np.float32)
    cache = pack_rows(key, value, fp8=True, k_scale=0.375, v_scale=2.5)
    cache = cache.reshape(pages, page_size, 1, 4, dim)
    stored = np.asarray(cache, np.float32).reshape(pages, page_size, -1, dim)
    head = kv_head_offset
    key = stored[:, :, 2 * head : 2 * head + 1] * 0.375
    value = stored[:, :, 2 * head + 1 : 2 * head + 2] * 2.5
    blocks = np.full((tokens, 64), -1, np.int32)
    for row, length in enumerate(lengths):
        selected = rng.permutation(length // ratio)[:64]
        blocks[row, : len(selected)] = selected
    expected = _dense_selected_reference(
        query, gate, key, value, blocks, lengths, table, ratio
    )
    run = jax.jit(
        functools.partial(
            block_sparse_gqa_attention,
            kv_heads=1,
            ratio=ratio,
            query_block=query_block,
            kv_tile=kv_tile,
            k_scale=0.375,
            v_scale=2.5,
            kv_head_offset=kv_head_offset,
            query_tile=query_tile,
        )
    )
    actual, _ = run(
        query,
        gate,
        jnp.asarray(blocks),
        jnp.asarray(lengths),
        jnp.asarray(table),
        cache,
        jnp.asarray(starts),
    )
    actual = np.asarray(actual, np.float32)
    assert np.isfinite(actual).all()
    np.testing.assert_array_equal(actual[lengths == 0], 0)
    error = np.linalg.norm((actual - expected).reshape(tokens, -1), axis=1)
    norm = np.linalg.norm(expected.reshape(tokens, -1), axis=1)
    assert np.max(error / np.maximum(norm, 1e-12)) < 0.01
